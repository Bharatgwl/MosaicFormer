"""Run fair comparisons: full-attention GPT, MosaicFormer, and ablations.

Copyright (c) 2026 Bharat Gusaiwal. All rights reserved.

Example:
python run_experiments.py --data_path ../the-verdict.txt --epochs 10
python run_experiments.py --data_path ../the-verdict.txt --run_recall
"""
import argparse
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import tiktoken

from baseline import GPTModel, find_parameter_matched_baseline_config
from data import create_dataloader, make_needle_recall_data, split_token_ids
from model import MosaicFormerModel, make_mosaic_config
from train_utils import count_parameters, run_needle_recall_comparison, train_model


PROJECT_DIR = Path(__file__).resolve().parent
RESULT_DIR = PROJECT_DIR / "RESULT"
MODEL_CHOICES = {
    "same_width",
    "matched",
    "local",
    "memory_no_gate",
    "full",
    "all",
}


def timestamped_result_path(path_text, run_timestamp):
    path = Path(path_text)
    if path.is_absolute():
        base_path = path
    else:
        base_path = RESULT_DIR / path

    suffix = base_path.suffix or ".csv"
    stem = base_path.stem
    candidate = base_path.with_name(f"{stem}_{run_timestamp}{suffix}")
    counter = 1
    while candidate.exists():
        candidate = base_path.with_name(f"{stem}_{run_timestamp}_{counter}{suffix}")
        counter += 1
    return candidate


def parse_model_selection(selection):
    selected = {item.strip() for item in selection.split(",") if item.strip()}
    unknown = selected - MODEL_CHOICES
    if unknown:
        raise ValueError(f"Unknown model names: {sorted(unknown)}")
    if not selected or "all" in selected:
        return {"same_width", "matched", "local", "memory_no_gate", "full"}
    return selected


def load_token_ids(data_path, tokenizer, token_cache=None):
    if token_cache is not None:
        cache_path = Path(token_cache)
        if cache_path.exists():
            print(f"Loading token cache: {cache_path}")
            return np.load(cache_path, mmap_mode="r")

    text_data = Path(data_path).read_text(encoding="utf-8")
    token_ids = tokenizer.encode(text_data, allowed_special={"<|endoftext|>"})

    if token_cache is not None:
        cache_path = Path(token_cache)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, np.asarray(token_ids, dtype=np.uint32))
        print(f"Saved token cache: {cache_path}")

    return token_ids


def slugify_name(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def checkpoint_root(args):
    path = Path(args.checkpoint_dir)
    if path.is_absolute():
        return path
    return PROJECT_DIR / path


def latest_checkpoint_path(name, args):
    return checkpoint_root(args) / f"{slugify_name(name)}_latest.pt"


def step_checkpoint_path(name, global_step, args):
    return checkpoint_root(args) / f"{slugify_name(name)}_step_{global_step}.pt"


def load_checkpoint_file(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def save_checkpoint(path, name, model, optimizer, state, args, model_config):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": name,
            "model_config": model_config,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": state["epoch"],
            "global_step": state["global_step"],
            "tokens_seen": state["tokens_seen"],
            "train_losses": state.get("train_losses", []),
            "val_losses": state.get("val_losses", []),
            "track_tokens_seen": state.get("track_tokens_seen", []),
            "final_val_loss": state.get("final_val_loss"),
            "args": vars(args),
            "saved_at": datetime.now().isoformat(timespec="seconds"),
        },
        path,
    )


def load_resume_state(name, model, optimizer, device, args):
    if args.resume_checkpoint is not None:
        path = Path(args.resume_checkpoint)
    elif args.resume:
        path = latest_checkpoint_path(name, args)
    else:
        return {
            "epoch": 0,
            "global_step": -1,
            "tokens_seen": 0,
            "path": None,
        }

    if not path.exists():
        print(f"No checkpoint found for {name}: {path}. Starting from scratch.")
        return {
            "epoch": 0,
            "global_step": -1,
            "tokens_seen": 0,
            "path": None,
        }

    checkpoint = load_checkpoint_file(path, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if args.override_lr_on_resume:
        for param_group in optimizer.param_groups:
            param_group["lr"] = args.learning_rate
        print(f"Override resume learning rate: {args.learning_rate}")
    print(
        f"Resumed {name} from {path} "
        f"(epoch {checkpoint.get('epoch', 0)}, step {checkpoint.get('global_step', -1)})"
    )
    return {
        "epoch": checkpoint.get("epoch", 0),
        "global_step": checkpoint.get("global_step", -1),
        "tokens_seen": checkpoint.get("tokens_seen", 0),
        "path": path,
    }


def make_checkpoint_callback(name, args, model_config):
    if not args.save_checkpoint:
        return None

    last_saved_step = {"value": None}

    def checkpoint_callback(model, optimizer, state, is_final=False):
        global_step = state["global_step"]
        should_save_step = (
            args.checkpoint_freq is not None
            and args.checkpoint_freq > 0
            and global_step > 0
            and global_step % args.checkpoint_freq == 0
            and last_saved_step["value"] != global_step
        )
        if not is_final and not should_save_step:
            return

        latest_path = latest_checkpoint_path(name, args)
        save_checkpoint(latest_path, name, model, optimizer, state, args, model_config)
        last_saved_step["value"] = global_step
        print(f"Checkpoint saved: {latest_path}")

        if should_save_step and args.keep_step_checkpoints:
            numbered_path = step_checkpoint_path(name, global_step, args)
            save_checkpoint(numbered_path, name, model, optimizer, state, args, model_config)
            print(f"Step checkpoint saved: {numbered_path}")

    return checkpoint_callback


def run_one_experiment(name, model, model_config, train_loader, val_loader, device, args):
    print(f"\n{'=' * 18} {name} {'=' * 18}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    resume_state = load_resume_state(name, model, optimizer, device, args)
    checkpoint_callback = make_checkpoint_callback(name, args, model_config)
    results = train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        args.epochs,
        args.eval_freq,
        args.eval_iter,
        args.max_train_batches,
        args.final_eval_iter,
        args.amp,
        start_epoch=resume_state["epoch"],
        start_global_step=resume_state["global_step"],
        start_tokens_seen=resume_state["tokens_seen"],
        checkpoint_callback=checkpoint_callback,
    )
    results["model"] = name
    results["start_step"] = resume_state["global_step"] + 1
    results["checkpoint_path"] = str(latest_checkpoint_path(name, args)) if args.save_checkpoint else ""
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--eval_freq", type=int, default=5)
    parser.add_argument("--eval_iter", type=int, default=3)
    parser.add_argument("--final_eval_iter", type=int, default=None)
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--materialize_dataset", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--models", default="all")
    parser.add_argument("--token_cache", default=None)
    parser.add_argument("--save_checkpoint", action="store_true")
    parser.add_argument("--checkpoint_dir", default="CHECKPOINT")
    parser.add_argument("--checkpoint_freq", type=int, default=1000)
    parser.add_argument("--keep_step_checkpoints", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume_checkpoint", default=None)
    parser.add_argument("--override_lr_on_resume", action="store_true")
    parser.add_argument("--output_csv", default="mosaicformer_experiment_results.csv")
    parser.add_argument("--config_size", choices=["small", "gpt2"], default="small")
    parser.add_argument("--run_recall", action="store_true")
    parser.add_argument("--recall_epochs", type=int, default=20)
    parser.add_argument("--recall_examples", type=int, default=256)
    parser.add_argument("--recall_batch_size", type=int, default=32)
    parser.add_argument("--recall_output_csv", default="needle_recall_results.csv")
    args = parser.parse_args()
    RESULT_DIR.mkdir(exist_ok=True)
    if args.save_checkpoint or args.resume or args.resume_checkpoint is not None:
        checkpoint_root(args).mkdir(parents=True, exist_ok=True)
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv = timestamped_result_path(args.output_csv, run_timestamp)
    recall_output_csv = timestamped_result_path(args.recall_output_csv, run_timestamp)
    if args.final_eval_iter is None:
        args.final_eval_iter = args.eval_iter

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = tiktoken.get_encoding("gpt2")
    token_ids = load_token_ids(args.data_path, tokenizer, token_cache=args.token_cache)
    cfg = make_mosaic_config(tokenizer.n_vocab, size=args.config_size)
    train_ids, val_ids = split_token_ids(token_ids)
    stride = args.stride if args.stride is not None else cfg["context_length"]
    lazy_dataset = not args.materialize_dataset
    train_loader = create_dataloader(train_ids, cfg["context_length"], stride, args.batch_size, True, True, lazy=lazy_dataset)
    val_loader = create_dataloader(val_ids, cfg["context_length"], stride, args.batch_size, False, True, lazy=lazy_dataset)
    print(f"Total tokens: {len(token_ids):,}")
    print(f"Train windows: {len(train_loader.dataset):,}")
    print(f"Validation windows: {len(val_loader.dataset):,}")
    print(f"Dataset mode: {'lazy slicing' if lazy_dataset else 'materialized tensors'}")
    print(f"AMP mixed precision: {'on' if args.amp and device.type == 'cuda' else 'off'}")
    print(f"Final eval batches: {args.final_eval_iter:,}")
    if args.max_train_batches is not None:
        print(f"Max train batches per epoch: {args.max_train_batches:,}")
    if args.save_checkpoint:
        print(f"Checkpoint dir: {checkpoint_root(args)}")
        if args.checkpoint_freq > 0:
            print(f"Checkpoint frequency: every {args.checkpoint_freq:,} steps")
        else:
            print("Checkpoint frequency: final checkpoint only")

    full_mosaic_cfg = cfg.copy()
    full_mosaic_cfg["use_memory"] = True
    full_mosaic_cfg["use_gate"] = True
    full_mosaic_params = count_parameters(MosaicFormerModel(full_mosaic_cfg))

    matched_baseline_cfg = find_parameter_matched_baseline_config(
        cfg,
        target_params=full_mosaic_params,
        max_emb_dim=max(512, cfg["emb_dim"] * 2),
    )
    matched_baseline_params = matched_baseline_cfg["matched_param_count"]
    matched_gap = matched_baseline_cfg["matched_param_gap_ratio"]
    print(
        "Parameter-matched baseline: "
        f"{matched_baseline_params / 1e6:.2f}M params "
        f"({matched_gap * 100:.2f}% gap vs full MosaicFormer)"
    )

    selected_models = parse_model_selection(args.models)
    experiment_builders = [
        ("same_width", "Baseline GPT same width", cfg.copy(), lambda model_cfg: GPTModel(model_cfg).to(device)),
        ("matched", "Baseline GPT parameter matched", matched_baseline_cfg, lambda model_cfg: GPTModel(model_cfg).to(device)),
        (
            "local",
            "MosaicFormer local only",
            make_mosaic_config(tokenizer.n_vocab, use_memory=False, size=args.config_size),
            lambda model_cfg: MosaicFormerModel(model_cfg).to(device),
        ),
        (
            "memory_no_gate",
            "MosaicFormer memory no gate",
            make_mosaic_config(tokenizer.n_vocab, use_memory=True, use_gate=False, size=args.config_size),
            lambda model_cfg: MosaicFormerModel(model_cfg).to(device),
        ),
        ("full", "MosaicFormer full", full_mosaic_cfg, lambda model_cfg: MosaicFormerModel(model_cfg).to(device)),
    ]
    experiments = [
        (name, model_config, build_model(model_config))
        for key, name, model_config, build_model in experiment_builders
        if key in selected_models
    ]
    if args.resume_checkpoint is not None and len(experiments) != 1:
        raise ValueError("--resume_checkpoint points to one file, so use it with exactly one selected model.")
    print("Selected models:", ", ".join(name for name, _, _ in experiments))

    results = []
    for name, model_config, model in experiments:
        results.append(run_one_experiment(name, model, model_config, train_loader, val_loader, device, args))

    report = pd.DataFrame([
        {
            "Model": result["model"],
            "Parameters (M)": result["num_params"] / 1e6,
            "Validation Loss": result["final_val_loss"],
            "Perplexity": result["perplexity"],
            "Train Time (s)": result["elapsed_sec"],
            "Peak GPU Memory (GB)": result["peak_gpu_mem_gb"],
            "Start Step": result["start_step"],
            "End Step": result["global_step"],
            "Checkpoint": result["checkpoint_path"],
        }
        for result in results
    ])
    report.to_csv(output_csv, index=False)
    print("\nFinal comparison")
    print(report.to_string(index=False))
    print(f"\nSaved: {output_csv}")

    if args.run_recall:
        recall_inputs, recall_targets = make_needle_recall_data(
            args.recall_examples,
            cfg["context_length"],
            vocab_size=128,
        )
        recall_cfg = make_mosaic_config(vocab_size=128, size=args.config_size)
        recall_baseline_cfg = find_parameter_matched_baseline_config(
            recall_cfg,
            target_params=count_parameters(MosaicFormerModel(recall_cfg)),
            max_emb_dim=max(512, recall_cfg["emb_dim"] * 2),
        )

        torch.manual_seed(42)
        recall_baseline = GPTModel(recall_baseline_cfg).to(device)
        torch.manual_seed(42)
        recall_mosaic = MosaicFormerModel(recall_cfg).to(device)
        recall_results = run_needle_recall_comparison(
            recall_baseline,
            recall_mosaic,
            recall_inputs,
            recall_targets,
            device,
            num_epochs=args.recall_epochs,
            batch_size=args.recall_batch_size,
            learning_rate=args.learning_rate,
        )
        recall_report = pd.DataFrame([
            {"Model": "Baseline GPT parameter matched", "Recall Accuracy": recall_results["baseline_accuracy"]},
            {"Model": "MosaicFormer full", "Recall Accuracy": recall_results["mosaic_accuracy"]},
        ])
        recall_report.to_csv(recall_output_csv, index=False)
        print("\nNeedle recall results")
        print(recall_report.to_string(index=False))
        print(f"\nSaved: {recall_output_csv}")


if __name__ == "__main__":
    main()
