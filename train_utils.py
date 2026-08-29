"""Training, evaluation, generation, and recall utilities for MosaicFormer.

Copyright (c) 2026 Bharat Gusaiwal. All rights reserved.
"""

import math
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())


def calc_loss_batch(model, input_batch, target_batch, device, use_amp=False):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    with torch.amp.autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
        logits = model(input_batch)
    return F.cross_entropy(logits.flatten(0, 1), target_batch.flatten())


def calc_loss_loader(model, dataloader, device, num_batches=None, use_amp=False):
    model.eval()
    if len(dataloader) == 0:
        raise ValueError("The dataloader is empty. Check context_length and data size.")

    num_batches = len(dataloader) if num_batches is None else min(num_batches, len(dataloader))
    total_loss = 0.0
    with torch.no_grad():
        for index, (input_batch, target_batch) in enumerate(dataloader):
            if index >= num_batches:
                break
            total_loss += calc_loss_batch(model, input_batch, target_batch, device, use_amp=use_amp).item()
    model.train()
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter, use_amp=False):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(model, train_loader, device, eval_iter, use_amp=use_amp)
        val_loss = calc_loss_loader(model, val_loader, device, eval_iter, use_amp=use_amp)
    model.train()
    return train_loss, val_loss


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs,
    eval_freq,
    eval_iter,
    max_train_batches=None,
    final_eval_iter=None,
    use_amp=False,
    start_epoch=0,
    start_global_step=-1,
    start_tokens_seen=0,
    checkpoint_callback=None,
):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen = start_tokens_seen
    global_step = start_global_step

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda")
    start_time = time.time()

    for epoch in range(num_epochs):
        display_epoch = start_epoch + epoch + 1
        model.train()
        for batch_index, (input_batch, target_batch) in enumerate(train_loader):
            if max_train_batches is not None and batch_index >= max_train_batches:
                break

            optimizer.zero_grad()
            loss = calc_loss_batch(model, input_batch, target_batch, device, use_amp=use_amp)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model,
                    train_loader,
                    val_loader,
                    device,
                    eval_iter,
                    use_amp=use_amp,
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(
                    f"Epoch {display_epoch} Step {global_step}: "
                    f"train loss {train_loss:.4f}, val loss {val_loss:.4f}"
                )

            if checkpoint_callback is not None:
                checkpoint_callback(
                    model,
                    optimizer,
                    {
                        "epoch": display_epoch,
                        "global_step": global_step,
                        "tokens_seen": tokens_seen,
                        "train_losses": train_losses,
                        "val_losses": val_losses,
                        "track_tokens_seen": track_tokens_seen,
                    },
                    is_final=False,
                )

    elapsed_sec = time.time() - start_time
    final_val_loss = calc_loss_loader(model, val_loader, device, final_eval_iter, use_amp=use_amp)
    peak_gpu_mem_gb = 0.0
    if device.type == "cuda":
        peak_gpu_mem_gb = torch.cuda.max_memory_allocated(device) / 1e9

    if checkpoint_callback is not None:
        checkpoint_callback(
            model,
            optimizer,
            {
                "epoch": start_epoch + num_epochs,
                "global_step": global_step,
                "tokens_seen": tokens_seen,
                "train_losses": train_losses,
                "val_losses": val_losses,
                "track_tokens_seen": track_tokens_seen,
                "final_val_loss": final_val_loss,
            },
            is_final=True,
        )

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "track_tokens_seen": track_tokens_seen,
        "elapsed_sec": elapsed_sec,
        "final_val_loss": final_val_loss,
        "perplexity": math.exp(final_val_loss),
        "peak_gpu_mem_gb": peak_gpu_mem_gb,
        "num_params": count_parameters(model),
        "global_step": global_step,
        "tokens_seen": tokens_seen,
    }


def generate(model, idx, max_new_tokens, context_size, temperature=0.8, top_k=20):
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_value = top_logits[:, -1]
            logits = torch.where(
                logits < min_value,
                torch.tensor(float("-inf"), device=logits.device),
                logits,
            )

        if temperature > 0.0:
            probabilities = torch.softmax(logits / temperature, dim=-1)
            idx_next = torch.multinomial(probabilities, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    model.train()
    return idx


def text_to_token_ids(text, tokenizer):
    return torch.tensor(tokenizer.encode(text)).unsqueeze(0)


def token_ids_to_text(token_ids, tokenizer):
    return tokenizer.decode(token_ids.squeeze(0).tolist())


def evaluate_needle_recall(model, input_ids, targets, device):
    """Accuracy for the last-token value in the synthetic recall task."""
    model.eval()
    with torch.no_grad():
        logits = model(input_ids.to(device))
        predictions = torch.argmax(logits[:, -1, :], dim=-1).cpu()
    return (predictions == targets).float().mean().item()


def train_needle_recall_task(model, input_ids, targets, device, num_epochs=20, batch_size=32, learning_rate=3e-4):
    """Train a model on the controlled key-value recall task.

    The loss is computed only at the final prediction position. This makes the
    result easy to interpret: can the model recover an early value after a long
    distractor sequence?
    """
    dataset = TensorDataset(input_ids, targets)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0.0
        for input_batch, target_batch in dataloader:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)
            optimizer.zero_grad()
            logits = model(input_batch)[:, -1, :]
            loss = F.cross_entropy(logits, target_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Recall epoch {epoch + 1}: loss {total_loss / len(dataloader):.4f}")

    accuracy = evaluate_needle_recall(model, input_ids, targets, device)
    print(f"Final recall accuracy: {accuracy:.4f}")
    return accuracy


def run_needle_recall_comparison(
    baseline_model,
    mosaic_model,
    input_ids,
    targets,
    device,
    num_epochs=20,
    batch_size=32,
    learning_rate=3e-4,
):
    """Train two models on the same recall data and return their accuracies."""
    print("\n========== Needle recall: baseline GPT ==========")
    baseline_accuracy = train_needle_recall_task(
        baseline_model,
        input_ids,
        targets,
        device,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )

    print("\n========== Needle recall: MosaicFormer ==========")
    mosaic_accuracy = train_needle_recall_task(
        mosaic_model,
        input_ids,
        targets,
        device,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )

    print("\nRecall comparison")
    print(f"Baseline GPT accuracy: {baseline_accuracy:.4f}")
    print(f"MosaicFormer accuracy: {mosaic_accuracy:.4f}")
    return {
        "baseline_accuracy": baseline_accuracy,
        "mosaic_accuracy": mosaic_accuracy,
    }
