# MosaicFormer

This folder is the full notebook-friendly implementation of the project idea:

> **MosaicFormer: A Multi-Scale Memory Transformer for Efficient Small Language Models**

It is intentionally written in the same simple PyTorch style as the `GPT` notebooks:
explicit tensor shapes, a custom `LayerNorm`, custom `Gelu`, readable attention code,
and no hidden framework magic.

## Core idea

The full-attention GPT baseline uses:

`Attention(Q, K, V) = softmax(QK^T / sqrt(d_h)) V`

MosaicFormer replaces long full attention with four components:

1. **Local causal attention** keeps exact detail inside a local window.
2. **Memory compression** turns each old chunk into a few learned vectors:
   `M = softmax(PK^T / sqrt(d)) V`.
3. **Memory cross-attention** lets the current chunk retrieve from compressed history.
4. **Gated fusion** learns the balance:
   `H_out = G * H_local + (1 - G) * H_memory`.

The approximate attention comparison is `O(T^2)` for full attention and
`O(Tw + Tm)` for local window `w` and memory size `m`.

## Folder map

- `model.py`: complete MosaicFormer architecture.
- `baseline.py`: ordinary full-causal-attention GPT for a fair baseline.
- `data.py`: GPT-style next-token dataset plus synthetic recall data.
- `train_utils.py`: loss, training, evaluation, generation, and recall accuracy.
- `run_experiments.py`: one command for the full baseline/ablation comparison.
- `Architecture_Diagram.md`: the project diagram for your report or presentation.
- `RUN_COMMANDS.md`: ready-to-run local and Colab commands.
- `NOTEBOOKS/`: all MosaicFormer notebooks.
- `RESULT/`: all CSV files, plots, and future experiment outputs.
- `tests/`: small correctness checks for masks, memory shapes, ablations, and parameter matching.
- `UNDERSTANDING_MOSAICFORMER.md`: flow-wise beginner explanation with Mermaid diagrams.

## Recommended order

1. Open `NOTEBOOKS/MosaicFormer_Architecture_Implementation.ipynb` and run all cells. Confirm all tensor-shape tests pass.
2. Open `NOTEBOOKS/MosaicFormer_Pretraining_and_Experiments.ipynb` and run the small The Verdict experiment.
3. Change `DATA_PATH` to `../GPT/tinystories_1000.txt` once the small run works.
4. Run the four experimental conditions fairly: baseline GPT, local only, memory without gate, full MosaicFormer.
5. Move to a larger dataset and repeat the long-context recall experiment for your final report.

## Command-line experiment

From this folder:

```powershell
python run_experiments.py --data_path ..\the-verdict.txt --epochs 10
```

New CSV results are saved into `RESULT/` by default. Existing old CSV and PNG
outputs are also kept there so the project root stays clean. Every new result
filename gets a timestamp, so older runs are not overwritten.

For TinyStories:

```powershell
python run_experiments.py --data_path ..\GPT\tinystories_1000.txt --epochs 5 --run_recall --recall_epochs 20 --output_csv tinystories_results.csv --recall_output_csv tinystories_recall_results.csv
```

For large local text files such as WikiText-103, keep lazy dataset slicing on
and cap the number of batches per epoch:

```powershell
python run_experiments.py --data_path DATA\wikitext103_train.txt --epochs 3 --max_train_batches 500 --eval_iter 20 --eval_freq 100 --output_csv wikitext103_lazy_results.csv
```

This uses the whole text file as the source, but it does not materialize every
training window in RAM and it does not require finishing millions of windows in
one epoch.

For faster Colab runs, compare only the parameter-matched GPT baseline and full
MosaicFormer, use mixed precision, and cache token IDs:

```powershell
python run_experiments.py --data_path DATA\wikitext103_train.txt --epochs 3 --max_train_batches 500 --eval_iter 20 --final_eval_iter 20 --eval_freq 250 --models matched,full --amp --token_cache DATA\wikitext103_tokens.npy --output_csv wikitext103_fast_results.csv
```

After the first run creates `DATA/wikitext103_tokens.npy`, later runs can load
the token cache directly and start faster with lower system RAM pressure.

For longer Colab runs, checkpoint commands are listed in `RUN_COMMANDS.md`.
Checkpointing is optional: normal local runs work without checkpoint flags.

For a 4GB GPU start with the default config: `emb_dim=128`, `n_layers=4`,
`n_heads=4`, `context_length=256`, `chunk_size=64`, and batch size 4.

## Tests

From this folder:

```powershell
python -m unittest discover -s tests
```

## Rights And Release Status

Copyright (c) 2026 Bharat Gusaiwal. All rights reserved.

This project is currently in a protected pre-publication stage. The source code,
documentation, diagrams, experiment outputs, and architecture explanations are
not licensed for copying, redistribution, public reuse, or commercial use unless
explicit written permission is given or a different license is added later.

See:

- `LICENSE` for the current rights notice.
- `AUTHORS.md` for authorship and AI-assistance disclosure.
- `MODEL_CARD.md` for the Hugging Face-ready model description.

## Research checklist

- Keep tokenizer, data split, context length, optimizer, learning rate, training tokens, and seed fixed.
- Report validation loss, perplexity, time, peak GPU memory, parameter count, and synthetic recall accuracy.
- Treat the first The Verdict run as a correctness check, not a publishable result.
- For a proper study, repeat each condition with at least three random seeds and report mean plus standard deviation.
