# MosaicFormer Run Commands

This file is your command menu. Pick one command, change only the values you
need, and run it from the MosaicFormer project folder.

## What The `--values` Mean

- `--data_path`: text file used for language-model training.
- `--epochs`: how many training rounds to run.
- `--max_train_batches`: how many batches to train per epoch.
- `--eval_iter`: validation batches checked during training.
- `--final_eval_iter`: validation batches checked at the end.
- `--eval_freq`: how often to print train/validation loss.
- `--models`: which models to train.
- `--amp`: use faster mixed precision on CUDA GPUs.
- `--token_cache`: save/load tokenized data so startup is faster.
- `--save_checkpoint`: turn on checkpoint saving.
- `--checkpoint_dir`: folder where checkpoints are saved.
- `--checkpoint_freq`: save every N training steps.
- `--resume`: resume each selected model from its latest checkpoint.
- `--resume_checkpoint`: resume one selected model from one exact checkpoint file.
- `--override_lr_on_resume`: use the command's new learning rate after resume.

If you do not pass `--save_checkpoint` or `--resume`, the run has no checkpoint
behavior. This is best for small local tests.

## Colab Smoke Test

Use this first after uploading files to Colab. It checks that everything works.

```bash
!python run_experiments.py \
  --data_path DATA/wikitext103_train.txt \
  --epochs 1 \
  --max_train_batches 200 \
  --eval_iter 10 \
  --final_eval_iter 10 \
  --eval_freq 100 \
  --models matched,full \
  --amp \
  --token_cache DATA/wikitext103_tokens.npy \
  --output_csv wikitext103_smoke.csv
```

## Colab Fast Comparison

Use this for a quick useful comparison between the fair GPT baseline and full
MosaicFormer.

```bash
!python run_experiments.py \
  --data_path DATA/wikitext103_train.txt \
  --epochs 3 \
  --max_train_batches 500 \
  --eval_iter 20 \
  --final_eval_iter 20 \
  --eval_freq 250 \
  --models matched,full \
  --amp \
  --token_cache DATA/wikitext103_tokens.npy \
  --output_csv wikitext103_fast_results.csv
```

## Colab Bigger Run Without Checkpoint

Use this when you have enough runtime and do not need resume.

```bash
!python run_experiments.py \
  --data_path DATA/wikitext103_train.txt \
  --epochs 5 \
  --max_train_batches 5000 \
  --eval_iter 30 \
  --final_eval_iter 100 \
  --eval_freq 500 \
  --models matched,full \
  --amp \
  --token_cache DATA/wikitext103_tokens.npy \
  --output_csv wikitext103_5k_results.csv
```

## Colab Bigger Run With Checkpoint

Use this when you expect Colab to disconnect. Save checkpoints to Google Drive,
not only `/content`.

```bash
!python run_experiments.py \
  --data_path DATA/wikitext103_train.txt \
  --epochs 5 \
  --max_train_batches 5000 \
  --eval_iter 30 \
  --final_eval_iter 100 \
  --eval_freq 500 \
  --models matched,full \
  --amp \
  --token_cache DATA/wikitext103_tokens.npy \
  --save_checkpoint \
  --checkpoint_dir CHECKPOINT \
  --checkpoint_freq 1000 \
  --output_csv wikitext103_5k_checkpoint_results.csv
```

This creates files such as:

```text
CHECKPOINT/baseline_gpt_parameter_matched_latest.pt
CHECKPOINT/mosaicformer_full_latest.pt
```

## Resume Latest Checkpoints

Use this after Colab runtime refills. It resumes both selected models from their
latest checkpoint files.

```bash
!python run_experiments.py \
  --data_path DATA/wikitext103_train.txt \
  --epochs 5 \
  --max_train_batches 5000 \
  --eval_iter 30 \
  --final_eval_iter 100 \
  --eval_freq 500 \
  --models matched,full \
  --amp \
  --token_cache DATA/wikitext103_tokens.npy \
  --save_checkpoint \
  --checkpoint_dir CHECKPOINT \
  --checkpoint_freq 1000 \
  --resume \
  --output_csv wikitext103_5k_resume_results.csv
```

`--epochs 5` means train 5 more epochs after loading the checkpoint.

## Resume With Smaller Learning Rate

Use this when validation loss is still improving but has become slow. The
important part is `--override_lr_on_resume`, because the checkpoint normally
restores the old optimizer learning rate.

```bash
!python run_experiments.py \
  --data_path DATA/wikitext103_train.txt \
  --epochs 5 \
  --max_train_batches 10000 \
  --learning_rate 1e-4 \
  --eval_iter 30 \
  --final_eval_iter 100 \
  --eval_freq 1000 \
  --models matched,full \
  --amp \
  --token_cache DATA/wikitext103_tokens.npy \
  --save_checkpoint \
  --checkpoint_dir CHECKPOINT \
  --checkpoint_freq 5000 \
  --resume \
  --override_lr_on_resume \
  --output_csv wikitext103_resume_lr1e4.csv
```

## Resume One Exact Checkpoint

Use this when you want to resume only one model from one file.

```bash
!python run_experiments.py \
  --data_path DATA/wikitext103_train.txt \
  --epochs 5 \
  --max_train_batches 5000 \
  --eval_iter 30 \
  --final_eval_iter 100 \
  --eval_freq 500 \
  --models full \
  --amp \
  --token_cache DATA/wikitext103_tokens.npy \
  --save_checkpoint \
  --checkpoint_dir CHECKPOINT \
  --checkpoint_freq 1000 \
  --resume_checkpoint CHECKPOINT/mosaicformer_full_latest.pt \
  --output_csv mosaicformer_resume_only.csv
```

## Recall Test

This runs the synthetic memory-recall experiment.

```bash
!python run_experiments.py \
  --data_path DATA/wikitext103_train.txt \
  --epochs 1 \
  --max_train_batches 1 \
  --models matched,full \
  --run_recall \
  --recall_epochs 50 \
  --recall_examples 1024 \
  --recall_batch_size 32 \
  --amp \
  --token_cache DATA/wikitext103_tokens.npy \
  --output_csv recall_dummy_main.csv \
  --recall_output_csv recall_50epoch_results.csv
```

## Small Local Run

PowerShell command for your laptop, without checkpoint:

```powershell
python run_experiments.py --data_path ..\GPT\tinystories_1000.txt --epochs 3 --max_train_batches 200 --eval_iter 10 --final_eval_iter 10 --eval_freq 50 --models matched,full --output_csv tinystories_small.csv
```

## Small Local Run With Checkpoint

PowerShell command for your laptop, with checkpoint:

```powershell
python run_experiments.py --data_path ..\GPT\tinystories_1000.txt --epochs 3 --max_train_batches 200 --eval_iter 10 --final_eval_iter 10 --eval_freq 50 --models matched,full --save_checkpoint --checkpoint_dir CHECKPOINT --checkpoint_freq 100 --output_csv tinystories_checkpoint.csv
```

## Important Notes

- Results always go to `RESULT/` when you pass a relative CSV filename.
- Result CSV files get timestamps automatically.
- Checkpoints go to `CHECKPOINT/` only when `--save_checkpoint` is used.
- `--resume` is for automatic latest checkpoint loading.
- `--resume_checkpoint` is for one exact checkpoint file and one selected model.
- For fair research, train the baseline and MosaicFormer for the same number of
  total steps.
