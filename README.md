# lamp

LaMP-5 / LaMP-7 experiments: `train.py`, `evaluate.py`, `requirements.txt`, and the `data/`, `ttt/`, and `util/` packages live at the **repository root** (same directory as this `README.md`). Run commands from that root so imports resolve.

If you already have **train**, **dev** (validation), and **test** data from the [LaMP benchmark downloads](https://lamp-benchmark.github.io/download), point `train.py` at train + dev and `evaluate.py` at test. Nothing here re-splits the benchmark; user-based separation comes from those files.

The benchmark often ships **two JSON files per split** (e.g. `train_questions.json` with `input` + `profile`, and `train_outputs.json` with gold `output` lines keyed by `id`). Upstream `LaMP/LaMP/data/datasets.py` still consumes **one merged list per split**; the LaMP repo merges splits with `LaMP/LaMP/utils/merge_with_rank.py`. This repo accepts **either** merged arrays (`--train_json` / `--val_json` / `--test_json`) **or** the paired question + output paths below.

## Layout

| Path | Role |
|------|------|
| `train.py` | **Global** supervised stage: LoRA + RAG (checkpoint for **M3 and M4**). |
| `evaluate.py` | **M1–M5** inference and metrics (M4/M5 add test-time training here). |
| `requirements.txt` | Python dependencies. |
| `data/` | Load test JSON and infer user keys for M4/M5 (`data_io.py`). |
| `ttt/` | Test-time training (`training.py`). |
| `util/` | Add upstream LaMP to `sys.path`, metrics, LoRA, prompts. |

## Model stages (one supervised `train.py`)

| Model | `train.py` (global LoRA + RAG) | `evaluate.py` |
|-------|--------------------------------|----------------|
| **M1, M2** | Skip | Base model; M2 uses long-context ICL. |
| **M3** | Run once; use `--output_dir` as `--adapter_dir` | Load adapter; RAG; generate. |
| **M4** | **Same** checkpoint as M3 | Load adapter → TTT per test user → generate; reset LoRA between users. |
| **M5** | Skip | Fresh LoRA on base → TTT per user → generate. |

## GPU (CUDA)

- **`evaluate.py`** uses CUDA when `torch.cuda.is_available()`; otherwise CPU.
- **`train.py`** uses `Seq2SeqTrainer`, which picks CUDA automatically when available (unless you force CPU, e.g. `CUDA_VISIBLE_DEVICES=""`).

With LaMP’s **Contriever** retriever, upstream code may assume CUDA; the default here is **BM25** (CPU).

## Setup

From the **repository root** (directory that contains `train.py`, `requirements.txt`, and `LaMP/`):

```bash
python3 -m pip install -r requirements.txt
```

Download the submodules:
```bash
git submodule update --init --recursive
```

## Base model: `google/flan-t5-small`

Weights are downloaded from the [Hugging Face Hub](https://huggingface.co/google/flan-t5-small) on first use; you do not need a copy inside this repo. Default cache: `~/.cache/huggingface/hub` (override with `HF_HOME` or `--cache_dir`).

Optional prefetch:

```bash
python3 -m pip install -U "huggingface_hub[cli]"
hf download google/flan-t5-small
```

Use `--base_model <hub_id_or_local_path>` to switch models.

## Train global LoRA + RAG (checkpoint for **M3 and M4**)

Use benchmark **train** and **dev** JSON. The `--output_dir` you pass here is the **`--adapter_dir`** for M3/M4 in `evaluate.py`.

Replace `path/to/...` with real paths (relative paths are resolved from your current working directory, usually the repo root).

```bash
python3 train.py --task LaMP-5 --train_json path/to/LaMP5_train.json --val_json path/to/LaMP5_dev.json --output_dir path/to/checkpoints/lamp5_global_lora
```

Official split files (written under `--output_dir` as `merged_train.json` / `merged_val.json` for the trainer):

```bash
python3 train.py --task LaMP-5 \
  --train_questions_json path/to/train_questions.json --train_outputs_json path/to/train_outputs.json \
  --val_questions_json path/to/dev_questions.json --val_outputs_json path/to/dev_outputs.json \
  --output_dir path/to/checkpoints/lamp5_global_lora
```

- For **LaMP-7** (personalized tweet paraphrasing), use `--task LaMP-7` and your LaMP-7 train/dev JSON paths; profile items use `text` (see upstream `LaMP/LaMP` prompts).
- The `python3 train.py …` line above uses **LaMP-5** paths as an example; swap `--task` and file names for LaMP-7.
- `--ranked` if profiles are pre-ranked (LaMP `merge_with_rank.py` workflow).

## Evaluate M1–M5

Use benchmark **test** data: either a single merged `--test_json` array or `--test_questions_json` plus `--test_outputs_json` (same layout as the train/dev split files). **Every mode below** accepts either form; swap in the two-file flags in place of `--test_json` when you only have the official split files.

For **M4/M5**, test rows are grouped by user (`user_id`-style fields, `--user_field`, or profile fingerprint) for TTT and LoRA resets.

For **M3 and M4**, use the same `--num_retrieved`, `--retriever`, and `--ranked` (or absence of `--ranked`) as you used in `train.py`, so retrieved profile items match the checkpoint.

### M1 — zero-shot

```bash
python3 evaluate.py --task LaMP-5 --test_json path/to/LaMP5_test.json --modes m1 --output_dir path/to/eval_out/m1
```

### M2 — ICL

```bash
python3 evaluate.py --task LaMP-5 --test_json path/to/LaMP5_test.json --modes m2 --output_dir path/to/eval_out/m2
```

### M3 — global LoRA + RAG

```bash
python3 evaluate.py --task LaMP-5 --test_json path/to/LaMP5_test.json --modes m3 --adapter_dir path/to/checkpoints/lamp5_global_lora --output_dir path/to/eval_out/m3
```

### M4 — global LoRA + TTT

```bash
python3 evaluate.py --task LaMP-5 --test_json path/to/LaMP5_test.json --modes m4 --adapter_dir path/to/checkpoints/lamp5_global_lora --ttt_steps 30 --ttt_lr 1e-4 --output_dir path/to/eval_out/m4
```

### M5 — clean TTT

```bash
python3 evaluate.py --task LaMP-5 --test_json path/to/LaMP5_test.json --modes m5 --ttt_steps 30 --ttt_lr 1e-4 --output_dir path/to/eval_out/m5
```

`--adapter_dir` is not used for M5.

### All modes (M1–M5)

```bash
python3 evaluate.py --task LaMP-5 --test_json path/to/LaMP5_test.json --modes m1,m2,m3,m4,m5 --adapter_dir path/to/checkpoints/lamp5_global_lora --output_dir path/to/eval_out/all
```

`--adapter_dir` is **required** whenever `m3` or `m4` appears in `--modes` (including this all-modes line). It is unused for `m1`, `m2`, and `m5` but may still be passed.

## Common `evaluate.py` flags

| Flag | Purpose |
|------|---------|
| `--base_model` | Default `google/flan-t5-small`. |
| `--num_retrieved`, `--retriever`, `--ranked` | RAG; align with training / LaMP ranking. |
| `--max_input_length`, `--max_new_tokens`, `--batch_size` | Generation / batching. |
| `--user_field` | JSON field for user id when grouping test rows (M4/M5). |
| `--cache_dir` | Hugging Face cache directory. |

LaMP data and papers: `LaMP/README.md` (inside your clone) and [lamp-benchmark.github.io](https://lamp-benchmark.github.io/).
