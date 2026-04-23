# lamp

LaMP-5 / LaMP-7 experiments: `train.py`, `run_evaluate.py`, `requirements.txt`, and the `data/`, `ttt/`, and `util/` packages live at the **repository root** (same directory as this `README.md`). Run commands from that root so imports resolve. The eval script is named `run_evaluate.py` so it does not shadow the HuggingFace **`evaluate`** metrics package when LaMP code does `import evaluate`.

Data follows the [LaMP benchmark](https://lamp-benchmark.github.io/download) layout (see also `LaMP/README.md` in the submodule):

- **Questions JSON** (`train_questions.json`, `dev_questions.json`, `test_questions.json`, …): each entry has at least `id`, `input`, and `profile`.
- **Outputs JSON** (`train_outputs.json`, …): gold `output` per `id`, either as a list of `{"id", "output"}` objects or wrapped as `{"task": "LaMP_5", "golds": [ ... ]}` (same shape as leaderboard gold/pred files).

For **LaMP-5**, `input` must be the full natural-language task (instruction + abstract), and `output` the real gold title. If `input` / `output` are only short strings like `091-002276-EM.txt`, you have **unexpanded corpus ids** (the same idea as Avocado *questions* before `LaMP/data/avocado/create_avocado_dataset.py` fills in real email bodies). The model will see those filenames as text; metrics can look near-perfect but meaningless. Use the official LaMP-5 download JSON, not id-only stubs.

`train.py` and `run_evaluate.py` merge questions + outputs by `id` (same idea as `LaMP/LaMP/utils/merge_with_rank.py` without profile reranking). Upstream `GeneralSeq2SeqDataset` still reads one merged list per split; this repo writes merged copies under `--output_dir` when training.

After merge, **`data_io`** drops unusable examples for the given `--task`: **LaMP-5** removes rows whose `input` contains “no abstract available” and strips profile papers without a real `abstract`; **LaMP-7** does the analogous check on tweet `text`. A short summary is printed to stderr when anything is removed.

## Layout

| Path | Role |
|------|------|
| `train.py` | **Global** supervised stage: LoRA + RAG (checkpoint for **M3 and M4**). |
| `run_evaluate.py` | **M1–M5** inference and metrics (M4/M5 add test-time training here). |
| `requirements.txt` | Python dependencies. |
| `data/` | Merge questions/outputs and infer user keys for M4/M5 (`data_io.py`). |
| `ttt/` | Test-time training (`training.py`). |
| `util/` | Add upstream LaMP to `sys.path`, metrics, LoRA, prompts. |

## Model stages (one supervised `train.py`)

| Model | `train.py` (global LoRA + RAG) | `run_evaluate.py` |
|-------|--------------------------------|----------------|
| **M1, M2** | Skip | Base model; M2 uses long-context ICL. |
| **M3** | Run once; use `--output_dir` as `--adapter_dir` | Load adapter; **RAG** prompt at decode. |
| **M4** | **Same** checkpoint as M3 | TTT on profile, then decode on **task input only** (no RAG); reset LoRA between users. |
| **M5** | Skip | Fresh LoRA; TTT on profile; decode on **task input only** (no RAG). |

## GPU (CUDA)

- **`run_evaluate.py`** uses CUDA when `torch.cuda.is_available()`; otherwise CPU.
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

**Required:** `train_questions.json` + `train_outputs.json`. **Optional:** `dev_questions.json` + `dev_outputs.json` for validation each epoch and `load_best_model_at_end`; if you omit dev, training runs without evaluation loops (checkpoints are still saved each epoch).

The `--output_dir` you pass here is the **`--adapter_dir`** for M3/M4 in `run_evaluate.py`. Merged `merged_train.json` (and `merged_dev.json` when dev is set) are written under `--output_dir` for transparency.

Replace `path/to/...` with real paths (relative paths resolve from your current working directory, usually the repo root).

**Train only (two JSON files):**

```bash
python3 train.py --task LaMP-5 \
  --train_questions_json path/to/train_questions.json \
  --train_outputs_json path/to/train_outputs.json \
  --output_dir path/to/checkpoints/lamp5_global_lora
```

**Train + dev (recommended when you have the benchmark dev split):**

```bash
python3 train.py --task LaMP-5 \
  --train_questions_json path/to/train_questions.json \
  --train_outputs_json path/to/train_outputs.json \
  --dev_questions_json path/to/dev_questions.json \
  --dev_outputs_json path/to/dev_outputs.json \
  --output_dir path/to/checkpoints/lamp5_global_lora
```

- For **LaMP-7** (personalized tweet paraphrasing), use `--task LaMP-7` and your LaMP-7 paths; profile items use `text` (see upstream `LaMP/LaMP` prompts).
- `--ranked` if profiles are pre-ranked (LaMP `merge_with_rank.py` workflow).

## Evaluate M1–M5

**Required:** `test_questions.json` and `test_outputs.json`. The model is run on question-side fields (`input`, `profile`); predictions are aligned by `id` with gold strings from the outputs file for BLEU, ROUGE, and METEOR (via LaMP `generation_metrics`).

For **M4/M5**, test rows are grouped by user (`user_id`-style fields, `--user_field`, or profile fingerprint) for TTT and LoRA resets. **M4/M5** decode with the same string as **M1** (LaMP `input` only); the profile is used only during TTT updates, not as a retrieved prompt at generation time—so you can contrast **M3 (LoRA + RAG at decode)** with **M4/M5 (LoRA adaptation + no retrieval at decode)**. The M4 checkpoint still comes from `train.py` (RAG-supervised); that is a deliberate train/decode asymmetry unless you add a separate non-RAG training stage.

For **M3** only, use the same `--num_retrieved`, `--retriever`, and `--ranked` (or absence of `--ranked`) as in `train.py`, so retrieval matches how the checkpoint was trained. **M4/M5** do not use RAG at decode (profile is used only inside TTT); those flags still apply if you run **M3** in the same command.

In the examples below, replace `path/to/test_questions.json` and `path/to/test_outputs.json` with your benchmark files.

### M1 — zero-shot

```bash
python3 run_evaluate.py --task LaMP-5 \
  --test_questions_json path/to/test_questions.json \
  --test_outputs_json path/to/test_outputs.json \
  --modes m1 --output_dir path/to/eval_out/m1
```

### M2 — ICL

```bash
python3 run_evaluate.py --task LaMP-5 \
  --test_questions_json path/to/test_questions.json \
  --test_outputs_json path/to/test_outputs.json \
  --modes m2 --output_dir path/to/eval_out/m2
```

### M3 — global LoRA + RAG

```bash
python3 run_evaluate.py --task LaMP-5 \
  --test_questions_json path/to/test_questions.json \
  --test_outputs_json path/to/test_outputs.json \
  --modes m3 --adapter_dir path/to/checkpoints/lamp5_global_lora \
  --output_dir path/to/eval_out/m3
```

### M4 — global LoRA + TTT

```bash
python3 run_evaluate.py --task LaMP-5 \
  --test_questions_json path/to/test_questions.json \
  --test_outputs_json path/to/test_outputs.json \
  --modes m4 --adapter_dir path/to/checkpoints/lamp5_global_lora \
  --ttt_steps 30 --ttt_lr 1e-4 --output_dir path/to/eval_out/m4
```

### M5 — clean TTT

```bash
python3 run_evaluate.py --task LaMP-5 \
  --test_questions_json path/to/test_questions.json \
  --test_outputs_json path/to/test_outputs.json \
  --modes m5 --ttt_steps 30 --ttt_lr 1e-4 --output_dir path/to/eval_out/m5
```

`--adapter_dir` is not used for M5.

### All modes (M1–M5)

```bash
python3 run_evaluate.py --task LaMP-5 \
  --test_questions_json path/to/test_questions.json \
  --test_outputs_json path/to/test_outputs.json \
  --modes m1,m2,m3,m4,m5 --adapter_dir path/to/checkpoints/lamp5_global_lora \
  --output_dir path/to/eval_out/all
```

`--adapter_dir` is **required** whenever `m3` or `m4` appears in `--modes` (including this all-modes line). It is unused for `m1`, `m2`, and `m5` but may still be passed.

## Common `run_evaluate.py` flags

| Flag | Purpose |
|------|---------|
| `--base_model` | Default `google/flan-t5-small`. |
| `--num_retrieved`, `--retriever`, `--ranked` | RAG; align with training / LaMP ranking. |
| `--max_input_length`, `--max_new_tokens`, `--batch_size` | Generation / batching. |
| `--user_field` | JSON field for user id when grouping test rows (M4/M5). |
| `--cache_dir` | Hugging Face cache directory. |
| `--verbose`, `--verbose_max_samples` | Print per-example inputs, profile counts, encoder preview, preds vs gold, and per-row BLEU/ROUGE/METEOR (cap rows with `verbose_max_samples`, `-1` = all). |

LaMP data and papers: `LaMP/README.md` (inside your clone) and [lamp-benchmark.github.io](https://lamp-benchmark.github.io/).
