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
| `train.py` | Optional supervised LoRA + RAG stage (checkpoint can be reused by **M3**). |
| `run_evaluate.py` | **M1/M2/M3/M4** inference and metrics. |
| `requirements.txt` | Python dependencies (includes **`higher`** for TTT-E2E meta-training). |
| `data/` | Merge questions/outputs and infer user keys (`data_io.py`). |
| `ttt/` | LoRA TTT (`training.py`); Flan **M4** (`flan_*.py` + `e2e.py` helpers); GPT-2 **TTT-E2E** (`mam_*.py`, `outer_meta.py`). |
| `train_mam_meta.py` | Optional **outer-loop** meta-training of `TTTGPT2` on LaMP train profiles. |
| `util/` | Add upstream LaMP to `sys.path`, metrics, LoRA, prompts. |

## Model stages (one supervised `train.py`)

| Model | `train.py` (global LoRA + RAG) | `run_evaluate.py` |
|-------|--------------------------------|----------------|
| **M1, M2** | Skip | Base model; M2 uses long-context ICL. |
| **M3** | Optional; use `--output_dir` as `--adapter_dir` if you train | **RAG** prompt at decode; if `--adapter_dir` is provided, loads the LoRA adapter, otherwise uses base model + RAG. |
| **M4** | Skip (no global LoRA) | **TTT-E2E:** GPT-2 + `TTTGPT2` (`--architecture causal_lm`) or Flan-T5 + `TTTFlanT5` (`seq2seq`); shared `--m4_inner_window`, `--m4_inner_stride`, `--m4_profile_max_tokens`. Optional checkpoints from `train_mam_meta.py` (GPT-2) and `train_flan_meta.py` (Flan). |

## GPU (CUDA)

- **`run_evaluate.py`** uses CUDA when `torch.cuda.is_available()`; otherwise CPU. The model and activations for generation run on that device. **`google/flan-t5-small`** is small, so wall time is often closer to CPU than you would see on a 7B model unless you raise **`--batch_size`** and use **`--fp16`** or **`--bf16`** (CUDA) to improve throughput. **`--verbose`** also runs extra metric passes per printed row (slow).
- **`train.py`** uses `Seq2SeqTrainer`, which picks CUDA automatically when available (unless you force CPU, e.g. `CUDA_VISIBLE_DEVICES=""`). Optional **`--fp16`** or **`--bf16`** (CUDA) turns on mixed-precision training; raise **`--batch_size`** when VRAM allows.

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

## Train global LoRA + RAG (optional checkpoint for **M3**)

**Required:** `train_questions.json` + `train_outputs.json`. **Optional:** `dev_questions.json` + `dev_outputs.json` for validation each epoch and `load_best_model_at_end`; if you omit dev, training runs without evaluation loops (checkpoints are still saved each epoch).

The `--output_dir` you pass here is the optional **`--adapter_dir`** for M3 in `run_evaluate.py`. Merged `merged_train.json` (and `merged_dev.json` when dev is set) are written under `--output_dir` for transparency.

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
- On **CUDA**, add e.g. `--fp16 --batch_size 8` (or `--bf16` on GPUs that support bfloat16) for faster steps than plain fp32.

## Evaluate M1/M2/M3/M4

**Required:** `test_questions.json` and `test_outputs.json`. The model is run on question-side fields (`input`, `profile`); predictions are aligned by `id` with gold strings from the outputs file for BLEU, ROUGE, and METEOR (via LaMP `generation_metrics`). Predictions are also written as **`pred_outputs.json`** in the same schema as the gold outputs file (`task` + `golds` with `id` / `output`, tab-indented). With multiple `--modes` in one run, files are named **`pred_outputs_<mode>.json`** so nothing is overwritten.

For **M4**, test rows are grouped by user (`user_id`-style fields, `--user_field`, or profile fingerprint) for per-user adaptation. M4 decodes on **`input` only**; the profile drives inner-loop updates only.

For **M3**, use the same `--num_retrieved`, `--retriever`, and `--ranked` (or absence of `--ranked`) as in `train.py` if you evaluate with a trained adapter.

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

### M3 — RAG (optional LoRA adapter)

```bash
python3 run_evaluate.py --task LaMP-5 \
  --test_questions_json path/to/test_questions.json \
  --test_outputs_json path/to/test_outputs.json \
  --modes m3 \
  --output_dir path/to/eval_out/m3
```

To evaluate M3 with finetuning, add `--adapter_dir path/to/checkpoints/lamp5_global_lora`.

### M4 — TTT-E2E on LaMP-5 / LaMP-7 (GPT-2 + MAM stack)

This is the **end-to-end TTT** setup merged from [MAM](https://github.com/evanly-gh/MAM): **DualMLP** (inner fast weights + static anchor), **test-time inner** = **one** left-to-right pass of sliding-window next-token CE on the user’s **profile-only** text (LaMP-5: titles/abstracts; LaMP-7: past tweets), matching the paper’s single stream over the context (§2.3). **Meta-training outer loop** (`higher` in `ttt/mam_outer.py`) is optional and separate from that single pass.

The benchmark **query** (`input`: instruction + text to personalize) is **never** used in TTT—only at generation—so you are not leaking the test prompt into adaptation. LaMP-7 uses **tweet-only** profile streams for inner TTT.

**Requirements**

- Install deps from the repo root: `python3 -m pip install -r requirements.txt` (pulls in **`higher`**).
- Use a **GPT-2** Hub id for `--base_model` (e.g. `gpt2` or `openai-community/gpt2-large`).
- Pass **`--architecture causal_lm`** (or rely on **`--architecture auto`**, which picks causal LM when the base model name contains `gpt2`).

**Step 1 — (Optional) Meta-train the outer loop on LaMP-5 or LaMP-7 train JSON**

This teaches the slow weights so held-out continuation NTP is good after inner TTT. Checkpoints and a CSV log are written to `--output_dir`.

```bash
python3 train_mam_meta.py --task LaMP-5 \
  --train_questions_json path/to/train_questions.json \
  --train_outputs_json path/to/train_outputs.json \
  --output_dir path/to/mam_ckpts \
  --model_name gpt2 \
  --meta_steps 500 \
  --ckpt_every 100
```

LaMP-7: use `--task LaMP-7` and your LaMP-7 train JSON paths (same flags). Meta-training uses **shorter** per-row floors for tweet-only profiles (`ttt/mam_data.py`).

For a quick wiring check, use `--meta_steps 20`. Larger `--model_name` (e.g. `openai-community/gpt2-large`) needs more RAM/VRAM.

**Step 1b — (Optional) Meta-train Flan-T5 outer loop (Dual-FFN)**

```bash
python3 train_flan_meta.py --task LaMP-5 \
  --train_questions_json path/to/train_questions.json \
  --train_outputs_json path/to/train_outputs.json \
  --output_dir path/to/flan_meta_ckpts \
  --model_name google/flan-t5-small \
  --meta_steps 500 \
  --ckpt_every 100
```

Use `--task LaMP-7` for tweet profiles. Pass `--m4_checkpoint path/to/flan_meta_ckpts/latest.pt` at evaluation time.

**Step 2 — Evaluate M4 on LaMP-5 or LaMP-7 test JSON**

Inner adaptation runs **per user** on the merged profile; inner weights are **reset** between users. Pass the meta checkpoint from Step 1 if you ran it; omit `--m4_checkpoint` to run **inner-only** TTT on top of pretrained GPT-2 with the same `TTTGPT2` architecture.

```bash
python3 run_evaluate.py --task LaMP-5 \
  --base_model gpt2 \
  --architecture causal_lm \
  --modes m4 \
  --test_questions_json path/to/test_questions.json \
  --test_outputs_json path/to/test_outputs.json \
  --m4_checkpoint path/to/mam_ckpts/latest.pt \
  --output_dir path/to/eval_out/m4_ttt_e2e
```

LaMP-7 example: set `--task LaMP-7` and LaMP-7 test paths. Tweet histories are often shorter than paper abstracts; try a **smaller** inner window if you want more windows per profile, e.g. `--m4_inner_window 128 --m4_inner_stride 64`.

Omit `--m4_checkpoint` if you did not run Step 1. Tune **`--ttt_lr`** and sliding-window flags for adaptation quality.

**Speed / paper alignment:** TTT-E2E at test time **streams the context once** (§2.3, [paper](https://arxiv.org/abs/2512.23675)). **GPT-2 M4:** ``inner_adapt_inplace`` — one SGD step per window. **Flan M4:** ``inner_adapt_t5_inplace`` — same one-pass idea over the profile stream. Cost per user is **~(#windows)**; widen **`--m4_inner_stride`** or shrink **`--m4_inner_window`** to reduce that. The official JAX stack is faster.

**M4 on Flan-T5 (paper-style sliding inner)**

With **`--base_model google/flan-t5-small`** and **`--architecture seq2seq`** (or **`auto`**), **`--modes m4`** uses **`TTTFlanT5`** + **`ttt/flan_inner.py`**: **one** left-to-right pass over the (truncated) profile stream, **one SGD step per sliding window** on the last-fraction **encoder+decoder FFN** trainable branch. Same **`--m4_inner_window`**, **`--m4_inner_stride`**, and **`--m4_profile_max_tokens`** as causal GPT-2 M4.

### All modes (M1/M2/M3/M4)

Flan-T5 line (M4 = **`TTTFlanT5` + `flan_inner`**; run GPT-2 M4 in a **separate** command with `--architecture causal_lm`):

```bash
python3 run_evaluate.py --task LaMP-5 \
  --test_questions_json path/to/test_questions.json \
  --test_outputs_json path/to/test_outputs.json \
  --modes m1,m2,m3,m4 \
  --output_dir path/to/eval_out/all
```

`--adapter_dir` is optional for `m3` and unused for `m1`, `m2`, and **`m4` in seq2seq mode**.

**Note:** You cannot mix **causal** M4 (`gpt2` + `TTTGPT2`) with **Flan** M1/M2/M3 in a single `run_evaluate.py` invocation, because `--base_model` and `--architecture` apply to the whole run. Run **TTT-E2E (GPT-2 M4)** as a separate command using the **M4 — TTT-E2E on LaMP-5 / LaMP-7** steps above.

## Common `run_evaluate.py` flags

| Flag | Purpose |
|------|---------|
| `--base_model` | Default `google/flan-t5-small`. M4 supports either Flan-T5 (`seq2seq`) or GPT-2 (`causal_lm`). |
| `--architecture` | `auto` (default), `seq2seq`, or `causal_lm`. **Causal** is required for GPT-2 M4 (`auto` selects it when `base_model` contains `gpt2`). |
| `--num_retrieved`, `--retriever`, `--ranked` | RAG; align with training / LaMP ranking. |
| `--max_input_length`, `--max_new_tokens`, `--batch_size` | Generation / batching; on GPU prefer larger `--batch_size` with `--fp16` or `--bf16`. **Causal LM** (`gpt2`, M1/M4): LaMP-5 decode treats **newline as an extra EOS** (stop after one line of title) plus a **``max_new_tokens`` ceiling** (64); LaMP-7 caps at 96 new tokens. Light post-processing only strips echoed ``Title:`` / whitespace. |
| `--fp16`, `--bf16` | CUDA half-precision inference (bf16 when the GPU supports it). **GPT-2 M4 (`TTTGPT2`)** loads in fp32 for stable inner steps. |
| `--ttt_steps`, `--ttt_lr` | **M4 (causal + seq2seq):** ``ttt_lr`` only for inner TTT; ``ttt_steps`` is legacy/unused. |
| `--m4_checkpoint` | **M4 (causal or seq2seq):** checkpoint path, e.g. `latest.pt` from `train_mam_meta.py` or `train_flan_meta.py` matching your `--architecture` and base model family. |
| `--m4_ttt_fraction` | M4 only: fraction of final blocks whose FFNs are adapted (default `0.25`, paper-style choice). |
| `--m4_inner_window`, `--m4_inner_stride` | **M4** sliding inner (Flan-T5 and GPT-2): window size and stride. For **GPT-2**, window must be ≤ `n_positions` (e.g. **1024**). Larger stride ⇒ fewer windows. |
| `--m4_profile_max_tokens` | **M4 (Flan + GPT-2):** tokenizer cap on the **merged profile** before inner sliding TTT. If unset, defaults to `min(4096, 8 × max_input_length)`. |
| `--user_field` | JSON field for user id when grouping test rows (**M4**). |
| `--cache_dir` | Hugging Face cache directory. |
| `--max_users` | If set to `K` (>0), only rows belonging to the **first K distinct users** (in merged test file order) are evaluated—handy for debugging without the full split. |
| `--verbose`, `--verbose_max_samples` | Print per-example inputs, profile counts, encoder preview, preds vs gold, and per-row BLEU/ROUGE/METEOR (cap rows with `verbose_max_samples`, `-1` = all). |

LaMP data and papers: `LaMP/README.md` (inside your clone) and [lamp-benchmark.github.io](https://lamp-benchmark.github.io/).
