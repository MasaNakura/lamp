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
| `run_evaluate.py` | **M1–M6** inference and metrics (M4/M5 LoRA+TTT; **M6** TTT-E2E-style runs). |
| `requirements.txt` | Python dependencies (includes **`higher`** for TTT-E2E meta-training). |
| `data/` | Merge questions/outputs and infer user keys for M4/M5/M6 (`data_io.py`). |
| `ttt/` | LoRA TTT (`training.py`); Flan **M6** (`t5_sliding_ttt.py` + `e2e.py` helpers); GPT-2 **TTT-E2E** (`mam_*.py`, `outer_meta.py`). |
| `train_mam_meta.py` | Optional **outer-loop** meta-training of `TTTGPT2` on LaMP train profiles. |
| `util/` | Add upstream LaMP to `sys.path`, metrics, LoRA, prompts. |

## Model stages (one supervised `train.py`)

| Model | `train.py` (global LoRA + RAG) | `run_evaluate.py` |
|-------|--------------------------------|----------------|
| **M1, M2** | Skip | Base model; M2 uses long-context ICL. |
| **M3** | Run once; use `--output_dir` as `--adapter_dir` | Load adapter; **RAG** prompt at decode. |
| **M4** | **Same** checkpoint as M3 | TTT on profile, then decode on **task input only** (no RAG); reset LoRA between users. |
| **M5** | Skip | Fresh LoRA; TTT on profile; decode on **task input only** (no RAG). |
| **M6** | Skip (no global LoRA) | **TTT-E2E:** GPT-2 + `TTTGPT2` (`--architecture causal_lm`) or Flan-T5 + `t5_sliding_ttt.py` (`seq2seq`); shared `--m6_inner_window`, `--m6_inner_stride`, `--m6_profile_max_tokens`. Optional GPT-2 meta checkpoint from `train_mam_meta.py`. |

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
- On **CUDA**, add e.g. `--fp16 --batch_size 8` (or `--bf16` on GPUs that support bfloat16) for faster steps than plain fp32.

## Evaluate M1–M6

**Required:** `test_questions.json` and `test_outputs.json`. The model is run on question-side fields (`input`, `profile`); predictions are aligned by `id` with gold strings from the outputs file for BLEU, ROUGE, and METEOR (via LaMP `generation_metrics`). Predictions are also written as **`pred_outputs.json`** in the same schema as the gold outputs file (`task` + `golds` with `id` / `output`, tab-indented). With multiple `--modes` in one run, files are named **`pred_outputs_<mode>.json`** so nothing is overwritten.

For **M4/M5/M6**, test rows are grouped by user (`user_id`-style fields, `--user_field`, or profile fingerprint) for per-user adaptation. **M4/M5** decode with the same string as **M1** (LaMP `input` only); the profile is used only during TTT updates, not as a retrieved prompt at generation time—so you can contrast **M3 (LoRA + RAG at decode)** with **M4/M5 (LoRA adaptation + no retrieval at decode)**. The M4 checkpoint still comes from `train.py` (RAG-supervised); that is a deliberate train/decode asymmetry unless you add a separate non-RAG training stage. **M6** also decodes on **`input` only**; the profile drives inner-loop updates only.

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

### M6 — TTT-E2E on LaMP-5 / LaMP-7 (GPT-2 + MAM stack)

This is the **end-to-end TTT** setup merged from [MAM](https://github.com/evanly-gh/MAM): **DualMLP** (inner fast weights + static anchor), **test-time inner** = **one** left-to-right pass of sliding-window next-token CE on the user’s **profile-only** text (LaMP-5: titles/abstracts; LaMP-7: past tweets), matching the paper’s single stream over the context (§2.3). **Meta-training outer loop** (`higher` in `ttt/mam_outer.py`) is optional and separate from that single pass.

The benchmark **query** (`input`: instruction + text to personalize) is **never** used in TTT—only at generation—so you are not leaking the test prompt into adaptation. LaMP-7 uses **tweet-only** profile streams for inner TTT (see `ttt/training.py` for pseudo-task construction used by M4/M5 LoRA TTT).

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

**Step 2 — Evaluate M6 on LaMP-5 or LaMP-7 test JSON**

Inner adaptation runs **per user** on the merged profile; inner weights are **reset** between users. Pass the meta checkpoint from Step 1 if you ran it; omit `--m6_mam_checkpoint` to run **inner-only** TTT on top of pretrained GPT-2 with the same `TTTGPT2` architecture.

```bash
python3 run_evaluate.py --task LaMP-5 \
  --base_model gpt2 \
  --architecture causal_lm \
  --modes m6 \
  --test_questions_json path/to/test_questions.json \
  --test_outputs_json path/to/test_outputs.json \
  --m6_mam_checkpoint path/to/mam_ckpts/latest.pt \
  --output_dir path/to/eval_out/m6_ttt_e2e
```

LaMP-7 example: set `--task LaMP-7` and LaMP-7 test paths. Tweet histories are often shorter than paper abstracts; try a **smaller** inner window if you want more windows per profile, e.g. `--m6_inner_window 128 --m6_inner_stride 64`.

Omit `--m6_mam_checkpoint` if you did not run Step 1. Tune **`--ttt_lr`** and sliding-window flags for adaptation quality.

**Speed / paper alignment:** TTT-E2E at test time **streams the context once** (§2.3, [paper](https://arxiv.org/abs/2512.23675)). **GPT-2 M6:** ``inner_adapt_inplace`` — one SGD step per window. **Flan M6:** ``t5_sliding_ttt`` — same idea on the profile stream. Cost per user is **~(#windows)**; widen **`--m6_inner_stride`** or shrink **`--m6_inner_window`** to reduce that. ``--ttt_steps`` affects **M4/M5** only, not M6 inner loops. The official JAX stack is faster.

**M6 on Flan-T5 (paper-style sliding inner)**

With **`--base_model google/flan-t5-small`** and **`--architecture seq2seq`** (or **`auto`**), **`--modes m6`** uses **`ttt/t5_sliding_ttt.py`**: **one** left-to-right pass over the (truncated) profile stream, **one SGD step per sliding window** on the last-fraction **encoder+decoder FFN** weights. Same **`--m6_inner_window`**, **`--m6_inner_stride`**, and **`--m6_profile_max_tokens`** as causal GPT-2 M6.

### All modes (M1–M6)

Flan-T5 line (M6 = **`t5_sliding_ttt`**; run GPT-2 M6 in a **separate** command with `--architecture causal_lm`):

```bash
python3 run_evaluate.py --task LaMP-5 \
  --test_questions_json path/to/test_questions.json \
  --test_outputs_json path/to/test_outputs.json \
  --modes m1,m2,m3,m4,m5,m6 --adapter_dir path/to/checkpoints/lamp5_global_lora \
  --output_dir path/to/eval_out/all
```

`--adapter_dir` is **required** whenever `m3` or `m4` appears in `--modes` (including this all-modes line). It is unused for `m1`, `m2`, `m5`, and **`m6` in seq2seq mode** but may still be passed.

**Note:** You cannot mix **causal** M6 (`gpt2` + `TTTGPT2`) with **Flan** M1–M5 in a single `run_evaluate.py` invocation, because `--base_model` and `--architecture` apply to the whole run. Run **TTT-E2E (GPT-2 M6)** as a separate command using the **M6 — TTT-E2E on LaMP-5 / LaMP-7** steps above.

## Common `run_evaluate.py` flags

| Flag | Purpose |
|------|---------|
| `--base_model` | Default `google/flan-t5-small`. For **TTT-E2E M6**, use a GPT-2 Hub id (e.g. `gpt2`). |
| `--architecture` | `auto` (default), `seq2seq`, or `causal_lm`. **Causal** is required for GPT-2 M6 (`auto` selects it when `base_model` contains `gpt2`). |
| `--num_retrieved`, `--retriever`, `--ranked` | RAG; align with training / LaMP ranking. |
| `--max_input_length`, `--max_new_tokens`, `--batch_size` | Generation / batching; on GPU prefer larger `--batch_size` with `--fp16` or `--bf16`. **Causal LM** (`gpt2`, M1/M6): LaMP-5 decode treats **newline as an extra EOS** (stop after one line of title) plus a **``max_new_tokens`` ceiling** (64); LaMP-7 caps at 96 new tokens. Light post-processing only strips echoed ``Title:`` / whitespace. |
| `--fp16`, `--bf16` | CUDA half-precision inference (bf16 when the GPU supports it). **GPT-2 M6 (`TTTGPT2`)** loads in fp32 for stable inner steps. |
| `--ttt_steps`, `--ttt_lr` | M4/M5: TTT minibatch count and LR. **Causal M6:** ``ttt_lr`` only (inner SGD); ``ttt_steps`` ignored for inner TTT (single pass per paper). |
| `--m6_mam_checkpoint` | **Causal M6 only:** `latest.pt` (or other) from `train_mam_meta.py`. |
| `--m6_inner_window`, `--m6_inner_stride` | **M6** sliding inner (Flan-T5 and GPT-2): window size and stride. For **GPT-2**, window must be ≤ `n_positions` (e.g. **1024**). Larger stride ⇒ fewer windows. |
| `--m6_profile_max_tokens` | **M6 (Flan + GPT-2):** tokenizer cap on the **merged profile** before inner sliding TTT. If unset, defaults to `min(4096, 8 × max_input_length)`. |
| `--user_field` | JSON field for user id when grouping test rows (M4/M5/**M6**). |
| `--cache_dir` | Hugging Face cache directory. |
| `--max_users` | If set to `K` (>0), only rows belonging to the **first K distinct users** (in merged test file order) are evaluated—handy for debugging without the full split. |
| `--verbose`, `--verbose_max_samples` | Print per-example inputs, profile counts, encoder preview, preds vs gold, and per-row BLEU/ROUGE/METEOR (cap rows with `verbose_max_samples`, `-1` = all). |

LaMP data and papers: `LaMP/README.md` (inside your clone) and [lamp-benchmark.github.io](https://lamp-benchmark.github.io/).
