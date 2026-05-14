# lamp

LaMP-5 / LaMP-7 and **SD-tooluse** / **SD-science** at the repo root. Run commands from this directory. The script is `run_evaluate.py` (not `evaluate.py`) so LaMP’s `import evaluate` resolves to HuggingFace metrics.

---

## Layout

| Path | Role |
|------|------|
| `train.py` | Global LoRA: `--prompt_style icl` (M2) or `rag` (M3). |
| `run_evaluate.py` | Eval **M1–M4** + metrics + `pred_outputs*.json`. |
| `train_mam_meta.py` | Optional GPT-2 M4 **meta-train** (`ttt/gpt2_outer.py`). |
| `train_flan_meta.py` | Optional Flan M4 **meta-train**. |
| `requirements.txt` | Deps (includes `higher` for meta-training). |
| `data/` | `data_io.py` — merge Q/O, user keys. |
| `ttt/` | Flan M4 (`flan_*.py`, `e2e.py`); GPT-2 M4 (`gpt2_*.py`). |
| `util/` | LaMP path, LoRA, prompts, SD export + metrics. |
| `Self-Distillation/` | Submodule → **SD** tasks (see below). |
| `LaMP/` | Submodule — upstream prompts, metrics, `LaMP/README.md`. |

---

## Setup

```bash
python3 -m pip install -r requirements.txt
git submodule update --init --recursive
```

Weights (e.g. `google/flan-t5-small`) download from HuggingFace on first use; optional `--cache_dir` or `HF_HOME`.

---

## Data

### LaMP

Download from [lamp-benchmark.github.io](https://lamp-benchmark.github.io/download). You need paired **questions** (`id`, `input`, `profile`) and **outputs** (`id`, `output`) per split. Details: `LaMP/README.md`.

### Self-Distillation (SD-tooluse / SD-science)

```bash
git submodule update --init Self-Distillation
python3 -m util.sd_self_distill --subtask tooluse --split train --out_dir data/sd_self_distill
python3 -m util.sd_self_distill --subtask tooluse --split eval  --out_dir data/sd_self_distill
python3 -m util.sd_self_distill --subtask science --split train --out_dir data/sd_self_distill
python3 -m util.sd_self_distill --subtask science --split eval  --out_dir data/sd_self_distill
```

Writes `*_questions.json` + `*_outputs.json`. Use `--task SD-tooluse` or `SD-science` in train/eval.

---

## Modes (eval)

| Mode | What |
|------|------|
| **M1** | `input` only (no profile). |
| **M2** | ICL: profile + instance in encoder; optional `--adapter_dir` (train with `train.py --prompt_style icl`). |
| **M3** | RAG over profile + optional LoRA (`--adapter_dir` from `train.py --prompt_style rag`). |
| **M4** | Test-time inner loop on profile, then generate from `input`. **Flan:** `seq2seq` + `TTTFlanT5`. **GPT-2:** `causal_lm` + `TTTGPT2`. Optional `--m4_checkpoint` from `train_flan_meta.py` / `train_mam_meta.py`. |

**M4 + RAG:** add `--m4_use_rag` and same `--num_retrieved` / `--retriever` / `--ranked` as M3.  
**Cannot** mix Flan M1–M3 with GPT-2 M4 in one run — use two commands (different `--base_model` / `--architecture`).

---

## Examples — `train.py`

```bash
# LaMP-5 global LoRA (M3-style prompts; add --prompt_style icl for M2)
python3 train.py --task LaMP-5 \
  --train_questions_json path/to/train_questions.json \
  --train_outputs_json path/to/train_outputs.json \
  --output_dir path/to/lora_ckpt

# + dev (epoch eval + load_best)
python3 train.py --task LaMP-5 \
  --train_questions_json path/to/train_questions.json \
  --train_outputs_json path/to/train_outputs.json \
  --dev_questions_json path/to/dev_questions.json \
  --dev_outputs_json path/to/dev_outputs.json \
  --output_dir path/to/lora_ckpt

# SD (same flags; paths from sd_self_distill export)
python3 train.py --task SD-tooluse \
  --train_questions_json data/sd_self_distill/tooluse_train_questions.json \
  --train_outputs_json data/sd_self_distill/tooluse_train_outputs.json \
  --output_dir exps/sd_lora
```

Use `--task LaMP-7` for tweet task. Pass `--output_dir` as `--adapter_dir` in eval for M2/M3.

---

## Examples — meta-train (optional M4)

```bash
# GPT-2 outer meta (LaMP)
python3 train_mam_meta.py --task LaMP-5 \
  --train_questions_json path/to/train_questions.json \
  --train_outputs_json path/to/train_outputs.json \
  --output_dir path/to/gpt2_meta --model_name gpt2 --meta_steps 500 --ckpt_every 100

# Flan outer meta
python3 train_flan_meta.py --task LaMP-5 \
  --train_questions_json path/to/train_questions.json \
  --train_outputs_json path/to/train_outputs.json \
  --output_dir path/to/flan_meta --model_name google/flan-t5-small --meta_steps 500 --ckpt_every 100
```

---

## Examples — `run_evaluate.py`

Replace `path/to/test_*.json` with your files. **`--test_questions_json`** + **`--test_outputs_json`** always required.

```bash
# M1
python3 run_evaluate.py --task LaMP-5 --modes m1 \
  --test_questions_json path/to/test_questions.json \
  --test_outputs_json path/to/test_outputs.json \
  --output_dir exps/eval_m1

# M2 (+ optional LoRA)
python3 run_evaluate.py --task LaMP-5 --modes m2 \
  --test_questions_json path/to/test_questions.json \
  --test_outputs_json path/to/test_outputs.json \
  --output_dir exps/eval_m2
# --adapter_dir path/to/lora_ckpt

# M3 (+ optional LoRA)
python3 run_evaluate.py --task LaMP-5 --modes m3 \
  --test_questions_json path/to/test_questions.json \
  --test_outputs_json path/to/test_outputs.json \
  --output_dir exps/eval_m3
# --adapter_dir path/to/lora_ckpt

# M4 Flan (all modes one run; adapter only affects m2/m3)
python3 run_evaluate.py --task LaMP-5 \
  --modes m1,m2,m3,m4 --base_model google/flan-t5-small --architecture seq2seq \
  --test_questions_json path/to/test_questions.json \
  --test_outputs_json path/to/test_outputs.json \
  --output_dir exps/eval_all
# --m4_checkpoint path/to/flan_meta/latest.pt

# M4 GPT-2 (separate command)
python3 run_evaluate.py --task LaMP-5 --modes m4 \
  --base_model gpt2 --architecture causal_lm \
  --test_questions_json path/to/test_questions.json \
  --test_outputs_json path/to/test_outputs.json \
  --output_dir exps/eval_m4_gpt2
# --m4_checkpoint path/to/gpt2_meta/latest.pt
```

SD eval example:

```bash
python3 run_evaluate.py --task SD-tooluse --modes m1,m4 \
  --test_questions_json data/sd_self_distill/tooluse_test_questions.json \
  --test_outputs_json data/sd_self_distill/tooluse_test_outputs.json \
  --base_model google/flan-t5-small \
  --m4_checkpoint exps/flan_meta_sd/latest.pt \
  --max_new_tokens 512 --output_dir exps/eval_sd
```

---

## Common flags (`run_evaluate.py`)

| Flag | Notes |
|------|--------|
| `--task` | `LaMP-5`, `LaMP-7`, `SD-tooluse`, `SD-science`. |
| `--modes` | `m1`, `m2`, `m3`, `m4` (comma list). |
| `--base_model` | Hub id; default Flan-small. GPT-2 for causal M4. |
| `--architecture` | `auto` / `seq2seq` / `causal_lm` (GPT-2 M4 needs causal). |
| `--adapter_dir` | LoRA dir from `train.py` — **M2/M3** only. |
| `--num_retrieved`, `--retriever`, `--ranked` | M3 / M4+RAG; match training. |
| `--max_input_length`, `--max_new_tokens`, `--batch_size` | Encode / decode / batch. |
| `--fp16`, `--bf16` | CUDA half/bfloat. |
| `--m4_checkpoint` | `latest.pt` from `train_flan_meta.py` or `train_mam_meta.py`. |
| `--m4_inner_window`, `--m4_inner_stride`, `--m4_profile_max_tokens` | M4 sliding inner. |
| `--m4_use_rag` | M4 with profile retrieval (same retriever flags as M3). |
| `--ttt_lr` | M4 inner SGD lr. |
| `--user_field` | User id column for M4 grouping. |
| `--cache_dir` | HF cache (models + Contriever if used). |
| `--verbose`, `--verbose_max_samples` | Print examples (slow if many). |
| `--save_encoder_prompts` | Write `encoder_prompts_<mode>.json` per mode. |

More: `--max_users`, `--sd_save_responses`, `--m4_ttt_fraction` — see `python run_evaluate.py -h`.
