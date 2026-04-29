"""
Evaluate personalization baselines on the LaMP **test** split: ``test_questions.json``
(inputs + profiles) and ``test_outputs.json`` (gold labels by ``id``), per the
[LaMP download](https://lamp-benchmark.github.io/download) and ``LaMP/README.md``.
Rows are aligned by ``id`` (same idea as ``LaMP/LaMP/utils/merge_with_rank.py`` without ranking).
The model only sees question-side fields; predictions are scored against the outputs file
(BLEU, ROUGE, METEOR via LaMP metrics). Writes **``pred_outputs.json``** (LaMP gold format:
``{"task": "LaMP_5"|"LaMP_7", "golds": [{"id", "output"}, ...]}`` with tab-indented JSON;
multiple ``--modes`` in one run use ``pred_outputs_<mode>.json``).

This script is named ``run_evaluate.py`` (not ``evaluate.py``) so LaMP's metric code can
``import evaluate`` and resolve the HuggingFace **evaluate** library instead of this file.

Models (paper storyboard):
  M1 Zero-shot base (task input only, no profile)
  M2 ICL (history serialized into the encoder budget)
  M3 Global LoRA + RAG (checkpoint from train.py; RAG prompt at decode)
  M4 Global LoRA + TTT (TTT on profile; decode uses **task input only**, no RAG—baseline TTT vs M3 RAG)
  M5 Clean TTT (same decode as M1/M4; fresh LoRA + TTT per user)
  M6 TTT-E2E: seq2seq uses ``ttt/t5_sliding_ttt.py`` (single-pass sliding FFN inner; shared ``--m6_*`` flags with causal GPT-2 M6).
  Causal GPT-2 uses ``ttt/mam_*.py`` (DualMLP + ``inner_adapt_inplace``; optional ``--m6_mam_checkpoint``). No global LoRA.

Metrics follow LaMP/LaMP/metrics/generation_metrics.py (BLEU, ROUGE, METEOR).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from typing import Callable

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

from util.lamp_paths import ensure_lamp_on_path

ensure_lamp_on_path()

from prompts.prompts import create_prompt_generator  # noqa: E402

_DATA_DIR = os.path.join(_ROOT, "data")
if _DATA_DIR not in sys.path:
    sys.path.append(_DATA_DIR)
import data_io  # noqa: E402

from ttt.training import run_ttt_steps  # noqa: E402
from util import metrics_eval, modeling_lora, prompting  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["LaMP-5", "LaMP-7"], required=True)
    p.add_argument(
        "--test_questions_json",
        required=True,
        help="LaMP test inputs + profiles (e.g. test_questions.json from the benchmark).",
    )
    p.add_argument(
        "--test_outputs_json",
        required=True,
        help="Gold labels keyed by id (e.g. test_outputs.json; list or {\"task\", \"golds\"}).",
    )
    p.add_argument(
        "--base_model",
        default="google/flan-t5-small",
        help="HF hub id, e.g. google/flan-t5-small or openai-community/gpt2-large.",
    )
    p.add_argument(
        "--architecture",
        choices=["auto", "seq2seq", "causal_lm"],
        default="auto",
        help="auto: use causal LM if base_model id contains 'gpt2'; else seq2seq (T5). "
        "Causal LM is supported for m1 and m6 only.",
    )
    p.add_argument("--adapter_dir", default=None, help="Checkpoint directory from train.py (M3/M4).")
    p.add_argument(
        "--modes",
        default="m1,m2,m3,m4,m5",
        help="Comma list among m1..m6 (m6 = TTT-E2E-style inner: Flan ``t5_sliding_ttt`` or GPT-2 ``mam_inner``; base model).",
    )
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--num_retrieved", type=int, default=3)
    p.add_argument("--retriever", default="bm25", choices=["bm25", "random", "recency"])
    p.add_argument("--ranked", action="store_true")
    p.add_argument("--max_input_length", type=int, default=512)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Generation batch size; on GPU try 16–32 with --fp16/--bf16 for higher throughput.",
    )
    p.add_argument(
        "--fp16",
        action="store_true",
        help="On CUDA: load the seq2seq model in float16 (usually faster and less VRAM than fp32).",
    )
    p.add_argument(
        "--bf16",
        action="store_true",
        help="On CUDA: use bfloat16 when supported (often best on Ampere+). Incompatible with --fp16.",
    )
    p.add_argument(
        "--ttt_steps",
        type=int,
        default=30,
        help="M4/M5: number of TTT minibatches. M6 (causal or seq2seq): **unused** for inner TTT (single sliding pass; use ``--m6_inner_window`` / ``--m6_inner_stride``).",
    )
    p.add_argument("--ttt_lr", type=float, default=1e-4)
    p.add_argument(
        "--m6_mam_checkpoint",
        default=None,
        help="For causal m6 only: optional ``.pt`` state_dict from ``train_mam_meta.py`` (TTTGPT2).",
    )
    p.add_argument(
        "--m6_inner_window",
        type=int,
        default=256,
        help="M6 sliding inner: token window size. **Causal (GPT-2):** must be ≤ ``n_positions`` (1024). **Seq2seq (Flan-T5):** ``ttt/t5_sliding_ttt.py`` profile pass.",
    )
    p.add_argument(
        "--m6_inner_stride",
        type=int,
        default=128,
        help="M6 sliding inner: stride between windows (seq2seq Flan path and causal GPT-2 path).",
    )
    p.add_argument(
        "--m6_profile_max_tokens",
        type=int,
        default=None,
        help="M6 (causal GPT-2 and seq2seq Flan-T5): max tokens for the merged **profile** stream before sliding inner TTT. "
        "If unset, uses min(4096, 8 × --max_input_length). Sliding uses ``--m6_inner_window`` / ``--m6_inner_stride``.",
    )
    p.add_argument("--user_field", default=None)
    p.add_argument(
        "--max_users",
        type=int,
        default=None,
        help="If set (>0), only evaluate rows whose user is among the first K distinct users "
        "(order = first appearance in the merged test file). All rows for those users are kept. "
        "Useful for quick debugging without scanning the full split.",
    )
    p.add_argument("--output_dir", default="eval_outputs")
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-example task input, profile size, encoder prompt preview, pred, gold, and string metrics.",
    )
    p.add_argument(
        "--verbose_max_samples",
        type=int,
        default=40,
        help="With --verbose, max rows to print per mode (-1 = all; can be slow on large test sets).",
    )
    return p.parse_args()


def task_leaderboard_name(task: str) -> str:
    return task.replace("-", "_")


def _clip_text(s: str, max_chars: int) -> str:
    t = s.replace("\n", " ").strip()
    if len(t) <= max_chars:
        return t
    return t[: max_chars - 3] + "..."


def _encoder_source_for_mode(
    mode: str,
    row: dict,
    *,
    task: str,
    tokenizer,
    max_in: int,
    rag_prompt: Callable[[dict], str],
) -> str:
    """Same text the model sees (pre-tokenization) as in ``run_for_mode``."""
    if mode in ("m1", "m4", "m5", "m6"):
        return row["input"]
    if mode == "m2":
        return prompting.build_icl_source(
            row, tokenizer, task=task, max_tokens=max_in, reserve_for_input=128
        )
    return rag_prompt(row)


def _verbose_report_mode(
    mode: str,
    *,
    task: str,
    tokenizer,
    max_in: int,
    rag_prompt: Callable[[dict], str],
    id_order: list,
    rows: list[dict],
    refs: list[str],
    pred_map: dict,
    corpus_metrics: dict[str, float],
    max_samples: int,
) -> None:
    n = len(id_order)
    limit = n if max_samples < 0 else min(n, max_samples)
    score_one = metrics_eval.make_per_example_string_metric()
    input_label = (
        "LaMP task input (instruction + paper abstract)"
        if task == "LaMP-5"
        else "LaMP task input (instruction + target tweet)"
    )
    print(
        f"\n{'=' * 72}\n[verbose] mode={mode}  task={task}  "
        f"printing {limit} of {n} examples  corpus_metrics={corpus_metrics}\n{'=' * 72}"
    )
    for i in range(limit):
        rid = id_order[i]
        row = rows[i]
        ref = refs[i]
        pred = pred_map[rid]
        prof = row.get("profile") or []
        enc_src = _encoder_source_for_mode(
            mode, row, task=task, tokenizer=tokenizer, max_in=max_in, rag_prompt=rag_prompt
        )
        per_ex = score_one(pred, ref)
        print(f"\n--- sample index={i}  id={rid!r} ---")
        print(f"  profile_items (history size): {len(prof)}")
        if data_io.looks_like_file_id_placeholder(str(row.get("input", ""))):
            print(
                "  NOTE: `input` looks like a corpus **file id** (e.g. *.txt), not a full LaMP prompt + abstract."
            )
        if data_io.looks_like_file_id_placeholder(str(ref)):
            print("  NOTE: gold `output` looks like a file id, not a real title string.")
        if task == "LaMP-5" and prof:
            p0 = prof[0] if isinstance(prof[0], dict) else {}
            ab = (p0.get("abstract") or p0.get("title") or "") if isinstance(p0, dict) else ""
            if isinstance(ab, str) and ab.strip():
                print(f"  first_profile title/abstract preview: {_clip_text(ab, 240)}")
        print(f"  {input_label} (preview): {_clip_text(row.get('input', ''), 420)}")
        print(f"  encoder_source preview ({mode}): {_clip_text(enc_src, 520)}")
        print(f"  gold_output preview: {_clip_text(ref, 320)}")
        print(f"  prediction preview: {_clip_text(pred, 320)}")
        print(f"  per_example_string_metrics: {per_ex}")
    if limit < n:
        print(f"\n[verbose] ... omitted {n - limit} further examples (see --verbose_max_samples).\n")


def _m6_profile_token_cap(max_in: int, m6_profile_max_tokens: int | None) -> int:
    """Upper bound on profile stream length (tokens) before sliding-window inner TTT."""
    if m6_profile_max_tokens is not None:
        return max(1, m6_profile_max_tokens)
    return min(4096, max_in * 8)


# LaMP-5 ``input`` already asks for a title, but raw GPT-2 does not learn a reliable ``<EOS>``
# after titles. We treat the first generated newline as end-of-title (same convention as many
# completion APIs): ``eos_token_id`` includes ``\\n`` so decoding stops there—no word-level clip.
# ``max_new_tokens`` is still a safety ceiling if the model never emits ``\\n``.
_CAUSAL_TITLE_MAX_NEW = 64
_CAUSAL_TWEET_MAX_NEW = 96


def _causal_decode_max_new_tokens(task: str, requested_max_new: int) -> int:
    if task == "LaMP-5":
        return max(8, min(requested_max_new, _CAUSAL_TITLE_MAX_NEW))
    if task == "LaMP-7":
        return max(8, min(requested_max_new, _CAUSAL_TWEET_MAX_NEW))
    return requested_max_new


def _lamp5_title_eos_token_ids(tokenizer) -> list[int]:
    """EOS ids for LaMP-5 title decode: model EOS plus newline (one line of title).

    Do **not** put ``pad_token_id`` here when it equals ``eos_token_id`` (common for GPT-2):
    the first “real” logit step can map to EOS/pad and end generation with **zero** visible
    title tokens. Newline still ends the line once at least one content token is forced via
    ``min_new_tokens`` in ``generate``.
    """
    out: list[int] = []
    if tokenizer.eos_token_id is not None:
        out.append(int(tokenizer.eos_token_id))
    for tid in tokenizer.encode("\n", add_special_tokens=False):
        tt = int(tid)
        if tt not in out:
            out.append(tt)
    return out


def _postprocess_causal_generation(task: str, text: str) -> str:
    """Light cleanup after decode (newline stopping does the heavy lifting for LaMP-5 titles)."""
    s = (text or "").strip()
    if not s:
        return s
    first = s.split("\n")[0].strip()
    if task == "LaMP-5":
        first = re.sub(r"^(Title|TITLE)\s*:\s*", "", first).strip()
        first = re.sub(r"\s+", " ", first).strip()
    elif task == "LaMP-7":
        first = re.sub(r"^(Tweet|TWEET)\s*:\s*", "", first).strip()
        first = re.sub(r"\s+", " ", first).strip()
    return first


def _restrict_to_first_k_users(
    merged: list[dict[str, object]],
    rows: list[dict[str, object]],
    refs: list[str],
    id_order: list[object],
    user_field: str | None,
    k: int,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[str],
    list[object],
    dict[str, list[dict[str, object]]],
]:
    """
    Keep every test row whose ``infer_user_id`` is one of the first ``k`` distinct user ids
    encountered when scanning ``rows`` in order (then rebuild ``user_to_rows``).
    """
    uid_order: list[str] = []
    seen: set[str] = set()
    for r in rows:
        uid = data_io.infer_user_id(r, user_field=user_field)
        if uid not in seen:
            seen.add(uid)
            uid_order.append(uid)
            if len(uid_order) >= k:
                break
    keep = set(uid_order)
    idx_kept = [
        i for i, r in enumerate(rows) if data_io.infer_user_id(r, user_field=user_field) in keep
    ]
    merged_f = [merged[i] for i in idx_kept]
    rows_f = [rows[i] for i in idx_kept]
    refs_f = [refs[i] for i in idx_kept]
    id_order_f = [id_order[i] for i in idx_kept]
    user_to_rows: dict[str, list[dict[str, object]]] = defaultdict(list)
    for r in rows_f:
        uid = data_io.infer_user_id(r, user_field=user_field)
        user_to_rows[uid].append(r)
    return merged_f, rows_f, refs_f, id_order_f, dict(user_to_rows)


def merge_profiles(rows: list[dict]) -> list[dict]:
    seen: set[str] = set()
    merged: list[dict] = []
    for r in rows:
        for item in r.get("profile") or []:
            key = str(item.get("id")) if item.get("id") is not None else json.dumps(item, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
    return merged


def _infer_torch_dtype(device: torch.device, *, want_fp16: bool, want_bf16: bool) -> torch.dtype | None:
    """Return ``torch_dtype`` for ``from_pretrained``, or ``None`` for default fp32."""
    if device.type != "cuda":
        if want_fp16 or want_bf16:
            print(
                "[run_evaluate] --fp16/--bf16 apply on CUDA only; running weights in fp32 on CPU.",
                file=sys.stderr,
            )
        return None
    if want_bf16:
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        print("[run_evaluate] --bf16 not supported on this GPU; using fp32.", file=sys.stderr)
        return None
    if want_fp16:
        return torch.float16
    return None


def resolved_architecture(base_model: str, architecture: str) -> str:
    if architecture == "causal_lm":
        return "causal_lm"
    if architecture == "seq2seq":
        return "seq2seq"
    if architecture == "auto":
        return "causal_lm" if "gpt2" in base_model.lower() else "seq2seq"
    raise ValueError(architecture)


@torch.inference_mode()
def batched_generate(model, tokenizer, sources: list[str], device: torch.device, max_in: int, max_new: int):
    enc = tokenizer(
        sources,
        truncation=True,
        max_length=max_in,
        padding=True,
        return_tensors="pt",
    ).to(device)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    out_ids = model.generate(
        **enc,
        max_new_tokens=max_new,
        pad_token_id=pad_id,
    )
    return tokenizer.batch_decode(out_ids, skip_special_tokens=True)


@torch.inference_mode()
def batched_generate_causal(
    model,
    tokenizer,
    sources: list[str],
    device: torch.device,
    max_in: int,
    max_new: int,
    *,
    task: str | None = None,
    repetition_penalty: float | None = None,
):
    enc = tokenizer(
        sources,
        truncation=True,
        max_length=max_in,
        padding=True,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    gen_kw: dict = {"max_new_tokens": max_new, "pad_token_id": pad_id}
    if repetition_penalty is not None and repetition_penalty > 1.0:
        gen_kw["repetition_penalty"] = repetition_penalty
    if task == "LaMP-5":
        gen_kw["eos_token_id"] = _lamp5_title_eos_token_ids(tokenizer)
        # Without this, the first sampled token can be ``\\n`` (in eos set) → immediate stop → "".
        gen_kw["min_new_tokens"] = 1
    out_ids = model.generate(**enc, **gen_kw)
    # HF returns [batch, prompt_padded_len + new_len]. New tokens always start *after* the
    # padded prompt width (same for every row). Using per-row attention_mask.sum() is wrong
    # for right-padded batches: it decodes pad ids as garbage and mis-aligns continuations.
    prompt_w = enc["input_ids"].shape[1]
    decoded: list[str] = []
    for i in range(len(sources)):
        new_part = out_ids[i, prompt_w:]
        decoded.append(tokenizer.decode(new_part, skip_special_tokens=True).strip())
    return decoded


def run_for_mode(
    mode: str,
    rows: list[dict],
    user_to_rows: dict[str, list[dict]],
    *,
    task: str,
    tokenizer,
    base_model_name: str,
    adapter_dir: str | None,
    cache_dir: str | None,
    device: torch.device,
    rag_prompt,
    max_in: int,
    max_new: int,
    batch_size: int,
    ttt_steps: int,
    ttt_lr: float,
    torch_dtype: torch.dtype | None,
    architecture: str = "seq2seq",
    m6_mam_checkpoint: str | None = None,
    m6_inner_window: int = 256,
    m6_inner_stride: int = 128,
    m6_profile_max_tokens: int | None = None,
) -> list[tuple[str, str]]:
    load_kw: dict = {"cache_dir": cache_dir}
    if torch_dtype is not None:
        load_kw["torch_dtype"] = torch_dtype

    if architecture == "causal_lm":
        if mode not in ("m1", "m6"):
            raise ValueError(
                f"Causal LM (--architecture causal_lm or a gpt2 base_model) supports m1 and m6 only; got {mode=!r}."
            )
        if mode == "m6":
            from ttt.mam_model import TTTGPT2

            load_plain = {k: v for k, v in load_kw.items() if k != "torch_dtype"}
            model = TTTGPT2(base_model_name, ttt_fraction=0.25)
            if m6_mam_checkpoint:
                try:
                    sd = torch.load(m6_mam_checkpoint, map_location="cpu", weights_only=False)
                except TypeError:
                    sd = torch.load(m6_mam_checkpoint, map_location="cpu")
                model.load_state_dict(sd, strict=True)
            model = model.to(device)
            max_pos = int(
                getattr(model.lm.config, "n_positions", None)
                or getattr(model.lm.config, "max_position_embeddings", 1024)
            )
            if m6_inner_window > max_pos:
                raise ValueError(
                    f"--m6_inner_window ({m6_inner_window}) exceeds model max positions ({max_pos}); "
                    f"GPT-2-style LMs are trained with that context cap per forward."
                )
        else:
            model = AutoModelForCausalLM.from_pretrained(base_model_name, **load_kw)
    else:
        base = AutoModelForSeq2SeqLM.from_pretrained(base_model_name, **load_kw)
        if mode in ("m3", "m4"):
            if not adapter_dir:
                raise ValueError(f"{mode} requires --adapter_dir (global LoRA+RAG checkpoint).")
            model = PeftModel.from_pretrained(base, adapter_dir)
        elif mode == "m5":
            model = modeling_lora.attach_lora(base)
        else:
            model = base

    model.to(device)
    model.eval()

    preds: list[tuple[str, str]] = []

    def handle_batch(sources: list[str], meta_ids: list[str]):
        gen_tok = getattr(model, "tokenizer", tokenizer)
        if architecture == "causal_lm":
            cap_new = _causal_decode_max_new_tokens(task, max_new)
            rep = 1.15 if task in ("LaMP-5", "LaMP-7") else None
            decoded = batched_generate_causal(
                model,
                gen_tok,
                sources,
                device,
                max_in,
                cap_new,
                task=task,
                repetition_penalty=rep,
            )
            decoded = [_postprocess_causal_generation(task, d) for d in decoded]
        else:
            decoded = batched_generate(model, tokenizer, sources, device, max_in, max_new)
        preds.extend(zip(meta_ids, decoded))

    if mode in ("m1", "m2", "m3"):
        batch_src: list[str] = []
        batch_ids: list[str] = []
        for row in tqdm(rows, desc=mode):
            if mode == "m1":
                src = row["input"]
            elif mode == "m2":
                src = prompting.build_icl_source(
                    row, tokenizer, task=task, max_tokens=max_in, reserve_for_input=128
                )
            else:
                src = rag_prompt(row)
            batch_src.append(src)
            batch_ids.append(row["id"])
            if len(batch_src) >= batch_size:
                handle_batch(batch_src, batch_ids)
                batch_src, batch_ids = [], []
        if batch_src:
            handle_batch(batch_src, batch_ids)
        return preds

    if mode in ("m4", "m5"):
        # Snapshot LoRA (global for M4, freshly initialized for M5), never leak across users.
        snapshot = modeling_lora.lora_state_snapshot(model)
        for _user, urows in tqdm(list(user_to_rows.items()), desc=mode):
            modeling_lora.restore_lora_snapshot(model, snapshot)
            for name, param in model.named_parameters():
                if "lora_" in name:
                    param.requires_grad = True
            prof = merge_profiles(urows)
            run_ttt_steps(
                model,
                tokenizer,
                task=task,
                profile=prof,
                device=device,
                max_input_length=max_in,
                micro_batch_size=2,
                steps=ttt_steps,
                lr=ttt_lr,
            )
            model.eval()

            batch_src, batch_ids = [], []
            for row in urows:
                # TTT uses profile in run_ttt_steps; decode matches M1 (no RAG) to separate TTT vs retrieval.
                batch_src.append(row["input"])
                batch_ids.append(row["id"])
                if len(batch_src) >= batch_size:
                    handle_batch(batch_src, batch_ids)
                    batch_src, batch_ids = [], []
            if batch_src:
                handle_batch(batch_src, batch_ids)

            modeling_lora.restore_lora_snapshot(model, snapshot)

        return preds

    if mode == "m6":
        if architecture == "causal_lm":
            from ttt import e2e as ttt_e2e
            from ttt.mam_inner import inner_adapt_inplace

            for _user, urows in tqdm(list(user_to_rows.items()), desc=mode):
                snap = model.snapshot_inner()
                try:
                    prof = merge_profiles(urows)
                    stream = ttt_e2e.build_flat_history_stream(task, prof)
                    gen_tok = model.tokenizer
                    prof_cap = _m6_profile_token_cap(max_in, m6_profile_max_tokens)
                    enc = gen_tok(
                        stream,
                        return_tensors="pt",
                        truncation=True,
                        max_length=prof_cap,
                    )
                    ctx_ids = enc["input_ids"].to(device)
                    if ctx_ids.shape[1] >= 2:
                        inner_adapt_inplace(
                            model,
                            ctx_ids,
                            lr=ttt_lr,
                            window=m6_inner_window,
                            stride=m6_inner_stride,
                        )
                    model.eval()
                    batch_src, batch_ids = [], []
                    for row in urows:
                        batch_src.append(row["input"])
                        batch_ids.append(row["id"])
                        if len(batch_src) >= batch_size:
                            handle_batch(batch_src, batch_ids)
                            batch_src, batch_ids = [], []
                    if batch_src:
                        handle_batch(batch_src, batch_ids)
                finally:
                    model.restore_inner(snap)
            return preds

        from ttt import e2e as ttt_e2e
        from ttt.t5_sliding_ttt import inner_adapt_t5_sliding_profile

        ffn = ttt_e2e.collect_inner_mlp_params(model, layer_fraction=0.25)
        if not ffn:
            raise RuntimeError(
                "Could not collect inner MLP parameters for m6 (unexpected model structure for this backbone)."
            )
        ffn_snap = ttt_e2e.snapshot_selected_params(ffn)
        for _user, urows in tqdm(list(user_to_rows.items()), desc=mode):
            ttt_e2e.restore_selected_params(ffn, ffn_snap)
            prof = merge_profiles(urows)
            prof_cap = _m6_profile_token_cap(max_in, m6_profile_max_tokens)
            inner_adapt_t5_sliding_profile(
                model,
                tokenizer,
                task=task,
                profile=prof,
                device=device,
                lr=ttt_lr,
                window=m6_inner_window,
                stride=m6_inner_stride,
                profile_token_cap=prof_cap,
                layer_fraction=0.25,
            )
            model.eval()
            batch_src, batch_ids = [], []
            for row in urows:
                batch_src.append(row["input"])
                batch_ids.append(row["id"])
                if len(batch_src) >= batch_size:
                    handle_batch(batch_src, batch_ids)
                    batch_src, batch_ids = [], []
            if batch_src:
                handle_batch(batch_src, batch_ids)
        ttt_e2e.restore_selected_params(ffn, ffn_snap)
        return preds

    raise ValueError(f"Unsupported mode: {mode!r}")


def main():
    args = parse_args()
    if args.fp16 and args.bf16:
        raise SystemExit("Use at most one of --fp16 and --bf16.")
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    torch_dtype = _infer_torch_dtype(device, want_fp16=args.fp16, want_bf16=args.bf16)

    merged = data_io.merge_questions_and_outputs(
        args.test_questions_json, args.test_outputs_json, task=args.task
    )
    data_io.warn_if_rows_look_like_unexpanded_placeholders(
        merged,
        task=args.task,
        context=f"test: {args.test_questions_json} + {args.test_outputs_json}",
    )
    refs = [r["output"] for r in merged]
    id_order = [r["id"] for r in merged]
    id_for_pred_json = data_io.gold_id_lookup(args.test_outputs_json)
    # Do not pass gold ``output`` into the model forward paths (M1–M5 only use input/profile).
    rows = [{k: v for k, v in r.items() if k != "output"} for r in merged]
    user_to_rows: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        uid = data_io.infer_user_id(r, user_field=args.user_field)
        user_to_rows[uid].append(r)

    if args.max_users is not None and args.max_users > 0:
        n_users_full = len(user_to_rows)
        n_rows_full = len(rows)
        merged, rows, refs, id_order, user_to_rows = _restrict_to_first_k_users(
            merged, rows, refs, id_order, args.user_field, args.max_users
        )
        print(
            f"[run_evaluate] --max_users={args.max_users}: "
            f"{len(user_to_rows)} user(s), {len(rows)} row(s) "
            f"(full split: {n_users_full} user(s), {n_rows_full} row(s)).",
            file=sys.stderr,
        )

    modes = [m.strip().lower() for m in args.modes.split(",") if m.strip()]
    arch = resolved_architecture(args.base_model, args.architecture)
    if arch == "causal_lm" and any(m in ("m2", "m3", "m4", "m5") for m in modes):
        raise ValueError(
            "Causal LM (gpt2-style) is only wired for m1 and m6; use --architecture seq2seq for m2–m5."
        )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, cache_dir=args.cache_dir, use_fast=False)
    if arch == "causal_lm" and tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    rag_gen, contriever = create_prompt_generator(
        args.num_retrieved,
        args.retriever,
        args.ranked,
        args.max_input_length,
        tokenizer,
    )
    if contriever is not None:
        contriever = contriever.to("cpu")

    def rag_prompt(row: dict) -> str:
        return rag_gen(row["input"], row["profile"], args.task)

    results_summary: dict[str, dict] = {}
    for mode in modes:
        if mode in ("m3", "m4") and not args.adapter_dir:
            raise ValueError(f"Mode {mode} requires --adapter_dir")
        pairs = run_for_mode(
            mode,
            rows,
            user_to_rows,
            task=args.task,
            tokenizer=tokenizer,
            base_model_name=args.base_model,
            adapter_dir=args.adapter_dir,
            cache_dir=args.cache_dir,
            device=device,
            rag_prompt=rag_prompt,
            max_in=args.max_input_length,
            max_new=args.max_new_tokens,
            batch_size=args.batch_size,
            ttt_steps=args.ttt_steps,
            ttt_lr=args.ttt_lr,
            torch_dtype=torch_dtype,
            architecture=arch,
            m6_mam_checkpoint=args.m6_mam_checkpoint,
            m6_inner_window=args.m6_inner_window,
            m6_inner_stride=args.m6_inner_stride,
            m6_profile_max_tokens=args.m6_profile_max_tokens,
        )
        pred_map = {i: p for i, p in pairs}
        preds_ordered = [pred_map[i] for i in id_order]
        metrics = metrics_eval.evaluate_strings(preds_ordered, refs)
        results_summary[mode] = metrics
        # LaMP leaderboard format (same as gold ``*_outputs.json``): ``pred_outputs.json``
        # when a single mode; otherwise one file per mode to avoid clobbering.
        pred_filename = "pred_outputs.json" if len(modes) == 1 else f"pred_outputs_{mode}.json"
        out_json = os.path.join(args.output_dir, pred_filename)
        metrics_eval.write_lamp_predictions(
            task_leaderboard_name(args.task),
            [(id_for_pred_json.get(str(i), i), pred_map[i]) for i in id_order],
            out_json,
        )
        with open(os.path.join(args.output_dir, f"metrics_{mode}.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(mode, metrics)
        if args.verbose:
            _verbose_report_mode(
                mode,
                task=args.task,
                tokenizer=tokenizer,
                max_in=args.max_input_length,
                rag_prompt=rag_prompt,
                id_order=id_order,
                rows=rows,
                refs=refs,
                pred_map=pred_map,
                corpus_metrics=metrics,
                max_samples=args.verbose_max_samples,
            )

    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)


if __name__ == "__main__":
    main()
