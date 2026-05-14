"""
Evaluate personalization baselines on the LaMP **test** split: ``test_questions.json``
(inputs + profiles) and ``test_outputs.json`` (gold labels by ``id``), per the
[LaMP download](https://lamp-benchmark.github.io/download) and ``LaMP/README.md``.
Rows are aligned by ``id`` (same idea as ``LaMP/LaMP/utils/merge_with_rank.py`` without ranking).
The model only sees question-side fields; predictions are scored against the outputs file.
**LaMP-5 / LaMP-7:** BLEU, ROUGE, METEOR via LaMP metrics.
**SD-tooluse / SD-science** (Self-Distillation submodule): task accuracy (same rules as
``Self-Distillation/eval_tooluse.py`` and ``eval_science.py``). Writes **``pred_outputs.json``**
(leaderboard-style ``task`` + ``golds`` with tab-indented JSON; multiple ``--modes`` use
``pred_outputs_<mode>.json``).

This script is named ``run_evaluate.py`` (not ``evaluate.py``) so LaMP's metric code can
``import evaluate`` and resolve the HuggingFace **evaluate** library instead of this file.

Models (paper storyboard):
  M1 Zero-shot base (task input only, no profile)
  M2 ICL (history serialized into the encoder budget; optional same LoRA adapter as M3).
  M3 RAG (+ optional LoRA adapter from train.py). **SD-tooluse / SD-science:** ``profile`` is
  many newline-split line rows from the long ``input``; M3 retrieves top‑K rows with ``sd_rag_query``,
  packs them under ``--max_input_length``, then appends ``Task:`` + full ``input`` (truncated
  only if still too long).
  M4 TTT-E2E: seq2seq uses ``ttt/flan_inner.py`` with ``TTTFlanT5`` Dual-FFN wrapper (single-pass sliding inner; shared ``--m4_*`` flags with causal GPT-2 M4). Optional ``--m4_use_rag``: per test row, retrieve top‑K history (M3 retriever flags), run sliding-window inner TTT on that text, then generate from that row's ``input``.
  Causal GPT-2 uses ``ttt/gpt2_*.py`` (DualMLP + ``inner_adapt_inplace``; optional ``--m4_checkpoint``). No global LoRA.

Metrics follow LaMP/LaMP/metrics/generation_metrics.py (BLEU, ROUGE, METEOR).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from typing import Any, Callable

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

from util.lamp_paths import ensure_lamp_on_path

ensure_lamp_on_path()

_DATA_DIR = os.path.join(_ROOT, "data")
if _DATA_DIR not in sys.path:
    sys.path.append(_DATA_DIR)
import data_io  # noqa: E402

from util import metrics_eval, prompting, sd_eval_metrics  # noqa: E402
from util import sd_self_distill  # noqa: E402
from util.encoder_prompt_dump import encoder_source_for_seq2seq_mode, write_encoder_prompts_json  # noqa: E402


def _is_sd_task(task: str) -> bool:
    return task in ("SD-tooluse", "SD-science")


def _apply_sd_truncation_side(tok) -> None:
    """
    Default HF truncation removes the **end** of the sequence. Self-distillation ``input`` is
    long documentation first and the real task at the **tail**; keep the tail when capping
    encoder length (e.g. Flan ``n_positions`` / ``--max_input_length``).
    """
    if hasattr(tok, "truncation_side"):
        tok.truncation_side = "left"


def _m4_rag_query(row: dict, task: str) -> str:
    """Profile retrieval query: short ``sd_rag_query`` on SD rows; full ``input`` on LaMP."""
    if _is_sd_task(task):
        return sd_self_distill.sd_rag_query_for_row(row)
    return row["input"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--task",
        choices=["LaMP-5", "LaMP-7", "SD-tooluse", "SD-science"],
        required=True,
    )
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
        "Causal LM is supported for m1 and m4 only.",
    )
    p.add_argument(
        "--adapter_dir",
        default=None,
        help="Optional LoRA adapter from train.py (**M2** ICL and **M3** RAG). Omit to use the base model. "
        "Ignored for M1 and seq2seq M4.",
    )
    p.add_argument(
        "--modes",
        default="m1,m2,m3,m4",
        help="Comma list among m1,m2,m3,m4 (m4 = TTT-E2E-style inner: Flan ``flan_inner`` or GPT-2 ``gpt2_inner``; base model).",
    )
    p.add_argument(
        "--cache_dir",
        default=None,
        help="Hugging Face hub cache directory (tokenizer, seq2seq weights, Contriever when using --retriever contriever; "
        "also HF ``datasets`` cache where used).",
    )
    p.add_argument("--num_retrieved", type=int, default=3)
    p.add_argument(
        "--retriever",
        default="contriever",
        choices=["contriever", "bm25", "random", "recency"],
    )
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
    p.add_argument("--ttt_lr", type=float, default=1e-4)
    p.add_argument(
        "--m4_checkpoint",
        default=None,
        help="For m4 (causal or seq2seq): optional ``.pt`` state_dict from ``train_mam_meta.py`` or ``train_flan_meta.py``.",
    )
    p.add_argument(
        "--m4_ttt_fraction",
        type=float,
        default=0.25,
        help="M4 only: fraction of final blocks whose FFNs are adapted (Dual-FFN trainable branch).",
    )
    p.add_argument(
        "--m4_inner_window",
        type=int,
        default=256,
        help="M4 sliding inner: token window size. **Causal (GPT-2):** must be ≤ ``n_positions`` (1024). **Seq2seq (Flan-T5):** ``ttt/flan_inner.py`` profile pass.",
    )
    p.add_argument(
        "--m4_inner_stride",
        type=int,
        default=128,
        help="M4 sliding inner: stride between windows (seq2seq Flan path and causal GPT-2 path).",
    )
    p.add_argument(
        "--m4_profile_max_tokens",
        type=int,
        default=None,
        help="M4: max tokens for the merged **profile** stream **before** sliding inner TTT (hard cap; "
        "first N tokens only). If unset: **SD-tooluse / SD-science** = no cap (full document, window-limited forwards); "
        "**LaMP** = min(4096, 8 × --max_input_length). Sliding still uses --m4_inner_window / --m4_inner_stride.",
    )
    p.add_argument(
        "--m4_use_rag",
        action="store_true",
        help="M4: for **each** test row, retrieve top‑K profile items (same as M3: --retriever, --num_retrieved, --ranked) "
        "using that row's ``input`` as the query, run sliding-window inner TTT on the retrieved history, then generate. "
        "Without this flag, inner TTT uses the full merged user profile once per user (no retrieval).",
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
    p.add_argument(
        "--sd_save_responses",
        action="store_true",
        help="For SD-tooluse / SD-science: write eval_responses_<mode>.json (prompt, pred, gold, correct).",
    )
    p.add_argument(
        "--save_encoder_prompts",
        action="store_true",
        help="After each mode, write encoder_prompts_<mode>.json: per-row id, raw task input, full encoder "
        "string passed to generate (M1/M2/M3/M4), approximate token length, and prediction. Same schema as "
        "``train.py --save_encoder_prompts`` (training dump adds ``gold_output``).",
    )
    return p.parse_args()


def task_leaderboard_name(task: str) -> str:
    return task.replace("-", "_")


def _clip_text(s: str, max_chars: int) -> str:
    t = s.replace("\n", " ").strip()
    if len(t) <= max_chars:
        return t
    return t[: max_chars - 3] + "..."


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
    architecture: str = "seq2seq",
    icl_model=None,
) -> None:
    n = len(id_order)
    limit = n if max_samples < 0 else min(n, max_samples)
    score_one = metrics_eval.make_per_example_string_metric()
    if task == "LaMP-5":
        input_label = "LaMP task input (instruction + paper abstract)"
    elif task == "LaMP-7":
        input_label = "LaMP task input (instruction + target tweet)"
    elif task == "SD-tooluse":
        input_label = "Self-Distillation tool-use prompt"
    elif task == "SD-science":
        input_label = "Self-Distillation science (flattened chat) prompt"
    else:
        input_label = "task input"
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
        enc_src = encoder_source_for_seq2seq_mode(
            mode,
            row,
            task=task,
            tokenizer=tokenizer,
            max_in=max_in,
            rag_prompt=rag_prompt,
            model=icl_model,
            architecture=architecture,
        )
        if _is_sd_task(task):
            if task == "SD-tooluse":
                try:
                    ok = sd_eval_metrics.tooluse_correct(pred, sd_eval_metrics.parse_tooluse_gold(ref))
                except (json.JSONDecodeError, ValueError, TypeError, KeyError):
                    ok = False
                per_ex = {"correct": float(ok)}
            else:
                ok = sd_eval_metrics.science_correct(pred, ref)
                per_ex = {"correct": float(ok)}
        else:
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
        print(f"  per_example_metrics: {per_ex}")
    if limit < n:
        print(f"\n[verbose] ... omitted {n - limit} further examples (see --verbose_max_samples).\n")


def _m4_profile_token_cap(max_in: int, m4_profile_max_tokens: int | None, task: str) -> int | None:
    """
    Token cap on the merged **profile** stream before Flan inner TTT sliding windows.

    **SD-tooluse / SD-science:** default ``None`` = tokenize the **full** profile (no pre-cut);
    sliding windows still limit each forward. Set ``--m4_profile_max_tokens`` to bound VRAM/time.

    **LaMP-5 / LaMP-7:** default ``min(4096, 8 × max_input_length)`` when unset (historical behavior).
    """
    if m4_profile_max_tokens is not None:
        return max(1, m4_profile_max_tokens)
    if _is_sd_task(task):
        return None
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
    from ttt.flan_inner import resolve_seq2seq_token_cap

    cap = resolve_seq2seq_token_cap(model, tokenizer, max_in)
    enc = tokenizer(
        sources,
        truncation=True,
        max_length=cap,
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
    ttt_lr: float,
    torch_dtype: torch.dtype | None,
    architecture: str = "seq2seq",
    m4_checkpoint: str | None = None,
    m4_ttt_fraction: float = 0.25,
    m4_inner_window: int = 256,
    m4_inner_stride: int = 128,
    m4_profile_max_tokens: int | None = None,
    m4_use_rag: bool = False,
    rag_retriever: str = "bm25",
    rag_num_retrieved: int = 3,
    rag_ranked: bool = False,
    encoder_prompts_path: str | None = None,
) -> list[tuple[str, str]]:
    load_kw: dict = {"cache_dir": cache_dir}
    if torch_dtype is not None:
        load_kw["torch_dtype"] = torch_dtype

    if architecture == "causal_lm":
        if mode not in ("m1", "m4"):
            raise ValueError(
                f"Causal LM (--architecture causal_lm or a gpt2 base_model) supports m1 and m4 only; got {mode=!r}."
            )
        if mode == "m4":
            from ttt.gpt2_model import TTTGPT2

            model = TTTGPT2(base_model_name, ttt_fraction=0.25)
            if m4_checkpoint:
                try:
                    sd = torch.load(m4_checkpoint, map_location="cpu", weights_only=False)
                except TypeError:
                    sd = torch.load(m4_checkpoint, map_location="cpu")
                model.load_state_dict(sd, strict=True)
            model = model.to(device)
            max_pos = int(
                getattr(model.lm.config, "n_positions", None)
                or getattr(model.lm.config, "max_position_embeddings", 1024)
            )
            if m4_inner_window > max_pos:
                raise ValueError(
                    f"--m4_inner_window ({m4_inner_window}) exceeds model max positions ({max_pos}); "
                    f"GPT-2-style LMs are trained with that context cap per forward."
                )
        else:
            model = AutoModelForCausalLM.from_pretrained(base_model_name, **load_kw)
    else:
        if mode == "m4":
            from ttt.flan_dual_mlp_model import TTTFlanT5

            model = TTTFlanT5(
                model_name=base_model_name,
                ttt_fraction=m4_ttt_fraction,
                cache_dir=cache_dir,
                torch_dtype=torch_dtype,
            )
            if m4_checkpoint:
                try:
                    sd = torch.load(m4_checkpoint, map_location="cpu", weights_only=False)
                except TypeError:
                    sd = torch.load(m4_checkpoint, map_location="cpu")
                model.load_state_dict(sd, strict=True)
        else:
            base = AutoModelForSeq2SeqLM.from_pretrained(base_model_name, **load_kw)
            if mode in ("m2", "m3") and adapter_dir:
                model = PeftModel.from_pretrained(base, adapter_dir)
            else:
                model = base

    model.to(device)
    model.eval()

    if _is_sd_task(task):
        _apply_sd_truncation_side(tokenizer)
        gen_tok_m = getattr(model, "tokenizer", None)
        if gen_tok_m is not None:
            _apply_sd_truncation_side(gen_tok_m)

    rag_selector = None
    if mode == "m4" and m4_use_rag:
        from ttt.lamp_profile_rag import LampProfileRAG

        rag_selector = LampProfileRAG(
            task,
            num_retrieved=rag_num_retrieved,
            retriever=rag_retriever,
            ranked=rag_ranked,
            device=device,
            cache_dir=cache_dir,
        )

    preds: list[tuple[str, str]] = []

    encode_max_len = max_in
    if mode == "m2" and architecture == "seq2seq":
        encode_max_len = prompting.icl_m2_max_encoder_tokens(
            task, tokenizer, max_in, model=model, architecture=architecture
        )

    def _maybe_dump_encoder_prompts() -> None:
        if not encoder_prompts_path:
            return
        write_encoder_prompts_json(
            encoder_prompts_path,
            mode=mode,
            task=task,
            rows=rows,
            preds=preds,
            tokenizer=tokenizer,
            max_in=max_in,
            encode_max_len=encode_max_len,
            rag_prompt=rag_prompt,
            model=model,
            architecture=architecture,
        )

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
            decoded = batched_generate(model, gen_tok, sources, device, encode_max_len, max_new)
        preds.extend(zip(meta_ids, decoded))

    if mode in ("m1", "m2", "m3"):
        batch_src: list[str] = []
        batch_ids: list[str] = []
        for row in tqdm(rows, desc=mode):
            if mode == "m1":
                src = row["input"]
            elif mode == "m2":
                src = prompting.icl_m2_encoder_text(
                    row,
                    tokenizer,
                    task=task,
                    max_input_length=max_in,
                    model=model,
                    architecture=architecture,
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
        _maybe_dump_encoder_prompts()
        return preds

    if mode == "m4":
        if architecture == "causal_lm":
            from ttt import e2e as ttt_e2e
            from ttt.gpt2_inner import inner_adapt_inplace

            prof_cap = _m4_profile_token_cap(max_in, m4_profile_max_tokens, task)

            if rag_selector is not None:
                for _user, urows in tqdm(list(user_to_rows.items()), desc=mode):
                    prof = merge_profiles(urows)
                    for row in urows:
                        snap = model.snapshot_inner()
                        try:
                            picked = rag_selector.select(_m4_rag_query(row, task), prof)
                            use_prof = picked if picked else prof
                            if _is_sd_task(task):
                                stream = sd_self_distill.sd_ttt_inner_stream_text([row], use_prof)
                            else:
                                stream = ttt_e2e.build_flat_history_stream(task, use_prof)
                            gen_tok = model.tokenizer
                            enc_kw: dict = {"return_tensors": "pt", "truncation": False}
                            if prof_cap is not None:
                                enc_kw["truncation"] = True
                                enc_kw["max_length"] = prof_cap
                            enc = gen_tok(stream, **enc_kw)
                            ctx_ids = enc["input_ids"].to(device)
                            if ctx_ids.shape[1] >= 2:
                                inner_adapt_inplace(
                                    model,
                                    ctx_ids,
                                    lr=ttt_lr,
                                    window=m4_inner_window,
                                    stride=m4_inner_stride,
                                )
                            model.eval()
                            handle_batch([row["input"]], [row["id"]])
                        finally:
                            model.restore_inner(snap)
                _maybe_dump_encoder_prompts()
                return preds

            for _user, urows in tqdm(list(user_to_rows.items()), desc=mode):
                snap = model.snapshot_inner()
                try:
                    prof = merge_profiles(urows)
                    if _is_sd_task(task):
                        stream = sd_self_distill.sd_ttt_inner_stream_text(urows, prof)
                    else:
                        stream = ttt_e2e.build_flat_history_stream(task, prof)
                    gen_tok = model.tokenizer
                    enc_kw2: dict = {"return_tensors": "pt", "truncation": False}
                    if prof_cap is not None:
                        enc_kw2["truncation"] = True
                        enc_kw2["max_length"] = prof_cap
                    enc = gen_tok(stream, **enc_kw2)
                    ctx_ids = enc["input_ids"].to(device)
                    if ctx_ids.shape[1] >= 2:
                        inner_adapt_inplace(
                            model,
                            ctx_ids,
                            lr=ttt_lr,
                            window=m4_inner_window,
                            stride=m4_inner_stride,
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
            _maybe_dump_encoder_prompts()
            return preds

        from ttt.flan_inner import inner_adapt_t5_inplace

        prof_cap = _m4_profile_token_cap(max_in, m4_profile_max_tokens, task)

        if rag_selector is not None:
            for _user, urows in tqdm(list(user_to_rows.items()), desc=mode):
                prof = merge_profiles(urows)
                for row in urows:
                    snap = model.snapshot_inner()
                    try:
                        picked = rag_selector.select(_m4_rag_query(row, task), prof)
                        use_prof = picked if picked else prof
                        ttt_txt = (
                            sd_self_distill.sd_ttt_inner_stream_text([row], use_prof)
                            if _is_sd_task(task)
                            else None
                        )
                        inner_adapt_t5_inplace(
                            model,
                            model.tokenizer,
                            task=task,
                            profile=use_prof,
                            device=device,
                            lr=ttt_lr,
                            window=m4_inner_window,
                            stride=m4_inner_stride,
                            profile_token_cap=prof_cap,
                            profile_rag=None if ttt_txt is not None else rag_selector,
                            rag_query=None if ttt_txt is not None else _m4_rag_query(row, task),
                            ttt_stream_text=ttt_txt,
                        )
                        model.eval()
                        handle_batch([row["input"]], [row["id"]])
                    finally:
                        model.restore_inner(snap)
            _maybe_dump_encoder_prompts()
            return preds

        for _user, urows in tqdm(list(user_to_rows.items()), desc=mode):
            snap = model.snapshot_inner()
            prof = merge_profiles(urows)
            try:
                ttt_txt = sd_self_distill.sd_ttt_inner_stream_text(urows, prof) if _is_sd_task(task) else None
                inner_adapt_t5_inplace(
                    model,
                    model.tokenizer,
                    task=task,
                    profile=prof,
                    device=device,
                    lr=ttt_lr,
                    window=m4_inner_window,
                    stride=m4_inner_stride,
                    profile_token_cap=prof_cap,
                    ttt_stream_text=ttt_txt,
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
        _maybe_dump_encoder_prompts()
        return preds

    raise ValueError(f"Unsupported mode: {mode!r}")


def main():
    args = parse_args()
    if args.fp16 and args.bf16:
        raise SystemExit("Use at most one of --fp16 and --bf16.")
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        from util.cuda_tf32 import enable_tf32

        enable_tf32()

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
    # Do not pass gold ``output`` into the model forward paths.
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
    if arch == "causal_lm" and any(m in ("m2", "m3") for m in modes):
        raise ValueError(
            "Causal LM (gpt2-style) is only wired for m1 and m4; use --architecture seq2seq for m2/m3."
        )

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.base_model, cache_dir=args.cache_dir, use_fast=False, legacy=False
        )
    except TypeError:
        tokenizer = AutoTokenizer.from_pretrained(
            args.base_model, cache_dir=args.cache_dir, use_fast=False
        )
    if arch == "causal_lm" and tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if _is_sd_task(args.task):
        _apply_sd_truncation_side(tokenizer)
    rag_prompt, contriever = prompting.m3_rag_prompt_and_contriever(
        args.task,
        tokenizer,
        num_retrieved=args.num_retrieved,
        retriever=args.retriever,
        ranked=args.ranked,
        max_length=args.max_input_length,
        device=device,
        cache_dir=args.cache_dir,
    )
    if contriever is not None:
        contriever = contriever.to(device)

    results_summary: dict[str, dict] = {}
    for mode in modes:
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
            ttt_lr=args.ttt_lr,
            torch_dtype=torch_dtype,
            architecture=arch,
            m4_checkpoint=args.m4_checkpoint,
            m4_ttt_fraction=args.m4_ttt_fraction,
            m4_inner_window=args.m4_inner_window,
            m4_inner_stride=args.m4_inner_stride,
            m4_profile_max_tokens=args.m4_profile_max_tokens,
            m4_use_rag=args.m4_use_rag,
            rag_retriever=args.retriever,
            rag_num_retrieved=args.num_retrieved,
            rag_ranked=args.ranked,
            encoder_prompts_path=(
                os.path.join(args.output_dir, f"encoder_prompts_{mode}.json")
                if args.save_encoder_prompts
                else None
            ),
        )
        pred_map = {i: p for i, p in pairs}
        preds_ordered = [pred_map[i] for i in id_order]
        scores_sd: list[int] | None = None
        if _is_sd_task(args.task):
            if args.task == "SD-tooluse":
                scores_sd, acc = sd_eval_metrics.tooluse_accuracy(preds_ordered, refs)
            else:
                scores_sd, acc = sd_eval_metrics.science_accuracy(preds_ordered, refs)
            metrics = {
                "accuracy": float(acc),
                "num_correct": int(sum(scores_sd)),
                "num_total": len(scores_sd),
            }
        else:
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
        if args.save_encoder_prompts:
            ep = os.path.join(args.output_dir, f"encoder_prompts_{mode}.json")
            print(f"Wrote encoder prompts for inspection: {ep}")
        if args.sd_save_responses and _is_sd_task(args.task) and scores_sd is not None:
            resp_path = os.path.join(args.output_dir, f"eval_responses_{mode}.json")
            with open(resp_path, "w", encoding="utf-8") as f:
                json.dump(
                    [
                        {
                            "id": id_order[j],
                            "input": rows[j]["input"],
                            "pred": preds_ordered[j],
                            "gold": refs[j],
                            "correct": bool(scores_sd[j]),
                        }
                        for j in range(len(id_order))
                    ],
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
                f.write("\n")
            print(f"Wrote {resp_path}")
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
                architecture=arch,
                icl_model=None,
            )

    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)


if __name__ == "__main__":
    main()
