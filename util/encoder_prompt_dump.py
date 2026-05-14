"""Shared encoder-string dump for ``run_evaluate.py`` and ``train.py`` (M2/M3 parity checks)."""
from __future__ import annotations

import json
import os
from typing import Any, Callable

from util import prompting


def encoder_source_for_seq2seq_mode(
    mode: str,
    row: dict,
    *,
    task: str,
    tokenizer,
    max_in: int,
    rag_prompt: Callable[[dict], str],
    model=None,
    architecture: str = "seq2seq",
) -> str:
    """Same pre-tokenization encoder text as ``run_for_mode`` for seq2seq (M1–M3 / M4 batch path)."""
    if mode in ("m1", "m4"):
        return row["input"]
    if mode == "m2":
        return prompting.icl_m2_encoder_text(
            row,
            tokenizer,
            task=task,
            max_input_length=max_in,
            model=model,
            architecture=architecture,
        )
    return rag_prompt(row)


def write_encoder_prompts_json(
    path: str,
    *,
    mode: str,
    task: str,
    rows: list[dict],
    preds: list[tuple[Any, str]],
    tokenizer,
    max_in: int,
    encode_max_len: int,
    rag_prompt: Callable[[dict], str] | None = None,
    model=None,
    architecture: str = "seq2seq",
    include_gold_output: bool = False,
    encoder_text_fn: Callable[[dict], str] | None = None,
) -> None:
    """
    Write one JSON array aligned with eval ``encoder_prompts_<mode>.json``.

    If ``include_gold_output`` (training dump), each record also has ``gold_output`` from the row.

    If ``encoder_text_fn`` is set, it supplies ``encoder_prompt`` per row (``rag_prompt`` / ``model`` are
    ignored for encoding). Otherwise ``rag_prompt`` is required for ``encoder_source_for_seq2seq_mode``.
    """
    pred_map = dict(preds)
    lm = model if architecture == "seq2seq" else None
    recs: list[dict] = []
    for row in rows:
        rid = row["id"]
        if encoder_text_fn is not None:
            enc = encoder_text_fn(row)
        else:
            if rag_prompt is None:
                raise ValueError("write_encoder_prompts_json: pass rag_prompt or encoder_text_fn")
            enc = encoder_source_for_seq2seq_mode(
                mode,
                row,
                task=task,
                tokenizer=tokenizer,
                max_in=max_in,
                rag_prompt=rag_prompt,
                model=lm,
                architecture=architecture,
            )
        ntok = len(
            tokenizer.encode(
                enc,
                add_special_tokens=False,
                truncation=True,
                max_length=131072,
            )
        )
        rec: dict[str, Any] = {
            "id": rid,
            "mode": mode,
            "task": task,
            "raw_task_input": row.get("input", ""),
            "encoder_prompt": enc,
            "approx_encoder_tokens_trunc": ntok,
            "seq2seq_encode_max_length": encode_max_len,
            "prediction": pred_map.get(rid, ""),
        }
        if include_gold_output:
            rec["gold_output"] = row.get("output", "")
        recs.append(rec)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(recs, f, ensure_ascii=False, indent=2)
        f.write("\n")


def flan_meta_profile_stream_text(row: dict[str, Any], task: str, profile_rag: Any | None = None) -> str:
    """
    Profile-side text that ``train_flan_meta`` / ``ttt.flan_outer`` meta streams are built from
    (flattened profile, optionally RAG-narrowed): same construction as ``meta_example_stream_lamp`` /
    ``meta_example_stream_lamp_rag`` document strings.
    """
    from ttt.e2e import build_flat_history_stream
    from util.sd_self_distill import sd_rag_query_for_row, sd_ttt_inner_stream_text

    prof = row.get("profile") or []
    if not isinstance(prof, list):
        prof = []
    use_prof = prof
    if profile_rag is not None:
        if task in ("SD-tooluse", "SD-science"):
            q = sd_rag_query_for_row(row)
        else:
            q = (row.get("input") or "").strip()
        if q:
            picked = profile_rag.select(q, prof)
            if picked:
                use_prof = picked
    if task in ("SD-tooluse", "SD-science"):
        return sd_ttt_inner_stream_text([row], use_prof)
    return build_flat_history_stream(task, use_prof)
