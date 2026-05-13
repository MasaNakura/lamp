"""Meta-training data: WikiText (MAM default) or LaMP profile streams (LaMP-5 / LaMP-7)."""
from __future__ import annotations

import os
import random
from typing import Any, Iterator

import torch


def _tokenize_and_cache(
    tokenizer,
    cache_path: str,
    dataset_name: str = "wikitext",
    config: str = "wikitext-103-raw-v1",
    split: str = "train",
    max_docs: int = 2000,
) -> torch.Tensor:
    from datasets import load_dataset

    if os.path.exists(cache_path):
        try:
            return torch.load(cache_path, weights_only=False)
        except TypeError:
            return torch.load(cache_path)

    ds = load_dataset(dataset_name, config, split=split, streaming=True)
    buf: list[int] = []
    docs = 0
    for row in ds:
        text = row["text"].strip()
        if not text:
            continue
        ids = tokenizer.encode(text)
        if len(ids) < 64:
            continue
        buf.extend(ids)
        buf.append(tokenizer.eos_token_id)
        docs += 1
        if docs >= max_docs:
            break

    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    tensor = torch.tensor(buf, dtype=torch.long)
    torch.save(tensor, cache_path)
    return tensor


def meta_example_stream(
    tokenizer,
    context_len: int = 256,
    continuation_len: int = 64,
    cache_path: str = ".cache/wikitext103_train.pt",
    seed: int = 0,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    flat = _tokenize_and_cache(tokenizer, cache_path)
    total = context_len + continuation_len
    rng = random.Random(seed)
    n = flat.size(0)
    while True:
        start = rng.randint(0, max(0, n - total - 1))
        chunk = flat[start : start + total]
        ctx = chunk[:context_len].unsqueeze(0)
        cont = chunk[context_len:].unsqueeze(0)
        yield ctx, cont


def _lamp_profile_document(task: str, profile: list[dict[str, Any]]) -> str:
    """Same flattened profile text as eval Flan M4 / ``build_flat_history_stream``."""
    from ttt.e2e import build_flat_history_stream

    prof = profile if isinstance(profile, list) else []
    return build_flat_history_stream(task, prof)


def _lamp_train_token_cache(
    tokenizer,
    rows: list[dict[str, Any]],
    task: str,
    cache_path: str,
) -> torch.Tensor:
    if os.path.exists(cache_path):
        try:
            return torch.load(cache_path, weights_only=False)
        except TypeError:
            return torch.load(cache_path)

    # LaMP-7 rows are short tweets per profile item; use lower floors than LaMP-5 title+abstract.
    if task in ("SD-tooluse", "SD-science"):
        min_doc_chars, min_row_tokens, min_total_tokens = 16, 8, 128
    elif task == "LaMP-7":
        min_doc_chars, min_row_tokens, min_total_tokens = 24, 12, 256
    else:
        min_doc_chars, min_row_tokens, min_total_tokens = 80, 32, 512

    buf: list[int] = []
    for row in rows:
        if task in ("SD-tooluse", "SD-science"):
            prof = row.get("profile") or []
            if not isinstance(prof, list):
                prof = []
            from util.sd_self_distill import sd_ttt_inner_stream_text

            doc = sd_ttt_inner_stream_text([row], prof)
        else:
            prof = row.get("profile") or []
            doc = _lamp_profile_document(task, prof if isinstance(prof, list) else [])
        if len(doc) < min_doc_chars:
            continue
        ids = tokenizer.encode(doc)
        if len(ids) < min_row_tokens:
            continue
        buf.extend(ids)
        buf.append(tokenizer.eos_token_id)

    if len(buf) < min_total_tokens:
        raise RuntimeError(
            f"LaMP meta cache: too few tokens after flattening profiles ({len(buf)} < {min_total_tokens}). "
            "Check that train JSON has real profile text, not placeholders. "
            "LaMP-7 needs enough tweets across the train split to fill the buffer."
        )
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    tensor = torch.tensor(buf, dtype=torch.long)
    torch.save(tensor, cache_path)
    return tensor


def meta_example_stream_lamp(
    tokenizer,
    train_rows: list[dict[str, Any]],
    task: str,
    *,
    context_len: int = 256,
    continuation_len: int = 64,
    cache_path: str = ".cache/lamp_train_profiles.pt",
    seed: int = 0,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    flat = _lamp_train_token_cache(tokenizer, train_rows, task, cache_path)
    total = context_len + continuation_len
    rng = random.Random(seed)
    n = flat.size(0)
    if n < total + 1:
        raise RuntimeError(f"LaMP token buffer too short ({n} < {total + 1}).")
    while True:
        start = rng.randint(0, n - total - 1)
        chunk = flat[start : start + total]
        ctx = chunk[:context_len].unsqueeze(0)
        cont = chunk[context_len:].unsqueeze(0)
        yield ctx, cont


def meta_example_stream_lamp_rag(
    tokenizer,
    train_rows: list[dict[str, Any]],
    task: str,
    rag: Any,
    *,
    context_len: int = 256,
    continuation_len: int = 64,
    seed: int = 0,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """
    Like ``meta_example_stream_lamp`` but each (context, continuation) span is drawn from a
    **single train row** after LaMP-style RAG narrows that row's profile. Aligns meta-training
    with query-conditioned TTT at test time.
    """
    from ttt.e2e import build_flat_history_stream
    from util.sd_self_distill import sd_rag_query_for_row, sd_ttt_inner_stream_text

    valid = [r for r in train_rows if (r.get("profile") or []) and (r.get("input") or "").strip()]
    if not valid:
        raise RuntimeError("meta_example_stream_lamp_rag: no rows with non-empty input and profile.")
    total = context_len + continuation_len
    rng = random.Random(seed)
    attempts = 0
    while True:
        attempts += 1
        if attempts > 50_000:
            raise RuntimeError(
                "meta_example_stream_lamp_rag: could not sample a span long enough after many tries. "
                "Increase train profile text, lower --context_len/--continuation_len, or raise "
                "--ttt_rag_num_retrieved."
            )
        row = rng.choice(valid)
        inp = sd_rag_query_for_row(row)
        prof = row.get("profile") or []
        subset = rag.select(inp, prof)
        use_prof = subset if subset else prof
        if task in ("SD-tooluse", "SD-science"):
            doc = sd_ttt_inner_stream_text([row], use_prof)
        else:
            doc = build_flat_history_stream(task, use_prof)
        ids = tokenizer.encode(doc, add_special_tokens=False)
        if len(ids) < total + 1:
            continue
        start = rng.randint(0, len(ids) - total - 1)
        chunk = ids[start : start + total]
        ctx = torch.tensor(chunk[:context_len], dtype=torch.long).unsqueeze(0)
        cont = torch.tensor(chunk[context_len:], dtype=torch.long).unsqueeze(0)
        yield ctx, cont
