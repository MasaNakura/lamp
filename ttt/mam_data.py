"""Meta-training data: WikiText (MAM default) or LaMP-5 profile streams."""
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
    parts: list[str] = []
    if task == "LaMP-5":
        for p in profile:
            t = (p.get("title") or "").strip()
            a = (p.get("abstract") or "").strip()
            if t:
                parts.append(f"[title] {t}")
            if a:
                parts.append(f"[abstract] {a}")
    elif task == "LaMP-7":
        for p in profile:
            tx = (p.get("text") or "").strip()
            if tx:
                parts.append(f"[tweet] {tx}")
    else:
        raise ValueError(task)
    return "\n\n".join(parts)


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

    buf: list[int] = []
    for row in rows:
        prof = row.get("profile") or []
        doc = _lamp_profile_document(task, prof if isinstance(prof, list) else [])
        if len(doc) < 80:
            continue
        ids = tokenizer.encode(doc)
        if len(ids) < 32:
            continue
        buf.extend(ids)
        buf.append(tokenizer.eos_token_id)

    if len(buf) < 512:
        raise RuntimeError(
            "LaMP meta cache: too few tokens after flattening profiles. "
            "Check that train JSON has real profile text, not placeholders."
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
