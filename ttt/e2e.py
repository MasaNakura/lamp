"""LaMP profile **stream text** and **sliding token windows** for Flan M4 inner TTT (see ``ttt/flan_inner.py``)."""
from __future__ import annotations

from typing import Any, Iterator


def build_flat_history_stream(task: str, profile: list[dict[str, Any]]) -> str:
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
    elif task in ("SD-tooluse", "SD-science"):
        for p in profile:
            tx = (p.get("text") or "").strip()
            if tx:
                parts.append(tx)
    else:
        raise ValueError(task)
    if task in ("SD-tooluse", "SD-science"):
        return "\n".join(parts)
    return "\n\n".join(parts)


def iter_history_token_id_windows(
    ids: list[int],
    *,
    window: int,
    stride: int,
) -> Iterator[list[int]]:
    """
    Left-to-right sliding windows over a **full** token sequence (no truncation here).

    Used by Flan inner TTT so every token of the profile stream can receive an update
    pass when combined with ``truncation=False`` upstream encoding.
    """
    if not ids:
        return
    if stride <= 0:
        stride = window
    n = len(ids)
    i = 0
    while i < n:
        yield ids[i : i + window]
        if i + window >= n:
            break
        i += stride
