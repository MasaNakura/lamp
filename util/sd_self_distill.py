"""Load Self-Distillation HF arrow datasets and expose LaMP-shaped rows + JSON export."""
from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def default_sd_root() -> str:
    return os.path.join(_ROOT, "Self-Distillation")


def chunk_text_to_profile_rows(
    text: str,
    *,
    max_chars: int = 480,
    overlap: int = 0,
) -> list[dict[str, Any]]:
    """
    One retrievable ``profile`` row per **non-empty line** (newline-separated), matching
    tool-use prompts where each line often describes a different tool / API, and science
    text where lines separate distinct facts or options.

    If a single line exceeds ``max_chars`` (rare), it is subdivided with a sliding window
    so BM25 rows stay bounded.
    """
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    if not text.strip():
        return []
    rows: list[str] = []
    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            continue
        if len(line) <= max_chars:
            rows.append(line)
        else:
            step = max(1, max_chars - max(0, overlap))
            start = 0
            while start < len(line):
                rows.append(line[start : start + max_chars])
                start += step
    return [{"text": r, "row_idx": i} for i, r in enumerate(rows)]


def science_user_content(prompt: Any) -> str:
    if isinstance(prompt, list):
        for m in reversed(prompt):
            if isinstance(m, dict) and (m.get("role") or "").lower() == "user":
                return (m.get("content") or "").strip()
    return ""


def sd_rag_query_for_row(row: dict[str, Any]) -> str:
    """Short query for BM25/Contriever; falls back to full ``input``."""
    q = (row.get("sd_rag_query") or "").strip()
    if q:
        return q
    return (row.get("input") or "").strip()


def sd_m2_preserving_tail(row: dict[str, Any], *, task: str) -> str:
    """
    **M2 (ICL):** text that must stay **un-truncated** (as much as possible): task framing,
    output-format instructions, and the concrete user question.

    **Tool-use:** keep everything through the ``Documentation:`` header (HF ``prompt`` layout),
    then append ``sd_rag_query`` (dataset ``instruction``) when it is not already included.

    **Science:** keep the flattened ``system:`` block plus the **last** ``user:`` block
    (the MC question), matching ``role:`` lines in ``input``.
    """
    inp = (row.get("input") or "").replace("\r\n", "\n").strip()
    inst = sd_rag_query_for_row(row)

    if task == "SD-tooluse":
        low = inp.lower()
        key = "documentation:"
        idx = low.find(key)
        if idx != -1:
            kept = inp[: idx + len(key)].rstrip()
        else:
            kept = inp[: min(1600, len(inp))].rstrip()
        if inst and inst.lower() not in kept.lower():
            return f"{kept}\n\n{inst}".strip()
        return kept or inst or inp[:1200]

    if task == "SD-science":
        matches = list(re.finditer(r"(?mi)^\s*user\s*:", inp))
        if matches:
            last_u = matches[-1].start()
            head = inp[:last_u].strip()
            user_block = inp[last_u:].strip()
            return f"{head}\n\n{user_block}".strip() if head else user_block
        return inp or inst

    raise ValueError(f"sd_m2_preserving_tail: unknown task {task!r}")


def sd_ttt_inner_stream_text(
    urows: list[dict[str, Any]],
    prof_use: list[dict[str, Any]],
) -> str:
    """
    Token stream for **M4 inner TTT** on Self-Distillation rows.

    When ``prof_use`` is exactly the same ordered line list as ``urows[0]['profile']``
    (typical single-user SD eval), returns the **original** dataset ``input`` string so
    TTT matches the HF export verbatim (including blank lines / spacing in ``input``).

    When ``prof_use`` is a **RAG subset** (or multiple users), returns ``text`` fields
    joined with newlines in retrieval/list order — the natural reconstruction of the
    selected documentation lines without ``[ctx]`` prefixes.
    """
    if not prof_use:
        return (urows[0].get("input") or "").strip() if urows else ""
    if len(urows) == 1:
        r0 = urows[0]
        full_prof = r0.get("profile") or []
        if len(prof_use) == len(full_prof) and all(
            isinstance(a, dict) and isinstance(b, dict)
            and (a.get("text") or "").strip() == (b.get("text") or "").strip()
            for a, b in zip(prof_use, full_prof)
        ):
            return (r0.get("input") or "").strip()
    return "\n".join(
        (p.get("text") or "").strip()
        for p in prof_use
        if isinstance(p, dict) and (p.get("text") or "").strip()
    )


def science_prompt_as_string(prompt: Any) -> str:
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        lines: list[str] = []
        for m in prompt:
            if not isinstance(m, dict):
                continue
            role = (m.get("role") or "user").strip()
            content = (m.get("content") or "").strip()
            lines.append(f"{role}: {content}")
        return "\n".join(lines)
    return str(prompt)


def _row_tooluse(
    ex: dict[str, Any],
    idx: int,
    *,
    chunk_chars: int,
) -> dict[str, Any]:
    rid = f"tu_{idx}"
    inp = (ex.get("prompt") or "").strip()
    gold = ex["golden_answer"]
    out = json.dumps(gold, ensure_ascii=False)
    rag_q = (ex.get("instruction") or "").strip()
    profile = chunk_text_to_profile_rows(inp, max_chars=chunk_chars)
    if not profile:
        profile = [{"text": inp[: chunk_chars * 2]}]
    return {
        "id": rid,
        "user_id": rid,
        "input": inp,
        "sd_rag_query": rag_q,
        "profile": profile,
        "output": out,
    }


def _row_science(
    ex: dict[str, Any],
    idx: int,
    *,
    chunk_chars: int,
) -> dict[str, Any]:
    rid = f"sc_{idx}"
    prompt_field = ex.get("prompt")
    if prompt_field is None:
        prompt_field = ex.get("messages")
    inp = science_prompt_as_string(prompt_field)
    gold = (ex.get("answer") or ex.get("output_text") or "").strip()
    rag_q = science_user_content(prompt_field)
    if not rag_q:
        rag_q = inp[:800]
    profile = chunk_text_to_profile_rows(inp, max_chars=chunk_chars)
    if not profile:
        profile = [{"text": inp[: chunk_chars * 2]}]
    return {
        "id": rid,
        "user_id": rid,
        "input": inp,
        "sd_rag_query": rag_q,
        "profile": profile,
        "output": gold,
    }


def hf_rows_for_split(
    subtask: str,
    split: str,
    *,
    sd_root: str | None = None,
    chunk_chars: int = 480,
) -> list[dict[str, Any]]:
    from datasets import load_from_disk

    root = sd_root or default_sd_root()
    if subtask == "tooluse":
        path = os.path.join(root, "data", "tooluse_data", f"{split}_data")
        builder = lambda ex, i: _row_tooluse(ex, i, chunk_chars=chunk_chars)
    elif subtask == "science":
        path = os.path.join(root, "data", "science_data", f"{split}_data")
        builder = lambda ex, i: _row_science(ex, i, chunk_chars=chunk_chars)
    else:
        raise ValueError(f"Unknown subtask {subtask!r}; use 'tooluse' or 'science'.")

    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"Missing Self-Distillation split at {path!r}. "
            "Run `git submodule update --init Self-Distillation` from the repo root."
        )
    ds = load_from_disk(path)
    return [builder(ds[i], i) for i in range(len(ds))]


def export_lamp_json(
    subtask: str,
    split: str,
    out_dir: str,
    *,
    sd_root: str | None = None,
    chunk_chars: int = 480,
) -> tuple[str, str]:
    rows = hf_rows_for_split(subtask, split, sd_root=sd_root, chunk_chars=chunk_chars)
    os.makedirs(out_dir, exist_ok=True)
    questions: list[dict[str, Any]] = []
    outputs: list[dict[str, Any]] = []
    for r in rows:
        q = {k: v for k, v in r.items() if k != "output"}
        questions.append(q)
        outputs.append({"id": r["id"], "output": r["output"]})
    split_tag = "train" if split == "train" else "test"
    q_path = os.path.join(out_dir, f"{subtask}_{split_tag}_questions.json")
    o_path = os.path.join(out_dir, f"{subtask}_{split_tag}_outputs.json")
    with open(q_path, "w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)
        f.write("\n")
    with open(o_path, "w", encoding="utf-8") as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return q_path, o_path


def _truncate_ids(tokenizer, text: str, max_tokens: int) -> str:
    ids = tokenizer(text, add_special_tokens=False, verbose=False)["input_ids"]
    if len(ids) <= max_tokens:
        return text
    return tokenizer.decode(ids[-max_tokens:], skip_special_tokens=True)


def build_sd_rag_prompt(
    row: dict[str, Any],
    *,
    task: str,
    num_retrieved: int,
    retriever: str,
    ranked: bool,
    max_length: int,
    tokenizer,
    device,
    cache_dir: str | None = None,
) -> str:
    """
    M3-style prompt: retrieve **top-k profile rows** (newline-split lines) with ``LampProfileRAG``,
    then concatenate as many as fit under ``max_length``, followed by ``Task:`` + full ``input``
    (``input`` is left-truncated by tokens only if the assembled string still exceeds ``max_length``).
    """
    inp = (row.get("input") or "").strip()
    query = sd_rag_query_for_row(row)
    prof = row.get("profile") or []
    if not prof:
        return _truncate_ids(tokenizer, inp, max_length)

    from ttt.lamp_profile_rag import LampProfileRAG

    r = LampProfileRAG(
        task,
        num_retrieved=num_retrieved,
        retriever=retriever,
        ranked=ranked,
        device=device,
        cache_dir=cache_dir,
    )
    chosen = r.select(query, prof)
    use_prof = chosen if chosen else prof[: max(1, num_retrieved)]

    header = "Retrieved context (relevant rows):\n"
    sep = "\n---\n"
    task_header = "\n\nTask:\n"
    tail = task_header + inp

    def _tok_len(s: str) -> int:
        return len(tokenizer(s, add_special_tokens=False, verbose=False)["input_ids"])

    pieces: list[str] = []
    for p in use_prof:
        chunk = (p.get("text") or "").strip()
        if not chunk:
            continue
        trial_ctx = header + sep.join(pieces + [chunk]) if pieces else header + chunk
        full = trial_ctx + tail
        if _tok_len(full) <= max_length:
            pieces.append(chunk)
            continue
        # Binary search longest prefix of ``chunk`` that still fits with current pieces + tail.
        lo, hi = 0, len(chunk)
        best = ""
        while lo <= hi:
            mid = (lo + hi) // 2
            sub = chunk[:mid]
            trial2 = header + sep.join(pieces + [sub]) if pieces else header + sub
            if sub.strip() and _tok_len(trial2 + tail) <= max_length:
                best = sub
                lo = mid + 1
            else:
                hi = mid - 1
        if best.strip():
            pieces.append(best.strip())
        break

    if pieces:
        ctx = header + sep.join(pieces)
    else:
        ctx = ""
    out = (ctx + tail) if ctx.strip() else tail.lstrip("\n")
    if _tok_len(out) <= max_length:
        return out
    return _truncate_ids(tokenizer, out, max_length)


def lamp_task_name(subtask: str) -> str:
    if subtask == "tooluse":
        return "SD-tooluse"
    if subtask == "science":
        return "SD-science"
    raise ValueError(subtask)


def parse_export_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export Self-Distillation splits to LaMP-style JSON.")
    p.add_argument("--subtask", choices=["tooluse", "science"], required=True)
    p.add_argument("--split", choices=["train", "eval"], required=True)
    p.add_argument(
        "--out_dir",
        default=None,
        help="Output directory (default: <repo>/data/sd_self_distill)",
    )
    p.add_argument("--sd_root", default=None, help="Path to Self-Distillation repo root.")
    p.add_argument(
        "--chunk_chars",
        type=int,
        default=480,
        help="Max characters per line before subdividing a single overlong line (default: 480). "
        "Profile rows are one non-empty line each.",
    )
    return p.parse_args()


def main_export() -> None:
    args = parse_export_args()
    out_dir = args.out_dir or os.path.join(_ROOT, "data", "sd_self_distill")
    q, o = export_lamp_json(
        args.subtask, args.split, out_dir, sd_root=args.sd_root, chunk_chars=args.chunk_chars
    )
    print(f"Wrote {q}\n{o}")


if __name__ == "__main__":
    main_export()
