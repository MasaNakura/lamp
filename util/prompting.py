"""Prompt construction: LaMP RAG prompts + long-context ICL concatenation."""
from __future__ import annotations

from typing import Any, Callable

import torch

from util.lamp_paths import ensure_lamp_on_path

ensure_lamp_on_path()

from prompts.prompts import create_prompt_generator  # noqa: E402

from util import sd_self_distill  # noqa: E402


def task_internal_name(task: str) -> str:
    if task in ("LaMP-5", "LaMP-7", "SD-tooluse", "SD-science"):
        return task
    raise ValueError(f"Unknown task for RAG helper: {task!r}")


def build_rag_prompt_fn(
    task: str,
    tokenizer,
    *,
    num_retrieved: int,
    retriever: str = "bm25",
    ranked: bool = False,
    max_length: int = 512,
) -> Callable[[dict[str, Any]], str]:
    if task in ("SD-tooluse", "SD-science"):
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def one_sd(sample: dict[str, Any]) -> str:
            return sd_self_distill.build_sd_rag_prompt(
                sample,
                task=task,
                num_retrieved=num_retrieved,
                retriever=retriever,
                ranked=ranked,
                max_length=max_length,
                tokenizer=tokenizer,
                device=dev,
                cache_dir=None,
            )

        return one_sd

    internal = task_internal_name(task)
    gen, _contriever = create_prompt_generator(
        num_retrieved, retriever, ranked, max_length, tokenizer
    )

    def one(sample: dict[str, Any]) -> str:
        return gen(sample["input"], sample["profile"], internal)

    return one


def build_icl_source(
    sample: dict[str, Any],
    tokenizer,
    *,
    task: str,
    max_tokens: int = 512,
    reserve_for_input: int = 128,
) -> str:
    """
    Model 2 (ICL): long-context encoder text from ``input`` + ``profile``.

    **LaMP-5 / LaMP-7:** history chunks (profile) then the instance tail.
    **SD-tooluse:** header first (through ``Documentation:`` via ``sd_m2_preserving_tail``),
    then the full post-header tool documentation from ``input`` (all tools in that block),
    truncating the **middle** only when over ``max_tokens``.
    **SD-science:** preserved prefix first, then profile lines packed under budget.
    """
    prof = sample.get("profile") or []
    if task == "LaMP-5":
        hist_chunks = [
            f'History paper: title "{p.get("title", "")}" abstract: {p.get("abstract", "")}'
            for p in prof
        ]
    elif task == "LaMP-7":
        hist_chunks = [f'History tweet: "{p.get("text", "")}"' for p in prof]
    elif task == "SD-tooluse":
        return sd_self_distill.build_sd_m2_tooluse_icl_encoder(sample, tokenizer, max_tokens=max_tokens)
    elif task == "SD-science":
        hist_chunks = [(p.get("text") or "").strip() for p in prof if (p.get("text") or "").strip()]
    else:
        raise ValueError(task)

    if task == "SD-science":
        tail = sd_self_distill.sd_m2_preserving_tail(sample, task=task)
        sep = "\n\n"
        tok = tokenizer
        tail_ids = tok(tail, add_special_tokens=False, verbose=False)["input_ids"]
        sep_ids = tok(sep, add_special_tokens=False, verbose=False)["input_ids"]
        min_doc_tokens = 16
        if len(tail_ids) + len(sep_ids) + min_doc_tokens > max_tokens:
            keep = max(64, max_tokens - len(sep_ids) - min_doc_tokens)
            tail = tok.decode(tail_ids[-keep:], skip_special_tokens=True)
            tail_ids = tok(tail, add_special_tokens=False, verbose=False)["input_ids"]
        budget = max_tokens - len(tail_ids) - len(sep_ids)
        budget = max(0, budget)
        text_parts: list[str] = []
        for chunk in reversed(hist_chunks):
            ids = tok(chunk, add_special_tokens=False, verbose=False)["input_ids"]
            if len(ids) > budget:
                if budget <= 0:
                    break
                chunk = tok.decode(ids[-budget:], skip_special_tokens=True)
                text_parts.append(chunk)
                break
            text_parts.append(chunk)
            budget -= len(ids)
        history = "\n".join(reversed(text_parts))
        # Read natural order: preserved system / task prefix first, then profile lines.
        return (tail + sep + history).strip() if history.strip() else tail

    tail = "\n\nNow personalize for this instance:\n" + sample["input"]
    budget = max_tokens - len(
        tokenizer(tail, add_special_tokens=False, verbose=False)["input_ids"]
    )
    text_parts: list[str] = []
    for chunk in reversed(hist_chunks):
        ids = tokenizer(chunk, add_special_tokens=False, verbose=False)["input_ids"]
        if len(ids) > budget:
            chunk = tokenizer.decode(ids[-budget:], skip_special_tokens=True)
            text_parts.append(chunk)
            break
        text_parts.append(chunk)
        budget -= len(ids)
    history = "\n".join(reversed(text_parts))
    return history + tail
