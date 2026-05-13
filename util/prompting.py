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
    **SD-tooluse / SD-science:** use ``input`` unchanged when its token length is within
    ``max_tokens``; otherwise keep the **prefix** (first ``max_tokens`` tokens), i.e.
    right-truncate the text (do not drop the beginning).
    """
    prof = sample.get("profile") or []
    if task == "LaMP-5":
        hist_chunks = [
            f'History paper: title "{p.get("title", "")}" abstract: {p.get("abstract", "")}'
            for p in prof
        ]
    elif task == "LaMP-7":
        hist_chunks = [f'History tweet: "{p.get("text", "")}"' for p in prof]
    elif task in ("SD-tooluse", "SD-science"):
        inp = (sample.get("input") or "").replace("\r\n", "\n").strip()
        return sd_self_distill.sd_m2_icl_encoder_from_raw_input(inp, tokenizer, max_tokens)
    else:
        raise ValueError(task)

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
