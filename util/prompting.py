"""Prompt construction: LaMP RAG prompts + long-context ICL concatenation."""
from __future__ import annotations

from typing import Any, Callable

from util.lamp_paths import ensure_lamp_on_path

ensure_lamp_on_path()

from prompts.prompts import create_prompt_generator  # noqa: E402


def task_internal_name(task: str) -> str:
    if task in ("LaMP-5", "LaMP-7"):
        return task
    raise ValueError("This experiment harness supports LaMP-5 and LaMP-7 only.")


def build_rag_prompt_fn(
    task: str,
    tokenizer,
    *,
    num_retrieved: int,
    retriever: str = "bm25",
    ranked: bool = False,
    max_length: int = 512,
) -> Callable[[dict[str, Any]], str]:
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
    Model 2 (ICL): concatenate serialized profile history, then the task input.
    Truncates from the left of history to respect a Flan-T5-style length budget.
    """
    prof = sample.get("profile") or []
    if task == "LaMP-5":
        hist_chunks = [
            f'History paper: title "{p.get("title", "")}" abstract: {p.get("abstract", "")}'
            for p in prof
        ]
    elif task == "LaMP-7":
        hist_chunks = [f'History tweet: "{p.get("text", "")}"' for p in prof]
    else:
        raise ValueError(task)

    tail = "\n\nNow personalize for this instance:\n" + sample["input"]
    budget = max_tokens - len(tokenizer(tail)["input_ids"])
    text_parts: list[str] = []
    for chunk in reversed(hist_chunks):
        ids = tokenizer(chunk)["input_ids"]
        if len(ids) > budget:
            chunk = tokenizer.decode(ids[-budget:], skip_special_tokens=True)
            text_parts.append(chunk)
            break
        text_parts.append(chunk)
        budget -= len(ids)
    history = "\n".join(reversed(text_parts))
    return history + tail
