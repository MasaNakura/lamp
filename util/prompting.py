"""Prompt construction: LaMP RAG prompts + long-context ICL concatenation."""
from __future__ import annotations

from typing import Any, Callable

import torch

from util.lamp_paths import ensure_lamp_on_path

ensure_lamp_on_path()

from prompts.prompts import create_prompt_generator  # noqa: E402

from util import sd_self_distill  # noqa: E402


def _sd_m2_encoder_max_length(
    tokenizer,
    cli_max: int,
    *,
    model=None,
    hard_cap: int = 8192,
) -> int:
    """
    Self-distillation M2 (ICL): encoder token budget implied by the CLI cap,
    ``tokenizer.model_max_length``, and (when ``model`` is set) seq2seq config fields.
    Shared by ``train.py`` (``--prompt_style icl``) and ``run_evaluate.py`` (M2).
    """
    candidates: list[int] = [cli_max]
    mml = getattr(tokenizer, "model_max_length", None)
    if isinstance(mml, int) and 128 <= mml < 1_000_000:
        candidates.append(mml)
    if model is not None:
        inner = model.get_base_model() if hasattr(model, "get_base_model") else model
        cfg = getattr(inner, "config", None)
        if cfg is not None:
            for name in ("max_source_positions", "n_positions", "max_position_embeddings"):
                v = getattr(cfg, name, None)
                if isinstance(v, int) and v > 0:
                    candidates.append(v)
    return min(hard_cap, max(candidates))


def icl_m2_max_encoder_tokens(
    task: str,
    tokenizer,
    cli_max: int,
    model=None,
    *,
    architecture: str = "seq2seq",
) -> int:
    """
    Token cap for M2 / ``--prompt_style icl`` encoder text (matches ``run_evaluate`` M2).
    """
    if architecture != "seq2seq":
        return cli_max
    if task in ("SD-tooluse", "SD-science"):
        return _sd_m2_encoder_max_length(tokenizer, cli_max, model=model)
    return cli_max


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
    cache_dir: str | None = None,
    device: torch.device | None = None,
) -> Callable[[dict[str, Any]], str]:
    if task in ("SD-tooluse", "SD-science"):
        dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
                cache_dir=cache_dir,
            )

        return one_sd

    internal = task_internal_name(task)
    gen, _contriever = create_prompt_generator(
        num_retrieved, retriever, ranked, max_length, tokenizer
    )

    def one(sample: dict[str, Any]) -> str:
        return gen(sample["input"], sample["profile"], internal)

    return one


def m3_rag_prompt_and_contriever(
    task: str,
    tokenizer,
    *,
    num_retrieved: int,
    retriever: str,
    ranked: bool,
    max_length: int,
    device: torch.device,
    cache_dir: str | None = None,
) -> tuple[Callable[[dict[str, Any]], str], Any]:
    """
    Same encoder-side RAG prompt as ``run_evaluate.py`` mode **m3** and ``train.py`` global
    LoRA (``--prompt_style rag``): one callable ``row -> str`` plus optional Contriever module
    for LaMP (move to device before forward).
    """
    if task in ("SD-tooluse", "SD-science"):
        fn = build_rag_prompt_fn(
            task,
            tokenizer,
            num_retrieved=num_retrieved,
            retriever=retriever,
            ranked=ranked,
            max_length=max_length,
            cache_dir=cache_dir,
            device=device,
        )
        return fn, None
    gen, contriever = create_prompt_generator(
        num_retrieved, retriever, ranked, max_length, tokenizer
    )
    internal = task_internal_name(task)

    def rag_row(row: dict[str, Any]) -> str:
        return gen(row["input"], row["profile"], internal)

    return rag_row, contriever


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
    **SD-tooluse / SD-science:** use ``input`` unchanged (only ``\\r``/``\\r\\n`` → ``\\n``)
    when it fits ``max_tokens``. **Tool-use** over budget: split at
    ``\\n\\nUse the following format:\\n``; keep the right segment verbatim and
    right-truncate the left segment using an **exact character prefix** of the raw string so
    newlines match; token counting only decides how much of that prefix to keep. **Science:** drop from the start until the suffix fits.
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
        return sd_self_distill.sd_m2_icl_encoder_from_raw_input(
            sample.get("input") or "", tokenizer, max_tokens, task=task
        )
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


def icl_m2_encoder_text(
    sample: dict[str, Any],
    tokenizer,
    *,
    task: str,
    max_input_length: int,
    model=None,
    architecture: str = "seq2seq",
    reserve_for_input: int = 128,
) -> str:
    """
    **Single entry point** for M2 ICL encoder strings and ``train.py --prompt_style icl``:
    same token cap and ``build_icl_source`` path as ``run_evaluate.py`` mode ``m2``.
    """
    cap = icl_m2_max_encoder_tokens(
        task, tokenizer, max_input_length, model=model, architecture=architecture
    )
    return build_icl_source(
        sample,
        tokenizer,
        task=task,
        max_tokens=cap,
        reserve_for_input=reserve_for_input,
    )
