"""
Paper-style **test-time** inner loop for **Flan-T5** (Sun et al., TTT-E2E, arXiv:2512.23675).

* **One pass** left-to-right over the **profile** token stream (``build_flat_history_stream``),
  optionally **truncated** to ``profile_token_cap`` tokens (same rule as causal M6:
  ``--m6_profile_max_tokens`` / ``--max_input_length``).
* **One SGD step per window** on the T5 objective with ``source == target ==`` window text.
* **Only** the last ``layer_fraction`` of **encoder and decoder FFN** weights are updated
  (``collect_inner_mlp_params`` in ``e2e.py``).

Decode uses each row’s ``input`` (LaMP instructions); that text is **not** in the inner stream.
"""
from __future__ import annotations

from typing import Any

import torch
from torch.optim import SGD
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from ttt.e2e import (
    build_flat_history_stream,
    collect_inner_mlp_params,
    iter_history_token_windows,
    train_only_selected_ffn,
)


@torch.enable_grad()
def inner_adapt_t5_sliding_profile(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    *,
    task: str,
    profile: list[dict[str, Any]],
    device: torch.device,
    lr: float,
    window: int,
    stride: int,
    profile_token_cap: int,
    layer_fraction: float = 0.25,
) -> None:
    """Single-pass sliding-window TTT on **profile only**; ``profile_token_cap`` bounds stream length."""
    inner = collect_inner_mlp_params(model, layer_fraction=layer_fraction)
    if not inner:
        return

    stream = build_flat_history_stream(task, profile)
    if not stream.strip():
        return

    cap = max(2, int(profile_token_cap))
    enc0 = tokenizer(stream, truncation=True, max_length=cap, add_special_tokens=False)
    ids0 = enc0["input_ids"]
    if not ids0 or len(ids0) < 2:
        return
    stream = tokenizer.decode(ids0, skip_special_tokens=True)
    if not stream.strip():
        return

    opt = SGD(inner, lr=lr)
    model.train()
    try:
        with train_only_selected_ffn(model, inner):
            for chunk_ids in iter_history_token_windows(
                tokenizer, stream, window=window, stride=stride
            ):
                if len(chunk_ids) < 2:
                    continue
                window_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
                if not window_text.strip():
                    continue
                batch = tokenizer(
                    [window_text],
                    text_target=[window_text],
                    truncation=True,
                    max_length=window,
                    padding=True,
                    return_tensors="pt",
                ).to(device)
                opt.zero_grad(set_to_none=True)
                out = model(**batch)
                loss = out.loss
                if loss is None or not torch.isfinite(loss):
                    continue
                loss.backward()
                opt.step()
    finally:
        model.eval()
        opt.zero_grad(set_to_none=True)
