"""Inner loop: test-time NTP adaptation on inner (trainable MLP) parameters only.

Adapted from MAM with no logic changes.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _ce_next_token(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )


def _iter_windows(ids: torch.Tensor, window: int, stride: int):
    n = ids.size(-1)
    if n <= window:
        yield ids
        return
    start = 0
    while start < n:
        end = min(start + window, n)
        yield ids[..., start:end]
        if end == n:
            break
        start += stride


@torch.enable_grad()
def inner_adapt_inplace(
    model,
    context_ids: torch.Tensor,
    steps: int = 1,
    lr: float = 1e-3,
    window: int = 256,
    stride: int | None = None,
):
    """SGD on ``model.inner_params()`` with next-token CE over sliding windows of ``context_ids``."""
    if stride is None:
        stride = window
    inner_params = list(model.inner_params())
    opt = torch.optim.SGD(inner_params, lr=lr)

    model.train()
    for _ in range(steps):
        for window_ids in _iter_windows(context_ids, window, stride):
            if window_ids.size(-1) < 2:
                continue
            opt.zero_grad()
            out = model(window_ids)
            loss = _ce_next_token(out.logits, window_ids)
            loss.backward()
            opt.step()
    model.eval()
    return model


def inner_adapt_functional(
    fmodel,
    diffopt,
    context_ids: torch.Tensor,
    steps: int = 1,
    window: int = 256,
    stride: int | None = None,
):
    """Differentiable inner loop for ``higher`` (outer meta-training)."""
    if stride is None:
        stride = window
    for _ in range(steps):
        for window_ids in _iter_windows(context_ids, window, stride):
            if window_ids.size(-1) < 2:
                continue
            out = fmodel(window_ids)
            loss = _ce_next_token(out.logits, window_ids)
            diffopt.step(loss)
    return fmodel
