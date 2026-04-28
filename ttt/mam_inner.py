"""Inner loop: test-time NTP on inner (trainable MLP) parameters.

Matches the TTT-E2E paper (Sun et al., arXiv:2512.23675) **test-time** picture: the model
**streams the given context once** (here: disjoint sliding chunks mimicking a long prompt),
taking **one gradient step per chunk** on next-token loss—no second pass over the same
tokens. (Mini-batch size and window sizes in the paper are large, e.g. k≈8K, b≈1K; this
repo uses smaller windows for LaMP-scale profiles.)

**Meta-training** still uses ``higher`` with this same single-pass inner on the support span.
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
    lr: float = 1e-3,
    window: int = 256,
    stride: int | None = None,
):
    """One left-to-right pass: one SGD step per sliding window over ``context_ids``."""
    if stride is None:
        stride = window
    inner_params = list(model.inner_params())
    opt = torch.optim.SGD(inner_params, lr=lr)

    model.train()
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
    window: int = 256,
    stride: int | None = None,
):
    """Differentiable single-pass inner loop for ``higher`` (same semantics as ``inner_adapt_inplace``)."""
    if stride is None:
        stride = window
    for window_ids in _iter_windows(context_ids, window, stride):
        if window_ids.size(-1) < 2:
            continue
        out = fmodel(window_ids)
        loss = _ce_next_token(out.logits, window_ids)
        diffopt.step(loss)
    return fmodel
