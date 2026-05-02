"""Inner adaptation loops for Flan-T5 Dual-FFN TTT-E2E."""
from __future__ import annotations

import torch

from ttt.e2e import build_flat_history_stream, iter_history_token_windows


def _cap_lm_seq_len(tokenizer, max_length: int) -> int:
    """T5/Flan: ``max_length`` alone does not cap ``labels``; also respect model max positions."""
    mm = getattr(tokenizer, "model_max_length", None)
    if mm is not None and mm < 100_000:
        return min(int(max_length), int(mm))
    return int(max_length)


def _tokenize_self_supervised_batch(
    tokenizer, text: str, device: torch.device, max_length: int
) -> dict[str, torch.Tensor]:
    cap = _cap_lm_seq_len(tokenizer, max_length)
    batch = tokenizer(
        [text],
        text_target=[text],
        truncation=True,
        max_length=cap,
        max_target_length=cap,
        padding=True,
        return_tensors="pt",
    )
    return {k: v.to(device) for k, v in batch.items()}


def _window_text_loss(model, tokenizer, window_ids: list[int], device: torch.device, window: int):
    text = tokenizer.decode(window_ids, skip_special_tokens=True)
    if not text.strip():
        return None
    batch = _tokenize_self_supervised_batch(tokenizer, text, device, window)
    out = model(**batch)
    return out.loss


@torch.enable_grad()
def inner_adapt_t5_inplace(
    model,
    tokenizer,
    *,
    task: str,
    profile: list[dict],
    device: torch.device,
    lr: float = 1e-4,
    window: int = 256,
    stride: int | None = None,
    profile_token_cap: int = 4096,
):
    """Single-pass sliding update on profile stream (one step per window)."""
    if stride is None:
        stride = window
    inner_params = list(model.inner_params())
    opt = torch.optim.SGD(inner_params, lr=lr)

    stream = build_flat_history_stream(task, profile)
    if not stream.strip():
        model.eval()
        return model

    cap = max(2, int(profile_token_cap))
    ids = tokenizer(stream, truncation=True, max_length=cap, add_special_tokens=False)["input_ids"]
    if not ids or len(ids) < 2:
        model.eval()
        return model
    stream = tokenizer.decode(ids, skip_special_tokens=True)
    if not stream.strip():
        model.eval()
        return model

    model.train()
    for window_ids in iter_history_token_windows(tokenizer, stream, window=window, stride=stride):
        if len(window_ids) < 2:
            continue
        loss = _window_text_loss(model, tokenizer, window_ids, device, window)
        if loss is None or not torch.isfinite(loss):
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    model.eval()
    return model


def inner_adapt_t5_functional(
    fmodel,
    diffopt,
    tokenizer,
    context_ids: torch.Tensor,
    *,
    window: int = 256,
    stride: int | None = None,
):
    """Differentiable single-pass inner loop over tokenized context ids."""
    if stride is None:
        stride = window
    ids = context_ids.detach().view(-1).tolist()
    n = len(ids)
    i = 0
    while i < n:
        chunk = ids[i : i + window]
        if len(chunk) >= 2:
            text = tokenizer.decode(chunk, skip_special_tokens=True)
            if text.strip():
                batch = _tokenize_self_supervised_batch(
                    tokenizer, text, context_ids.device, window
                )
                out = fmodel(**batch)
                loss = out.loss
                if loss is not None and torch.isfinite(loss):
                    diffopt.step(loss)
        if i + window >= n:
            break
        i += stride
    return fmodel
