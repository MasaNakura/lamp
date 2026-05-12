"""Inner adaptation loops for Flan-T5 Dual-FFN TTT-E2E."""
from __future__ import annotations

import torch

from ttt.e2e import build_flat_history_stream, iter_history_token_id_windows
from ttt.lamp_profile_rag import LampProfileRAG


def _seq2seq_positions_cap(model) -> int | None:
    """Read explicit position cap from config when present (e.g. ``n_positions`` on Flan-T5)."""
    if model is None:
        return None
    lm = getattr(model, "lm", model)
    base = lm.get_base_model() if hasattr(lm, "get_base_model") else lm
    cfg = getattr(base, "config", None)
    if cfg is None:
        return None
    for name in ("n_positions", "max_position_embeddings", "max_source_positions"):
        v = getattr(cfg, name, None)
        if isinstance(v, int) and v > 0:
            return int(v)
    return None


def _config_looks_like_t5(cfg) -> bool:
    if cfg is None:
        return False
    mt = (getattr(cfg, "model_type", None) or "").lower()
    if mt in ("t5", "mt5"):
        return True
    archs = getattr(cfg, "architectures", None) or []
    if isinstance(archs, (list, tuple)):
        return any("T5" in str(a) for a in archs)
    return False


def resolve_seq2seq_token_cap(model, tokenizer, requested: int) -> int:
    """
    Hard cap for encoder/decoder token length on seq2seq forwards (T5/Flan, PEFT-wrapped, ``TTTFlanT5``).

    Some checkpoints omit ``n_positions``; standard Flan-T5 is still 512. ``--max_input_length``
    can be larger (e.g. SD M2), but the LM must not see more than this cap or HF warns / misbehaves.
    """
    cap = max(2, int(requested))
    mpos = _seq2seq_positions_cap(model)
    if mpos is not None:
        cap = min(cap, mpos)
    else:
        lm = getattr(model, "lm", model)
        base = lm.get_base_model() if hasattr(lm, "get_base_model") else lm
        cfg = getattr(base, "config", None)
        if _config_looks_like_t5(cfg):
            cap = min(cap, 512)
    mm = getattr(tokenizer, "model_max_length", None)
    if isinstance(mm, int) and 128 <= mm < 100_000:
        cap = min(cap, mm)
    return max(2, cap)


def _cap_lm_seq_len(tokenizer, max_length: int, *, model=None) -> int:
    """Clamp LM batch length for inner self-supervised steps."""
    if model is None:
        cap = int(max_length)
        mm = getattr(tokenizer, "model_max_length", None)
        if mm is not None and mm < 100_000:
            cap = min(cap, int(mm))
        return max(cap, 2)
    return resolve_seq2seq_token_cap(model, tokenizer, max_length)


def _tokenize_self_supervised_batch(
    tokenizer, text: str, device: torch.device, max_length: int, *, model=None
) -> dict[str, torch.Tensor]:
    cap = _cap_lm_seq_len(tokenizer, max_length, model=model)
    # One ``max_length`` for both sides (``text_target``); HF tokenizers no longer accept
    # ``max_target_length`` here (warning: keyword not recognized).
    batch = tokenizer(
        [text],
        text_target=[text],
        truncation=True,
        max_length=cap,
        padding=True,
        return_tensors="pt",
    )
    if model is not None:
        hard = resolve_seq2seq_token_cap(model, tokenizer, 1_000_000)
        if batch["input_ids"].shape[1] > hard:
            for key in ("input_ids", "attention_mask"):
                if key in batch:
                    batch[key] = batch[key][:, :hard]
            if "labels" in batch:
                batch["labels"] = batch["labels"][:, :hard]
    return {k: v.to(device) for k, v in batch.items()}


def _window_text_loss(ttt_model, tokenizer, window_ids: list[int], device: torch.device, window: int):
    text = tokenizer.decode(window_ids, skip_special_tokens=True)
    if not text.strip():
        return None
    batch = _tokenize_self_supervised_batch(tokenizer, text, device, window, model=ttt_model)
    out = ttt_model(**batch)
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
    profile_token_cap: int | None = None,
    profile_rag: LampProfileRAG | None = None,
    rag_query: str | None = None,
    ttt_stream_text: str | None = None,
):
    """Single-pass sliding update on profile stream (one step per window).

    ``profile_token_cap``: if ``None``, tokenize the **entire** flattened profile with
    ``truncation=False`` and slide over **all** token ids (only each forward is window-limited).
    If a positive int, keep only the **first** that many tokens before sliding (legacy / VRAM cap).

    ``ttt_stream_text``: if set, use this string as the inner-loop stream. For SD+RAG, the
    caller runs retrieval, then passes ``sd_ttt_inner_stream_text`` here and sets
    ``profile_rag=None``. For LaMP, leave unset and use ``profile`` / ``profile_rag`` as before.
    """
    if stride is None:
        stride = window
    inner_params = list(model.inner_params())
    opt = torch.optim.SGD(inner_params, lr=lr)

    if ttt_stream_text is not None:
        stream = (ttt_stream_text or "").strip()
    else:
        prof_use = profile
        if profile_rag is not None and rag_query is not None and (rag_query.strip()):
            picked = profile_rag.select(rag_query, profile)
            if picked:
                prof_use = picked
        stream = build_flat_history_stream(task, prof_use)
    if not stream.strip():
        model.eval()
        return model

    # Unbounded ``encode`` can exceed the LM's context (e.g. 591 > 512) and HF warns before
    # sliding windows run. Cap the **stream** tokenization to the same limit as inner forwards.
    stream_cap = resolve_seq2seq_token_cap(model, tokenizer, 1_000_000)
    ids = tokenizer.encode(
        stream,
        add_special_tokens=False,
        truncation=True,
        max_length=stream_cap,
    )
    if profile_token_cap is not None and int(profile_token_cap) > 0:
        ids = ids[: min(int(profile_token_cap), len(ids))]
    if not ids or len(ids) < 2:
        model.eval()
        return model

    eff_window = resolve_seq2seq_token_cap(model, tokenizer, int(window))
    eff_stride = min(int(stride), eff_window) if stride is not None else eff_window

    # eval(): disable dropout / stochastic depth while still backpropping inner FFNs (train() adds noise
    # and can destabilize short RAG streams).
    model.eval()
    for window_ids in iter_history_token_id_windows(ids, window=eff_window, stride=eff_stride):
        if len(window_ids) < 2:
            continue
        loss = _window_text_loss(model, tokenizer, window_ids, device, eff_window)
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
    eff_window = resolve_seq2seq_token_cap(fmodel, tokenizer, int(window))
    eff_stride = min(int(stride), eff_window)
    ids = context_ids.detach().view(-1).tolist()
    n = len(ids)
    i = 0
    while i < n:
        chunk = ids[i : i + eff_window]
        if len(chunk) >= 2:
            text = tokenizer.decode(chunk, skip_special_tokens=True)
            if text.strip():
                batch = _tokenize_self_supervised_batch(
                    tokenizer, text, context_ids.device, eff_window, model=fmodel
                )
                out = fmodel(**batch)
                loss = out.loss
                if loss is not None and torch.isfinite(loss):
                    diffopt.step(loss)
        if i + eff_window >= n:
            break
        i += eff_stride
    return fmodel
