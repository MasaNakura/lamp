"""
TTT-E2E-style **simulation** for LaMP (Sun et al., *End-to-End Test-Time Training for Long Context*,
`arXiv:2512.23675 <https://arxiv.org/abs/2512.23675>`_).

The official implementation is JAX in `test-time-training/e2e` on GitHub; this module is a **PyTorch +
HuggingFace** analogue. For **GPT-2** with paper-style **DualMLP + higher outer loop**, use ``ttt/mam_*.py``
and ``train_mam_meta.py``; this file keeps the **T5** path and generic helpers (e.g. ``build_flat_history_stream``).

**Backbones**

* **Causal LM (GPT-2):** inner loop is standard shifted next-token prediction on token windows; only the
  last ``layer_fraction`` of transformer blocks' **MLP** parameters are trainable during inner TTT.
* **Seq2seq (Flan-T5):** inner loop uses the same pseudo-seq2seq objectives as ``ttt/training.py``, with
  MLP-only updates on the last fraction of **encoder and decoder** FFN stacks.

**Bilevel (paper):** meta-learning at *training* time optimizes the initialization so that post-inner NTP
loss is low; see ``ttt/outer_meta.py`` for a differentiable **K=1 inner-step** surrogate on GPT-2. **Eval**
(``run_evaluate.py`` m6) runs the **inner** loop only unless you load a checkpoint produced with that
meta stage.
"""
from __future__ import annotations

from contextlib import contextmanager
from itertools import cycle
from typing import Any, Iterator

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, PreTrainedModel

from ttt.training import ProfileSFTDataset, build_profile_training_pairs


def backbone_kind(model: PreTrainedModel) -> str:
    if isinstance(model, GPT2LMHeadModel):
        return "gpt2"
    mt = getattr(getattr(model, "config", None), "model_type", "") or ""
    if mt in ("t5", "mt5"):
        return "t5"
    raise TypeError(
        f"TTT-E2E sim supports GPT2LMHeadModel or T5 seq2seq; got {type(model).__name__} (model_type={mt!r})."
    )


def _seq2seq_core(model: PreTrainedModel):
    m = model
    if hasattr(m, "get_base_model"):
        try:
            m = m.get_base_model()
        except Exception:
            pass
    if hasattr(m, "base_model"):
        m = m.base_model
    enc = getattr(m, "encoder", None)
    dec = getattr(m, "decoder", None)
    if enc is None or dec is None:
        raise TypeError("Expected encoder/decoder on seq2seq model for TTT-E2E T5 path.")
    return enc, dec


def collect_t5_encoder_ffn_params(model: PreTrainedModel, *, layer_fraction: float = 0.25) -> list[torch.nn.Parameter]:
    enc, _ = _seq2seq_core(model)
    blocks = getattr(enc, "block", None)
    if blocks is None:
        return []
    n = len(blocks)
    k = max(1, int(n * layer_fraction))
    params: list[torch.nn.Parameter] = []
    for block in blocks[-k:]:
        ff = block.layer[1]
        params.extend(ff.parameters())
    return params


def collect_t5_decoder_ffn_params(model: PreTrainedModel, *, layer_fraction: float = 0.25) -> list[torch.nn.Parameter]:
    _, dec = _seq2seq_core(model)
    blocks = getattr(dec, "block", None)
    if blocks is None:
        return []
    n = len(blocks)
    k = max(1, int(n * layer_fraction))
    params: list[torch.nn.Parameter] = []
    for block in blocks[-k:]:
        ff = block.layer[2]
        params.extend(p for p in ff.parameters())
    return params


def collect_t5_ffn_params_union(model: PreTrainedModel, *, layer_fraction: float = 0.25) -> list[torch.nn.Parameter]:
    a = collect_t5_encoder_ffn_params(model, layer_fraction=layer_fraction)
    b = collect_t5_decoder_ffn_params(model, layer_fraction=layer_fraction)
    seen: set[int] = set()
    out: list[torch.nn.Parameter] = []
    for p in a + b:
        pid = id(p)
        if pid in seen:
            continue
        seen.add(pid)
        out.append(p)
    return out


def collect_gpt2_mlp_params(model: GPT2LMHeadModel, *, layer_fraction: float = 0.25) -> list[torch.nn.Parameter]:
    blocks = model.transformer.h
    n = len(blocks)
    k = max(1, int(n * layer_fraction))
    params: list[torch.nn.Parameter] = []
    for block in blocks[-k:]:
        params.extend(p for p in block.mlp.parameters())
    return params


def collect_inner_mlp_params(model: PreTrainedModel, *, layer_fraction: float = 0.25) -> list[torch.nn.Parameter]:
    kind = backbone_kind(model)
    if kind == "gpt2":
        return collect_gpt2_mlp_params(model, layer_fraction=layer_fraction)  # type: ignore[arg-type]
    return collect_t5_ffn_params_union(model, layer_fraction=layer_fraction)


def dynamic_param_names_in_order(model: PreTrainedModel, params: list[torch.nn.Parameter]) -> list[str]:
    id_to_name = {id(p): n for n, p in model.named_parameters()}
    return [id_to_name[id(p)] for p in params]


def snapshot_selected_params(params: list[torch.nn.Parameter]) -> list[torch.Tensor]:
    return [p.detach().clone() for p in params]


def restore_selected_params(params: list[torch.nn.Parameter], snap: list[torch.Tensor]) -> None:
    for p, s in zip(params, snap):
        p.data.copy_(s.to(device=p.device, dtype=p.dtype))


@contextmanager
def train_only_selected_ffn(model: PreTrainedModel, trainable: list[torch.nn.Parameter]):
    train_ids = {id(p) for p in trainable}
    backup: dict[int, bool] = {}
    try:
        for p in model.parameters():
            backup[id(p)] = p.requires_grad
            p.requires_grad = id(p) in train_ids
        yield
    finally:
        for p in model.parameters():
            p.requires_grad = backup.get(id(p), p.requires_grad)


def build_flat_history_stream(task: str, profile: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    if task == "LaMP-5":
        for p in profile:
            t = (p.get("title") or "").strip()
            a = (p.get("abstract") or "").strip()
            if t:
                parts.append(f"[title] {t}")
            if a:
                parts.append(f"[abstract] {a}")
    elif task == "LaMP-7":
        for p in profile:
            tx = (p.get("text") or "").strip()
            if tx:
                parts.append(f"[tweet] {tx}")
    else:
        raise ValueError(task)
    return "\n\n".join(parts)


def iter_history_token_windows(
    tokenizer,
    stream: str,
    *,
    window: int,
    stride: int,
) -> Iterator[list[int]]:
    ids = tokenizer(stream, add_special_tokens=False, return_attention_mask=False)["input_ids"]
    if not ids:
        return
    if stride <= 0:
        stride = window
    i = 0
    while i < len(ids):
        yield ids[i : i + window]
        i += stride


def _causal_lm_loss_on_ids(
    model: GPT2LMHeadModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    labels = input_ids.clone()
    if attention_mask is not None:
        labels = labels.masked_fill(attention_mask == 0, -100)
    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    if out.loss is None:
        return torch.tensor(0.0, device=input_ids.device, requires_grad=True)
    return out.loss


def run_sliding_history_ntp_pass_gpt2(
    model: GPT2LMHeadModel,
    tokenizer,
    stream: str,
    optimizer: AdamW,
    *,
    device: torch.device,
    window: int,
    stride: int,
    max_windows: int,
) -> None:
    if not stream.strip():
        return
    n = 0
    for chunk in iter_history_token_windows(tokenizer, stream, window=window, stride=stride):
        if not chunk or len(chunk) < 2:
            continue
        input_ids = torch.tensor([chunk], device=device, dtype=torch.long)
        with torch.enable_grad():
            loss = _causal_lm_loss_on_ids(model, input_ids, None)
            if not torch.isfinite(loss):
                continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        n += 1
        if n >= max_windows:
            break


def run_sliding_history_ntp_pass_t5(
    model: PreTrainedModel,
    tokenizer,
    stream: str,
    optimizer: AdamW,
    *,
    device: torch.device,
    window: int,
    stride: int,
    max_windows: int,
) -> None:
    """Self-supervised copy objective on each window (seq2seq NTP analogue)."""
    if not stream.strip():
        return
    text = stream
    n = 0
    for chunk_ids in iter_history_token_windows(tokenizer, text, window=window, stride=stride):
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
        with torch.enable_grad():
            out = model(**batch)
            loss = out.loss
            if loss is None or not torch.isfinite(loss):
                continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        n += 1
        if n >= max_windows:
            break


def run_sliding_history_ntp_pass(
    model: PreTrainedModel,
    tokenizer,
    stream: str,
    optimizer: AdamW,
    *,
    device: torch.device,
    window: int,
    stride: int,
    max_windows: int,
) -> None:
    kind = backbone_kind(model)
    if kind == "gpt2":
        run_sliding_history_ntp_pass_gpt2(
            model, tokenizer, stream, optimizer, device=device, window=window, stride=stride, max_windows=max_windows
        )
    else:
        run_sliding_history_ntp_pass_t5(
            model, tokenizer, stream, optimizer, device=device, window=window, stride=stride, max_windows=max_windows
        )


def _gpt2_phase_b_step(
    model: GPT2LMHeadModel,
    tokenizer,
    full_text: str,
    device: torch.device,
    max_length: int,
    optimizer: AdamW,
) -> None:
    enc = tokenizer(full_text, truncation=True, max_length=max_length, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attn = enc.get("attention_mask")
    if attn is not None:
        attn = attn.to(device)
    if input_ids.shape[1] < 2:
        return
    with torch.enable_grad():
        loss = _causal_lm_loss_on_ids(model, input_ids, attn)
        if not torch.isfinite(loss):
            return
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


def run_ttt_e2e_simulation(
    model: PreTrainedModel,
    tokenizer,
    *,
    task: str,
    profile: list[dict[str, Any]],
    device: torch.device,
    max_input_length: int,
    micro_batch_size: int = 2,
    steps: int = 50,
    lr: float = 1e-4,
    layer_fraction: float = 0.25,
    sliding_window: int | None = 256,
    sliding_stride: int = 128,
    max_sliding_windows: int = 16,
) -> None:
    """
    Inner-loop only: MLP-only Adam steps on (optional) sliding history NTP then profile pseudo-tasks.

    Matches the paper's *inner* recipe at a high level: freeze non-MLP stacks; update last-layer-fraction
    MLPs on next-token / self-supervised objectives. **Outer** meta-training is ``ttt.outer_meta``.
    """
    inner = collect_inner_mlp_params(model, layer_fraction=layer_fraction)
    if not inner:
        return

    opt = AdamW(inner, lr=lr, weight_decay=0.0)
    model.train()
    try:
        with train_only_selected_ffn(model, inner):
            if sliding_window is not None:
                stream = build_flat_history_stream(task, profile)
                if stream.strip():
                    run_sliding_history_ntp_pass(
                        model,
                        tokenizer,
                        stream,
                        opt,
                        device=device,
                        window=sliding_window,
                        stride=sliding_stride,
                        max_windows=max_sliding_windows,
                    )

            kind = backbone_kind(model)
            if kind == "gpt2":
                pairs = build_profile_training_pairs(task, profile)
                if not pairs:
                    return
                texts: list[str] = []
                for src, tgt in pairs:
                    texts.append(f"{src.strip()}\n{tgt.strip()}")
                stream_b = cycle(texts)
                for _ in range(steps):
                    t = next(stream_b)
                    _gpt2_phase_b_step(model, tokenizer, t, device, max_input_length, opt)
            else:
                pairs = build_profile_training_pairs(task, profile)
                if pairs:
                    ds = ProfileSFTDataset(pairs)
                    dl = DataLoader(ds, batch_size=micro_batch_size, shuffle=True, drop_last=False)
                    if len(dl) > 0:
                        stream_batches = cycle(dl)
                        for _ in range(steps):
                            batch = next(stream_batches)
                            model_inputs = tokenizer(
                                batch["source"],
                                text_target=batch["target"],
                                truncation=True,
                                max_length=max_input_length,
                                padding=True,
                                return_tensors="pt",
                            ).to(device)
                            with torch.enable_grad():
                                out = model(**model_inputs)
                                loss = out.loss
                                if loss is None or not torch.isfinite(loss):
                                    continue
                                opt.zero_grad(set_to_none=True)
                                loss.backward()
                                opt.step()
    finally:
        model.eval()
        opt.zero_grad(set_to_none=True)
