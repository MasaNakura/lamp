"""
TTT-E2E-style **simulation** for LaMP (Sun et al., *End-to-End Test-Time Training for Long Context*,
`arXiv:2512.23675 <https://arxiv.org/abs/2512.23675>`_).

The official implementation is JAX in `test-time-training/e2e` on GitHub; this module is a **PyTorch +
HuggingFace** helper library: **FFN subset collection**, **profile stream text**, **sliding token windows**,
and **GPT-2 NTP loss** (for ``ttt/outer_meta.py``). **Flan-T5** sliding inner TTT lives in ``ttt/flan_inner.py``.

**Backbones**

* **Causal LM (GPT-2):** ``ttt/mam_inner.py`` + ``ttt/mam_model.py`` (DualMLP); ``outer_meta`` uses ``_causal_lm_loss_on_ids`` here.
* **Seq2seq (Flan-T5):** ``ttt/flan_inner.py`` (single-pass sliding FFN inner from ``run_evaluate.py`` M6).

**Bilevel (paper):** meta-learning at *training* time optimizes the initialization so that post-inner NTP
loss is low; see ``ttt/outer_meta.py`` for a differentiable **K=1 inner-step** surrogate on GPT-2. **Eval**
(``run_evaluate.py`` m6) runs the **inner** loop only unless you load a checkpoint produced with that
meta stage.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

import torch
from transformers import GPT2LMHeadModel, PreTrainedModel


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
    ids = tokenizer(
        stream,
        add_special_tokens=False,
        return_attention_mask=False,
        verbose=False,
    )["input_ids"]
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
