"""
**Outer loop** (meta-learning) for TTT-E2E-style training, aligned with Sun et al. (arXiv:2512.23675).

The paper trains with an **inner** loop (TTT on NTP over a prefix) nested inside an **outer** loop that
minimizes the **post-inner** loss—implemented with *gradients of gradients* in the official JAX repo
`https://github.com/test-time-training/e2e`.

This file provides a **PyTorch** surrogate for **GPT-2** causal LMs:

* Inner: one (or few) gradient steps on **last-fraction MLP weights only**, on NTP loss over a
  *support* span, with ``create_graph=True`` so the step is differentiable.
* Outer: NTP loss on a *query* span evaluated via ``torch.func.functional_call`` with the **adapted**
  MLP tensors, so ``loss_outer.backward()`` matches a **single-step** bilevel chain (same structure as
  MAML-style meta-objectives discussed in the paper).

**Scope:** T5/encoder–decoder is not wired through ``functional_call`` here; use the JAX reference or
extend this module if you need full parity on seq2seq.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from transformers import GPT2LMHeadModel

from .e2e import (
    _causal_lm_loss_on_ids,
    collect_gpt2_mlp_params,
    dynamic_param_names_in_order,
    train_only_selected_ffn,
)

try:
    from torch.func import functional_call
except ImportError:  # pragma: no cover
    functional_call = None  # type: ignore[misc, assignment]


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


def bilevel_gpt2_ntp_k1_loss(
    model: GPT2LMHeadModel,
    tokenizer: "PreTrainedTokenizerBase",
    *,
    support_text: str,
    query_text: str,
    device: torch.device,
    layer_fraction: float = 0.25,
    inner_lr: float = 1e-4,
    max_seq_len: int = 512,
) -> torch.Tensor:
    """
    Scalar **outer** loss = query NTP after **one** inner SGD step on support NTP (MLP subset only).

    Matches the paper's bilevel *shape*: outer minimizes loss after inner TTT; implementation is the
    standard single-step implicit-gradient chain (see Finn et al., MAML; paper cites nested grads).
    """
    if functional_call is None:
        raise RuntimeError(
            "bilevel_gpt2_ntp_k1_loss requires PyTorch 2.0+ (``torch.func.functional_call``). "
            "For full TTT-E2E pretraining use the official JAX repo test-time-training/e2e."
        )

    dyn = collect_gpt2_mlp_params(model, layer_fraction=layer_fraction)
    if not dyn:
        raise RuntimeError("No MLP parameters collected for bilevel inner step.")

    names = dynamic_param_names_in_order(model, dyn)

    enc_s = tokenizer(support_text, truncation=True, max_length=max_seq_len, return_tensors="pt")
    input_ids_s = enc_s["input_ids"].to(device)
    attn_s = enc_s.get("attention_mask")
    if attn_s is not None:
        attn_s = attn_s.to(device)
    if input_ids_s.shape[1] < 2:
        return torch.zeros((), device=device, requires_grad=True)

    enc_q = tokenizer(query_text, truncation=True, max_length=max_seq_len, return_tensors="pt")
    input_ids_q = enc_q["input_ids"].to(device)
    attn_q = enc_q.get("attention_mask")
    if attn_q is not None:
        attn_q = attn_q.to(device)
    if input_ids_q.shape[1] < 2:
        return torch.zeros((), device=device, requires_grad=True)

    with train_only_selected_ffn(model, dyn):
        loss_inner = _causal_lm_loss_on_ids(model, input_ids_s, attn_s)
        grads = torch.autograd.grad(
            loss_inner,
            dyn,
            create_graph=True,
            retain_graph=False,
            allow_unused=False,
        )

    adapted = {n: p - inner_lr * g for n, p, g in zip(names, dyn, grads)}

    labels_q = input_ids_q.clone()
    if attn_q is not None:
        labels_q = labels_q.masked_fill(attn_q == 0, -100)

    # Outside ``train_only_selected_ffn`` so non-MLP weights recover ``requires_grad`` and receive outer grads.
    fc_kw: dict = {"input_ids": input_ids_q, "labels": labels_q}
    if attn_q is not None:
        fc_kw["attention_mask"] = attn_q
    out_q = functional_call(model, adapted, args=(), kwargs=fc_kw)
    if out_q.loss is None:
        return torch.zeros((), device=device, requires_grad=True)
    return out_q.loss
