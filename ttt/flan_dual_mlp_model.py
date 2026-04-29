"""Flan-T5 wrapper with Dual-FFN blocks for TTT-E2E-style inner updates.

This mirrors the GPT-2 ``TTTGPT2`` split:
* ``trainable`` branch = fast inner-loop weights
* ``static`` branch = frozen anchor branch
"""
from __future__ import annotations

import copy
from typing import Iterator

import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, PreTrainedModel


class DualFFNCore(nn.Module):
    """Two FFN branches with summed outputs."""

    def __init__(self, base_ffn_core: nn.Module):
        super().__init__()
        self.trainable = copy.deepcopy(base_ffn_core)
        self.static = copy.deepcopy(base_ffn_core)
        for p in self.static.parameters():
            p.requires_grad = False

    def forward(self, hidden_states):
        return self.trainable(hidden_states) + self.static(hidden_states)


class TTTFlanT5(nn.Module):
    """T5/Flan model whose last-layer-fraction FFNs use ``DualFFNCore``."""

    def __init__(
        self,
        model_name: str = "google/flan-t5-small",
        ttt_fraction: float = 0.25,
        *,
        cache_dir: str | None = None,
        torch_dtype: torch.dtype | None = None,
    ):
        super().__init__()
        load_kw: dict = {"cache_dir": cache_dir}
        if torch_dtype is not None:
            load_kw["torch_dtype"] = torch_dtype
        self.lm = AutoModelForSeq2SeqLM.from_pretrained(model_name, **load_kw)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, use_fast=False)

        self._replace_ffn_cores(ttt_fraction)
        self._mark_param_roles()

    def _iter_selected_blocks(self, blocks, frac: float):
        n = len(blocks)
        k = max(1, int(round(n * frac)))
        start = n - k
        return range(start, n)

    def _replace_ffn_cores(self, frac: float) -> None:
        base: PreTrainedModel = self.lm
        core = base.get_base_model() if hasattr(base, "get_base_model") else base
        if hasattr(core, "base_model"):
            core = core.base_model
        enc_blocks = core.encoder.block
        dec_blocks = core.decoder.block

        self.ttt_encoder_block_indices: list[int] = []
        self.ttt_decoder_block_indices: list[int] = []

        for idx in self._iter_selected_blocks(enc_blocks, frac):
            ff_layer = enc_blocks[idx].layer[1]
            ff_layer.DenseReluDense = DualFFNCore(ff_layer.DenseReluDense)
            self.ttt_encoder_block_indices.append(idx)

        for idx in self._iter_selected_blocks(dec_blocks, frac):
            ff_layer = dec_blocks[idx].layer[2]
            ff_layer.DenseReluDense = DualFFNCore(ff_layer.DenseReluDense)
            self.ttt_decoder_block_indices.append(idx)

    def _mark_param_roles(self) -> None:
        for name, p in self.named_parameters():
            if ".DenseReluDense.trainable." in name:
                p._ttt_role = "inner"
            else:
                p._ttt_role = "outer"

    def inner_params(self) -> Iterator[nn.Parameter]:
        for p in self.parameters():
            if getattr(p, "_ttt_role", "outer") == "inner":
                yield p

    def outer_params(self) -> Iterator[nn.Parameter]:
        for p in self.parameters():
            if getattr(p, "_ttt_role", "outer") == "outer":
                yield p

    def snapshot_inner(self) -> list[torch.Tensor]:
        return [p.detach().clone() for p in self.inner_params()]

    def restore_inner(self, snapshot: list[torch.Tensor]) -> None:
        with torch.no_grad():
            for p, s in zip(self.inner_params(), snapshot):
                p.copy_(s.to(device=p.device, dtype=p.dtype))

    def forward(self, *args, **kwargs):
        return self.lm(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.lm.generate(*args, **kwargs)
