"""TTT wrapper around pretrained GPT-2 (from MAM / TTT-E2E paper design).

Canonical copy lives under ``lamp/ttt/``; see ``ttt-MAM/README.md`` for the old layout.

* Last-fraction block MLPs become ``DualMLP`` (trainable inner branch + static outer anchor).
* ``inner_params`` / ``outer_params`` split matches the paper: inner = fast weights at test time.
"""
from __future__ import annotations

import copy
from typing import Iterator

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers.models.gpt2.modeling_gpt2 import GPT2MLP


class DualMLP(nn.Module):
    """Trainable MLP branch (inner TTT) + frozen static branch (outer meta); outputs summed."""

    def __init__(self, base_mlp: GPT2MLP):
        super().__init__()
        self.trainable = copy.deepcopy(base_mlp)
        self.static = copy.deepcopy(base_mlp)

    def forward(self, hidden_states):
        return self.trainable(hidden_states) + self.static(hidden_states)


class TTTGPT2(nn.Module):
    """GPT-2 with the last ``ttt_fraction`` of block MLPs replaced by ``DualMLP``."""

    def __init__(self, model_name: str = "gpt2", ttt_fraction: float = 0.25):
        super().__init__()
        self.lm = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        blocks = self.lm.transformer.h
        n_blocks = len(blocks)
        n_ttt = max(1, int(round(n_blocks * ttt_fraction)))
        self.ttt_block_indices = list(range(n_blocks - n_ttt, n_blocks))

        for idx in self.ttt_block_indices:
            blocks[idx].mlp = DualMLP(blocks[idx].mlp)

        self._mark_param_roles()

    def _mark_param_roles(self):
        for name, p in self.named_parameters():
            if ".mlp.trainable." in name:
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

    def forward(self, input_ids, labels=None, **kw):
        return self.lm(input_ids=input_ids, labels=labels, **kw)

    def generate(self, *args, **kw):
        return self.lm.generate(*args, **kw)
