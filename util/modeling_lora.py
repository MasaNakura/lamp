"""Flan-T5 + LoRA helpers (global training and per-user reset for TTT)."""
from __future__ import annotations

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def load_base(model_name: str, cache_dir: str | None = None):
    kwargs = {}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **kwargs)
    return tokenizer, model


def attach_lora(model, *, r: int = 8, alpha: int = 32, dropout: float = 0.05):
    cfg = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=("q", "v"),
    )
    return get_peft_model(model, cfg)


def lora_state_snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    snap: dict[str, torch.Tensor] = {}
    for name, p in model.named_parameters():
        if "lora_" in name:
            snap[name] = p.detach().float().cpu().clone()
    return snap


def restore_lora_snapshot(model: torch.nn.Module, snap: dict[str, torch.Tensor]) -> None:
    for name, p in model.named_parameters():
        if name in snap:
            p.data.copy_(snap[name].to(dtype=p.dtype, device=p.device))


def merge_and_unload(model: PeftModel) -> torch.nn.Module:
    """Optional: merge LoRA into base for faster inference (not used for TTT reset flows)."""
    m = model.merge_and_unload()
    return m
