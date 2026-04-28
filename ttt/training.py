"""
Test-time training on user history without using the held-out target.

Uses seq2seq examples mined from profile fields only (LaMP-5: title/abstract;
LaMP-7: tweet text, with cross-tweet style prompts when multiple tweets exist),
remaining label-free for the actual test query.
"""
from __future__ import annotations

from itertools import cycle
from typing import Any

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSeq2SeqLM


class ProfileSFTDataset(Dataset):
    def __init__(self, pairs: list[tuple[str, str]]):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int):
        s, t = self.pairs[idx]
        return {"source": s, "target": t}


def build_profile_training_pairs(task: str, profile: list[dict[str, Any]]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    if task == "LaMP-5":
        for p in profile:
            title = (p.get("title") or "").strip()
            abstract = (p.get("abstract") or "").strip()
            if not title or not abstract:
                continue
            src = (
                "Generate a concise scholarly paper title for the abstract below.\n\n"
                f"Abstract: {abstract}\n\nTitle:"
            )
            tgt = title
            pairs.append((src, tgt))
    elif task == "LaMP-7":
        texts = [(p.get("text") or "").strip() for p in profile if (p.get("text") or "").strip()]
        for i, tgt in enumerate(texts):
            others = [t for j, t in enumerate(texts) if j != i][:8]
            if others:
                ob = "\n".join(f"- {x}" for x in others)
                src = (
                    "Past tweets from the same user (style only; do not copy them verbatim):\n"
                    f"{ob}\n\n"
                    "Write one new tweet in the same voice, tone, and length habits.\n\nTweet:"
                )
                pairs.append((src, tgt))
            else:
                src = (
                    "Write a new tweet in the same user's voice and style as the reference below.\n\n"
                    f"Reference tweet: {tgt}\n\nNew tweet:"
                )
                pairs.append((src, tgt))
    else:
        raise ValueError(task)
    return pairs


def run_ttt_steps(
    model: AutoModelForSeq2SeqLM,
    tokenizer,
    *,
    task: str,
    profile: list[dict[str, Any]],
    device: torch.device,
    max_input_length: int,
    micro_batch_size: int = 2,
    steps: int = 50,
    lr: float = 1e-4,
) -> None:
    pairs = build_profile_training_pairs(task, profile)
    if not pairs:
        return

    ds = ProfileSFTDataset(pairs)
    dl = DataLoader(ds, batch_size=micro_batch_size, shuffle=True, drop_last=False)
    if len(dl) == 0:
        return

    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = AdamW(trainable, lr=lr, weight_decay=0.0)

    stream = cycle(dl)
    for _ in range(steps):
        batch = next(stream)
        model_inputs = tokenizer(
            batch["source"],
            text_target=batch["target"],
            truncation=True,
            max_length=max_input_length,
            padding=True,
            return_tensors="pt",
        ).to(device)
        out = model(**model_inputs)
        loss = out.loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    model.eval()
    opt.zero_grad(set_to_none=True)
