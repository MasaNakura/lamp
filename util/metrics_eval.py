"""Generation metrics aligned with LaMP/LaMP/metrics/generation_metrics.py."""
from __future__ import annotations

import json
import os
from typing import Any

from util.lamp_paths import ensure_lamp_on_path

ensure_lamp_on_path()

from metrics.generation_metrics import create_metric_bleu_rouge_meteor  # noqa: E402


def build_compute_metrics(tokenizer):
    return create_metric_bleu_rouge_meteor(tokenizer=tokenizer)


def evaluate_strings(preds: list[str], refs: list[str]) -> dict[str, float]:
    """BLEU / ROUGE / METEOR without token ids (string-level)."""
    from metrics.generation_metrics import create_metric_bleu_rouge_meteor_chatgpt

    metric = create_metric_bleu_rouge_meteor_chatgpt()
    return metric(preds, refs)


def make_per_example_string_metric():
    """Return ``score(pred, ref) -> dict`` sharing one load of sacrebleu/rouge/meteor (for verbose logging)."""
    from metrics.generation_metrics import create_metric_bleu_rouge_meteor_chatgpt

    metric = create_metric_bleu_rouge_meteor_chatgpt()

    def score_one(pred: str, ref: str) -> dict[str, float]:
        return metric([pred], [ref])

    return score_one


def write_lamp_predictions(task_underscore: str, pairs: list[tuple[str, str]], out_path: str) -> None:
    """
    Write {"task": "LaMP_5", "golds": [{"id": ..., "output": ...}]} style predictions
    (same shape as LaMP leaderboard / eval README).
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    payload: dict[str, Any] = {
        "task": task_underscore,
        "golds": [{"id": i, "output": o} for i, o in pairs],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
