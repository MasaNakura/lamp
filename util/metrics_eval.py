"""Generation metrics aligned with LaMP/LaMP/metrics/generation_metrics.py."""
from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

from util.lamp_paths import ensure_lamp_on_path

ensure_lamp_on_path()

from metrics.generation_metrics import create_metric_bleu_rouge_meteor  # noqa: E402


def _sanitize_prediction_token_ids(preds, tokenizer):
    """
    Avoid Rust ``tokenizer.batch_decode`` OverflowError on invalid ids (e.g. logits
    as 3D array, negatives, or values past ``len(tokenizer) - 1``). LaMP's metric
    helper stays unchanged; we normalize here before delegating to it.
    """
    x = np.asarray(preds)
    if x.size == 0:
        return x.astype(np.int64)
    if x.ndim == 3:
        x = np.argmax(x, axis=-1)
    elif x.ndim != 2:
        raise ValueError(f"Expected predictions of rank 2 or 3, got shape {x.shape}")
    if np.issubdtype(x.dtype, np.floating):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = np.rint(x).astype(np.int64, copy=False)
    else:
        x = x.astype(np.int64, copy=False)
    pad = tokenizer.pad_token_id
    if pad is None:
        pad = 0
    pad = int(pad)
    max_id = max(0, len(tokenizer) - 1)
    x = np.where(x < 0, pad, x)
    x = np.where(x > max_id, pad, x)
    return x


def build_compute_metrics(tokenizer):
    inner = create_metric_bleu_rouge_meteor(tokenizer=tokenizer)

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = _sanitize_prediction_token_ids(preds, tokenizer)
        return inner((preds, labels))

    return compute_metrics


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


def write_lamp_predictions(task_underscore: str, pairs: list[tuple[Any, str]], out_path: str) -> None:
    """
    Write leaderboard gold-style predictions (LaMP README / ``eval/eval_task.py``):

        {"task": "LaMP_5", "golds": [{"id": ..., "output": ...}, ...]}

    Same schema as official ``*_outputs.json`` gold files; ``id`` types match the
    values passed in ``pairs`` (typically same as question rows).
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    payload: dict[str, Any] = {
        "task": task_underscore,
        "golds": [{"id": i, "output": o} for i, o in pairs],
    }
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, ensure_ascii=False, indent="\t")
        f.write("\n")
