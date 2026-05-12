"""Accuracy metrics for Self-Distillation tool-use and science (medical QA) tasks."""
from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any


def extract_actions(text: str) -> list[str]:
    return re.findall(r"Action:\s*(\w+)", text)


def extract_action_inputs(text: str) -> dict[str, Any]:
    json_blocks = re.findall(r"Action Input:\s*({.*?})", text, re.DOTALL)
    combined: dict[str, Any] = {}
    for block in json_blocks:
        try:
            combined.update(json.loads(block))
        except json.JSONDecodeError:
            continue
    return combined


def tooluse_correct(pred: str, golden_answer: list[dict[str, Any]]) -> bool:
    pred_actions = extract_actions(pred)
    pred_inputs = extract_action_inputs(pred)
    gt_actions = [item["Action"] for item in golden_answer]
    gt_inputs: dict[str, Any] = {}
    for item in golden_answer:
        try:
            gt_inputs.update(json.loads(item["Action_Input"]))
        except (json.JSONDecodeError, TypeError, KeyError):
            pass
    actions_match = Counter(pred_actions) == Counter(gt_actions)
    inputs_match = pred_inputs == gt_inputs
    return bool(actions_match and inputs_match)


def extract_xml_answer(text: str) -> str:
    if "<answer>" not in text:
        return text.strip()
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def science_correct(pred: str, gold_answer: str) -> bool:
    return extract_xml_answer(pred) == (gold_answer or "").strip()


def parse_tooluse_gold(ref: str) -> list[dict[str, Any]]:
    data = json.loads(ref)
    if not isinstance(data, list):
        raise ValueError("SD-tooluse gold must decode to a JSON list.")
    return data


def tooluse_accuracy(preds: list[str], refs: list[str]) -> tuple[list[int], float]:
    scores: list[int] = []
    for pred, ref in zip(preds, refs):
        try:
            gold = parse_tooluse_gold(ref)
            scores.append(1 if tooluse_correct(pred, gold) else 0)
        except (json.JSONDecodeError, ValueError, TypeError, KeyError):
            scores.append(0)
    acc = sum(scores) / max(1, len(scores))
    return scores, acc


def science_accuracy(preds: list[str], refs: list[str]) -> tuple[list[int], float]:
    scores = [1 if science_correct(p, r) else 0 for p, r in zip(preds, refs)]
    acc = sum(scores) / max(1, len(scores))
    return scores, acc
