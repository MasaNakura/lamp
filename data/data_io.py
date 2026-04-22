"""Load LaMP JSON lists and infer user keys for per-user evaluation (e.g. M4/M5 TTT)."""
from __future__ import annotations

import hashlib
import json
from typing import Any


def load_json_list(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON array of samples.")
    return data


def _unwrap_lamp_records(data: Any, path: str) -> list[dict[str, Any]]:
    """
    LaMP downloads use either a bare JSON list of samples or a leaderboard-style
    object: {"task": "LaMP_5", "golds": [ ... ]} (see LaMP README / eval_task.py).
    """
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("golds"), list):
        return data["golds"]
    raise ValueError(
        f"{path} must be a JSON array of objects or an object with a 'golds' array."
    )


def merge_questions_and_outputs(questions_path: str, outputs_path: str) -> list[dict[str, Any]]:
    """
    Build rows like upstream ``GeneralSeq2SeqDataset`` expects: each item has at least
    ``id``, ``input``, ``output``, and usually ``profile``.

    Official LaMP splits store **inputs** (``train_questions.json`` / ``*_questions.json``)
    and **labels** (``train_outputs.json`` / ``*_outputs.json``) in separate files; see
    ``LaMP/LaMP/utils/merge_with_rank.py`` and ``LaMP/data/avocado/create_avocado_dataset.py``.
    """
    with open(questions_path, encoding="utf-8") as f:
        q_raw = json.load(f)
    with open(outputs_path, encoding="utf-8") as f:
        o_raw = json.load(f)

    questions = _unwrap_lamp_records(q_raw, questions_path)
    outputs = _unwrap_lamp_records(o_raw, outputs_path)

    out_by_id: dict[str, str] = {}
    for o in outputs:
        if "id" not in o or "output" not in o:
            raise ValueError(
                f"Each entry in {outputs_path} must include 'id' and 'output' (after unwrapping 'golds')."
            )
        out_by_id[str(o["id"])] = o["output"]

    merged: list[dict[str, Any]] = []
    missing: list[str] = []
    for row in questions:
        if "id" not in row:
            raise ValueError(f"Each question row in {questions_path} must include 'id'.")
        rid = str(row["id"])
        if rid not in out_by_id:
            missing.append(rid)
            continue
        m = dict(row)
        m["output"] = out_by_id[rid]
        merged.append(m)

    if missing:
        raise KeyError(
            f"No matching output for {len(missing)} question id(s) (showing up to 10): {missing[:10]}"
        )

    return merged


def _profile_fingerprint(profile: list[dict[str, Any]]) -> str:
    keys: list[str] = []
    for item in profile:
        if "id" in item:
            keys.append(str(item["id"]))
        else:
            keys.append(
                hashlib.sha256(
                    json.dumps(item, sort_keys=True, ensure_ascii=False).encode("utf-8")
                ).hexdigest()
            )
    raw = json.dumps(sorted(keys), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def infer_user_id(row: dict[str, Any], user_field: str | None = None) -> str:
    """
    Stable user key for grouping test rows (M4/M5: TTT + reset per user).

    Prefer an explicit field (`user_field`, or `user_id` / `user` / `author_id` / `author`).
    Otherwise fingerprint the profile. Official user-based splits already keep users
    out of train; this only partitions the **test** file you pass to `evaluate.py`.
    """
    if user_field and user_field in row:
        return str(row[user_field])
    for k in ("user_id", "user", "author_id", "author"):
        if k in row:
            return str(row[k])
    prof = row.get("profile") or []
    if isinstance(prof, list) and prof:
        return _profile_fingerprint(prof)
    rid = str(row.get("id", ""))
    if "_" in rid:
        return rid.split("_", 1)[0]
    return rid or "unknown_user"
