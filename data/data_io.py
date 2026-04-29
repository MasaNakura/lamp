"""Load LaMP JSON lists and infer user keys for per-user evaluation."""
from __future__ import annotations

import hashlib
import json
import re
import sys
from typing import Any

# Heuristic: LaMP-6 Avocado *questions* use raw corpus file names as `input` until you run
# ``create_avocado_dataset.py``. LaMP-5 release JSON should have full prompts + abstracts.
_PLACEHOLDER_ID_RE = re.compile(
    r"^[\w.\-]{6,120}$",
    re.ASCII,
)


def looks_like_file_id_placeholder(text: str) -> bool:
    """
    True if ``text`` looks like a lone corpus filename / id (not a natural-language LaMP prompt).
    """
    s = (text or "").strip()
    if not s or len(s) > 200:
        return False
    if "\n" in s:
        return False
    sl = s.lower()
    if any(
        w in sl
        for w in (
            "generate",
            "abstract",
            "title",
            "paper",
            "tweet",
            "scholarly",
            "instruction",
            "paraphrase",
            "voice",
        )
    ):
        return False
    if s.count(" ") >= 2:
        return False
    if s.endswith((".txt", ".json")) and _PLACEHOLDER_ID_RE.match(s):
        return True
    if re.fullmatch(r"[\d\-]+-[A-Za-z0-9_.]+\.txt", s):
        return True
    return False


def warn_if_rows_look_like_unexpanded_placeholders(
    rows: list[dict[str, Any]],
    *,
    task: str,
    context: str,
) -> None:
    """Print a clear stderr warning when JSON looks like id-only placeholders, not real text."""
    if not rows:
        return
    n = min(64, len(rows))
    sample = rows[:n]
    bad_in = sum(1 for r in sample if looks_like_file_id_placeholder(str(r.get("input", ""))))
    bad_out = sum(1 for r in sample if looks_like_file_id_placeholder(str(r.get("output", ""))))
    if bad_in < n * 0.4 and bad_out < n * 0.4:
        return
    print(
        "\n".join(
            [
                "",
                "*" * 78,
                f"WARNING ({context}): Many rows look like **file-id placeholders** (e.g. '*.txt')",
                f"  in `input` and/or `output`, not natural-language LaMP text for **{task}**.",
                "  The model will encode those short strings as-is; metrics can look artificially high",
                "  when prediction and gold are the same placeholder.",
                "",
                "  For **LaMP-5**, use the official benchmark JSON where `input` is the full task",
                "  string (instruction + abstract) and `output` is the gold title.",
                "  File-name-only fields usually mean Avocado-style **pre-merge** ids; see",
                "  `LaMP/data/avocado/create_avocado_dataset.py` and the LaMP README download notes.",
                "*" * 78,
                "",
            ]
        ),
        file=sys.stderr,
    )


def load_json_list(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON array of samples.")
    return data


def _lamp5_query_abstract_missing(inp: str) -> bool:
    """True if the LaMP-5 task string has no usable abstract (benchmark placeholder)."""
    t = (inp or "").strip()
    if not t:
        return True
    return bool(re.search(r"\bno abstract available\b", t, re.IGNORECASE))


def _lamp5_profile_item_has_abstract(item: dict[str, Any]) -> bool:
    raw = item.get("abstract")
    if raw is None:
        return False
    ab = str(raw).strip()
    if not ab:
        return False
    if re.fullmatch(r"no abstract available\.?", ab, re.IGNORECASE):
        return False
    return True


def _lamp7_query_missing(inp: str) -> bool:
    t = (inp or "").strip()
    if not t:
        return True
    return bool(re.search(r"\bno (tweet|text) available\b", t, re.IGNORECASE))


def _lamp7_profile_item_has_text(item: dict[str, Any]) -> bool:
    raw = item.get("text")
    if raw is None:
        return False
    txt = str(raw).strip()
    if not txt:
        return False
    if re.fullmatch(r"no (tweet|text) available\.?", txt, re.IGNORECASE):
        return False
    return True


def filter_invalid_lamp_samples(rows: list[dict[str, Any]], task: str) -> list[dict[str, Any]]:
    """
    Drop unusable LaMP rows and strip bad profile entries (no abstract / no tweet text).

    Used after merging questions + outputs so train and eval stay consistent.
    """
    if task == "LaMP-5":
        return _filter_lamp5_rows(rows)
    if task == "LaMP-7":
        return _filter_lamp7_rows(rows)
    return list(rows)


def _filter_lamp5_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    dropped_rows = 0
    removed_profile_items = 0
    for row in rows:
        if _lamp5_query_abstract_missing(str(row.get("input", ""))):
            dropped_rows += 1
            continue
        prof = row.get("profile")
        new_row = dict(row)
        if isinstance(prof, list):
            newp = [dict(p) for p in prof if isinstance(p, dict) and _lamp5_profile_item_has_abstract(p)]
            removed_profile_items += len(prof) - len(newp)
            new_row["profile"] = newp
        else:
            new_row["profile"] = prof if isinstance(prof, list) else []
        kept.append(new_row)
    if dropped_rows or removed_profile_items:
        print(
            "[data_io] LaMP-5: dropped "
            f"{dropped_rows} sample(s) whose `input` has no real abstract "
            f"(e.g. 'No abstract available'); removed {removed_profile_items} profile "
            f"paper(s) without a usable abstract. {len(rows)} → {len(kept)} rows.",
            file=sys.stderr,
        )
    return kept


def _filter_lamp7_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    dropped_rows = 0
    removed_profile_items = 0
    for row in rows:
        if _lamp7_query_missing(str(row.get("input", ""))):
            dropped_rows += 1
            continue
        prof = row.get("profile")
        new_row = dict(row)
        if isinstance(prof, list):
            newp = [dict(p) for p in prof if isinstance(p, dict) and _lamp7_profile_item_has_text(p)]
            removed_profile_items += len(prof) - len(newp)
            new_row["profile"] = newp
        else:
            new_row["profile"] = prof if isinstance(prof, list) else []
        kept.append(new_row)
    if dropped_rows or removed_profile_items:
        print(
            "[data_io] LaMP-7: dropped "
            f"{dropped_rows} sample(s) with unavailable query text; removed "
            f"{removed_profile_items} profile tweet(s) without usable `text`. "
            f"{len(rows)} → {len(kept)} rows.",
            file=sys.stderr,
        )
    return kept


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


def gold_id_lookup(outputs_path: str) -> dict[str, Any]:
    """
    Map ``str(id)`` -> the ``id`` value as stored in the gold ``*_outputs.json`` file.

    LaMP's ``eval/evaluation.py`` compares prediction ids to gold ids with set equality;
    JSON may use strings vs integers, so predictions should reuse the gold file's ``id``
    representation when writing leaderboard-format JSON.
    """
    with open(outputs_path, encoding="utf-8") as f:
        o_raw = json.load(f)
    outputs = _unwrap_lamp_records(o_raw, outputs_path)
    return {str(o["id"]): o["id"] for o in outputs}


def merge_questions_and_outputs(
    questions_path: str, outputs_path: str, *, task: str | None = None
) -> list[dict[str, Any]]:
    """
    Build rows like upstream ``GeneralSeq2SeqDataset`` expects: each item has at least
    ``id``, ``input``, ``output``, and usually ``profile``.

    Official LaMP splits store **inputs** (``train_questions.json`` / ``*_questions.json``)
    and **labels** (``train_outputs.json`` / ``*_outputs.json``) in separate files; see
    ``LaMP/LaMP/utils/merge_with_rank.py`` and ``LaMP/data/avocado/create_avocado_dataset.py``.

    If ``task`` is ``LaMP-5`` or ``LaMP-7``, rows with missing query text and profile items
    without usable abstract / tweet ``text`` are removed (see ``filter_invalid_lamp_samples``).
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

    if task:
        merged = filter_invalid_lamp_samples(merged, task)
    if not merged:
        raise ValueError(
            f"No rows left after merging and filtering ({questions_path!r} + {outputs_path!r}, task={task!r})."
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
    Stable user key for grouping test rows.

    Prefer an explicit field (`user_field`, or `user_id` / `user` / `author_id` / `author`).
    Otherwise fingerprint the profile. Official user-based splits already keep users
    out of train; this only partitions the **test** file you pass to `run_evaluate.py`.
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
