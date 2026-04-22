"""
Evaluate personalization baselines and TTT variants on a held-out split. Pass either a
single merged JSON array (`--test_json`) or benchmark-style `*_questions.json` +
`*_outputs.json` paths (merged by id, like upstream ``merge_with_rank.py``).

Models (paper storyboard):
  M1 Zero-shot base (task input only, no profile)
  M2 ICL (history serialized into the encoder budget)
  M3 Global LoRA + RAG (checkpoint from train.py)
  M4 Global LoRA + TTT (reset LoRA to global snapshot each user; TTT on profile)
  M5 Clean TTT (fresh LoRA on base; reset each user)

Metrics follow LaMP/LaMP/metrics/generation_metrics.py (BLEU, ROUGE*, METEOR).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from util.lamp_paths import ensure_lamp_on_path

ensure_lamp_on_path()

from prompts.prompts import create_prompt_generator  # noqa: E402

_DATA_DIR = os.path.join(_ROOT, "data")
if _DATA_DIR not in sys.path:
    sys.path.append(_DATA_DIR)
import data_io  # noqa: E402

from ttt.training import run_ttt_steps  # noqa: E402
from util import metrics_eval, modeling_lora, prompting  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["LaMP-5", "LaMP-7"], required=True)
    p.add_argument(
        "--test_json",
        default=None,
        help="Single merged JSON array (id, input, output, profile per row).",
    )
    p.add_argument(
        "--test_questions_json",
        default=None,
        help="LaMP benchmark-style test inputs (see *_questions.json on the download page).",
    )
    p.add_argument(
        "--test_outputs_json",
        default=None,
        help="Gold outputs for the same ids (list or {\"task\", \"golds\"} like *_outputs.json).",
    )
    p.add_argument("--base_model", default="google/flan-t5-small")
    p.add_argument("--adapter_dir", default=None, help="Checkpoint directory from train.py (M3/M4).")
    p.add_argument("--modes", default="m1,m2,m3,m4,m5", help="Comma list among m1..m5.")
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--num_retrieved", type=int, default=3)
    p.add_argument("--retriever", default="bm25", choices=["bm25", "random", "recency"])
    p.add_argument("--ranked", action="store_true")
    p.add_argument("--max_input_length", type=int, default=512)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--ttt_steps", type=int, default=30)
    p.add_argument("--ttt_lr", type=float, default=1e-4)
    p.add_argument("--user_field", default=None)
    p.add_argument("--output_dir", default="eval_outputs")
    return p.parse_args()


def task_leaderboard_name(task: str) -> str:
    return task.replace("-", "_")


def merge_profiles(rows: list[dict]) -> list[dict]:
    seen: set[str] = set()
    merged: list[dict] = []
    for r in rows:
        for item in r.get("profile") or []:
            key = str(item.get("id")) if item.get("id") is not None else json.dumps(item, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
    return merged


@torch.no_grad()
def batched_generate(model, tokenizer, sources: list[str], device: torch.device, max_in: int, max_new: int):
    enc = tokenizer(
        sources,
        truncation=True,
        max_length=max_in,
        padding=True,
        return_tensors="pt",
    ).to(device)
    out_ids = model.generate(**enc, max_new_tokens=max_new)
    return tokenizer.batch_decode(out_ids, skip_special_tokens=True)


def run_for_mode(
    mode: str,
    rows: list[dict],
    user_to_rows: dict[str, list[dict]],
    *,
    task: str,
    tokenizer,
    base_model_name: str,
    adapter_dir: str | None,
    cache_dir: str | None,
    device: torch.device,
    rag_prompt,
    max_in: int,
    max_new: int,
    batch_size: int,
    ttt_steps: int,
    ttt_lr: float,
) -> list[tuple[str, str]]:
    base = AutoModelForSeq2SeqLM.from_pretrained(base_model_name, cache_dir=cache_dir)
    if mode in ("m3", "m4"):
        if not adapter_dir:
            raise ValueError(f"{mode} requires --adapter_dir (global LoRA+RAG checkpoint).")
        model = PeftModel.from_pretrained(base, adapter_dir)
    elif mode == "m5":
        model = modeling_lora.attach_lora(base)
    else:
        model = base

    model.to(device)
    model.eval()

    preds: list[tuple[str, str]] = []

    def handle_batch(sources: list[str], meta_ids: list[str]):
        decoded = batched_generate(model, tokenizer, sources, device, max_in, max_new)
        preds.extend(zip(meta_ids, decoded))

    if mode in ("m1", "m2", "m3"):
        batch_src: list[str] = []
        batch_ids: list[str] = []
        for row in tqdm(rows, desc=mode):
            if mode == "m1":
                src = row["input"]
            elif mode == "m2":
                src = prompting.build_icl_source(
                    row, tokenizer, task=task, max_tokens=max_in, reserve_for_input=128
                )
            else:
                src = rag_prompt(row)
            batch_src.append(src)
            batch_ids.append(row["id"])
            if len(batch_src) >= batch_size:
                handle_batch(batch_src, batch_ids)
                batch_src, batch_ids = [], []
        if batch_src:
            handle_batch(batch_src, batch_ids)
        return preds

    # M4 / M5: snapshot LoRA (global for M4, freshly initialized for M5), never leak across users.
    snapshot = modeling_lora.lora_state_snapshot(model)
    for _user, urows in tqdm(list(user_to_rows.items()), desc=mode):
        modeling_lora.restore_lora_snapshot(model, snapshot)
        prof = merge_profiles(urows)
        run_ttt_steps(
            model,
            tokenizer,
            task=task,
            profile=prof,
            device=device,
            max_input_length=max_in,
            micro_batch_size=2,
            steps=ttt_steps,
            lr=ttt_lr,
        )
        model.eval()

        batch_src, batch_ids = [], []
        for row in urows:
            batch_src.append(rag_prompt(row))
            batch_ids.append(row["id"])
            if len(batch_src) >= batch_size:
                handle_batch(batch_src, batch_ids)
                batch_src, batch_ids = [], []
        if batch_src:
            handle_batch(batch_src, batch_ids)

        modeling_lora.restore_lora_snapshot(model, snapshot)

    return preds


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.test_json:
        rows = data_io.load_json_list(args.test_json)
    elif args.test_questions_json and args.test_outputs_json:
        rows = data_io.merge_questions_and_outputs(
            args.test_questions_json, args.test_outputs_json
        )
    else:
        raise SystemExit(
            "Provide either --test_json (merged array) or both "
            "--test_questions_json and --test_outputs_json."
        )
    user_to_rows: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        uid = data_io.infer_user_id(r, user_field=args.user_field)
        user_to_rows[uid].append(r)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, cache_dir=args.cache_dir, use_fast=False)
    rag_gen, contriever = create_prompt_generator(
        args.num_retrieved,
        args.retriever,
        args.ranked,
        args.max_input_length,
        tokenizer,
    )
    if contriever is not None:
        contriever = contriever.to("cpu")

    def rag_prompt(row: dict) -> str:
        return rag_gen(row["input"], row["profile"], args.task)

    modes = [m.strip().lower() for m in args.modes.split(",") if m.strip()]
    refs = [r["output"] for r in rows]
    id_order = [r["id"] for r in rows]

    results_summary: dict[str, dict] = {}
    for mode in modes:
        if mode in ("m3", "m4") and not args.adapter_dir:
            raise ValueError(f"Mode {mode} requires --adapter_dir")
        pairs = run_for_mode(
            mode,
            rows,
            user_to_rows,
            task=args.task,
            tokenizer=tokenizer,
            base_model_name=args.base_model,
            adapter_dir=args.adapter_dir,
            cache_dir=args.cache_dir,
            device=device,
            rag_prompt=rag_prompt,
            max_in=args.max_input_length,
            max_new=args.max_new_tokens,
            batch_size=args.batch_size,
            ttt_steps=args.ttt_steps,
            ttt_lr=args.ttt_lr,
        )
        pred_map = {i: p for i, p in pairs}
        preds_ordered = [pred_map[i] for i in id_order]
        metrics = metrics_eval.evaluate_strings(preds_ordered, refs)
        results_summary[mode] = metrics
        out_json = os.path.join(args.output_dir, f"preds_{mode}.json")
        metrics_eval.write_lamp_predictions(
            task_leaderboard_name(args.task),
            [(i, pred_map[i]) for i in id_order],
            out_json,
        )
        with open(os.path.join(args.output_dir, f"metrics_{mode}.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(mode, metrics)

    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)


if __name__ == "__main__":
    main()
