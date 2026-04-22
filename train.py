"""
Global supervised training: LoRA + LaMP RAG prompts (same stage for **M3 and M4**).

The saved checkpoint is used by `evaluate.py` for **M3** (inference only) and **M4**
(inference after per-user test-time training). **M5** skips this script and starts
from the base model with a fresh LoRA at evaluation time.

Each training/validation example must include ``id``, ``input``, ``output``, and
``profile``. Pass either merged JSON (``--train_json`` / ``--val_json``) or the
official split files (``*_questions.json`` + ``*_outputs.json``); the latter are
merged the same way as upstream ``LaMP/LaMP/utils/merge_with_rank.py``. Uses
``GeneralSeq2SeqDataset`` and metrics from the LaMP submodule via ``util.lamp_paths``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from util.lamp_paths import ensure_lamp_on_path

ensure_lamp_on_path()

from data.datasets import GeneralSeq2SeqDataset, convert_to_hf_dataset, create_preprocessor  # noqa: E402
from metrics.generation_metrics import create_metric_bleu_rouge_meteor  # noqa: E402
from prompts.prompts import create_prompt_generator  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from util import modeling_lora

_DATA_DIR = os.path.join(_ROOT, "data")
if _DATA_DIR not in sys.path:
    sys.path.append(_DATA_DIR)
import data_io  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["LaMP-5", "LaMP-7"], required=True)
    p.add_argument(
        "--train_json",
        default=None,
        help="Merged LaMP JSON (array): each row has id, input, output, profile.",
    )
    p.add_argument(
        "--val_json",
        default=None,
        help="Merged validation JSON (same schema as --train_json).",
    )
    p.add_argument(
        "--train_questions_json",
        default=None,
        help="LaMP split: inputs + profiles only (see benchmark *_questions.json).",
    )
    p.add_argument(
        "--train_outputs_json",
        default=None,
        help="LaMP split: gold labels (list or {\"task\", \"golds\"}; see *_outputs.json).",
    )
    p.add_argument("--val_questions_json", default=None)
    p.add_argument("--val_outputs_json", default=None)
    p.add_argument("--base_model", default="google/flan-t5-small")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--num_retrieved", type=int, default=3)
    p.add_argument("--retriever", default="bm25", choices=["bm25", "random", "recency"])
    p.add_argument("--ranked", action="store_true", help="Profile items are pre-ranked (LaMP merge step).")
    p.add_argument("--max_input_length", type=int, default=512)
    p.add_argument("--max_target_length", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _resolve_train_val_merged_paths(args) -> tuple[str, str]:
    """Return (train_path, val_path) on disk for ``GeneralSeq2SeqDataset`` (single file each)."""
    merged_train = os.path.join(args.output_dir, "merged_train.json")
    merged_val = os.path.join(args.output_dir, "merged_val.json")

    if args.train_json and args.val_json:
        return args.train_json, args.val_json

    pair_train = args.train_questions_json and args.train_outputs_json
    pair_val = args.val_questions_json and args.val_outputs_json
    if pair_train and pair_val:
        train_rows = data_io.merge_questions_and_outputs(
            args.train_questions_json, args.train_outputs_json
        )
        val_rows = data_io.merge_questions_and_outputs(
            args.val_questions_json, args.val_outputs_json
        )
        with open(merged_train, "w", encoding="utf-8") as f:
            json.dump(train_rows, f, ensure_ascii=False)
        with open(merged_val, "w", encoding="utf-8") as f:
            json.dump(val_rows, f, ensure_ascii=False)
        return merged_train, merged_val

    raise SystemExit(
        "Provide either (--train_json and --val_json) merged files, or all four: "
        "--train_questions_json, --train_outputs_json, --val_questions_json, --val_outputs_json."
    )


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    train_path, val_path = _resolve_train_val_merged_paths(args)

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, cache_dir=args.cache_dir, use_fast=False
    )
    base = AutoModelForSeq2SeqLM.from_pretrained(
        args.base_model, cache_dir=args.cache_dir
    )
    model = modeling_lora.attach_lora(
        base, r=args.lora_r, alpha=args.lora_alpha, dropout=0.05
    )
    model.print_trainable_parameters()

    prompt_generator, contriever = create_prompt_generator(
        args.num_retrieved,
        args.retriever,
        args.ranked,
        args.max_input_length,
        tokenizer,
    )
    if contriever is not None:
        contriever = contriever.to("cpu")

    train_ds = GeneralSeq2SeqDataset(
        train_path, use_profile=True, task=args.task, create_prompt=prompt_generator
    )
    val_ds = GeneralSeq2SeqDataset(
        val_path, use_profile=True, task=args.task, create_prompt=prompt_generator
    )

    train_hf = convert_to_hf_dataset(train_ds, cache_dir=args.cache_dir).map(
        create_preprocessor(tokenizer, args.max_input_length), batched=True
    )
    val_hf = convert_to_hf_dataset(val_ds, cache_dir=args.cache_dir).map(
        create_preprocessor(tokenizer, args.max_input_length), batched=True
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model, padding="longest", max_length=args.max_input_length
    )
    compute_metrics = create_metric_bleu_rouge_meteor(tokenizer=tokenizer)

    targs = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        predict_with_generate=True,
        generation_max_length=args.max_target_length,
        logging_steps=50,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="rouge-1",
        greater_is_better=True,
        save_total_limit=2,
        seed=args.seed,
        report_to=[],
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=targs,
        train_dataset=train_hf,
        eval_dataset=val_hf,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    metrics = trainer.evaluate(val_hf)
    with open(os.path.join(args.output_dir, "val_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(metrics)


if __name__ == "__main__":
    main()
