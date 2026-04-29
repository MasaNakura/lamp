"""
Global supervised training: LoRA + LaMP RAG prompts (optional checkpoint for **M3**).

The saved checkpoint can be used by `run_evaluate.py` for **M3** (RAG + adapter).
M3 can also run without this stage (base model + RAG).

**Training data** follows the LaMP release layout: ``train_questions.json`` (``input`` +
``profile`` per ``id``) and ``train_outputs.json`` (gold ``output`` per ``id``), merged
by id like ``LaMP/LaMP/utils/merge_with_rank.py`` (see ``LaMP/README.md`` and the
benchmark site). Optional ``dev_questions.json`` / ``dev_outputs.json`` enable
per-epoch validation; if omitted, training runs without a dev split (no
``load_best_model_at_end``).

Uses ``GeneralSeq2SeqDataset`` and metrics from the LaMP submodule via ``util.lamp_paths``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

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
        "--train_questions_json",
        required=True,
        help="LaMP train inputs + profiles (e.g. train_questions.json).",
    )
    p.add_argument(
        "--train_outputs_json",
        required=True,
        help="LaMP train gold labels (e.g. train_outputs.json; list or {\"task\", \"golds\"}).",
    )
    p.add_argument(
        "--dev_questions_json",
        default=None,
        help="Optional dev_questions.json for validation during training.",
    )
    p.add_argument(
        "--dev_outputs_json",
        default=None,
        help="Optional dev_outputs.json (pair with --dev_questions_json).",
    )
    p.add_argument("--base_model", default="google/flan-t5-small")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--num_retrieved", type=int, default=3)
    p.add_argument(
        "--retriever",
        default="contriever",
        choices=["contriever", "bm25", "random", "recency"],
    )
    p.add_argument("--ranked", action="store_true", help="Profile items are pre-ranked (LaMP merge step).")
    p.add_argument("--max_input_length", type=int, default=512)
    p.add_argument("--max_target_length", type=int, default=128)
    p.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Per-device train/eval batch size; on GPU you can often increase this with --fp16/--bf16.",
    )
    p.add_argument(
        "--fp16",
        action="store_true",
        help="Mixed-precision training on CUDA (Trainer fp16 + loss scaling).",
    )
    p.add_argument(
        "--bf16",
        action="store_true",
        help="bfloat16 training on CUDA when supported (often best on Ampere+). Mutually exclusive with --fp16.",
    )
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _write_merged_train_and_maybe_dev(
    args,
) -> tuple[str, str | None]:
    """
    Write merged JSON files for ``GeneralSeq2SeqDataset`` (upstream expects one path per split).

    Returns (train_merged_path, dev_merged_path_or_None).
    """
    merged_train = os.path.join(args.output_dir, "merged_train.json")
    merged_dev = os.path.join(args.output_dir, "merged_dev.json")

    train_rows = data_io.merge_questions_and_outputs(
        args.train_questions_json, args.train_outputs_json, task=args.task
    )
    data_io.warn_if_rows_look_like_unexpanded_placeholders(
        train_rows,
        task=args.task,
        context=f"train: {args.train_questions_json} + {args.train_outputs_json}",
    )
    with open(merged_train, "w", encoding="utf-8") as f:
        json.dump(train_rows, f, ensure_ascii=False)

    has_dev = bool(args.dev_questions_json and args.dev_outputs_json)
    if has_dev:
        dev_rows = data_io.merge_questions_and_outputs(
            args.dev_questions_json, args.dev_outputs_json, task=args.task
        )
        data_io.warn_if_rows_look_like_unexpanded_placeholders(
            dev_rows,
            task=args.task,
            context=f"dev: {args.dev_questions_json} + {args.dev_outputs_json}",
        )
        with open(merged_dev, "w", encoding="utf-8") as f:
            json.dump(dev_rows, f, ensure_ascii=False)
        return merged_train, merged_dev

    if args.dev_questions_json or args.dev_outputs_json:
        raise SystemExit("Pass both --dev_questions_json and --dev_outputs_json, or neither.")

    return merged_train, None


def main():
    args = parse_args()
    if args.fp16 and args.bf16:
        raise SystemExit("Use at most one of --fp16 and --bf16.")
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    os.makedirs(args.output_dir, exist_ok=True)
    train_path, dev_path = _write_merged_train_and_maybe_dev(args)

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, cache_dir=args.cache_dir, use_fast=False
    )
    base = AutoModelForSeq2SeqLM.from_pretrained(
        args.base_model, cache_dir=args.cache_dir
    )
    model = modeling_lora.attach_lora(
        base, r=args.lora_r, alpha=args.lora_alpha, dropout=0.05
    )
    model = model.to(device)
    model.print_trainable_parameters()

    prompt_generator, contriever = create_prompt_generator(
        args.num_retrieved,
        args.retriever,
        args.ranked,
        args.max_input_length,
        tokenizer,
    )
    if contriever is not None:
        contriever = contriever.to(device)

    train_ds = GeneralSeq2SeqDataset(
        train_path, use_profile=True, task=args.task, create_prompt=prompt_generator
    )
    val_hf = None
    if dev_path is not None:
        val_ds = GeneralSeq2SeqDataset(
            dev_path, use_profile=True, task=args.task, create_prompt=prompt_generator
        )
        val_hf = convert_to_hf_dataset(val_ds, cache_dir=args.cache_dir).map(
            create_preprocessor(tokenizer, args.max_input_length), batched=True
        )

    train_hf = convert_to_hf_dataset(train_ds, cache_dir=args.cache_dir).map(
        create_preprocessor(tokenizer, args.max_input_length), batched=True
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model, padding="longest", max_length=args.max_input_length
    )
    compute_metrics = create_metric_bleu_rouge_meteor(tokenizer=tokenizer)

    use_eval = val_hf is not None
    cuda = use_cuda
    use_bf16 = bool(args.bf16 and cuda and torch.cuda.is_bf16_supported())
    if args.bf16 and cuda and not use_bf16:
        print("[train] --bf16 not supported on this GPU; training in fp32 (bf16 disabled).", file=sys.stderr)
    use_fp16 = bool(args.fp16 and cuda)
    if (args.fp16 or args.bf16) and not cuda:
        print("[train] --fp16/--bf16 apply on CUDA only; training in fp32 on CPU.", file=sys.stderr)

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
        eval_strategy="epoch" if use_eval else "no",
        load_best_model_at_end=use_eval,
        metric_for_best_model="rouge-1" if use_eval else None,
        greater_is_better=True if use_eval else None,
        save_total_limit=2,
        seed=args.seed,
        report_to=[],
        fp16=use_fp16,
        bf16=use_bf16,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=targs,
        train_dataset=train_hf,
        eval_dataset=val_hf,
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics if use_eval else None,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if use_eval:
        metrics = trainer.evaluate(val_hf)
        with open(os.path.join(args.output_dir, "dev_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(metrics)


if __name__ == "__main__":
    main()
