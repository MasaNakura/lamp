"""Meta-train Flan-T5 TTT-E2E outer loop on LaMP profile streams.

Optional ``--save_encoder_prompts`` writes ``encoder_prompts_flan_meta.json`` after meta-training
(profile stream text aligned with ``meta_example_stream_lamp`` / ``_lamp_rag``), same schema as
``train.py`` / ``run_evaluate.py`` encoder prompt dumps (``prediction`` empty).
"""
from __future__ import annotations

import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_DATA_DIR = os.path.join(_ROOT, "data")
if _DATA_DIR not in sys.path:
    sys.path.append(_DATA_DIR)

import data_io  # noqa: E402
import torch  # noqa: E402

from ttt.flan_outer import run_lamp  # noqa: E402
from util.encoder_prompt_dump import flan_meta_profile_stream_text, write_encoder_prompts_json  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--task",
        choices=["LaMP-5", "LaMP-7", "SD-tooluse", "SD-science"],
        default="LaMP-5",
    )
    p.add_argument("--train_questions_json", required=True)
    p.add_argument("--train_outputs_json", required=True)
    p.add_argument("--output_dir", default="flan_meta_checkpoints")
    p.add_argument("--model_name", default="google/flan-t5-small")
    p.add_argument("--ttt_fraction", type=float, default=0.25)
    p.add_argument("--meta_steps", type=int, default=2000)
    p.add_argument("--inner_lr", type=float, default=1e-3)
    p.add_argument("--outer_lr_outer", type=float, default=1e-5)
    p.add_argument("--outer_lr_inner_init", type=float, default=1e-4)
    p.add_argument("--context_len", type=int, default=256)
    p.add_argument("--continuation_len", type=int, default=64)
    p.add_argument("--window", type=int, default=256)
    p.add_argument("--ckpt_every", type=int, default=200)
    p.add_argument("--log_every", type=int, default=50, help="Print loss/EMA every N meta steps (0 disables).")
    p.add_argument(
        "--lamp_cache_path",
        default=None,
        help="Cache for tokenized train profiles; default: <output_dir>/lamp_profile_token_cache_flan.pt",
    )
    amp = p.add_mutually_exclusive_group()
    amp.add_argument(
        "--fp16",
        action="store_true",
        help="CUDA only: fp32 weights; float16 autocast on inner windows only; meta CE in fp32 + GradScaler (avoids NaN from fp16 softmax).",
    )
    amp.add_argument(
        "--bf16",
        action="store_true",
        help="CUDA only: bfloat16 autocast when supported; no GradScaler (often best on Ampere+).",
    )
    p.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Trade speed for VRAM: HF gradient checkpointing on the Flan-T5 stack (helps a lot with meta+higher).",
    )
    p.add_argument(
        "--no_cuda_expandable_segments",
        action="store_true",
        help="By default we set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True before GPU use; pass this to disable.",
    )
    p.add_argument("--cache_dir", default=None, help="HF cache dir (Flan-T5 and Contriever when using --ttt_rag contriever).")
    p.add_argument(
        "--ttt_rag",
        action="store_true",
        help="Meta-train with query-conditioned RAG profile narrowing (same retrieval as LaMP M3); "
        "matches M4 eval with --m4_use_rag. Disables the flat-profile token cache stream.",
    )
    p.add_argument(
        "--ttt_rag_retriever",
        default="bm25",
        choices=["contriever", "bm25", "random", "recency"],
        help="Used with --ttt_rag (LaMP-aligned).",
    )
    p.add_argument("--ttt_rag_num_retrieved", type=int, default=3)
    p.add_argument("--ttt_rag_ranked", action="store_true")
    p.add_argument(
        "--save_encoder_prompts",
        action="store_true",
        help="After run_lamp finishes, write encoder_prompts_flan_meta.json: per-row M4-meta profile stream "
        "(same text family as the meta token stream), gold_output, empty prediction.",
    )
    p.add_argument(
        "--save_encoder_prompts_max_rows",
        type=int,
        default=2000,
        help="With --save_encoder_prompts, max rows (-1 = all). First rows in merged train order.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if not args.no_cuda_expandable_segments:
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = bool(args.fp16 and device.type == "cuda")
    use_bf16 = bool(args.bf16 and device.type == "cuda" and torch.cuda.is_bf16_supported())
    if args.fp16 and not use_fp16:
        print("[train_flan_meta] --fp16 requires CUDA; running in fp32.", file=sys.stderr)
    if args.bf16 and device.type != "cuda":
        print("[train_flan_meta] --bf16 requires CUDA; running in fp32.", file=sys.stderr)
    elif args.bf16 and device.type == "cuda" and not use_bf16:
        print("[train_flan_meta] --bf16 not supported on this GPU; running in fp32.", file=sys.stderr)

    merged = data_io.merge_questions_and_outputs(
        args.train_questions_json, args.train_outputs_json, task=args.task
    )
    rows = [{k: v for k, v in r.items() if k != "output"} for r in merged]

    log_path = os.path.join(args.output_dir, "meta_loss_lamp_flan.csv")
    lamp_cache = args.lamp_cache_path or os.path.join(args.output_dir, "lamp_profile_token_cache_flan.pt")

    profile_rag = None
    if args.ttt_rag:
        from ttt.lamp_profile_rag import LampProfileRAG

        profile_rag = LampProfileRAG(
            args.task,
            num_retrieved=args.ttt_rag_num_retrieved,
            retriever=args.ttt_rag_retriever,
            ranked=args.ttt_rag_ranked,
            device=device,
            cache_dir=args.cache_dir,
        )

    run_lamp(
        rows,
        task=args.task,
        device=device,
        model_name=args.model_name,
        ttt_fraction=args.ttt_fraction,
        meta_steps=args.meta_steps,
        inner_lr=args.inner_lr,
        outer_lr_outer=args.outer_lr_outer,
        outer_lr_inner_init=args.outer_lr_inner_init,
        context_len=args.context_len,
        continuation_len=args.continuation_len,
        window=args.window,
        ckpt_dir=args.output_dir,
        log_path=log_path,
        ckpt_every=args.ckpt_every,
        log_every=args.log_every,
        lamp_cache_path=lamp_cache,
        use_fp16=use_fp16,
        use_bf16=use_bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        profile_rag=profile_rag,
        cache_dir=args.cache_dir,
    )

    if args.save_encoder_prompts:
        from transformers import AutoTokenizer

        lim = args.save_encoder_prompts_max_rows
        dump_rows = merged if lim < 0 else merged[:lim]
        tok = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir, use_fast=False)
        if args.task in ("SD-tooluse", "SD-science") and hasattr(tok, "truncation_side"):
            tok.truncation_side = "left"
        enc_max = min(getattr(tok, "model_max_length", 512) or 512, 8192)
        win_max = max(args.window, args.context_len + args.continuation_len)

        def _enc_fn(row: dict) -> str:
            return flan_meta_profile_stream_text(row, args.task, profile_rag)

        out_path = os.path.join(args.output_dir, "encoder_prompts_flan_meta.json")
        write_encoder_prompts_json(
            out_path,
            mode="m4_flan_meta_profile",
            task=args.task,
            rows=dump_rows,
            preds=[(str(r["id"]), "") for r in dump_rows],
            tokenizer=tok,
            max_in=win_max,
            encode_max_len=enc_max,
            rag_prompt=None,
            model=None,
            architecture="seq2seq",
            include_gold_output=True,
            encoder_text_fn=_enc_fn,
        )
        print(f"[train_flan_meta] Wrote encoder prompt dump: {out_path} ({len(dump_rows)} rows)", file=sys.stderr)

    print(f"Done. Checkpoints and log under {args.output_dir}")


if __name__ == "__main__":
    main()
