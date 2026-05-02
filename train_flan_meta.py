"""Meta-train Flan-T5 TTT-E2E outer loop on LaMP profile streams."""
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["LaMP-5", "LaMP-7"], default="LaMP-5")
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
    p.add_argument(
        "--fp16",
        action="store_true",
        help="CUDA only: load Flan-T5 in float16, autocast forwards, and GradScaler on the meta backward.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = bool(args.fp16 and device.type == "cuda")
    if args.fp16 and not use_fp16:
        print("[train_flan_meta] --fp16 requires CUDA; running in fp32.", file=sys.stderr)

    merged = data_io.merge_questions_and_outputs(
        args.train_questions_json, args.train_outputs_json, task=args.task
    )
    rows = [{k: v for k, v in r.items() if k != "output"} for r in merged]

    log_path = os.path.join(args.output_dir, "meta_loss_lamp_flan.csv")
    lamp_cache = args.lamp_cache_path or os.path.join(args.output_dir, "lamp_profile_token_cache_flan.pt")

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
    )
    print(f"Done. Checkpoints and log under {args.output_dir}")


if __name__ == "__main__":
    main()
