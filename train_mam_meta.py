"""
Meta-train TTT-E2E (outer loop) for GPT-2 with DualMLP on LaMP-5 profile text.

Uses ``higher`` + ``ttt.mam_outer.run_lamp`` (same structure as the MAM reference).
Run from the repo root::

    py -3 train_mam_meta.py --task LaMP-5 \\
      --train_questions_json path/to/train_questions.json \\
      --train_outputs_json path/to/train_outputs.json \\
      --output_dir path/to/mam_ckpts \\
      --model_name gpt2 \\
      --meta_steps 500
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

from ttt.mam_outer import run_lamp  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["LaMP-5", "LaMP-7"], default="LaMP-5")
    p.add_argument("--train_questions_json", required=True)
    p.add_argument("--train_outputs_json", required=True)
    p.add_argument("--output_dir", default="mam_checkpoints")
    p.add_argument("--model_name", default="gpt2", help="HF hub id for GPT-2 (e.g. gpt2, openai-community/gpt2-large).")
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
        help="Cache for tokenized train profiles; default: <output_dir>/lamp_profile_token_cache.pt",
    )
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    merged = data_io.merge_questions_and_outputs(
        args.train_questions_json, args.train_outputs_json, task=args.task
    )
    rows = [{k: v for k, v in r.items() if k != "output"} for r in merged]

    log_path = os.path.join(args.output_dir, "meta_loss_lamp.csv")
    lamp_cache = args.lamp_cache_path or os.path.join(args.output_dir, "lamp_profile_token_cache.pt")

    run_lamp(
        rows,
        task=args.task,
        device=device,
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
        model_name=args.model_name,
        lamp_cache_path=lamp_cache,
    )
    print(f"Done. Checkpoints and log under {args.output_dir}")


if __name__ == "__main__":
    main()
