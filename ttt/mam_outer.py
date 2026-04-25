"""Outer meta-loop (``higher``): train init so post-inner NTP on a held-out span is good.

Includes the original WikiText driver (``run``) and ``run_lamp`` for LaMP-5 profile corpora.
"""
from __future__ import annotations

import argparse
import csv
import os
import time
from typing import Any

import higher
import torch
from tqdm import tqdm

from .mam_data import meta_example_stream, meta_example_stream_lamp
from .mam_inner import inner_adapt_functional
from .mam_model import TTTGPT2


def _meta_step(
    model: TTTGPT2,
    outer_opt: torch.optim.Optimizer,
    inner_opt: torch.optim.Optimizer,
    ctx: torch.Tensor,
    cont: torch.Tensor,
    *,
    inner_steps: int,
    window: int,
    clip: float = 1.0,
) -> float:
    ctx = ctx.to(next(model.parameters()).device)
    cont = cont.to(next(model.parameters()).device)

    outer_opt.zero_grad()
    with higher.innerloop_ctx(
        model,
        inner_opt,
        copy_initial_weights=False,
        track_higher_grads=True,
    ) as (fmodel, diffopt):
        inner_adapt_functional(fmodel, diffopt, ctx, steps=inner_steps, window=window)

        full = torch.cat([ctx, cont], dim=-1)
        logits = fmodel(full).logits
        cl = cont.size(-1)
        pred_logits = logits[:, -cl - 1 : -1, :]
        meta_loss = torch.nn.functional.cross_entropy(
            pred_logits.reshape(-1, pred_logits.size(-1)),
            cont.reshape(-1),
        )
        meta_loss.backward()

    torch.nn.utils.clip_grad_norm_(
        list(model.outer_params()) + list(model.inner_params()),
        clip,
    )
    outer_opt.step()
    return float(meta_loss.detach().cpu())


def run(
    *,
    device: torch.device | None = None,
    meta_steps: int = 2000,
    inner_steps: int = 1,
    inner_lr: float = 1e-3,
    outer_lr_outer: float = 1e-5,
    outer_lr_inner_init: float = 1e-4,
    context_len: int = 256,
    continuation_len: int = 64,
    window: int = 256,
    ckpt_dir: str = "checkpoints",
    log_path: str = "logs/meta_loss.csv",
    ckpt_every: int = 200,
    model_name: str = "gpt2",
):
    dev = device or torch.device("cpu")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

    model = TTTGPT2(model_name).to(dev)

    outer_opt = torch.optim.Adam(
        [
            {"params": list(model.outer_params()), "lr": outer_lr_outer},
            {"params": list(model.inner_params()), "lr": outer_lr_inner_init},
        ]
    )
    inner_opt = torch.optim.SGD(model.inner_params(), lr=inner_lr)

    stream = meta_example_stream(
        model.tokenizer,
        context_len=context_len,
        continuation_len=continuation_len,
    )

    log_file = open(log_path, "w", newline="")
    log = csv.writer(log_file)
    log.writerow(["meta_step", "meta_loss", "elapsed_s"])
    t0 = time.time()

    pbar = tqdm(range(meta_steps), desc="meta-train (wikitext)")
    for step in pbar:
        ctx, cont = next(stream)
        loss_v = _meta_step(
            model, outer_opt, inner_opt, ctx, cont, inner_steps=inner_steps, window=window
        )
        elapsed = time.time() - t0
        log.writerow([step, loss_v, f"{elapsed:.1f}"])
        log_file.flush()
        pbar.set_postfix(loss=f"{loss_v:.3f}")

        if (step + 1) % ckpt_every == 0 or step == meta_steps - 1:
            path = os.path.join(ckpt_dir, f"ttt_gpt2_meta_{step + 1}.pt")
            torch.save(model.state_dict(), path)
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "latest.pt"))

    log_file.close()


def run_lamp(
    train_rows: list[dict[str, Any]],
    *,
    task: str,
    device: torch.device | None = None,
    meta_steps: int = 2000,
    inner_steps: int = 1,
    inner_lr: float = 1e-3,
    outer_lr_outer: float = 1e-5,
    outer_lr_inner_init: float = 1e-4,
    context_len: int = 256,
    continuation_len: int = 64,
    window: int = 256,
    ckpt_dir: str = "checkpoints",
    log_path: str = "logs/meta_loss_lamp.csv",
    ckpt_every: int = 200,
    model_name: str = "gpt2",
    lamp_cache_path: str = ".cache/lamp_train_profiles.pt",
):
    """Meta-train on random (context, continuation) slices from flattened LaMP profiles."""
    dev = device or torch.device("cpu")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

    model = TTTGPT2(model_name).to(dev)
    outer_opt = torch.optim.Adam(
        [
            {"params": list(model.outer_params()), "lr": outer_lr_outer},
            {"params": list(model.inner_params()), "lr": outer_lr_inner_init},
        ]
    )
    inner_opt = torch.optim.SGD(model.inner_params(), lr=inner_lr)

    stream = meta_example_stream_lamp(
        model.tokenizer,
        train_rows,
        task,
        context_len=context_len,
        continuation_len=continuation_len,
        cache_path=lamp_cache_path,
    )

    log_file = open(log_path, "w", newline="")
    log = csv.writer(log_file)
    log.writerow(["meta_step", "meta_loss", "elapsed_s"])
    t0 = time.time()
    pbar = tqdm(range(meta_steps), desc=f"meta-train (LaMP {task})")
    for step in pbar:
        ctx, cont = next(stream)
        loss_v = _meta_step(
            model, outer_opt, inner_opt, ctx, cont, inner_steps=inner_steps, window=window
        )
        elapsed = time.time() - t0
        log.writerow([step, loss_v, f"{elapsed:.1f}"])
        log_file.flush()
        pbar.set_postfix(loss=f"{loss_v:.3f}")

        if (step + 1) % ckpt_every == 0 or step == meta_steps - 1:
            torch.save(model.state_dict(), os.path.join(ckpt_dir, f"ttt_lamp_meta_{step + 1}.pt"))
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "latest.pt"))

    log_file.close()


def main():
    ap = argparse.ArgumentParser(description="TTT-E2E outer loop (WikiText default).")
    ap.add_argument("--meta-steps", type=int, default=2000)
    ap.add_argument("--inner-steps", type=int, default=1)
    ap.add_argument("--inner-lr", type=float, default=1e-3)
    ap.add_argument("--context-len", type=int, default=256)
    ap.add_argument("--continuation-len", type=int, default=64)
    ap.add_argument("--ckpt-every", type=int, default=200)
    ap.add_argument("--model-name", type=str, default="gpt2")
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run(
        device=dev,
        meta_steps=args.meta_steps,
        inner_steps=args.inner_steps,
        inner_lr=args.inner_lr,
        context_len=args.context_len,
        continuation_len=args.continuation_len,
        ckpt_every=args.ckpt_every,
        model_name=args.model_name,
    )


if __name__ == "__main__":
    main()
