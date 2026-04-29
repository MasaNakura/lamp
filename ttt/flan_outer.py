"""Outer meta-loop for Flan-T5 Dual-FFN TTT-E2E."""
from __future__ import annotations

import argparse
import csv
import os
import time
from typing import Any

import higher
import torch
from tqdm import tqdm

from .flan_inner import inner_adapt_t5_functional
from .flan_dual_mlp_model import TTTFlanT5
from .mam_data import meta_example_stream, meta_example_stream_lamp


def _loss_text_copy(model, tokenizer, ids: torch.Tensor, max_length: int) -> torch.Tensor:
    text = tokenizer.decode(ids.view(-1).tolist(), skip_special_tokens=True)
    if not text.strip():
        return torch.zeros((), device=ids.device, requires_grad=True)
    batch = tokenizer(
        [text],
        text_target=[text],
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="pt",
    )
    batch = {k: v.to(ids.device) for k, v in batch.items()}
    out = model(**batch)
    if out.loss is None:
        return torch.zeros((), device=ids.device, requires_grad=True)
    return out.loss


def _meta_step(
    model: TTTFlanT5,
    tokenizer,
    outer_opt: torch.optim.Optimizer,
    inner_opt: torch.optim.Optimizer,
    ctx: torch.Tensor,
    cont: torch.Tensor,
    *,
    window: int,
    max_seq_len: int,
    clip: float = 1.0,
) -> float:
    dev = next(model.parameters()).device
    ctx = ctx.to(dev)
    cont = cont.to(dev)

    outer_opt.zero_grad()
    with higher.innerloop_ctx(
        model,
        inner_opt,
        copy_initial_weights=False,
        track_higher_grads=True,
    ) as (fmodel, diffopt):
        inner_adapt_t5_functional(
            fmodel,
            diffopt,
            tokenizer,
            ctx,
            window=window,
            stride=window,
        )
        meta_loss = _loss_text_copy(fmodel, tokenizer, cont, max_length=max_seq_len)
        meta_loss.backward()

    torch.nn.utils.clip_grad_norm_(list(model.outer_params()) + list(model.inner_params()), clip)
    outer_opt.step()
    return float(meta_loss.detach().cpu())


def run_lamp(
    train_rows: list[dict[str, Any]],
    *,
    task: str,
    device: torch.device | None = None,
    model_name: str = "google/flan-t5-small",
    ttt_fraction: float = 0.25,
    meta_steps: int = 2000,
    inner_lr: float = 1e-3,
    outer_lr_outer: float = 1e-5,
    outer_lr_inner_init: float = 1e-4,
    context_len: int = 256,
    continuation_len: int = 64,
    window: int = 256,
    ckpt_dir: str = "checkpoints_flan",
    log_path: str = "logs/meta_loss_lamp_flan.csv",
    ckpt_every: int = 200,
    lamp_cache_path: str = ".cache/lamp_train_profiles_flan.pt",
    log_every: int = 50,
):
    dev = device or torch.device("cpu")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

    model = TTTFlanT5(model_name=model_name, ttt_fraction=ttt_fraction).to(dev)
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

    log_file = open(log_path, "w", newline="", encoding="utf-8")
    log = csv.writer(log_file)
    log.writerow(["meta_step", "meta_loss", "meta_loss_ema", "elapsed_s"])
    t0 = time.time()
    ema_loss: float | None = None
    ema_decay = 0.98

    pbar = tqdm(range(meta_steps), desc=f"meta-train flan (LaMP {task})")
    for step in pbar:
        ctx, cont = next(stream)
        loss_v = _meta_step(
            model,
            model.tokenizer,
            outer_opt,
            inner_opt,
            ctx,
            cont,
            window=window,
            max_seq_len=max(window, continuation_len),
        )
        elapsed = time.time() - t0
        ema_loss = loss_v if ema_loss is None else ema_decay * ema_loss + (1.0 - ema_decay) * loss_v
        log.writerow([step, loss_v, f"{ema_loss:.6f}", f"{elapsed:.1f}"])
        log_file.flush()
        pbar.set_postfix(loss=f"{loss_v:.3f}", ema=f"{ema_loss:.3f}")
        if log_every > 0 and ((step + 1) % log_every == 0 or step == 0 or step == meta_steps - 1):
            print(
                f"[meta-train flan lamp] step={step + 1}/{meta_steps} "
                f"loss={loss_v:.4f} ema={ema_loss:.4f} elapsed_s={elapsed:.1f}"
            )

        if (step + 1) % ckpt_every == 0 or step == meta_steps - 1:
            torch.save(model.state_dict(), os.path.join(ckpt_dir, f"ttt_flan_meta_{step + 1}.pt"))
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "latest.pt"))
    log_file.close()


def run(
    *,
    device: torch.device | None = None,
    model_name: str = "google/flan-t5-small",
    ttt_fraction: float = 0.25,
    meta_steps: int = 2000,
    inner_lr: float = 1e-3,
    outer_lr_outer: float = 1e-5,
    outer_lr_inner_init: float = 1e-4,
    context_len: int = 256,
    continuation_len: int = 64,
    window: int = 256,
    ckpt_dir: str = "checkpoints_flan",
    log_path: str = "logs/meta_loss_flan.csv",
    ckpt_every: int = 200,
    cache_path: str = ".cache/wikitext103_train_flan.pt",
):
    dev = device or torch.device("cpu")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

    model = TTTFlanT5(model_name=model_name, ttt_fraction=ttt_fraction).to(dev)
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
        cache_path=cache_path,
    )

    log_file = open(log_path, "w", newline="", encoding="utf-8")
    log = csv.writer(log_file)
    log.writerow(["meta_step", "meta_loss", "elapsed_s"])
    t0 = time.time()
    pbar = tqdm(range(meta_steps), desc="meta-train flan (wikitext)")
    for step in pbar:
        ctx, cont = next(stream)
        loss_v = _meta_step(
            model,
            model.tokenizer,
            outer_opt,
            inner_opt,
            ctx,
            cont,
            window=window,
            max_seq_len=max(window, continuation_len),
        )
        elapsed = time.time() - t0
        log.writerow([step, loss_v, f"{elapsed:.1f}"])
        log_file.flush()
        pbar.set_postfix(loss=f"{loss_v:.3f}")
        if (step + 1) % ckpt_every == 0 or step == meta_steps - 1:
            torch.save(model.state_dict(), os.path.join(ckpt_dir, f"ttt_flan_meta_{step + 1}.pt"))
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "latest.pt"))
    log_file.close()


def main():
    ap = argparse.ArgumentParser(description="TTT-E2E outer meta loop for Flan-T5 Dual-FFN.")
    ap.add_argument("--meta-steps", type=int, default=2000)
    ap.add_argument("--inner-lr", type=float, default=1e-3)
    ap.add_argument("--context-len", type=int, default=256)
    ap.add_argument("--continuation-len", type=int, default=64)
    ap.add_argument("--ckpt-every", type=int, default=200)
    ap.add_argument("--model-name", type=str, default="google/flan-t5-small")
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run(
        device=dev,
        meta_steps=args.meta_steps,
        inner_lr=args.inner_lr,
        context_len=args.context_len,
        continuation_len=args.continuation_len,
        ckpt_every=args.ckpt_every,
        model_name=args.model_name,
    )


if __name__ == "__main__":
    main()
