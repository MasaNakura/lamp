"""Enable TF32 on Ampere+ without PyTorch 2.9+ deprecation warnings."""
from __future__ import annotations

import torch


def enable_tf32() -> None:
    if not torch.cuda.is_available():
        return
    try:
        torch.backends.cuda.matmul.fp32_precision = "tf32"  # type: ignore[attr-defined]
        conv = getattr(torch.backends.cudnn, "conv", None)
        if conv is not None and hasattr(conv, "fp32_precision"):
            conv.fp32_precision = "tf32"  # type: ignore[attr-defined]
        return
    except (AttributeError, TypeError):
        pass
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
