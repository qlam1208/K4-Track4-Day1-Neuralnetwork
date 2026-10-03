"""optimizer.py — Cấu hình bộ tối ưu hoá, scheduler và cắt gradient.

Được dùng torch.optim.* và torch.nn.utils.clip_grad_norm_ (xem README mục 5).
File này gom việc chọn bộ tối ưu và cắt gradient để `train.py` gọn và mọi thí nghiệm công bằng.
"""
from __future__ import annotations

import torch
import torch.nn.utils

OPTIMIZERS = ("sgd", "sgd_momentum", "adam", "adamw")


def build_optimizer(name: str, params, lr: float, weight_decay: float = 0.0,
                    momentum: float = 0.9, betas=(0.9, 0.999), eps: float = 1e-8):
    """Trả về một torch.optim.Optimizer."""
    name_clean = name.lower()
    if name_clean not in OPTIMIZERS:
        raise ValueError(f"Optimizer '{name}' không hợp lệ. Phải thuộc {OPTIMIZERS}")

    if name_clean == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay)
    elif name_clean == "sgd_momentum":
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif name_clean == "adam":
        return torch.optim.Adam(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    elif name_clean == "adamw":
        return torch.optim.AdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)


def build_scheduler(optimizer, name: str | None, total_steps: int, **kwargs):
    """(Tuỳ chọn) Bộ lập lịch tốc độ học, ví dụ cosine."""
    if name is None or name.lower() in ("none", ""):
        return None

    name_clean = name.lower()
    if name_clean == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, **kwargs)
    elif name_clean == "step":
        step_size = kwargs.get("step_size", total_steps // 3)
        gamma = kwargs.get("gamma", 0.1)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    else:
        raise ValueError(f"Scheduler '{name}' chưa được hỗ trợ.")


def clip_gradients(params, max_norm: float | None) -> float:
    """Cắt gradient theo chuẩn L2 toàn cục, và TRẢ VỀ chuẩn gradient TRƯỚC KHI cắt.

    Khi dùng mixed precision FP16 + GradScaler: phải scaler.unscale_(optimizer) TRƯỚC khi gọi hàm này.
    """
    if isinstance(params, torch.Tensor):
        p_list = [params]
    else:
        p_list = [p for p in params if p.grad is not None]

    if not p_list:
        return 0.0

    if max_norm is None:
        total_norm = torch.nn.utils.clip_grad_norm_(p_list, max_norm=float("inf"))
    else:
        total_norm = torch.nn.utils.clip_grad_norm_(p_list, max_norm=float(max_norm))

    return float(total_norm.item() if hasattr(total_norm, "item") else total_norm)
