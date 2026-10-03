"""plots.py — Vẽ biểu đồ huấn luyện từng thí nghiệm và so sánh nhóm thí nghiệm.

Ảnh biểu đồ là sản phẩm nộp (xem README mục 6): mỗi thí nghiệm một ảnh figures/<exp_id>.png.
"""
from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt


def plot_run(result: dict, path: str) -> None:
    """Vẽ MỘT thí nghiệm thành một ảnh PNG có ít nhất 3 ô:
         (1) train_loss và val_loss theo epoch (cùng một trục)
         (2) val_acc (và val_macro_f1) theo epoch
         (3) grad_norm theo epoch (đo TRƯỚC khi clip)
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    cfg = result["cfg"]
    hist = result["history"]
    epochs = hist["epoch"]

    if not epochs:
        return

    best_epoch = result["summary"].get("best_epoch", 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    exp_id = cfg.get("exp_id", "exp")
    sub_title = (
        f"[{exp_id}] {cfg.get('description', '')} | opt={cfg.get('optimizer')}, "
        f"lr={cfg.get('lr')}, batch={cfg.get('batch')}, drop={cfg.get('dropout')}, init={cfg.get('init')}"
    )
    fig.suptitle(sub_title, fontsize=12, y=1.02)

    # Ô 1: Loss
    axes[0].plot(epochs, hist["train_loss"], label="Train Loss", marker="o", markersize=3)
    axes[0].plot(epochs, hist["val_loss"], label="Val Loss", marker="s", markersize=3)
    axes[0].axvline(best_epoch, color="gray", linestyle="--", alpha=0.7, label=f"Best Ep ({best_epoch})")
    axes[0].set_title("Loss vs. Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, linestyle=":", alpha=0.6)

    # Ô 2: Metrics
    axes[1].plot(epochs, hist["val_acc"], label="Val Accuracy", color="green", marker="^", markersize=3)
    if "val_macro_f1" in hist and hist["val_macro_f1"]:
        axes[1].plot(epochs, hist["val_macro_f1"], label="Val Macro-F1", color="orange", marker="d", markersize=3)
    axes[1].axvline(best_epoch, color="gray", linestyle="--", alpha=0.7, label=f"Best Ep ({best_epoch})")
    axes[1].set_title("Validation Metrics vs. Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].legend()
    axes[1].grid(True, linestyle=":", alpha=0.6)

    # Ô 3: Gradient Norm
    axes[2].plot(epochs, hist["grad_norm"], label="Grad Norm (pre-clip)", color="purple", marker="x", markersize=3)
    axes[2].set_title("Gradient Norm vs. Epoch")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("L2 Norm")
    axes[2].legend()
    axes[2].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_compare(results: list[dict], metric: str, path: str, title: str = "") -> None:
    """Vẽ chồng một chỉ số (ví dụ "val_loss", "val_macro_f1", "grad_norm") của nhiều thí nghiệm
    trên cùng một trục, mỗi thí nghiệm một đường, chú thích bằng exp_id.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    for res in results:
        cfg = res["cfg"]
        hist = res["history"]
        exp_id = cfg.get("exp_id", "exp")
        epochs = hist.get("epoch", [])
        vals = hist.get(metric, [])
        if epochs and vals:
            ax.plot(epochs, vals, marker="o", markersize=3, label=f"{exp_id}")

    chart_title = title if title else f"Comparison of {metric}"
    ax.set_title(chart_title, fontsize=13)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
