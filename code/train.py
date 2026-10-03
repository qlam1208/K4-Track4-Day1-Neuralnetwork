"""train.py — Vòng lặp huấn luyện, đánh giá, dự đoán và lưu kết quả.

Gồm: đặt seed, đánh giá, vòng huấn luyện `run_experiment(cfg, data)`, dự đoán và ghi file nộp.
Mọi chỉ số (loss, accuracy, macro-F1) dùng cùng định nghĩa với scripts/evaluate.py.
"""
from __future__ import annotations

import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from data import iterate_batches
from model import MLP, EXPECTED_PARAMS, count_params
from optimizer import build_optimizer, clip_gradients

# Cấu hình mặc định = BASELINE (M-base).
DEFAULT_CFG = dict(
    exp_id="base-s1", group="baseline", description="Baseline M-base",
    loss="ce",                 # "ce" | "mse"
    optimizer="sgd_momentum",  # "sgd" | "sgd_momentum" | "adam" | "adamw"
    lr=0.05,                   # Học sinh chọn bằng val
    weight_decay=0.0, momentum=0.9,
    batch=512, epochs=20,
    hidden=(256, 128), dropout=0.0, init="he",
    clip_norm=None,            # None = không clip; hoặc số, ví dụ 1.0
    precision="fp32",          # "fp32" | "fp16" | "bf16"
    seed=1,
)


def set_seed(seed: int) -> None:
    """Đặt seed cho random, numpy, torch (và torch.cuda nếu có)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    """macro-F1 = trung bình cộng F1 của 7 lớp; F1_c = 2PR/(P+R), bằng 0 nếu P+R = 0.

    cm: ma trận nhầm lẫn (7, 7), hàng = nhãn thật, cột = dự đoán.
    """
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    return float(f1.mean())


@torch.no_grad()
def predict(model: torch.nn.Module, X: torch.Tensor, batch_size: int = 8192) -> torch.Tensor:
    """Trả về nhãn dự đoán int64 (N,) = argmax của logits."""
    model.eval()
    preds = []
    n = len(X)
    for i in range(0, n, batch_size):
        xb = X[i : i + batch_size]
        logits = model(xb)
        preds.append(logits.argmax(dim=1))
    return torch.cat(preds, dim=0)


def compute_loss(logits: torch.Tensor, y: torch.Tensor, loss_name: str) -> torch.Tensor:
    """"ce"  : cross-entropy nhận logit thô và nhãn int64 (F.cross_entropy).
       "mse" : MSE giữa logit và one-hot của y.
    """
    if loss_name == "ce":
        return F.cross_entropy(logits, y)
    elif loss_name == "mse":
        y_one_hot = F.one_hot(y, num_classes=logits.shape[1]).float()
        return F.mse_loss(logits, y_one_hot)
    else:
        raise ValueError(f"Hàm mất mát '{loss_name}' không được hỗ trợ (chỉ chọn 'ce' hoặc 'mse').")


@torch.no_grad()
def evaluate(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor,
             loss_name: str = "ce", batch_size: int = 8192) -> dict:
    """Trả về dict(loss, acc, macro_f1) ở chế độ eval() (dropout tắt) và no_grad."""
    model.eval()
    n = len(X)
    total_loss = 0.0
    all_preds = []

    for i in range(0, n, batch_size):
        xb = X[i : i + batch_size]
        yb = y[i : i + batch_size]
        logits = model(xb)

        if loss_name == "ce":
            loss = F.cross_entropy(logits, yb, reduction="sum")
            total_loss += float(loss.item())
        elif loss_name == "mse":
            yb_one_hot = F.one_hot(yb, num_classes=logits.shape[1]).float()
            loss = F.mse_loss(logits, yb_one_hot, reduction="sum")
            total_loss += float(loss.item())

        all_preds.append(logits.argmax(dim=1))

    num_classes = 7
    mean_loss = total_loss / (n * num_classes if loss_name == "mse" else n)

    all_preds_cat = torch.cat(all_preds, dim=0)
    acc = float((all_preds_cat == y).float().mean().item())

    # Ma trận nhầm lẫn
    y_np = y.cpu().numpy()
    pred_np = all_preds_cat.cpu().numpy()
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (y_np, pred_np), 1)
    macro_f1 = macro_f1_from_confusion(cm)

    return {"loss": float(mean_loss), "acc": float(acc), "macro_f1": float(macro_f1)}


def run_experiment(cfg: dict, data: dict) -> dict:
    """Huấn luyện một cấu hình và trả về lịch sử + tóm tắt."""
    set_seed(cfg["seed"])

    device = data["X_tr"].device
    hidden = tuple(cfg.get("hidden", (256, 128)))
    dropout = float(cfg.get("dropout", 0.0))
    init = str(cfg.get("init", "he"))

    model = MLP(hidden=hidden, dropout=dropout, init=init).to(device)
    if hidden in EXPECTED_PARAMS:
        assert count_params(model) == EXPECTED_PARAMS[hidden], (
            f"Số params {count_params(model)} không khớp {EXPECTED_PARAMS[hidden]}"
        )

    lr = float(cfg["lr"])
    optimizer = build_optimizer(
        cfg.get("optimizer", "sgd_momentum"),
        model.parameters(),
        lr=lr,
        weight_decay=float(cfg.get("weight_decay", 0.0)),
        momentum=float(cfg.get("momentum", 0.9)),
    )

    precision = str(cfg.get("precision", "fp32")).lower()
    is_cuda = device.type == "cuda"
    use_amp = (precision in ("fp16", "bf16")) and is_cuda
    amp_dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=(precision == "fp16" and is_cuda))

    # Loss bước 0
    step0_val = evaluate(model, data["X_val"], data["y_val"], loss_name=cfg.get("loss", "ce"))
    step0_loss = step0_val["loss"]

    epochs = int(cfg.get("epochs", 20))
    batch_size = int(cfg.get("batch", 512))
    clip_norm = cfg.get("clip_norm", None)
    loss_name = str(cfg.get("loss", "ce"))

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
        "grad_norm": [],
        "epoch_time_s": [],
    }

    best_val_loss = float("inf")
    best_epoch = 1
    best_state = None
    best_val_acc = 0.0
    best_val_macro_f1 = 0.0
    diverged = False

    # Tập con cố định để tính train loss sau mỗi epoch (50,000 mẫu để tiết kiệm thời gian)
    train_subset_size = min(50000, len(data["X_tr"]))
    X_tr_sub = data["X_tr"][:train_subset_size]
    y_tr_sub = data["y_tr"][:train_subset_size]

    generator = torch.Generator(device="cpu")
    generator.manual_seed(cfg["seed"])

    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(1, epochs + 1):
        if is_cuda:
            torch.cuda.synchronize(device)
        t0 = time.time()

        model.train()
        epoch_grad_norms: list[float] = []

        for xb, yb in iterate_batches(data["X_tr"], data["y_tr"], batch_size=batch_size,
                                      generator=generator, shuffle=True):
            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.autocast("cuda", dtype=amp_dtype):
                    logits = model(xb)
                    loss = compute_loss(logits, yb, loss_name)
            else:
                logits = model(xb)
                loss = compute_loss(logits, yb, loss_name)

            if torch.isnan(loss) or torch.isinf(loss):
                diverged = True
                print(f"[{cfg.get('exp_id')}] Loss phân kỳ (NaN/Inf) tại epoch {epoch}!")
                break

            if precision == "fp16" and is_cuda:
                scaler.scale(loss).backward()
                if clip_norm is not None:
                    scaler.unscale_(optimizer)
                gn = clip_gradients(model.parameters(), clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                gn = clip_gradients(model.parameters(), clip_norm)
                optimizer.step()

            epoch_grad_norms.append(gn)

        if diverged:
            break

        if is_cuda:
            torch.cuda.synchronize(device)
        epoch_time = time.time() - t0

        # Đánh giá cuối epoch ở chế độ eval
        tr_eval = evaluate(model, X_tr_sub, y_tr_sub, loss_name=loss_name)
        val_eval = evaluate(model, data["X_val"], data["y_val"], loss_name=loss_name)

        mean_gn = float(np.mean(epoch_grad_norms)) if epoch_grad_norms else 0.0

        history["epoch"].append(epoch)
        history["train_loss"].append(tr_eval["loss"])
        history["val_loss"].append(val_eval["loss"])
        history["val_acc"].append(val_eval["acc"])
        history["val_macro_f1"].append(val_eval["macro_f1"])
        history["grad_norm"].append(mean_gn)
        history["epoch_time_s"].append(epoch_time)

        if val_eval["loss"] < best_val_loss:
            best_val_loss = val_eval["loss"]
            best_epoch = epoch
            best_val_acc = val_eval["acc"]
            best_val_macro_f1 = val_eval["macro_f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    avg_time_per_epoch = float(np.mean(history["epoch_time_s"])) if history["epoch_time_s"] else 0.0
    peak_mem_MB = float(torch.cuda.max_memory_allocated(device) / (1024 * 1024)) if is_cuda else 0.0

    summary = {
        "step0_loss": float(step0_loss),
        "best_val_loss": float(best_val_loss) if not diverged else float("nan"),
        "best_epoch": int(best_epoch) if not diverged else 0,
        "final_train_loss": float(history["train_loss"][-1]) if history["train_loss"] else float("nan"),
        "final_val_loss": float(history["val_loss"][-1]) if history["val_loss"] else float("nan"),
        "val_acc": float(best_val_acc) if not diverged else 0.0,
        "val_macro_f1": float(best_val_macro_f1) if not diverged else 0.0,
        "time_per_epoch_s": float(avg_time_per_epoch),
        "peak_mem_MB": float(peak_mem_MB),
        "diverged": diverged,
    }

    if best_state is None:
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    return {
        "cfg": cfg,
        "history": history,
        "summary": summary,
        "best_state": best_state,
    }


def write_predictions(row_id: np.ndarray, preds: np.ndarray, path: str) -> None:
    """Ghi file nộp cho scripts/evaluate.py: CSV có tiêu đề `row_id,pred`."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"row_id": row_id.astype(int), "pred": preds.astype(int)})
    df.to_csv(p, index=False)
    print(f"Đã ghi {len(df)} dòng dự đoán ra {path}")


def final_eval(cfg: dict, result: dict, data: dict, pred_path: str) -> None:
    """Dùng MỘT LẦN cho cấu hình cuối cùng (và baseline): nạp best_state, dự đoán eval, ghi predictions."""
    device = data["X_eval"].device
    hidden = tuple(cfg.get("hidden", (256, 128)))
    model = MLP(hidden=hidden, dropout=0.0, init=cfg.get("init", "he")).to(device)
    model.load_state_dict(result["best_state"])

    preds = predict(model, data["X_eval"])
    write_predictions(data["eval_row_id"], preds.cpu().numpy(), pred_path)
    print(f"Để đánh giá, hãy chạy:\n  python scripts/evaluate.py --pred {pred_path}")
