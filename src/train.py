"""
Обучение и инференс классификатора медицинских снимков на GPU.

Три конфигурации (выбираются флагом --config):
  baseline   — исходная («наивная») версия программы: GPU используется,
               но конвейер данных и цикл обучения написаны неэффективно;
  optimized  — та же модель, оптимизированный конвейер для GPU;
  optimized_dw — оптимизированный конвейер + облегчённая архитектура
               (depthwise-separable свёртки).

Запуск одной конфигурации:
  python src/train.py --config optimized --seed 0 --out results/run.json
"""
import argparse
import json
import os
import resource
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from data import load_splits  # noqa: E402
from model import MedCNN, n_params  # noqa: E402

MEAN, STD = 0.25, 0.20  # статистики яркости набора (посчитаны заранее)

# Ширины слоёв подобраны под Apple M1 с 8 ГБ общей памяти (CPU и GPU делят одну память).
W = (16, 32, 64, 128)
CONFIGS = {
    "baseline": dict(block="conv", widths=W, batch=32, eval_batch=32, lr=1e-3, amp=False, channels_last=False),
    # AMP (fp16) и channels_last проверены отдельно (см. experiments.py) и на M1 ускорения
    # не дали, поэтому в итоговой оптимизированной конфигурации они выключены.
    "optimized": dict(block="conv", widths=W, batch=128, eval_batch=256, lr=2e-3, amp=False, channels_last=False),
    # Кейс 5: исправление просадки качества — батч 64 (по experiments.md он так же быстр, как 128),
    # вдвое больше шагов оптимизатора за эпоху.
    "optimized_v2": dict(block="conv", widths=W, batch=64, eval_batch=256, lr=1.5e-3, amp=False, channels_last=False),
    "optimized_dw": dict(block="dw", widths=W, batch=128, eval_batch=256, lr=3e-3, amp=False, channels_last=False),
}


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def sync(dev):
    if dev.type == "mps":
        torch.mps.synchronize()
    elif dev.type == "cuda":
        torch.cuda.synchronize()


def gpu_mem(dev):
    if dev.type == "mps":
        return torch.mps.current_allocated_memory()
    if dev.type == "cuda":
        return torch.cuda.memory_allocated()
    return 0


# ----------------------------------------------------------------------------
# 1. BASELINE: типичный «наивный» код
# ----------------------------------------------------------------------------
class NumpyDataset(torch.utils.data.Dataset):
    """Предобработка и аугментация каждого снимка по одному на CPU (numpy)."""

    def __init__(self, x, y, train):
        self.x, self.y, self.train = x, y, train

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        img = self.x[i].astype(np.float32) / 255.0
        img = (img - MEAN) / STD
        if self.train and np.random.rand() < 0.5:
            img = img[:, ::-1].copy()  # горизонтальное отражение
        return torch.from_numpy(img[None]), int(self.y[i])


def train_baseline(model, data, cfg, dev, epochs, stats):
    (xtr, ytr), _ = data
    loader = torch.utils.data.DataLoader(NumpyDataset(xtr, ytr, True), batch_size=cfg["batch"], shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, cfg["lr"], total_steps=epochs * len(loader))
    model.train()
    for ep in range(epochs):
        t_data = t_gpu = 0.0
        t0 = time.perf_counter()
        for xb, yb in loader:
            t1 = time.perf_counter()
            t_data += t1 - t0
            xb, yb = xb.to(dev), yb.to(dev)  # копирование каждого батча CPU -> GPU
            out = model(xb)
            loss = F.cross_entropy(out, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            # .item() на каждом шаге => принудительная синхронизация CPU и GPU
            stats["loss_hist"].append(loss.item())
            acc = (out.argmax(1) == yb).float().mean().item()  # noqa: F841
            stats["peak_mem"] = max(stats["peak_mem"], gpu_mem(dev))
            t0 = time.perf_counter()
            t_gpu += t0 - t1
        stats["epoch_data_s"].append(t_data)
        stats["epoch_gpu_s"].append(t_gpu)


def predict_baseline(model, x, cfg, dev):
    model.eval()
    probs = []
    loader = torch.utils.data.DataLoader(NumpyDataset(x, np.zeros(len(x)), False), batch_size=cfg["eval_batch"])
    for xb, _ in loader:  # без torch.no_grad(): строится лишний граф вычислений
        out = model(xb.to(dev))
        probs.append(F.softmax(out, 1)[:, 1].detach().cpu().numpy())
    return np.concatenate(probs)


# ----------------------------------------------------------------------------
# 2. OPTIMIZED: конвейер, построенный вокруг GPU
# ----------------------------------------------------------------------------
def gpu_preprocess(xb_u8, train, channels_last):
    """Нормализация и аугментация целого батча прямо на GPU."""
    x = xb_u8.unsqueeze(1).float().mul_(1.0 / 255.0).sub_(MEAN).div_(STD)
    if train:
        flip = torch.rand(x.shape[0], device=x.device) < 0.5
        x = torch.where(flip[:, None, None, None], x.flip(-1), x)
    if channels_last:
        x = x.contiguous(memory_format=torch.channels_last)
    return x


def train_optimized(model, data, cfg, dev, epochs, stats):
    (xtr, ytr), _ = data
    # весь набор один раз загружается в память GPU в компактном uint8 (4000*128*128 = 62 МБ)
    X = torch.from_numpy(xtr).to(dev)
    Y = torch.from_numpy(ytr).to(dev)
    n, bs = len(X), cfg["batch"]
    steps = (n + bs - 1) // bs
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, cfg["lr"], total_steps=epochs * steps)
    # GradScaler на MPS (torch 2.5) падает: MPS не поддерживает float64 -> используем только на CUDA
    scaler = torch.amp.GradScaler(dev.type, enabled=cfg["amp"] and dev.type == "cuda")
    model.train()
    for ep in range(epochs):
        t0 = time.perf_counter()
        perm = torch.randperm(n, device=dev)
        loss_sum = torch.zeros((), device=dev)
        for s in range(steps):
            idx = perm[s * bs:(s + 1) * bs]
            xb = gpu_preprocess(X[idx], True, cfg["channels_last"])
            yb = Y[idx]
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=cfg["amp"]):
                out = model(xb)
                loss = F.cross_entropy(out, yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            loss_sum += loss.detach()  # накопление на GPU, без синхронизации
            if s % 10 == 0:
                stats["peak_mem"] = max(stats["peak_mem"], gpu_mem(dev))
        mean_loss = (loss_sum / steps).item()  # одна синхронизация за эпоху
        stats["loss_hist"].append(mean_loss)
        sync(dev)
        stats["epoch_data_s"].append(0.0)
        stats["epoch_gpu_s"].append(time.perf_counter() - t0)


@torch.inference_mode()
def predict_optimized(model, x, cfg, dev):
    model.eval()
    X = torch.from_numpy(x).to(dev)
    probs = []
    for i in range(0, len(X), cfg["eval_batch"]):
        xb = gpu_preprocess(X[i:i + cfg["eval_batch"]], False, cfg["channels_last"])
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=cfg["amp"]):
            out = model(xb)
        probs.append(F.softmax(out.float(), 1)[:, 1])
    return torch.cat(probs).cpu().numpy()  # одна передача GPU -> CPU


# ----------------------------------------------------------------------------
def roc_auc(y, p):
    """ROC AUC через статистику Манна-Уитни (без scikit-learn)."""
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p))
    ranks[order] = np.arange(1, len(p) + 1)
    # усреднение рангов для совпадающих значений
    _, inv, cnt = np.unique(p, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    ranks = (sums / cnt)[inv]
    n1 = int((y == 1).sum())
    n0 = len(y) - n1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def metrics(y, p):
    pred = (p >= 0.5).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return dict(
        accuracy=(tp + tn) / len(y),
        sensitivity=tp / max(tp + fn, 1),  # доля найденных патологий
        specificity=tn / max(tn + fp, 1),
        f1=2 * tp / max(2 * tp + fp + fn, 1),
        roc_auc=roc_auc(y, p),
        has_nan=bool(np.isnan(p).any()),
    )


def run(config, seed, epochs, n_train, n_test):
    cfg = CONFIGS[config]
    dev = get_device()
    torch.manual_seed(seed)
    np.random.seed(seed)
    data = load_splits(n_train, n_test)
    (_, _), (xte, yte) = data

    model = MedCNN(block=cfg["block"], widths=cfg["widths"]).to(dev)
    if cfg["channels_last"]:
        model = model.to(memory_format=torch.channels_last)

    stats = dict(loss_hist=[], epoch_data_s=[], epoch_gpu_s=[], peak_mem=0)
    sync(dev)
    t0 = time.perf_counter()
    (train_baseline if config == "baseline" else train_optimized)(model, data, cfg, dev, epochs, stats)
    sync(dev)
    train_s = time.perf_counter() - t0

    predict = predict_baseline if config == "baseline" else predict_optimized
    predict(model, xte[:256], cfg, dev)  # прогрев
    sync(dev)
    lat = []
    for _ in range(3):
        t0 = time.perf_counter()
        probs = predict(model, xte, cfg, dev)
        sync(dev)
        lat.append(time.perf_counter() - t0)
    infer_s = float(np.median(lat))

    # задержка на один снимок (сценарий «врач загрузил один срез»)
    one = xte[:1]
    single = []
    for _ in range(30):
        t0 = time.perf_counter()
        predict(model, one, cfg, dev)
        sync(dev)
        single.append(time.perf_counter() - t0)

    return dict(
        config=config, seed=seed, device=str(dev), epochs=epochs, n_train=n_train, n_test=n_test,
        params=n_params(model),
        train_time_s=train_s,
        epoch_times_s=[a + b for a, b in zip(stats["epoch_data_s"], stats["epoch_gpu_s"])],
        epoch_data_wait_s=stats["epoch_data_s"],
        train_throughput_img_s=n_train * epochs / train_s,
        infer_time_s=infer_s,
        infer_throughput_img_s=n_test / infer_s,
        single_image_latency_ms=float(np.median(single) * 1000),
        peak_gpu_mem_mb=stats["peak_mem"] / 2 ** 20,
        driver_gpu_mem_mb=(torch.mps.driver_allocated_memory() / 2 ** 20) if dev.type == "mps" else None,
        max_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20,  # macOS: байты
        final_loss=stats["loss_hist"][-1],
        quality=metrics(yte, probs),
        probs_head=[round(float(v), 4) for v in probs[:20]],
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=CONFIGS, default="optimized")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = run(a.config, a.seed, a.epochs, a.n_train, a.n_test)
    txt = json.dumps(res, ensure_ascii=False, indent=2)
    if a.out:
        with open(a.out, "w") as f:
            f.write(txt)
    print(txt)
