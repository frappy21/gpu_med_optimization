"""
Отдельные эксперименты и профилирование (поиск узких мест):
  1) точность вычислений (fp32 / fp16-autocast) и формат памяти (NCHW / channels_last);
  2) влияние размера батча на скорость шага обучения и инференса;
  3) профилирование baseline-цикла через torch.profiler.

  python src/experiments.py   -> results/experiments.md
"""
import os
import sys
import time
import warnings

import numpy as np
import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from model import MedCNN  # noqa: E402
from train import CONFIGS, NumpyDataset  # noqa: E402
from data import make_dataset  # noqa: E402

dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
W = CONFIGS["baseline"]["widths"]
RES = os.path.join(os.path.dirname(__file__), "..", "results")


def sync():
    if dev.type == "mps":
        torch.mps.synchronize()


def step_time(bs, amp=False, cl=False, block="conv", iters=8):
    torch.manual_seed(0)
    m = MedCNN(block=block, widths=W).to(dev)
    x = torch.randn(bs, 1, 128, 128, device=dev)
    if cl:
        m = m.to(memory_format=torch.channels_last)
        x = x.contiguous(memory_format=torch.channels_last)
    y = torch.randint(0, 2, (bs,), device=dev)
    opt = torch.optim.Adam(m.parameters())
    for i in range(iters + 3):
        if i == 3:
            sync()
            t = time.perf_counter()
        with torch.autocast(dev.type, dtype=torch.float16, enabled=amp):
            loss = F.cross_entropy(m(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    sync()
    return (time.perf_counter() - t) / iters / bs * 1000  # мс на изображение


@torch.inference_mode()
def infer_time(bs, amp=False, block="conv", iters=8):
    m = MedCNN(block=block, widths=W).to(dev).eval()
    x = torch.randn(bs, 1, 128, 128, device=dev)
    for i in range(iters + 3):
        if i == 3:
            sync()
            t = time.perf_counter()
        with torch.autocast(dev.type, dtype=torch.float16, enabled=amp):
            m(x)
    sync()
    return (time.perf_counter() - t) / iters / bs * 1000


def main():
    out = [f"# Эксперименты и профилирование ({dev})", ""]

    out += ["## 1. Точность вычислений и формат памяти (обучение, батч 128, мс/изобр.)", "",
            "| Вариант | conv | dw |", "|---|---|---|"]
    for name, amp, cl in [("fp32, NCHW", False, False), ("fp32, channels_last", False, True),
                          ("fp16 autocast, NCHW", True, False), ("fp16 autocast, channels_last", True, True)]:
        out.append(f"| {name} | {step_time(128, amp, cl):.2f} | {step_time(128, amp, cl, 'dw'):.2f} |")
        print(out[-1], flush=True)

    out += ["", "## 2. Размер батча (fp32)", "",
            "| Батч | обучение conv, мс/изобр. | обучение dw | инференс conv | инференс conv fp16 | инференс dw |",
            "|---|---|---|---|---|---|"]
    for bs in [16, 32, 64, 128, 256]:
        out.append(f"| {bs} | {step_time(bs):.2f} | {step_time(bs, block='dw'):.2f} | {infer_time(bs):.3f} | "
                   f"{infer_time(bs, amp=True):.3f} | {infer_time(bs, block='dw'):.3f} |")
        print(out[-1], flush=True)

    # 3. Профилирование исходного цикла
    x, y = make_dataset(640, 7)
    loader = torch.utils.data.DataLoader(NumpyDataset(x, y, True), batch_size=32, shuffle=True)
    m = MedCNN(widths=W).to(dev)
    opt = torch.optim.Adam(m.parameters())
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            o = m(xb)
            loss = F.cross_entropy(o, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss.item()
            (o.argmax(1) == yb).float().mean().item()
    table = prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=15)
    out += ["", "## 3. torch.profiler: исходный (baseline) цикл обучения, топ-15 операций по времени CPU", "",
            "```", table, "```"]
    print(table)
    open(os.path.join(RES, "experiments.md"), "w").write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
