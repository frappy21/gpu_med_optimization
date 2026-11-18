"""
Сравнение конфигураций: каждая запускается в отдельном процессе (чистая память),
несколько раз с разными seed. Итоги -> results/summary.json и results/summary.md

  python src/benchmark.py --repeats 3
"""
import argparse
import json
import os
import statistics as st
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
CONFIGS = ["baseline", "optimized", "optimized_v2", "optimized_dw"]


def ms(vals):
    return st.mean(vals), (st.stdev(vals) if len(vals) > 1 else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(RES, exist_ok=True)

    runs = {c: [] for c in CONFIGS}
    for seed in range(a.repeats):
        for c in CONFIGS:
            out = os.path.join(RES, f"run_{c}_seed{seed}.json")
            print(f"-> {c} seed={seed}", flush=True)
            subprocess.run([sys.executable, os.path.join(HERE, "train.py"), "--config", c, "--seed", str(seed),
                            "--epochs", str(a.epochs), "--out", out], check=True, stdout=subprocess.DEVNULL)
            runs[c].append(json.load(open(out)))

    summary = {}
    for c, rs in runs.items():
        q = lambda k: ms([r["quality"][k] for r in rs])  # noqa: E731
        summary[c] = dict(
            params=rs[0]["params"],
            train_time_s=ms([r["train_time_s"] for r in rs]),
            train_throughput=ms([r["train_throughput_img_s"] for r in rs]),
            data_wait_share=ms([sum(r["epoch_data_wait_s"]) / r["train_time_s"] for r in rs]),
            infer_throughput=ms([r["infer_throughput_img_s"] for r in rs]),
            latency_ms=ms([r["single_image_latency_ms"] for r in rs]),
            peak_gpu_mem_mb=ms([r["peak_gpu_mem_mb"] for r in rs]),
            max_rss_mb=ms([r["max_rss_mb"] for r in rs]),
            accuracy=q("accuracy"), sensitivity=q("sensitivity"), specificity=q("specificity"),
            f1=q("f1"), roc_auc=q("roc_auc"),
            nan_runs=sum(r["quality"]["has_nan"] for r in rs),
            epoch_time_cv=ms([st.pstdev(r["epoch_times_s"]) / st.mean(r["epoch_times_s"]) for r in rs]),
        )
    json.dump(dict(device=runs["baseline"][0]["device"], repeats=a.repeats, epochs=a.epochs, summary=summary),
              open(os.path.join(RES, "summary.json"), "w"), ensure_ascii=False, indent=2)

    b = summary["baseline"]
    rows = [
        ("Параметров модели", "params", "{:,}", False),
        ("Время обучения, с", "train_time_s", "{:.1f}", True),
        ("Скорость обучения, изобр./с", "train_throughput", "{:.0f}", True),
        ("Доля ожидания данных (CPU), %", "data_wait_share", "{:.1%}", True),
        ("Инференс, изобр./с", "infer_throughput", "{:.0f}", True),
        ("Задержка 1 снимка, мс", "latency_ms", "{:.1f}", True),
        ("Пик памяти GPU (тензоры), МБ", "peak_gpu_mem_mb", "{:.0f}", True),
        ("Пик RSS процесса, МБ", "max_rss_mb", "{:.0f}", True),
        ("Accuracy", "accuracy", "{:.3f}", True),
        ("Чувствительность", "sensitivity", "{:.3f}", True),
        ("Специфичность", "specificity", "{:.3f}", True),
        ("F1", "f1", "{:.3f}", True),
        ("ROC AUC", "roc_auc", "{:.4f}", True),
        ("Разброс времени эпох (CV)", "epoch_time_cv", "{:.1%}", True),
        ("Запусков с NaN", "nan_runs", "{}", False),
    ]
    lines = [f"Устройство: {b and runs['baseline'][0]['device']}, запусков на конфигурацию: {a.repeats}, эпох: {a.epochs}", "",
             "| Показатель | " + " | ".join(CONFIGS) + " |", "|---|" + "---|" * len(CONFIGS)]
    for title, k, fmt, has_sd in rows:
        cells = []
        for c in CONFIGS:
            v = summary[c][k]
            cells.append(f"{fmt.format(v[0])} ± {fmt.format(v[1])}" if has_sd else fmt.format(v))
        lines.append(f"| {title} | " + " | ".join(cells) + " |")
    lines += ["", "Ускорение относительно baseline:"]
    for c in CONFIGS[1:]:
        s = summary[c]
        lines.append(f"- {c}: обучение x{b['train_time_s'][0] / s['train_time_s'][0]:.2f}, "
                     f"инференс x{s['infer_throughput'][0] / b['infer_throughput'][0]:.2f}")
    md = "\n".join(lines)
    open(os.path.join(RES, "summary.md"), "w").write(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
