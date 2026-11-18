"""
Генератор синтетического набора «медицинских» снимков (имитация срезов МРТ/КТ).

Почему синтетика: реальные медицинские данные являются персональными
(ФЗ-152), поэтому для учебно-исследовательской задачи используется
воспроизводимый генератор. Каждый снимок — 1 канал, 128x128, uint8:
  * класс 0 «норма»     — эллипс «мозга» со структурой тканей и шумом;
  * класс 1 «патология» — то же + одно-два ярких очага («опухоль») случайного
    размера, формы и контраста.
"""
import numpy as np

IMG = 128


def _make_one(rng: np.random.Generator, label: int) -> np.ndarray:
    yy, xx = np.mgrid[0:IMG, 0:IMG].astype(np.float32)
    cy, cx = IMG / 2 + rng.normal(0, 4, 2)
    ry, rx = rng.uniform(42, 56), rng.uniform(36, 50)
    head = (((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2) <= 1.0

    img = np.zeros((IMG, IMG), np.float32)
    img[head] = 0.45
    # «ткани»: низкочастотная текстура
    for _ in range(6):
        fy, fx = rng.uniform(0.03, 0.12, 2)
        ph = rng.uniform(0, 2 * np.pi, 2)
        img += head * 0.05 * np.sin(fy * yy + ph[0]) * np.cos(fx * xx + ph[1])
    # «желудочки» — тёмные области в центре
    vent = (((yy - cy) / 10) ** 2 + ((xx - cx) / 5) ** 2) <= 1.0
    img[vent] = 0.2

    if label == 1:
        for _ in range(rng.integers(1, 3)):
            ang = rng.uniform(0, 2 * np.pi)
            dist = rng.uniform(0, 0.6)
            ly = cy + np.sin(ang) * dist * ry
            lx = cx + np.cos(ang) * dist * rx
            r = rng.uniform(3, 9)
            contrast = rng.uniform(0.12, 0.35)
            blob = np.exp(-(((yy - ly) ** 2 + (xx - lx) ** 2) / (2 * r ** 2)))
            img += head * contrast * blob

    img += rng.normal(0, 0.06, img.shape).astype(np.float32)  # шум аппарата
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def make_dataset(n: int, seed: int):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, n).astype(np.int64)
    images = np.stack([_make_one(rng, int(l)) for l in labels])
    return images, labels


def load_splits(n_train=4000, n_test=1000, seed=42):
    xtr, ytr = make_dataset(n_train, seed)
    xte, yte = make_dataset(n_test, seed + 1)
    return (xtr, ytr), (xte, yte)
