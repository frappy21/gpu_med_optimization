"""Свёрточные сети для бинарной классификации снимков (норма / патология)."""
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Обычный блок: Conv3x3 -> BN -> ReLU (x2) -> MaxPool."""

    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.net(x)


class DWBlock(nn.Module):
    """Облегчённый блок: depthwise-separable свёртки (идея MobileNet)."""

    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cin, 3, padding=1, groups=cin, bias=False), nn.BatchNorm2d(cin), nn.ReLU(inplace=True),
            nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, groups=cout, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.net(x)


class MedCNN(nn.Module):
    def __init__(self, block="conv", widths=(32, 64, 128, 256)):
        super().__init__()
        B = ConvBlock if block == "conv" else DWBlock
        # первый блок всегда обычный: у входа 1 канал, depthwise там бессмыслен
        layers = [ConvBlock(1, widths[0])]
        for a, b in zip(widths[:-1], widths[1:]):
            layers.append(B(a, b))
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.2), nn.Linear(widths[-1], 2))

    def forward(self, x):
        return self.head(self.features(x))


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
