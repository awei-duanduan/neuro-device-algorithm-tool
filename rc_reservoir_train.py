#!/usr/bin/env python
"""Reservoir-computing readout training for device pulse-response tables.

The flow mirrors the useful core of rc_pndi:
1. binarize image pixels,
2. reshape them into short pulse codes,
3. map each code to a measured/simulated device response,
4. train only a linear readout layer.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import EMNIST, FashionMNIST, MNIST


CLASS_LABELS = {
    "mnist": [str(i) for i in range(10)],
    "fashion": [
        "T-shirt",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "AnkleBoot",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["single", "paper-multitask"], default="single")
    parser.add_argument("--dataset", choices=["mnist", "fashion"], default="mnist")
    parser.add_argument("--device-response", required=True)
    parser.add_argument("--output-dir", default="rc_results_gui")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--train-limit", type=int, default=10000)
    parser.add_argument("--test-limit", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_response_table(path: str | os.PathLike[str]) -> np.ndarray:
    rows: dict[int, float] = {}
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = {field.lower(): field for field in (reader.fieldnames or [])}
        state_col = fields.get("input_state_5bit") or fields.get("input_state_4bit") or fields.get("state") or fields.get("code")
        response_col = fields.get("response") or fields.get("value") or fields.get("epsp")
        if not state_col or not response_col:
            raise ValueError("Device response CSV must contain input_state_5bit/input_state_4bit and response columns.")
        for row in reader:
            state = str(row[state_col]).strip()
            if set(state) <= {"0", "1"}:
                idx = int(state, 2)
            else:
                idx = int(float(state))
            rows[idx] = float(row[response_col])
    n = max(rows) + 1
    if n not in (16, 32):
        raise ValueError("Device response table must contain 16 states for 4-bit or 32 states for 5-bit coding.")
    values = np.full(n, np.nan, dtype=np.float32)
    for idx, val in rows.items():
        if idx < 0 or idx >= n:
            raise ValueError(f"Pulse state out of range: {idx}")
        values[idx] = val
    if np.isnan(values).any():
        width = int(np.log2(n))
        missing = [format(i, f"0{width}b") for i, val in enumerate(values) if np.isnan(val)]
        raise ValueError(f"Device response table is missing states: {', '.join(missing)}")
    std = float(values.std())
    if std <= 0:
        raise ValueError("Device responses must not all be identical.")
    return ((values - values.mean()) / std).astype(np.float32)


def load_raw_response_table(path: str | os.PathLike[str]) -> np.ndarray:
    rows: dict[int, float] = {}
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = {field.lower(): field for field in (reader.fieldnames or [])}
        state_col = fields.get("input_state_5bit") or fields.get("input_state_4bit") or fields.get("state") or fields.get("code")
        response_col = fields.get("response") or fields.get("value") or fields.get("epsp")
        for row in reader:
            state = str(row[state_col]).strip()
            idx = int(state, 2) if set(state) <= {"0", "1"} else int(float(state))
            rows[idx] = float(row[response_col])
    n = max(rows) + 1
    values = np.full(n, np.nan, dtype=np.float32)
    for idx, val in rows.items():
        values[idx] = val
    if np.isnan(values).any():
        raise ValueError("Raw device response table is incomplete.")
    return values


def load_images(dataset: str, data_root: Path, train: bool, limit: int) -> tuple[torch.Tensor, torch.Tensor]:
    cls = FashionMNIST if dataset == "fashion" else MNIST
    ds = cls(root=str(data_root), train=train, download=True)
    images = ds.data.float() / 255.0
    labels = torch.as_tensor(ds.targets, dtype=torch.long)
    if limit > 0:
        images = images[:limit]
        labels = labels[:limit]
    return images, labels


def reservoir_features(
    images: torch.Tensor,
    responses: np.ndarray,
    threshold: float,
    device: torch.device,
    pulse_bits: int = 4,
    stats: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    # Match the rc_pndi image preprocessing: crop 28x28 to 22x20 = 440 pixels,
    # then each row is split into 4 pulse words. 4-bit gives 110 states;
    # 5-bit gives the Nature Communications p-NDI setting with 88 states.
    imgs = images.to(device)
    crop = imgs[:, 4:26, 5:25]
    binary = (crop > threshold).to(torch.long)
    if crop.shape[1] * crop.shape[2] % pulse_bits != 0:
        raise ValueError("Cropped image pixel count must be divisible by pulse_bits.")
    chunks = binary.reshape(binary.shape[0], -1, pulse_bits)
    weights = torch.tensor([2 ** i for i in range(pulse_bits - 1, -1, -1)], device=device, dtype=torch.long)
    codes = (chunks * weights).sum(dim=2)
    response_tensor = torch.tensor(responses, device=device, dtype=torch.float32)
    feat = response_tensor[codes]

    if stats is None:
        mean = feat.mean(dim=0, keepdim=True)
        std = feat.std(dim=0, keepdim=True).clamp_min(1e-6)
        stats = (mean, std)
    feat = (feat - stats[0]) / stats[1]
    return feat.detach().cpu(), stats


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, float, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    conf = np.zeros((10, 10), dtype=np.int64)
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            pred = logits.argmax(dim=1)
            total_loss += float(loss.item()) * y.numel()
            total += int(y.numel())
            correct += int((pred == y).sum().item())
            for t, p in zip(y.cpu().numpy(), pred.cpu().numpy()):
                conf[int(t), int(p)] += 1
    return total_loss / max(total, 1), 100.0 * correct / max(total, 1), conf


def save_csv(path: Path, header: list[str], rows: list[list[float | int | str]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def plot_loss(history: list[dict[str, float]], path: Path) -> None:
    epochs = [row["epoch"] for row in history]
    plt.figure(figsize=(7, 4.8))
    plt.plot(epochs, [row["train_loss"] for row in history], "k-", label="training")
    plt.plot(epochs, [row["test_loss"] for row in history], "r-", label="testing")
    plt.plot(epochs, [row["val_loss"] for row in history], "b-", label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_accuracy(history: list[dict[str, float]], path: Path) -> None:
    epochs = [row["epoch"] for row in history]
    plt.figure(figsize=(7, 4.8))
    plt.plot(epochs, [row["train_acc"] for row in history], "k-", label="training")
    plt.plot(epochs, [row["test_acc"] for row in history], "r-", label="testing")
    plt.plot(epochs, [row["val_acc"] for row in history], "b-", label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy %")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_confusion(percent: np.ndarray, labels: list[str], accuracy: float, path: Path) -> None:
    n_class = len(labels)
    plt.figure(figsize=(7.2, 6.2))
    im = plt.imshow(percent, cmap="Blues", vmin=0, vmax=1)
    plt.title(f"Accuracy: {accuracy:.2f}%")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(np.arange(n_class), labels, rotation=45, ha="right")
    plt.yticks(np.arange(n_class), labels)
    for i in range(n_class):
        for j in range(n_class):
            val = percent[i, j] * 100.0
            color = "white" if percent[i, j] > 0.55 else "black"
            plt.text(j, i, f"{val:.1f}%", ha="center", va="center", color=color, fontsize=8)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def apply_paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 9,
            "axes.linewidth": 1.0,
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def plot_device_response(raw_responses: np.ndarray, normalized_responses: np.ndarray, path: Path) -> None:
    apply_paper_style()
    pulse_bits = int(np.log2(len(raw_responses)))
    states = [format(i, f"0{pulse_bits}b") for i in range(len(raw_responses))]
    x = np.arange(len(raw_responses))
    fig, ax1 = plt.subplots(figsize=(7.2, 3.8))
    ax1.plot(x, raw_responses, "k.", ms=7)
    for y in raw_responses:
        ax1.axhline(y, color="#f07c7c", lw=0.6, ls="--", alpha=0.45)
    ax1.set_xlabel("Input State")
    ax1.set_ylabel("Device response")
    ax1.set_xticks(x)
    ax1.set_xticklabels(states, rotation=60, ha="right")
    ax2 = ax1.twinx()
    ax2.plot(x, normalized_responses, color="#1f77b4", lw=1.4, alpha=0.8)
    ax2.set_ylabel("Normalized response", color="#1f77b4")
    ax2.tick_params(axis="y", colors="#1f77b4")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_feature_heatmap(features: torch.Tensor, labels: torch.Tensor, path: Path, max_samples: int = 600) -> None:
    apply_paper_style()
    x = features.numpy()
    y = labels.numpy()
    order = np.argsort(y)
    if order.size > max_samples:
        keep = np.linspace(0, order.size - 1, max_samples).astype(int)
        order = order[keep]
    x = x[order]
    y = y[order]
    fig, ax = plt.subplots(figsize=(8.4, 3.4))
    im = ax.imshow(x.T, aspect="auto", cmap="coolwarm", interpolation="nearest")
    boundaries = np.where(np.diff(y) != 0)[0]
    for b in boundaries:
        ax.axvline(b + 0.5, color="white", lw=0.7, alpha=0.8)
    ax.set_xlabel("Samples sorted by class")
    ax.set_ylabel("Reservoir state index")
    ax.set_title("Stacked reservoir feature vectors")
    cbar = fig.colorbar(im, ax=ax, fraction=0.018, pad=0.015)
    cbar.set_label("Feature value")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def compute_pca(features: torch.Tensor, n_components: int = 3, max_samples: int = 2500) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = features.numpy()
    if x.shape[0] > max_samples:
        idx = np.linspace(0, x.shape[0] - 1, max_samples).astype(int)
        x = x[idx]
    else:
        idx = np.arange(x.shape[0])
    x = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    coords = x @ vt[:n_components].T
    return coords, vt[:n_components], idx


def plot_feature_pca(features: torch.Tensor, labels: torch.Tensor, path: Path, max_samples: int = 2500) -> None:
    apply_paper_style()
    x = features.numpy()
    y = labels.numpy()
    if x.shape[0] > max_samples:
        idx = np.linspace(0, x.shape[0] - 1, max_samples).astype(int)
        x = x[idx]
        y = y[idx]
    x = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    coords = x @ vt[:3].T
    fig = plt.figure(figsize=(5.8, 4.8))
    ax = fig.add_subplot(111, projection="3d")
    cmap = plt.get_cmap("tab10")
    for cls in range(10):
        mask = y == cls
        if np.any(mask):
            ax.scatter(coords[mask, 0], coords[mask, 1], coords[mask, 2], s=7, alpha=0.65, color=cmap(cls), label=str(cls))
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title("PCA of reservoir features")
    ax.legend(ncol=2, fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_feature_matrix_csv(
    path: Path,
    features: torch.Tensor,
    labels: torch.Tensor,
    label_names: list[str],
    max_samples: int = 5000,
) -> None:
    sample_count = min(max_samples, int(features.shape[0]))
    indices = np.linspace(0, int(features.shape[0]) - 1, sample_count).astype(int)
    feat = features[indices].numpy()
    lbl = labels[indices].numpy()
    rows = []
    for out_idx, src_idx in enumerate(indices):
        label_id = int(lbl[out_idx])
        label_name = label_names[label_id] if 0 <= label_id < len(label_names) else str(label_id)
        rows.append([int(src_idx), label_id, label_name] + [float(v) for v in feat[out_idx].tolist()])
    save_csv(
        path,
        ["sample_index", "label_id", "label_name"] + [f"state_{i:03d}" for i in range(features.shape[1])],
        rows,
    )


def save_feature_pca_csv(
    path: Path,
    features: torch.Tensor,
    labels: torch.Tensor,
    label_names: list[str],
    max_samples: int = 5000,
) -> None:
    coords, _, indices = compute_pca(features, n_components=3, max_samples=max_samples)
    lbl = labels[indices].numpy()
    rows = []
    for out_idx, src_idx in enumerate(indices):
        label_id = int(lbl[out_idx])
        label_name = label_names[label_id] if 0 <= label_id < len(label_names) else str(label_id)
        rows.append([
            int(src_idx),
            label_id,
            label_name,
            float(coords[out_idx, 0]),
            float(coords[out_idx, 1]),
            float(coords[out_idx, 2]),
        ])
    save_csv(path, ["sample_index", "label_id", "label_name", "pc1", "pc2", "pc3"], rows)


def plot_paper_summary(
    history: list[dict[str, float]],
    features: torch.Tensor,
    labels: torch.Tensor,
    conf_percent: np.ndarray,
    class_labels: list[str],
    accuracy: float,
    path: Path,
) -> None:
    apply_paper_style()
    x = features.numpy()
    y = labels.numpy()
    order = np.argsort(y)
    if order.size > 480:
        keep = np.linspace(0, order.size - 1, 480).astype(int)
        order = order[keep]
    heat = x[order].T

    pca_x = x
    pca_y = y
    if pca_x.shape[0] > 1800:
        idx = np.linspace(0, pca_x.shape[0] - 1, 1800).astype(int)
        pca_x = pca_x[idx]
        pca_y = pca_y[idx]
    pca_x = pca_x - pca_x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(pca_x, full_matrices=False)
    coords = pca_x @ vt[:3].T

    epochs = [row["epoch"] for row in history]
    fig = plt.figure(figsize=(11.0, 7.8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], width_ratios=[1.15, 1.0], hspace=0.36, wspace=0.30)

    ax_a = fig.add_subplot(gs[0, 0])
    im_a = ax_a.imshow(heat, aspect="auto", cmap="coolwarm", interpolation="nearest")
    ax_a.set_xlabel("Samples sorted by class")
    ax_a.set_ylabel("Reservoir state")
    ax_a.set_title("a  Reservoir feature map", loc="left", fontweight="bold")
    fig.colorbar(im_a, ax=ax_a, fraction=0.026, pad=0.015)

    ax_b = fig.add_subplot(gs[0, 1], projection="3d")
    cmap = plt.get_cmap("tab10")
    for cls in range(10):
        mask = pca_y == cls
        if np.any(mask):
            ax_b.scatter(coords[mask, 0], coords[mask, 1], coords[mask, 2], s=6, alpha=0.62, color=cmap(cls))
    ax_b.set_xlabel("PC1")
    ax_b.set_ylabel("PC2")
    ax_b.set_zlabel("PC3")
    ax_b.set_title("b  Feature distribution", loc="left", fontweight="bold")

    ax_c = fig.add_subplot(gs[1, 0])
    ax_c.plot(epochs, [row["train_loss"] for row in history], "k-", lw=1.6, label="training")
    ax_c.plot(epochs, [row["test_loss"] for row in history], "r-", lw=1.6, label="testing")
    ax_c.plot(epochs, [row["val_loss"] for row in history], "b-", lw=1.6, label="validation")
    ax_c.set_xlabel("Epoch")
    ax_c.set_ylabel("Cross-Entropy")
    ax_c.set_title("c  Readout training", loc="left", fontweight="bold")
    ax_c.legend(frameon=False)
    ax_c.grid(alpha=0.2)
    ax_c2 = ax_c.twinx()
    ax_c2.plot(epochs, [row["test_acc"] for row in history], color="#2ca25f", lw=1.5, ls="--", label="testing accuracy")
    ax_c2.set_ylabel("Accuracy (%)", color="#2ca25f")
    ax_c2.tick_params(axis="y", colors="#2ca25f")

    ax_d = fig.add_subplot(gs[1, 1])
    im_d = ax_d.imshow(conf_percent, cmap="YlGnBu", vmin=0, vmax=1)
    ax_d.set_title(f"d  Confusion matrix  Accuracy: {accuracy:.2f}%", loc="left", fontweight="bold")
    ax_d.set_xlabel("Predicted")
    ax_d.set_ylabel("True")
    ax_d.set_xticks(np.arange(10))
    ax_d.set_yticks(np.arange(10))
    ax_d.set_xticklabels(class_labels, rotation=45, ha="right", fontsize=7)
    ax_d.set_yticklabels(class_labels, fontsize=7)
    for i in range(10):
        for j in range(10):
            val = conf_percent[i, j] * 100.0
            color = "white" if conf_percent[i, j] > 0.55 else "black"
            ax_d.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=6, color=color)
    fig.colorbar(im_d, ax=ax_d, fraction=0.046, pad=0.03)
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def split_train_val(
    feat: torch.Tensor,
    labels: torch.Tensor,
    val_fraction: float = 0.15,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n_val = max(1, int(val_fraction * feat.shape[0]))
    return feat[:-n_val], labels[:-n_val], feat[-n_val:], labels[-n_val:]


def limit_tensors(images: torch.Tensor, labels: torch.Tensor, limit: int) -> tuple[torch.Tensor, torch.Tensor]:
    if limit > 0 and limit < labels.numel():
        keep_indices = []
        classes = torch.unique(labels).tolist()
        base = max(1, limit // len(classes))
        remainder = max(0, limit - base * len(classes))
        generator = torch.Generator().manual_seed(1234)
        for pos, cls in enumerate(classes):
            cls_idx = torch.where(labels == int(cls))[0]
            cls_idx = cls_idx[torch.randperm(cls_idx.numel(), generator=generator)]
            take = min(cls_idx.numel(), base + (1 if pos < remainder else 0))
            keep_indices.append(cls_idx[:take])
        indices = torch.cat(keep_indices)
        indices = indices[torch.randperm(indices.numel(), generator=generator)]
        return images[indices], labels[indices]
    return images, labels


def load_paper_task_images(
    task_name: str,
    data_root: Path,
    train: bool,
    limit: int,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    if task_name == "MNIST":
        ds = MNIST(root=str(data_root), train=train, download=True)
        images = ds.data.float() / 255.0
        labels = torch.as_tensor(ds.targets, dtype=torch.long)
        class_labels = [str(i) for i in range(10)]
    elif task_name == "EMNIST":
        ds = EMNIST(root=str(data_root), split="letters", train=train, download=True)
        images = ds.data.float() / 255.0
        images = torch.rot90(images, k=3, dims=(1, 2)).flip(2)
        labels = torch.as_tensor(ds.targets, dtype=torch.long)
        keep_values = torch.tensor([12, 13, 19], dtype=torch.long)
        keep = (labels[:, None] == keep_values[None, :]).any(dim=1)
        images = images[keep]
        labels = labels[keep]
        remap = {12: 0, 13: 1, 19: 2}
        labels = torch.tensor([remap[int(x)] for x in labels], dtype=torch.long)
        class_labels = ["L", "M", "S"]
    elif task_name == "FMNIST":
        ds = FashionMNIST(root=str(data_root), train=train, download=True)
        images = ds.data.float() / 255.0
        labels = torch.as_tensor(ds.targets, dtype=torch.long)
        keep_values = torch.tensor([0, 1, 3, 8, 9], dtype=torch.long)
        keep = (labels[:, None] == keep_values[None, :]).any(dim=1)
        images = images[keep]
        labels = labels[keep]
        remap = {0: 0, 1: 1, 3: 2, 8: 3, 9: 4}
        labels = torch.tensor([remap[int(x)] for x in labels], dtype=torch.long)
        class_labels = ["T-shirt", "Pants", "Dress", "Bag", "Shoes"]
    else:
        raise ValueError(f"Unknown paper task: {task_name}")
    images, labels = limit_tensors(images, labels, limit)
    return images, labels, class_labels


def make_loaders(
    train_feat: torch.Tensor,
    train_labels: torch.Tensor,
    val_feat: torch.Tensor,
    val_labels: torch.Tensor,
    test_feat: torch.Tensor,
    test_labels: torch.Tensor,
    batch_size: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    return (
        DataLoader(TensorDataset(train_feat, train_labels), batch_size=batch_size, shuffle=True),
        DataLoader(TensorDataset(val_feat, val_labels), batch_size=batch_size, shuffle=False),
        DataLoader(TensorDataset(test_feat, test_labels), batch_size=batch_size, shuffle=False),
    )


def train_readout(
    task_name: str,
    num_classes: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    feature_dim: int,
    epochs: int,
    lr: float,
    device: torch.device,
    progress_start: int,
    progress_total: int,
) -> tuple[nn.Module, list[dict[str, float]], np.ndarray]:
    model = nn.Linear(feature_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history: list[dict[str, float]] = []
    final_conf = np.zeros((num_classes, num_classes), dtype=np.int64)
    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        train_loss, train_acc, _ = evaluate(model, train_loader, criterion, device)
        val_loss, val_acc, _ = evaluate(model, val_loader, criterion, device)
        test_loss, test_acc, final_conf = evaluate(model, test_loader, criterion, device)
        row = {
            "epoch": float(epoch),
            "train_loss": train_loss,
            "test_loss": test_loss,
            "val_loss": val_loss,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "val_acc": val_acc,
        }
        history.append(row)
        global_epoch = progress_start + epoch
        print(
            "RC Epoch "
            f"{global_epoch}/{progress_total} task={task_name} local_epoch={epoch}/{epochs} "
            f"train_loss={train_loss:.6f} test_loss={test_loss:.6f} val_loss={val_loss:.6f} "
            f"train_acc={train_acc:.2f} test_acc={test_acc:.2f} val_acc={val_acc:.2f}",
            flush=True,
        )
    return model, history, final_conf


def save_task_outputs(
    output_dir: Path,
    task_name: str,
    history: list[dict[str, float]],
    conf: np.ndarray,
    class_labels: list[str],
) -> dict[str, str | float]:
    prefix = task_name.lower()
    loss_csv = output_dir / f"{prefix}_loss_history.csv"
    acc_csv = output_dir / f"{prefix}_accuracy_history.csv"
    conf_counts_csv = output_dir / f"{prefix}_confusion_matrix_counts.csv"
    conf_percent_csv = output_dir / f"{prefix}_confusion_matrix_percent.csv"
    conf_png = output_dir / f"{prefix}_confusion_matrix.png"
    loss_rows = [[int(r["epoch"]), r["train_loss"], r["test_loss"], r["val_loss"]] for r in history]
    acc_rows = [[int(r["epoch"]), r["train_acc"], r["test_acc"], r["val_acc"]] for r in history]
    save_csv(loss_csv, ["epoch", "training", "testing", "validation"], loss_rows)
    save_csv(acc_csv, ["epoch", "training", "testing", "validation"], acc_rows)
    save_csv(conf_counts_csv, ["true\\pred"] + class_labels, [[class_labels[i]] + conf[i].tolist() for i in range(len(class_labels))])
    percent = conf / conf.sum(axis=1, keepdims=True).clip(min=1)
    save_csv(conf_percent_csv, ["true\\pred"] + class_labels, [[class_labels[i]] + [f"{v:.6f}" for v in percent[i]] for i in range(len(class_labels))])
    plot_confusion(percent, class_labels, history[-1]["test_acc"], conf_png)
    return {
        "task": task_name,
        "final_test_accuracy_percent": history[-1]["test_acc"],
        "final_test_loss": history[-1]["test_loss"],
        "loss_history": str(loss_csv),
        "accuracy_history": str(acc_csv),
        "confusion_counts": str(conf_counts_csv),
        "confusion_percent": str(conf_percent_csv),
        "confusion_chart": str(conf_png),
    }


def plot_paper_multitask_summary(
    output_dir: Path,
    task_results: dict[str, dict[str, object]],
    path: Path,
) -> None:
    apply_paper_style()
    fig = plt.figure(figsize=(12.0, 8.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0], hspace=0.36, wspace=0.35)

    heat_feat = task_results["FMNIST"]["test_feat"].numpy()
    heat_labels = task_results["FMNIST"]["test_labels"].numpy()
    order = np.argsort(heat_labels)
    if order.size > 520:
        order = order[np.linspace(0, order.size - 1, 520).astype(int)]
    ax_a = fig.add_subplot(gs[0, 0:2])
    im_a = ax_a.imshow(heat_feat[order].T, aspect="auto", cmap="coolwarm", interpolation="nearest")
    ax_a.set_title("a  Stacked reservoir features", loc="left", fontweight="bold")
    ax_a.set_xlabel("Garment samples sorted by class")
    ax_a.set_ylabel("Reservoir state")
    fig.colorbar(im_a, ax=ax_a, fraction=0.020, pad=0.012)

    ax_b = fig.add_subplot(gs[0, 2])
    names = ["MNIST", "EMNIST", "FMNIST"]
    accs = [task_results[n]["summary"]["final_test_accuracy_percent"] for n in names]
    colors = ["#4c78a8", "#f58518", "#54a24b"]
    ax_b.bar(names, accs, color=colors, edgecolor="black", linewidth=0.8)
    ax_b.set_ylim(0, 100)
    ax_b.set_ylabel("Accuracy (%)")
    ax_b.set_title("b  Independent tasks", loc="left", fontweight="bold")
    for x, acc in enumerate(accs):
        ax_b.text(x, acc + 1.2, f"{acc:.1f}%", ha="center", va="bottom", fontsize=8)

    for idx, name in enumerate(names):
        ax = fig.add_subplot(gs[1, idx])
        conf = task_results[name]["conf_percent"]
        labels = task_results[name]["labels"]
        im = ax.imshow(conf, cmap="YlGnBu", vmin=0, vmax=1)
        ax.set_title(f"{chr(ord('c') + idx)}  {name}", loc="left", fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(labels, fontsize=7)
        for i in range(len(labels)):
            for j in range(len(labels)):
                val = conf[i, j] * 100
                color = "white" if conf[i, j] > 0.55 else "black"
                ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=6, color=color)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def run_paper_multitask(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path("TorchData")
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"RC Device: {gpu_name}", flush=True)

    raw_responses = load_raw_response_table(args.device_response)
    responses = load_response_table(args.device_response)
    if len(responses) != 32:
        raise ValueError("Paper multitask mode requires a 32-state 5-bit device response CSV.")
    pulse_bits = 5
    plot_device_response(raw_responses, responses, output_dir / "device_response_map.png")
    save_csv(
        output_dir / "device_response_used.csv",
        ["input_state_5bit", "raw_response", "normalized_response"],
        [[format(i, "05b"), float(raw_responses[i]), float(responses[i])] for i in range(32)],
    )

    tasks = ["MNIST", "EMNIST", "FMNIST"]
    task_results: dict[str, dict[str, object]] = {}
    total_progress = args.epochs * len(tasks)
    for task_index, task_name in enumerate(tasks):
        tr_images, tr_labels, labels = load_paper_task_images(task_name, data_root, True, args.train_limit)
        te_images, te_labels, _ = load_paper_task_images(task_name, data_root, False, args.test_limit)
        all_train_feat, stats = reservoir_features(tr_images, responses, args.threshold, device, pulse_bits=pulse_bits)
        test_feat, _ = reservoir_features(te_images, responses, args.threshold, device, pulse_bits=pulse_bits, stats=stats)
        train_feat, train_labels, val_feat, val_labels = split_train_val(all_train_feat, tr_labels)
        train_loader, val_loader, test_loader = make_loaders(
            train_feat,
            train_labels,
            val_feat,
            val_labels,
            test_feat,
            te_labels,
            args.batch_size,
        )
        model, history, conf = train_readout(
            task_name,
            len(labels),
            train_loader,
            val_loader,
            test_loader,
            feature_dim=train_feat.shape[1],
            epochs=args.epochs,
            lr=args.lr,
            device=device,
            progress_start=task_index * args.epochs,
            progress_total=total_progress,
        )
        summary = save_task_outputs(output_dir, task_name, history, conf, labels)
        conf_percent = conf / conf.sum(axis=1, keepdims=True).clip(min=1)
        task_results[task_name] = {
            "model": model,
            "history": history,
            "conf": conf,
            "conf_percent": conf_percent,
            "labels": labels,
            "summary": summary,
            "test_feat": test_feat,
            "test_labels": te_labels,
        }
        if task_name == "FMNIST":
            plot_feature_heatmap(test_feat, te_labels, output_dir / "reservoir_feature_heatmap.png")
            plot_feature_pca(test_feat, te_labels, output_dir / "reservoir_feature_pca.png")
            save_feature_matrix_csv(output_dir / "reservoir_feature_matrix.csv", test_feat, te_labels, labels)
            save_feature_pca_csv(output_dir / "reservoir_feature_pca_coordinates.csv", test_feat, te_labels, labels)

    plot_paper_multitask_summary(output_dir, task_results, output_dir / "paper_multitask_summary.png")
    final_summary = {
        "task": "paper-multitask",
        "dataset": "MNIST digits + EMNIST L/M/S + Fashion-MNIST 5 garment classes",
        "gpu_name": gpu_name,
        "device": str(device),
        "pulse_bits": 5,
        "num_device_states": 32,
        "reservoir_feature_dim": int(task_results["MNIST"]["test_feat"].shape[1]),
        "preprocessing": "28x28 -> crop rows 4:26 cols 5:25 -> 22x20 pixels -> 5-bit pulse words",
        "readout": "three independent linear readouts: MNIST 10, EMNIST 3, FMNIST 5",
        "epochs_per_task": args.epochs,
        "train_limit_per_task": args.train_limit,
        "test_limit_per_task": args.test_limit,
        "device_response_file": str(Path(args.device_response).resolve()),
        "tasks": {name: task_results[name]["summary"] for name in tasks},
        "paper_style_outputs": [
            "device_response_map.png",
            "reservoir_feature_heatmap.png",
            "reservoir_feature_pca.png",
            "paper_multitask_summary.png",
        ],
        "paper_style_data": [
            "device_response_used.csv",
            "reservoir_feature_matrix.csv",
            "reservoir_feature_pca_coordinates.csv",
        ],
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(final_summary, handle, indent=2)
    mean_acc = float(np.mean([task_results[name]["summary"]["final_test_accuracy_percent"] for name in tasks]))
    print(f"RC Complete final_test_accuracy={mean_acc:.2f}", flush=True)


def main() -> None:
    args = parse_args()
    if args.task == "paper-multitask":
        run_paper_multitask(args)
        return
    if args.epochs <= 0:
        raise ValueError("epochs must be positive.")
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path("TorchData")
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"RC Device: {gpu_name}", flush=True)

    raw_responses = load_raw_response_table(args.device_response)
    responses = load_response_table(args.device_response)
    pulse_bits = int(np.log2(len(responses)))
    train_images, train_labels = load_images(args.dataset, data_root, True, args.train_limit)
    test_images, test_labels = load_images(args.dataset, data_root, False, args.test_limit)

    n_val = max(1, int(0.15 * train_images.shape[0]))
    val_images = train_images[-n_val:]
    val_labels = train_labels[-n_val:]
    tr_images = train_images[:-n_val]
    tr_labels = train_labels[:-n_val]

    train_feat, stats = reservoir_features(tr_images, responses, args.threshold, device, pulse_bits=pulse_bits)
    val_feat, _ = reservoir_features(val_images, responses, args.threshold, device, pulse_bits=pulse_bits, stats=stats)
    test_feat, _ = reservoir_features(test_images, responses, args.threshold, device, pulse_bits=pulse_bits, stats=stats)

    train_loader = DataLoader(TensorDataset(train_feat, tr_labels), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_feat, val_labels), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(test_feat, test_labels), batch_size=args.batch_size, shuffle=False)

    model = nn.Linear(train_feat.shape[1], 10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history: list[dict[str, float]] = []
    final_conf = np.zeros((10, 10), dtype=np.int64)
    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        train_loss, train_acc, _ = evaluate(model, train_loader, criterion, device)
        val_loss, val_acc, _ = evaluate(model, val_loader, criterion, device)
        test_loss, test_acc, final_conf = evaluate(model, test_loader, criterion, device)
        row = {
            "epoch": float(epoch),
            "train_loss": train_loss,
            "test_loss": test_loss,
            "val_loss": val_loss,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "val_acc": val_acc,
        }
        history.append(row)
        print(
            "RC Epoch "
            f"{epoch}/{args.epochs} train_loss={train_loss:.6f} "
            f"test_loss={test_loss:.6f} val_loss={val_loss:.6f} "
            f"train_acc={train_acc:.2f} test_acc={test_acc:.2f} val_acc={val_acc:.2f}",
            flush=True,
        )

    loss_rows = [
        [int(row["epoch"]), row["train_loss"], row["test_loss"], row["val_loss"]]
        for row in history
    ]
    acc_rows = [
        [int(row["epoch"]), row["train_acc"], row["test_acc"], row["val_acc"]]
        for row in history
    ]
    save_csv(output_dir / "loss_history.csv", ["epoch", "training", "testing", "validation"], loss_rows)
    save_csv(output_dir / "accuracy_history.csv", ["epoch", "training", "testing", "validation"], acc_rows)
    save_csv(output_dir / "confusion_matrix_counts.csv", ["true\\pred"] + CLASS_LABELS[args.dataset], [[CLASS_LABELS[args.dataset][i]] + final_conf[i].tolist() for i in range(10)])
    row_sums = final_conf.sum(axis=1, keepdims=True).clip(min=1)
    conf_percent = final_conf / row_sums
    save_csv(output_dir / "confusion_matrix_percent.csv", ["true\\pred"] + CLASS_LABELS[args.dataset], [[CLASS_LABELS[args.dataset][i]] + [f"{v:.6f}" for v in conf_percent[i]] for i in range(10)])
    save_csv(
        output_dir / "device_response_used.csv",
        [f"input_state_{pulse_bits}bit", "raw_response", "normalized_response"],
        [[format(i, f"0{pulse_bits}b"), float(raw_responses[i]), float(responses[i])] for i in range(len(responses))],
    )
    preview_count = min(200, int(test_feat.shape[0]))
    preview_rows = []
    for idx in range(preview_count):
        preview_rows.append([idx, int(test_labels[idx].item())] + [float(v) for v in test_feat[idx].tolist()])
    save_csv(
        output_dir / "reservoir_feature_preview.csv",
        ["sample_index", "label"] + [f"state_{i:03d}" for i in range(test_feat.shape[1])],
        preview_rows,
    )

    plot_loss(history, output_dir / "loss_curve.png")
    plot_accuracy(history, output_dir / "accuracy_curve.png")
    plot_confusion(conf_percent, CLASS_LABELS[args.dataset], history[-1]["test_acc"], output_dir / "confusion_matrix.png")
    plot_device_response(raw_responses, responses, output_dir / "device_response_map.png")
    plot_feature_heatmap(test_feat, test_labels, output_dir / "reservoir_feature_heatmap.png")
    plot_feature_pca(test_feat, test_labels, output_dir / "reservoir_feature_pca.png")
    save_feature_matrix_csv(output_dir / "reservoir_feature_matrix.csv", test_feat, test_labels, CLASS_LABELS[args.dataset])
    save_feature_pca_csv(output_dir / "reservoir_feature_pca_coordinates.csv", test_feat, test_labels, CLASS_LABELS[args.dataset])
    plot_paper_summary(
        history,
        test_feat,
        test_labels,
        conf_percent,
        CLASS_LABELS[args.dataset],
        history[-1]["test_acc"],
        output_dir / "paper_style_summary.png",
    )

    summary = {
        "dataset": args.dataset,
        "gpu_name": gpu_name,
        "device": str(device),
        "reservoir_feature_dim": int(train_feat.shape[1]),
        "pulse_bits": pulse_bits,
        "crop": "28x28 -> rows 4:26, cols 5:25 -> 22x20 = 440 pixels",
        "readout": "linear 110-to-10",
        "epochs": args.epochs,
        "train_samples": int(tr_labels.numel()),
        "validation_samples": int(val_labels.numel()),
        "test_samples": int(test_labels.numel()),
        "final_test_accuracy_percent": history[-1]["test_acc"],
        "final_test_loss": history[-1]["test_loss"],
        "device_response_file": str(Path(args.device_response).resolve()),
        "paper_style_outputs": [
            "device_response_map.png",
            "reservoir_feature_heatmap.png",
            "reservoir_feature_pca.png",
            "paper_style_summary.png",
        ],
        "paper_style_data": [
            "device_response_used.csv",
            "reservoir_feature_matrix.csv",
            "reservoir_feature_pca_coordinates.csv",
        ],
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"RC Complete final_test_accuracy={history[-1]['test_acc']:.2f}", flush=True)


if __name__ == "__main__":
    main()
