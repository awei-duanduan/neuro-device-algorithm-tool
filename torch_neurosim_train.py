import argparse
import csv
import json
import math
import re
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


ROOT = Path(__file__).resolve().parent


class MLP(nn.Module):
    def __init__(self, input_neurons: int, hidden_neurons: int, output_neurons: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_neurons, hidden_neurons),
            nn.Sigmoid(),
            nn.Linear(hidden_neurons, output_neurons),
        )

    def forward(self, x):
        return self.net(x)


class DeviceMLP(nn.Module):
    def __init__(self, input_neurons: int, hidden_neurons: int, output_neurons: int = 10):
        super().__init__()
        self.weight1 = nn.Parameter(torch.randint(0, 4, (hidden_neurons, input_neurons), dtype=torch.float32) / 3.0)
        self.weight2 = nn.Parameter(torch.randint(0, 4, (output_neurons, hidden_neurons), dtype=torch.float32) / 3.0)

    def forward(self, x):
        x = torch.flatten(x, 1)
        hidden = torch.sigmoid(torch.matmul(x, (2.0 * self.weight1 - 1.0).t()))
        return torch.sigmoid(torch.matmul(hidden, (2.0 * self.weight2 - 1.0).t()))


def read_fit_params(path: Path):
    params = {
        "sigmaCtoCNorm": 0.0,
        "maxNumLevelLTP": 0,
        "maxNumLevelLTD": 0,
        "maxConductance": 1.0,
        "minConductance": 0.0,
        "paramALTP": 0.0,
        "paramALTD": 0.0,
    }
    if not path.exists():
        return params
    text = path.read_text(encoding="utf-8", errors="ignore")
    patterns = {
        "sigmaCtoCNorm": r"sigmaCtoC normalized\s*=\s*([0-9.eE+-]+)",
        "maxNumLevelLTP": r"maxNumLevelLTP\s*=\s*(\d+)",
        "maxNumLevelLTD": r"maxNumLevelLTD\s*=\s*(\d+)",
        "maxConductance": r"maxConductance\s*=\s*([0-9.eE+-]+)",
        "minConductance": r"minConductance\s*=\s*([0-9.eE+-]+)",
        "paramALTP": r"paramALTP C\+\+\s*=\s*([0-9.eE+-]+)",
        "paramALTD": r"paramALTD C\+\+\s*=\s*([0-9.eE+-]+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            value = float(match.group(1))
            params[key] = int(value) if key.startswith("maxNum") else value
    return params


def torch_truncate(x, num_level: int, threshold: float = 0.5):
    if num_level <= 0:
        return x
    sign = torch.where(x < 0, -1.0, 1.0)
    val = x * num_level * sign
    floored = torch.floor(val)
    rounded = torch.where(val - floored >= threshold, floored + 1.0, floored)
    return rounded * sign / num_level


@torch.no_grad()
def realdevice_write_normalized(weight_norm, delta_weight_norm, fit_params, generator=None):
    max_g = float(fit_params.get("maxConductance", 1.0))
    min_g = float(fit_params.get("minConductance", 0.0))
    g_range = max(max_g - min_g, 1e-30)
    ltp_levels = max(int(fit_params.get("maxNumLevelLTP", 0)), 0)
    ltd_levels = max(int(fit_params.get("maxNumLevelLTD", 0)), 0)
    a_ltp = float(fit_params.get("paramALTP", 0.0))
    a_ltd = float(fit_params.get("paramALTD", 0.0))
    sigma = float(fit_params.get("sigmaCtoCNorm", 0.0)) * g_range

    conductance = torch.clamp(weight_norm, 0.0, 1.0) * g_range + min_g
    conductance_new = conductance.clone()

    pos = delta_weight_norm > 0
    neg = delta_weight_norm < 0
    num_pulse = torch.zeros_like(delta_weight_norm)

    if pos.any() and ltp_levels > 0:
        delta = torch_truncate(delta_weight_norm[pos], ltp_levels)
        pulses = delta * ltp_levels
        num_pulse[pos] = pulses
        if abs(a_ltp) > 1e-30:
            b_ltp = g_range / (1.0 - math.exp(-ltp_levels / a_ltp))
            inside = torch.clamp(1.0 - (conductance[pos] - min_g) / b_ltp, min=1e-12)
            x_pulse = -a_ltp * torch.log(inside)
            conductance_new[pos] = b_ltp * (1.0 - torch.exp(-(x_pulse + pulses) / a_ltp)) + min_g
        else:
            x_pulse = (conductance[pos] - min_g) / g_range * ltp_levels
            conductance_new[pos] = (x_pulse + pulses) / ltp_levels * g_range + min_g

    if neg.any() and ltd_levels > 0:
        delta = torch_truncate(delta_weight_norm[neg], ltd_levels)
        pulses = delta * ltd_levels
        num_pulse[neg] = pulses
        if abs(a_ltd) > 1e-30:
            b_ltd = g_range / (1.0 - math.exp(-ltd_levels / a_ltd))
            inside = torch.clamp(1.0 - (conductance[neg] - min_g) / b_ltd, min=1e-12)
            x_pulse = -a_ltd * torch.log(inside)
            conductance_new[neg] = b_ltd * (1.0 - torch.exp(-(x_pulse + pulses) / a_ltd)) + min_g
        else:
            x_pulse = (conductance[neg] - min_g) / g_range * ltd_levels
            conductance_new[neg] = (x_pulse + pulses) / ltd_levels * g_range + min_g

    if sigma > 0:
        changed = num_pulse != 0
        if changed.any():
            noise = torch.randn(
                conductance_new.shape,
                device=conductance_new.device,
                dtype=conductance_new.dtype,
                generator=generator,
            )
            conductance_new[changed] += noise[changed] * sigma * torch.sqrt(torch.abs(num_pulse[changed]))

    conductance_new = torch.clamp(conductance_new, min_g, max_g)
    weight_norm.copy_((conductance_new - min_g) / g_range)
    return int(torch.count_nonzero(num_pulse).item()), int(num_pulse.numel())


@torch.no_grad()
def realdevice_optimizer_step(model, lr: float, fit_params, generator=None):
    changed = 0
    total = 0
    for param in model.parameters():
        if param.grad is None:
            continue
        delta = -lr * param.grad
        changed_now, total_now = realdevice_write_normalized(param.data, delta, fit_params, generator=generator)
        changed += changed_now
        total += total_now
    return changed, total


def build_loaders(dataset_name: str, batch_size: int, seed: int):
    transform = transforms.Compose([transforms.ToTensor()])
    if dataset_name == "mnist":
        dataset_cls = datasets.MNIST
    elif dataset_name == "fashion":
        dataset_cls = datasets.FashionMNIST
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    train_full = dataset_cls(ROOT / "TorchData", train=True, download=True, transform=transform)
    test_set = dataset_cls(ROOT / "TorchData", train=False, download=True, transform=transform)

    train_size = int(len(train_full) * 0.9)
    val_size = len(train_full) - train_size
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(train_full, [train_size, val_size], generator=generator)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, generator=generator, pin_memory=pin)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, pin_memory=pin)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, pin_memory=pin)
    return train_loader, val_loader, test_loader


@torch.no_grad()
def apply_device_mapping(model, fit_params, enabled: bool):
    if not enabled:
        return
    sigma = float(fit_params.get("sigmaCtoCNorm", 0.0))
    ltp_levels = max(int(fit_params.get("maxNumLevelLTP", 0)), 0)
    ltd_levels = max(int(fit_params.get("maxNumLevelLTD", 0)), 0)
    levels = max(ltp_levels, ltd_levels, 0)
    for name, param in model.named_parameters():
        if "weight" not in name:
            continue
        if levels > 1:
            clipped = torch.clamp(param.data, -1.0, 1.0)
            normalized = (clipped + 1.0) / 2.0
            quantized = torch.round(normalized * levels) / levels
            param.data.copy_(quantized * 2.0 - 1.0)
        if sigma > 0:
            param.data.add_(torch.randn_like(param.data) * sigma * 0.01)


def train_one_epoch(model, loader, optimizer, criterion, device, fit_params, mapping_mode, lr, generator=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    pulse_changed = 0
    pulse_total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        else:
            model.zero_grad(set_to_none=True)
        logits = model(images)
        if mapping_mode == "train-realdevice":
            target = torch.zeros(labels.numel(), 10, device=device, dtype=logits.dtype)
            target.scatter_(1, labels.view(-1, 1), 1.0)
            loss = torch.mean((target - logits) ** 2)
        else:
            loss = criterion(logits, labels)
        loss.backward()
        if mapping_mode == "train-realdevice":
            changed_now, total_now = realdevice_optimizer_step(model, lr * labels.numel(), fit_params, generator=generator)
            pulse_changed += changed_now
            pulse_total += total_now
        else:
            optimizer.step()
            apply_device_mapping(model, fit_params, mapping_mode == "train")
        running_loss += loss.item() * labels.numel()
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.numel()
    pulse_fraction = pulse_changed / max(pulse_total, 1) * 100.0
    return running_loss / total, correct / total * 100.0, pulse_fraction


@torch.no_grad()
def evaluate(model, loader, criterion, device, need_confusion=False):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    confusion = torch.zeros((10, 10), dtype=torch.int64)
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        preds = logits.argmax(dim=1)
        running_loss += loss.item() * labels.numel()
        correct += (preds == labels).sum().item()
        total += labels.numel()
        if need_confusion:
            for truth, pred in zip(labels.cpu(), preds.cpu()):
                confusion[int(truth), int(pred)] += 1
    accuracy = correct / total * 100.0
    if need_confusion:
        return running_loss / total, accuracy, confusion.numpy()
    return running_loss / total, accuracy


def write_history(path: Path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "training_cross_entropy",
            "testing_cross_entropy",
            "validation_cross_entropy",
            "training_accuracy_percent",
            "testing_accuracy_percent",
            "validation_accuracy_percent",
        ])
        writer.writerows(rows)


def plot_loss(path: Path, rows):
    arr = np.array(rows, dtype=float)
    plt.figure(figsize=(6, 4.5), dpi=180)
    plt.plot(arr[:, 0], arr[:, 1], "k-", label="training", linewidth=1.8)
    plt.plot(arr[:, 0], arr[:, 2], "r-", label="testing", linewidth=1.8)
    plt.plot(arr[:, 0], arr[:, 3], "b-", label="validation", linewidth=1.8)
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_accuracy(path: Path, rows):
    arr = np.array(rows, dtype=float)
    plt.figure(figsize=(6, 4.5), dpi=180)
    plt.plot(arr[:, 0], arr[:, 4], "k-", label="training", linewidth=1.8)
    plt.plot(arr[:, 0], arr[:, 5], "r-", label="testing", linewidth=1.8)
    plt.plot(arr[:, 0], arr[:, 6], "b-", label="validation", linewidth=1.8)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy %")
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def write_confusion_csv(count_path: Path, percent_path: Path, confusion):
    row_sum = confusion.sum(axis=1, keepdims=True)
    percent = np.divide(confusion, np.maximum(row_sum, 1)) * 100
    for path, matrix, fmt in [(count_path, confusion, "{:d}"), (percent_path, percent, "{:.6f}")]:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["true_label"] + [f"predicted_{i}" for i in range(10)])
            for i in range(10):
                writer.writerow([i] + [fmt.format(int(v) if fmt == "{:d}" else float(v)) for v in matrix[i]])
    return percent


def plot_confusion(path: Path, percent, accuracy):
    plt.figure(figsize=(6, 5.2), dpi=180)
    plt.imshow(percent, cmap="Blues", vmin=0, vmax=100)
    plt.title(f"Accuracy: {accuracy:.2f}%")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(10), range(10))
    plt.yticks(range(10), range(10))
    plt.colorbar(fraction=0.046, pad=0.04)
    for r in range(10):
        for c in range(10):
            color = "white" if percent[r, c] >= 50 else "black"
            plt.text(c, r, f"{percent[r, c]:.1f}%", ha="center", va="center", color=color, fontsize=7)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    parser = argparse.ArgumentParser(description="CUDA/PyTorch MLP training for NeuroSim-style experiments")
    parser.add_argument("--dataset", choices=["mnist", "fashion"], default="mnist")
    parser.add_argument("--hidden", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device-update-gain", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mapping-mode", choices=["none", "train", "post", "train-realdevice"], default="none")
    parser.add_argument("--device-mapping", action="store_true", help="Shortcut for --mapping-mode train")
    parser.add_argument("--output-dir", default="torch_results")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    fit_params = read_fit_params(ROOT / "fit_device_result.txt")
    mapping_mode = "train" if args.device_mapping else args.mapping_mode
    if mapping_mode == "train-realdevice" and args.optimizer != "sgd":
        args.optimizer = "sgd"

    train_loader, val_loader, test_loader = build_loaders(args.dataset, args.batch_size, args.seed)
    if mapping_mode == "train-realdevice":
        model = DeviceMLP(input_neurons=28 * 28, hidden_neurons=args.hidden).to(device)
    else:
        model = MLP(input_neurons=28 * 28, hidden_neurons=args.hidden).to(device)
    criterion = nn.CrossEntropyLoss()
    if mapping_mode == "train-realdevice":
        optimizer = None
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.0)
    torch_generator = None
    if device.type == "cuda":
        torch_generator = torch.Generator(device=device).manual_seed(args.seed)

    rows = []
    start = time.time()
    print(f"Device: {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"Network: 784-{args.hidden}-10", flush=True)
    print(f"Dataset: {args.dataset}", flush=True)
    print(f"Optimizer: {args.optimizer}, lr={args.lr}", flush=True)
    print(f"Device mapping mode: {mapping_mode}", flush=True)

    for epoch in range(1, args.epochs + 1):
        update_loss, update_acc, pulse_fraction = train_one_epoch(
            model, train_loader, optimizer, criterion, device, fit_params, mapping_mode, args.lr * args.device_update_gain, generator=torch_generator
        )
        train_loss, train_acc = evaluate(model, train_loader, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        rows.append([epoch, train_loss, test_loss, val_loss, train_acc, test_acc, val_acc])
        pulse_text = f" pulse_update={pulse_fraction:.4f}%" if mapping_mode == "train-realdevice" else ""
        print(
            f"Epoch {epoch}/{args.epochs} "
            f"train_loss={train_loss:.6f} test_loss={test_loss:.6f} val_loss={val_loss:.6f} "
            f"train_acc={train_acc:.2f}% test_acc={test_acc:.2f}% val_acc={val_acc:.2f}%"
            f"{pulse_text}",
            flush=True,
        )

    pre_mapping_loss, pre_mapping_acc = evaluate(model, test_loader, criterion, device)
    if mapping_mode == "post":
        apply_device_mapping(model, fit_params, True)
    final_loss, final_acc, confusion = evaluate(model, test_loader, criterion, device, need_confusion=True)
    history_csv = output_dir / "loss_history.csv"
    loss_png = output_dir / "loss_curve.png"
    accuracy_csv = output_dir / "accuracy_history.csv"
    accuracy_png = output_dir / "accuracy_curve.png"
    conf_counts_csv = output_dir / "confusion_matrix_counts.csv"
    conf_percent_csv = output_dir / "confusion_matrix_percent.csv"
    conf_png = output_dir / "confusion_matrix.png"
    summary_json = output_dir / "summary.json"

    write_history(history_csv, rows)
    write_history(accuracy_csv, rows)
    plot_loss(loss_png, rows)
    plot_accuracy(accuracy_png, rows)
    percent = write_confusion_csv(conf_counts_csv, conf_percent_csv, confusion)
    plot_confusion(conf_png, percent, final_acc)
    summary = {
        "dataset": args.dataset,
        "network": f"784-{args.hidden}-10",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "optimizer": args.optimizer,
        "lr": args.lr,
        "device_update_gain": args.device_update_gain,
        "seed": args.seed,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "mapping_mode": mapping_mode,
        "pre_mapping_test_loss": pre_mapping_loss,
        "pre_mapping_test_accuracy_percent": pre_mapping_acc,
        "final_test_loss": final_loss,
        "final_test_accuracy_percent": final_acc,
        "elapsed_seconds": time.time() - start,
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Final test accuracy: {final_acc:.2f}%", flush=True)
    print(f"Saved: {history_csv}", flush=True)
    print(f"Saved: {loss_png}", flush=True)
    print(f"Saved: {accuracy_csv}", flush=True)
    print(f"Saved: {accuracy_png}", flush=True)
    print(f"Saved: {conf_counts_csv}", flush=True)
    print(f"Saved: {conf_percent_csv}", flush=True)
    print(f"Saved: {conf_png}", flush=True)
    print(f"Saved: {summary_json}", flush=True)


if __name__ == "__main__":
    main()
