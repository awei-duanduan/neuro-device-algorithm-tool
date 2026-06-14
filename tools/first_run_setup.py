#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


PROJECT = Path(__file__).resolve().parents[1]


def run(cmd: list[str], *, timeout: int | None = None, check: bool = True) -> int:
    print("\n> " + " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(PROJECT), check=check, timeout=timeout)
    return proc.returncode


def prepare_neurosim_datasets(python_exe: str) -> None:
    jobs = [
        ["--dataset", "mnist", "--image-size", "28", "--root-copy"],
        ["--dataset", "fashion", "--image-size", "28"],
        ["--dataset", "mnist", "--image-size", "20"],
        ["--dataset", "fashion", "--image-size", "20"],
    ]
    for args in jobs:
        run([python_exe, "prepare_neurosim_dataset.py", *args])


def prepare_torch_cache(python_exe: str, include_emnist: bool) -> None:
    code = (
        "from torchvision import datasets\n"
        "root='TorchData'\n"
        "datasets.MNIST(root=root, train=True, download=True)\n"
        "datasets.MNIST(root=root, train=False, download=True)\n"
        "datasets.FashionMNIST(root=root, train=True, download=True)\n"
        "datasets.FashionMNIST(root=root, train=False, download=True)\n"
        + (
            "datasets.EMNIST(root=root, split='letters', train=True, download=True)\n"
            "datasets.EMNIST(root=root, split='letters', train=False, download=True)\n"
            if include_emnist
            else ""
        )
        + "print('Torchvision dataset cache is ready.')\n"
    )
    run([python_exe, "-c", code])


def main() -> int:
    parser = argparse.ArgumentParser(description="First-run setup for the neural-device algorithm tool.")
    parser.add_argument("--python", default=sys.executable, help="Python executable to use.")
    parser.add_argument("--skip-neurosim-data", action="store_true", help="Do not download/generate NeuroSim train/test text files.")
    parser.add_argument("--torch-cache", action="store_true", help="Pre-download MNIST and Fashion-MNIST through torchvision.")
    parser.add_argument("--emnist", action="store_true", help="Also pre-download EMNIST. This can be large and slow.")
    parser.add_argument("--check-only", action="store_true", help="Only run the environment check.")
    args = parser.parse_args()

    first_check = run([args.python, "tools/check_env.py"], check=False)
    if first_check != 0:
        print("\nEnvironment check reported missing required components.")
        print("Dataset generation will still continue because it only needs standard Python.")
        print("Install MATLAB/MSYS2/PyTorch requirements before running the full GUI workflow.")
    if args.check_only:
        return 0

    if not args.skip_neurosim_data:
        prepare_neurosim_datasets(args.python)

    if args.torch_cache or args.emnist:
        prepare_torch_cache(args.python, include_emnist=args.emnist)

    run([args.python, "tools/check_env.py"], check=False)
    print("\nFirst-run setup finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
