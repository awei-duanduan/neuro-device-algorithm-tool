#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


MATLAB_CANDIDATES = [
    Path(os.environ["MATLAB_EXE"]) if os.environ.get("MATLAB_EXE") else None,
    Path(r"C:\Program Files\MATLAB\R2022b\bin\matlab.exe"),
    Path(r"C:\Program Files\MATLAB\R2023a\bin\matlab.exe"),
    Path(r"C:\Program Files\MATLAB\R2023b\bin\matlab.exe"),
    Path(r"C:\Program Files\MATLAB\R2024a\bin\matlab.exe"),
    Path(r"C:\Program Files\MATLAB\R2024b\bin\matlab.exe"),
    Path(r"C:\Program Files\MATLAB\R2025a\bin\matlab.exe"),
]


def run(cmd: list[str], cwd: Path | None = None, timeout: int = 30) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
        return proc.returncode, proc.stdout.strip()
    except Exception as exc:
        return 999, str(exc)


def status(ok: bool, name: str, detail: str) -> dict[str, object]:
    return {"ok": ok, "name": name, "detail": detail}


def looks_like_project(path: Path) -> bool:
    return (path / "fit_device_gui.m").exists() and (path / "rc_reservoir_train.py").exists()


def discover_project(arg_project: str | None) -> Path:
    if arg_project:
        return Path(arg_project).expanduser().resolve()
    env_project = os.environ.get("NEURO_DEVICE_PROJECT")
    if env_project:
        return Path(env_project).expanduser().resolve()
    cwd = Path.cwd().resolve()
    if looks_like_project(cwd):
        return cwd
    for parent in cwd.parents:
        if looks_like_project(parent):
            return parent
    return cwd


def find_matlab() -> Path | None:
    found = shutil.which("matlab")
    if found:
        return Path(found)
    for path in MATLAB_CANDIDATES:
        if path and path.exists():
            return path
    return None


def find_executable(names: list[str], extra_dirs: list[Path]) -> Path | None:
    for base in extra_dirs:
        for name in names:
            p = base / name
            if p.exists():
                return p
    for name in names:
        found = shutil.which(name)
        if found:
            return Path(found)
    return None


def check_project(project: Path) -> list[dict[str, object]]:
    files = [
        "fit_device_gui.m",
        "fit_device_to_realsim.m",
        "Cell.cpp",
        "Param.cpp",
        "torch_neurosim_train.py",
        "rc_reservoir_train.py",
        "reservoir_device_response_pndi_05s.csv",
        "reservoir_device_response_pei2025.csv",
    ]
    results = [status(project.exists(), "project directory", str(project))]
    for file in files:
        p = project / file
        results.append(status(p.exists(), file, str(p)))
    return results


def check_matlab(project: Path, deep: bool) -> list[dict[str, object]]:
    results = []
    matlab = find_matlab()
    results.append(status(matlab is not None and matlab.exists(), "MATLAB", str(matlab) if matlab else "not found"))
    if deep and matlab and matlab.exists() and project.exists():
        escaped = str(project).replace("'", "''")
        cmd = [str(matlab), "-batch", f"cd('{escaped}'); fit_device_gui; drawnow; pause(0.5); close(gcf);"]
        code, out = run(cmd, timeout=120)
        results.append(status(code == 0, "MATLAB GUI open/close", out[-1000:] if out else "ok"))
    return results


def check_msys2() -> list[dict[str, object]]:
    msys_root = Path(os.environ.get("MSYS2_ROOT", r"C:\msys64"))
    candidates = [
        msys_root / "ucrt64" / "bin",
        msys_root / "mingw64" / "bin",
        msys_root / "usr" / "bin",
    ]
    gpp = find_executable(["g++.exe", "g++"], candidates)
    make = find_executable(["make.exe", "make"], candidates)
    return [
        status(gpp is not None and gpp.exists(), "MSYS2/MinGW g++", str(gpp) if gpp else "not found"),
        status(make is not None and make.exists(), "MSYS2/MinGW make", str(make) if make else "not found"),
    ]


def find_project_python(project: Path) -> Path | None:
    candidates = [
        project / ".venv_torch" / "Scripts" / "python.exe",
        project / ".venv_torch" / "bin" / "python",
        project / "venv" / "Scripts" / "python.exe",
        project / "venv" / "bin" / "python",
    ]
    for p in candidates:
        if p.exists():
            return p
    found = shutil.which("python")
    return Path(found) if found else None


def check_python(project: Path) -> list[dict[str, object]]:
    py = find_project_python(project)
    results = [status(py is not None and py.exists(), "Python for PyTorch", str(py) if py else "not found")]
    if not py:
        return results
    code = (
        "import json, importlib.util\n"
        "mods=['torch','torchvision','numpy','matplotlib','pandas','openpyxl','PIL','fitz']\n"
        "info={m:(importlib.util.find_spec(m) is not None) for m in mods}\n"
        "try:\n"
        "    import torch\n"
        "    info['torch_version']=torch.__version__\n"
        "    info['cuda_available']=torch.cuda.is_available()\n"
        "    info['cuda_device']=torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''\n"
        "except Exception as exc:\n"
        "    info['torch_error']=str(exc)\n"
        "print(json.dumps(info, ensure_ascii=False))\n"
    )
    rc, out = run([str(py), "-c", code], cwd=project if project.exists() else None, timeout=60)
    if rc != 0:
        results.append(status(False, "Python imports", out))
        return results
    try:
        info = json.loads(out)
    except Exception:
        results.append(status(False, "Python imports", out))
        return results
    for mod in ["torch", "torchvision", "numpy", "matplotlib", "pandas", "openpyxl", "PIL", "fitz"]:
        results.append(status(bool(info.get(mod)), f"Python module {mod}", "installed" if info.get(mod) else "missing"))
    results.append(status("torch_version" in info, "PyTorch version", str(info.get("torch_version", info.get("torch_error", "unavailable")))))
    results.append(status(bool(info.get("cuda_available")), "PyTorch CUDA", str(info.get("cuda_device") or "unavailable")))
    return results


def check_nvidia() -> list[dict[str, object]]:
    smi = shutil.which("nvidia-smi")
    results = [status(smi is not None, "nvidia-smi", smi or "not found")]
    if smi:
        rc, out = run([smi, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], timeout=20)
        results.append(status(rc == 0, "NVIDIA GPU/driver", out))
    return results


def check_dataset_cache(project: Path) -> list[dict[str, object]]:
    root = project / "TorchData"
    paths = [root / "MNIST", root / "FashionMNIST", root / "EMNIST"]
    results = [status(p.exists(), f"dataset cache {p.name}", str(p)) for p in paths]

    neurosim_files = [
        project / "patch60000_train.txt",
        project / "patch10000_test.txt",
        project / "label60000_train.txt",
        project / "label10000_test.txt",
        project / "MNIST_data.zip",
    ]
    for p in neurosim_files:
        results.append(status(p.exists(), f"generated NeuroSim data {p.name}", str(p)))

    dataset_root = project / "Datasets"
    results.append(status(dataset_root.exists(), "generated dataset directory", str(dataset_root)))
    return results


def print_table(results: list[dict[str, object]]) -> None:
    width = max(len(str(r["name"])) for r in results) if results else 10
    for r in results:
        mark = "OK" if r["ok"] else "MISS"
        print(f"[{mark:<4}] {str(r['name']).ljust(width)}  {r['detail']}")
    missing = [r for r in results if not r["ok"]]
    print()
    print(f"Summary: {len(results) - len(missing)}/{len(results)} checks passed.")
    if missing:
        print("Missing or mismatched items:")
        for r in missing:
            print(f"- {r['name']}: {r['detail']}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=None, help="Path to the project root.")
    parser.add_argument("--deep", action="store_true", help="Also open and close the MATLAB GUI.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project = discover_project(args.project)
    results: list[dict[str, object]] = []
    results += check_project(project)
    results += check_matlab(project, args.deep)
    results += check_msys2()
    results += check_python(project)
    results += check_nvidia()
    results += check_dataset_cache(project)

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        print_table(results)
    optional_prefixes = ("dataset cache", "generated NeuroSim data", "generated dataset directory")
    required_ok = all(r["ok"] for r in results if not str(r["name"]).startswith(optional_prefixes))
    return 0 if required_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
