---
name: neuro-device-algorithm-tool
description: Maintain, run, debug, and extend a local nonvolatile neural-device algorithm recognition tool based on MATLAB GUI, MLP_NeuroSim fitting, C++/MSYS2 NeuroSim execution, PyTorch CUDA recognition, and reservoir-computing workflows. Use when the user asks to check the required environment, run the GUI, modify the GUI, fit LTP/LTD or device pulse-response data, run MNIST/Fashion/EMNIST neural recognition, inspect generated plots/data, or package the device-to-algorithm workflow.
---

# Neuro Device Algorithm Tool

## First Step

Before running or modifying the tool, run the environment check.

If the current directory is the project root, run first-run setup before normal work:

```powershell
python tools/first_run_setup.py
```

This automatically checks the environment and downloads/generates MNIST/Fashion-MNIST NeuroSim train/test files from public sources. For a lighter check without downloads:

```powershell
python tools/first_run_setup.py --check-only
```

Use `python tools/first_run_setup.py --torch-cache` to also pre-download torchvision MNIST/Fashion-MNIST cache. Use `--emnist` only when the user explicitly wants the larger EMNIST cache downloaded at setup time.

Read `references/environment-and-workflow.md` when you need paths, modes, datasets, output files, literature settings, or troubleshooting notes.

## Operating Workflow

1. Check the environment and report mismatches clearly: MATLAB, MSYS2/MinGW, Python venv, PyTorch, CUDA, NVIDIA driver, project files, device-response CSVs, and dataset cache.
2. If train/test files are missing after clone, run `prepare_neurosim_dataset.py` from the project root to download public datasets and generate local NeuroSim text files.
3. For MATLAB GUI work, edit `fit_device_gui.m` and verify with MATLAB batch startup when available.
4. For MLP/NeuroSim work, keep the LTP/LTD fit path separate from reservoir-computing paths. The fitting workflow updates `Cell.cpp` RealDevice parameters.
5. For PyTorch CUDA neural-network work, use `.venv_torch\Scripts\python.exe` when present and confirm CUDA with `torch.cuda.is_available()` plus the GPU name.
6. For reservoir computing, use `rc_reservoir_train.py`. Default paper mode is p-NDI 5-bit/32-state multitask: MNIST digits, EMNIST L/M/S, and Fashion-MNIST five classes.
7. Always tell the user where result images and corresponding CSV/JSON data were written.

## Common Commands

Open the MATLAB GUI:

```powershell
matlab -batch "cd('<path-to-project>'); fit_device_gui"
```

Smoke-test reservoir computing on CUDA:

```powershell
cd "<path-to-project>"
.\.venv_torch\Scripts\python.exe -u rc_reservoir_train.py --task paper-multitask --device-response reservoir_device_response_pndi_05s.csv --epochs 1 --train-limit 150 --test-limit 45 --batch-size 64 --output-dir rc_results_skill_smoke
```

Smoke-test single-task mode:

```powershell
cd "<path-to-project>"
.\.venv_torch\Scripts\python.exe -u rc_reservoir_train.py --task single --dataset mnist --device-response reservoir_device_response_pei2025.csv --epochs 1 --train-limit 300 --test-limit 100 --batch-size 128 --output-dir rc_results_single_smoke
```

## Editing Rules

- Use `apply_patch` for code edits.
- Do not overwrite user data files.
- Keep MATLAB GUI controls visible and avoid overlapping text fields.
- After changing Python training code, run a small CUDA smoke test.
- After changing MATLAB GUI code, run a MATLAB batch open/close check.
