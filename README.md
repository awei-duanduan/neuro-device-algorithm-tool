# 非易失性神经器件算法识别工具

中文 | [English](#nonvolatile-neural-device-algorithm-tool)

这是一个面向神经形态器件研究的完整工具仓库。它基于原始 `MLP+NeuroSim V2.0`，并扩展了 MATLAB 图形界面、器件 LTP/LTD 曲线拟合、C++ NeuroSim 运行、PyTorch CUDA 后映射识别，以及水库计算流程。

这个工具的核心用途是把器件实验曲线连接到算法功能：

1. 导入 LTP/LTD 或脉冲响应曲线。
2. 拟合器件非线性参数。
3. 自动写入 `Cell.cpp` 中的 `RealDevice`。
4. 运行 MNIST/Fashion-MNIST 分类或水库计算任务。
5. 输出准确率、损失曲线、混淆矩阵、特征图和对应 CSV/JSON 数据。

## 适合谁使用

- 做忆阻器、突触晶体管、光电突触、离子器件等非易失性神经器件的研究者。
- 希望把器件曲线映射到神经网络识别性能的同学。
- 希望复现实验器件到算法功能流程的新手。

## 仓库包含什么

```text
.
|-- fit_device_gui.m                 # MATLAB GUI 主入口
|-- fit_device_to_realsim.m           # LTP/LTD 拟合与 RealDevice 参数映射
|-- nonlinear_fit.m                   # 原始 NeuroSim 非线性拟合脚本
|-- Cell.cpp                          # RealDevice 参数写入位置
|-- Param.cpp / Train.cpp / Test.cpp  # MLP+NeuroSim C++ 源码
|-- torch_neurosim_train.py           # PyTorch CUDA 后映射识别
|-- rc_reservoir_train.py             # 水库计算脚本
|-- prepare_neurosim_dataset.py       # 下载公开数据集并生成 NeuroSim 文本数据
|-- prepare_fashion_mnist.py          # Fashion-MNIST 数据准备脚本
|-- volatile_LTP_example.csv          # 示例 LTP 曲线
|-- volatile_LTD_example.csv          # 示例 LTD 曲线
|-- reservoir_device_response_*.csv   # 示例水库器件响应
|-- tools/check_env.py                # 环境检查
|-- tools/first_run_setup.py          # 首次启动初始化
|-- docs/datasets.md                  # 数据集来源和生成说明
|-- docs/codex-start.md               # 可以直接发给 Codex 的启动提示词
|-- .codex/skills/...                 # 仓库自带 Codex skill
```

## 重要说明：数据集不直接上传

MNIST、Fashion-MNIST、EMNIST 都是公开数据集。为了让 GitHub 仓库保持轻量，本仓库不上传训练集/测试集本体，而是提供脚本自动从公开源下载并生成本地文件。

生成后会在本地出现：

- `patch60000_train.txt`
- `label60000_train.txt`
- `patch10000_test.txt`
- `label10000_test.txt`
- `Datasets/`
- `TorchData/`

这些文件会被 `.gitignore` 忽略，不会上传到 GitHub。

## 环境要求

推荐 Windows 环境：

- MATLAB R2022b 或兼容版本
- MSYS2/MinGW，包含 `make` 和 `g++`
- Python 3.10 或更高版本
- PyTorch，如需 GPU 加速请安装 CUDA 版本
- NVIDIA 显卡和驱动，如需运行 CUDA

## 新手快速开始

克隆仓库：

```powershell
git clone https://github.com/awei-duanduan/neuro-device-algorithm-tool.git
cd neuro-device-algorithm-tool
```

创建 Python 虚拟环境：

```powershell
python -m venv .venv_torch
.\.venv_torch\Scripts\python.exe -m pip install --upgrade pip
.\.venv_torch\Scripts\python.exe -m pip install -r requirements.txt
```

如需 CUDA，请安装 CUDA 版 PyTorch，例如：

```powershell
.\.venv_torch\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

首次初始化，自动检查环境并下载/生成 NeuroSim 数据：

```powershell
.\.venv_torch\Scripts\python.exe tools\first_run_setup.py
```

如果想提前下载 PyTorch 的 MNIST/Fashion-MNIST 缓存：

```powershell
.\.venv_torch\Scripts\python.exe tools\first_run_setup.py --torch-cache
```

EMNIST 比较大，需要时再下载：

```powershell
.\.venv_torch\Scripts\python.exe tools\first_run_setup.py --emnist
```

## 打开 MATLAB GUI

在 MATLAB 中进入仓库目录，然后运行：

```matlab
fit_device_gui
```

或者在 PowerShell 中运行：

```powershell
matlab -batch "fit_device_gui"
```

GUI 中包含两个主要页面：

- `MLP / NeuroSim`：LTP/LTD 拟合、更新 `Cell.cpp`、运行 C++ NeuroSim、运行 PyTorch CUDA。
- `Reservoir Computing`：使用器件脉冲响应做水库计算。

## LTP/LTD 拟合流程

1. 选择 LTP CSV 文件。
2. 选择 LTD CSV 文件。
3. 点击 `Fit Preview` 查看拟合曲线。
4. 确认后点击 `Fit and Update`。
5. 工具会把拟合参数写入 `Cell.cpp` 的 `RealDevice`。
6. 点击 `Run Neural Network` 或 `Run PyTorch CUDA` 运行识别任务。

CSV 文件建议包含两列：

```text
pulse,conductance
0,0.10
1,0.18
2,0.26
...
```

## 运行神经网络

C++ NeuroSim：

```powershell
make
.\main.exe
```

PyTorch CUDA 示例：

```powershell
.\.venv_torch\Scripts\python.exe -u torch_neurosim_train.py --dataset mnist --epochs 1 --hidden-neurons 30 --output-dir torch_results_smoke
```

## 运行水库计算

小规模快速测试：

```powershell
.\.venv_torch\Scripts\python.exe -u rc_reservoir_train.py --task paper-multitask --device-response reservoir_device_response_pndi_05s.csv --epochs 1 --train-limit 150 --test-limit 45 --batch-size 64 --output-dir rc_results_smoke
```

## 结果在哪里

运行结果会保存到你设置的输出目录，例如：

- `torch_results_*`
- `rc_results_*`

常见输出文件：

- `loss_curve.png`
- `accuracy_curve.png`
- `loss_history.csv`
- `accuracy_history.csv`
- `confusion_matrix.png`
- `confusion_matrix_counts.csv`
- `confusion_matrix_percent.csv`
- `summary.json`

水库计算还会输出：

- `paper_multitask_summary.png`
- `device_response_map.png`
- `reservoir_feature_heatmap.png`
- `reservoir_feature_pca.png`
- `reservoir_feature_matrix.csv`
- `reservoir_feature_pca_coordinates.csv`

## 如何让 Codex 帮你自动初始化

你可以把下面这段直接发给 Codex：

```text
请从 GitHub 克隆并初始化这个非易失性神经器件算法识别工具仓库：

https://github.com/awei-duanduan/neuro-device-algorithm-tool.git

请执行 clone、创建 Python 虚拟环境、安装依赖、读取 .codex/skills/neuro-device-algorithm-tool、运行 tools/first_run_setup.py。
如果缺少 MATLAB、MSYS2/MinGW、PyTorch CUDA 或 NVIDIA 驱动，请告诉我具体缺什么以及如何安装。
初始化完成后，打开 MATLAB GUI，并说明如何进行 LTP/LTD 拟合、Run Neural Network、Run PyTorch CUDA 和 Reservoir Computing。
```

更多提示词见 `docs/codex-start.md`。

## 原始 MLP+NeuroSim 说明

原始 MLP+NeuroSim 框架由 Prof. Shimeng Yu 团队开发，并以非商业方式公开。原始模型遵循 Creative Commons Attribution-NonCommercial 4.0 International Public License。

如果使用或改写原始工具，请引用：

P.-Y. Chen, X. Peng, S. Yu, "NeuroSim+: An integrated device-to-algorithm framework for benchmarking synaptic devices and array architectures," IEEE International Electron Devices Meeting (IEDM), 2017.

原始模拟器细节请参考 `Documents/Manual.pdf`。

---

# Nonvolatile Neural Device Algorithm Tool

[中文](#非易失性神经器件算法识别工具) | English

This repository packages a complete device-to-algorithm workflow for neuromorphic device research. It extends the original `MLP+NeuroSim V2.0` with a MATLAB GUI, LTP/LTD curve fitting, C++ NeuroSim execution, PyTorch CUDA post-mapping recognition, and reservoir-computing workflows.

The practical goal is to connect measured device curves to algorithm-level performance:

1. Import LTP/LTD or pulse-response curves.
2. Fit nonlinear device parameters.
3. Write fitted values into `Cell.cpp` `RealDevice`.
4. Run MNIST/Fashion-MNIST recognition or reservoir-computing tasks.
5. Export accuracy, loss curves, confusion matrices, feature plots, and matching CSV/JSON data.

## Who This Is For

- Researchers working on memristors, synaptic transistors, optoelectronic synapses, iontronic devices, and related nonvolatile neural devices.
- Users who want to map device characteristics to neural-network recognition performance.
- Beginners who want a guided device-to-algorithm workflow.

## What Is Included

```text
.
|-- fit_device_gui.m                 # MATLAB GUI entry point
|-- fit_device_to_realsim.m           # LTP/LTD fitting and RealDevice mapping
|-- nonlinear_fit.m                   # Original NeuroSim nonlinear fitting script
|-- Cell.cpp                          # RealDevice parameter target
|-- Param.cpp / Train.cpp / Test.cpp  # MLP+NeuroSim C++ sources
|-- torch_neurosim_train.py           # PyTorch CUDA post-mapping recognition
|-- rc_reservoir_train.py             # Reservoir-computing runner
|-- prepare_neurosim_dataset.py       # Download public datasets and generate NeuroSim text files
|-- volatile_LTP_example.csv          # Example LTP curve
|-- volatile_LTD_example.csv          # Example LTD curve
|-- reservoir_device_response_*.csv   # Example reservoir device responses
|-- tools/check_env.py                # Environment checker
|-- tools/first_run_setup.py          # First-run setup
|-- docs/datasets.md                  # Dataset source notes
|-- docs/codex-start.md               # Copy-paste Codex prompt
|-- .codex/skills/...                 # Built-in Codex skill
```

## Datasets Are Downloaded, Not Committed

MNIST, Fashion-MNIST, and EMNIST are public datasets. To keep the GitHub repository lightweight, dataset files are not committed. Scripts download public sources and generate local NeuroSim train/test files.

Generated local files may include:

- `patch60000_train.txt`
- `label60000_train.txt`
- `patch10000_test.txt`
- `label10000_test.txt`
- `Datasets/`
- `TorchData/`

These files are ignored by Git.

## Requirements

Recommended Windows environment:

- MATLAB R2022b or compatible
- MSYS2/MinGW with `make` and `g++`
- Python 3.10+
- PyTorch, CUDA build if GPU acceleration is needed
- NVIDIA GPU and driver for CUDA

## Quick Start

Clone the repository:

```powershell
git clone https://github.com/awei-duanduan/neuro-device-algorithm-tool.git
cd neuro-device-algorithm-tool
```

Create a Python virtual environment:

```powershell
python -m venv .venv_torch
.\.venv_torch\Scripts\python.exe -m pip install --upgrade pip
.\.venv_torch\Scripts\python.exe -m pip install -r requirements.txt
```

Install CUDA PyTorch if GPU acceleration is needed:

```powershell
.\.venv_torch\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Run first-time setup:

```powershell
.\.venv_torch\Scripts\python.exe tools\first_run_setup.py
```

Optionally pre-download torchvision MNIST/Fashion-MNIST cache:

```powershell
.\.venv_torch\Scripts\python.exe tools\first_run_setup.py --torch-cache
```

Download EMNIST only when needed:

```powershell
.\.venv_torch\Scripts\python.exe tools\first_run_setup.py --emnist
```

## Open The MATLAB GUI

In MATLAB, switch to the repository folder and run:

```matlab
fit_device_gui
```

Or from PowerShell:

```powershell
matlab -batch "fit_device_gui"
```

The GUI has two main pages:

- `MLP / NeuroSim`: LTP/LTD fitting, `Cell.cpp` update, C++ NeuroSim run, PyTorch CUDA run.
- `Reservoir Computing`: reservoir-computing workflows based on device pulse responses.

## LTP/LTD Fitting Workflow

1. Select the LTP CSV file.
2. Select the LTD CSV file.
3. Click `Fit Preview`.
4. Click `Fit and Update`.
5. The fitted parameters are written into `Cell.cpp` `RealDevice`.
6. Click `Run Neural Network` or `Run PyTorch CUDA`.

Recommended CSV format:

```text
pulse,conductance
0,0.10
1,0.18
2,0.26
...
```

## Run Neural Recognition

C++ NeuroSim:

```powershell
make
.\main.exe
```

PyTorch CUDA smoke test:

```powershell
.\.venv_torch\Scripts\python.exe -u torch_neurosim_train.py --dataset mnist --epochs 1 --hidden-neurons 30 --output-dir torch_results_smoke
```

## Run Reservoir Computing

Small smoke test:

```powershell
.\.venv_torch\Scripts\python.exe -u rc_reservoir_train.py --task paper-multitask --device-response reservoir_device_response_pndi_05s.csv --epochs 1 --train-limit 150 --test-limit 45 --batch-size 64 --output-dir rc_results_smoke
```

## Outputs

Common output files:

- `loss_curve.png`
- `accuracy_curve.png`
- `loss_history.csv`
- `accuracy_history.csv`
- `confusion_matrix.png`
- `confusion_matrix_counts.csv`
- `confusion_matrix_percent.csv`
- `summary.json`

Reservoir-computing outputs may also include:

- `paper_multitask_summary.png`
- `device_response_map.png`
- `reservoir_feature_heatmap.png`
- `reservoir_feature_pca.png`
- `reservoir_feature_matrix.csv`
- `reservoir_feature_pca_coordinates.csv`

## Codex Prompt

Users can give Codex this prompt:

```text
Please clone and initialize this nonvolatile neural-device algorithm tool:

https://github.com/awei-duanduan/neuro-device-algorithm-tool.git

Clone the repository, create the Python virtual environment, install dependencies, read .codex/skills/neuro-device-algorithm-tool, and run tools/first_run_setup.py.
If MATLAB, MSYS2/MinGW, PyTorch CUDA, or NVIDIA drivers are missing, tell me exactly what is missing and how to install it.
After initialization, open the MATLAB GUI and explain how to run LTP/LTD fitting, Run Neural Network, Run PyTorch CUDA, and Reservoir Computing.
```

More prompts are available in `docs/codex-start.md`.

## Original MLP+NeuroSim Notice

The original MLP+NeuroSim framework was developed by Prof. Shimeng Yu's group and released for non-commercial use under the Creative Commons Attribution-NonCommercial 4.0 International Public License.

If you use or adapt the original tool, cite:

P.-Y. Chen, X. Peng, S. Yu, "NeuroSim+: An integrated device-to-algorithm framework for benchmarking synaptic devices and array architectures," IEEE International Electron Devices Meeting (IEDM), 2017.

See `Documents/Manual.pdf` for baseline MLP+NeuroSim details.
