# Codex Start Prompt

After cloning this repository, users can ask Codex to initialize and operate the tool with this prompt:

```text
请使用这个仓库里的 .codex/skills/neuro-device-algorithm-tool 作为工作说明。
先运行 tools/first_run_setup.py 完成环境检查和公开数据集下载/生成。
如果缺少 MATLAB、MSYS2/MinGW、PyTorch CUDA 或 Python 依赖，请告诉我具体缺什么以及怎么安装。
初始化完成后，打开 MATLAB GUI，并说明如何进行 LTP/LTD 拟合、Run Neural Network、Run PyTorch CUDA 和 Reservoir Computing。
```

If the user wants a faster check without downloading datasets:

```text
请读取这个仓库里的 .codex/skills/neuro-device-algorithm-tool。
只运行 tools/first_run_setup.py --check-only 检查环境，不下载数据。
```

If the user wants PyTorch cache prepared at startup:

```text
请读取这个仓库里的 .codex/skills/neuro-device-algorithm-tool。
运行 tools/first_run_setup.py --torch-cache，让 MNIST 和 Fashion-MNIST 的 PyTorch 缓存也提前下载。
EMNIST 很大，除非我明确要求，否则不要预下载 EMNIST。
```

To install the skill into a local Codex profile, copy:

```text
.codex/skills/neuro-device-algorithm-tool
```

to:

```text
%USERPROFILE%\.codex\skills\neuro-device-algorithm-tool
```
