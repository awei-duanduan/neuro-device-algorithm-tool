# Environment And Workflow

## Project Discovery

The tool is portable. Do not assume a fixed drive letter or user name.

Use one of these methods to locate the project:

- Run commands from the project root.
- Pass `--project "<path-to-project>"` to `tools/check_env.py`.
- Set `NEURO_DEVICE_PROJECT=<path-to-project>`.

The project root should contain `fit_device_gui.m`, `Cell.cpp`, `torch_neurosim_train.py`, and `rc_reservoir_train.py`.

Generated NeuroSim train/test data may include:

- `patch60000_train.txt`
- `patch10000_test.txt`
- `label60000_train.txt`
- `label10000_test.txt`
- `MNIST_data.zip`
- `Datasets/` variants for MNIST/Fashion-MNIST and image-size options

These files are intentionally not committed. They are generated from public dataset sources.

## Required Environment Pairing

| Capability | Required component | Check |
|---|---|---|
| Train/test data | Public dataset download | `prepare_neurosim_dataset.py` generates NeuroSim text files locally |
| MATLAB GUI | MATLAB R2022b or compatible | `matlab` on PATH, `MATLAB_EXE`, or a common Windows install path |
| C++ NeuroSim | MSYS2/MinGW UCRT64 | `make` and `g++` available |
| PyTorch training | Python virtual environment | `torch`, `torchvision`, `matplotlib`, `numpy`, `pandas`, `openpyxl`, `PIL`, `fitz` import |
| CUDA acceleration | NVIDIA driver + CUDA PyTorch | `torch.cuda.is_available()` is true |
| Reservoir paper mode | 32-state p-NDI CSV | `reservoir_device_response_pndi_05s.csv` exists |
| Reservoir single mode | 16-state Pei CSV | `reservoir_device_response_pei2025.csv` exists |

## GUI Pages

- `MLP / NeuroSim`: LTP/LTD fitting, `Cell.cpp` RealDevice update, C++ run, PyTorch CUDA post-mapping run.
- `Reservoir Computing`: single-task or paper multitask reservoir computing.

## Reservoir Paper Mode

- Dataset: MNIST digits, EMNIST letters `L/M/S`, and Fashion-MNIST classes `T-shirt/Pants/Dress/Bag/Shoes`
- Device coding: p-NDI 5-bit, 32 states
- Default response file: `reservoir_device_response_pndi_05s.csv`
- Readout: three independent linear readouts
- Default GUI settings: epochs `100`, batch size `2048`, train/test sample fields `0` for full available data

Outputs include:

- `paper_multitask_summary.png`
- `device_response_map.png`
- `reservoir_feature_heatmap.png`
- `reservoir_feature_pca.png`
- `reservoir_feature_matrix.csv`
- `reservoir_feature_pca_coordinates.csv`
- per-task loss, accuracy, and confusion matrix CSV/PNG files
- `summary.json`

## Troubleshooting

- If GPU appears idle in Windows Task Manager, check the NVIDIA GPU panel rather than Intel GPU 0.
- If PyTorch says CUDA is unavailable, verify the PyTorch wheel was installed with a CUDA index URL.
- If MATLAB reports `ProcessStartInfo.Arguments` type errors, ensure process arguments are passed as `System.String`.
- If EMNIST logs many percent progress lines, it is downloading/extracting the dataset for the first time.
- If GUI controls overlap, adjust normalized positions in `fit_device_gui.m` and verify with a screenshot.
