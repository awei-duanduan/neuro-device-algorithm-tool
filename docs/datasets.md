# Dataset Sources And Local Generation

Datasets are intentionally not committed to this repository. They are public datasets that can be downloaded again, and committing generated train/test text files would make the GitHub repository unnecessarily large.

## Public Sources

- MNIST: https://yann.lecun.com/exdb/mnist/
- Fashion-MNIST: https://github.com/zalandoresearch/fashion-mnist
- EMNIST: https://www.nist.gov/itl/products-and-services/emnist-dataset

The project uses public mirrors where needed for scripted downloads.

## Generate NeuroSim Text Files

From the project root:

```powershell
.\.venv_torch\Scripts\python.exe tools\first_run_setup.py
```

Or generate each NeuroSim dataset explicitly:

```powershell
.\.venv_torch\Scripts\python.exe prepare_neurosim_dataset.py --dataset mnist --image-size 28 --root-copy
.\.venv_torch\Scripts\python.exe prepare_neurosim_dataset.py --dataset fashion --image-size 28
.\.venv_torch\Scripts\python.exe prepare_neurosim_dataset.py --dataset mnist --image-size 20
.\.venv_torch\Scripts\python.exe prepare_neurosim_dataset.py --dataset fashion --image-size 20
```

`--root-copy` copies the MNIST 28x28 files to the project root for the original C++ MLP+NeuroSim runner:

- `patch60000_train.txt`
- `label60000_train.txt`
- `patch10000_test.txt`
- `label10000_test.txt`

Generated dataset folders are written under `Datasets/`.

## PyTorch Cache

The PyTorch CUDA and reservoir-computing workflows use `torchvision` datasets. MNIST, Fashion-MNIST, and EMNIST are downloaded automatically into `TorchData/` on first run.

`TorchData/`, `Datasets/`, and generated NeuroSim text files are ignored by Git.
