from pathlib import Path
import argparse
import gzip
import shutil
import struct
import urllib.request


ROOT = Path(__file__).resolve().parent
DATASETS = {
    "mnist": {
        "raw_dir": ROOT / "Datasets" / "MNIST_raw",
        "out_prefix": "MNIST",
        "urls": [
            "https://storage.googleapis.com/cvdf-datasets/mnist/",
            "https://github.com/fgnt/mnist/raw/master/",
        ],
        "files": {
            "train_images": "train-images-idx3-ubyte.gz",
            "train_labels": "train-labels-idx1-ubyte.gz",
            "test_images": "t10k-images-idx3-ubyte.gz",
            "test_labels": "t10k-labels-idx1-ubyte.gz",
        },
    },
    "fashion": {
        "raw_dir": ROOT / "Datasets" / "FashionMNIST_raw",
        "out_prefix": "FashionMNIST",
        "urls": [
            "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/",
            "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/",
        ],
        "files": {
            "train_images": "train-images-idx3-ubyte.gz",
            "train_labels": "train-labels-idx1-ubyte.gz",
            "test_images": "t10k-images-idx3-ubyte.gz",
            "test_labels": "t10k-labels-idx1-ubyte.gz",
        },
    },
}


def download_file(raw_dir: Path, urls, name: str) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / name
    if target.exists() and target.stat().st_size > 0:
        return target

    errors = []
    for base_url in urls:
        url = base_url + name
        try:
            print(f"Downloading {url}")
            urllib.request.urlretrieve(url, target)
            return target
        except Exception as exc:
            errors.append(f"{url}: {exc}")
            if target.exists():
                target.unlink()
    raise RuntimeError("Could not download dataset file:\n" + "\n".join(errors))


def read_idx_images(path: Path):
    with gzip.open(path, "rb") as f:
        magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"{path} is not an IDX image file")
        data = f.read(count * rows * cols)
    return data, count, rows, cols


def read_idx_labels(path: Path):
    with gzip.open(path, "rb") as f:
        magic, count = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"{path} is not an IDX label file")
        data = f.read(count)
    return data, count


def sample_pixel(data: bytes, image_index: int, pixel_index: int, rows: int, cols: int, image_size: int) -> float:
    row_out = pixel_index // image_size
    col_out = pixel_index % image_size
    row_in = row_out + (rows - image_size) // 2
    col_in = col_out + (cols - image_size) // 2
    value = data[image_index * rows * cols + row_in * cols + col_in]
    return value / 255.0


def write_patch_file(images: bytes, count: int, rows: int, cols: int, image_size: int, out_path: Path):
    pixel_count = image_size * image_size
    with out_path.open("w", encoding="ascii", newline="\n") as f:
        for pixel_index in range(pixel_count):
            values = [
                f"{sample_pixel(images, image_index, pixel_index, rows, cols, image_size):.4f}"
                for image_index in range(count)
            ]
            f.write(" ".join(values))
            f.write("\n")


def write_label_file(labels: bytes, count: int, out_path: Path):
    with out_path.open("w", encoding="ascii", newline="\n") as f:
        for i in range(count):
            f.write(f"{labels[i]}\n")


def copy_root_neurosim_files(out_dir: Path):
    mappings = [
        ("patch60000_train.txt", ROOT / "patch60000_train.txt"),
        ("label60000_train.txt", ROOT / "label60000_train.txt"),
        ("patch10000_test.txt", ROOT / "patch10000_test.txt"),
        ("label10000_test.txt", ROOT / "label10000_test.txt"),
    ]
    for name, target in mappings:
        shutil.copyfile(out_dir / name, target)
    print("Copied MNIST 28x28 files to the project root for the original C++ NeuroSim runner.")


def prepare(dataset: str, image_size: int, root_copy: bool = False) -> Path:
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset}")
    if image_size < 1 or image_size > 28:
        raise ValueError("image_size must be between 1 and 28")

    config = DATASETS[dataset]
    out_dir = ROOT / "Datasets" / f"{config['out_prefix']}_{image_size}x{image_size}"
    outputs = [
        out_dir / "patch60000_train.txt",
        out_dir / "label60000_train.txt",
        out_dir / "patch10000_test.txt",
        out_dir / "label10000_test.txt",
    ]
    if all(p.exists() and p.stat().st_size > 0 for p in outputs):
        print(f"{dataset} {image_size}x{image_size} is already prepared in {out_dir}")
        if root_copy:
            if dataset != "mnist" or image_size != 28:
                raise ValueError("--root-copy is only valid for MNIST 28x28")
            copy_root_neurosim_files(out_dir)
        return out_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        key: download_file(config["raw_dir"], config["urls"], name)
        for key, name in config["files"].items()
    }

    train_images, train_count, rows, cols = read_idx_images(paths["train_images"])
    train_labels, train_label_count = read_idx_labels(paths["train_labels"])
    test_images, test_count, test_rows, test_cols = read_idx_images(paths["test_images"])
    test_labels, test_label_count = read_idx_labels(paths["test_labels"])

    if (train_count, train_label_count, test_count, test_label_count) != (60000, 60000, 10000, 10000):
        raise ValueError("Unexpected data size")
    if (rows, cols, test_rows, test_cols) != (28, 28, 28, 28):
        raise ValueError("Unexpected image shape")

    print(f"Writing {dataset} {image_size}x{image_size} NeuroSim patch files...")
    write_patch_file(train_images, train_count, rows, cols, image_size, outputs[0])
    write_label_file(train_labels, train_count, outputs[1])
    write_patch_file(test_images, test_count, test_rows, test_cols, image_size, outputs[2])
    write_label_file(test_labels, test_label_count, outputs[3])
    print(f"Prepared in {out_dir}")
    if root_copy:
        if dataset != "mnist" or image_size != 28:
            raise ValueError("--root-copy is only valid for MNIST 28x28")
        copy_root_neurosim_files(out_dir)
    return out_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--image-size", type=int, required=True)
    parser.add_argument("--root-copy", action="store_true", help="Copy MNIST 28x28 outputs to the project root for the original C++ NeuroSim runner.")
    args = parser.parse_args()
    prepare(args.dataset, args.image_size, args.root_copy)


if __name__ == "__main__":
    main()
