from pathlib import Path
import gzip
import shutil
import struct
import urllib.request


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "Datasets" / "FashionMNIST_raw"
OUT = ROOT / "Datasets" / "FashionMNIST"

FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

BASE_URLS = [
    "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/",
    "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/",
]


def download_file(name: str) -> Path:
    RAW.mkdir(parents=True, exist_ok=True)
    target = RAW / name
    if target.exists() and target.stat().st_size > 0:
        return target

    errors = []
    for base_url in BASE_URLS:
        url = base_url + name
        try:
            print(f"Downloading {url}")
            urllib.request.urlretrieve(url, target)
            return target
        except Exception as exc:
            errors.append(f"{url}: {exc}")
            if target.exists():
                target.unlink()
    raise RuntimeError("Could not download Fashion-MNIST file:\n" + "\n".join(errors))


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


def pixel_20x20(data: bytes, image_index: int, pixel_index: int, rows: int, cols: int) -> float:
    row20 = pixel_index // 20
    col20 = pixel_index % 20
    row28 = row20 + 4
    col28 = col20 + 4
    value = data[image_index * rows * cols + row28 * cols + col28]
    return value / 255.0


def write_patch_file(images: bytes, count: int, rows: int, cols: int, out_path: Path):
    with out_path.open("w", encoding="ascii", newline="\n") as f:
        for pixel_index in range(400):
            values = [
                f"{pixel_20x20(images, image_index, pixel_index, rows, cols):.4f}"
                for image_index in range(count)
            ]
            f.write(" ".join(values))
            f.write("\n")


def write_label_file(labels: bytes, count: int, out_path: Path):
    with out_path.open("w", encoding="ascii", newline="\n") as f:
        for i in range(count):
            f.write(f"{labels[i]}\n")


def ensure_mnist_snapshot():
    mnist_dir = ROOT / "Datasets" / "MNIST"
    mnist_dir.mkdir(parents=True, exist_ok=True)
    for name in [
        "patch60000_train.txt",
        "label60000_train.txt",
        "patch10000_test.txt",
        "label10000_test.txt",
    ]:
        source = ROOT / name
        target = mnist_dir / name
        if source.exists() and not target.exists():
            print(f"Saving MNIST snapshot: {target}")
            shutil.copy2(source, target)


def main():
    ensure_mnist_snapshot()
    OUT.mkdir(parents=True, exist_ok=True)

    outputs = [
        OUT / "patch60000_train.txt",
        OUT / "label60000_train.txt",
        OUT / "patch10000_test.txt",
        OUT / "label10000_test.txt",
    ]
    if all(p.exists() and p.stat().st_size > 0 for p in outputs):
        print(f"Fashion-MNIST is already prepared in {OUT}")
        return

    paths = {key: download_file(name) for key, name in FILES.items()}

    train_images, train_count, rows, cols = read_idx_images(paths["train_images"])
    train_labels, train_label_count = read_idx_labels(paths["train_labels"])
    test_images, test_count, test_rows, test_cols = read_idx_images(paths["test_images"])
    test_labels, test_label_count = read_idx_labels(paths["test_labels"])

    if (train_count, train_label_count, test_count, test_label_count) != (60000, 60000, 10000, 10000):
        raise ValueError("Unexpected Fashion-MNIST data size")
    if (rows, cols, test_rows, test_cols) != (28, 28, 28, 28):
        raise ValueError("Unexpected Fashion-MNIST image shape")

    print("Writing Fashion-MNIST NeuroSim patch files...")
    write_patch_file(train_images, train_count, rows, cols, outputs[0])
    write_label_file(train_labels, train_count, outputs[1])
    write_patch_file(test_images, test_count, test_rows, test_cols, outputs[2])
    write_label_file(test_labels, test_count, outputs[3])
    print(f"Fashion-MNIST prepared in {OUT}")


if __name__ == "__main__":
    main()
