"""Normalize the raw images and split off a validation set.

Pixels are scaled to [0, 1] and then standardized to zero mean and unit
variance using statistics from the training split only. Reads data/raw/
and writes the train, val and test splits to data/processed/.
Hyperparameters come from the `preprocess` section of params.yaml.
"""

import argparse
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"


def load_params(path: Path) -> dict:
    """Return the `preprocess` section of the params file."""
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)["preprocess"]


def load_raw(name: str) -> tuple[np.ndarray, np.ndarray]:
    """Load one raw split as an (images, labels) pair."""
    with np.load(RAW_DIR / f"{name}.npz") as data:
        return data["images"], data["labels"]


def scale(images: np.ndarray) -> np.ndarray:
    """Scale uint8 pixel values from [0, 255] to float32 in [0, 1]."""
    return images.astype(np.float32) / 255.0


def standardize(images: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Shift and scale pixels to zero mean and unit variance."""
    return ((images - mean) / std).astype(np.float32)


def save(name: str, images: np.ndarray, labels: np.ndarray) -> None:
    """Save one processed split and report its shape."""
    np.savez_compressed(PROCESSED_DIR / f"{name}.npz",
                        images=images, labels=labels)
    print(f"{name:5s}: images {images.shape} {images.dtype}")


def main() -> None:
    """Scale pixels, hold out a validation set, standardize, save splits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", type=Path, default=ROOT / "params.yaml")
    args = parser.parse_args()
    params = load_params(args.params)

    x_train, y_train = load_raw("train")
    x_test, y_test = load_raw("test")
    x_train, x_test = scale(x_train), scale(x_test)

    # Shuffle first so the validation set is a random sample.
    rng = np.random.default_rng(params["seed"])
    perm = rng.permutation(len(x_train))
    n_val = int(len(x_train) * params["val_split"])
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    # Fit the statistics on the training split only so val and test stay unseen.
    mean, std = x_train[train_idx].mean(), x_train[train_idx].std()
    x_train, x_test = standardize(x_train, mean, std), standardize(x_test, mean, std)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    save("train", x_train[train_idx], y_train[train_idx])
    save("val", x_train[val_idx], y_train[val_idx])
    save("test", x_test, y_test)


if __name__ == "__main__":
    main()
