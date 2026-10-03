"""Normalize the raw images and split off a validation set.

Pixels are rescaled to [-1, 1]. Reads data/raw/ and writes the train,
val and test splits to data/processed/. Hyperparameters come from the
`preprocess` section of params.yaml.
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


def normalize(images: np.ndarray) -> np.ndarray:
    """Scale uint8 pixel values from [0, 255] to float32 in [-1, 1]."""
    return images.astype(np.float32) / 127.5 - 1.0


def save(name: str, images: np.ndarray, labels: np.ndarray) -> None:
    """Save one processed split and report its shape."""
    np.savez_compressed(PROCESSED_DIR / f"{name}.npz",
                        images=images, labels=labels)
    print(f"{name:5s}: images {images.shape} {images.dtype}")


def main() -> None:
    """Normalize pixels, hold out a validation set, save all splits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", type=Path, default=ROOT / "params.yaml")
    args = parser.parse_args()
    params = load_params(args.params)

    x_train, y_train = load_raw("train")
    x_test, y_test = load_raw("test")
    x_train, x_test = normalize(x_train), normalize(x_test)

    # Shuffle first so the validation set is a random sample.
    rng = np.random.default_rng(params["seed"])
    perm = rng.permutation(len(x_train))
    n_val = int(len(x_train) * params["val_split"])
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    save("train", x_train[train_idx], y_train[train_idx])
    save("val", x_train[val_idx], y_train[val_idx])
    save("test", x_test, y_test)


if __name__ == "__main__":
    main()
