"""Normalize raw Fashion-MNIST arrays and split a validation set off the training data.

Reads data/raw/{train,test}.npz and writes data/processed/{train,val,test}.npz.
Hyperparameters come from the `preprocess` section of params.yaml:

    preprocess:
      val_split: 0.1   # fraction of the training data held out for validation
      seed: 42         # RNG seed for the shuffle before splitting

Run standalone:  python src/preprocess.py [--params params.yaml]
"""

import argparse
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"


def load_params(path: Path) -> dict:
    with open(path) as f:
        params = yaml.safe_load(f)
    try:
        section = params["preprocess"]
        return {"val_split": float(section["val_split"]), "seed": int(section["seed"])}
    except (TypeError, KeyError) as e:
        raise SystemExit(f"{path}: missing 'preprocess.val_split' / 'preprocess.seed' ({e!r})")


def load_raw(name: str) -> tuple[np.ndarray, np.ndarray]:
    with np.load(RAW_DIR / f"{name}.npz") as data:
        return data["images"], data["labels"]


def normalize(images: np.ndarray) -> np.ndarray:
    return images.astype(np.float32) / 255.0


def save(name: str, images: np.ndarray, labels: np.ndarray) -> None:
    np.savez_compressed(PROCESSED_DIR / f"{name}.npz", images=images, labels=labels)
    print(f"{name:5s}: images {images.shape} {images.dtype}, labels {labels.shape}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--params", type=Path, default=ROOT / "params.yaml")
    args = parser.parse_args()

    params = load_params(args.params)
    if not 0.0 < params["val_split"] < 1.0:
        raise SystemExit(f"preprocess.val_split must be in (0, 1), got {params['val_split']}")

    x_train, y_train = load_raw("train")
    x_test, y_test = load_raw("test")
    x_train, x_test = normalize(x_train), normalize(x_test)

    # Shuffle before splitting so the validation set is a random sample.
    perm = np.random.default_rng(params["seed"]).permutation(len(x_train))
    n_val = int(len(x_train) * params["val_split"])
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    save("train", x_train[train_idx], y_train[train_idx])
    save("val", x_train[val_idx], y_train[val_idx])
    save("test", x_test, y_test)
    print(f"saved to {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
