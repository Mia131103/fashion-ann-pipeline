"""Download Fashion-MNIST and save the raw arrays to data/raw/."""

from pathlib import Path

import numpy as np
from tensorflow.keras.datasets import fashion_mnist

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def main() -> None:
    """Save the raw train and test splits as compressed .npz files."""
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(RAW_DIR / "train.npz", images=x_train, labels=y_train)
    np.savez_compressed(RAW_DIR / "test.npz", images=x_test, labels=y_test)
    print(f"train {x_train.shape}, test {x_test.shape} -> {RAW_DIR}")


if __name__ == "__main__":
    main()
