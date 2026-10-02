"""Download Fashion-MNIST and save the raw train/test arrays to data/raw/.

Run standalone:  python src/prepare.py
"""

from pathlib import Path

import numpy as np
from tensorflow.keras.datasets import fashion_mnist

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def main() -> None:
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(RAW_DIR / "train.npz", images=x_train, labels=y_train)
    np.savez_compressed(RAW_DIR / "test.npz", images=x_test, labels=y_test)

    print(f"train: images {x_train.shape} {x_train.dtype}, labels {y_train.shape}")
    print(f"test:  images {x_test.shape} {x_test.dtype}, labels {y_test.shape}")
    print(f"saved to {RAW_DIR}")


if __name__ == "__main__":
    main()
