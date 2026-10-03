"""Train the Fashion-MNIST ANN and save the model and its history.

Reads data/processed/ and writes models/model.h5 and
models/history.csv. Hyperparameters come from the `train` section of
params.yaml.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
NUM_CLASSES = 10


def load_params(path: Path) -> dict:
    """Return the `train` section of the params file."""
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)["train"]


def load_split(name: str) -> tuple[np.ndarray, np.ndarray]:
    """Load one processed split as an (images, labels) pair."""
    with np.load(PROCESSED_DIR / f"{name}.npz") as data:
        return data["images"], data["labels"]


def build_model(input_shape: tuple[int, ...], params: dict) -> tf.keras.Model:
    """Build and compile the Flatten-Dense-Dropout-Softmax model."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(params["hidden_units"], activation="relu"),
        tf.keras.layers.Dropout(params["dropout_rate"]),
        tf.keras.layers.Dense(NUM_CLASSES, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(params["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_history(history: dict[str, list[float]], path: Path) -> None:
    """Write the per-epoch metrics to a CSV file, one row per epoch."""
    rows = zip(*history.values(), strict=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", *history])
        writer.writerows([i, *row] for i, row in enumerate(rows, start=1))


def main() -> None:
    """Build the model, train it and save the model and history."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", type=Path, default=ROOT / "params.yaml")
    args = parser.parse_args()
    params = load_params(args.params)
    tf.keras.utils.set_random_seed(params["seed"])

    x_train, y_train = load_split("train")
    x_val, y_val = load_split("val")

    model = build_model(x_train.shape[1:], params)
    model.summary()
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=params["epochs"],
        batch_size=params["batch_size"],
        verbose=2,
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODELS_DIR / "model.h5")
    save_history(history.history, MODELS_DIR / "history.csv")


if __name__ == "__main__":
    main()
