"""Train a Sequential ANN on the processed Fashion-MNIST data.

Model: Flatten -> Dense(ReLU) -> Dropout -> Dense(10, Softmax), compiled with Adam and
sparse_categorical_crossentropy. Reads data/processed/{train,val}.npz and writes
models/model.h5 and models/history.csv.
Hyperparameters come from the `train` section of params.yaml:

    train:
      hidden_units: 128     # neurons in the Dense (ReLU) layer
      dropout_rate: 0.2     # fraction of units dropped after the hidden layer
      learning_rate: 0.001  # Adam learning rate
      epochs: 10
      batch_size: 32
      seed: 42              # seed for Python / NumPy / TensorFlow RNGs

Run standalone:  python src/train.py [--params params.yaml]
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
PARAM_TYPES = {
    "hidden_units": int,
    "dropout_rate": float,
    "learning_rate": float,
    "epochs": int,
    "batch_size": int,
    "seed": int,
}


def load_params(path: Path) -> dict:
    with open(path) as f:
        params = yaml.safe_load(f)
    try:
        section = params["train"]
        return {key: cast(section[key]) for key, cast in PARAM_TYPES.items()}
    except (TypeError, KeyError) as e:
        raise SystemExit(f"{path}: missing or invalid 'train' section ({e!r}); need {list(PARAM_TYPES)}")


def load_split(name: str) -> tuple[np.ndarray, np.ndarray]:
    path = PROCESSED_DIR / f"{name}.npz"
    if not path.exists():
        raise SystemExit(f"{path} not found; run src/preprocess.py first")
    with np.load(path) as data:
        return data["images"], data["labels"]


def build_model(input_shape: tuple[int, ...], params: dict) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(params["hidden_units"], activation="relu"),
            tf.keras.layers.Dropout(params["dropout_rate"]),
            tf.keras.layers.Dense(NUM_CLASSES, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=params["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_history(history: dict[str, list[float]], path: Path) -> None:
    keys = list(history)
    n_epochs = len(history[keys[0]])
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", *keys])
        for i in range(n_epochs):
            writer.writerow([i + 1, *(history[k][i] for k in keys)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
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
    print(f"saved model and history to {MODELS_DIR}")


if __name__ == "__main__":
    main()
