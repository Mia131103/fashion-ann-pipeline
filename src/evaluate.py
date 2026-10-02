"""Evaluate the trained model on the processed Fashion-MNIST test set.

Reads models/model.h5 and data/processed/test.npz, then writes metrics.json (project root)
and models/confusion_matrix.png. The batch size comes from the `train` section of params.yaml:

    train:
      batch_size: 32

Run standalone:  python src/evaluate.py [--params params.yaml]
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend: only writes files, never opens a window
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import yaml
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "model.h5"
CONFUSION_MATRIX_PATH = MODELS_DIR / "confusion_matrix.png"
METRICS_PATH = ROOT / "metrics.json"

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]  # fmt: skip


def load_params(path: Path) -> dict:
    with open(path) as f:
        params = yaml.safe_load(f)
    try:
        return {"batch_size": int(params["train"]["batch_size"])}
    except (TypeError, KeyError) as e:
        raise SystemExit(f"{path}: missing 'train.batch_size' ({e!r})")


def load_test_set() -> tuple[np.ndarray, np.ndarray]:
    path = PROCESSED_DIR / "test.npz"
    if not path.exists():
        raise SystemExit(f"{path} not found; run src/preprocess.py first")
    with np.load(path) as data:
        return data["images"], data["labels"]


def load_model() -> tf.keras.Model:
    if not MODEL_PATH.exists():
        raise SystemExit(f"{MODEL_PATH} not found; run src/train.py first")
    return tf.keras.models.load_model(MODEL_PATH)


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, path: Path) -> np.ndarray:
    labels = list(range(len(CLASS_NAMES)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(9, 9))
    ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(
        ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False
    )
    ax.set_title("Fashion-MNIST test set confusion matrix")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return cm


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--params", type=Path, default=ROOT / "params.yaml")
    args = parser.parse_args()

    params = load_params(args.params)
    x_test, y_test = load_test_set()
    model = load_model()

    # compile=True restores the training loss/metrics, so evaluate() returns [loss, accuracy].
    loss, accuracy = model.evaluate(x_test, y_test, batch_size=params["batch_size"], verbose=2)

    y_pred = np.argmax(model.predict(x_test, batch_size=params["batch_size"], verbose=0), axis=1)
    save_confusion_matrix(y_test, y_pred, CONFUSION_MATRIX_PATH)

    metrics = {
        "test_loss": float(loss),
        "test_accuracy": float(accuracy),
        "num_test_samples": int(len(y_test)),
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
        f.write("\n")
    print(json.dumps(metrics, indent=2))
    print(f"saved {METRICS_PATH} and {CONFUSION_MATRIX_PATH}")


if __name__ == "__main__":
    main()
