"""Evaluate the trained model on the test set.

Reads models/model.h5 and data/processed/test.npz, then writes
metrics.json (project root) and models/confusion_matrix.png.
"""

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from matplotlib.figure import Figure
from sklearn.metrics import ConfusionMatrixDisplay

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
METRICS_PATH = ROOT / "metrics.json"

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                          path: Path) -> None:
    """Plot the confusion matrix and save it as an image."""
    # Figure (not pyplot) needs no GUI backend, so this runs headless.
    fig = Figure(figsize=(9, 9))
    ax = fig.subplots()
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=CLASS_NAMES, ax=ax,
        cmap="Blues", xticks_rotation=45, colorbar=False,
    )
    ax.set_title("Fashion-MNIST test set confusion matrix")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main() -> None:
    """Compute test loss and accuracy, then write metrics and plot."""
    with np.load(ROOT / "data" / "processed" / "test.npz") as data:
        x_test, y_test = data["images"], data["labels"]
    model = tf.keras.models.load_model(MODELS_DIR / "model.h5")

    loss, accuracy = model.evaluate(x_test, y_test, verbose=2)
    y_pred = model.predict(x_test, verbose=0).argmax(axis=1)
    save_confusion_matrix(y_test, y_pred, MODELS_DIR / "confusion_matrix.png")

    metrics = {
        "test_loss": float(loss),
        "test_accuracy": float(accuracy),
        "num_test_samples": len(y_test),
    }
    text = json.dumps(metrics, indent=2)
    METRICS_PATH.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
