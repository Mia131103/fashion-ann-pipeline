# fashion-ann-pipeline

The goal of this assignment is to build a fully-connected Artificial Neural Network (ANN — not a CNN) that classifies a Fashion-MNIST image into one of the 10 categories. The target is ≥ 85% test accuracy. The pipeline must be reproducible end-to-end via a single dvc repro call, and every intermediate artifact (raw data, processed data, trained model, metrics) must be versioned through DVC and pushed to a Google Drive remote.

The assignment report is [`report/REPORT.pdf`](report/REPORT.pdf) (source: [`report/REPORT.md`](report/REPORT.md)). The full, uncropped evidence screenshots are in [`screenshots/`](screenshots/), named by assignment part.

## Results

| Version | `train.hidden_units` | Test accuracy | Test loss |
|---|---|---|---|
| tag `v1` | 128 | 0.8774 | 0.3467 |
| tag `v2` | 256 | 0.8814 | 0.3404 |
| `main` (after the Part E merge; standardized inputs) | 256 | 0.8763 | 0.3642 |

`dvc metrics diff v1 v2` and `dvc params diff v1 v2` reproduce the first two rows.

## Pipeline

| Stage | Script | Output |
|---|---|---|
| prepare | `src/prepare.py` | `data/raw/{train,test}.npz` |
| preprocess | `src/preprocess.py` | `data/processed/{train,val,test}.npz` |
| train | `src/train.py` | `models/model.h5`, `models/history.csv` |
| evaluate | `src/evaluate.py` | `metrics.json`, `models/confusion_matrix.png` |

Hyperparameters live in `params.yaml`, and stages, dependencies and outputs in `dvc.yaml`. `dvc.lock` records the hashes of the current data and model, which are stored in the Google Drive remote (`gdriveremote`).

## Branches and tags

- `main`: final state, including the merge of the simulated teammate's branch (Part E)
- `dev`: feature work for the scripts, DVC setup and pipeline (Parts B–D)
- `teammate-sim`: the simulated collaborator's conflicting normalization change (Part E)
- `v1`, `v2`: the pipeline before and after the `hidden_units` change (Part D)

## Reproduce

Requires Python 3.14.

```bash
uv sync                   # or: pip install tensorflow "dvc[gdrive]" pyyaml scikit-learn matplotlib "cryptography<44"
source .venv/bin/activate

# Your own Google OAuth client; stored in .dvc/config.local, which Git ignores
dvc remote modify --local gdriveremote gdrive_client_id <client-id>
dvc remote modify --local gdriveremote gdrive_client_secret <client-secret>

dvc pull                  # fetch data and model (the Drive folder must be shared with you)
dvc repro                 # reports "up to date" after a pull; without one, rebuilds every stage
dvc metrics show
```
