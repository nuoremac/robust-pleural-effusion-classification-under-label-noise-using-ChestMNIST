# Robust pleural effusion classification under label noise on ChestMNIST

TP4, *Robust Classification and Uncertainty Modeling*. Computer Engineering
Department, National Advanced School of Engineering, University of Yaoundé I.

The clinical target is **pleural effusion** on chest radiographs, read from
column 2 of the ChestMNIST label matrix. The question the study asks is not only
which classical classifier scores best, but how much each of them loses when a
fraction of the training annotations is wrong, and whether tightening the
regularisation of a model buys back any of that loss.

---

## Contribution covered by this branch

This branch contains **Phase III: evaluation, robustness to label noise and
experiment tracking**, assigned to **FOMETHE SOBMBANANG MAXIMILIEN** after the
redistribution of the Phase III workload. Concretely:

| Lab task | What is delivered | Where |
| --- | --- | --- |
| 3.1 | One evaluation layer for every model and every condition: accuracy, precision, recall, F1, ROC AUC, average precision, MCC and specificity, with `decision_function` used for the margin based models | [`src/evaluate.py`](src/evaluate.py) |
| 3.1 | Baseline sweep over the seven families, hyperparameter choice and top three selection on validation average precision, plus the precision-recall and ROC curves | [`src/run_baselines.py`](src/run_baselines.py), [`src/make_figures.py`](src/make_figures.py) |
| 3.2 | Corruption of 5 percent of the minority class training labels with persisted indices, and the full 2x2 design (clean or noisy labels, standard or robust configuration) over three seeds | [`src/noise.py`](src/noise.py), [`src/run_robustness.py`](src/run_robustness.py) |
| 3.3 | MLflow experiment on a local SQLite backend: parameters, seed, noise level, metrics, candidate traces, flipped indices and the fitted estimators as artifacts | [`src/tracking.py`](src/tracking.py) |
| Writing | Experimental protocol, results and discussion sections of the eight page article, with every table generated from the result files | [`paper/`](paper), [`src/export_tables.py`](src/export_tables.py) |

Phase I and Phase II belong to Nkolo Atangana and Phase IV to Ossombe Raoul. The
repository was empty when this branch was started, so [`src/data.py`](src/data.py)
and [`src/models.py`](src/models.py) implement the *common protocol* that the
team fixed, in order to make Phase III runnable end to end. They are the
interface Phase I and Phase II are expected to replace or extend, not a claim on
those phases: the decision boundary figures, the feature importance analysis and
the multiclass demonstration asked of Phase II are not part of this branch.

---

## Common protocol

Everything below is fixed in [`src/config.py`](src/config.py) and read by every
script, so the three contributors report comparable numbers.

- **Data.** ChestMNIST (MedMNIST v2, derived from NIH ChestX-ray14), 28x28
  grayscale, 14 binary findings per image. Target: `labels[:, 2]`, effusion.
- **Splits.** The official MedMNIST partition is kept: 78 468 train, 11 219
  validation, 22 433 test. Only the training split is subsampled, to 10 000
  images by stratified sampling with seed 42; the retained indices are saved to
  `results/train_indices_seed42.npy`.
- **Representation.** Images are flattened to 784 intensities in [0, 1] and
  standardized per feature. The scaler, and the 50 component PCA basis used in
  the representation check, are fitted on the training subsample only.
- **Metrics.** Accuracy, precision, recall, F1, ROC AUC and average precision.
  Model selection always uses **validation average precision**; the test split
  is read once per configuration, after the choice.
- **Label noise.** 5 percent of the minority class training labels are flipped.
  The flipped row indices are drawn once per seed and reused by every model
  (`results/flipped_indices_seed*.npy`). Validation and test labels are never
  touched.
- **Seeds.** 42 (primary), 7 and 2024. The seed controls the training
  subsample, the corrupted indices and the estimators' internal randomness.

---

## Getting the data

The six arrays are not committed. Place them in `data/`:

```
data/train_images.npy   data/train_labels.npy
data/val_images.npy     data/val_labels.npy
data/test_images.npy    data/test_labels.npy
```

They are the contents of `chestmnist.npz` as distributed by MedMNIST v2
(<https://medmnist.com>). From the archive:

```bash
python - <<'PY'
import numpy as np
bundle = np.load("chestmnist.npz")
for key in bundle.files:
    np.save(f"data/{key}.npy", bundle[key])
PY
```

---

## Reproducing the results

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python tests/test_phase3.py              # 12 invariant checks, a few seconds
python src/run_baselines.py              # Task 3.1, about 20 min on 4 CPU cores
python src/run_robustness.py             # Task 3.2 and 3.3, about 70 min
python src/make_figures.py               # figures 1 to 5
python src/export_tables.py              # LaTeX fragments for the article
```

`run_robustness.py` appends each completed cell to
`results/runs_robustness.jsonl` and skips cells already present, so an
interrupted sweep resumes where it stopped. `--tables-only` rebuilds the CSV
tables from that log without refitting anything. Both run scripts accept
`--no-mlflow` if the tracking server is not wanted.

Browse the tracked runs with:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

---

## Layout

```
src/config.py            protocol constants shared by the three phases
src/data.py              loading, stratified subsampling, scaling, PCA
src/models.py            seven families, standard grids, robust grids, SGA logistic regression
src/noise.py             minority class label flipping, with an auditable record
src/evaluate.py          the single evaluation layer: scores, metrics, curves, operating points
src/tracking.py          MLflow wrapper, degrades to a no-op when MLflow is absent
src/run_baselines.py     Task 3.1
src/run_robustness.py    Tasks 3.2 and 3.3
src/make_figures.py      figures 1 to 5
src/export_tables.py     LaTeX tables and the consolidated results.csv
tests/test_phase3.py     invariants the reported numbers rely on
results/                 CSV tables, JSON records, figures, LaTeX fragments
paper/                   the Phase III sections of the article
```

`tests/test_phase3.py` is not a test of scikit-learn. It checks the three
properties a reader cannot verify from a table: that the metrics are computed on
the intended vectors (including the degenerate case of a model that predicts no
positive at all), that the corruption flips exactly the requested number of
minority training labels and leaves the clean vector intact, and that the scaler
is fitted on the training subsample alone. It also checks that every robust
candidate really constrains capacity more than the standard one it replaces,
without which the 2x2 design would mean nothing.

## Reported numbers and their limits

Effusion prevalence is 12.3 percent on the test split, so a classifier that
always answers "no effusion" reaches 0.877 accuracy. Accuracy is reported for
completeness only; the conclusions of Phase III rest on average precision, whose
no-skill level is the prevalence itself. The absolute scores are those of
classical classifiers on 28x28 thumbnails of radiographs, far below what a
convolutional network reaches on full resolution images, and the label
corruption is simulated rather than observed. Both limits are stated in the
article.
