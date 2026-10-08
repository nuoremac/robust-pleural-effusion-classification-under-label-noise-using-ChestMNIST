"""Common evaluation layer for Phase III (Task 3.1).

Every number that appears in the article passes through this module, which is
the point: seven model families, four experimental conditions and three seeds
cannot be compared unless the metrics are computed by one piece of code with one
set of conventions. Those conventions are:

Thresholded metrics.
    Accuracy, precision, recall and F1 are read off the confusion matrix
    produced by the estimator's own ``predict``, that is at the operating point
    each family uses by default (posterior 0.5 for the probabilistic models,
    signed margin 0 for the support vector machines). No threshold is tuned on
    the test split.

Ranking metrics.
    ROC AUC and average precision are threshold free and are computed from a
    continuous score. The score is the positive class posterior when the
    estimator exposes ``predict_proba``, and the signed distance to the
    separating hyperplane for ``LinearSVC`` and ``SVC``, which expose only
    ``decision_function``. Both quantities induce the same ordering semantics
    (larger means more likely positive), which is all these metrics require, so
    the two families remain comparable without fitting Platt scaling on top of
    the SVMs.

Average precision as the selection metric.
    With an effusion prevalence of about 0.12, a classifier that never predicts
    the positive class already reaches 0.88 accuracy, and the large negative
    pool makes the false positive rate of the ROC curve insensitive to the
    errors that matter clinically. Average precision summarises the
    precision-recall curve, whose no-skill level is the prevalence itself, so it
    moves when the minority class is handled better or worse. It is therefore
    used for every model selection decision in Phase III.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

METRIC_KEYS = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "average_precision",
    "mcc",
    "specificity",
)

METRIC_LABELS = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "roc_auc": "ROC AUC",
    "average_precision": "AP",
    "mcc": "MCC",
    "specificity": "Specificity",
}


# ------------------------------------------------------------------ score access
def model_scores(model, X: np.ndarray) -> tuple[np.ndarray, str]:
    """Continuous positive-class score for ``X`` and the name of its source.

    ``predict_proba`` is preferred because it is bounded and interpretable; the
    signed margin is the fallback for the support vector machines. A model
    exposing neither cannot be ranked and is rejected rather than silently
    scored with its hard predictions, which would collapse the ROC and
    precision-recall curves to a single point.
    """
    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(X))
        classes = np.asarray(getattr(model, "classes_", [0, 1]))
        positive_column = int(np.flatnonzero(classes == classes.max())[0])
        return proba[:, positive_column], "predict_proba"
    if hasattr(model, "decision_function"):
        margin = np.asarray(model.decision_function(X)).ravel()
        return margin, "decision_function"
    raise TypeError(
        f"{type(model).__name__} exposes neither predict_proba nor decision_function, "
        "so ranking metrics cannot be computed for it"
    )


# ---------------------------------------------------------------------- metrics
def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray
) -> dict[str, float]:
    """The eight scalar metrics reported for every run.

    ``zero_division=0`` is explicit: a heavily regularised model trained on
    corrupted labels can stop predicting the positive class altogether, and that
    degenerate behaviour must be reported as precision 0 rather than crash the
    experiment loop.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    y_score = np.asarray(y_score, dtype=np.float64).ravel()

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "average_precision": float(average_precision_score(y_true, y_score)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def metrics_at_threshold(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float
) -> dict[str, float]:
    """Metrics obtained by cutting the score vector at ``threshold``."""
    y_pred = (np.asarray(y_score, dtype=np.float64).ravel() >= threshold).astype(int)
    return compute_metrics(y_true, y_pred, y_score)


def best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    """Threshold maximising F1 on ``(y_true, y_score)``, and the F1 it reaches.

    This is used only on the validation split. The default cut-off of each
    family (0.5 on a posterior, 0 on a margin) is calibrated for balanced data
    and pushes several models to predict no positive case at all on a 12 percent
    prevalence task, which makes their F1 uninformative about ranking quality.
    Selecting the operating point on validation and reporting the resulting test
    F1 separates two questions that the default threshold conflates: how well a
    model orders patients, and where its decision boundary happens to sit.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    # precision_recall_curve returns one more point than thresholds; the last
    # point is the degenerate (recall 0, precision 1) corner and has no threshold.
    precision, recall = precision[:-1], recall[:-1]
    denominator = precision + recall
    f1 = np.where(denominator > 0, 2 * precision * recall / np.maximum(denominator, 1e-12), 0.0)
    best = int(np.argmax(f1))
    return float(thresholds[best]), float(f1[best])


@dataclass
class EvalResult:
    """Metrics plus the raw vectors the figures need, for one model on one split."""

    split: str
    metrics: dict[str, float]
    y_true: np.ndarray
    y_pred: np.ndarray
    y_score: np.ndarray
    score_source: str
    fit_seconds: float = 0.0
    predict_seconds: float = 0.0
    extra: dict = field(default_factory=dict)

    @property
    def prevalence(self) -> float:
        """No-skill average precision of the split, the floor of its PR curve."""
        return float(np.mean(self.y_true))

    def pr_curve(self) -> tuple[np.ndarray, np.ndarray]:
        precision, recall, _ = precision_recall_curve(self.y_true, self.y_score)
        return recall, precision

    def roc_curve(self) -> tuple[np.ndarray, np.ndarray]:
        fpr, tpr, _ = roc_curve(self.y_true, self.y_score)
        return fpr, tpr

    def flat(self, prefix: str | None = None) -> dict[str, float]:
        """Metrics as a flat dictionary, optionally prefixed by the split name."""
        tag = f"{prefix or self.split}_"
        return {f"{tag}{k}": v for k, v in self.metrics.items()}


def evaluate_on_split(
    model, X: np.ndarray, y: np.ndarray, split: str, predict_seconds: float | None = None
) -> EvalResult:
    """Run ``model`` on one split and return its metrics and score vectors."""
    import time

    start = time.perf_counter()
    y_pred = np.asarray(model.predict(X)).ravel()
    y_score, source = model_scores(model, X)
    elapsed = time.perf_counter() - start

    return EvalResult(
        split=split,
        metrics=compute_metrics(y, y_pred, y_score),
        y_true=np.asarray(y).ravel(),
        y_pred=y_pred,
        y_score=y_score,
        score_source=source,
        predict_seconds=float(predict_seconds if predict_seconds is not None else elapsed),
    )


def fit_and_evaluate(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    splits: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[object, dict[str, EvalResult]]:
    """Fit once, then evaluate on each named split.

    ``splits`` maps a split name to its ``(X, y)`` pair. The training labels
    passed here may be the corrupted ones; the labels inside ``splits`` never
    are, which is how the clean and noisy conditions of Task 3.2 stay separated.
    """
    import time

    start = time.perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - start

    results: dict[str, EvalResult] = {}
    for name, (X, y) in splits.items():
        result = evaluate_on_split(model, X, y, name)
        result.fit_seconds = fit_seconds
        results[name] = result
    return model, results
