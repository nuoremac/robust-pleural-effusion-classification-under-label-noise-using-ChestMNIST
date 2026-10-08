"""Model zoo for the effusion task (Phase II interface) plus the robustness knobs.

Seven families are declared, each with the small hyperparameter grid fixed in the
common protocol. On top of that, every family exposes a *robust* grid whose
candidates are strictly more constrained than the standard pick: stronger L2 for
the logistic model, a smaller margin penalty C for the support vector machines,
shallower trees with larger leaves for the tree based models, a wider
neighbourhood for KNN and heavier variance smoothing for Naive Bayes. Phase III
uses the standard grid for Table 2 and the robust grid for the 2x2 design of
Table 3.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_is_fitted


# --------------------------------------------------------------------------- SGA
def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Logistic function, evaluated without overflow for large negative z."""
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


class LogisticRegressionSGA(BaseEstimator, ClassifierMixin):
    """Logistic regression trained by stochastic gradient ascent on the penalised
    conditional log-likelihood.

    The update follows the template given in the lab sheet, with three additions
    that the model needs to be usable as a baseline on 784 standardized features:
    the training set is reshuffled at every epoch, the step size decays as
    eta_0 / (1 + decay * epoch), and an L2 term is subtracted from the gradient.
    The L2 strength is the knob used for the robust variant of Task 3.2.

    For a single example (x, y) the ascent direction of the penalised
    log-likelihood is (y - sigma(w.x + b)) x - lambda w, which is what the inner
    loop applies.
    """

    def __init__(
        self,
        learning_rate: float = 0.05,
        epochs: int = 20,
        l2: float = 1e-4,
        decay: float = 0.5,
        shuffle: bool = True,
        random_state: int | None = None,
    ) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2 = l2
        self.decay = decay
        self.shuffle = shuffle
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionSGA":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).ravel()
        self.classes_ = np.unique(y)
        if self.classes_.size != 2:
            raise ValueError("LogisticRegressionSGA handles binary targets only")
        y01 = (y == self.classes_[1]).astype(np.float64)

        n_samples, n_features = X.shape
        rng = np.random.default_rng(self.random_state)
        w = np.zeros(n_features, dtype=np.float64)
        b = 0.0
        self.loglik_path_: list[float] = []

        for epoch in range(self.epochs):
            eta = self.learning_rate / (1.0 + self.decay * epoch)
            order = rng.permutation(n_samples) if self.shuffle else np.arange(n_samples)
            for i in order:
                xi = X[i]
                z = float(xi @ w) + b
                if z >= 0.0:
                    h = 1.0 / (1.0 + np.exp(-z))
                else:
                    ez = np.exp(z)
                    h = ez / (1.0 + ez)
                error = y01[i] - h
                w += eta * (error * xi - self.l2 * w)
                b += eta * error
            self.loglik_path_.append(self._mean_loglik(X, y01, w, b))

        self.coef_ = w.reshape(1, -1)
        self.intercept_ = np.array([b])
        self.n_features_in_ = n_features
        self.n_iter_ = self.epochs
        return self

    @staticmethod
    def _mean_loglik(X: np.ndarray, y01: np.ndarray, w: np.ndarray, b: float) -> float:
        z = X @ w + b
        return float(np.mean(y01 * z - np.logaddexp(0.0, z)))

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "coef_")
        return np.asarray(X, dtype=np.float64) @ self.coef_.ravel() + self.intercept_[0]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p1 = _sigmoid(self.decision_function(X))
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        idx = (self.decision_function(X) >= 0.0).astype(int)
        return self.classes_[idx]


# ------------------------------------------------------------------- model zoo
def _logreg_sga(seed: int, **params: Any) -> LogisticRegressionSGA:
    return LogisticRegressionSGA(random_state=seed, **params)


def _gaussian_nb(seed: int, **params: Any) -> GaussianNB:
    return GaussianNB(**params)                       # deterministic, seed unused


def _knn(seed: int, **params: Any) -> KNeighborsClassifier:
    return KNeighborsClassifier(n_jobs=-1, **params)  # deterministic, seed unused


def _decision_tree(seed: int, **params: Any) -> DecisionTreeClassifier:
    return DecisionTreeClassifier(random_state=seed, **params)


def _boosting(seed: int, **params: Any) -> HistGradientBoostingClassifier:
    defaults: dict[str, Any] = dict(max_iter=200, learning_rate=0.1, early_stopping=False)
    defaults.update(params)
    return HistGradientBoostingClassifier(random_state=seed, **defaults)


def _svm_linear(seed: int, **params: Any) -> LinearSVC:
    defaults: dict[str, Any] = dict(dual="auto", max_iter=5000)
    defaults.update(params)
    return LinearSVC(random_state=seed, **defaults)


def _svm_rbf(seed: int, **params: Any) -> SVC:
    defaults: dict[str, Any] = dict(kernel="rbf", gamma="scale", cache_size=1000)
    defaults.update(params)
    return SVC(random_state=seed, **defaults)


FACTORIES: dict[str, Callable[..., Any]] = {
    "logreg_sga": _logreg_sga,
    "gaussian_nb": _gaussian_nb,
    "knn": _knn,
    "decision_tree": _decision_tree,
    "boosting": _boosting,
    "svm_linear": _svm_linear,
    "svm_rbf": _svm_rbf,
}

DISPLAY_NAMES: dict[str, str] = {
    "logreg_sga": "Logistic regression (SGA)",
    "gaussian_nb": "Gaussian Naive Bayes",
    "knn": "K-nearest neighbours",
    "decision_tree": "Decision tree (CART)",
    "boosting": "Histogram gradient boosting",
    "svm_linear": "Linear SVM (soft margin)",
    "svm_rbf": "Kernel SVM (RBF)",
}

# Score semantics, needed by the evaluation module: models exposing a calibrated
# posterior are ranked with predict_proba, the two margin based models with
# their signed distance to the separating hyperplane.
SCORE_SOURCES: dict[str, str] = {
    "logreg_sga": "predict_proba",
    "gaussian_nb": "predict_proba",
    "knn": "predict_proba",
    "decision_tree": "predict_proba",
    "boosting": "predict_proba",
    "svm_linear": "decision_function",
    "svm_rbf": "decision_function",
}

# Standard grids. The protocol caps the search at three candidates per family;
# Naive Bayes and boosting are run in a single fixed configuration.
STANDARD_GRIDS: dict[str, list[dict[str, Any]]] = {
    "logreg_sga": [{"l2": 1e-5}, {"l2": 1e-4}, {"l2": 1e-3}],
    "gaussian_nb": [{"var_smoothing": 1e-9}],
    "knn": [{"n_neighbors": 3}, {"n_neighbors": 5}, {"n_neighbors": 11}],
    "decision_tree": [{"max_depth": 3}, {"max_depth": 5}, {"max_depth": 10}],
    "boosting": [{"max_leaf_nodes": 31}],
    "svm_linear": [{"C": 0.1}, {"C": 1.0}, {"C": 10.0}],
    "svm_rbf": [{"C": 0.1}, {"C": 1.0}, {"C": 10.0}],
}

# Robust grids. Every candidate constrains the hypothesis space more than the
# standard configuration it replaces; which candidate is retained is decided on
# validation average precision, never on test.
ROBUST_GRIDS: dict[str, list[dict[str, Any]]] = {
    "logreg_sga": [{"l2": 1e-2}, {"l2": 1e-1}, {"l2": 1.0}],
    "gaussian_nb": [{"var_smoothing": 1e-3}, {"var_smoothing": 1e-1}],
    "knn": [{"n_neighbors": 25}, {"n_neighbors": 51}, {"n_neighbors": 101}],
    "decision_tree": [
        {"max_depth": 3, "min_samples_leaf": 50},
        {"max_depth": 4, "min_samples_leaf": 100},
        {"max_depth": 2, "min_samples_leaf": 200},
    ],
    "boosting": [
        {"max_depth": 3, "max_leaf_nodes": 8, "min_samples_leaf": 50, "l2_regularization": 1.0},
        {"max_depth": 2, "max_leaf_nodes": 4, "min_samples_leaf": 100, "l2_regularization": 10.0},
    ],
    "svm_linear": [{"C": 0.01}, {"C": 0.001}],
    "svm_rbf": [{"C": 0.01}, {"C": 0.001}],
}


def build_model(family: str, params: dict[str, Any], seed: int):
    """Instantiate one model of ``family`` with ``params`` and the given seed."""
    if family not in FACTORIES:
        raise KeyError(f"unknown model family {family!r}")
    return FACTORIES[family](seed, **params)


def params_label(params: dict[str, Any]) -> str:
    """Compact, stable textual key for a hyperparameter dictionary."""
    if not params:
        return "default"
    return ", ".join(f"{k}={v}" for k, v in sorted(params.items()))
