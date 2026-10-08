"""Checks on the Phase III invariants.

These are not tests of scikit-learn. They test the three properties the reported
numbers rely on and that a reader cannot verify by looking at a table: that the
metrics are computed on the intended vectors, that the corruption touches only
the training labels of the minority class, and that nothing from validation or
test enters the representation.

Run with ``python tests/test_phase3.py`` (no test runner required) or with
``pytest tests/test_phase3.py``.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from sklearn.metrics import average_precision_score

import evaluate
import models
import noise


def test_metrics_match_a_hand_computed_confusion_matrix() -> None:
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 1, 1, 0, 1, 1, 1])
    y_score = np.array([0.1, 0.2, 0.6, 0.7, 0.3, 0.8, 0.9, 0.95])
    m = evaluate.compute_metrics(y_true, y_pred, y_score)
    # tp = 3, fp = 2, fn = 1, tn = 2
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (3, 2, 1, 2)
    assert np.isclose(m["accuracy"], 5 / 8)
    assert np.isclose(m["precision"], 3 / 5)
    assert np.isclose(m["recall"], 3 / 4)
    assert np.isclose(m["f1"], 2 * (3 / 5) * (3 / 4) / ((3 / 5) + (3 / 4)))
    assert np.isclose(m["specificity"], 2 / 4)


def test_degenerate_prediction_is_reported_not_raised() -> None:
    """A model that predicts no positive case must score precision 0, not crash."""
    y_true = np.array([0, 0, 0, 1])
    y_pred = np.zeros(4, dtype=int)
    y_score = np.array([0.1, 0.2, 0.3, 0.4])
    m = evaluate.compute_metrics(y_true, y_pred, y_score)
    assert m["precision"] == 0.0 and m["recall"] == 0.0 and m["f1"] == 0.0
    # The ranking metrics still see the perfect ordering of this toy example.
    assert np.isclose(m["average_precision"], 1.0)


def test_average_precision_is_invariant_to_a_monotone_rescaling_of_the_score() -> None:
    """Why a posterior and a signed margin can be compared on AP and AUC."""
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=400)
    margin = rng.normal(size=400) + y_true
    posterior = 1.0 / (1.0 + np.exp(-margin))          # strictly increasing map
    assert np.isclose(
        average_precision_score(y_true, margin), average_precision_score(y_true, posterior)
    )


def test_score_source_follows_the_estimator_interface() -> None:
    rng = np.random.default_rng(1)
    X = rng.normal(size=(120, 6))
    y = (X[:, 0] + 0.4 * rng.normal(size=120) > 0).astype(int)

    probabilistic = models.build_model("decision_tree", {"max_depth": 3}, 0).fit(X, y)
    _, source = evaluate.model_scores(probabilistic, X)
    assert source == "predict_proba"

    margin_based = models.build_model("svm_linear", {"C": 1.0}, 0).fit(X, y)
    scores, source = evaluate.model_scores(margin_based, X)
    assert source == "decision_function"
    # The margin is signed and unbounded, unlike a posterior.
    assert scores.min() < 0.0 < scores.max()


def test_best_f1_threshold_is_at_least_as_good_as_the_default() -> None:
    rng = np.random.default_rng(2)
    y_true = (rng.random(600) < 0.12).astype(int)      # the prevalence of the study
    y_score = np.clip(0.1 + 0.25 * y_true + 0.1 * rng.normal(size=600), 0, 1)
    threshold, f1 = evaluate.best_f1_threshold(y_true, y_score)
    default = evaluate.metrics_at_threshold(y_true, y_score, 0.5)
    assert f1 >= default["f1"]
    tuned = evaluate.metrics_at_threshold(y_true, y_score, threshold)
    assert np.isclose(tuned["f1"], f1)


def test_noise_flips_the_requested_number_of_minority_labels() -> None:
    y = np.zeros(1000, dtype=np.int8)
    y[:120] = 1                                        # minority class is 1
    y_noisy, record = noise.inject_label_noise(y, seed=42, rate=0.05)
    assert record.minority_label == 1
    assert record.n_minority == 120
    assert record.flipped_indices.size == 6            # floor(0.05 * 120)
    assert np.array_equal(np.flatnonzero(y != y_noisy), record.flipped_indices)
    assert np.all(y[record.flipped_indices] == 1)
    assert np.all(y_noisy[record.flipped_indices] == 0)
    assert y_noisy.sum() == 114


def test_noise_leaves_the_input_untouched_and_is_reproducible() -> None:
    y = np.concatenate([np.ones(200, dtype=np.int8), np.zeros(800, dtype=np.int8)])
    before = y.copy()
    first, record_a = noise.inject_label_noise(y, seed=7)
    second, record_b = noise.inject_label_noise(y, seed=7)
    assert np.array_equal(y, before)                   # the clean condition stays available
    assert np.array_equal(first, second)
    assert np.array_equal(record_a.flipped_indices, record_b.flipped_indices)
    third, record_c = noise.inject_label_noise(y, seed=2024)
    assert not np.array_equal(record_a.flipped_indices, record_c.flipped_indices)


def test_sga_recovers_a_separable_direction_and_increases_the_log_likelihood() -> None:
    rng = np.random.default_rng(3)
    X = rng.normal(size=(500, 4))
    y = (X @ np.array([2.0, -1.0, 0.0, 0.0]) > 0).astype(int)
    model = models.LogisticRegressionSGA(epochs=12, learning_rate=0.1, l2=1e-6,
                                         random_state=0).fit(X, y)
    assert model.predict(X).mean() > 0.3
    assert (model.predict(X) == y).mean() > 0.95
    path = model.loglik_path_
    assert path[-1] > path[0]                          # ascent, as the name says
    # The two informative coordinates must dominate the two noise coordinates.
    weights = np.abs(model.coef_.ravel())
    assert weights[:2].min() > weights[2:].max()


def test_stronger_l2_shrinks_the_weights() -> None:
    """The robustness knob of the logistic model has to act on the weights."""
    rng = np.random.default_rng(4)
    X = rng.normal(size=(400, 8))
    y = (X[:, 0] > 0).astype(int)
    weak = models.LogisticRegressionSGA(epochs=8, l2=1e-6, random_state=0).fit(X, y)
    strong = models.LogisticRegressionSGA(epochs=8, l2=1.0, random_state=0).fit(X, y)
    assert np.linalg.norm(strong.coef_) < np.linalg.norm(weak.coef_)


def test_robust_grids_are_strictly_more_constrained_than_the_standard_ones() -> None:
    """A robust candidate must reduce capacity, otherwise Table 3 means nothing."""
    assert min(c["l2"] for c in models.ROBUST_GRIDS["logreg_sga"]) > max(
        c["l2"] for c in models.STANDARD_GRIDS["logreg_sga"]
    )
    for family in ("svm_linear", "svm_rbf"):
        assert max(c["C"] for c in models.ROBUST_GRIDS[family]) < min(
            c["C"] for c in models.STANDARD_GRIDS[family]
        )
    assert min(c["n_neighbors"] for c in models.ROBUST_GRIDS["knn"]) > max(
        c["n_neighbors"] for c in models.STANDARD_GRIDS["knn"]
    )
    for candidate in models.ROBUST_GRIDS["decision_tree"]:
        assert candidate["max_depth"] <= 4 and candidate["min_samples_leaf"] >= 50


def test_pipeline_fits_the_representation_on_the_training_subsample_only() -> None:
    """Skipped when ``data/`` is empty; otherwise it checks the real pipeline.

    If the scaler were refitted per split, each split would come out exactly
    centred. The training subsample must be centred to machine precision and the
    two held out splits must not be, since they are only transformed.
    """
    import config
    import data as data_module

    if not (config.DATA_DIR / "train_images.npy").exists():
        print("      (skipped: data/ is empty)")
        return

    dataset = data_module.build_dataset(seed=config.PRIMARY_SEED, subset_size=2000)
    assert dataset.X_train.shape[1] == 784
    assert abs(float(dataset.X_train.mean())) < 1e-4
    assert abs(float(dataset.X_train.std()) - 1.0) < 1e-2
    assert abs(float(dataset.X_val.mean())) > 1e-3
    assert abs(float(dataset.X_test.mean())) > 1e-3
    # The held out splits keep their official size, only the training split is capped.
    assert dataset.X_val.shape[0] == 11219 and dataset.X_test.shape[0] == 22433
    assert dataset.train_indices.size == 2000
    # Stratification: the subsample reproduces the prevalence of the full split.
    assert abs(float(dataset.y_train.mean()) - 0.118) < 0.005


def test_subsampling_is_reproducible_across_calls_and_varies_with_the_seed() -> None:
    import config
    import data as data_module

    if not (config.DATA_DIR / "train_labels.npy").exists():
        print("      (skipped: data/ is empty)")
        return

    labels = np.load(config.DATA_DIR / "train_labels.npy")[:, config.TARGET_INDEX]
    first = data_module.stratified_subset_indices(labels, 5000, seed=42)
    again = data_module.stratified_subset_indices(labels, 5000, seed=42)
    other = data_module.stratified_subset_indices(labels, 5000, seed=7)
    assert np.array_equal(first, again)
    assert not np.array_equal(first, other)
    assert first.size == 5000 and np.unique(first).size == 5000


def main() -> int:
    functions = [
        value for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    failures = 0
    for function in functions:
        try:
            function()
        except AssertionError as error:
            failures += 1
            print(f"FAIL  {function.__name__}: {error}")
        except Exception as error:                     # noqa: BLE001
            failures += 1
            print(f"ERROR {function.__name__}: {type(error).__name__}: {error}")
        else:
            print(f"ok    {function.__name__}")
    print(f"\n{len(functions) - failures}/{len(functions)} checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
