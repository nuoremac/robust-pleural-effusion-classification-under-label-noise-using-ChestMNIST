"""Task 3.2: robustness of the three selected models to uncertain labels.

Design. For each of the three families retained by ``run_baselines.py``, four
runs are executed, crossing the training labels with the capacity of the
hypothesis class:

                       standard configuration      robust configuration
  clean labels         fit and evaluate            fit and evaluate
  5 percent of the
  minority class
  flipped              fit and evaluate            fit and evaluate

The clean / standard cell reproduces the Table 2 baseline, which is what makes
the three other cells readable: the clean / robust cell measures the price of
extra regularisation when nothing is wrong with the labels, the
noisy / standard cell measures the damage done by the corruption, and the
noisy / robust cell says whether constraining the model recovers part of that
damage. Reporting the noisy / robust cell alone would not separate a genuine
robustness gain from a configuration that happens to be better everywhere.

Three rules keep the comparison honest:

* the corrupted indices are drawn once per seed and shared by the three
  families, so the gaps between families cannot come from different corruptions;
* validation and test labels are never corrupted, so every metric is measured
  against the reference annotation;
* the robust hyperparameter is chosen once, on the validation average precision
  of the clean primary-seed run, and is then frozen for the corrupted condition
  and for the replication seeds. Reselecting it under corruption would be
  defensible as a description of what a practitioner does, but it would make the
  difference between the two cells of a row mix a change of labels with a change
  of configuration, and the question asked here is whether one fixed, more
  constrained model degrades less than one fixed, less constrained one.

Outputs: ``results/results_robustness.csv``, ``results/results_deltas.csv``,
``results/robust_selection.json``, score vectors under ``results/scores/``,
flipped indices under ``results/``, and one MLflow run per cell.

Usage: python src/run_robustness.py [--seeds 42 7 2024] [--no-mlflow]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import data
import evaluate
import models
import noise
import tracking
from config import (
    NOISE_RATE, N_SELECTED_MODELS, PRIMARY_SEED, RESULTS_DIR, SEEDS,
    SELECTION_METRIC, TRAIN_SUBSET_SIZE,
)

SCORES_DIR = RESULTS_DIR / "scores"
RUNS_JSONL = RESULTS_DIR / "runs_robustness.jsonl"
VAL_METRIC = f"val_{SELECTION_METRIC}"


# --------------------------------------------------------------------- helpers
def load_selection() -> dict:
    path = RESULTS_DIR / "selection.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Run `python src/run_baselines.py` first: Task 3.2 "
            "re-trains the models that Task 3.1 selected."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def read_done_keys() -> set[tuple]:
    """Keys of the runs already stored, so an interrupted sweep can be resumed."""
    if not RUNS_JSONL.exists():
        return set()
    done = set()
    for line in RUNS_JSONL.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        done.add((row["seed"], row["family"], row["label_condition"], row["configuration"]))
    return done


def append_run(row: dict) -> None:
    with RUNS_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, default=float) + "\n")


def fit_candidates(
    family: str,
    grid: list[dict[str, Any]],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
) -> tuple[dict, Any, evaluate.EvalResult, list[dict]]:
    """Fit every candidate of ``grid`` and return the one with the best validation AP.

    The fitted estimators are kept in memory during the search so that the
    winner does not have to be refitted before its test evaluation. Only
    validation scores drive the choice.
    """
    trace: list[dict] = []
    best: tuple[float, dict, Any, evaluate.EvalResult] | None = None

    for params in grid:
        model = models.build_model(family, params, seed)
        started = time.perf_counter()
        model.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - started
        result = evaluate.evaluate_on_split(model, X_val, y_val, "val")
        result.fit_seconds = fit_seconds
        ap = result.metrics[SELECTION_METRIC]
        trace.append(
            {
                "params": models.params_label(params),
                VAL_METRIC: ap,
                "val_roc_auc": result.metrics["roc_auc"],
                "fit_seconds": round(fit_seconds, 2),
            }
        )
        print(
            f"      candidate {models.params_label(params):40s} "
            f"val AP={ap:.4f} ({fit_seconds:.1f}s fit)"
        )
        if best is None or ap > best[0]:
            best = (ap, params, model, result)

    assert best is not None
    return best[1], best[2], best[3], trace


def run_cell(
    family: str,
    configuration: str,
    label_condition: str,
    grid: list[dict[str, Any]],
    dataset: data.Dataset,
    y_train: np.ndarray,
    seed: int,
    noise_record: noise.NoiseRecord | None,
    use_mlflow: bool,
) -> dict:
    """Execute one cell of the 2x2 design and return its CSV row."""
    print(f"    {family} / {label_condition} / {configuration}")
    params, model, val_result, trace = fit_candidates(
        family, grid, dataset.X_train, y_train, dataset.X_val, dataset.y_val, seed
    )
    test_result = evaluate.evaluate_on_split(model, dataset.X_test, dataset.y_test, "test")
    test_result.fit_seconds = val_result.fit_seconds

    threshold, val_f1_tuned = evaluate.best_f1_threshold(val_result.y_true, val_result.y_score)
    tuned_test = evaluate.metrics_at_threshold(test_result.y_true, test_result.y_score, threshold)

    label = models.params_label(params)
    row: dict = {
        "seed": seed,
        "family": family,
        "model": models.DISPLAY_NAMES[family],
        "label_condition": label_condition,
        "configuration": configuration,
        "params": label,
        "score_source": val_result.score_source,
        "noise_rate": NOISE_RATE if label_condition == "noisy" else 0.0,
        "n_flipped": int(noise_record.flipped_indices.size) if noise_record else 0,
        "train_prevalence": float(np.mean(y_train)),
        "fit_seconds": round(val_result.fit_seconds, 3),
        "val_selected_threshold": round(float(threshold), 6),
        "val_f1_at_val_threshold": val_f1_tuned,
        "test_f1_at_val_threshold": tuned_test["f1"],
        "test_precision_at_val_threshold": tuned_test["precision"],
        "test_recall_at_val_threshold": tuned_test["recall"],
        "candidate_trace": json.dumps(trace),
    }
    row.update(val_result.flat("val"))
    row.update(test_result.flat("test"))

    print(
        f"      retained {label:40s} val AP={row[VAL_METRIC]:.4f} "
        f"test AP={row['test_average_precision']:.4f} "
        f"test AUC={row['test_roc_auc']:.4f} test F1*={row['test_f1_at_val_threshold']:.4f}"
    )

    np.savez_compressed(
        SCORES_DIR / f"robust_{family}_{label_condition}_{configuration}_seed{seed}.npz",
        val_score=val_result.y_score,
        val_true=val_result.y_true,
        test_score=test_result.y_score,
        test_true=test_result.y_true,
    )

    if use_mlflow:
        with tracking.run(
            f"robust-{family}-{label_condition}-{configuration}-seed{seed}",
            tags={
                "phase": "III",
                "task": "3.2",
                "family": family,
                "label_condition": label_condition,
                "configuration": configuration,
            },
        ):
            tracking.log_params(
                {
                    "family": family,
                    **params,
                    "seed": seed,
                    "feature_space": dataset.feature_space,
                    "n_features": dataset.n_features,
                    "train_subset_size": TRAIN_SUBSET_SIZE,
                    "label_condition": label_condition,
                    "configuration": configuration,
                    "score_source": val_result.score_source,
                    "selection_metric": SELECTION_METRIC,
                    **(noise_record.as_params() if noise_record else {"noise_rate": 0.0}),
                }
            )
            tracking.log_metrics(
                {
                    **val_result.flat("val"),
                    **test_result.flat("test"),
                    "fit_seconds": val_result.fit_seconds,
                    "val_selected_threshold": float(threshold),
                    "test_f1_at_val_threshold": tuned_test["f1"],
                }
            )
            tracking.log_json({"candidates": trace}, f"candidate_trace_{family}.json")
            if noise_record is not None:
                tracking.log_json(
                    {"flipped_indices": noise_record.flipped_indices.tolist()},
                    f"flipped_indices_seed{seed}.json",
                )
            # The estimator itself is logged for the primary seed only. A fitted
            # RBF machine or a KNN index on 10 000 by 784 features weighs tens of
            # megabytes, and the replication seeds exist to quantify variance,
            # not to be deployed; their metrics and parameters are enough to
            # rebuild them from the scripts.
            if seed == PRIMARY_SEED:
                tracking.log_model(model)

    return row


# ------------------------------------------------------------------------ sweep
def run(seeds: tuple[int, ...] = SEEDS, use_mlflow: bool = True) -> pd.DataFrame:
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    selection = load_selection()
    families = selection["selected_families"][:N_SELECTED_MODELS]
    standard_params = {f: selection["standard_params"][f] for f in families}
    print(f"Selected families: {families}")
    print(f"Standard configurations: {standard_params}")

    if use_mlflow and not tracking.init_tracking():
        print("MLflow is not installed: metrics are written to CSV only.")

    done = read_done_keys()
    if done:
        print(f"Resuming: {len(done)} run(s) already recorded in {RUNS_JSONL.name}")

    robust_choice_path = RESULTS_DIR / "robust_selection.json"
    robust_choice: dict = (
        json.loads(robust_choice_path.read_text(encoding="utf-8"))
        if robust_choice_path.exists() else {}
    )

    for seed in seeds:
        print(f"\n=== seed {seed} ===")
        dataset = data.build_dataset(seed=seed, feature_space="standardized")
        y_clean = dataset.y_train
        y_noisy, record = noise.inject_label_noise(y_clean, seed=seed, rate=NOISE_RATE)
        np.save(RESULTS_DIR / f"flipped_indices_seed{seed}.npy", record.flipped_indices)
        print(f"  corruption: {record.as_params()}")

        conditions = {"clean": (y_clean, None), "noisy": (y_noisy, record)}

        for family in families:
            for label_condition, (y_train, rec) in conditions.items():
                for configuration in ("standard", "robust"):
                    key = (seed, family, label_condition, configuration)
                    if key in done:
                        print(f"    skip (already recorded): {key}")
                        continue

                    if configuration == "standard":
                        grid = [standard_params[family]]
                    elif family in robust_choice:
                        # Already decided: reuse it, so that the only thing that
                        # changes between the two label conditions is the labels.
                        grid = [robust_choice[family]["params_dict"]]
                    else:
                        grid = models.ROBUST_GRIDS[family]

                    row = run_cell(
                        family, configuration, label_condition, grid, dataset,
                        y_train, seed, rec, use_mlflow,
                    )
                    append_run(row)
                    done.add(key)

                    if configuration == "robust" and len(grid) > 1:
                        chosen = next(
                            p for p in models.ROBUST_GRIDS[family]
                            if models.params_label(p) == row["params"]
                        )
                        robust_choice[family] = {
                            "params_dict": chosen,
                            "params": row["params"],
                            "selected_on_seed": seed,
                            "selected_on_label_condition": label_condition,
                            VAL_METRIC: row[VAL_METRIC],
                            "candidates": json.loads(row["candidate_trace"]),
                        }
                        robust_choice_path.write_text(
                            json.dumps(robust_choice, indent=2), encoding="utf-8"
                        )

    frame = build_tables()
    return frame


# ----------------------------------------------------------------- aggregation
def build_tables() -> pd.DataFrame:
    """Turn the run log into the two tables used in the article."""
    rows = [
        json.loads(line)
        for line in RUNS_JSONL.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    frame = pd.DataFrame(rows)
    order = {"clean": 0, "noisy": 1}
    frame = frame.sort_values(
        ["family", "configuration", "label_condition", "seed"],
        key=lambda s: s.map(order).fillna(s) if s.name == "label_condition" else s,
    )
    frame.to_csv(RESULTS_DIR / "results_robustness.csv", index=False)

    metric_columns = [
        c for c in frame.columns
        if c.startswith(("val_", "test_")) and frame[c].dtype.kind in "fi"
    ]

    # Degradation caused by the corruption, per family, configuration and seed.
    pivot = frame.pivot_table(
        index=["family", "configuration", "seed"],
        columns="label_condition",
        values=metric_columns,
    )
    deltas = []
    for (family, configuration, seed), group in pivot.iterrows():
        record: dict = {"family": family, "configuration": configuration, "seed": seed}
        for metric in metric_columns:
            try:
                clean, noisy = group[(metric, "clean")], group[(metric, "noisy")]
            except KeyError:
                continue
            if pd.isna(clean) or pd.isna(noisy):
                continue
            record[f"{metric}_clean"] = float(clean)
            record[f"{metric}_noisy"] = float(noisy)
            record[f"{metric}_delta"] = float(noisy) - float(clean)
            record[f"{metric}_relative_delta"] = (
                (float(noisy) - float(clean)) / float(clean) if clean else float("nan")
            )
        # A group that is missing one of its two label conditions, which happens
        # while a sweep is still running, yields no difference at all and would
        # otherwise appear as a row of blanks in the table.
        if any(key.endswith("_delta") for key in record):
            deltas.append(record)

    delta_frame = pd.DataFrame(deltas)
    delta_frame.to_csv(RESULTS_DIR / "results_deltas.csv", index=False)

    print(f"\nWritten: {RESULTS_DIR / 'results_robustness.csv'}")
    print(f"Written: {RESULTS_DIR / 'results_deltas.csv'}")
    if not delta_frame.empty:
        columns = ["family", "configuration", "seed",
                   "test_average_precision_clean", "test_average_precision_noisy",
                   "test_average_precision_delta"]
        available = [c for c in columns if c in delta_frame.columns]
        print("\nTest AP degradation under 5 percent minority label noise")
        print(delta_frame[available].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("--tables-only", action="store_true",
                        help="rebuild the CSV tables from the existing run log")
    args = parser.parse_args()
    if args.tables_only:
        build_tables()
        return
    run(seeds=tuple(args.seeds), use_mlflow=not args.no_mlflow)


if __name__ == "__main__":
    main()
