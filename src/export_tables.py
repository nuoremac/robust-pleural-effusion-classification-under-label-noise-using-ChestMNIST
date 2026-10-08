"""Export the result tables of Phase III as LaTeX fragments.

The fragments are written to ``results/tables/`` and included verbatim by the
article, so a number can never be retyped by hand into the text. Every table
uses ``booktabs`` rules and the ``\\input`` target contains the tabular
environment only, which lets the article choose the float placement and the
caption.

Usage: python src/export_tables.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import models
from config import RESULTS_DIR, TABLES_DIR

HEADLINE = ("accuracy", "precision", "recall", "f1", "roc_auc", "average_precision")
HEADLINE_TEX = {
    "accuracy": "Acc.",
    "precision": "Prec.",
    "recall": "Rec.",
    "f1": "F1",
    "roc_auc": "AUC",
    "average_precision": "AP",
    "mcc": "MCC",
}


def _write(name: str, body: str) -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    path = TABLES_DIR / name
    path.write_text(body.rstrip() + "\n", encoding="utf-8")
    print(f"  wrote {path.name}")


def _escape(text: str) -> str:
    return str(text).replace("_", r"\_").replace("%", r"\%")


# Hyperparameter names as a paper prints them, not as scikit-learn spells them.
# Keeping the columns narrow is what lets the tables fit a two-column layout.
_MATH_SYMBOLS = {
    "l2": r"\lambda",
    "l2_regularization": r"\lambda",
    "C": "C",
    "n_neighbors": "K",
    "var_smoothing": r"\varepsilon",
}
_WORD_SYMBOLS = {
    "max_depth": "depth",
    "min_samples_leaf": "leaf",
    "max_leaf_nodes": "leaves",
}


_ABBREVIATIONS = {
    "knn": "KNN",
    "svm_rbf": "Kernel SVM",
    "svm_linear": "Linear SVM",
    "boosting": "Boosting",
    "decision_tree": "Decision tree",
    "gaussian_nb": "Naive Bayes",
    "logreg_sga": "LogReg SGA",
}


def _short_name(family: str) -> str:
    """Abbreviated model name, for the narrow single-column tables."""
    return _ABBREVIATIONS.get(family, models.DISPLAY_NAMES[family].split(" (")[0])


def _math_number(value: str) -> str:
    """Scientific notation as a power of ten, plain digits otherwise."""
    if "e-" in value or "e+" in value:
        mantissa, exponent = value.split("e")
        exponent = int(exponent)
        if float(mantissa) == 1.0:
            return rf"10^{{{exponent}}}"
        return rf"{mantissa}\times 10^{{{exponent}}}"
    return value.rstrip("0").rstrip(".") if "." in value else value


def _pretty_params(label: str) -> str:
    """Rewrite ``max_depth=3, min_samples_leaf=50`` as ``depth 3, leaf 50``."""
    if not isinstance(label, str) or not label or label == "default":
        return "default"
    parts = []
    for item in label.split(", "):
        if "=" not in item:
            parts.append(_escape(item))
            continue
        key, value = item.split("=", 1)
        if key in _MATH_SYMBOLS:
            parts.append(f"${_MATH_SYMBOLS[key]} = {_math_number(value)}$")
        elif key in _WORD_SYMBOLS:
            parts.append(f"{_WORD_SYMBOLS[key]} {_math_number(value)}")
        else:
            parts.append(f"{_escape(key)} {_math_number(value)}")
    return ", ".join(parts)


# ------------------------------------------------------------------ table 1
def table_dataset() -> None:
    description = json.loads(
        (RESULTS_DIR / "dataset_description.json").read_text(encoding="utf-8")
    )
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Split & Images & Effusion positive & Negative & Prevalence \\",
        r"\midrule",
    ]
    for split in ("train", "val", "test"):
        info = description["splits"][split]
        lines.append(
            f"{split.capitalize()} & {info['n_samples']:,} & {info['n_positive']:,} & "
            f"{info['n_negative']:,} & {info['prevalence']:.4f} \\\\".replace(",", r"\,")
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write("table_dataset.tex", "\n".join(lines))


# ------------------------------------------------------------------ table 2
def table_baselines() -> None:
    frame = pd.read_csv(RESULTS_DIR / "results_baselines.csv")
    frame = frame.sort_values("val_average_precision", ascending=False)
    best_val = frame["val_average_precision"].max()
    best_test = frame["test_average_precision"].max()

    lines = [
        r"\begin{tabular}{llcccccccc}",
        r"\toprule",
        r"\multirow{2}{*}{Model} & \multirow{2}{*}{Configuration} & "
        r"\multirow{2}{*}{Score type} & Val. & \multicolumn{6}{c}{Test split} \\",
        r"\cmidrule(l){5-10}",
        r" & & & AP & Acc. & Prec. & Rec. & F1 & AUC & AP \\",
        r"\midrule",
    ]
    for _, row in frame.iterrows():
        source = "prob." if row["score_source"] == "predict_proba" else "margin"
        val_ap = f"{row['val_average_precision']:.4f}"
        test_ap = f"{row['test_average_precision']:.4f}"
        if np.isclose(row["val_average_precision"], best_val):
            val_ap = rf"\textbf{{{val_ap}}}"
        if np.isclose(row["test_average_precision"], best_test):
            test_ap = rf"\textbf{{{test_ap}}}"
        lines.append(
            f"{_escape(row['model'])} & {_pretty_params(row['params'])} & {source} & {val_ap} & "
            f"{row['test_accuracy']:.4f} & {row['test_precision']:.4f} & "
            f"{row['test_recall']:.4f} & {row['test_f1']:.4f} & "
            f"{row['test_roc_auc']:.4f} & {test_ap} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write("table_baselines.tex", "\n".join(lines))


# ------------------------------------------------------------------ table 3
def table_operating_point() -> None:
    """Default cut-off against a cut-off selected on validation."""
    frame = pd.read_csv(RESULTS_DIR / "results_baselines.csv")
    frame = frame.sort_values("val_average_precision", ascending=False)
    lines = [
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"\multirow{2}{*}{Model} & \multicolumn{3}{c}{Default cut-off} & "
        r"\multicolumn{3}{c}{Cut-off chosen on validation} \\",
        r"\cmidrule(lr){2-4}\cmidrule(l){5-7}",
        r" & Prec. & Rec. & F1 & Prec. & Rec. & F1 \\",
        r"\midrule",
    ]
    for _, row in frame.iterrows():
        lines.append(
            f"{_escape(row['model'])} & {row['test_precision']:.3f} & {row['test_recall']:.3f} & "
            f"{row['test_f1']:.3f} & {row['test_precision_at_val_threshold']:.3f} & "
            f"{row['test_recall_at_val_threshold']:.3f} & "
            f"{row['test_f1_at_val_threshold']:.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write("table_operating_point.tex", "\n".join(lines))


# ------------------------------------------------------------------ table 4
def table_representation() -> None:
    path = RESULTS_DIR / "results_representation.csv"
    if not path.exists():
        return
    ablation = pd.read_csv(path)
    # The reference row must carry the SAME hyperparameters as the projected
    # one, otherwise the table would compare a representation change with a
    # hyperparameter change. The full candidate sweep is used for that reason,
    # not the per-family best of results_baselines.csv.
    candidates = pd.read_csv(RESULTS_DIR / "results_candidates.csv")

    # One row per model, the two representations side by side: this keeps the
    # table inside a single column and puts the comparison on one line.
    lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\multirow{2}{*}{Model} & \multicolumn{2}{c}{Pixels (784)} & "
        r"\multicolumn{2}{c}{PCA (50)} \\",
        r"\cmidrule(lr){2-3}\cmidrule(l){4-5}",
        r" & AUC & AP & AUC & AP \\",
        r"\midrule",
    ]
    for _, reduced in ablation.iterrows():
        reference = candidates[
            (candidates["family"] == reduced["family"])
            & (candidates["params"] == reduced["params"])
        ]
        if reference.empty:
            print(f"  no matching 784 dimensional run for {reduced['family']} "
                  f"{reduced['params']}: row skipped")
            continue
        full = reference.iloc[0]
        lines.append(
            f"{_short_name(reduced['family'])}, {_pretty_params(reduced['params'])} & "
            f"{full['test_roc_auc']:.4f} & {full['test_average_precision']:.4f} & "
            f"{reduced['test_roc_auc']:.4f} & {reduced['test_average_precision']:.4f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write("table_representation.tex", "\n".join(lines))


# ------------------------------------------------------------------ table 5
def table_robustness() -> None:
    path = RESULTS_DIR / "results_robustness.csv"
    if not path.exists():
        print("  results_robustness.csv not found: robustness tables skipped")
        return
    frame = pd.read_csv(path)
    selection = json.loads((RESULTS_DIR / "selection.json").read_text(encoding="utf-8"))
    families = selection["selected_families"]
    n_seeds = frame["seed"].nunique()

    def cell(values: pd.Series, digits: int = 4) -> str:
        if len(values) == 1 or n_seeds == 1:
            return f"{values.mean():.{digits}f}"
        return rf"{values.mean():.{digits}f}\,$\pm$\,{values.std(ddof=1):.{digits}f}"

    lines = [
        r"\begin{tabular}{llllcccc}",
        r"\toprule",
        r"Model & Config. & Training labels & Hyperparameters & Test F1$^{\star}$ & "
        r"Test AUC & Test AP & $\Delta$AP \\",
        r"\midrule",
    ]
    blocks: list[str] = []
    for family in families:
        block = frame[frame["family"] == family]
        rows: list[str] = []
        for configuration in ("standard", "robust"):
            sub = block[block["configuration"] == configuration]
            clean = sub[sub["label_condition"] == "clean"]
            noisy = sub[sub["label_condition"] == "noisy"]
            delta = noisy["test_average_precision"].mean() - clean["test_average_precision"].mean()
            for condition, group in (("clean", clean), ("5\\% flipped", noisy)):
                if group.empty:
                    continue
                parameters = _pretty_params(group["params"].iloc[0])
                delta_cell = f"{delta:+.4f}" if condition != "clean" else ""
                rows.append(
                    f"{_short_name(family)} & {configuration} & {condition} & "
                    f"{parameters} & {cell(group['test_f1_at_val_threshold'], 3)} & "
                    f"{cell(group['test_roc_auc'])} & {cell(group['test_average_precision'])} & "
                    f"{delta_cell} \\\\"
                )
        if rows:
            blocks.append("\n".join(rows))
    lines.append("\n\\addlinespace\n".join(blocks))
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write("table_robustness.tex", "\n".join(lines))

    # Compact degradation table
    deltas = pd.read_csv(RESULTS_DIR / "results_deltas.csv")
    lines = [
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Model & Config. & AP clean & AP noisy & $\Delta$AP & Relative $\Delta$AP \\",
        r"\midrule",
    ]
    blocks = []
    for family in families:
        rows = []
        for configuration in ("standard", "robust"):
            sub = deltas[(deltas["family"] == family) & (deltas["configuration"] == configuration)]
            if sub.empty:
                continue
            rows.append(
                f"{_short_name(family)} & {configuration} & "
                f"{sub['test_average_precision_clean'].mean():.4f} & "
                f"{sub['test_average_precision_noisy'].mean():.4f} & "
                f"{sub['test_average_precision_delta'].mean():+.4f} & "
                f"{100 * sub['test_average_precision_relative_delta'].mean():+.1f}\\% \\\\"
            )
        if rows:
            blocks.append("\n".join(rows))
    lines.append("\n\\addlinespace\n".join(blocks))
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write("table_degradation.tex", "\n".join(lines))


def consolidate_results() -> None:
    """Merge every run of Phase III into one long-format ``results/results.csv``.

    The per-script files stay as they are, but the team's deliverable list asks
    for a single table. One row is one fitted model on one split, identified by
    the experiment it belongs to, the family, the hyperparameters, the seed, the
    label condition and the configuration.
    """
    frames = []
    baseline_path = RESULTS_DIR / "results_candidates.csv"
    if baseline_path.exists():
        frame = pd.read_csv(baseline_path)
        frame["experiment"] = "3.1-baselines"
        frame["label_condition"] = "clean"
        frame["configuration"] = "standard"
        frames.append(frame)

    representation_path = RESULTS_DIR / "results_representation.csv"
    if representation_path.exists():
        frame = pd.read_csv(representation_path)
        frame["experiment"] = "3.1-representation"
        frame["label_condition"] = "clean"
        frame["configuration"] = "standard"
        frames.append(frame)

    robustness_path = RESULTS_DIR / "results_robustness.csv"
    if robustness_path.exists():
        frame = pd.read_csv(robustness_path).drop(columns=["candidate_trace"], errors="ignore")
        frame["experiment"] = "3.2-robustness"
        frame["feature_space"] = "standardized"
        frames.append(frame)

    if not frames:
        return

    merged = pd.concat(frames, ignore_index=True, sort=False)
    lead = [
        "experiment", "family", "model", "params", "feature_space", "seed",
        "label_condition", "configuration", "noise_rate", "n_flipped", "score_source",
    ]
    ordered = [c for c in lead if c in merged.columns]
    ordered += [c for c in merged.columns if c.startswith("val_")]
    ordered += [c for c in merged.columns if c.startswith("test_")]
    ordered += [c for c in merged.columns if c not in ordered]
    merged = merged[ordered]
    merged.to_csv(RESULTS_DIR / "results.csv", index=False)
    print(f"  wrote results.csv ({len(merged)} rows across {merged['experiment'].nunique()} "
          f"experiment groups)")


def main() -> None:
    print("Consolidated table:")
    consolidate_results()
    print("LaTeX tables:")
    table_dataset()
    table_baselines()
    table_operating_point()
    table_representation()
    table_robustness()
    print(f"Output directory: {TABLES_DIR}")


if __name__ == "__main__":
    main()
