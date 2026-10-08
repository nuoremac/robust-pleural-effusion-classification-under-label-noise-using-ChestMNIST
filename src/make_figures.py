"""Figures for the Phase III sections of the article.

Four figures are produced, each from the score vectors persisted by the two
experiment scripts rather than from a fresh fit, so a figure can never disagree
with the table next to it:

fig1_pr_baselines
    Precision-recall curves of the seven baselines on the test split, with the
    no-skill line drawn at the test prevalence.
fig2_roc_baselines
    The same seven models as ROC curves. Printed beside figure 1 it shows why
    the two views disagree about how much progress the models make.
fig3_pr_robustness
    One panel per selected family, four curves per panel (the 2x2 design).
fig4_ap_degradation
    Test average precision before and after corruption, standard against robust
    configuration, averaged over the seeds with the spread shown as error bars.
fig5_threshold
    F1 as a function of the decision threshold for the best model, with the
    default cut-off and the validation selected cut-off marked.

Both a PDF (for the article) and a PNG (for quick inspection) are written.

Usage: python src/make_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.metrics import (
    average_precision_score, precision_recall_curve, roc_auc_score, roc_curve,
)

import models
from config import FIGURES_DIR, PRIMARY_SEED, RESULTS_DIR

SCORES_DIR = RESULTS_DIR / "scores"

# Colour-blind safe sequence (Okabe and Ito), kept identical across figures so
# that a family keeps its colour from one figure to the next.
PALETTE = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#444444")
CONDITION_STYLE = {
    ("clean", "standard"): dict(color="#0072B2", linestyle="-"),
    ("clean", "robust"): dict(color="#56B4E9", linestyle="--"),
    ("noisy", "standard"): dict(color="#D55E00", linestyle="-"),
    ("noisy", "robust"): dict(color="#E69F00", linestyle="--"),
}
# The precision axis of the PR plots is truncated: on a 12 percent prevalence
# task the leftmost part of the curve is decided by a handful of examples and
# swings between 0 and 1, which compresses the region that actually matters.
# Every caption states the truncation.
PR_YMAX = 0.70

CONDITION_LABEL = {
    ("clean", "standard"): "clean labels, standard",
    ("clean", "robust"): "clean labels, robust",
    ("noisy", "standard"): "noisy labels, standard",
    ("noisy", "robust"): "noisy labels, robust",
}


def use_paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8.5,
            "legend.fontsize": 6.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 1.2,
            "figure.dpi": 150,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def save(fig: plt.Figure, stem: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        fig.savefig(FIGURES_DIR / f"{stem}.{extension}", dpi=300)
    plt.close(fig)
    print(f"  wrote {stem}.pdf and {stem}.png")


def _load_scores(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as handle:
        return {k: handle[k] for k in handle.files}


def _baseline_score_file(family: str, params_label: str, seed: int) -> Path:
    return SCORES_DIR / f"baseline_{family}_{params_label.replace(' ', '')}_seed{seed}.npz"


# -------------------------------------------------------------------- figure 1-2
def figure_baseline_curves(seed: int = PRIMARY_SEED) -> None:
    baselines = pd.read_csv(RESULTS_DIR / "results_baselines.csv")
    baselines = baselines.sort_values("test_average_precision", ascending=False)

    curves = []
    for colour, (_, row) in zip(PALETTE, baselines.iterrows()):
        path = _baseline_score_file(row["family"], row["params"], seed)
        if not path.exists():
            print(f"  missing score file for {row['family']}: {path.name}")
            continue
        scores = _load_scores(path)
        curves.append((row, colour, scores["test_true"], scores["test_score"]))

    if not curves:
        return
    prevalence = float(np.mean(curves[0][2]))

    fig, ax = plt.subplots(figsize=(3.45, 2.75))
    for row, colour, y_true, y_score in curves:
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        # Step drawing, not a straight join: linear interpolation between two
        # points of a precision-recall curve is not achievable by any classifier
        # and overstates the area, which is visible on the saturated posterior of
        # naive Bayes. Average precision itself is a step-wise sum.
        ax.plot(recall, precision, color=colour, drawstyle="steps-post",
                label=f"{models.DISPLAY_NAMES[row['family']]} (AP {ap:.3f})")
    ax.axhline(prevalence, color="black", linestyle=":", linewidth=0.9,
               label=f"no skill (prevalence {prevalence:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, PR_YMAX)
    ax.set_title("Precision-recall on the test split")
    ax.legend(loc="upper right", frameon=True, framealpha=0.92, edgecolor="none",
              facecolor="white", handlelength=1.6)
    save(fig, "fig1_pr_baselines")

    fig, ax = plt.subplots(figsize=(3.45, 2.75))
    for row, colour, y_true, y_score in curves:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
        ax.plot(fpr, tpr, color=colour,
                label=f"{models.DISPLAY_NAMES[row['family']]} (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="black", linestyle=":", linewidth=0.9, label="no skill")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("ROC on the test split")
    # The ROC curves crowd the lower right corner, so the legend needs an opaque
    # backing to stay readable where it sits.
    ax.legend(loc="lower right", frameon=True, framealpha=0.92, edgecolor="none",
              facecolor="white", handlelength=1.6)
    save(fig, "fig2_roc_baselines")


# ---------------------------------------------------------------------- figure 3
def figure_robustness_curves(seed: int = PRIMARY_SEED) -> None:
    selection = json.loads((RESULTS_DIR / "selection.json").read_text(encoding="utf-8"))
    families = selection["selected_families"]

    fig, axes = plt.subplots(1, len(families), figsize=(7.0, 2.5), sharey=True)
    axes = np.atleast_1d(axes)
    prevalence = None

    for ax, family in zip(axes, families):
        for (condition, configuration), style in CONDITION_STYLE.items():
            path = SCORES_DIR / f"robust_{family}_{condition}_{configuration}_seed{seed}.npz"
            if not path.exists():
                continue
            scores = _load_scores(path)
            y_true, y_score = scores["test_true"], scores["test_score"]
            prevalence = float(np.mean(y_true))
            precision, recall, _ = precision_recall_curve(y_true, y_score)
            ap = average_precision_score(y_true, y_score)
            ax.plot(recall, precision, drawstyle="steps-post",
                    label=f"{CONDITION_LABEL[(condition, configuration)]} "
                          f"(AP {ap:.3f})", **style)
        if prevalence is not None:
            ax.axhline(prevalence, color="black", linestyle=":", linewidth=0.9)
        ax.set_title(models.DISPLAY_NAMES[family])
        ax.set_xlabel("Recall")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, PR_YMAX)
        ax.legend(loc="upper right", frameon=False, handlelength=1.8)
    axes[0].set_ylabel("Precision")
    fig.suptitle(
        f"Test precision-recall under the 2x2 design, seed {seed} "
        "(dotted line: no-skill prevalence)", fontsize=8.5, y=1.04,
    )
    save(fig, "fig3_pr_robustness")


# ---------------------------------------------------------------------- figure 4
def figure_ap_degradation() -> None:
    frame = pd.read_csv(RESULTS_DIR / "results_robustness.csv")
    selection = json.loads((RESULTS_DIR / "selection.json").read_text(encoding="utf-8"))
    families = selection["selected_families"]

    grouped = (
        frame.groupby(["family", "configuration", "label_condition"])["test_average_precision"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(3.45, 2.75))
    width = 0.2
    positions = np.arange(len(families), dtype=float)
    bars = [
        ("standard", "clean", "#0072B2", "//"),
        ("standard", "noisy", "#D55E00", "//"),
        ("robust", "clean", "#56B4E9", ".."),
        ("robust", "noisy", "#E69F00", ".."),
    ]
    for offset, (configuration, condition, colour, hatch) in zip(
        np.linspace(-1.5, 1.5, 4) * width, bars
    ):
        heights, errors = [], []
        for family in families:
            match = grouped[
                (grouped["family"] == family)
                & (grouped["configuration"] == configuration)
                & (grouped["label_condition"] == condition)
            ]
            heights.append(float(match["mean"].iloc[0]) if len(match) else np.nan)
            spread = float(match["std"].iloc[0]) if len(match) else np.nan
            errors.append(0.0 if np.isnan(spread) else spread)
        ax.bar(positions + offset, heights, width * 0.92, yerr=errors, capsize=1.8,
               color=colour, edgecolor="white", linewidth=0.4,
               error_kw=dict(elinewidth=0.7),
               label=f"{configuration}, {condition} labels")

    prevalence = float(selection["test_prevalence"])
    ax.axhline(prevalence, color="black", linestyle=":", linewidth=0.9)
    ax.annotate(f"no skill ({prevalence:.3f})", xy=(len(families) - 0.5, prevalence),
                xytext=(0, 3), textcoords="offset points", ha="right", fontsize=6.5)
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [models.DISPLAY_NAMES[f].replace(" (", "\n(") for f in families], fontsize=6.8
    )
    ax.set_ylabel("Test average precision")
    ax.set_title("Average precision before and after label corruption")
    ax.legend(loc="lower left", frameon=False, ncols=2, columnspacing=1.0)
    seeds = sorted(frame["seed"].unique().tolist())
    ax.text(0.5, -0.33, f"bars: mean over seeds {seeds}; whiskers: standard deviation",
            transform=ax.transAxes, ha="center", fontsize=6.2)
    save(fig, "fig4_ap_degradation")


# ---------------------------------------------------------------------- figure 5
def figure_threshold_sensitivity(seed: int = PRIMARY_SEED) -> None:
    """F1 against the decision threshold for the best baseline.

    The point of the figure is the distance between the default cut-off each
    family uses and the cut-off that actually maximises F1 on this prevalence.
    """
    baselines = pd.read_csv(RESULTS_DIR / "results_baselines.csv")
    best = baselines.sort_values("val_average_precision", ascending=False).iloc[0]
    path = _baseline_score_file(best["family"], best["params"], seed)
    if not path.exists():
        return
    scores = _load_scores(path)

    fig, ax = plt.subplots(figsize=(3.45, 2.4))
    for split, colour in (("val", "#0072B2"), ("test", "#D55E00")):
        y_true, y_score = scores[f"{split}_true"], scores[f"{split}_score"]
        precision, recall, thresholds = precision_recall_curve(y_true, y_score)
        precision, recall = precision[:-1], recall[:-1]
        denominator = precision + recall
        f1 = np.where(denominator > 0, 2 * precision * recall / np.maximum(denominator, 1e-12), 0)
        ax.plot(thresholds, f1, color=colour, label=f"{split} F1")

    default_cut = 0.5 if best["score_source"] == "predict_proba" else 0.0
    ax.axvline(default_cut, color="black", linestyle=":", linewidth=0.9)
    ax.axvline(float(best["val_selected_threshold"]), color="#009E73",
               linestyle="--", linewidth=0.9)
    handles, labels = ax.get_legend_handles_labels()
    handles += [
        Line2D([], [], color="black", linestyle=":", linewidth=0.9),
        Line2D([], [], color="#009E73", linestyle="--", linewidth=0.9),
    ]
    labels += [f"default cut-off ({default_cut:g})",
               f"validation choice ({float(best['val_selected_threshold']):.3f})"]
    ax.set_xlabel(f"Decision threshold on the {best['score_source']} score")
    ax.set_ylabel("F1")
    ax.set_title(f"Operating point of {models.DISPLAY_NAMES[best['family']]}")
    ax.legend(handles, labels, loc="upper right", frameon=False, handlelength=1.8)
    save(fig, "fig5_threshold_sensitivity")


def main() -> None:
    use_paper_style()
    print("Figures:")
    figure_baseline_curves()
    if (RESULTS_DIR / "results_robustness.csv").exists():
        figure_robustness_curves()
        figure_ap_degradation()
    else:
        print("  results_robustness.csv not found: figures 3 and 4 skipped")
    figure_threshold_sensitivity()
    print(f"Output directory: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
