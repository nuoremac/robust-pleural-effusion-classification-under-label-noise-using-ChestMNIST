"""Simulation of uncertain labels on the training split (Task 3.2).

The corruption model is deliberately the simplest one that matches the clinical
failure mode described in the target paper: a radiograph showing a small
effusion is reported as negative. It is therefore an asymmetric,
minority-to-majority flip applied to a fixed random subset of the positive
training cases.

Three properties matter for the validity of the comparison and are enforced
here rather than left to the caller:

* only training labels are altered; validation and test labels are returned
  untouched by construction, since this module never sees them;
* the flipped row indices are produced once per seed and reused by every model,
  so the performance gaps reported in Table 3 cannot be attributed to different
  corruptions;
* the flip is recorded, not guessed: the returned record carries the indices and
  the resulting prevalence, which the evaluation script logs to MLflow.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import NOISE_RATE


@dataclass(frozen=True)
class NoiseRecord:
    """Everything needed to reproduce and audit one corruption."""

    minority_label: int
    rate: float
    flipped_indices: np.ndarray
    n_minority: int
    prevalence_before: float
    prevalence_after: float
    seed: int

    def as_params(self) -> dict:
        return {
            "noise_minority_label": int(self.minority_label),
            "noise_rate": float(self.rate),
            "noise_n_flipped": int(self.flipped_indices.size),
            "noise_n_minority": int(self.n_minority),
            "noise_prevalence_before": round(float(self.prevalence_before), 6),
            "noise_prevalence_after": round(float(self.prevalence_after), 6),
            "noise_seed": int(self.seed),
        }


def identify_minority_label(y: np.ndarray) -> int:
    """Return the less frequent of the two labels present in ``y``."""
    values, counts = np.unique(y, return_counts=True)
    if values.size != 2:
        raise ValueError(f"expected a binary target, found values {values.tolist()}")
    return int(values[int(np.argmin(counts))])


def inject_label_noise(
    y_train: np.ndarray, seed: int, rate: float = NOISE_RATE
) -> tuple[np.ndarray, NoiseRecord]:
    """Flip ``rate`` of the minority-class training labels.

    Returns a new label vector; the input is left unmodified so that the clean
    condition of the 2x2 design can be rerun from the same object.
    """
    if not 0.0 <= rate < 1.0:
        raise ValueError(f"rate must lie in [0, 1), got {rate}")

    y_noisy = np.array(y_train, copy=True)
    minority = identify_minority_label(y_train)
    minority_rows = np.flatnonzero(y_train == minority)
    n_flip = int(np.floor(rate * minority_rows.size))

    rng = np.random.default_rng(seed)
    flipped = np.sort(rng.choice(minority_rows, size=n_flip, replace=False))
    y_noisy[flipped] = 1 - minority                      # binary complement

    record = NoiseRecord(
        minority_label=minority,
        rate=rate,
        flipped_indices=flipped,
        n_minority=int(minority_rows.size),
        prevalence_before=float(y_train.mean()),
        prevalence_after=float(y_noisy.mean()),
        seed=seed,
    )
    return y_noisy, record
