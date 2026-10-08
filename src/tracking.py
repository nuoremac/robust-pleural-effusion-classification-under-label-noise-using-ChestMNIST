"""MLflow tracking for Phase III (Task 3.3).

The experiment is backed by a local SQLite database (``mlflow.db``) with its
artifact tree in ``mlartifacts/``, both inside the repository, so the run history
travels with the code and can be inspected by the two other contributors with
``mlflow ui --backend-store-uri sqlite:///mlflow.db``. The plain file store is
avoided on purpose: MLflow 3 deprecates it.

What is logged, and why it is logged rather than written down in the report:

* parameters: model family, every hyperparameter, the seed, the feature space,
  the training subset size, the noise rate and the number of flipped labels.
  These seven fields are exactly what is needed to rebuild a run;
* metrics: the eight evaluation metrics on validation and on test, plus fit and
  prediction wall clock time;
* tags: the experimental condition of the 2x2 design (clean or noisy labels,
  standard or robust configuration), so runs can be filtered from the UI;
* artifacts: the serialised estimator, the flipped label indices and the
  training subset indices.

Tracking is optional. If MLflow is not installed the helpers degrade to no-ops
and the experiment still produces its CSV files, because the reproducibility of
the results must not depend on the availability of the tracking server.
"""
from __future__ import annotations

import json
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from config import MLFLOW_ARTIFACT_URI, MLFLOW_EXPERIMENT, MLFLOW_TRACKING_URI

try:
    import mlflow
    import mlflow.sklearn

    MLFLOW_AVAILABLE = True
except ImportError:                                    # pragma: no cover
    mlflow = None                                      # type: ignore[assignment]
    MLFLOW_AVAILABLE = False


_INITIALISED = False


def init_tracking(
    experiment: str = MLFLOW_EXPERIMENT,
    uri: str = MLFLOW_TRACKING_URI,
    artifact_uri: str = MLFLOW_ARTIFACT_URI,
) -> bool:
    """Point MLflow at the local store and select the experiment. Idempotent."""
    global _INITIALISED
    if not MLFLOW_AVAILABLE:
        return False
    if not _INITIALISED:
        mlflow.set_tracking_uri(uri)
        if mlflow.get_experiment_by_name(experiment) is None:
            mlflow.create_experiment(experiment, artifact_location=artifact_uri)
        mlflow.set_experiment(experiment)
        _INITIALISED = True
    return True


@contextmanager
def run(name: str, tags: dict[str, Any] | None = None, nested: bool = False) -> Iterator[Any]:
    """Open one MLflow run, or yield ``None`` when tracking is unavailable."""
    if not init_tracking():
        yield None
        return
    with mlflow.start_run(run_name=name, nested=nested) as active:
        if tags:
            mlflow.set_tags({k: str(v) for k, v in tags.items()})
        yield active


def log_params(params: dict[str, Any]) -> None:
    if MLFLOW_AVAILABLE and mlflow.active_run() is not None:
        mlflow.log_params({k: str(v) for k, v in params.items()})


def log_metrics(metrics: dict[str, float]) -> None:
    if MLFLOW_AVAILABLE and mlflow.active_run() is not None:
        numeric = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
        mlflow.log_metrics(numeric)


def log_json(payload: dict[str, Any], filename: str) -> None:
    """Persist a small dictionary as a JSON artifact of the active run."""
    if not (MLFLOW_AVAILABLE and mlflow.active_run() is not None):
        return
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / filename
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        mlflow.log_artifact(str(path))


def log_file(path: str | Path, artifact_path: str | None = None) -> None:
    if MLFLOW_AVAILABLE and mlflow.active_run() is not None and Path(path).exists():
        mlflow.log_artifact(str(path), artifact_path=artifact_path)


def log_model(model: Any, name: str = "model") -> None:
    """Log the fitted estimator as a run artifact (Task 3.3, third item)."""
    if not (MLFLOW_AVAILABLE and mlflow.active_run() is not None):
        return
    try:
        mlflow.sklearn.log_model(model, name=name)
    except TypeError:
        # MLflow renamed artifact_path to name in version 3; support both.
        mlflow.sklearn.log_model(model, artifact_path=name)
    except Exception as exc:                           # pragma: no cover
        print(f"  [tracking] could not log the model as an artifact: {exc}")
