PHASE III — ROBUSTNESS, EVALUATION AND MLOPS

Open phase_3_robustness.ipynb and use phase_1/.venv/bin/python as the kernel.
From TP1, install additional requirements if necessary:
  phase_1/.venv/bin/python -m pip install -r phase_3/requirements.txt
Run all notebook cells in order after completing Phases I and II.

Task 3.1: reuse all baseline metrics, evaluate accuracy, precision, recall,
F1, ROC-AUC and average precision; plot precision–recall curves.
Task 3.2: select the top three Phase II validation families, flip 5% of the
minority-class training labels, and compare clean/noisy standard/robust models.
Task 3.3: log hyperparameters, metrics, label indices and complete inference
models in local MLflow runs.

Selected families: gradient boosting, Linear SVM, Gaussian Naive Bayes.
Robust variants: shallower boosted trees, smaller SVM C, larger Naive Bayes
variance smoothing. These are hypotheses to evaluate, not guaranteed fixes.
Three seeds yield 36 training runs and paired performance drops. The same
corrupted indices are used across models for each seed. Original data is intact.
Do not tune on test results. Keep the fixed Phase I–II preprocessing/subset.

outputs/ stores raw and summary metrics, noise indices, protocol and plots.
mlflow.db and mlartifacts/ store local experiment tracking and fitted models.
To view MLflow, run from TP1:
  phase_1/.venv/bin/python -m mlflow ui --backend-store-uri sqlite:///phase_3/mlflow.db --host 127.0.0.1 --port 5000
Open http://127.0.0.1:5000 in your browser.

Interpret absolute noisy performance as well as drops. Default-threshold
recall is low. Seed variability does not quantify uncertainty across patients.
Five percent of minority labels is not five percent of all training examples.

VERIFICATION
The notebook executed successfully: 36 training runs, 72 validation/test
metric rows, fitted model artifacts, summary tables and both figures.
The tracking history also retains an initial failed serialization attempt.
