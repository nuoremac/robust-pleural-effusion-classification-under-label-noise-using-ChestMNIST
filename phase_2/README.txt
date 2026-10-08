PHASE II — BOUNDARY-BASED AND NON-LINEAR CLASSIFICATION

Open phase_2_classification.ipynb in VS Code or JupyterLab.
Select the same Python kernel as Phase I: ../phase_1/.venv/bin/python.
All dependencies are already included in Phase I requirements.
Run cells in order. Phase I outputs must exist first, especially the training
indices, phase1_config.json, selected_models.json and metrics CSVs.

Task 2.1: CART decision tree, feature importance, complexity explanation,
and gradient boosting with shallow trees.
Task 2.2: linear soft-margin SVM, support vectors, C regularization,
and RBF kernel SVM with C and gamma selection.
LinearSVC uses a dedicated linear solver. Support-vector counts are reported
for RBF SVC; LinearSVC does not expose support-vector indices.
Task 2.3: separate two-component PCA demonstration of logistic regression
and tree decision boundaries. These illustration models are not the main
benchmark models; they use balanced weights to show prediction regions.

Data: ChestMNIST effusion column 2; identical Phase I training subset.
Main preprocessing: pixels / 255, training-fitted StandardScaler and PCA(50).
Tune on validation average precision only. Keep test data for final evaluation.
No label corruption is added in Phase II.

outputs/ contains metrics, figures, fitted pipelines, selected configurations,
and a validation ranking combining Phase I and II for Phase III handoff.
Timing scopes differ: Phase I fit time included preprocessing; Phase II model
fit time excludes shared preprocessing. Do not directly rank these times.

Do not treat PCA importance as anatomical explanation. Interpret results as
a low-resolution annotation benchmark, not evidence of clinical suitability.

VALIDATION STATUS
The notebook was executed successfully in the Phase I environment.
Measured results and figures are saved in outputs/ and embedded in the notebook.
