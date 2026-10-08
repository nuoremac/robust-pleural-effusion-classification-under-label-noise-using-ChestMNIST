PHASE IV — EVIDENTIAL DEEP LEARNING EXPERIMENTS

Open phase_4_edl.ipynb with the Phase I Python environment.
Requires completed Phase I–III artifacts, including saved preprocessing,
training indices and noisy labels for seeds 42, 43 and 44.

Install CPU-only PyTorch from TP1 if necessary:
  phase_1/.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
Other dependencies are provided by Phases I and III.

Compare matched 50-64-32-2 MLPs with softmax cross-entropy or evidential
expected squared-error, predictive variance and annealed KL objectives.
EDL evidence is generated using softplus; alpha=evidence+1.
Predictive probabilities are alpha/sum(alpha); vacuity is 2/sum(alpha).
Reference: Sensoy et al. (NeurIPS 2018), https://arxiv.org/abs/1806.01768
This is a lightweight feature-based adaptation, not a medical CNN reproduction.

Twelve runs: two objectives, clean/noisy labels and three seeds. Both objectives
start from matching weights and use matching minibatch orders. Select checkpoints
with validation AP. Never tune on test results. Maximum 50 epochs; early stopping.

outputs/ contains metrics, calibration summaries, paired changes, uncertainty
plots, risk–coverage curves and weight checkpoints. MLflow uses phase_3/mlflow.db
with a separate experiment; this phase's model artifacts are in mlartifacts/.
Brier and ECE assess predictive calibration. Error-detection AUC and selective
prediction assess uncertainty usefulness on this dataset; no OOD test is included.
The four-paper literature comparison and final article remain separate work.

VERIFICATION
Executed successfully: 12 neural-network runs, 24 validation/test metric rows,
weight checkpoints, paired changes and uncertainty/calibration figures.
Dirichlet KL was checked against PyTorch's distribution implementation;
finite gradients, saved-weight reload and probability bounds were verified.
