PHASE I — DATA PREPARATION AND PROBABILISTIC BASELINES

Project: robust pleural effusion classification with ChestMNIST.

PURPOSE
Prepare reproducible inputs and establish reference performance before
introducing label corruption in Phase III.

DATA ALREADY AVAILABLE
The original arrays are in ../chestmnist(1)/:
  train_images.npy, train_labels.npy
  val_images.npy, val_labels.npy
  test_images.npy, test_labels.npy
Each .npy file stores a NumPy array. Keep the original files unchanged.
Check shapes, types and label values after loading; do not assume them silently.
Use label column 2 (zero-based), named "effusion", as the binary target:
  0 = effusion absent; 1 = effusion present.
ChestMNIST is derived from NIH ChestXray14, not CheXpert. Its labels do not
include CheXpert's explicit uncertainty category. Labels are the dataset's
annotations, not a guarantee of clinical ground truth.

TASK 1.1 — DATA AND PREPROCESSING
1. Load the six arrays with numpy.load(..., allow_pickle=False).
2. Verify image/label counts, expected image dimensions, label values,
   missing/non-finite values, and class counts in each split.
3. Preserve the provided train/validation/test splits. For tonight's runtime,
   select up to 10,000 training images with stratified sampling, seed 42.
   Save selected indices; retain validation and test data.
4. Flatten each 28 x 28 grayscale image into 784 numerical features.
5. If imputation is needed, fit it on training data only. Fit feature scaling
   on training data only and apply the fitted transformation to validation/test.
6. If dimensionality reduction is needed, fit PCA on training data only.
   This is optional, not a substitute for documenting preprocessing.

TASK 1.2 — LOGISTIC REGRESSION
Implement binary logistic regression with stochastic gradient ascent (SGA).
The model learns weights and a bias, then uses sigmoid to estimate P(y=1|x).
Shuffle training examples each epoch; update parameters using training data.
Choose learning rate, epochs and any regularization using validation data.
The assignment also requests multiclass logistic regression. Demonstrate it
on a separate synthetic three-class dataset. ChestMNIST's 14 findings may
coexist, so they must not be treated as 14 mutually exclusive classes.

TASK 1.3 — NAIVE BAYES AND KNN
Gaussian Naive Bayes: estimate each feature's distribution within each class.
Explain its assumption of feature independence conditional on the class.
KNN: predict by voting among K nearby training examples. Test K = 3, 5, 11
on validation data. Explain scaling, distance and high dimensionality.

RESULTS TO SAVE
Dataset checks and class counts; preprocessing configuration; subset indices;
model parameters; validation metrics; final test metrics when choices are fixed.
Use accuracy, precision, recall, F1, ROC-AUC and average precision (AP).
The positive class is effusion present. Never tune using test results.
Do not introduce synthetic label corruption in this phase.

FOLDER STRUCTURE
src/       Future loading, preprocessing and baseline implementation.
outputs/   Generated checks, metrics, saved transformations and figures.
requirements.txt   Python dependencies for Phase I.

PYTHON ENVIRONMENT SETUP — BEFORE RUNNING THE NOTEBOOK
Run these commands in the VS Code terminal, from the project root (TP1),
not inside a notebook cell:

  sudo apt install python3.12-venv
  python3 -m venv phase_1/.venv
  phase_1/.venv/bin/python -m pip install -r phase_1/requirements.txt

The first command is for Ubuntu/Debian with Python 3.12. It requires an
administrator password and is needed only if the venv package is missing.
The final command installs ipykernel and the other Phase I dependencies into
the project environment. Package downloads require an internet connection.

In VS Code, open phase_1_baselines.ipynb and click Select Kernel:
  1. Choose Python Environments.
  2. Select phase_1/.venv/bin/python.
  3. If it is not listed, use the interpreter-path selection to locate it.
  4. Run notebook cells in order.

If VS Code says that Python requires ipykernel, complete the terminal setup
above before running cells. The notebook's optional %pip cell cannot run
without a working kernel. After installing requirements.txt, skip that cell:
the required packages are already installed in the selected environment.

NOTEBOOK
The notebook phase_1_baselines.ipynb contains explanations and runnable cells
for data checks, shared preprocessing, binary logistic SGA, Gaussian Naive
Bayes, KNN, final evaluation and a synthetic multiclass SGA demonstration.
Open it in JupyterLab or VS Code and select the project environment above.
Notebook code syntax was checked at creation, but training and runtime
validation were not performed. Numerical results are generated when you run it.
