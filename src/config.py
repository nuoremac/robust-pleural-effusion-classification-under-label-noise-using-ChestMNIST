"""Shared experimental protocol for TP4.

Every constant in this module is part of the protocol the team fixed before
running anything. Nothing here may be changed by a single phase in isolation:
the three phases must read the same values so that the numbers reported in the
article are comparable across contributors.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
ARTIFACTS_DIR = RESULTS_DIR / "artifacts"
# MLflow is backed by a local SQLite database rather than the plain file store,
# which MLflow 3 deprecates. Both the database and the artifact tree stay inside
# the repository so the run history can be inspected with
#   mlflow ui --backend-store-uri sqlite:///mlflow.db
MLFLOW_TRACKING_URI = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
MLFLOW_ARTIFACT_URI = (ROOT / "mlartifacts").as_uri()

# ---------------------------------------------------------------- task definition
# ChestMNIST ships 14 binary thorax findings per radiograph. The team's clinical
# target is pleural effusion, which is column 2 of the label matrix. The order
# follows NIH ChestX-ray14 as redistributed by MedMNIST v2.
LABEL_NAMES = (
    "atelectasis", "cardiomegaly", "effusion", "infiltration", "mass", "nodule",
    "pneumonia", "pneumothorax", "consolidation", "edema", "emphysema",
    "fibrosis", "pleural_thickening", "hernia",
)
TARGET_INDEX = 2
TARGET_NAME = LABEL_NAMES[TARGET_INDEX]

# ---------------------------------------------------------------- sampling
# The official MedMNIST train/val/test partition is preserved. Only the training
# split is subsampled, to keep seven model families and a 2x2 robustness design
# inside the time budget of the lab session.
TRAIN_SUBSET_SIZE = 10_000
PRIMARY_SEED = 42
SEEDS = (42, 7, 2024)

# ---------------------------------------------------------------- representation
IMAGE_SIDE = 28
N_FEATURES = IMAGE_SIDE * IMAGE_SIDE          # 784 raw pixel intensities
PCA_COMPONENTS = 50                           # secondary representation only

# ---------------------------------------------------------------- label noise
# Task 3.2: flip a fraction of the training labels of the minority class. The
# selected row indices are persisted so that every model sees exactly the same
# corrupted training set. Validation and test labels are never touched.
NOISE_RATE = 0.05

# ---------------------------------------------------------------- model selection
SELECTION_METRIC = "average_precision"        # computed on the validation split
N_SELECTED_MODELS = 3

# ---------------------------------------------------------------- tracking
MLFLOW_EXPERIMENT = "tp4-pleural-effusion-phase3"
