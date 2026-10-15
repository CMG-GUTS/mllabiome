from pathlib import Path

from curated_microbiota.collections import lampp_crc
from sklearn.linear_model import SGDClassifier

from mllabiome import Transform
from mllabiome.federated import Federated


EXPERIMENT_DIR = Path("runs/fl_lampp_lodo_crc")
HELD_OUT_STUDY = None
NUM_ROUNDS = 10
LOCAL_EPOCHS = 2
NUM_SHARES = 3
RECONSTRUCTION_THRESHOLD = 2
CLIPPING_RANGE = 8.0
LEARNING_RATE = 0.01
L2_ALPHA = 0.0001
RANDOM_STATE = 17
CLIENT_CPUS = 1
FLOWER_PROJECT = Path(__file__).resolve().parent / "flower-secagg"


DATA = lampp_crc.mllabiome()
EVALUATION = lampp_crc.splits().mllabiome(
    optimize_metric="log_loss",
    n_jobs="auto",
)
RESOLUTIONS = (("raw", ("all",)),)
COUNT_TRANSFORMATIONS = (Transform("arcsine_sqrt"),)
MODELS = (
    (
        "fed_sgd_logistic",
        SGDClassifier(
            loss="log_loss",
            penalty="l2",
            learning_rate="constant",
            alpha=L2_ALPHA,
            eta0=LEARNING_RATE,
            random_state=RANDOM_STATE,
        ),
    ),
)
FEDERATED = Federated(
    num_rounds=NUM_ROUNDS,
    local_epochs=LOCAL_EPOCHS,
    num_shares=NUM_SHARES,
    reconstruction_threshold=RECONSTRUCTION_THRESHOLD,
    clipping_range=CLIPPING_RANGE,
    held_out_study=HELD_OUT_STUDY,
    random_state=RANDOM_STATE,
    client_cpus=CLIENT_CPUS,
    flower_project=FLOWER_PROJECT,
)
