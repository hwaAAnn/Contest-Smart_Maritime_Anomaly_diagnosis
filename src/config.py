"""Global settings shared by the whole pipeline."""

SEED = 45
N_FOLDS = 5

# Class groups used by the 3-stage cascade (original label values).
HARD_CLASSES = [0, 3, 9, 15, 16, 19]
MID_CLASSES = [0, 3, 9, 10, 11, 14, 15, 16, 19]  # HARD + {10, 11, 14}

# XGBoost hyper-parameters used by every stage (num_class is set per stage).
XGB_PARAMS = dict(
    objective="multi:softprob",
    n_estimators=2500,
    learning_rate=0.03,
    max_depth=7,
    min_child_weight=2,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.5,
    reg_lambda=1.5,
    tree_method="hist",
    n_jobs=-1,
)

# Ridge added to every class covariance before inversion (Gaussian stats).
COV_EPS = 1e-5
