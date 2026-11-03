"""3-stage cascade: all classes -> MID classes -> HARD classes."""
import numpy as np

from .config import HARD_CLASSES, MID_CLASSES, N_FOLDS, SEED
from .models import StageClassifier


class CascadeClassifier:
    """Stage 1 predicts all 21 classes. Samples routed to a MID class are
    re-classified by a MID-only model (Stage 2); those that land on a HARD
    class are refined once more by a HARD-only model (Stage 3)."""

    def __init__(self, all_classes, mid=MID_CLASSES, hard=HARD_CLASSES,
                 n_folds=N_FOLDS, seed=SEED, use_gpu=False, **xgb_overrides):
        kw = dict(n_folds=n_folds, seed=seed, use_gpu=use_gpu, **xgb_overrides)
        self.mid, self.hard = list(mid), list(hard)
        self.stage1 = StageClassifier("stage1-all", all_classes, **kw)
        self.stage2 = StageClassifier("stage2-mid", self.mid, **kw)
        self.stage3 = StageClassifier("stage3-hard", self.hard, **kw)

    def fit(self, X, y):
        for stage in (self.stage1, self.stage2, self.stage3):
            stage.fit(X, y)  # each stage keeps only the rows of its own classes
        return self

    def predict(self, X):
        X = np.asarray(X)
        pred = self.stage1.predict(X)

        idx2 = np.where(np.isin(pred, self.mid))[0]
        if len(idx2):
            pred[idx2] = self.stage2.predict(X[idx2])

        idx3 = np.where(np.isin(pred, self.hard) & np.isin(np.arange(len(pred)), idx2))[0]
        if len(idx3):
            pred[idx3] = self.stage3.predict(X[idx3])
        return pred
