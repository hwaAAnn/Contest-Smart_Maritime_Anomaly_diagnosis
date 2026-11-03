"""XGBoost factory and a CV-bagged stage classifier."""
import numpy as np
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from .config import N_FOLDS, SEED, XGB_PARAMS
from .features import FeatureStore


def make_xgb(n_classes, seed=SEED, use_gpu=False, **overrides):
    params = {**XGB_PARAMS, **overrides}
    params.update(num_class=n_classes, random_state=seed, device="cuda" if use_gpu else "cpu")
    return XGBClassifier(**params)


class StageClassifier:
    """One cascade stage: 5-fold CV bagging of (FeatureStore -> XGBoost).

    Each fold fits its own FeatureStore on the fold's training part, so the
    scaler / PCA / LDA / Gaussian statistics never see the validation part.
    Predictions are the average of the fold models' probabilities.
    """

    def __init__(self, name, classes, n_folds=N_FOLDS, seed=SEED, use_gpu=False, **xgb_overrides):
        self.name = name
        self.classes = np.sort(np.asarray(classes))
        self.n_folds, self.seed, self.use_gpu = n_folds, seed, use_gpu
        self.xgb_overrides = xgb_overrides
        self.folds = []  # [(FeatureStore, XGBClassifier)]

    def fit(self, X, y, verbose=True):
        X, y = np.asarray(X), np.asarray(y)
        keep = np.isin(y, self.classes)
        X, y = X[keep], y[keep]
        y_idx = np.searchsorted(self.classes, y)  # XGBoost needs labels 0..K-1
        oof = np.zeros(len(y), dtype=int)

        skf = StratifiedKFold(self.n_folds, shuffle=True, random_state=self.seed)
        for k, (tr, va) in enumerate(skf.split(X, y_idx), 1):
            store = FeatureStore(self.classes, seed=self.seed).fit(X[tr], y[tr])
            model = make_xgb(len(self.classes), self.seed, self.use_gpu, **self.xgb_overrides)
            model.fit(store.transform(X[tr]), y_idx[tr])
            oof[va] = model.predict(store.transform(X[va]))
            self.folds.append((store, model))
            if verbose:
                f1 = f1_score(y_idx[va], oof[va], average="macro")
                print(f"[{self.name}] fold {k}: macro-F1 = {f1:.4f}")

        self.oof_f1 = f1_score(y_idx, oof, average="macro")
        if verbose:
            print(f"[{self.name}] OOF macro-F1 = {self.oof_f1:.4f}")
            print(classification_report(y, self.classes[oof], digits=4))
        return self

    def predict_proba(self, X):
        """Mean fold probability, columns ordered as ``self.classes``."""
        X = np.asarray(X)
        return np.mean([m.predict_proba(s.transform(X)) for s, m in self.folds], axis=0)

    def predict(self, X):
        return self.classes[np.argmax(self.predict_proba(X), axis=1)]
