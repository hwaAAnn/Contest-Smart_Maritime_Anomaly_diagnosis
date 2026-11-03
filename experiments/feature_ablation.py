"""Feature-set ablation on the difficult classes (XGBoost or MLP).

Consolidates the "전문의고도화" notebooks. Eight feature sets are compared on a
stratified 70/30 split of the selected classes:

    1 scaled          2 PCA            3 LDA          4 scaled+PCA
    5 scaled+LDA      6 PCA+LDA        7 scaled+PCA+LDA   8 PCA -> LDA

    python experiments/feature_ablation.py --data-dir data --classes hard --model xgb
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import HARD_CLASSES, SEED  # noqa: E402

MID_ONLY = [10, 11, 14]
LONG_SCHEDULE = {6, 7}  # the notebooks trained these two with lr=0.03 / 4000 trees


def select_classes(df, mode):
    if mode == "hard":
        return df[df["target"].isin(HARD_CLASSES)].copy()
    if mode == "hard9":  # HARD + MID classes, labels kept
        return df[df["target"].isin(HARD_CLASSES + MID_ONLY)].copy()
    if mode == "mid_vs_hard":  # HARD merged into a single label 0
        sub = df[df["target"].isin(HARD_CLASSES + MID_ONLY)].copy()
        sub["target"] = sub["target"].where(~sub["target"].isin(HARD_CLASSES), 0)
        return sub
    raise ValueError(mode)


def build_feature_sets(X_tr, X_te, y_tr, n_lda):
    """Return {type_id: (train_df, test_df)}."""
    sc = StandardScaler().fit(X_tr)
    s_tr = pd.DataFrame(sc.transform(X_tr), index=X_tr.index, columns=X_tr.columns)
    s_te = pd.DataFrame(sc.transform(X_te), index=X_te.index, columns=X_te.columns)

    n_pc = X_tr.shape[1]
    pca = PCA(n_components=n_pc).fit(s_tr)
    cols = [f"PC{i + 1}" for i in range(n_pc)]
    p_tr = pd.DataFrame(pca.transform(s_tr), index=X_tr.index, columns=cols)
    p_te = pd.DataFrame(pca.transform(s_te), index=X_te.index, columns=cols)

    lcols = [f"LDA{i + 1}" for i in range(n_lda)]
    lda = LinearDiscriminantAnalysis(n_components=n_lda).fit(s_tr, y_tr)
    l_tr = pd.DataFrame(lda.transform(s_tr), index=X_tr.index, columns=lcols)
    l_te = pd.DataFrame(lda.transform(s_te), index=X_te.index, columns=lcols)

    lda_p = LinearDiscriminantAnalysis(n_components=n_lda).fit(p_tr, y_tr)
    lp_tr = pd.DataFrame(lda_p.transform(p_tr), index=X_tr.index, columns=lcols)
    lp_te = pd.DataFrame(lda_p.transform(p_te), index=X_te.index, columns=lcols)

    cat = lambda *a: pd.concat(a, axis=1)  # noqa: E731
    return {
        1: (s_tr, s_te),
        2: (p_tr, p_te),
        3: (l_tr, l_te),
        4: (cat(s_tr, p_tr), cat(s_te, p_te)),
        5: (cat(s_tr, l_tr), cat(s_te, l_te)),
        6: (cat(p_tr, l_tr), cat(p_te, l_te)),
        7: (cat(s_tr, p_tr, l_tr), cat(s_te, p_te, l_te)),
        8: (lp_tr, lp_te),
    }


def make_model(kind, type_id, n_classes):
    if kind == "mlp":
        return MLPClassifier(hidden_layer_sizes=(128, 64, 32), activation="relu",
                             solver="adam", max_iter=200, random_state=42)
    lr, n_est = (0.03, 4000) if type_id in LONG_SCHEDULE else (0.1, 100)
    return XGBClassifier(objective="multi:softprob", num_class=n_classes, max_depth=4,
                         learning_rate=lr, n_estimators=n_est, subsample=0.8,
                         colsample_bytree=0.8, reg_alpha=0.0, reg_lambda=1.0,
                         eval_metric="mlogloss", random_state=SEED)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--classes", default="hard", choices=["hard", "hard9", "mid_vs_hard"])
    ap.add_argument("--model", default="xgb", choices=["xgb", "mlp"])
    ap.add_argument("--out-dir", default="outputs/ablation")
    args = ap.parse_args()

    df = pd.read_csv(os.path.join(args.data_dir, "train.csv")).drop(columns=["ID"])
    df = select_classes(df, args.classes)
    X, y = df.drop(columns=["target"]), df["target"]

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y_enc, test_size=0.3, random_state=SEED, stratify=y)
    n_cls = len(le.classes_)
    feats = build_feature_sets(X_tr, X_te, y_tr, n_lda=min(5, n_cls - 1))

    os.makedirs(args.out_dir, exist_ok=True)
    rows = []
    for t, (a, b) in feats.items():
        model = make_model(args.model, t, n_cls).fit(a, y_tr)
        pred = model.predict(b)
        f1 = f1_score(y_te, pred, average="macro")
        rows.append({"type": t, "n_features": a.shape[1], "macro_f1": f1})
        print(f"type{t}: features={a.shape[1]:3d}  macro-F1={f1:.4f}")
        pd.DataFrame(confusion_matrix(y_te, pred), index=le.classes_, columns=le.classes_) \
            .to_csv(os.path.join(args.out_dir, f"cm_type{t}.csv"))
        proba = pd.DataFrame(model.predict_proba(b), index=b.index,
                             columns=[f"prob_{c}" for c in le.classes_])
        proba["true_label"] = le.inverse_transform(y_te)
        proba.to_csv(os.path.join(args.out_dir, f"proba_type{t}.csv"))

    pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, "summary.csv"), index=False)


if __name__ == "__main__":
    main()
