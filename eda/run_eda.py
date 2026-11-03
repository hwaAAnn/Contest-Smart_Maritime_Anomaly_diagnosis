"""Exploratory analysis used before modelling. Figures/CSVs go to --out-dir.

    python eda/run_eda.py --data-dir data --steps importance pca corr vif dist hard

Steps
    importance  XGBoost feature importance (+ SHAP summary if `shap` is installed)
    pca         2-D PCA projection coloured by target
    corr        correlation heatmap and |corr| >= 0.8 pairs
    vif         VIF per feature and VIF-based pruning of correlated groups
    dist        per-feature histograms, boxplots by target and feature-vs-target scatter
    hard        hard-class views: pairwise scatter, histograms, X_01 x target counts
"""
import argparse
import itertools
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import HARD_CLASSES, SEED  # noqa: E402

CORR_THRESHOLD = 0.8


def _save(fig_name, out):
    plt.tight_layout()
    plt.savefig(os.path.join(out, fig_name), dpi=120)
    plt.close()


def feature_importance(X, y, out):
    y_idx = pd.factorize(pd.Series(y), sort=True)[0]
    model = XGBClassifier(objective="multi:softprob", eval_metric="mlogloss", random_state=SEED).fit(X, y_idx)
    score = model.get_booster().get_score(importance_type="weight")
    imp = pd.DataFrame({"feature": list(score), "importance": list(score.values())}) \
        .sort_values("importance", ascending=False)
    imp.to_csv(os.path.join(out, "xgb_feature_importance.csv"), index=False)

    plt.figure(figsize=(8, 12))
    sns.barplot(data=imp, x="importance", y="feature", color="steelblue")
    _save("xgb_feature_importance.png", out)

    try:
        import shap
    except ImportError:
        print("shap not installed - skipping SHAP summary")
        return
    shap.summary_plot(shap.TreeExplainer(model).shap_values(X), X, show=False)
    _save("shap_summary.png", out)


def pca_projection(X, y, out):
    Z = PCA(n_components=2).fit_transform(X)
    plt.figure(figsize=(9, 7))
    sc = plt.scatter(Z[:, 0], Z[:, 1], c=y, cmap="tab20", s=12, alpha=0.8)
    plt.colorbar(sc, label="target")
    plt.xlabel("PCA1")
    plt.ylabel("PCA2")
    _save("pca_2d.png", out)


def correlation(X, out):
    corr = X.corr()
    plt.figure(figsize=(14, 11))
    sns.heatmap(corr, cmap="coolwarm", center=0, linewidths=0.3)
    _save("corr_heatmap.png", out)

    upper = corr.abs().where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    pairs = upper.stack()
    pairs = pairs[pairs >= CORR_THRESHOLD].sort_values(ascending=False)
    pairs.to_csv(os.path.join(out, "high_corr_pairs.csv"), header=["abs_corr"])
    print(f"{len(pairs)} feature pairs with |corr| >= {CORR_THRESHOLD}")


def vif(X, out):
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    def compute(df):
        return pd.Series([variance_inflation_factor(df.values, i) for i in range(df.shape[1])], index=df.columns)

    compute(X).sort_values(ascending=False).to_csv(os.path.join(out, "vif.csv"), header=["VIF"])

    # group correlated features (union of pairs) and drop the highest-VIF member of each group
    upper = X.corr().abs().where(np.triu(np.ones((X.shape[1],) * 2), k=1).astype(bool))
    groups = []
    for a, b in upper.stack()[lambda s: s >= CORR_THRESHOLD].index:
        for g in groups:
            if a in g or b in g:
                g.update([a, b])
                break
        else:
            groups.append({a, b})
    dropped = [compute(X[list(g)]).idxmax() for g in groups]
    pd.Series(dropped, name="dropped_feature").to_csv(os.path.join(out, "vif_pruned.csv"), index=False)
    print(f"VIF pruning would drop {len(dropped)} features: {dropped}")


def distributions(X, y, out):
    os.makedirs(os.path.join(out, "box"), exist_ok=True)
    X.hist(bins=50, figsize=(20, 15))
    _save("feature_hist.png", out)

    cols, n_col = list(X.columns), 4
    fig, axes = plt.subplots((len(cols) + n_col - 1) // n_col, n_col, figsize=(20, 4 * ((len(cols) + n_col - 1) // n_col)))
    for ax, c in zip(axes.ravel(), cols):
        ax.scatter(X[c], y, alpha=0.5, s=6)
        ax.set_title(f"{c} vs target")
    _save("feature_vs_target.png", out)

    for c in cols:
        plt.figure(figsize=(12, 4))
        sns.boxplot(x=y, y=X[c])
        _save(f"box/{c}.png", out)


def hard_class_views(df, out, max_pair_features):
    hard = df[df["target"].isin(HARD_CLASSES)].sort_values("target")
    feats = hard.drop(columns="target")
    os.makedirs(os.path.join(out, "hard_pairs"), exist_ok=True)

    feats.hist(bins=50, figsize=(20, 15))
    _save("hard_hist.png", out)

    for a, b in itertools.permutations(feats.columns[:max_pair_features], 2):
        plt.figure(figsize=(6, 6))
        sns.scatterplot(data=hard, x=a, y=b, hue="target", alpha=0.7, s=10, palette="tab10")
        plt.legend(ncol=3, fontsize=8)
        _save(f"hard_pairs/{a}_{b}.png", out)

    df.groupby(["X_01", "target"]).size().unstack(fill_value=0) \
        .reindex(columns=HARD_CLASSES, fill_value=0) \
        .to_csv(os.path.join(out, "X01_vs_hard_classes.csv"), encoding="utf-8-sig")


STEPS = ["importance", "pca", "corr", "vif", "dist", "hard"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="outputs/eda")
    ap.add_argument("--max-pair-features", type=int, default=10, help="pairwise scatter is O(n^2): only the first N features")
    ap.add_argument("--steps", nargs="+", default=STEPS, choices=STEPS)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(os.path.join(args.data_dir, "train.csv")).drop(columns=["ID"])
    X, y = df.drop(columns="target"), df["target"]

    for step in args.steps:
        print(f"== {step}")
        {"importance": lambda: feature_importance(X, y, args.out_dir),
         "pca": lambda: pca_projection(X, y, args.out_dir),
         "corr": lambda: correlation(X, args.out_dir),
         "vif": lambda: vif(X, args.out_dir),
         "dist": lambda: distributions(X, y, args.out_dir),
         "hard": lambda: hard_class_views(df, args.out_dir, args.max_pair_features)}[step]()


if __name__ == "__main__":
    main()
