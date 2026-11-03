"""Train the 3-stage cascade and write a submission file.

    python scripts/train_predict.py --data-dir data --out outputs/submission.csv
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import N_FOLDS, SEED  # noqa: E402
from src.data import load_submission, load_test, load_train  # noqa: E402
from src.pipeline import CascadeClassifier  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="outputs/submission.csv")
    ap.add_argument("--n-estimators", type=int, default=None, help="override XGB trees (use a small value for a quick run)")
    ap.add_argument("--n-folds", type=int, default=N_FOLDS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--gpu", action="store_true")
    args = ap.parse_args()

    X, y = load_train(args.data_dir)
    ids, X_test = load_test(args.data_dir)

    overrides = {"n_estimators": args.n_estimators} if args.n_estimators else {}
    model = CascadeClassifier(np.unique(y), n_folds=args.n_folds, seed=args.seed,
                              use_gpu=args.gpu, **overrides).fit(X, y)

    sub = load_submission(args.data_dir)
    sub["target"] = model.predict(X_test)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    sub.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
