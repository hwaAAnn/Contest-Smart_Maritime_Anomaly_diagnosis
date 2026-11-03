"""Data loading helpers."""
import os

import pandas as pd


def load_train(data_dir):
    """Return (X, y) from train.csv. X keeps the anonymised columns X_01..X_52."""
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    assert {"ID", "target"}.issubset(train.columns), "train.csv needs ID and target"
    return train.drop(columns=["ID", "target"]), train["target"].to_numpy()


def load_test(data_dir):
    """Return (ID series, X) from test.csv."""
    test = pd.read_csv(os.path.join(data_dir, "test.csv"))
    assert "ID" in test.columns, "test.csv needs ID"
    return test["ID"], test.drop(columns=["ID"])


def load_submission(data_dir):
    return pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))
