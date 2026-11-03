"""Scaling and full-rank PCA rotation."""
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .config import SEED


class ScalePCA:
    """StandardScaler followed by a PCA that keeps every component.

    The PCA is used as an orthogonal rotation (decorrelation), not as a
    dimensionality reduction, so no information is dropped.
    """

    def __init__(self, n_components=None, seed=SEED):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components, random_state=seed)

    def fit(self, X):
        self.pca.fit(self.scaler.fit_transform(X))
        return self

    def scale(self, X):
        return self.scaler.transform(X)

    def transform(self, X):
        """Return (scaled X, PCA scores)."""
        Xs = self.scaler.transform(X)
        return Xs, self.pca.transform(Xs)
