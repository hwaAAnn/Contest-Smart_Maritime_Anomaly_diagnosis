"""Feature store: raw, PCA, LDA, Gaussian log-likelihood and Mahalanobis features."""
import numpy as np
from numpy.linalg import inv, slogdet
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from .config import COV_EPS, SEED
from .preprocessing import ScalePCA


def fit_class_gaussians(Z, y, classes, eps=COV_EPS):
    """Fit N(mu_c, Sigma_c + eps*I) per class in the PCA space."""
    eye = np.eye(Z.shape[1])
    stats = {}
    for c in classes:
        Zc = Z[y == c]
        cov = np.cov(Zc, rowvar=False) + eps * eye
        stats[c] = (Zc.mean(axis=0), inv(cov), slogdet(cov)[1])  # mu, Sigma^-1, log|Sigma|
    return stats


def _quad_form(Z, mu, inv_cov):
    Zm = Z - mu
    return np.sum(Zm @ inv_cov * Zm, axis=1)


def gaussian_loglik(Z, mu, inv_cov, logdet):
    """-1/2 [(z-mu)^T Sigma^-1 (z-mu) + log|Sigma|]  (constant term dropped)."""
    return -0.5 * (_quad_form(Z, mu, inv_cov) + logdet)


def mahalanobis(Z, mu, inv_cov):
    return np.sqrt(np.clip(_quad_form(Z, mu, inv_cov), 0, None))


class FeatureStore:
    """Builds the concatenated feature matrix fitted on a training set.

    Column blocks (in order): scaled raw (optional) | PCA | LDA | log-likelihood (K) | Mahalanobis (K).
    """

    def __init__(self, classes, include_raw=True, n_pca=None, seed=SEED):
        self.classes = np.asarray(classes)
        self.include_raw = include_raw
        self.rotation = ScalePCA(n_components=n_pca, seed=seed)

    def fit(self, X, y):
        self.rotation.fit(X)
        _, Z = self.rotation.transform(X)
        self.lda = LinearDiscriminantAnalysis(
            n_components=min(len(self.classes) - 1, Z.shape[1])
        ).fit(Z, y)
        self.gauss = fit_class_gaussians(Z, y, self.classes)
        return self

    def transform(self, X):
        Xs, Z = self.rotation.transform(X)
        blocks = [Xs] if self.include_raw else []
        blocks += [Z, self.lda.transform(Z)]
        blocks.append(np.column_stack([gaussian_loglik(Z, *self.gauss[c]) for c in self.classes]))
        blocks.append(np.column_stack([mahalanobis(Z, *self.gauss[c][:2]) for c in self.classes]))
        return np.concatenate(blocks, axis=1)
