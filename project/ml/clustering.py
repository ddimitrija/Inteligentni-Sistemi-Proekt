from __future__ import annotations

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def choose_best_k(X, min_k: int = 2, max_k: int = 8, random_state: int = 42):
    """
    Pick the cluster count with the best silhouette score.

    To keep the project responsive on small classroom datasets, we score a
    sampled subset instead of the full matrix.
    Returns (best_k, scores_dict).
    """
    n_samples = len(X)
    if n_samples < min_k:
        return 1, {}

    upper = min(max_k, n_samples - 1)
    if upper < min_k:
        return min_k, {}

    best_k = min_k
    best_score = float("-inf")
    scores = {}

    sample_size = min(150, n_samples)

    for k in range(min_k, upper + 1):
        model = KMeans(
            n_clusters=k,
            random_state=random_state,
            n_init=3,
            max_iter=100,
            algorithm="lloyd",
        )
        labels = model.fit_predict(X)

        if len(set(labels)) < 2:
            continue

        score = silhouette_score(
            X,
            labels,
            sample_size=sample_size,
            random_state=random_state,
        )
        scores[k] = float(score)

        if score > best_score:
            best_score = score
            best_k = k

    if not scores:
        best_k = min(max(2, min_k), upper)

    return best_k, scores


def train_kmeans(X, k: int = 5, random_state: int = 42):
    model = KMeans(
        n_clusters=k,
        random_state=random_state,
        n_init=3,
        max_iter=100,
        algorithm="lloyd",
    )
    labels = model.fit_predict(X)
    return model, labels
