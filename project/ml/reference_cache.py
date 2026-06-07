"""
Pre-compute and cache KMeans clustering for reference.csv.

Run this script once:
    python -m ml.reference_cache

It writes spotify/reference_cache.joblib next to reference.csv.
main.py loads it automatically on future runs — no re-fitting needed.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from ml.clustering import choose_best_k, train_kmeans
from ml.preprocessing import scale_features
from spotify.fetch import load_exportify_csv

REFERENCE_CSV = Path(__file__).parents[1] / "spotify" / "reference.csv"
CACHE_PATH = REFERENCE_CSV.with_suffix(".joblib")


def build_cache(
    reference_csv: Path = REFERENCE_CSV,
    cache_path: Path = CACHE_PATH,
    min_k: int = 2,
    max_k: int = 8,
) -> dict:
    print(f"Loading {reference_csv.name} ...")
    pool_df, feature_cols = load_exportify_csv(reference_csv)

    print(f"Scaling {len(pool_df)} tracks × {len(feature_cols)} features ...")
    X_scaled, scaler = scale_features(pool_df, feature_cols)

    print(f"Selecting best k in [{min_k}, {max_k}] by silhouette score ...")
    best_k, scores = choose_best_k(X_scaled, min_k=min_k, max_k=max_k)
    print(f"  → best k = {best_k}  (scores: {', '.join(f'k={k}: {s:.3f}' for k, s in sorted(scores.items()))})")

    model, labels = train_kmeans(X_scaled, k=best_k)

    # Compute centroid distances for every track
    centroids = model.cluster_centers_
    distances = np.linalg.norm(X_scaled - centroids[labels], axis=1)

    cache = {
        "pool_df": pool_df,
        "feature_cols": feature_cols,
        "scaler": scaler,
        "model": model,
        "labels": labels,
        "distances": distances,
        "best_k": best_k,
        "scores": scores,
    }

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(cache, cache_path)
    print(f"Cache saved → {cache_path}")
    return cache


def load_cache(cache_path: Path = CACHE_PATH) -> dict | None:
    """Return cached data if the cache file exists, otherwise None."""
    if cache_path.exists():
        return joblib.load(cache_path)
    return None


if __name__ == "__main__":
    build_cache()
