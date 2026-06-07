from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import StandardScaler


def scale_features(df: pd.DataFrame, feature_cols: list[str]):
    cols = [c for c in feature_cols if c in df.columns]
    if not cols:
        raise ValueError("No valid feature columns found for scaling.")

    X = df[cols].copy()
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True)).fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, scaler
