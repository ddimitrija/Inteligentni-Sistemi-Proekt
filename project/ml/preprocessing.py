from sklearn.preprocessing import StandardScaler

def scale_features(df, feature_cols):
    scaler = StandardScaler()
    X = scaler.fit_transform(df[feature_cols])
    return X, scaler