from sklearn.cluster import KMeans

def train_kmeans(X, k=5):
    model = KMeans(n_clusters=k, random_state=42)
    labels = model.fit_predict(X)
    return model, labels