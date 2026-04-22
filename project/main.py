import os
from dotenv import load_dotenv
from pathlib import Path

from spotify.fetch import fetch_playlist
from ml.preprocessing import scale_features
from ml.clustering import train_kmeans

# load .env
load_dotenv(dotenv_path=Path(__file__).parent / '.env')

FEATURE_COLS = [
    'danceability', 'energy', 'loudness', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
]

def main():
    playlist_id = "64723tL7qpQhJCMgSWHHnQ"

    print("Fetching playlist...")
    df, sp = fetch_playlist(playlist_id)

    print("Scaling features...")
    X, scaler = scale_features(df, FEATURE_COLS)

    print("Clustering...")
    model, labels = train_kmeans(X, k=5)

    df['cluster'] = labels

    print("\nSample results:")
    print(df[['name', 'artist', 'cluster']].head(10))


if __name__ == "__main__":
    main()