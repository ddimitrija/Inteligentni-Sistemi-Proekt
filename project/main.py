import os
from pathlib import Path
from dotenv import load_dotenv

from spotify.fetch import fetch_playlist
from ml.preprocessing import scale_features
from ml.clustering import train_kmeans


# =========================
# CONFIG
# =========================
PLAYLIST_ID = "64723tL7qpQhJCMgSWHHnQ"
NUM_CLUSTERS = 5

FEATURE_COLS = [
    'danceability',
    'energy',
    'loudness',
    'speechiness',
    'acousticness',
    'instrumentalness',
    'liveness',
    'valence',
    'tempo'
]


# load environment variables
load_dotenv(dotenv_path=Path(__file__).parent / '.env')


def print_cluster_stats(df):
    print("\nSongs per cluster:")

    cluster_counts = df['cluster'].value_counts().sort_index()

    for cluster, count in cluster_counts.items():
        print(f"Cluster {cluster}: {count} songs")


def show_sample_songs(df, amount=3):
    print("\nExample songs from each cluster:\n")

    for cluster in sorted(df['cluster'].unique()):
        print(f"--- Cluster {cluster} ---")

        songs = df[df['cluster'] == cluster][['name', 'artist']].head(amount)

        for _, row in songs.iterrows():
            print(f"{row['name']} - {row['artist']}")

        print()


def main():
    print("Starting playlist analysis...\n")

    # fetch spotify playlist
    print("Fetching playlist data...")
    df, sp = fetch_playlist(PLAYLIST_ID)

    print(f"Loaded {len(df)} songs.\n")

    # preprocess data
    print("Scaling audio features...")
    X, scaler = scale_features(df, FEATURE_COLS)

    # train clustering model
    print(f"Training KMeans model with {NUM_CLUSTERS} clusters...")
    model, labels = train_kmeans(X, k=NUM_CLUSTERS)

    # save labels
    df['cluster'] = labels

    # results
    print_cluster_stats(df)
    show_sample_songs(df)

    print("Finished.")


if __name__ == "__main__":
    main()
