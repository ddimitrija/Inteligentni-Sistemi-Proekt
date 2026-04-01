import os
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv
import pandas as pd

# Load credentials from .env file so we never hardcode secrets in source code
load_dotenv()

# These are the 9 audio features Spotify computes for every track.
# We define them here as a constant so every other module can import
AUDIO_FEATURES = [
    'danceability', 'energy', 'loudness', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
]


def get_spotify_client():
    """
    Creates and returns an authenticated Spotify client.
    Uses the Authorization Code Flow, which lets users log in
    and grant access to their own playlists.
    """
    return spotipy.Spotify(auth_manager=SpotifyOAuth(
        client_id=os.getenv("SPOTIPY_CLIENT_ID"),
        client_secret=os.getenv("SPOTIPY_CLIENT_SECRET"),
        redirect_uri=os.getenv("SPOTIPY_REDIRECT_URI", "http://localhost:8888/callback"),
        scope="playlist-read-private playlist-read-collaborative"
    ))


def get_playlist_tracks(sp, playlist_id):
    """
    Fetches all tracks from a playlist, handling Spotify's pagination.

    Spotify only returns 100 tracks per request, so for longer playlists
    we need to keep asking for the next "page" until there are no more.
    Returns a list of track objects.
    """
    results = sp.playlist_tracks(playlist_id)
    tracks = results['items']

    # Keep fetching as long as Spotify tells us there's a next page
    while results['next']:
        results = sp.next(results)
        tracks.extend(results['items'])

    # Filter out any None entries (can happen if a track was removed)
    return [t for t in tracks if t['track'] and t['track']['id']]


def get_audio_features(sp, tracks):
    """
    Given a list of track objects, fetches audio features for all of them.

    The Spotify API limits audio feature requests to 100 tracks at a time,
    so we batch them. We also filter out any None results (Spotify occasionally
    returns None for tracks it can't analyze, like local files or podcasts).
    """
    track_ids = [t['track']['id'] for t in tracks]
    all_features = []

    # Process in batches of 100 — Spotify's hard limit per request
    for i in range(0, len(track_ids), 100):
        batch = sp.audio_features(track_ids[i:i + 100])
        # Filter out None values before extending
        all_features.extend([f for f in batch if f is not None])

    return all_features


def build_dataframe(tracks, audio_features):
    """
    Combines track metadata and audio features into a single clean DataFrame.

    Each row is one song. We include the track ID as the index (so we can
    look songs up later), plus the name and artist for human-readable output,
    plus all 9 audio features for the ML model.
    """
    # Build a lookup dict from track_id -> metadata for fast merging
    metadata = {
        t['track']['id']: {
            'name': t['track']['name'],
            'artist': t['track']['artists'][0]['name'],
            'popularity': t['track']['popularity']
        }
        for t in tracks
    }

    rows = []
    for feature in audio_features:
        track_id = feature['id']
        if track_id in metadata:
            row = {'id': track_id}
            row.update(metadata[track_id])  # name, artist, popularity
            row.update({k: feature[k] for k in AUDIO_FEATURES})  # audio features
            rows.append(row)

    df = pd.DataFrame(rows).set_index('id')
    return df


def fetch_playlist(playlist_id):
    """
    The main public function — the only one most of your code will call.
    Pass in a playlist ID, get back a clean DataFrame ready for ML.
    """
    sp = get_spotify_client()
    tracks = get_playlist_tracks(sp, playlist_id)
    features = get_audio_features(sp, tracks)
    df = build_dataframe(tracks, features)

    print(f"Fetched {len(df)} tracks from playlist.")
    return df, sp  # We return sp too so other modules can reuse the connection
