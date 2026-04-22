from .client import get_spotify_client
import pandas as pd

AUDIO_FEATURES = [
    'danceability', 'energy', 'loudness', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
]

def get_playlist_tracks(sp, playlist_id):
    results = sp.playlist_tracks(playlist_id)

    print("DEBUG FIRST RESPONSE:")
    print(results)  # 👈 ADD THIS

    tracks = []

    while results:
        for item in results['items']:
            track = item.get('track')

            if not track or not track.get('id'):
                continue

            tracks.append(track)

        results = sp.next(results) if results.get('next') else None

    return tracks

def get_audio_features(sp, tracks):
    track_ids = [t['id'] for t in tracks if t and t.get('id')]

    all_features = []

    for i in range(0, len(track_ids), 50):
        batch_ids = track_ids[i:i + 50]

        try:
            batch = sp.audio_features(batch_ids)

            if batch:
                all_features.extend([f for f in batch if f is not None])

        except Exception as e:
            print("Audio features error:", e)

    return all_features


def build_dataframe(tracks, audio_features):
    metadata = {
        t['id']: {
            'name': t['name'],
            'artist': t['artists'][0]['name'],
            'popularity': t.get('popularity', 0)
        }
        for t in tracks if t
    }
    print(f"Tracks count: {len(tracks)}")
    print(f"Audio features count: {len(audio_features)}")

    rows = []
    for feature in audio_features:
        track_id = feature['id']
        if track_id in metadata:
            row = {'id': track_id}
            row.update(metadata[track_id])
            row.update({k: feature.get(k, 0) for k in AUDIO_FEATURES})
            rows.append(row)

    df = pd.DataFrame(rows).set_index('id')
    return df


def fetch_playlist(playlist_id):
    playlist_id = clean_playlist_id(playlist_id)
    sp = get_spotify_client()

    tracks = get_playlist_tracks(sp, playlist_id)
    features = get_audio_features(sp, tracks)
    df = build_dataframe(tracks, features)

    if df.empty:
        print("No audio features returned")
        return df, sp   # ✅ ALWAYS return 2 values

    return df, sp

def clean_playlist_id(playlist_id):
    if "playlist/" in playlist_id:
        return playlist_id.split("playlist/")[1].split("?")[0]
    return playlist_id