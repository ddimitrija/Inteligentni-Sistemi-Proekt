import os
import spotipy
from spotipy.oauth2 import SpotifyOAuth

def get_spotify_client():
    scope = "playlist-read-private playlist-read-collaborative user-read-private"

    auth_manager = SpotifyOAuth(
        client_id=os.getenv("SPOTIPY_CLIENT_ID"),
        client_secret=os.getenv("SPOTIPY_CLIENT_SECRET"),
        redirect_uri=os.getenv("SPOTIPY_REDIRECT_URI"),
        scope=scope,
        cache_path=".cache",
        show_dialog=True,
        open_browser=True
    )

    sp = spotipy.Spotify(auth_manager=auth_manager)

    try:
        user = sp.current_user()
        print("AUTH OK:", user["display_name"])
    except Exception as e:
        print("AUTH FAILED:", e)

    return sp