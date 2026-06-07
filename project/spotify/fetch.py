from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_SOURCE_CSV = Path(__file__).parent / "my_playlist.csv"

AUDIO_FEATURES = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
]

DEFAULT_FEATURE_COLUMNS = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "popularity",
    "duration_ms",
    "explicit",
    "artist_count",
    "release_year",
]

COLUMN_MAP = {
    "Track URI": "track_id",
    "Track Name": "name",
    "Artist Name(s)": "artist",
    "Popularity": "popularity",
    "Duration (ms)": "duration_ms",
    "Explicit": "explicit",
    "Release Date": "release_date",
    "Danceability": "danceability",
    "Energy": "energy",
    "Loudness": "loudness",
    "Speechiness": "speechiness",
    "Acousticness": "acousticness",
    "Instrumentalness": "instrumentalness",
    "Liveness": "liveness",
    "Valence": "valence",
    "Tempo": "tempo",
}


def _clean_bool(value) -> int:
    if pd.isna(value):
        return 0
    if isinstance(value, bool):
        return int(value)
    text = str(value).strip().lower()
    return int(text in {"true", "1", "yes", "y"})


def _artist_count(value) -> int:
    if pd.isna(value):
        return 0
    artists = [part.strip() for part in str(value).split(",")]
    artists = [artist for artist in artists if artist]
    return max(len(artists), 1)


def load_exportify_csv(csv_path: str | Path):
    """
    Load an Exportify CSV and return a cleaned dataframe plus usable feature columns.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV file not found: {csv_path}\n"
            "Export your playlist from Exportify and place the file at this path "
            "or pass --source / --pool on the command line."
        )

    raw = pd.read_csv(csv_path)

    rename_map = {source: target for source, target in COLUMN_MAP.items() if source in raw.columns}
    df = raw[list(rename_map.keys())].rename(columns=rename_map).copy()

    if "track_id" not in df.columns:
        raise ValueError(
            f"{csv_path.name} does not contain a Track URI column, so it cannot be used as an Exportify export."
        )

    df["track_id"] = df["track_id"].astype(str).str.replace("spotify:track:", "", regex=False)
    df["name"] = df.get("name", "").fillna("").astype(str)
    df["artist"] = df.get("artist", "").fillna("").astype(str)

    # Extra features derived from the Exportify export
    df["artist_count"] = raw.get("Artist Name(s)", pd.Series([None] * len(df))).apply(_artist_count)
    df["release_year"] = pd.to_datetime(raw.get("Release Date"), errors="coerce").dt.year

    if "explicit" in df.columns:
        df["explicit"] = df["explicit"].apply(_clean_bool)
    else:
        df["explicit"] = 0

    numeric_cols = [
        "popularity",
        "duration_ms",
        "explicit",
        "artist_count",
        "release_year",
        *AUDIO_FEATURES,
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop unusable rows only if the track id or title is missing.
    df = df[df["track_id"].astype(str).str.len() > 0]
    df = df[df["name"].astype(str).str.len() > 0]

    df = df.drop_duplicates(subset=["track_id"]).set_index("track_id", drop=False)

    feature_cols = [col for col in DEFAULT_FEATURE_COLUMNS if col in df.columns]
    if not feature_cols:
        raise ValueError(f"No usable numeric features found in {csv_path.name}.")

    return df, feature_cols
