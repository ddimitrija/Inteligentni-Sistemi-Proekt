from __future__ import annotations

import argparse
import webbrowser
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

from ml.clustering import choose_best_k, train_kmeans
from ml.preprocessing import scale_features
from ml.reference_cache import CACHE_PATH, build_cache, load_cache
from spotify.fetch import DEFAULT_SOURCE_CSV, load_exportify_csv

OUTPUT_HTML = Path(__file__).with_name("recommendation_report.html")
OUTPUT_CSV  = Path(__file__).with_name("recommendations.csv")


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_source(source_csv: Path) -> tuple[pd.DataFrame, list[str]]:
    df, feature_cols = load_exportify_csv(source_csv)
    return df, feature_cols


def get_pool(cache_path: Path = CACHE_PATH) -> dict:
    """Return the pre-computed reference cache, building it if missing."""
    cache = load_cache(cache_path)
    if cache is None:
        print("No reference cache found — building it now (runs once) ...")
        cache = build_cache()
    return cache


def recommend(
    source_df: pd.DataFrame,
    source_features: list[str],
    pool_cache: dict,
    recs_per_cluster: int = 5,
) -> tuple[pd.DataFrame, dict]:
    """
    Cluster source tracks using the pre-fitted pool scaler/model,
    then pick the closest pool tracks from each cluster as recommendations.
    """
    pool_df     = pool_cache["pool_df"].copy()
    scaler      = pool_cache["scaler"]
    model       = pool_cache["model"]
    feature_cols = [c for c in pool_cache["feature_cols"] if c in source_features]

    if not feature_cols:
        raise ValueError("No shared features between source playlist and reference cache.")

    # Scale source tracks with the already-fitted pool scaler
    source_X = source_df[feature_cols].apply(pd.to_numeric, errors="coerce")
    source_X = source_X.fillna(source_X.median()).fillna(0)
    source_scaled = scaler.transform(source_X)

    # Assign source tracks to pool clusters
    source_labels = model.predict(source_scaled)
    source_df = source_df.copy()
    source_df["cluster"] = source_labels

    # Pre-compute pool distances (already done in cache, but re-derive for shared feature subset)
    pool_X = pool_df[feature_cols].apply(pd.to_numeric, errors="coerce")
    pool_X = pool_X.fillna(pool_X.median()).fillna(0)
    pool_scaled = scaler.transform(pool_X)
    pool_labels = model.predict(pool_scaled)
    pool_df["cluster"] = pool_labels
    centroids = model.cluster_centers_
    pool_df["distance"] = np.linalg.norm(pool_scaled - centroids[pool_labels], axis=1)

    source_ids = set(source_df["track_id"].astype(str))
    candidates = pool_df[~pool_df["track_id"].astype(str).isin(source_ids)].copy()

    # Index pool_scaled rows by pool_df's positional index for fast lookup
    pool_idx_map = {idx: pos for pos, idx in enumerate(pool_df.index)}

    all_recs = []
    cluster_payloads = []

    for cid in sorted(source_df["cluster"].unique()):
        src_rows  = source_df[source_df["cluster"] == cid]
        pool_rows = candidates[candidates["cluster"] == cid].copy()

        # Centroid of THIS source playlist's tracks in this cluster
        src_positions = [i for i, idx in enumerate(source_df.index) if source_df.at[idx, "cluster"] == cid]
        source_centroid = source_scaled[src_positions].mean(axis=0)

        # Re-rank pool tracks by distance to the source centroid (not the global pool centroid)
        if not pool_rows.empty:
            positions = [pool_idx_map[idx] for idx in pool_rows.index]
            pool_rows = pool_rows.copy()
            pool_rows["distance"] = np.linalg.norm(pool_scaled[positions] - source_centroid, axis=1)
            pool_rows = pool_rows.sort_values(
                ["distance", "popularity"], ascending=[True, False], na_position="last"
            ).head(recs_per_cluster)

        for _, row in pool_rows.iterrows():
            all_recs.append({
                "track_id":           str(row["track_id"]),
                "name":               str(row.get("name", "")),
                "artist":             str(row.get("artist", "")),
                "cluster":            int(cid),
                "distance":           round(float(row["distance"]), 4),
                "popularity":         None if pd.isna(row.get("popularity")) else int(row["popularity"]),
                "source_cluster_size": int(len(src_rows)),
            })

        cluster_payloads.append({
            "cluster_id":    int(cid),
            "source_count":  int(len(src_rows)),
            "pool_count":    int(len(pool_rows)),
            "source_examples": src_rows.head(5),
            "recommendations": pool_rows,
        })

    # Deduplicate by track_id, keep best distance
    seen: dict[str, dict] = {}
    for r in all_recs:
        tid = r["track_id"]
        if tid not in seen or r["distance"] < seen[tid]["distance"]:
            seen[tid] = r

    recs_df = pd.DataFrame(list(seen.values()))
    if not recs_df.empty:
        recs_df = recs_df.sort_values(
            ["cluster", "distance", "popularity"], ascending=[True, True, False], na_position="last"
        ).reset_index(drop=True)

    report = {
        "source_count":    len(source_df),
        "pool_count":      len(pool_df),
        "best_k":          model.n_clusters,
        "scores":          pool_cache["scores"],
        "cluster_payloads": cluster_payloads,
        "feature_cols":    feature_cols,
        "recommendations_df": recs_df,
    }
    return recs_df, report


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

_CSS = """
:root {
    --bg: #0f172a; --panel: #111827; --text: #e5e7eb;
    --muted: #9ca3af; --accent: #7c3aed; --border: rgba(255,255,255,0.08);
}
* { box-sizing: border-box; }
body {
    margin: 0;
    font-family: Inter, ui-sans-serif, system-ui, sans-serif;
    background: radial-gradient(circle at top, #1e293b 0%, var(--bg) 55%);
    color: var(--text); line-height: 1.5;
}
.wrap { max-width: 1100px; margin: 0 auto; padding: 32px 20px 48px; }
.hero {
    background: linear-gradient(135deg, rgba(124,58,237,.28), rgba(6,182,212,.18));
    border: 1px solid var(--border); border-radius: 24px;
    padding: 28px; box-shadow: 0 20px 60px rgba(0,0,0,.25);
}
h1 { margin: 0 0 8px; font-size: 2rem; }
.muted { color: var(--muted); }
.grid { display: grid; grid-template-columns: repeat(auto-fit,minmax(200px,1fr)); gap:14px; margin-top:18px; }
.card {
    background: rgba(17,24,39,.9); border: 1px solid var(--border);
    border-radius: 20px; padding: 18px; box-shadow: 0 16px 40px rgba(0,0,0,.18);
}
.metric-title { color: var(--muted); font-size:.9rem; }
.metric-value { font-size:1.8rem; font-weight:800; margin-top:6px; }
.section { margin-top: 24px; }
.section h2 { margin: 0 0 12px; font-size:1.3rem; }
.pill {
    display:inline-block; padding:5px 10px; border-radius:999px;
    background:rgba(124,58,237,.16); border:1px solid rgba(124,58,237,.35);
    font-size:.85rem; margin: 0 6px 6px 0;
}
.cluster-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); gap:14px; }
.cluster-head { display:flex; justify-content:space-between; align-items:baseline; }
.cluster-head h3 { margin:0; font-size:1.1rem; }
.badge {
    padding:5px 10px; border-radius:999px;
    background:rgba(6,182,212,.12); border:1px solid rgba(6,182,212,.3);
    color:#cffafe; font-size:.8rem;
}
.song-list { margin:8px 0 0; padding-left:18px; }
.song-list li { margin:5px 0; font-size:.92rem; }
.empty { color:var(--muted); font-style:italic; padding:8px 0; }
.table-wrap { overflow-x:auto; border-radius:18px; border:1px solid var(--border); }
table { width:100%; border-collapse:collapse; background:rgba(17,24,39,.92); }
th,td { padding:11px 14px; border-bottom:1px solid var(--border); text-align:left; }
th { color:#f9fafb; background:rgba(255,255,255,.02); }
tr:last-child td { border-bottom:none; }
code { background:rgba(255,255,255,.08); padding:2px 6px; border-radius:6px; font-size:.88rem; }
.footer { margin-top:20px; color:var(--muted); font-size:.88rem; }
"""


def _song_list(rows: pd.DataFrame) -> str:
    if rows.empty:
        return "<div class='empty'>None</div>"
    items = "".join(
        f"<li><strong>{escape(str(r.get('name','')))} </strong>"
        f"<span class='muted'>— {escape(str(r.get('artist','Unknown')))}</span></li>"
        for _, r in rows.iterrows()
    )
    return f"<ul class='song-list'>{items}</ul>"


def render_html(report: dict, source_name: str, output_html: Path) -> None:
    recs   = report["recommendations_df"]
    scores = report["scores"]
    k      = report["best_k"]

    score_pills = "".join(
        f"<span class='pill'>k={k}: {s:.3f}</span>"
        for k, s in sorted(scores.items())
    )

    # Metric cards
    def card(title, value, sub=""):
        return (f"<div class='card'>"
                f"<div class='metric-title'>{escape(title)}</div>"
                f"<div class='metric-value'>{escape(str(value))}</div>"
                f"<div class='muted' style='font-size:.85rem'>{escape(sub)}</div></div>")

    metrics = "".join([
        card("Source tracks",          report["source_count"], "tracks analyzed"),
        card("Reference pool",         report["pool_count"],   "candidate tracks"),
        card("Clusters",               k,                      "chosen by silhouette"),
        card("Recommendations",        0 if recs.empty else len(recs), "new songs"),
    ])

    # Recommendations table
    if recs.empty:
        rec_html = "<div class='card empty'>No recommendations found — the pool may overlap entirely with your source playlist.</div>"
    else:
        rows_html = "".join(
            f"<tr><td><strong>{escape(str(r['name']))}</strong>"
            f"<div class='muted'>{escape(str(r['artist']))}</div></td>"
            f"<td>Cluster {int(r['cluster'])}</td>"
            f"<td>{float(r['distance']):.3f}</td>"
            f"<td>{'' if pd.isna(r.get('popularity')) else int(r['popularity'])}</td></tr>"
            for _, r in recs.iterrows()
        )
        rec_html = (
            "<div class='table-wrap'><table>"
            "<thead><tr><th>Song</th><th>Cluster</th><th>Distance</th><th>Popularity</th></tr></thead>"
            f"<tbody>{rows_html}</tbody></table></div>"
        )

    # Cluster cards
    cluster_cards = ""
    for p in report["cluster_payloads"]:
        cluster_cards += (
            f"<div class='card'>"
            f"<div class='cluster-head'><h3>Cluster {p['cluster_id']}</h3>"
            f"<span class='badge'>{p['source_count']} source · {p['pool_count']} recs</span></div>"
            f"<p style='font-size:.88rem;margin:10px 0 4px'><strong>Source songs</strong></p>"
            f"{_song_list(p['source_examples'])}"
            f"<p style='font-size:.88rem;margin:10px 0 4px'><strong>Recommended</strong></p>"
            f"{_song_list(p['recommendations'])}"
            f"</div>"
        )

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Music Recommendation Report</title>
<style>{_CSS}</style>
</head>
<body>
<div class="wrap">
  <div class="hero">
    <h1>Music Recommendation Report</h1>
    <p class="muted">Source: <code>{escape(source_name)}</code> · Pool: <code>reference.csv</code></p>
    <p class="muted">Features: {escape(", ".join(report["feature_cols"]))}</p>
    <div style="margin-top:10px">{score_pills}</div>
    <div class="grid">{metrics}</div>
  </div>

  <div class="section">
    <h2>Recommendations</h2>
    {rec_html}
  </div>

  <div class="section">
    <h2>Clusters</h2>
    <div class="cluster-grid">{cluster_cards}</div>
  </div>

  <div class="footer">Generated from Exportify CSVs. Reference pool clustered once and cached.</div>
</div>
</body>
</html>"""

    output_html.write_text(html, encoding="utf-8")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Cluster a playlist and recommend new songs from reference.csv.")
    parser.add_argument("--source",          default=str(DEFAULT_SOURCE_CSV), help="Exportify CSV to analyse.")
    parser.add_argument("--recommendations", type=int, default=5,             help="Recommendations per cluster.")
    parser.add_argument("--rebuild-cache",   action="store_true",             help="Force-rebuild the reference cache.")
    parser.add_argument("--output-html",     default=str(OUTPUT_HTML),        help="HTML report path.")
    parser.add_argument("--output-csv",      default=str(OUTPUT_CSV),         help="CSV output path.")
    parser.add_argument("--open-browser",    action="store_true",             help="Open report in browser.")
    args = parser.parse_args()

    source_csv  = Path(args.source)
    output_html = Path(args.output_html)
    output_csv  = Path(args.output_csv)

    # Load / build reference cache
    if args.rebuild_cache and CACHE_PATH.exists():
        CACHE_PATH.unlink()
        print("Deleted old cache.")

    pool_cache = get_pool()

    # Load source playlist
    print(f"\nLoading source playlist: {source_csv.name} ...")
    source_df, source_features = load_source(source_csv)
    print(f"  {len(source_df)} tracks, {len(source_features)} features")

    # Run recommendations
    print("\nMatching source tracks to reference clusters ...")
    recs_df, report = recommend(source_df, source_features, pool_cache, recs_per_cluster=args.recommendations)
    print(f"  {len(recs_df)} unique recommendations across {report['best_k']} clusters")

    # Save outputs
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_html.parent.mkdir(parents=True, exist_ok=True)

    cols = ["track_id", "name", "artist", "cluster", "distance", "popularity", "source_cluster_size"]
    (recs_df if not recs_df.empty else pd.DataFrame(columns=cols)).to_csv(output_csv, index=False)

    render_html(report, source_csv.name, output_html)
    print(f"\nSaved: {output_csv.name}, {output_html.name}")

    if args.open_browser:
        webbrowser.open(output_html.resolve().as_uri())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
