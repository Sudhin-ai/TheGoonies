"""
features.py
============
Feature-Extraktion aus Gaze-Daten.
Wird von extract_features.py (Training) und predict.py (Vorhersage) geteilt.

Datenformat – Stream-Dict:
    {
        "time_stamps": np.array([t0, t1, ...]),
        "time_series": np.array([[x0,y0], [x1,y1], ...])
    }
"""
import numpy as np
import pandas as pd

FIXATION_VELOCITY_THRESHOLD = 1.0  # normierte Einheiten/Sekunde


# ---------------------------------------------------------------------------
# CSV-Ladefunktionen
# ---------------------------------------------------------------------------

def gaze_csv_to_stream(path):
    """Lädt eine Gaze-CSV als Stream-Dict.

    Unterstützt beide Formate:
      - alt: timestamp,x,y,radius_px
      - neu: timestamp,x,y,radius_px,in_text_region
    """
    df = pd.read_csv(path)
    df = df.rename(columns=str.strip)
    cols = ["x", "y"]
    if "in_text_region" in df.columns:
        cols.append("in_text_region")
    return {
        "info": {"name": ["gaze_csv"], "type": ["gaze"]},
        "time_stamps": df["timestamp"].to_numpy(dtype=float),
        "time_series": df[cols].to_numpy(dtype=float),
    }


def markers_csv_to_segments(path):
    """Liest eine Marker-CSV (timestamp,label,text_id) und gibt eine Liste
    von (text_id, t_start, t_end)-Tupeln zurück."""
    df = pd.read_csv(path)
    df = df.rename(columns=str.strip)
    segments = []
    pending = {}  # text_id → t_start

    for _, row in df.iterrows():
        label = str(row["label"]).strip().lower()
        tid = int(row["text_id"])
        ts = float(row["timestamp"])
        if label == "text_start":
            pending[tid] = ts
        elif label == "text_end" and tid in pending:
            segments.append((tid, pending.pop(tid), ts))

    for tid in pending:
        print(f"  Warnung: Text {tid} hat kein text_end – übersprungen.")

    return sorted(segments)


# ---------------------------------------------------------------------------
# Stream-Hilfsfunktionen
# ---------------------------------------------------------------------------

def crop_stream_to_window(stream, t_start, t_end):
    """Schneidet einen Stream auf ein Zeitfenster zu."""
    t = np.asarray(stream["time_stamps"], dtype=float)
    mask = (t >= t_start) & (t <= t_end)
    return {
        **stream,
        "time_stamps": t[mask],
        "time_series": np.asarray(stream["time_series"])[mask],
    }


# ---------------------------------------------------------------------------
# Feature-Extraktion
# ---------------------------------------------------------------------------

def _stream_to_dataframe(stream, column_names):
    data = np.asarray(stream["time_series"], dtype=float)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    n_channels = data.shape[1] if data.size else 0
    if n_channels != len(column_names):
        raise ValueError(
            f"Stream hat {n_channels} Kanäle, erwartet {len(column_names)} ({column_names})."
        )
    df = pd.DataFrame(data, columns=column_names)
    df["t"] = np.asarray(stream["time_stamps"], dtype=float)
    return df


def _fixation_durations(velocity, timestamps):
    durations = []
    in_fix, fix_start = False, None
    for v, t in zip(velocity < FIXATION_VELOCITY_THRESHOLD, timestamps):
        if v and not in_fix:
            in_fix, fix_start = True, t
        elif not v and in_fix:
            in_fix = False
            durations.append(t - fix_start)
    if in_fix and fix_start is not None:
        durations.append(timestamps.iloc[-1] - fix_start)
    return durations


def extract_gaze(stream):
    """Features: reading_time, fixation_count, fixation_duration_mean,
    gaze_dispersion, gaze_valid_ratio, text_gaze_ratio.

    text_gaze_ratio = Anteil der Samples mit Blick im Textbereich (0–1).
    NaN wenn in_text_region nicht in den Daten vorhanden.
    """
    n_channels = np.asarray(stream["time_series"]).shape[1] if np.asarray(stream["time_series"]).ndim > 1 else 1
    has_region = n_channels >= 3
    col_names = ["x", "y", "in_text_region"] if has_region else ["x", "y"]

    df = _stream_to_dataframe(stream, col_names)
    n_total = len(df)
    df = df.dropna(subset=["x", "y"]).reset_index(drop=True)
    valid_ratio = (len(df) / n_total) if n_total > 0 else np.nan

    text_gaze_ratio = df["in_text_region"].mean() if has_region and len(df) > 0 else np.nan

    empty = {
        "reading_time": np.nan, "fixation_count": np.nan,
        "fixation_duration_mean": np.nan, "gaze_dispersion": np.nan,
        "gaze_valid_ratio": valid_ratio, "text_gaze_ratio": text_gaze_ratio,
    }
    if len(df) < 2:
        return empty

    duration = df["t"].iloc[-1] - df["t"].iloc[0]
    dt = df["t"].diff()
    dist = np.sqrt(df["x"].diff()**2 + df["y"].diff()**2)
    velocity = (dist / dt).replace([np.inf, -np.inf], np.nan).fillna(0)

    fix_dur = _fixation_durations(velocity, df["t"])

    return {
        "reading_time": duration,
        "fixation_count": int((velocity < FIXATION_VELOCITY_THRESHOLD).sum()),
        "fixation_duration_mean": np.mean(fix_dur) if fix_dur else np.nan,
        "gaze_dispersion": df["x"].std() + df["y"].std(),
        "gaze_valid_ratio": valid_ratio,
        "text_gaze_ratio": text_gaze_ratio,
    }
