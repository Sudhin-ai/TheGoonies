"""
features.py
============
Feature-Extraktion aus Gaze- und Kopfbewegungs-Daten.
Wird von extract_features.py (Training) und predict.py (Vorhersage) geteilt.

Datenformat – Stream-Dict:
    {
        "time_stamps": np.array([t0, t1, ...]),
        "time_series": np.array([[x0,y0], [x1,y1], ...])  # Gaze: x,y
    }                                                       # Head: yaw,pitch,roll

Dieses Format wird sowohl von pyxdf als auch von den CSV-Ladefunktionen
in diesem Modul erzeugt, sodass die Feature-Funktionen unverändert bleiben.
"""
import numpy as np
import pandas as pd

FIXATION_VELOCITY_THRESHOLD = 1.0  # normierte Einheiten/Sekunde


# ---------------------------------------------------------------------------
# CSV-Ladefunktionen (ersetzen pyxdf für die reine CSV-Pipeline)
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


def head_csv_to_stream(data_path, sync_path=None):
    """Lädt eine AirPods/Sensor-Logger-CSV als Stream-Dict.

    Wenn sync_path angegeben: rechnet Sensor-Logger-Unix-Zeit in dieselbe
    Zeitbasis wie die Gaze-CSV um (beide verwenden time.time()).
    Bei gleichem Zeitbasis-System (selber PC) ist kein Offset nötig.
    """
    import json, os

    df = pd.read_csv(data_path)
    df = df.rename(columns=str.strip)

    # Spaltennamen normalisieren (Sensor Logger variiert)
    col_map = {
        "time":  ["time", "timestamp", "Time", "Timestamp", "seconds_elapsed"],
        "yaw":   ["yaw", "attitude.yaw", "Yaw (°)", "Yaw", "yaw (rad)"],
        "pitch": ["pitch", "attitude.pitch", "Pitch (°)", "Pitch", "pitch (rad)"],
        "roll":  ["roll", "attitude.roll", "Roll (°)", "Roll", "roll (rad)"],
    }

    def find(candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    t_col = find(col_map["time"])
    y_col = find(col_map["yaw"])
    p_col = find(col_map["pitch"])
    r_col = find(col_map["roll"])

    missing = [k for k, c in [("time", t_col), ("yaw", y_col), ("pitch", p_col), ("roll", r_col)] if c is None]
    if missing:
        raise ValueError(
            f"Spalten nicht gefunden: {missing}\n"
            f"Verfügbare Spalten: {list(df.columns)}\n"
            "Passe col_map in features.py → head_csv_to_stream() an."
        )

    timestamps = pd.to_numeric(df[t_col], errors="coerce").to_numpy(dtype=float)

    # Zeitoffset anwenden falls Sensor Logger andere Epoch verwendet
    if sync_path and os.path.exists(sync_path):
        with open(sync_path) as f:
            sync = json.load(f)
        # Offset: LSL/Unix-Zeit zur Sensor-Logger-Zeit
        offset = sync.get("unix_time", sync.get("lsl_clock", 0)) - timestamps[0]
        # Nur korrigieren wenn Offset > 1 Sekunde (sonst selbe Zeitbasis)
        if abs(offset) > 1.0:
            timestamps = timestamps + offset

    data = df[[y_col, p_col, r_col]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    return {
        "info": {"name": ["head_csv"], "type": ["IMU"]},
        "time_stamps": timestamps,
        "time_series": data,
    }


# ---------------------------------------------------------------------------
# Stream-Hilfsfunktionen (kompatibel mit pyxdf-Format)
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


def crop_to_overlap(stream_a, stream_b):
    """Schneidet zwei Streams auf ihr gemeinsames Zeitfenster zu."""
    ta = np.asarray(stream_a["time_stamps"], dtype=float)
    tb = np.asarray(stream_b["time_stamps"], dtype=float)

    if len(ta) == 0 or len(tb) == 0:
        return stream_a, stream_b

    t0 = max(ta[0], tb[0])
    t1 = min(ta[-1], tb[-1])

    if t0 >= t1:
        print("  Warnung: Streams überlappen sich zeitlich nicht – kein Zuschnitt.")
        return stream_a, stream_b

    def _crop(stream, t):
        mask = (t >= t0) & (t <= t1)
        return {**stream, "time_stamps": t[mask], "time_series": np.asarray(stream["time_series"])[mask]}

    return _crop(stream_a, ta), _crop(stream_b, tb)


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

    text_gaze_ratio = Anteil der Samples, bei denen der Blick im Textbereich
    war. 1.0 = immer auf Text geschaut, 0.0 = nie. NaN wenn in_text_region
    nicht in den Daten vorhanden (ältere CSVs).
    """
    # Kanal-Layout: x, y, [in_text_region optional]
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


def extract_head(stream):
    """Features: head_motion, head_std."""
    df = _stream_to_dataframe(stream, ["yaw", "pitch", "roll"])
    df = df.dropna(subset=["yaw", "pitch", "roll"]).reset_index(drop=True)
    if len(df) < 2:
        return {"head_motion": np.nan, "head_std": np.nan}
    motion = np.sqrt(df["yaw"].diff()**2 + df["pitch"].diff()**2 + df["roll"].diff()**2).fillna(0)
    return {"head_motion": motion.mean(), "head_std": motion.std()}
