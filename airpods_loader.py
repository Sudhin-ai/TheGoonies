"""
airpods_loader.py
==================
Liest einen Sensor-Logger-Export (CSV oder JSON) und gibt ein Objekt
zurück, das wie ein pyxdf-Stream aussieht – damit features.py ihn
ohne Änderungen verarbeiten kann.

Sensor Logger exportiert Orientierungsdaten mit Unix-Zeitstempeln.
Da LabRecorder eine eigene LSL-Uhr verwendet, wird ein Sync-Punkt
benötigt (sync_*.json, erzeugt von send_markers.py).

WICHTIG – Sensor Logger Spaltennamen:
  Gehe in Sensor Logger → Aufzeichnung → CSV öffnen und prüfe, wie
  die Yaw/Pitch/Roll-Spalten heißen. Passe COLUMN_MAP unten an falls
  nötig. Typische Varianten:
    "yaw" / "pitch" / "roll"            (einfach)
    "attitude.yaw" / "attitude.pitch"   (mit Präfix)
    "Yaw (°)" / "Pitch (°)"             (mit Einheit)

Nutzung (wird von extract_features.py automatisch aufgerufen, kann
aber auch standalone getestet werden):
    python airpods_loader.py sync_20240601_120000.json aufnahme_head.csv
"""
import json
import os
import sys

import numpy as np
import pandas as pd

# Spaltennamen im Sensor-Logger-Export anpassen falls nötig
COLUMN_MAP = {
    "time": ["time", "timestamp", "Time", "Timestamp", "seconds_elapsed"],
    "yaw":  ["yaw", "attitude.yaw", "Yaw (°)", "Yaw", "yaw (rad)"],
    "pitch": ["pitch", "attitude.pitch", "Pitch (°)", "Pitch", "pitch (rad)"],
    "roll": ["roll", "attitude.roll", "Roll (°)", "Roll", "roll (rad)"],
}


def _find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_sensor_logger(data_path, sync_path):
    """Lädt eine Sensor-Logger-Datei (CSV oder JSON) und gibt einen
    simulierten LSL-Stream zurück.

    Parameters
    ----------
    data_path : str
        Pfad zur CSV- oder JSON-Datei von Sensor Logger.
    sync_path : str
        Pfad zur sync_*.json-Datei (von send_markers.py).

    Returns
    -------
    dict mit 'time_stamps' (LSL-Uhr) und 'time_series' [[yaw, pitch, roll], ...]
    """
    # Sync-Offset laden
    with open(sync_path) as f:
        sync = json.load(f)
    lsl_offset = sync["lsl_clock"] - sync["unix_time"]  # LSL = Unix + offset

    # Datei einlesen
    ext = os.path.splitext(data_path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(data_path)
    elif ext == ".json":
        raw = json.load(open(data_path))
        # Sensor Logger JSON: {"payload": [{"time": ..., "yaw": ..., ...}, ...]}
        # oder direkt eine Liste
        if isinstance(raw, dict):
            key = next(k for k in raw if isinstance(raw[k], list))
            df = pd.DataFrame(raw[key])
        else:
            df = pd.DataFrame(raw)
    else:
        raise ValueError(f"Unbekanntes Dateiformat: {ext}  (erwartet .csv oder .json)")

    # Spalten zuordnen
    time_col  = _find_col(df, COLUMN_MAP["time"])
    yaw_col   = _find_col(df, COLUMN_MAP["yaw"])
    pitch_col = _find_col(df, COLUMN_MAP["pitch"])
    roll_col  = _find_col(df, COLUMN_MAP["roll"])

    missing = [name for name, col in
               [("time", time_col), ("yaw", yaw_col), ("pitch", pitch_col), ("roll", roll_col)]
               if col is None]
    if missing:
        raise ValueError(
            f"Spalten nicht gefunden: {missing}\n"
            f"Verfügbare Spalten: {list(df.columns)}\n"
            "Passe COLUMN_MAP in airpods_loader.py an."
        )

    unix_times = pd.to_numeric(df[time_col], errors="coerce").values
    lsl_times = unix_times + lsl_offset

    data = df[[yaw_col, pitch_col, roll_col]].apply(
        pd.to_numeric, errors="coerce"
    ).values

    return {
        "info": {"name": ["AirPods_SensorLogger"], "type": ["IMU"]},
        "time_stamps": lsl_times,
        "time_series": data,
    }


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Nutzung: python airpods_loader.py sync_*.json aufnahme.csv")
        sys.exit(1)
    stream = load_sensor_logger(sys.argv[2], sys.argv[1])
    print(f"Geladen: {len(stream['time_stamps'])} Samples")
    print(f"Zeitraum: {stream['time_stamps'][0]:.2f} – {stream['time_stamps'][-1]:.2f} (LSL-Uhr)")
    print(f"Erste Zeile yaw/pitch/roll: {stream['time_series'][0]}")
