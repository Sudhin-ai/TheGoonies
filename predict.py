"""
predict.py
==========
Sagt einen Score für eine Person aus einer Gaze-CSV vorher.

Nutzung:
    python predict.py sudhin demo   (Demo-Werte)

Einzelner Text (ohne Marker):
    python predict.py sudhin raw_data/sudhin_gaze.csv
    → wertet die gesamte CSV als einen Text aus

Mit Marker-Segmentierung:
    python predict.py sudhin raw_data/sudhin_gaze.csv raw_data/sudhin_markers.csv
    → gibt einen Score pro Text aus
"""
import sys

import joblib
import numpy as np
import pandas as pd

from features import (
    crop_stream_to_window,
    extract_gaze,
    gaze_csv_to_stream,
    markers_csv_to_segments,
)

MODELS_DIR = "models"

DEMO_SAMPLE = {
    "reading_time": 180.0,
    "fixation_count": 400,
    "fixation_duration_mean": 0.25,
    "gaze_dispersion": 0.12,
    "gaze_valid_ratio": 0.95,
    "text_gaze_ratio": 0.88,
}


def predict_segment(model, feature_cols, gaze_stream, label=""):
    row = extract_gaze(gaze_stream)
    sample = pd.DataFrame([row])[feature_cols]
    if sample.isna().any(axis=None):
        print(f"  Warnung {label}: mindestens ein Feature ist NaN.")
    score = model.predict(sample)[0]
    print(f"  {label}Vorhergesagter Score: {score:.2f}")
    return score


def main():
    if len(sys.argv) < 3:
        print("Nutzung: python predict.py <participant> <gaze.csv|demo> [markers.csv]")
        sys.exit(1)

    participant = sys.argv[1]
    model_path  = f"{MODELS_DIR}/{participant}.pkl"

    try:
        bundle = joblib.load(model_path)
    except FileNotFoundError:
        pooled = f"{MODELS_DIR}/pooled.pkl"
        if os.path.exists(pooled):
            print(f"Kein within-subject Modell für '{participant}' – verwende gepooltes Modell.")
            bundle = joblib.load(pooled)
        else:
            print(f"Kein Modell gefunden ({model_path}). Zuerst train_model.py ausführen.")
            sys.exit(1)

    model        = bundle["model"]
    feature_cols = bundle["feature_cols"]

    if sys.argv[2] == "demo":
        sample = pd.DataFrame([DEMO_SAMPLE])[feature_cols]
        print(f"Demo – vorhergesagter Score ({participant}): {model.predict(sample)[0]:.2f}")
        return

    gaze_path    = sys.argv[2]
    markers_path = sys.argv[3] if len(sys.argv) > 3 else None

    gaze_stream = gaze_csv_to_stream(gaze_path)

    if markers_path:
        segments = markers_csv_to_segments(markers_path)
        for text_id, t_start, t_end in segments:
            seg = crop_stream_to_window(gaze_stream, t_start, t_end)
            predict_segment(model, feature_cols, seg, label=f"Text {text_id}: ")
    else:
        predict_segment(model, feature_cols, gaze_stream)


if __name__ == "__main__":
    main()
