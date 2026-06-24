"""
predict.py
==========
Sagt einen Score für eine Person aus einer Gaze-CSV vorher.

Nutzung:
    python predict.py <participant> <gaze.csv> [<markers.csv>] [<head.csv>]
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
    crop_to_overlap,
    extract_gaze,
    extract_head,
    gaze_csv_to_stream,
    head_csv_to_stream,
    markers_csv_to_segments,
)

MODELS_DIR = "models"

DEMO_SAMPLE = {
    "reading_time": 180.0,
    "fixation_count": 400,
    "fixation_duration_mean": 0.25,
    "gaze_dispersion": 0.12,
    "gaze_valid_ratio": 0.95,
    "head_motion": 1.2,
    "head_std": 0.5,
}


def predict_segment(model, feature_cols, gaze_stream, head_stream=None, label=""):
    if head_stream is not None:
        gaze_stream, head_stream = crop_to_overlap(gaze_stream, head_stream)

    row = {}
    row.update(extract_gaze(gaze_stream))
    if head_stream is not None:
        row.update(extract_head(head_stream))

    sample = pd.DataFrame([row])[feature_cols]
    if sample.isna().any(axis=None):
        print(f"  Warnung {label}: mindestens ein Feature ist NaN.")
    score = model.predict(sample)[0]
    print(f"  {label}Vorhergesagter Score: {score:.2f}")
    return score


def main():
    if len(sys.argv) < 3:
        print("Nutzung: python predict.py <participant> <gaze.csv|demo> [markers.csv] [head.csv]")
        sys.exit(1)

    participant = sys.argv[1]
    model_path = f"{MODELS_DIR}/{participant}.pkl"

    try:
        bundle = joblib.load(model_path)
    except FileNotFoundError:
        print(f"Kein Modell für '{participant}' ({model_path}). Zuerst train_model.py ausführen.")
        sys.exit(1)

    model = bundle["model"]
    feature_cols = bundle["feature_cols"]

    if sys.argv[2] == "demo":
        sample = pd.DataFrame([DEMO_SAMPLE])[feature_cols]
        print(f"Demo – vorhergesagter Score ({participant}): {model.predict(sample)[0]:.2f}")
        return

    gaze_path    = sys.argv[2]
    markers_path = sys.argv[3] if len(sys.argv) > 3 else None
    head_path    = sys.argv[4] if len(sys.argv) > 4 else None

    gaze_stream = gaze_csv_to_stream(gaze_path)
    head_stream = head_csv_to_stream(head_path) if head_path else None

    if markers_path:
        segments = markers_csv_to_segments(markers_path)
        for text_id, t_start, t_end in segments:
            seg_gaze = crop_stream_to_window(gaze_stream, t_start, t_end)
            seg_head = crop_stream_to_window(head_stream, t_start, t_end) if head_stream else None
            predict_segment(model, feature_cols, seg_gaze, seg_head, label=f"Text {text_id}: ")
    else:
        predict_segment(model, feature_cols, gaze_stream, head_stream)


if __name__ == "__main__":
    main()
