"""
train_model.py
===============
Trainiert je einen RandomForestRegressor pro Versuchsperson (within-subject)
auf dataset.csv. Verwendet Leave-One-Out-CV für die Modellbewertung.

Nutzung:
    python train_model.py
    python train_model.py --dataset mein_dataset.csv
"""
import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut, cross_val_score

RANDOM_STATE = 42
MODELS_DIR = "models"

FEATURE_COLS = [
    "reading_time",
    "fixation_count",
    "fixation_duration_mean",
    "gaze_dispersion",
    "gaze_valid_ratio",
    "text_gaze_ratio",   # Anteil der Zeit mit Blick auf den Text (0-1)
    "head_motion",
    "head_std",
]

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="dataset.csv")
args = parser.parse_args()

os.makedirs(MODELS_DIR, exist_ok=True)
df = pd.read_csv(args.dataset)

for participant, group in df.groupby("participant"):
    print(f"\n=== {participant} ===")

    # Nur Features verwenden die für diesen Teilnehmer tatsächlich Daten haben.
    # Komplett-NaN-Spalten (z.B. head_motion ohne AirPods) werden ausgeschlossen.
    available = [c for c in FEATURE_COLS if c in group.columns and group[c].notna().any()]
    missing = [c for c in FEATURE_COLS if c not in available]
    if missing:
        print(f"  Fehlende Features (werden ignoriert): {missing}")

    sub = group[available + ["score"]].dropna().reset_index(drop=True)

    dropped = len(group) - len(sub)
    if dropped:
        print(f"  {dropped} Zeile(n) mit NaN entfernt.")

    n = len(sub)
    if n < 5:
        print(f"  Zu wenige Samples ({n}) – übersprungen.")
        continue
    if n < 20:
        print(f"  Warnung: nur {n} Samples – Metriken sind orientierend.")

    X, y = sub[available], sub["score"]

    loo = LeaveOneOut()
    cv_model = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE)
    r2  = cross_val_score(cv_model, X, y, cv=loo, scoring="r2")
    mae = -cross_val_score(cv_model, X, y, cv=loo, scoring="neg_mean_absolute_error")

    print(f"  LOO-CV R²:  {r2.mean():.3f}  (±{r2.std():.3f})")
    print(f"  LOO-CV MAE: {mae.mean():.3f}  (±{mae.std():.3f})")

    model = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE)
    model.fit(X, y)

    out = os.path.join(MODELS_DIR, f"{participant}.pkl")
    joblib.dump({"model": model, "feature_cols": available}, out)
    print(f"  Gespeichert: {out}")

print("\nFertig.")
