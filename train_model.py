"""
train_model.py
===============
Trainiert je einen RandomForestRegressor pro Versuchsperson (within-subject)
sowie ein gepooltes Modell über alle Teilnehmer (cross-subject).
Verwendet Leave-One-Out-CV für die Modellbewertung.

Nutzung:
    python train_model.py
    python train_model.py --dataset mein_dataset.csv
"""
import argparse
import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut, cross_val_score

RANDOM_STATE = 42
MODELS_DIR = "models"
MIN_SAMPLES = 3

FEATURE_COLS = [
    "reading_time",
    "fixation_count",
    "fixation_duration_mean",
    "gaze_dispersion",
    "gaze_valid_ratio",
    "text_gaze_ratio",
]

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="dataset.csv")
args = parser.parse_args()

os.makedirs(MODELS_DIR, exist_ok=True)
df = pd.read_csv(args.dataset)


def evaluate_and_save(name, sub, available, filename):
    n = len(sub)
    if n < MIN_SAMPLES:
        print(f"  Zu wenige Samples ({n}) – übersprungen.")
        return

    # Echte Samples für Evaluation, alle für Training
    has_aug = "augmented" in sub.columns
    real = sub[~sub["augmented"].astype(bool)] if has_aug else sub
    n_real = len(real)
    n_aug  = n - n_real

    if n_real < MIN_SAMPLES:
        print(f"  Zu wenige echte Samples ({n_real}) – übersprungen.")
        return

    label = f"{n_real} echte"
    if n_aug:
        label += f" + {n_aug} augmentierte"
    print(f"  Samples: {label}")

    X_all, y_all   = sub[available],  sub["score"]
    X_real, y_real = real[available], real["score"]

    loo = LeaveOneOut()
    cv_model = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mae = -cross_val_score(cv_model, X_real, y_real, cv=loo,
                               scoring="neg_mean_absolute_error")
        if n_real >= 5:
            r2 = cross_val_score(cv_model, X_real, y_real, cv=loo, scoring="r2")
            r2_val = f"{np.nanmean(r2):.3f}  (±{np.nanstd(r2):.3f})"
        else:
            r2_val = "n/a"

    print(f"  LOO-CV MAE (echte): {mae.mean():.3f}  (±{mae.std():.3f})")
    print(f"  LOO-CV R²  (echte): {r2_val}")

    # Modell auf allen Daten trainieren (echte + augmentierte)
    model = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE)
    model.fit(X_all, y_all)

    out = os.path.join(MODELS_DIR, filename)
    joblib.dump({"model": model, "feature_cols": available,
                 "mae": float(mae.mean())}, out)
    print(f"  Gespeichert: {out}")


# --- Within-subject Modelle ---
for participant, group in df.groupby("participant"):
    print(f"\n=== {participant} ===")

    available = [c for c in FEATURE_COLS if c in group.columns and group[c].notna().any()]
    sub = group[available + ["score"]].dropna().reset_index(drop=True)

    dropped = len(group) - len(sub)
    if dropped:
        print(f"  {dropped} Zeile(n) ohne Gaze-Daten übersprungen.")

    evaluate_and_save(participant, sub, available, f"{participant}.pkl")

# --- Gepooltes Cross-Subject-Modell ---
print(f"\n=== GEPOOLTES MODELL (alle Teilnehmer) ===")
available_all = [c for c in FEATURE_COLS if c in df.columns and df[c].notna().any()]
pooled = df[available_all + ["score"]].dropna().reset_index(drop=True)
print(f"  {len(pooled)} gültige Samples über alle Teilnehmer.")
evaluate_and_save("pooled", pooled, available_all, "pooled.pkl")

print("\nFertig.")
