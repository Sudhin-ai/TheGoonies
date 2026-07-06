"""
augment_data.py
===============
Augmentiert dataset.csv durch Gaussian Noise auf alle Gaze-Features.
Die echten Daten bleiben in dataset.csv unverändert.
Ausgabe: dataset_augmented.csv (echte + synthetische Samples).

HINWEIS: Augmentierte Daten nur für Training verwenden.
In der Präsentation transparent als "data augmentation" kennzeichnen.

Nutzung:
    python augment_data.py
    python augment_data.py --factor 2   # 2x so viele Samples 
"""

import argparse
import numpy as np
import pandas as pd

FEATURE_COLS = [
    "reading_time",
    "fixation_count",
    "fixation_duration_mean",
    "gaze_dispersion",
    "gaze_valid_ratio",
    "text_gaze_ratio",
]

# Rauschstärke pro Feature (Anteil der Standardabweichung)
NOISE_SCALE = 0.08

parser = argparse.ArgumentParser()
parser.add_argument("--input",  default="dataset.csv")
parser.add_argument("--output", default="dataset_augmented.csv")
parser.add_argument("--factor", type=int, default=2,
                    help="Wie viele synthetische Kopien pro echtem Sample")
parser.add_argument("--target", type=int, default=None,
                    help="Ziel-Gesamtanzahl pro Teilnehmer (überschreibt --factor)")
parser.add_argument("--seed",   type=int, default=42)
args = parser.parse_args()

rng = np.random.default_rng(args.seed)

df = pd.read_csv(args.input)
real = df.dropna(subset=FEATURE_COLS).copy()
real["augmented"] = False

print(f"Echte Samples: {len(real)}")
print(f"  dario:  {(real['participant']=='dario').sum()}")
print(f"  kushal: {(real['participant']=='kushal').sum()}")
print(f"  sudhin: {(real['participant']=='sudhin').sum()}")

synthetic_rows = []

for participant, group in real.groupby("participant"):
    n_real = len(group)
    if args.target is not None:
        n_needed = max(0, args.target - n_real)
    else:
        n_needed = n_real * args.factor

    # Gleichmäßig auf die echten Samples verteilen
    base, rest = divmod(n_needed, n_real)
    copies = [base + (1 if i < rest else 0) for i in range(n_real)]

    for (_, row), n_copies in zip(group.iterrows(), copies):
        for i in range(n_copies):
            new = row.copy()
            new["augmented"] = True
            new["text_id"] = f"{int(row['text_id'])}_aug{i+1}"

            for col in FEATURE_COLS:
                val = row[col]
                std = real[col].std()
                noise = rng.normal(0, NOISE_SCALE * std)
                new_val = val + noise

                # Bounds einhalten
                if col == "gaze_valid_ratio" or col == "text_gaze_ratio":
                    new_val = float(np.clip(new_val, 0.0, 1.0))
                elif col in ("fixation_count",):
                    new_val = max(1.0, round(new_val))
                elif col == "reading_time":
                    new_val = max(10.0, new_val)
                elif col == "fixation_duration_mean":
                    new_val = max(0.05, new_val)
                elif col == "gaze_dispersion":
                    new_val = max(0.01, new_val)

                new[col] = new_val

            # Score leicht variieren (±1, gerundet, 0-10 begrenzt)
            score_noise = rng.normal(0, 0.4)
            new["score"] = int(np.clip(round(row["score"] + score_noise), 0, 10))

            synthetic_rows.append(new)

synthetic = pd.DataFrame(synthetic_rows)
combined  = pd.concat([real, synthetic], ignore_index=True)

cols = ["participant", "text_id", "augmented", "score"] + FEATURE_COLS
combined = combined[cols]
combined.to_csv(args.output, index=False)

total = len(combined)
print(f"\nAugmentierte Samples (Faktor {args.factor}x): {len(synthetic)}")
print(f"Gesamt in {args.output}: {total} Samples")
print(f"  dario:  {(combined['participant']=='dario').sum()}")
print(f"  kushal: {(combined['participant']=='kushal').sum()}")
print(f"  sudhin: {(combined['participant']=='sudhin').sum()}")
print("\nHinweis: train_model.py mit --dataset dataset_augmented.csv aufrufen.")
