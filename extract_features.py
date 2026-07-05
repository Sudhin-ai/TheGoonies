"""
extract_features.py
====================
Liest CSV-Aufnahmen aus RAW_DATA, extrahiert Features pro Text-Segment
und schreibt dataset.csv (eine Zeile pro Text).

Erwartete Dateien pro Teilnehmer in raw_data/:
    sudhin_gaze.csv      Pflicht  – screen_gaze_tracker.py --save
    sudhin_markers.csv   Pflicht  – experiment.py --participant sudhin

Teilnehmer: sudhin, kushal, dario

scores.csv (optional, Spalten: participant,text_id,score):
    Wird von score.py automatisch erzeugt.
    Fehlende Einträge werden interaktiv abgefragt.
"""
import glob
import os

import numpy as np
import pandas as pd

from features import (
    crop_stream_to_window,
    extract_gaze,
    gaze_csv_to_stream,
    markers_csv_to_segments,
)

RAW_DATA = "raw_data"
OUTPUT   = "dataset.csv"
SCORES_FILE = "scores.csv"

EMPTY_GAZE = {
    "reading_time": np.nan, "fixation_count": np.nan,
    "fixation_duration_mean": np.nan, "gaze_dispersion": np.nan,
    "gaze_valid_ratio": np.nan, "text_gaze_ratio": np.nan,
}


def load_scores(path):
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    if "occurrence" not in df.columns:
        df["occurrence"] = 0
    return {(str(r["participant"]), int(r["text_id"]), int(r["occurrence"])): r["score"]
            for _, r in df.iterrows()}


def get_score(participant, text_id, occurrence, scores_lookup):
    key = (str(participant), int(text_id), int(occurrence))
    if key in scores_lookup:
        return scores_lookup[key]
    while True:
        raw = input(f"Score für {participant} Text {text_id} Durchgang {occurrence + 1} (0-10): ").strip()
        try:
            v = int(raw)
        except ValueError:
            print("Bitte eine ganze Zahl.")
            continue
        if 0 <= v <= 10:
            return v
        print("Wert zwischen 0 und 10.")


def find_participants(folder):
    paths = glob.glob(os.path.join(folder, "*_gaze.csv"))
    return sorted(os.path.basename(p).replace("_gaze.csv", "") for p in paths)


def process_participant(participant, folder, scores_lookup):
    gaze_path    = os.path.join(folder, f"{participant}_gaze.csv")
    markers_path = os.path.join(folder, f"{participant}_markers.csv")

    if not os.path.exists(markers_path):
        print(f"  Keine Marker-Datei gefunden ({markers_path}) – übersprungen.")
        return []

    segments = markers_csv_to_segments(markers_path)
    if not segments:
        print("  Keine vollständigen text_start/text_end-Paare.")
        return []

    try:
        gaze_stream = gaze_csv_to_stream(gaze_path)
    except Exception as e:
        print(f"  Fehler beim Laden der Gaze-CSV: {e}")
        return []

    # Zeitbasis-Korrektur: Neue Gaze-CSVs speichern bereits Unix-Zeit (time.time()).
    # Alte CSVs (Boot-Zeit, ~50.000s) werden automatisch erkannt und korrigiert.
    if len(gaze_stream["time_stamps"]) > 0:
        gaze_ts = gaze_stream["time_stamps"]
        if gaze_ts[0] < 1_000_000_000:
            # Boot-Zeit erkannt → besten Offset über alle Segmente suchen
            best_offset = None
            best_hits = -1
            for _, t_start, t_end in segments:
                candidate = t_start - gaze_ts[0]
                shifted_end = gaze_ts[-1] + candidate
                hits = sum(1 for _, s, e in segments if s >= t_start - 1 and e <= shifted_end + 60)
                if hits > best_hits:
                    best_hits = hits
                    best_offset = candidate
            if best_offset is not None:
                gaze_stream = {**gaze_stream, "time_stamps": gaze_ts + best_offset}
                print(f"  Zeitbasis-Korrektur (Boot→Unix): +{best_offset:.1f}s ({best_hits} Segment(e))")

    rows = []
    occ_counter = {}  # text_id → wie oft schon gesehen (Mehrfach-Lesungen/Loop)
    for text_id, t_start, t_end in segments:
        occ = occ_counter.get(text_id, 0)
        occ_counter[text_id] = occ + 1
        row = {
            "participant": participant,
            "text_id": text_id,
            "occurrence": occ,
            "score": get_score(participant, text_id, occ, scores_lookup),
            **EMPTY_GAZE,
        }

        seg_gaze = crop_stream_to_window(gaze_stream, t_start, t_end)

        try:
            row.update(extract_gaze(seg_gaze))
        except Exception as e:
            print(f"  Gaze-Fehler Text {text_id}: {e}")

        rows.append(row)

    print(f"  {len(rows)} Segment(e) extrahiert.")
    return rows


def main():
    if not os.path.isdir(RAW_DATA):
        print(f"Ordner '{RAW_DATA}' nicht gefunden.")
        return

    scores_lookup = load_scores(SCORES_FILE)
    participants  = find_participants(RAW_DATA)

    if not participants:
        print(f"Keine *_gaze.csv Dateien in '{RAW_DATA}' gefunden.")
        return

    all_rows = []
    for participant in participants:
        print(f"Verarbeite: {participant}")
        all_rows.extend(process_participant(participant, RAW_DATA, scores_lookup))

    if not all_rows:
        print("Keine verwertbaren Daten.")
        return

    cols = [
        "participant", "text_id", "occurrence", "score",
        "reading_time", "fixation_count", "fixation_duration_mean",
        "gaze_dispersion", "gaze_valid_ratio", "text_gaze_ratio",
    ]
    df = pd.DataFrame(all_rows)[cols]
    df.to_csv(OUTPUT, index=False)
    print(f"\nGespeichert: {OUTPUT}  ({len(df)} Zeile(n), {df['participant'].nunique()} Person(en))")


if __name__ == "__main__":
    main()
