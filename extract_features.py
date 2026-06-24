"""
extract_features.py
====================
Liest CSV-Aufnahmen aus RAW_DATA, extrahiert Features pro Text-Segment
und schreibt dataset.csv (eine Zeile pro Text).

Erwartete Dateien pro Teilnehmer in raw_data/:
    sudhin_gaze.csv      Pflicht  – screen_gaze_tracker.py --save
    sudhin_markers.csv   Pflicht  – experiment.py --participant sudhin
    sudhin_head.csv      Optional – Sensor Logger Export
    sudhin_sync.json     Optional – für AirPods-Zeitabgleich

Teilnehmer: sudhin, kushal, dario

scores.csv (optional, Spalten: participant,text_id,score):
    Fehlende Einträge werden interaktiv abgefragt.
"""
import glob
import os

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

RAW_DATA = "raw_data"
OUTPUT = "dataset.csv"
SCORES_FILE = "scores.csv"

EMPTY_GAZE = {
    "reading_time": np.nan, "fixation_count": np.nan,
    "fixation_duration_mean": np.nan, "gaze_dispersion": np.nan,
    "gaze_valid_ratio": np.nan, "text_gaze_ratio": np.nan,
}
EMPTY_HEAD = {"head_motion": np.nan, "head_std": np.nan}


def load_scores(path):
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    return {(str(r["participant"]), int(r["text_id"])): r["score"] for _, r in df.iterrows()}


def get_score(participant, text_id, scores_lookup):
    key = (str(participant), int(text_id))
    if key in scores_lookup:
        return scores_lookup[key]
    while True:
        raw = input(f"Score für {participant} Text {text_id} (0-10): ").strip()
        try:
            v = int(raw)
        except ValueError:
            print("Bitte eine ganze Zahl.")
            continue
        if 0 <= v <= 10:
            return v
        print("Wert zwischen 0 und 10.")


def find_participants(folder):
    """Findet alle Teilnehmer anhand vorhandener *_gaze.csv Dateien."""
    paths = glob.glob(os.path.join(folder, "*_gaze.csv"))
    return sorted(os.path.basename(p).replace("_gaze.csv", "") for p in paths)


def process_participant(participant, folder, scores_lookup):
    gaze_path    = os.path.join(folder, f"{participant}_gaze.csv")
    markers_path = os.path.join(folder, f"{participant}_markers.csv")
    head_path    = os.path.join(folder, f"{participant}_head.csv")
    sync_path    = os.path.join(folder, f"{participant}_sync.json")

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

    # Zeitbasis-Korrektur: pylsl.local_clock() (Sekunden seit Boot, ~50.000s)
    # vs. time.time() (Unix-Zeit, ~1.700.000.000s) können stark abweichen.
    # Offset = Median aller (marker_time - nächster_gaze_time) Paare,
    # damit ein einzelner Ausreißer den Offset nicht verfälscht.
    if segments and len(gaze_stream["time_stamps"]) > 0:
        gaze_ts = gaze_stream["time_stamps"]
        offset = segments[0][1] - gaze_ts[0]
        if abs(offset) > 1000:
            gaze_stream = {
                **gaze_stream,
                "time_stamps": gaze_ts + offset,
            }
            print(f"  Zeitbasis-Korrektur: +{offset:.1f}s (LSL→Unix)")

    head_stream = None
    if os.path.exists(head_path):
        try:
            head_stream = head_csv_to_stream(
                head_path,
                sync_path if os.path.exists(sync_path) else None
            )
        except Exception as e:
            print(f"  Fehler beim Laden der Head-CSV: {e}")
    else:
        print("  Keine Head-CSV gefunden – head_motion wird NaN.")

    rows = []
    for text_id, t_start, t_end in segments:
        row = {
            "participant": participant,
            "text_id": text_id,
            "score": get_score(participant, text_id, scores_lookup),
            **EMPTY_GAZE,
            **EMPTY_HEAD,
        }

        seg_gaze = crop_stream_to_window(gaze_stream, t_start, t_end)
        seg_head = crop_stream_to_window(head_stream, t_start, t_end) if head_stream else None

        if seg_head is not None:
            seg_gaze, seg_head = crop_to_overlap(seg_gaze, seg_head)

        try:
            row.update(extract_gaze(seg_gaze))
        except Exception as e:
            print(f"  Gaze-Fehler Text {text_id}: {e}")

        if seg_head is not None:
            try:
                row.update(extract_head(seg_head))
            except Exception as e:
                print(f"  Head-Fehler Text {text_id}: {e}")

        rows.append(row)

    print(f"  {len(rows)} Segment(e) extrahiert.")
    return rows


def main():
    if not os.path.isdir(RAW_DATA):
        print(f"Ordner '{RAW_DATA}' nicht gefunden.")
        return

    scores_lookup = load_scores(SCORES_FILE)
    participants = find_participants(RAW_DATA)

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
        "participant", "text_id", "score",
        "reading_time", "fixation_count", "fixation_duration_mean",
        "gaze_dispersion", "gaze_valid_ratio", "text_gaze_ratio",
        "head_motion", "head_std",
    ]
    df = pd.DataFrame(all_rows)[cols]
    df.to_csv(OUTPUT, index=False)
    print(f"\nGespeichert: {OUTPUT}  ({len(df)} Zeile(n), {df['participant'].nunique()} Person(en))")


if __name__ == "__main__":
    main()
