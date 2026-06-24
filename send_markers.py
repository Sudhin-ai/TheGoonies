"""
send_markers.py
===============
Sendet text_start / text_end Marker und speichert sie als CSV.
Läuft auf dem Aufnahme-PC neben screen_gaze_tracker.py.

Nutzung:
    python send_markers.py --participant sudhin
    python send_markers.py --participant kushal
    python send_markers.py --participant dario

Erzeugt:
    sudhin_markers.csv   → Marker-Zeitstempel (Unix-Zeit)
    sudhin_sync.json     → Zeitabgleich für AirPods-Daten

Bedienung:
    Enter           → Text starten / beenden
    q + Enter       → Beenden

Installation:
    pip install pylsl   (optional – nur wenn auch LSL-Stream gewünscht)
"""
import argparse
import os
import csv
import json
import time
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--participant", required=True, help="sudhin, kushal oder dario")
parser.add_argument("--lsl", action="store_true", help="Zusätzlich LSL-Stream senden")
args = parser.parse_args()

os.makedirs("raw_data", exist_ok=True)
markers_path = os.path.join("raw_data", f"{args.participant}_markers.csv")
sync_path    = os.path.join("raw_data", f"{args.participant}_sync.json")

outlet = None
if args.lsl:
    import pylsl
    info = pylsl.StreamInfo("TextMarkers", "Markers", 1, pylsl.IRREGULAR_RATE, "string", "text_markers_v1")
    outlet = pylsl.StreamOutlet(info)
    print("LSL-Stream aktiv.")

print(f"Teilnehmer: {args.participant}")
print(f"Marker-Datei: {markers_path}")
print("Enter = Text starten/beenden | q + Enter = Beenden\n")

text_id = 0
in_text = False
sync_saved = False
rows = []

try:
    while True:
        raw = input().strip().lower()
        ts = time.time()

        if raw == "q":
            if in_text:
                rows.append([f"{ts:.6f}", "text_end", text_id])
                print(f"  → text_end (Text {text_id})")
            break

        # Beim ersten Marker Sync-Datei schreiben
        if not sync_saved:
            lsl_ts = pylsl.local_clock() if outlet else ts
            with open(sync_path, "w") as f:
                json.dump({"unix_time": ts, "lsl_clock": lsl_ts}, f)
            print(f"  Sync gespeichert: {sync_path}")
            sync_saved = True

        if outlet:
            outlet.push_sample(["text_start" if not in_text else "text_end"], pylsl.local_clock())

        if not in_text:
            text_id += 1
            rows.append([f"{ts:.6f}", "text_start", text_id])
            in_text = True
            print(f"  → text_start (Text {text_id}) – Enter wenn fertig gelesen")
        else:
            rows.append([f"{ts:.6f}", "text_end", text_id])
            in_text = False
            print(f"  → text_end   (Text {text_id}) – Enter für nächsten Text\n")

finally:
    with open(markers_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "label", "text_id"])
        w.writerows(rows)
    print(f"Gespeichert: {markers_path} ({len(rows)} Marker)")
