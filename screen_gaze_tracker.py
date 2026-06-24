"""
screen_gaze_tracker.py
=======================
Workaround-Lösung für den Tobii Tracker 5, der keinen direkten SDK-Zugriff
bietet, sondern nur einen halbtransparenten weißen Kreis auf dem Bildschirm
anzeigt.

ERKENNUNGSANSATZ - MOG2-Hintergrundseparation:
Statt eines statischen Referenzbildes wird ein adaptives Hintergrundmodell
(MOG2) verwendet. MOG2 lernt langsam alles Statische (Text, Fenster, Icons)
und meldet nur, was sich bewegt - also den Tobii-Kreis, der dem Blick folgt.

Warum MOG2 statt statischer Referenz?
  Die statische Referenz enthielt keinen Text. Sobald Text geöffnet wurde,
  erschien er als riesige Diff-Region und wurde fälschlicherweise als Kreis
  erkannt. MOG2 lernt Text als Hintergrund (~3-5 Sekunden) und ignoriert
  ihn danach automatisch.

ABLAUF:
1. Skript starten - Tobii-Kreis BEREITS aktivieren (MOG2 braucht keine
   separate Referenzphase ohne Kreis)
2. Lesetext öffnen und 5 Sekunden warten (MOG2 lernt den Hintergrund)
3. Ab dann läuft die Erkennung stabil

Installation:
    pip install mss opencv-python pylsl numpy

Kalibrierung (Vorschau, kein LSL-Stream):
    python screen_gaze_tracker.py --calibrate

Echte Aufnahme (sendet LSL-Stream):
    python screen_gaze_tracker.py
"""
import argparse
import math
import time

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Einstellungen
# ---------------------------------------------------------------------------

# Lernrate für das MOG2-Hintergrundmodell (0-1).
# 0.005 bei 25 Hz: statischer Text wird in ~3-5 Sekunden als Hintergrund
# gelernt. Niedriger = stabilere Erkennung aber langsamere Anpassung.
MOG2_LEARNING_RATE = 0.005

# MOG2-Empfindlichkeit: höher = weniger empfindlich (weniger Rauschen,
# aber schwächere Signale werden übersehen). 25-50 ist ein guter Bereich.
MOG2_VAR_THRESHOLD = 25

# Radius-Grenzen in Bildschirm-Pixeln.
MIN_RADIUS_PX = 50
MAX_RADIUS_PX = 300

# Minimale Hull-Kreisförmigkeit (0-1).
# Rechtecke/Fensterränder haben ~0.1-0.4, Kreise >0.7.
MIN_CIRCULARITY = 0.55

TARGET_HZ = 25

# Textbereich – nur Erkennungen INNERHALB dieses Rechtecks werden gesendet/
# gespeichert. None = ganzer Bildschirm (kein Filter).
# Format: (x, y, breite, höhe) in Bildschirm-Pixeln.
# Im Kalibrierungsmodus als grünes Rechteck sichtbar → Werte hier anpassen.
TEXT_REGION = (960, 0, 960, 1080)  # Chrome/Google-Docs-Textbereich

# Bereiche die aus der MOG2-Erkennung ausgeschlossen werden sollen
# (z.B. Tobii-Experience-Fenster in der Ecke).
# Format: (x, y, breite, höhe) in Bildschirm-Pixeln.
EXCLUDE_REGIONS = []


def grab_frame(sct, monitor):
    raw = sct.grab(monitor)
    return np.ascontiguousarray(np.array(raw)[:, :, :3])  # BGRA -> BGR


def apply_exclusions(mask):
    for (x, y, w, h) in EXCLUDE_REGIONS:
        mask[y:y+h, x:x+w] = 0
    return mask


def hull_circularity(contour):
    """Kreisförmigkeit des Convex Hull: 4π*Fläche/Umfang². Max = 1.0.

    Hull-basiert damit ein Ring (niedriger roher circ-Wert) korrekt als
    scheibenartig erkannt wird.
    """
    hull = cv2.convexHull(contour)
    area = cv2.contourArea(hull)
    perimeter = cv2.arcLength(hull, True)
    if perimeter == 0:
        return 0.0
    return (4 * math.pi * area) / (perimeter ** 2)


def detect(fg_mask):
    """Findet den Tobii-Kreis in der MOG2-Vordergrundmaske.

    Bewertet Kandidaten nach score = hull_circularity × radius.
    Gibt (x_px, y_px, radius_px, circ) zurück oder None.
    """
    # Ring-Lücken schließen, Rauschen entfernen
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_close)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)

    mask = apply_exclusions(mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    best = None
    best_score = -1.0

    for c in contours:
        (x, y), radius = cv2.minEnclosingCircle(c)
        if not (MIN_RADIUS_PX <= radius <= MAX_RADIUS_PX):
            continue
        circ = hull_circularity(c)
        if circ < MIN_CIRCULARITY:
            continue
        score = circ * radius
        if score > best_score:
            best_score = score
            best = (x, y, radius, circ)

    if best is None:
        return None
    x, y, radius, circ = best
    return float(x), float(y), float(radius), circ


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Live-Vorschau anzeigen. Sendet keinen LSL-Stream, speichert keine CSV.",
    )
    parser.add_argument(
        "--save",
        metavar="DATEI",
        default=None,
        help="Gaze-Daten in eine CSV-Datei speichern, z.B. --save sudhin_gaze.csv. "
             "Kann zusammen mit LSL-Stream verwendet werden.",
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=3,
        help="Wartezeit in Sekunden bevor Erkennung startet (Standard: 3).",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Tobii Screen Gaze Tracker  (MOG2)")
    print("=" * 60)
    print(f"\nTobii-Kreis jetzt aktivieren, dann Lesetext öffnen.")
    print(f"In {args.delay} Sekunden startet die Erkennung.\n")
    print("Hinweis: MOG2 lernt den Hintergrund in ~5 Sekunden.")
    print("In dieser Zeit können noch Fehlerkennungen auftreten.\n")

    for i in range(args.delay, 0, -1):
        print(f"  {i}...", end="\r", flush=True)
        time.sleep(1)
    print("  Los!                ")

    import mss

    bg_sub = cv2.createBackgroundSubtractorMOG2(
        history=int(TARGET_HZ / MOG2_LEARNING_RATE),  # ~5000 frames
        varThreshold=MOG2_VAR_THRESHOLD,
        detectShadows=False,
    )

    with mss.MSS() as sct:
        monitor = sct.monitors[1]
        width, height = monitor["width"], monitor["height"]

        outlet = None
        if not args.calibrate:
            import pylsl
            info = pylsl.StreamInfo(
                name="ScreenGazeTracker_gaze",
                type="gaze",
                channel_count=2,
                nominal_srate=pylsl.IRREGULAR_RATE,
                channel_format="float32",
                source_id="screen_gaze_tracker_v1",
            )
            outlet = pylsl.StreamOutlet(info)
            print("LSL-Stream 'ScreenGazeTracker_gaze' (type=gaze) ist aktiv.")

        csv_file = None
        csv_writer = None
        if args.save and not args.calibrate:
            import csv as _csv
            csv_file = open(args.save, "w", newline="", encoding="utf-8")
            csv_writer = _csv.writer(csv_file)
            csv_writer.writerow(["timestamp", "x", "y", "radius_px", "in_text_region"])
            print(f"CSV-Aufnahme: {args.save}")

        if not args.calibrate:
            print("Druecke Strg+C zum Beenden.\n")

        period = 1.0 / TARGET_HZ
        no_detection_count = 0

        try:
            while True:
                t0 = time.perf_counter()
                frame = grab_frame(sct, monitor)

                # MOG2: lernt langsam den Hintergrund, gibt Vordergrundmaske zurück
                fg_mask = bg_sub.apply(frame, learningRate=MOG2_LEARNING_RATE)

                result = detect(fg_mask)

                if result is not None:
                    x_px, y_px, r_px, circ = result
                    no_detection_count = 0

                    # Textbereich-Filter
                    in_region = True
                    if TEXT_REGION is not None:
                        rx, ry, rw, rh = TEXT_REGION
                        in_region = (rx <= x_px <= rx + rw) and (ry <= y_px <= ry + rh)

                    x_norm = x_px / width
                    y_norm = y_px / height
                    ts = pylsl.local_clock() if outlet is not None else time.time()

                    if outlet is not None:
                        outlet.push_sample([x_norm, y_norm], ts)

                    if csv_writer is not None:
                        csv_writer.writerow([
                            f"{ts:.6f}", f"{x_norm:.6f}", f"{y_norm:.6f}",
                            f"{r_px:.1f}", "1" if in_region else "0"
                        ])
                        csv_file.flush()

                    if not args.calibrate:
                        print(
                            f"Kreis bei x={x_norm:.3f} y={y_norm:.3f} "
                            f"r={r_px:.0f}px circ={circ:.2f}"
                            + (" [Text]" if in_region else " [außerhalb]"),
                            end="\r",
                        )

                    if args.calibrate:
                        # rot = im Bereich, orange = außerhalb
                        color = (0, 0, 255) if in_region else (0, 128, 255)
                        cv2.circle(frame, (int(x_px), int(y_px)), int(r_px), color, 3)
                        cv2.circle(frame, (int(x_px), int(y_px)), 4, color, -1)
                        label = f"r={r_px:.0f}px circ={circ:.2f}" + ("" if in_region else " [außerhalb]")
                        cv2.putText(frame, label,
                                    (int(x_px) + int(r_px) + 5, int(y_px)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                else:
                    no_detection_count += 1
                    if args.calibrate and no_detection_count % 10 == 0:
                        print("Kein Kreis erkannt...           ", end="\r")

                if args.calibrate:
                    # MOG2-Vordergrundmaske klein einblenden (oben links)
                    fg_color = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
                    fh, fw = fg_color.shape[:2]
                    small_fg = cv2.resize(fg_color, (fw // 6, fh // 6))
                    sfh, sfw = small_fg.shape[:2]
                    frame[10:10+sfh, 10:10+sfw] = small_fg
                    cv2.putText(frame, "MOG2", (10, 10+sfh+15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                    # Textbereich grün einzeichnen
                    if TEXT_REGION is not None:
                        rx, ry, rw, rh = TEXT_REGION
                        cv2.rectangle(frame, (rx, ry), (rx+rw, ry+rh), (0, 255, 0), 2)
                        cv2.putText(frame, "Textbereich", (rx+4, ry+20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    for (ex, ey, ew, eh) in EXCLUDE_REGIONS:
                        cv2.rectangle(frame, (ex, ey), (ex+ew, ey+eh), (0, 255, 255), 2)

                    preview = cv2.resize(frame, (width // 2, height // 2))
                    cv2.imshow("Kalibrierung - 'q' beenden", preview)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break

                elapsed = time.perf_counter() - t0
                time.sleep(max(0, period - elapsed))

        except KeyboardInterrupt:
            print("\nBeendet.")
        finally:
            if csv_file is not None:
                csv_file.close()
                print(f"CSV gespeichert: {args.save}")
            if args.calibrate:
                cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
