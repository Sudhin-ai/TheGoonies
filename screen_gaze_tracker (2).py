"""
screen_gaze_tracker.py
=======================
Workaround-Lösung für den Tobii Tracker 5, der keinen direkten SDK-Zugriff
bietet, sondern nur einen halbtransparenten weißen Kreis auf dem Bildschirm
anzeigt.

ERKENNUNGSANSATZ - Differenzbild (robuster als HoughCircles):
Beim Start wird einmalig ein Referenz-Screenshot OHNE den Tobii-Kreis
aufgenommen. Jeder folgende Frame wird mit diesem Referenzbild verglichen.
Der Unterschied zeigt genau wo der Kreis ist - unabhängig von Farbe,
Größe oder Hintergrund. Das funktioniert auch bei weißen/transparenten
Kreisen auf beliebigen Hintergründen zuverlässig.

ABLAUF:
1. Skript starten - Countdown beginnt
2. Während Countdown: Tobii Experience App öffnen, Kreis NOCH NICHT aktivieren
3. Nach "REFERENZ wird aufgenommen": Kreis WEITERHIN deaktiviert lassen
4. Nach "JETZT Kreis aktivieren": Kreis in Tobii Experience einschalten
5. Tobii-Fenster in Ecke schieben, Lesetext öffnen -> fertig

Installation:
    pip install mss opencv-python pylsl numpy

Kalibrierung (Vorschau, kein LSL-Stream):
    python screen_gaze_tracker.py --calibrate

Echte Aufnahme (sendet LSL-Stream):
    python screen_gaze_tracker.py

Delay anpassen (z.B. 10 Sekunden):
    python screen_gaze_tracker.py --calibrate --delay 10
"""
import argparse
import time

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Einstellungen - normalerweise nichts ändern nötig
# ---------------------------------------------------------------------------

# Wie stark muss sich ein Pixel unterscheiden, um als "Kreis-Pixel" zu gelten?
# Höher = weniger empfindlich (weniger Falscherkennungen durch Bildschirm-
# Flackern), aber der Kreis muss deutlicher sein. 15-25 ist ein guter Bereich.
DIFF_THRESHOLD = 20

# Minimale Fläche in Pixeln, die als Kreis gewertet wird.
# Verhindert, dass kleine Artefakte (Cursor, Animationen) erkannt werden.
# Bei einem großen Tobii-Kreis (~60-80px Radius) entspricht das ~11000-20000px².
MIN_AREA_PX = 5000
MAX_AREA_PX = 80000

TARGET_HZ = 25

# Bereiche die ignoriert werden sollen (z.B. Tobii-Experience-Fenster in Ecke).
# Format: (x, y, breite, höhe) in Bildschirm-Pixeln.
# Beispiel: EXCLUDE_REGIONS = [(0, 800, 300, 280)]  # unten links
EXCLUDE_REGIONS = []


def grab_frame(sct, monitor):
    raw = sct.grab(monitor)
    return np.ascontiguousarray(np.array(raw)[:, :, :3])  # BGRA -> BGR


def apply_exclusions(mask):
    """Löscht konfigurierte Ausschlussbereiche aus der Differenzmaske."""
    for (x, y, w, h) in EXCLUDE_REGIONS:
        mask[y:y+h, x:x+w] = 0
    return mask


def detect_from_diff(frame, reference):
    """Findet den Tobii-Kreis über das Differenzbild zur Referenz.

    Gibt (x_px, y_px, radius_px) zurück oder None falls nichts gefunden.
    """
    # Differenzbild: wo hat sich etwas verändert?
    diff = cv2.absdiff(frame, reference)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # Schwellwert: nur deutliche Unterschiede zählen
    _, thresh = cv2.threshold(gray, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)

    # Rauschen entfernen
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    thresh = apply_exclusions(thresh)

    # Konturen finden - der Kreis ist die größte zusammenhängende Fläche
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Größte Kontur nehmen (= Tobii-Kreis)
    best = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(best)

    if not (MIN_AREA_PX <= area <= MAX_AREA_PX):
        return None

    # Mittelpunkt und Radius aus der umschließenden Kreisform
    (x, y), radius = cv2.minEnclosingCircle(best)
    return float(x), float(y), float(radius)


def take_reference(sct, monitor):
    """Nimmt den Referenz-Screenshot auf (ohne Tobii-Kreis)."""
    print("\n>>> REFERENZ wird jetzt aufgenommen - Tobii-Kreis muss DEAKTIVIERT sein! <<<")
    time.sleep(0.5)
    frame = grab_frame(sct, monitor)
    print(">>> Referenz aufgenommen. JETZT Tobii-Kreis in der Experience App aktivieren! <<<")
    print(">>> Danach Tobii-Fenster in die Ecke schieben und Lesetext öffnen. <<<\n")
    time.sleep(2)  # Kurz warten damit der Kreis sichtbar wird
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Live-Vorschau anzeigen (Differenzbild + erkannter Kreis). "
             "Sendet keinen LSL-Stream.",
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=5,
        help="Countdown in Sekunden vor der Referenzaufnahme (Standard: 5). "
             "Nutze diese Zeit um PowerShell zu minimieren.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Tobii Screen Gaze Tracker")
    print("=" * 60)
    print(f"\nCountdown: {args.delay} Sekunden.")
    print(">> Nutze diese Zeit um PowerShell zu minimieren und")
    print("   sicherzustellen dass der Tobii-Kreis NOCH NICHT aktiv ist. <<\n")

    for i in range(args.delay, 0, -1):
        print(f"  {i}...", end="\r", flush=True)
        time.sleep(1)
    print("  Los!                ")

    import mss

    with mss.MSS() as sct:
        monitor = sct.monitors[1]
        width, height = monitor["width"], monitor["height"]

        # Referenzbild aufnehmen (ohne Kreis)
        reference = take_reference(sct, monitor)

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
            print("Druecke Strg+C zum Beenden.\n")

        prev_pos = None
        period = 1.0 / TARGET_HZ
        no_detection_count = 0

        try:
            while True:
                t0 = time.perf_counter()
                frame = grab_frame(sct, monitor)
                result = detect_from_diff(frame, reference)

                if result is not None:
                    x_px, y_px, r_px = result
                    prev_pos = (x_px, y_px)
                    no_detection_count = 0
                    x_norm = x_px / width
                    y_norm = y_px / height

                    if outlet is not None:
                        outlet.push_sample([x_norm, y_norm], pylsl.local_clock())
                    else:
                        print(f"Kreis bei x={x_norm:.3f} y={y_norm:.3f} r={r_px:.0f}px", end="\r")

                    if args.calibrate:
                        # Erkannten Kreis rot einzeichnen
                        cv2.circle(frame, (int(x_px), int(y_px)), int(r_px), (0, 0, 255), 3)
                        cv2.circle(frame, (int(x_px), int(y_px)), 4, (0, 0, 255), -1)
                else:
                    no_detection_count += 1
                    if args.calibrate and no_detection_count % 10 == 0:
                        print("Kein Kreis erkannt...           ", end="\r")

                if args.calibrate:
                    # Differenzbild als kleine Vorschau einblenden (oben links)
                    diff = cv2.absdiff(frame, reference)
                    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                    _, diff_thresh = cv2.threshold(diff_gray, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
                    diff_color = cv2.cvtColor(diff_thresh, cv2.COLOR_GRAY2BGR)
                    # Differenzbild klein einblenden
                    dh, dw = diff_color.shape[:2]
                    small_diff = cv2.resize(diff_color, (dw // 6, dh // 6))
                    sdh, sdw = small_diff.shape[:2]
                    frame[10:10+sdh, 10:10+sdw] = small_diff
                    cv2.putText(frame, "Diff (oben links)", (10, 10+sdh+15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                    for (ex, ey, ew, eh) in EXCLUDE_REGIONS:
                        cv2.rectangle(frame, (ex, ey), (ex+ew, ey+eh), (0, 255, 255), 2)

                    preview = cv2.resize(frame, (width // 2, height // 2))
                    cv2.imshow("Kalibrierung - 'q' zum Beenden, 'r' neue Referenz", preview)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    elif key == ord("r"):
                        # Neue Referenz aufnehmen (z.B. wenn sich Hintergrund geändert hat)
                        print("\nNeue Referenz wird aufgenommen - Kreis kurz deaktivieren!")
                        time.sleep(2)
                        reference = grab_frame(sct, monitor)
                        print("Neue Referenz aufgenommen.")

                elapsed = time.perf_counter() - t0
                time.sleep(max(0, period - elapsed))

        except KeyboardInterrupt:
            print("\nBeendet.")
        finally:
            if args.calibrate:
                cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
