"""
screen_gaze_tracker.py
=======================
Erkennt die Position des Tobii-Gaze-Kreises per Differenzbild-Methode
und sendet sie als LSL-Stream.

METHODE (Differenzbild statt HoughCircles):
  1. Einmalig ein Referenz-Screenshot aufnehmen (OHNE den Tobii-Kreis).
  2. Jeden Frame mit dem Referenzbild vergleichen -> nur die Pixel, die
     sich verändert haben (= der Kreis), leuchten im Differenzbild auf.
  3. Den Mittelpunkt der größten zusammenhängenden Veränderungsregion
     als Kreisposition nehmen.
  Vorteil: Funktioniert zuverlässig bei weißen/transparenten Kreisen,
  unabhängig vom Hintergrund. Keine Falscherkennungen durch Buttons,
  Icons etc., da diese sich zwischen den Frames nicht verändern.

ABLAUF:
  1. Skript starten (--delay gibt Zeit zum Wechseln):
       python screen_gaze_tracker.py --calibrate --delay 5
  2. Während des Countdowns: Tobii Experience App in den Vordergrund,
     Kreis aktivieren, dann Tobii-Fenster in die Ecke schieben.
  3. Referenzbild wird automatisch aufgenommen (Kreis muss kurz
     NICHT auf dem Bildschirm sein - Augen kurz schließen oder
     wegschauen reicht NICHT; stattdessen 'r' drücken wenn der
     Kreis gerade nicht sichtbar ist).
  4. Ab jetzt wird die Kreisposition getrackt.
  'r' drücken: neues Referenzbild aufnehmen (ohne Kreis).
  'q' drücken: Kalibrierungsmodus beenden.

Installation:
    pip install mss opencv-python pylsl numpy
"""
import argparse
import time

import cv2
import numpy as np

TARGET_HZ = 25
DIFF_THRESHOLD = 30      # Minimale Helligkeitsänderung um als "Kreis" zu gelten (0-255)
MIN_AREA_PX = 500        # Minimale Fläche der Änderungsregion in Pixeln
BLUR_KERNEL = (5, 5)     # Weichzeichnen vor Differenzbildung (reduziert Rauschen)

EXCLUDE_REGIONS = [
    # (x, y, breite, höhe) in Bildschirm-Pixeln
    # Beispiel: Tobii-Experience-Fenster oben links ausschließen:
    # (0, 0, 300, 200),
]


def grab_frame(sct, monitor):
    raw = sct.grab(monitor)
    return np.ascontiguousarray(np.array(raw)[:, :, :3])  # BGRA -> BGR


def apply_exclusions(mask):
    """Setzt Ausschlusszonen im Differenzbild auf 0."""
    for (x, y, w, h) in EXCLUDE_REGIONS:
        mask[y:y+h, x:x+w] = 0
    return mask


def detect_from_diff(frame, reference):
    """Gibt (cx, cy) des Kreismittelpunkts zurück, oder None.

    Vergleicht den aktuellen Frame mit dem Referenzbild (aufgenommen
    ohne Kreis). Der Tobii-Kreis ist die größte Region, die sich
    verändert hat.
    """
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, BLUR_KERNEL, 0)

    gray_ref = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    gray_ref = cv2.GaussianBlur(gray_ref, BLUR_KERNEL, 0)

    diff = cv2.absdiff(gray_frame, gray_ref)
    _, thresh = cv2.threshold(diff, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
    thresh = apply_exclusions(thresh)

    # Morphologisches Schließen: kleine Lücken im Kreis füllen
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, thresh

    # Größte zusammenhängende Veränderungsregion = Tobii-Kreis
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < MIN_AREA_PX:
        return None, thresh

    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None, thresh

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy), thresh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Live-Vorschau anzeigen (kein LSL-Stream).",
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=5,
        help="Wartezeit in Sekunden vor dem Start (Standard: 5). "
             "Tobii Experience App in dieser Zeit in den Vordergrund bringen.",
    )
    args = parser.parse_args()

    print(f"Startet in {args.delay} Sekunden ...")
    print(">> Tobii Experience App in den Vordergrund bringen und Kreis aktivieren! <<")
    print(">> Danach Tobii-Fenster in die Ecke schieben. <<")
    for i in range(args.delay, 0, -1):
        print(f"   {i}...", end="\r", flush=True)
        time.sleep(1)
    print("Los!                    ")

    import mss

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
            print("Druecke Strg+C zum Beenden.")

        # --- Referenzbild aufnehmen ---
        # Idealerweise kurz den Kreis verdecken oder 'r' druecken wenn
        # der Kreis gerade nicht sichtbar ist.
        print()
        print("Referenzbild wird aufgenommen (Kreis sollte kurz nicht sichtbar sein)...")
        print("Tipp: kurz auf einen leeren Bereich des Bildschirms schauen und")
        print("      dann 'r' druecken um das Referenzbild manuell zu setzen.")
        time.sleep(1)
        reference = grab_frame(sct, monitor)
        print("Referenzbild aufgenommen. Tracking laeuft.")
        if args.calibrate:
            print("Vorschau: roter Punkt = erkannte Position | 'r' = neues Referenz | 'q' = Beenden")

        period = 1.0 / TARGET_HZ

        try:
            while True:
                t0 = time.perf_counter()
                frame = grab_frame(sct, monitor)
                pos, diff_vis = detect_from_diff(frame, reference)

                if pos is not None:
                    cx, cy = pos
                    x_norm = cx / width
                    y_norm = cy / height

                    if outlet is not None:
                        import pylsl
                        outlet.push_sample([x_norm, y_norm], pylsl.local_clock())
                    else:
                        print(f"Kreis bei x={x_norm:.3f} y={y_norm:.3f}", end="\r")

                    if args.calibrate:
                        cv2.circle(frame, (cx, cy), 20, (0, 0, 255), 3)
                        cv2.drawMarker(frame, (cx, cy), (0, 0, 255),
                                       cv2.MARKER_CROSS, 30, 2)
                else:
                    if args.calibrate:
                        cv2.putText(frame, "Kein Kreis erkannt", (30, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                if args.calibrate:
                    # Ausschlusszonen gelb markieren
                    for (ex, ey, ew, eh) in EXCLUDE_REGIONS:
                        cv2.rectangle(frame, (ex, ey), (ex+ew, ey+eh), (0, 255, 255), 2)

                    preview = cv2.resize(frame, (width // 2, height // 2))

                    # Differenzbild klein einblenden (oben rechts im Vorschaufenster)
                    diff_color = cv2.cvtColor(diff_vis, cv2.COLOR_GRAY2BGR)
                    diff_small = cv2.resize(diff_color, (width // 6, height // 6))
                    h_p, w_p = preview.shape[:2]
                    h_d, w_d = diff_small.shape[:2]
                    preview[10:10+h_d, w_p-w_d-10:w_p-10] = diff_small

                    cv2.imshow("Kalibrierung - 'r' Referenz | 'q' Beenden", preview)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    elif key == ord("r"):
                        reference = grab_frame(sct, monitor)
                        print("\nNeues Referenzbild aufgenommen.")

                elapsed = time.perf_counter() - t0
                time.sleep(max(0, period - elapsed))

        except KeyboardInterrupt:
            print("\nBeendet.")
        finally:
            if args.calibrate:
                cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
