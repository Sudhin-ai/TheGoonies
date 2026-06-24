"""
screen_gaze_tracker.py
=======================
Workaround-Lösung, falls kein Zugriff auf die offizielle Tobii Stream
Engine SDK (tobii_stream_engine.dll) besteht.

Idee: Tobiis eigene Software zeigt bereits einen Kreis auf dem
Bildschirm, der die aktuelle Blickposition visualisiert. Dieses Skript
macht laufend Screenshots, erkennt die Position dieses Kreises per
Bildverarbeitung (OpenCV) und sendet die erkannte Position als ganz
normalen LSL-Gaze-Stream - im selben Format ("type"="gaze", Kanäle
"x","y"), das auch App-TobiiStreamEngine verwenden würde.

Dadurch müsst ihr an extract_features.py / features.py NICHTS ändern:
LabRecorder zeichnet diesen Stream genauso auf wie den offiziellen,
und find_stream_by_type(streams, "gaze") findet ihn automatisch.

WICHTIGE EINSCHRÄNKUNGEN (bitte unbedingt vorher lesen):
- Funktioniert nur, wenn der Kreis WIRKLICH systemweit über jedem
  Programm angezeigt wird (z. B. auch während ihr ein PDF oder eine
  Webseite lest) - nicht nur innerhalb eines Tobii-eigenen
  Kalibrierungsfensters. Bitte das unbedingt selbst prüfen, BEVOR ihr
  Zeit in dieses Skript investiert.
- Die Genauigkeit/Sampling-Rate ist deutlich schlechter als die echte
  Tobii-Hardware (abhängig von Bildschirmaufnahme-Geschwindigkeit,
  typischerweise 15-30 Hz statt 60-120 Hz).
- Keine Information über Blinzeln, Pupillendurchmesser, linkes/rechtes
  Auge getrennt o. ä. - nur die x/y-Position des Kreises.
- Die Erkennung muss höchstwahrscheinlich an Farbe/Größe eures
  konkreten Kreises angepasst werden (siehe Kalibrierungsmodus unten).
- Das misst weiterhin denselben von Tobiis eigener (lizenzpflichtiger)
  Software berechneten Blickpunkt, nur über einen Umweg statt über die
  offizielle API - an der grundsätzlichen Lizenzfrage ändert die
  Aufnahme-Methode nichts, nur der technische Weg ist anders.

Installation (auf dem Windows-PC):
    pip install mss opencv-python pylsl numpy

Kalibrierung (zeigt Live-Vorschau mit Erkennung, sendet noch NICHTS):
    python screen_gaze_tracker.py --calibrate

Echte Aufnahme (sendet LSL-Stream, keine Vorschau):
    python screen_gaze_tracker.py
"""
import argparse
import time

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Diese Werte müsst ihr wahrscheinlich an euren Kreis anpassen!
# Nutzt den Kalibrierungsmodus (--calibrate), um die richtigen Werte zu finden.
# ---------------------------------------------------------------------------
# Falls der Kreis eine bestimmte, gut von der Umgebung unterscheidbare Farbe
# hat, hier den HSV-Bereich eintragen (reduziert Fehlerkennungen stark).
# Auf None setzen, um stattdessen reine Formerkennung (ohne Farbfilter) zu
# nutzen - das funktioniert nur bei eher ruhigem Hintergrund gut.
# Tobii Tracker 5: Der Kreis ist weiß/halbtransparent.
# HSV-Weiß = niedriger Sättigungswert (S), hoher Helligkeitswert (V).
# Falls die Erkennung zu viele weiße UI-Elemente erfasst, S-Untergrenze
# erhöhen (z. B. auf 10) oder V-Untergrenze senken (z. B. auf 180).
COLOR_HSV_LOWER = np.array([0, 0, 200])    # H beliebig, S fast 0, V sehr hell
COLOR_HSV_UPPER = np.array([180, 40, 255]) # H beliebig, S max 40, V max

MIN_RADIUS_PX = 30   # Kleine UI-Elemente (Buttons, Icons) ausschließen
MAX_RADIUS_PX = 100  # Tobii-Kreis ist groß (~60-80px geschätzt aus Screenshot)
TARGET_HZ = 25

# Bereiche, die von der Erkennung ausgeschlossen werden (z. B. das kleine
# Tobii-Experience-Fenster in der Ecke, das eigene runde UI-Elemente haben
# kann, die sonst fälschlich als Gaze-Kreis erkannt werden könnten).
# Format: (x, y, breite, höhe) in Bildschirm-Pixeln. Mit --calibrate die
# richtigen Werte ermitteln (im Vorschaufenster wird die Zone gelb markiert).
EXCLUDE_REGIONS = [
    # (0, 0, 400, 300),  # Beispiel: oben links, 400x300 Pixel
]


def apply_exclusions(gray, scale_x=1.0, scale_y=1.0):
    """Schwärzt die konfigurierten Ausschlusszonen, damit dort keine
    Kreise erkannt werden."""
    for (x, y, w, h) in EXCLUDE_REGIONS:
        x0, y0 = int(x * scale_x), int(y * scale_y)
        x1, y1 = int((x + w) * scale_x), int((y + h) * scale_y)
        gray[y0:y1, x0:x1] = 0
    return gray


def grab_frame(sct, monitor):
    raw = sct.grab(monitor)
    return np.ascontiguousarray(np.array(raw)[:, :, :3])  # BGRA -> BGR


def detect_circle(frame, prev_pos=None):
    """Gibt (x_px, y_px, radius_px) zurück, oder None falls nichts gefunden."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if COLOR_HSV_LOWER is not None and COLOR_HSV_UPPER is not None:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, COLOR_HSV_LOWER, COLOR_HSV_UPPER)
        gray = cv2.bitwise_and(gray, gray, mask=mask)

    gray = apply_exclusions(gray)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=30,
        param1=60,
        param2=20,
        minRadius=MIN_RADIUS_PX,
        maxRadius=MAX_RADIUS_PX,
    )

    if circles is None:
        return None

    circles = circles[0]

    if prev_pos is not None:
        # Den Kreis nehmen, der am nächsten an der letzten Position liegt.
        # Das reduziert Fehlerkennungen durch andere runde Formen im
        # Hintergrund (Buttons, Icons, etc.), da der Blickpunkt sich
        # zwischen zwei Frames nur wenig bewegen sollte.
        dists = np.hypot(circles[:, 0] - prev_pos[0], circles[:, 1] - prev_pos[1])
        best = circles[np.argmin(dists)]
    else:
        best = circles[0]

    return float(best[0]), float(best[1]), float(best[2])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Live-Vorschau mit Erkennung anzeigen, statt einen LSL-Stream zu senden.",
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=5,
        help="Wartezeit in Sekunden vor dem Start (Standard: 5). "
             "Nutze diese Zeit, um die Tobii Experience App in den "
             "Vordergrund zu bringen und den Kreis zu aktivieren.",
    )
    args = parser.parse_args()

    print(f"Startet in {args.delay} Sekunden ...")
    print(">> Jetzt Tobii Experience App in den Vordergrund bringen und Kreis aktivieren! <<")
    for i in range(args.delay, 0, -1):
        print(f"   {i}...", end="\r", flush=True)
        time.sleep(1)
    print("Los!                    ")

    import mss  # Import hier, damit --calibrate-Hilfe auch ohne mss läuft

    with mss.MSS() as sct:
        monitor = sct.monitors[1]  # Hauptbildschirm; bei Mehrschirm-Setup ggf. anpassen
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

        prev_pos = None
        period = 1.0 / TARGET_HZ

        try:
            while True:
                t0 = time.perf_counter()
                frame = grab_frame(sct, monitor)
                result = detect_circle(frame, prev_pos)

                if result is not None:
                    x_px, y_px, r_px = result
                    prev_pos = (x_px, y_px)
                    x_norm = x_px / width
                    y_norm = y_px / height

                    if outlet is not None:
                        import pylsl

                        outlet.push_sample([x_norm, y_norm], pylsl.local_clock())
                    else:
                        print(f"Kreis erkannt bei x={x_norm:.3f} y={y_norm:.3f} r={r_px:.0f}px")

                    if args.calibrate:
                        cv2.circle(frame, (int(x_px), int(y_px)), int(r_px), (0, 0, 255), 2)
                else:
                    prev_pos = None
                    if args.calibrate:
                        print("Kein Kreis erkannt...")

                if args.calibrate:
                    for (ex, ey, ew, eh) in EXCLUDE_REGIONS:
                        cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (0, 255, 255), 2)
                    preview = cv2.resize(frame, (width // 2, height // 2))
                    cv2.imshow("Kalibrierung - 'q' zum Beenden", preview)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                elapsed = time.perf_counter() - t0
                time.sleep(max(0, period - elapsed))
        except KeyboardInterrupt:
            print("Beendet.")
        finally:
            if args.calibrate:
                cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
