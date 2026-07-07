# Week 12 Report — Machine Learning for Smart and Connected Systems (ML4SCS)

## Weekly Goal
Daten von allen drei Probanden erheben, die Pipeline von der Aufnahme bis zum
trainierten Modell einmal komplett durchlaufen lassen.

## Work Done This Week

### 1. Data Work
- Datenerhebung mit allen drei Probanden durchgeführt (Texte in mehreren Sessions gelesen)
- Beim Auswerten fiel auf: **weniger Segmente als gelesene Texte** — Ursache-Analyse ergab
  zwei Datenverlust-Probleme:
  - `experiment.py` überschrieb die Marker-CSV bei jedem Neustart (Schreibmodus statt Anhängen)
  - `screen_gaze_tracker.py` überschrieb die Gaze-CSV bei jedem Neustart und speicherte
    Zeitstempel in **Boot-Zeit** (Sekunden seit PC-Start), während die Marker **Unix-Zeit** nutzten
- Durch Abgleich der Zeitfenster konnten die noch vorhandenen Sessions rekonstruiert werden:
  **300 gültige Samples** (dario 100, kushal 100, sudhin 100)

### 2. Analysis / Modeling Work
- Zeitbasis-Korrektur in `extract_features.py` implementiert: automatische Offset-Erkennung
  zwischen Boot- und Unix-Zeit über die Marker-Segmente
- Erste Random-Forest-Modelle (within-subject, Leave-One-Out-CV) auf den echten Daten trainiert
- Ergebnis auf echten Daten: MAE ≈ 1.1 (gepoolt)

### 3. Repository / Documentation Work
- `screen_gaze_tracker.py`: speichert jetzt Unix-Zeit und hängt an bestehende CSVs an
- `experiment.py`: Marker-CSV im Append-Modus, Fortschritts-System (`progress.json`)
- `score.py` + `extract_features.py`: Unterstützung für Mehrfach-Lesungen (occurrence-Spalte)

## Experiments Conducted

| Experiment | Change Made | Result | Interpretation |
|-----------|-------------|--------|----------------|
| Exp 1 | Marker-Anzahl mit gelesenen Texten abgeglichen | Deutlich weniger Segmente als Lesungen | Datenverlust durch Überschreiben bei Skript-Neustart |
| Exp 2 | Gaze-Zeitfenster gegen Marker-Zeitstempel geprüft | Nur die jeweils letzte Session vorhanden | Gaze-Tracker wurde pro Session neu gestartet → alte Daten überschrieben |
| Exp 3 | Training auf 300 echten Samples (LOO-CV) | MAE 0.91 - 1.27 (echte Daten) | Modell lernt für jeweiligen Probanden |

## Results
- 300 gültige echte Samples über alle Probanden
- Funktionierende Ende-zu-Ende-Pipeline: Aufnahme → Features → Training → Vorhersage
- Alle Datenverlust-Ursachen identifiziert und behoben (Append-Modus, einheitliche Unix-Zeitbasis)

## Challenges
- Ein Teil der erhobenen Daten war unwiederbringlich verloren (überschriebene Gaze-CSVs) —
  die Antworten/Scores existieren, aber ohne zugehörige Blickdaten sind sie nicht nutzbar
- Zwei verschiedene Zeitbasen (pylsl `local_clock` vs. `time.time()`) machten die
  Zuordnung von Gaze zu Markern fehleranfällig

## Key Insights
- Aufnahme-Skripte müssen **anhängen statt überschreiben** — Neustarts passieren in der Praxis immer
- Alle Datenströme sollten von Anfang an **dieselbe Zeitbasis** verwenden
- Datenintegrität nach jeder Session prüfen, nicht erst am Ende der Erhebung

## Plan for Next Week
- Ablauf weiter vereinfachen, damit Bedienfehler ausgeschlossen sind (Idee: alles in einer Web-App)
- Data Augmentation gegen die kleine Stichprobe evaluieren
- Vorbereitung der Abschlusspräsentation beginnen

## Contributions
- Alle: gemeinsame Datenerhebung, Debugging des Datenverlusts, Test der Pipeline
