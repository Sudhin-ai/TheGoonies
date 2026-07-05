# Week 10 Report — Machine Learning for Smart and Connected Systems (ML4SCS)

## Weekly Goal
Das Eye-Tracking-Skript finalisieren, Skripte zur Datenextraktion erstellen und
erste Daten erfassen.

## Work Done This Week

### 3. Repository / Documentation Work
- `screen_gaze_tracker (3)` und `(4)` — Weiterentwicklung des Tracking-Skripts, um den
  Eye-Tracker zu verfolgen und Gaze-Daten zu generieren
- `extract_features.py` und `features.py` — Extraktion der Features aus den Gaze-Daten

## Results
- **IMU-Daten_1 bis IMU-Daten_10** (siehe `docs/images/`): gesammelte AirPods-IMU-Daten
  während der Lesetests

Die AirPods-Daten schwanken beim Lesen kaum und bleiben auch bei unterschiedlichen
Scores sehr ähnlich — sie besitzen damit keine Aussagekraft über die Aufmerksamkeit
beim Lesen eines Textes.

## Challenges
- Die Skripte müssen so angepasst werden, dass sie keine AirPods-Daten mehr einbeziehen

## Key Insights
- Die AirPods-Daten zeigen zu wenig Aussagekraft und werden in diesem Projekt nicht
  mehr einbezogen

## Plan for Next Week
- Skripte anpassen (AirPods-Daten entfernen)
- Zusätzliche Skripte zum Trainieren des Modells und zur Auswertung der Daten erstellen
- Weitere Texte für Testdurchläufe schreiben

## Contributions
- Alle: gemeinsame Datenerfassung und Skript-Entwicklung
