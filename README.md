# Aufmerksamkeitsanalyse beim Lesen — ML4SCS Gruppenprojekt

Semesterprojekt für **Machine Learning for Smart and Connected Systems (ML4SCS)**.

Wir sagen das Textverständnis (Score 0–10 in einem Multiple-Choice-Quiz) allein aus
dem **Blickverhalten beim Lesen** vorher — aufgezeichnet mit einem **Tobii Eye Tracker 5**
und einem Random-Forest-Modell.

---

## Team — The Goonies

- Dario Pino
- Sudhin Hegde
- Kushal Shiva Patel

---

## Projektfrage

> Inwiefern lassen sich Blickbewegungsdaten (Tobii ET5) nutzen, um das
> Leseverständnis während einer Leseaufgabe vorherzusagen?

Ursprünglich war zusätzlich Kopf-Kinematik (AirPods Motion Data) geplant. Die
IMU-Analyse (Woche 5 & 10) zeigte jedoch zu wenig Aussagekraft beim ruhigen Lesen,
daher wurde das Projekt auf reine Gaze-Daten fokussiert. Die AirPods-Skripte und
-Auswertungen liegen im [archive/](archive/)-Ordner.

**Gültigkeitsbereich:** Alle Trainingsdaten stammen aus Durchläufen mit normaler
Leseintention. Das Modell sagt Verständnis-Scores *innerhalb aufmerksamen Lesens*
vorher — die Erkennung bewusster Unaufmerksamkeit war nicht Teil der Datenerhebung
(siehe Weekly Report Woche 13).

---

## Experiment-Setup

- **3 Probanden** (sudhin, kushal, dario)
- **80 Texte** (`texte.json`) mit je **10 Multiple-Choice-Fragen**
- Text wird auf der **rechten Bildschirmhälfte** gelesen (links läuft der
  Tobii-Gaze-Kreis über die Tobii Experience App)
- Der Tobii ET5 bietet keinen freien SDK-Zugriff → der Blickpunkt wird über
  **MOG2-Hintergrundsubtraktion** aus dem sichtbaren Gaze-Kreis auf dem
  Bildschirm rekonstruiert (`screen_gaze_tracker.py`)
- Nach jedem Text beantwortet der Proband die MC-Fragen → echter Score (0–10)

## Features (pro gelesenem Text)

| Feature | Bedeutung |
|---|---|
| `reading_time` | Lesedauer in Sekunden |
| `fixation_count` | Anzahl Fixationen (Geschwindigkeits-Schwellwert) |
| `fixation_duration_mean` | mittlere Fixationsdauer |
| `gaze_dispersion` | Streuung des Blicks (std x + std y) |
| `gaze_valid_ratio` | Anteil gültiger Gaze-Samples |
| `text_gaze_ratio` | Anteil der Zeit mit Blick im Textbereich |

## Modell

- **Random Forest Regressor** (scikit-learn, 200 Bäume)
- je ein **within-subject Modell** pro Proband + ein **gepooltes Modell**
- Evaluation per **Leave-One-Out-Kreuzvalidierung** (MAE)
- **Data Augmentation** (Gaussian Noise) gegen die kleine Stichprobe —
  transparent gekennzeichnet, Metriken werden auf echten Samples berechnet

---

## Der schnellste Weg: Web-App

```bash
pip install -r requirements.txt
python app.py
```

Die App (`app.py`) streamlinet den kompletten Ablauf in einer Website:
Probanden-Auswahl → Gaze-Tracker startet automatisch → Text lesen →
MC-Fragen → nächster Text oder Stopp. Im **Demo-Modus** wird nach dem Lesen
der vorhergesagte Score direkt mit dem echten Quiz-Ergebnis verglichen.

## Pipeline (manuell)

| Schritt | Skript | Ausgabe |
|---|---|---|
| 1. Gaze aufzeichnen | `screen_gaze_tracker.py --save raw_data/<name>_gaze.csv` | Gaze-CSV |
| 2. Experiment | `experiment.py --participant <name>` + `server.py` | Marker, Antworten |
| 3. Scores berechnen | `score.py` | `scores.csv` |
| 4. Features extrahieren | `extract_features.py` | `dataset.csv` |
| 5. (optional) Augmentieren | `augment_data.py --target 200` | `dataset_augmented.csv` |
| 6. Modell trainieren | `train_model.py --dataset dataset_final.csv` | `models/*.pkl` |
| 7. Vorhersage | `predict.py <name> <gaze.csv>` | Score 0–10 |

---

## Repository-Struktur

```
├── app.py                  # Streamlined Web-App (empfohlener Einstieg)
├── screen_gaze_tracker.py  # MOG2-basiertes Gaze-Tracking des Tobii-Kreises
├── experiment.py           # Tkinter-Experiment (Alternative zur Web-App)
├── server.py               # Flask-Server für MC-Fragen (Alternative)
├── features.py             # Geteilte Feature-Extraktion
├── extract_features.py     # Gaze + Marker + Scores → dataset.csv
├── augment_data.py         # Data Augmentation (Gaussian Noise)
├── train_model.py          # Random Forest + LOO-CV
├── predict.py              # Score-Vorhersage aus Gaze-CSV
├── score.py                # MC-Antworten → Scores
├── texte.json              # 80 Texte mit je 10 MC-Fragen
├── answer_key.csv          # Antwortschlüssel
├── raw_data/               # Gaze- und Marker-CSVs der Probanden
├── antworten/              # MC-Antworten pro Proband
├── scores/                 # Scores pro Proband
├── dataset*.csv            # Feature-Datensätze (echt / augmentiert)
├── docs/images/            # Screenshots & IMU-Plots
├── weekly reports/         # Wöchentliche Projektberichte
└── archive/                # AirPods-Skripte & alte Skript-Versionen
```
