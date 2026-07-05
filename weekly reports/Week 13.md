# Week 13 Report — Machine Learning for Smart and Connected Systems (ML4SCS)

## Weekly Goal
Experiment-Ablauf in einer Web-App streamlinen, Modell finalisieren und die
Abschlusspräsentation erstellen.

## Work Done This Week

### 1. Data Work
- **Data Augmentation** implementiert (`augment_data.py`): Gaussian Noise auf alle
  Gaze-Features, Scores leicht variiert, physikalische Grenzen eingehalten
  (z. B. Ratios auf 0–1 begrenzt)
- Datensatz pro Proband auf **200 Samples** aufgestockt (600 gesamt), echte und
  synthetische Samples über eine `augmented`-Spalte unterscheidbar
- Augmentierung wird transparent dokumentiert — synthetische Daten dienen nur dem
  Training, nicht der Bewertung

### 2. Analysis / Modeling Work
- Finale Random-Forest-Modelle trainiert: je ein within-subject Modell pro Proband
  plus ein **gepooltes Modell** über alle Teilnehmer
- **Negativtest** durchgeführt: absichtlich unaufmerksamer Durchlauf (Text kaum gelesen,
  zufällig geantwortet) → echtes Ergebnis 3/10, Modell sagte 10/10 vorher
- Erkenntnis: Das Modell kennt nur aufmerksames Lesen (range restriction) — bei den
  Trainingsdaten korrelierte kurze Lesezeit mit *hohen* Scores, daher wird überfliegendes
  Lesen falsch eingeordnet. Diese Limitation wird in der Präsentation offen benannt.

### 3. Repository / Documentation Work
- **`app.py`** erstellt: der komplette Ablauf läuft jetzt in einer einzigen Web-App —
  Probanden-Auswahl, automatischer Start des Gaze-Trackers, Textanzeige, MC-Fragen,
  Weiter/Stopp, alles auf einer Seite; nur noch ein Terminal-Befehl nötig
- **Demo-Modus** in der App: nach dem Lesen wird der vorhergesagte Score direkt neben dem
  echten Quiz-Ergebnis angezeigt, inklusive Fehlertoleranz (MAE) des Modells
- Anzeige-Logik für Vorhersagen: Runden auf ganze Scores, bei Unsicherheit zwischen zwei
  Werten Anzeige als Halbschritt (z. B. „7.5")
- Repository aufgeräumt: alte Skript-Versionen und AirPods-Dateien nach `archive/`,
  Screenshots nach `docs/images/`, README vollständig überarbeitet, `requirements.txt` ergänzt
- **Abschlusspräsentation (PowerPoint)** erstellt: Fragestellung, Methodik (Tobii + MOG2,
  Feature-Extraktion, Random Forest), Ergebnisse mit Grafiken, Live-Demo, Limitationen und Fazit

## Experiments Conducted

| Experiment | Change Made | Result | Interpretation |
|-----------|-------------|--------|----------------|
| Exp 1 | Augmentation (Gaussian Noise, 200 Samples/Proband) | Training stabiler, LOO-CV MAE ≈ 0.37–0.40 | Metrik durch synthetische Zwillinge optimistisch — ehrliche Referenz bleibt die MAE auf echten Daten |
| Exp 2 | Live-Vorhersage über Demo-Modus (aufmerksames Lesen) | Vorhersage innerhalb der Fehlertoleranz | Pipeline funktioniert Ende-zu-Ende in Echtzeit |
| Exp 3 | Negativtest: absichtlich unaufmerksames Lesen | Echt 3/10, vorhergesagt 10/10 | Modell kann nur den trainierten Bereich (aufmerksames Lesen) abdecken |

## Results
- Vollständig automatisierter Experiment- und Demo-Ablauf über eine Website
- Finale Modelle: within-subject (je 200 Samples) + gepooltes Modell (600 Samples)
- Live-Demo funktioniert: Text lesen → Vorhersage vs. echtes Quiz-Ergebnis in Sekunden
- Abschlusspräsentation fertiggestellt

## Challenges
- Vorhersagen außerhalb des Trainingsbereichs (unaufmerksames Lesen) sind nicht möglich —
  dafür wären gezielt „schlechte" Durchläufe als Trainingsdaten nötig gewesen
- Augmentierte Daten verbessern das Training, dürfen aber nicht unreflektiert in die
  Evaluation einfließen

## Key Insights
- Ein streamlineter Ablauf (ein Befehl, eine Website) verhindert genau die Bedienfehler,
  die uns in den Vorwochen Daten gekostet haben
- Data Augmentation ersetzt keine echten Daten — jede echte Lesung ist mehr wert als
  viele synthetische Kopien
- Limitationen offen zu benennen (Gültigkeitsbereich: aufmerksames Lesen) ist stärker,
  als eine scheinbar perfekte Demo zu zeigen

## Plan for Next Week
- Abschlusspräsentation halten
- Projekt abschließen und Repository finalisieren

## Contributions
- Alle: gemeinsame Arbeit an Web-App, Modell-Finalisierung, Tests und Präsentation
