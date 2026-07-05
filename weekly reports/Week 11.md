# Week 11 Report — Machine Learning for Smart and Connected Systems (ML4SCS)

## Weekly Goal
Die Skripte finalisieren, weitere Daten sammeln und eine Automation erstellen, um den
Testablauf zu verbessern und zu beschleunigen.

## Work Done This Week

### 3. Repository / Documentation Work
- `screen_gaze_tracker.py` — finales Tracking-Skript
- `train_model.py` — Training des Random-Forest-Modells
- `predict.py` — Score-Vorhersage aus Gaze-Daten
- `experiment.py` — Automation des Testablaufs (Textanzeige, Marker)
- `server.py` — Automation der MC-Fragen über einen lokalen Webserver
- `send_markers.py` — sendet nach dem Beantworten der MC-Fragen die Ergebnisse, die mit
  dem Antwortschlüssel abgeglichen werden, um die Anzahl richtiger Antworten zu ermitteln

## Results
- Der Ordner `antworten/` enthält die Antworten der jeweiligen Probanden
- `scores.csv` zeigt das Ergebnis der MC-Fragen pro Proband (x/10)

## Challenges
- Wir sind mit der Vielfältigkeit der Texte nicht zufrieden — da sehr viele Texte
  gelesen werden müssen, sollen sie abwechslungsreicher aufgebaut werden

## Key Insights
- Die Automation war sehr hilfreich, um Zeit zu sparen und den Ablauf zu vereinfachen

## Plan for Next Week
- Mehr Daten erheben
- Texte vielfältiger aufbauen und finalisieren

## Contributions
- Alle: gemeinsame Arbeit an Automation und Datenerhebung
