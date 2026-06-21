# Week 04 Report — Machine Learning for Smart and Connected Systems (ML4SCS)

## Weekly Goal
-Weitere Recherche und bestellen vom Tobii Eyetracker 5. Erste Version unser Texte erstellen, mit welcher Konzentration getestet werden soll.

**Examples**
- Define the project question
- Explore the dataset
- Build a baseline pipeline

## Work Done This Week

### 0. Project setup
- Project Question: 
   -Inwiefern lassen sich Blickbewegungsdaten (Tobii ET5) und Kopf-Kinematik (AirPods Motion Data) kombinieren, um die kognitive Konzentration während einer Leseaufgabe vorherzusagen?
- Test SetUp: 
    -Probant liest 5 Texte von jeweils 3-5 Minuten und muss zu jedem Text ca 10. fragen beantworten. Texte müsse bei hohre Konzentration gerade so in dem Zeitintervall lesbar sein.
- Data Used: 
    -Eyetracking Data: 
      -Gaze Points ($x, y$): Die normierten Koordinaten (meist von 0 bis 1) auf dem Bildschirm, an denen sich der Blick des Nutzers gerade befindet.
      -Fixationen und Sakkaden: Die Dauer, wie lange ein Bereich fixiert wird, sowie die Geschwindigkeit der schnellen Augenbewegungen zwischen zwei Punkten
    -Airpods:
      -Beschleunigung (Linear Acceleration): Die Bewegung des Kopfes entlang der drei Raumachsen ($x, y, z$)
      -Winkelgeschwindigkeit (Gyroscope / Angular Velocity): Die Rotationsgeschwindigkeit, mit der der Kopf gedreht, geneigt oder geschüttelt wird.
      -Head Tracking-Daten: Die von den AirPods erzeugten Positionsdaten, die erkennen, ob der Kopf ruhig gehalten oder unruhig hin und her bewegt wird.
- What tools or libraries are used?
      -Um die Sensordaten aufzuzeichnen, benötigen wir die offizielle Tobii Stream Engine API für den Eye-Tracker sowie eine Schnittstelle wie das Apple CoreMotion Framework für die AirPods. Zum anschließenden Speichern und Loggen der asynchronen Datenströme in eine CSV-Datei reichen warscheinlich die Standard-Bibliotheken von Python aus.


### 1. Data Work
-

### 2. Analysis / Modeling Work
-

### 3. Repository / Documentation Work
-

## Experiments Conducted
-

## Results
-

## Challenges
- Kein automatischer Datenexport (API-Stream): Der Tobii speichert oder loggt von sich aus keine Daten. Ihr müsst die asynchronen Datenströme (60–90 Hz) aktiv abfangen und in Echtzeit in eine Datei (CSV/Text) schreiben.
- Fehlende Feature-Berechnung: Der Tracker liefert nur rohe Gaze-Koordinaten. Metriken wie Fixationsdauer oder Sakkaden müssen von euch über Algorithmen (z.B. I-VT Filter) selbst berechnet werden.
- Datensynchronisation: Da die AirPods über Bluetooth und der Tobii über USB/Stream Engine laufen, müssen die asynchronen Datenströme zeitlich auf eine gemeinsame Zeitachse abgeglichen werden.

## Key Insights
- What did you learn this week?

## Plan for Next Week
- Den Tobii EyeTracker 5 ausprobieren und mit Datenextrahierung auseinandersetzen
- Daten aus AIrpods extrahieren und überlegen wie wir diese einsetzen
- Texte und Fragen für Experiment ausarbeiten

## Contributions
- Jonah: Recherche zur Datenextraktion des Tobii Eye-Tracker 5
- Sudhin: erste version der Tex
