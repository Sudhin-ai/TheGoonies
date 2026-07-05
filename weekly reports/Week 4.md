# Week 04 Report — Machine Learning for Smart and Connected Systems (ML4SCS)

## Weekly Goal
Weitere Recherche betreiben und den Tobii Eye Tracker 5 bestellen. Eine erste Version
der Texte erstellen, mit denen die Konzentration getestet werden soll.

## Work Done This Week

### 0. Project Setup

**Projektfrage:**
Inwiefern lassen sich Blickbewegungsdaten (Tobii ET5) und Kopf-Kinematik (AirPods
Motion Data) kombinieren, um die kognitive Konzentration während einer Leseaufgabe
vorherzusagen?

**Test-Setup:**
Der Proband liest 5 Texte von jeweils 3–5 Minuten Länge und beantwortet zu jedem Text
ca. 10 Fragen. Die Texte sollen bei hoher Konzentration gerade so im Zeitintervall
lesbar sein.

**Geplante Daten:**

*Eye-Tracking-Daten:*
- **Gaze Points (x, y):** normierte Bildschirmkoordinaten (0 bis 1), an denen sich der
  Blick des Nutzers gerade befindet
- **Fixationen und Sakkaden:** Dauer, wie lange ein Bereich fixiert wird, sowie die
  Geschwindigkeit der schnellen Augenbewegungen zwischen zwei Punkten

*AirPods (IMU):*
- **Lineare Beschleunigung:** Bewegung des Kopfes entlang der drei Raumachsen (x, y, z)
- **Winkelgeschwindigkeit (Gyroskop):** Rotationsgeschwindigkeit, mit der der Kopf
  gedreht, geneigt oder geschüttelt wird
- **Head-Tracking-Daten:** Positionsdaten der AirPods, die erkennen, ob der Kopf ruhig
  gehalten oder unruhig bewegt wird

**Werkzeuge und Bibliotheken:**
Zur Aufzeichnung der Sensordaten benötigen wir die offizielle Tobii Stream Engine API
für den Eye-Tracker sowie eine Schnittstelle wie das Apple-CoreMotion-Framework für die
AirPods. Zum Speichern und Loggen der asynchronen Datenströme in CSV-Dateien reichen
voraussichtlich die Standard-Bibliotheken von Python aus.

## Challenges
- **Kein automatischer Datenexport (API-Stream):** Der Tobii speichert oder loggt von
  sich aus keine Daten. Die asynchronen Datenströme (60–90 Hz) müssen aktiv abgefangen
  und in Echtzeit in eine Datei (CSV/Text) geschrieben werden.
- **Fehlende Feature-Berechnung:** Der Tracker liefert nur rohe Gaze-Koordinaten.
  Metriken wie Fixationsdauer oder Sakkaden müssen über eigene Algorithmen
  (z. B. I-VT-Filter) berechnet werden.
- **Datensynchronisation:** Da die AirPods über Bluetooth und der Tobii über
  USB/Stream Engine laufen, müssen die asynchronen Datenströme zeitlich auf eine
  gemeinsame Zeitachse abgeglichen werden.

## Plan for Next Week
- Den Tobii Eye Tracker 5 ausprobieren und sich mit der Datenextraktion auseinandersetzen
- Daten aus den AirPods extrahieren und überlegen, wie diese eingesetzt werden können
- Texte und Fragen für das Experiment ausarbeiten

## Contributions
- Jonah: Recherche zur Datenextraktion des Tobii Eye Tracker 5
- Sudhin: erste Version der Texte
