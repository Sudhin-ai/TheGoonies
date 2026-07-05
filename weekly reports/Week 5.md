# Week 05 Report — Machine Learning for Smart and Connected Systems (ML4SCS)

## Weekly Goal
- Den Tobii Eye Tracker 5 ausprobieren und sich mit der Datenextraktion auseinandersetzen
- Daten aus den AirPods extrahieren und überlegen, wie diese eingesetzt werden können
- Texte und Fragen für das Experiment ausarbeiten

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

### 1. Data Work
Erste Bewegungsdaten wurden mit den AirPods gesammelt. In einem Durchlauf wurde sich
kaum bewegt, um konzentriertes Verhalten zu simulieren; im zweiten Durchlauf wurde sich
bewusst mehr bewegt, um unruhiges/unkonzentriertes Verhalten zu simulieren.

### 2. Analysis / Modeling Work
Die beiden Bewegungsaufnahmen der AirPods wurden in Python ausgewertet und visualisiert:

| Feature | Bedeutung | ruhig | unruhig | Δ |
|---|---|---|---|---|
| `magnitude_mean` | durchschnittliche Bewegungsstärke | 0.9971 | 0.9980 | +0.1 % |
| `magnitude_std` | Schwankung der Bewegungsstärke | 0.0041 | 0.0342 | +738.5 % |
| `stillness_ratio` | Anteil echter Ruhe-Momente | 0.0000 | 0.0000 | +0.0 % |
| `pitch_range` | Gesamtbereich der Vor-/Rückneigung | 0.1288 | 0.9524 | +639.3 % |
| `yaw_range` | Gesamtbereich der Links-/Rechtsdrehung | 0.2505 | 1.9468 | +677.1 % |

Es ist ein klarer Unterschied zwischen ruhigem und unruhigem Verhalten erkennbar.

**Eignung zur Unterscheidung:**

| Feature | Geeignet? |
|---|---|
| `magnitude_mean` | Nein (nur 0.1 % Unterschied) |
| `magnitude_std` | Ja (+738 %) |
| `stillness_ratio` | Nein (Schwellenwert zu niedrig) |
| `pitch_range` | Ja (+639 %) |
| `yaw_range` | Ja (+677 %) |

<img width="1417" height="495" alt="image" src="https://github.com/user-attachments/assets/99a600b5-4d60-453f-a4ea-c77687b4c8a1" />

### 3. Repository / Documentation Work
- Testdatei zur Auswertung der AirPods-Daten erstellt

## Key Insights
- Wie man AirPods-Daten ausliest und weiterverarbeitet

## Plan for Next Week
- Setup des Tobii Eye Tracker 5
- Quellenarbeit
- Test genauer definieren

## Contributions
- Jonah: AirPods-Analyse
- Sudhin: Recherche zur Umsetzung
- Kushal: Recherche nach Quellen
