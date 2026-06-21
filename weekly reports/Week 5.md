# Week 05 Report — Machine Learning for Smart and Connected Systems (ML4SCS)

## Weekly Goal
-Den Tobii EyeTracker 5 ausprobieren und mit Datenextrahierung auseinandersetzen

-Daten aus AIrpods extrahieren und überlegen wie wir diese einsetzen

-Texte und Fragen für Experiment ausarbeiten

## Work Done This Week

### 0. Project setup

Project Question:

  -Inwiefern lassen sich Blickbewegungsdaten (Tobii ET5) und Kopf-Kinematik (AirPods Motion Data) kombinieren, um die kognitive Konzentration während einer Leseaufgabe vorherzusagen?

Test SetUp: 

  -Probant liest 5 Texte von jeweils 3-5 Minuten und muss zu jedem Text ca 10. fragen beantworten. Texte müsse bei hohre Konzentration gerade so in dem Zeitintervall lesbar sein.

### 1. Data Work
-Erste Bewegungsdaten wurden mit den Airpods gesammelt. Einmal wurde sich dabei kaum bewegt, um Konzentration zu simulieren. Beim zweiten mal wurde sich mehr bewegt um ein unruhiges/unkonzentriertes Verhalten zu simulieren.

### 2. Analysis / Modeling Work
-Die Beiden Bewegungsdaten der AirPods wurden in Pythen ausgwertet und visualisiert:


magnitude_mean-Durchschnittliche Bewegungsstärke                            ruhig=0.9971  unruhig=0.9980  Δ=+0.1%

magnitude_std-Schwankung der Bewegungsstärke                                ruhig=0.0041  unruhig=0.0342  Δ=+738.5%

stillness_ratio-Anteil echter Ruhe-Momente                                  ruhig=0.0000  unruhig=0.0000  Δ=+0.0%

pitch_range-pitch_range — Gesamtbereich der Vor-/Rückneigung                ruhig=0.1288  unruhig=0.9524  Δ=+639.3%

yaw_range-Gesamtbereich der Links-/Rechtsdrehung                            ruhig=0.2505  unruhig=1.9468  Δ=+677.1%

Es ist ein klarer Unterschied zwischen ruhigem und unruhigem Verhalten erkennbar. 

Taugt zur Unterscheidung?

magnitude_mean	Nein (0.1% Unterschied)

magnitude_std	Ja (738%)

stillness_ratio	Nein (Schwellenwert zu niedrig)

pitch_range	Ja (639%)

yaw_range	Ja (677%)
<img width="1417" height="495" alt="image" src="https://github.com/user-attachments/assets/99a600b5-4d60-453f-a4ea-c77687b4c8a1" />

### 3. Repository / Documentation Work
- Testdatei um AirPoddaten auszuwerten.


## Challenges

## Key Insights
-Wie man AirPod daten bekommt/benutzt

## Plan for Next Week
-Tobi EyeTracker 5 SetUp
-Quellarbeit
-Test genauer definieren

## Contributions
- Jonah: AirPods Anlayse
- Sudhin: Recherche nach Umsetzung
- Kushal: Recherche nach Quellen
  
