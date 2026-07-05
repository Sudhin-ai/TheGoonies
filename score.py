"""
score.py
========
Wertet die Antworten eines Teilnehmers aus und erstellt scores.csv.

Nutzung:
    python score.py --participant sudhin
    python score.py --participant kushal
    python score.py --participant dario
    python score.py --participant sudhin --all   # alle Teilnehmer zusammen

Liest:
    antworten/<participant>_antworten.csv   (text_id, question_id, answer)
    answer_key.csv                          (text_id, title, question_id, correct)
        Falls answer_key.csv fehlt, wird sie automatisch aus texte.json generiert.

Schreibt:
    scores/<participant>_scores.csv         (text_id, score)
    scores.csv                              (participant, text_id, score) – für train_model.py
"""

import argparse
import csv
import json
import os

# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------

TEXTE_JSON    = "texte.json"
ANSWER_KEY    = "answer_key.csv"
ANTWORTEN_DIR = "antworten"
SCORES_DIR    = "scores"
COMBINED_CSV  = "scores.csv"          # Eingabe für train_model.py

# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def generate_answer_key(texte_path, out_path):
    """Erstellt answer_key.csv aus texte.json."""
    with open(texte_path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    if content.startswith("{"):
        raw = [json.loads(line) for line in content.splitlines() if line.strip()]
    else:
        raw = json.loads(content)

    if isinstance(raw, dict):
        for key in ("texte", "texts", "items"):
            if key in raw:
                raw = raw[key]
                break

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text_id", "title", "question_id", "correct"])
        for entry in raw:
            tid   = entry["text_id"]
            title = entry.get("title", "")
            for q in entry.get("questions", []):
                writer.writerow([tid, title, q["question_id"], q["correct"]])

    print(f"answer_key.csv generiert ({out_path})")


def load_answer_key(path):
    """Gibt dict (text_id, question_id) → correct zurück."""
    key = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tid = int(row["text_id"])
            qid = int(row["question_id"])
            key[(tid, qid)] = str(row["correct"]).strip().upper()
    return key


def load_antworten(path):
    """Gibt liste von (text_id, question_id, answer) zurück."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append((
                int(row["text_id"]),
                int(row["question_id"]),
                str(row["answer"]).strip().upper(),
            ))
    return rows


def compute_scores(antworten, answer_key):
    """Gibt Liste von (text_id, occurrence, score) zurück.

    Wird ein Text mehrfach gelesen (Loop nach Text 80), stehen seine
    Antworten als getrennte aufeinanderfolgende Blöcke in der CSV.
    Jeder Block wird als eigener Durchgang (occurrence 0, 1, ...) gewertet."""
    runs = []          # Liste von (tid, [(qid, answer), ...])
    for tid, qid, answer in antworten:
        if runs and runs[-1][0] == tid:
            runs[-1][1].append((qid, answer))
        else:
            runs.append((tid, [(qid, answer)]))

    results = []
    occ_counter = {}
    for tid, block in runs:
        occ = occ_counter.get(tid, 0)
        occ_counter[tid] = occ + 1
        score = 0
        for qid, answer in block:
            correct = answer_key.get((tid, qid))
            if correct is None:
                print(f"  Warnung: Kein Schlüssel für Text {tid} Frage {qid} – übersprungen.")
                continue
            if answer == correct:
                score += 1
        results.append((tid, occ, score))
    return results


# ---------------------------------------------------------------------------
# Hauptprogramm
# ---------------------------------------------------------------------------

def process(participant, answer_key):
    antworten_path = os.path.join(ANTWORTEN_DIR, f"{participant}_antworten.csv")
    if not os.path.exists(antworten_path):
        print(f"  Nicht gefunden: {antworten_path}")
        return []

    antworten = load_antworten(antworten_path)
    scores    = compute_scores(antworten, answer_key)

    os.makedirs(SCORES_DIR, exist_ok=True)
    out_path = os.path.join(SCORES_DIR, f"{participant}_scores.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text_id", "occurrence", "score"])
        for tid, occ, score in scores:
            writer.writerow([tid, occ, score])
            occ_label = f" (Durchgang {occ + 1})" if occ else ""
            print(f"  Text {tid:2d}{occ_label}: {score:2d}/10")

    print(f"  Gespeichert: {out_path}")
    return [(participant, tid, occ, score) for tid, occ, score in scores]


def main():
    parser = argparse.ArgumentParser(description="Antworten auswerten")
    parser.add_argument("--participant", default=None,
                        help="Teilnehmer-ID (sudhin, kushal, dario). "
                             "Ohne Angabe werden alle ausgewertet.")
    parser.add_argument("--all", action="store_true",
                        help="Alle Teilnehmer auswerten")
    parser.add_argument("--texte", default=TEXTE_JSON,
                        help=f"Pfad zur texte.json (Standard: {TEXTE_JSON})")
    args = parser.parse_args()

    # answer_key.csv erzeugen falls nicht vorhanden
    if not os.path.exists(ANSWER_KEY):
        if not os.path.exists(args.texte):
            print(f"Fehler: Weder {ANSWER_KEY} noch {args.texte} gefunden.")
            return
        generate_answer_key(args.texte, ANSWER_KEY)

    answer_key = load_answer_key(ANSWER_KEY)
    print(f"Antwortschlüssel: {len(answer_key)} Einträge")

    # Teilnehmer ermitteln
    if args.all or args.participant is None:
        # alle *_antworten.csv im Ordner
        if not os.path.isdir(ANTWORTEN_DIR):
            print(f"Ordner '{ANTWORTEN_DIR}' nicht gefunden.")
            return
        import glob
        paths = glob.glob(os.path.join(ANTWORTEN_DIR, "*_antworten.csv"))
        participants = [os.path.basename(p).replace("_antworten.csv", "")
                        for p in sorted(paths)
                        if not p.endswith("_demo_antworten.csv")]
    else:
        participants = [args.participant]

    if not participants:
        print("Keine Antwortdateien gefunden.")
        return

    # Auswerten
    all_rows = []
    for p in participants:
        print(f"\n=== {p} ===")
        all_rows.extend(process(p, answer_key))

    # Kombinierte scores.csv für train_model.py aktualisieren
    if all_rows:
        # Bestehende Einträge laden (andere Teilnehmer behalten)
        existing = {}
        if os.path.exists(COMBINED_CSV):
            with open(COMBINED_CSV, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    occ = int(row.get("occurrence", 0) or 0)
                    key = (str(row["participant"]), int(row["text_id"]), occ)
                    existing[key] = int(row["score"])

        # Neue Einträge eintragen (überschreiben)
        for participant, tid, occ, score in all_rows:
            existing[(str(participant), tid, occ)] = score

        with open(COMBINED_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["participant", "text_id", "occurrence", "score"])
            for (p, tid, occ), score in sorted(existing.items()):
                writer.writerow([p, tid, occ, score])

        print(f"\nscores.csv aktualisiert ({len(existing)} Einträge gesamt)")

    print("\nFertig.")


if __name__ == "__main__":
    main()
