"""
server.py
=========
Lokaler Flask-Webserver für die MC-Fragen des Leseexperiments.

Starten (einmalig, läuft während des gesamten Experiments):
    python server.py

Routen:
    GET  /text/<text_id>?participant=<name>  → Fragebogen anzeigen
    POST /text/<text_id>                     → Antworten speichern
    GET  /done                               → Bestätigungsseite

Speichert:
    antworten/<participant>_antworten.csv   (text_id, question_id, answer)

Installation:
    pip install flask
"""

import csv
import json
import os

from flask import Flask, redirect, render_template_string, request, url_for

# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------

TEXTE_JSON   = "texte.json"
ANTWORTEN_DIR = "antworten"

# ---------------------------------------------------------------------------
# Texte laden
# ---------------------------------------------------------------------------

def load_texte(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        for key in ("texte", "texts", "items"):
            if key in raw:
                raw = raw[key]
                break
    return {int(entry["text_id"]): entry for entry in raw}


TEXTE = load_texte(TEXTE_JSON)

# ---------------------------------------------------------------------------
# Flask-App
# ---------------------------------------------------------------------------

app = Flask(__name__)

# ---------------------------------------------------------------------------
# HTML-Templates
# ---------------------------------------------------------------------------

FRAGEN_HTML = """
<!doctype html>
<html lang="de">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Fragen – Text {{ text_id }}</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: "Segoe UI", sans-serif;
      background: #f4f4f4;
      color: #111;
      padding: 40px 20px;
    }
    .container {
      max-width: 760px;
      margin: 0 auto;
      background: #fff;
      border-radius: 8px;
      padding: 40px;
      box-shadow: 0 2px 12px rgba(0,0,0,.12);
    }
    h1 {
      font-size: 1.3rem;
      margin-bottom: 6px;
      color: #222;
    }
    .subtitle {
      font-size: 0.9rem;
      color: #666;
      margin-bottom: 32px;
    }
    .question-block {
      margin-bottom: 28px;
      padding-bottom: 24px;
      border-bottom: 1px solid #eee;
    }
    .question-block:last-of-type {
      border-bottom: none;
    }
    .question-text {
      font-weight: 600;
      margin-bottom: 12px;
      line-height: 1.5;
    }
    .option label {
      display: flex;
      align-items: flex-start;
      gap: 10px;
      padding: 8px 12px;
      border-radius: 5px;
      cursor: pointer;
      transition: background .15s;
      line-height: 1.4;
    }
    .option label:hover { background: #f0f0f0; }
    .option input[type=radio] { margin-top: 3px; flex-shrink: 0; }
    .btn-submit {
      display: block;
      width: 100%;
      margin-top: 36px;
      padding: 14px;
      background: #2563eb;
      color: #fff;
      font-size: 1rem;
      font-weight: 700;
      border: none;
      border-radius: 6px;
      cursor: pointer;
      letter-spacing: .3px;
      transition: background .15s;
    }
    .btn-submit:hover { background: #1d4ed8; }
    .progress {
      font-size: 0.85rem;
      color: #888;
      text-align: right;
      margin-bottom: 20px;
    }
  </style>
</head>
<body>
<div class="container">
  <h1>{{ title }}</h1>
  <p class="subtitle">Text {{ text_id }} &nbsp;|&nbsp; Teilnehmer: <strong>{{ participant }}</strong></p>

  <form method="POST" action="/text/{{ text_id }}">
    <input type="hidden" name="participant" value="{{ participant }}">

    {% for q in questions %}
    <div class="question-block">
      <p class="question-text">{{ loop.index }}. {{ q.question }}</p>
      {% for key, option_text in q.options.items() %}
      <div class="option">
        <label>
          <input type="radio" name="q{{ q.question_id }}" value="{{ key }}" required>
          <span><strong>{{ key }}</strong> &nbsp;{{ option_text }}</span>
        </label>
      </div>
      {% endfor %}
    </div>
    {% endfor %}

    <button type="submit" class="btn-submit">Antworten absenden →</button>
  </form>
</div>
</body>
</html>
"""

DONE_HTML = """
<!doctype html>
<html lang="de">
<head>
  <meta charset="utf-8">
  <title>Antworten gespeichert</title>
  <style>
    body {
      font-family: "Segoe UI", sans-serif;
      display: flex;
      align-items: center;
      justify-content: center;
      min-height: 100vh;
      background: #f4f4f4;
      margin: 0;
    }
    .box {
      text-align: center;
      background: #fff;
      border-radius: 10px;
      padding: 60px 80px;
      box-shadow: 0 2px 16px rgba(0,0,0,.12);
    }
    .check { font-size: 3rem; margin-bottom: 16px; }
    h1 { font-size: 1.5rem; color: #111; margin-bottom: 10px; }
    p  { color: #555; font-size: 1rem; }
  </style>
</head>
<body>
<div class="box">
  <div class="check">✓</div>
  <h1>Antworten gespeichert</h1>
  <p>Kehre zum Experiment-Fenster zurück<br>und drücke die <strong>Leertaste</strong> für den nächsten Text.</p>
</div>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# Routen
# ---------------------------------------------------------------------------

@app.route("/text/<int:text_id>", methods=["GET"])
def show_questions(text_id):
    if text_id not in TEXTE:
        return f"Text {text_id} nicht gefunden.", 404

    entry       = TEXTE[text_id]
    participant = request.args.get("participant", "unbekannt")

    return render_template_string(
        FRAGEN_HTML,
        text_id=text_id,
        title=entry.get("title", ""),
        participant=participant,
        questions=entry.get("questions", []),
    )


@app.route("/text/<int:text_id>", methods=["POST"])
def save_answers(text_id):
    if text_id not in TEXTE:
        return f"Text {text_id} nicht gefunden.", 404

    participant = request.form.get("participant", "unbekannt")
    questions   = TEXTE[text_id].get("questions", [])

    os.makedirs(ANTWORTEN_DIR, exist_ok=True)
    csv_path = os.path.join(ANTWORTEN_DIR, f"{participant}_antworten.csv")

    # Datei anlegen falls neu, sonst anhängen
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["text_id", "question_id", "answer"])
        for q in questions:
            qid    = q["question_id"]
            answer = request.form.get(f"q{qid}", "")
            writer.writerow([text_id, qid, answer])

    print(f"  Antworten gespeichert: {csv_path} (Text {text_id}, {participant})")
    return render_template_string(DONE_HTML)


@app.route("/")
def index():
    return "<h2>Server läuft. Öffne <code>/text/&lt;text_id&gt;?participant=&lt;name&gt;</code></h2>"


# ---------------------------------------------------------------------------
# Start
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 50)
    print("  Fragen-Server  →  http://localhost:5000")
    print("=" * 50)
    app.run(host="127.0.0.1", port=5000, debug=False)
