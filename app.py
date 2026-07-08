"""
app.py
======
Streamlinete Web-Oberfläche für das gesamte Leseexperiment.
Ersetzt experiment.py + server.py + predict.py durch EINE Website.

Ablauf:
    1. python app.py               → Browser öffnet sich auf rechter Bildschirmhälfte
    2. Teilnehmer + Modus wählen   → Gaze-Tracker startet automatisch
    3. Text lesen → "Weiter"       → MC-Fragen auf derselben Seite
    4. Normal:  [Weiter] nächster Text  /  [Stopp] speichern & beenden
       Demo:    Vorhersage vs. echtes Ergebnis → [Beenden] zurück zum Start

Die alten Skripte (experiment.py, server.py, predict.py) bleiben unverändert
als Fallback nutzbar. Datenformate sind identisch.
"""

import csv
import ctypes
import json
import math
import os
import subprocess
import sys
import threading
import time

import joblib
from flask import Flask, redirect, render_template_string, request

# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------

TEXTE_JSON      = "texte.json"
DEMO_TEXTE_JSON = "demo_texte.json"
PROGRESS_FILE = "progress.json"
ANSWER_KEY    = "answer_key.csv"
ANTWORTEN_DIR = "antworten"
RAW_DATA      = "raw_data"
MODELS_DIR    = "models"
PARTICIPANTS  = ["sudhin", "kushal", "dario"]
PORT          = 5000

PYTHON  = sys.executable
TRACKER = "screen_gaze_tracker.py"

# ---------------------------------------------------------------------------
# Texte & Progress
# ---------------------------------------------------------------------------

def load_texte(path):
    with open(path, "r", encoding="utf-8") as f:
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
    return {int(e["text_id"]): e for e in raw}


TEXTE = load_texte(TEXTE_JSON)
TEXT_IDS = sorted(TEXTE.keys())

# Eigene Textsammlung nur für den Demo-Modus (falls vorhanden)
DEMO_TEXTE = load_texte(DEMO_TEXTE_JSON) if os.path.exists(DEMO_TEXTE_JSON) else {}
DEMO_TEXT_IDS = sorted(DEMO_TEXTE.keys())


def active_texte():
    """Liefert die passende Textsammlung je nach Modus."""
    if S.mode == "demo" and DEMO_TEXTE:
        return DEMO_TEXTE
    return TEXTE


def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r", encoding="utf-8-sig") as f:
            return json.load(f)
    return {p: 0 for p in PARTICIPANTS}


def save_progress(progress):
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2)


def next_text_id(participant):
    """Nächster Text laut progress.json. Nach dem letzten Text wird wieder
    von vorne begonnen (Endlos-Loop)."""
    done = load_progress().get(participant, 0)
    return TEXT_IDS[done % len(TEXT_IDS)]

# ---------------------------------------------------------------------------
# Marker & Antworten
# ---------------------------------------------------------------------------

def write_marker(markers_path, label, text_id):
    file_exists = os.path.exists(markers_path)
    with open(markers_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(["timestamp", "label", "text_id"])
        w.writerow([f"{time.time():.6f}", label, text_id])


def save_answers(participant, text_id, form, demo=False):
    os.makedirs(ANTWORTEN_DIR, exist_ok=True)
    suffix = "_demo_antworten.csv" if demo else "_antworten.csv"
    path = os.path.join(ANTWORTEN_DIR, f"{participant}{suffix}")
    write_header = not os.path.exists(path)
    answers = []
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["text_id", "question_id", "answer"])
        for q in active_texte()[text_id].get("questions", []):
            qid = q["question_id"]
            ans = form.get(f"q{qid}", "")
            w.writerow([text_id, qid, ans])
            answers.append((qid, ans))
    return answers


def compute_actual_score(text_id, answers):
    correct = {q["question_id"]: str(q["correct"]).strip().upper()
               for q in active_texte()[text_id].get("questions", [])}
    return sum(1 for qid, ans in answers
               if correct.get(qid) == str(ans).strip().upper())

# ---------------------------------------------------------------------------
# Gaze-Tracker Subprozess
# ---------------------------------------------------------------------------

class Session:
    """Globaler Zustand des aktuellen Durchlaufs (eine Person zur Zeit)."""
    participant = None
    mode        = None    # "normal" | "demo"
    model       = "personal"  # "personal" | "pooled"
    text_id     = None
    tracker     = None    # subprocess.Popen
    gaze_path   = None
    markers_path = None
    demo_start  = None    # Unix-Zeit Lesebeginn (Demo)
    demo_end    = None


S = Session()


def start_tracker(gaze_path):
    stop_tracker()
    proc = subprocess.Popen(
        [PYTHON, TRACKER, "--save", gaze_path],
        cwd=os.path.dirname(os.path.abspath(__file__)),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    time.sleep(2.0)  # MOG2 braucht kurz zum Initialisieren
    if proc.poll() is not None:
        raise RuntimeError("Gaze-Tracker sofort beendet – Tobii-Kreis aktiv? "
                           "Test: python screen_gaze_tracker.py --save test.csv")
    return proc


def stop_tracker():
    if S.tracker is not None and S.tracker.poll() is None:
        S.tracker.terminate()
        try:
            S.tracker.wait(timeout=5)
        except subprocess.TimeoutExpired:
            S.tracker.kill()
    S.tracker = None

# ---------------------------------------------------------------------------
# Vorhersage (Demo-Modus)
# ---------------------------------------------------------------------------

def predict_score(participant, gaze_path, t_start, t_end, model="personal"):
    """Gibt (score, mae, model_name) zurück. mae ist None wenn nicht gespeichert."""
    from features import crop_stream_to_window, extract_gaze, gaze_csv_to_stream
    import pandas as pd

    if model == "pooled":
        model_path = os.path.join(MODELS_DIR, "pooled.pkl")
        model_name = "Gepoolt"
    else:
        model_path = os.path.join(MODELS_DIR, f"{participant}.pkl")
        model_name = participant
        if not os.path.exists(model_path):
            model_path = os.path.join(MODELS_DIR, "pooled.pkl")
            model_name = "Gepoolt (Fallback)"
    bundle = joblib.load(model_path)

    stream = gaze_csv_to_stream(gaze_path)
    seg = crop_stream_to_window(stream, t_start, t_end)
    row = extract_gaze(seg)
    sample = pd.DataFrame([row])[bundle["feature_cols"]]
    return float(bundle["model"].predict(sample)[0]), bundle.get("mae"), model_name


def display_score(value):
    """Rundet die Vorhersage auf eine ganze Zahl (0–10). Der echte Quiz-Score
    ist ganzzahlig, und bei einer MAE von ~1 Punkt täuschte eine Halbschritt-
    Anzeige eine Präzision vor, die das Modell nicht hat."""
    result = int(round(value))
    return str(max(0, min(10, result)))

# ---------------------------------------------------------------------------
# HTML (gemeinsames Layout)
# ---------------------------------------------------------------------------

BASE_CSS = """
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: "Segoe UI", sans-serif; background: #fff; color: #111;
         padding: 40px 32px; line-height: 1.6; }
  h1 { font-size: 1.4rem; margin-bottom: 20px; }
  .muted { color: #777; font-size: .9rem; margin-bottom: 24px; }
  .btn { display: inline-block; padding: 13px 30px; font-size: 1rem;
         font-weight: 700; border: none; border-radius: 6px; cursor: pointer;
         text-decoration: none; text-align: center; }
  .btn-blue  { background: #2563eb; color: #fff; }
  .btn-green { background: #16a34a; color: #fff; }
  .btn-red   { background: #dc2626; color: #fff; }
  .btn-gray  { background: #6b7280; color: #fff; }
  .btn:hover { filter: brightness(.9); }
  .row { display: flex; gap: 14px; margin-top: 28px; }
  .card { max-width: 860px; margin: 0 auto; }
  select { font-size: 1.05rem; padding: 10px 14px; border-radius: 6px;
           border: 1px solid #bbb; width: 100%; margin-bottom: 18px; }
  label.radio { display: block; padding: 10px 14px; border: 1px solid #ddd;
                border-radius: 6px; margin-bottom: 10px; cursor: pointer; }
  label.radio:hover { background: #f3f4f6; }
  .text-body { font-size: 1.12rem; white-space: pre-wrap; margin-bottom: 30px; }
  .qblock { margin-bottom: 24px; padding-bottom: 18px; border-bottom: 1px solid #eee; }
  .qtext { font-weight: 600; margin-bottom: 10px; }
  .opt label { display: flex; gap: 10px; padding: 7px 10px; border-radius: 5px;
               cursor: pointer; }
  .opt label:hover { background: #f0f0f0; }
  .result-box { text-align: center; padding: 40px; border-radius: 10px;
                background: #f3f4f6; margin: 30px 0; }
  .score-big { font-size: 3rem; font-weight: 800; }
</style>
"""

HOME_HTML = BASE_CSS + """
<div class="card">
  <h1>Aufmerksamkeitsanalyse – Experiment</h1>
  <p class="muted">Teilnehmer und Modus wählen. Der Gaze-Tracker startet automatisch —
     Tobii-Kreis vorher in der Tobii Experience App aktivieren!</p>
  <form method="POST" action="/start">
    <select name="participant" required>
      <option value="" disabled selected>Teilnehmer wählen…</option>
      {% for p in participants %}
      <option value="{{ p }}">{{ p }}</option>
      {% endfor %}
    </select>
    <label class="radio"><input type="radio" name="mode" value="normal" checked>
      <strong>Normaler Durchlauf</strong> – Daten für das Training sammeln</label>
    <label class="radio"><input type="radio" name="mode" value="demo">
      <strong>Vorhersage-Demo</strong> – Modell sagt den Score voraus</label>

    <p class="muted" style="margin:18px 0 8px;">Modell für die Vorhersage-Demo:</p>
    <label class="radio"><input type="radio" name="model" value="personal" checked>
      <strong>Persönliches Modell</strong> – trainiert nur auf diesem Teilnehmer</label>
    <label class="radio"><input type="radio" name="model" value="pooled">
      <strong>Gepooltes Modell</strong> – trainiert auf allen Teilnehmern</label>

    <div class="row"><button class="btn btn-blue" type="submit">Durchlauf starten →</button></div>
  </form>
  {% if error %}<p style="color:#dc2626; margin-top:20px;"><strong>Fehler:</strong> {{ error }}</p>{% endif %}
</div>
"""

READ_HTML = BASE_CSS + """
<div class="card">
  <p class="muted">Text {{ text_id }} · {{ participant }} · {{ mode }}</p>
  <h1>{{ title }}</h1>
  <div class="text-body">{{ body }}</div>
  <form method="POST" action="/finish_reading">
    <button class="btn btn-blue" type="submit">Fertig gelesen – weiter zu den Fragen →</button>
  </form>
</div>
"""

QUESTIONS_HTML = BASE_CSS + """
<div class="card">
  <p class="muted">Fragen zu Text {{ text_id }} · {{ participant }}</p>
  <h1>{{ title }}</h1>
  <form method="POST" action="/answers">
    {% for q in questions %}
    <div class="qblock">
      <p class="qtext">{{ loop.index }}. {{ q.question }}</p>
      {% for key, option in q.options.items() %}
      <div class="opt"><label>
        <input type="radio" name="q{{ q.question_id }}" value="{{ key }}" required>
        <span><strong>{{ key }}</strong>&nbsp; {{ option }}</span>
      </label></div>
      {% endfor %}
    </div>
    {% endfor %}
    <button class="btn btn-blue" type="submit">Antworten absenden →</button>
  </form>
</div>
"""

NORMAL_DONE_HTML = BASE_CSS + """
<div class="card">
  <h1>Antworten gespeichert ✓</h1>
  <p class="muted">Text {{ text_id }} abgeschlossen · {{ participant }}</p>
  <div class="row">
    <form method="POST" action="/next"><button class="btn btn-green" type="submit">Weiter – nächster Text →</button></form>
    <form method="POST" action="/stop"><button class="btn btn-red" type="submit">Stopp – speichern &amp; beenden</button></form>
  </div>
</div>
"""

DEMO_RESULT_HTML = BASE_CSS + """
<div class="card">
  <h1>Vorhersage-Demo – Ergebnis</h1>
  <p class="muted">Text {{ text_id }} · {{ participant }}{% if model_name %} · Modell: {{ model_name }}{% endif %}</p>
  <div class="row" style="gap:24px;">
    <div class="result-box" style="flex:1;">
      <p>Vorhergesagt (Modell)</p>
      <p class="score-big" style="color:#2563eb;">{{ predicted }}{% if mae %} <span style="font-size:1.2rem; color:#888;">±{{ mae }}</span>{% endif %}</p>
      <p class="muted">aus Blickverhalten{% if mae %} · mittlere Abweichung des Modells: {{ mae }} Punkte{% endif %}</p>
    </div>
    <div class="result-box" style="flex:1;">
      <p>Tatsächlich (Quiz)</p>
      <p class="score-big" style="color:#16a34a;">{{ actual }} / 10</p>
      <p class="muted">aus MC-Antworten</p>
    </div>
  </div>
  {% if error %}<p style="color:#dc2626;"><strong>Vorhersage-Fehler:</strong> {{ error }}</p>{% endif %}
  <div class="row"><a class="btn btn-gray" href="/finish_demo">Beenden – zurück zum Start</a></div>
</div>
"""

PIPELINE_RUNNING_HTML = BASE_CSS + """
<meta http-equiv="refresh" content="4">
<div class="card" style="text-align:center; padding-top:80px;">
  <h1>Daten werden verarbeitet…</h1>
  <p class="muted" style="font-size:1.1rem; margin-top:16px;">
    Schritt {{ step }} von {{ total }}: <strong>{{ current }}</strong></p>
  <p class="muted" style="margin-top:24px;">Die Seite aktualisiert sich automatisch.
    Das Modell-Update dauert ca. 10–20 Sekunden.</p>
</div>
"""

STOPPED_HTML = BASE_CSS + """
<div class="card">
  <h1>Durchlauf beendet ✓</h1>
  <p class="muted">Scores ausgewertet, Features extrahiert, Daten augmentiert, Modell aktualisiert.</p>
  <pre style="background:#f3f4f6; padding:16px; border-radius:8px; font-size:.85rem;
       max-height:300px; overflow:auto;">{{ log }}</pre>
  <div class="row"><a class="btn btn-blue" href="/">Zurück zum Start</a></div>
</div>
"""

# ---------------------------------------------------------------------------
# Flask
# ---------------------------------------------------------------------------

app = Flask(__name__)


@app.route("/")
def home():
    stop_tracker()
    return render_template_string(
        HOME_HTML, participants=PARTICIPANTS, progress=load_progress(),
        n_texts=len(TEXT_IDS), error=request.args.get("error"))


@app.route("/start", methods=["POST"])
def start():
    S.participant = request.form["participant"]
    S.mode        = request.form["mode"]
    S.model       = request.form.get("model", "personal")

    if S.mode == "demo":
        S.gaze_path    = os.path.join(RAW_DATA, "demo_gaze.csv")
        S.markers_path = os.path.join(RAW_DATA, "demo_markers.csv")
        for p in (S.gaze_path, S.markers_path):
            if os.path.exists(p):
                os.remove(p)
    else:
        S.gaze_path    = os.path.join(RAW_DATA, f"{S.participant}_gaze.csv")
        S.markers_path = os.path.join(RAW_DATA, f"{S.participant}_markers.csv")

    if S.mode == "demo":
        import random
        S.text_id = random.choice(DEMO_TEXT_IDS if DEMO_TEXTE else TEXT_IDS)
    else:
        S.text_id = next_text_id(S.participant)

    try:
        S.tracker = start_tracker(S.gaze_path)
    except RuntimeError as e:
        return redirect(f"/?error={e}")

    return redirect("/read")


@app.route("/read")
def read():
    entry = active_texte()[S.text_id]
    write_marker(S.markers_path, "text_start", S.text_id)
    S.demo_start = time.time()
    return render_template_string(
        READ_HTML, text_id=S.text_id, participant=S.participant,
        mode="Demo" if S.mode == "demo" else "Normal",
        title=entry.get("title", ""),
        body=entry.get("content", entry.get("text", "")))


@app.route("/finish_reading", methods=["POST"])
def finish_reading():
    write_marker(S.markers_path, "text_end", S.text_id)
    S.demo_end = time.time()
    return redirect("/questions")


@app.route("/questions")
def questions():
    entry = active_texte()[S.text_id]
    return render_template_string(
        QUESTIONS_HTML, text_id=S.text_id, participant=S.participant,
        title=entry.get("title", ""), questions=entry.get("questions", []))


@app.route("/answers", methods=["POST"])
def answers():
    answers = save_answers(S.participant, S.text_id, request.form,
                           demo=(S.mode == "demo"))
    actual  = compute_actual_score(S.text_id, answers)

    if S.mode == "demo":
        stop_tracker()
        predicted, mae, model_name, error = None, None, None, None
        try:
            raw_score, mae, model_name = predict_score(
                S.participant, S.gaze_path, S.demo_start, S.demo_end, S.model)
            predicted = display_score(raw_score)
        except Exception as e:
            error = str(e)
        return render_template_string(
            DEMO_RESULT_HTML, text_id=S.text_id, participant=S.participant,
            predicted=predicted or "–", actual=actual, error=error,
            model_name=model_name,
            mae=f"{math.ceil(mae * 10) / 10:.1f}" if mae else None)

    # Normal: Fortschritt hochzählen
    progress = load_progress()
    progress[S.participant] = progress.get(S.participant, 0) + 1
    save_progress(progress)
    return render_template_string(
        NORMAL_DONE_HTML, text_id=S.text_id, participant=S.participant)


@app.route("/next", methods=["POST"])
def next_text():
    S.text_id = next_text_id(S.participant)
    return redirect("/read")


PIPELINE = {"running": False, "log": "", "done_steps": 0}

PIPELINE_STEPS = [
    ("Scores auswerten",        lambda p: [PYTHON, "score.py", "--participant", p]),
    ("Features extrahieren",    lambda p: [PYTHON, "extract_features.py", "--auto"]),
    ("Daten augmentieren",      lambda p: [PYTHON, "augment_data.py", "--target", "200"]),
    ("Modell aktualisieren",    lambda p: [PYTHON, "train_model.py",
                                           "--dataset", "dataset_augmented.csv", "--no-cv"]),
]


def run_pipeline(participant):
    cwd = os.path.dirname(os.path.abspath(__file__))
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    PIPELINE["running"] = True
    PIPELINE["log"] = ""
    PIPELINE["done_steps"] = 0
    for name, cmd in PIPELINE_STEPS:
        PIPELINE["log"] += f"\n=== {name} ===\n"
        try:
            result = subprocess.run(cmd(participant), capture_output=True,
                                    text=True, encoding="utf-8", errors="replace",
                                    cwd=cwd, env=env, timeout=600)
            PIPELINE["log"] += (result.stdout or "") + (result.stderr or "")
        except Exception as e:
            PIPELINE["log"] += f"FEHLER: {e}\n"
        PIPELINE["done_steps"] += 1
    PIPELINE["running"] = False


@app.route("/stop", methods=["POST"])
@app.route("/stop_get", methods=["GET"])
def stop():
    stop_tracker()
    if not PIPELINE["running"]:
        threading.Thread(target=run_pipeline, args=(S.participant,),
                         daemon=True).start()
    return redirect("/pipeline")


@app.route("/pipeline")
def pipeline_status():
    if PIPELINE["running"]:
        step = PIPELINE["done_steps"]
        names = [n for n, _ in PIPELINE_STEPS]
        current = names[step] if step < len(names) else "…"
        return render_template_string(PIPELINE_RUNNING_HTML,
                                      step=step + 1, total=len(names),
                                      current=current)
    return render_template_string(
        STOPPED_HTML, log=PIPELINE["log"].strip() or "Keine Ausgabe.")


@app.route("/finish_demo")
def finish_demo():
    stop_tracker()
    return redirect("/")

# ---------------------------------------------------------------------------
# Browser auf rechter Bildschirmhälfte öffnen
# ---------------------------------------------------------------------------

def open_browser_right_half():
    url = f"http://localhost:{PORT}"
    try:
        sw = ctypes.windll.user32.GetSystemMetrics(0)
        sh = ctypes.windll.user32.GetSystemMetrics(1)
    except Exception:
        sw, sh = 1920, 1080
    half_x, w, h = sw // 2, sw // 2, sh

    candidates = [
        os.path.expandvars(r"%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe"),
        os.path.expandvars(r"%ProgramFiles%\Microsoft\Edge\Application\msedge.exe"),
        os.path.expandvars(r"%ProgramFiles%\Google\Chrome\Application\chrome.exe"),
        os.path.expandvars(r"%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"),
        os.path.expandvars(r"%LocalAppData%\Google\Chrome\Application\chrome.exe"),
    ]
    for exe in candidates:
        if os.path.exists(exe):
            subprocess.Popen([
                exe, f"--app={url}",
                f"--window-position={half_x},0", f"--window-size={w},{h}",
            ])
            return
    import webbrowser
    webbrowser.open(url)


if __name__ == "__main__":
    print("=" * 56)
    print(f"  Experiment-App  →  http://localhost:{PORT}")
    print("  WICHTIG: Tobii-Kreis in Tobii Experience aktivieren!")
    print("=" * 56)
    import threading
    threading.Timer(1.2, open_browser_right_half).start()
    try:
        app.run(host="127.0.0.1", port=PORT, debug=False)
    finally:
        stop_tracker()
