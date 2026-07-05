"""
experiment.py
=============
Steuerskript für das Eye-Tracking-Leseexperiment.

Ablauf pro Text:
  1. Wartebildschirm ("Weiter mit Leertaste")
  2. Rechte Bildschirmhälfte: Textanzeige + text_start Marker
  3. "Weiter"-Button → text_end Marker + Browser öffnet localhost:5000/text/<id>
  4. Nächster Wartebildschirm

Voraussetzung: server.py muss laufen (python server.py).

Nutzung:
    python experiment.py --participant sudhin
    python experiment.py --participant kushal
    python experiment.py --participant dario --start 5   # ab text_id 5 weitermachen

Erzeugt:
    raw_data/sudhin_markers.csv
    raw_data/sudhin_sync.json

Installation:
    pip install pylsl flask
"""
import argparse
import csv
import json
import os
import sys
import time
import webbrowser
import tkinter as tk
from tkinter import font as tkfont
from tkinter import messagebox

# ---------------------------------------------------------------------------
# Einstellungen (hier anpassen falls nötig)
# ---------------------------------------------------------------------------

TEXTE_JSON     = "texte.json"
RAW_DATA       = "raw_data"
SERVER_URL     = "http://localhost:5000"   # server.py muss laufen
PROGRESS_FILE  = "progress.json"           # speichert Fortschritt pro Teilnehmer

# Schrift
FONT_FAMILY    = "Segoe UI"   # Windows-Standard, lesbar
FONT_SIZE_TEXT = 16
FONT_SIZE_UI   = 14

# Farben
BG_COLOR       = "#FFFFFF"
TEXT_COLOR     = "#000000"
BTN_BG         = "#DDDDDD"
BTN_FG         = "#000000"
BTN_ACTIVE_BG  = "#BBBBBB"

# Zeilenbreite (Zeichen pro Zeile) – kleinerer Wert = schmalere Textspalte
WRAP_CHARS     = 90

# Sekunden warten nach Öffnen des Browsers bevor nächster Wartebildschirm erscheint
BROWSER_WAIT   = 1.5

# ---------------------------------------------------------------------------
# Marker-System
# ---------------------------------------------------------------------------

class MarkerLogger:
    """Schreibt Marker in CSV und optional in einen LSL-Stream."""

    def __init__(self, participant):
        os.makedirs(RAW_DATA, exist_ok=True)
        self.markers_path = os.path.join(RAW_DATA, f"{participant}_markers.csv")
        self.sync_path    = os.path.join(RAW_DATA, f"{participant}_sync.json")
        self._rows        = []
        self._sync_saved  = False

        # LSL-Outlet (optional – kein Fehler wenn pylsl fehlt)
        self.outlet = None
        try:
            import pylsl
            info = pylsl.StreamInfo(
                "TextMarkers", "Markers", 1,
                pylsl.IRREGULAR_RATE, "string", "experiment_v1"
            )
            self.outlet   = pylsl.StreamOutlet(info)
            self._pylsl   = pylsl
            print("LSL-Stream 'TextMarkers' aktiv.")
        except ImportError:
            print("pylsl nicht gefunden – kein LSL-Stream (nur CSV).")

    def _sync(self):
        if self._sync_saved:
            return
        ts     = time.time()
        lsl_ts = self._pylsl.local_clock() if self.outlet else ts
        with open(self.sync_path, "w") as f:
            json.dump({"unix_time": ts, "lsl_clock": lsl_ts}, f)
        self._sync_saved = True

    def send(self, label, text_id):
        self._sync()
        ts = time.time()
        self._rows.append([f"{ts:.6f}", label, text_id])
        if self.outlet:
            self.outlet.push_sample([label], self._pylsl.local_clock())

    def save(self):
        # Anhängen falls Datei schon existiert, sonst neu anlegen mit Header
        file_exists = os.path.exists(self.markers_path)
        with open(self.markers_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if not file_exists:
                w.writerow(["timestamp", "label", "text_id"])
            w.writerows(self._rows)
        print(f"Gespeichert: {self.markers_path} ({len(self._rows)} Marker)")


# ---------------------------------------------------------------------------
# Datei-Laden
# ---------------------------------------------------------------------------

def load_texte(path):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    # JSONL: ein Objekt pro Zeile
    if content.startswith("{"):
        raw = [json.loads(line) for line in content.splitlines() if line.strip()]
    else:
        raw = json.loads(content)

    if isinstance(raw, dict):
        for key in ("texte", "texts", "items"):
            if key in raw:
                raw = raw[key]
                break
    if not isinstance(raw, list):
        raise ValueError(f"Unerwartetes Format in {path}.")
    return raw



# ---------------------------------------------------------------------------
# Fortschritt speichern / laden
# ---------------------------------------------------------------------------

def load_progress(participant):
    """Gibt den gespeicherten Listen-Index für diesen Teilnehmer zurück (0 wenn neu)."""
    if not os.path.exists(PROGRESS_FILE):
        return 0
    with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get(participant, 0)


def save_progress(participant, next_index):
    """Speichert den nächsten Listen-Index für diesen Teilnehmer."""
    data = {}
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    data[participant] = next_index
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Tkinter-Fenster
# ---------------------------------------------------------------------------

class ExperimentApp:
    def __init__(self, root, texte, participant, marker_logger, start_index=0):
        self.root          = root
        self.texte         = texte
        self.participant   = participant
        self.logger        = marker_logger
        self.current_index = start_index
        self.state         = "waiting"   # "waiting" | "reading"

        # Rechte Bildschirmhälfte (Tobii Experience App bleibt links sichtbar)
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        half_w = sw // 2
        self.root.geometry(f"{half_w}x{sh}+{half_w}+0")
        self.root.resizable(False, False)
        self.root.configure(bg=BG_COLOR)
        self.root.bind("<Escape>", self._on_escape)
        self.root.focus_set()

        # Schriften
        self.font_text  = tkfont.Font(family=FONT_FAMILY, size=FONT_SIZE_TEXT)
        self.font_title = tkfont.Font(family=FONT_FAMILY, size=FONT_SIZE_TEXT + 4, weight="bold")
        self.font_ui    = tkfont.Font(family=FONT_FAMILY, size=FONT_SIZE_UI)
        self.font_small = tkfont.Font(family=FONT_FAMILY, size=FONT_SIZE_UI - 2)

        self._build_waiting_screen()
        self._build_reading_screen()

        self._show_waiting()

    # --- Layout-Aufbau ---

    def _build_waiting_screen(self):
        self.frame_wait = tk.Frame(self.root, bg=BG_COLOR)
        self.frame_wait.place(relx=0, rely=0, relwidth=1, relheight=1)

        tk.Label(
            self.frame_wait,
            text="Leseexperiment",
            bg=BG_COLOR, fg=TEXT_COLOR,
            font=tkfont.Font(family=FONT_FAMILY, size=28, weight="bold"),
        ).pack(expand=True, pady=(0, 10))

        self.lbl_progress = tk.Label(
            self.frame_wait, text="", bg=BG_COLOR, fg="#555555",
            font=self.font_ui,
        )
        self.lbl_progress.pack()

        self.lbl_wait_hint = tk.Label(
            self.frame_wait,
            text="Drücke die Leertaste um den nächsten Text zu starten.",
            bg=BG_COLOR, fg="#333333",
            font=self.font_ui, wraplength=700,
        )
        self.lbl_wait_hint.pack(pady=(30, 0))

        tk.Label(
            self.frame_wait,
            text="[ESC] Experiment beenden",
            bg=BG_COLOR, fg="#888888",
            font=self.font_small,
        ).pack(side=tk.BOTTOM, pady=15)

        self.root.bind("<space>", self._on_space)

    def _build_reading_screen(self):
        self.frame_read = tk.Frame(self.root, bg=BG_COLOR)
        self.frame_read.place(relx=0, rely=0, relwidth=1, relheight=1)

        # Titelzeile
        self.lbl_title = tk.Label(
            self.frame_read, text="", bg=BG_COLOR, fg=TEXT_COLOR,
            font=self.font_title, wraplength=1400, justify=tk.CENTER,
        )
        self.lbl_title.pack(pady=(40, 10), padx=80)

        # Scrollbarer Textbereich
        text_frame = tk.Frame(self.frame_read, bg=BG_COLOR)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=100, pady=10)

        scrollbar = tk.Scrollbar(text_frame, bg="#CCCCCC", troughcolor="#EEEEEE",
                                 activebackground="#AAAAAA")
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.text_widget = tk.Text(
            text_frame,
            bg=BG_COLOR, fg=TEXT_COLOR,
            font=self.font_text,
            wrap=tk.WORD,
            relief=tk.FLAT,
            bd=0,
            highlightthickness=0,
            state=tk.DISABLED,
            cursor="arrow",
            width=WRAP_CHARS,
            yscrollcommand=scrollbar.set,
            spacing1=4, spacing2=2, spacing3=4,
        )
        self.text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.text_widget.yview)

        # "Weiter"-Button
        btn_frame = tk.Frame(self.frame_read, bg=BG_COLOR)
        btn_frame.pack(pady=(10, 30))

        self.btn_weiter = tk.Button(
            btn_frame,
            text="Weiter  →",
            bg=BTN_BG, fg=BTN_FG,
            activebackground=BTN_ACTIVE_BG, activeforeground=BTN_FG,
            font=tkfont.Font(family=FONT_FAMILY, size=FONT_SIZE_UI + 2, weight="bold"),
            relief=tk.FLAT, padx=30, pady=10,
            command=self._on_weiter,
        )
        self.btn_weiter.pack()

        tk.Label(
            self.frame_read,
            text="[ESC] Experiment beenden",
            bg=BG_COLOR, fg="#888888",
            font=self.font_small,
        ).pack(side=tk.BOTTOM, pady=8)

    # --- Zustandswechsel ---

    def _show_waiting(self):
        self.state = "waiting"
        total = len(self.texte)
        idx   = self.current_index

        if idx >= total:
            self._finish()
            return

        self.lbl_progress.config(
            text=f"Text {idx + 1} von {total}"
        )

        if idx == 0:
            self.lbl_wait_hint.config(
                text="Drücke die Leertaste um den ersten Text zu starten."
            )
        else:
            self.lbl_wait_hint.config(
                text="Bitte beantworte zunächst die Fragen im Browser.\n"
                     "Danach: Leertaste drücken für den nächsten Text."
            )

        self.frame_read.lower()
        self.frame_wait.lift()
        # Fenster wieder in den Vordergrund holen damit Leertaste ankommt
        self.root.lift()
        self.root.focus_force()

    def _show_text(self, text_entry):
        self.state = "reading"

        title   = text_entry.get("title", "")
        content = text_entry.get("content", "")

        self.lbl_title.config(text=title)
        self.text_widget.config(state=tk.NORMAL)
        self.text_widget.delete("1.0", tk.END)
        self.text_widget.insert(tk.END, content)
        self.text_widget.yview_moveto(0)
        self.text_widget.config(state=tk.DISABLED)

        self.frame_wait.lower()
        self.frame_read.lift()
        # Fokus auf Root (nicht auf Button) – sonst löst die Leertaste
        # sofort den Weiter-Button aus, weil Tkinter Space = Button-Klick
        self.root.focus_set()

        # Marker senden
        text_id = text_entry.get("text_id", self.current_index + 1)
        self.logger.send("text_start", text_id)
        print(f"  → text_start (Text {text_id}: {title[:50]})")

    # --- Ereignis-Handler ---

    def _on_space(self, event=None):
        if self.state != "waiting":
            return
        if self.current_index >= len(self.texte):
            self._finish()
            return
        entry = self.texte[self.current_index]
        self._show_text(entry)

    def _on_weiter(self):
        if self.state != "reading":
            return

        entry   = self.texte[self.current_index]
        text_id = entry.get("text_id", self.current_index + 1)

        self.logger.send("text_end", text_id)
        print(f"  → text_end   (Text {text_id})")

        # Fragebogen im Browser öffnen
        url = f"{SERVER_URL}/text/{text_id}?participant={self.participant}"
        webbrowser.open(url)
        print(f"  Browser: {url}")

        self.current_index += 1
        save_progress(self.participant, self.current_index)

        # Kurz warten damit Browser Zeit hat sich zu öffnen
        self.root.after(int(BROWSER_WAIT * 1000), self._show_waiting)

    def _on_escape(self, event=None):
        if messagebox.askyesno(
            "Beenden?",
            "Experiment jetzt beenden?\n\n(Bisher aufgezeichnete Daten werden gespeichert.)",
            parent=self.root,
        ):
            self._finish()

    def _finish(self):
        self.logger.save()
        self.root.destroy()


# ---------------------------------------------------------------------------
# Hauptprogramm
# ---------------------------------------------------------------------------

def find_start_index(texte, start_text_id):
    """Gibt den Listen-Index zurück, bei dem text_id == start_text_id.
    Falls nicht gefunden, wird 0 zurückgegeben und eine Warnung ausgegeben."""
    for i, entry in enumerate(texte):
        if int(entry.get("text_id", i + 1)) == start_text_id:
            return i
    print(f"  Warnung: text_id {start_text_id} nicht gefunden – starte von vorne.")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Eye-Tracking Leseexperiment")
    parser.add_argument("--participant", required=True,
                        help="Teilnehmer-ID: sudhin, kushal oder dario")
    parser.add_argument("--texte", default=TEXTE_JSON,
                        help=f"Pfad zur texte.json (Standard: {TEXTE_JSON})")
    parser.add_argument("--start", type=int, default=None,
                        help="text_id ab der fortgesetzt wird, z.B. --start 12 "
                             "(nützlich beim Probandenwechsel oder nach Unterbrechung)")
    args = parser.parse_args()

    # Texte laden
    if not os.path.exists(args.texte):
        print(f"Fehler: {args.texte} nicht gefunden.")
        sys.exit(1)

    print(f"Lade Texte aus {args.texte} ...")
    texte = load_texte(args.texte)
    print(f"  {len(texte)} Texte geladen.")

    # Startposition bestimmen
    if args.start is not None:
        # Manuelle Angabe überschreibt gespeicherten Fortschritt
        start_index = find_start_index(texte, args.start)
        save_progress(args.participant, start_index)
        print(f"  Starte ab text_id {args.start} (Listen-Index {start_index}).")
    else:
        start_index = load_progress(args.participant)
        if start_index > 0:
            done_text_id = texte[start_index - 1].get("text_id", start_index) if start_index <= len(texte) else "?"
            next_text_id = texte[start_index].get("text_id", start_index + 1) if start_index < len(texte) else "–"
            print(f"  Fortschritt geladen: {args.participant} hat bereits {start_index} Text(e) absolviert.")
            print(f"  Weiter ab text_id {next_text_id} (Listen-Index {start_index}).")
        else:
            print(f"  Kein gespeicherter Fortschritt für {args.participant} – starte von vorne.")

    # Marker-Logger
    logger = MarkerLogger(args.participant)

    # Tkinter starten
    root = tk.Tk()
    root.title("Leseexperiment")

    app = ExperimentApp(
        root,
        texte=texte,
        participant=args.participant,
        marker_logger=logger,
        start_index=start_index,
    )

    try:
        root.mainloop()
    except KeyboardInterrupt:
        logger.save()


if __name__ == "__main__":
    main()
