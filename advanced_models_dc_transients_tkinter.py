#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modele zaawansowane maszyn elektrycznych + stany nieustalone maszyn DC
======================================================================
Interaktywna aplikacja Tkinter do rozdziałów 7 i 8 (I. Boldea, "Electric Machines:
Steady State, Transients, and Design with MATLAB"):

  Rozdział 7 – modele dq (Park), wektor przestrzenny, nasycenie magnetyczne,
               efekt naskórkowy, modele wysokiej częstotliwości (prądy łożyskowe),
               zadania 7.1–7.7
  Rozdział 8 – Przykład 8.1 (generator DC: skok V_F i zwarcie),
               Przykład 8.2 (silnik PM: skok napięcia, dwie bezwładności),
               8.4.2 osłabianie strumienia, 8.4.3 silnik szeregowy (Zad. 8.4),
               8.5 regulacja kaskadowa PI, 8.6 przekształtnik DC-DC (Zad. 8.5),
               8.7 wyznaczanie parametrów z prób, Zadania 8.1–8.3

Każda zakładka ma: suwaki z parametrami, wyniki liczbowe, wykresy (matplotlib)
oraz wyjaśnienie "jak dla ucznia liceum" z komentarzem doświadczonego inżyniera.

Uruchomienie:  python3 advanced_models_dc_transients_tkinter.py
Eksport wyjaśnień do Markdown:  python3 advanced_models_dc_transients_tkinter.py --export-md plik.md
Wymagania: numpy, scipy, matplotlib, tkinter (pakiet python3-tk)
"""

import math
import queue
import sys
import threading
import traceback

import numpy as np

MU0 = 4e-7 * math.pi          # przenikalność magnetyczna próżni [H/m]
TWO_PI_3 = 2.0 * math.pi / 3.0

# ── Paleta Catppuccin Mocha ────────────────────────────────────────────────
BG = "#1e1e2e"; MANTLE = "#181825"; CRUST = "#11111b"; SURF0 = "#313244"
SURF1 = "#45475a"; TEXT = "#cdd6f4"; SUBTEXT = "#a6adc8"
BLUE = "#89b4fa"; GREEN = "#a6e3a1"; RED = "#f38ba8"; YELLOW = "#f9e2af"
MAUVE = "#cba6f7"; PEACH = "#fab387"; TEAL = "#94e2d5"; SKY = "#89dceb"
PINK = "#f5c2e7"; LAVENDER = "#b4befe"
CYCLE = [SKY, GREEN, PEACH, MAUVE, YELLOW, RED, TEAL, PINK]


def setup_matplotlib():
    """Ciemny motyw wykresów spójny z interfejsem."""
    import matplotlib
    matplotlib.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": MANTLE, "savefig.facecolor": BG,
        "axes.edgecolor": SURF1, "axes.labelcolor": TEXT, "axes.titlecolor": TEXT,
        "axes.titlesize": 10, "axes.labelsize": 9, "axes.grid": True,
        "grid.color": SURF0, "grid.linewidth": 0.7, "text.color": TEXT,
        "xtick.color": SUBTEXT, "ytick.color": SUBTEXT, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "legend.fontsize": 8, "legend.facecolor": SURF0,
        "legend.edgecolor": SURF1, "legend.framealpha": 0.85, "lines.linewidth": 1.6,
        "axes.prop_cycle": matplotlib.cycler(color=CYCLE),
    })


def fmt(x, n=4):
    """Czytelny zapis liczby (bez notacji naukowej dla typowych wartości)."""
    if x is None:
        return "—"
    if isinstance(x, complex):
        return f"{fmt(x.real, n)} {'+' if x.imag >= 0 else '−'} j{fmt(abs(x.imag), n)}"
    if not np.isfinite(x):
        return "∞"
    ax = abs(x)
    if ax != 0 and (ax >= 1e5 or ax < 1e-3):
        return f"{x:.{n - 1}e}"
    return f"{x:.{n}g}"


def rk4(f, x0, t_end, dt, record_every=1):
    """Klasyczny Runge-Kutta 4. rzędu na listach floatów (szybki dla małych układów).

    f(t, x) -> lista pochodnych. Zwraca (t, X) jako tablice numpy.
    """
    n = int(round(t_end / dt))
    x = list(x0)
    ts = [0.0]
    xs = [list(x)]
    m = len(x)
    t = 0.0
    for k in range(1, n + 1):
        k1 = f(t, x)
        x2 = [x[i] + 0.5 * dt * k1[i] for i in range(m)]
        k2 = f(t + 0.5 * dt, x2)
        x3 = [x[i] + 0.5 * dt * k2[i] for i in range(m)]
        k3 = f(t + 0.5 * dt, x3)
        x4 = [x[i] + dt * k3[i] for i in range(m)]
        k4 = f(t + dt, x4)
        x = [x[i] + dt / 6.0 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) for i in range(m)]
        t = k * dt
        if k % record_every == 0:
            ts.append(t)
            xs.append(list(x))
    return np.array(ts), np.array(xs)


def park(a, b, c, th):
    """Transformacja Parka (niezmiennik amplitudy, współczynnik 2/3) – Równ. 7.36."""
    d = 2.0 / 3.0 * (a * np.cos(th) + b * np.cos(th - TWO_PI_3) + c * np.cos(th + TWO_PI_3))
    q = -2.0 / 3.0 * (a * np.sin(th) + b * np.sin(th - TWO_PI_3) + c * np.sin(th + TWO_PI_3))
    z = (a + b + c) / 3.0
    return d, q, z


# ═══════════════════════════════════════════════════════════════════════════
#  GUI – szkielet zakładek
# ═══════════════════════════════════════════════════════════════════════════
try:
    import tkinter as tk
    from tkinter import ttk
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    from matplotlib.figure import Figure
    HAVE_TK = True
except Exception:          # pozwala eksportować teksty bez środowiska graficznego
    HAVE_TK = False


class ScrollFrame(ttk.Frame if HAVE_TK else object):
    """Ramka z pionowym paskiem przewijania (Canvas + Scrollbar)."""

    def __init__(self, parent, width=360):
        super().__init__(parent)
        self.canvas = tk.Canvas(self, width=width, bg=BG, highlightthickness=0)
        sb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas)
        self.inner.bind("<Configure>",
                        lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self._win = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.bind("<Configure>",
                         lambda e: self.canvas.itemconfigure(self._win, width=e.width))
        self.canvas.configure(yscrollcommand=sb.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self.inner.bind("<Enter>", self._bind_wheel)
        self.inner.bind("<Leave>", self._unbind_wheel)

    def _bind_wheel(self, _e):
        self.canvas.bind_all("<MouseWheel>", self._on_wheel)
        self.canvas.bind_all("<Button-4>", lambda e: self.canvas.yview_scroll(-2, "units"))
        self.canvas.bind_all("<Button-5>", lambda e: self.canvas.yview_scroll(2, "units"))

    def _unbind_wheel(self, _e):
        for s in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.canvas.unbind_all(s)

    def _on_wheel(self, e):
        self.canvas.yview_scroll(int(-e.delta / 120) or (-1 if e.delta > 0 else 1), "units")


def render_markup(txt, content):
    """Wstawia prosty „markdown” do widżetu Text (nagłówki, wzory, uwagi)."""
    txt.configure(state="normal")
    txt.delete("1.0", "end")
    for raw in content.strip("\n").split("\n"):
        line = raw.rstrip()
        if line.startswith("### "):
            txt.insert("end", line[4:] + "\n", "h3")
        elif line.startswith("## "):
            txt.insert("end", line[3:] + "\n", "h2")
        elif line.startswith("» "):
            txt.insert("end", "    " + line[2:] + "\n", "formula")
        elif line.startswith("⚙"):
            txt.insert("end", line + "\n", "eng")
        elif line.startswith("💡"):
            txt.insert("end", line + "\n", "tip")
        elif line.startswith("• ") or line.startswith("  – "):
            txt.insert("end", line + "\n", "bullet")
        else:
            txt.insert("end", line + "\n", "body")
    txt.configure(state="disabled")


class BaseTab:
    """Zakładka: suwaki (lewa), wykresy + wyjaśnienie (prawa).

    Podklasa definiuje: TITLE, SLIDERS, EXPLAIN, calc(p) -> dane (czysto numeryczne,
    wykonywane w wątku roboczym) oraz draw(p, dane) -> rysowanie (wątek GUI).
    SLIDERS: (klucz, etykieta, min, max, start, jednostka[, krok])
    """
    TITLE = "?"
    SLIDERS = []
    EXPLAIN = ""
    ANIM = False

    def __init__(self, notebook, app):
        self.app = app
        self.vars = {}
        self.extras = {}
        self._init = {}
        self._job = None
        self._gen = 0
        self.drawn = False
        self.anim_on = False
        self.frame = ttk.Frame(notebook)
        notebook.add(self.frame, text=self.TITLE)
        self._build()

    # ── budowa interfejsu ──────────────────────────────────────────────────
    def _build(self):
        pw = ttk.PanedWindow(self.frame, orient="horizontal")
        pw.pack(fill="both", expand=True)
        left = ttk.Frame(pw)
        pw.add(left, weight=0)
        right = ttk.PanedWindow(pw, orient="vertical")
        pw.add(right, weight=1)

        resf = ttk.LabelFrame(left, text=" Wyniki obliczeń ")
        resf.pack(side="bottom", fill="x", padx=4, pady=4)
        self.res = tk.Text(resf, height=15, width=46, bg=CRUST, fg=GREEN, relief="flat",
                           font=("DejaVu Sans Mono", 9), wrap="none", insertbackground=TEXT)
        self.res.pack(fill="both", expand=True, padx=2, pady=2)

        sc = ScrollFrame(left, width=350)
        sc.pack(side="top", fill="both", expand=True)
        box = ttk.LabelFrame(sc.inner, text=" Parametry (suwak lub wpisz + Enter) ")
        box.pack(fill="x", padx=4, pady=4)
        for spec in self.SLIDERS:
            self._slider(box, *spec)
        extra = ttk.Frame(sc.inner)
        extra.pack(fill="x", padx=4, pady=2)
        self.build_extra(extra)
        btns = ttk.Frame(sc.inner)
        btns.pack(fill="x", padx=4, pady=6)
        ttk.Button(btns, text="⟳ Przelicz", command=self.request).pack(side="left", padx=2)
        ttk.Button(btns, text="↺ Reset", command=self.reset).pack(side="left", padx=2)
        if self.ANIM:
            ttk.Button(btns, text="▶ Start", command=self.anim_start).pack(side="left", padx=2)
            ttk.Button(btns, text="■ Stop", command=self.anim_stop).pack(side="left", padx=2)

        pf = ttk.Frame(right)
        right.add(pf, weight=3)
        self.fig = Figure(figsize=(8.5, 5.6), dpi=90, layout="constrained")
        self.canvas = FigureCanvasTkAgg(self.fig, master=pf)
        tb = NavigationToolbar2Tk(self.canvas, pf, pack_toolbar=False)
        tb.update()
        for w in [tb] + list(tb.winfo_children()):
            try:
                w.configure(background=SURF0)
            except tk.TclError:
                pass
        tb.pack(side="bottom", fill="x")
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        ef = ttk.LabelFrame(right, text=" Wyjaśnienie – krok po kroku ")
        right.add(ef, weight=2)
        self.exp = tk.Text(ef, wrap="word", bg=MANTLE, fg=TEXT, relief="flat",
                           font=("DejaVu Sans", 10), padx=10, pady=6, height=10)
        esb = ttk.Scrollbar(ef, orient="vertical", command=self.exp.yview)
        self.exp.configure(yscrollcommand=esb.set)
        esb.pack(side="right", fill="y")
        self.exp.pack(side="left", fill="both", expand=True)
        self.exp.tag_configure("h2", font=("DejaVu Sans", 12, "bold"), foreground=SKY,
                               spacing1=8, spacing3=4)
        self.exp.tag_configure("h3", font=("DejaVu Sans", 10, "bold"), foreground=MAUVE,
                               spacing1=6, spacing3=2)
        self.exp.tag_configure("formula", font=("DejaVu Sans Mono", 10), foreground=YELLOW,
                               background=CRUST)
        self.exp.tag_configure("eng", foreground=PEACH, lmargin1=6, lmargin2=20)
        self.exp.tag_configure("tip", foreground=GREEN, lmargin1=6, lmargin2=20)
        self.exp.tag_configure("bullet", lmargin1=12, lmargin2=26)
        self.exp.tag_configure("body", spacing3=2)
        render_markup(self.exp, self.EXPLAIN)

    def _slider(self, parent, key, label, lo, hi, init, unit, step=None):
        row = ttk.Frame(parent)
        row.pack(fill="x", padx=4, pady=1)
        row.columnconfigure(0, weight=1)
        ttk.Label(row, text=f"{label} [{unit}]" if unit else label).grid(
            row=0, column=0, columnspan=2, sticky="w")
        var = tk.DoubleVar(value=init)
        ent_var = tk.StringVar(value=fmt(init))

        def on_move(_v=None):
            v = var.get()
            if step:
                v = round(v / step) * step
                var.set(v)
            ent_var.set(fmt(v))
            self.request()

        def on_entry(_e=None):
            try:
                v = float(ent_var.get().replace(",", "."))
            except ValueError:
                ent_var.set(fmt(var.get()))
                return
            v = min(max(v, lo), hi)          # walidacja zakresu
            var.set(v)
            ent_var.set(fmt(v))
            self.request()

        sl = ttk.Scale(row, variable=var, from_=lo, to=hi, orient="horizontal",
                       command=on_move)
        sl.grid(row=1, column=0, sticky="ew")
        ent = ttk.Entry(row, textvariable=ent_var, width=9)
        ent.grid(row=1, column=1, padx=(4, 0))
        ent.bind("<Return>", on_entry)
        ent.bind("<FocusOut>", on_entry)
        self.vars[key] = (var, ent_var, lo, hi)
        self._init[key] = init

    def add_combo(self, parent, key, label, values, init=0):
        f = ttk.Frame(parent)
        f.pack(fill="x", pady=2)
        ttk.Label(f, text=label).pack(anchor="w")
        v = tk.StringVar(value=values[init])
        cb = ttk.Combobox(f, textvariable=v, values=values, state="readonly")
        cb.pack(fill="x")
        cb.bind("<<ComboboxSelected>>", lambda e: self.request())
        self.extras[key] = v
        self._init[key] = values[init]

    def add_check(self, parent, key, label, init=True):
        v = tk.BooleanVar(value=init)
        ttk.Checkbutton(parent, text=label, variable=v, command=self.request).pack(anchor="w")
        self.extras[key] = v
        self._init[key] = init

    def build_extra(self, parent):
        pass

    # ── parametry / obliczenia ─────────────────────────────────────────────
    def params(self):
        p = {}
        for k, (var, _e, lo, hi) in self.vars.items():
            p[k] = float(min(max(var.get(), lo), hi))
        for k, v in self.extras.items():
            p[k] = v.get()
        return p

    def reset(self):
        for k, (var, ent, _lo, _hi) in self.vars.items():
            var.set(self._init[k])
            ent.set(fmt(self._init[k]))
        for k, v in self.extras.items():
            v.set(self._init[k])
        self.request()

    def request(self):
        if self._job is not None:
            self.frame.after_cancel(self._job)
        self._job = self.frame.after(160, self._launch)

    def _launch(self):
        self._job = None
        self._gen += 1
        gen = self._gen
        p = self.params()
        self.app.set_status(f"Obliczam: {self.TITLE} …")

        def work():
            try:
                data, err = self.calc(p), None
            except Exception:
                data, err = None, traceback.format_exc()
            self.app.results.put((self, gen, p, data, err))

        threading.Thread(target=work, daemon=True).start()

    def finish(self, gen, p, data, err):
        if gen != self._gen:          # wynik przestarzały (suwak poruszony ponownie)
            return
        if err:
            self.write_results(["BŁĄD OBLICZEŃ:", *err.splitlines()[-6:]])
            self.app.set_status("Błąd – sprawdź parametry")
            return
        try:
            self.fig.clear()
            lines = self.draw(p, data)
            self.canvas.draw_idle()
            if lines:
                self.write_results(lines)
            self.drawn = True
            self.app.set_status(f"Gotowe: {self.TITLE}")
        except Exception:
            self.write_results(["BŁĄD RYSOWANIA:", *traceback.format_exc().splitlines()[-6:]])

    def run_sync(self):
        """Obliczenia synchroniczne (używane w testach)."""
        p = self.params()
        data = self.calc(p)
        self.fig.clear()
        lines = self.draw(p, data)
        self.canvas.draw()
        if lines:
            self.write_results(lines)
        return p, data

    def write_results(self, lines):
        self.res.configure(state="normal")
        self.res.delete("1.0", "end")
        self.res.insert("end", "\n".join(lines))
        self.res.configure(state="disabled")

    def ensure_drawn(self):
        if not self.drawn:
            self.request()

    # ── animacja ───────────────────────────────────────────────────────────
    def anim_start(self):
        if not self.anim_on:
            self.anim_on = True
            self._anim_loop()

    def anim_stop(self):
        self.anim_on = False

    def _anim_loop(self):
        if not self.anim_on:
            return
        try:
            self.anim_step()
            self.canvas.draw_idle()
        except Exception:
            self.anim_on = False
            return
        self.frame.after(45, self._anim_loop)

    def anim_step(self):
        pass

    def calc(self, p):
        return None

    def draw(self, p, d):
        return []


def legend(ax, **kw):
    h, _l = ax.get_legend_handles_labels()
    if h:
        ax.legend(**kw)


# ═══════════════════════════════════════════════════════════════════════════
#  START
# ═══════════════════════════════════════════════════════════════════════════
class TabStart(BaseTab):
    TITLE = "Start – mapa materiału"
    SLIDERS = []
    EXPLAIN = """
## Jak korzystać z aplikacji
Aplikacja ilustruje rozdziały 7 („Zaawansowane modele maszyn elektrycznych”) i 8 („Stany nieustalone maszyn prądu stałego z komutatorem”). Każda zakładka ma trzy części:
• po lewej – suwaki z parametrami (możesz też wpisać liczbę i nacisnąć Enter) oraz wyniki liczbowe,
• u góry po prawej – wykresy (pasek narzędzi pozwala powiększać i zapisywać rysunek),
• na dole – wyjaśnienie krok po kroku, najpierw „po ludzku”, potem wzory i komentarz inżyniera (⚙).

## O co w ogóle chodzi? (wersja dla licealisty)
Silnik elektryczny w podręczniku do fizyki działa „w stanie ustalonym”: stałe napięcie, stała prędkość, stały prąd. W prawdziwym świecie silnik jest włączany, hamowany, obciążany, zasilany z falownika, który przełącza napięcie tysiące razy na sekundę. Wtedy prądy i prędkość się ZMIENIAJĄ – to są stany nieustalone (przejściowe).
Żeby je policzyć, inżynier potrzebuje modelu – zestawu równań różniczkowych („jak szybko zmienia się prąd, jeśli…”). Rozdział 7 pokazuje, jak zbudować taki model dla każdej maszyny, a rozdział 8 liczy go w praktyce dla najprostszej maszyny – silnika prądu stałego.

### Trzy najważniejsze idee
• Stała czasowa τ – „jak długo coś się rozpędza”. Obwód z cewką L i opornikiem R: τ = L/R. Po czasie τ zmiana osiąga ok. 63%, po 3τ – 95%, po 5τ – 99%.
• Transformacja dq (Parka) – patrzymy na maszynę z obracającego się „karuzelowego” układu współrzędnych. Wtedy prądy przemienne stają się stałymi liczbami, a równania mają stałe współczynniki. To jak zdjęcie karuzeli zrobione z samej karuzeli – konie stoją w miejscu.
• Wartości własne (bieguny) – liczby, które mówią, czy odpowiedź układu będzie gładka (liczby rzeczywiste ujemne), czy z oscylacjami (liczby zespolone).

## Wykresy na tej stronie
Lewy wykres: porównanie stałych czasowych z przykładów (skala logarytmiczna!). Widać, że w jednej maszynie żyją zjawiska różniące się szybkością nawet 2000 razy – obwód twornika reaguje w ułamku milisekundy, a obwód wzbudzenia w pół sekundy.
Prawy wykres: zakresy częstotliwości, w których stosuje się różne modele (Rozdz. 7.11): do ok. 400 Hz model R-L-E, od 20 kHz model pojemnościowy (wysokiej częstotliwości), pomiędzy – model uniwersalny.
⚙ Inżynier: dobór modelu to zawsze kompromis „dokładność kontra czas obliczeń”. Model dq wystarcza do sterowania i większości stanów przejściowych; MES (metoda elementów skończonych) do projektowania szczegółów; model HF do kompatybilności elektromagnetycznej (EMC) i prądów łożyskowych.
"""

    def calc(self, p):
        return None

    def draw(self, p, d):
        ax = self.fig.add_subplot(1, 2, 1)
        names = ["Twornik gen.\nz obciąż. (8.1)", "Te silnika PM\n(8.2)", "Zwarcie\nLa/Ra (8.1)",
                 "Tem, J=2e-4\n(8.2)", "Tem, J=1e-3\n(8.2)", "Wzbudzenie\nLF/RF (8.1)"]
        vals = [0.5e-3 / 2.1, 2e-3, 5e-3, 4.52e-3, 22.6e-3, 0.5]
        cols = [SKY, GREEN, RED, PEACH, YELLOW, MAUVE]
        ax.barh(names, np.array(vals) * 1e3, color=cols)
        ax.set_xscale("log")
        ax.set_xlabel("stała czasowa [ms] (skala log)")
        ax.set_title("Skale czasu w maszynie DC (Rozdz. 8)")
        for i, v in enumerate(vals):
            ax.text(v * 1e3 * 1.1, i, f"{v * 1e3:.3g} ms", va="center", fontsize=8)
        ax.set_xlim(0.1, 3000)
        ax2 = self.fig.add_subplot(1, 2, 2)
        bands = [(1, 400, GREEN, "model R-L-E\n(dq, obwodowy)"),
                 (400, 2e4, YELLOW, "model uniwersalny\n(kable + maszyna)"),
                 (2e4, 1e7, RED, "model HF\n(pojemności pasożytnicze)")]
        for lo, hi, c, lab in bands:
            ax2.axvspan(lo, hi, color=c, alpha=0.25)
            ax2.text(math.sqrt(lo * hi), 0.5, lab, ha="center", va="center", fontsize=9)
        for f, lab in [(50, "sieć 50 Hz"), (5e3, "PWM 5 kHz"), (1 / (1e-6 * math.pi), "zbocze IGBT 1 µs")]:
            ax2.axvline(f, color=TEXT, lw=1, ls="--")
            ax2.text(f, 0.93, lab, rotation=90, va="top", ha="right", fontsize=8)
        ax2.set_xscale("log")
        ax2.set_xlim(1, 1e7)
        ax2.set_ylim(0, 1)
        ax2.set_yticks([])
        ax2.set_xlabel("częstotliwość [Hz]")
        ax2.set_title("Który model kiedy? (Rozdz. 7.1 i 7.11)")
        return ["Witaj! Wybierz zakładkę rozdziału 7 lub 8.",
                "",
                "Rozdz. 7: modele dq, Park, nasycenie,",
                "          naskórkowość, HF, zadania 7.x",
                "Rozdz. 8: przykłady 8.1, 8.2, sekcje",
                "          8.4–8.7, zadania 8.1–8.5",
                "",
                "Wszystkie wykresy reagują na suwaki."]


# ═══════════════════════════════════════════════════════════════════════════
#  7.2–7.4  Model fizyczny dq, napięcia transformacji i rotacji
# ═══════════════════════════════════════════════════════════════════════════
FRAMES = ["ωb = 0 (osie stojana – maszyna DC)", "ωb = ωr (osie wirnika – SM)",
          "ωb = ω1 (osie synchroniczne – IM)"]


class Tab72(BaseTab):
    TITLE = "7.2–7.4 Model dq i SEM"
    ANIM = True
    SLIDERS = [
        ("psi0", "Amplituda strumienia skojarzonego Ψ0", 0.1, 2.0, 1.0, "Wb"),
        ("m", "Głębokość pulsacji strumienia m", 0.0, 1.0, 0.5, "–"),
        ("fp", "Częstotliwość pulsacji fp", 0.0, 50.0, 5.0, "Hz"),
        ("n", "Prędkość względna cewki n", 0.0, 3000.0, 600.0, "obr/min"),
        ("p1", "Liczba par biegunów p1", 1, 4, 1, "–", 1),
    ]
    EXPLAIN = """
## Po co model dq? (7.1–7.2)
Wyobraź sobie karuzelę. Stojąc obok, widzisz konia, który ciągle zmienia położenie – jego współrzędne x i y to sinusoidy. Gdy wskoczysz na karuzelę, koń stoi nieruchomo obok ciebie. Model dq to właśnie „wskoczenie na karuzelę”: patrzymy na maszynę z układu osi d (oś bieguna, „direct”) i q (oś poprzeczna, „quadrature”, 90° dalej), który obraca się z prędkością ωb.
Autor proponuje model FIZYCZNY: maszynę z uzwojeniami komutatorowymi, których szczotki leżą w osiach d i q i obracają się z prędkością ωb. Dzięki temu pole każdego uzwojenia zawsze stoi w osi szczotek, a indukcyjności NIE zależą od kąta wirnika. Równania mają stałe współczynniki – ogromne uproszczenie.
### Jaką prędkość osi wybrać? (Tabela 7.1)
• maszyna synchroniczna (SM): ωb = ωr – osie przyklejone do wirnika, bo wirnik ma wystające bieguny (asymetrię magnetyczną),
• maszyna indukcyjna (IM): szczelina równomierna, więc wolno wybrać dowolne ωb; najczęściej 0, ωr lub ω1,
• maszyna DC: ωb = 0 – asymetria (bieguny, magnes) jest w stojanie, szczotki stoją.
Kliknij ▶ Start – na lewym wykresie osie d-q obracają się zgodnie z wybraną opcją, wirnik (pomarańczowa szprycha) i pole wirujące (zielona strzałka) też się kręcą.

## Dwa rodzaje napięcia indukowanego (7.3)
Z prawa Faradaya: SEM = −dΨ/dt. Strumień skojarzony z cewką Ψ(θ, t) zależy od czasu I od położenia, więc pochodna ma dwie części:
» e = −∂Ψ/∂t − (∂Ψ/∂θ)·(dθ/dt) = e_p + e_m
• e_p – napięcie pulsacyjne (transformatorowe): strumień „pompuje” się w czasie, cewka stoi. Tak działa transformator.
• e_m – napięcie rotacji (ruchu): strumień jest stały, ale cewka się przez niego przesuwa z prędkością ω. Tak działa prądnica rowerowa.
Dla Ψ = Ψm(t)·cos θ dostajemy:
» e_p = −(dΨm/dt)·cos θ         e_m = +Ψm·ω·sin θ
Ważny wniosek (Równ. 7.3): SEM rotacji w osi q pochodzi od strumienia osi d i odwrotnie – osie „rozmawiają ze sobą” tylko przez ruch. I tylko SEM rotacji wytwarza moment!
Na wykresach: górny prawy – strumień Ψ(t); dolny lewy – e_p, e_m, suma oraz (kropki) −dΨ/dt policzone numerycznie. Kropki leżą dokładnie na sumie – to dowód, że rozkład jest poprawny. Dolny prawy – wartości skuteczne obu składników.
💡 Spróbuj: ustaw n = 0 – zostaje tylko e_p (transformator). Ustaw m = 0 – zostaje tylko e_m (prądnica). Zwiększ fp – rośnie e_p.

## Silnik DC z magnesem jako model dq (7.4)
Wystarczy zostawić jedno uzwojenie wirnika w osi q (twornik) i magnes w osi d stojana. Równanie:
» V_qr = R_a·I_qr + L_a·dI_qr/dt + ω_r·Ψ_dr,    T_e = p1·Ψ_dr·I_qr
Ψ_dr to strumień od magnesu, „widziany” przez fikcyjne uzwojenie w osi d. Stąd klasyczne wzory E = k·Φ·n i T = k·Φ·I. „Maszyna DC to uproszczony model dq!”
⚙ Inżynier: w maszynie DC komutator robi fizycznie to, co w napędzie z falownikiem robi procesor (transformacja Parka w czasie rzeczywistym). Dlatego sterowanie wektorowe silników AC nazywa się „robieniem z silnika AC silnika DC”.
"""

    def build_extra(self, parent):
        self.add_combo(parent, "frame", "Prędkość osi dq w animacji:", FRAMES, 1)
        self.th_r = 0.0

    def calc(self, p):
        w = p["p1"] * 2 * math.pi * p["n"] / 60.0
        t = np.linspace(0, 0.2, 4001)
        psim = p["psi0"] * (1 + p["m"] * np.sin(2 * math.pi * p["fp"] * t))
        dpsim = p["psi0"] * p["m"] * 2 * math.pi * p["fp"] * np.cos(2 * math.pi * p["fp"] * t)
        th = w * t
        psi = psim * np.cos(th)
        ep = -dpsim * np.cos(th)
        em = psim * w * np.sin(th)
        num = -np.gradient(psi, t)
        return dict(t=t, psi=psi, ep=ep, em=em, num=num, w=w)

    def draw(self, p, d):
        gs = self.fig.add_gridspec(2, 2)
        a0 = self.fig.add_subplot(gs[0, 0])
        self._schematic(a0)
        a1 = self.fig.add_subplot(gs[0, 1])
        a1.plot(d["t"] * 1e3, d["psi"], color=SKY, label="Ψ(t) = Ψm(t)·cos θ")
        a1.set_xlabel("t [ms]"); a1.set_ylabel("Ψ [Wb]"); a1.set_title("Strumień skojarzony z cewką")
        legend(a1, loc="upper right")
        a2 = self.fig.add_subplot(gs[1, 0])
        t = d["t"] * 1e3
        a2.plot(t, d["ep"], color=PEACH, label="e_p pulsacyjne")
        a2.plot(t, d["em"], color=GREEN, label="e_m rotacji")
        a2.plot(t, d["ep"] + d["em"], color=TEXT, lw=2, label="e = e_p + e_m")
        a2.plot(t[::80], d["num"][::80], "o", ms=3, color=RED, label="−dΨ/dt (numer.)")
        a2.set_xlabel("t [ms]"); a2.set_ylabel("e [V]"); a2.set_title("Rozkład SEM (Równ. 7.1–7.3)")
        legend(a2, loc="upper right", ncol=2)
        a3 = self.fig.add_subplot(gs[1, 1])
        rms = [np.sqrt(np.mean(d["ep"] ** 2)), np.sqrt(np.mean(d["em"] ** 2)),
               np.sqrt(np.mean((d["ep"] + d["em"]) ** 2))]
        a3.bar(["e_p (RMS)", "e_m (RMS)", "e (RMS)"], rms, color=[PEACH, GREEN, TEXT])
        a3.set_ylabel("[V]"); a3.set_title("Wartości skuteczne składników")
        return ["ω względne   = " + fmt(d["w"]) + " rad/s",
                "e_p RMS      = " + fmt(rms[0]) + " V",
                "e_m RMS      = " + fmt(rms[1]) + " V",
                "e   RMS      = " + fmt(rms[2]) + " V",
                "max |e−(−dΨ/dt)| = " + fmt(np.max(np.abs(d["ep"] + d["em"] - d["num"])[5:-5])) + " V",
                "",
                "Tabela 7.1 (osie dq):",
                "  SM : ωb=ωr, wirnik DC, stojan AC(ω1)",
                "  IM : ωb=ω1, wirnik AC(ω1−ωr)",
                "  DC : ωb=0,  wirnik AC(ωr), stojan DC",
                "Moment tworzą tylko SEM rotacji!"]

    def _angles(self):
        fr = FRAMES.index(self.extras["frame"].get())
        th1 = self.th_r * 1.25                     # pole wiruje nieco szybciej (poślizg)
        thb = [0.0, self.th_r, th1][fr]
        return thb, th1

    def _schematic(self, ax):
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_xlim(-1.35, 1.35); ax.set_ylim(-1.35, 1.35)
        tt = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.cos(tt), np.sin(tt), color=SURF1, lw=6)
        ax.plot(0.62 * np.cos(tt), 0.62 * np.sin(tt), color=SURF1, lw=3)
        ax.set_title("Model fizyczny dq (Rys. 7.1) – ▶ animacja", fontsize=9)
        thb, th1 = self._angles()
        self._ld, = ax.plot([], [], color=SKY, lw=2)
        self._lq, = ax.plot([], [], color=MAUVE, lw=2)
        self._rot, = ax.plot([], [], color=PEACH, lw=4)
        self._fld = ax.annotate("", xy=(0, 0), xytext=(0, 0),
                                arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=2))
        self._td = ax.text(0, 0, "d", color=SKY, fontsize=11, weight="bold")
        self._tq = ax.text(0, 0, "q", color=MAUVE, fontsize=11, weight="bold")
        ax.text(-1.3, -1.3, "— wirnik ωr", color=PEACH, fontsize=8)
        ax.text(0.15, -1.3, "→ pole ω1", color=GREEN, fontsize=8)
        self._update_schematic()

    def _update_schematic(self):
        thb, th1 = self._angles()
        c, s = math.cos(thb), math.sin(thb)
        self._ld.set_data([-1.15 * c, 1.15 * c], [-1.15 * s, 1.15 * s])
        self._lq.set_data([1.15 * s, -1.15 * s], [-1.15 * c, 1.15 * c])
        self._td.set_position((1.22 * c, 1.22 * s))
        self._tq.set_position((-1.22 * s, 1.22 * c))
        self._rot.set_data([0, 0.6 * math.cos(self.th_r)], [0, 0.6 * math.sin(self.th_r)])
        self._fld.xy = (0.9 * math.cos(th1), 0.9 * math.sin(th1))

    def anim_step(self):
        self.th_r += 0.06
        self._update_schematic()


# ═══════════════════════════════════════════════════════════════════════════
#  7.5 Maszyna synchroniczna w osiach dq – PMSM (Zadanie 7.3)
# ═══════════════════════════════════════════════════════════════════════════
class TabPMSM(BaseTab):
    TITLE = "7.5 SM/PMSM (Zad. 7.3)"
    SLIDERS = [
        ("psi", "Strumień magnesu Ψ_PM", 0.02, 0.3, 0.10, "Wb"),
        ("Ld", "Indukcyjność osi d, L_d", 1.0, 30.0, 5.0, "mH"),
        ("Lq", "Indukcyjność osi q, L_q", 1.0, 40.0, 12.0, "mH"),
        ("Rs", "Rezystancja stojana R_s", 0.01, 1.0, 0.2, "Ω"),
        ("p1", "Pary biegunów p1", 1, 8, 4, "–", 1),
        ("I", "Amplituda prądu I", 1.0, 50.0, 20.0, "A"),
        ("n", "Prędkość n", 0.0, 3000.0, 1000.0, "obr/min"),
        ("gam", "Kąt prądu γ (od osi q)", -90.0, 90.0, 30.0, "°"),
    ]
    EXPLAIN = """
## Maszyna synchroniczna w osiach wirnika (7.5)
W maszynie synchronicznej wirnik kręci się dokładnie z prędkością pola. Przyklejamy więc osie dq do wirnika (ωb = ωr): oś d – w stronę bieguna (magnesu), oś q – 90° elektrycznych dalej. W takim układzie w stanie ustalonym wszystkie wielkości są STAŁE (prąd „stały” zamiast sinusoidy) – idealnie do sterowania.
Zadanie 7.3 każe usunąć z wirnika klatkę i uzwojenie wzbudzenia i wstawić magnesy w osi d. Zamiast L_dm·I_F piszemy Ψ_PM. Równania upraszczają się do:
» V_d = R_s·I_d + L_d·dI_d/dt − ω_r·L_q·I_q
» V_q = R_s·I_q + L_q·dI_q/dt + ω_r·(L_d·I_d + Ψ_PM)
» Ψ_d = L_d·I_d + Ψ_PM,     Ψ_q = L_q·I_q
» T_e = 3/2·p1·(Ψ_d·I_q − Ψ_q·I_d) = 3/2·p1·[Ψ_PM·I_q + (L_d − L_q)·I_d·I_q]
Czynnik 3/2 wynika z transformacji Parka z niezmiennikiem amplitudy (patrz zakładka 7.9–7.10).
### Dwa „silniki w jednym”
• moment magnesu 3/2·p1·Ψ_PM·I_q – jak siła na przewód w polu magnesu (siła Lorentza),
• moment reluktancyjny 3/2·p1·(L_d−L_q)·I_d·I_q – żelazo „chce” ustawić się tak, by strumień miał najłatwiejszą drogę (jak spinacz przyciągany przez magnes).
Gdy L_q > L_d (magnesy zagłębione – IPMSM, zadania 7.2 i 7.4), ujemne I_d (γ > 0) dodaje moment reluktancyjny. Istnieje kąt γ dający największy moment przy danym prądzie – MTPA (Maximum Torque Per Ampere) – zaznaczony na wykresie.
### Wykresy
• lewy górny: moment w funkcji kąta prądu γ – składowa magnesu, reluktancyjna i suma,
• prawy górny: wykres wektorowy w osiach dq – prąd I, strumień Ψ_s, napięcie V (przeskalowane),
• dolne: stan nieustalony po nagłym przyłożeniu napięć V_d, V_q (przy stałej prędkości). Prądy dążą do wartości zadanych z oscylacjami o częstotliwości zbliżonej do ω_r – to sprzężenie osi przez człony ω_r·L·I (SEM rotacji z zakładki 7.2–7.4!).
💡 Spróbuj: ustaw L_d = L_q (magnesy powierzchniowe) – moment reluktancyjny znika, MTPA to γ = 0. Zmniejsz R_s – oscylacje zanikają wolniej.
⚙ Inżynier: w praktyce regulator prądu „odsprzęga” osie, odejmując człony ω_r·L·I (tzw. feed-forward), dlatego w napędach nie widać tych oscylacji. Napięcie |V| rośnie z prędkością – przy granicy napięcia falownika trzeba zwiększyć −I_d (osłabianie pola).
"""

    def calc(self, p):
        Ld, Lq = p["Ld"] * 1e-3, p["Lq"] * 1e-3
        p1, psi, I, Rs = p["p1"], p["psi"], p["I"], p["Rs"]
        wr = p1 * 2 * math.pi * p["n"] / 60
        g = np.radians(np.linspace(-90, 90, 721))
        Tpm = 1.5 * p1 * psi * I * np.cos(g)
        Trel = 1.5 * p1 * (Ld - Lq) * (-I * np.sin(g)) * (I * np.cos(g))
        k = int(np.argmax(Tpm + Trel))
        gam = math.radians(p["gam"])
        id0, iq0 = -I * math.sin(gam), I * math.cos(gam)
        Vd = Rs * id0 - wr * Lq * iq0
        Vq = Rs * iq0 + wr * (Ld * id0 + psi)
        tau = max(Ld, Lq) / Rs
        t_end = min(max(5 * tau, 0.01), 0.4)
        dt = min(t_end / 4000, 2e-5)

        def f(_t, x):
            i_d, i_q = x
            return [(Vd - Rs * i_d + wr * Lq * i_q) / Ld,
                    (Vq - Rs * i_q - wr * (Ld * i_d + psi)) / Lq]
        t, X = rk4(f, [0.0, 0.0], t_end, dt, record_every=max(1, int(t_end / dt / 2000)))
        Te = 1.5 * p1 * (psi * X[:, 1] + (Ld - Lq) * X[:, 0] * X[:, 1])
        return dict(g=np.degrees(g), Tpm=Tpm, Trel=Trel, gm=np.degrees(g[k]), Tm=(Tpm + Trel)[k],
                    id0=id0, iq0=iq0, Vd=Vd, Vq=Vq, t=t, X=X, Te=Te, wr=wr, Ld=Ld, Lq=Lq)

    def draw(self, p, d):
        a0 = self.fig.add_subplot(2, 2, 1)
        a0.plot(d["g"], d["Tpm"], color=SKY, label="od magnesu")
        a0.plot(d["g"], d["Trel"], color=PEACH, label="reluktancyjny")
        a0.plot(d["g"], d["Tpm"] + d["Trel"], color=GREEN, lw=2.2, label="całkowity")
        a0.axvline(p["gam"], color=TEXT, ls="--", lw=1)
        a0.plot([d["gm"]], [d["Tm"]], "*", color=YELLOW, ms=12, label=f"MTPA γ={d['gm']:.1f}°")
        a0.set_xlabel("γ [°]"); a0.set_ylabel("T_e [N·m]"); a0.set_title("Moment vs kąt prądu")
        legend(a0, loc="lower center", fontsize=7)
        a1 = self.fig.add_subplot(2, 2, 2)
        a1.set_aspect("equal")
        psd = d["Ld"] * d["id0"] + p["psi"]
        psq = d["Lq"] * d["iq0"]
        Vmag = math.hypot(d["Vd"], d["Vq"]) or 1
        Imag = p["I"]
        Pmag = math.hypot(psd, psq) or 1
        sc = 1.0
        for (x, y), c, lab in [((d["id0"] / Imag, d["iq0"] / Imag), SKY, "I"),
                               ((psd / Pmag, psq / Pmag), GREEN, "Ψs"),
                               ((p["psi"] / Pmag, 0), MAUVE, "Ψ_PM"),
                               ((d["Vd"] / Vmag, d["Vq"] / Vmag), RED, "V")]:
            a1.annotate("", xy=(x * sc, y * sc), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="-|>", color=c, lw=2))
            a1.text(x * sc * 1.08, y * sc * 1.08, lab, color=c, fontsize=10, weight="bold")
        a1.set_xlim(-1.3, 1.3); a1.set_ylim(-0.4, 1.3)
        a1.set_xlabel("oś d"); a1.set_ylabel("oś q")
        a1.set_title("Wektory w osiach dq (znormalizowane)")
        a2 = self.fig.add_subplot(2, 2, 3)
        t = d["t"] * 1e3
        a2.plot(t, d["X"][:, 0], color=SKY, label="i_d")
        a2.plot(t, d["X"][:, 1], color=PEACH, label="i_q")
        a2.axhline(d["id0"], color=SKY, ls=":"); a2.axhline(d["iq0"], color=PEACH, ls=":")
        a2.set_xlabel("t [ms]"); a2.set_ylabel("[A]"); a2.set_title("Skok napięć V_d, V_q przy ω_r = const")
        legend(a2)
        a3 = self.fig.add_subplot(2, 2, 4)
        a3.plot(t, d["Te"], color=GREEN)
        a3.set_xlabel("t [ms]"); a3.set_ylabel("T_e [N·m]"); a3.set_title("Moment w stanie przejściowym")
        Te0 = 1.5 * p["p1"] * (p["psi"] * d["iq0"] + (d["Ld"] - d["Lq"]) * d["id0"] * d["iq0"])
        cosphi = math.cos(math.atan2(d["Vq"], d["Vd"]) - math.atan2(d["iq0"], d["id0"]))
        return [f"ω_r (elektr.) = {fmt(d['wr'])} rad/s",
                f"I_d = {fmt(d['id0'])} A   I_q = {fmt(d['iq0'])} A",
                f"V_d = {fmt(d['Vd'])} V   V_q = {fmt(d['Vq'])} V",
                f"|V| (amplituda fazowa) = {fmt(math.hypot(d['Vd'], d['Vq']))} V",
                f"T_e ustalony = {fmt(Te0)} N·m",
                f"MTPA: γ = {d['gm']:.1f}°,  T_max = {fmt(d['Tm'])} N·m",
                f"cos φ = {fmt(cosphi)}",
                f"P = 3/2(VdId+VqIq) = {fmt(1.5 * (d['Vd'] * d['id0'] + d['Vq'] * d['iq0']))} W",
                f"τ_d = Ld/Rs = {fmt(d['Ld'] / p['Rs'] * 1e3)} ms",
                f"τ_q = Lq/Rs = {fmt(d['Lq'] / p['Rs'] * 1e3)} ms"]


# ═══════════════════════════════════════════════════════════════════════════
#  7.6 Maszyna indukcyjna w osiach dq – rozruch bezpośredni
# ═══════════════════════════════════════════════════════════════════════════
def im_simulate(p, frame_idx):
    """Model dq silnika indukcyjnego (Równ. 7.21–7.23) w osiach wirujących z ωb.

    Zmienne stanu: strumienie Ψds, Ψqs, Ψdr, Ψqr, prędkość elektryczna ωr, kąt osi θb.
    """
    Rs, Rr = p["Rs"], p["Rr"]
    Lls = Llr = p["Lls"] * 1e-3
    Lm = p["Lm"] * 1e-3
    Ls, Lr = Lls + Lm, Llr + Lm
    D = Ls * Lr - Lm * Lm
    p1, J = p["p1"], p["J"]
    w1 = 2 * math.pi * p["f"]
    Vm = p["V"] * math.sqrt(2.0 / 3.0)          # amplituda napięcia fazowego
    TLv, tL = p["TL"], p["tL"]

    def f(t, x):
        pds, pqs, pdr, pqr, wr, thb = x
        ids = (Lr * pds - Lm * pdr) / D
        iqs = (Lr * pqs - Lm * pqr) / D
        idr = (Ls * pdr - Lm * pds) / D
        iqr = (Ls * pqr - Lm * pqs) / D
        wb = (0.0, wr, w1)[frame_idx]
        ang = w1 * t - thb
        vds, vqs = Vm * math.cos(ang), Vm * math.sin(ang)
        Te = 1.5 * p1 * (pds * iqs - pqs * ids)
        TL = TLv if t >= tL else 0.0
        return [vds - Rs * ids + wb * pqs,
                vqs - Rs * iqs - wb * pds,
                -Rr * idr + (wb - wr) * pqr,
                -Rr * iqr - (wb - wr) * pdr,
                p1 / J * (Te - TL),
                wb]

    dt = 1e-4
    t, X = rk4(f, [0, 0, 0, 0, 0, 0], p["tend"], dt, record_every=2)
    pds, pqs, pdr, pqr, wr, thb = X.T
    ids = (Lr * pds - Lm * pdr) / D
    iqs = (Lr * pqs - Lm * pqr) / D
    Te = 1.5 * p1 * (pds * iqs - pqs * ids)
    isa = np.real((ids + 1j * iqs) * np.exp(1j * thb))    # prąd fazy A
    return dict(t=t, ids=ids, iqs=iqs, Te=Te, n=wr / p1 * 60 / (2 * math.pi), ia=isa)


class TabIM(BaseTab):
    TITLE = "7.6 IM w osiach dq"
    SLIDERS = [
        ("V", "Napięcie międzyprzewodowe V", 50.0, 500.0, 400.0, "V"),
        ("f", "Częstotliwość f1", 10.0, 60.0, 50.0, "Hz"),
        ("Rs", "R_s", 0.2, 5.0, 1.405, "Ω"),
        ("Rr", "R_r' (sprowadzona)", 0.2, 5.0, 1.395, "Ω"),
        ("Lls", "L_sl = L_rl (rozproszenie)", 1.0, 20.0, 5.839, "mH"),
        ("Lm", "L_m (magnesowanie)", 50.0, 400.0, 172.2, "mH"),
        ("p1", "Pary biegunów p1", 1, 4, 2, "–", 1),
        ("J", "Moment bezwładności J", 0.005, 0.2, 0.0131, "kg·m²"),
        ("TL", "Moment obciążenia T_L", 0.0, 40.0, 20.0, "N·m"),
        ("tL", "Chwila załączenia T_L", 0.1, 1.2, 0.5, "s"),
        ("tend", "Czas symulacji", 0.4, 1.5, 0.8, "s"),
    ]
    EXPLAIN = """
## Silnik indukcyjny w osiach dq (7.6)
Silnik indukcyjny („klatkowy”) ma symetryczny stojan i symetryczny wirnik, a szczelina jest równa dookoła. Dlatego – jak pisze autor – prędkość osi ωb możemy wybrać DOWOLNIE, a indukcyjności i tak będą stałe. Równania (7.21–7.22) w zapisie strumieniowym:
» dΨ_ds/dt = V_ds − R_s·I_ds + ω_b·Ψ_qs         dΨ_qs/dt = V_qs − R_s·I_qs − ω_b·Ψ_ds
» dΨ_dr/dt = −R_r·I_dr + (ω_b − ω_r)·Ψ_qr     dΨ_qr/dt = −R_r·I_qr − (ω_b − ω_r)·Ψ_dr
» Ψ_s = L_sl·I_s + L_m·(I_s + I_r),   Ψ_r = L_rl·I_r + L_m·(I_s + I_r)
» T_e = 3/2·p1·(Ψ_ds·I_qs − Ψ_qs·I_ds),     (J/p1)·dω_r/dt = T_e − T_L
Człony ω_b·Ψ to właśnie SEM rotacji: stojan „porusza się” względem osi z prędkością −ω_b, a wirnik z prędkością ω_r − ω_b.
### Co pokazuje symulacja?
Silnik 4 kW, 400 V, 50 Hz załączamy bezpośrednio do sieci (rozruch bezpośredni), a w chwili t_L dokładamy obciążenie.
• Lewy górny: prądy i_d, i_q w wybranych osiach. W osiach synchronicznych (ω_b = ω1) po rozruchu są STAŁE – to „prąd stały” na karuzeli. W osiach stojana (ω_b = 0) są sinusoidami 50 Hz, w osiach wirnika – sinusoidami o częstotliwości poślizgu.
• Prawy górny: prąd fazy A – taki sam niezależnie od wybranych osi (bo to prawdziwy prąd w kablu!). Widać 5–7-krotny prąd rozruchowy.
• Lewy dolny: moment – linia ciągła dla wybranych osi, przerywana dla osi synchronicznych. Pokrywają się – moment jest wielkością fizyczną, nie zależy od „punktu widzenia”. Oscylacje na początku to efekt składowej stałej strumienia po załączeniu.
• Prawy dolny: prędkość – rozpędzanie, a potem spadek po obciążeniu (poślizg rośnie).
💡 Zmień „osie” w liście – zmienia się tylko lewy górny wykres! Zwiększ J – rozruch trwa dłużej. Zmniejsz V – maleje moment (∝ V²).
⚙ Inżynier: osie ω_b = 0 stosuje się do badania softstartów i falowników (Podsumowanie 7.12), ω_b = ω1 do sterowania polowo-zorientowanego (wszystko jest DC → proste regulatory PI), ω_b = ω_r dla maszyn pierścieniowych zasilanych od strony wirnika (np. generator DFIG w elektrowni wiatrowej).
"""

    def build_extra(self, parent):
        self.add_combo(parent, "frame", "Osie dq (ωb):", FRAMES, 2)

    def calc(self, p):
        fi = FRAMES.index(p["frame"])
        d = im_simulate(p, fi)
        ref = d if fi == 2 else im_simulate(p, 2)
        d["Te_ref"] = ref["Te"]
        return d

    def draw(self, p, d):
        t = d["t"]
        a0 = self.fig.add_subplot(2, 2, 1)
        a0.plot(t, d["ids"], color=SKY, label="i_ds"); a0.plot(t, d["iqs"], color=PEACH, label="i_qs")
        a0.set_title("Prądy stojana w osiach: " + p["frame"].split("(")[0]); a0.set_xlabel("t [s]"); a0.set_ylabel("[A]")
        legend(a0)
        a1 = self.fig.add_subplot(2, 2, 2)
        a1.plot(t, d["ia"], color=GREEN, lw=1); a1.set_title("Prąd fazy A (niezależny od osi)")
        a1.set_xlabel("t [s]"); a1.set_ylabel("i_a [A]")
        a2 = self.fig.add_subplot(2, 2, 3)
        a2.plot(t, d["Te"], color=MAUVE, label="T_e (wybrane osie)")
        a2.plot(t, d["Te_ref"], "--", color=YELLOW, lw=1, label="T_e (osie ω1)")
        a2.set_xlabel("t [s]"); a2.set_ylabel("[N·m]"); a2.set_title("Moment elektromagnetyczny")
        legend(a2)
        a3 = self.fig.add_subplot(2, 2, 4)
        a3.plot(t, d["n"], color=SKY); a3.set_xlabel("t [s]"); a3.set_ylabel("n [obr/min]")
        a3.set_title("Prędkość obrotowa")
        ns = 60 * p["f"] / p["p1"]
        k = len(t) - 1
        ia_tail = d["ia"][int(0.85 * k):]
        return [f"n_sync = {fmt(ns)} obr/min",
                f"n końcowe = {fmt(d['n'][-1])} obr/min",
                f"poślizg s = {fmt((ns - d['n'][-1]) / ns * 100)} %",
                f"I szczyt. rozruchu = {fmt(np.max(np.abs(d['ia'])))} A",
                f"I skut. końcowy  ≈ {fmt(np.sqrt(np.mean(ia_tail ** 2)))} A",
                f"T_e max = {fmt(np.max(d['Te']))} N·m",
                f"T_e końcowy = {fmt(d['Te'][-1])} N·m",
                f"różnica T_e między osiami: {fmt(np.max(np.abs(d['Te'] - d['Te_ref'])))} N·m",
                "(≈0 → moment nie zależy od osi)"]


# ═══════════════════════════════════════════════════════════════════════════
#  7.7 Nasycenie magnetyczne w modelu dq
# ═══════════════════════════════════════════════════════════════════════════
class TabSat(BaseTab):
    TITLE = "7.7 Nasycenie"
    SLIDERS = [
        ("Lu", "Indukcyjność nienasycona L_u", 50.0, 500.0, 200.0, "mH"),
        ("Ps", "Strumień nasycenia Ψ_s", 0.3, 2.0, 1.0, "Wb"),
        ("Li", "Indukcyjność „powietrzna” L_∞", 0.0, 50.0, 5.0, "mH"),
        ("id", "Prąd magnesujący i_dm", 0.0, 20.0, 6.0, "A"),
        ("iq", "Prąd magnesujący i_qm", 0.0, 20.0, 4.0, "A"),
        ("kq", "Współczynnik osi q: Ψq* = kq·Ψd*", 0.3, 1.0, 1.0, "–"),
    ]
    EXPLAIN = """
## Nasycenie – żelazo „się zapycha” (7.7)
Żelazo wzmacnia pole magnetyczne tysiące razy, ale tylko do pewnego momentu. Gdy wszystkie „domeny magnetyczne” są już ustawione, dalsze zwiększanie prądu prawie nie zwiększa strumienia – jak gąbka, która nasiąkła wodą. Krzywa Ψ(i) się wygina (lewy górny wykres).
Model z książki: każda oś ma własną krzywą magnesowania Ψ*_dm(i_m), Ψ*_qm(i_m), ale ZALEŻY ONA TYLKO od wypadkowego prądu magnesującego:
» i_m = √(i_dm² + i_qm²),   i_dm = i_d + i_dr + i_F,   i_qm = i_q + i_qr
» L_dm(i_m) = Ψ*_dm(i_m)/i_m     (indukcyjność „cięciwowa”, do stanu ustalonego)
### Stan nieustalony – trzy różne indukcyjności
Przy zmianie prądów potrzebna jest pochodna strumienia (Równ. 7.27–7.31):
» dΨ_dm/dt = L_dmt·di_dm/dt + L_dqm·di_qm/dt
» dΨ_qm/dt = L_qdm·di_dm/dt + L_qmt·di_qm/dt
» L_dmt = L_dm + (i_dm²/i_m)·dL_dm/di_m        L_qmt = L_qm + (i_qm²/i_m)·dL_qm/di_m
» L_dqm = (i_dm·i_qm/i_m)·dL_dm/di_m
• L_dmt, L_qmt – indukcyjności „przejściowe” (styczne w kierunku danej osi),
• L_dqm – sprzężenie skrośne: zmiana prądu w osi q zmienia strumień w osi d, choć osie są prostopadłe! Pojawia się TYLKO przy nasyceniu (dL/di ≠ 0) i gdy oba prądy są niezerowe.
### Wykresy
• lewy górny: krzywa magnesowania, punkt pracy, cięciwa (L_dm) i styczna (L_dyn = dΨ/di),
• prawy górny: L_dm i L_dyn w funkcji prądu – obie spadają, styczna szybciej,
• lewy dolny: L_dmt, L_qmt, L_dqm w funkcji kąta wektora i_m (przy stałym |i_m|) – przy 0° i 90° sprzężenie znika, największe (co do modułu) jest przy 45°. L_dqm wychodzi UJEMNE, bo przy nasyceniu dL_dm/di_m < 0,
• prawy dolny: mapa sprzężenia skrośnego L_dqm(i_dm, i_qm).
💡 Ustaw małe prądy (np. 1 A) – nasycenia prawie nie ma, L_dqm ≈ 0. Zwiększ prądy do 15 A – sprzężenie rośnie.
⚙ Inżynier: nowoczesne maszyny (np. silniki samochodów elektrycznych) pracują „na granicy” nasycenia, więc pominięcie go daje błędy momentu rzędu 10–30%. W sterownikach używa się tablic L(i_d, i_q) z pomiarów lub MES. Przy małych sygnałach AC (testy postojowe) zamiast L_dmt występują indukcyjności przyrostowe z lokalnej pętli histerezy, μ_i ≈ (120–150)·μ0. Uwaga: dla kq ≠ 1 wzory dają L_dqm ≠ L_qdm – książka zakłada tu wzajemność, która ściśle zachodzi dla jednej wspólnej krzywej (kq = 1).
"""

    @staticmethod
    def psi_curve(i, Lu, Ps, Li):
        return Ps * np.tanh(Lu * i / Ps) + Li * i

    @staticmethod
    def dpsi(i, Lu, Ps, Li):
        return Lu / np.cosh(Lu * i / Ps) ** 2 + Li

    def calc(self, p):
        Lu, Li, Ps, kq = p["Lu"] * 1e-3, p["Li"] * 1e-3, p["Ps"], p["kq"]
        i = np.linspace(1e-4, 30, 600)
        psi = self.psi_curve(i, Lu, Ps, Li)
        Lm = psi / i
        Ldyn = self.dpsi(i, Lu, Ps, Li)
        dLdi = (Ldyn - Lm) / i                    # d(Ψ/i)/di = (Ψ' − Ψ/i)/i

        def trans(idm, iqm):
            im = np.maximum(np.hypot(idm, iqm), 1e-6)
            L = self.psi_curve(im, Lu, Ps, Li) / im
            dL = (self.dpsi(im, Lu, Ps, Li) - L) / im
            Ldmt = L + idm ** 2 / im * dL
            Lqmt = kq * (L + iqm ** 2 / im * dL)
            Ldqm = idm * iqm / im * dL
            Lqdm = kq * Ldqm
            return L, Ldmt, Lqmt, Ldqm, Lqdm
        im0 = max(math.hypot(p["id"], p["iq"]), 1e-3)
        ang = np.linspace(0, math.pi / 2, 181)
        ang_res = trans(im0 * np.cos(ang), im0 * np.sin(ang))
        g = np.linspace(0, 20, 81)
        GD, GQ = np.meshgrid(g, g)
        grid_res = trans(GD, GQ)
        op = trans(np.array([p["id"]]), np.array([p["iq"]]))
        return dict(i=i, psi=psi, Lm=Lm, Ldyn=Ldyn, dLdi=dLdi, im0=im0, ang=np.degrees(ang),
                    A=ang_res, GD=GD, GQ=GQ, G=grid_res, op=[float(x[0]) for x in op])

    def draw(self, p, d):
        Lu, Li, Ps = p["Lu"] * 1e-3, p["Li"] * 1e-3, p["Ps"]
        a0 = self.fig.add_subplot(2, 2, 1)
        a0.plot(d["i"], d["psi"], color=SKY, lw=2, label="Ψ*_dm(i_m)")
        a0.plot(d["i"], p["kq"] * d["psi"], color=MAUVE, lw=1.2, label="Ψ*_qm = kq·Ψ*_dm")
        a0.plot(d["i"], Lu * d["i"], ":", color=SUBTEXT, label="bez nasycenia")
        im0 = d["im0"]
        ps0 = float(self.psi_curve(im0, Lu, Ps, Li))
        a0.plot([0, im0], [0, ps0], color=YELLOW, lw=1.5, label="cięciwa → L_dm")
        sl = float(self.dpsi(im0, Lu, Ps, Li))
        xx = np.array([max(0, im0 - 6), im0 + 6])
        a0.plot(xx, ps0 + sl * (xx - im0), color=RED, lw=1.5, label="styczna → L_dyn")
        a0.plot([im0], [ps0], "o", color=TEXT)
        a0.set_ylim(0, max(d["psi"]) * 1.15)
        a0.set_xlabel("i_m [A]"); a0.set_ylabel("Ψ [Wb]"); a0.set_title("Krzywa magnesowania (Rys. 7.5)")
        legend(a0, fontsize=7)
        a1 = self.fig.add_subplot(2, 2, 2)
        a1.plot(d["i"], d["Lm"] * 1e3, color=YELLOW, label="L_dm = Ψ/i (cięciwowa)")
        a1.plot(d["i"], d["Ldyn"] * 1e3, color=RED, label="L_dyn = dΨ/di (styczna)")
        a1.axvline(im0, color=TEXT, ls="--", lw=1)
        a1.set_xlabel("i_m [A]"); a1.set_ylabel("[mH]"); a1.set_title("Indukcyjności vs prąd")
        legend(a1)
        a2 = self.fig.add_subplot(2, 2, 3)
        L, Ldmt, Lqmt, Ldqm, Lqdm = d["A"]
        a2.plot(d["ang"], Ldmt * 1e3, color=SKY, label="L_dmt")
        a2.plot(d["ang"], Lqmt * 1e3, color=MAUVE, label="L_qmt")
        a2.plot(d["ang"], Ldqm * 1e3, color=PEACH, lw=2, label="L_dqm (sprzężenie)")
        a2.plot(d["ang"], L * 1e3, ":", color=YELLOW, label="L_dm")
        a2.set_xlabel("kąt wektora i_m [°]"); a2.set_ylabel("[mH]")
        a2.set_title(f"Indukcyjności przejściowe przy |i_m| = {im0:.2f} A")
        legend(a2, fontsize=7)
        a3 = self.fig.add_subplot(2, 2, 4)
        cs = a3.contourf(d["GD"], d["GQ"], d["G"][3] * 1e3, levels=20, cmap="magma")
        self.fig.colorbar(cs, ax=a3, label="L_dqm [mH]")
        a3.plot([p["id"]], [p["iq"]], "o", color=GREEN, ms=8)
        a3.set_xlabel("i_dm [A]"); a3.set_ylabel("i_qm [A]"); a3.set_title("Mapa sprzężenia skrośnego L_dqm")
        L0, Ldmt0, Lqmt0, Ldqm0, Lqdm0 = d["op"]
        return [f"|i_m| = {fmt(im0)} A",
                f"Ψ_m   = {fmt(ps0)} Wb",
                f"L_dm  = {fmt(L0 * 1e3)} mH  (L_u = {fmt(p['Lu'])})",
                f"L_dyn = {fmt(sl * 1e3)} mH",
                f"L_dmt = {fmt(Ldmt0 * 1e3)} mH",
                f"L_qmt = {fmt(Lqmt0 * 1e3)} mH",
                f"L_dqm = {fmt(Ldqm0 * 1e3)} mH",
                f"L_qdm = {fmt(Lqdm0 * 1e3)} mH",
                f"stopień nasycenia k_s = L_u/L_dm = {fmt(p['Lu'] * 1e-3 / L0)}",
                "",
                "Sprawdzenie: dla i_qm=0 → L_dmt = L_dyn,",
                "L_dqm = 0 (brak sprzężenia skrośnego)."]


# ═══════════════════════════════════════════════════════════════════════════
#  7.8 Efekt naskórkowości – obwody równoległe o stałych parametrach
# ═══════════════════════════════════════════════════════════════════════════
MATERIALS = {"Miedź (75°C) σ = 4.7e7 S/m": 4.7e7, "Aluminium odlewane (75°C) σ = 2.6e7 S/m": 2.6e7,
             "Mosiądz σ = 1.5e7 S/m": 1.5e7}


def deep_bar_z(f, h, sigma):
    """Impedancja głębokiego pręta w żłobku / R_dc (klasyczny wzór Fieldsa)."""
    w = 2 * np.pi * np.asarray(f, dtype=float)
    k = (1 + 1j) * np.sqrt(w * MU0 * sigma / 2) * h        # (1+j)·ξ
    k = np.where(np.abs(k) < 1e-8, 1e-8, k)
    return k / np.tanh(k)


def ladder_z(x, w, n):
    R, L = np.exp(x[:n]), np.exp(x[n:])
    Y = np.zeros_like(w, dtype=complex)
    for k in range(n):
        Y += 1 / (R[k] + 1j * w * L[k])
    return 1 / Y


def fit_ladder(f, Zt, Ldc, n):
    from scipy.optimize import least_squares
    w = 2 * np.pi * f

    def res(x):
        e = (ladder_z(x, w, n) - Zt) / np.abs(Zt)
        return np.concatenate([e.real, e.imag])
    best = None
    for spread in (3.0, 6.0, 12.0):
        R0 = np.array([n * spread ** k for k in range(n)], dtype=float)
        L0 = np.array([n * Ldc * 0.8 / spread ** (0.5 * k) for k in range(n)], dtype=float)
        x0 = np.log(np.concatenate([R0, L0]))
        try:
            r = least_squares(res, x0, method="trf", max_nfev=3000)
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r
    return best.x


class TabSkin(BaseTab):
    TITLE = "7.8 Naskórkowość"
    SLIDERS = [
        ("h", "Wysokość pręta klatki h", 5.0, 60.0, 30.0, "mm"),
        ("fmax", "Maks. częstotliwość dopasowania", 60.0, 5000.0, 1000.0, "Hz"),
        ("fsel", "Częstotliwość do rozkładu prądu", 0.5, 1000.0, 50.0, "Hz"),
    ]
    EXPLAIN = """
## Efekt naskórkowości (7.8) – prąd ucieka na powierzchnię
Prąd przemienny w grubym przewodzie nie płynie równomiernie. Zmienne pole magnetyczne indukuje w przewodzie prądy wirowe, które „wypychają” prąd w stronę powierzchni (w pręcie klatki – w stronę szczeliny, do góry żłobka). Im wyższa częstotliwość, tym cieńsza „skórka”, w której płynie prąd. Grubość tej skórki to głębokość wnikania:
» δ = √(2/(ω·μ0·σ))       (dla miedzi przy 50 Hz: ok. 1 cm)
Skutek dla pręta klatki silnika:
• rezystancja ROŚNIE: R_r(ω) = k_R·R_dc, k_R > 1 (prąd ma węższą drogę),
• indukcyjność rozproszenia MALEJE: L_rl(ω) = k_X·L_dc, k_X < 1.
Przy rozruchu (poślizg s = 1, częstotliwość w wirniku 50 Hz) rezystancja wirnika jest duża → duży moment rozruchowy i mniejszy prąd. Przy pracy normalnej (1–3 Hz w wirniku) rezystancja jest mała → wysoka sprawność. Konstruktorzy robią to celowo – silniki z głębokim żłobkiem lub z podwójną klatką.
### Jak to wstawić do modelu dq?
Model dq lubi stałe parametry, a tu R i L zależą od częstotliwości. Sztuczka z Rys. 7.6: zastępujemy jeden obwód o zmiennych parametrach kilkoma (2–3) obwodami R-L o STAŁYCH parametrach połączonymi równolegle. Dobieramy je (regresją, jak tu – metodą najmniejszych kwadratów), aby ich impedancja pasowała do rzeczywistej w całym zakresie częstotliwości.
» Z_pręta/R_dc = (1+j)ξ·coth((1+j)ξ),   ξ = h/δ
» Z_zast = 1 / Σ_k 1/(R_k + jωL_k)
### Wykresy
• górne: k_R(f) i k_X(f) – dokładne (biała linia) i przybliżone 1, 2, 3 obwodami,
• lewy dolny: rozkład gęstości prądu wzdłuż wysokości pręta (0 = dno żłobka, 1 = strona szczeliny) przy wybranej częstotliwości,
• prawy dolny: błąd dopasowania |Z| w % – jeden obwód jest bardzo zły, trzy wystarczają (co potwierdza Podsumowanie 7.12: „trzy obwody w praktyce wystarczają dla wszystkich SM i IM”).
💡 Zmień materiał na aluminium – δ rośnie (gorzej przewodzi), efekt słabnie. Zwiększ h – efekt rośnie jak h².
⚙ Inżynier: parametry obwodów wyznacza się MES lub z próby postojowej odpowiedzi częstotliwościowej (SSFR). W dużych turbogeneratorach litej stali wirnika potrzeba nawet 3 obwodów w osi d i q. Straty w żelazie modeluje się podobnie – dodatkowymi zwartymi uzwojeniami dq (Boldea & Nasar 1987).
"""

    def build_extra(self, parent):
        self.add_combo(parent, "mat", "Materiał pręta:", list(MATERIALS), 0)
        self.add_combo(parent, "nsel", "Wyróżniona liczba obwodów:", ["1", "2", "3"], 2)

    def calc(self, p):
        sigma = MATERIALS[p["mat"]]
        h = p["h"] * 1e-3
        f = np.logspace(-1, math.log10(p["fmax"]), 90)
        Zt = deep_bar_z(f, h, sigma)
        Ldc = MU0 * sigma * h * h / 3            # L_dc / R_dc [s]
        fits = {}
        for n in (1, 2, 3):
            x = fit_ladder(f, Zt, Ldc, n)
            fits[n] = (x, ladder_z(x, 2 * np.pi * f, n))
        y = np.linspace(0, 1, 200)
        ks = (1 + 1j) * math.sqrt(2 * math.pi * p["fsel"] * MU0 * sigma / 2) * h
        J = np.abs(ks * np.cosh(ks * y) / np.sinh(ks))
        return dict(f=f, Zt=Zt, Ldc=Ldc, fits=fits, y=y, J=J, sigma=sigma, h=h)

    def draw(self, p, d):
        f, Zt, Ldc = d["f"], d["Zt"], d["Ldc"]
        w = 2 * np.pi * f
        nsel = int(p["nsel"])
        cols = {1: RED, 2: YELLOW, 3: GREEN}
        a0 = self.fig.add_subplot(2, 2, 1)
        a1 = self.fig.add_subplot(2, 2, 2)
        a0.semilogx(f, Zt.real, color=TEXT, lw=2.5, label="dokładnie")
        a1.semilogx(f, Zt.imag / (w * Ldc), color=TEXT, lw=2.5, label="dokładnie")
        for n, (x, Zf) in d["fits"].items():
            lw = 2 if n == nsel else 1
            ls = "-" if n == nsel else "--"
            a0.semilogx(f, Zf.real, ls, color=cols[n], lw=lw, label=f"{n} obw.")
            a1.semilogx(f, Zf.imag / (w * Ldc), ls, color=cols[n], lw=lw, label=f"{n} obw.")
        for a, tt, yl in [(a0, "k_R = R_r(f)/R_dc", "k_R"), (a1, "k_X = L_rl(f)/L_dc", "k_X")]:
            a.axvline(50, color=SUBTEXT, ls=":", lw=1)
            a.set_xlabel("częstotliwość w wirniku f2 [Hz]"); a.set_ylabel(yl); a.set_title(tt)
            legend(a, fontsize=7)
        a2 = self.fig.add_subplot(2, 2, 3)
        a2.plot(d["y"], d["J"], color=SKY, lw=2)
        a2.axhline(1, color=SUBTEXT, ls=":")
        a2.set_xlabel("położenie w pręcie y/h (0 = dno, 1 = szczelina)")
        a2.set_ylabel("|J|/J_dc"); a2.set_title(f"Rozkład gęstości prądu przy {p['fsel']:.3g} Hz")
        a3 = self.fig.add_subplot(2, 2, 4)
        lines = []
        for n, (x, Zf) in d["fits"].items():
            err = np.abs(Zf - Zt) / np.abs(Zt) * 100
            a3.semilogx(f, err, color=cols[n], lw=2 if n == nsel else 1, label=f"{n} obw.: max {err.max():.2g}%")
        a3.set_xlabel("f [Hz]"); a3.set_ylabel("błąd |ΔZ|/|Z| [%]"); a3.set_title("Jakość zastępowania")
        a3.set_yscale("log")
        legend(a3)
        x = d["fits"][nsel][0]
        R, L = np.exp(x[:nsel]), np.exp(x[nsel:])
        delta50 = math.sqrt(2 / (2 * math.pi * 50 * MU0 * d["sigma"]))
        z50 = deep_bar_z(np.array([50.0]), d["h"], d["sigma"])[0]
        lines += [f"δ(50 Hz) = {fmt(delta50 * 1e3)} mm,  ξ = h/δ = {fmt(d['h'] / delta50)}",
                  f"k_R(50 Hz) = {fmt(z50.real)}",
                  f"k_X(50 Hz) = {fmt(z50.imag / (2 * math.pi * 50 * Ldc))}",
                  f"L_dc/R_dc = {fmt(Ldc * 1e3)} ms",
                  f"Obwody zastępcze ({nsel}) w j.w. R_dc:"]
        for k in range(nsel):
            lines.append(f"  R{k + 1} = {fmt(R[k])} ,  L{k + 1}/R_dc = {fmt(L[k] * 1e3)} ms")
        lines.append(f"  R_dc wypadkowe = {fmt(1 / np.sum(1 / R))} (≈1)")
        return lines


# ═══════════════════════════════════════════════════════════════════════════
#  7.9–7.10 Transformacja Parka i wektor przestrzenny
# ═══════════════════════════════════════════════════════════════════════════
class TabPark(BaseTab):
    TITLE = "7.9–7.10 Park i wektor"
    ANIM = True
    SLIDERS = [
        ("Im", "Amplituda prądu I_m", 1.0, 20.0, 10.0, "A"),
        ("Vm", "Amplituda napięcia fazowego V_m", 10.0, 400.0, 325.0, "V"),
        ("f1", "Częstotliwość f1", 1.0, 100.0, 50.0, "Hz"),
        ("s", "Poślizg s (ωr = (1−s)ω1)", 0.0, 1.0, 0.05, "–"),
        ("phi", "Kąt fazowy φ (V wyprzedza I)", -90.0, 90.0, 30.0, "°"),
        ("unb", "Asymetria amplitudy fazy C", -50.0, 50.0, 0.0, "%"),
        ("h5", "5. harmoniczna prądu", 0.0, 30.0, 0.0, "%"),
        ("i0", "Składowa zerowa (3. harm.)", 0.0, 5.0, 0.0, "A"),
    ]
    EXPLAIN = """
## Równoważność maszyny 3-fazowej i modelu dq (7.9)
Trzy fazy A, B, C przesunięte o 120° wytwarzają razem jedno pole wirujące. To samo pole można wytworzyć DWOMA prostopadłymi uzwojeniami d i q. Warunek: przepływy (MMF, „siła magnesująca” = prąd × zwoje) muszą dawać ten sam wynik – rzutujemy prądy faz na osie d i q (Rys. 7.7). Wychodzi transformacja Parka (Równ. 7.36–7.37), tutaj w wersji z niezmiennikiem amplitudy (czynnik 2/3):
» i_d = 2/3·[i_A·cos θ + i_B·cos(θ − 2π/3) + i_C·cos(θ + 2π/3)]
» i_q = −2/3·[i_A·sin θ + i_B·sin(θ − 2π/3) + i_C·sin(θ + 2π/3)]
» i_0 = (i_A + i_B + i_C)/3          (składowa zerowa)
Dwie zmienne d, q nie wystarczą do opisu trzech faz – trzecią jest składowa zerowa i_0. Nie wytwarza ona pola wirującego, płynie tylko przez rezystancję i indukcyjność rozproszenia. Przy połączeniu w gwiazdę bez przewodu neutralnego i_0 = 0.
Moc: przy współczynniku 2/3 moc w modelu dq trzeba pomnożyć przez 3/2 (Równ. 7.42):
» p_abc = v_A·i_A + v_B·i_B + v_C·i_C = 3/2·(v_d·i_d + v_q·i_q) + 3·v_0·i_0
» Q = 3/2·(v_q·i_d − v_d·i_q)   (moc bierna – mimo że w stanie ustalonym wszystko jest DC!)

## Wektor przestrzenny (7.10)
Zamiast pary liczb (i_d, i_q) piszemy jedną liczbę zespoloną I_s = i_d + j·i_q. Równanie napięć stojana skraca się do (Równ. 7.51):
» V_s = R_s·I_s + dΨ_s/dt + j·ω_b·Ψ_s
W osiach stojana (ω_b = 0) wektor prądu kręci się po okręgu z prędkością ω1 (Rys. 7.8a). W osiach synchronicznych (ω_b = ω1) ten sam wektor STOI w miejscu (Rys. 7.8b).
### Wykresy (kliknij ▶ Start)
• lewy górny: prądy faz – kursor pokazuje chwilę animacji,
• prawy górny: i_d, i_q, i_0 w wybranych osiach – dla ω1 i symetrii to proste linie!
• lewy dolny: trajektoria wektora prądu w osiach stojana (αβ, szara) i w osiach wybranych (kolorowa); strzałka i obracająca się oś d,
• prawy dolny: moc chwilowa policzona z faz (linia) i z dq0 (kropki) – idealnie się pokrywają.
💡 Dodaj 5. harmoniczną – w osiach synchronicznych pojawiają się oscylacje 6·f1 (5. harm. wiruje w przeciwną stronę: −5 − 1 = −6). Dodaj asymetrię – oscylacje 2·f1 (składowa przeciwna). Dodaj i_0 – pojawia się tylko w i_0, nie w d, q.
⚙ Inżynier: wariant √(2/3) zachowuje moc bez czynnika 3/2 (transformacja ortonormalna), wariant 2/3 zachowuje amplitudy i dominuje w sterowaniu napędami. Dla maszyn 6-, 9-, 12-fazowych stosuje się 2, 3, 4 pary osi dq plus składowe zerowe.
"""

    def build_extra(self, parent):
        self.add_combo(parent, "frame", "Osie dq (ωb):", FRAMES, 2)
        self.k = 0

    def calc(self, p):
        w1 = 2 * math.pi * p["f1"]
        wr = (1 - p["s"]) * w1
        t = np.linspace(0, 2 / p["f1"], 721)
        Im, h5 = p["Im"], p["h5"] / 100
        ia = Im * np.cos(w1 * t) + h5 * Im * np.cos(5 * w1 * t)
        ib = Im * np.cos(w1 * t - TWO_PI_3) + h5 * Im * np.cos(5 * (w1 * t - TWO_PI_3))
        ic = Im * (1 + p["unb"] / 100) * np.cos(w1 * t + TWO_PI_3) + h5 * Im * np.cos(5 * (w1 * t + TWO_PI_3))
        z = p["i0"] * np.cos(3 * w1 * t)
        ia, ib, ic = ia + z, ib + z, ic + z
        ph = math.radians(p["phi"])
        va = p["Vm"] * np.cos(w1 * t + ph)
        vb = p["Vm"] * np.cos(w1 * t + ph - TWO_PI_3)
        vc = p["Vm"] * np.cos(w1 * t + ph + TWO_PI_3)
        fi = FRAMES.index(p["frame"])
        thb = [0 * t, wr * t, w1 * t][fi]
        idq = park(ia, ib, ic, thb)
        vdq = park(va, vb, vc, thb)
        ial, ibe, _ = park(ia, ib, ic, 0 * t)
        pabc = va * ia + vb * ib + vc * ic
        pdq = 1.5 * (vdq[0] * idq[0] + vdq[1] * idq[1]) + 3 * vdq[2] * idq[2]
        qdq = 1.5 * (vdq[1] * idq[0] - vdq[0] * idq[1])
        return dict(t=t, ia=ia, ib=ib, ic=ic, idq=idq, thb=thb, ial=ial, ibe=ibe,
                    pabc=pabc, pdq=pdq, qdq=qdq)

    def draw(self, p, d):
        t = d["t"] * 1e3
        a0 = self.fig.add_subplot(2, 2, 1)
        for y, c, lab in [(d["ia"], SKY, "i_A"), (d["ib"], GREEN, "i_B"), (d["ic"], PEACH, "i_C")]:
            a0.plot(t, y, color=c, label=lab)
        self._c0 = a0.axvline(0, color=TEXT, lw=1)
        a0.set_xlabel("t [ms]"); a0.set_ylabel("[A]"); a0.set_title("Prądy fazowe abc")
        legend(a0, ncol=3, loc="upper right")
        a1 = self.fig.add_subplot(2, 2, 2)
        a1.plot(t, d["idq"][0], color=SKY, label="i_d")
        a1.plot(t, d["idq"][1], color=PEACH, label="i_q")
        a1.plot(t, d["idq"][2], color=MAUVE, label="i_0")
        self._c1 = a1.axvline(0, color=TEXT, lw=1)
        a1.set_xlabel("t [ms]"); a1.set_ylabel("[A]"); a1.set_title("Po transformacji Parka")
        legend(a1, ncol=3, loc="upper right")
        a2 = self.fig.add_subplot(2, 2, 3)
        a2.set_aspect("equal")
        a2.plot(d["ial"], d["ibe"], color=SURF1, lw=3, label="osie stojana (αβ)")
        a2.plot(d["idq"][0], d["idq"][1], color=YELLOW, lw=1.5, label="osie wybrane (dq)")
        R = 1.35 * max(np.max(np.hypot(d["ial"], d["ibe"])), 1e-3)
        a2.set_xlim(-R, R); a2.set_ylim(-R, R)
        self._axd, = a2.plot([], [], color=SKY, lw=1, ls="--")
        self._arr = a2.annotate("", xy=(0, 0), xytext=(0, 0),
                                arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=2))
        self._pt, = a2.plot([], [], "o", color=YELLOW)
        a2.set_title("Wektor przestrzenny I_s (Rys. 7.8)"); legend(a2, fontsize=7, loc="lower right")
        self._R = R
        a3 = self.fig.add_subplot(2, 2, 4)
        a3.plot(t, d["pabc"] / 1e3, color=GREEN, label="p z faz abc")
        a3.plot(t[::12], d["pdq"][::12] / 1e3, "o", ms=3, color=RED, label="p = 3/2(vd·id+vq·iq)+3v0i0")
        a3.plot(t, d["qdq"] / 1e3, color=MAUVE, label="q = 3/2(vq·id−vd·iq)")
        a3.set_xlabel("t [ms]"); a3.set_ylabel("[kW], [kvar]"); a3.set_title("Równoważność mocy (Równ. 7.42–7.45)")
        legend(a3, fontsize=7)
        self._d = d
        self.k = 0
        self._upd()
        P = 1.5 * p["Vm"] * p["Im"] * math.cos(math.radians(p["phi"]))
        Q = 1.5 * p["Vm"] * p["Im"] * math.sin(math.radians(p["phi"]))
        return [f"średnie i_d = {fmt(np.mean(d['idq'][0]))} A",
                f"średnie i_q = {fmt(np.mean(d['idq'][1]))} A",
                f"tętnienia i_d (p-p) = {fmt(np.ptp(d['idq'][0]))} A",
                f"max |p_abc − p_dq0| = {fmt(np.max(np.abs(d['pabc'] - d['pdq'])))} W",
                f"P średnie = {fmt(np.mean(d['pabc']))} W",
                f"  (teoria symetr.: 3/2·Vm·Im·cosφ = {fmt(P)} W)",
                f"Q średnie = {fmt(np.mean(d['qdq']))} var",
                f"  (teoria: 3/2·Vm·Im·sinφ = {fmt(Q)} var)",
                f"i_0 RMS = {fmt(np.sqrt(np.mean(d['idq'][2] ** 2)))} A"]

    def _upd(self):
        d = self._d
        k = self.k % len(d["t"])
        tk = d["t"][k] * 1e3
        self._c0.set_xdata([tk, tk]); self._c1.set_xdata([tk, tk])
        x, y = d["ial"][k], d["ibe"][k]
        self._arr.xy = (x, y)
        th = d["thb"][k]
        R = self._R
        self._axd.set_data([0, R * math.cos(th)], [0, R * math.sin(th)])
        self._pt.set_data([d["idq"][0][k]], [d["idq"][1][k]])

    def anim_step(self):
        self.k += 3
        self._upd()


# ═══════════════════════════════════════════════════════════════════════════
#  7.11 Modele wysokiej częstotliwości – napięcie wspólne i prądy łożyskowe
# ═══════════════════════════════════════════════════════════════════════════
class TabHF(BaseTab):
    TITLE = "7.11 Model HF / łożyska"
    SLIDERS = [
        ("Vdc", "Napięcie obwodu DC falownika", 300.0, 800.0, 560.0, "V"),
        ("fsw", "Częstotliwość łączeń PWM", 1.0, 20.0, 5.0, "kHz"),
        ("m", "Współczynnik modulacji m", 0.1, 1.0, 0.8, "–"),
        ("f1", "Częstotliwość podstawowa f1", 10.0, 100.0, 50.0, "Hz"),
        ("tr", "Czas narastania zbocza IGBT", 50.0, 2000.0, 200.0, "ns"),
        ("Csf", "C_sf uzwojenie–obudowa (całk.)", 0.5, 20.0, 5.0, "nF"),
        ("Csr", "C_sr stojan–wirnik", 10.0, 500.0, 80.0, "pF"),
        ("Crf", "C_rf wirnik–obudowa", 200.0, 5000.0, 1500.0, "pF"),
        ("Cb", "C_b łożysk (film olejowy)", 20.0, 1000.0, 200.0, "pF"),
        ("Vth", "Napięcie przebicia filmu olejowego", 2.0, 40.0, 20.0, "V"),
        ("L", "Indukcyjność fazy L", 1.0, 50.0, 10.0, "mH"),
        ("R", "Rezystancja fazy R", 0.1, 5.0, 1.0, "Ω"),
        ("Csw", "Pojemność międzyzwojowa C_sw", 10.0, 2000.0, 200.0, "pF"),
    ]
    EXPLAIN = """
## Gdy liczą się mikrosekundy (7.11)
Tranzystor IGBT w falowniku przełącza napięcie setek woltów w czasie 0,5–2 µs. Dla tak szybkich zmian cewki uzwojeń stanowią „ścianę” (duża reaktancja ωL), a do głosu dochodzą malutkie pojemności pasożytnicze: między zwojami, między uzwojeniem a obudową, między stojanem a wirnikiem. To jak w telefonie „z puszek”: wolne zmiany idą sznurkiem (drutem), a szybkie drgania przeskakują przez powietrze.
• do ok. 400 Hz wystarczy model R-L-E (dq),
• powyżej 20 kHz – model pojemnościowy (Rys. 7.9: rozłożone L i C wzdłuż uzwojenia),
• 400 Hz – 20 kHz – model uniwersalny (Rys. 7.10), razem z kablem.
### Napięcie wspólne (common mode)
Falownik 3-fazowy łączy każdą fazę do +V_dc/2 albo −V_dc/2. Suma trzech napięć NIGDY nie jest zerem – średnia skacze schodkami ±V_dc/6, ±V_dc/2:
» v_cm = (v_A0 + v_B0 + v_C0)/3
Punkt neutralny silnika „podskakuje” względem ziemi – to źródło V_nsg z Rys. 7.10.
### Napięcie na wale i prądy łożyskowe
Wirnik jest połączony z uzwojeniem przez C_sr, a z obudową przez C_rf i łożyska C_b. To dzielnik pojemnościowy:
» v_wału = BVR·v_cm,     BVR = C_sr/(C_sr + C_rf + C_b)   (zwykle 2–10%)
Film olejowy w łożysku jest izolatorem tylko do pewnego napięcia. Gdy v_wału przekroczy próg, następuje wyładowanie (EDM – jak mini-spawarka) i mikroskopijny krater na bieżni. Po milionach takich iskier łożysko ma „tarkę” (fluting) i hałasuje. To jedna z głównych przyczyn awarii silników zasilanych z falowników!
Prąd przez pojemność uzwojenie–obudowa: i_cm = C_sf·dv/dt ≈ C_sf·ΔV/t_r – krótkie impulsy o amplitudzie amperów.
### Wykresy
• lewy górny: 2 okresy nośnej – napięcia gałęzi (cienkie), v_cm (grube) i impulsy prądu i_cm,
• prawy górny: napięcie wału w całym okresie podstawowym z zaznaczonymi wyładowaniami EDM (czerwone ×),
• lewy dolny: impedancja fazy |Z(f)| – indukcyjna do rezonansu, potem pojemnościowa; tło pokazuje zakresy modeli,
• prawy dolny: widmo v_cm – prążki wokół wielokrotności f_sw.
💡 Skróć t_r → rosną impulsy i_cm (dv/dt!). Zmniejsz C_rf lub zwiększ C_sr → rośnie BVR i liczba wyładowań. Obniż próg V_th (zużyty smar) → więcej EDM.
⚙ Inżynier: środki zaradcze – filtr sinus/dU-dt lub filtr common-mode, szczotka uziemiająca wał, łożysko izolowane (ceramiczne kulki) od strony N, ekranowany kabel symetryczny, klatka Faradaya (ekran elektrostatyczny) w szczelinie zmniejszająca C_sr.
"""

    def calc(self, p):
        fsw, f1, m, Vdc = p["fsw"] * 1e3, p["f1"], p["m"], p["Vdc"]
        npc = 200
        N = int(round(fsw / f1 * npc))
        t = np.arange(N) / (fsw * npc)
        car = 2 * np.abs(2 * ((t * fsw) % 1.0) - 1) - 1              # nośna trójkątna ±1
        w1 = 2 * math.pi * f1
        legs = [Vdc / 2 * np.where(m * np.cos(w1 * t - k * TWO_PI_3) > car, 1.0, -1.0) for k in range(3)]
        vcm = (legs[0] + legs[1] + legs[2]) / 3
        BVR = p["Csr"] / (p["Csr"] + p["Crf"] + p["Cb"])
        rng = np.random.default_rng(7)
        vsh = np.zeros_like(vcm)
        v = 0.0
        ev_t, ev_v = [], []
        dv = np.diff(vcm, prepend=vcm[0]) * BVR
        th = p["Vth"] * (0.7 + 0.6 * rng.random())
        for k in range(N):
            v += dv[k]
            if abs(v) > th:
                ev_t.append(t[k]); ev_v.append(v)
                v = 0.0
                th = p["Vth"] * (0.7 + 0.6 * rng.random())
            vsh[k] = v
        # powiększenie: 2 okresy nośnej z realnym czasem narastania
        tr = p["tr"] * 1e-9
        tz = np.arange(0, 2 / fsw, tr / 8)
        carz = 2 * np.abs(2 * ((tz * fsw) % 1.0) - 1) - 1
        ph0 = 0.3
        legz = [Vdc / 2 * np.where(m * np.cos(ph0 - k * TWO_PI_3) > carz, 1.0, -1.0) for k in range(3)]
        nk = max(1, int(round(tr / (tr / 8))))
        ker = np.ones(nk) / nk
        legz = [np.convolve(x, ker, mode="same") for x in legz]
        vcmz = sum(legz) / 3
        icm = p["Csf"] * 1e-9 * np.gradient(vcmz, tz)
        fz = np.logspace(1, 7, 600)
        wz = 2 * np.pi * fz
        Zrl = p["R"] + 1j * wz * p["L"] * 1e-3
        Zc = 1 / (1j * wz * p["Csw"] * 1e-12)
        Zdm = Zrl * Zc / (Zrl + Zc)
        Zcm = 1 / (1j * wz * p["Csf"] * 1e-9)
        spec = np.abs(np.fft.rfft(vcm)) / N * 2
        fsp = np.fft.rfftfreq(N, t[1] - t[0])
        return dict(t=t, vsh=vsh, ev_t=np.array(ev_t), ev_v=np.array(ev_v), BVR=BVR, tz=tz,
                    legz=legz, vcmz=vcmz, icm=icm, fz=fz, Zdm=Zdm, Zcm=Zcm, spec=spec, fsp=fsp,
                    fres=1 / (2 * math.pi * math.sqrt(p["L"] * 1e-3 * p["Csw"] * 1e-12)))

    def draw(self, p, d):
        a0 = self.fig.add_subplot(2, 2, 1)
        tz = d["tz"] * 1e6
        for x, c in zip(d["legz"], [SKY, GREEN, PEACH]):
            a0.plot(tz, x, color=c, lw=0.8, alpha=0.6)
        a0.plot(tz, d["vcmz"], color=YELLOW, lw=2, label="v_cm")
        a0.set_xlabel("t [µs]"); a0.set_ylabel("[V]"); a0.set_title("Napięcie wspólne i prąd i_cm")
        b0 = a0.twinx()
        b0.plot(tz, d["icm"], color=RED, lw=1, label="i_cm")
        b0.set_ylabel("i_cm [A]", color=RED); b0.tick_params(axis="y", colors=RED); b0.grid(False)
        legend(a0, loc="upper left")
        a1 = self.fig.add_subplot(2, 2, 2)
        a1.plot(d["t"] * 1e3, d["vsh"], color=MAUVE, lw=0.8)
        if len(d["ev_t"]):
            a1.plot(d["ev_t"] * 1e3, d["ev_v"], "x", color=RED, ms=6, label=f"EDM: {len(d['ev_t'])}")
        a1.axhline(p["Vth"], color=SUBTEXT, ls=":"); a1.axhline(-p["Vth"], color=SUBTEXT, ls=":")
        a1.set_xlabel("t [ms]"); a1.set_ylabel("v_wału [V]"); a1.set_title(f"Napięcie wału (BVR = {d['BVR'] * 100:.2f}%)")
        legend(a1, loc="upper right")
        a2 = self.fig.add_subplot(2, 2, 3)
        a2.axvspan(10, 400, color=GREEN, alpha=0.12); a2.axvspan(400, 2e4, color=YELLOW, alpha=0.12)
        a2.axvspan(2e4, 1e7, color=RED, alpha=0.12)
        a2.loglog(d["fz"], np.abs(d["Zdm"]), color=SKY, label="|Z| fazy (R+jωL)‖C_sw")
        a2.loglog(d["fz"], np.abs(d["Zcm"]), color=PEACH, label="|Z| wspólna 1/(jωC_sf)")
        a2.axvline(d["fres"], color=TEXT, ls="--", lw=1)
        a2.set_xlabel("f [Hz]"); a2.set_ylabel("|Z| [Ω]"); a2.set_title("Impedancja vs częstotliwość")
        legend(a2, fontsize=7)
        a3 = self.fig.add_subplot(2, 2, 4)
        k = d["fsp"] <= 6 * p["fsw"] * 1e3
        a3.plot(d["fsp"][k] / 1e3, d["spec"][k], color=YELLOW, lw=0.8)
        a3.set_xlabel("f [kHz]"); a3.set_ylabel("|V_cm| [V]"); a3.set_title("Widmo napięcia wspólnego")
        dvdt = p["Vdc"] / (p["tr"] * 1e-9)
        return [f"BVR = C_sr/(C_sr+C_rf+C_b) = {fmt(d['BVR'] * 100)} %",
                f"v_cm max = V_dc/2 = {fmt(p['Vdc'] / 2)} V",
                f"skok v_wału na 1 łączenie = BVR·V_dc/3 = {fmt(d['BVR'] * p['Vdc'] / 3)} V",
                f"pełna huśtawka v_wału = BVR·V_dc = {fmt(d['BVR'] * p['Vdc'])} V",
                f"liczba wyładowań EDM / okres = {len(d['ev_t'])}",
                f"  → ok. {fmt(len(d['ev_t']) * p['f1'])} na sekundę",
                f"dv/dt gałęzi = {fmt(dvdt / 1e9)} kV/µs",
                f"i_cm szczyt. ≈ {fmt(np.max(np.abs(d['icm'])))} A",
                f"rezonans L–C_sw = {fmt(d['fres'] / 1e3)} kHz"]


# ═══════════════════════════════════════════════════════════════════════════
#  Zadania 7.1–7.7 (SRM z indukcyjnością sinusoidalną/trapezową)
# ═══════════════════════════════════════════════════════════════════════════
def trap(x, flat):
    """Okresowy „cosinus trapezowy”: płaskie części o szerokości flat·π wokół 0 i π."""
    x = (np.asarray(x) + np.pi) % (2 * np.pi) - np.pi
    a = flat * np.pi
    ax_ = np.abs(x)
    y = 1 - 2 * (ax_ - a) / (np.pi - 2 * a)
    return np.clip(y, -1, 1)


class TabProb7(BaseTab):
    TITLE = "Zadania 7.1–7.7"
    SLIDERS = [
        ("L0", "Indukcyjność średnia L0", 5.0, 50.0, 20.0, "mH"),
        ("L1", "Amplituda zmian L1", 1.0, 20.0, 10.0, "mH"),
        ("I", "Amplituda prądu fazy I", 1.0, 30.0, 10.0, "A"),
        ("phi", "Przesunięcie prądu φ", -90.0, 90.0, 45.0, "°"),
        ("flat", "Udział płaskich odcinków (trapez)", 0.0, 0.45, 0.25, "–"),
    ]
    EXPLAIN = """
## Zadania 7.1–7.7 – rozwiązania z komentarzem
### 7.1 Czy obcowzbudny silnik DC pasuje do modelu dq0?
TAK. Uzwojenie wzbudzenia leży w osi d stojana, twornik (ze szczotkami w strefie neutralnej) – w osi q wirnika, ω_b = 0. W uzwojeniu wzbudzenia nie ma SEM rotacji (ono stoi, a jego oś nie ma „prostopadłego partnera” wytwarzającego strumień w ruchu), a prostopadłość osi wyklucza sprzężenie transformatorowe. Dwa równania:
» V_F = R_F·I_F + L_F·dI_F/dt
» V_a = R_a·I_a + L_a·dI_a/dt + ω_r·L_dm·I_F
### 7.2 Wirnik z barierą strumienia
Model dq jest ścisły dla STAŁEJ szczeliny. Wirnik jawnobiegunowy zastępujemy cylindrycznym z cienką średnicową barierą (szczeliną „nadprzewodzącą” dla strumienia). Dla L_dm > L_qm bariera leży WZDŁUŻ osi d – przecina drogę strumienia osi q, więc L_qm maleje. Dla L_dm < L_qm bariera leży wzdłuż osi q i wypełniamy ją magnesami magnesowanymi w osi d (tak powstaje wirnik IPMSM). Zasada: bariera stoi w poprzek drogi strumienia, który ma być „trudny”.
### 7.3 PMSM – patrz zakładka „7.5 SM/PMSM”
Zamiast L_dm·I_F piszemy Ψ_PM, usuwamy równania klatki i wzbudzenia; T_e = 3/2·p1·[Ψ_PM·I_q + (L_d − L_q)·I_d·I_q].
### 7.4 Dwufazowa maszyna PM o różnych liczbach zwojów
Uzwojenia o RÓŻNEJ liczbie zwojów, ale tej samej masie miedzi, są po sprowadzeniu do wspólnej liczby zwojów symetryczne (rezystancja i indukcyjność skalują się z kwadratem przekładni, a przepływ ∝ N·I jest taki sam). Model dq w osiach wirnika (ω_b = ω_r) jest więc poprawny. Składowej zerowej NIE ma – dwie prostopadłe fazy to dokładnie osie α, β (składowa zerowa pojawia się dopiero przy 3 fazach).
Po usunięciu jednego uzwojenia stojana asymetria jest i w stojanie, i w wirniku (jawne bieguny) – żaden układ osi nie da stałych indukcyjności, potrzebny jest model fazowy. Natomiast dla wersji z symetryczną klatką (Rys. 7.11b) jedno uzwojenie stojana nie przeszkadza: model dq działa w osiach stojana (ω_b = 0) – tak liczy się silnik jednofazowy. Reguła: osie przyklejamy do części NIESYMETRYCZNEJ, druga część musi być symetryczna.
### 7.5 Dwufazowy silnik indukcyjny
TAK – to wprost model dq bez składowej zerowej. Dowolne ω_b (0, ω_r, ω1), bo obie strony są symetryczne.
### 7.6 PMSM 6 żłobków / 4 bieguny z uzwojeniem skupionym
TAK, w osiach wirnika (ω_b = ω_r), bo strumień magnesu skojarzony z cewkami jest sinusoidalny w funkcji kąta. Harmoniczna przepływu rzędu 2p1 = 4 tworzy moment; pozostałe harmoniczne nie dają momentu średniego i trafiają do indukcyjności rozproszenia.
### 7.7 Silnik reluktancyjny przełączalny (SRM) 6/4 – wykresy obok
Wzajemne indukcyjności ≈ 0, własne zmieniają się z kątem. Jeśli zmiany są SINUSOIDALNE, SRM jest w istocie synchroniczną maszyną reluktancyjną → model dq w osiach wirnika działa. Moment z energii:
» T = Σ ½·i_k²·dL_k/dθ
Z prądami sinusoidalnymi (3-fazowymi) moment jest STAŁY – brak tętnień (zielona linia, prawy dolny wykres). Przy indukcyjności trapezowej (liniowe narastanie + płaski odcinek) składniki się już nie znoszą → tętnienia momentu, a model dq NIE jest ścisły (indukcyjności nie są sinusoidalne) – trzeba stosować model we współrzędnych fazowych.
💡 Suwak „udział płaskich odcinków”: 0 = trójkąt, 0,45 = prawie prostokąt. Obserwuj, jak rosną tętnienia. Kąt φ zmienia moment średni (maksimum przy 45°, jak w maszynie reluktancyjnej sin 2φ).
⚙ Inżynier: prawdziwe SRM zasila się impulsami prądu (nie sinusoidą) i sterownik „profiluje” prąd, żeby zmniejszyć tętnienia i hałas – to temat na osobną zakładkę sterowania.
"""

    def calc(self, p):
        L0, L1, I, flat = p["L0"] * 1e-3, p["L1"] * 1e-3, p["I"], p["flat"]
        phi = math.radians(p["phi"])
        th = np.linspace(0, math.pi, 2401)                 # 0..180° mech. (2 okresy L)
        a = TWO_PI_3
        out = {}
        for key, shape in (("sin", np.cos), ("trap", lambda x: trap(x, flat))):
            T = np.zeros_like(th)
            Ls, Is = [], []
            for k in range(3):
                L = L0 + L1 * shape(4 * th - k * a)
                dL = np.gradient(L, th)
                i = I * np.cos(2 * th - k * a / 2 + phi)
                T += 0.5 * i ** 2 * dL
                Ls.append(L); Is.append(i)
            out[key] = (Ls, Is, T)
        return dict(th=np.degrees(th), out=out)

    def draw(self, p, d):
        th = d["th"]
        cols = [SKY, GREEN, PEACH]
        a0 = self.fig.add_subplot(2, 2, 1)
        for k in range(3):
            a0.plot(th, d["out"]["sin"][0][k] * 1e3, color=cols[k], label=f"L_{'ABC'[k]} sinus.")
            a0.plot(th, d["out"]["trap"][0][k] * 1e3, "--", color=cols[k], lw=1)
        a0.set_xlabel("θ_r [° mech.]"); a0.set_ylabel("[mH]"); a0.set_title("Indukcyjności faz SRM 6/4 (– – trapez)")
        legend(a0, fontsize=7, ncol=3)
        a1 = self.fig.add_subplot(2, 2, 2)
        for k in range(3):
            a1.plot(th, d["out"]["sin"][1][k], color=cols[k], label=f"i_{'ABC'[k]}")
        a1.set_xlabel("θ_r [° mech.]"); a1.set_ylabel("[A]"); a1.set_title("Prądy sinusoidalne")
        legend(a1, ncol=3, fontsize=7)
        a2 = self.fig.add_subplot(2, 2, 3)
        Ts, Tt = d["out"]["sin"][2], d["out"]["trap"][2]
        a2.plot(th, Ts, color=GREEN, lw=2, label="L sinusoidalne")
        a2.plot(th, Tt, color=RED, label="L trapezowe")
        a2.set_xlabel("θ_r [° mech.]"); a2.set_ylabel("T [N·m]"); a2.set_title("Moment T = Σ ½ i² dL/dθ")
        legend(a2)
        a3 = self.fig.add_subplot(2, 2, 4)
        rip = lambda T: np.ptp(T[5:-5]) / max(abs(np.mean(T)), 1e-9) * 100
        vals = [np.mean(Ts), np.mean(Tt)]
        a3.bar(["T śr. sinus", "T śr. trapez"], vals, color=[GREEN, RED])
        b3 = a3.twinx()
        b3.plot([0, 1], [rip(Ts), rip(Tt)], "D", color=YELLOW, ms=9)
        b3.set_ylabel("tętnienia [%]", color=YELLOW); b3.grid(False); b3.tick_params(axis="y", colors=YELLOW)
        a3.set_ylabel("T [N·m]"); a3.set_title("Moment średni i tętnienia (◆)")
        Tth = 1.5 * p["L1"] * 1e-3 * p["I"] ** 2 * math.sin(2 * math.radians(p["phi"]))
        return [f"T śr. (sinus)  = {fmt(vals[0])} N·m",
                f"  teoria 3/2·L1·I²·sin2φ = {fmt(Tth)} N·m",
                f"tętnienia (sinus)  = {fmt(rip(Ts))} %",
                f"T śr. (trapez) = {fmt(vals[1])} N·m",
                f"tętnienia (trapez) = {fmt(rip(Tt))} %",
                "",
                "Wniosek 7.7: sinusoidalne L → model dq",
                "OK i moment bez tętnień; trapezowe L →",
                "model fazowy + tętnienia momentu."]


# ═══════════════════════════════════════════════════════════════════════════
#  PRZYKŁAD 8.1 – generator obcowzbudny: skok V_F i nagłe zwarcie
# ═══════════════════════════════════════════════════════════════════════════
class TabEx81(BaseTab):
    TITLE = "Przykład 8.1"
    SLIDERS = [
        ("Ra", "Rezystancja twornika R_a", 0.01, 1.0, 0.1, "Ω"),
        ("RF", "Rezystancja wzbudzenia R_F", 0.2, 10.0, 1.0, "Ω"),
        ("La", "Indukcyjność twornika L_a", 0.05, 10.0, 0.5, "mH"),
        ("LF", "Indukcyjność wzbudzenia L_F", 0.05, 5.0, 0.5, "H"),
        ("Van", "Napięcie znamionowe V_an", 50.0, 500.0, 200.0, "V"),
        ("Ian", "Prąd znamionowy |I_an|", 10.0, 300.0, 100.0, "A"),
        ("IFn", "Prąd wzbudzenia I_Fn", 1.0, 20.0, 5.0, "A"),
        ("step", "Skok napięcia wzbudzenia ΔV_F", -50.0, 100.0, 20.0, "%"),
        ("tend", "Czas obserwacji", 0.5, 5.0, 3.0, "s"),
    ]
    EXPLAIN = """
## Przykład 8.1 – prądnica prądu stałego: skok wzbudzenia i zwarcie
Dane: R_a = 0,1 Ω, R_F = 1 Ω, L_a = 0,5 mH, L_F = 0,5 H, V_an = 200 V, I_an = 100 A (w książce −100 A, bo przyjęto konwencję silnikową – minus oznacza pracę prądnicową), I_Fn = 5 A, n = 1500 obr/min, obciążenie rezystancyjne.
### Po ludzku
Prądnica to „pompa elektryczna”: wirnik kręcony z zewnątrz przecina pole magnetyczne wytworzone przez prąd wzbudzenia I_F i indukuje SEM E. Prędkość jest stała (ciężki napęd, np. turbina), więc liczymy tylko zjawiska elektromagnetyczne – to są SZYBKIE stany nieustalone (8.3).
Co się dzieje, gdy podniesiemy napięcie wzbudzenia o 20%? Prąd wzbudzenia nie skoczy od razu – cewka L_F = 0,5 H działa jak bezwładność („koło zamachowe prądu”). Rośnie powoli ze stałą czasową T_F = L_F/R_F = 0,5 s. Za nim rośnie SEM i napięcie wyjściowe.
### Krok 1: rezystancja obciążenia i SEM
» R_load = V_an/|I_an| = 200/100 = 2 Ω
» E = V_an + R_a·|I_an| = 200 + 0,1·100 = 210 V
» ω_r·L_dm = E/I_Fn = 210/5 = 42 V/A       („ile woltów na amper wzbudzenia”)
» V_F0 = R_F·I_Fn = 5 V
### Krok 2: transmitancja (Równ. 8.9)
» V_a(s)/V_F(s) = ω_r·L_dm·R_load / [(R_F + s·L_F)·(R_a + R_load + s·L_a)]
» = 42·2 / [(1 + 0,5·s)·(2,1 + 0,0005·s)]
Dwa bieguny rzeczywiste ujemne: s1 = −R_F/L_F = −2 1/s oraz s2 = −(R_a+R_load)/L_a = −4200 1/s. Oba ujemne i rzeczywiste → odpowiedź stabilna i bez oscylacji (aperiodyczna).
### Krok 3: odpowiedź na skok ΔV_F = 0,2·5 = 1 V
» ΔV_a(∞) = 42·2/(1·2,1)·1 = 40 V   →   V_a: 200 → 240 V
» ΔI_a(∞) = 40/2 = 20 A              →   I_a: 100 → 120 A
» Δi_a(t) = 20·[1 − (T_F·e^(−t/T_F) − T_a·e^(−t/T_a))/(T_F − T_a)],  T_a = L_a/(R_a+R_load) = 0,238 ms
Ponieważ T_a jest 2000 razy mniejsze od T_F, praktycznie Δi_a(t) ≈ 20·(1 − e^(−t/0,5 s)). Po 3·T_F = 1,5 s osiągamy 95% zmiany.
### Krok 4: nagłe zwarcie zacisków (V_a = 0, I_F = I_Fn)
SEM zostaje (strumień wzbudzenia się nie zmienia), a prąd ogranicza TYLKO mała rezystancja twornika:
» L_a·di/dt + R_a·i = E    →    i(t) = E/R_a + (I_n − E/R_a)·e^(−t/T_e)
» i(∞) = 210/0,1 = 2100 A = 21·I_n (!),   T_e = L_a/R_a = 5 ms
W 15 ms (3·T_e) prąd rośnie do ponad 2000 A – dlatego zwarcie jest niebezpieczne i wymaga zabezpieczeń działających w milisekundach (wyłączniki szybkie).
### Wykresy
• lewy górny: prąd wzbudzenia i_F(t) – powolny wykładniczy wzrost,
• prawy górny: napięcie zacisków V_a(t),  • lewy dolny: prąd twornika i_a(t),
• prawy dolny: prąd zwarcia w milisekundach.
💡 Zwiększ R_F przy stałym L_F – T_F maleje, odpowiedź przyspiesza. Dlatego w praktyce stosuje się „forsowanie wzbudzenia” – chwilowo dużo wyższe V_F.
⚙ Inżynier: ta sama logika (wolny obwód strumienia, szybki obwód momentu) doprowadziła do sterowania polowo-zorientowanego maszyn AC: strumień trzymamy stały, a momentem sterujemy szybkim prądem.
"""

    def calc(self, p):
        Ra, RF, La, LF = p["Ra"], p["RF"], p["La"] * 1e-3, p["LF"]
        Rl = p["Van"] / p["Ian"]
        E = p["Van"] + Ra * p["Ian"]
        K = E / p["IFn"]
        VF0 = RF * p["IFn"]
        dVF = p["step"] / 100 * VF0
        TF, Ta = LF / RF, La / (Ra + Rl)
        t = np.linspace(0, p["tend"], 3000)
        diF = dVF / RF * (1 - np.exp(-t / TF))
        dIa_inf = K * dVF / (RF * (Ra + Rl))
        if abs(TF - Ta) > 1e-12:
            dia = dIa_inf * (1 - (TF * np.exp(-t / TF) - Ta * np.exp(-t / Ta)) / (TF - Ta))
        else:
            dia = dIa_inf * (1 - (1 + t / TF) * np.exp(-t / TF))
        ia = p["Ian"] + dia
        Te = La / Ra
        tsc = np.linspace(0, 6 * Te, 800)
        Isc = E / Ra
        isc = Isc + (p["Ian"] - Isc) * np.exp(-tsc / Te)
        return dict(t=t, iF=p["IFn"] + diF, ia=ia, Va=Rl * ia, tsc=tsc, isc=isc, Rl=Rl, E=E, K=K,
                    VF0=VF0, dVF=dVF, TF=TF, Ta=Ta, Te=Te, Isc=Isc, dIa=dIa_inf)

    def draw(self, p, d):
        t = d["t"]
        a0 = self.fig.add_subplot(2, 2, 1)
        a0.plot(t, d["iF"], color=MAUVE)
        a0.axvline(d["TF"], color=SUBTEXT, ls=":", lw=1)
        a0.text(d["TF"], a0.get_ylim()[0], " T_F", color=SUBTEXT, va="bottom")
        a0.set_xlabel("t [s]"); a0.set_ylabel("i_F [A]"); a0.set_title("Prąd wzbudzenia po skoku V_F")
        a1 = self.fig.add_subplot(2, 2, 2)
        a1.plot(t, d["Va"], color=SKY); a1.set_xlabel("t [s]"); a1.set_ylabel("V_a [V]")
        a1.set_title("Napięcie zacisków V_a(t)")
        a2 = self.fig.add_subplot(2, 2, 3)
        a2.plot(t, d["ia"], color=GREEN); a2.set_xlabel("t [s]"); a2.set_ylabel("i_a [A]")
        a2.set_title("Prąd obciążenia i_a(t)")
        a3 = self.fig.add_subplot(2, 2, 4)
        a3.plot(d["tsc"] * 1e3, d["isc"], color=RED, lw=2, label="i_zw(t)")
        a3.axhline(p["Ian"], color=SUBTEXT, ls=":", label="I_n")
        a3.axhline(d["Isc"], color=YELLOW, ls="--", lw=1, label="E/R_a")
        a3.set_xlabel("t [ms]"); a3.set_ylabel("[A]"); a3.set_title("Nagłe zwarcie zacisków")
        legend(a3, loc="center right")
        num = d["K"] * d["Rl"]
        return ["R_load = V_an/I_an   = " + fmt(d["Rl"]) + " Ω",
                "E = V_an + R_a·I_an  = " + fmt(d["E"]) + " V",
                "ω_r·L_dm = E/I_Fn    = " + fmt(d["K"]) + " V/A",
                "V_F0 = R_F·I_Fn      = " + fmt(d["VF0"]) + " V,  ΔV_F = " + fmt(d["dVF"]) + " V",
                "",
                "V_a/V_F = " + fmt(num) + " /",
                f"  [({fmt(p['RF'])} + {fmt(p['LF'])}s)({fmt(p['Ra'] + d['Rl'])} + {fmt(p['La'] * 1e-3)}s)]",
                "bieguny: " + fmt(-1 / d["TF"]) + " ; " + fmt(-1 / d["Ta"]) + " 1/s",
                "T_F = " + fmt(d["TF"]) + " s,  T_a = " + fmt(d["Ta"] * 1e3) + " ms",
                "ΔV_a(∞) = " + fmt(d["dIa"] * d["Rl"]) + " V,  ΔI_a(∞) = " + fmt(d["dIa"]) + " A",
                "",
                "Zwarcie: T_e = L_a/R_a = " + fmt(d["Te"] * 1e3) + " ms",
                "I_zw(∞) = E/R_a = " + fmt(d["Isc"]) + " A = " + fmt(d["Isc"] / p["Ian"]) + "·I_n"]


# ═══════════════════════════════════════════════════════════════════════════
#  PRZYKŁAD 8.2 – silnik DC PM: skok napięcia 12 → 10 V, dwie bezwładności
# ═══════════════════════════════════════════════════════════════════════════
SCEN82 = ["Skok napięcia V_a (stały T_L)", "Skok momentu obciążenia (stałe V_a)"]


def pm_motor(Va, TL, Ra, La, psi, p1, J, w0, i0, t_end, n=1500):
    """Równ. 8.16: L_a di/dt = V_a − R_a i − ω_r Ψ ;  (J/p1) dω_r/dt = p1 Ψ i − T_L."""
    from scipy.integrate import solve_ivp

    def f(_t, x):
        i, w = x
        return [(Va - Ra * i - w * psi) / La, p1 / J * (p1 * psi * i - TL)]
    t = np.linspace(0, t_end, n)
    s = solve_ivp(f, (0, t_end), [i0, w0], t_eval=t, method="Radau", rtol=1e-8, atol=1e-9)
    return s.t, s.y[0], s.y[1]


class TabEx82(BaseTab):
    TITLE = "Przykład 8.2"
    SLIDERS = [
        ("Pn", "Moc znamionowa P_n", 10.0, 500.0, 50.0, "W"),
        ("Vn", "Napięcie znamionowe V_n", 6.0, 48.0, 12.0, "V"),
        ("eta", "Sprawność η_n", 0.5, 0.98, 0.9, "–"),
        ("Ra", "Rezystancja twornika R_a", 0.02, 1.0, 0.12, "Ω"),
        ("Te", "Elektryczna stała czasowa T_e", 0.2, 10.0, 2.0, "ms"),
        ("p1", "Pary biegunów p1", 1, 4, 1, "–", 1),
        ("nn", "Prędkość znamionowa n_n", 500.0, 5000.0, 1500.0, "obr/min"),
        ("V2", "Nowe napięcie V_a (scenariusz 1)", 0.0, 24.0, 10.0, "V"),
        ("dT", "Zmiana T_L (scenariusz 2)", -50.0, 100.0, 20.0, "%"),
        ("J1", "Bezwładność J1 (×10⁻⁴)", 0.5, 50.0, 2.0, "kg·m²"),
        ("J2", "Bezwładność J2 (×10⁻⁴)", 0.5, 50.0, 10.0, "kg·m²"),
        ("tend", "Czas obserwacji", 20.0, 500.0, 100.0, "ms"),
    ]
    EXPLAIN = """
## Przykład 8.2 – silnik DC z magnesami: nagłe obniżenie napięcia
Dane: P_n = 50 W, V_n = 12 V, η_n = 0,9, R_a = 0,12 Ω, T_e = 2 ms, p1 = 1, n_n = 1500 obr/min; dwie bezwładności J = 2·10⁻⁴ i J' = 10⁻³ kg·m²; napięcie spada z 12 do 10 V przy stałym momencie obciążenia.
### Po ludzku
Mały silniczek (np. wentylator w samochodzie). Obniżamy napięcie – silnik zwolni. Pytanie: czy zwolni „gładko”, czy z przeregulowaniem (spadnie poniżej nowej prędkości i wróci, jak samochód hamujący z zawieszeniem na sprężynach)? Odpowiedź zależy od stosunku dwóch stałych czasowych.
### Krok 1: stan znamionowy (d/dt = 0)
» I_an = P_n/(η_n·V_n) = 50/(0,9·12) = 4,63 A
» E = V_n − R_a·I_an = 12 − 0,12·4,63 = 11,44 V
» ω_r = 2π·1500/60 = 157,08 rad/s
» Ψ_PM = E/ω_r = 0,0729 Wb      T_e = p1·Ψ_PM·I_an = 0,337 N·m
» L_a = T_e·R_a = 0,002·0,12 = 0,24 mH
### Krok 2: model (Równ. 8.16–8.21)
» L_a·di_a/dt = V_a − R_a·i_a − ω_r·Ψ_PM
» (J/p1)·dω_r/dt = p1·Ψ_PM·i_a − T_L
Po wyeliminowaniu prądu dostajemy równanie 2. rzędu z dwiema stałymi czasowymi:
» T_e·T_em·d²ω/dt² + T_em·dω/dt + ω = (V_a − R_a·T_L/(p1Ψ))/Ψ
» T_e = L_a/R_a (elektryczna),   T_em = J·R_a/(p1·Ψ_PM)² (elektromechaniczna)
» bieguny: s1,2 = [−1 ± √(1 − 4T_e/T_em)]/(2T_e)
### Krok 3: dwie bezwładności
• J = 2·10⁻⁴: T_em = 4,52 ms < 4·T_e = 8 ms → pierwiastek z liczby ujemnej → bieguny ZESPOLONE → oscylacje tłumione (przeregulowanie prędkości i prądu),
• J' = 10⁻³: T_em = 22,6 ms > 8 ms → bieguny rzeczywiste → odpowiedź aperiodyczna (gładka, wolniejsza).
### Krok 4: stan końcowy
Moment obciążenia stały → prąd końcowy taki sam jak początkowy: i_a(∞) = 4,63 A.
» ω_rf = (V_a − R_a·I_a)/Ψ_PM = (10 − 0,556)/0,0729 = 129,6 rad/s    (książka: 129,55)
Czyli prędkość spada o ok. 17,5% (z 157 do 130 rad/s) – prawie proporcjonalnie do napięcia.
Warunki początkowe (Równ. 8.22): ω(0) = 157 rad/s, dω/dt(0) = 0 (prąd nie zmienia się skokowo, bo jest cewka). W chwili skoku prąd zaczyna gwałtownie spadać (a nawet zmienia znak – silnik chwilowo hamuje prądnicowo!).
### Wykresy
• lewy górny: ω_r(t) dla J i J' (Rys. 8.4a),  • prawy górny: i_a(t) (Rys. 8.4b),
• lewy dolny: bieguny na płaszczyźnie zespolonej – na osi rzeczywistej brak oscylacji,
• prawy dolny: współczynnik tłumienia ζ = ½·√(T_em/T_e) w funkcji J; ζ < 1 → oscylacje.
💡 Scenariusz 2 (uwaga w książce): skok momentu przy stałym napięciu. Teraz di_a/dt(0) = 0, a nie dω/dt(0) = 0. Prędkość spada o R_a·ΔT_L/(p1Ψ)² – mało, bo silnik PM ma „sztywną” charakterystykę.
⚙ Inżynier: silnik PM ma wbudowane sprzężenie zwrotne (SEM ∝ prędkości) – dlatego zawsze jest stabilny w pętli otwartej. Małe J (np. twornik bezżłobkowy) daje najszybszą odpowiedź momentu, ale z przeregulowaniem – tę rolę przejmuje regulator (zakładka 8.5).
"""

    def build_extra(self, parent):
        self.add_combo(parent, "scen", "Scenariusz:", SCEN82, 0)

    def calc(self, p):
        Ra, p1 = p["Ra"], p["p1"]
        Ia = p["Pn"] / (p["eta"] * p["Vn"])
        E = p["Vn"] - Ra * Ia
        w0 = p1 * 2 * math.pi * p["nn"] / 60
        psi = E / w0
        T0 = p1 * psi * Ia
        La = p["Te"] * 1e-3 * Ra
        if p["scen"] == SCEN82[0]:
            Va, TL = p["V2"], T0
        else:
            Va, TL = p["Vn"], T0 * (1 + p["dT"] / 100)
        res = []
        for J in (p["J1"] * 1e-4, p["J2"] * 1e-4):
            t, i, w = pm_motor(Va, TL, Ra, La, psi, p1, J, w0, Ia, p["tend"] * 1e-3)
            Tem = J * Ra / (p1 * psi) ** 2
            disc = complex(1 - 4 * p["Te"] * 1e-3 / Tem)
            s1 = (-1 + disc ** 0.5) / (2 * p["Te"] * 1e-3)
            s2 = (-1 - disc ** 0.5) / (2 * p["Te"] * 1e-3)
            res.append(dict(J=J, t=t, i=i, w=w, Tem=Tem, s=(s1, s2)))
        Jg = np.logspace(-5, -2, 200)
        zeta = 0.5 * np.sqrt(Jg * Ra / (p1 * psi) ** 2 / (p["Te"] * 1e-3))
        iaf = TL / (p1 * psi)
        return dict(Ia=Ia, E=E, w0=w0, psi=psi, T0=T0, La=La, Va=Va, TL=TL, res=res, Jg=Jg,
                    zeta=zeta, iaf=iaf, wf=(Va - Ra * iaf) / psi)

    def draw(self, p, d):
        cols = [PEACH, SKY]
        a0 = self.fig.add_subplot(2, 2, 1)
        a1 = self.fig.add_subplot(2, 2, 2)
        a2 = self.fig.add_subplot(2, 2, 3)
        for r, c in zip(d["res"], cols):
            lab = f"J = {r['J']:.1e} kg·m²"
            a0.plot(r["t"] * 1e3, r["w"], color=c, label=lab)
            a1.plot(r["t"] * 1e3, r["i"], color=c, label=lab)
            a2.plot([s.real for s in r["s"]], [s.imag for s in r["s"]], "x", ms=10, mew=2.5, color=c, label=lab)
        a0.axhline(d["wf"], color=SUBTEXT, ls=":", lw=1)
        a0.set_xlabel("t [ms]"); a0.set_ylabel("ω_r [rad/s]"); a0.set_title("Prędkość (Rys. 8.4a)")
        legend(a0)
        a1.axhline(d["iaf"], color=SUBTEXT, ls=":", lw=1)
        a1.set_xlabel("t [ms]"); a1.set_ylabel("i_a [A]"); a1.set_title("Prąd twornika (Rys. 8.4b)")
        legend(a1)
        a2.axvline(0, color=RED, lw=1)
        a2.set_xlabel("Re s [1/s]"); a2.set_ylabel("Im s [1/s]"); a2.set_title("Bieguny (wartości własne)")
        legend(a2, fontsize=7)
        a3 = self.fig.add_subplot(2, 2, 4)
        a3.semilogx(d["Jg"], d["zeta"], color=GREEN)
        a3.axhline(1, color=RED, ls="--", lw=1)
        a3.fill_between(d["Jg"], 0, 1, color=RED, alpha=0.1)
        a3.text(d["Jg"][3], 0.5, "oscylacje (T_em < 4T_e)", color=RED, fontsize=8)
        for r, c in zip(d["res"], cols):
            z = 0.5 * math.sqrt(r["Tem"] / (p["Te"] * 1e-3))
            a3.plot([r["J"]], [z], "o", color=c, ms=8)
        a3.set_xlabel("J [kg·m²]"); a3.set_ylabel("ζ"); a3.set_title("Współczynnik tłumienia ζ(J)")
        out = [f"I_an = P/(ηV) = {fmt(d['Ia'])} A",
               f"E = {fmt(d['E'])} V,  ω_r0 = {fmt(d['w0'])} rad/s",
               f"Ψ_PM = {fmt(d['psi'])} Wb,  T_e = {fmt(d['T0'])} N·m",
               f"L_a = T_e·R_a = {fmt(d['La'] * 1e3)} mH",
               f"V_a = {fmt(d['Va'])} V,  T_L = {fmt(d['TL'])} N·m",
               f"ω_rf = {fmt(d['wf'])} rad/s,  i_af = {fmt(d['iaf'])} A",
               f"4·T_e = {fmt(4 * p['Te'])} ms"]
        for r in d["res"]:
            kind = "oscylacyjna" if r["Tem"] < 4 * p["Te"] * 1e-3 else "aperiodyczna"
            out += [f"J={r['J']:.1e}: T_em={fmt(r['Tem'] * 1e3)} ms → {kind}",
                    f"   s1,2 = {fmt(r['s'][0], 4)}",
                    f"   i_a min/max = {fmt(np.min(r['i']))} / {fmt(np.max(r['i']))} A"]
        return out


# ═══════════════════════════════════════════════════════════════════════════
#  8.4.2 Silnik obcowzbudny – zmiana strumienia (osłabianie pola)
# ═══════════════════════════════════════════════════════════════════════════
class TabVarFlux(BaseTab):
    TITLE = "8.4.2 Zmienny strumień"
    SLIDERS = [
        ("Va", "Napięcie twornika V_a", 50.0, 440.0, 220.0, "V"),
        ("Ra", "R_a", 0.05, 3.0, 0.6, "Ω"),
        ("La", "L_a", 1.0, 100.0, 12.0, "mH"),
        ("RF", "R_F", 20.0, 300.0, 110.0, "Ω"),
        ("LF", "L_F", 1.0, 100.0, 22.0, "H"),
        ("VF", "Napięcie wzbudzenia V_F0", 50.0, 440.0, 220.0, "V"),
        ("Ldm", "Indukcyjność wzajemna L_dm", 0.05, 2.0, 0.4, "H"),
        ("p1", "Pary biegunów p1", 1, 4, 2, "–", 1),
        ("J", "Bezwładność J", 0.02, 3.0, 0.3, "kg·m²"),
        ("TL", "Moment obciążenia T_L", 0.0, 60.0, 20.0, "N·m"),
        ("step", "Skok V_F (−: osłabianie)", -60.0, 50.0, -20.0, "%"),
        ("tend", "Czas symulacji", 0.5, 5.0, 2.0, "s"),
    ]
    EXPLAIN = """
## 8.4.2 Zmiana strumienia – osłabianie pola
Chcemy, by silnik kręcił się SZYBCIEJ niż znamionowo, ale napięcie twornika jest już maksymalne. Sztuczka: osłabiamy pole (zmniejszamy prąd wzbudzenia I_F). Wtedy do wytworzenia tej samej SEM potrzebna jest większa prędkość: E = ω_r·L_dm·I_F.
### Po ludzku
To jak jazda na rowerze na lżejszym przełożeniu: łatwiej kręcić szybko, ale „siła” (moment na amper) jest mniejsza.
### Model (Równ. 8.27–8.28) – układ nieliniowy 3. rzędu
» L_F·dI_F/dt = V_F − R_F·I_F
» L_a·dI_a/dt = V_a − R_a·I_a − ω_r·L_dm·I_F
» (J/p1)·dω_r/dt = p1·L_dm·I_F·I_a − T_L
Iloczyny zmiennych (I_F·I_a, ω_r·I_F) czynią układ NIELINIOWYM. Inżynier linearyzuje go wokół punktu pracy (teoria małych odchyleń, Równ. 8.29–8.31): x = x0 + Δx i odrzuca iloczyny małych Δ.
### Wartości własne (Równ. 8.32)
» γ3 = −R_F/L_F         (obwód wzbudzenia – oddzielnie, bo osie d i q są prostopadłe!)
» γ1,2 – jak dla stałego strumienia (Przykład 8.2), ale policzone dla I_F0
Obwód wzbudzenia ma dużą stałą czasową (tu 0,2 s), więc zmiany strumienia są wolne.
### Wykresy
• lewy górny: I_F(t) – wolny wykładniczy spadek,
• prawy górny: I_a(t) – ciekawe! Gdy strumień maleje, SEM spada, więc prąd twornika gwałtownie ROŚNIE (niebezpieczny udar prądu), bo V_a − E rośnie. Linia przerywana – model zlinearyzowany,
• lewy dolny: prędkość rośnie do nowej, wyższej wartości,
• prawy dolny: wartości własne na płaszczyźnie zespolonej.
💡 Zwiększ skok do −50%: model liniowy (przerywany) zaczyna się wyraźnie różnić od nieliniowego – linearyzacja jest dobra tylko dla MAŁYCH zmian. Zmniejsz L_F – strumień zmienia się szybciej, udar prądu większy.
⚙ Inżynier: w praktyce osłabianie pola robi się powoli z ograniczeniem prądu twornika. Analogicznie w silnikach AC ze sterowaniem wektorowym: prąd i_d (strumień) zmieniamy wolno, a i_q (moment) szybko. To podobieństwo dało początek sterowaniu polowo-zorientowanemu, które zrewolucjonizowało napędy.
"""

    def calc(self, p):
        from scipy.integrate import solve_ivp
        Ra, La, RF, LF, Ldm, p1, J = p["Ra"], p["La"] * 1e-3, p["RF"], p["LF"], p["Ldm"], p["p1"], p["J"]
        IF0 = p["VF"] / RF
        ia0 = p["TL"] / (p1 * Ldm * IF0)
        w0 = (p["Va"] - Ra * ia0) / (Ldm * IF0)
        VF1 = p["VF"] * (1 + p["step"] / 100)
        t0 = 0.05 * p["tend"]

        def vf(t):
            return VF1 if t >= t0 else p["VF"]

        def f(t, x):
            iF, ia, w = x
            return [(vf(t) - RF * iF) / LF, (p["Va"] - Ra * ia - w * Ldm * iF) / La,
                    p1 / J * (p1 * Ldm * iF * ia - p["TL"])]
        t = np.linspace(0, p["tend"], 2500)
        s = solve_ivp(f, (0, p["tend"]), [IF0, ia0, w0], t_eval=t, method="LSODA", rtol=1e-7,
                      atol=1e-8, max_step=p["tend"] / 500)
        A = np.array([[-RF / LF, 0, 0],
                      [-Ldm * w0 / La, -Ra / La, -Ldm * IF0 / La],
                      [p1 * p1 * Ldm * ia0 / J, p1 * p1 * Ldm * IF0 / J, 0]])
        B = np.array([1 / LF, 0, 0])
        ev = np.linalg.eigvals(A)

        def fl(tt, x):
            return A @ x + B * (VF1 - p["VF"] if tt >= t0 else 0.0)
        sl = solve_ivp(fl, (0, p["tend"]), [0, 0, 0], t_eval=t, method="LSODA", rtol=1e-7,
                       atol=1e-9, max_step=p["tend"] / 500)
        IF1 = VF1 / RF
        ia1 = p["TL"] / (p1 * Ldm * IF1)
        w1 = (p["Va"] - Ra * ia1) / (Ldm * IF1)
        return dict(t=s.t, X=s.y, XL=sl.y, IF0=IF0, ia0=ia0, w0=w0, ev=ev, IF1=IF1, ia1=ia1, w1=w1)

    def draw(self, p, d):
        t = d["t"]
        k = 60 / (2 * math.pi) / p["p1"]
        a0 = self.fig.add_subplot(2, 2, 1)
        a0.plot(t, d["X"][0], color=MAUVE); a0.set_title("Prąd wzbudzenia I_F"); a0.set_xlabel("t [s]"); a0.set_ylabel("[A]")
        a1 = self.fig.add_subplot(2, 2, 2)
        a1.plot(t, d["X"][1], color=GREEN, label="nieliniowy")
        a1.plot(t, d["ia0"] + d["XL"][1], "--", color=YELLOW, label="zlinearyzowany")
        a1.set_title("Prąd twornika I_a"); a1.set_xlabel("t [s]"); a1.set_ylabel("[A]"); legend(a1)
        a2 = self.fig.add_subplot(2, 2, 3)
        a2.plot(t, d["X"][2] * k, color=SKY, label="nieliniowy")
        a2.plot(t, (d["w0"] + d["XL"][2]) * k, "--", color=YELLOW, label="zlinearyzowany")
        a2.set_title("Prędkość n"); a2.set_xlabel("t [s]"); a2.set_ylabel("[obr/min]"); legend(a2)
        a3 = self.fig.add_subplot(2, 2, 4)
        evs = sorted(d["ev"], key=lambda z: abs(z.real + p["RF"] / p["LF"]))
        a3.plot([evs[0].real], [evs[0].imag], "s", ms=10, color=MAUVE, label="γ3 = −R_F/L_F")
        a3.plot([z.real for z in evs[1:]], [z.imag for z in evs[1:]], "x", ms=10, mew=2.5, color=PEACH, label="γ1,2")
        a3.axvline(0, color=RED, lw=1)
        a3.set_xscale("symlog", linthresh=1)
        a3.set_title("Wartości własne (Równ. 8.32)"); a3.set_xlabel("Re [1/s] (symlog)"); a3.set_ylabel("Im [1/s]")
        legend(a3, fontsize=7)
        return [f"Punkt pracy: I_F0 = {fmt(d['IF0'])} A",
                f"  I_a0 = {fmt(d['ia0'])} A,  n0 = {fmt(d['w0'] * k)} obr/min",
                f"Po zmianie: I_F1 = {fmt(d['IF1'])} A",
                f"  I_a1 = {fmt(d['ia1'])} A,  n1 = {fmt(d['w1'] * k)} obr/min",
                f"I_a max w stanie przejść. = {fmt(np.max(d['X'][1]))} A",
                f"T_F = L_F/R_F = {fmt(p['LF'] / p['RF'])} s",
                "Wartości własne:"] + [f"  {fmt(z)} 1/s" for z in d["ev"]]


# ═══════════════════════════════════════════════════════════════════════════
#  8.4.3 Silnik szeregowy DC – Zadanie 8.4
# ═══════════════════════════════════════════════════════════════════════════
class TabSeries(BaseTab):
    TITLE = "8.4.3 Szeregowy (Zad. 8.4)"
    SLIDERS = [
        ("V", "Napięcie zasilania V", 100.0, 800.0, 500.0, "V"),
        ("Ra", "R_a", 0.1, 3.0, 1.0, "Ω"),
        ("RF", "R_Fs (wzbudzenie szeregowe)", 0.05, 2.0, 0.5, "Ω"),
        ("La", "L_a", 1.0, 100.0, 10.0, "mH"),
        ("LF", "L_Fs", 0.05, 2.0, 0.5, "H"),
        ("p1", "Pary biegunów p1", 1, 4, 2, "–", 1),
        ("nn", "Prędkość znamionowa n_n", 300.0, 3000.0, 1500.0, "obr/min"),
        ("In", "Prąd znamionowy I_an", 20.0, 300.0, 100.0, "A"),
        ("n2", "Druga prędkość do analizy", 200.0, 3000.0, 750.0, "obr/min"),
        ("J", "Bezwładność J (założona)", 0.1, 20.0, 2.0, "kg·m²"),
        ("step", "Zmiana momentu obciążenia", -50.0, 100.0, 20.0, "%"),
        ("tend", "Czas symulacji", 0.5, 10.0, 3.0, "s"),
    ]
    EXPLAIN = """
## 8.4.3 Silnik szeregowy – Zadanie 8.4
Dane: R_a = 2·R_F = 1 Ω (czyli R_Fs = 0,5 Ω), L_a = 10 mH, L_Fs = 0,5 H, p1 = 2, n_n = 1500 obr/min, V = 500 V, I_a = 100 A. Straty poza miedzią pomijamy. Moment bezwładności nie jest podany – przyjmujemy J = 2 kg·m² (suwak).
### Po ludzku
W silniku szeregowym TEN SAM prąd płynie przez twornik i przez wzbudzenie. Im większe obciążenie, tym większy prąd, tym silniejsze pole i tym większy moment (∝ I²!). To idealny silnik trakcyjny – tramwaj ruszający pod górę dostaje ogromny moment. Wada: bez obciążenia silnik „ucieka” (rozbiegnie się), bo słabe pole → bardzo duża prędkość.
### Krok 1: stan znamionowy
» E = V − (R_a + R_Fs)·I_a = 500 − 1,5·100 = 350 V
» ω_r = p1·2π·n/60 = 2·157,08 = 314,16 rad/s (elektryczne)
» L_dm = E/(ω_r·I_a) = 350/(314,16·100) = 11,14 mH
» T_e = p1·L_dm·I_a² = 2·0,01114·100² = 222,8 N·m     (sprawdzenie: E·I/Ω_mech = 35000/157,08 ✓)
### Krok 2: linearyzacja (Równ. 8.33–8.36)
» (L_a + L_Fs)·di/dt = V − (R_a + R_Fs)·i − ω_r·L_dm·i
» (J/p1)·dω_r/dt = p1·L_dm·i² − T_L
Małe odchylenia wokół (I0, ω0):
» [(L_a+L_Fs)·s + R_a + R_Fs + ω0·L_dm]·Δi + L_dm·I0·Δω = ΔV
» (J/p1)·s·Δω = 2·p1·L_dm·I0·Δi − ΔT_L
Równanie charakterystyczne:
» (L·s + R + ω0·L_dm)·(J/p1)·s + 2·p1·L_dm²·I0² = 0
Zastępcza elektryczna stała czasowa (Równ. 8.38):
» T_es = (L_a + L_Fs)/(R_a + R_Fs + ω0·L_dm)
Zauważ: SEM rotacji działa jak dodatkowa rezystancja ω0·L_dm – dlatego T_es < (L_a+L_Fs)/(R_a+R_Fs), a odpowiedź jest szybsza niż „prosta” stała L/R.
### Krok 3: 1500 i 750 obr/min
Przy 750 obr/min (V = 500 V, bez nasycenia) prąd jest większy: I0 = V/(R + ω0·L_dm) = 500/(1,5 + 1,75) = 153,8 A. Wartości własne zmieniają się z prędkością (prawy dolny wykres).
### Krok 4: wzrost obciążenia o 20%
Nowy prąd: I = √(T_L/(p1·L_dm)) = √(1,2)·100 = 109,5 A, prędkość spada. Symulacja nieliniowa (ciągła) vs zlinearyzowana (przerywana).
### Wykresy
• lewy górny: prąd,  • prawy górny: prędkość,
• lewy dolny: charakterystyka mechaniczna T(n) – „hiperbola” silnika szeregowego i linie obciążenia,
• prawy dolny: części rzeczywiste wartości własnych i T_es w funkcji prędkości.
⚙ Inżynier: w rzeczywistości L_dm(i) zależy od prądu (nasycenie jest „nieuniknione”, bo prąd twornika to prąd wzbudzenia) – wtedy T_es zmienia się z prędkością jeszcze silniej. Silnik szeregowy nie może pracować bez obciążenia (np. przy zerwanym pasku) – zabezpieczenie nadobrotowe jest obowiązkowe.
"""

    def calc(self, p):
        from scipy.integrate import solve_ivp
        R, L = p["Ra"] + p["RF"], p["La"] * 1e-3 + p["LF"]
        p1, J, V = p["p1"], p["J"], p["V"]
        wn = p1 * 2 * math.pi * p["nn"] / 60
        E = V - R * p["In"]
        if E <= 0:
            raise ValueError("E ≤ 0: za duży spadek napięcia na rezystancjach")
        Ldm = E / (wn * p["In"])
        Tn = p1 * Ldm * p["In"] ** 2

        def eig(w0):
            I0 = V / (R + w0 * Ldm)
            a2, a1, a0 = L * J / p1, (R + w0 * Ldm) * J / p1, 2 * p1 * Ldm ** 2 * I0 ** 2
            return np.roots([a2, a1, a0]), L / (R + w0 * Ldm), I0
        e1, Tes1, _ = eig(wn)
        w2 = p1 * 2 * math.pi * p["n2"] / 60
        e2, Tes2, I2 = eig(w2)
        TL1 = Tn * (1 + p["step"] / 100)
        t0 = 0.1

        def f(t, x):
            i, w = x
            TL = TL1 if t >= t0 else Tn
            return [(V - R * i - w * Ldm * i) / L, p1 / J * (p1 * Ldm * i * i - TL)]
        t = np.linspace(0, p["tend"], 2000)
        s = solve_ivp(f, (0, p["tend"]), [p["In"], wn], t_eval=t, method="LSODA", rtol=1e-8, atol=1e-8,
                      max_step=p["tend"] / 400)
        A = np.array([[-(R + wn * Ldm) / L, -Ldm * p["In"] / L], [2 * p1 * p1 * Ldm * p["In"] / J, 0]])

        def fl(tt, x):
            return A @ x + np.array([0, -p1 / J * (TL1 - Tn) if tt >= t0 else 0.0])
        sl = solve_ivp(fl, (0, p["tend"]), [0, 0], t_eval=t, method="LSODA", rtol=1e-8, atol=1e-9,
                       max_step=p["tend"] / 400)
        ns = np.linspace(100, 3000, 150)
        ws = p1 * 2 * math.pi * ns / 60
        evs = np.array([eig(w)[0] for w in ws])
        tes = L / (R + ws * Ldm)
        Tchar = p1 * Ldm * (V / (R + ws * Ldm)) ** 2
        I1 = math.sqrt(TL1 / (p1 * Ldm))
        n1 = (V - R * I1) / (Ldm * I1) / p1 * 60 / (2 * math.pi)
        return dict(t=s.t, X=s.y, XL=sl.y, Ldm=Ldm, E=E, Tn=Tn, TL1=TL1, e1=e1, e2=e2, Tes1=Tes1,
                    Tes2=Tes2, I2=I2, ns=ns, evs=evs, tes=tes, Tchar=Tchar, I1=I1, n1=n1, wn=wn,
                    R=R, L=L)

    def draw(self, p, d):
        k = 60 / (2 * math.pi) / p["p1"]
        t = d["t"]
        a0 = self.fig.add_subplot(2, 2, 1)
        a0.plot(t, d["X"][0], color=GREEN, label="nieliniowy")
        a0.plot(t, p["In"] + d["XL"][0], "--", color=YELLOW, label="zlinearyzowany")
        a0.set_title("Prąd i_a"); a0.set_xlabel("t [s]"); a0.set_ylabel("[A]"); legend(a0)
        a1 = self.fig.add_subplot(2, 2, 2)
        a1.plot(t, d["X"][1] * k, color=SKY, label="nieliniowy")
        a1.plot(t, (d["wn"] + d["XL"][1]) * k, "--", color=YELLOW, label="zlinearyzowany")
        a1.set_title("Prędkość n"); a1.set_xlabel("t [s]"); a1.set_ylabel("[obr/min]"); legend(a1)
        a2 = self.fig.add_subplot(2, 2, 3)
        a2.plot(d["ns"], d["Tchar"], color=PEACH, lw=2, label="T_e(n) przy V = const")
        a2.axhline(d["Tn"], color=SUBTEXT, ls=":", label="T_L przed")
        a2.axhline(d["TL1"], color=RED, ls="--", lw=1, label="T_L po")
        a2.plot([p["nn"], d["n1"]], [d["Tn"], d["TL1"]], "o", color=TEXT)
        a2.set_ylim(0, min(np.max(d["Tchar"]), 4 * max(d["Tn"], d["TL1"])))
        a2.set_xlabel("n [obr/min]"); a2.set_ylabel("T [N·m]"); a2.set_title("Charakterystyka mechaniczna"); legend(a2, fontsize=7)
        a3 = self.fig.add_subplot(2, 2, 4)
        a3.plot(d["ns"], d["evs"][:, 0].real, color=PEACH, label="Re γ1")
        a3.plot(d["ns"], d["evs"][:, 1].real, color=MAUVE, label="Re γ2")
        a3.axvline(p["nn"], color=SUBTEXT, ls=":"); a3.axvline(p["n2"], color=SUBTEXT, ls=":")
        a3.set_xlabel("n [obr/min]"); a3.set_ylabel("Re γ [1/s]"); a3.set_title("Wartości własne vs prędkość")
        b3 = a3.twinx()
        b3.plot(d["ns"], d["tes"] * 1e3, color=YELLOW, ls="--", label="T_es")
        b3.set_ylabel("T_es [ms]", color=YELLOW); b3.grid(False); b3.tick_params(axis="y", colors=YELLOW)
        legend(a3, fontsize=7, loc="lower right")
        return [f"E = {fmt(d['E'])} V",
                f"L_dm = E/(ω_r·I_a) = {fmt(d['Ldm'] * 1e3)} mH",
                f"T_en = p1·L_dm·I² = {fmt(d['Tn'])} N·m",
                f"L/R = {fmt(d['L'] / d['R'] * 1e3)} ms (bez SEM)",
                f"n = {fmt(p['nn'])}: T_es = {fmt(d['Tes1'] * 1e3)} ms",
                f"  γ1,2 = {fmt(complex(d['e1'][0]))}",
                f"         {fmt(complex(d['e1'][1]))}",
                f"n = {fmt(p['n2'])}: I0 = {fmt(d['I2'])} A, T_es = {fmt(d['Tes2'] * 1e3)} ms",
                f"  γ1,2 = {fmt(complex(d['e2'][0]))}",
                f"         {fmt(complex(d['e2'][1]))}",
                f"Po zmianie T_L: I = {fmt(d['I1'])} A, n = {fmt(d['n1'])} obr/min"]


# ═══════════════════════════════════════════════════════════════════════════
#  8.5 Regulacja kaskadowa (PI prądu + PI prędkości) silnika DC PM
# ═══════════════════════════════════════════════════════════════════════════
class TabCascade(BaseTab):
    TITLE = "8.5 Regulacja PI"
    SLIDERS = [
        ("Vdc", "Napięcie zasilania przekształtnika", 12.0, 48.0, 24.0, "V"),
        ("Ra", "R_a", 0.02, 1.0, 0.12, "Ω"),
        ("La", "L_a", 0.05, 5.0, 0.24, "mH"),
        ("psi", "Ψ_PM (= p1Ψ, V·s/rad)", 0.02, 0.2, 0.0729, "Wb"),
        ("J", "Bezwładność J (×10⁻⁴)", 0.5, 50.0, 2.0, "kg·m²"),
        ("wci", "Pasmo pętli prądu ω_ci", 300.0, 10000.0, 3000.0, "rad/s"),
        ("wcw", "Pasmo pętli prędkości ω_cω", 10.0, 600.0, 150.0, "rad/s"),
        ("Imax", "Ograniczenie prądu I_max", 2.0, 60.0, 15.0, "A"),
        ("nref", "Prędkość zadana n*", 0.0, 3000.0, 1500.0, "obr/min"),
        ("TL", "Skok momentu obciążenia", 0.0, 1.0, 0.34, "N·m"),
        ("tL", "Chwila skoku T_L", 0.05, 0.4, 0.25, "s"),
    ]
    EXPLAIN = """
## 8.5 Podstawowa regulacja zamknięta – kaskada PI
Silnik z Przykładu 8.2 (12 V, Ψ = 0,0729 Wb) chcemy rozpędzić do zadanej prędkości i utrzymać ją mimo zmian obciążenia. Układ z Rys. 8.7 ma DWIE pętle:
• wewnętrzna – szybka pętla PRĄDU (moment), regulator PI_i porównuje prąd zadany i*_a z mierzonym,
• zewnętrzna – wolniejsza pętla PRĘDKOŚCI, regulator PI_ω na podstawie błędu prędkości wylicza, jaki prąd (moment) jest potrzebny.
### Po ludzku
Kierowca (pętla prędkości) mówi: „potrzebuję więcej mocy”. Pedał gazu (pętla prądu) szybko i dokładnie ją dostarcza, ale nigdy ponad limit (I_max) – żeby nie spalić silnika. Bez regulatora, przy bezpośrednim podaniu 11,4 V na stojący silnik, prąd rozruchu wynosi V/R_a ≈ 95 A (20 × znamionowy!) – patrz linia przerywana.
### Regulator PI
» u(t) = K_p·e(t) + K_i·∫e(t)dt
Część P reaguje na bieżący błąd, część I „pamięta” błąd z przeszłości i usuwa uchyb ustalony (np. po obciążeniu).
### Dobór nastaw (kompensacja biegunów)
Pętla prądu: obiekt 1/(R_a + s·L_a); wybieramy zero regulatora w biegunie obiektu:
» K_pi = L_a·ω_ci,   K_ii = R_a·ω_ci   →   pętla otwarta = ω_ci/s   (idealny integrator)
Pętla prędkości: obiekt k_t/(J·s), k_t = p1·Ψ:
» K_pω = J·ω_cω/k_t,   K_iω = K_pω·ω_cω/4
Zasada kaskady: pętla zewnętrzna co najmniej 5–10 razy wolniejsza od wewnętrznej.
Anti-windup: gdy prąd jest ograniczony, integrator prędkości zostaje „zamrożony” – inaczej po rozruchu pojawiłoby się duże przeregulowanie.
Przekształtnik PWM modelujemy jako stałe wzmocnienie z ograniczeniem ±V_dc (dopuszczalne przy prądzie ciągłym – patrz zakładka 8.6).
### Wykresy
• lewy górny: prędkość – zadana, z regulacją i w pętli otwartej,
• prawy górny: prąd – z regulacją ograniczony do I_max; w pętli otwartej ogromny udar,
• lewy dolny: napięcie sterujące V_a,
• prawy dolny: charakterystyka Bodego pętli prędkości z zapasem fazy (PM > 45° = dobre tłumienie).
💡 Zmniejsz ω_cω – odpowiedź wolniejsza i łagodniejsza. Zwiększ ją bardzo – zapas fazy spada, pojawiają się oscylacje. Zmniejsz I_max – rozruch trwa dłużej (mniejszy moment).
⚙ Inżynier: w rzeczywistych napędach dochodzą filtry pomiarowe, próbkowanie (opóźnienie ~1,5 okresu PWM) i kompensacja SEM (feed-forward). Nastawy weryfikuje się w laboratorium odpowiedzią skokową i sprawdza przy maksymalnym oraz minimalnym J.
"""

    def calc(self, p):
        Ra, La, psi, J = p["Ra"], p["La"] * 1e-3, p["psi"], p["J"] * 1e-4
        kt = psi
        Kpi, Kii = La * p["wci"], Ra * p["wci"]
        Kpw = J * p["wcw"] / kt
        Kiw = Kpw * p["wcw"] / 4
        wref = p["nref"] * 2 * math.pi / 60
        dt = 1e-5
        tend = max(0.45, p["tL"] + 0.2)
        N = int(tend / dt)
        rec = 10
        i = w = 0.0
        Iw = Ii = 0.0
        io = wo = 0.0
        Vo = min(kt * wref, p["Vdc"])
        out = np.zeros((N // rec + 1, 6))
        Imax, Vdc = p["Imax"], p["Vdc"]
        for k in range(N + 1):
            t = k * dt
            TL = p["TL"] if t >= p["tL"] else 0.0
            ew = wref - w
            iref = Kpw * ew + Iw
            if iref > Imax:
                iref = Imax
            elif iref < -Imax:
                iref = -Imax
            else:
                Iw += Kiw * ew * dt                       # anti-windup: całkuj tylko bez nasycenia
            ei = iref - i
            v = Kpi * ei + Ii
            if v > Vdc:
                v = Vdc
            elif v < -Vdc:
                v = -Vdc
            else:
                Ii += Kii * ei * dt
            if k % rec == 0:
                out[k // rec] = (t, w, i, v, wo, io)
            # obiekt (Euler, krok 10 µs << L_a/R_a)
            di = (v - Ra * i - kt * w) / La
            dw = (kt * i - TL) / J
            i += di * dt
            w += dw * dt
            dio = (Vo - Ra * io - kt * wo) / La
            dwo = (kt * io - TL) / J
            io += dio * dt
            wo += dwo * dt
        # Bode pętli prędkości: L(s) = PIω(s)·[ωci/(s+ωci)]·kt/(Js)
        wg = np.logspace(0, 5, 600)
        s = 1j * wg
        Lw = (Kpw + Kiw / s) * (p["wci"] / (s + p["wci"])) * kt / (J * s)
        mag = 20 * np.log10(np.abs(Lw))
        ph = np.degrees(np.unwrap(np.angle(Lw)))
        kc = np.argmin(np.abs(mag))
        return dict(o=out, Kpi=Kpi, Kii=Kii, Kpw=Kpw, Kiw=Kiw, wref=wref, wg=wg, mag=mag, ph=ph,
                    wc=wg[kc], pm=180 + ph[kc], Vo=Vo)

    def draw(self, p, d):
        o = d["o"]
        t = o[:, 0] * 1e3
        k = 60 / (2 * math.pi)
        a0 = self.fig.add_subplot(2, 2, 1)
        a0.axhline(p["nref"], color=SUBTEXT, ls=":", label="n*")
        a0.plot(t, o[:, 1] * k, color=GREEN, label="regulacja PI")
        a0.plot(t, o[:, 4] * k, "--", color=PEACH, lw=1, label="pętla otwarta")
        a0.set_xlabel("t [ms]"); a0.set_ylabel("n [obr/min]"); a0.set_title("Prędkość"); legend(a0)
        a1 = self.fig.add_subplot(2, 2, 2)
        a1.plot(t, o[:, 5], "--", color=PEACH, lw=1, label="pętla otwarta")
        a1.plot(t, o[:, 2], color=SKY, label="regulacja PI")
        a1.axhline(p["Imax"], color=RED, ls=":", lw=1)
        a1.set_ylim(min(-2, np.min(o[:, 2]) * 1.2), max(np.max(o[:, 2]) * 1.4, 5))
        a1.set_xlabel("t [ms]"); a1.set_ylabel("i_a [A]"); a1.set_title("Prąd twornika (oś ucięta)"); legend(a1)
        a2 = self.fig.add_subplot(2, 2, 3)
        a2.plot(t, o[:, 3], color=MAUVE); a2.set_xlabel("t [ms]"); a2.set_ylabel("V_a [V]")
        a2.set_title("Napięcie z przekształtnika")
        a3 = self.fig.add_subplot(2, 2, 4)
        a3.semilogx(d["wg"], d["mag"], color=YELLOW, label="|L| [dB]")
        a3.axhline(0, color=SUBTEXT, lw=1)
        a3.axvline(d["wc"], color=SUBTEXT, ls=":")
        a3.set_xlabel("ω [rad/s]"); a3.set_ylabel("|L(jω)| [dB]", color=YELLOW)
        b3 = a3.twinx()
        b3.semilogx(d["wg"], d["ph"], color=SKY)
        b3.set_ylabel("faza [°]", color=SKY); b3.grid(False)
        a3.set_title(f"Bode pętli prędkości: PM = {d['pm']:.1f}°")
        o_ = o[:, 1] * k
        tr = t[np.argmax(o_ >= 0.9 * p["nref"])] if np.any(o_ >= 0.9 * p["nref"]) else float("nan")
        mask = o[:, 0] < p["tL"]
        over = (np.max(o_[mask]) - p["nref"]) / max(p["nref"], 1) * 100
        drop = p["nref"] - np.min(o_[~mask]) if np.any(~mask) else 0
        return [f"K_pi = {fmt(d['Kpi'])} V/A,  K_ii = {fmt(d['Kii'])} V/(A·s)",
                f"K_pω = {fmt(d['Kpw'])} A·s/rad,  K_iω = {fmt(d['Kiw'])}",
                f"ω_ci/ω_cω = {fmt(p['wci'] / p['wcw'])} (≥5 zalecane)",
                f"czas do 90% n*: {fmt(tr)} ms",
                f"przeregulowanie: {fmt(over)} %",
                f"spadek prędkości po T_L: {fmt(drop)} obr/min",
                f"i_a max (PI): {fmt(np.max(o[:, 2]))} A",
                f"i_a max (pętla otwarta): {fmt(np.max(o[:, 5]))} A",
                f"zapas fazy PM = {fmt(d['pm'])}°,  ω_c = {fmt(d['wc'])} rad/s"]


# ═══════════════════════════════════════════════════════════════════════════
#  8.6 Przekształtnik DC-DC – prąd ciągły / przerywany (Zadanie 8.5)
# ═══════════════════════════════════════════════════════════════════════════
def chopper_periodic(V, R, tau, E, Ts, alpha, periods=400):
    """Analityczna symulacja łącznika jednokwadrantowego (Równ. 8.39–8.43).
    Zwraca prąd na początku okresu w stanie ustalonym i parametry okresu."""
    Ton = alpha * Ts
    i0 = 0.0
    for _ in range(periods):
        A = (V - E) / R
        i1 = A + (i0 - A) * math.exp(-Ton / tau) if Ton > 0 else i0
        if i1 < 0:
            i1 = 0.0
        B = -E / R
        Toff = Ts - Ton
        if E > 0 and i1 > 0:
            tx = tau * math.log((i1 - B) / (-B))
        else:
            tx = float("inf") if i1 > 0 else 0.0
        if tx < Toff:
            i_next = 0.0
        else:
            i_next = B + (i1 - B) * math.exp(-Toff / tau)
        if abs(i_next - i0) < 1e-12:
            i0 = i_next
            break
        i0 = i_next
    return i0


def chopper_wave(V, R, tau, E, Ts, alpha, n_per=3, npts=600):
    i0 = chopper_periodic(V, R, tau, E, Ts, alpha)
    Ton = alpha * Ts
    t = np.linspace(0, Ts, npts, endpoint=False)
    A, B = (V - E) / R, -E / R
    i = np.empty_like(t)
    v = np.empty_like(t)
    on = t < Ton
    i[on] = A + (i0 - A) * np.exp(-t[on] / tau)
    i[on] = np.maximum(i[on], 0)
    v[on] = V
    i1 = max(A + (i0 - A) * math.exp(-Ton / tau), 0) if Ton > 0 else i0
    toff = t[~on] - Ton
    ioff = B + (i1 - B) * np.exp(-toff / tau)
    zero = ioff <= 0
    ioff[zero] = 0
    i[~on] = ioff
    v[~on] = np.where(zero, E, 0.0)
    if E > 0 and i1 > 0:
        tx = tau * math.log((i1 - B) / (-B))
    else:
        tx = float("inf")
    lam = min(1.0, (Ton + tx) / Ts) if i1 > 0 else alpha
    T = np.concatenate([t + k * Ts for k in range(n_per)])
    Vav = alpha * V + (1 - lam) * E             # Równ. 8.44 (λ = 1 dla prądu ciągłego)
    Iav = (Vav - E) / R                         # średnio L·di/dt = 0 w stanie ustalonym
    return T, np.tile(i, n_per), np.tile(v, n_per), lam, Iav, Vav


class TabChopper(BaseTab):
    TITLE = "8.6 DC-DC (Zad. 8.5)"
    SLIDERS = [
        ("V", "Napięcie źródła V", 6.0, 48.0, 12.0, "V"),
        ("Ra", "R_a", 0.1, 5.0, 1.0, "Ω"),
        ("tau", "T_e = L_a/R_a", 0.5, 20.0, 5.0, "ms"),
        ("k", "Stała SEM k_E (= p1Ψ)", 0.01, 0.2, 0.05, "V·s/rad"),
        ("w", "Prędkość ω_r", 0.0, 400.0, 120.0, "rad/s"),
        ("Ts", "Okres łączeń T_s", 0.1, 5.0, 1.0, "ms"),
        ("a", "Współczynnik wypełnienia α", 0.0, 1.0, 0.5, "–"),
    ]
    EXPLAIN = """
## 8.6 Silnik DC zasilany z przekształtnika DC-DC – Zadanie 8.5
Dane: R_a = 1 Ω, L_a/R_a = 5 ms, p1 = 1, SEM 0,05 V/(rad/s), ω_r = 120 rad/s, T_s = 1 ms, V = 12 V. Prędkość stała (mechanika jest dużo wolniejsza niż przełączenia). Współczynnika wypełnienia α w treści nie podano – ustaw go suwakiem.
### Po ludzku
Przekształtnik to bardzo szybki włącznik (tranzystor IGBT): przez czas α·T_s silnik jest podłączony do 12 V, przez resztę okresu – odłączony, a prąd płynie dalej przez diodę zwrotną (bo cewka „nie lubi” przerw w prądzie). Średnio silnik „widzi” napięcie V_av ≈ α·V. Tak działa ściemniacz LED czy regulator hulajnogi.
### Równania (8.39–8.40)
» IGBT włączony (0 < t < αT_s):   V = R_a·i + L_a·di/dt + E
» Dioda przewodzi (αT_s < t < λT_s):   0 = R_a·i + L_a·di/dt + E
Rozwiązania – wykładnicze dążenie do (V−E)/R_a, a potem do −E/R_a:
» i(t) = (V−E)/R_a + [i(0) − (V−E)/R_a]·e^(−t/T_e)
» i(t) = −E/R_a + [i(αT_s) + E/R_a]·e^(−(t−αT_s)/T_e)
### Prąd ciągły czy przerywany?
Tu E = 0,05·120 = 6 V. Jeśli w czasie wyłączenia prąd spadnie do zera przed końcem okresu (dioda nie przepuści ujemnego prądu), zaczyna się przerwa – prąd PRZERYWANY, λ < 1. W przerwie na zaciskach jest samo E (silnik „pokazuje” swoją SEM).
» V_av = α·V + (1 − λ)·E    (prąd przerywany)       V_av = α·V   (prąd ciągły, λ = 1)
» I_av = (V_av − E)/R_a,     T_av = k_E·I_av
Wniosek książki: przy prądzie przerywanym średnie napięcie jest WIĘKSZE niż α·V – „wzmocnienie” przekształtnika się zmienia, a regulator dostaje nieliniowy obiekt → sterowanie staje się ospałe (szczególnie przy małych prędkościach). Rozwiązanie: wykryć tryb przerywany i dodać Δα z tablicy.
### Wykresy
• lewy górny: prąd i_a(t) w 3 okresach (stan ustalony),  • prawy górny: napięcie na zaciskach,
• lewy dolny: V_av(α) – rzeczywiste vs „idealne” α·V; szary obszar = prąd przerywany,
• prawy dolny: I_av(α) i tętnienia prądu (p-p).
💡 Domyślne α = 0,5 daje α·V = 6 V = E → według „idealnego” wzoru prąd średni byłby zero, a w rzeczywistości płyną krótkie impulsy prądu. Zwiększ T_s (niższa częstotliwość) → większe tętnienia, łatwiej o przerywanie.
⚙ Inżynier: typowe częstotliwości łączeń to kilka–kilkadziesiąt kHz. Tętnienia prądu ≈ V·α(1−α)·T_s/L_a (dla T_s << T_e) – aby je zmniejszyć, podnosimy częstotliwość albo dodajemy dławik szeregowy.
"""

    def calc(self, p):
        V, R, tau, Ts = p["V"], p["Ra"], p["tau"] * 1e-3, p["Ts"] * 1e-3
        E = p["k"] * p["w"]
        T, i, v, lam, Iav, Vav = chopper_wave(V, R, tau, E, Ts, p["a"])
        al = np.linspace(0.0, 1.0, 101)
        Vs, Is, Rp, Ls = [], [], [], []
        for a in al:
            _T, ii, _vv, l_, ia_, va_ = chopper_wave(V, R, tau, E, Ts, a, n_per=1, npts=300)
            Vs.append(va_); Is.append(ia_); Rp.append(np.ptp(ii)); Ls.append(l_)
        return dict(T=T, i=i, v=v, lam=lam, Iav=Iav, Vav=Vav, E=E, al=al, Vs=np.array(Vs),
                    Is=np.array(Is), Rp=np.array(Rp), Ls=np.array(Ls))

    def draw(self, p, d):
        T = d["T"] * 1e3
        a0 = self.fig.add_subplot(2, 2, 1)
        a0.plot(T, d["i"], color=GREEN); a0.axhline(d["Iav"], color=YELLOW, ls="--", lw=1, label=f"I_av={d['Iav']:.3g} A")
        a0.set_xlabel("t [ms]"); a0.set_ylabel("i_a [A]")
        a0.set_title("Prąd twornika – " + ("PRZERYWANY" if d["lam"] < 0.999 else "ciągły") + f" (λ={d['lam']:.3f})")
        legend(a0)
        a1 = self.fig.add_subplot(2, 2, 2)
        a1.plot(T, d["v"], color=SKY); a1.axhline(d["E"], color=PEACH, ls=":", label="E")
        a1.axhline(d["Vav"], color=YELLOW, ls="--", lw=1, label=f"V_av={d['Vav']:.3g} V")
        a1.set_xlabel("t [ms]"); a1.set_ylabel("v_a [V]"); a1.set_title("Napięcie na zaciskach silnika"); legend(a1)
        a2 = self.fig.add_subplot(2, 2, 3)
        disc = d["Ls"] < 0.999
        a2.fill_between(d["al"], 0, p["V"], where=disc, color=SURF1, alpha=0.5, label="prąd przerywany")
        a2.plot(d["al"], d["Vs"], color=GREEN, lw=2, label="V_av rzeczywiste")
        a2.plot(d["al"], d["al"] * p["V"], "--", color=PEACH, label="α·V (prąd ciągły)")
        a2.axvline(p["a"], color=TEXT, lw=1, ls=":")
        a2.set_xlabel("α"); a2.set_ylabel("V_av [V]"); a2.set_title("Średnie napięcie vs wypełnienie"); legend(a2, fontsize=7)
        a3 = self.fig.add_subplot(2, 2, 4)
        a3.plot(d["al"], d["Is"], color=SKY, lw=2, label="I_av")
        a3.plot(d["al"], (d["al"] * p["V"] - d["E"]) / p["Ra"], "--", color=PEACH, label="(αV−E)/R_a")
        a3.plot(d["al"], d["Rp"], color=MAUVE, label="tętnienia p-p")
        a3.axvline(p["a"], color=TEXT, lw=1, ls=":")
        a3.set_ylim(bottom=min(0, np.min(d["Is"])) - 0.2)
        a3.set_xlabel("α"); a3.set_ylabel("[A]"); a3.set_title("Prąd średni i tętnienia"); legend(a3, fontsize=7)
        return [f"E = k_E·ω_r = {fmt(d['E'])} V",
                f"L_a = {fmt(p['tau'] * p['Ra'])} mH",
                f"α = {fmt(p['a'])},  λ = {fmt(d['lam'])}",
                "tryb: " + ("PRZERYWANY" if d["lam"] < 0.999 else "ciągły"),
                f"V_av = {fmt(d['Vav'])} V  (α·V = {fmt(p['a'] * p['V'])} V)",
                f"I_av = {fmt(d['Iav'])} A",
                f"T_av = k·I_av = {fmt(p['k'] * d['Iav'])} N·m",
                f"i max / min = {fmt(np.max(d['i']))} / {fmt(np.min(d['i']))} A",
                f"granica ciągłości: α ≈ {fmt(d['al'][np.argmax(d['Ls'] >= 0.999)] if np.any(d['Ls'] >= 0.999) else float('nan'))}"]


# ═══════════════════════════════════════════════════════════════════════════
#  8.7 Parametry z prób: zanik prądu przy postoju, swobodny wybieg
# ═══════════════════════════════════════════════════════════════════════════
class TabTests(BaseTab):
    TITLE = "8.7 Próby (Lab 8.1)"
    SLIDERS = [
        ("Ra", "Prawdziwe R_a", 0.05, 2.0, 0.5, "Ω"),
        ("La", "Prawdziwe L_a (przy małym prądzie)", 0.5, 50.0, 5.0, "mH"),
        ("sat", "Nasycenie: spadek L_a na 10 A", 0.0, 0.6, 0.2, "–"),
        ("Vd", "Spadek napięcia na diodzie V_d", 0.0, 2.0, 0.8, "V"),
        ("i0", "Prąd początkowy i_0", 1.0, 50.0, 10.0, "A"),
        ("noise", "Szum pomiarowy", 0.0, 5.0, 1.0, "%"),
        ("Jt", "Prawdziwe J", 0.005, 0.5, 0.05, "kg·m²"),
        ("B", "Tarcie lepkie B", 0.0, 0.02, 0.002, "N·m·s"),
        ("Tc", "Tarcie suche T_c", 0.0, 1.0, 0.2, "N·m"),
        ("n0", "Prędkość początkowa wybiegu", 300.0, 3000.0, 1500.0, "obr/min"),
    ]
    EXPLAIN = """
## 8.7 Jak zmierzyć parametry modelu? (Lab 8.1)
Model jest tyle wart, ile jego parametry. Książka opisuje próby POSTOJOWE (wirnik stoi) – nie trzeba drugiej maszyny, a zużycie energii jest małe.
### Próba zaniku prądu (Rys. 8.9)
1. Przez przekształtnik DC-DC ustalamy w tworniku prąd i_0. Napięcie V_0 i prąd mierzymy → R_a = V_0/i_0 (w stanie ustalonym cewka nie ma spadku napięcia).
2. Wyłączamy IGBT. Prąd płynie dalej przez diodę zwrotną i zanika. Rejestrujemy i(t) i napięcie diody V_d.
3. Równanie obwodu (Równ. 8.46):  0 = R_a·i + L_a·di/dt + V_d
4. Całkujemy od 0 do t_da (gdy i ≈ 0,01·i_0) (Równ. 8.47):
» L_a·i_0 = R_a·∫i·dt + V_d·t_da      →      L_a = (R_a·∫i dt + V_d·t_da)/i_0
Po ludzku: cewka zgromadziła energię (prąd „rozpędzony”); ile „pracy” wykonał zanik prądu na rezystancji i diodzie, tyle wynosiła jej „bezwładność” L_a·i_0.
Pomiar dla kilku i_0 pokazuje, czy L_a zależy od prądu (nasycenie). Pominięcie V_d daje błąd – szczególnie przy małych napięciach i dużych prądach (prawy górny wykres). Przy nasyceniu próba daje indukcyjność „uśrednioną energetycznie” po całym zaniku (od i_0 do 0), a nie wartość dokładnie przy i_0 – stąd różnica zielonej i białej krzywej. Ustaw nasycenie = 0, a zgodność będzie bardzo dobra.
### Próba wybiegu – moment bezwładności J
Rozpędzamy silnik, wyłączamy zasilanie (I_F = 0 – brak strat w żelazie) i mierzymy spadek prędkości. Straty mechaniczne p_mec(Ω) znamy z prób biegu jałowego. Z bilansu energii (Równ. 8.49):
» p_mec = −J·Ω·dΩ/dt     →     J = −p_mec/(Ω·dΩ/dt)
Po ludzku: bąk zwalnia tym wolniej, im jest „cięższy” (większe J) przy tych samych oporach.
Nachylenie dΩ/dt wyznaczamy z danych pomiarowych (regresja liniowa lokalnie) – na wykresie styczna.
### Wykresy
• lewy górny: zanik prądu (z szumem) i napięcie diody,
• prawy górny: L_a zmierzone dla różnych i_0 (z V_d i bez) vs prawdziwe,
• lewy dolny: wybieg z zaznaczoną styczną,
• prawy dolny: błędy wyznaczenia parametrów w %.
💡 Ustaw V_d = 0 – obie metody się zgadzają. Zwiększ szum – rośnie błąd J (różniczkowanie wzmacnia szum!), a L_a prawie nie (całkowanie uśrednia szum). To ważna lekcja: inżynier woli całkować niż różniczkować.
⚙ Inżynier: w maszynie z magnesami (PM) próba wybiegu nie oddzieli strat mechanicznych od strat w żelazie (magnesu nie da się wyłączyć) – wtedy J mierzy się metodą wahadła. Pełne procedury opisują normy IEEE, IEC, NEMA.
"""

    def build_extra(self, parent):
        self.seed = 3

    def _decay(self, p, i0, rng):
        Ra, Vd = p["Ra"], p["Vd"]
        L0 = p["La"] * 1e-3

        def La(i):
            return L0 / (1 + p["sat"] * min(i, 30) / 10)
        tau = L0 / Ra
        dt = tau / 400
        t, i = [0.0], [i0]
        while i[-1] > 0.005 * i0 and len(t) < 20000:
            x = i[-1]
            k1 = -(Ra * x + Vd) / La(x)
            x2 = x + 0.5 * dt * k1
            k2 = -(Ra * x2 + Vd) / La(max(x2, 0))
            x = x + dt * k2
            t.append(t[-1] + dt); i.append(max(x, 0.0))
        t, i = np.array(t), np.array(i)
        im = i + rng.normal(0, p["noise"] / 100 * i0, len(i))
        V0m = Ra * i0 * (1 + rng.normal(0, p["noise"] / 300))
        Ra_est = V0m / i0
        k = np.argmax(im <= 0.01 * i0) if np.any(im <= 0.01 * i0) else len(im) - 1
        tda = t[k]
        integ = np.trapezoid(im[:k + 1], t[:k + 1]) if hasattr(np, "trapezoid") else np.trapz(im[:k + 1], t[:k + 1])
        La_est = (Ra_est * integ + Vd * tda) / i0
        La_novd = Ra_est * integ / i0
        return t, i, im, Ra_est, La_est, La_novd, La(i0), tda

    def calc(self, p):
        rng = np.random.default_rng(self.seed)
        t, i, im, Ra_e, La_e, La_n, La_true, tda = self._decay(p, p["i0"], rng)
        sweep = np.linspace(1, 50, 20)
        sw = [self._decay(p, x, rng) for x in sweep]
        J, B, Tc = p["Jt"], p["B"], p["Tc"]
        W0 = p["n0"] * 2 * math.pi / 60
        # analityczny wybieg: J dΩ/dt = −(BΩ + Tc)
        if B > 0:
            tstop = J / B * math.log((B * W0 + Tc) / Tc) if Tc > 0 else 5 * J / B
        else:
            tstop = J * W0 / Tc if Tc > 0 else 10.0
        tw = np.linspace(0, tstop, 800)
        if B > 0:
            Wt = (W0 + Tc / B) * np.exp(-B * tw / J) - Tc / B
        else:
            Wt = W0 - Tc / J * tw
        Wt = np.maximum(Wt, 0)
        Wm = Wt + rng.normal(0, p["noise"] / 100 * W0 * 0.2, len(Wt))
        Ws = 0.6 * W0                                 # punkt, w którym liczymy J
        kc = int(np.argmin(np.abs(Wt - Ws)))
        win = slice(max(0, kc - 40), min(len(tw), kc + 40))
        slope, icpt = np.polyfit(tw[win], Wm[win], 1)
        pmec = (B * Ws + Tc) * Ws                     # z próby biegu jałowego (znane)
        J_est = -pmec / (Ws * slope) if slope < 0 else float("nan")
        return dict(t=t, i=i, im=im, Ra_e=Ra_e, La_e=La_e, La_n=La_n, La_true=La_true, tda=tda,
                    sweep=sweep, sw=sw, tw=tw, Wm=Wm, kc=kc, slope=slope, icpt=icpt, J_est=J_est, Ws=Ws)

    def draw(self, p, d):
        a0 = self.fig.add_subplot(2, 2, 1)
        a0.plot(d["t"] * 1e3, d["im"], ".", ms=2, color=SKY, label="i zmierzone")
        a0.plot(d["t"] * 1e3, d["i"], color=GREEN, label="i prawdziwe")
        a0.axvline(d["tda"] * 1e3, color=SUBTEXT, ls=":")
        a0.set_xlabel("t [ms]"); a0.set_ylabel("i [A]"); a0.set_title("Zanik prądu po wyłączeniu IGBT"); legend(a0)
        a1 = self.fig.add_subplot(2, 2, 2)
        sw = d["sw"]
        a1.plot(d["sweep"], [s[6] * 1e3 for s in sw], color=TEXT, lw=2, label="L_a prawdziwe")
        a1.plot(d["sweep"], [s[4] * 1e3 for s in sw], "o-", color=GREEN, ms=4, label="z V_d (Równ. 8.47)")
        a1.plot(d["sweep"], [s[5] * 1e3 for s in sw], "s--", color=RED, ms=4, label="bez V_d")
        a1.axvline(p["i0"], color=SUBTEXT, ls=":")
        a1.set_xlabel("i_0 [A]"); a1.set_ylabel("L_a [mH]"); a1.set_title("L_a z próby zaniku vs prąd"); legend(a1, fontsize=7)
        a2 = self.fig.add_subplot(2, 2, 3)
        k = 60 / (2 * math.pi)
        a2.plot(d["tw"], d["Wm"] * k, ".", ms=2, color=MAUVE, label="pomiar")
        tt = d["tw"][max(0, d["kc"] - 150):d["kc"] + 150]
        a2.plot(tt, (d["slope"] * tt + d["icpt"]) * k, color=YELLOW, lw=2, label="styczna dΩ/dt")
        a2.set_xlabel("t [s]"); a2.set_ylabel("n [obr/min]"); a2.set_title("Próba wybiegu"); legend(a2)
        a3 = self.fig.add_subplot(2, 2, 4)
        errs = [(d["Ra_e"] / p["Ra"] - 1) * 100, (d["La_e"] / d["La_true"] - 1) * 100,
                (d["La_n"] / d["La_true"] - 1) * 100, (d["J_est"] / p["Jt"] - 1) * 100]
        a3.bar(["R_a", "L_a (z V_d)", "L_a (bez V_d)", "J"], errs, color=[SKY, GREEN, RED, MAUVE])
        a3.axhline(0, color=TEXT, lw=1)
        a3.set_ylabel("błąd [%]"); a3.set_title("Dokładność wyznaczenia parametrów")
        return [f"R_a zmierzone = {fmt(d['Ra_e'])} Ω (prawdz. {fmt(p['Ra'])})",
                f"t_da = {fmt(d['tda'] * 1e3)} ms",
                f"L_a(i0) prawdziwe = {fmt(d['La_true'] * 1e3)} mH",
                f"L_a z V_d    = {fmt(d['La_e'] * 1e3)} mH",
                f"L_a bez V_d  = {fmt(d['La_n'] * 1e3)} mH",
                f"dΩ/dt przy Ω = {fmt(d['Ws'])} rad/s: {fmt(d['slope'])} rad/s²",
                f"p_mec = {fmt((p['B'] * d['Ws'] + p['Tc']) * d['Ws'])} W",
                f"J = −p_mec/(Ω·dΩ/dt) = {fmt(d['J_est'])} kg·m²",
                f"   (prawdziwe {fmt(p['Jt'])})"]


# ═══════════════════════════════════════════════════════════════════════════
#  Zadania 8.1–8.3
# ═══════════════════════════════════════════════════════════════════════════
class TabP81(BaseTab):
    TITLE = "Zad. 8.1"
    SLIDERS = [
        ("RL", "Rezystancja obciążenia R_l", 0.1, 10.0, 1.0, "Ω"),
        ("L", "Indukcyjność obciążenia L", 0.01, 5.0, 1.0, "H"),
        ("Ra", "R_a (L_a = 0)", 0.0, 1.0, 0.1, "Ω"),
        ("RF", "R_F", 5.0, 200.0, 50.0, "Ω"),
        ("LF", "L_F", 0.1, 20.0, 5.0, "H"),
        ("VF", "Napięcie wzbudzenia V_F", 10.0, 300.0, 120.0, "V"),
        ("K", "E/i_F = ω_r·L_dm", 5.0, 100.0, 40.0, "V/A"),
        ("tend", "Czas obserwacji", 0.5, 10.0, 5.0, "s"),
    ]
    EXPLAIN = """
## Zadanie 8.1 – narastanie prądu prądnicy po załączeniu wzbudzenia
Dane: obciążenie R_l = 1 Ω, L = 1 H; R_a = 0,1 Ω, L_a = 0; wzbudzenie R_F = 50 Ω, L_F = 5 H nagle przyłączone do 120 V; E/i_F = ω_r·L_dm = 40 V/A; prędkość stała.
### Po ludzku
Dwa „zbiorniki” napełniane jeden po drugim: najpierw prąd wzbudzenia napełnia się ze stałą czasową T_F, a SEM (proporcjonalna do i_F) napełnia drugi zbiornik – obwód twornika z obciążeniem – ze stałą czasową T_a. Wynik to krzywa „S” (na początku prąd rośnie bardzo wolno).
### Krok 1: obwód wzbudzenia
» i_F(t) = (V_F/R_F)·(1 − e^(−t/T_F)),   V_F/R_F = 2,4 A,   T_F = L_F/R_F = 0,1 s
### Krok 2: obwód twornika + obciążenia
» (R_a + R_l)·i_a + L·di_a/dt = E(t) = 40·i_F(t)
» T_a = L/(R_a + R_l) = 1/1,1 = 0,909 s,   I_a(∞) = 40·2,4/1,1 = 87,27 A
### Krok 3: rozwiązanie (dwie stałe czasowe)
» i_a(t) = I_a∞·[1 − (T_a·e^(−t/T_a) − T_F·e^(−t/T_F))/(T_a − T_F)]
Na starcie di_a/dt = 0 (krzywa „S”), po ok. 5·T_a ≈ 4,5 s prąd osiąga wartość ustaloną. Napięcie na zaciskach: V = E − R_a·i_a.
💡 Jak w Przykładzie 8.1 – ale teraz wolniejszy jest obwód twornika (duże L obciążenia), a nie wzbudzenia. Zamień wartości tak, by T_a ≈ T_F – krzywa nadal jest poprawna (przypadek graniczny t·e^(−t/T)).
"""

    def calc(self, p):
        from scipy.integrate import solve_ivp
        RF, LF, Ra, RL, L, K = p["RF"], p["LF"], p["Ra"], p["RL"], p["L"], p["K"]

        def f(_t, x):
            iF, ia = x
            return [(p["VF"] - RF * iF) / LF, (K * iF - (Ra + RL) * ia) / L]
        t = np.linspace(0, p["tend"], 1500)
        s = solve_ivp(f, (0, p["tend"]), [0, 0], t_eval=t, method="LSODA", rtol=1e-8, atol=1e-9)
        return dict(t=s.t, iF=s.y[0], ia=s.y[1], TF=LF / RF, Ta=L / (Ra + RL),
                    Iinf=K * p["VF"] / RF / (Ra + RL))

    def draw(self, p, d):
        t = d["t"]
        a0 = self.fig.add_subplot(2, 2, 1)
        a0.plot(t, d["iF"], color=MAUVE); a0.set_title("Prąd wzbudzenia i_F(t)"); a0.set_xlabel("t [s]"); a0.set_ylabel("[A]")
        a1 = self.fig.add_subplot(2, 2, 2)
        a1.plot(t, d["ia"], color=GREEN); a1.axhline(d["Iinf"], color=SUBTEXT, ls=":")
        a1.set_title("Prąd twornika i_a(t) – krzywa „S”"); a1.set_xlabel("t [s]"); a1.set_ylabel("[A]")
        a2 = self.fig.add_subplot(2, 2, 3)
        E = p["K"] * d["iF"]
        a2.plot(t, E, color=PEACH, label="E = 40·i_F"); a2.plot(t, E - p["Ra"] * d["ia"], color=SKY, label="V zacisków")
        a2.set_title("SEM i napięcie"); a2.set_xlabel("t [s]"); a2.set_ylabel("[V]"); legend(a2)
        a3 = self.fig.add_subplot(2, 2, 4)
        a3.plot(t, d["iF"] / max(d["iF"][-1], 1e-9), color=MAUVE, label="i_F / i_F∞")
        a3.plot(t, d["ia"] / d["Iinf"], color=GREEN, label="i_a / i_a∞")
        a3.set_title("Porównanie (znormalizowane)"); a3.set_xlabel("t [s]"); legend(a3)
        k63 = np.argmax(d["ia"] >= 0.632 * d["Iinf"]) if np.any(d["ia"] >= 0.632 * d["Iinf"]) else -1
        return [f"i_F∞ = V_F/R_F = {fmt(p['VF'] / p['RF'])} A",
                f"T_F = L_F/R_F = {fmt(d['TF'])} s",
                f"T_a = L/(R_a+R_l) = {fmt(d['Ta'])} s",
                f"I_a∞ = {fmt(d['Iinf'])} A",
                f"i_a(t_koniec) = {fmt(d['ia'][-1])} A",
                f"63% I_a∞ po {fmt(t[k63]) if k63 >= 0 else '—'} s",
                f"E∞ = {fmt(p['K'] * p['VF'] / p['RF'])} V"]


class TabP82(BaseTab):
    TITLE = "Zad. 8.2"
    SLIDERS = [
        ("V", "Napięcie zasilania V", 1.0, 24.0, 12.0, "V"),
        ("Ra", "R_a", 0.02, 1.0, 0.12, "Ω"),
        ("La", "L_a (w zadaniu 0)", 0.0, 2.0, 0.0, "mH"),
        ("kt", "p1·Ψ_PM", 0.02, 0.2, 0.073, "N·m/A"),
        ("J", "J (×10⁻⁴)", 0.5, 50.0, 2.0, "kg·m²"),
        ("T0", "Moment stały obciążenia", 0.0, 1.0, 0.2, "N·m"),
        ("B", "Współczynnik ×10⁻³ (T_L = T0 + B·ω)", 0.0, 5.0, 1.0, "N·m·s"),
        ("tend", "Czas obserwacji", 10.0, 200.0, 40.0, "ms"),
    ]
    EXPLAIN = """
## Zadanie 8.2 – rozruch silnika PM pod obciążeniem
Dane: R_a = 0,12 Ω, L_a = 0, 2p1 = 2, p1·Ψ_PM = 0,073 N·m/A, J = 2·10⁻⁴ kg·m², obciążenie T_L = 0,2 + 10⁻³·ω_r, silnik włączony wprost na 12 V.
### Po ludzku
W chwili włączenia silnik stoi – nie ma SEM, więc prąd ogranicza tylko R_a: 12/0,12 = 100 A (!). Daje to ogromny moment 7,3 N·m i silnik gwałtownie przyspiesza. W miarę rozpędzania SEM rośnie, prąd maleje, aż moment silnika zrówna się z obciążeniem.
### Krok 1: równania (L_a = 0 → Równ. 8.21 z T_e = 0)
» i_a = (V − k·ω)/R_a,    k = p1·Ψ_PM = 0,073
» J·dω/dt = k·i_a − T0 − B·ω = k·V/R_a − T0 − (k²/R_a + B)·ω
To równanie 1. rzędu: ω dąży wykładniczo do wartości końcowej.
### Krok 2: stała czasowa i prędkość końcowa
» τ_m = J/(k²/R_a + B) = 2·10⁻⁴/(0,04441 + 0,001) = 4,40 ms
» ω_f = (k·V/R_a − T0)/(k²/R_a + B) = (7,3 − 0,2)/0,04541 = 156,3 rad/s ≈ 1493 obr/min
### Krok 3: przebiegi
» ω(t) = ω_f·(1 − e^(−t/τ_m)),    i_a(t) = (V − k·ω)/R_a,    T_e = k·i_a
Prąd startuje od 100 A i spada do ok. 4,9 A. Po 5·τ_m ≈ 22 ms rozruch zakończony.
💡 Ustaw L_a = 0,24 mH (T_e = 2 ms jak w Przykładzie 8.2) – prąd nie skacze już do 100 A natychmiast, a przebieg ma drugi rząd (lekkie oscylacje). Zwiększ J – rozruch dłuższy, ale prąd rozruchowy ten sam.
⚙ Inżynier: 20-krotny prąd rozruchowy grzeje szczotki i może rozmagnesować magnesy – w praktyce rozruch robi się przez przekształtnik z ograniczeniem prądu (zakładka 8.5).
"""

    def calc(self, p):
        from scipy.integrate import solve_ivp
        V, Ra, La, k, J, T0, B = p["V"], p["Ra"], p["La"] * 1e-3, p["kt"], p["J"] * 1e-4, p["T0"], p["B"] * 1e-3
        t = np.linspace(0, p["tend"] * 1e-3, 1500)

        def TL(w):
            return (T0 if w > 1e-9 else min(T0, k * V / Ra)) + B * w
        if La <= 1e-9:
            def f(_t, x):
                w = x[0]
                i = (V - k * w) / Ra
                dw = (k * i - TL(w)) / J
                return [dw if (w > 0 or dw > 0) else 0.0]
            s = solve_ivp(f, (0, t[-1]), [0.0], t_eval=t, method="LSODA", rtol=1e-8, atol=1e-10)
            w = s.y[0]
            i = (V - k * w) / Ra
        else:
            def f(_t, x):
                i, w = x
                dw = (k * i - TL(w)) / J
                return [(V - Ra * i - k * w) / La, dw if (w > 0 or dw > 0) else 0.0]
            s = solve_ivp(f, (0, t[-1]), [0.0, 0.0], t_eval=t, method="LSODA", rtol=1e-8, atol=1e-10)
            i, w = s.y
        taum = J / (k * k / Ra + B)
        wf = (k * V / Ra - T0) / (k * k / Ra + B)
        return dict(t=s.t, w=w, i=i, taum=taum, wf=wf, TLt=T0 + B * w)

    def draw(self, p, d):
        t = d["t"] * 1e3
        a0 = self.fig.add_subplot(2, 2, 1)
        a0.plot(t, d["w"], color=SKY); a0.axhline(d["wf"], color=SUBTEXT, ls=":")
        a0.set_title("Prędkość ω_r(t)"); a0.set_xlabel("t [ms]"); a0.set_ylabel("[rad/s]")
        a1 = self.fig.add_subplot(2, 2, 2)
        a1.plot(t, d["i"], color=GREEN); a1.set_title("Prąd twornika i_a(t)"); a1.set_xlabel("t [ms]"); a1.set_ylabel("[A]")
        a2 = self.fig.add_subplot(2, 2, 3)
        a2.plot(t, p["kt"] * d["i"], color=PEACH, label="T_e = k·i_a")
        a2.plot(t, d["TLt"], color=RED, label="T_L = T0 + Bω")
        a2.set_title("Momenty"); a2.set_xlabel("t [ms]"); a2.set_ylabel("[N·m]"); a2.set_yscale("log"); legend(a2)
        a3 = self.fig.add_subplot(2, 2, 4)
        a3.plot(d["w"], p["kt"] * d["i"], color=PEACH, label="trajektoria T_e(ω)")
        a3.plot(d["w"], d["TLt"], color=RED, label="T_L(ω)")
        a3.set_title("Płaszczyzna moment–prędkość"); a3.set_xlabel("ω_r [rad/s]"); a3.set_ylabel("[N·m]"); legend(a3)
        return [f"I rozruchowy = V/R_a = {fmt(p['V'] / p['Ra'])} A",
                f"T_e rozruchowy = {fmt(p['kt'] * p['V'] / p['Ra'])} N·m",
                f"τ_m = J/(k²/R_a + B) = {fmt(d['taum'] * 1e3)} ms",
                f"ω_f = {fmt(d['wf'])} rad/s = {fmt(d['wf'] * 60 / (2 * math.pi))} obr/min",
                f"i_a końcowy = {fmt(d['i'][-1])} A",
                f"T_L końcowy = {fmt(d['TLt'][-1])} N·m"]


class TabP83(BaseTab):
    TITLE = "Zad. 8.3"
    SLIDERS = [
        ("n0", "Prędkość początkowa", 500.0, 3000.0, 1500.0, "obr/min"),
        ("Ra", "R_a", 0.02, 1.0, 0.12, "Ω"),
        ("La", "L_a (T_e = 2 ms → 0,24)", 0.01, 2.0, 0.24, "mH"),
        ("kt", "p1·Ψ_PM", 0.02, 0.2, 0.073, "N·m/A"),
        ("J", "J (×10⁻⁴)", 0.5, 50.0, 2.0, "kg·m²"),
        ("T0", "Moment stały obciążenia", 0.0, 1.0, 0.2, "N·m"),
        ("B", "Współczynnik ×10⁻³", 0.0, 5.0, 1.0, "N·m·s"),
        ("step", "Zmiana momentu obciążenia", -80.0, 50.0, -20.0, "%"),
        ("tend", "Czas obserwacji", 10.0, 200.0, 40.0, "ms"),
    ]
    EXPLAIN = """
## Zadanie 8.3 – zrzut obciążenia o 20%
Silnik z Zadania 8.2 pracuje w stanie ustalonym przy 1500 obr/min. Moment obciążenia maleje skokowo o 20%. Szukamy nowego prądu ustalonego oraz ω_r(t), T_e(t).
### Krok 1: stan początkowy
» ω0 = 2π·1500/60 = 157,08 rad/s
» T_L0 = 0,2 + 10⁻³·157,08 = 0,357 N·m     →   I_a0 = T_L0/k = 4,89 A
» napięcie potrzebne do 1500 obr/min: V = R_a·I_a0 + k·ω0 = 0,587 + 11,467 = 12,05 V (≈ 12 V z Zad. 8.2)
### Krok 2: nowy stan ustalony
Obciążenie: T_L = 0,8·(0,2 + 10⁻³·ω). Z równań V = R_a·I + k·ω oraz k·I = T_L:
» ω_f = (k·V/R_a − 0,8·0,2)/(k²/R_a + 0,8·10⁻³),   I_af = (V − k·ω_f)/R_a
Wynik: prąd spada z 4,89 do ok. 3,93 A, a prędkość rośnie tylko o ok. 15 obr/min (do ok. 1515 obr/min, +1%) – silnik PM ma „sztywną” charakterystykę.
### Krok 3: warunek początkowy (wskazówka książki)
Teraz skok dotyczy MOMENTU, a nie napięcia. Prąd w chwili t = 0⁺ jest ciągły i jego pochodna jest zerowa: (di_a/dt)(0) = 0, bo napięcie i SEM się nie zmieniły. Zmienia się natomiast od razu przyspieszenie: dω/dt(0) = (k·I_a0 − T_L)/J ≠ 0.
Równanie 2. rzędu jak w Przykładzie 8.2 – z L_a = 0,24 mH i J = 2·10⁻⁴ mamy T_em < 4T_e, więc odpowiedź jest lekko oscylacyjna.
💡 Porównaj z Przykładem 8.2: tam skakało napięcie (i prąd reagował gwałtownie), tu skacze moment (prąd reaguje łagodnie, bo po drodze jest mechanika).
"""

    def calc(self, p):
        Ra, La, k, J = p["Ra"], p["La"] * 1e-3, p["kt"], p["J"] * 1e-4
        T0, B = p["T0"], p["B"] * 1e-3
        w0 = p["n0"] * 2 * math.pi / 60
        TL0 = T0 + B * w0
        I0 = TL0 / k
        V = Ra * I0 + k * w0
        c = 1 + p["step"] / 100
        from scipy.integrate import solve_ivp

        def f(_t, x):
            i, w = x
            return [(V - Ra * i - k * w) / La, (k * i - c * (T0 + B * w)) / J]
        t = np.linspace(0, p["tend"] * 1e-3, 1500)
        s = solve_ivp(f, (0, t[-1]), [I0, w0], t_eval=t, method="Radau", rtol=1e-9, atol=1e-10)
        wf = (k * V / Ra - c * T0) / (k * k / Ra + c * B)
        If = (V - k * wf) / Ra
        return dict(t=s.t, i=s.y[0], w=s.y[1], V=V, I0=I0, TL0=TL0, wf=wf, If=If, c=c, w0=w0,
                    Tem=J * Ra / k ** 2, Te=La / Ra)

    def draw(self, p, d):
        t = d["t"] * 1e3
        a0 = self.fig.add_subplot(2, 2, 1)
        a0.plot(t, d["w"] * 60 / (2 * math.pi), color=SKY); a0.axhline(d["wf"] * 60 / (2 * math.pi), color=SUBTEXT, ls=":")
        a0.set_title("Prędkość"); a0.set_xlabel("t [ms]"); a0.set_ylabel("[obr/min]")
        a1 = self.fig.add_subplot(2, 2, 2)
        a1.plot(t, d["i"], color=GREEN); a1.axhline(d["If"], color=SUBTEXT, ls=":")
        a1.set_title("Prąd twornika (di/dt(0) = 0)"); a1.set_xlabel("t [ms]"); a1.set_ylabel("[A]")
        a2 = self.fig.add_subplot(2, 2, 3)
        a2.plot(t, p["kt"] * d["i"], color=PEACH, label="T_e")
        a2.plot(t, d["c"] * (p["T0"] + p["B"] * 1e-3 * d["w"]), color=RED, label="T_L")
        a2.set_title("Momenty"); a2.set_xlabel("t [ms]"); a2.set_ylabel("[N·m]"); legend(a2)
        a3 = self.fig.add_subplot(2, 2, 4)
        a3.plot(d["w"], d["i"], color=MAUVE)
        a3.plot([d["w0"]], [d["I0"]], "o", color=TEXT); a3.plot([d["wf"]], [d["If"]], "*", color=YELLOW, ms=12)
        a3.set_title("Trajektoria w płaszczyźnie (ω, i)"); a3.set_xlabel("ω [rad/s]"); a3.set_ylabel("i_a [A]")
        return [f"T_L0 = {fmt(d['TL0'])} N·m,  I_a0 = {fmt(d['I0'])} A",
                f"V potrzebne = {fmt(d['V'])} V",
                f"T_L po zmianie = {fmt(d['c'])}·T_L(ω)",
                f"ω_f = {fmt(d['wf'])} rad/s = {fmt(d['wf'] * 60 / (2 * math.pi))} obr/min",
                f"I_af = {fmt(d['If'])} A",
                f"T_e = {fmt(d['Te'] * 1e3)} ms, T_em = {fmt(d['Tem'] * 1e3)} ms",
                "odpowiedź: " + ("oscylacyjna" if d["Tem"] < 4 * d["Te"] else "aperiodyczna") +
                " (T_em " + ("<" if d["Tem"] < 4 * d["Te"] else ">") + " 4T_e)"]


# ═══════════════════════════════════════════════════════════════════════════
#  Główne okno
# ═══════════════════════════════════════════════════════════════════════════
CH7 = [Tab72, TabPMSM, TabIM, TabSat, TabSkin, TabPark, TabHF, TabProb7]
CH8 = [TabEx81, TabEx82, TabVarFlux, TabSeries, TabCascade, TabChopper, TabTests, TabP81, TabP82, TabP83]


class App:
    def __init__(self, root):
        self.root = root
        root.title("Modele zaawansowane maszyn + stany nieustalone DC (Boldea, rozdz. 7–8)")
        root.geometry("1500x960")
        root.configure(bg=BG)
        self._style()
        self.results = queue.Queue()
        top = ttk.Notebook(root)
        top.pack(fill="both", expand=True)
        self.tabs = []
        self.frame2tab = {}
        start = TabStart(top, self)
        self._reg(start)
        for title, classes in (("Rozdział 7 – modele zaawansowane", CH7),
                               ("Rozdział 8 – stany nieustalone maszyn DC", CH8)):
            fr = ttk.Frame(top)
            top.add(fr, text=title)
            nb = ttk.Notebook(fr, style="Sub.TNotebook")
            nb.pack(fill="both", expand=True)
            for cls in classes:
                self._reg(cls(nb, self))
            nb.bind("<<NotebookTabChanged>>", self._on_tab)
        top.bind("<<NotebookTabChanged>>", self._on_tab)
        self.status = tk.StringVar(value="Gotowy")
        ttk.Label(root, textvariable=self.status, style="Status.TLabel").pack(fill="x", side="bottom")
        start.request()
        root.after(50, self._poll)

    def _reg(self, tab):
        self.tabs.append(tab)
        self.frame2tab[str(tab.frame)] = tab

    def _style(self):
        st = ttk.Style(self.root)
        try:
            st.theme_use("clam")
        except tk.TclError:
            pass
        st.configure(".", background=BG, foreground=TEXT, fieldbackground=SURF0, bordercolor=SURF1,
                     lightcolor=SURF0, darkcolor=CRUST, troughcolor=SURF0, font=("DejaVu Sans", 9))
        st.configure("TNotebook", background=CRUST, borderwidth=0)
        st.configure("TNotebook.Tab", background=SURF0, foreground=SUBTEXT, padding=(10, 4))
        st.map("TNotebook.Tab", background=[("selected", BG)], foreground=[("selected", SKY)])
        st.configure("Sub.TNotebook", background=MANTLE)
        st.configure("Sub.TNotebook.Tab", background=SURF0, foreground=SUBTEXT, padding=(7, 3))
        st.map("Sub.TNotebook.Tab", background=[("selected", BG)], foreground=[("selected", GREEN)])
        st.configure("TLabelframe", background=BG, bordercolor=SURF1)
        st.configure("TLabelframe.Label", background=BG, foreground=MAUVE, font=("DejaVu Sans", 9, "bold"))
        st.configure("TButton", background=SURF0, foreground=TEXT, padding=(8, 3))
        st.map("TButton", background=[("active", SURF1)])
        st.configure("TEntry", fieldbackground=SURF0, foreground=TEXT, insertcolor=TEXT)
        st.configure("TCombobox", fieldbackground=SURF0, foreground=TEXT, background=SURF0, arrowcolor=TEXT)
        st.map("TCombobox", fieldbackground=[("readonly", SURF0)], foreground=[("readonly", TEXT)])
        self.root.option_add("*TCombobox*Listbox.background", SURF0)
        self.root.option_add("*TCombobox*Listbox.foreground", TEXT)
        st.configure("TCheckbutton", background=BG, foreground=TEXT)
        st.configure("Horizontal.TScale", background=BG, troughcolor=SURF0)
        st.configure("Status.TLabel", background=CRUST, foreground=SUBTEXT, padding=(8, 2))

    def _on_tab(self, event):
        nb = event.widget
        try:
            cur = nb.nametowidget(nb.select())
        except Exception:
            return
        for t in self.tabs:
            if t.anim_on and str(t.frame) != str(cur):
                t.anim_stop()
        tab = self.frame2tab.get(str(cur))
        if tab is None:                      # wybrano zakładkę rozdziału → aktywna podzakładka
            for child in cur.winfo_children():
                if isinstance(child, ttk.Notebook) and child.select():
                    tab = self.frame2tab.get(str(child.nametowidget(child.select())))
        if tab is not None:
            tab.ensure_drawn()

    def _poll(self):
        try:
            while True:
                tab, gen, p, data, err = self.results.get_nowait()
                tab.finish(gen, p, data, err)
        except queue.Empty:
            pass
        self.root.after(50, self._poll)

    def set_status(self, s):
        self.status.set(s)


def export_markdown(path):
    """Zapisuje wszystkie wyjaśnienia do pliku Markdown."""
    parts = ["# Modele zaawansowane maszyn i stany nieustalone maszyn DC – wyjaśnienia\n",
             "_Wygenerowano z aplikacji `advanced_models_dc_transients_tkinter.py` "
             "(opcja `--export-md`). Te same teksty są widoczne w zakładkach aplikacji._\n"]
    for cls in [TabStart] + CH7 + CH8:
        parts.append(f"\n---\n\n# Zakładka: {cls.TITLE}\n")
        for line in cls.EXPLAIN.strip("\n").split("\n"):
            if line.startswith("» "):
                parts.append("    " + line[2:])
            elif line.startswith("## ") or line.startswith("### "):
                parts.append("\n" + line + "\n")
            elif line.startswith("⚙") or line.startswith("💡"):
                parts.append("\n> " + line + "\n")
            elif line.startswith("• "):
                parts.append("- " + line[2:])
            else:
                parts.append("\n" + line)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts) + "\n")


def main():
    if "--export-md" in sys.argv:
        k = sys.argv.index("--export-md")
        path = sys.argv[k + 1] if len(sys.argv) > k + 1 else "WYJASNIENIA_MODELE_ZAAWANSOWANE.md"
        export_markdown(path)
        print("Zapisano", path)
        return
    if not HAVE_TK:
        print("Brak tkinter/matplotlib – zainstaluj python3-tk oraz: pip install numpy scipy matplotlib")
        sys.exit(1)
    setup_matplotlib()
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
