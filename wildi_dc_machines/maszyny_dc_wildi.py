#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MASZYNY PRĄDU STAŁEGO – interaktywny podręcznik do przykładów z Wildiego
=========================================================================
(T. Wildi, „Electrical Machines, Drives and Power Systems",
 rozdz. „Direct-Current Generators" oraz „Direct-Current Motors")

Aplikacja zawiera WSZYSTKIE przykłady rachunkowe z załącznika:

  Prądnice (generatory) DC
    G1 – napięcie indukowane prądnicy 6-biegunowej (E0 = Z·n·Φ/60)
    G2 – prądy szczotek i cewek oraz napięcie na cewkę w prądnicy 12-biegunowej
  Silniki DC
    M1 – prąd rozruchowy i siła przeciwelektromotoryczna (SEM) silnika z magnesami
    M2 – prąd, moment i strumień silnika 225 kW
    M3 – układ Ward-Leonarda: praca silnikowa i hamowanie odzyskowe
    M4 – silnik bocznikowy: prądy, SEM, bilans mocy
    M5 – silnik szeregowy: charakterystyki w jednostkach względnych (pu), sprawność
    M6 – hamowanie dynamiczne, mechaniczna stała czasowa T0
    M7 – hamowanie przeciwprądem (plugging)

oraz zakładkę „Analiza sterowania", w której dla układów dynamicznych
(M1, M4, M3, M6, M7) wyznaczane są: transmitancja, postać zpk, bieguny/zera,
charakterystyki Bodego, Nyquista i odpowiedź skokowa (otwarta pętla oraz
zamknięta z regulatorem PI) – z automatycznym, słownym objaśnieniem wyników.

Wymagania:  Python ≥ 3.9, numpy, scipy, matplotlib, tkinter
Uruchomienie:  python maszyny_dc_wildi.py
"""

import textwrap
import tkinter as tk
from tkinter import ttk

import numpy as np
from scipy import signal
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure                                   # noqa: E402
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,      # noqa: E402
                                               NavigationToolbar2Tk)
from cycler import cycler                                              # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
#  STAŁE I KOLORY
# ─────────────────────────────────────────────────────────────────────────────
C = dict(bg="#1e1e2e", panel="#181825", surf="#313244", surf2="#45475a",
         fg="#cdd6f4", sub="#a6adc8", cyan="#89dceb", green="#a6e3a1",
         red="#f38ba8", yellow="#f9e2af", mauve="#cba6f7", peach="#fab387",
         blue="#89b4fa", teal="#94e2d5", pink="#f5c2e7")

TWO_PI = 2.0 * np.pi
RPM2RAD = TWO_PI / 60.0          # [rad/s] na 1 obr/min
RAD2RPM = 60.0 / TWO_PI          # [obr/min] na 1 rad/s
K_WILDI = (30.0 / np.pi) ** 2 / np.log(2.0)   # ≈ 131.5 – stała ze wzoru na T0
HP = 746.0                       # [W] 1 KM (koń mechaniczny, hp)

FONT = "DejaVu Sans"
MONO = "DejaVu Sans Mono"


def setup_matplotlib():
    """Ciemny motyw wykresów (Catppuccin Mocha)."""
    matplotlib.rcParams.update({
        "figure.facecolor": C["bg"], "axes.facecolor": C["panel"],
        "savefig.facecolor": C["bg"], "axes.edgecolor": C["sub"],
        "axes.labelcolor": C["fg"], "xtick.color": C["sub"],
        "ytick.color": C["sub"], "text.color": C["fg"],
        "grid.color": C["surf2"], "axes.grid": True, "grid.alpha": 0.55,
        "grid.linestyle": "--", "legend.facecolor": C["surf"],
        "legend.edgecolor": C["surf2"], "legend.fontsize": 8,
        "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
        "axes.titlecolor": C["cyan"], "lines.linewidth": 2.0,
        "axes.prop_cycle": cycler(color=[C["cyan"], C["green"], C["peach"],
                                         C["mauve"], C["yellow"], C["red"],
                                         C["blue"]]),
    })


def f4(x, n=4):
    """Krótki zapis liczby (n cyfr znaczących), z obsługą ±inf/nan."""
    if x is None:
        return "—"
    if isinstance(x, complex):
        if abs(x.imag) < 1e-12 * max(1.0, abs(x.real)):
            return f4(x.real, n)
        sign = "+" if x.imag >= 0 else "−"
        return f"{x.real:.{n}g} {sign} j{abs(x.imag):.{n}g}"
    if np.isinf(x):
        return "∞" if x > 0 else "−∞"
    if np.isnan(x):
        return "—"
    return f"{x:.{n}g}"


# ─────────────────────────────────────────────────────────────────────────────
#  „MINI-MARKUP" DLA PÓL TEKSTOWYCH  (objaśnienia i wyniki)
# ─────────────────────────────────────────────────────────────────────────────
#  "# "  nagłówek 1      "## " nagłówek 2     "$ "  wzór (żółty, mono)
#  "- "  punktor         "> "  wynik (zielony, pogrubiony)
#  "+ "  OK (zielony)    "! "  ostrzeżenie     "x "  błąd/niebezpieczeństwo
#  "~ "  przypis         pusta linia = nowy akapit, inne linie łączone w akapit
def setup_text_tags(t, size=10):
    t.configure(bg=C["panel"], fg=C["fg"], insertbackground=C["fg"],
                wrap="word", font=(FONT, size), padx=12, pady=8,
                relief="flat", highlightthickness=0, borderwidth=0)
    t.tag_configure("h1", font=(FONT, size + 3, "bold"), foreground=C["cyan"],
                    spacing1=10, spacing3=4)
    t.tag_configure("h2", font=(FONT, size + 1, "bold"), foreground=C["mauve"],
                    spacing1=8, spacing3=3)
    t.tag_configure("eq", font=(MONO, size, "bold"), foreground=C["yellow"],
                    lmargin1=26, lmargin2=26, spacing1=2, spacing3=2)
    t.tag_configure("li", lmargin1=12, lmargin2=28, spacing1=1)
    t.tag_configure("res", font=(MONO, size, "bold"), foreground=C["green"],
                    lmargin1=16, lmargin2=16, spacing1=2)
    t.tag_configure("ok", foreground=C["green"], lmargin1=12, lmargin2=28)
    t.tag_configure("warn", foreground=C["peach"], lmargin1=12, lmargin2=28)
    t.tag_configure("bad", foreground=C["red"], font=(FONT, size, "bold"),
                    lmargin1=12, lmargin2=28)
    t.tag_configure("note", foreground=C["sub"], font=(FONT, size - 1, "italic"),
                    lmargin1=12, lmargin2=12)
    t.tag_configure("p", spacing1=3, spacing3=3)


_PREFIX = [("## ", "h2", ""), ("# ", "h1", ""), ("$ ", "eq", ""),
           ("- ", "li", "•  "), ("> ", "res", "▶ "), ("+ ", "ok", "✔ "),
           ("! ", "warn", "⚠ "), ("x ", "bad", "✘ "), ("~ ", "note", "")]


def render_markup(t, markup):
    """Wypełnia widżet Text tekstem w mini-markupie."""
    t.configure(state="normal")
    t.delete("1.0", "end")
    para = []

    def flush():
        if para:
            t.insert("end", " ".join(para) + "\n", "p")
            para.clear()

    for raw in textwrap.dedent(markup).strip("\n").split("\n"):
        line = raw.strip()
        if not line:
            flush()
            continue
        for pre, tag, bullet in _PREFIX:
            if line.startswith(pre):
                flush()
                t.insert("end", bullet + line[len(pre):] + "\n", tag)
                break
        else:
            para.append(line)
    flush()
    t.configure(state="disabled")


# ─────────────────────────────────────────────────────────────────────────────
#  PANEL SUWAKÓW (przewijany)
# ─────────────────────────────────────────────────────────────────────────────
class ParamPanel(ttk.Frame):
    """Przewijana lista suwaków ttk.Scale + pola Entry do wpisania wartości."""

    def __init__(self, parent, on_change, height=260):
        super().__init__(parent)
        self.on_change = on_change
        self.canvas = tk.Canvas(self, bg=C["bg"], highlightthickness=0,
                                height=height)
        sb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas)
        self.inner.bind("<Configure>", lambda e: self.canvas.configure(
            scrollregion=self.canvas.bbox("all")))
        self._win = self.canvas.create_window((0, 0), window=self.inner,
                                              anchor="nw")
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(
            self._win, width=e.width))
        self.canvas.configure(yscrollcommand=sb.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self.canvas.bind("<Enter>", self._bind_wheel)
        self.canvas.bind("<Leave>", self._unbind_wheel)
        self.specs, self.vars = {}, {}

    # — kółko myszy —
    def _bind_wheel(self, _e):
        self.canvas.bind_all("<MouseWheel>", self._wheel)
        self.canvas.bind_all("<Button-4>", self._wheel)
        self.canvas.bind_all("<Button-5>", self._wheel)

    def _unbind_wheel(self, _e):
        for ev in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.canvas.unbind_all(ev)

    def _wheel(self, e):
        if getattr(e, "num", None) == 4 or getattr(e, "delta", 0) > 0:
            self.canvas.yview_scroll(-1, "units")
        else:
            self.canvas.yview_scroll(1, "units")

    @staticmethod
    def _fmt(v, integer):
        return str(int(round(v))) if integer else f"{v:.5g}"

    def add(self, key, label, lo, hi, init, unit="", integer=False):
        row = ttk.Frame(self.inner)
        row.pack(fill="x", padx=4, pady=3)
        txt = f"{label}  [{unit}]" if unit else label
        ttk.Label(row, text=txt).grid(row=0, column=0, columnspan=2, sticky="w")
        var = tk.DoubleVar(value=init)
        svar = tk.StringVar(value=self._fmt(init, integer))
        sc = ttk.Scale(row, from_=lo, to=hi, variable=var, orient="horizontal",
                       command=lambda _v, k=key: self._moved(k))
        sc.grid(row=1, column=0, sticky="ew")
        ent = ttk.Entry(row, textvariable=svar, width=10)
        ent.grid(row=1, column=1, padx=(6, 0))
        row.columnconfigure(0, weight=1)
        ent.bind("<Return>", lambda _e, k=key: self._typed(k))
        ent.bind("<FocusOut>", lambda _e, k=key: self._typed(k))
        self.specs[key] = (lo, hi, init, integer)
        self.vars[key] = (var, svar)

    def _moved(self, key):
        lo, hi, _init, integer = self.specs[key]
        var, svar = self.vars[key]
        v = float(np.clip(var.get(), lo, hi))
        if integer:
            v = float(round(v))
            var.set(v)
        svar.set(self._fmt(v, integer))
        self.on_change()

    def _typed(self, key):
        lo, hi, _init, integer = self.specs[key]
        var, svar = self.vars[key]
        try:
            v = float(svar.get().replace(",", "."))
        except ValueError:
            v = var.get()
        v = float(np.clip(v, lo, hi))
        if integer:
            v = float(round(v))
        if abs(v - var.get()) > 1e-15 or svar.get() != self._fmt(v, integer):
            var.set(v)
            svar.set(self._fmt(v, integer))
            self.on_change()

    def values(self):
        out = {}
        for k, (var, _s) in self.vars.items():
            lo, hi, _i, integer = self.specs[k]
            v = float(np.clip(var.get(), lo, hi))
            out[k] = int(round(v)) if integer else v
        return out

    def reset(self):
        for k, (lo, hi, init, integer) in self.specs.items():
            var, svar = self.vars[k]
            var.set(init)
            svar.set(self._fmt(init, integer))
        self.on_change()


# ─────────────────────────────────────────────────────────────────────────────
#  OBLICZENIA – czyste funkcje (używane przez zakładki i tabelę weryfikacji)
# ─────────────────────────────────────────────────────────────────────────────
def calc_g1(slots, turns, phi, n):
    """G1: napięcie indukowane prądnicy z uzwojeniem pętlicowym."""
    Z = slots * turns * 2                    # 1 zwój = 2 przewodniki
    E = Z * n * phi / 60.0                   # E0 = Z·n·Φ/60
    return dict(Z=Z, E=E)


def calc_g2(poles, coils, V, I):
    """G2: prądnica wielobiegunowa, uzwojenie pętlicowe (a = 2p ścieżek)."""
    brush_sets = poles                       # tyle zestawów szczotek, ile biegunów
    pos_sets = poles / 2
    I_brush = I / pos_sets                   # prąd jednego zestawu szczotek
    I_coil = I_brush / 2                     # szczotka zbiera prąd z 2 stron
    coils_between = coils / poles            # cewek między sąsiednimi szczotkami
    E_avg = V / coils_between
    paths = poles                            # liczba gałęzi równoległych
    return dict(brush_sets=brush_sets, pos_sets=pos_sets, I_brush=I_brush,
                I_coil=I_coil, coils_between=coils_between, E_avg=E_avg,
                paths=paths, P=V * I)


def calc_m1(Es, R, E_ref, n_ref, n):
    """M1: silnik z magnesami trwałymi – SEM i prąd przy prędkości n."""
    k = E_ref / n_ref                        # V na obr/min
    Eo = k * n
    I = (Es - Eo) / R
    K = E_ref / (n_ref * RPM2RAD)            # V·s/rad = N·m/A
    return dict(k=k, K=K, I_start=Es / R, Eo=Eo, I=I, n0=Es / k,
                T=K * I, P=Eo * I, Pcu=I ** 2 * R)


def calc_m2(P, V, n, coils, turns, slots, poles, D, L):
    """M2: znamionowy prąd, przewodniki/żłobek, moment, strumień."""
    I = P / V
    Z = 2 * coils * turns
    cond_slot = Z / slots
    T = P / (n * RPM2RAD)                    # T = 9,55·P/n
    phi = TWO_PI * T / (Z * I)               # Φ = 6,28·T/(Z·I)
    B_av = poles * phi / (np.pi * D * L)     # średnia indukcja w szczelinie
    I_cond = I / poles                       # pętlicowe: a = liczba biegunów
    A_lin = Z * I_cond / (np.pi * D)         # okładzina prądowa [A/m]
    return dict(I=I, Z=Z, cond_slot=cond_slot, T=T, phi=phi, B_av=B_av,
                I_cond=I_cond, A_lin=A_lin)


def calc_m3(Es, Eo, R, n_nom, E_nom):
    """M3: Ward-Leonard – prąd, moc, prędkość, moment."""
    I = (Es - Eo) / R
    P = Eo * I
    n = n_nom * Eo / E_nom
    T = P / (n * RPM2RAD) if n > 0 else 0.0
    return dict(I=I, P=P, n=n, T=T, Pin=Es * I, Ploss=I ** 2 * R)


def calc_m4(Es, IL, Rf, Ra, n):
    """M4: silnik bocznikowy – prądy, SEM, bilans mocy."""
    If = Es / Rf
    Ia = IL - If
    Eo = Es - Ia * Ra
    Pin = Es * IL
    Pf = Es * If
    Pcu = Ia ** 2 * Ra
    Pm = Eo * Ia
    T = Pm / (n * RPM2RAD)
    return dict(If=If, Ia=Ia, IR=Ia * Ra, Eo=Eo, Pin=Pin, Pf=Pf, Pa=Es * Ia,
                Pcu=Pcu, Pm=Pm, T=T, K=Eo / (n * RPM2RAD))


# Charakterystyki silnika szeregowego (Wildi, rys. 11) – przybliżone
# (odczytane/aproksymowane tak, by przechodziły przez punkty z przykładu:
#  T=0,4 pu → n=1,4 pu, I=0,6 pu  oraz punkt znamionowy T=1 → n=1, I=1)
SER_T = np.array([0.10, 0.20, 0.30, 0.40, 0.60, 0.80, 1.00, 1.20, 1.50, 2.00])
SER_N = np.array([2.80, 1.90, 1.60, 1.40, 1.20, 1.08, 1.00, 0.93, 0.85, 0.75])
SER_I = np.array([0.33, 0.45, 0.53, 0.60, 0.75, 0.88, 1.00, 1.12, 1.30, 1.60])
_ser_n = PchipInterpolator(SER_T, SER_N)
_ser_i = PchipInterpolator(SER_T, SER_I)


def calc_m5(hp, V, n_r, I_fl, TL):
    """M5: silnik szeregowy w jednostkach względnych (per-unit)."""
    PB = hp * HP
    TB = PB / (n_r * RPM2RAD)
    Tpu = TL / TB
    Tq = float(np.clip(Tpu, SER_T[0], SER_T[-1]))
    n_pu = float(_ser_n(Tq))
    I_pu = float(_ser_i(Tq))
    n = n_pu * n_r
    I = I_pu * I_fl
    Pi = V * I
    Po = n * RPM2RAD * TL
    return dict(PB=PB, TB=TB, Tpu=Tpu, n_pu=n_pu, I_pu=I_pu, n=n, I=I,
                Pi=Pi, Po=Po, eta=Po / Pi, clipped=(Tq != Tpu))


def calc_m6(J, n1, Es, Rb, Pw, n2):
    """M6: hamowanie dynamiczne – stała półtrwania T0 (Wildi, wzór 9)."""
    P1 = Es ** 2 / Rb                        # E0 ≈ Es w chwili przełączenia
    To = J * n1 ** 2 / (K_WILDI * P1)
    T = To / np.log(2.0)                     # klasyczna stała czasowa (36,8 %)
    halvings = np.log2(n1 / n2)
    t2 = To * halvings
    To_c = J * n1 ** 2 / (K_WILDI * Pw)      # tylko straty mech. i w żelazie
    t2c = To_c * halvings
    To_both = J * n1 ** 2 / (K_WILDI * (P1 + Pw))
    return dict(P1=P1, I1=Es / Rb, To=To, T=T, halvings=halvings, t2=t2,
                To_c=To_c, t2c=t2c, ratio=t2c / t2, To_both=To_both,
                t2_both=To_both * halvings)


def calc_m7(J, n1, Es, Rb):
    """M7: hamowanie przeciwprądem (plugging)."""
    Eo = Es                                  # tuż przed przełączeniem E0 ≈ Es
    U_net = Eo + Es
    I1 = U_net / Rb
    P_motor = Eo * I1                        # moc oddawana przez wirujące masy
    P_res = U_net * I1                       # moc wydzielana w rezystorze
    To = J * n1 ** 2 / (K_WILDI * P_motor)
    ts = 2 * To
    # rozwiązanie ścisłe (równanie różniczkowe): t_s = τ·ln(1 + E0/Es)
    w1 = n1 * RPM2RAD
    K = Eo / w1
    tau = J * Rb / K ** 2
    ts_exact = tau * np.log(1 + Eo / Es)
    return dict(Eo=Eo, U_net=U_net, I1=I1, P_motor=P_motor, P_res=P_res,
                To=To, ts=ts, tau=tau, K=K, ts_exact=ts_exact,
                I_stop=Es / Rb)


# ─────────────────────────────────────────────────────────────────────────────
#  BAZOWA ZAKŁADKA PRZYKŁADU
# ─────────────────────────────────────────────────────────────────────────────
class ExampleTab(ttk.Frame):
    """Szablon: suwaki | wyniki krok-po-kroku | wykres | objaśnienie."""
    tab_title = "Przykład"
    params = []          # (klucz, etykieta, min, max, wartość_książkowa, jedn., int?)
    explanation = ""

    def __init__(self, nb, app):
        super().__init__(nb)
        self.app = app
        self._job = None
        self._marks = []

        paned = ttk.PanedWindow(self, orient="horizontal")
        paned.pack(fill="both", expand=True)
        left = ttk.Frame(paned)
        right = ttk.PanedWindow(paned, orient="vertical")
        paned.add(left, weight=0)
        paned.add(right, weight=1)

        lf = ttk.LabelFrame(left, text=" Parametry – przesuń suwak lub wpisz wartość ")
        lf.pack(fill="x", padx=6, pady=(6, 3))
        self.pp = ParamPanel(lf, self.schedule,
                             height=min(56 * max(len(self.params), 1) + 8, 340))
        self.pp.pack(fill="both", expand=True, padx=2, pady=2)
        for spec in self.params:
            self.pp.add(*spec)
        bf = ttk.Frame(left)
        bf.pack(fill="x", padx=6)
        ttk.Button(bf, text="↺  Dane z książki", command=self.pp.reset).pack(
            side="left", padx=2, pady=3)
        ttk.Label(bf, text="LPM na wykresie = znacznik, PPM = usuń",
                  style="Sub.TLabel").pack(side="left", padx=8)

        rf = ttk.LabelFrame(left, text=" Rozwiązanie krok po kroku (na żywo) ")
        rf.pack(fill="both", expand=True, padx=6, pady=(3, 6))
        self.res = tk.Text(rf, width=52, height=12)
        setup_text_tags(self.res, size=10)
        rsb = ttk.Scrollbar(rf, orient="vertical", command=self.res.yview)
        self.res.configure(yscrollcommand=rsb.set)
        self.res.pack(side="left", fill="both", expand=True)
        rsb.pack(side="right", fill="y")

        ff = ttk.Frame(right)
        self.fig = Figure(figsize=(8, 5), dpi=100, layout="constrained")
        self.canvas = FigureCanvasTkAgg(self.fig, master=ff)
        self.toolbar = NavigationToolbar2Tk(self.canvas, ff, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(side="bottom", fill="x")
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
        self.canvas.mpl_connect("button_press_event", self._on_click)

        ef = ttk.LabelFrame(right, text=" Objaśnienie (dla ucznia liceum) + komentarz inżyniera ")
        self.exp = tk.Text(ef, height=10)
        setup_text_tags(self.exp, size=10)
        esb = ttk.Scrollbar(ef, orient="vertical", command=self.exp.yview)
        self.exp.configure(yscrollcommand=esb.set)
        self.exp.pack(side="left", fill="both", expand=True)
        esb.pack(side="right", fill="y")
        render_markup(self.exp, self.explanation)

        right.add(ff, weight=3)
        right.add(ef, weight=2)
        self._paned, self._right = paned, right
        self.after(300, self._init_sashes)
        self.update_all()

    def _init_sashes(self):
        try:
            self._paned.sashpos(0, 440)
            h = self._right.winfo_height()
            if h > 100:
                self._right.sashpos(0, int(h * 0.60))
        except tk.TclError:
            pass

    # — interakcja —
    def _on_click(self, ev):
        if ev.inaxes is None or str(self.toolbar.mode):
            return
        if ev.button == 1:
            a = ev.inaxes.annotate(
                f"x = {ev.xdata:.4g}\ny = {ev.ydata:.4g}", (ev.xdata, ev.ydata),
                xytext=(12, 12), textcoords="offset points", fontsize=8,
                color=C["bg"], bbox=dict(boxstyle="round", fc=C["yellow"], alpha=0.9),
                arrowprops=dict(arrowstyle="->", color=C["yellow"]))
            self._marks.append(a)
        elif ev.button == 3:
            for a in self._marks:
                try:
                    a.remove()
                except (ValueError, AttributeError):
                    pass
            self._marks.clear()
        self.canvas.draw_idle()

    def schedule(self, *_):
        if self._job is not None:
            self.after_cancel(self._job)
        self._job = self.after(70, self.update_all)

    def update_all(self):
        self._job = None
        p = self.pp.values()
        try:
            render_markup(self.res, self.solve(p))
        except Exception as e:                       # noqa: BLE001
            render_markup(self.res, f"x Błąd obliczeń: {e}")
        try:
            self._marks.clear()
            self.fig.clear()
            self.draw(p)
            self.canvas.draw_idle()
            self.app.status(f"{self.tab_title}: zaktualizowano.")
        except Exception as e:                       # noqa: BLE001
            self.app.status(f"Błąd rysowania ({self.tab_title}): {e}")

    # do nadpisania
    def solve(self, p):
        return ""

    def draw(self, p):
        pass


def mark(ax, x, y, text, color=None, dx=10, dy=10):
    """Punkt pracy z podpisem."""
    color = color or C["yellow"]
    ax.plot([x], [y], "o", color=color, ms=8, zorder=5,
            markeredgecolor=C["bg"])
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points",
                fontsize=8, color=color,
                bbox=dict(boxstyle="round,pad=0.25", fc=C["surf"], ec=color, alpha=0.9))


# ─────────────────────────────────────────────────────────────────────────────
#  G1 – NAPIĘCIE INDUKOWANE
# ─────────────────────────────────────────────────────────────────────────────
class TabG1(ExampleTab):
    tab_title = "G1 – napięcie prądnicy"
    tab_short = "G1 · E₀"
    params = [("slots", "Liczba żłobków (= liczba cewek)", 10, 200, 90, "", True),
              ("turns", "Zwoje na cewkę", 1, 20, 4, "", True),
              ("phi", "Strumień na biegun Φ", 0.005, 0.12, 0.04, "Wb"),
              ("n", "Prędkość obrotowa n", 0, 2000, 600, "obr/min"),
              ("poles", "Liczba biegunów (pętlicowe: nie wpływa na E0)", 2, 24, 6, "", True)]
    explanation = """
    # Przykład G1 – ile woltów wytworzy prądnica?
    Treść (Wildi): Twornik 6-biegunowej prądnicy obracającej się z prędkością 600 obr/min
    ma 90 żłobków. Każda cewka ma 4 zwoje, a strumień na biegun wynosi 0,04 Wb.
    Oblicz napięcie indukowane.

    ## 1. Fizyka „na chłopski rozum"
    Gdy przewodnik (drut) porusza się w polu magnetycznym i „przecina" linie pola,
    na jego końcach pojawia się napięcie – to prawo indukcji Faradaya, to samo, które
    działa w dynamie rowerowym. Im szybciej tniesz linie pola i im jest ich więcej,
    tym większe napięcie. Prądnica to po prostu setki takich drutów ułożonych w żłobkach
    obracającego się walca (twornika) i połączonych tak, by ich napięcia się sumowały.
    W pojedynczym drucie napięcie jest zmienne (raz jest pod biegunem N, raz pod S),
    ale komutator – obracający się pierścień z wycinków miedzi – działa jak mechaniczny
    prostownik i „przełącza" przewody tak, by na szczotkach napięcie miało stały znak.

    ## 2. Wzór (uzwojenie pętlicowe, szczotki w strefie neutralnej)
    $ E₀ = Z · n · Φ / 60
    - E₀ – napięcie między szczotkami [V]
    - Z – liczba wszystkich przewodników na tworniku
    - n – prędkość [obr/min]; dzielimy przez 60, żeby przejść na obroty na sekundę
    - Φ – strumień magnetyczny jednego bieguna [Wb] (ile „linii pola" wychodzi z bieguna)

    ## 3. Rozwiązanie krok po kroku
    - Krok 1. Każda cewka ma dwa boki: jeden leży na górze jednego żłobka, drugi na dole innego.
      W każdym żłobku mieszczą się więc dwa boki → cewek jest tyle, ile żłobków: 90.
    - Krok 2. Jeden zwój to pętla – ma DWA przewodniki czynne (idący „tam" i „z powrotem").
    $ Z = 90 cewek × 4 zwoje × 2 przewodniki = 720
    - Krok 3. Podstawiamy:
    $ E₀ = 720 × 600 × 0,04 / 60 = 288 V
    > Wynik z książki: 288 V (bez obciążenia, szczotki w strefie neutralnej)

    ## 4. Dlaczego we wzorze nie ma liczby biegunów?
    Ogólny wzór to E₀ = (p/a)·Z·n·Φ/60, gdzie p – liczba biegunów, a – liczba gałęzi
    równoległych uzwojenia. W uzwojeniu pętlicowym a = p, więc p/a = 1 i bieguny „skracają się".
    Więcej biegunów daje więcej przecięć linii pola na obrót, ale jednocześnie przewodniki
    dzielą się na więcej równoległych gałęzi – efekty się znoszą. (W uzwojeniu falistym a = 2
    i wynik byłby p/2 = 3 razy większy!)

    ## 5. Komentarz inżyniera – sprawdzenie rzędu wielkości
    W jednej gałęzi jest Z/a = 720/6 = 120 przewodników w szeregu, więc średnio na jeden
    przewodnik przypada 288/120 = 2,4 V. To rozsądne: przewodnik przecina na obrót strumień
    p·Φ = 0,24 Wb, a obrotów na sekundę jest 10 → 2,4 V. Sprawdzenie zgadza się z Faradayem
    (e = dΦ/dt). Napięcie jest wprost proporcjonalne do n i Φ – to dwa „pokrętła", którymi
    w praktyce regulujemy prądnicę (prędkość napędu i prąd wzbudzenia).

    ## 6. Co pokazują wykresy?
    - Lewy: E₀(n) – prosta przez zero. Nachylenie zależy od Φ (pokazane Φ·0,5 i Φ·1,5).
      W rzeczywistej maszynie Φ nie rośnie liniowo z prądem wzbudzenia (nasycenie żelaza).
    - Prawy: dlaczego napięcie jest „gładkie". Jedna cewka daje napięcie przemienne;
      komutator je prostuje (pulsuje od 0 do max); wiele cewek przesuniętych w czasie
      sumuje się w prawie stałe napięcie – tętnienia maleją z liczbą cewek.

    ## 7. Eksperymentuj
    - Podwój prędkość → napięcie się podwaja. Zmniejsz Φ o połowę → napięcie spada o połowę.
    - Zmień liczbę biegunów – E₀ się nie zmienia (pętlica!), zmienia się tylko liczba gałęzi.
    """

    def solve(self, p):
        r = calc_g1(p["slots"], p["turns"], p["phi"], p["n"])
        per_path = r["Z"] / p["poles"]
        e_c = r["E"] / per_path if per_path else 0
        return f"""
        # Dane
        - żłobki = cewki = {p['slots']},  zwoje/cewkę = {p['turns']}
        - Φ = {p['phi']:.4g} Wb,  n = {p['n']:.0f} obr/min,  2p = {p['poles']}
        ## Krok 1 – liczba przewodników
        $ Z = {p['slots']} × {p['turns']} × 2 = {r['Z']}
        ## Krok 2 – napięcie
        $ E₀ = Z·n·Φ/60 = {r['Z']}·{p['n']:.0f}·{p['phi']:.4g}/60
        > E₀ = {r['E']:.2f} V
        ## Kontrola fizyczna
        - przewodników w jednej gałęzi: Z/a = {per_path:.0f}
        - średnie napięcie na przewodnik: {e_c:.3g} V
        - częstotliwość przemagnesowania twornika: f = p·n/120 = {p['poles'] * p['n'] / 120:.3g} Hz
        ~ Książka: 288 V dla danych domyślnych.
        """

    def draw(self, p):
        r = calc_g1(p["slots"], p["turns"], p["phi"], p["n"])
        ax1, ax2 = self.fig.subplots(1, 2)
        n = np.linspace(0, 2000, 200)
        for f, ls in ((0.5, ":"), (1.0, "-"), (1.5, "--")):
            ax1.plot(n, r["Z"] * n * p["phi"] * f / 60, ls,
                     label=f"Φ = {p['phi'] * f:.3g} Wb")
        mark(ax1, p["n"], r["E"], f"{r['E']:.1f} V")
        ax1.set(title="E₀ = Z·n·Φ/60", xlabel="n [obr/min]", ylabel="E₀ [V]")
        ax1.legend()

        th = np.linspace(0, 4 * np.pi, 1500)
        E = r["E"]
        ax2.plot(np.degrees(th), E * np.pi / 2 * np.sin(th), ":", color=C["sub"],
                 lw=1.2, label="1 cewka bez komutatora (AC)")
        k_series = max(int(round(p["slots"] / p["poles"])), 1)
        for k, col in ((1, C["peach"]), (2, C["mauve"]), (k_series, C["green"])):
            s = sum(np.abs(np.sin(th - j * np.pi / k)) for j in range(k))
            s = s / s.mean() * E
            rip = (s.max() - s.min()) / E * 100
            ax2.plot(np.degrees(th), s, color=col,
                     label=f"{k} cewk{'a' if k == 1 else 'i'} z komutatorem (tętnienia {rip:.1f} %)")
        ax2.axhline(E, color=C["yellow"], lw=1, ls="--")
        ax2.set(title="Rola komutatora i liczby cewek", xlabel="kąt obrotu [°]",
                ylabel="napięcie [V]", xlim=(0, 720))
        ax2.legend(loc="lower right")


# ─────────────────────────────────────────────────────────────────────────────
#  G2 – PRĄDNICA 12-BIEGUNOWA
# ─────────────────────────────────────────────────────────────────────────────
class TabG2(ExampleTab):
    tab_title = "G2 – prądnica 12-biegunowa"
    tab_short = "G2 · szczotki"
    params = [("poles", "Liczba biegunów", 2, 24, 12, "", True),
              ("coils", "Liczba cewek (wycinków komutatora)", 12, 240, 72, "", True),
              ("V", "Napięcie między sąsiednimi szczotkami", 20, 600, 240, "V"),
              ("I", "Prąd obciążenia", 100, 5000, 2400, "A")]
    explanation = """
    # Przykład G2 – kto niesie te 2400 A?
    Treść (Wildi, rys. 38): Prądnica 12-biegunowa z 72 cewkami wytwarza 240 V między
    sąsiednimi szczotkami i oddaje do obciążenia 2400 A. Oblicz: a) prąd na jeden zestaw
    szczotek, b) prąd płynący w każdej cewce, c) średnie napięcie indukowane na cewkę.

    ## 1. Intuicja
    Duże prądnice mają wiele par biegunów (tu 6 par = 12 biegunów). Pod każdym biegunem
    jest komplet szczotek: na przemian „+" i „−". Wszystkie szczotki „+" łączy się razem
    (zacisk dodatni), a wszystkie „−" razem (zacisk ujemny). To jak 12 baterii połączonych
    równolegle – każda daje to samo napięcie, a prądy się dodają.

    ## 2. Rozwiązanie
    - a) Zestawów szczotek jest tyle co biegunów: 12, z czego 6 dodatnich. Prąd 2400 A
      wypływa z zacisku „+", czyli dzieli się na 6 szczotek:
    $ I_szczotki = 2400 / 6 = 400 A
    - b) Każda szczotka dodatnia zbiera prąd z cewek po lewej i po prawej stronie
      (dwie gałęzie spotykają się na szczotce):
    $ I_cewki = 400 / 2 = 200 A
    - c) Między sąsiednimi szczotkami leży 72/12 = 6 cewek połączonych szeregowo,
      których napięcia sumują się do 240 V:
    $ E_śr = 240 / 6 = 40 V na cewkę
    > Książka: 400 A, 200 A, 40 V

    ## 3. Sprawdzenie (bilans)
    Uzwojenie pętlicowe ma tyle gałęzi równoległych co biegunów: 12 gałęzi × 200 A = 2400 A ✔.
    Moc oddawana: P = 240 V × 2400 A = 576 kW – to już maszyna przemysłowa (huta, galwanizernia).

    ## 4. Komentarz inżyniera
    - Napięcie „40 V na cewkę" jest wartością ŚREDNIĄ. Cewka pod środkiem bieguna ma
      większe napięcie (przy rozkładzie sinusoidalnym ok. π/2·40 ≈ 63 V), a cewka w strefie
      neutralnej – prawie zero. Właśnie cewki z zerowym napięciem są zwierane przez szczotki
      (komutacja bez iskrzenia).
    - Napięcie między sąsiednimi wycinkami komutatora projektuje się zwykle poniżej
      ~ 20–30 V (średnio), bo wyższe grozi przeskokiem łuku po komutatorze („flashover").
      Przy 40 V średnio projektant powinien sprawdzić izolację i uzwojenie kompensacyjne.
    - Prąd szczotki 400 A wymaga wielu szczotek równolegle (typowo 6–12 A/cm² węgla).

    ## 5. Co pokazują wykresy?
    - Lewy: przybliżony rozkład napięć w 6 cewkach między szczotkami (sinus indukcji pod
      biegunem). Suma słupków = napięcie między szczotkami, linia przerywana = średnia.
    - Prawy: jak prąd szczotki i prąd cewki maleją, gdy rośnie liczba biegunów (przy tym
      samym prądzie całkowitym). To jeden z powodów budowy maszyn wielobiegunowych.
    """

    def solve(self, p):
        r = calc_g2(p["poles"], p["coils"], p["V"], p["I"])
        return f"""
        # Dane
        - 2p = {p['poles']}, cewki = {p['coils']}, U = {p['V']:.0f} V, I = {p['I']:.0f} A
        ## a) prąd zestawu szczotek
        $ zestawy szczotek = {r['brush_sets']}, dodatnich = {r['pos_sets']:.0f}
        $ I_sz = {p['I']:.0f}/{r['pos_sets']:.0f}
        > I_szczotki = {r['I_brush']:.1f} A
        ## b) prąd cewki
        $ I_c = I_sz/2
        > I_cewki = {r['I_coil']:.1f} A
        ## c) napięcie na cewkę
        $ cewek między szczotkami = {p['coils']}/{p['poles']} = {r['coils_between']:.3g}
        > E_śr = {r['E_avg']:.2f} V
        ## Kontrola
        - gałęzie równoległe a = {r['paths']}:  a·I_c = {r['paths'] * r['I_coil']:.0f} A
        - moc P = U·I = {r['P'] / 1e3:.1f} kW
        - szczyt napięcia cewki ≈ π/2·E_śr = {np.pi / 2 * r['E_avg']:.1f} V
        """ + ("! Liczba cewek nie dzieli się przez liczbę biegunów – w praktyce uzwojenie "
               "byłoby niesymetryczne." if p["coils"] % p["poles"] else "")

    def draw(self, p):
        r = calc_g2(p["poles"], p["coils"], p["V"], p["I"])
        ax1, ax2 = self.fig.subplots(1, 2)
        N = max(int(round(r["coils_between"])), 1)
        k = np.arange(N)
        w = np.sin(np.pi * (k + 0.5) / N)
        e = p["V"] * w / w.sum()
        ax1.bar(k + 1, e, color=C["cyan"], edgecolor=C["bg"])
        ax1.axhline(r["E_avg"], color=C["yellow"], ls="--",
                    label=f"średnio {r['E_avg']:.1f} V")
        for i, v in zip(k + 1, e):
            ax1.text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
        ax1.set(title=f"Napięcia {N} cewek między szczotkami (Σ = {p['V']:.0f} V)",
                xlabel="numer cewki (od szczotki „−” do „+”)", ylabel="napięcie cewki [V]")
        ax1.legend()

        P = np.arange(2, 26, 2)
        ax2.plot(P, p["I"] / (P / 2), "o-", label="prąd zestawu szczotek")
        ax2.plot(P, p["I"] / P, "s-", label="prąd cewki (gałęzi)")
        mark(ax2, p["poles"], r["I_brush"], f"{r['I_brush']:.0f} A")
        mark(ax2, p["poles"], r["I_coil"], f"{r['I_coil']:.0f} A", C["green"], dy=-22)
        ax2.set(title=f"Podział prądu {p['I']:.0f} A", xlabel="liczba biegunów 2p",
                ylabel="prąd [A]", yscale="log")
        ax2.legend()


# ─────────────────────────────────────────────────────────────────────────────
#  M1 – ROZRUCH, SEM
# ─────────────────────────────────────────────────────────────────────────────
class TabM1(ExampleTab):
    tab_title = "M1 – rozruch i SEM"
    tab_short = "M1 · rozruch"
    params = [("Es", "Napięcie zasilania Es", 10, 300, 150, "V"),
              ("R", "Rezystancja twornika R", 0.1, 5, 1.0, "Ω"),
              ("E_ref", "SEM wzorcowa (jako prądnica)", 10, 200, 50, "V"),
              ("n_ref", "… przy prędkości", 100, 3000, 500, "obr/min"),
              ("n", "Prędkość do obliczeń", 0, 1500, 1000, "obr/min"),
              ("L", "Indukcyjność twornika L (założona)", 0.001, 0.2, 0.02, "H"),
              ("J", "Moment bezwładności J (założony)", 0.005, 0.5, 0.05, "kg·m²"),
              ("TL", "Moment obciążenia (symulacja)", 0, 5, 0.5, "N·m")]
    explanation = """
    # Przykład M1 – dlaczego silnik przy rozruchu „pobiera" ogromny prąd?
    Treść (Wildi): Twornik prądnicy z magnesami trwałymi ma rezystancję 1 Ω i wytwarza 50 V
    przy 500 obr/min. Twornik podłączono do źródła 150 V (pracuje teraz jako silnik). Oblicz:
    a) prąd rozruchowy, b) SEM przy 1000 i 1460 obr/min, c) prąd twornika przy tych prędkościach.

    ## 1. Kluczowa idea: silnik to też prądnica!
    Kiedy wirnik silnika się kręci, jego przewody tną pole magnesów – więc indukuje się
    w nich napięcie, dokładnie jak w prądnicy. To napięcie działa PRZECIWNIE do napięcia
    zasilania, dlatego nazywa się je siłą przeciwelektromotoryczną (SEM, ang. cemf).
    Prąd płynie pod wpływem RÓŻNICY napięć:
    $ I = (Es − E₀) / R          E₀ = k · n
    W chwili startu wirnik stoi, E₀ = 0, a prąd ogranicza tylko mała rezystancja uzwojenia.

    ## 2. Rozwiązanie
    - a) Rozruch: E₀ = 0
    $ I_rozr = 150 V / 1 Ω = 150 A
    - b) SEM jest proporcjonalna do prędkości: 50 V przy 500 obr/min → k = 0,1 V/(obr/min)
    $ E₀(1000) = 100 V,     E₀(1460) = 146 V
    - c) Prąd:
    $ I(1000) = (150 − 100)/1 = 50 A,    I(1460) = (150 − 146)/1 = 4 A
    > Książka: 150 A; 100 V i 146 V; 50 A i 4 A

    ## 3. Komentarz inżyniera
    - Prąd rozruchowy jest ~37 razy większy niż prąd przy 1460 obr/min. W prawdziwych
      maszynach bywa to 20–30× prądu znamionowego – bezpieczniki by zadziałały, a komutator
      iskrzyłby niszcząco. Dlatego stosuje się rozruszniki (rezystor szeregowy stopniowo
      zwierany, p. rozdz. 14) lub przekształtnik ograniczający prąd.
    - Prędkość biegu jałowego (I ≈ 0): n₀ = Es/k = 1500 obr/min. Silnik „sam" ustala prędkość,
      przy której SEM prawie równa się napięciu zasilania – to naturalne sprzężenie zwrotne.
    - Moc mechaniczna wewnętrzna P = E₀·I (przy 1460 obr/min: 146·4 = 584 W), straty w miedzi I²R = 16 W.

    ## 4. Model dynamiczny (prawe wykresy)
    Żeby zobaczyć PRZEBIEG rozruchu w czasie, potrzebne są dwa równania różniczkowe:
    $ L·di/dt = Es − R·i − K·ω          (obwód elektryczny)
    $ J·dω/dt = K·i − T_obc             (II zasada dynamiki dla ruchu obrotowego)
    gdzie K = E₀/ω = 0,955 V·s/rad (= N·m/A). L i J nie są podane w książce – przyjęto
    rozsądne wartości (można je zmieniać). Widać, że prąd startuje od ~150 A
    (zaokrąglone przez indukcyjność) i maleje, a prędkość rośnie wykładniczo.
    Stała czasowa elektromechaniczna τm = J·R/K² decyduje o czasie rozpędzania.

    ## 5. Eksperymentuj
    - Zwiększ R (rezystor rozruchowy) → mniejszy prąd szczytowy, wolniejszy rozruch.
    - Zwiększ J (koło zamachowe) → dłuższy rozruch, ale prąd szczytowy prawie bez zmian.
    """

    def solve(self, p):
        r = calc_m1(p["Es"], p["R"], p["E_ref"], p["n_ref"], p["n"])
        r1 = calc_m1(p["Es"], p["R"], p["E_ref"], p["n_ref"], 1000)
        r2 = calc_m1(p["Es"], p["R"], p["E_ref"], p["n_ref"], 1460)
        tau_m = p["J"] * p["R"] / r["K"] ** 2
        tau_e = p["L"] / p["R"]
        warn = ("! Przy tej prędkości E₀ > Es: maszyna pracuje jako PRĄDNICA (I < 0)."
                if r["I"] < 0 else "")
        return f"""
        # a) Prąd rozruchowy (n = 0, E₀ = 0)
        $ I_rozr = Es/R = {p['Es']:.4g}/{p['R']:.4g}
        > I_rozr = {r['I_start']:.2f} A
        # b), c) SEM i prąd
        $ k = {p['E_ref']:.4g}/{p['n_ref']:.4g} = {r['k']:.4g} V/(obr/min)
        $ E₀(1000) = {r1['Eo']:.2f} V   I = {r1['I']:.2f} A
        $ E₀(1460) = {r2['Eo']:.2f} V   I = {r2['I']:.2f} A
        ## Dla suwaka n = {p['n']:.0f} obr/min
        > E₀ = {r['Eo']:.2f} V,  I = {r['I']:.2f} A
        - moment T = K·I = {r['T']:.3g} N·m,  P_mech = E₀·I = {r['P']:.4g} W
        - straty I²R = {r['Pcu']:.4g} W
        {warn}
        ## Parametry dynamiczne
        - K = {r['K']:.4g} V·s/rad,  n₀ (bieg jałowy) = {r['n0']:.0f} obr/min
        - τe = L/R = {tau_e * 1e3:.3g} ms,  τm = J·R/K² = {tau_m * 1e3:.3g} ms
        - stosunek prądu rozruchowego do prądu przy 1460 obr/min: {r['I_start'] / max(r2['I'], 1e-9):.1f}×
        """

    def draw(self, p):
        r = calc_m1(p["Es"], p["R"], p["E_ref"], p["n_ref"], p["n"])
        axs = self.fig.subplot_mosaic([["IE", "si"], ["IE", "sn"]])
        ax = axs["IE"]
        n = np.linspace(0, max(r["n0"] * 1.05, 10), 300)
        E = r["k"] * n
        I = (p["Es"] - E) / p["R"]
        ax.plot(n, I, label="prąd twornika I(n)")
        ax.set(title="Prąd i SEM w funkcji prędkości", xlabel="n [obr/min]",
               ylabel="I [A]")
        axb = ax.twinx()
        axb.plot(n, E, color=C["green"], ls="--", label="SEM E₀(n)")
        axb.axhline(p["Es"], color=C["red"], lw=1, ls=":", label="Es")
        axb.set_ylabel("E₀ [V]", color=C["green"])
        axb.grid(False)
        for nn in (0, 1000, 1460):
            rr = calc_m1(p["Es"], p["R"], p["E_ref"], p["n_ref"], nn)
            mark(ax, nn, rr["I"], f"{nn} obr/min\nI={rr['I']:.0f} A", C["peach"])
        mark(ax, p["n"], r["I"], f"suwak: {r['I']:.1f} A", dy=-30)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = axb.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="upper right")

        K, R, L, J = r["K"], p["R"], p["L"], p["J"]
        tau_m = J * R / K ** 2
        t_end = 6 * tau_m + 6 * L / R

        def f(_t, x):
            i, w = x
            return [(p["Es"] - R * i - K * w) / L, (K * i - p["TL"] * np.sign(w)) / J]

        sol = solve_ivp(f, (0, t_end), [0, 0], max_step=t_end / 800, rtol=1e-6)
        t = sol.t * 1e3
        axs["si"].plot(t, sol.y[0], color=C["peach"])
        axs["si"].axhline(r["I_start"], color=C["red"], ls=":", lw=1,
                          label=f"Es/R = {r['I_start']:.0f} A")
        axs["si"].set(title="Symulacja rozruchu (ODE)", ylabel="i [A]")
        axs["si"].legend()
        axs["sn"].plot(t, sol.y[1] * RAD2RPM, color=C["green"])
        for nn in (1000, 1460):
            if nn < sol.y[1].max() * RAD2RPM:
                axs["sn"].axhline(nn, color=C["sub"], ls=":", lw=1)
        axs["sn"].set(xlabel="t [ms]", ylabel="n [obr/min]")


# ─────────────────────────────────────────────────────────────────────────────
#  M2 – SILNIK 225 kW
# ─────────────────────────────────────────────────────────────────────────────
class TabM2(ExampleTab):
    tab_title = "M2 – moment i strumień"
    tab_short = "M2 · moment"
    params = [("P", "Moc znamionowa P", 10e3, 500e3, 225e3, "W"),
              ("V", "Napięcie U", 50, 600, 250, "V"),
              ("n", "Prędkość n", 100, 3000, 1200, "obr/min"),
              ("coils", "Cewki twornika", 20, 500, 243, "", True),
              ("turns", "Zwoje na cewkę", 1, 10, 1, "", True),
              ("slots", "Żłobki", 10, 200, 81, "", True),
              ("poles", "Bieguny", 2, 16, 6, "", True),
              ("D", "Średnica twornika D", 0.1, 1.5, 0.559, "m"),
              ("L", "Długość osiowa twornika", 0.05, 1.0, 0.235, "m")]
    explanation = """
    # Przykład M2 – jak duży moment daje silnik 225 kW?
    Treść (Wildi, rys. 4–5): silnik 225 kW (≈300 KM), 250 V, 1200 obr/min; 243 cewki po 1 zwoju,
    uzwojenie pętlicowe, 81 żłobków, 243 wycinki komutatora, 6 biegunów, średnica twornika
    559 mm, długość 235 mm. Oblicz: a) prąd znamionowy twornika, b) liczbę przewodników
    w żłobku, c) strumień na biegun.

    ## 1. Moc, moment i prędkość – jak w rowerze
    Moc to „moment × prędkość kątowa": P = T·ω. Na rowerze możesz jechać szybko z małą siłą
    na pedałach albo wolno z dużą – moc może być ta sama. Silnik zamienia moc elektryczną
    E₀·I na mechaniczną T·ω. Stąd dwa podstawowe wzory:
    $ P = E₀ · I                      T = 9,55 · P / n
    $ T = Z · Φ · I / (2π)          →   Φ = 2π · T / (Z · I)
    (9,55 = 60/2π – przelicznik obr/min na rad/s.)

    ## 2. Rozwiązanie
    - a) Spadek napięcia na rezystancji twornika jest mały, więc E₀ ≈ U = 250 V:
    $ I = P / E₀ = 225 000 / 250 = 900 A
    - b) Każda cewka (1 zwój) ma 2 przewodniki: Z = 243 × 2 = 486. Na żłobek:
    $ 486 / 81 = 6 przewodników (6 boków cewek) w żłobku
    - c) Moment i strumień:
    $ T = 9,55 × 225 000 / 1200 = 1791 N·m
    $ Φ = 6,28 × 1791 / (486 × 900) = 25,7 mWb
    > Książka: 900 A, 6 przewodników/żłobek, 1791 N·m, 25,7 mWb

    ## 3. Komentarz inżyniera – czy ten silnik jest „rozsądnie" zaprojektowany?
    - Średnia indukcja w szczelinie B_śr = 2p·Φ / (π·D·L) ≈ 0,37 T (pod biegunem ok. 0,55–0,6 T,
      bo biegun obejmuje ~65 % podziałki). Typowo 0,6–0,9 T pod biegunem – wartość w normie.
    - Okładzina prądowa (ile amperów przewodników przypada na metr obwodu twornika):
      A = Z·I_przew/(π·D) ≈ 41 kA/m – typowo 25–50 kA/m dla maszyn tej mocy ✔.
    - Moment 1791 N·m to tyle, co siła ~180 kG na ramieniu 1 m – a silnik waży ok. tony.
    - Ten sam strumień możemy policzyć „z drugiej strony": Φ = 60·E₀/(Z·n) = 25,7 mWb ✔
      (dwa niezależne wzory – zgodność potwierdza wynik).

    ## 4. Co pokazują wykresy?
    - Lewy: moment rośnie liniowo z prądem (przy stałym Φ) – T = (ZΦ/2π)·I.
    - Prawy: hiperbola stałej mocy T = 9,55·P/n – przy tej samej mocy niższa prędkość
      oznacza większy moment (przekładnia, dźwigi, walcarki).
    """

    def solve(self, p):
        r = calc_m2(**p)
        phi2 = 60 * p["V"] / (r["Z"] * p["n"])
        return f"""
        # a) prąd znamionowy (E₀ ≈ U)
        $ I = P/E₀ = {p['P']:.0f}/{p['V']:.0f}
        > I = {r['I']:.1f} A
        # b) przewodniki w żłobku
        $ Z = 2·{p['coils']}·{p['turns']} = {r['Z']}
        > Z/żłobki = {r['cond_slot']:.3g}
        # c) moment i strumień
        $ T = 9,55·P/n = {r['T']:.1f} N·m
        $ Φ = 2π·T/(Z·I)
        > Φ = {r['phi'] * 1e3:.2f} mWb
        ## Kontrola
        - Φ = 60·E₀/(Z·n) = {phi2 * 1e3:.2f} mWb (inny wzór – zgodny)
        - prąd przewodnika (a = {p['poles']}): {r['I_cond']:.0f} A
        - B_śr w szczelinie = {r['B_av']:.3f} T
        - okładzina prądowa A = {r['A_lin'] / 1e3:.1f} kA/m
        """ + ("! Liczba przewodników na żłobek nie jest całkowita!"
               if abs(r["cond_slot"] - round(r["cond_slot"])) > 1e-6 else "+ Liczba przewodników w żłobku jest całkowita.")

    def draw(self, p):
        r = calc_m2(**p)
        ax1, ax2 = self.fig.subplots(1, 2)
        I = np.linspace(0, 2 * r["I"], 100)
        kT = r["Z"] * r["phi"] / TWO_PI
        ax1.plot(I, kT * I, label=f"Φ = {r['phi'] * 1e3:.1f} mWb")
        ax1.plot(I, 0.8 * kT * I, "--", label="Φ · 0,8 (osłabione pole)")
        mark(ax1, r["I"], r["T"], f"{r['I']:.0f} A\n{r['T']:.0f} N·m")
        ax1.set(title="T = Z·Φ·I / 2π", xlabel="I [A]", ylabel="T [N·m]")
        ax1.legend()
        n = np.linspace(0.2 * p["n"], 2 * p["n"], 200)
        for f, ls in ((0.5, ":"), (1, "-"), (1.5, "--")):
            ax2.plot(n, f * p["P"] / (n * RPM2RAD), ls, label=f"P = {f * p['P'] / 1e3:.0f} kW")
        mark(ax2, p["n"], r["T"], f"{p['n']:.0f} obr/min")
        ax2.set(title="Hiperbole stałej mocy T = 9,55·P/n", xlabel="n [obr/min]",
                ylabel="T [N·m]")
        ax2.legend()


# ─────────────────────────────────────────────────────────────────────────────
#  M3 – WARD-LEONARD
# ─────────────────────────────────────────────────────────────────────────────
class TabM3(ExampleTab):
    tab_title = "M3 – Ward-Leonard"
    tab_short = "M3 · Ward-Leonard"
    params = [("Es1", "Napięcie prądnicy – przypadek a) Es", 300, 600, 400, "V"),
              ("Eo", "SEM silnika E₀ (stan początkowy)", 100, 550, 380, "V"),
              ("Es2", "Napięcie prądnicy – przypadek b) Es", 200, 600, 350, "V"),
              ("R", "Rezystancja obwodu twornika", 0.002, 0.05, 0.010, "Ω"),
              ("n_nom", "Prędkość znamionowa (przy 500 V)", 100, 1000, 300, "obr/min"),
              ("J", "Moment bezwładności J (założony)", 2000, 100000, 20000, "kg·m²"),
              ("Lm", "Indukcyjność obwodu twornika (założona)", 0.00005, 0.005, 0.0005, "H")]
    explanation = """
    # Przykład M3 – układ Ward-Leonarda: silnik, który umie też hamować
    Treść (Wildi, rys. 6): silnik 2000 kW, 500 V zasilany z prądnicy 2500 kW w układzie
    Ward-Leonarda. Rezystancja obwodu twornika (silnik + prądnica) 10 mΩ, prędkość
    znamionowa 300 obr/min przy E₀ = 500 V. Oblicz moment i prędkość, gdy:
    a) Es = 400 V i E₀ = 380 V,  b) Es = 350 V i E₀ = 380 V.

    ## 1. Na czym polega układ?
    Silnik prądu przemiennego napędza prądnicę DC. Napięcie prądnicy Es reguluje się
    prądem jej wzbudzenia (od zera do maksimum, także ze zmianą znaku). To napięcie zasila
    twornik silnika DC, którego pole jest stałe. Ponieważ prędkość silnika ≈ proporcjonalna
    do napięcia, kręcimy „gałką" wzbudzenia prądnicy i płynnie zmieniamy prędkość
    w obie strony – stosowane w walcowniach, kopalniach, wysokich windach.

    ## 2. Rozwiązanie a) – praca silnikowa
    $ I = (Es − E₀)/R = (400 − 380)/0,01 = 2000 A
    $ P = E₀·I = 380 × 2000 = 760 kW
    $ n = 300 × 380/500 = 228 obr/min
    $ T = 9,55 × 760 000 / 228 = 31,8 kN·m
    ## 3. Rozwiązanie b) – hamowanie odzyskowe
    Prędkość nie może zmienić się natychmiast (bezwładność!), więc E₀ nadal = 380 V, n = 228 obr/min.
    $ I = (350 − 380)/0,01 = −3000 A     ← prąd zmienił kierunek!
    $ P = 380 × 3000 = 1140 kW  (silnik ODDAJE moc do prądnicy)
    $ T = 9,55 × 1 140 000 / 228 = 47,7 kN·m  hamującego
    > Książka: a) 31,8 kN·m, 228 obr/min;  b) −47,7 kN·m (hamujący), 228 obr/min

    ## 4. Co tu jest najważniejsze?
    Wystarczy obniżyć Es tylko o 50 V (10 %), żeby moment zmienił się z +31,8 na −47,7 kN·m!
    Powód: rezystancja jest bardzo mała (10 mΩ), więc niewielka różnica napięć daje ogromny
    prąd. Silnik staje się prądnicą, a prądnica – silnikiem, który napędza maszynę AC jako
    generator asynchroniczny i zwraca energię do sieci. To hamowanie odzyskowe (rekuperacja)
    – ta sama idea co w tramwajach i samochodach elektrycznych.

    ## 5. Komentarz inżyniera
    - Prąd 3000 A to 75 % ponad znamionowe ~4000 A? Sprawdźmy: I_n = 2000 kW/500 V = 4000 A,
      więc 3000 A jest dopuszczalne. Ale zmiana Es o np. 100 V dałaby −8000 A – za dużo!
      W praktyce Es zmienia się płynnie (rampa) albo układ ma regulator prądu.
    - Symulacja (prawe wykresy, ODE): obciążenie ma stały moment równy momentowi z punktu a).
      Po skoku Es silnik hamuje, aż osiągnie nową równowagę: E₀ = Es₂ − R·I = 330 V → 198 obr/min.
      Indukcyjność twornika (założona) sprawia, że prąd zmienia się w ciągu milisekund, a nie natychmiast.
    - Lewy wykres to płaszczyzna moment–prędkość (4 ćwiartki). Proste to charakterystyki
      mechaniczne dla stałych Es – są prawie pionowe (bardzo „sztywne"), bo R jest małe.
      Trajektoria pokazuje przejście: A (silnik) → B (hamowanie) → C (nowy punkt pracy).
    """

    def _K(self, p):
        return 500.0 / (p["n_nom"] * RPM2RAD)

    def solve(self, p):
        a = calc_m3(p["Es1"], p["Eo"], p["R"], p["n_nom"], 500)
        b = calc_m3(p["Es2"], p["Eo"], p["R"], p["n_nom"], 500)
        K = self._K(p)
        TL = K * a["I"]
        Eo_new = p["Es2"] - p["R"] * a["I"]
        tau_m = p["J"] * p["R"] / K ** 2

        def reg(x):
            return "praca SILNIKOWA" if x["I"] >= 0 else "HAMOWANIE odzyskowe (prądnicowa)"
        return f"""
        # a) Es = {p['Es1']:.0f} V, E₀ = {p['Eo']:.0f} V
        $ I = ({p['Es1']:.0f} − {p['Eo']:.0f})/{p['R']:.4g} = {a['I']:.0f} A
        $ P = E₀·I = {a['P'] / 1e3:.1f} kW,  n = {a['n']:.1f} obr/min
        > T = {a['T'] / 1e3:.2f} kN·m  → {reg(a)}
        # b) Es = {p['Es2']:.0f} V, E₀ = {p['Eo']:.0f} V (prędkość jeszcze ta sama)
        $ I = ({p['Es2']:.0f} − {p['Eo']:.0f})/{p['R']:.4g} = {b['I']:.0f} A
        $ P = E₀·I = {b['P'] / 1e3:.1f} kW,  n = {b['n']:.1f} obr/min
        > T = {b['T'] / 1e3:.2f} kN·m  → {reg(b)}
        - straty w R: {b['Ploss'] / 1e3:.1f} kW; do prądnicy płynie {-b['Pin'] / 1e3:.1f} kW
        ## Dynamika (symulacja)
        - K = kΦ = {K:.3f} V·s/rad,  τm = J·R/K² = {tau_m:.3f} s
        - moment obciążenia T_L = {TL / 1e3:.2f} kN·m
        - nowy stan ustalony: E₀ = {Eo_new:.0f} V → n = {p['n_nom'] * Eo_new / 500:.0f} obr/min
        """ + ("x Prąd hamowania przekracza 2× prąd znamionowy (4000 A)!"
               if abs(b["I"]) > 8000 else "+ Prąd hamowania w granicach 2× I_n (4000 A).")

    def draw(self, p):
        K = self._K(p)
        R, L, J = p["R"], p["Lm"], p["J"]
        a = calc_m3(p["Es1"], p["Eo"], R, p["n_nom"], 500)
        b = calc_m3(p["Es2"], p["Eo"], R, p["n_nom"], 500)
        TL = K * a["I"]
        w0 = p["Eo"] / K
        t_step = 0.5
        tau_m = J * R / K ** 2
        t_end = t_step + 6 * tau_m + 5 * L / R

        def Es(t):
            return p["Es1"] if t < t_step else p["Es2"]

        def f(t, x):
            i, w = x
            return [(Es(t) - R * i - K * w) / L, (K * i - TL) / J]

        sol = solve_ivp(f, (0, t_end), [a["I"], w0], max_step=min(t_end / 1500, L / R / 5),
                        rtol=1e-7, atol=[1e-3, 1e-6])
        axs = self.fig.subplot_mosaic([["pl", "n"], ["pl", "i"]])
        ax = axs["pl"]
        n_all = sol.y[1] * RAD2RPM
        nmin, nmax = n_all.min(), n_all.max()
        nn = np.linspace(nmin - 10, nmax + 10, 50)
        for E_s, col, nm in ((p["Es1"], C["cyan"], "Es (a)"), (p["Es2"], C["peach"], "Es (b)")):
            T = K * (E_s - K * nn * RPM2RAD) / R
            ax.plot(T / 1e3, nn, color=col, label=f"charakterystyka {nm} = {E_s:.0f} V")
        T_traj = K * sol.y[0]
        ax.plot(T_traj / 1e3, n_all, color=C["yellow"], lw=1.2, ls="--",
                label="trajektoria (symulacja)")
        mark(ax, a["T"] / 1e3, a["n"], "A: silnik")
        mark(ax, b["T"] / 1e3, b["n"], "B: hamowanie", C["red"])
        mark(ax, T_traj[-1] / 1e3, n_all[-1], "C: nowy stan", C["green"], dy=-25)
        ax.axvline(0, color=C["fg"], lw=0.8)
        ax.set(title="Płaszczyzna moment–prędkość", xlabel="T [kN·m]",
               ylabel="n [obr/min]")
        ax.legend(loc="lower left", fontsize=7)
        axs["n"].plot(sol.t, n_all, color=C["green"])
        axs["n"].axvline(t_step, color=C["sub"], ls=":")
        axs["n"].set(title=f"Skok Es: {p['Es1']:.0f} → {p['Es2']:.0f} V w t = {t_step} s",
                     ylabel="n [obr/min]")
        axs["i"].plot(sol.t, sol.y[0] / 1e3, color=C["peach"], label="prąd twornika")
        axs["i"].axhline(0, color=C["fg"], lw=0.8)
        axs["i"].set(xlabel="t [s]", ylabel="I [kA]")
        axi = axs["i"].twinx()
        axi.plot(sol.t, K * sol.y[0] / 1e3, color=C["mauve"], lw=1, ls="--", label="moment")
        axi.set_ylabel("T [kN·m]", color=C["mauve"])
        axi.grid(False)
        h1, l1 = axs["i"].get_legend_handles_labels()
        h2, l2 = axi.get_legend_handles_labels()
        axs["i"].legend(h1 + h2, l1 + l2, loc="lower right")


# ─────────────────────────────────────────────────────────────────────────────
#  M4 – SILNIK BOCZNIKOWY
# ─────────────────────────────────────────────────────────────────────────────
class TabM4(ExampleTab):
    tab_title = "M4 – silnik bocznikowy"
    tab_short = "M4 · bocznikowy"
    params = [("Es", "Napięcie zasilania", 20, 300, 120, "V"),
              ("IL", "Prąd z sieci (liniowy)", 2, 150, 51, "A"),
              ("Rf", "Rezystancja uzwojenia bocznikowego", 20, 500, 120, "Ω"),
              ("Ra", "Rezystancja twornika", 0.01, 1.0, 0.1, "Ω"),
              ("n", "Prędkość", 200, 3000, 1500, "obr/min")]
    explanation = """
    # Przykład M4 – dokąd idzie moc w silniku bocznikowym?
    Treść (Wildi, rys. 9): silnik bocznikowy obraca się z prędkością 1500 obr/min, zasilany
    ze źródła 120 V. Prąd z sieci 51 A, rezystancja uzwojenia wzbudzenia (bocznikowego) 120 Ω,
    rezystancja twornika 0,1 Ω. Oblicz: a) prąd twornika, b) SEM, c) moc mechaniczną.

    ## 1. Budowa
    „Bocznikowy" znaczy, że uzwojenie wzbudzenia (elektromagnes tworzący pole) jest podłączone
    RÓWNOLEGLE do twornika – bezpośrednio do sieci. Prąd z sieci rozdziela się jak woda
    w rozgałęzieniu rury: mała część płynie przez wzbudzenie, reszta przez twornik
    (I prawo Kirchhoffa). Pole jest więc stałe niezależnie od obciążenia → prędkość prawie
    stała (charakterystyka „sztywna" – obrabiarki, wentylatory, pompy).

    ## 2. Rozwiązanie
    - a) Prąd wzbudzenia z prawa Ohma, a prąd twornika z prawa Kirchhoffa:
    $ I_f = 120 V / 120 Ω = 1 A          I_a = 51 − 1 = 50 A
    - b) Spadek napięcia na tworniku i SEM:
    $ I_a·R_a = 50 × 0,1 = 5 V            E₀ = 120 − 5 = 115 V
    - c) Bilans mocy:
    $ P_pobrana = 120 × 51 = 6120 W
    $ P_twornika = 120 × 50 = 6000 W,    straty w tworniku I²R = 50² × 0,1 = 250 W
    $ P_mech = 6000 − 250 = 5750 W  (= E₀·I_a = 115 × 50)  ≈ 7,7 KM
    > Książka: 50 A, 115 V, 5750 W

    ## 3. Komentarz inżyniera
    - Moc na wale jest jeszcze nieco mniejsza niż 5750 W: część zjadają tarcie w łożyskach,
      opory powietrza (wentylacja) i straty w żelazie twornika (histereza, prądy wirowe).
    - Sprawność „elektryczna" (bez strat mechanicznych): 5750/6120 = 94 % – realnie ok. 85–90 %.
    - Moment wewnętrzny: T = P/ω = 5750/(2π·1500/60) ≈ 36,6 N·m.
    - Charakterystyka n(T) spada tylko o kilka procent od biegu jałowego do obciążenia
      znamionowego. Osłabienie pola (większa rezystancja wzbudzenia) PODNOSI prędkość –
      to „regulacja polem" z rozdz. 6. Nigdy nie wolno przerwać obwodu wzbudzenia silnika
      pracującego – strumień spada do szczątkowego i silnik może się „rozbiegać".

    ## 4. Wykresy
    - Lewy: wykres bilansu mocy (gdzie „ucieka" energia).
    - Prawy: charakterystyka mechaniczna n(T) dla nominalnego wzbudzenia i dla osłabionego
      pola (R_f × 1,25), przy założeniu liniowej zależności Φ od I_f (bez nasycenia).
    """

    def solve(self, p):
        r = calc_m4(**p)
        eta = r["Pm"] / r["Pin"] if r["Pin"] else 0
        bad = "x I_a ≤ 0 – prąd sieci mniejszy niż prąd wzbudzenia!" if r["Ia"] <= 0 else ""
        return f"""
        # a) prądy
        $ I_f = Es/R_f = {p['Es']:.4g}/{p['Rf']:.4g} = {r['If']:.3f} A
        > I_a = I_L − I_f = {r['Ia']:.2f} A
        # b) SEM
        $ I_a·R_a = {r['IR']:.3f} V
        > E₀ = Es − I_a·R_a = {r['Eo']:.2f} V
        # c) moce
        $ P_we = Es·I_L = {r['Pin']:.0f} W
        $ P_wzbudzenia = Es·I_f = {r['Pf']:.0f} W
        $ P_tw = Es·I_a = {r['Pa']:.0f} W,  I²R = {r['Pcu']:.1f} W
        > P_mech = E₀·I_a = {r['Pm']:.0f} W  ({r['Pm'] / HP:.2f} KM)
        - moment T = {r['T']:.2f} N·m,  sprawność (bez strat mech.) {eta * 100:.1f} %
        {bad}
        """

    def draw(self, p):
        r = calc_m4(**p)
        ax1, ax2 = self.fig.subplots(1, 2)
        names = ["pobrana z sieci", "wzbudzenie (I_f²R_f)", "miedź twornika (I_a²R_a)",
                 "mechaniczna E₀·I_a"]
        vals = [r["Pin"], r["Pf"], r["Pcu"], r["Pm"]]
        cols = [C["blue"], C["peach"], C["red"], C["green"]]
        bars = ax1.barh(names[::-1], vals[::-1], color=cols[::-1], edgecolor=C["bg"])
        for b_, v in zip(bars, vals[::-1]):
            ax1.text(b_.get_width(), b_.get_y() + b_.get_height() / 2,
                     f" {v:.0f} W ({v / r['Pin'] * 100:.1f} %)", va="center", fontsize=8)
        ax1.set(title="Bilans mocy", xlabel="P [W]",
                xlim=(0, r["Pin"] * 1.45 if r["Pin"] > 0 else 1))
        K = r["K"]
        Tn = max(r["T"], 1e-3)
        T = np.linspace(0, 2 * Tn, 100)
        for f, ls, lab in ((1.0, "-", "R_f nominalne"), (1 / 1.25, "--", "R_f × 1,25 (osłabione pole)")):
            Kf = K * f
            n = (p["Es"] - T / Kf * p["Ra"]) / Kf * RAD2RPM
            ax2.plot(T, n, ls, label=lab)
        mark(ax2, r["T"], p["n"], f"{r['T']:.1f} N·m\n{p['n']:.0f} obr/min")
        ax2.set(title="Charakterystyka mechaniczna n(T)", xlabel="T [N·m]",
                ylabel="n [obr/min]")
        ax2.legend()


# ─────────────────────────────────────────────────────────────────────────────
#  M5 – SILNIK SZEREGOWY
# ─────────────────────────────────────────────────────────────────────────────
class TabM5(ExampleTab):
    tab_title = "M5 – silnik szeregowy"
    tab_short = "M5 · szeregowy"
    params = [("hp", "Moc znamionowa", 1, 100, 15, "KM (hp)"),
              ("V", "Napięcie", 50, 600, 240, "V"),
              ("n_r", "Prędkość znamionowa", 300, 4000, 1780, "obr/min"),
              ("I_fl", "Prąd znamionowy", 5, 300, 54, "A"),
              ("TL", "Moment obciążenia", 5, 120, 24, "N·m")]
    explanation = """
    # Przykład M5 – silnik szeregowy i „jednostki względne"
    Treść (Wildi, rys. 11): silnik szeregowy 15 KM, 240 V, 1780 obr/min, prąd znamionowy 54 A.
    Jego charakterystyki podano w jednostkach względnych (pu). Oblicz: a) prąd i prędkość
    przy momencie obciążenia 24 N·m, b) sprawność w tych warunkach.

    ## 1. Czym jest silnik szeregowy?
    Uzwojenie wzbudzenia połączone jest SZEREGOWO z twornikiem – płynie przez nie ten sam prąd.
    Gdy obciążenie rośnie → rośnie prąd → rośnie pole → rośnie moment (z grubsza T ~ I²) –
    stąd ogromny moment rozruchowy (tramwaje, dźwigi, rozruszniki samochodowe).
    Ale gdy obciążenie maleje → słabnie pole → silnik przyspiesza. Bez obciążenia prędkość
    rośnie „bez końca" – silnik może się rozerwać! Dlatego silnika szeregowego NIGDY
    nie uruchamia się bez obciążenia (np. nie łączy się go z maszyną przez pasek).

    ## 2. Jednostki względne (per unit, pu)
    Zamiast pamiętać wartości dla każdego silnika osobno, dzielimy je przez wartości
    znamionowe („bazowe"). 1 pu = wartość znamionowa. Wtedy jedna krzywa opisuje całą rodzinę
    podobnych silników – wygodne jak procenty.
    $ P_B = 15 × 746 = 11 190 W,   n_B = 1780 obr/min,   I_B = 54 A
    $ T_B = 9,55 × P_B / n_B = 9,55 × 11 190 / 1780 = 60 N·m

    ## 3. Rozwiązanie
    $ T(pu) = 24 / 60 = 0,4 pu
    - z wykresu: przy T = 0,4 pu → n = 1,4 pu i I = 0,6 pu
    $ n = 1,4 × 1780 = 2492 obr/min,     I = 0,6 × 54 = 32,4 A
    $ P_we = 240 × 32,4 = 7776 W,       P_wy = n·T/9,55 = 2492 × 24 / 9,55 = 6263 W
    $ η = 6263 / 7776 = 80,5 %
    > Książka: 32,4 A, 2492 obr/min, η = 80,5 %

    ## 4. Komentarz inżyniera
    - Krzywe z rys. 11 zostały tu odwzorowane przybliżoną aproksymacją (PCHIP) przechodzącą
      przez punkty z przykładu – odczyty dla innych momentów są orientacyjne (±3 %).
    - Przy 40 % momentu silnik kręci się 40 % szybciej niż znamionowo – w trakcji to zaleta:
      lekki pociąg jedzie szybko, ciężki pod górę – wolno, ale z dużą siłą (naturalna
      „hiperboliczna" charakterystyka, bliska stałej mocy).
    - Sprawność maleje przy małych obciążeniach, bo stałe straty (tarcie, żelazo) stanowią
      większą część mocy; maksimum zwykle w pobliżu obciążenia znamionowego.

    ## 5. Wykresy
    - Lewy: charakterystyki n(T) i I(T) w pu (zaznaczony punkt pracy z suwaka).
    - Prawy: sprawność i moc wyjściowa w funkcji momentu w jednostkach fizycznych.
    """

    def solve(self, p):
        r = calc_m5(**p)
        warn = ("! Moment poza zakresem krzywych (0,1–2 pu) – wynik przycięty do krańca krzywej."
                if r["clipped"] else "")
        return f"""
        # Wielkości bazowe
        $ P_B = {p['hp']:.4g}·746 = {r['PB']:.0f} W
        $ T_B = 9,55·P_B/n_B = {r['TB']:.2f} N·m
        # a) punkt pracy
        $ T(pu) = {p['TL']:.4g}/{r['TB']:.3g} = {r['Tpu']:.3f} pu
        $ z krzywych: n = {r['n_pu']:.3f} pu,  I = {r['I_pu']:.3f} pu
        > n = {r['n']:.0f} obr/min,   I = {r['I']:.2f} A
        # b) sprawność
        $ P_we = U·I = {r['Pi']:.0f} W
        $ P_wy = n·T/9,55 = {r['Po']:.0f} W
        > η = {r['eta'] * 100:.1f} %
        {warn}
        """

    def draw(self, p):
        r = calc_m5(**p)
        ax1, ax2 = self.fig.subplots(1, 2)
        T = np.linspace(SER_T[0], SER_T[-1], 300)
        ax1.plot(T, _ser_n(T), label="prędkość n (pu)")
        ax1.plot(T, _ser_i(T), label="prąd I (pu)")
        ax1.plot(SER_T, SER_N, ".", color=C["sub"])
        ax1.plot(SER_T, SER_I, ".", color=C["sub"])
        Tq = float(np.clip(r["Tpu"], SER_T[0], SER_T[-1]))
        ax1.axvline(Tq, color=C["yellow"], ls=":", lw=1)
        mark(ax1, Tq, r["n_pu"], f"n = {r['n_pu']:.2f} pu")
        mark(ax1, Tq, r["I_pu"], f"I = {r['I_pu']:.2f} pu", C["green"], dy=-25)
        ax1.set(title="Charakterystyki silnika szeregowego (pu)", xlabel="T [pu]",
                ylabel="n, I [pu]", ylim=(0, 3))
        ax1.legend()
        TB = r["TB"]
        Tn = T * TB
        n = _ser_n(T) * p["n_r"]
        I = _ser_i(T) * p["I_fl"]
        eta = n * RPM2RAD * Tn / (p["V"] * I) * 100
        ax2.plot(Tn, eta, color=C["green"], label="sprawność η [%]")
        mark(ax2, Tq * TB, r["eta"] * 100, f"η = {r['eta'] * 100:.1f} %")
        ax2.set(title="Sprawność i moc wyjściowa", xlabel="T [N·m]", ylabel="η [%]",
                ylim=(50, 100))
        axb = ax2.twinx()
        axb.plot(Tn, n * RPM2RAD * Tn / 1e3, color=C["peach"], ls="--", label="P_wy [kW]")
        axb.set_ylabel("P_wy [kW]", color=C["peach"])
        axb.grid(False)
        h1, l1 = ax2.get_legend_handles_labels()
        h2, l2 = axb.get_legend_handles_labels()
        ax2.legend(h1 + h2, l1 + l2, loc="lower right")


# ─────────────────────────────────────────────────────────────────────────────
#  M6 – HAMOWANIE DYNAMICZNE
# ─────────────────────────────────────────────────────────────────────────────
class TabM6(ExampleTab):
    tab_title = "M6 – hamowanie dynamiczne"
    tab_short = "M6 · ham. dynam."
    params = [("J", "Moment bezwładności J", 10, 1000, 177, "kg·m²"),
              ("n1", "Prędkość początkowa n₁", 200, 3000, 1280, "obr/min"),
              ("Es", "Napięcie twornika przed hamowaniem", 50, 400, 210, "V"),
              ("Rb", "Rezystor hamujący R", 0.05, 2.0, 0.2, "Ω"),
              ("Pw", "Straty wentylacyjne, tarcie, żelazo", 500, 30000, 8000, "W"),
              ("n2", "Prędkość końcowa n₂", 1, 500, 20, "obr/min")]
    explanation = """
    # Przykład M6 – jak szybko zatrzymać ciężkie koło zamachowe?
    Treść (Wildi): silnik 225 kW, 250 V, 1280 obr/min; straty wentylacyjne, tarcia i w żelazie
    8 kW. Napędza duże koło zamachowe; łączny moment bezwładności 177 kg·m². Silnik jest
    zasilany 210 V i kręci się z prędkością 1280 obr/min, gdy twornik przełączamy na rezystor
    hamujący 0,2 Ω. Oblicz: a) mechaniczną stałą czasową T₀, b) czas spadku prędkości do
    20 obr/min, c) ten czas, gdy hamują tylko straty mechaniczne i w żelazie.

    ## 1. Hamowanie dynamiczne – silnik zamienia się w prądnicę
    Odłączamy twornik od sieci i podłączamy do rezystora, pozostawiając wzbudzenie. Wirujące
    koło zamachowe kręci teraz maszynę jak prądnicę: indukuje się napięcie, przez rezystor
    płynie prąd w przeciwną stronę → powstaje moment hamujący. Energia ruchu (Ek = ½Jω²)
    zamienia się w ciepło w rezystorze. Im wolniej się kręci, tym mniejsze napięcie, mniejszy
    prąd i słabsze hamowanie – dlatego prędkość maleje wykładniczo, jak napięcie
    rozładowującego się kondensatora.

    ## 2. Stała „połowicznego zaniku" T₀
    T₀ to czas, po którym prędkość spada o połowę (jak czas połowicznego rozpadu w fizyce
    jądrowej!). Wildi podaje:
    $ T₀ = J · n₁² / (131,5 · P₁)        T = T₀ / 0,693  (klasyczna stała czasowa, 36,8 %)
    gdzie P₁ – początkowa moc oddawana do rezystora, 131,5 = (30/π)²/ln 2.

    ## 3. Rozwiązanie
    - a) Tuż po przełączeniu E₀ ≈ 210 V (prędkość jeszcze się nie zmieniła):
    $ P₁ = E₀² / R = 210² / 0,2 = 220 500 W
    $ T₀ = 177 × 1280² / (131,5 × 220 500) = 10 s
    - b) Prędkość spada o połowę co 10 s: 1280 → 640 → 320 → 160 → 80 → 40 → 20 obr/min,
      czyli 6 „połówek":
    $ t = 6 × T₀ = 60 s
    - c) Tylko straty 8 kW (przybliżenie – zachowują się podobnie jak rezystor):
    $ T₀ = 177 × 1280² / (131,5 × 8000) = 275 s     t = 6 × 275 = 1650 s ≈ 27,5 min
    > Książka: T₀ = 10 s, 60 s, ≈ 1650 s (28× dłużej niż z hamowaniem dynamicznym)

    ## 4. Skąd ten wzór? (dla dociekliwych)
    Moc w rezystorze P = E²/R = (Kω)²/R, a moment hamujący T = P/ω = (K²/R)·ω – proporcjonalny
    do prędkości jak tarcie lepkie. Równanie ruchu J·dω/dt = −(K²/R)·ω ma rozwiązanie
    ω(t) = ω₁·e^(−t/T), T = J·R/K². Podstawiając K = E/ω₁ dostajemy T = Jω₁²/P₁, a T₀ = T·ln2. ✔

    ## 5. Komentarz inżyniera
    - Teoretycznie prędkość nigdy nie osiąga zera (wykładnicza!). Praktycznie przyjmuje się,
      że maszyna stoi po ok. 5·T₀. Na końcu zatrzymanie dokańcza hamulec mechaniczny.
    - Prąd początkowy: 210/0,2 = 1050 A ≈ 1,2 × I_n (I_n = 225 kW/250 V = 900 A) – Wildi zaleca
      ok. 2 × I_n, więc jest bezpiecznie dla komutatora.
    - W rzeczywistości oba efekty (rezystor + straty) działają jednocześnie – krzywa „łącznie".
    - Energia do rozproszenia: ½·177·(134 rad/s)² ≈ 1,59 MJ – rezystor musi wytrzymać krótko
      dużą moc (220 kW na starcie!).
    """

    def solve(self, p):
        r = calc_m6(**p)
        w1 = p["n1"] * RPM2RAD
        Ek = 0.5 * p["J"] * w1 ** 2
        seq = []
        n, t = p["n1"], 0.0
        while n > p["n2"] * 1.0001 and len(seq) < 12:
            seq.append(f"{t:7.1f} s → {n:7.1f} obr/min")
            n /= 2
            t += r["To"]
        seq.append(f"{t:7.1f} s → {n:7.1f} obr/min")
        seq_txt = "\n".join("$ " + s for s in seq)
        return f"""
        # a) stała czasowa
        $ P₁ = Es²/R = {p['Es']:.4g}²/{p['Rb']:.4g} = {r['P1'] / 1e3:.1f} kW
        $ T₀ = J·n₁²/(131,5·P₁)
        > T₀ = {r['To']:.2f} s   (T = T₀/0,693 = {r['T']:.2f} s)
        # b) czas do n₂ = {p['n2']:.0f} obr/min
        $ liczba „połówek" = log₂(n₁/n₂) = {r['halvings']:.2f}
        {seq_txt}
        > t = {r['t2']:.1f} s
        # c) tylko straty {p['Pw'] / 1e3:.1f} kW
        $ T₀' = {r['To_c']:.1f} s
        > t' = {r['t2c']:.0f} s = {r['t2c'] / 60:.1f} min  ({r['ratio']:.1f}× dłużej)
        ## Dodatkowo
        - oba efekty razem: T₀ = {r['To_both']:.2f} s, t = {r['t2_both']:.1f} s
        - prąd początkowy {r['I1']:.0f} A, energia kinetyczna {Ek / 1e6:.3f} MJ
        """

    def draw(self, p):
        r = calc_m6(**p)
        ax1, ax2 = self.fig.subplots(1, 2)
        w1 = p["n1"] * RPM2RAD
        K = p["Es"] / w1
        Beq = K ** 2 / p["Rb"]
        Bw = p["Pw"] / w1 ** 2
        t_max = min(max(r["t2"] * 1.3, 5 * r["To"]), 3 * r["t2c"])
        t = np.linspace(0, t_max, 800)
        n_dyn = p["n1"] * np.exp(-t * Beq / p["J"])
        n_both = p["n1"] * np.exp(-t * (Beq + Bw) / p["J"])
        n_coast = p["n1"] * np.exp(-t * Bw / p["J"])
        ax1.plot(t, n_dyn, label=f"hamowanie dynamiczne (T₀={r['To']:.1f} s)")
        ax1.plot(t, n_both, "--", label=f"dynamiczne + straty (T₀={r['To_both']:.1f} s)")
        ax1.plot(t, n_coast, label=f"wybieg – tylko straty (T₀={r['To_c']:.0f} s)")
        k = np.arange(0, int(t_max / r["To"]) + 1)
        ax1.plot(k * r["To"], p["n1"] / 2 ** k, "o", color=C["yellow"], ms=5,
                 label="co T₀ prędkość ÷2")
        ax1.axhline(p["n2"], color=C["red"], ls=":", lw=1, label=f"n₂ = {p['n2']:.0f} obr/min")
        if r["t2"] <= t_max:
            mark(ax1, r["t2"], p["n2"], f"t = {r['t2']:.1f} s", C["red"], dy=20)
        ax1.set(title="Prędkość podczas hamowania", xlabel="t [s]", ylabel="n [obr/min]",
                xlim=(0, t_max), ylim=(0, p["n1"] * 1.05))
        ax1.legend(fontsize=7)
        i = K * n_dyn * RPM2RAD / p["Rb"]
        ax2.plot(t, i, color=C["peach"], label="prąd hamowania I [A]")
        ax2.set(title="Prąd i moment hamujący", xlabel="t [s]", ylabel="I [A]",
                xlim=(0, t_max))
        axb = ax2.twinx()
        axb.plot(t, K * i / 1e3, color=C["mauve"], ls="--", label="moment hamujący [kN·m]")
        axb.set_ylabel("T [kN·m]", color=C["mauve"])
        axb.grid(False)
        h1, l1 = ax2.get_legend_handles_labels()
        h2, l2 = axb.get_legend_handles_labels()
        ax2.legend(h1 + h2, l1 + l2)


# ─────────────────────────────────────────────────────────────────────────────
#  M7 – HAMOWANIE PRZECIWPRĄDEM
# ─────────────────────────────────────────────────────────────────────────────
class TabM7(ExampleTab):
    tab_title = "M7 – hamowanie przeciwprądem"
    tab_short = "M7 · przeciwprąd"
    params = [("J", "Moment bezwładności J", 10, 1000, 177, "kg·m²"),
              ("n1", "Prędkość początkowa n₁", 200, 3000, 1280, "obr/min"),
              ("Es", "Napięcie zasilania", 50, 400, 210, "V"),
              ("Rb", "Rezystor w obwodzie przeciwprądu R", 0.1, 3.0, 0.4, "Ω")]
    explanation = """
    # Przykład M7 – hamowanie przeciwprądem (ang. plugging)
    Treść (Wildi): silnik z przykładu M6 hamujemy przeciwprądem; rezystor zwiększono do 0,4 Ω,
    tak aby początkowy prąd hamowania był taki sam jak poprzednio. Oblicz: a) początkowy
    prąd i moc hamowania, b) czas zatrzymania.

    ## 1. Na czym polega?
    Nagle zamieniamy końcówki zasilania twornika. Teraz SEM wirującego silnika i napięcie
    sieci DODAJĄ SIĘ (wcześniej się odejmowały). Bez rezystora prąd byłby nawet 50× większy
    od znamionowego – łuk na komutatorze zniszczyłby maszynę. Dlatego wstawiamy rezystor.
    Hamuje i silnik (oddaje energię kinetyczną), i sieć (dostarcza energię!) – dlatego
    hamowanie jest szybsze niż dynamiczne, a moment hamujący istnieje nawet przy n = 0.

    ## 2. Rozwiązanie
    $ U_netto = E₀ + Es = 210 + 210 = 420 V
    $ I₁ = 420 / 0,4 = 1050 A        (tyle samo co w M6: 210/0,2)
    $ P (od silnika) = E₀·I₁ = 210 × 1050 = 220,5 kW
    $ (w rezystorze wydziela się 420 × 1050 = 441 kW – połowa z sieci!)
    - Ponieważ moc oddawana przez silnik jest taka sama jak w M6, T₀ = 10 s. Wildi wykazuje,
      że przy przeciwprądzie prędkość spada do ZERA po czasie:
    $ t_s = 2 · T₀ = 20 s
    > Książka: I = 1050 A, P = 220,5 kW (od silnika), t_s = 20 s

    ## 3. Sprawdzenie równaniem różniczkowym
    $ J·dω/dt = −K·i,     i = (Es + K·ω)/R
    $ ω(t) = (ω₁ + Es/K)·e^(−t/τ) − Es/K,   τ = J·R/K²
    Zero prędkości: e^(−t/τ) = ½ → t_s = τ·ln 2. Ponieważ R jest 2× większe niż w M6, τ jest 2×
    większe, więc t_s = 2·T₀ ✔ (aplikacja liczy oba sposoby).

    ## 4. Komentarz inżyniera
    - W chwili zatrzymania prąd = Es/R = 525 A (połowa początkowego) – silnik dalej wytwarza
      moment i zacząłby kręcić się W DRUGĄ STRONĘ! Dlatego obwód musi być natychmiast
      przerwany przez czujnik zerowej prędkości (ang. null-speed switch, „plugging switch").
    - Wykres (przerywana linia po t_s) pokazuje, co by się stało bez wyłącznika.
    - Porównanie z hamowaniem dynamicznym (ten sam prąd początkowy): po czasie 2·T₀
      dynamiczne zostawia jeszcze 25 % prędkości, przeciwprąd – zero. Za to rezystor
      przeciwprądu grzeje się prawie 2× bardziej, a sieć dokłada energii. Dynamiczne jest
      prostsze i popularniejsze.
    """

    def solve(self, p):
        r = calc_m7(**p)
        return f"""
        # a) prąd i moc
        $ U_netto = E₀ + Es = {r['U_net']:.0f} V
        > I₁ = {r['U_net']:.0f}/{p['Rb']:.4g} = {r['I1']:.0f} A
        $ P od silnika = E₀·I₁ = {r['P_motor'] / 1e3:.1f} kW
        $ P w rezystorze = {r['P_res'] / 1e3:.1f} kW
        # b) czas zatrzymania
        $ T₀ = J·n₁²/(131,5·P) = {r['To']:.2f} s
        > t_s = 2·T₀ = {r['ts']:.2f} s
        ## Sprawdzenie ODE
        - K = {r['K']:.4g} V·s/rad, τ = J·R/K² = {r['tau']:.2f} s
        - t_s = τ·ln(1+E₀/Es) = {r['ts_exact']:.2f} s
        - prąd w chwili zatrzymania: {r['I_stop']:.0f} A  (!) wyłączyć obwód!
        - dynamiczne przy tym samym I₁ wymaga R = {p['Rb'] / 2:.3g} Ω
        """

    def draw(self, p):
        r = calc_m7(**p)
        K, J, R, Es = r["K"], p["J"], p["Rb"], p["Es"]
        w1 = p["n1"] * RPM2RAD
        tau = r["tau"]
        ts = r["ts_exact"]
        t = np.linspace(0, 2.5 * ts, 800)
        w_pl = (w1 + Es / K) * np.exp(-t / tau) - Es / K
        Rd = R / 2
        tau_d = J * Rd / K ** 2
        w_dy = w1 * np.exp(-t / tau_d)
        ax1, ax2 = self.fig.subplots(1, 2)
        m = t <= ts
        ax1.plot(t[m], w_pl[m] * RAD2RPM, label="przeciwprąd")
        ax1.plot(t[~m], w_pl[~m] * RAD2RPM, ":", color=C["cyan"],
                 label="…bez wyłącznika (rewers!)")
        ax1.plot(t, w_dy * RAD2RPM, label=f"dynamiczne (R = {Rd:.3g} Ω, ten sam I₁)")
        ax1.axhline(0, color=C["fg"], lw=0.8)
        mark(ax1, ts, 0, f"stop: t_s = {ts:.1f} s", C["red"], dy=25)
        mark(ax1, ts, w1 * np.exp(-ts / tau_d) * RAD2RPM,
             f"{np.exp(-ts / tau_d) * 100:.0f} % prędkości", C["green"])
        ax1.set(title="Prędkość: przeciwprąd vs dynamiczne", xlabel="t [s]",
                ylabel="n [obr/min]")
        ax1.legend(fontsize=7)
        i_pl = -(Es + K * w_pl) / R
        i_dy = -K * w_dy / Rd
        ax2.plot(t[m], i_pl[m], label="przeciwprąd")
        ax2.plot(t[~m], i_pl[~m], ":", color=C["cyan"])
        ax2.plot(t, i_dy, label="dynamiczne")
        ax2.axvline(ts, color=C["red"], ls=":", lw=1)
        ax2.set(title="Prąd twornika (ujemny = hamujący)", xlabel="t [s]", ylabel="I [A]")
        ax2.legend()


# ─────────────────────────────────────────────────────────────────────────────
#  ANALIZA STEROWANIA
# ─────────────────────────────────────────────────────────────────────────────
SYSTEMS = {
    "M1 – silnik z magnesami trwałymi: ω(s)/U(s)": dict(
        kind="motor", u_nom=150, u_name="U", u_unit="V", closed=True,
        params=[("R", "Rezystancja twornika R", 0.1, 5, 1.0, "Ω"),
                ("L", "Indukcyjność twornika L (zał.)", 0.001, 0.2, 0.02, "H"),
                ("K", "Stała maszyny K = kΦ", 0.2, 3, 0.9549, "V·s/rad"),
                ("J", "Moment bezwładności J (zał.)", 0.005, 0.5, 0.05, "kg·m²"),
                ("B", "Tarcie lepkie B (zał.)", 0.0, 0.05, 0.001, "N·m·s/rad")],
        kp=(0.0, 5.0, 0.4), ki=(0.0, 100.0, 13.0),
        desc="Wejście: napięcie twornika U [V]; wyjście: prędkość ω [rad/s]. "
             "R i K z przykładu M1 (K = 50 V / 52,36 rad/s); L, J, B przyjęte."),
    "M4 – silnik bocznikowy (Φ = const): ω(s)/U_a(s)": dict(
        kind="motor", u_nom=120, u_name="U", u_unit="V", closed=True,
        params=[("R", "Rezystancja twornika R", 0.02, 1.0, 0.1, "Ω"),
                ("L", "Indukcyjność twornika L (zał.)", 0.0005, 0.05, 0.005, "H"),
                ("K", "Stała maszyny K = kΦ", 0.2, 2.0, 0.7321, "V·s/rad"),
                ("J", "Moment bezwładności J (zał.)", 0.02, 2.0, 0.25, "kg·m²"),
                ("B", "Tarcie lepkie B (zał.)", 0.0, 0.05, 0.005, "N·m·s/rad")],
        kp=(0.0, 5.0, 0.05), ki=(0.0, 100.0, 5.0),
        desc="Wejście: napięcie twornika [V]; wyjście ω [rad/s]. K = E₀/ω = 115/157,1 "
             "z przykładu M4; L, J, B przyjęte."),
    "M3 – układ Ward-Leonarda: ω(s)/U_f(s)": dict(
        kind="wl", u_nom=220, u_name="U_f", u_unit="V", closed=True,
        params=[("Tf", "Stała czasowa wzbudzenia prądnicy τ_f (zał.)", 0.1, 5.0, 1.0, "s"),
                ("Kg", "Wzmocnienie prądnicy Es/U_f (zał.)", 0.5, 5.0, 2.273, "V/V"),
                ("R", "Rezystancja obwodu twornika R", 0.002, 0.05, 0.01, "Ω"),
                ("L", "Indukcyjność obwodu twornika (zał.)", 0.00005, 0.005, 0.0005, "H"),
                ("K", "Stała silnika K = kΦ", 5, 40, 15.915, "V·s/rad"),
                ("J", "Moment bezwładności J (zał.)", 2000, 100000, 20000, "kg·m²"),
                ("B", "Tarcie lepkie B (zał.)", 0.0, 500, 10, "N·m·s/rad")],
        kp=(0.0, 100.0, 12.0), ki=(0.0, 50.0, 5.0),
        desc="Wejście: napięcie wzbudzenia prądnicy U_f [V]; wyjście ω silnika [rad/s]. "
             "R i K z przykładu M3 (K = 500 V / 31,42 rad/s); τ_f, K_g, L, J, B przyjęte."),
    "M6 – hamowanie dynamiczne: ω(s)/T_z(s)": dict(
        kind="brake_T", u_nom=1000, u_name="T_z", u_unit="N·m", closed=False, n0=1280,
        params=[("R", "Rezystor hamujący R", 0.05, 2.0, 0.2, "Ω"),
                ("K", "Stała maszyny K = E/ω", 0.5, 3.0, 1.5667, "V·s/rad"),
                ("J", "Moment bezwładności J", 10, 1000, 177, "kg·m²"),
                ("B", "Straty jako tarcie lepkie B = P_w/ω₁²", 0.0, 3.0, 0.4453, "N·m·s/rad")],
        desc="Wejście: dodatkowy moment zewnętrzny T_z [N·m]; wyjście ω [rad/s]. "
             "Twornik zwarty przez rezystor (hamowanie dynamiczne). Dane z przykładu M6."),
    "M7 – hamowanie przeciwprądem: ω(s)/U(s)": dict(
        kind="brake_U", u_nom=-210, u_name="U", u_unit="V", closed=False, n0=1280,
        total=True,
        params=[("R", "Rezystor w obwodzie R", 0.1, 3.0, 0.4, "Ω"),
                ("K", "Stała maszyny K = E/ω", 0.5, 3.0, 1.5667, "V·s/rad"),
                ("J", "Moment bezwładności J", 10, 1000, 177, "kg·m²"),
                ("B", "Tarcie lepkie B", 0.0, 3.0, 0.0, "N·m·s/rad")],
        desc="Wejście: napięcie zasilania [V] (przeciwprąd = skok −210 V); wyjście ω [rad/s]. "
             "Pełny przebieg = odpowiedź swobodna od n₀ + wymuszona. Dane z przykładu M7."),
}


def plant_tf(kind, p):
    """Transmitancja obiektu G(s) = num/den (wyjście: ω [rad/s])."""
    if kind in ("motor", "wl"):
        den = np.polyadd(np.polymul([p["L"], p["R"]], [p["J"], p["B"]]), [p["K"] ** 2])
        num = np.array([p["K"]])
        if kind == "wl":
            num = num * p["Kg"]
            den = np.polymul([p["Tf"], 1.0], den)
        return num, den
    a = p["K"] ** 2 / p["R"] + p["B"]
    if kind == "brake_T":
        return np.array([1.0]), np.array([p["J"], a])
    return np.array([p["K"] / p["R"]]), np.array([p["J"], a])


def poly_str(c, var="s"):
    """Wielomian jako tekst: 0.001 s² + 0.05 s + 0.913"""
    c = np.atleast_1d(np.asarray(c, float))
    n = len(c) - 1
    sup = {2: "²", 3: "³", 4: "⁴", 5: "⁵"}
    parts = []
    for i, a in enumerate(c):
        pw = n - i
        if abs(a) < 1e-14:
            continue
        s = f"{abs(a):.4g}"
        if pw == 1:
            s += f"·{var}"
        elif pw > 1:
            s += f"·{var}{sup.get(pw, '^' + str(pw))}"
        parts.append(("− " if a < 0 else "+ ") + s)
    if not parts:
        return "0"
    out = " ".join(parts)
    return out[2:] if out.startswith("+ ") else "−" + out[2:]


def factor_str(roots):
    """Iloczyn czynników (s − r) z łączeniem par sprzężonych."""
    out, used = [], np.zeros(len(roots), bool)
    for i, r in enumerate(roots):
        if used[i]:
            continue
        used[i] = True
        if abs(r.imag) > 1e-9 * max(1, abs(r)):
            for j in range(i + 1, len(roots)):
                if not used[j] and abs(roots[j] - np.conj(r)) < 1e-6 * max(1, abs(r)):
                    used[j] = True
                    break
            out.append(f"(s² {'+' if -2 * r.real >= 0 else '−'} {abs(2 * r.real):.4g}·s + {abs(r) ** 2:.4g})")
        else:
            x = r.real
            if abs(x) < 1e-12:
                out.append("s")
            else:
                out.append(f"(s {'+' if x < 0 else '−'} {abs(x):.4g})")
    return "·".join(out) if out else "1"


def step_info(t, y):
    """Parametry odpowiedzi skokowej (y znormalizowane do skoku jednostkowego)."""
    yf = y[-1]
    info = dict(final=yf, tr=np.nan, ts=np.nan, Mp=0.0, tp=np.nan)
    if not np.isfinite(yf) or abs(yf) < 1e-15:
        return info
    yn = y / yf
    try:
        t10 = t[np.argmax(yn >= 0.1)]
        t90 = t[np.argmax(yn >= 0.9)]
        info["tr"] = t90 - t10
    except (ValueError, IndexError):
        pass
    out = np.where(np.abs(yn - 1) > 0.02)[0]
    info["ts"] = t[out[-1] + 1] if len(out) and out[-1] + 1 < len(t) else (t[0] if not len(out) else np.nan)
    ip = int(np.argmax(yn))
    info["Mp"] = max(0.0, (yn[ip] - 1) * 100)
    info["tp"] = t[ip]
    return info


def freq_range(*root_sets):
    mags = np.concatenate([np.abs(np.atleast_1d(r)) for r in root_sets if len(np.atleast_1d(r))])
    mags = mags[mags > 1e-9]
    if not len(mags):
        return 1e-2, 1e2
    return mags.min() / 100, mags.max() * 100


def margins(numL, denL, w):
    """Zapas wzmocnienia i fazy pętli L(jω) (sprzężenie jednostkowe, ujemne)."""
    _, H = signal.freqs(numL, denL, worN=w)
    mag = np.abs(H)
    ph = np.degrees(np.unwrap(np.angle(H)))
    res = dict(GM=np.inf, wpc=np.nan, PM=np.inf, wgc=np.nan)
    idx = np.where(np.diff(np.sign(mag - 1.0)) != 0)[0]
    if len(idx):
        i = idx[0]
        lw = np.interp(0, [np.log10(mag[i]), np.log10(mag[i + 1])][::(1 if mag[i] < mag[i + 1] else -1)],
                       [np.log10(w[i]), np.log10(w[i + 1])][::(1 if mag[i] < mag[i + 1] else -1)])
        res["wgc"] = 10 ** lw
        phc = np.interp(lw, np.log10(w), ph)
        res["PM"] = 180.0 + phc
    for k in (-180, -540, 180):
        idx2 = np.where(np.diff(np.sign(ph - k)) != 0)[0]
        if len(idx2):
            i = idx2[0]
            lw = np.interp(k, sorted([ph[i], ph[i + 1]]),
                           [np.log10(w[i]), np.log10(w[i + 1])] if ph[i] < ph[i + 1]
                           else [np.log10(w[i + 1]), np.log10(w[i])])
            res["wpc"] = 10 ** lw
            res["GM"] = -20 * np.log10(np.interp(lw, np.log10(w), mag))
            break
    return res, mag, ph, H


class ControlTab(ttk.Frame):
    """Bode / zpk / Nyquist / step dla modeli dynamicznych z przykładów."""
    tab_title = "Analiza sterowania"
    tab_short = "⚙ Bode · zpk · Nyquist · step"

    def __init__(self, nb, app):
        super().__init__(nb)
        self.app = app
        self._job = None
        paned = ttk.PanedWindow(self, orient="horizontal")
        paned.pack(fill="both", expand=True)
        left = ttk.Frame(paned)
        right = ttk.PanedWindow(paned, orient="vertical")
        paned.add(left, weight=0)
        paned.add(right, weight=1)

        sf = ttk.LabelFrame(left, text=" Układ (model z przykładu) ")
        sf.pack(fill="x", padx=6, pady=(6, 3))
        self.sys_var = tk.StringVar(value=list(SYSTEMS)[0])
        cb = ttk.Combobox(sf, textvariable=self.sys_var, values=list(SYSTEMS),
                          state="readonly", width=48)
        cb.pack(fill="x", padx=4, pady=3)
        cb.bind("<<ComboboxSelected>>", lambda e: self.rebuild())
        self.loop_var = tk.StringVar(value="open")
        lfr = ttk.Frame(sf)
        lfr.pack(fill="x", padx=4)
        self.rb_open = ttk.Radiobutton(lfr, text="Pętla otwarta G(s)", value="open",
                                       variable=self.loop_var, command=self.rebuild_ctrl)
        self.rb_closed = ttk.Radiobutton(lfr, text="Pętla zamknięta z regulatorem PI",
                                         value="closed", variable=self.loop_var,
                                         command=self.rebuild_ctrl)
        self.rb_open.pack(side="left")
        self.rb_closed.pack(side="left", padx=8)

        self.plant_box = ttk.LabelFrame(left, text=" Parametry obiektu ")
        self.plant_box.pack(fill="x", padx=6, pady=3)
        self.ctrl_box = ttk.LabelFrame(left, text=" Regulator / wymuszenie ")
        self.ctrl_box.pack(fill="x", padx=6, pady=3)
        self.pp_plant = None
        self.pp_ctrl = None

        rf = ttk.LabelFrame(left, text=" Wyniki: transmitancja, zpk, zapasy, odpowiedź skokowa ")
        rf.pack(fill="both", expand=True, padx=6, pady=(3, 6))
        self.res = tk.Text(rf, width=52, height=10)
        setup_text_tags(self.res, 10)
        rsb = ttk.Scrollbar(rf, orient="vertical", command=self.res.yview)
        self.res.configure(yscrollcommand=rsb.set)
        self.res.pack(side="left", fill="both", expand=True)
        rsb.pack(side="right", fill="y")

        ff = ttk.Frame(right)
        self.fig = Figure(figsize=(9, 6.5), dpi=100, layout="constrained")
        self.canvas = FigureCanvasTkAgg(self.fig, master=ff)
        self.toolbar = NavigationToolbar2Tk(self.canvas, ff, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(side="bottom", fill="x")
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
        ef = ttk.LabelFrame(right, text=" Interpretacja wyników (generowana automatycznie) + teoria ")
        self.exp = tk.Text(ef, height=10)
        setup_text_tags(self.exp, 10)
        esb = ttk.Scrollbar(ef, orient="vertical", command=self.exp.yview)
        self.exp.configure(yscrollcommand=esb.set)
        self.exp.pack(side="left", fill="both", expand=True)
        esb.pack(side="right", fill="y")
        right.add(ff, weight=3)
        right.add(ef, weight=2)
        self._paned, self._right = paned, right
        self.after(300, self._init_sashes)
        self.rebuild()

    def _init_sashes(self):
        try:
            self._paned.sashpos(0, 440)
            h = self._right.winfo_height()
            if h > 100:
                self._right.sashpos(0, int(h * 0.62))
        except tk.TclError:
            pass

    # — budowa paneli —
    def rebuild(self):
        sd = SYSTEMS[self.sys_var.get()]
        if not sd["closed"]:
            self.loop_var.set("open")
            self.rb_closed.state(["disabled"])
        else:
            self.rb_closed.state(["!disabled"])
        if self.pp_plant is not None:
            self.pp_plant.destroy()
        self.pp_plant = ParamPanel(self.plant_box, self.schedule,
                                   height=min(56 * len(sd["params"]) + 8, 290))
        for spec in sd["params"]:
            self.pp_plant.add(*spec)
        self.pp_plant.pack(fill="x", padx=2, pady=2)
        self.rebuild_ctrl()

    def rebuild_ctrl(self):
        sd = SYSTEMS[self.sys_var.get()]
        if self.pp_ctrl is not None:
            self.pp_ctrl.destroy()
        self.pp_ctrl = ParamPanel(self.ctrl_box, self.schedule, height=180)
        if self.loop_var.get() == "closed":
            self.pp_ctrl.add("Kp", "Wzmocnienie proporcjonalne Kp", *sd["kp"], "V/(rad/s)")
            self.pp_ctrl.add("Ki", "Wzmocnienie całkujące Ki", *sd["ki"], "V/rad")
            self.pp_ctrl.add("ref", "Skok prędkości zadanej", 10, 3000, 300, "obr/min")
        else:
            u = sd["u_nom"]
            lo, hi = (min(u * 2, -abs(u) * 0.1), 0) if u < 0 else (abs(u) * 0.05, abs(u) * 2)
            self.pp_ctrl.add("amp", f"Amplituda skoku {sd['u_name']}", lo, hi, u, sd["u_unit"])
            if "n0" in sd:
                self.pp_ctrl.add("n0", "Prędkość początkowa n₀", 0, 3000, sd["n0"], "obr/min")
        self.pp_ctrl.pack(fill="x", padx=2, pady=2)
        self.schedule()

    def schedule(self, *_):
        if self._job is not None:
            self.after_cancel(self._job)
        self._job = self.after(90, self.update_all)

    # — obliczenia —
    def compute(self):
        sd = SYSTEMS[self.sys_var.get()]
        p = self.pp_plant.values()
        c = self.pp_ctrl.values()
        closed = self.loop_var.get() == "closed"
        num, den = plant_tf(sd["kind"], p)
        d = dict(sd=sd, p=p, c=c, closed=closed, numG=num, denG=den)
        if closed:
            Kp, Ki = c["Kp"], c["Ki"]
            if Kp == 0 and Ki == 0:
                Kp = 1e-6
            if Ki > 0:
                numC, denC = np.array([Kp, Ki]), np.array([1.0, 0.0])
            else:
                numC, denC = np.array([Kp]), np.array([1.0])
            numL = np.polymul(numC, num)
            denL = np.polymul(denC, den)
            numT = numL
            denT = np.polyadd(denL, numL)
            numU = np.polymul(numC, den)
            denU = denT
            d.update(numC=numC, denC=denC, numL=numL, denL=denL, numT=numT, denT=denT,
                     numU=numU, denU=denU)
            num_a, den_a = numT, denT
        else:
            d.update(numL=num, denL=den)
            num_a, den_a = num, den
        d["num_a"], d["den_a"] = num_a, den_a
        z, pl, k = signal.tf2zpk(num_a, den_a)
        d["z"], d["poles"], d["k"] = z, pl, k
        zG, pG, kG = signal.tf2zpk(num, den)
        d["zG"], d["pG"], d["kG"] = zG, pG, kG
        # pętla zamknięta (do Nyquista – dla trybu otwartego: hipotetyczne sprzężenie jedn.)
        clp = np.roots(np.polyadd(d["denL"], d["numL"]))
        d["cl_poles"] = clp
        d["P_rhp"] = int(np.sum(np.roots(d["denL"]).real > 1e-9))
        d["Z_rhp"] = int(np.sum(clp.real > 1e-9))
        # częstotliwości
        wlo, whi = freq_range(pl, z, pG, clp)
        w = np.logspace(np.log10(wlo), np.log10(whi), 4000)
        d["w"] = w
        mg, magL, phL, HL = margins(d["numL"], d["denL"], w)
        d.update(mg=mg, magL=magL, phL=phL, HL=HL)
        _, Ha = signal.freqs(num_a, den_a, worN=w)
        d["Ha"] = Ha
        d["Ms"] = 1.0 / np.min(np.abs(1 + HL))
        # odpowiedź skokowa
        re = pl.real
        stable = np.all(re < -1e-12)
        d["stable"] = stable
        if stable:
            tmax = 7.0 / np.min(-re)
        else:
            unst = re[re >= -1e-12]
            tmax = 5.0 / max(np.max(np.abs(unst)), 1e-2) if np.max(unst) > 1e-9 else 50.0
        tmax = min(tmax, 1e5)
        t = np.linspace(0, tmax, 3000)
        t, y = signal.step((num_a, den_a), T=t)
        d["t"], d["y"] = t, y
        d["info"] = step_info(t, y) if stable else dict(final=np.nan, tr=np.nan, ts=np.nan,
                                                         Mp=np.nan, tp=np.nan)
        if closed:
            _, yu = signal.step((d["numU"], d["denU"]), T=t)
            d["yu"] = yu
        den0 = np.polyval(den_a, 0)
        d["dc"] = np.polyval(num_a, 0) / den0 if abs(den0) > 1e-15 else np.inf
        bw = np.nan
        if stable and np.isfinite(d["dc"]) and abs(d["dc"]) > 0:
            below = np.where(np.abs(Ha) < abs(d["dc"]) / np.sqrt(2))[0]
            if len(below):
                bw = w[below[0]]
        d["bw"] = bw
        return d

    def update_all(self):
        self._job = None
        try:
            d = self.compute()
        except Exception as e:                       # noqa: BLE001
            render_markup(self.res, f"x Błąd obliczeń: {e}")
            return
        try:
            render_markup(self.res, self.results_text(d))
            render_markup(self.exp, self.interpretation(d) + "\n" + CONTROL_THEORY)
        except Exception as e:                       # noqa: BLE001
            render_markup(self.res, f"x Błąd opisu: {e}")
        try:
            self.fig.clear()
            self.draw(d)
            self.canvas.draw_idle()
            self.app.status("Analiza sterowania: zaktualizowano.")
        except Exception as e:                       # noqa: BLE001
            self.app.status(f"Błąd rysowania: {e}")

    # — tekst wyników —
    @staticmethod
    def pole_lines(poles):
        lines = []
        for r in poles:
            if abs(r.imag) > 1e-9 * max(1, abs(r)) and r.imag < 0:
                continue
            if abs(r.imag) > 1e-9 * max(1, abs(r)):
                wn = abs(r)
                z = -r.real / wn
                lines.append(f"- p = {f4(r)}:  ωn = {wn:.4g} rad/s, ζ = {z:.3f}, "
                             f"f_d = {abs(r.imag) / TWO_PI:.4g} Hz")
            elif abs(r.real) < 1e-12:
                lines.append("- p = 0: całkowanie (integrator)")
            else:
                lines.append(f"- p = {r.real:.4g}:  τ = {1 / abs(r.real):.4g} s"
                             + ("  (NIESTABILNY)" if r.real > 0 else ""))
        return "\n".join(lines)

    def results_text(self, d):
        sd, info, mg = d["sd"], d["info"], d["mg"]
        out = [f"# {self.sys_var.get()}", f"~ {sd['desc']}",
               "## Transmitancja obiektu G(s)",
               f"$ licznik:   {poly_str(d['numG'])}",
               f"$ mianownik: {poly_str(d['denG'])}"]
        if d["closed"]:
            out += ["## Regulator PI",
                    f"$ C(s) = Kp + Ki/s = ({poly_str(d['numC'])}) / ({poly_str(d['denC'])})",
                    "## Pętla zamknięta T(s) = C·G / (1 + C·G)",
                    f"$ licznik:   {poly_str(d['numT'])}",
                    f"$ mianownik: {poly_str(d['denT'])}"]
        nm = "T(s)" if d["closed"] else "G(s)"
        out += [f"## Postać zpk {nm}",
                f"$ k = {d['k']:.5g}",
                "$ zera: " + (", ".join(f4(x) for x in d["z"]) if len(d["z"]) else "brak"),
                "$ bieguny: " + ", ".join(f4(x) for x in d["poles"]),
                f"$ {nm} = {d['k']:.4g}·{factor_str(d['z'])} / [{factor_str(d['poles'])}]",
                self.pole_lines(d["poles"]),
                "## Odpowiedź skokowa (wartości dla skoku jednostkowego)"]
        if d["stable"]:
            unit = "(rad/s)/(rad/s)" if d["closed"] else f"(rad/s)/{sd['u_unit']}"
            out += [f"$ wzmocnienie statyczne K(0) = {d['dc']:.5g} {unit}",
                    f"$ czas narastania 10–90 %: t_r = {f4(info['tr'])} s",
                    f"$ czas ustalania (±2 %): t_s = {f4(info['ts'])} s",
                    f"$ przeregulowanie: M_p = {info['Mp']:.2f} %  (t_p = {f4(info['tp'])} s)"]
        else:
            out.append("x Układ NIESTABILNY – odpowiedź narasta bez ograniczeń.")
        lab = "L(s) = C·G" if d["closed"] else "G(s) (przy hipot. sprzężeniu jednostkowym)"
        out += [f"## Bode / Nyquist dla {lab}",
                f"$ zapas fazy PM = {f4(mg['PM'])}°  przy ω_c = {f4(mg['wgc'])} rad/s",
                f"$ zapas wzmocnienia GM = {f4(mg['GM'])} dB  przy ω_180 = {f4(mg['wpc'])} rad/s",
                f"$ szczyt wrażliwości M_s = 1/min|1+L| = {d['Ms']:.3f}",
                f"$ pasmo −3 dB {nm}: ω_B = {f4(d['bw'])} rad/s ({f4(d['bw'] / TWO_PI)} Hz)",
                f"$ Nyquist: P = {d['P_rhp']}, Z = {d['Z_rhp']}, N = Z − P = {d['Z_rhp'] - d['P_rhp']}"]
        return "\n".join(out)

    # — interpretacja słowna —
    def interpretation(self, d):
        sd, p, info, mg = d["sd"], d["p"], d["info"], d["mg"]
        kind = sd["kind"]
        L = ["# Co oznaczają te wyniki? (interpretacja dla aktualnych suwaków)"]
        # fizyka modelu
        if kind in ("motor", "wl"):
            tau_e = p["L"] / p["R"]
            tau_m = p["J"] * p["R"] / p["K"] ** 2
            wn = np.sqrt((p["R"] * p["B"] + p["K"] ** 2) / (p["L"] * p["J"]))
            zeta = (p["L"] * p["B"] + p["R"] * p["J"]) / (2 * np.sqrt(p["L"] * p["J"] * (p["R"] * p["B"] + p["K"] ** 2)))
            L += ["## 1. Skąd się wzięła transmitancja?",
                  "Model silnika to dwa równania: elektryczne L·di/dt = u − R·i − K·ω oraz "
                  "mechaniczne J·dω/dt = K·i − B·ω. Po transformacji Laplace'a (zamiana "
                  "„d/dt” na „s”) i wyeliminowaniu prądu dostajemy:",
                  "$ ω(s)/U(s) = K / [(L·s + R)(J·s + B) + K²]",
                  f"Elektryczna stała czasowa τe = L/R = {tau_e * 1e3:.3g} ms mówi, jak szybko "
                  f"narasta prąd; elektromechaniczna τm = J·R/K² = {tau_m * 1e3:.4g} ms – jak "
                  f"szybko rozpędza się wirnik. Ich porównanie decyduje o charakterze odpowiedzi: "
                  f"ωn = {wn:.4g} rad/s, ζ = {zeta:.3f}."]
            if zeta >= 1:
                L.append(f"+ ζ ≥ 1 (τm ≥ 4·τe: {tau_m / tau_e:.2f} ≥ 4): sam silnik jest nadtłumiony "
                         "– prędkość narasta bez przeregulowania (jak samochód łagodnie dojeżdżający do celu).")
            else:
                L.append(f"! ζ < 1 (τm/τe = {tau_m / tau_e:.2f} < 4): bieguny są zespolone – silnik "
                         "lekko „przestrzeliwuje” prędkość ustaloną i oscyluje (duża indukcyjność "
                         "w stosunku do bezwładności).")
            if kind == "wl":
                L.append(f"W układzie Ward-Leonarda dochodzi jeszcze wolny biegun wzbudzenia "
                         f"prądnicy s = −1/τ_f = {-1 / p['Tf']:.3g} 1/s. Pole magnetyczne prądnicy "
                         "narasta powoli (duża indukcyjność uzwojenia wzbudzenia), więc to ono "
                         "zwykle dominuje w dynamice całego napędu.")
        else:
            a = p["K"] ** 2 / p["R"] + p["B"]
            tau = p["J"] / a
            L += ["## 1. Skąd się wzięła transmitancja?",
                  "Przy hamowaniu twornik jest zamknięty przez rezystor, więc i = (u − K·ω)/R "
                  "(indukcyjność pomijamy – zmienia się dużo szybciej niż prędkość). Z równania "
                  "J·dω/dt = K·i − B·ω + T_z dostajemy człon INERCYJNY I RZĘDU:",
                  "$ G(s) = k / (J·s + K²/R + B),   τ = J / (K²/R + B)",
                  f"Biegun s = −1/τ = {-1 / tau:.4g} 1/s, τ = {tau:.3f} s, a „połowiczny czas” "
                  f"T₀ = τ·ln2 = {tau * np.log(2):.3f} s – dla B = 0 jest to dokładnie stała z Wildiego "
                  "(wzór 9); B > 0 dodaje straty mechaniczne, które dodatkowo skracają hamowanie. "
                  "Rezystor hamujący działa jak tłumik: im mniejsze R, tym większe K²/R "
                  "i tym szybsze hamowanie."]
        # bieguny i stabilność
        L.append("## 2. Bieguny, zera, postać zpk")
        L.append("Postać zpk (zera–bieguny–wzmocnienie) zapisuje transmitancję jako iloczyn "
                 "prostych czynników. Każdy biegun odpowiada jednemu „składnikowi” ruchu: "
                 "biegun rzeczywisty −a daje składową e^(−at) (zanik z τ = 1/a), para "
                 "zespolona −σ ± jω_d daje gasnące drgania e^(−σt)·sin(ω_d·t).")
        if d["stable"]:
            dom = d["poles"][np.argmin(np.abs(d["poles"].real))]
            L.append(f"+ Wszystkie bieguny leżą w lewej półpłaszczyźnie → układ STABILNY. "
                     f"Biegun dominujący (najbliżej osi urojonej) {f4(dom)} – to on decyduje "
                     f"o czasie ustalania (≈ 4/|Re p| = {4 / abs(dom.real):.4g} s).")
        else:
            L.append("x Co najmniej jeden biegun ma Re ≥ 0 → układ NIESTABILNY: każde zakłócenie rośnie.")
        if len(d["z"]):
            L.append(f"Zero {', '.join(f4(x) for x in d['z'])} pochodzi od regulatora PI "
                     "(s = −Ki/Kp). Zero „przyspiesza” początek odpowiedzi i może zwiększać "
                     "przeregulowanie, nawet jeśli bieguny są dobrze tłumione.")
        # step
        L.append("## 3. Odpowiedź skokowa (step)")
        if d["stable"]:
            if d["closed"]:
                L.append(f"Zadajemy nagle prędkość {d['c']['ref']:.0f} obr/min. Dzięki członowi "
                         "całkującemu (Ki > 0) uchyb ustalony jest ZEROWY – regulator „dokręca” "
                         "napięcie tak długo, aż prędkość dokładnie osiągnie zadaną."
                         if d["c"]["Ki"] > 0 else
                         "Regulator jest tylko proporcjonalny (Ki = 0) – zostaje uchyb ustalony "
                         f"{(1 - d['dc']) * 100:.1f} % (statyzm).")
                Mp = info["Mp"]
                if Mp < 1:
                    L.append(f"+ Brak przeregulowania; czas ustalania {f4(info['ts'])} s.")
                elif Mp < 15:
                    L.append(f"+ Przeregulowanie {Mp:.1f} % – typowe, akceptowalne dla napędu.")
                else:
                    L.append(f"! Przeregulowanie {Mp:.1f} % – za duże dla większości napędów "
                             "(udary mechaniczne). Zmniejsz Ki lub zwiększ Kp.")
                umax = np.max(np.abs(d["yu"])) * d["c"]["ref"] * RPM2RAD
                unom = abs(sd["u_nom"]) * (2 if kind == "motor" else 1.5)
                if umax > unom:
                    L.append(f"x Szczytowe napięcie sterujące {umax:.0f} {sd['u_unit']} przekracza "
                             f"realny zakres (~{unom:.0f} {sd['u_unit']}). W rzeczywistości wystąpi "
                             "nasycenie i odpowiedź będzie wolniejsza – potrzebny anti-windup lub "
                             "mniejsze wzmocnienia/rampa wartości zadanej.")
                else:
                    L.append(f"+ Szczytowe napięcie sterujące {umax:.0f} {sd['u_unit']} mieści się "
                             "w realnym zakresie zasilania.")
            else:
                amp = d["c"]["amp"]
                dn = d["dc"] * amp * RAD2RPM
                L.append(f"Skok wejścia {amp:.4g} {sd['u_unit']} powoduje zmianę prędkości "
                         f"o {dn:.4g} obr/min w stanie ustalonym (wzmocnienie statyczne "
                         f"{d['dc']:.4g} (rad/s)/{sd['u_unit']}). Czas narastania {f4(info['tr'])} s, "
                         f"czas ustalania {f4(info['ts'])} s, przeregulowanie {info['Mp']:.1f} %.")
                if kind == "brake_U":
                    L.append("Całkowity przebieg hamowania przeciwprądem = odpowiedź SWOBODNA "
                             "(prędkość początkowa zanika wykładniczo) + odpowiedź WYMUSZONA "
                             "(skok napięcia −Es dąży do biegu w przeciwną stronę). Zero prędkości "
                             "wypada dokładnie w t_s z przykładu M7 – wtedy trzeba wyłączyć zasilanie.")
                if kind == "brake_T":
                    L.append("Krzywa „hamowanie od n₀” to odpowiedź swobodna – dokładnie przebieg "
                             "z przykładu M6: prędkość maleje o połowę co T₀.")
        # bode
        mgp = mg["PM"]
        L.append("## 4. Charakterystyka Bodego")
        L.append("Bode pokazuje, jak układ reaguje na sinusoidalne wymuszenie o różnych "
                 "częstotliwościach: górny wykres – ile razy wzmacnia (w decybelach, 20·log|G|), "
                 "dolny – o ile opóźnia (przesunięcie fazy). Przy niskich częstotliwościach "
                 "silnik nadąża za zmianami napięcia; przy wysokich bezwładność „nie pozwala” – "
                 f"wzmocnienie spada o 20 dB/dekadę na każdy biegun (tu nadmiar biegunów "
                 f"{len(d['poles']) - len(d['z'])} → nachylenie "
                 f"{-20 * (len(d['poles']) - len(d['z']))} dB/dek, faza dąży do "
                 f"{-90 * (len(d['poles']) - len(d['z']))}°).")
        if np.isfinite(d["bw"]):
            L.append(f"Pasmo −3 dB: ω_B = {d['bw']:.4g} rad/s – zmiany wolniejsze niż "
                     f"~{d['bw'] / TWO_PI:.3g} Hz układ przenosi prawie bez tłumienia.")
        if np.isfinite(mgp):
            if mgp >= 60:
                L.append(f"+ Zapas fazy {mgp:.1f}° – bardzo bezpieczny, odpowiedź spokojna (bez lub z małym przeregulowaniem).")
            elif mgp >= 45:
                L.append(f"+ Zapas fazy {mgp:.1f}° – dobry kompromis szybkość/stabilność (zalecane 45–60°).")
            elif mgp >= 30:
                L.append(f"! Zapas fazy {mgp:.1f}° – dopuszczalny, ale odpowiedź będzie oscylacyjna.")
            elif mgp > 0:
                L.append(f"x Zapas fazy {mgp:.1f}° – układ blisko granicy stabilności!")
            else:
                L.append(f"x Zapas fazy ujemny ({mgp:.1f}°) – zamknięty układ niestabilny.")
        else:
            L.append("|L(jω)| nie przecina 0 dB (wzmocnienie pętli < 1 dla wszystkich ω) – "
                     "zapas fazy nieokreślony (∞): sprzężenie zwrotne nie może tu wywołać niestabilności.")
        if np.isinf(mg["GM"]):
            L.append("+ Zapas wzmocnienia = ∞: faza nigdy nie osiąga −180° (dla obiektu I lub II "
                     "rzędu, lub PI z obiektem II rzędu faza dąży do −180°, ale jej nie przekracza), "
                     "więc teoretycznie wzmocnienie można zwiększać bez utraty stabilności – w praktyce "
                     "ograniczają nas nasycenie napięcia i pominięte opóźnienia (przekształtnik, czujnik).")
        else:
            L.append(f"Zapas wzmocnienia {mg['GM']:.1f} dB – tyle razy (10^(GM/20) = "
                     f"{10 ** (mg['GM'] / 20):.3g}) można zwiększyć wzmocnienie pętli, zanim układ "
                     "stanie się niestabilny. Zalecane ≥ 6 dB.")
        # nyquist
        L.append("## 5. Charakterystyka Nyquista")
        L.append("To ta sama informacja co Bode, ale narysowana na płaszczyźnie zespolonej: "
                 "każdy punkt to L(jω) dla innej częstotliwości. Kluczowy jest punkt krytyczny −1. "
                 "Kryterium Nyquista: liczba okrążeń N punktu −1 (zgodnie z ruchem wskazówek zegara) "
                 "= Z − P, gdzie P – liczba niestabilnych biegunów pętli otwartej, Z – zamkniętej.")
        if d["Z_rhp"] == 0:
            L.append(f"+ Krzywa nie obejmuje punktu −1 (P = {d['P_rhp']}, N = 0 → Z = 0): układ ze "
                     "sprzężeniem zwrotnym jest stabilny.")
        else:
            L.append(f"x Krzywa obejmuje punkt −1: Z = {d['Z_rhp']} biegunów zamkniętej pętli w prawej półpłaszczyźnie.")
        Ms = d["Ms"]
        L.append(f"Najmniejsza odległość krzywej od −1 wynosi 1/M_s = {1 / Ms:.3f} "
                 f"(M_s = {Ms:.2f}). " + ("Wartość M_s < 2 oznacza dobrą odporność na zmiany parametrów."
                                          if Ms < 2 else "M_s ≥ 2 – mała odporność; układ czuły na zmiany parametrów."))
        if not d["closed"]:
            L.append("~ W trybie pętli otwartej Nyquist i zapasy dotyczą hipotetycznego sprzężenia "
                     "jednostkowego u = ω_zad − ω (regulator P o wzmocnieniu 1).")
        return "\n".join(L)

    # — wykresy —
    def draw(self, d):
        sd = d["sd"]
        gs = self.fig.add_gridspec(3, 2)
        ax_m = self.fig.add_subplot(gs[0, 0])
        ax_p = self.fig.add_subplot(gs[1, 0], sharex=ax_m)
        ax_n = self.fig.add_subplot(gs[0:2, 1])
        ax_s = self.fig.add_subplot(gs[2, 0])
        ax_z = self.fig.add_subplot(gs[2, 1])
        w, mg = d["w"], d["mg"]
        # Bode
        labL = "L(jω) = C·G" if d["closed"] else "G(jω)"
        ax_m.semilogx(w, 20 * np.log10(d["magL"]), label=labL)
        ax_p.semilogx(w, d["phL"], label=labL)
        if d["closed"]:
            ax_m.semilogx(w, 20 * np.log10(np.abs(d["Ha"])), "--", color=C["green"],
                          label="T(jω) zamknięta")
            ax_p.semilogx(w, np.degrees(np.unwrap(np.angle(d["Ha"]))), "--",
                          color=C["green"], label="T(jω)")
        ax_m.axhline(0, color=C["red"], lw=0.8, ls=":")
        ax_p.axhline(-180, color=C["red"], lw=0.8, ls=":")
        if np.isfinite(mg["wgc"]):
            ax_m.axvline(mg["wgc"], color=C["yellow"], lw=0.8, ls=":")
            ax_p.axvline(mg["wgc"], color=C["yellow"], lw=0.8, ls=":")
            phc = np.interp(np.log10(mg["wgc"]), np.log10(w), d["phL"])
            ax_p.plot([mg["wgc"]] * 2, [-180, phc], color=C["yellow"], lw=3, alpha=0.7)
            ax_p.annotate(f"PM = {mg['PM']:.1f}°", (mg["wgc"], phc), xytext=(8, -14),
                          textcoords="offset points", color=C["yellow"], fontsize=8)
        if np.isfinite(mg["wpc"]):
            ax_m.annotate(f"GM = {mg['GM']:.1f} dB", (mg["wpc"], -mg["GM"]), xytext=(6, 6),
                          textcoords="offset points", color=C["peach"], fontsize=8)
            ax_m.plot([mg["wpc"]] * 2, [0, -mg["GM"]], color=C["peach"], lw=3, alpha=0.7)
        for pz in d["pG"]:
            if abs(pz) > 1e-9:
                ax_m.axvline(abs(pz), color=C["sub"], lw=0.6, ls="-.", alpha=0.6)
        ax_m.set(title="Bode – moduł", ylabel="|·| [dB]")
        ax_p.set(title="Bode – faza", ylabel="faza [°]", xlabel="ω [rad/s]")
        ax_m.legend(fontsize=7)
        # Nyquist
        H = d["HL"]
        ax_n.plot(H.real, H.imag, label="ω > 0")
        ax_n.plot(H.real, -H.imag, "--", color=C["sub"], lw=1.2, label="ω < 0")
        n_arr = len(H)
        for i in (n_arr // 4, n_arr // 2, 3 * n_arr // 4):
            if i + 1 < n_arr:
                ax_n.annotate("", xy=(H[i + 1].real, H[i + 1].imag), xytext=(H[i].real, H[i].imag),
                              arrowprops=dict(arrowstyle="->", color=C["cyan"], lw=1.6))
        th = np.linspace(0, TWO_PI, 200)
        ax_n.plot(np.cos(th), np.sin(th), ":", color=C["surf2"], lw=1)
        ax_n.plot([-1], [0], "x", color=C["red"], ms=12, mew=3, label="punkt krytyczny −1")
        Ms = d["Ms"]
        ax_n.plot(-1 + np.cos(th) / Ms, np.sin(th) / Ms, color=C["red"], lw=0.8, alpha=0.6,
                  label=f"okrąg 1/M_s = {1 / Ms:.2f}")
        R_ = np.abs(H)
        lim = max(1.6, min(np.max(R_) * 1.1, 3 * max(1.0, np.percentile(R_, 50) * 2)))
        lim = min(lim, max(1.6, 1.1 * np.max(R_)))
        ax_n.set_xlim(-lim, lim)
        ax_n.set_ylim(-lim, lim)
        ax_n.set_aspect("equal", adjustable="datalim")
        ax_n.axhline(0, color=C["fg"], lw=0.5)
        ax_n.axvline(0, color=C["fg"], lw=0.5)
        ax_n.set(title=f"Nyquist – {labL}", xlabel="Re", ylabel="Im")
        ax_n.legend(fontsize=7, loc="lower right")
        # Step
        t, y = d["t"], d["y"]
        if d["closed"]:
            ref = d["c"]["ref"]
            ax_s.plot(t, ref * y, label="prędkość n(t)")
            ax_s.axhline(ref, color=C["red"], ls=":", lw=1, label="zadana")
            ax_s.set(ylabel="n [obr/min]")
            axu = ax_s.twinx()
            axu.plot(t, d["yu"] * ref * RPM2RAD, color=C["peach"], lw=1, ls="--", label="sterowanie u(t)")
            axu.set_ylabel(f"u [{sd['u_unit']}]", color=C["peach"])
            axu.grid(False)
            h1, l1 = ax_s.get_legend_handles_labels()
            h2, l2 = axu.get_legend_handles_labels()
            ax_s.legend(h1 + h2, l1 + l2, fontsize=7, loc="lower right")
        else:
            amp = d["c"]["amp"]
            ax_s.plot(t, amp * y * RAD2RPM, label=f"wymuszona: skok {amp:.4g} {sd['u_unit']}")
            if "n0" in d["c"]:
                pole = d["poles"][0].real
                n_free = d["c"]["n0"] * np.exp(pole * t)
                ax_s.plot(t, n_free, color=C["green"], label="swobodna: hamowanie od n₀")
                if sd.get("total"):
                    tot = n_free + amp * y * RAD2RPM
                    ax_s.plot(t, tot, color=C["yellow"], lw=2.4, label="pełny przebieg")
                    zc = np.where(np.diff(np.sign(tot)) != 0)[0]
                    if len(zc):
                        tz = t[zc[0]]
                        mark(ax_s, tz, 0, f"n = 0 po {tz:.1f} s", C["red"], dy=14)
                ax_s.axhline(0, color=C["fg"], lw=0.6)
            ax_s.set(ylabel="n [obr/min]")
            ax_s.legend(fontsize=7)
        info = d["info"]
        if d["stable"] and np.isfinite(info["ts"]):
            ax_s.axvline(info["ts"], color=C["mauve"], ls=":", lw=1)
        ax_s.set(title="Odpowiedź skokowa", xlabel="t [s]")
        # pz map
        ax_z.axvspan(-1e9, 0, color=C["green"], alpha=0.06)
        pl, z = d["poles"], d["z"]
        ax_z.plot(pl.real, pl.imag, "x", color=C["red"], ms=10, mew=2.5, label="bieguny")
        if len(z):
            ax_z.plot(z.real, z.imag, "o", mfc="none", color=C["cyan"], ms=9, mew=2, label="zera")
        if d["closed"]:
            ax_z.plot(d["pG"].real, d["pG"].imag, "+", color=C["sub"], ms=9, label="bieguny G(s)")
        pts = np.concatenate([pl, z, d["pG"]]) if len(z) else np.concatenate([pl, d["pG"]])
        xr = max(np.max(np.abs(pts.real)), 1e-3) * 1.3
        yr = max(np.max(np.abs(pts.imag)), xr * 0.3) * 1.3
        ax_z.set_xlim(-xr, max(xr * 0.15, 0.1 * xr))
        ax_z.set_ylim(-yr, yr)
        ax_z.axvline(0, color=C["fg"], lw=0.8)
        ax_z.axhline(0, color=C["fg"], lw=0.5)
        if np.max(np.abs(pts.real)) / max(np.min(np.abs(pts.real[np.abs(pts.real) > 1e-12])), 1e-12) > 200:
            ax_z.set_xscale("symlog", linthresh=max(np.min(np.abs(pts.real[np.abs(pts.real) > 1e-12])), 1e-6))
        ax_z.set(title="Mapa zer i biegunów (zpk)", xlabel="Re [1/s]", ylabel="Im [rad/s]")
        ax_z.legend(fontsize=7, loc="upper left")


CONTROL_THEORY = """
# Teoria w pigułce (dla ucznia liceum)
## Transmitancja G(s)
Transmitancja to „przepis”, jak układ przerabia sygnał wejściowy na wyjściowy. Zamiast
rozwiązywać równania różniczkowe, inżynierowie używają transformacji Laplace'a: pochodna
d/dt zamienia się w mnożenie przez s, a równania różniczkowe – w zwykłe ułamki
(wielomian/wielomian). Dzięki temu „sklejanie” elementów (silnik + prądnica + regulator)
sprowadza się do mnożenia ułamków.
## zpk – zera, bieguny, wzmocnienie
Rozkładając licznik i mianownik na czynniki dostajemy G(s) = k·(s − z₁)…/((s − p₁)(s − p₂)…).
Bieguny p to „naturalne tryby” układu: ich część rzeczywista mówi, jak szybko coś zanika
(im bardziej na lewo, tym szybciej), część urojona – jak szybko drga. Biegun w prawej
półpłaszczyźnie = niestabilność (np. sprzężenie mikrofonu z głośnikiem – pisk rośnie).
## Odpowiedź skokowa (step)
Co się dzieje, gdy nagle przełączymy wejście z 0 na stałą wartość (np. włączamy napięcie).
Parametry: czas narastania (od 10 do 90 %), czas ustalania (wejście w pas ±2 %),
przeregulowanie (o ile procent przestrzeli), wartość ustalona.
## Charakterystyka Bodego
Wymuszamy sinusoidą o częstotliwości ω i patrzymy, jak duża i jak opóźniona jest odpowiedź.
Oś częstotliwości jest logarytmiczna, moduł w decybelach (20 dB = 10×). Zapas fazy (PM)
i zapas wzmocnienia (GM) mówią, jak daleko jesteśmy od granicy stabilności:
PM = 180° + faza przy |L| = 1;  GM = −|L| [dB] przy fazie −180°.
Reguły kciuka: PM ≈ 45–60°, GM ≥ 6 dB.  Przybliżenie: przeregulowanie ≈ 70 − PM [%] (dla PM 30–70°).
## Charakterystyka Nyquista
Wykres L(jω) na płaszczyźnie zespolonej. Jeżeli pętla otwarta jest stabilna, to układ
zamknięty jest stabilny wtedy i tylko wtedy, gdy krzywa NIE obejmuje punktu −1. Odległość
od −1 to miara „zapasu bezpieczeństwa” (odporności).
## Regulator PI
u(t) = Kp·e(t) + Ki·∫e dt. Część P reaguje na bieżący uchyb, część I „pamięta” uchyb
z przeszłości i usuwa go do zera w stanie ustalonym. To standardowy regulator prędkości
w napędach DC (w przekształtnikach tyrystorowych i tranzystorowych).
~ Uwaga: w modelach przyjęto liniowość (Φ = const, brak nasycenia, brak ograniczeń napięcia/prądu).
~ Parametry oznaczone „(zał.)” nie występują w książce – przyjęto typowe wartości.
"""


# ─────────────────────────────────────────────────────────────────────────────
#  ZAKŁADKA STARTOWA – przegląd + weryfikacja z książką
# ─────────────────────────────────────────────────────────────────────────────
def verification_rows():
    """(przykład, wielkość, wartość z książki, wartość obliczona, jednostka)"""
    g1 = calc_g1(90, 4, 0.04, 600)
    g2 = calc_g2(12, 72, 240, 2400)
    m1a = calc_m1(150, 1, 50, 500, 1000)
    m1b = calc_m1(150, 1, 50, 500, 1460)
    m2 = calc_m2(225e3, 250, 1200, 243, 1, 81, 6, 0.559, 0.235)
    m3a = calc_m3(400, 380, 0.01, 300, 500)
    m3b = calc_m3(350, 380, 0.01, 300, 500)
    m4 = calc_m4(120, 51, 120, 0.1, 1500)
    m5 = calc_m5(15, 240, 1780, 54, 24)
    m6 = calc_m6(177, 1280, 210, 0.2, 8000, 20)
    m7 = calc_m7(177, 1280, 210, 0.4)
    return [
        ("G1", "napięcie indukowane E₀", 288, g1["E"], "V"),
        ("G2", "prąd zestawu szczotek", 400, g2["I_brush"], "A"),
        ("G2", "prąd cewki", 200, g2["I_coil"], "A"),
        ("G2", "średnie napięcie na cewkę", 40, g2["E_avg"], "V"),
        ("M1", "prąd rozruchowy", 150, m1a["I_start"], "A"),
        ("M1", "SEM przy 1000 obr/min", 100, m1a["Eo"], "V"),
        ("M1", "SEM przy 1460 obr/min", 146, m1b["Eo"], "V"),
        ("M1", "prąd przy 1000 obr/min", 50, m1a["I"], "A"),
        ("M1", "prąd przy 1460 obr/min", 4, m1b["I"], "A"),
        ("M2", "prąd znamionowy", 900, m2["I"], "A"),
        ("M2", "przewodniki w żłobku", 6, m2["cond_slot"], "—"),
        ("M2", "moment", 1791, m2["T"], "N·m"),
        ("M2", "strumień na biegun", 25.7, m2["phi"] * 1e3, "mWb"),
        ("M3a", "prąd twornika", 2000, m3a["I"], "A"),
        ("M3a", "moc", 760, m3a["P"] / 1e3, "kW"),
        ("M3a", "prędkość", 228, m3a["n"], "obr/min"),
        ("M3a", "moment", 31.8, m3a["T"] / 1e3, "kN·m"),
        ("M3b", "prąd twornika", -3000, m3b["I"], "A"),
        ("M3b", "moc oddawana", 1140, -m3b["P"] / 1e3, "kW"),
        ("M3b", "moment hamujący", 47.7, -m3b["T"] / 1e3, "kN·m"),
        ("M4", "prąd twornika", 50, m4["Ia"], "A"),
        ("M4", "SEM", 115, m4["Eo"], "V"),
        ("M4", "moc mechaniczna", 5750, m4["Pm"], "W"),
        ("M5", "moment bazowy", 60, m5["TB"], "N·m"),
        ("M5", "prędkość", 2492, m5["n"], "obr/min"),
        ("M5", "prąd", 32.4, m5["I"], "A"),
        ("M5", "sprawność", 80.5, m5["eta"] * 100, "%"),
        ("M6", "moc początkowa P₁", 220.5, m6["P1"] / 1e3, "kW"),
        ("M6", "stała T₀", 10, m6["To"], "s"),
        ("M6", "czas do 20 obr/min", 60, m6["t2"], "s"),
        ("M6", "czas – tylko straty", 1650, m6["t2c"], "s"),
        ("M7", "prąd początkowy", 1050, m7["I1"], "A"),
        ("M7", "moc od silnika", 220.5, m7["P_motor"] / 1e3, "kW"),
        ("M7", "czas zatrzymania", 20, m7["ts"], "s"),
    ]


OVERVIEW = """
# Maszyny prądu stałego – wszystkie przykłady z załącznika (Wildi)
Aplikacja to interaktywny „zeszyt” do rozdziałów o prądnicach i silnikach prądu stałego.
Każdy przykład ma własną zakładkę z: suwakami (dane z książki są wartościami domyślnymi),
rozwiązaniem krok po kroku liczonym na żywo, wykresami oraz objaśnieniem na poziomie
liceum z komentarzem doświadczonego inżyniera.

## Jak korzystać
- Przesuń suwak albo wpisz wartość w pole obok i naciśnij Enter.
- „↺ Dane z książki” przywraca wartości z przykładu.
- Pasek narzędzi pod wykresem: lupa (zoom), krzyżyk (przesuwanie), dyskietka (zapis PNG).
- Kliknij lewym przyciskiem na wykresie, aby postawić znacznik z odczytem; prawym – usuń znaczniki.
- Zakładka „Analiza sterowania”: Bode, zpk, Nyquist i odpowiedź skokowa dla układów
  dynamicznych (silniki M1, M4, napęd Ward-Leonarda M3, hamowanie M6 i M7), w pętli otwartej
  i zamkniętej z regulatorem PI – z automatyczną interpretacją wyników.

## Najważniejsze wzory rozdziałów
$ E₀ = Z·n·Φ/60                (napięcie indukowane / SEM)
$ I = (Es − E₀)/R             (prąd twornika silnika)
$ P = E₀·I,   T = Z·Φ·I/2π = 9,55·P/n
$ n = 60·(Es − I·R)/(Z·Φ) ≈ 60·Es/(Z·Φ)   (prędkość ~ napięcie / strumień)
$ T₀ = J·n₁²/(131,5·P₁)       (hamowanie: czas spadku prędkości o połowę)

## Weryfikacja: wynik z książki vs obliczenia aplikacji (tabela poniżej)
Wszystkie przykłady obliczono tymi samymi funkcjami, których używają zakładki.
Różnice < 1 % wynikają z zaokrągleń w książce (np. 9,55 zamiast 60/2π, 131,5 zamiast 131,56).
"""


class OverviewTab(ttk.Frame):
    tab_title = "Start"
    tab_short = "Start ✔"

    def __init__(self, nb, app):
        super().__init__(nb)
        pw = ttk.PanedWindow(self, orient="vertical")
        pw.pack(fill="both", expand=True)
        tf = ttk.Frame(pw)
        txt = tk.Text(tf, height=14)
        setup_text_tags(txt, 11)
        sb = ttk.Scrollbar(tf, orient="vertical", command=txt.yview)
        txt.configure(yscrollcommand=sb.set)
        txt.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        render_markup(txt, OVERVIEW)
        bf = ttk.Frame(pw)
        cols = ("ex", "q", "book", "app", "unit", "ok")
        tree = ttk.Treeview(bf, columns=cols, show="headings", height=14)
        for c, h, wd in zip(cols, ("Przykład", "Wielkość", "Książka", "Aplikacja", "Jedn.", "Zgodność"),
                            (80, 260, 110, 110, 80, 110)):
            tree.heading(c, text=h)
            tree.column(c, width=wd, anchor="w" if c == "q" else "center")
        tree.tag_configure("ok", foreground=C["green"])
        tree.tag_configure("bad", foreground=C["red"])
        for ex, q, book, val, unit in verification_rows():
            err = abs(val - book) / max(abs(book), 1e-12) * 100
            ok = err < 1.5
            tree.insert("", "end", values=(ex, q, f"{book:g}", f"{val:.4g}", unit,
                                           f"{'✔' if ok else '✘'} {err:.2f} %"),
                        tags=("ok" if ok else "bad",))
        tsb = ttk.Scrollbar(bf, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=tsb.set)
        tree.pack(side="left", fill="both", expand=True)
        tsb.pack(side="right", fill="y")
        pw.add(tf, weight=1)
        pw.add(bf, weight=1)


# ─────────────────────────────────────────────────────────────────────────────
#  APLIKACJA
# ─────────────────────────────────────────────────────────────────────────────
class App:
    TABS = [OverviewTab, TabG1, TabG2, TabM1, TabM2, TabM3, TabM4, TabM5, TabM6, TabM7,
            ControlTab]

    def __init__(self, root):
        self.root = root
        root.title("Maszyny prądu stałego – przykłady z Wildiego (prądnice i silniki DC)")
        root.geometry("1500x930")
        root.minsize(1100, 700)
        root.configure(bg=C["bg"])
        self._style()
        self.nb = ttk.Notebook(root)
        self.nb.pack(fill="both", expand=True)
        self._status = tk.StringVar(value="Gotowe.")
        ttk.Label(root, textvariable=self._status, style="Status.TLabel",
                  anchor="w").pack(fill="x", side="bottom")
        self.tabs = []
        for T in self.TABS:
            tab = T(self.nb, self)
            self.nb.add(tab, text=getattr(T, "tab_short", T.tab_title))
            self.tabs.append(tab)

    def status(self, msg):
        self._status.set(msg)

    def _style(self):
        s = ttk.Style(self.root)
        try:
            s.theme_use("clam")
        except tk.TclError:
            pass
        s.configure(".", background=C["bg"], foreground=C["fg"], fieldbackground=C["surf"],
                    bordercolor=C["surf2"], lightcolor=C["surf"], darkcolor=C["bg"],
                    troughcolor=C["surf"], font=(FONT, 10))
        s.configure("TFrame", background=C["bg"])
        s.configure("TLabel", background=C["bg"], foreground=C["fg"])
        s.configure("Sub.TLabel", background=C["bg"], foreground=C["sub"], font=(FONT, 9, "italic"))
        s.configure("Status.TLabel", background=C["panel"], foreground=C["cyan"], padding=4)
        s.configure("TLabelframe", background=C["bg"], bordercolor=C["surf2"])
        s.configure("TLabelframe.Label", background=C["bg"], foreground=C["mauve"],
                    font=(FONT, 10, "bold"))
        s.configure("TNotebook", background=C["panel"], borderwidth=0)
        s.configure("TNotebook.Tab", background=C["surf"], foreground=C["sub"],
                    padding=(7, 4), font=(FONT, 10))
        s.map("TNotebook.Tab", background=[("selected", C["bg"])],
              foreground=[("selected", C["cyan"])])
        s.configure("TButton", background=C["surf"], foreground=C["fg"], padding=(8, 4))
        s.map("TButton", background=[("active", C["surf2"])])
        s.configure("TEntry", fieldbackground=C["surf"], foreground=C["yellow"],
                    insertcolor=C["fg"])
        s.configure("Horizontal.TScale", background=C["cyan"], troughcolor=C["surf"])
        s.configure("TRadiobutton", background=C["bg"], foreground=C["fg"])
        s.map("TRadiobutton", foreground=[("disabled", C["surf2"])],
              background=[("active", C["bg"])])
        s.configure("TCombobox", fieldbackground=C["surf"], foreground=C["fg"],
                    background=C["surf"], arrowcolor=C["cyan"])
        s.map("TCombobox", fieldbackground=[("readonly", C["surf"])],
              foreground=[("readonly", C["fg"])])
        self.root.option_add("*TCombobox*Listbox.background", C["surf"])
        self.root.option_add("*TCombobox*Listbox.foreground", C["fg"])
        s.configure("Treeview", background=C["panel"], fieldbackground=C["panel"],
                    foreground=C["fg"], rowheight=24, font=(FONT, 10))
        s.configure("Treeview.Heading", background=C["surf"], foreground=C["cyan"],
                    font=(FONT, 10, "bold"))
        s.configure("TPanedwindow", background=C["surf2"])
        s.configure("Vertical.TScrollbar", background=C["surf"], troughcolor=C["bg"],
                    arrowcolor=C["sub"])


def main():
    setup_matplotlib()
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
