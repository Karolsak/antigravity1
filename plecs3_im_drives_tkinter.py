"""
Silnik indukcyjny: modele (rozdz. 8) i sterowanie (rozdz. 9) - aplikacja Tkinter
================================================================================
Interaktywne przyklady z zalacznika "Plecs3" (De Doncker, Pulle, Veltman,
"Advanced Electrical Drives", rozdz. 8 "Induction Machine Modeling Concepts"
i rozdz. 9 "Control of Induction Machine Drives").

Kazda zakladka = jeden przyklad:
  * suwaki / pola z parametrami (po lewej) - wykresy odswiezaja sie same,
  * wykresy Matplotlib z paskiem narzedzi (zoom, przesuwanie, zapis PNG),
  * wyniki liczbowe,
  * "Wyjasnienie" - opis jezykiem zrozumialym dla ucznia liceum,
  * dla ukladow sterowania dodatkowo zakladka "Analiza LTI": charakterystyki
    Bodego, postac zpk, mapa zer i biegunow, wykres Nyquista, odpowiedz
    skokowa + automatyczna interpretacja wynikow.

Uruchomienie:   python plecs3_im_drives_tkinter.py
Test bez okna:  python plecs3_im_drives_tkinter.py --selftest [katalog_png]
Wymagania:      numpy, matplotlib (tkinter jest w standardowym Pythonie)
"""
import math
import os
import sys
import traceback
import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.patches import Ellipse

import plecs3_models as M
from plecs3_teksty import TEXTS, LTI_INFO, INTRO

PI = math.pi
RPM = M.RPM
matplotlib.rcParams.update({"font.size": 8.5, "axes.grid": True, "grid.alpha": 0.35,
                            "legend.fontsize": 7.5, "axes.titlesize": 9.5,
                            "lines.linewidth": 1.4})
COL = dict(b="#1f77b4", o="#ff7f0e", g="#2ca02c", r="#d62728", p="#9467bd", br="#8c564b",
           pk="#e377c2", gr="#7f7f7f", ol="#bcbd22", c="#17becf", k="#222222")
MONO = ("Consolas", 9) if sys.platform.startswith("win") else ("DejaVu Sans Mono", 9)


def fv(v):
    """Formatowanie wartosci suwaka."""
    if v == 0:
        return "0"
    a = abs(v)
    if a >= 1000:
        return f"{v:.0f}"
    if a >= 100:
        return f"{v:.1f}"
    if a >= 1:
        return f"{v:.3g}" if a < 10 else f"{v:.2f}".rstrip("0").rstrip(".")
    if a >= 1e-3:
        return f"{v:.4g}"
    return f"{v:.3e}"


def arrow(ax, z, color, label=None, origin=0j, lw=2.0, ls="-"):
    ax.annotate("", xy=(origin.real + z.real, origin.imag + z.imag),
                xytext=(origin.real, origin.imag),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls=ls,
                                shrinkA=0, shrinkB=0, mutation_scale=12))
    ax.plot([origin.real, origin.real + z.real], [origin.imag, origin.imag + z.imag],
             alpha=0.0)                      # niewidoczne punkty -> poprawne autoskalowanie
    if label:
        ax.plot([], [], color=color, lw=lw, label=label)


def twin(ax, color):
    ax2 = ax.twinx()
    ax2.grid(False)
    ax2.tick_params(axis="y", colors=color)
    ax2.yaxis.label.set_color(color)
    return ax2


# =============================================================================
# BAZA ZAKLADKI
# =============================================================================

class Tab(ttk.Frame):
    KEY = ""
    TITLE = ""
    HAS_LTI = False
    FIGSIZE = (9.5, 6.4)

    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        self.vars = {}
        self._job = None
        self.done = False
        pw = ttk.PanedWindow(self, orient="horizontal")
        pw.pack(fill="both", expand=True)
        left = ttk.Frame(pw)
        pw.add(left, weight=0)
        self.right = ttk.Notebook(pw)
        pw.add(self.right, weight=1)

        lpw = ttk.PanedWindow(left, orient="vertical")
        lpw.pack(fill="both", expand=True)
        cframe = ttk.Frame(lpw)
        lpw.add(cframe, weight=3)
        cv = tk.Canvas(cframe, width=330, highlightthickness=0)
        sb = ttk.Scrollbar(cframe, orient="vertical", command=cv.yview)
        self.ctrl = ttk.Frame(cv)
        self.ctrl.bind("<Configure>", lambda e: cv.configure(scrollregion=cv.bbox("all")))
        cv.create_window((0, 0), window=self.ctrl, anchor="nw", width=318)
        cv.configure(yscrollcommand=sb.set)
        cv.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        def _wheel(e):
            cv.yview_scroll(int(-1 * (e.delta / 120)) if e.delta else (1 if e.num == 5 else -1), "units")
        cv.bind("<Enter>", lambda e: (cv.bind_all("<MouseWheel>", _wheel),
                                      cv.bind_all("<Button-4>", _wheel),
                                      cv.bind_all("<Button-5>", _wheel)))
        cv.bind("<Leave>", lambda e: (cv.unbind_all("<MouseWheel>"), cv.unbind_all("<Button-4>"),
                                      cv.unbind_all("<Button-5>")))
        rframe = ttk.LabelFrame(lpw, text="Wyniki liczbowe")
        lpw.add(rframe, weight=2)
        self.res = ScrolledText(rframe, width=44, height=14, font=MONO, wrap="word")
        self.res.pack(fill="both", expand=True)

        ttl = ttk.Label(self.ctrl, text=self.TITLE, font=("TkDefaultFont", 10, "bold"),
                        wraplength=300, justify="left")
        ttl.pack(anchor="w", padx=6, pady=(6, 2))

        f1 = ttk.Frame(self.right)
        self.right.add(f1, text="  Wykresy  ")
        self.fig = Figure(figsize=self.FIGSIZE, layout="constrained")
        self.canvas = FigureCanvasTkAgg(self.fig, master=f1)
        tb = NavigationToolbar2Tk(self.canvas, f1, pack_toolbar=False)
        tb.update()
        tb.pack(side="bottom", fill="x")
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        if self.HAS_LTI:
            self.lti = LTIPanel(self.right, self)
            self.right.add(self.lti, text="  Analiza LTI: Bode · zpk · Nyquist · step  ")
        f3 = ttk.Frame(self.right)
        self.right.add(f3, text="  Wyjaśnienie (dla ucznia)  ")
        tx = ScrolledText(f3, wrap="word", font=("TkDefaultFont", 10), padx=12, pady=8)
        tx.pack(fill="both", expand=True)
        tx.insert("1.0", TEXTS.get(self.KEY, "(brak opisu)"))
        tx.configure(state="disabled")
        self.build()
        ttk.Button(self.ctrl, text="Przelicz / symuluj", command=self.refresh).pack(
            fill="x", padx=6, pady=8)

    # --- widgety ------------------------------------------------------------
    def section(self, text):
        ttk.Label(self.ctrl, text=text, font=("TkDefaultFont", 9, "bold"),
                  foreground="#1f4e79").pack(anchor="w", padx=6, pady=(8, 0))

    def note(self, text):
        ttk.Label(self.ctrl, text=text, wraplength=300, justify="left",
                  foreground="#555").pack(anchor="w", padx=8)

    def slider(self, key, label, lo, hi, val, res=None, unit=""):
        fr = ttk.Frame(self.ctrl)
        fr.pack(fill="x", padx=6, pady=1)
        var = tk.DoubleVar(value=val)
        top = ttk.Frame(fr)
        top.pack(fill="x")
        lab = ttk.Label(top, text=label)
        lab.pack(side="left")
        ent = ttk.Entry(top, width=9, justify="right")
        ent.pack(side="right")

        def snap(v):
            if res:
                v = round(v / res) * res
            return min(max(v, lo), hi)

        def show(*_):
            ent.delete(0, "end")
            ent.insert(0, fv(snap(var.get())))
            lab.config(text=f"{label} [{unit}]" if unit else label)

        def moved(_e=None):
            show()
            self.schedule()

        def typed(_e=None):
            try:
                v = float(ent.get().replace(",", "."))
                var.set(min(max(v, lo), hi))
                show()
                self.schedule(50)
            except ValueError:
                show()
        sc = ttk.Scale(fr, from_=lo, to=hi, variable=var, command=moved)
        sc.pack(fill="x")
        ent.bind("<Return>", typed)
        ent.bind("<FocusOut>", typed)
        show()
        self.vars[key] = ("s", var, snap)

    def combo(self, key, label, values, val):
        fr = ttk.Frame(self.ctrl)
        fr.pack(fill="x", padx=6, pady=2)
        ttk.Label(fr, text=label).pack(anchor="w")
        var = tk.StringVar(value=val)
        cb = ttk.Combobox(fr, textvariable=var, values=values, state="readonly")
        cb.pack(fill="x")
        cb.bind("<<ComboboxSelected>>", lambda e: self.schedule(10))
        self.vars[key] = ("c", var, None)

    def check(self, key, label, val):
        var = tk.BooleanVar(value=val)
        ttk.Checkbutton(self.ctrl, text=label, variable=var,
                        command=lambda: self.schedule(10)).pack(anchor="w", padx=6, pady=1)
        self.vars[key] = ("b", var, None)

    def v(self, key):
        kind, var, snap = self.vars[key]
        if kind == "s":
            return snap(var.get())
        return var.get()

    def setv(self, key, val):
        self.vars[key][1].set(val)

    # --- odswiezanie ---------------------------------------------------------
    def schedule(self, ms=300):
        if self._job is not None:
            self.after_cancel(self._job)
        self._job = self.after(ms, self.refresh)

    def refresh(self):
        self._job = None
        self.done = True
        try:
            self.app.status(f"Obliczam: {self.TITLE} …")
            self.config(cursor="watch")
            self.update_idletasks()
            self.fig.clear()
            self.compute()
            self.canvas.draw_idle()
            if self.HAS_LTI:
                self.lti.update_plots()
            self.app.status("Gotowe. Zmieniaj suwaki – wykresy przeliczą się automatycznie.")
        except Exception:
            self.set_result("BŁĄD OBLICZEŃ:\n" + traceback.format_exc())
            self.app.status("Błąd – szczegóły w polu wyników.")
        finally:
            self.config(cursor="")

    def set_result(self, text):
        self.res.configure(state="normal")
        self.res.delete("1.0", "end")
        self.res.insert("1.0", text)
        self.res.configure(state="disabled")

    def axes(self, nr, nc, **kw):
        return self.fig.subplots(nr, nc, squeeze=False, **kw)

    # do nadpisania
    def build(self):
        pass

    def compute(self):
        pass

    def lti_systems(self):
        return []


# =============================================================================
# PANEL ANALIZY LTI
# =============================================================================

class LTIPanel(ttk.Frame):
    def __init__(self, master, tab):
        super().__init__(master)
        self.tab = tab
        top = ttk.Frame(self)
        top.pack(fill="x", padx=4, pady=3)
        ttk.Label(top, text="Badana transmitancja:").pack(side="left")
        self.sel = ttk.Combobox(top, state="readonly", width=95)
        self.sel.pack(side="left", padx=4, fill="x", expand=True)
        self.sel.bind("<<ComboboxSelected>>", lambda e: self.update_plots(keep=True))
        pw = ttk.PanedWindow(self, orient="vertical")
        pw.pack(fill="both", expand=True)
        ff = ttk.Frame(pw)
        pw.add(ff, weight=3)
        self.fig = Figure(figsize=(9.5, 4.8), layout="constrained")
        self.canvas = FigureCanvasTkAgg(self.fig, master=ff)
        tb = NavigationToolbar2Tk(self.canvas, ff, pack_toolbar=False)
        tb.update()
        tb.pack(side="bottom", fill="x")
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        tf = ttk.LabelFrame(pw, text="Wyniki analizy i ich wyjaśnienie")
        pw.add(tf, weight=2)
        self.txt = ScrolledText(tf, font=MONO, wrap="word", height=12)
        self.txt.pack(fill="both", expand=True)
        self.systems = []

    def update_plots(self, keep=False):
        idx = self.sel.current() if self.sel.current() >= 0 else 0
        self.systems = self.tab.lti_systems()
        names = [s["name"] for s in self.systems]
        self.sel["values"] = names
        if not names:
            return
        idx = min(idx, len(names) - 1)
        self.sel.current(idx)
        s = self.systems[idx]
        try:
            self.draw(s)
        except Exception:
            self.txt.delete("1.0", "end")
            self.txt.insert("1.0", traceback.format_exc())
        self.canvas.draw_idle()

    def draw(self, s):
        G = s["G"]
        kind = s.get("kind", "plant")
        f = self.fig
        f.clear()
        gs = f.add_gridspec(2, 3)
        axm = f.add_subplot(gs[0, 0])
        axp = f.add_subplot(gs[1, 0], sharex=axm)
        axn = f.add_subplot(gs[0, 1])
        axs = f.add_subplot(gs[1, 1])
        axz = f.add_subplot(gs[:, 2])
        w = G.wrange(800)
        H = G.freqresp(w)
        mag = 20 * np.log10(np.abs(H) + 1e-300)
        ph = np.unwrap(np.angle(H)) * 180 / PI
        if ph[0] > 90:
            ph -= 360 * math.ceil((ph[0] - 90) / 360)
        if ph[0] < -270:
            ph += 360 * math.ceil((-270 - ph[0]) / 360)
        axm.semilogx(w, mag, color=COL["b"])
        axm.set_ylabel("|G| [dB]")
        axm.set_title("Bode – amplituda")
        axp.semilogx(w, ph, color=COL["r"])
        if ph.min() < -600:
            axp.set_ylim(-600, max(ph.max(), -150) + 20)
        axp.set_ylabel("faza [°]")
        axp.set_xlabel("ω [rad/s]")
        axp.set_title("Bode – faza")
        if kind == "loop":
            m = M.margins(G)
            axm.axhline(0, color=COL["k"], lw=0.8)
            axp.axhline(-180, color=COL["k"], lw=0.8, ls="--")
            if m["wc"]:
                axm.axvline(m["wc"], color=COL["g"], ls=":")
                axp.axvline(m["wc"], color=COL["g"], ls=":")
                axp.annotate(f"PM={m['pm']:.1f}°", xy=(m["wc"], -180 + m["pm"]),
                             xytext=(5, 10), textcoords="offset points", color=COL["g"])
                axp.plot([m["wc"], m["wc"]], [-180, -180 + m["pm"]], color=COL["g"], lw=2.5)
            if m["gm"] is not None:
                axm.axvline(m["w180"], color=COL["p"], ls=":")
                axm.annotate(f"GM={m['gm']:.1f} dB", xy=(m["w180"], -m["gm"]),
                             xytext=(5, -12), textcoords="offset points", color=COL["p"])
                axm.plot([m["w180"], m["w180"]], [0, -m["gm"]], color=COL["p"], lw=2.5)
        else:
            dc = G.dcgain()
            if math.isfinite(dc) and dc != 0:
                axm.axhline(20 * math.log10(abs(dc)) - 3, color=COL["gr"], ls=":", lw=0.9)
        # Nyquist
        Hn = H
        lim = np.percentile(np.abs(Hn), 85) * 1.6 + 1e-12
        if kind == "loop":
            lim = 2.5                         # okolica punktu krytycznego -1 (lupa: zoom)
        axn.plot(Hn.real, Hn.imag, color=COL["b"], label="ω > 0")
        axn.plot(Hn.real, -Hn.imag, color=COL["b"], ls="--", lw=0.9, label="ω < 0")
        for frac in (0.3, 0.55, 0.8):
            i = int(frac * (len(w) - 2))
            if abs(Hn[i]) < lim:
                axn.annotate("", xy=(Hn[i + 1].real, Hn[i + 1].imag), xytext=(Hn[i].real, Hn[i].imag),
                             arrowprops=dict(arrowstyle="->", color=COL["b"]))
        if kind == "loop":
            axn.plot([-1], [0], "x", color=COL["r"], ms=9, mew=2, label="punkt −1")
            th = np.linspace(0, 2 * PI, 200)
            axn.plot(np.cos(th), np.sin(th), color=COL["gr"], lw=0.6, ls=":")
        axn.set_xlim(-lim, lim)
        axn.set_ylim(-lim, lim)
        axn.set_aspect("equal", adjustable="box")
        axn.axhline(0, color=COL["k"], lw=0.6)
        axn.axvline(0, color=COL["k"], lw=0.6)
        axn.set_title("Nyquist " + ("L(jω)" if kind == "loop" else "G(jω)"))
        axn.set_xlabel("Re")
        axn.set_ylabel("Im")
        axn.legend(loc="best")
        # Step
        if kind == "loop":
            T = G.feedback()
            t, y = T.step(s.get("t_end"))
            axs.set_title("Odpowiedź skokowa układu zamkniętego T = L/(1+L)")
            if t is not None:
                axs.plot(t, y, color=COL["g"])
                axs.axhline(1.0, color=COL["gr"], ls=":")
        else:
            t, y = G.step(s.get("t_end"))
            axs.set_title("Odpowiedź na skok jednostkowy " + s.get("in", ""))
            if t is not None:
                axs.plot(t, y, color=COL["g"])
            else:
                axs.text(0.5, 0.5, "transmitancja niewłaściwa:\nskok → impuls Diraca",
                         ha="center", va="center", transform=axs.transAxes)
        axs.set_xlabel("t [s]")
        axs.set_ylabel(s.get("out", "y"))
        # mapa zer i biegunow
        p = G.poles()
        z = G.zeros()
        axz.axvspan(0, 1e9, color=COL["r"], alpha=0.06)
        axz.plot(p.real, p.imag, "x", color=COL["r"], ms=9, mew=2, label="bieguny G/L")
        if len(z):
            axz.plot(z.real, z.imag, "o", mfc="none", color=COL["b"], ms=8, mew=1.6, label="zera")
        if kind == "loop":
            pc = G.feedback().poles()
            axz.plot(pc.real, pc.imag, "s", mfc="none", color=COL["g"], ms=8, label="bieguny T (zamk.)")
        pts = np.concatenate([p, z]) if len(z) else p
        if kind == "loop":
            pts = np.concatenate([pts, G.feedback().poles()])
        pts = pts[np.isfinite(pts)]
        if len(pts):
            core = pts[np.abs(pts) <= 30 * (np.median(np.abs(pts)) + 1e-9)]
            if len(core) == 0:
                core = pts
            r = max(np.max(np.abs(core.real)), np.max(np.abs(core.imag)), 1e-3) * 1.25
            axz.set_xlim(-r, r * 0.6)
            axz.set_ylim(-r, r)
        axz.axhline(0, color=COL["k"], lw=0.6)
        axz.axvline(0, color=COL["k"], lw=0.8)
        axz.set_title("Mapa zer i biegunów (zpk)")
        axz.set_xlabel("Re s [1/s]")
        axz.set_ylabel("Im s [rad/s]")
        axz.legend(loc="lower left")
        info = s.get("info", "")
        if s.get("key") in LTI_INFO:
            info = (info + "\n\n" if info else "") + LTI_INFO[s["key"]]
        rep = M.lti_report(G, kind, extra="")
        hdr = f"{s['name']}\nwejście: {s.get('in', '-')}   wyjście: {s.get('out', '-')}\n"
        self.txt.delete("1.0", "end")
        self.txt.insert("1.0", hdr + "\n" + rep + ("\n\nINTERPRETACJA INŻYNIERSKA:\n" + info if info else ""))


# =============================================================================
# ROZDZIAL 8 - MODELE
# =============================================================================

class T81Tab(Tab):
    KEY = "t81"
    TITLE = "8.6.1 Tutorial 1 – uproszczony model maszyny (bez rozproszeń)"

    def build(self):
        self.section("Wirujący strumień (wejście modelu)")
        self.slider("psi", "amplituda strumienia ψm", 0.5, 1.6, 1.25, 0.01, "Vs")
        self.slider("f", "częstotliwość f", 5, 60, 50, 0.5, "Hz")
        self.section("Parametry maszyny")
        self.slider("Rs", "rezystancja stojana Rs", 1, 12, 6.9, 0.1, "Ω")
        self.slider("Rr", "rezystancja wirnika Rr", 0.5, 8, 3.0, 0.1, "Ω")
        self.slider("Lm", "indukcyjność magnesująca Lm", 0.1, 0.6, 0.3409, 0.0001, "H")
        self.slider("J", "moment bezwładności J", 0.0005, 0.02, 0.001, 0.0005, "kg·m²")
        self.section("Obciążenie wentylatorowe T = k·Ω²")
        self.slider("TL", "moment obciążenia przy 1500 obr/min", 0, 25, 12.7, 0.1, "Nm")
        self.slider("Tend", "czas symulacji", 0.002, 3, 0.01, 0.001, "s")
        self.note("Przy J = 0,001 kg·m² (jak w książce) rozruch trwa ułamek milisekundy – "
                  "zwiększ J, żeby zobaczyć go wyraźniej.")

    def compute(self):
        k = self.v("TL") / (1500 / RPM) ** 2
        t, n, Te, Tl, ss = M.t81_simplified(self.v("psi"), self.v("f"), self.v("Rs"), self.v("Rr"),
                                            self.v("Lm"), 2, self.v("J"), k, self.v("Tend"))
        ax = self.axes(2, 2)
        a = ax[0, 0]
        a.plot(t, Te, color=COL["b"], label="Te – moment silnika")
        a.plot(t, Tl, color=COL["o"], label="Tl – moment obciążenia")
        a.set_xlabel("t [s]")
        a.set_ylabel("moment [Nm]")
        a.set_title("Rozruch: moment i prędkość")
        a2 = twin(a, COL["g"])
        a2.plot(t, n, color=COL["g"], label="n")
        a2.set_ylabel("n [obr/min]")
        a.legend(loc="center right")
        # charakterystyka mechaniczna
        a = ax[0, 1]
        ws = 2 * PI * self.v("f")
        ns = ws / 2 * RPM
        nn = np.linspace(0, ns * 1.05, 300)
        Te_n = ss["kT"] * (ws - 2 * nn / RPM)
        a.plot(nn, Te_n, color=COL["b"], label="silnik: Te = 3/2·p·ψ²·ωsl/Rr")
        a.plot(nn, k * (nn / RPM) ** 2, color=COL["o"], label="wentylator k·Ω²")
        a.plot([ss["n"]], [ss["Te"]], "o", color=COL["r"], ms=8, label="punkt pracy")
        a.set_ylim(min(-1, np.min(Te_n) * 0.0), max(ss["Te"] * 3, 5))
        a.set_xlim(nn[0], nn[-1])
        a.set_xlabel("n [obr/min]")
        a.set_ylabel("moment [Nm]")
        a.set_title("Charakterystyka mechaniczna (prosta – brak rozproszeń)")
        a.legend()
        # bilans mocy
        a = ax[1, 0]
        names = ["P pobrana", "straty Cu stojana", "straty Cu wirnika", "P mechaniczna", "Q bierna/10"]
        vals = [ss["P"], ss["Pcu_s"], ss["Pcu_r"], ss["Pmech"], ss["Q"] / 10]
        bars = a.bar(names, vals, color=[COL["b"], COL["r"], COL["o"], COL["g"], COL["p"]])
        for b_, v_ in zip(bars, vals):
            a.text(b_.get_x() + b_.get_width() / 2, v_, f"{v_:.0f}", ha="center", va="bottom")
        a.set_ylabel("[W]  (Q w var/10)")
        a.set_title("Bilans mocy w stanie ustalonym")
        a.tick_params(axis="x", labelrotation=15)
        # wykres wektorowy
        a = ax[1, 1]
        psi = self.v("psi")
        im = psi / self.v("Lm")
        ir = ss["i_r"]
        isv = complex(im, ir)
        us = self.v("Rs") * isv + 1j * ws * psi
        arrow(a, complex(psi * 4, 0), COL["k"], "ψm ×4 [Vs]")
        arrow(a, complex(im, 0), COL["b"], "i_m (magnesujący) [A]")
        arrow(a, complex(0, ir), COL["o"], "−i_r (z wirnika) [A]", origin=complex(im, 0))
        arrow(a, isv, COL["r"], "i_s = i_m − i_r [A]")
        arrow(a, us / 100, COL["g"], "u_s /100 [V]")
        a.set_aspect("equal", adjustable="datalim")
        a.set_title("Wykres wektorowy w układzie wirującym ze strumieniem")
        a.set_xlabel("oś d (wzdłuż ψm)")
        a.set_ylabel("oś q")
        a.legend(loc="lower right")
        a.autoscale()
        self.set_result(
            f"Stan ustalony (analitycznie):\n"
            f"  prędkość n        = {ss['n']:.1f} obr/min\n"
            f"  poślizg s         = {ss['slip'] * 100:.3f} %\n"
            f"  ω poślizgu        = {ss['wsl']:.3f} rad/s\n"
            f"  moment Te         = {ss['Te']:.2f} Nm\n"
            f"  sztywność kT      = {ss['kT']:.3f} Nm/(rad/s)\n"
            f"  prąd |is|         = {ss['i_s']:.3f} A (ampl.)\n"
            f"     magnesujący    = {ss['i_m']:.3f} A, wirnika = {ss['i_r']:.3f} A\n"
            f"  napięcie |us|     = {ss['u_s']:.1f} V (ampl.)\n"
            f"  P = {ss['P']:.1f} W,  Q = {ss['Q']:.1f} var\n"
            f"  cos φ             = {ss['cosphi']:.3f}\n"
            f"  sprawność η       = {ss['eta'] * 100:.1f} %\n"
            f"Symulacja: n(t_końc) = {n[-1]:.1f} obr/min\n")


class UniversalTab(Tab):
    KEY = "t82"
    TITLE = "8.6.2 / 8.6.3 Tutoriale 2–3 – uniwersalny model IRTF z parametrem a"

    def build(self):
        amin, amax = M.a_limits(M.SMALL)
        self.amin, self.amax = amin, amax
        self.section("Współczynnik transformacji a")
        self.slider("a", f"a  ({amin:.4f} … {amax:.4f})", amin, amax, 1.0, 0.0001)
        self.note("a = Lm/Lr → model 'na strumień wirnika', a = 1 → klasyczny, a = Ls/Lm → "
                  "'na strumień stojana'.")
        self.section("Zasilanie U/f (û = ωs·ψ*)")
        self.slider("psi", "ψ* generatora U/f", 0.5, 1.5, 1.25, 0.01, "Vs")
        self.slider("f", "częstotliwość f", 10, 60, 50, 0.5, "Hz")
        self.section("Obciążenie")
        self.combo("lt", "rodzaj obciążenia", ["stały moment (skok)", "kwadratowy (wentylator)"],
                   "stały moment (skok)")
        self.slider("TL", "moment obciążenia", 0, 15, 8.0, 0.1, "Nm")
        self.note("Książka (Fig. 8.42) używa 1 Nm; domyślne 8 Nm lepiej rozsuwa wektory na wykresie.")
        self.slider("tL", "chwila załączenia obciążenia", 0, 2, 0.5, 0.05, "s")
        self.slider("J", "J", 0.001, 0.05, 0.001, 0.0005, "kg·m²")
        self.slider("Tend", "czas symulacji", 0.3, 3, 1.0, 0.1, "s")

    def compute(self):
        par = dict(M.SMALL)
        par["J"] = self.v("J")
        TL, tL = self.v("TL"), self.v("tL")
        quad = self.v("lt").startswith("kw")
        kq = TL / (1500 / RPM) ** 2

        def load(t, W):
            if t < tL:
                return 0.0
            return kq * W * abs(W) if quad else TL
        f = self.v("f")
        u = 2 * PI * f * self.v("psi")
        res = {}
        for a in (self.amin, self.v("a"), self.amax):
            res[a] = M.sim_voltage_im(par, a, u, f, self.v("Tend"), 1e-4, load)
        r = res[self.v("a")]
        a_sel = self.v("a")
        ax = self.axes(2, 2)
        x = ax[0, 0]
        x.plot(r["t"], r["Te"], color=COL["b"], label=f"Te (a={a_sel:.4f})")
        x.plot(r["t"], r["Tl"], color=COL["o"], label="Tl")
        for aa, ls in ((self.amin, ":"), (self.amax, "--")):
            x.plot(res[aa]["t"], res[aa]["Te"], color=COL["k"], ls=ls, lw=0.8, label=f"Te (a={aa:.4f})")
        x.set_ylabel("moment [Nm]")
        x.set_xlabel("t [s]")
        x.set_title("Moment i prędkość – NIE zależą od a (linie się pokrywają)")
        x2 = twin(x, COL["g"])
        x2.plot(r["t"], r["n"], color=COL["g"])
        x2.set_ylabel("n [obr/min]")
        x.legend(loc="lower right")
        x = ax[0, 1]
        x.plot(r["t"], np.abs(r["psis"]), color=COL["r"], label="|ψs| stojana")
        x.plot(r["t"], np.abs(r["psiM"]), color=COL["p"], label="|ψM| (zależy od a)")
        x.plot(r["t"], np.abs(r["psiR"]), color=COL["b"], label="|ψR| (zależy od a)")
        x.set_xlabel("t [s]")
        x.set_ylabel("strumień [Vs]")
        x.set_title("Moduły strumieni w modelu uniwersalnym")
        x.legend()
        # wykres wektorowy w ukladzie psi_m
        x = ax[1, 0]
        ps, pM, pR, isr = r["psis"][-1], r["psiM"][-1], r["psiR"][-1], r["isr"][-1]
        upar = M.universal(par, a_sel)
        psir_phys = pR / a_sel
        aa = np.linspace(self.amin, self.amax, 50)
        locM = [ps - M.universal(par, q)["LsS"] * isr for q in aa]
        locR = [q * psir_phys for q in aa]
        x.plot([z.real for z in locM], [z.imag for z in locM], color=COL["p"], lw=4, alpha=0.25,
               label="miejsce końca ψM gdy a zmienia się")
        x.plot([z.real for z in locR], [z.imag for z in locR], color=COL["b"], lw=4, alpha=0.25,
               label="miejsce końca ψR gdy a zmienia się")
        arrow(x, ps, COL["r"], "ψs")
        arrow(x, pM, COL["p"], "ψM")
        arrow(x, pR, COL["b"], "ψR")
        arrow(x, -upar["LsS"] * isr, COL["gr"], "−LσS·is", origin=ps, lw=1.2)
        arrow(x, isr * 0.05, COL["o"], "is × 0,05 [A]", lw=1.2)
        tips = np.array([ps, pM, pR] + locM + locR)
        cx, cy = tips.real.mean(), tips.imag.mean()
        R = max(np.ptp(tips.real), np.ptp(tips.imag)) * 0.75 + 0.01
        x.set_xlim(cx - R, cx + R)
        x.set_ylim(cy - R, cy + R)
        x.set_aspect("equal", adjustable="box")
        x.set_title("Końce wektorów strumienia – POWIĘKSZENIE (układ ψm)")
        x.set_xlabel("d [Vs]")
        x.set_ylabel("q [Vs]")
        x.legend(loc="upper left", fontsize=6.5, framealpha=0.6)
        x = ax[1, 1]
        m = r["t"] > r["t"][-1] - 3 / f
        x.plot(r["t"][m], r["isa"][m], color=COL["r"], label="i_a")
        x.plot(r["t"][m], r["isb"][m], color=COL["g"], label="i_b")
        x.plot(r["t"][m], r["isc"][m], color=COL["b"], label="i_c")
        x.set_xlabel("t [s]")
        x.set_ylabel("prąd fazowy [A]")
        x.set_title("Prądy fazowe – ostatnie 3 okresy")
        x.legend()
        txt = "Porównanie dla trzech wartości a (dowód niezmienniczości):\n"
        for aa_, rr in res.items():
            txt += (f"  a={aa_:.4f}: Te={rr['Te'][-1]:8.4f} Nm  n={rr['n'][-1]:9.3f} obr/min  "
                    f"|is|={rr['is_abs'][-1]:.4f} A\n")
        txt += (f"\nParametry modelu dla a = {a_sel:.4f}:\n"
                f"  LσS = {upar['LsS'] * 1e3:.3f} mH,  LM = {upar['LM'] * 1e3:.2f} mH\n"
                f"  LσR = {upar['LsR'] * 1e3:.3f} mH,  RR = {upar['RR']:.4f} Ω\n"
                f"Strumienie na końcu: |ψs|={abs(ps):.4f}  |ψM|={abs(pM):.4f}  |ψR|={abs(pR):.4f} Vs\n"
                f"  |ψm| (szczelina, a=1) = {r['psim'][-1]:.4f} Vs")
        self.set_result(txt)


class T84Tab(Tab):
    KEY = "t84"
    TITLE = "8.6.4 Tutorial 4 – model prądowy bez rozproszeń (zasada sterowania polowego)"
    HAS_LTI = True

    def build(self):
        self.section("Prądy zadane (wymuszenie)")
        self.slider("isd", "prąd osi d  i_d", 0.5, 8, 4.0, 0.1, "A")
        self.slider("isq", "skok prądu osi q  i_q", 0, 8, 4.0, 0.1, "A")
        self.slider("tq", "chwila skoku i_q", 0.1, 1.8, 0.8, 0.05, "s")
        self.section("Maszyna i obciążenie")
        self.slider("Rr", "Rr", 0.5, 8, 3.0, 0.1, "Ω")
        self.slider("Lm", "Lm", 0.1, 0.6, 0.3409, 0.0001, "H")
        self.slider("J", "J", 0.01, 0.5, 0.1, 0.01, "kg·m²")
        self.slider("TL", "moment obciążenia", 0, 10, 0, 0.1, "Nm")
        self.slider("Tend", "czas symulacji", 0.5, 5, 2.0, 0.1, "s")
        self.section("Analiza: regulator prędkości PI (rozszerzenie)")
        self.slider("Kp", "Kp  [A/(rad/s)]", 0.001, 1.0, 0.05, 0.001)
        self.slider("Ki", "Ki  [A/(rad/s)/s]", 0.0, 5.0, 0.1, 0.005)

    def compute(self):
        t, r = M.t84_zero_leakage(self.v("isd"), self.v("isq"), self.v("tq"), self.v("Rr"),
                                  self.v("Lm"), 2, self.v("J"), self.v("TL"), self.v("Tend"))
        ax = self.axes(3, 1, sharex=True)
        a = ax[0, 0]
        a.plot(t, r[:, 2], color=COL["g"], label="ωm – prędkość wału (elektr.)")
        a.plot(t, r[:, 3], color=COL["r"], label="ωsl – poślizg")
        a.plot(t, r[:, 4], color=COL["b"], label="ωs = ωm + ωsl – stojan")
        a.set_ylabel("[rad/s]")
        a.set_title("Prędkości: synchroniczna, wirnika, poślizgu")
        a.legend()
        a = ax[1, 0]
        a.plot(t, r[:, 0], color=COL["p"], label="ψm")
        a.axhline(self.v("Lm") * self.v("isd"), color=COL["gr"], ls=":", label="Lm·i_d (ustalony)")
        tau = self.v("Lm") / self.v("Rr")
        a.axvline(tau, color=COL["o"], ls="--", lw=0.8, label=f"τ = Lm/Rr = {tau * 1e3:.0f} ms")
        a.set_ylabel("strumień [Vs]")
        a.set_title("Strumień narasta wykładniczo (człon inercyjny I rzędu)")
        a.legend()
        a = ax[2, 0]
        a.plot(t, r[:, 1], color=COL["b"], label="Te")
        a.plot(t, r[:, 5], color=COL["o"], ls="--", label="i_q [A]")
        a.set_ylabel("Te [Nm]")
        a.set_xlabel("t [s]")
        a.set_title("Moment – zmienia się skokowo razem z i_q")
        a.legend()
        psi = r[-1, 0]
        self.set_result(
            f"Stała czasowa strumienia τ = Lm/Rr = {tau * 1e3:.1f} ms\n"
            f"Strumień ustalony ψ = Lm·i_d = {self.v('Lm') * self.v('isd'):.4f} Vs\n"
            f"Moment po skoku  Te = 3/2·p·ψ·i_q = {1.5 * 2 * psi * self.v('isq'):.3f} Nm\n"
            f"Poślizg ωsl = Rr·i_q/ψ = {self.v('Rr') * self.v('isq') / psi:.3f} rad/s\n"
            f"Przyspieszenie dωm/dt = p·(Te−Tl)/J = "
            f"{2 * (1.5 * 2 * psi * self.v('isq') - self.v('TL')) / self.v('J'):.1f} rad/s²\n"
            f"Na końcu: ωm = {r[-1, 2]:.1f}, ωs = {r[-1, 4]:.1f} rad/s")

    def lti_systems(self):
        Lm, Rr, J, p = self.v("Lm"), self.v("Rr"), self.v("J"), 2
        tau = Lm / Rr
        psi0 = Lm * self.v("isd")
        G1 = M.TF([Lm], [tau, 1.0], name="Gψ")
        G2 = M.TF([1.5 * p * psi0 * p], [J, 0.0], name="Gω")
        C = M.TF([self.v("Kp"), self.v("Ki")], [1.0, 0.0], name="C")
        L = C * G2
        L.name = "L"
        return [dict(name="Gψ(s) = Δψm/Δi_d = Lm/(τs+1)  – tor strumienia", G=G1, key="t84_psi",
                     **{"in": "Δi_d [A]", "out": "Δψm [Vs]"}),
                dict(name="Gω(s) = Δωm/Δi_q = 3/2·p²·ψ/(J·s)  – tor momentu i prędkości", G=G2,
                     key="t84_w", t_end=1.0, **{"in": "Δi_q [A]", "out": "Δωm [rad/s]"}),
                dict(name="L(s) = (Kp + Ki/s)·Gω(s)  – pętla regulacji prędkości (rozszerzenie)", G=L,
                     kind="loop", key="t84_loop", **{"in": "ω* [rad/s]", "out": "ωm/ω*"})]


class T85Tab(Tab):
    KEY = "t85"
    TITLE = "8.6.5 Tutorial 5 – model UFO z rozproszeniami (orientacja: wirnik ↔ stojan)"
    HAS_LTI = True

    def build(self):
        amin, amax = M.a_limits(M.SMALL)
        self.section("Współczynnik transformacji a")
        self.slider("a", f"a  ({amin:.4f} … {amax:.4f})", amin, amax, amin, 0.0001)
        self.note("a = Lm/Lr (min) → strumień wirnika: pełne odsprzężenie;  "
                  "a = Ls/Lm (max) → strumień stojana: sprzężenia i 'szpilka' poślizgu.")
        self.section("Prądy zadane (rampy)")
        self.slider("isd", "i_d", 0.5, 8, 4.0, 0.1, "A")
        self.slider("isq", "i_q (skok/rampa)", 0, 8, 4.0, 0.1, "A")
        self.slider("tq", "chwila zmiany i_q", 0.1, 1.8, 0.8, 0.05, "s")
        self.slider("tr", "czas narastania rampy", 0.001, 0.1, 0.01, 0.001, "s")
        self.slider("J", "J", 0.01, 0.5, 0.1, 0.01, "kg·m²")
        self.slider("Tend", "czas symulacji", 1.0, 4.0, 2.0, 0.1, "s")

    def compute(self):
        r, up = M.t85_ufo(M.SMALL, self.v("a"), self.v("isd"), self.v("isq"), self.v("tq"),
                          self.v("tr"), self.v("J"), 0.0, self.v("Tend"))
        self.up = up
        ax = self.axes(2, 2)
        a = ax[0, 0]
        a.plot(r["t"], r["psiM"], color=COL["p"], lw=2, label="ψM (oś d modelu)")
        a.plot(r["t"], r["psis"], color=COL["r"], ls="--", label="|ψs|")
        a.plot(r["t"], r["psiR"], color=COL["b"], ls=":", label="|ψR|")
        a.set_title("Strumienie")
        a.set_ylabel("[Vs]")
        a.legend()
        a = ax[0, 1]
        a.plot(r["t"], r["Te"], color=COL["b"])
        a.set_title("Moment Te = 3/2·p·ψM·i_q")
        a.set_ylabel("[Nm]")
        a = ax[1, 0]
        a.plot(r["t"], r["wsl"], color=COL["r"])
        a.set_title("Pulsacja poślizgu ωsl (przy a = Ls/Lm widać impuls)")
        a.set_ylabel("[rad/s]")
        a.set_xlabel("t [s]")
        a = ax[1, 1]
        a.plot(r["t"], r["wm"], color=COL["g"], label="ωm")
        a.plot(r["t"], r["ws"], color=COL["b"], ls="--", label="ωs")
        a.set_title("Prędkości")
        a.set_ylabel("[rad/s]")
        a.set_xlabel("t [s]")
        a.legend()
        i0 = np.searchsorted(r["t"], self.v("tq") - 0.01)
        self.set_result(
            f"a = {self.v('a'):.4f}\n"
            f"  LM = {up['LM'] * 1e3:.2f} mH   LσS = {up['LsS'] * 1e3:.3f} mH\n"
            f"  LσR = {up['LsR'] * 1e3:.3f} mH  RR = {up['RR']:.4f} Ω\n"
            f"  stała czasowa strumienia L_R/R_R = {up['LR'] / up['RR'] * 1e3:.1f} ms\n"
            f"ψM przed skokiem i_q : {r['psiM'][i0]:.4f} Vs\n"
            f"ψM na końcu          : {r['psiM'][-1]:.4f} Vs  "
            f"(zmiana {100 * (r['psiM'][-1] / r['psiM'][i0] - 1):+.2f} %)\n"
            f"max ωsl              : {r['wsl'].max():.2f} rad/s\n"
            f"ωsl na końcu         : {r['wsl'][-1]:.3f} rad/s\n"
            f"Te na końcu          : {r['Te'][-1]:.3f} Nm")

    def lti_systems(self):
        up = M.universal(M.SMALL, self.v("a"))
        LM, LsR, RR, LR = up["LM"], up["LsR"], up["RR"], up["LR"]
        isd = self.v("isd")
        psi0 = LM * isd
        G1 = M.TF([LM / LR * LsR, LM / LR * RR], [1.0, RR / LR], name="Gψ")
        den = psi0 * LR / LM - LsR * isd
        G2 = M.TF([LsR / den, RR / den], [self.v("tr"), 1.0], name="Gsl")
        G3 = M.TF([1.5 * 2 * psi0 * 2], [self.v("J"), 0.0], name="Gω")
        return [dict(name="Gψ(s) = ΔψM/Δi_d = (L_M/L_R)(LσR·s + R_R)/(s + R_R/L_R)", G=G1, key="t85_psi",
                     **{"in": "Δi_d [A]", "out": "ΔψM [Vs]"}),
                dict(name="Gsl(s) = Δωsl/Δi_q* = (LσR·s + R_R)/(ψM·L_R/L_M − LσR·i_d) · 1/(Tr·s+1)",
                     G=G2, key="t85_sl", **{"in": "Δi_q* [A]", "out": "Δωsl [rad/s]"}),
                dict(name="Gω(s) = Δωm/Δi_q = 3/2·p²·ψM/(J·s)", G=G3, key="t84_w", t_end=1.0,
                     **{"in": "Δi_q [A]", "out": "Δωm [rad/s]"})]


class HeylandUFOTab(Tab):
    KEY = "heyl"
    TITLE = "8.3.5 Wykres Heylanda w układzie synchronicznym (stałe ψM)"

    def build(self):
        amin, amax = M.a_limits(M.SMALL)
        self.amin, self.amax = amin, amax
        self.slider("a", f"a  ({amin:.4f} … {amax:.4f})", amin, amax, amax, 0.0001)
        self.slider("psi", "strumień ψM", 0.3, 1.5, 1.0, 0.01, "Vs")
        self.slider("wmax", "zakres poślizgu ±ωsl", 10, 2000, 400, 10, "rad/s")
        self.slider("wsl", "zaznaczony poślizg ωsl", -400, 400, 20, 1, "rad/s")

    def compute(self):
        psi = self.v("psi")
        ax = self.axes(1, 2)
        a = ax[0, 0]
        b = ax[0, 1]
        for aa, col, lw in ((self.amax, COL["r"], 1.0), (0.5 * (self.amin + self.amax), COL["o"], 1.0),
                            (self.amin, COL["b"], 1.0), (self.v("a"), COL["k"], 2.4)):
            w, isd, isq, Te, up, circ = M.heyland_ufo(M.SMALL, aa, psi, self.v("wmax"), 1601)
            lab = f"a = {aa:.4f}" + ("  (wybrane)" if col == COL["k"] else "")
            a.plot(isd, isq, color=col, lw=lw, label=lab)
            b.plot(w, Te, color=col, lw=lw, label=lab)
        w, isd, isq, Te, up, circ = M.heyland_ufo(M.SMALL, self.v("a"), psi, self.v("wmax"), 1601)
        ws_ = self.v("wsl")
        x = ws_ * up["LsR"] / up["RR"]
        q0 = ws_ * psi / (up["RR"] * (1 + x * x))
        d0 = psi / up["LM"] + x * q0
        a.plot([0, d0], [0, q0], color=COL["g"], lw=2)
        a.plot([d0], [q0], "o", color=COL["g"], ms=8, label=f"is przy ωsl = {ws_:.0f} rad/s")
        b.plot([ws_], [1.5 * 2 * psi * q0], "o", color=COL["g"], ms=8)
        if circ:
            a.plot([circ["cx"]], [0], "+", color=COL["k"], ms=10)
        a.set_aspect("equal", adjustable="datalim")
        a.set_xlabel("i_sd [A]")
        a.set_ylabel("i_sq [A]")
        a.set_title("Miejsce geometryczne prądu stojana (okrąg Heylanda)")
        a.legend(loc="upper right")
        b.set_xlabel("ωsl [rad/s]")
        b.set_ylabel("Te [Nm]")
        b.set_title("Moment w funkcji poślizgu przy stałym ψM")
        b.legend()
        txt = f"a = {self.v('a'):.4f}:  LσR = {up['LsR'] * 1e3:.3f} mH, LM = {up['LM'] * 1e3:.1f} mH, RR = {up['RR']:.3f} Ω\n"
        if circ:
            txt += (f"Środek okręgu i_d = ψM/2·(1/LσR + 2/LM) = {circ['cx']:.2f} A\n"
                    f"Promień       r  = ψM/(2·LσR)        = {circ['r']:.2f} A\n"
                    f"Moment maks. Te_max = 3/2·p·ψM·r     = {circ['Tmax']:.1f} Nm\n"
                    f"   przy ωsl = RR/LσR                 = {circ['wsl_Tmax']:.1f} rad/s\n")
        else:
            txt += "LσR = 0 → promień nieskończony: okrąg staje się prostą i_d = ψM/LM.\n"
        txt += (f"\nZaznaczony punkt: i_d = {d0:.3f} A, i_q = {q0:.3f} A, |is| = {math.hypot(d0, q0):.3f} A, "
                f"Te = {1.5 * 2 * psi * q0:.2f} Nm")
        self.set_result(txt)


class SteadyTab(Tab):
    KEY = "steady"
    TITLE = "8.3.6 + Tutorial 8.6.8 – stan ustalony maszyny 17,3 kW zasilanej z sieci"

    def build(self):
        self.slider("U", "napięcie przewodowe U (trójkąt)", 100, 480, 415, 1, "V")
        self.slider("f", "częstotliwość f", 5, 60, 50, 0.5, "Hz")
        self.slider("kRr", "mnożnik rezystancji wirnika", 0.5, 3.0, 1.0, 0.05)
        self.slider("n", "zaznaczona prędkość n", 0, 3000, 1465, 1, "obr/min")
        self.check("norm", "wykres Heylanda znormalizowany (jak Fig. 8.32)", True)

    def compute(self):
        par = dict(M.M17)
        par["Rr"] *= self.v("kRr")
        f = self.v("f")
        u = math.sqrt(2) * self.v("U")
        ns = 60 * f / par["p"]
        n = np.linspace(-ns, 2 * ns, 1801)
        sv = M.steady_voltage(par, u, f, n)
        mk = M.steady_voltage(par, u, f, [self.v("n")])
        ax = self.axes(2, 2)
        a = ax[0, 0]
        motor = (sv["s"] > 0) & (sv["s"] <= 1)
        gen = sv["s"] < 0
        brake = sv["s"] > 1
        z = sv["i_n"] if self.v("norm") else sv["i_s"]
        zm = mk["i_n"][0] if self.v("norm") else mk["i_s"][0]
        for msk, col, lab in ((motor, COL["b"], "silnik 0<s≤1"), (gen, COL["g"], "prądnica s<0"),
                              (brake, COL["r"], "hamowanie s>1")):
            a.plot(-z.imag[msk], z.real[msk], ".", ms=2, color=col, label=lab)
        a.plot([0, -zm.imag], [0, zm.real], color=COL["k"], lw=2)
        a.plot([-zm.imag], [zm.real], "o", color=COL["k"])
        a.set_aspect("equal", adjustable="datalim")
        a.set_xlabel("−Im(is) – składowa bierna (magnesująca)" + (" [–]" if self.v("norm") else " [A]"))
        a.set_ylabel("Re(is) – składowa czynna" + (" [–]" if self.v("norm") else " [A]"))
        a.set_title("Wykres Heylanda (okrąg prądu stojana)")
        a.legend(loc="lower left", markerscale=4)
        a = ax[0, 1]
        a.plot(n, sv["Te"], color=COL["b"])
        a.axhline(113, color=COL["gr"], ls=":", label="Tn = 113 Nm")
        a.axvline(ns, color=COL["gr"], ls="--", lw=0.8)
        i_max = np.argmax(sv["Te"] * motor)
        a.plot([n[i_max]], [sv["Te"][i_max]], "v", color=COL["r"], label=f"moment krytyczny {sv['Te'][i_max]:.0f} Nm")
        a.plot([self.v("n")], [mk["Te"][0]], "o", color=COL["k"])
        a.set_xlim(0, 2 * ns)
        a.set_ylim(np.percentile(sv["Te"], 1) * 1.1, np.max(sv["Te"]) * 1.1)
        a.set_xlabel("n [obr/min]")
        a.set_ylabel("Te [Nm]")
        a.set_title("Charakterystyka mechaniczna Te(n)")
        a.legend()
        a = ax[1, 0]
        a.plot(n, sv["I_line"], color=COL["r"], label="prąd przewodowy I_l (RMS)")
        a.set_xlim(0, 2 * ns)
        a.set_xlabel("n [obr/min]")
        a.set_ylabel("[A]")
        a.set_title("Prąd i strumień stojana")
        a.plot([self.v("n")], [mk["I_line"][0]], "o", color=COL["k"])
        a2 = twin(a, COL["p"])
        a2.plot(n, sv["psi_s"], color=COL["p"])
        a2.set_ylabel("|ψs| [Vs]")
        a.legend(loc="upper right")
        a = ax[1, 1]
        a.plot(n, sv["Pout"] / 1e3, color=COL["g"], label="P mech [kW]")
        a.plot(n, sv["Pin"] / 1e3, color=COL["b"], ls="--", label="P elektr. [kW]")
        a.set_xlim(0, 2 * ns)
        a.set_ylim(-60, 60)
        a.set_xlabel("n [obr/min]")
        a.set_ylabel("[kW]")
        a.set_title("Moc i sprawność")
        a2 = twin(a, COL["o"])
        eta = np.where(sv["Pout"] > 0.02 * 17332, sv["eta"], np.nan)
        a2.plot(n, eta * 100, color=COL["o"])
        a2.set_ylim(0, 100)
        a2.set_ylabel("η [%]")
        a.legend(loc="lower left")
        s0 = M.steady_voltage(par, u, f, [0.0])
        self.set_result(
            f"Zaznaczony punkt n = {self.v('n'):.0f} obr/min (poślizg {mk['s'][0] * 100:.2f} %):\n"
            f"  Te = {mk['Te'][0]:.1f} Nm, I_l = {mk['I_line'][0]:.2f} A, |ψs| = {mk['psi_s'][0]:.3f} Vs\n"
            f"  P_mech = {mk['Pout'][0] / 1e3:.2f} kW, P_el = {mk['Pin'][0] / 1e3:.2f} kW, "
            f"η = {np.nan_to_num(mk['eta'][0]) * 100:.1f} %, cos φ = {mk['pf'][0]:.3f}\n\n"
            f"Rozruch (n = 0): Te = {s0['Te'][0]:.1f} Nm, I_l = {s0['I_line'][0]:.1f} A\n"
            f"Moment krytyczny {sv['Te'][i_max]:.1f} Nm przy n = {n[i_max]:.0f} obr/min\n"
            f"Prędkość synchroniczna {ns:.0f} obr/min\n"
            f"Model 'rotor flux': LσS = {sv['up']['LsS'] * 1e3:.1f} mH, LM = {sv['up']['LM'] * 1e3:.1f} mH, "
            f"RR = {sv['up']['RR']:.3f} Ω")


class EstimTab(Tab):
    KEY = "estim"
    TITLE = "8.4 + Tutorial 8.6.6 – estymacja parametrów z tabliczki znamionowej"

    def build(self):
        self.section("Tabliczka znamionowa (Tab. 8.1)")
        self.slider("U", "napięcie przewodowe U_l", 200, 690, 415, 1, "V")
        self.slider("I", "prąd przewodowy I_l", 5, 100, 30, 0.1, "A")
        self.slider("P", "moc znamionowa P", 1000, 60000, 17332, 1, "W")
        self.slider("n", "prędkość znamionowa n", 600, 2990, 1465, 1, "obr/min")
        self.slider("pf", "współczynnik mocy cos φ", 0.5, 0.95, 0.80, 0.01)
        self.slider("f", "częstotliwość f", 50, 60, 50, 10, "Hz")
        self.slider("p", "liczba par biegunów p", 1, 4, 2, 1)
        self.section("Pomiary")
        self.slider("I0", "prąd biegu jałowego I_0", 1, 40, 8.41, 0.01, "A")
        self.slider("Rl", "rezystancja między zaciskami R_l", 0.05, 3, 0.40, 0.01, "Ω")

    def compute(self):
        st, par = M.estimate_parameters(self.v("U"), self.v("I"), self.v("I0"), self.v("Rl"),
                                        self.v("P"), self.v("n"), self.v("f"), self.v("pf"),
                                        int(self.v("p")))
        ax = self.axes(1, 2)
        a = ax[0, 0]
        a.set_axis_off()
        a.set_xlim(0, 10)
        a.set_ylim(0, 6)
        a.set_title("Schemat zastępczy (a = 1, Tab. 8.2) i model 'rotor flux' (Tab. 8.3)")

        def box(x, y, w, h, txt, col):
            a.add_patch(matplotlib.patches.Rectangle((x, y), w, h, fc="white", ec=col, lw=1.6))
            a.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=7.5)

        def schem(y0, items, title):
            a.text(0.1, y0 + 2.15, title, fontsize=8.5, weight="bold")
            a.plot([0.3, 9.7], [y0 + 1.6, y0 + 1.6], color=COL["k"], lw=1)
            a.plot([0.3, 9.7], [y0, y0], color=COL["k"], lw=1)
            box(0.9, y0 + 1.35, 1.6, 0.5, items[0], COL["r"])
            box(3.0, y0 + 1.35, 1.6, 0.5, items[1], COL["b"])
            box(4.6, y0 + 0.5, 1.6, 0.6, items[2], COL["p"])
            a.plot([5.4, 5.4], [y0 + 1.1, y0 + 1.6], color=COL["k"])
            a.plot([5.4, 5.4], [y0, y0 + 0.5], color=COL["k"])
            if items[3]:
                box(6.4, y0 + 1.35, 1.6, 0.5, items[3], COL["b"])
            box(8.0, y0 + 0.5, 1.6, 0.6, items[4], COL["o"])
            a.plot([8.8, 8.8], [y0 + 1.1, y0 + 1.6], color=COL["k"])
            a.plot([8.8, 8.8], [y0, y0 + 0.5], color=COL["k"])
            a.plot([0.3], [y0 + 1.6], "o", color=COL["k"])
            a.plot([0.3], [y0], "o", color=COL["k"])
        Lss = par["Lss"]
        schem(3.3, [f"Rs={par['Rs']:.3f} Ω", f"Lσs={Lss * 1e3:.1f} mH", f"Lm={par['Lm'] * 1e3:.0f} mH",
                    f"Lσr={Lss * 1e3:.1f} mH", f"Rr/s\nRr={par['Rr']:.3f} Ω"], "a = 1 (klasyczny schemat T)")
        up = M.universal(dict(Rs=par["Rs"], Rr=par["Rr"], Lm=par["Lm"], Ls=par["Ls"], Lr=par["Lr"], p=2),
                         par["Lm"] / par["Lr"])
        schem(0.4, [f"Rs={par['Rs']:.3f} Ω", f"LσS={up['LsS'] * 1e3:.1f} mH", f"LM={up['LM'] * 1e3:.1f} mH",
                    "", f"RR/s\nRR={up['RR']:.3f} Ω"], f"a = Lm/Lr = {up['a']:.4f} (model 'rotor flux')")
        b = ax[0, 1]
        names = ["Ls [mH]", "Lm [mH]", "Lσs [mH]", "Rr [mΩ]", "Rs [mΩ]", "ψs [Vs·100]", "ψR [Vs·100]"]
        mine = [par["Ls"] * 1e3, par["Lm"] * 1e3, Lss * 1e3, par["Rr"] * 1e3, par["Rs"] * 1e3,
                par["psi_s"] * 100, par["psiR"] * 100]
        book = [272, 260, 12, 540, 600, 183, 167]
        xx = np.arange(len(names))
        b.bar(xx - 0.2, mine, 0.4, color=COL["b"], label="obliczone (suwaki)")
        b.bar(xx + 0.2, book, 0.4, color=COL["o"], label="książka (Tab. 8.2/8.3)")
        b.set_xticks(xx, names, rotation=25)
        b.set_title("Porównanie z tabelami w książce")
        b.legend()
        lines = ["Obliczenia krok po kroku (Listing 8.1):"]
        for desc, sym, val, unit in st:
            if desc.startswith("---"):
                lines.append(desc)
                continue
            lines.append(f"{sym:>13s} = {val:10.5g} {unit:5s} | {desc}")
        lines.append("\nUwaga: w Tab. 8.3 ψR = 1,67 Vs pochodzi z linii 46 listingu, która używa "
                     "nietypowego wzoru; fizycznie ψR = |ψs − LσS·is| ≈ 1,59 Vs.")
        self.set_result("\n".join(lines))


class GridTab(Tab):
    KEY = "grid"
    TITLE = "8.6.7 Tutorial 7 – maszyna 17,3 kW załączona bezpośrednio do sieci"
    HAS_LTI = True

    def build(self):
        self.slider("U", "napięcie przewodowe U (trójkąt)", 200, 480, 415, 1, "V")
        self.slider("TL", "skok momentu obciążenia", 0, 200, 113, 1, "Nm")
        self.slider("tL", "chwila skoku", 0.2, 2.5, 1.0, 0.05, "s")
        self.slider("J", "J", 0.02, 1.0, 0.1, 0.01, "kg·m²")
        amin, amax = M.a_limits(M.M17)
        self.slider("a", "a (nie zmienia wyników)", amin, amax, amin, 0.0001)
        self.slider("Tend", "czas symulacji", 0.5, 3.0, 2.0, 0.1, "s")

    def compute(self):
        par = dict(M.M17)
        par["J"] = self.v("J")
        u = math.sqrt(2) * self.v("U")
        TL, tL = self.v("TL"), self.v("tL")
        r = M.sim_voltage_im(par, self.v("a"), u, 50.0, self.v("Tend"), 1e-4,
                             lambda t, W: TL if t >= tL else 0.0, n_out=4000)
        self.r = r
        ax = self.axes(2, 2)
        a = ax[0, 0]
        a.plot(r["t"], r["Te"], color=COL["b"], label="Te")
        a.plot(r["t"], r["Tl"], color=COL["o"], label="Tl")
        a.set_ylabel("moment [Nm]")
        a.set_title("Moment (oscylacje po załączeniu i po skoku obciążenia)")
        a2 = twin(a, COL["g"])
        a2.plot(r["t"], r["n"], color=COL["g"])
        a2.set_ylabel("n [obr/min]")
        a.legend(loc="lower right")
        a = ax[0, 1]
        nn = np.linspace(0, 1500, 600)
        sv = M.steady_voltage(par, u, 50.0, nn)
        a.plot(nn, sv["Te"], color=COL["gr"], lw=2.5, alpha=0.6, label="charakterystyka statyczna")
        a.plot(r["n"], r["Te"], color=COL["b"], lw=0.8, label="trajektoria dynamiczna")
        a.set_xlabel("n [obr/min]")
        a.set_ylabel("Te [Nm]")
        a.set_title("Dynamika ≠ statyka: moment krąży wokół charakterystyki")
        a.legend()
        a = ax[1, 0]
        a.plot(r["t"], r["isa"], color=COL["r"], lw=0.6)
        a.set_xlabel("t [s]")
        a.set_ylabel("i_a fazy [A]")
        a.set_title("Prąd fazowy – prąd rozruchowy kilka razy większy od znamionowego")
        a = ax[1, 1]
        a.plot(r["t"], np.abs(r["psis"]), color=COL["p"], label="|ψs|")
        a.plot(r["t"], np.abs(r["psiR"]), color=COL["b"], label="|ψR|")
        a.set_xlabel("t [s]")
        a.set_ylabel("[Vs]")
        a.set_title("Strumienie")
        a.legend()
        m = r["t"] > r["t"][-1] - 0.1
        self.set_result(
            f"Na końcu symulacji: n = {r['n'][-1]:.1f} obr/min, Te = {r['Te'][-1]:.1f} Nm\n"
            f"Szczyt prądu rozruchowego: {np.max(np.abs(r['isa'])):.0f} A (ampl. fazowa)\n"
            f"   prąd znamionowy fazy (ampl.) = {math.sqrt(2) * 30 / math.sqrt(3):.1f} A\n"
            f"Maks. moment w czasie rozruchu: {np.max(r['Te']):.0f} Nm, min: {np.min(r['Te']):.0f} Nm\n"
            f"|ψs| ustalony = {np.mean(np.abs(r['psis'][m])):.3f} Vs\n"
            "Zakładka 'Analiza LTI' – linearyzacja w punkcie pracy po skoku obciążenia.")

    def lti_systems(self):
        par = dict(M.M17)
        par["J"] = self.v("J")
        u = math.sqrt(2) * self.v("U")
        ws = 2 * PI * 50
        psi = u / math.hypot(ws, par["Rs"] / par["Ls"])
        d = M.vf_linear(par, 1500.0, psi_ref=psi, T_ext=self.v("TL"))
        info = (f"Punkt pracy: n0 = {d['n0']:.1f} obr/min, Te0 = {d['Te0']:.1f} Nm.\n"
                "Wartości własne zlinearyzowanego modelu: "
                + ", ".join(M.fmt(complex(e)) for e in d["eig"]))
        return [dict(name="G_l(s) = Δn/ΔT_load – reakcja prędkości na skok obciążenia", G=d["G_l"],
                     key="grid_l", info=info, **{"in": "ΔT_load [Nm]", "out": "Δn [obr/min]"}),
                dict(name="L(s) = −G_Te,Ω(s)/(J s) – pętla elektromechaniczna (sztywność 'sprężyny' pola)",
                     G=d["L"], kind="loop", key="grid_L", info=info, **{"in": "-", "out": "-"})]


class Single1Tab(Tab):
    KEY = "single1"
    TITLE = "8.5 + Tutorial 8.6.9 – rozruch silnika jednofazowego z kondensatorem pracy"

    def build(self):
        self.slider("U", "napięcie sieci U (RMS)", 60, 130, 110, 1, "V")
        self.slider("C", "kondensator C", 0, 80, 31, 0.5, "µF")
        self.slider("TL", "skok obciążenia", 0, 3, 2.0, 0.05, "Nm")
        self.slider("tL", "chwila skoku", 0.2, 2.5, 1.0, 0.05, "s")
        self.slider("J", "J", 0.002, 0.05, 0.0146, 0.0002, "kg·m²")
        self.slider("Tend", "czas symulacji", 0.5, 3, 2.0, 0.1, "s")
        self.check("vec", "wykres wektorowy ψR i is w układzie wirnika (Fig. 8.51)", True)

    def compute(self):
        P = dict(M.SINGLE)
        P["U"] = self.v("U")
        P["J"] = self.v("J")
        C = self.v("C") * 1e-6
        r = M.sim_single_phase(P, self.v("TL"), self.v("tL"), self.v("Tend"), 5e-5, 6000, C=C)
        ax = self.axes(2, 2)
        a = ax[0, 0]
        a.plot(r["t"], r["Te"], color=COL["b"], lw=0.6, label="Te (tętniący!)")
        a.plot(r["t"], r["Tl"], color=COL["o"], label="Tl")
        a.set_ylabel("moment [Nm]")
        a.set_title("Moment i prędkość")
        a2 = twin(a, COL["g"])
        a2.plot(r["t"], r["n"], color=COL["g"])
        a2.set_ylabel("n [obr/min]")
        a.legend(loc="lower right")
        a = ax[0, 1]
        m = r["t"] > r["t"][-1] - 0.06
        a.plot(r["t"][m], r["i_run"][m], color=COL["r"], label="i_run (główne)")
        a.plot(r["t"][m], r["i_aux"][m], color=COL["b"], label="i_aux (pomocnicze)")
        a.set_title("Prądy uzwojeń – ostatnie 60 ms")
        a.set_ylabel("[A]")
        a.legend()
        a = ax[1, 0]
        a.plot(r["t"][m], r["u_run"][m], color=COL["r"], label="u_run = u_sieci")
        a.plot(r["t"][m], r["u_aux"][m], color=COL["b"], label="u_aux")
        a.plot(r["t"][m], r["u_c"][m], color=COL["gr"], ls="--", label="u_C")
        a.set_title("Napięcia – kondensator przesuwa fazę")
        a.set_xlabel("t [s]")
        a.set_ylabel("[V]")
        a.legend()
        a = ax[1, 1]
        if self.v("vec"):
            rot = np.exp(-1j * r["th"])
            pR = (r["pRa"] + 1j * r["pRb"]) * rot * 50
            isx = (r["isa"] + 1j * r["isb"]) * rot
            a.plot(pR.real, pR.imag, color=COL["g"], lw=0.5, label="ψR^xy ·50")
            a.plot(isx.real, isx.imag, color=COL["r"], lw=0.5, label="is^xy")
            a.set_aspect("equal", adjustable="datalim")
            a.set_title("Wektory w układzie wirnika (Fig. 8.51)")
            a.legend()
        else:
            a.plot(r["isa"] , r["isb"], color=COL["r"], lw=0.4)
            a.set_aspect("equal", adjustable="datalim")
            a.set_title("Hodograf prądu stojana (elipsa = pole eliptyczne)")
        mm = r["t"] > r["t"][-1] - 0.2
        self.set_result(
            f"C = {self.v('C'):.1f} µF\n"
            f"Prędkość końcowa  n = {r['n'][-1]:.1f} obr/min\n"
            f"Moment średni     = {np.mean(r['Te'][mm]):.3f} Nm\n"
            f"Tętnienia (ampl.) = {(np.max(r['Te'][mm]) - np.min(r['Te'][mm])) / 2:.3f} Nm  "
            f"(częstotliwość 2f = 100 Hz)\n"
            f"Prąd główny (ampl.)    = {np.max(np.abs(r['i_run'][mm])):.3f} A\n"
            f"Prąd pomocniczy (ampl.)= {np.max(np.abs(r['i_aux'][mm])):.3f} A\n"
            f"Napięcie na C (ampl.)  = {np.max(np.abs(r['u_c'][mm])):.1f} V\n"
            f"Szczyt prądu rozruchu  = {np.max(np.abs(r['i_run'])):.2f} A")


class Single2Tab(Tab):
    KEY = "single2"
    TITLE = "8.5.1 – stan ustalony silnika jednofazowego (składowe zgodna i przeciwna)"

    def build(self):
        self.slider("U", "napięcie U (RMS)", 60, 130, 110, 1, "V")
        self.slider("C", "kondensator C", 0, 80, 31, 0.5, "µF")
        self.slider("k", "przekładnia k_aux", 0.6, 2.0, 1.18, 0.01)
        self.slider("Ra", "rezystancja R_aux", 1, 20, 7.14, 0.01, "Ω")
        self.slider("n", "zaznaczona prędkość", 0, 1499, 1400, 1, "obr/min")

    def compute(self):
        P = dict(M.SINGLE)
        P.update(U=self.v("U"), kaux=self.v("k"), Raux=self.v("Ra"))
        C = self.v("C") * 1e-6
        ns = 60 * P["f"] / P["p"]
        n = np.linspace(-ns * 0.999, ns * 0.999, 301)
        r = M.steady_single_phase(P, n, C=C)
        ax = self.axes(2, 2)
        a = ax[0, 0]
        a.plot(n / ns, r["Tav"], color=COL["b"], label="moment średni")
        a.plot(n / ns, r["Trip"], color=COL["r"], label="amplituda tętnień (2f)")
        a.axvline(self.v("n") / ns, color=COL["gr"], ls=":")
        a.set_xlabel("n / n_s  (s = 2 … 0)")
        a.set_ylabel("[Nm]")
        a.set_title("Moment średni i tętnienia (Fig. 8.38)")
        a.legend()
        a = ax[0, 1]
        a.plot(n / ns, r["Irun"], color=COL["r"], label="Î_run")
        a.plot(n / ns, r["Iaux"], color=COL["b"], label="Î_aux")
        a.set_xlabel("n / n_s")
        a.set_ylabel("[A]")
        a.set_title("Amplitudy prądów uzwojeń")
        a.legend()
        # skan pojemnosci
        a = ax[1, 0]
        Cs = np.linspace(1, 80, 80)
        trip = []
        tav = []
        for c in Cs:
            q = M.steady_single_phase(P, [self.v("n")], C=c * 1e-6)
            trip.append(q["Trip"][0])
            tav.append(q["Tav"][0])
        a.plot(Cs, trip, color=COL["r"], label="tętnienia")
        a.plot(Cs, tav, color=COL["b"], label="moment średni")
        ib = int(np.argmin(trip))
        a.plot([Cs[ib]], [trip[ib]], "o", color=COL["k"], label=f"min tętnień: C ≈ {Cs[ib]:.0f} µF")
        a.axvline(self.v("C"), color=COL["gr"], ls=":")
        a.set_xlabel("C [µF]")
        a.set_ylabel("[Nm]")
        a.set_title(f"Dobór kondensatora przy n = {self.v('n'):.0f} obr/min")
        a.legend()
        # wykres wskazowy
        a = ax[1, 1]
        q = M.steady_single_phase(P, [self.v("n")], C=C)
        wm = self.v("n") / RPM * P["p"]
        w = 2 * PI * P["f"]
        U = math.sqrt(2) * P["U"]
        # ponowne wyliczenie fazorow (prosto - jak w modelu)
        s = (w - wm) / w

        def Zseq(sl):
            ZM = 1j * w * P["LM"]
            ZR = P["RR"] / sl
            return 1j * w * P["LsS"] + ZM * ZR / (ZM + ZR)
        Tinv = np.array([[1, 1], [-1j, 1j]])
        Zab = Tinv @ np.diag([Zseq(s), Zseq(2 - s)]) @ np.linalg.inv(Tinv)
        Mx = Zab.copy()
        Mx[0, 0] += P["Raux"] / P["kaux"] ** 2 + (1 / (1j * w * C) if C > 0 else 0) / P["kaux"] ** 2
        Mx[1, 1] += P["Rrun"]
        I = np.linalg.solve(Mx, np.array([U / P["kaux"], U]))
        Iaux = I[0] / P["kaux"]
        Irun = I[1]
        sc = max(abs(Iaux), abs(Irun))
        arrow(a, complex(1, 0) * sc * 1.2, COL["k"], "U sieci (kierunek)")
        arrow(a, Irun, COL["r"], "I_run")
        arrow(a, Iaux, COL["b"], "I_aux")
        a.set_aspect("equal", adjustable="datalim")
        ang = math.degrees(np.angle(Iaux) - np.angle(Irun))
        a.set_title(f"Wskazy prądów: przesunięcie {ang:.0f}° (ideał 90°)")
        a.legend(loc="lower left")
        self.set_result(
            f"Przy n = {self.v('n'):.0f} obr/min, C = {self.v('C'):.1f} µF:\n"
            f"  moment średni   = {q['Tav'][0]:.3f} Nm\n"
            f"  tętnienia 100 Hz= {q['Trip'][0]:.3f} Nm\n"
            f"  Î_run = {q['Irun'][0]:.3f} A,  Î_aux = {q['Iaux'][0]:.3f} A\n"
            f"  przesunięcie fazowe prądów = {ang:.1f}°\n"
            f"Moment rozruchowy (n=0): {M.steady_single_phase(P, [0.0], C=C)['Tav'][0]:.3f} Nm\n"
            f"Najmniejsze tętnienia przy C ≈ {Cs[ib]:.0f} µF")


# =============================================================================
# ROZDZIAL 9 - STEROWANIE
# =============================================================================

class KlossTab(Tab):
    KEY = "kloss"
    TITLE = "9.1 Idea sterowania V/f – wzór Klossa i punkty pracy"

    def build(self):
        self.slider("psi", "strumień stojana ψs (stały)", 0.5, 2.2, 1.83, 0.01, "Vs")
        self.section("Trzy częstotliwości zadane (jako prędkość synchroniczna)")
        self.slider("n1", "n_s1*", 50, 3000, 500, 10, "obr/min")
        self.slider("n2", "n_s2*", 50, 3000, 1000, 10, "obr/min")
        self.slider("n3", "n_s3*", 50, 3000, 1500, 10, "obr/min")
        self.section("Obciążenie wentylatorowe")
        self.slider("TL", "moment obciążenia przy 1465 obr/min", 0, 300, 113, 1, "Nm")

    def compute(self):
        par = M.M17
        ws_list = [self.v(k) / RPM * 2 for k in ("n1", "n2", "n3")]
        curves, That, wslh, up = M.kloss_curves(par, self.v("psi"), ws_list)
        kq = M.quad_load_k(self.v("TL"), 1465)
        ax = self.axes(1, 2)
        a = ax[0, 0]
        cols = [COL["b"], COL["g"], COL["r"]]
        ops = []
        for (ws, n, Te), c in zip(curves, cols):
            a.plot(n, Te, color=c, label=f"ωs = {ws:.0f} rad/s ({ws / 2 * RPM:.0f} obr/min)")
            Tl = kq * (n / RPM) ** 2
            dif = Te - Tl
            idx = np.nonzero((dif[:-1] > 0) & (dif[1:] <= 0) & (n[:-1] > 0))[0]
            if len(idx):
                i = idx[-1]
                ops.append((n[i], Te[i], ws))
                a.plot([n[i]], [Te[i]], "o", color=c, ms=8)
        nn = np.linspace(0, max(c[1].max() for c in curves), 300)
        a.plot(nn, kq * (nn / RPM) ** 2, color=COL["o"], lw=2, label="obciążenie T_l(n)")
        a.axhline(That, color=COL["gr"], ls=":", label=f"moment krytyczny T̂e = {That:.0f} Nm")
        a.set_ylim(-0.3 * That, 1.15 * That)
        a.set_xlabel("n [obr/min]")
        a.set_ylabel("Te [Nm]")
        a.set_title("Rodzina charakterystyk V/f (przesunięte, ten sam T̂e)")
        a.legend(loc="lower left")
        b = ax[0, 1]
        w = np.linspace(-6 * wslh, 6 * wslh, 600)
        Te = np.where(np.abs(w) > 1e-9, 2 * That / (w / wslh + wslh / np.where(w == 0, 1, w)), 0)
        b.plot(w, Te, color=COL["b"])
        b.axvspan(-wslh, wslh, color=COL["g"], alpha=0.08, label="praca stabilna |ωsl| < ω̂sl")
        b.axvline(wslh, color=COL["r"], ls="--", label=f"ω̂sl = RR/LσR = {wslh:.1f} rad/s")
        b.axvline(-wslh, color=COL["r"], ls="--")
        b.set_xlabel("ωsl [rad/s]")
        b.set_ylabel("Te [Nm]")
        b.set_title("Wzór Klossa: Te = 2T̂e /(ωsl/ω̂sl + ω̂sl/ωsl)")
        b.legend()
        txt = (f"Model 'stator flux' (a = Ls/Lm):\n  LσR = {up['LsR'] * 1e3:.2f} mH, RR = {up['RR']:.3f} Ω\n"
               f"Moment krytyczny  T̂e = 3/2·p·ψs²/(2·LσR) = {That:.1f} Nm\n"
               f"Poślizg krytyczny ω̂sl = RR/LσR = {wslh:.2f} rad/s "
               f"(= {wslh / 2 * RPM:.0f} obr/min)\n\nPunkty pracy (przecięcie z obciążeniem):\n")
        for n, T, ws in ops:
            txt += f"  ωs={ws:6.1f}: n = {n:7.1f} obr/min, T = {T:6.1f} Nm, poślizg {ws / 2 * RPM - n:5.1f} obr/min\n"
        self.set_result(txt)


class VF1Tab(Tab):
    KEY = "vf1"
    TITLE = "9.1.1 + Tutorial 9.7.1 – prosty napęd V/f (bez czujnika) i jego niestabilność"
    HAS_LTI = True

    def build(self):
        self.combo("mode", "scenariusz", ["Część 1: rozruch do n* (obciążenie wentylatorowe)",
                                          "Część 2: rampa 0→900 obr/min + zakłócenie (niestabilność)"],
                   "Część 1: rozruch do n* (obciążenie wentylatorowe)")
        self.slider("psi", "strumień zadany ψs*", 0.5, 2.2, 1.83, 0.01, "Vs")
        self.slider("n", "prędkość zadana n* (część 1)", 100, 1800, 1465, 5, "obr/min")
        self.slider("kRR", "mnożnik rezystancji wirnika", 0.5, 1.5, 0.9, 0.01)
        self.note("Książka: część 2 liczona z Rr o 10 % mniejszą (0,9), by lepiej pokazać rezonans.")
        self.section("Część 2")
        self.slider("nend", "koniec rampy", 200, 1500, 900, 10, "obr/min")
        self.slider("rate", "szybkość rampy", 5, 200, 30, 1, "obr/min/s")
        self.slider("da", "zakłócenie: amplituda prostokąta", 0, 10, 1.0, 0.1, "Nm")
        self.slider("df", "zakłócenie: częstotliwość", 0.2, 10, 1.0, 0.1, "Hz")
        self.section("Analiza LTI (punkt pracy)")
        self.slider("nop", "n*_s punktu linearyzacji", 20, 1500, 450, 5, "obr/min")
        self.check("lq", "w analizie LTI uwzględnij obciążenie wentylatorowe", False)

    def compute(self):
        part2 = self.v("mode").startswith("Część 2")
        par = M.M17
        if part2:
            Tend = self.v("nend") / self.v("rate") + 3.0
            r = M.sim_vf(par, "speed", self.v("psi"), n_ramp=(self.v("nend"), self.v("rate")),
                         kRR=self.v("kRR"), load_quad=False, dist_amp=self.v("da"), dist_f=self.v("df"),
                         T_end=Tend, dt=5e-4, n_out=6000)
        else:
            r = M.sim_vf(par, "speed", self.v("psi"), self.v("n"), kRR=1.0, load_quad=True,
                         T_end=2.0, dt=2e-4)
        ax = self.axes(2, 2)
        a = ax[0, 0]
        a.plot(r["t"], r["Te"], color=COL["b"], lw=0.7, label="Te")
        a.plot(r["t"], r["Tl"], color=COL["o"], label="Tl")
        a.set_ylabel("moment [Nm]")
        a.set_title("Moment i prędkość")
        a2 = twin(a, COL["g"])
        a2.plot(r["t"], r["n"], color=COL["g"], label="n")
        a2.plot(r["t"], r["nref"], color=COL["g"], ls=":", label="n*_s (synchr.)")
        a2.set_ylabel("n [obr/min]")
        a.legend(loc="upper left")
        a2.legend(loc="lower right")
        a = ax[0, 1]
        a.plot(r["t"], r["psis"], color=COL["p"], label="|ψs|")
        a.axhline(self.v("psi"), color=COL["gr"], ls=":", label="ψs*")
        a.set_title("Strumień stojana")
        a.set_ylabel("[Vs]")
        a.legend()
        a = ax[1, 0]
        a.plot(r["t"], r["ia"], color=COL["r"], lw=0.5)
        a.set_title("Prąd fazowy i_a")
        a.set_xlabel("t [s]")
        a.set_ylabel("[A]")
        a = ax[1, 1]
        nl = np.linspace(20, 1500, 90)
        eg = M.vf_eig_sweep(par, nl, self.v("psi"), self.v("kRR"))
        e1 = M.vf_eig_sweep(par, nl, self.v("psi"), 1.0)
        self.eig_max = eg.real.max(axis=1)
        a.plot(nl, e1.real.max(axis=1), color=COL["gr"], ls="--", label="Rr nominalne")
        a.plot(nl, self.eig_max, color=COL["r"], label=f"Rr × {self.v('kRR'):.2f}")
        a.axhline(0, color=COL["k"], lw=0.8)
        un = nl[self.eig_max > 0]
        if len(un):
            a.axvspan(un.min(), un.max(), color=COL["r"], alpha=0.1, label="obszar niestabilny")
        a.axvline(self.v("nop"), color=COL["b"], ls=":", label="punkt analizy LTI")
        a.set_xlabel("n*_s [obr/min] (bieg jałowy)")
        a.set_ylabel("max Re(λ) [1/s]")
        a.set_title("Stabilność: największa część rzeczywista biegunów")
        a.legend()
        txt = ""
        if part2:
            ws_ = np.abs(r["n"] - r["nref"])
            i = int(np.argmax(ws_))
            txt += (f"Rampa do {self.v('nend'):.0f} obr/min z {self.v('rate'):.0f} obr/min/s, Rr×{self.v('kRR'):.2f}\n"
                    f"Największa odchyłka prędkości {ws_[i]:.0f} obr/min przy n*≈{r['nref'][i]:.0f} obr/min\n"
                    f"Maks. |i_a| = {np.max(np.abs(r['ia'])):.1f} A, maks. |Te| = {np.max(np.abs(r['Te'])):.0f} Nm\n")
        else:
            txt += (f"Rozruch: n_końc = {r['n'][-1]:.1f} obr/min (zadane {self.v('n'):.0f}), "
                    f"Te = {r['Te'][-1]:.1f} Nm\n|ψs| = {r['psis'][-1]:.3f} Vs\n"
                    f"Maks. |i_a| podczas rozruchu = {np.max(np.abs(r['ia'])):.0f} A\n")
        if len(un):
            txt += f"\nAnaliza biegunów: NIESTABILNY dla n*_s ≈ {un.min():.0f} … {un.max():.0f} obr/min"
        else:
            txt += "\nAnaliza biegunów: stabilny w całym zakresie."
        self.set_result(txt)

    def lti_systems(self):
        d = M.vf_linear(M.M17, self.v("nop"), self.v("psi"), self.v("kRR"), "speed",
                        load_quad=self.v("lq"))
        info = (f"Punkt pracy: n*_s = {self.v('nop'):.0f} obr/min → n0 = {d['n0']:.1f} obr/min, "
                f"Te0 = {d['Te0']:.2f} Nm. Wartości własne: "
                + ", ".join(M.fmt(complex(e)) for e in d["eig"]))
        return [dict(name="G_n(s) = Δn/Δn*_s – odpowiedź prędkości na zmianę częstotliwości", G=d["G_n"],
                     key="vf_n", info=info, **{"in": "Δn*_s [obr/min]", "out": "Δn [obr/min]"}),
                dict(name="G_l(s) = Δn/ΔT_load – wpływ zakłócenia obciążenia", G=d["G_l"], key="vf_l",
                     info=info, **{"in": "ΔT_l [Nm]", "out": "Δn [obr/min]"}),
                dict(name="L(s) – pętla elektromechaniczna (pole ↔ wirnik), kryterium Nyquista", G=d["L"],
                     kind="loop", key="vf_L", info=info, **{"in": "-", "out": "-"})]


class VF2Tab(Tab):
    KEY = "vf2"
    TITLE = "9.1.2 + Tutorial 9.7.2 – napęd V/f sterowany momentem, z czujnikiem prędkości"
    HAS_LTI = True

    def build(self):
        self.slider("T", "moment zadany T*", 0, 250, 113, 1, "Nm")
        self.slider("psi", "strumień zadany ψs*", 0.5, 2.2, 1.868, 0.001, "Vs")
        self.check("lq", "obciążenie wentylatorowe (113 Nm przy 1465 obr/min)", True)
        self.slider("kRR", "mnożnik rezystancji wirnika (maszyna)", 0.5, 1.5, 1.0, 0.01)
        self.slider("Tend", "czas symulacji", 0.3, 3.0, 1.5, 0.1, "s")
        self.slider("nop", "prędkość punktu linearyzacji", 100, 1500, 1413, 1, "obr/min")

    def compute(self):
        r = M.sim_vf(M.M17, "torque", self.v("psi"), T_ref=self.v("T"), kRR=self.v("kRR"),
                     load_quad=self.v("lq"), T_end=self.v("Tend"), dt=2e-4)
        self.r = r
        ax = self.axes(2, 2)
        a = ax[0, 0]
        a.plot(r["t"], r["Te"], color=COL["b"], label="Te")
        a.plot(r["t"], r["Tl"], color=COL["o"], label="Tl")
        a.axhline(self.v("T"), color=COL["gr"], ls=":", label="T*")
        a.set_ylabel("moment [Nm]")
        a.set_title("Moment i prędkość")
        a2 = twin(a, COL["g"])
        a2.plot(r["t"], r["n"], color=COL["g"])
        a2.set_ylabel("n [obr/min]")
        a.legend(loc="lower right")
        a = ax[0, 1]
        a.plot(r["t"], r["psis"], color=COL["p"], label="|ψs|")
        a.axhline(self.v("psi"), color=COL["gr"], ls=":", label="ψs*")
        a.set_ylabel("[Vs]")
        a.set_title("Strumień stojana – oscylacje w stanach przejściowych")
        a.legend()
        a = ax[1, 0]
        a.plot(r["t"], r["ws"], color=COL["b"], label="ωs* (sterownik)")
        a.plot(r["t"], r["n"] / RPM * 2, color=COL["g"], ls="--", label="ωm (czujnik)")
        a.set_ylabel("[rad/s]")
        a.set_xlabel("t [s]")
        a.set_title("ωs* = ωm + ωsl*  – częstotliwość 'nadąża' za wałem")
        a.legend()
        a = ax[1, 1]
        a.plot(r["t"], r["ia"], color=COL["r"], lw=0.5)
        a.set_xlabel("t [s]")
        a.set_ylabel("[A]")
        a.set_title("Prąd fazowy")
        upS = M.stator_model(M.M17)
        isq = 2 * self.v("T") / (3 * 2 * self.v("psi"))
        self.set_result(
            f"Sterownik: isq* = 2T*/(3pψs*) = {isq:.2f} A,  ωsl* = RR·isq*/ψs* = "
            f"{upS['RR'] * isq / self.v('psi'):.3f} rad/s\n"
            f"  (RR = {upS['RR']:.4f} Ω z modelu a = Ls/Lm)\n"
            f"Stan ustalony (koniec):\n"
            f"  Te = {r['Te'][-1]:.2f} Nm   (zadane {self.v('T'):.0f} Nm; książka: 105,1 Nm)\n"
            f"  |ψs| = {r['psis'][-1]:.4f} Vs (książka: 1,871 Vs)\n"
            f"  n = {r['n'][-1]:.1f} obr/min\n"
            "Różnica momentu wynika z pominięcia LσR we wzorze (9.7) na poślizg.")

    def lti_systems(self):
        d = M.vf_linear(M.M17, self.v("nop"), self.v("psi"), self.v("kRR"), "torque", self.v("T"),
                        load_quad=self.v("lq"))
        info = (f"Punkt pracy: n0 = {d['n0']:.0f} obr/min, Te0 = {d['Te0']:.1f} Nm (T* = {self.v('T'):.0f}). "
                "Wartości własne: " + ", ".join(M.fmt(complex(e)) for e in d["eig"]))
        return [dict(name="G_T(s) = ΔTe/ΔT* – jak wiernie moment silnika śledzi zadany", G=d["G_te"],
                     key="vf2_T", info=info, t_end=0.6, **{"in": "ΔT* [Nm]", "out": "ΔTe [Nm]"}),
                dict(name="G_n(s) = Δn/ΔT* – prędkość po zmianie momentu zadanego", G=d["G_n"], key="vf2_n",
                     info=info, **{"in": "ΔT* [Nm]", "out": "Δn [obr/min]"}),
                dict(name="G_l(s) = Δn/ΔT_load – wpływ zakłócenia obciążenia", G=d["G_l"], key="vf_l",
                     info=info, **{"in": "ΔT_l [Nm]", "out": "Δn [obr/min]"}),
                dict(name="L(s) – pętla elektromechaniczna", G=d["L"], kind="loop", key="vf2_L", info=info,
                     **{"in": "-", "out": "-"})]


class DetuneTab(Tab):
    KEY = "detune"
    TITLE = "9.2.1 – zasada FOC, kąt pola (CFO) i odstrojenie (detuning)"

    def build(self):
        self.section("Błąd kąta orientacji (Fig. 9.8)")
        self.slider("dth", "błąd kąta Δθ = θc − θ", -40, 40, 5, 0.5, "°")
        self.section("Odstrojenie rezystancji wirnika w IFO")
        self.slider("k", "estymata RR_est / RR", 0.4, 2.0, 1.3, 0.01)
        self.slider("T", "moment zadany T*", 10, 200, 113, 1, "Nm")
        self.slider("psi", "strumień zadany ψR*", 0.8, 1.8, 1.593, 0.001, "Vs")
        self.check("dyn", "pokaż symulację dynamiczną (UFO + model prądowy)", True)

    def compute(self):
        par = M.M17
        up = M.rotor_model(par)
        ax = self.axes(2, 2)
        a = ax[0, 0]
        isd_c = self.v("psi") / up["LM"]
        isq_c = 2 * self.v("T") / (3 * 2 * self.v("psi"))
        ic = complex(isd_c, isq_c)
        dth = math.radians(self.v("dth"))
        L = abs(ic) * 1.3
        for ang, col, lab in ((0.0, COL["b"], "układ sterownika (d_c, q_c)"),
                              (-dth, COL["r"], "układ maszyny (d, q) – prawdziwy strumień")):
            e = complex(math.cos(ang), math.sin(ang))
            arrow(a, e * L, col, lab, lw=1.2)
            arrow(a, 1j * e * L, col, None, lw=1.2, ls="--")
        arrow(a, ic, COL["g"], "wektor prądu is")
        e = complex(math.cos(-dth), math.sin(-dth))
        isd_m = (ic * e.conjugate()).real
        isq_m = (ic * e.conjugate()).imag
        arrow(a, e * isd_m, COL["o"], "i_sd w maszynie", lw=3)
        a.set_aspect("equal", adjustable="datalim")
        a.set_title("Rozstrojenie układów odniesienia")
        a.legend(loc="lower right")
        # krzywe odstrojenia
        kk = np.linspace(0.4, 2.0, 161)
        dc, info = M.detuning_curves(par, kk, self.v("psi"), self.v("T"))
        b = ax[0, 1]
        b.plot(kk, dc["psi"], color=COL["p"], label="ψR / ψR*")
        b.plot(kk, dc["Te"], color=COL["b"], label="Te / T*")
        b.axvline(1, color=COL["gr"], ls=":")
        b.axvline(self.v("k"), color=COL["r"], ls="--", label="wybrane k")
        b.set_xlabel("RR_est / RR")
        b.set_ylabel("[–]")
        b.set_title("Stan ustalony IFO przy błędnej estymacie RR")
        b.legend()
        c = ax[1, 0]
        c.plot(kk, dc["dth"], color=COL["r"])
        c.axvline(1, color=COL["gr"], ls=":")
        c.set_xlabel("RR_est / RR")
        c.set_ylabel("Δθ [°]")
        c.set_title("Błąd kąta orientacji pola")
        d = ax[1, 1]
        if self.v("dyn"):
            a_r = par["Lm"] / par["Lr"]
            r1, _ = M.sim_foc_current(par, a_r, self.v("psi"), self.v("T"), 0.3, 2.5, 0.02, kRR=1.0,
                                      T_end=1.8, dt=2e-4, load_frac=1.0)
            r2, _ = M.sim_foc_current(par, a_r, self.v("psi"), self.v("T"), 0.3, 2.5, 0.02, kRR=self.v("k"),
                                      T_end=1.8, dt=2e-4, load_frac=1.0)
            d.plot(r1["t"], r1["Te"], color=COL["b"], label="Te – dostrojony")
            d.plot(r2["t"], r2["Te"], color=COL["r"], label=f"Te – RR_est = {self.v('k'):.2f}·RR")
            d.set_ylabel("Te [Nm]")
            d2 = twin(d, COL["p"])
            d2.plot(r1["t"], r1["psiR"], color=COL["p"], ls="--")
            d2.plot(r2["t"], r2["psiR"], color=COL["p"], ls=":")
            d2.set_ylabel("|ψR| [Vs] (kreskowane)")
            d.set_xlabel("t [s]")
            d.set_title("Dynamika: odstrojenie zmienia moment i strumień")
            d.legend(loc="lower right")
        i = int(np.argmin(np.abs(kk - self.v("k"))))
        self.set_result(
            f"Błąd kąta Δθ = {self.v('dth'):.1f}°:\n"
            f"  zadane i_sd = {isd_c:.2f} A, i_sq = {isq_c:.2f} A\n"
            f"  w maszynie  i_sd = {isd_m:.2f} A, i_sq = {isq_m:.2f} A\n"
            f"  strumień ustalony ψR = LM·i_sd = {up['LM'] * isd_m:.3f} Vs (zadany {self.v('psi'):.3f})\n"
            f"  moment ustalony Te = {1.5 * 2 * up['LM'] * isd_m * isq_m:.1f} Nm (zadany {self.v('T'):.0f})\n\n"
            f"Odstrojenie RR_est = {self.v('k'):.2f}·RR (stała czasowa τR = {info['tauR']:.3f} s):\n"
            f"  ψR/ψR* = {dc['psi'][i]:.3f},  Te/T* = {dc['Te'][i]:.3f},  Δθ = {dc['dth'][i]:.1f}°")


class CFOTab(Tab):
    KEY = "cfo"
    TITLE = "9.2.4 – 9.2.8 Metody wyznaczania kąta pola (CFO): IFO i DFO"

    def build(self):
        self.slider("kRs", "estymata Rs_est / Rs (model napięciowy)", 0.5, 1.5, 1.2, 0.01)
        self.slider("kRR", "estymata RR_est / RR (model prądowy)", 0.5, 1.5, 1.2, 0.01)
        self.slider("T", "moment obciążenia T", 0, 200, 113, 1, "Nm")
        self.note("Prędkość przełączenia obserwatora łączonego wyznaczana jest automatycznie: "
                  "tam, gdzie błąd modelu napięciowego staje się mniejszy od błędu modelu prądowego.")

    def compute(self):
        n = np.logspace(0, math.log10(1600), 400)
        r = M.observer_errors(M.M17, n, self.v("kRs"), self.v("kRR"), T_ref=self.v("T"))
        idx = np.nonzero(np.abs(r["ev"]) <= np.abs(r["ec"]))[0]
        nx = n[idx[0]] if len(idx) else n[-1]
        cur_better_low = len(idx) == 0 or idx[0] > 0
        ax = self.fig.subplots(2, 2, gridspec_kw=dict(height_ratios=[1, 1]))
        a = ax[0, 0]
        a.plot(n, r["ev"], color=COL["r"], label="model napięciowy (u, i, Rs) – 9.2.7")
        a.plot(n, r["ec"], color=COL["b"], label="model prądowy (i, ωm, RR) – 9.2.8")
        comb = np.where(n < nx, r["ec"], r["ev"])
        a.plot(n, comb, color=COL["g"], lw=3, alpha=0.4, label="obserwator łączony")
        a.axvline(nx, color=COL["gr"], ls=":")
        a.set_ylim(-60, 60)
        a.set_xscale("log")
        a.set_xlabel("n [obr/min] (skala log.)")
        a.set_ylabel("błąd |ψR| [%]")
        a.set_title("Błąd modułu estymowanego strumienia wirnika")
        a.legend(fontsize=7)
        b = ax[0, 1]
        b.plot(n, r["av"], color=COL["r"], label="model napięciowy")
        b.plot(n, r["ac"], color=COL["b"], label="model prądowy")
        b.set_ylim(-45, 45)
        b.set_xscale("log")
        b.set_xlabel("n [obr/min] (skala log.)")
        b.set_ylabel("błąd kąta [°]")
        b.set_title("Błąd kąta pola (orientacji)")
        b.legend()
        gs = ax[1, 0].get_gridspec()
        ax[1, 0].remove()
        ax[1, 1].remove()
        t = self.fig.add_subplot(gs[1, :])
        t.set_axis_off()
        rows = [["IFO + tachometr (Fig. 9.11)", "ωm, is", "LM, RR, LσR", "tak*", "dryf/offset ωm"],
                ["IFO + enkoder (Fig. 9.12)", "θm, is", "LM, RR, LσR", "TAK", "czujnik na wale"],
                ["DFO hallotrony (Fig. 9.13)", "ψm, is", "brak (a=1)", "TAK", "<120 °C, specjalny silnik"],
                ["DFO cewki pomiarowe (Fig. 9.14)", "e_s (całkowanie)", "LσS", "NIE", "całkowanie przy f→0"],
                ["DFO napięcie+prąd (Fig. 9.15)", "us, is", "Rs, LσS", "NIE", "Rs zależy od temperatury"],
                ["DFO prąd+prędkość (Fig. 9.16)", "is, ωm", "RR, LM", "TAK", "RR zależy od temperatury"]]
        tb = t.table(cellText=rows, colLabels=["Metoda CFO", "Pomiar", "Potrzebne parametry",
                                               "Działa przy f = 0?", "Główna słabość"],
                     loc="center", cellLoc="left")
        tb.auto_set_font_size(False)
        tb.set_fontsize(7.5)
        tb.scale(1, 1.35)
        t.set_title("Porównanie metod wyznaczania kąta pola")
        i1 = int(np.argmin(np.abs(n - 50)))
        i2 = int(np.argmin(np.abs(n - 1465)))
        if cur_better_low:
            concl = (f"Dla tych błędów parametrów model prądowy jest lepszy poniżej ok. {nx:.0f} obr/min,\n"
                     "a model napięciowy powyżej – tu przełącza się obserwator łączony.\n")
        else:
            concl = ("Dla tych błędów parametrów model napięciowy jest lepszy w całym zakresie\n"
                     "(duży błąd RR albo duży moment). Zmniejsz błąd RR lub zwiększ błąd Rs.\n")
        self.set_result(
            f"Model napięciowy (Rs_est = {self.v('kRs'):.2f}·Rs):\n"
            f"  błąd przy 1 obr/min    : {r['ev'][0]:+.1f} %, kąt {r['av'][0]:+.1f}°\n"
            f"  błąd przy 50 obr/min   : {r['ev'][i1]:+.1f} %, kąt {r['av'][i1]:+.1f}°\n"
            f"  błąd przy 1465 obr/min : {r['ev'][i2]:+.2f} %, kąt {r['av'][i2]:+.2f}°\n"
            f"Model prądowy (RR_est = {self.v('kRR'):.2f}·RR):\n"
            f"  błąd (prawie niezależny od n): {r['ec'][i2]:+.1f} %, kąt {r['ac'][i2]:+.1f}°\n\n"
            + concl +
            "Uwaga: przy T = 0 i n → 0 częstotliwość ωs → 0, a błąd modelu napięciowego rośnie\n"
            "bez ograniczeń (dzielenie przez ωs) – bezczujnikowe DFO nie działa przy f = 0.")


class LimitsTab(Tab):
    KEY = "limits"
    TITLE = "9.3 – 9.4 Granice pracy napędu FOC (MA, MF, MTPF) i osłabianie pola"

    def build(self):
        self.slider("u", "maks. napięcie û_max (ampl. fazowa)", 300, 800, 586.9, 0.1, "V")
        self.slider("i", "maks. prąd î_max (ampl.)", 10, 80, 36.7, 0.1, "A")
        self.slider("psi", "maks. strumień wirnika ψR_max", 0.8, 1.8, 1.593, 0.001, "Vs")
        self.check("didak", "dydaktycznie zwiększone χi (jak w Fig. 9.17)", False)
        self.slider("chi", "χi dydaktyczne", 0.1, 0.6, 0.3, 0.01)
        self.slider("Tp", "moment zadany (% T_max w bazie)", 5, 100, 50, 1, "%")
        self.slider("wx", "zaznaczona prędkość ωs/ωs_b", 0.05, 5.0, 1.6, 0.01)

    def compute(self):
        L = M.op_limits(M.M17, self.v("u"), self.v("i"), self.v("psi"),
                        self.v("chi") if self.v("didak") else None)
        chi, Ls, kT = L["chi"], L["Ls"], L["kT"]
        Tcmd = self.v("Tp") / 100 * L["Tmax_b"]
        wb = L["ws_b"]
        wr = np.linspace(0.02, 5.0, 500) * wb
        cur = np.array([M.fw_currents(L, w, Tcmd) for w in wr], dtype=object)
        isd = cur[:, 0].astype(float)
        isq = cur[:, 1].astype(float)
        Te = cur[:, 2].astype(float)
        env = np.array([M.fw_currents(L, w, 1e9)[2] for w in wr])
        ax = self.fig.subplots(1, 3, squeeze=False, gridspec_kw=dict(width_ratios=[1.35, 1, 1]))
        a = ax[0, 0]
        im = L["i_max"]
        th = np.linspace(0, 2 * PI, 300)
        a.plot(im * np.cos(th), im * np.sin(th), color=COL["r"], lw=2, label="MA: |is| = î_max")
        a.axvline(L["isd_max"], color=COL["p"], lw=2, label="ψR_max: i_sd = ψR_max/LM")
        for k, c in ((1.0, COL["b"]), (1.6, COL["c"]), (2.5, COL["g"]), (4.0, COL["ol"])):
            ws = k * wb
            rd = L["u_max"] / (ws * Ls)
            a.add_patch(Ellipse((0, 0), 2 * rd, 2 * rd / chi, fill=False, ec=c, lw=1.2, ls="--"))
            a.plot([], [], color=c, ls="--", label=f"MF elipsa ωs = {k:.1f}·ωs_b")
        xs = np.linspace(0.05, im * 1.2, 300)
        for frac in (0.25, 0.5, 1.0):
            T = frac * L["Tmax_b"]
            a.plot(xs, T / (kT * xs), color=COL["gr"], lw=0.8)
            a.text(xs[-1], T / (kT * xs[-1]), f"{frac * 100:.0f}% T", fontsize=7, color=COL["gr"])
        a.plot(xs, xs / chi, color=COL["k"], lw=1, ls="-.", label="MTPF: i_sd = χi·i_sq")
        a.plot(xs, xs, color=COL["o"], lw=1, ls=":", label="MTPA: i_sd = i_sq")
        a.plot(isd, isq, color=COL["g"], lw=3, label="trajektoria prądu")
        ws_x = self.v("wx") * wb
        px = M.fw_currents(L, ws_x, Tcmd)
        a.plot([px[0]], [px[1]], "o", color=COL["r"], ms=9)
        a.set_xlim(0, im * 1.15)
        a.set_ylim(0, im * 1.15)
        a.set_box_aspect(1)
        a.set_xlabel("i_sd [A]")
        a.set_ylabel("i_sq [A]")
        a.set_title("Płaszczyzna prądu dq (Fig. 9.17/9.18)")
        a.legend(fontsize=6.5, loc="upper right")
        b = ax[0, 1]
        x = wr / wb
        b.plot(x, isd, color=COL["b"], label="i_sd")
        b.plot(x, isq, color=COL["r"], label="i_sq")
        b.plot(x, np.hypot(isd, isq), color=COL["k"], ls="--", label="|is|")
        b.axvline(1, color=COL["gr"], ls=":")
        b.axvline(self.v("wx"), color=COL["r"], ls=":")
        b.set_xlabel("ωs / ωs_b")
        b.set_ylabel("[A]")
        b.set_title("Prądy w funkcji prędkości (Fig. 9.19)")
        b.legend()
        c = ax[0, 2]
        c.plot(x, env, color=COL["gr"], lw=2.5, alpha=0.6, label="maks. moment (obwiednia)")
        c.plot(x, Te, color=COL["b"], label="moment dla zadanego T*")
        c.set_xlabel("ωs / ωs_b")
        c.set_ylabel("Te [Nm]")
        c.set_title("Moment i moc")
        c2 = twin(c, COL["g"])
        c2.plot(x, env * wr / 2 / 1e3, color=COL["g"], lw=2, alpha=0.6)
        c2.plot(x, Te * wr / 2 / 1e3, color=COL["g"], ls="--")
        c2.set_ylabel("P ≈ Te·ωs/p [kW]")
        c.legend(loc="upper right")
        kappa = L["isd_max"] / L["i_max"]
        self.set_result(
            f"χi = LσS/Ls = {chi:.4f}{' (dydaktyczne)' if self.v('didak') else ''}\n"
            f"κi = i_sd,max / î_max = {kappa:.3f}\n"
            f"i_sd,max = ψR_max/LM = {L['isd_max']:.2f} A,  i_sq w punkcie D = {L['isq_D']:.2f} A\n"
            f"Prędkość bazowa ωs_b = {wb:.1f} rad/s → n_b ≈ {wb / 2 * RPM:.0f} obr/min\n"
            f"Maks. moment w bazie T_max = {L['Tmax_b']:.1f} Nm\n"
            f"Moment zadany T* = {Tcmd:.1f} Nm\n\n"
            f"Przy ωs = {self.v('wx'):.2f}·ωs_b ({ws_x / 2 * RPM:.0f} obr/min): obszar '{px[3]}'\n"
            f"   i_sd = {px[0]:.2f} A, i_sq = {px[1]:.2f} A, Te = {px[2]:.1f} Nm, "
            f"P = {px[2] * ws_x / 2 / 1e3:.1f} kW")


class FOC1Tab(Tab):
    KEY = "foc1"
    TITLE = "9.5 + Tutorial 9.7.3 – sterowanie UFO z modelem prądowym i enkoderem"
    HAS_LTI = True

    def build(self):
        amin, amax = M.a_limits(M.M17)
        self.slider("a", f"a ({amin:.4f} wirnik … {amax:.4f} stojan)", amin, amax, amin, 0.0001)
        self.slider("psi", "strumień zadany ψM*", 0.6, 2.0, 1.83, 0.01, "Vs")
        self.slider("T", "skok momentu T*", 0, 200, 113, 1, "Nm")
        self.slider("tau", "filtr momentu τT", 0.0005, 0.1, 0.02, 0.0005, "s")
        self.slider("load", "obciążenie wentylatorowe (× znam.)", 0, 2, 1.0, 0.05)
        self.section("Odstrojenie parametrów sterownika")
        self.slider("kRR", "Rr_est / Rr", 0.5, 2.0, 1.0, 0.01)
        self.slider("kLm", "Lm_est / Lm", 0.7, 1.3, 1.0, 0.01)
        self.section("Analiza: regulator prędkości PI (rozszerzenie)")
        self.slider("Kp", "Kp [Nm/(obr/min)]", 0.01, 5, 0.5, 0.01)
        self.slider("Ki", "Ki [Nm/(obr/min)/s]", 0.0, 50, 5, 0.1)

    def compute(self):
        par = M.M17
        r, upc = M.sim_foc_current(par, self.v("a"), self.v("psi"), self.v("T"), 0.5, 1.5, self.v("tau"),
                                   kRR=self.v("kRR"), kLm=self.v("kLm"), load_frac=self.v("load"))
        self.r = r
        ax = self.axes(2, 2, sharex=True)
        a = ax[0, 0]
        a.plot(r["t"], r["Te"], color=COL["b"], label="Te")
        a.plot(r["t"], r["Tref"], color=COL["k"], ls=":", label="T* (po filtrze)")
        a.plot(r["t"], r["Tl"], color=COL["o"], label="Tl")
        a.set_ylabel("[Nm]")
        a.set_title("Moment i prędkość")
        a2 = twin(a, COL["g"])
        a2.plot(r["t"], r["n"], color=COL["g"])
        a2.set_ylabel("n [obr/min]")
        a.legend(loc="lower left")
        a = ax[0, 1]
        a.plot(r["t"], r["wsl"], color=COL["r"])
        a.set_ylabel("ωsl* [rad/s]")
        a.set_title("Poślizg z sterownika UFO")
        a = ax[1, 0]
        a.plot(r["t"], r["isd"], color=COL["b"], label="i_sd*")
        a.plot(r["t"], r["isq"], color=COL["r"], label="i_sq*")
        a.set_ylabel("[A]")
        a.set_xlabel("t [s]")
        a.set_title("Prądy zadane (wyjście UFO)")
        a.legend()
        a = ax[1, 1]
        a.plot(r["t"], r["psiM"], color=COL["p"], label="|ψM| maszyny")
        a.axhline(self.v("psi"), color=COL["k"], ls=":", label="ψM*")
        a.plot(r["t"], r["psiR"], color=COL["b"], ls="--", lw=0.9, label="|ψR|")
        a.plot(r["t"], r["psis"], color=COL["r"], ls="--", lw=0.9, label="|ψs|")
        a.set_xlabel("t [s]")
        a.set_ylabel("[Vs]")
        a.set_title("Strumienie – stałe mimo zmian momentu")
        a.legend()
        i = np.searchsorted(r["t"], 1.4)
        self.set_result(
            f"a = {self.v('a'):.4f}: sterownik LM = {upc['LM'] * 1e3:.1f} mH, LσR = {upc['LsR'] * 1e3:.2f} mH, "
            f"RR = {upc['RR']:.4f} Ω\n"
            f"t = 1,4 s: Te = {r['Te'][i]:.2f} Nm (T* = {r['Tref'][i]:.1f}), |ψM| = {r['psiM'][i]:.4f} Vs\n"
            f"           n = {r['n'][i]:.1f} obr/min, ωsl* = {r['wsl'][i]:.3f} rad/s\n"
            f"           i_sd* = {r['isd'][i]:.2f} A, i_sq* = {r['isq'][i]:.2f} A\n"
            f"maks. ωsl* = {r['wsl'].max():.2f} rad/s (impuls przy a = Ls/Lm)\n"
            f"t = 2,4 s: Te = {r['Te'][-10]:.1f} Nm, n = {r['n'][-10]:.0f} obr/min")

    def lti_systems(self):
        par = M.M17
        up = M.rotor_model(par)
        J = par["J"]
        kq = M.quad_load_k(113, 1465) * self.v("load")
        W0 = 1465 / RPM * math.sqrt(max(self.v("load"), 1e-6)) if self.v("load") > 0 else 0.0
        B = 2 * kq * W0
        GT = M.TF([1.0], [self.v("tau"), 1.0], name="G_T")
        mech = M.TF([RPM], [J, B], name="G_mech")
        Gpsi = M.TF([up["LM"]], [up["LM"] / up["RR"], 1.0], name="G_ψR")
        C = M.TF([self.v("Kp"), self.v("Ki")], [1.0, 0.0], name="C")
        L = C * GT * mech
        L.name = "L"
        Gn = GT * mech
        Gn.name = "G_n"
        return [dict(name="G_T(s) = ΔTe/ΔT* = 1/(τT·s + 1) – UFO odwraca model maszyny", G=GT, key="foc_T",
                     **{"in": "ΔT* [Nm]", "out": "ΔTe [Nm]"}),
                dict(name="G_n(s) = Δn/ΔT* – moment → prędkość (z linearyzowanym wentylatorem)", G=Gn,
                     key="foc_n", **{"in": "ΔT* [Nm]", "out": "Δn [obr/min]"}),
                dict(name="G_ψR(s) = ΔψR/Δi_sd = LM/(τR·s+1) – dlaczego strumienia nie zmienia się szybko",
                     G=Gpsi, key="foc_psi", **{"in": "Δi_sd [A]", "out": "ΔψR [Vs]"}),
                dict(name="L(s) = PI(s)·G_T(s)·G_mech(s) – projekt pętli prędkości (rozszerzenie)", G=L,
                     kind="loop", key="foc_L", **{"in": "n* [obr/min]", "out": "n/n*"})]


class FOC2Tab(Tab):
    KEY = "foc2"
    TITLE = "9.6 + Tutorial 9.7.4 – UFO z regulatorem prądu i maszyną zasilaną napięciem"
    HAS_LTI = True

    def build(self):
        amin, amax = M.a_limits(M.M17)
        self.slider("a", f"a ({amin:.4f} … {amax:.4f})", amin, amax, amax, 0.0001)
        self.slider("psi", "strumień zadany ψM*", 0.6, 2.0, 1.83, 0.01, "Vs")
        self.slider("T", "skok momentu T*", 0, 200, 113, 1, "Nm")
        self.slider("tau", "filtr momentu τT", 0.0005, 0.05, 0.002, 0.0005, "s")
        self.slider("load", "obciążenie wentylatorowe (× znam.)", 0, 2, 1.0, 0.05)
        self.section("Regulator prądu PI w osiach dq")
        self.slider("Ts", "okres próbkowania Ts", 50, 500, 100, 10, "µs")
        self.slider("kp", "Kp / Kp_opt", 0.1, 2.5, 1.0, 0.01)
        self.slider("ki", "Ki / Ki_opt", 0.0, 3.0, 1.0, 0.01)
        self.check("ff", "kompensacja SEM ψM·ωm (feed-forward)", True)
        self.slider("Udc", "napięcie DC falownika", 600, 1500, 1200, 10, "V")

    def gains(self):
        up = M.rotor_model(M.M17)
        Ts = self.v("Ts") * 1e-6
        Td = 1.5 * Ts
        R = M.M17["Rs"] + up["RR"]
        Kp0 = up["LsS"] / (2 * Td)
        Ki0 = R / (2 * Td)
        return up["LsS"], R, Kp0 * self.v("kp"), Ki0 * self.v("ki"), Td, Ts, Kp0, Ki0

    def compute(self):
        LsS, R, Kp, Ki, Td, Ts, Kp0, Ki0 = self.gains()
        r, info = M.sim_foc_voltage(M.M17, self.v("a"), self.v("psi"), self.v("T"), 0.5, 1.5, self.v("tau"),
                                    Ts, Kp, Ki, self.v("Udc"), self.v("ff"), self.v("load"))
        ax = self.axes(2, 2, sharex=True)
        a = ax[0, 0]
        a.plot(r["t"], r["Te"], color=COL["b"], label="Te")
        a.plot(r["t"], r["Tref"], color=COL["k"], ls=":", label="T*")
        a.plot(r["t"], r["Tl"], color=COL["o"], label="Tl")
        a.set_ylabel("[Nm]")
        a.set_title("Moment i prędkość")
        a2 = twin(a, COL["g"])
        a2.plot(r["t"], r["n"], color=COL["g"])
        a2.set_ylabel("n [obr/min]")
        a.legend(loc="lower left")
        a = ax[0, 1]
        a.plot(r["t"], r["isd_ref"], color=COL["b"], ls=":", label="i_sd*")
        a.plot(r["t"], r["isd"], color=COL["b"], lw=0.9, label="i_sd")
        a.plot(r["t"], r["isq_ref"], color=COL["r"], ls=":", label="i_sq*")
        a.plot(r["t"], r["isq"], color=COL["r"], lw=0.9, label="i_sq")
        a.set_ylabel("[A]")
        a.set_title("Regulacja prądu – zadane i zmierzone")
        a.legend()
        a = ax[1, 0]
        a.plot(r["t"], r["psis"], color=COL["r"], label="|ψs|")
        a.plot(r["t"], r["psiR"], color=COL["b"], label="|ψR|")
        a.axhline(self.v("psi"), color=COL["k"], ls=":", label="ψM*")
        a.set_ylabel("[Vs]")
        a.set_xlabel("t [s]")
        a.set_title("Strumienie stojana i wirnika")
        a.legend()
        a = ax[1, 1]
        a.plot(r["t"], r["u"], color=COL["p"], label="|u| zadane")
        a.axhline(info["umax"], color=COL["r"], ls="--", label="granica Udc/√3")
        a.set_ylabel("[V]")
        a.set_xlabel("t [s]")
        a.set_title("Napięcie falownika")
        a.legend()
        i = np.searchsorted(r["t"], 1.4)
        self.set_result(
            f"Regulator: Kp = {Kp:.2f} Ω (opt {Kp0:.2f}), Ki = {Ki:.0f} Ω/s (opt {Ki0:.0f})\n"
            f"  Ts = {Ts * 1e6:.0f} µs, opóźnienie Td = 1,5·Ts = {Td * 1e6:.0f} µs\n"
            f"  obiekt: LσS = {LsS * 1e3:.2f} mH, R = Rs + RR = {R:.3f} Ω\n"
            f"t = 1,4 s: Te = {r['Te'][i]:.2f} Nm, |ψs| = {r['psis'][i]:.4f} Vs, |ψR| = {r['psiR'][i]:.4f} Vs\n"
            f"   i_sd = {r['isd'][i]:.2f} A (zad. {r['isd_ref'][i]:.2f}), "
            f"i_sq = {r['isq'][i]:.2f} A (zad. {r['isq_ref'][i]:.2f})\n"
            f"   n = {r['n'][i]:.0f} obr/min, |u| = {r['u'][i]:.0f} V\n"
            f"Maks. |u| = {r['u'].max():.0f} V (granica {info['umax']:.0f} V)")

    def lti_systems(self):
        LsS, R, Kp, Ki, Td, Ts, Kp0, Ki0 = self.gains()
        L = M.current_loop_tf(LsS, R, Kp, Ki, Td)
        G = M.TF([1.0], [LsS, R], name="G_i")
        Lr = L.rational(2)
        # G_z = G/(1+L) = G * den_L / (den_L + num_L)
        dist = M.TF(np.polymul(G.num, Lr.den), np.polymul(G.den, np.polyadd(Lr.den, Lr.num)), name="G_z").minreal(1e-4)
        return [dict(name="L(s) = (Kp + Ki/s)·e^(−s·1,5Ts)/(LσS·s + R) – pętla regulacji prądu", G=L,
                     kind="loop", key="cur_L", **{"in": "i* [A]", "out": "i/i*"}),
                dict(name="G_i(s) = Δi/Δu = 1/(LσS·s + R) – obiekt (uzwojenie maszyny)", G=G, key="cur_G",
                     **{"in": "Δu [V]", "out": "Δi [A]"}),
                dict(name="G_z(s) = Δi/Δe = G/(1+L) – tłumienie zakłócenia SEM przez regulator", G=dist,
                     key="cur_z", **{"in": "Δe (SEM) [V]", "out": "Δi [A]"})]


# =============================================================================
# OKNO GLOWNE
# =============================================================================

CH8 = [T81Tab, UniversalTab, T84Tab, T85Tab, HeylandUFOTab, SteadyTab, EstimTab, GridTab,
       Single1Tab, Single2Tab]
CH9 = [KlossTab, VF1Tab, VF2Tab, DetuneTab, CFOTab, LimitsTab, FOC1Tab, FOC2Tab]


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Silnik indukcyjny – modele (rozdz. 8) i sterowanie (rozdz. 9)  ·  Plecs3")
        self.geometry("1500x930")
        try:
            ttk.Style(self).theme_use("clam")
        except tk.TclError:
            pass
        self.sb = ttk.Label(self, text="", anchor="w", relief="sunken")
        self.sb.pack(side="bottom", fill="x")
        main = ttk.Notebook(self)
        main.pack(fill="both", expand=True)
        intro = ttk.Frame(main)
        main.add(intro, text=" Start ")
        tx = ScrolledText(intro, wrap="word", font=("TkDefaultFont", 11), padx=16, pady=10)
        tx.pack(fill="both", expand=True)
        tx.insert("1.0", INTRO)
        tx.configure(state="disabled")
        self.tabs = []
        for title, classes in (("Rozdział 8 – Modele maszyny", CH8), ("Rozdział 9 – Sterowanie napędem", CH9)):
            fr = ttk.Frame(main)
            main.add(fr, text=f" {title} ")
            nb = ttk.Notebook(fr)
            nb.pack(fill="both", expand=True)
            for cls in classes:
                t = cls(nb, self)
                nb.add(t, text=" " + short_name(cls) + " ")
                self.tabs.append(t)
            nb.bind("<<NotebookTabChanged>>", self.on_tab)
        main.bind("<<NotebookTabChanged>>", self.on_main)
        self.main = main
        self.status("Wybierz rozdział i przykład. Obliczenia uruchamiają się przy pierwszym otwarciu zakładki.")

    def status(self, t):
        self.sb.config(text="  " + t)

    def on_main(self, e):
        w = self.main.nametowidget(self.main.select())
        for child in w.winfo_children():
            if isinstance(child, ttk.Notebook):
                self.on_tab(None, child)

    def on_tab(self, e, nb=None):
        nb = nb or e.widget
        try:
            t = nb.nametowidget(nb.select())
        except Exception:
            return
        if isinstance(t, Tab) and not t.done:
            t.after(30, t.refresh)


def short_name(cls):
    names = {"T81Tab": "8.6.1 Uproszczony", "UniversalTab": "8.6.2 Uniwersalny (a)",
             "T84Tab": "8.6.4 Prądowy i_d/i_q", "T85Tab": "8.6.5 UFO", "HeylandUFOTab": "8.3.5 Heyland",
             "SteadyTab": "8.3.6 Stan ustalony", "EstimTab": "8.6.6 Parametry",
             "GridTab": "8.6.7 Rozruch z sieci", "Single1Tab": "8.6.9 1-faz. rozruch",
             "Single2Tab": "8.5.1 1-faz. ustalony", "KlossTab": "9.1 Kloss",
             "VF1Tab": "9.7.1 V/f prosty", "VF2Tab": "9.7.2 V/f + czujnik", "DetuneTab": "9.2.1 Odstrojenie",
             "CFOTab": "9.2.4 Metody CFO", "LimitsTab": "9.3 Granice / osł. pola",
             "FOC1Tab": "9.7.3 UFO prądowy", "FOC2Tab": "9.7.4 UFO + reg. prądu"}
    return names.get(cls.__name__, cls.__name__)


def selftest(outdir):
    """Przelicza wszystkie zakladki i zapisuje wykresy do PNG (test bez interakcji)."""
    os.makedirs(outdir, exist_ok=True)
    app = App()
    app.update()
    ok = True
    for t in app.tabs:
        t.refresh()
        app.update()
        txt = t.res.get("1.0", "end")
        bad = "BŁĄD" in txt
        ok &= not bad
        t.fig.savefig(os.path.join(outdir, f"{t.KEY}.png"), dpi=80)
        if t.HAS_LTI:
            for i in range(len(t.lti.systems)):
                t.lti.sel.current(i)
                t.lti.update_plots(keep=True)
                t.lti.fig.savefig(os.path.join(outdir, f"{t.KEY}_lti{i}.png"), dpi=80)
                rep = t.lti.txt.get("1.0", "end")
                if "Traceback" in rep:
                    ok = False
                    print(t.KEY, "LTI", i, rep[:2000])
        print(f"{'OK ' if not bad else 'ERR'} {t.KEY:8s} {t.TITLE}")
        if bad:
            print(txt[:3000])
    app.destroy()
    print("SELFTEST", "PASSED" if ok else "FAILED")
    return ok


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        i = sys.argv.index("--selftest")
        out = sys.argv[i + 1] if len(sys.argv) > i + 1 else "selftest_png"
        sys.exit(0 if selftest(out) else 1)
    App().mainloop()
