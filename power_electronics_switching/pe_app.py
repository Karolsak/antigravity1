"""
Strategie przełączania przekształtników energoelektronicznych - aplikacja Tkinter
==================================================================================
Interaktywne wykresy (Matplotlib, z zoomem) dla WSZYSTKICH przykładów z rozdziałów:
  4. Mostek H (full-bridge): stany przewodzenia, PWM bipolarne/unipolarne,
     wektory napięcia, dc-dc 4-ćwiartkowe, dwa mostki LC-RC, phase-shift PWM
  5. Przekształtniki trójfazowe: układy 3-f, sine-PWM, Clarke, wektory, SVPWM
  6. Przekształtniki wielopoziomowe NPC: gałąź 3/4-poziomowa, mapa wektorów,
     SVPWM 3-poziomowe, balansowanie kondensatorów
oraz analizę UKŁADÓW STEROWANIA (Bode, zpk, Nyquist, step, bieguny, przemiatanie
parametrów) dla: regulatora PI prądu, regulatora PR, kaskady napięcia DC i
regulatora balansu kondensatorów NPC.

Każda zakładka: suwaki parametrów, wykresy, wyniki liczbowe, objaśnienie dla
ucznia liceum.

Uruchomienie:   python pe_app.py
Wymagania:      Python 3.8+, numpy, matplotlib  (tkinter jest w standardowym Pythonie)
"""
import math
import os
import sys
import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pe_core as pc           # noqa: E402
import pe_control as ct        # noqa: E402
from pe_texts import EXPL, CONTROL_INTRO   # noqa: E402

PI = math.pi
SQ3 = math.sqrt(3.0)
COL = dict(a="#d62728", b="#2ca02c", c="#1f77b4", k="#333333", o="#ff7f0e",
           p="#9467bd", g="#7f7f7f", on="#2ca02c", off="#dddddd", path="#e31a1c")
PH_COL = [COL["a"], COL["b"], COL["c"]]


def fmt(x, d=4):
    return "-" if x is None else f"{x:.{d}g}"


# =============================================================================
# Klasa bazowa zakładki przykładu
# =============================================================================

class ExampleTab(ttk.Frame):
    """Lewy panel: suwaki, opcje, wyniki.  Prawy: wykresy + objaśnienie.
    Podklasy definiują PARAMS, key, extra_controls(), update_plot()."""
    PARAMS = []   # (nazwa, etykieta, min, max, domyślna, krok)
    key = ""

    def __init__(self, master):
        super().__init__(master)
        self.vars = {}
        self.opts = {}
        self._pending = None
        pw = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        pw.pack(fill=tk.BOTH, expand=True)
        left_outer = ttk.Frame(pw, width=320)
        right = ttk.Frame(pw)
        pw.add(left_outer, weight=0)
        pw.add(right, weight=1)

        # przewijany panel lewy
        cnv = tk.Canvas(left_outer, width=310, highlightthickness=0)
        sb = ttk.Scrollbar(left_outer, orient=tk.VERTICAL, command=cnv.yview)
        left = ttk.Frame(cnv)
        left.bind("<Configure>", lambda e: cnv.configure(scrollregion=cnv.bbox("all")))
        cnv.create_window((0, 0), window=left, anchor="nw")
        cnv.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        cnv.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        ttk.Label(left, text="Parametry", font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=6, pady=(6, 2))
        for name, label, lo, hi, val, res in self.PARAMS:
            fr = ttk.Frame(left)
            fr.pack(fill=tk.X, padx=6, pady=0)
            var = tk.DoubleVar(value=val)
            self.vars[name] = var
            ttk.Label(fr, text=label).pack(anchor="w")
            sc = tk.Scale(fr, from_=lo, to=hi, resolution=res, orient=tk.HORIZONTAL,
                          variable=var, showvalue=True, length=270,
                          command=lambda _e: self.schedule())
            sc.pack(fill=tk.X)
        self.extra_controls(left)
        ttk.Button(left, text="Przywróć domyślne", command=self.reset).pack(fill=tk.X, padx=6, pady=6)
        ttk.Label(left, text="Wyniki", font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=6)
        self.result = tk.Text(left, height=18, width=40, font=("Consolas", 9), bg="#f4f6fa",
                              wrap=tk.WORD)
        self.result.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))

        vpw = ttk.PanedWindow(right, orient=tk.VERTICAL)
        vpw.pack(fill=tk.BOTH, expand=True)
        figfr = ttk.Frame(vpw)
        self.build_figure_area(figfr)
        txtfr = ttk.Frame(vpw)
        self.expl = ScrolledText(txtfr, height=12, wrap=tk.WORD, font=("Segoe UI", 10), bg="#fffdf3")
        self.expl.pack(fill=tk.BOTH, expand=True)
        self.expl.insert("1.0", self.explanation())
        self.expl.configure(state=tk.DISABLED)
        vpw.add(figfr, weight=3)
        vpw.add(txtfr, weight=1)
        self.refresh()

    # --- elementy do nadpisania ---
    def build_figure_area(self, parent):
        self.fig = Figure(figsize=(9, 6), dpi=90)
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        NavigationToolbar2Tk(self.canvas, parent).update()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def explanation(self):
        return EXPL.get(self.key, "")

    def extra_controls(self, parent):
        pass

    # --- pomocnicze kontrolki ---
    def add_check(self, parent, name, label, val=False):
        var = tk.BooleanVar(value=val)
        self.opts[name] = var
        ttk.Checkbutton(parent, text=label, variable=var, command=self.schedule).pack(anchor="w", padx=8)
        return var

    def add_radio(self, parent, name, label, choices, val):
        var = tk.StringVar(value=val)
        self.opts[name] = var
        fr = ttk.LabelFrame(parent, text=label)
        fr.pack(fill=tk.X, padx=6, pady=3)
        for txt, v in choices:
            ttk.Radiobutton(fr, text=txt, value=v, variable=var, command=self.schedule).pack(anchor="w")
        return var

    def add_combo(self, parent, name, label, values, val):
        var = tk.StringVar(value=val)
        self.opts[name] = var
        ttk.Label(parent, text=label).pack(anchor="w", padx=6)
        cb = ttk.Combobox(parent, textvariable=var, values=values, state="readonly")
        cb.pack(fill=tk.X, padx=6, pady=2)
        cb.bind("<<ComboboxSelected>>", lambda _e: self.schedule())
        return var

    def o(self, name):
        return self.opts[name].get()

    def reset(self):
        for name, _l, _lo, _hi, val, _r in self.PARAMS:
            self.vars[name].set(val)
        self.refresh()

    def p(self, name):
        return self.vars[name].get()

    def schedule(self):
        if self._pending:
            self.after_cancel(self._pending)
        self._pending = self.after(150, self.refresh)

    def refresh(self):
        self._pending = None
        self.fig.clear()
        try:
            lines = self.update_plot() or []
        except Exception as exc:     # nie zabijaj GUI przy nietypowych parametrach
            lines = [f"Błąd obliczeń: {exc!r}"]
        self.result.delete("1.0", tk.END)
        self.result.insert("1.0", "\n".join(lines))
        try:
            self.fig.tight_layout()
        except Exception:
            pass
        self.canvas.draw_idle()

    def update_plot(self):
        raise NotImplementedError


def wave_ax(ax, title, ylabel, xlabel="t [ms]"):
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.grid(True, alpha=0.3)


# =============================================================================
# Rysowanie schematów
# =============================================================================

def _box(ax, x, y, w, h, txt, on=False, hl=False, fs=8):
    fc = "#c7f0c7" if on else "#f0f0f0"
    ec = COL["path"] if hl else "#555555"
    ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h, fc=fc, ec=ec, lw=3 if hl else 1, zorder=3))
    ax.text(x, y, txt, ha="center", va="center", fontsize=fs, zorder=4,
            color=COL["path"] if hl else "black", fontweight="bold" if hl else "normal")


def draw_hbridge(ax, S, res, i_dir):
    """Schemat mostka H z zaznaczoną drogą prądu."""
    ax.set_xlim(-0.8, 7.2)
    ax.set_ylim(-0.6, 3.8)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.plot([0, 6.5], [3.2, 3.2], "k-", lw=1.5)
    ax.plot([0, 6.5], [0.0, 0.0], "k-", lw=1.5)
    ax.plot([0, 0], [0, 3.2], "k-", lw=1.5)
    ax.add_patch(Circle((0, 1.6), 0.35, fc="white", ec="k", zorder=3))
    ax.text(0, 1.6, "Vdc", ha="center", va="center", fontsize=8, zorder=4)
    ax.text(0.15, 3.35, "P (+)", fontsize=8)
    ax.text(0.15, -0.35, "N (-)", fontsize=8)
    path = res.get("path", []) if not res.get("forbidden") else []
    for x, (up, dn, du, dd, nm) in zip([2.0, 5.0], [("S1", "S2", "D1", "D2", "a"),
                                                       ("S3", "S4", "D3", "D4", "b")]):
        ax.plot([x, x], [0, 3.2], "k-", lw=1)
        ax.plot([x + 0.7, x + 0.7], [0.3, 2.9], "k-", lw=0.6)
        su = S[up]
        sd = S[dn]
        _box(ax, x, 2.4, 0.7, 0.55, up, on=su, hl=up in path)
        _box(ax, x, 0.8, 0.7, 0.55, dn, on=sd, hl=dn in path)
        _box(ax, x + 0.7, 2.4, 0.45, 0.45, du, hl=du in path, fs=7)
        _box(ax, x + 0.7, 0.8, 0.45, 0.45, dd, hl=dd in path, fs=7)
        ax.plot([x], [1.6], "ko", ms=5)
        ax.text(x - 0.35, 1.6, nm, fontsize=10, ha="right", va="center", fontweight="bold")
    # obciążenie R-L
    ax.plot([2.0, 2.9], [1.6, 1.6], "k-", lw=1)
    ax.plot([4.1, 5.0], [1.6, 1.6], "k-", lw=1)
    ax.add_patch(Rectangle((2.9, 1.4), 1.2, 0.4, fc="#fff3c4", ec="k", zorder=3))
    ax.text(3.5, 1.6, "R - L", ha="center", va="center", fontsize=8, zorder=4)
    if not res.get("forbidden"):
        x0, x1 = (2.4, 4.6) if i_dir > 0 else (4.6, 2.4)
        ax.add_patch(FancyArrowPatch((x0, 1.98), (x1, 1.98), arrowstyle="-|>",
                                     mutation_scale=15, color=COL["path"], lw=2))
        ax.text(3.5, 2.15, "i", color=COL["path"], ha="center", fontsize=10)
    if res.get("forbidden"):
        ax.text(3.25, 3.55, res["msg"], ha="center", color="red", fontsize=10, fontweight="bold")
    else:
        ax.text(3.25, 3.55, f"vo = v_a - v_b = {res['vo']:+.0f}·Vdc", ha="center",
                fontsize=11, fontweight="bold", color="#1f3d7a")


def draw_npc_leg(ax, S, res, i_dir):
    """Schemat gałęzi 3-poziomowej NPC."""
    ax.set_xlim(-1.6, 4.6)
    ax.set_ylim(-0.5, 4.6)
    ax.set_aspect("equal")
    ax.axis("off")
    path = res.get("path", [])
    forb = res.get("forbidden")
    ax.plot([-1, 2], [4.2, 4.2], "k-")
    ax.plot([-1, 2], [-0.2, -0.2], "k-")
    ax.plot([-1, -1], [-0.2, 4.2], "k-")
    _box(ax, -1, 3.1, 0.8, 0.45, "C góra", fs=7)
    _box(ax, -1, 0.9, 0.8, 0.45, "C dół", fs=7)
    ax.plot([-1, 0.9], [2.0, 2.0], "k-", lw=1)
    ax.text(-1.5, 2.0, "M", fontsize=10, fontweight="bold", va="center")
    ax.text(-1.5, 4.2, "P", fontsize=10, fontweight="bold", va="center")
    ax.text(-1.5, -0.2, "N", fontsize=10, fontweight="bold", va="center")
    ax.plot([2, 2], [-0.2, 4.2], "k-", lw=1)
    ys = [3.7, 2.7, 1.3, 0.3]
    for k, y in enumerate(ys):
        nm = f"S{k + 1}"
        _box(ax, 2, y, 0.7, 0.5, nm, on=S[k], hl=(nm in path), fs=8)
        _box(ax, 2.75, y, 0.45, 0.4, f"D{k + 1}", hl=(f"D{k + 1}" in path), fs=7)
    _box(ax, 0.9, 2.7, 0.6, 0.4, "Dm1", hl="Dm1" in path, fs=7)
    _box(ax, 0.9, 1.3, 0.6, 0.4, "Dm2", hl="Dm2" in path, fs=7)
    ax.plot([0.9, 0.9], [2.0, 2.5], "k-", lw=1)
    ax.plot([0.9, 0.9], [1.5, 2.0], "k-", lw=1)
    ax.plot([0.9, 0.9, 2], [2.9, 3.2, 3.2], "k-", lw=1)
    ax.plot([0.9, 0.9, 2], [1.1, 0.8, 0.8], "k-", lw=1)
    ax.plot([2, 3.9], [2.0, 2.0], "k-", lw=1.5)
    ax.plot([3.9], [2.0], "ko")
    ax.text(4.0, 2.0, "o", fontsize=11, va="center", fontweight="bold")
    if not forb:
        y0, y1 = (2.25, 2.25)
        xa, xb = (2.9, 3.8) if i_dir > 0 else (3.8, 2.9)
        ax.add_patch(FancyArrowPatch((xa, y0), (xb, y1), arrowstyle="-|>", mutation_scale=14,
                                     color=COL["path"], lw=2))
        lv = {0: "0", 1: "Vdc/2", 2: "Vdc"}[res["level"]]
        ax.text(1.5, 4.45, f"v_oN = {lv}", ha="center", fontsize=12, fontweight="bold", color="#1f3d7a")
    else:
        ax.text(1.5, 4.45, "ZWARCIE!", ha="center", fontsize=12, fontweight="bold", color="red")


# =============================================================================
# ROZDZIAŁ 4
# =============================================================================

class TabH1(ExampleTab):
    key = "H1"
    PARAMS = []

    def extra_controls(self, parent):
        fr = ttk.LabelFrame(parent, text="Tranzystory (zaznacz = ON)")
        fr.pack(fill=tk.X, padx=6, pady=4)
        for nm, v in [("S1", True), ("S2", False), ("S3", False), ("S4", True)]:
            var = tk.BooleanVar(value=v)
            self.opts[nm] = var
            ttk.Checkbutton(fr, text=nm, variable=var, command=self.schedule).pack(side=tk.LEFT, padx=4)
        self.add_radio(parent, "dir", "Kierunek prądu obciążenia",
                       [("a -> b (dodatni)", "+"), ("b -> a (ujemny)", "-")], "+")
        ttk.Label(parent, text="Szybki wybór stanu:").pack(anchor="w", padx=6)
        fr2 = ttk.Frame(parent)
        fr2.pack(fill=tk.X, padx=6)
        for txt, st in [("+Vdc", (1, 0, 0, 1)), ("-Vdc", (0, 1, 1, 0)),
                        ("0 (S1S3)", (1, 0, 1, 0)), ("0 (S2S4)", (0, 1, 0, 1))]:
            ttk.Button(fr2, text=txt, width=8,
                       command=lambda s=st: self.set_state(s)).pack(side=tk.LEFT, padx=1)

    def set_state(self, st):
        for nm, v in zip(["S1", "S2", "S3", "S4"], st):
            self.opts[nm].set(bool(v))
        self.refresh()

    def update_plot(self):
        S = {nm: self.o(nm) for nm in ["S1", "S2", "S3", "S4"]}
        idir = 1 if self.o("dir") == "+" else -1
        res = pc.hbridge_state(S["S1"], S["S2"], S["S3"], S["S4"], idir)
        ax = self.fig.add_subplot(1, 2, 1)
        draw_hbridge(ax, S, res, idir)
        ax.set_title("Mostek H - elementy przewodzące (czerwone)", fontsize=10)
        # tabela wszystkich 16 kombinacji
        ax2 = self.fig.add_subplot(1, 2, 2)
        ax2.axis("off")
        rows = []
        for k in range(16):
            s = [(k >> (3 - j)) & 1 for j in range(4)]
            rp = pc.hbridge_state(*s, 1)
            rn = pc.hbridge_state(*s, -1)
            if rp["forbidden"]:
                rows.append(["".join(map(str, s)), "ZABRONIONY", "", ""])
            else:
                sg = lambda v: "0" if v == 0 else f"{v:+.0f}"
                rows.append(["".join(map(str, s)), sg(rp['vo']), sg(rn['vo']),
                             "zamierz." if rp["vo"] == rn["vo"] else "zależy od i"])
        tb = ax2.table(cellText=rows, colLabels=["S1S2S3S4", "vo (i>0)", "vo (i<0)", "uwagi"],
                       loc="center", cellLoc="center")
        tb.auto_set_font_size(False)
        tb.set_fontsize(8)
        tb.scale(1, 1.15)
        cur = "".join(str(int(S[n])) for n in ["S1", "S2", "S3", "S4"])
        for r_i, r in enumerate(rows):
            col = "#ffd6d6" if r[1] == "ZABRONIONY" else ("#d6f5d6" if r[3] == "zamierz." else "#fff6d6")
            if r[0] == cur:
                col = "#9fc5ff"
            for c_i in range(4):
                tb[(r_i + 1, c_i)].set_facecolor(col)
        ax2.set_title("Wszystkie 16 kombinacji (niebieski = bieżąca)", fontsize=10)
        lines = [res["msg"]]
        if not res["forbidden"]:
            lines += [f"v_aN = {res['vaN']:.0f}·Vdc, v_bN = {res['vbN']:.0f}·Vdc",
                      f"vo = {res['vo']:+.0f}·Vdc"]
        lines += ["", "Zielone = stan zamierzony (wynik nie zależy",
                  "od kierunku prądu); żółte = zależy od prądu;",
                  "czerwone = zwarcie szyny DC."]
        return lines


class TabH2(ExampleTab):
    key = "H2"
    PARAMS = [("Vdc", "Vdc [V]", 6, 100, 24, 1),
              ("R", "R [Ω]", 0.5, 30, 10, 0.5),
              ("L", "L [mH]", 0.1, 20, 1, 0.1),
              ("ma", "ma (indeks modulacji)", 0.05, 1.2, 0.8, 0.01),
              ("fsw", "fsw [Hz]", 500, 10000, 2500, 50),
              ("f1", "f1 [Hz]", 10, 100, 50, 1)]

    def update_plot(self):
        Vdc, R, L, ma, fsw, f1 = (self.p(k) for k in ["Vdc", "R", "L", "ma", "fsw", "f1"])
        L *= 1e-3
        r = pc.hbridge_bipolar(Vdc, R, L, f1, fsw, ma)
        t = r["t"] * 1e3
        ax1 = self.fig.add_subplot(4, 1, 1)
        ax1.plot(t, r["c"], color=COL["g"], lw=0.6, label="nośna c(t)")
        ax1.plot(t, r["m"], color=COL["a"], lw=1.5, label="m(t)")
        wave_ax(ax1, "Porównanie sinus-trójkąt", "")
        ax1.legend(fontsize=7, loc="upper right")
        ax2 = self.fig.add_subplot(4, 1, 2, sharex=ax1)
        ax2.plot(t, r["vo"], color=COL["c"], lw=0.6, label="vo")
        V1 = pc.fundamental(r["vo"], 1)
        ax2.plot(t, V1 * np.sin(2 * PI * f1 * r["t"]), "k--", lw=1, label="1. harm.")
        wave_ax(ax2, "Napięcie mostka (rys. 4.6) - tylko ±Vdc", "V")
        ax2.legend(fontsize=7, loc="upper right")
        ax3 = self.fig.add_subplot(4, 1, 3, sharex=ax1)
        ax3.plot(t, r["iL"], color=COL["a"], lw=1)
        wave_ax(ax3, "Prąd obciążenia iL (rys. 4.5)", "A")
        ax3b = self.fig.add_subplot(4, 2, 7)
        ax3b.plot(t, r["idc"], color=COL["p"], lw=0.6)
        ax3b.axhline(np.mean(r["idc"]), color="k", ls="--", lw=1)
        wave_ax(ax3b, "Prąd szyny DC idc (rys. 4.7)", "A")
        ax4 = self.fig.add_subplot(4, 2, 8)
        h, amp = pc.spectrum(r["vo"], 1)
        mask = h <= 3 * fsw / f1 + 5
        ax4.bar(h[mask], amp[mask], width=0.8, color=COL["c"])
        wave_ax(ax4, "Widmo vo", "V", "rząd harmonicznej")
        I1 = pc.fundamental(r["iL"], 1)
        Irms = pc.rms(r["iL"])
        P = Irms ** 2 * R
        Idc = float(np.mean(r["idc"]))
        Z = math.hypot(R, 2 * PI * f1 * L)
        ripple = np.max(np.abs(r["iL"] - I1 * np.sin(2 * PI * f1 * r["t"] - math.atan2(2 * PI * f1 * L, R))))
        return [f"V1 (symul.) = {V1:.3f} V   (teoria ma·Vdc = {ma * Vdc:.3f} V)",
                f"|Z| = {Z:.3f} Ω,  φ = {math.degrees(math.atan2(2 * PI * f1 * L, R)):.2f}°",
                f"I1 = {I1:.3f} A  (teoria {min(ma, 1) * Vdc / Z:.3f} A)",
                f"I_rms = {Irms:.3f} A",
                f"tętnienie prądu (max odchyłka) ≈ {ripple:.3f} A",
                f"P_R = I_rms²·R = {P:.3f} W",
                f"Vdc·Idc_śr = {Vdc * Idc:.3f} W   (bilans energii ✓)",
                f"THD napięcia = {pc.thd_from_spectrum(h, amp):.1f} %",
                f"THD prądu = {pc.thd_from_spectrum(*pc.spectrum(r['iL'], 1)):.2f} %",
                f"mf = fsw/f1 = {fsw / f1:.0f}"]


class TabH3(ExampleTab):
    key = "H3"
    PARAMS = [("Vdc", "Vdc [V]", 6, 100, 24, 1),
              ("Vreq", "Vreq [V]", -100, 100, 6, 0.5),
              ("Tsw", "Tsw [µs]", 20, 1000, 200, 10)]

    def update_plot(self):
        Vdc, Vreq, Tsw = self.p("Vdc"), self.p("Vreq"), self.p("Tsw")
        Vreq = float(np.clip(Vreq, -Vdc, Vdc))
        lines = [f"Vreq (ograniczone do ±Vdc) = {Vreq:.2f} V", ""]
        ax0 = self.fig.add_subplot(4, 1, 1)
        ax0.set_xlim(-1.3, 1.3)
        ax0.set_ylim(-0.6, 0.6)
        ax0.axhline(0, color="k", lw=1)
        for x, lab in [(-1, "-Vdc\nS2-S3"), (0, "0\nS1-S3 / S2-S4"), (1, "+Vdc\nS1-S4")]:
            ax0.plot(x, 0, "o", ms=10, color=COL["c"])
            ax0.text(x, -0.45, lab, ha="center", fontsize=8)
        ax0.annotate("", xy=(Vreq / Vdc, 0.0), xytext=(0, 0),
                     arrowprops=dict(arrowstyle="-|>", color=COL["a"], lw=2.5))
        ax0.text(Vreq / Vdc, 0.2, "Vreq", color=COL["a"], ha="center")
        ax0.set_yticks([])
        ax0.set_title("Wektory napięcia mostka H (rys. 4.8)", fontsize=10)
        strategies = [("bipolar", "BIPOLARNE (T0 = 0)"), ("unipolar", "UNIPOLARNE (+V/0 lub -V/0)"),
                      ("ps", "PHASE-SHIFT (T+ = T-, średnia 0)")]
        for n, (st, title) in enumerate(strategies):
            ax = self.fig.add_subplot(4, 1, n + 2)
            if st == "ps":
                dd = min(abs(Vreq) / Vdc, 1.0) / 2
                Tp = Tm = dd * Tsw
                T0 = Tsw - 2 * Tp
                seg = [(0, T0 / 3), (Vdc, Tp), (0, T0 / 3), (-Vdc, Tm), (0, T0 / 3)]
                lines.append(f"{title}:\n  d = {dd:.3f}, T+ = T- = {Tp:.1f} µs, T0 = {T0:.1f} µs (3 części)")
                lines.append(f"  średnia = 0 V (transformator!), Vrms = {Vdc * math.sqrt(2 * dd):.2f} V")
            else:
                Tp, T0, Tm = pc.vector_intervals(Vreq, Vdc, Tsw, st)
                if st == "bipolar":
                    seg = [(-Vdc, Tm / 2), (Vdc, Tp), (-Vdc, Tm / 2)]
                else:
                    vv = Vdc if Vreq >= 0 else -Vdc
                    Tn = Tp if Vreq >= 0 else Tm
                    seg = [(0, T0 / 2), (vv, Tn), (0, T0 / 2)]
                avg = (Vdc * Tp - Vdc * Tm) / Tsw
                lines.append(f"{title}:\n  T+ = {Tp:.1f} µs, T0 = {T0:.1f} µs, T- = {Tm:.1f} µs"
                             f"\n  średnia = {avg:.3f} V ✓")
            tt, vv_ = [0.0], []
            for v, d in seg:
                vv_.append(v)
                tt.append(tt[-1] + d)
            tt2 = np.repeat(tt, 2)[1:-1]
            vv2 = np.repeat(vv_, 2)
            for cyc in range(2):
                ax.plot(tt2 + cyc * Tsw, vv2, color=COL["c"], lw=2)
            ax.axhline(Vreq if st != "ps" else 0, color=COL["a"], ls="--", lw=1)
            ax.set_ylim(-Vdc * 1.25, Vdc * 1.25)
            wave_ax(ax, title, "vo [V]", "t [µs]")
        return lines


class TabH4(ExampleTab):
    key = "H4"
    PARAMS = [("Vdc", "Vdc [V]", 6, 100, 24, 1),
              ("d", "d (wypełnienie, -1..1)", -1, 1, 0.5, 0.01),
              ("E", "E - SEM obciążenia [V]", -50, 50, 0, 0.5),
              ("R", "R [Ω]", 0.2, 20, 2, 0.1),
              ("L", "L [mH]", 0.2, 20, 2, 0.1),
              ("fsw", "fsw [Hz]", 500, 20000, 5000, 100)]

    def update_plot(self):
        Vdc, d, E, R, L, fsw = (self.p(k) for k in ["Vdc", "d", "E", "R", "L", "fsw"])
        r = pc.single_dcdc(Vdc, d, fsw, R, L * 1e-3, E, n_show=4)
        t = r["t"] * 1e3
        ax1 = self.fig.add_subplot(4, 1, 1)
        ax1.plot(t, r["c1"], color=COL["g"], lw=1, label="c1 (piła 0..1)")
        ax1.plot(t, r["c2"], color=COL["p"], lw=1, label="c2 = -c1")
        ax1.axhline(d, color=COL["a"], lw=2, label="d")
        wave_ax(ax1, "Nośne i wypełnienie (rys. 4.11-4.12)", "")
        ax1.legend(fontsize=7, loc="upper right", ncol=3)
        ax2 = self.fig.add_subplot(4, 1, 2, sharex=ax1)
        ax2.step(t, r["S1"] + 1.5, where="post", color=COL["a"], label="S1 (S2 = negacja)")
        ax2.step(t, r["S3"] + 0.0, where="post", color=COL["c"], label="S3 (S4 = negacja)")
        ax2.set_yticks([0, 1, 1.5, 2.5])
        ax2.set_yticklabels(["S3 off", "S3 on", "S1 off", "S1 on"])
        wave_ax(ax2, "Impulsy bramkowe", "")
        ax3 = self.fig.add_subplot(4, 1, 3, sharex=ax1)
        ax3.plot(t, r["vo"], color=COL["c"], lw=1.2)
        ax3.axhline(np.mean(r["vo"]), color="k", ls="--", lw=1, label="średnia")
        wave_ax(ax3, "Napięcie mostka vo", "V")
        ax3.legend(fontsize=7)
        ax4 = self.fig.add_subplot(4, 1, 4, sharex=ax1)
        ax4.plot(t, r["iL"], color=COL["a"])
        ax4.axhline(0, color="k", lw=0.6)
        wave_ax(ax4, "Prąd obciążenia R-L-E", "A")
        Vo = float(np.mean(r["vo"]))
        I = float(np.mean(r["iL"]))
        quad = {(True, True): "I (napędzanie w przód)", (True, False): "II (hamowanie odzyskowe, przód)",
                (False, False): "III (napędzanie w tył)", (False, True): "IV (hamowanie odzyskowe, tył)"}
        q = quad[(Vo >= 0, I >= 0)]
        return [f"Vo średnie = {Vo:.3f} V  (teoria d·Vdc = {d * Vdc:.3f} V)",
                f"I średnie = {I:.3f} A  (teoria (dVdc-E)/R = {(d * Vdc - E) / R:.3f} A)",
                f"tętnienie ΔI (p-p) = {np.ptp(r['iL']):.3f} A",
                f"moc z DC = {Vo * I:.2f} W  ({'DC -> obciążenie' if Vo * I >= 0 else 'obciążenie -> DC'})",
                f"ćwiartka pracy: {q}",
                f"L/R = {L / R:.2f} ms,  Tsw = {1e3 / fsw:.3f} ms"]


class TabH5(ExampleTab):
    key = "H5"
    PARAMS = [("Vdc1", "Vdc LC [V]", 6, 60, 24, 1),
              ("Vdc2", "Vdc RC [V]", 6, 60, 24, 1),
              ("d1", "d1 (LC)", -1, 1, 0.5, 0.01),
              ("d2", "d2 (RC)", -1, 1, 0.3, 0.01),
              ("R", "R połączenia [Ω]", 0.1, 10, 1.0, 0.1),
              ("L", "L dławika [mH]", 0.2, 20, 2.0, 0.1),
              ("fsw", "fsw [Hz]", 500, 20000, 5000, 100),
              ("shift", "przesunięcie nośnej RC [okres]", 0, 1, 0, 0.01)]

    def extra_controls(self, parent):
        fr = ttk.Frame(parent)
        fr.pack(fill=tk.X, padx=6)
        ttk.Button(fr, text="Przypadek 1 (0,5 / 0,3)", command=lambda: self.case(0.5, 0.3)).pack(side=tk.LEFT)
        ttk.Button(fr, text="Przypadek 2 (-0,5 / 0,3)", command=lambda: self.case(-0.5, 0.3)).pack(side=tk.LEFT)

    def case(self, d1, d2):
        self.vars["d1"].set(d1)
        self.vars["d2"].set(d2)
        self.refresh()

    def update_plot(self):
        V1, V2, d1, d2, R, L, fsw, sh = (self.p(k) for k in
                                         ["Vdc1", "Vdc2", "d1", "d2", "R", "L", "fsw", "shift"])
        r = pc.dual_dcdc(V1, V2, d1, d2, fsw, R, L * 1e-3, sh, n_show=4)
        t = r["t"] * 1e3
        ax1 = self.fig.add_subplot(4, 1, 1)
        ax1.plot(t, r["v1"], color=COL["a"], lw=1.2, label="v_brg1 (LC)")
        ax1.plot(t, r["v2"], color=COL["c"], lw=1.2, ls="--", label="v_brg2 (RC)")
        wave_ax(ax1, "Napięcia mostków (rys. 4.13 / 4.16)", "V")
        ax1.legend(fontsize=7, loc="upper right")
        ax2 = self.fig.add_subplot(4, 1, 2, sharex=ax1)
        ax2.plot(t, r["v1"] - r["v2"], color=COL["p"], lw=1)
        wave_ax(ax2, "Napięcie na gałęzi L-R:  v1 - v2", "V")
        ax3 = self.fig.add_subplot(4, 1, 3, sharex=ax1)
        ax3.plot(t, r["i"], color=COL["a"])
        ax3.axhline(np.mean(r["i"]), color="k", ls="--", lw=1)
        wave_ax(ax3, "Prąd między przekształtnikami (rys. 4.14 / 4.17)", "A")
        ax4 = self.fig.add_subplot(4, 1, 4, sharex=ax1)
        ax4.step(t, r["S1"] + 1.5, where="post", color=COL["a"], label="S1 (LC)")
        ax4.step(t, r["S3"] + 0.0, where="post", color=COL["c"], label="S3 (LC)")
        ax4.set_yticks([0, 1, 1.5, 2.5])
        ax4.set_yticklabels(["S3 off", "S3 on", "S1 off", "S1 on"])
        wave_ax(ax4, "PWM przekształtnika LC (rys. 4.15 / 4.18)", "")
        I = float(np.mean(r["i"]))
        dv = np.round(np.unique(np.round(r["v1"] - r["v2"], 6)), 3)
        return [f"V1 śr = {np.mean(r['v1']):.2f} V,  V2 śr = {np.mean(r['v2']):.2f} V",
                f"I śr = {I:.3f} A  (teoria {(d1 * V1 - d2 * V2) / R:.3f} A)",
                f"tętnienie p-p = {np.ptp(r['i']):.3f} A",
                f"kierunek mocy: {'LC -> RC' if I > 0 else 'RC -> LC'}",
                f"moc RC: {np.mean(r['v2'] * r['i']):.2f} W,  straty R: {np.mean(r['i'] ** 2) * R:.2f} W",
                f"poziomy napięcia na L-R: {', '.join(f'{v:g}' for v in dv)} V",
                "(przypadek 2: trzy poziomy -> trzy nachylenia di/dt)"]


class TabH6(ExampleTab):
    key = "H6"
    PARAMS = TabH2.PARAMS

    def update_plot(self):
        Vdc, R, L, ma, fsw, f1 = (self.p(k) for k in ["Vdc", "R", "L", "ma", "fsw", "f1"])
        L *= 1e-3
        u = pc.hbridge_unipolar_ac(Vdc, R, L, f1, fsw, ma)
        b = pc.hbridge_bipolar(Vdc, R, L, f1, fsw, ma)
        t = u["t"] * 1e3
        ax1 = self.fig.add_subplot(4, 1, 1)
        ax1.plot(t, u["c1"], color=COL["g"], lw=0.6, label="c1 (0..1)")
        ax1.plot(t, u["c2"], color=COL["p"], lw=0.6, label="c2 = -c1")
        ax1.plot(t, u["m"], color=COL["a"], lw=1.5, label="m(t)")
        wave_ax(ax1, "Unipolarne PWM - dwie nośne (rys. 4.19)", "")
        ax1.legend(fontsize=7, loc="upper right", ncol=3)
        ax2 = self.fig.add_subplot(4, 1, 2, sharex=ax1)
        ax2.plot(t, u["vo"], color=COL["c"], lw=0.6)
        wave_ax(ax2, "Napięcie mostka: 3 poziomy +Vdc / 0 / -Vdc (rys. 4.21)", "V")
        ax3 = self.fig.add_subplot(4, 1, 3, sharex=ax1)
        ax3.plot(t, b["iL"], color=COL["g"], lw=0.8, label="bipolarne")
        ax3.plot(t, u["iL"], color=COL["a"], lw=1, label="unipolarne")
        wave_ax(ax3, "Prąd iL: porównanie (rys. 4.20 vs 4.5)", "A")
        ax3.legend(fontsize=7, loc="upper right")
        ax4 = self.fig.add_subplot(4, 1, 4)
        hb, ab = pc.spectrum(b["vo"], 1)
        hu, au = pc.spectrum(u["vo"], 1)
        mask = hb <= 3 * fsw / f1 + 5
        ax4.bar(hb[mask] - 0.25, ab[mask], width=0.5, color=COL["g"], label="bipolarne")
        ax4.bar(hu[mask] + 0.25, au[mask], width=0.5, color=COL["a"], label="unipolarne")
        wave_ax(ax4, "Widmo napięcia - unipolarne przesuwa harmoniczne do 2·mf", "V", "rząd harmonicznej")
        ax4.legend(fontsize=7)
        rb = np.ptp(b["iL"] - pc.fundamental(b["iL"], 1) * np.sin(2 * PI * f1 * b["t"] - math.atan2(2 * PI * f1 * L, R)))
        ru = np.ptp(u["iL"] - pc.fundamental(u["iL"], 1) * np.sin(2 * PI * f1 * u["t"] - math.atan2(2 * PI * f1 * L, R)))
        return [f"V1 unipolarne = {pc.fundamental(u['vo'], 1):.3f} V",
                f"V1 bipolarne  = {pc.fundamental(b['vo'], 1):.3f} V",
                f"THD napięcia: bip. {pc.thd_from_spectrum(hb, ab):.1f} %, unip. {pc.thd_from_spectrum(hu, au):.1f} %",
                f"THD prądu:   bip. {pc.thd_from_spectrum(*pc.spectrum(b['iL'], 1)):.2f} %, "
                f"unip. {pc.thd_from_spectrum(*pc.spectrum(u['iL'], 1)):.2f} %",
                f"tętnienie prądu p-p: bip. {rb:.3f} A, unip. {ru:.3f} A",
                f"stosunek tętnień ≈ {rb / max(ru, 1e-9):.2f}×",
                f"skok napięcia przy łączeniu: bip. {2 * Vdc:.0f} V, unip. {Vdc:.0f} V"]


class TabH7(ExampleTab):
    key = "H7"
    PARAMS = [("Vdc", "Vdc [V]", 6, 100, 24, 1),
              ("d", "d - przesunięcie (0..0,5)", 0, 0.5, 0.2, 0.01),
              ("fsw", "fsw [Hz]", 1000, 100000, 20000, 500),
              ("R", "R [Ω] (zastępcze obciążenie)", 0.5, 20, 5, 0.5),
              ("L", "L [µH] (rozproszenie)", 5, 2000, 100, 5)]

    def update_plot(self):
        Vdc, d, fsw, R, L = (self.p(k) for k in ["Vdc", "d", "fsw", "R", "L"])
        r = pc.phase_shift_pwm(Vdc, d, fsw, R, L * 1e-6, n_show=3)
        t = r["t"] * 1e6
        ax1 = self.fig.add_subplot(4, 1, 1)
        ax1.plot(t, r["c1"], color=COL["g"], label="c1 (piła)")
        ax1.plot(t, r["c2"], color=COL["p"], label="c2 (piła przesunięta o d·T, ujemna)")
        ax1.axhline(0.5, color=COL["a"], ls="--", lw=1)
        ax1.axhline(-0.5, color=COL["c"], ls="--", lw=1)
        wave_ax(ax1, "Nośne przesunięte fazowo (rys. 4.23)", "", "t [µs]")
        ax1.legend(fontsize=7, loc="upper right")
        ax2 = self.fig.add_subplot(4, 1, 2, sharex=ax1)
        ax2.step(t, r["S1"] + 1.5, where="post", color=COL["a"], label="S1")
        ax2.step(t, r["S3"] + 0.0, where="post", color=COL["c"], label="S3")
        ax2.set_yticks([0, 1, 1.5, 2.5])
        ax2.set_yticklabels(["S3 off", "S3 on", "S1 off", "S1 on"])
        wave_ax(ax2, "Impulsy - każda gałąź 50 %", "", "t [µs]")
        ax3 = self.fig.add_subplot(4, 1, 3, sharex=ax1)
        ax3.plot(t, r["vo"], color=COL["c"], lw=1.5)
        wave_ax(ax3, "Napięcie mostka (rys. 4.24): +Vdc, 0, -Vdc, 0", "V", "t [µs]")
        ax4 = self.fig.add_subplot(4, 2, 7)
        ax4.plot(t, r["iL"], color=COL["a"])
        wave_ax(ax4, "Prąd (średnia 0)", "A", "t [µs]")
        ax5 = self.fig.add_subplot(4, 2, 8)
        dd = np.linspace(0, 0.5, 101)
        ax5.plot(dd, Vdc * np.sqrt(2 * dd), label="Vrms")
        ax5.plot(dd, 4 / PI * Vdc * np.sin(PI * dd) / math.sqrt(2), label="V1 rms")
        ax5.axvline(d, color="k", ls="--")
        wave_ax(ax5, "Napięcie w funkcji d", "V", "d")
        ax5.legend(fontsize=7)
        n = int(len(r["vo"]) / 3)
        V1 = pc.fundamental(r["vo"][:n], 1)
        return [f"średnia vo = {np.mean(r['vo']):.4f} V  (zawsze 0 ✓)",
                f"Vrms = {pc.rms(r['vo']):.3f} V  (teoria Vdc√(2d) = {Vdc * math.sqrt(2 * d):.3f} V)",
                f"V1 (fsw) = {V1:.3f} V  (teoria 4/π·Vdc·sin(πd) = {4 / PI * Vdc * math.sin(PI * d):.3f} V)",
                f"średnia prądu = {np.mean(r['iL']):.4f} A",
                f"impuls +Vdc trwa d·T = {d / fsw * 1e6:.2f} µs"]


# =============================================================================
# ROZDZIAŁ 5
# =============================================================================

class TabT1(ExampleTab):
    key = "T1"
    PARAMS = [("Vm", "Amplituda Vm [V]", 10, 400, 325, 5),
              ("f", "f [Hz]", 10, 100, 50, 1),
              ("wt", "kąt ωt [°]", 0, 360, 30, 1),
              ("kc", "niesymetria: amplituda fazy c [%]", 0, 150, 100, 1),
              ("dphi", "niesymetria: przesunięcie fazy b [°]", -60, 60, 0, 1)]

    def update_plot(self):
        Vm, f, wt, kc, dphi = (self.p(k) for k in ["Vm", "f", "wt", "kc", "dphi"])
        th = np.radians(np.linspace(0, 720, 1441))
        amps = np.array([Vm, Vm, Vm * kc / 100])
        phs = np.radians([0, -120 + dphi, 120])
        v = amps[:, None] * np.cos(th[None, :] + phs[:, None])
        ax1 = self.fig.add_subplot(2, 2, 1)
        for k in range(3):
            ax1.plot(np.degrees(th), v[k], color=PH_COL[k], label="vabc"[k + 1] if False else ["va", "vb", "vc"][k])
        ax1.plot(np.degrees(th), v.sum(axis=0), "k--", lw=1, label="va+vb+vc")
        ax1.axvline(wt, color="k", lw=0.8)
        wave_ax(ax1, "Napięcia fazowe (rys. 5.3)", "V", "ωt [°]")
        ax1.legend(fontsize=7, ncol=4)
        ax2 = self.fig.add_subplot(2, 2, 2)
        vll = np.array([v[0] - v[1], v[1] - v[2], v[2] - v[0]])
        for k, nm in enumerate(["vab", "vbc", "vca"]):
            ax2.plot(np.degrees(th), vll[k], color=PH_COL[k], label=nm)
        ax2.axvline(wt, color="k", lw=0.8)
        wave_ax(ax2, "Napięcia międzyprzewodowe (√3 razy większe, +30°)", "V", "ωt [°]")
        ax2.legend(fontsize=7, ncol=3)
        ax3 = self.fig.add_subplot(2, 2, 3)
        a = math.radians(wt)
        lim = max(amps.max(), Vm) * 1.9
        for k in range(3):
            z = amps[k] * np.exp(1j * (a + phs[k]))
            ax3.annotate("", xy=(z.real, z.imag), xytext=(0, 0),
                         arrowprops=dict(arrowstyle="-|>", color=PH_COL[k], lw=2))
            ax3.plot([z.real, z.real], [z.imag, 0], ":", color=PH_COL[k])
            ax3.plot(z.real, 0, "o", color=PH_COL[k])
            ax3.text(z.real * 1.08, z.imag * 1.08, ["Va", "Vb", "Vc"][k], color=PH_COL[k])
        zs = [amps[k] * np.exp(1j * (a + phs[k])) for k in range(3)]
        for (i1, i2), nm in zip([(0, 1), (1, 2), (2, 0)], ["Vab", "Vbc", "Vca"]):
            z = zs[i1] - zs[i2]
            ax3.annotate("", xy=(z.real, z.imag), xytext=(0, 0),
                         arrowprops=dict(arrowstyle="-|>", color=PH_COL[i1], lw=1, ls="--"))
            ax3.text(z.real * 1.03, z.imag * 1.03, nm, color=PH_COL[i1], fontsize=7)
        ax3.add_patch(Circle((0, 0), Vm, fill=False, ls=":", color="k"))
        ax3.set_xlim(-lim, lim)
        ax3.set_ylim(-lim, lim)
        ax3.set_aspect("equal")
        ax3.grid(True, alpha=0.3)
        ax3.set_title("Wektory wirujące (rzut na oś poziomą = wartość chwilowa)", fontsize=9)
        ax4 = self.fig.add_subplot(2, 2, 4)
        ax4.axis("off")
        vsum = sum(zs)
        V_ph = Vm / math.sqrt(2)
        txt = ("Połączenia (rys. 5.2):\n"
               f"  GWIAZDA:  V_LL = √3·V_ph = {SQ3 * V_ph:.1f} V rms, I_L = I_ph\n"
               f"  TRÓJKĄT:  V_ph = V_LL,  I_L = √3·I_ph\n"
               f"  GWIAZDA + N:  prąd neutralny ~ |Va+Vb+Vc|\n\n"
               f"Chwila ωt = {wt:.0f}°:\n"
               f"  va = {v[0][int(wt * 2)]:.1f} V, vb = {v[1][int(wt * 2)]:.1f} V, vc = {v[2][int(wt * 2)]:.1f} V\n"
               f"  suma = {v[:, int(wt * 2)].sum():.2f} V\n\n"
               f"|Va + Vb + Vc| (amplituda) = {abs(vsum):.1f} V\n"
               + ("  -> układ SYMETRYCZNY (suma = 0)" if abs(vsum) < 1e-6 else
                  "  -> NIESYMETRIA: w gwieździe bez N punkt\n     neutralny przesuwa się o (Va+Vb+Vc)/3"))
        ax4.text(0, 1, txt, va="top", fontsize=9, family="monospace")
        return [f"V_ph rms = {V_ph:.1f} V,  V_LL rms = {SQ3 * V_ph:.1f} V",
                f"ω = 2πf = {2 * PI * f:.1f} rad/s,  T = {1000 / f:.2f} ms",
                f"przesunięcie napięcia punktu neutralnego (amplituda) = {abs(vsum) / 3:.2f} V"]


class TabT2(ExampleTab):
    key = "T2"
    PARAMS = [("Vdc", "Vdc [V]", 6, 100, 24, 1),
              ("R", "R [Ω]", 0.5, 30, 10, 0.5),
              ("L", "L [mH]", 0.1, 20, 1, 0.1),
              ("ma", "ma", 0.05, 1.3, 0.8, 0.01),
              ("fsw", "fsw [Hz]", 500, 10000, 2500, 50),
              ("f1", "f1 [Hz]", 10, 100, 50, 1)]

    def update_plot(self):
        Vdc, R, L, ma, fsw, f1 = (self.p(k) for k in ["Vdc", "R", "L", "ma", "fsw", "f1"])
        r = pc.three_phase_sim(Vdc, R, L * 1e-3, f1, fsw, ma * Vdc / 2, mode="spwm", pps=120)
        t = r["t"] * 1e3
        ax1 = self.fig.add_subplot(4, 1, 1)
        for k in range(3):
            ax1.plot(t, r["i"][k], color=PH_COL[k], label=f"i{'abc'[k]}")
        wave_ax(ax1, "Prądy obciążenia (rys. 5.6)", "A")
        ax1.legend(fontsize=7, ncol=3, loc="upper right")
        ax2 = self.fig.add_subplot(4, 1, 2, sharex=ax1)
        for k in range(3):
            ax2.step(t, r["S"][k] * 0.8 + 2 - k, where="post", color=PH_COL[k])
        ax2.set_yticks([0.4, 1.4, 2.4])
        ax2.set_yticklabels(["S5 (c)", "S3 (b)", "S1 (a)"])
        wave_ax(ax2, "PWM gałęzi (rys. 5.7)", "")
        ax3 = self.fig.add_subplot(4, 1, 3, sharex=ax1)
        ax3.plot(t, r["vph"][0], color=COL["a"], lw=0.7, label="v_an")
        ax3.plot(t, r["vnN"], color=COL["g"], lw=0.7, label="v_nN")
        wave_ax(ax3, "Napięcie fazowe obciążenia - 5 poziomów (rys. 5.8)", "V")
        ax3.legend(fontsize=7, loc="upper right")
        ax4 = self.fig.add_subplot(4, 2, 7)
        h, amp = pc.spectrum(r["vph"][0], 1)
        mask = h <= 2.5 * r["fsw"] / f1
        ax4.bar(h[mask], amp[mask], width=0.8, color=COL["c"])
        wave_ax(ax4, "Widmo v_an", "V", "rząd")
        ax5 = self.fig.add_subplot(4, 2, 8)
        al, be, _ = pc.clarke(*r["i"])
        ax5.plot(al, be, color=COL["p"], lw=0.8)
        ax5.set_aspect("equal")
        wave_ax(ax5, "Prąd na płaszczyźnie α-β", "iβ", "iα")
        V1 = pc.fundamental(r["vph"][0], 1)
        Z = math.hypot(R, 2 * PI * f1 * L * 1e-3)
        return [f"V1 fazowe = {V1:.3f} V  (teoria ma·Vdc/2 = {min(ma, 1) * Vdc / 2:.3f} V)",
                f"I1 = {pc.fundamental(r['i'][0], 1):.3f} A  (teoria {min(ma, 1) * Vdc / 2 / Z:.3f} A)",
                f"V_LL rms = {V1 * SQ3 / math.sqrt(2):.2f} V",
                f"THD v_an = {pc.thd_from_spectrum(h, amp):.1f} %",
                f"THD i_a = {pc.thd_from_spectrum(*pc.spectrum(r['i'][0], 1)):.2f} %",
                f"poziomy v_an: {sorted(set(np.round(r['vph'][0] / Vdc * 3).astype(int).tolist()))} ·Vdc/3",
                f"suma prądów (max) = {np.max(np.abs(r['i'].sum(axis=0))):.2e} A",
                "ma > 1 -> przemodulowanie (zobacz THD)" if ma > 1 else ""]


class TabT3(ExampleTab):
    key = "T3"
    PARAMS = [("Vm", "Amplituda Vm", 0.1, 1.0, 0.3, 0.01),
              ("wt", "kąt ωt [°] (znacznik)", 0, 360, 30, 1),
              ("kb", "amplituda fazy b [%]", 0, 150, 100, 1),
              ("dphi", "błąd kąta fazy c [°]", -60, 60, 0, 1),
              ("h5", "5. harmoniczna [%]", 0, 40, 0, 1),
              ("off", "składowa wspólna (offset)", -0.5, 0.5, 0, 0.01)]

    def update_plot(self):
        Vm, wt, kb, dphi, h5, off = (self.p(k) for k in ["Vm", "wt", "kb", "dphi", "h5", "off"])
        th = np.radians(np.linspace(0, 360, 721))
        ph = np.radians([0, -120, 120 + dphi])
        amp = np.array([Vm, Vm * kb / 100, Vm])
        v = amp[:, None] * np.cos(th[None, :] + ph[:, None])
        v += (h5 / 100) * amp[:, None] * np.cos(5 * (th[None, :] + ph[:, None]))
        v += off
        al, be, z = pc.clarke(*v)
        ax1 = self.fig.add_subplot(2, 2, 1)
        for k in range(3):
            ax1.plot(np.degrees(th), v[k], color=PH_COL[k], label=["va", "vb", "vc"][k])
        ax1.axvline(wt, color="k", lw=0.8)
        wave_ax(ax1, "Wejście: trzy fazy", "", "ωt [°]")
        ax1.legend(fontsize=7, ncol=3)
        ax2 = self.fig.add_subplot(2, 2, 2)
        ax2.plot(np.degrees(th), al, color=COL["o"], label="vα")
        ax2.plot(np.degrees(th), be, color=COL["p"], label="vβ")
        ax2.plot(np.degrees(th), z, "k--", label="v0")
        ax2.axvline(wt, color="k", lw=0.8)
        wave_ax(ax2, "Wyjście Clarke'a: α, β (90° od siebie), 0", "", "ωt [°]")
        ax2.legend(fontsize=7, ncol=3)
        ax3 = self.fig.add_subplot(2, 2, 3)
        ax3.plot(al, be, color=COL["c"], lw=1.5)
        k = int(round(wt * 2)) % 721
        ax3.annotate("", xy=(al[k], be[k]), xytext=(0, 0),
                     arrowprops=dict(arrowstyle="-|>", color=COL["a"], lw=2))
        ax3.text(al[k], be[k], f"  V = {al[k]:.3f} {be[k]:+.3f}j", fontsize=8)
        ax3.set_aspect("equal")
        lim = max(0.1, np.max(np.abs(np.r_[al, be]))) * 1.3
        ax3.set_xlim(-lim, lim)
        ax3.set_ylim(-lim, lim)
        wave_ax(ax3, "Płaszczyzna zespolona V = vα + j vβ (rys. 5.10)", "β", "α")
        ax4 = self.fig.add_subplot(2, 2, 4)
        ax4.axis("off")
        # składowe symetryczne (zgodna / przeciwna) z 1. harmonicznej V
        V = al + 1j * be
        Vp = np.mean(V * np.exp(-1j * th))
        Vn = np.mean(V * np.exp(1j * th))
        txt = ("Transformacja (amplitudowo niezmiennicza):\n"
               "  vα = 2/3 (va - vb/2 - vc/2)\n"
               "  vβ = (vb - vc)/√3\n"
               "  v0 = (va + vb + vc)/3\n\n"
               f"Dla ωt = {wt:.0f}°:\n  vα = {al[k]:.4f},  vβ = {be[k]:.4f},  v0 = {z[k]:.4f}\n\n"
               f"Składowa zgodna |V+| = {abs(Vp):.4f}  (wiruje w lewo)\n"
               f"Składowa przeciwna |V-| = {abs(Vn):.4f}  (wiruje w prawo)\n"
               f"-> trajektoria: {'OKRĄG' if abs(Vn) < 1e-4 and h5 == 0 else 'elipsa / kształt złożony'}\n"
               f"offset {off:+.2f} trafia TYLKO do v0")
        ax4.text(0, 1, txt, va="top", family="monospace", fontsize=9)
        return [f"Przykład z książki: Vm = 0,3, ωt = 30° -> 0,26 + j0,15",
                f"bieżące: V = {al[k]:.4f} {be[k]:+.4f}j,  |V| = {abs(V[k]):.4f}",
                f"|V+| = {abs(Vp):.4f}, |V-| = {abs(Vn):.4f}",
                f"max |v0| = {np.max(np.abs(z)):.4f}"]


class TabT4(ExampleTab):
    key = "T4"
    PARAMS = [("Vdc", "Vdc [V] (1 = jednostki względne)", 1, 600, 1, 1)]

    def extra_controls(self, parent):
        self.add_combo(parent, "state", "Wybrany stan (a b c):",
                       [f"V{k} = {pc.VEC2[k]}" for k in range(8)], "V1 = (1, 0, 0)")

    def update_plot(self):
        Vdc = self.p("Vdc")
        rows = pc.two_level_table(Vdc)
        sel = int(self.o("state").split()[0][1:])
        ax1 = self.fig.add_subplot(2, 2, 1)
        ax2 = self.fig.add_subplot(2, 2, 2)
        for ax, keyv, title in [(ax1, "V", "Napięcia fazowe (rys. 5.12): |V| = 2/3·Vdc"),
                                (ax2, "VLL", "Napięcia międzyprzewodowe (rys. 5.13): |V| = 2/√3·Vdc")]:
            pts = [r[keyv] for r in rows if r["k"] in (1, 6, 2, 4, 3, 5)]
            order = [1, 6, 2, 4, 3, 5, 1]
            hexv = [rows[k][keyv] for k in order]
            ax.plot([z.real for z in hexv], [z.imag for z in hexv], "k--", lw=0.8)
            for r in rows:
                z = r[keyv]
                hl = r["k"] == sel
                if r["k"] in (0, 7):
                    continue
                ax.annotate("", xy=(z.real, z.imag), xytext=(0, 0),
                            arrowprops=dict(arrowstyle="-|>", color=COL["a"] if hl else COL["c"],
                                            lw=2.5 if hl else 1.2))
                ax.text(z.real * 1.12, z.imag * 1.12, f"V{r['k']}\n{''.join(map(str, r['s']))}",
                        ha="center", va="center", fontsize=8, color=COL["a"] if hl else "k")
            ax.plot(0, 0, "o", ms=10 if sel in (0, 7) else 6, color=COL["a"] if sel in (0, 7) else "k")
            ax.text(0.05 * Vdc, -0.12 * Vdc, "V0, V7", fontsize=8)
            m = max(abs(z) for z in pts) * 1.35
            ax.set_xlim(-m, m)
            ax.set_ylim(-m, m)
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)
            ax.set_title(title, fontsize=9)
        ax3 = self.fig.add_subplot(2, 2, 3)
        r = rows[sel]
        ax3.bar(["van", "vbn", "vcn", "vab", "vbc", "vca"],
                [r["van"], r["vbn"], r["vcn"], r["vab"], r["vbc"], r["vca"]],
                color=PH_COL * 2)
        ax3.axhline(0, color="k", lw=0.8)
        wave_ax(ax3, f"Stan V{sel} = {r['s']}: napięcia", "V", "")
        ax4 = self.fig.add_subplot(2, 2, 4)
        ax4.axis("off")
        cell = [[f"V{q['k']}", "".join(map(str, q['s'])), f"{q['van'] / Vdc:+.3f}", f"{q['vbn'] / Vdc:+.3f}",
                 f"{q['vcn'] / Vdc:+.3f}", f"{q['vab'] / Vdc:+.0f}", f"{q['vbc'] / Vdc:+.0f}", f"{q['vca'] / Vdc:+.0f}",
                 f"{q['V'].real / Vdc:+.3f}{q['V'].imag / Vdc:+.3f}j"] for q in rows]
        tb = ax4.table(cellText=cell, colLabels=["V", "abc", "van", "vbn", "vcn", "vab", "vbc", "vca", "vα+jvβ"],
                       loc="center", cellLoc="center")
        tb.auto_set_font_size(False)
        tb.set_fontsize(7)
        tb.scale(1, 1.2)
        for c in range(9):
            tb[(sel + 1, c)].set_facecolor("#9fc5ff")
        ax4.set_title("Tabele 5.1 / 5.2 (wartości ·Vdc)", fontsize=9)
        return [f"Stan V{sel} = (a,b,c) = {r['s']}",
                f"v_nN = {Vdc * sum(r['s']) / 3:.3f} V",
                f"van = {r['van']:.3f}, vbn = {r['vbn']:.3f}, vcn = {r['vcn']:.3f} V",
                f"vab = {r['vab']:.3f}, vbc = {r['vbc']:.3f}, vca = {r['vca']:.3f} V",
                f"V = {r['V'].real:.4f} {r['V'].imag:+.4f}j, |V| = {abs(r['V']):.4f}, "
                f"∠ = {math.degrees(math.atan2(r['V'].imag, r['V'].real)) if abs(r['V']) > 0 else 0:.0f}°",
                f"V_LL: |V| = {abs(r['VLL']):.4f}"]


class TabT5(ExampleTab):
    key = "T5"
    PARAMS = [("Vr", "|Vr| / Vdc", 0, 0.7, 0.3, 0.005),
              ("ang", "kąt wektora zadanego [°]", 0, 359, 30, 1),
              ("T", "okres przełączania T [µs]", 50, 1000, 400, 10)]

    def extra_controls(self, parent):
        fr = ttk.Frame(parent)
        fr.pack(fill=tk.X, padx=6)
        ttk.Button(fr, text="Książka: 0,3∠30°", command=lambda: self.ex(0.3, 30)).pack(side=tk.LEFT)
        ttk.Button(fr, text="Granica: 2/3∠30°", command=lambda: self.ex(0.6667, 30)).pack(side=tk.LEFT)
        ttk.Button(fr, text="Na V6: 0,5∠60°", command=lambda: self.ex(0.5, 60)).pack(side=tk.LEFT)

    def ex(self, m, a):
        self.vars["Vr"].set(m)
        self.vars["ang"].set(a)
        self.refresh()

    def update_plot(self):
        Vm, ang, T = self.p("Vr"), self.p("ang"), self.p("T")
        Vr = Vm * np.exp(1j * math.radians(ang))
        r = pc.svpwm_times(Vr, T)
        ax = self.fig.add_subplot(1, 2, 1)
        order = [1, 6, 2, 4, 3, 5, 1]
        hexv = [pc.vec2(k) for k in order]
        ax.fill([z.real for z in hexv], [z.imag for z in hexv], color="#eef4ff")
        ax.plot([z.real for z in hexv], [z.imag for z in hexv], "k-", lw=1)
        for k in range(1, 7):
            z = pc.vec2(k)
            hl = k in (r["x"], r["y"])
            ax.annotate("", xy=(z.real, z.imag), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="-|>", color=COL["c"] if hl else COL["g"], lw=2 if hl else 1))
            ax.text(z.real * 1.13, z.imag * 1.13, f"V{k}\n{''.join(map(str, pc.VEC2[k]))}",
                    ha="center", va="center", fontsize=8, fontweight="bold" if hl else "normal")
        for rad, ls, lab in [(1 / SQ3, "-", "0,577 (granica liniowa)"), (2 / 3, ":", "2/3 (wierzchołki)"),
                             (0.5, "--", "0,5 (granica SPWM)")]:
            ax.add_patch(Circle((0, 0), rad, fill=False, ls=ls, color="k", lw=0.8))
            ax.text(rad * 0.71, -rad * 0.71, lab, fontsize=6)
        ax.add_patch(Circle((0, 0), Vm, fill=False, color=COL["a"], ls="--", lw=1))
        Vx, Vy = pc.vec2(r["x"]), pc.vec2(r["y"])
        p1 = Vx * r["t1"] / T
        p2 = p1 + Vy * r["t2"] / T
        ax.annotate("", xy=(p1.real, p1.imag), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=COL["o"], lw=2))
        ax.annotate("", xy=(p2.real, p2.imag), xytext=(p1.real, p1.imag),
                    arrowprops=dict(arrowstyle="-|>", color=COL["b"], lw=2))
        ax.annotate("", xy=(Vr.real, Vr.imag), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=COL["a"], lw=3))
        ax.text(Vr.real, Vr.imag + 0.04, "Vr", color=COL["a"], fontweight="bold")
        ax.set_xlim(-0.85, 0.85)
        ax.set_ylim(-0.85, 0.85)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_title(f"Sektor {r['sector']}: Vr = (t1/T)·V{r['x']} + (t2/T)·V{r['y']}", fontsize=10)
        # sekwencja przełączeń
        seq = [(0, r["t0"] / 4), (r["x"], r["t1"] / 2), (r["y"], r["t2"] / 2), (7, r["t0"] / 4),
               (7, r["t0"] / 4), (r["y"], r["t2"] / 2), (r["x"], r["t1"] / 2), (0, r["t0"] / 4)]
        ax2 = self.fig.add_subplot(2, 2, 2)
        tcur = 0.0
        for vk, dur in seq:
            s = pc.VEC2[vk]
            for leg in range(3):
                ax2.fill_between([tcur, tcur + dur], 2 - leg, 2 - leg + 0.8 * s[leg],
                                 color=PH_COL[leg], step="pre", alpha=0.8)
            ax2.axvline(tcur, color="k", lw=0.4)
            if dur > T * 0.03:
                ax2.text(tcur + dur / 2, 3.0, f"V{vk}", ha="center", fontsize=8)
            tcur += dur
        ax2.set_ylim(-0.2, 3.3)
        ax2.set_xlim(0, T)
        ax2.set_yticks([0.4, 1.4, 2.4])
        ax2.set_yticklabels(["c", "b", "a"])
        wave_ax(ax2, "Sekwencja w okresie T (Tabela 5.3)", "gałąź", "t [µs]")
        ax3 = self.fig.add_subplot(2, 2, 4)
        ax3.barh(["t0", "t2", "t1"], [r["t0"], r["t2"], r["t1"]], color=[COL["g"], COL["b"], COL["o"]])
        if r["over"]:
            ax3.barh(["t1+t2 (żądane)"], [sum(r["raw"])], color="red")
        ax3.axvline(T, color="k", ls="--")
        wave_ax(ax3, "Czasy wektorów", "", "µs")
        lines = [f"Vr = {Vr.real:.4f} {Vr.imag:+.4f}j  (|Vr| = {Vm:.3f}·Vdc)",
                 f"para wektorów: (V{r['x']}, V{r['y']}), sektor {r['sector']}",
                 f"t1 = {r['t1']:.2f} µs ({r['t1'] / T:.4f} T)",
                 f"t2 = {r['t2']:.2f} µs ({r['t2'] / T:.4f} T)",
                 f"t0 = {r['t0']:.2f} µs ({r['t0'] / T:.4f} T)",
                 "sekwencja: " + " - ".join(f"V{v}" for v, _ in seq)]
        if r["over"]:
            lines += ["", f"!! PRZEMODULOWANIE: t1+t2 = {sum(r['raw']) / T:.3f}·T > T",
                      "   czasy przeskalowane do T - napięcie zniekształcone"]
        lines += ["", f"wypełnienia gałęzi: d = {np.round(pc.svpwm_leg_duties(Vr)[0], 4)}"]
        return lines


class TabT6(ExampleTab):
    key = "T6"
    PARAMS = [("Vdc", "Vdc [V]", 6, 100, 24, 1),
              ("R", "R [Ω]", 0.5, 30, 10, 0.5),
              ("L", "L [mH]", 0.1, 20, 1, 0.1),
              ("m", "amplituda napięcia fazowego / Vdc", 0.02, 0.7, 0.45, 0.005),
              ("fsw", "fsw [Hz]", 500, 10000, 2500, 50),
              ("f1", "f1 [Hz]", 10, 100, 50, 1)]

    def extra_controls(self, parent):
        self.add_radio(parent, "mode", "Modulacja", [("SVPWM (rozdz. 5.6-5.7)", "svpwm"),
                                                      ("sine-PWM (rozdz. 5.3)", "spwm")], "svpwm")

    def update_plot(self):
        Vdc, R, L, m, fsw, f1 = (self.p(k) for k in ["Vdc", "R", "L", "m", "fsw", "f1"])
        mode = self.o("mode")
        r = pc.three_phase_sim(Vdc, R, L * 1e-3, f1, fsw, m * Vdc, mode=mode, pps=120)
        t = r["t"] * 1e3
        ax1 = self.fig.add_subplot(3, 2, 1)
        for k in range(3):
            ax1.plot(t, r["i"][k], color=PH_COL[k], lw=0.9)
        wave_ax(ax1, "Prądy (rys. 5.16)", "A")
        ax2 = self.fig.add_subplot(3, 2, 3, sharex=ax1)
        for k in range(3):
            ax2.step(t, r["S"][k] * 0.8 + 2 - k, where="post", color=PH_COL[k], lw=0.6)
        ax2.set_yticks([0.4, 1.4, 2.4])
        ax2.set_yticklabels(["c", "b", "a"])
        wave_ax(ax2, "PWM gałęzi (rys. 5.17)", "")
        ax3 = self.fig.add_subplot(3, 2, 5, sharex=ax1)
        ax3.plot(t, r["vph"][0], color=COL["a"], lw=0.6)
        wave_ax(ax3, "Napięcie fazowe v_an (rys. 5.18)", "V")
        ax4 = self.fig.add_subplot(3, 2, 2)
        nsw = r["D"].shape[1] // 3
        D = r["D"][:, -nsw:]
        ang = np.linspace(0, 360, nsw, endpoint=False)
        for k in range(3):
            ax4.plot(ang, D[k], color=PH_COL[k], label=f"d_{'abc'[k]}")
        wave_ax(ax4, "Wypełnienia gałęzi ('siodło' = sinus + składowa zerowa)", "d", "ωt [°]")
        ax4.legend(fontsize=7, ncol=3)
        ax5 = self.fig.add_subplot(3, 2, 4)
        al, be, _ = pc.clarke(*r["i"])
        ax5.plot(al, be, color=COL["p"], lw=0.8)
        ax5.set_aspect("equal")
        wave_ax(ax5, "Trajektoria prądu α-β", "iβ", "iα")
        ax6 = self.fig.add_subplot(3, 2, 6)
        mm = np.linspace(0.02, 0.7, 30)
        res_sv, res_sp = [], []
        for mv in mm:
            res_sv.append(pc.fundamental(pc.three_phase_sim(1, 10, 1e-3, 50, 1500, mv, "svpwm", 1, pps=40)["vph"][0], 1))
            res_sp.append(pc.fundamental(pc.three_phase_sim(1, 10, 1e-3, 50, 1500, mv, "spwm", 1, pps=40)["vph"][0], 1))
        ax6.plot(mm, mm, "k:", label="ideał")
        ax6.plot(mm, res_sv, color=COL["a"], label="SVPWM")
        ax6.plot(mm, res_sp, color=COL["c"], label="SPWM")
        ax6.axvline(0.5, color=COL["c"], ls="--", lw=0.8)
        ax6.axvline(1 / SQ3, color=COL["a"], ls="--", lw=0.8)
        ax6.axvline(m, color="k", lw=1)
        wave_ax(ax6, "Charakterystyka V1 = f(żądane) - granice 0,5 i 0,577", "V1/Vdc", "żądane/Vdc")
        ax6.legend(fontsize=7)
        V1 = pc.fundamental(r["vph"][0], 1)
        h, amp = pc.spectrum(r["vph"][0], 1)
        return [f"tryb: {mode.upper()}",
                f"żądane V1 = {m * Vdc:.3f} V, uzyskane V1 = {V1:.3f} V",
                f"granica liniowa: SPWM {0.5 * Vdc:.2f} V, SVPWM {Vdc / SQ3:.2f} V",
                f"V_ph rms max (SVPWM) = {0.408 * Vdc:.2f} V,  V_LL rms max = {0.707 * Vdc:.2f} V",
                f"THD v_an = {pc.thd_from_spectrum(h, amp):.1f} %",
                f"THD i_a = {pc.thd_from_spectrum(*pc.spectrum(r['i'][0], 1)):.2f} %",
                f"I1 = {pc.fundamental(r['i'][0], 1):.3f} A",
                f"Vdc dla sieci 400 V: ≥ {400 / 0.7071:.0f} V"]


# =============================================================================
# ROZDZIAŁ 6
# =============================================================================

class TabN1(ExampleTab):
    key = "N1"
    PARAMS = []

    def extra_controls(self, parent):
        fr = ttk.LabelFrame(parent, text="Tranzystory gałęzi (ON)")
        fr.pack(fill=tk.X, padx=6, pady=4)
        for nm, v in [("S1", False), ("S2", True), ("S3", True), ("S4", False)]:
            var = tk.BooleanVar(value=v)
            self.opts[nm] = var
            ttk.Checkbutton(fr, text=nm, variable=var, command=self.schedule).pack(side=tk.LEFT, padx=4)
        self.add_radio(parent, "dir", "Kierunek prądu wyjściowego",
                       [("wypływa z zacisku o", "+"), ("wpływa do zacisku o", "-")], "+")
        fr2 = ttk.Frame(parent)
        fr2.pack(fill=tk.X, padx=6)
        for txt, st in [("Vdc", (1, 1, 0, 0)), ("Vdc/2", (0, 1, 1, 0)), ("0", (0, 0, 1, 1))]:
            ttk.Button(fr2, text=txt, width=8, command=lambda s=st: self.set_state(s)).pack(side=tk.LEFT, padx=1)

    def set_state(self, st):
        for nm, v in zip(["S1", "S2", "S3", "S4"], st):
            self.opts[nm].set(bool(v))
        self.refresh()

    def update_plot(self):
        S = [self.o(n) for n in ["S1", "S2", "S3", "S4"]]
        idir = 1 if self.o("dir") == "+" else -1
        res = pc.npc3_leg(S, idir)
        ax = self.fig.add_subplot(1, 2, 1)
        draw_npc_leg(ax, S, res, idir)
        ax.set_title("Gałąź 3-poziomowa NPC (rys. 6.2)", fontsize=10)
        ax2 = self.fig.add_subplot(1, 2, 2)
        ax2.axis("off")
        rows = []
        cur = "".join(str(int(s)) for s in S)
        names = {0: "0", 1: "Vdc/2", 2: "Vdc", None: "ZWARCIE"}
        for k in range(16):
            s = [(k >> (3 - j)) & 1 for j in range(4)]
            rp, rn = pc.npc3_leg(s, 1), pc.npc3_leg(s, -1)
            uw = "zabroniony" if rp["forbidden"] else ("zamierzony" if rp.get("intentional") else "zależy od i")
            rows.append(["".join(map(str, s)), names[rp["level"]], names[rn["level"]], uw])
        tb = ax2.table(cellText=rows, colLabels=["S1S2S3S4", "i wypływa", "i wpływa", "uwagi"],
                       loc="upper center", cellLoc="center")
        tb.auto_set_font_size(False)
        tb.set_fontsize(8)
        tb.scale(1, 1.1)
        for i_r, r in enumerate(rows):
            col = {"zabroniony": "#ffd6d6", "zamierzony": "#d6f5d6"}.get(r[3], "#fff6d6")
            if r[0] == cur:
                col = "#9fc5ff"
            for c in range(4):
                tb[(i_r + 1, c)].set_facecolor(col)
        ax2.set_title("Wszystkie kombinacje 3-poziomowej gałęzi", fontsize=10)
        t4 = ("Tabela 6.2 - gałąź 4-poziomowa (S1..S6):\n"
              "  000111 -> 0        001110 -> Vdc/3\n"
              "  011100 -> 2Vdc/3   111000 -> Vdc\n"
              "Zawsze 3 sąsiednie tranzystory ON;\n"
              ">3 włączone -> zwarcie kondensatora(ów).")
        return [res["msg"], "", "Tabela 6.1 - 3 poziomy:",
                "  S1S2 ON -> Vdc,  S2S3 ON -> Vdc/2,  S3S4 ON -> 0",
                "Każdy tranzystor blokuje tylko Vdc/2.", "", t4]


class TabN2(ExampleTab):
    key = "N2"
    PARAMS = [("idx", "numer wektora (3-poz.: 0..26, 4-poz.: 0..63)", 0, 63, 12, 1)]

    def extra_controls(self, parent):
        self.add_radio(parent, "k", "Liczba poziomów", [("2 poziomy", "2"), ("3 poziomy (rys. 6.5)", "3"),
                                                        ("4 poziomy (rys. 6.6)", "4")], "3")

    def update_plot(self):
        k = int(self.o("k"))
        vecs = pc.multilevel_vectors(k)
        idx = int(min(self.p("idx"), k ** 3 - 1))
        groups = {}
        for n, (s, z) in enumerate(vecs):
            groups.setdefault((round(z.real, 6), round(z.imag, 6)), []).append((n, s))
        ax = self.fig.add_subplot(1, 2, 1)
        sel_z = vecs[idx][1]
        for (x, y), lst in groups.items():
            red = len(lst)
            ax.plot(x, y, "o", ms=4 + 3 * red, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"][min(red, 5) - 1],
                    alpha=0.85)
            if k <= 3:
                ax.text(x, y + 0.03, "/".join(f"V{n}" for n, _ in lst), ha="center", fontsize=6)
        ax.annotate("", xy=(sel_z.real, sel_z.imag), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color="red", lw=2))
        ax.add_patch(Circle((0, 0), 1 / SQ3, fill=False, ls="--", color="k", lw=0.8))
        ax.text(0.42, -0.45, "0,577·Vdc", fontsize=7)
        # siatka trójkątów
        pts = np.array([[z.real, z.imag] for _, z in vecs])
        step = 2 / 3 / (k - 1)
        for (x, y) in {(round(p[0], 6), round(p[1], 6)) for p in pts}:
            for d in [0, 60, 120]:
                e = step * np.exp(1j * math.radians(d))
                xe, ye = x + e.real, y + e.imag
                if np.min(np.hypot(pts[:, 0] - xe, pts[:, 1] - ye)) < 1e-6:
                    ax.plot([x, xe], [y, ye], color="#bbbbbb", lw=0.5, zorder=0)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)
        ax.set_title(f"Mapa wektorów {k}-poziomowego (rozmiar kropki = redundancja)", fontsize=9)
        ax2 = self.fig.add_subplot(2, 2, 2)
        mags = {}
        for (x, y), lst in groups.items():
            mags.setdefault(round(math.hypot(x, y), 4), []).append(len(lst))
        mk = sorted(mags)
        ax2.bar([f"{m:.3f}" for m in mk], [len(mags[m]) for m in mk], color=COL["c"])
        wave_ax(ax2, "Liczba różnych wektorów wg długości", "liczba", "|V|/Vdc")
        ax2.tick_params(axis="x", labelrotation=60, labelsize=7)
        ax3 = self.fig.add_subplot(2, 2, 4)
        ax3.axis("off")
        same = groups[(round(sel_z.real, 6), round(sel_z.imag, 6))]
        txt = (f"Wybrany V{idx} = stan faz (La,Lb,Lc) = {vecs[idx][0]}\n"
               f"V = {sel_z.real:.4f} {sel_z.imag:+.4f}j,  |V| = {abs(sel_z):.4f}·Vdc\n\n"
               f"Realizacje tego samego wektora ({len(same)}):\n" +
               "\n".join(f"   V{n}: {s}  -> poziomy faz {tuple(round(l / (k - 1), 3) for l in s)}·Vdc"
                         for n, s in same))
        ax3.text(0, 1, txt, va="top", family="monospace", fontsize=8)
        return [f"stany: {k}³ = {k ** 3}", f"różnych wektorów: {len(groups)}",
                f"rozkład redundancji: " + ", ".join(f"{r}×: {sum(1 for l in groups.values() if len(l) == r)}"
                                                     for r in sorted({len(l) for l in groups.values()})),
                f"długości: {', '.join(f'{m:.4f}' for m in mk)}",
                "", "numer = La·k² + Lb·k + Lc (Tabela 6.3)"]


class TabN3(ExampleTab):
    key = "N3"
    PARAMS = [("Vr", "|Vr| / Vdc", 0, 0.577, 0.5, 0.005),
              ("ang", "kąt [°]", 0, 359, 40, 1)]

    def extra_controls(self, parent):
        self.add_radio(parent, "seq", "Wybór sekwencji redundantnej",
                       [("dolny kondensator (δ min)", "lo"), ("górny kondensator (δ max)", "hi"),
                        ("środek (symetryczna)", "mid")], "lo")
        fr = ttk.Frame(parent)
        fr.pack(fill=tk.X, padx=6)
        ttk.Button(fr, text="|Vr| = 0,2 (wewn.)", command=lambda: self.ex(0.2, 40)).pack(side=tk.LEFT)
        ttk.Button(fr, text="Trójkąt V9-V12-V21", command=lambda: self.ex(0.30, 20)).pack(side=tk.LEFT)

    def ex(self, m, a):
        self.vars["Vr"].set(m)
        self.vars["ang"].set(a)
        self.refresh()

    def update_plot(self):
        Vm, ang = self.p("Vr"), self.p("ang")
        th = math.radians(ang)
        Vr = Vm * np.exp(1j * th)
        # napięcia faz w jednostkach poziomu (Vdc/2), środek = 1
        va, vb, vc = pc.inv_clarke(Vr.real, Vr.imag)
        u = 1.0 + np.array([va, vb, vc]) / 0.5
        opt = pc.ml_delta_options(u, 3)
        lo, hi = opt["rng"]
        delta = opt[self.o("seq")]
        states, d = pc.ml_sequence(u, delta)
        vecs = dict(pc.multilevel_vectors(3))
        ax = self.fig.add_subplot(1, 2, 1)
        allv = pc.multilevel_vectors(3)
        pts = {}
        for n, (s, z) in enumerate(allv):
            pts.setdefault((round(z.real, 6), round(z.imag, 6)), []).append(n)
        step = 1 / 3
        P = np.array([[x, y] for x, y in pts])
        for (x, y) in pts:
            for dd in [0, 60, 120]:
                e = step * np.exp(1j * math.radians(dd))
                if np.min(np.hypot(P[:, 0] - x - e.real, P[:, 1] - y - e.imag)) < 1e-6:
                    ax.plot([x, x + e.real], [y, y + e.imag], color="#bbbbbb", lw=0.6, zorder=0)
        for (x, y), lst in pts.items():
            ax.plot(x, y, "o", color=COL["c"], ms=4)
            ax.text(x, y + 0.025, "/".join(f"V{n}" for n in lst), fontsize=6, ha="center")
        tri_pts = [vecs[s] for s in states[:3]]
        ax.fill([z.real for z in tri_pts], [z.imag for z in tri_pts], color="#ffd27f", alpha=0.7)
        ax.add_patch(Circle((0, 0), Vm, fill=False, ls="--", color=COL["a"]))
        ax.annotate("", xy=(Vr.real, Vr.imag), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=COL["a"], lw=2.5))
        Vz = tri_pts[0]
        ax.annotate("", xy=(Vr.real, Vr.imag), xytext=(Vz.real, Vz.imag),
                    arrowprops=dict(arrowstyle="-|>", color=COL["p"], lw=1.5, ls="--"))
        ax.text(Vr.real, Vr.imag + 0.05, "Vr", color=COL["a"], fontweight="bold")
        ax.set_aspect("equal")
        ax.set_xlim(-0.75, 0.75)
        ax.set_ylim(-0.7, 0.7)
        ax.grid(True, alpha=0.2)
        ax.set_title("Trójkąt najbliższych wektorów (fiolet: Vr' = Vr - Vz)", fontsize=9)
        # sekwencja poziomów
        ax2 = self.fig.add_subplot(2, 2, 2)
        seq = states + states[::-1]
        dur = [x / 2 for x in d] + [x / 2 for x in d[::-1]]
        tc = 0.0
        for st, du in zip(seq, dur):
            for leg in range(3):
                ax2.fill_between([tc, tc + du], 3 * (2 - leg), 3 * (2 - leg) + st[leg],
                                 color=PH_COL[leg], alpha=0.8, step="pre")
            ax2.axvline(tc, color="k", lw=0.3)
            tc += du
        ax2.set_yticks([0, 1, 2, 3, 4, 5, 6, 7, 8])
        ax2.set_yticklabels(["c:0", "1", "2", "b:0", "1", "2", "a:0", "1", "2"], fontsize=7)
        ax2.set_xlim(0, 1)
        wave_ax(ax2, "Poziomy faz w okresie (symetrycznie)", "poziom", "t/T")
        ax3 = self.fig.add_subplot(2, 2, 4)
        ax3.axis("off")
        names = [f"V{pc.ml_index(s, 3)}{s}" for s in states]
        Vx, Vy = vecs[states[1]], vecs[states[2]]
        Vrp, Vxp, Vyp = Vr - Vz, Vx - Vz, Vy - Vz
        det = Vxp.real * Vyp.imag - Vyp.real * Vxp.imag
        t1 = (Vrp.real * Vyp.imag - Vyp.real * Vrp.imag) / det
        t2 = (Vxp.real * Vrp.imag - Vrp.real * Vxp.imag) / det
        uses_low = sum(1 for s in states for L in s if L == 1)
        txt = ("Sekwencja (1. i 4. = ten sam wektor Vz):\n  " + " - ".join(names) + "\n\n"
               f"czasy (rozkład ułamkowy):  t_z = {d[0] + d[3]:.4f}T (Vz1 {d[0]:.4f} + Vz2 {d[3]:.4f})\n"
               f"                           t1 = {d[1]:.4f}T,  t2 = {d[2]:.4f}T\n"
               f"wzór 5.32 z przesunięciem: t1 = {t1:.4f}T,  t2 = {t2:.4f}T  ✓\n\n"
               f"Vz = {Vz.real:.4f}{Vz.imag:+.4f}j,  Vr' = Vr - Vz = {Vrp.real:.4f}{Vrp.imag:+.4f}j\n"
               f"δ ∈ [{lo:.3f}, {hi:.3f}],  wybrane δ = {delta:.3f}\n"
               f"liczba różnych sekwencji redundantnych w tym trójkącie: {opt['n']}")
        ax3.text(0, 1, txt, va="top", family="monospace", fontsize=8)
        inner = Vm <= 1 / 3 * math.cos(math.radians(30)) + 1e-9
        return [f"Vr = {Vr.real:.4f} {Vr.imag:+.4f}j",
                f"napięcia faz [poziomy]: {np.round(u, 4)}",
                f"stany: {', '.join(names)}",
                f"Vz w środku? {'TAK (trójkąt wewnętrzny)' if abs(Vz) < 1e-9 else 'NIE - zewnętrzny trójkąt'}",
                f"wewnętrzny sześciokąt: {'tak' if inner else 'nie'}",
                f"połączeń faz z punktem M w sekwencji: {uses_low}",
                "Zmień tryb sekwencji: czasy t1, t2 bez zmian,", "zmieniają się tylko realizacje Vz."]


class TabN4(ExampleTab):
    key = "N4"
    PARAMS = [("Vdc", "Vdc [V]", 12, 600, 24, 1),
              ("m", "amplituda napięcia fazowego / Vdc", 0.05, 0.577, 0.42, 0.005),
              ("R", "R [Ω]", 0.5, 30, 10, 0.5),
              ("L", "L [mH]", 0.1, 20, 1, 0.1),
              ("C", "C każdego kondensatora [µF]", 100, 10000, 2200, 100),
              ("band", "pasmo histerezy vc_max [% Vdc]", 0.2, 10, 2, 0.1),
              ("vc0", "początkowa niesymetria vc0 [V]", -10, 10, 2, 0.1),
              ("Kp", "Kp (tryb P)", 0.0, 5, 0.5, 0.05),
              ("fsw", "fsw [Hz]", 500, 10000, 2500, 50),
              ("ncyc", "liczba okresów 50 Hz", 2, 20, 8, 1)]

    def extra_controls(self, parent):
        self.add_radio(parent, "mode", "Balansowanie kondensatorów",
                       [("histereza (książka, rozdz. 6.6)", "hyst"), ("predykcyjne (znak i_M)", "pred"),
                        ("regulator P przesunięcia δ", "pi"), ("brak (zawsze 'dolna')", "none")], "hyst")
        self.add_check(parent, "cmp2", "Porównaj z falownikiem 2-poziomowym", True)

    def update_plot(self):
        Vdc, m, R, L, C, band, vc0, Kp, fsw, ncyc = (self.p(k) for k in
                                                       ["Vdc", "m", "R", "L", "C", "band", "vc0", "Kp", "fsw", "ncyc"])
        r = pc.npc3_sim(Vdc, R, L * 1e-3, C * 1e-6, 50, fsw, m * Vdc, mode=self.o("mode"),
                        band_pct=band, Kp=Kp, vc0=vc0, n_cyc=int(ncyc), pps=80)
        t = r["t"] * 1e3
        ppc = r["ppc"]
        last = slice(len(t) - ppc, len(t))
        ax1 = self.fig.add_subplot(3, 2, 1)
        ax1.plot(t[last], r["vph"][0][last], color=COL["a"], lw=0.6)
        wave_ax(ax1, "Napięcie fazowe v_an (rys. 6.12) - wiele poziomów", "V")
        ax2 = self.fig.add_subplot(3, 2, 3, sharex=ax1)
        for k in range(3):
            ax2.plot(t[last], r["i"][k][last], color=PH_COL[k], lw=0.9)
        wave_ax(ax2, "Prądy (rys. 6.13)", "A")
        ax3 = self.fig.add_subplot(3, 2, 5, sharex=ax1)
        ax3.step(t[last], r["lev"][0][last], where="post", color=COL["a"])
        ax3.set_yticks([0, 1, 2])
        ax3.set_yticklabels(["0", "Vdc/2", "Vdc"])
        wave_ax(ax3, "Poziom gałęzi a (v_aN)", "")
        ax4 = self.fig.add_subplot(3, 2, 2)
        vup = Vdc / 2 + r["vc"] / 2
        vlo = Vdc / 2 - r["vc"] / 2
        ax4.plot(t, vup, color=COL["a"], label="v_góra")
        ax4.plot(t, vlo, color=COL["c"], label="v_dół")
        ax4.axhline(Vdc / 2, color="k", ls=":", lw=0.8)
        wave_ax(ax4, "Napięcia kondensatorów (rys. 6.14)", "V")
        ax4.legend(fontsize=7)
        ax5 = self.fig.add_subplot(3, 2, 4, sharex=ax4)
        ax5.plot(t, r["vc"], color=COL["p"], lw=0.8)
        if self.o("mode") == "hyst":
            ax5.axhline(r["band"], color="r", ls="--", lw=0.8)
            ax5.axhline(-r["band"], color="r", ls="--", lw=0.8)
        wave_ax(ax5, "vc = v_góra - v_dół (pasmo histerezy)", "V")
        ax6 = self.fig.add_subplot(3, 2, 6)
        h3, a3 = pc.spectrum(r["i"][0][last], 1)
        mf = r["fsw"] / 50

        def bands(h, amp):
            edges = [1.5] + [mf * (k + 0.5) for k in range(4)]
            return [math.sqrt(np.sum(amp[(h > lo_) & (h <= hi_)] ** 2 / 2)) for lo_, hi_ in zip(edges[:-1], edges[1:])]
        labels = ["2..mf/2", "≈ mf", "≈ 2mf", "≈ 3mf"]
        xb = np.arange(4)
        ax6.bar(xb + 0.2, bands(h3, a3), width=0.4, color=COL["a"], label="3-poziomowy NPC")
        lines_cmp = []
        if self.opts["cmp2"].get():
            r2 = pc.three_phase_sim(Vdc, R, L * 1e-3, 50, fsw, m * Vdc, "svpwm", 2, pps=80)
            h2, a2 = pc.spectrum(r2["i"][0], 1)
            ax6.bar(xb - 0.2, bands(h2, a2), width=0.4, color=COL["g"], label="2-poziomowy SVPWM")
            lines_cmp = [f"THD i_a 2-poziomowy = {pc.thd_from_spectrum(h2, a2):.2f} %"]
        ax6.set_xticks(xb)
        ax6.set_xticklabels(labels)
        wave_ax(ax6, "Wartość skuteczna harmonicznych prądu w pasmach", "A rms", "pasmo (mf = fsw/f1)")
        ax6.legend(fontsize=7)
        lev_count = len(np.unique(np.round(r["vph"][0][last], 2)))
        return [f"tryb: {self.o('mode')}",
                f"V1 = {pc.fundamental(r['vph'][0][last], 1):.3f} V (żądane {m * Vdc:.3f} V)",
                f"I1 = {pc.fundamental(r['i'][0][last], 1):.3f} A",
                f"THD i_a 3-poziomowy = {pc.thd_from_spectrum(h3, a3):.2f} %"] + lines_cmp + [
                f"liczba różnych wartości v_an ≈ {lev_count}",
                f"vc: start {vc0:.2f} V, ostatni okres {r['vc'][last].min():.3f} .. {r['vc'][last].max():.3f} V",
                f"pasmo histerezy ±{r['band']:.3f} V",
                f"śr. |i_M| = {np.mean(np.abs(r['iM'])):.3f} A,  dvc/dt ≈ {np.mean(np.abs(r['iM'])) / (C * 1e-6):.0f} V/s"]


# =============================================================================
# ZAKŁADKI ANALIZY STEROWANIA
# =============================================================================

class ControlTab(ttk.Frame):
    """Analiza układu regulacji:  Bode, Nyquist, Step (zadana + zakłócenie),
    mapa biegunów i zer, ZPK, przemiatanie dowolnego parametru, opcjonalnie
    symulacja czasowa.  Podklasa definiuje PARAMS, key, build(p) -> dict."""
    PARAMS = []
    key = ""
    FREQ = (1.0, 1e5)          # zakres częstotliwości [Hz]
    TIME_SIM = False

    def __init__(self, master):
        super().__init__(master)
        self.vars = {}
        self._pending = None
        pw = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        pw.pack(fill=tk.BOTH, expand=True)
        left_outer = ttk.Frame(pw, width=340)
        right = ttk.Frame(pw)
        pw.add(left_outer, weight=0)
        pw.add(right, weight=1)
        cnv = tk.Canvas(left_outer, width=330, highlightthickness=0)
        sb = ttk.Scrollbar(left_outer, orient=tk.VERTICAL, command=cnv.yview)
        left = ttk.Frame(cnv)
        left.bind("<Configure>", lambda e: cnv.configure(scrollregion=cnv.bbox("all")))
        cnv.create_window((0, 0), window=left, anchor="nw")
        cnv.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        cnv.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ttk.Label(left, text="Parametry układu", font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=6, pady=(6, 2))
        for name, label, lo, hi, val, res in self.PARAMS:
            fr = ttk.Frame(left)
            fr.pack(fill=tk.X, padx=6)
            var = tk.DoubleVar(value=val)
            self.vars[name] = var
            ttk.Label(fr, text=label).pack(anchor="w")
            tk.Scale(fr, from_=lo, to=hi, resolution=res, orient=tk.HORIZONTAL, variable=var,
                     length=290, command=lambda _e: self.schedule()).pack(fill=tk.X)
        self.extra_controls(left)
        ttk.Button(left, text="Przywróć domyślne", command=self.reset).pack(fill=tk.X, padx=6, pady=4)
        ttk.Label(left, text="Przemiatany parametr:").pack(anchor="w", padx=6)
        self.sweep_var = tk.StringVar(value=self.PARAMS[0][1])
        cb = ttk.Combobox(left, textvariable=self.sweep_var, values=[p[1] for p in self.PARAMS], state="readonly")
        cb.pack(fill=tk.X, padx=6)
        cb.bind("<<ComboboxSelected>>", lambda _e: self.do_sweep())
        ttk.Button(left, text="Przelicz przemiatanie", command=self.do_sweep).pack(fill=tk.X, padx=6, pady=3)
        ttk.Label(left, text="Wyniki (ZPK, zapasy, step)", font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=6)
        self.result = tk.Text(left, height=30, width=44, font=("Consolas", 8), bg="#f4f6fa", wrap=tk.WORD)
        self.result.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))

        vpw = ttk.PanedWindow(right, orient=tk.VERTICAL)
        vpw.pack(fill=tk.BOTH, expand=True)
        nbf = ttk.Frame(vpw)
        self.nb = ttk.Notebook(nbf)
        self.nb.pack(fill=tk.BOTH, expand=True)
        self.fig, self.canvas = self._mkfig(self.nb, "Bode / Nyquist / Step / Bieguny")
        self.fig_sw, self.canvas_sw = self._mkfig(self.nb, "Przemiatanie parametru")
        if self.TIME_SIM:
            self.fig_ts, self.canvas_ts = self._mkfig(self.nb, "Symulacja czasowa")
        txtfr = ttk.Frame(vpw)
        ex = ScrolledText(txtfr, height=12, wrap=tk.WORD, font=("Segoe UI", 10), bg="#fffdf3")
        ex.pack(fill=tk.BOTH, expand=True)
        ex.insert("1.0", EXPL.get(self.key, "") + "\n" + "=" * 70 + "\n" + CONTROL_INTRO)
        ex.configure(state=tk.DISABLED)
        vpw.add(nbf, weight=3)
        vpw.add(txtfr, weight=1)
        self.refresh()
        self.do_sweep()

    def _mkfig(self, nb, title):
        fr = ttk.Frame(nb)
        nb.add(fr, text=title)
        fig = Figure(figsize=(9, 6), dpi=90)
        canvas = FigureCanvasTkAgg(fig, master=fr)
        NavigationToolbar2Tk(canvas, fr).update()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        return fig, canvas

    def extra_controls(self, parent):
        pass

    def reset(self):
        for name, _l, _lo, _hi, val, _r in self.PARAMS:
            self.vars[name].set(val)
        self.refresh()
        self.do_sweep()

    def p(self, name):
        return self.vars[name].get()

    def params(self):
        return {k: v.get() for k, v in self.vars.items()}

    def schedule(self):
        if self._pending:
            self.after_cancel(self._pending)
        self._pending = self.after(200, self.refresh)

    def build(self, p):
        """Zwraca dict: C (TF regulatora), P (TF obiektu), fsw (lub None),
        y (nazwa wyjścia), u (nazwa zakłócenia), dist_scale."""
        raise NotImplementedError

    def time_sim(self, fig, p):
        pass

    # --- analiza jednego zestawu parametrów ---
    def analyse(self, p, step=True):
        sysd = self.build(p)
        C, P = sysd["C"], sysd["P"]
        Lp = C * P
        w = 2 * PI * np.logspace(math.log10(self.FREQ[0]), math.log10(self.FREQ[1]), 2500)
        mg = ct.margins(Lp, w)
        T, Sd = ct.closed_loop(None, C, P)
        out = dict(sys=sysd, L=Lp, w=w, mg=mg, T=T, Sd=Sd, stable=ct.is_stable(T))
        if step:
            te = sysd.get("t_end") or ct.settle_horizon(T, mg["wc"])
            t, y = ct.step_response(T, te)
            out["t"], out["y"] = t, y
            out["info"] = ct.step_info(t, y, T.dc_gain() if out["stable"] else None)
            td, yd = ct.step_response(Sd, te * 1.5)
            out["td"], out["yd"] = td, yd * sysd.get("dist_scale", 1.0)
        return out

    def refresh(self):
        self._pending = None
        p = self.params()
        try:
            a = self.analyse(p)
        except Exception as exc:
            self.result.delete("1.0", tk.END)
            self.result.insert("1.0", f"Błąd obliczeń: {exc!r}")
            return
        self.draw_main(a)
        self.write_results(a)
        if self.TIME_SIM:
            self.fig_ts.clear()
            try:
                self.time_sim(self.fig_ts, p)
                self.fig_ts.tight_layout()
            except Exception as exc:
                self.fig_ts.text(0.1, 0.5, f"Błąd symulacji: {exc!r}")
            self.canvas_ts.draw_idle()

    def draw_main(self, a):
        fig = self.fig
        fig.clear()
        sysd, mg, w = a["sys"], a["mg"], a["w"]
        f = w / (2 * PI)
        HL = ct.freq_resp(a["L"], w)
        HP = ct.freq_resp(sysd["P"], w)
        HC = ct.freq_resp(sysd["C"], w)
        HT = ct.freq_resp(a["T"], w)
        # --- Bode amplituda ---
        ax1 = fig.add_subplot(2, 3, 1)
        ax1.semilogx(f, 20 * np.log10(np.abs(HL)), color=COL["k"], lw=2, label="L = C·P")
        ax1.semilogx(f, 20 * np.log10(np.abs(HP)), color=COL["c"], lw=1, ls="--", label="P (obiekt)")
        ax1.semilogx(f, 20 * np.log10(np.abs(HC)), color=COL["o"], lw=1, ls="--", label="C (regulator)")
        ax1.semilogx(f, 20 * np.log10(np.abs(HT) + 1e-12), color=COL["b"], lw=1, label="T (zamknięta)")
        ax1.axhline(0, color="r", lw=0.8)
        if mg["wc"]:
            ax1.axvline(mg["wc"] / 2 / PI, color="r", ls=":", lw=1)
        if mg["w180"]:
            ax1.axvline(mg["w180"] / 2 / PI, color=COL["p"], ls=":", lw=1)
            ax1.annotate(f"GM={mg['gm']:.1f} dB", xy=(mg["w180"] / 2 / PI, -mg["gm"]),
                         xytext=(mg["w180"] / 2 / PI, -mg["gm"] - 25), fontsize=7,
                         arrowprops=dict(arrowstyle="->", color=COL["p"]))
        lm = 20 * np.log10(np.abs(HL))
        ax1.set_ylim(max(np.min(lm), -100) - 5, min(np.max(lm), 120) + 10)
        wave_ax(ax1, "Bode - amplituda", "dB", "f [Hz]")
        ax1.legend(fontsize=6, loc="lower left")
        # --- Bode faza ---
        ax4 = fig.add_subplot(2, 3, 4, sharex=ax1)
        ax4.semilogx(f, mg["ph"], color=COL["k"], lw=2, label="faza L")
        php = np.unwrap(np.angle(HP)) * 180 / PI
        ax4.semilogx(f, php - 360 * round(php[0] / 360), color=COL["c"], lw=1, ls="--", label="faza P")
        ax4.axhline(-180, color="r", lw=0.8)
        if mg["wc"]:
            fc = mg["wc"] / 2 / PI
            phc = -180 + mg["pm"]
            ax4.plot([fc, fc], [-180, phc], color="g", lw=3)
            ax4.text(fc * 1.1, (phc - 180) / 2, f"PM={mg['pm']:.1f}°", fontsize=8, color="g")
        ax4.set_ylim(max(np.min(mg["ph"]), -540) - 10, max(np.max(mg["ph"]), 0) + 20)
        wave_ax(ax4, "Bode - faza", "°", "f [Hz]")
        ax4.legend(fontsize=6, loc="lower left")
        # --- Nyquist ---
        ax2 = fig.add_subplot(2, 3, 2)
        ax2.plot(HL.real, HL.imag, color=COL["k"], lw=1.5, label="ω > 0")
        ax2.plot(HL.real, -HL.imag, color=COL["g"], lw=1, ls="--", label="ω < 0")
        ax2.plot(-1, 0, "r+", ms=14, mew=2)
        ax2.add_patch(Circle((0, 0), 1, fill=False, ls=":", color="r", lw=0.8))
        if mg["wc"]:
            z = ct.freq_resp(a["L"], np.array([mg["wc"]]))[0]
            ax2.plot([0, z.real], [0, z.imag], "g-", lw=1)
            ax2.plot(z.real, z.imag, "go")
        if mg["w180"]:
            z = ct.freq_resp(a["L"], np.array([mg["w180"]]))[0]
            ax2.plot(z.real, z.imag, "o", color=COL["p"])
        R = 2.5
        ax2.set_xlim(-R, R)
        ax2.set_ylim(-R, R)
        ax2.set_aspect("equal")
        wave_ax(ax2, "Nyquist L(jω) (powiększenie wokół -1)", "Im", "Re")
        ax2.legend(fontsize=6, loc="upper right")
        # --- mapa biegunów/zer ---
        ax5 = fig.add_subplot(2, 3, 5)
        pT, zT = a["T"].poles(), a["T"].zeros()
        Lr = a["L"].rational()
        pL = Lr.poles()
        ax5.plot(pL.real, pL.imag, "x", color=COL["g"], ms=8, label="bieguny L (otwarta)")
        ax5.plot(pT.real, pT.imag, "x", color="r", ms=10, mew=2, label="bieguny T (zamknięta)")
        ax5.plot(zT.real, zT.imag, "o", mfc="none", color=COL["c"], ms=9, label="zera T")
        ax5.axvline(0, color="k", lw=0.8)
        ax5.axhline(0, color="k", lw=0.5)
        allp = np.r_[pT, zT, pL]
        sc = max(np.max(np.abs(allp)) if allp.size else 1, 1) * 1.1
        for zeta in [0.2, 0.5, 0.707]:
            ang = math.acos(zeta)
            ax5.plot([0, -sc * math.cos(ang)], [0, sc * math.sin(ang)], ":", color="#999999", lw=0.7)
            ax5.plot([0, -sc * math.cos(ang)], [0, -sc * math.sin(ang)], ":", color="#999999", lw=0.7)
            ax5.text(-sc * math.cos(ang) * 0.9, sc * math.sin(ang) * 0.9, f"ζ={zeta}", fontsize=6, color="#777777")
        wave_ax(ax5, "Bieguny i zera (Padé dla opóźnienia)", "Im [rad/s]", "Re [rad/s]")
        ax5.legend(fontsize=6, loc="lower left")
        # --- odpowiedzi skokowe ---
        ax3 = fig.add_subplot(2, 3, 3)
        if "t" in a:
            sc_t, unit = (1e3, "ms") if a["t"][-1] < 5 else (1, "s")
            ax3.plot(a["t"] * sc_t, a["y"], color=COL["a"], lw=1.5, label="y (skok zadanej)")
            ax3.axhline(1.0, color="k", ls=":", lw=0.8)
            ax3.axhspan(0.98, 1.02, color="g", alpha=0.1)
            inf = a["info"]
            if inf.get("ts"):
                ax3.axvline(inf["ts"] * sc_t, color="g", ls="--", lw=0.8)
            wave_ax(ax3, f"Step: {sysd.get('y', 'y')} / zadana", "y/y*", f"t [{unit}]")
            ax3.legend(fontsize=6)
            ax6 = fig.add_subplot(2, 3, 6)
            ax6.plot(a["td"] * sc_t, a["yd"], color=COL["p"], lw=1.5)
            ax6.axhline(0, color="k", lw=0.6)
            wave_ax(ax6, f"Zakłócenie: {sysd.get('u', 'd')}", sysd.get("yd_unit", ""), f"t [{unit}]")
        try:
            fig.tight_layout()
        except Exception:
            pass
        self.canvas.draw_idle()

    def write_results(self, a):
        s, mg = a["sys"], a["mg"]
        L = a["L"]
        lines = ["=== ZPK ==="]
        lines.append(s["P"].zpk_text("P"))
        lines.append(s["C"].zpk_text("C"))
        lines.append(L.zpk_text("L"))
        lines.append(a["T"].zpk_text("T") + "   [Padé]")
        lines.append("")
        lines.append("=== Bieguny pętli zamkniętej ===")
        for pk in sorted(a["T"].poles(), key=lambda z: -z.real):
            zeta, wn = ct.damping_of(pk)
            lines.append(f"  {pk.real:+.4g} {pk.imag:+.4g}j   ζ={zeta:.3f}  fn={wn / 2 / PI:.4g} Hz")
        lines.append("UKŁAD STABILNY ✓" if a["stable"] else "!!! UKŁAD NIESTABILNY !!!")
        lines.append("")
        lines.append("=== Zapasy stabilności (Bode/Nyquist) ===")
        lines.append(f"fc = {fmt(mg['wc'] and mg['wc'] / 2 / PI)} Hz,  PM = {fmt(mg['pm'])}°")
        lines.append(f"f180 = {fmt(mg['w180'] and mg['w180'] / 2 / PI)} Hz,  GM = {fmt(mg['gm'])} dB")
        if len(mg["all_wc"]) > 1:
            lines.append("  wszystkie przecięcia 0 dB: " +
                         ", ".join(f"{w_ / 2 / PI:.1f} Hz (PM {pm_:.0f}°)" for w_, pm_ in mg["all_wc"]))
        lines.append("")
        if "info" in a:
            inf = a["info"]
            lines.append("=== Odpowiedź skokowa ===")
            lines.append(f"tr(10-90%) = {fmt(inf['tr'] and inf['tr'] * 1e3)} ms")
            lines.append(f"przeregulowanie = {fmt(inf['os'], 3)} %")
            lines.append(f"ts(2%) = {fmt(inf['ts'] and inf['ts'] * 1e3)} ms")
            lines.append(f"wartość ustalona = {fmt(a['T'].dc_gain())}")
            yd = a["yd"]
            lines.append(f"zakłócenie: max odchyłka = {np.nanmax(np.abs(yd)):.4g} {s.get('yd_unit', '')}, "
                         f"końcowa = {yd[-1]:.3g}")
        lines.append("")
        lines.append("=== Interpretacja ===")
        lines += ct.interpret_margins(mg["pm"], mg["gm"], mg["wc"], s.get("fsw"))
        lines += s.get("notes", [])
        self.result.delete("1.0", tk.END)
        self.result.insert("1.0", "\n".join(lines))

    def do_sweep(self):
        label = self.sweep_var.get()
        par = next(pp for pp in self.PARAMS if pp[1] == label)
        name, _, lo, hi, _, _ = par
        base = self.params()
        cur = base[name]
        xs = np.linspace(lo, hi, 26)
        if lo <= 0:
            xs = np.linspace(lo + (hi - lo) * 0.02, hi, 26)
        PM, GM, FC, OS, TS = [], [], [], [], []
        for x in xs:
            pp = dict(base)
            pp[name] = x
            try:
                a = self.analyse(pp, step=True)
                PM.append(a["mg"]["pm"] if a["mg"]["pm"] is not None else np.nan)
                GM.append(a["mg"]["gm"] if a["mg"]["gm"] is not None else np.nan)
                FC.append(a["mg"]["wc"] / 2 / PI if a["mg"]["wc"] else np.nan)
                inf = a["info"]
                OS.append(inf["os"] if (a["stable"] and inf["os"] is not None) else np.nan)
                TS.append(inf["ts"] * 1e3 if (a["stable"] and inf["ts"]) else np.nan)
            except Exception:
                PM.append(np.nan); GM.append(np.nan); FC.append(np.nan); OS.append(np.nan); TS.append(np.nan)
        fig = self.fig_sw
        fig.clear()
        for k, (data, ttl, unit, ref) in enumerate([(PM, "Zapas fazy PM", "°", [45, 60]),
                                                   (GM, "Zapas wzmocnienia GM", "dB", [6]),
                                                   (FC, "Częstotliwość odcięcia fc", "Hz", []),
                                                   (OS, "Przeregulowanie (step)", "%", [])]):
            ax = fig.add_subplot(2, 2, k + 1)
            ax.plot(xs, data, "o-", color=PH_COL[k % 3] if k < 3 else COL["p"], ms=3)
            for rv in ref:
                ax.axhline(rv, color="g", ls=":", lw=0.8)
            if k < 2:
                ax.axhline(0, color="r", lw=0.8)
            ax.axvline(cur, color="k", ls="--", lw=1)
            wave_ax(ax, f"{ttl} vs {label}", unit, label)
            if k == 3:
                ax2 = ax.twinx()
                ax2.plot(xs, TS, "s--", color=COL["g"], ms=3)
                ax2.set_ylabel("ts [ms]", color=COL["g"])
        try:
            fig.tight_layout()
        except Exception:
            pass
        self.canvas_sw.draw_idle()


class TabC1(ControlTab):
    key = "C1"
    FREQ = (1.0, 2e5)
    PARAMS = [("R", "R obciążenia [Ω]", 0.1, 30, 10, 0.1),
              ("L", "L [mH]", 0.1, 20, 1, 0.1),
              ("fsw", "fsw [Hz] (Ts = 1/fsw)", 1000, 40000, 10000, 500),
              ("kd", "opóźnienie Td / Ts", 0, 3, 1.5, 0.05),
              ("Kp", "Kp [V/A]", 0.1, 30, 3.0, 0.1),
              ("Ki", "Ki [V/(A·s)]", 0, 200000, 30000, 500),
              ("fbw", "pasmo docelowe dla przycisku [Hz]", 50, 5000, 500, 10)]

    def extra_controls(self, parent):
        ttk.Button(parent, text="Dobierz Kp, Ki (kompensacja bieguna R/L)", command=self.tune).pack(fill=tk.X, padx=6, pady=3)

    def tune(self):
        wc = 2 * PI * self.p("fbw")
        self.vars["Kp"].set(round(wc * self.p("L") * 1e-3, 1))
        self.vars["Ki"].set(round(wc * self.p("R") / 500) * 500)
        self.refresh()
        self.do_sweep()

    def build(self, p):
        L = p["L"] * 1e-3
        Ts = 1 / p["fsw"]
        P = ct.TF([1.0], [L, p["R"]], delay=p["kd"] * Ts)
        C = ct.pi_tf(p["Kp"], p["Ki"])
        z = p["Ki"] / max(p["Kp"], 1e-9)
        return dict(C=C, P=P, fsw=p["fsw"], y="prąd i", u="1 V na wejściu obiektu",
                    yd_unit="A",
                    notes=[f"biegun obiektu -R/L = {-p['R'] / L:.4g} rad/s, zero PI = {-z:.4g} rad/s",
                           f"kompensacja: Ki/Kp = {z:.4g} vs R/L = {p['R'] / L:.4g}",
                           f"Td = {p['kd'] * Ts * 1e6:.1f} µs -> strata fazy przy 1 kHz: {360 * 1e3 * p['kd'] * Ts:.1f}°"])


class TabC2(ControlTab):
    key = "C2"
    FREQ = (1.0, 1e5)
    TIME_SIM = True
    PARAMS = [("R", "R dławika [Ω]", 0.01, 2, 0.2, 0.01),
              ("L", "L dławika [mH]", 0.5, 20, 5, 0.1),
              ("fsw", "fsw [Hz]", 2000, 40000, 10000, 500),
              ("kd", "opóźnienie Td / Ts", 0, 3, 1.5, 0.05),
              ("Kp", "Kp [V/A]", 0.5, 80, 15, 0.5),
              ("Kr", "Kr [V/A]", 0, 2000, 300, 10),
              ("wc", "ωc rezonansu [rad/s]", 0.5, 50, 5, 0.5),
              ("f0", "f rezonansu regulatora [Hz]", 45, 65, 50, 0.5),
              ("fg", "f sieci / prądu zadanego [Hz] (symulacja)", 45, 65, 50, 0.5)]

    def build(self, p):
        L = p["L"] * 1e-3
        Ts = 1 / p["fsw"]
        P = ct.TF([1.0], [L, p["R"]], delay=p["kd"] * Ts)
        C = ct.pr_tf(p["Kp"], p["Kr"], 2 * PI * p["f0"], p["wc"])
        return dict(C=C, P=P, fsw=p["fsw"], y="prąd sieci", u="skok napięcia sieci 1 V", yd_unit="A",
                    t_end=0.1,
                    notes=[f"pasmo P: fc ≈ Kp/(2πL) = {p['Kp'] / (2 * PI * L):.0f} Hz",
                           f"|C(j2π·{p['f0']:.0f})| = {abs(C(1j * 2 * PI * p['f0'])):.4g} (rezonans)"])

    def time_sim(self, fig, p):
        """Śledzenie sinusoidy 10 A: PR vs PI o tym samym Kp (Ki = Kr·2ωc)."""
        L = p["L"] * 1e-3
        R = p["R"]
        Ts = 1 / p["fsw"]
        nd = max(1, int(round(p["kd"])))
        N = int(0.1 / Ts)
        t = np.arange(N) * Ts
        iref = 10 * np.sin(2 * PI * p["fg"] * t)
        eg = 0.0 * t + 20 * np.sin(2 * PI * p["fg"] * t + 0.3)   # napięcie sieci (zakłócenie)
        w0 = 2 * PI * p["f0"]
        res = {}
        for mode in ["PR", "PI"]:
            i = 0.0
            x1 = x2 = 0.0      # stan członu rezonansowego / całki
            ubuf = [0.0] * (nd + 1)
            iv = np.empty(N)
            a = math.exp(-R * Ts / L)
            b = (1 - a) / R
            for k in range(N):
                iv[k] = i
                e = iref[k] - i
                if mode == "PR":
                    # dyskretyzacja (Euler wsteczny, krok Ts) członu 2ωc·Kr·s/(s²+2ωc s+ω0²)
                    x1n = x1 + Ts * x2
                    x2n = x2 + Ts * (-w0 * w0 * x1n - 2 * p["wc"] * x2 + e)
                    x1, x2 = x1n, x2n
                    u = p["Kp"] * e + 2 * p["wc"] * p["Kr"] * x2
                else:
                    x1 += Ts * e
                    u = p["Kp"] * e + 2 * p["wc"] * p["Kr"] * x1
                ubuf.append(u + eg[k])          # sprzężenie w przód napięcia sieci
                u_app = ubuf.pop(0)
                i = a * i + b * (u_app - eg[k])
            res[mode] = iv
        ax1 = fig.add_subplot(2, 1, 1)
        ax1.plot(t * 1e3, iref, "k--", lw=1, label="i* (zadany)")
        ax1.plot(t * 1e3, res["PR"], color=COL["a"], lw=1, label="PR")
        ax1.plot(t * 1e3, res["PI"], color=COL["c"], lw=1, label="PI (Ki = 2ωc·Kr)")
        wave_ax(ax1, "Śledzenie prądu sinusoidalnego (model dyskretny z opóźnieniem)", "A", "t [ms]")
        ax1.legend(fontsize=7)
        ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)
        ax2.plot(t * 1e3, iref - res["PR"], color=COL["a"], label="błąd PR")
        ax2.plot(t * 1e3, iref - res["PI"], color=COL["c"], label="błąd PI")
        wave_ax(ax2, "Uchyb śledzenia", "A", "t [ms]")
        ax2.legend(fontsize=7)


class TabC3(ControlTab):
    key = "C3"
    FREQ = (0.1, 1e4)
    PARAMS = [("C", "C szyny DC [µF]", 100, 10000, 2200, 50),
              ("fci", "pasmo pętli prądu fci [Hz]", 50, 3000, 500, 10),
              ("Kp", "Kp napięcia [A/V]", 0.01, 10, 1.4, 0.01),
              ("Ki", "Ki napięcia [A/(V·s)]", 0, 1000, 110, 1),
              ("a", "a - optimum symetryczne (przycisk)", 1.5, 6, 3, 0.1),
              ("dI", "skok prądu obciążenia [A]", 0.1, 20, 5, 0.1)]

    def extra_controls(self, parent):
        ttk.Button(parent, text="Optimum symetryczne: Kp = C/(a τi), Ti = a² τi",
                   command=self.tune).pack(fill=tk.X, padx=6, pady=3)

    def tune(self):
        C = self.p("C") * 1e-6
        taui = 1 / (2 * PI * self.p("fci"))
        a = self.p("a")
        Kp = C / (a * taui)
        Ki = Kp / (a * a * taui)
        self.vars["Kp"].set(round(Kp, 2))
        self.vars["Ki"].set(round(Ki))
        self.refresh()
        self.do_sweep()

    def build(self, p):
        C = p["C"] * 1e-6
        taui = 1 / (2 * PI * p["fci"])
        P = ct.TF([1.0], [C, 0.0]) * ct.first_order(1.0, taui)
        Cc = ct.pi_tf(p["Kp"], p["Ki"])
        a = p["a"]
        return dict(C=Cc, P=P, fsw=None, y="napięcie DC", u=f"prąd obciążenia +{p['dI']} A",
                    yd_unit="V", dist_scale=-p["dI"],
                    notes=[f"τi = {taui * 1e3:.3f} ms",
                           f"optimum symetryczne dla a = {a}: PM teoret. = {math.degrees(math.asin((a * a - 1) / (a * a + 1))):.1f}°,"
                           f" fc = {1 / (a * taui) / 2 / PI:.1f} Hz",
                           f"stosunek pasm fci/fc ≈ {p['fci'] / max(1e-9, (p['Kp'] / C) / 2 / PI):.1f} (zalecane ≥ 5)"])


class TabC4(ControlTab):
    key = "C4"
    FREQ = (0.05, 1e4)
    TIME_SIM = True
    PARAMS = [("Vdc", "Vdc [V]", 12, 600, 24, 1),
              ("m", "amplituda napięcia fazowego / Vdc", 0.05, 0.577, 0.42, 0.005),
              ("R", "R obciążenia [Ω]", 0.5, 30, 10, 0.5),
              ("L", "L obciążenia [mH]", 0.1, 100, 1, 0.1),
              ("C", "C kondensatora [µF]", 100, 10000, 2200, 100),
              ("Kp", "Kp (δ na jedn. vc/(Vdc/2))", 0.01, 20, 0.5, 0.01),
              ("Ki", "Ki [1/s]", 0, 500, 0, 1),
              ("ff", "filtr pomiaru vc fc [Hz]", 5, 2000, 300, 5),
              ("fsw", "fsw [Hz]", 500, 10000, 2500, 50)]

    def kb(self, p):
        L = p["L"] * 1e-3
        Z = math.hypot(p["R"], 2 * PI * 50 * L)
        I = p["m"] * p["Vdc"] / Z
        phi = math.atan2(2 * PI * 50 * L, p["R"])
        return pc.npc_balance_gain(p["m"] / 0.5, I, phi, n=360), I, phi

    def build(self, p):
        Kb, I, phi = self.kb(p)
        C = p["C"] * 1e-6
        Ts = 1 / p["fsw"]
        tf_ = 1 / (2 * PI * p["ff"])
        Pl = ct.TF([abs(Kb) / (p["Vdc"] / 2)], [C, 0.0], delay=Ts) * ct.first_order(1.0, tf_)
        Cc = ct.pi_tf(p["Kp"], p["Ki"]) if p["Ki"] > 0 else ct.TF([p["Kp"]], [1.0])
        return dict(C=Cc, P=Pl, fsw=None, y="vc/(Vdc/2)", u="stały prąd 1 A z punktu M",
                    yd_unit="V", dist_scale=(p["Vdc"] / 2) / max(abs(Kb), 1e-9),
                    notes=[f"Kb = d<i_M>/dδ = {Kb:.4f} A  (I = {I:.3f} A, cos φ = {math.cos(phi):.3f})",
                           f"fc ≈ Kp·|Kb|/(C·Vdc/2·2π) = {p['Kp'] * abs(Kb) / (C * p['Vdc'] / 2) / 2 / PI:.2f} Hz",
                           "regulator P -> δ = -Kp·vc/(Vdc/2) (znak wg Kb)"])

    def time_sim(self, fig, p):
        r = pc.npc3_sim(p["Vdc"], p["R"], p["L"] * 1e-3, p["C"] * 1e-6, 50, p["fsw"], p["m"] * p["Vdc"],
                        mode="pi", Kp=p["Kp"], vc0=0.1 * p["Vdc"], n_cyc=10, pps=60)
        t = r["t"] * 1e3
        a = self.analyse(p, step=False)
        ax = fig.add_subplot(2, 1, 1)
        ax.plot(t, r["vc"], color=COL["p"], lw=0.8, label="pełny model przełączany (regulator P)")
        # przewidywanie liniowe: odpowiedź swobodna z vc0 = odpowiedź T na skok ujemny
        T = a["T"]
        tt, yy = ct.step_response(T, r["t"][-1], n=800)
        ax.plot(tt * 1e3, 0.1 * p["Vdc"] * (1 - yy), "k--", lw=1.2, label="model liniowy (zanik vc0)")
        wave_ax(ax, "Zanik początkowej niesymetrii vc0 = 10 % Vdc", "vc [V]", "t [ms]")
        ax.legend(fontsize=7)
        ax2 = fig.add_subplot(2, 1, 2)
        last = slice(len(t) - r["ppc"], len(t))
        for k in range(3):
            ax2.plot(t[last], r["i"][k][last], color=PH_COL[k], lw=0.8)
        wave_ax(ax2, "Prądy fazowe (ostatni okres)", "A", "t [ms]")


# =============================================================================
# Aplikacja
# =============================================================================

GROUPS = [
    ("Rozdz. 4: Mostek H", [("4.2 Stany przewodzenia", TabH1), ("4.2 PWM bipolarne dc-ac", TabH2),
                            ("4.3 Wektory T+ T0 T-", TabH3), ("4.4 dc-dc unipolarne", TabH4),
                            ("4.4 Dwa mostki LC-RC", TabH5), ("4.5 PWM unipolarne dc-ac", TabH6),
                            ("4.5 Phase-shift PWM", TabH7)]),
    ("Rozdz. 5: Trójfazowe / SVPWM", [("5.2 Układy 3-fazowe", TabT1), ("5.3 Falownik sine-PWM", TabT2),
                                      ("5.4 Clarke", TabT3), ("5.5 Wektory przekształtnika", TabT4),
                                      ("5.6 Algorytm SVPWM", TabT5), ("5.7 Symulacja SVPWM", TabT6)]),
    ("Rozdz. 6: Wielopoziomowe NPC", [("6.3 Gałąź NPC 3/4-poz.", TabN1), ("6.4 Mapa wektorów", TabN2),
                                      ("6.5 SVPWM 3-poziomowe", TabN3), ("6.6-6.7 Balans kond. + symulacja", TabN4)]),
    ("Układy sterowania (Bode/ZPK/Nyquist/Step)", [("C1 PI prądu mostka H", TabC1), ("C2 PR prądu sieci", TabC2),
                                                   ("C3 Kaskada napięcia DC", TabC3), ("C4 Balans NPC (regulator)", TabC4)]),
]


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Strategie przełączania przekształtników energoelektronicznych - mostek H, SVPWM, NPC, sterowanie")
        self.geometry("1480x940")
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        outer = ttk.Notebook(self)
        outer.pack(fill=tk.BOTH, expand=True)
        self.lazy = {}
        self.built = set()
        for gname, tabs in GROUPS:
            gfr = ttk.Frame(outer)
            outer.add(gfr, text=gname)
            nb = ttk.Notebook(gfr)
            nb.pack(fill=tk.BOTH, expand=True)
            for name, cls in tabs:
                fr = ttk.Frame(nb)
                nb.add(fr, text=name)
                self.lazy[str(fr)] = (fr, cls)
            nb.bind("<<NotebookTabChanged>>", self._on_tab)
        outer.bind("<<NotebookTabChanged>>", self._on_outer)
        self.outer = outer

    def _on_outer(self, _e=None):
        gfr = self.nametowidget(self.outer.select())
        for ch in gfr.winfo_children():
            if isinstance(ch, ttk.Notebook):
                self._build(ch.select())

    def _on_tab(self, e):
        self._build(e.widget.select())

    def _build(self, sel):
        """Zakładki budowane przy pierwszym otwarciu - szybki start."""
        if sel in self.lazy and sel not in self.built:
            fr, cls = self.lazy[sel]
            self.built.add(sel)
            tab = cls(fr)
            tab.pack(fill=tk.BOTH, expand=True)


if __name__ == "__main__":
    App().mainloop()
