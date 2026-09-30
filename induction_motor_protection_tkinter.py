"""
Zabezpieczenia silnika indukcyjnego - interaktywna aplikacja Tkinter
====================================================================
Na podstawie rozdziału 12 "Induction Motor Protection"
(Mbungu, Bansal, Naidoo, Tungadio - Power System Protection in Smart Grid Environment).

Każda zakładka zawiera:
  * suwaki z parametrami (lewa kolumna) i wyniki liczone na żywo,
  * interaktywne wykresy Matplotlib (zoom / przesuwanie z paska narzędzi),
  * wyjaśnienie krok po kroku - jak dla ucznia liceum.

Zakładki:
  0. Przegląd           - schemat zabezpieczeń, kody ANSI, podsumowanie wyników
  1. Obwód zastępczy    - składowa zgodna i przeciwna, moment T1 i T2 (rozdz. 12.2-12.3)
  2. Rozruch i utyk     - symulacja ODE rozruchu + zabezpieczenia 49/48/51LR (rozdz. 12.5)
  3. Model cieplny      - model 3-węzłowy, przekaźnik cieplny, żywotność izolacji (rozdz. 12.4)
  4. Przykł. 12.1-12.3  - prądy silnika 13 KM i nastawy zabezpieczeń + krzywe TCC
  5. Przykł. 12.4-12.5  - zanik napięcia szczątkowego po odłączeniu zasilania
  6. Przykł. 12.6       - prąd równoważny cieplny I_eq = sqrt(I1^2 + K*I2^2)
  7. Przykł. 12.7-12.8  - asymetria napięć (NEMA, VUF), składowe symetryczne prądów
  8. Składowa przeciwna - rozdz. 12.8-12.9: I2/I1 = (Ist/Irun)*(V2/V1), grzanie wirnika
  9. Zadania 12.9-12.10 - moc składowej zgodnej, skutki przepięć (zadania nierozwiązane)

Uruchomienie:
    pip install numpy matplotlib
    python induction_motor_protection_tkinter.py
"""
import math
import cmath
import tkinter as tk
from tkinter import ttk

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle
from cycler import cycler

# =============================================================================
# Kolory (Catppuccin Mocha)
# =============================================================================
BG = "#1e1e2e"        # tło okna
BG2 = "#181825"       # tło wykresów
SURF = "#313244"      # panele
OVER = "#45475a"
FG = "#cdd6f4"        # tekst
SUB = "#a6adc8"
CYAN = "#89dceb"
GREEN = "#a6e3a1"
RED = "#f38ba8"
YELLOW = "#f9e2af"
PEACH = "#fab387"
MAUVE = "#cba6f7"
BLUE = "#89b4fa"
PINK = "#f5c2e7"

matplotlib.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG2, "axes.edgecolor": "#585b70",
    "axes.labelcolor": FG, "axes.titlecolor": CYAN, "axes.titleweight": "bold",
    "xtick.color": SUB, "ytick.color": SUB, "text.color": FG,
    "grid.color": OVER, "grid.alpha": 0.6, "grid.linestyle": ":",
    "legend.facecolor": SURF, "legend.edgecolor": "#585b70", "legend.fontsize": 8,
    "font.size": 9, "axes.titlesize": 10, "lines.linewidth": 1.8,
    "axes.prop_cycle": cycler(color=[CYAN, GREEN, PEACH, MAUVE, YELLOW, RED, BLUE]),
})

A_OP = cmath.exp(2j * math.pi / 3)   # operator a = 1∠120°
SQ3 = math.sqrt(3.0)


# =============================================================================
# RDZEŃ OBLICZENIOWY (fizyka) - funkcje niezależne od GUI
# =============================================================================

def wildi_full_load_current(hp, v_line):
    """Przybliżony prąd znamionowy silnika 3-f wg Wildiego: I_n ≈ 600·P[KM] / U[V].

    Wzór zawiera 'typowe' η·cosφ ≈ 0,72 (746 W/KM / (√3·0,72) ≈ 600)."""
    return 600.0 * hp / v_line


def motor_currents(hp, v_line, k_nl=0.5, k_lr_min=5.0, k_lr_max=6.0):
    """Prądy: znamionowy, jałowy (k_nl·In) i rozruchowy/zablokowanego wirnika (5-6·In)."""
    i_n = wildi_full_load_current(hp, v_line)
    return dict(In=i_n, Inl=k_nl * i_n, Ilr_min=k_lr_min * i_n, Ilr_max=k_lr_max * i_n)


def sc_setting(i_st, margin=1.25, ct_ratio=40.0):
    """Nastawa członu zwarciowego (50): I_sc = 1,25·I_st; wtórnie I_sc / przekładnia CT."""
    i_sc = margin * i_st
    return i_sc, i_sc / ct_ratio


def voltage_decay(v0, t, xm, x2, r2, f):
    """Zanik napięcia szczątkowego: V(t) = V0·exp(-t/T_op), T_op = (xm + x2)/(2πf·r2)."""
    t_op = (xm + x2) / (2 * math.pi * f * r2)
    return v0 * np.exp(-np.asarray(t, dtype=float) / t_op), t_op


def i_equivalent(i1, i2, k=3.0):
    """Prąd równoważny cieplnie (równ. 12.70): I_eq = sqrt(I1² + K·I2²)."""
    return math.sqrt(i1 ** 2 + k * i2 ** 2)


def seq_from_magnitudes(ma, mb, mc):
    """Składowe symetryczne przy założeniu, że kąty są symetryczne (0°, -120°, +120°).

    Zwraca (X0, X1, X2) - liczby zespolone."""
    xa = ma
    xb = mb * A_OP ** 2
    xc = mc * A_OP
    x0 = (xa + xb + xc) / 3
    x1 = (xa + A_OP * xb + A_OP ** 2 * xc) / 3
    x2 = (xa + A_OP ** 2 * xb + A_OP * xc) / 3
    return x0, x1, x2


def nema_unbalance(v):
    """Asymetria wg NEMA MG-1: max odchyłka od średniej / średnia · 100 %."""
    avg = sum(v) / len(v)
    return max(abs(x - avg) for x in v) / avg * 100.0, avg


def vuf_exact(vab, vbc, vca):
    """Dokładny współczynnik asymetrii VUF = |V2|/|V1| z samych modułów napięć
    międzyfazowych (trójkąt napięć musi być zamknięty): wzór IEC z parametrem β."""
    beta = (vab ** 4 + vbc ** 4 + vca ** 4) / (vab ** 2 + vbc ** 2 + vca ** 2) ** 2
    r = math.sqrt(max(0.0, 3 - 6 * beta))
    return math.sqrt((1 - r) / (1 + r)) * 100.0


def closed_triangle(vab, vbc, vca):
    """Rzeczywiste (zamknięte) wskazy napięć międzyfazowych o zadanych modułach.

    Vab + Vbc + Vca = 0  →  kąty z twierdzenia cosinusów. Zwraca 3 liczby zespolone."""
    # kąt między Vab a -Vca (wierzchołek A trójkąta)
    cos_a = (vab ** 2 + vca ** 2 - vbc ** 2) / (2 * vab * vca)
    cos_a = max(-1.0, min(1.0, cos_a))
    ang = math.acos(cos_a)
    p_a = 0j
    p_b = vab + 0j
    p_c = vca * cmath.exp(-1j * ang)         # punkt C (kolejność zgodna: C poniżej osi)
    # Vab = B - A, Vbc = C - B, Vca = A - C
    return p_b - p_a, p_c - p_b, p_a - p_c


def neg_seq_current_ratio(ist_over_irun, v2_over_v1):
    """Równ. 12.80-12.81: I2/I1 ≈ (Ist/Irun)·(V2/V1)."""
    return ist_over_irun * v2_over_v1


def overvoltage_effects(dv_pct, n_core=2.0):
    """Skutki przepięcia: przeciążenie ≈ ΔU %, straty w rdzeniu ∝ U^n (n=2 bez nasycenia,
    n≈2,6 z nasyceniem). Zwraca (przeciążenie %, wzrost strat w rdzeniu %)."""
    r = 1 + dv_pct / 100.0
    return dv_pct, (r ** n_core - 1) * 100.0


def thermal_trip_time(k, tau, a_init=0.0):
    """Czas zadziałania przekaźnika cieplnego (IEC 60255-8, równ. 12.69/12.71):

        t = τ·ln((k² - A)/(k² - 1)),   k = I_eq / I_th,  A - stan początkowy (0 = zimny).
    Dla k ≤ 1 przekaźnik nie zadziała (zwraca inf)."""
    k = np.asarray(k, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = tau * np.log((k ** 2 - a_init) / (k ** 2 - 1.0))
    return np.where(k > 1.0, t, np.inf)


def relative_life(temp, t_class=155.0):
    """Reguła Montsingera (rys. 12.7): każde +10 °C skraca życie izolacji o połowę."""
    return 2.0 ** ((t_class - np.asarray(temp, dtype=float)) / 10.0)


TABLE_12_2 = dict(unb=[0, 2, 3.5, 5], cur=[1.0, 1.01, 1.04, 1.075],
                  loss=[0, 4, 12.5, 25], heat=[100, 105, 114, 128])


# --- obwód zastępczy Steinmetza -------------------------------------------------

def seq_circuit(v, slip, rs, xs, xm, rr, xr):
    """Obwód zastępczy jednej składowej (bez Rm). Zwraca (Is, Ir, P_ag) w j.w.

    Z = Rs + jXs + [ jXm || (Rr/slip + jXr) ];  P_ag = |Ir|²·Rr/slip (moc w szczelinie)."""
    zr = rr / slip + 1j * xr
    zm = 1j * xm
    zin = rs + 1j * xs + zr * zm / (zr + zm)
    i_s = v / zin
    i_r = i_s * zm / (zr + zm)
    p_ag = abs(i_r) ** 2 * rr / slip
    return i_s, i_r, p_ag


class MotorPU:
    """Silnik klatkowy w jednostkach względnych, z rezystancją wirnika zależną od
    częstotliwości w wirniku (efekt wypierania prądu): Rr(s) = Rr0 + dRr·s."""

    def __init__(self, rs=0.03, xs=0.09, xm=3.0, rr0=0.025, drr=0.04, xr=0.09):
        self.rs, self.xs, self.xm, self.rr0, self.drr, self.xr = rs, xs, xm, rr0, drr, xr

    def rr(self, slip):
        return self.rr0 + self.drr * slip

    def operate(self, s, v1, v2=0.0):
        """Punkt pracy przy poślizgu s, napięciach składowej zgodnej v1 i przeciwnej v2."""
        s = max(s, 1e-4)
        s2 = 2.0 - s
        is1, ir1, pag1 = seq_circuit(v1, s, self.rs, self.xs, self.xm, self.rr(s), self.xr)
        if v2 > 0:
            is2, ir2, pag2 = seq_circuit(v2, s2, self.rs, self.xs, self.xm, self.rr(s2), self.xr)
        else:
            is2 = ir2 = 0j
            pag2 = 0.0
        ia = is1 + is2
        ib = A_OP ** 2 * is1 + A_OP * is2
        ic = A_OP * is1 + A_OP ** 2 * is2
        q_rotor = abs(ir1) ** 2 * self.rr(s) + abs(ir2) ** 2 * self.rr(s2)
        return dict(T1=pag1, T2=-pag2, T=pag1 - pag2, Is1=is1, Is2=is2, Ir1=ir1, Ir2=ir2,
                    Imax=max(abs(ia), abs(ib), abs(ic)), q=q_rotor, Pag1=pag1, Pag2=pag2)


def simulate_start(m, v=1.0, unb=0.0, tl0=0.8, fan=True, h=2.0, t_end=25.0,
                   jam_t=0.0, locked=False, theta0=0.0, t_sw=20.0,
                   i_set=2.0, t_lr=15.0, t_stall=6.5, i_sc=6.0, dt=0.004):
    """Symulacja rozruchu (równanie ruchu 12.17 w j.w.):

        2H·dω/dt = T_e(ω) - T_L(ω)
        dθ/dt   = q(ω)/C - θ/τ_ch          (model cieplny wirnika Zocholla, θ=1 → wyłączenie)
    C dobrane tak, by zablokowany wirnik przy U=1 z zimnego stanu osiągnął θ=1 po t_sw.

    Zabezpieczenia: 49 (cieplne), 50 (zwarciowe), 51LR (zablokowany wirnik/za długi
    rozruch - czas t_lr), 48 (utyk w czasie pracy - czas t_stall, aktywny po rozruchu)."""
    q_lr = m.operate(1.0, 1.0)["q"]
    c_th = q_lr * t_sw
    tau_cool = 1800.0
    n = int(t_end / dt) + 1
    t = np.linspace(0, t_end, n)
    w_arr = np.zeros(n); i_arr = np.zeros(n); th_arr = np.zeros(n)
    te_arr = np.zeros(n); tl_arr = np.zeros(n); i2_arr = np.zeros(n)
    w, th = 0.0, theta0
    timer, start_done, trip = 0.0, False, None
    t_run = None
    for k in range(n):
        tk_ = t[k]
        jam = locked or (jam_t > 0 and tk_ >= jam_t)
        tl = tl0 * (0.1 + 0.9 * w * w) if fan else tl0
        if jam:
            tl = 5.0
        if trip is None:
            op = m.operate(1.0 - w, v, v * unb / 100.0)
            te, imax, q = op["T"], op["Imax"], op["q"]
            i2 = abs(op["Is2"])
        else:
            te = imax = q = i2 = 0.0
        # --- mechanika ---
        if locked:
            w = 0.0
        else:
            w += (te - tl) / (2.0 * h) * dt
            w = min(max(w, 0.0), 1.0)
        # --- cieplny model wirnika ---
        th += (q / c_th - th / tau_cool) * dt
        # --- zabezpieczenia ---
        if trip is None:
            if imax >= i_sc:
                trip = (tk_, "50 - zwarciowe")
            elif th >= 1.0:
                trip = (tk_, "49 - przeciążenie cieplne")
            else:
                if imax > i_set:
                    timer += dt
                else:
                    if not start_done and tk_ > 0.05:
                        start_done, t_run = True, tk_
                    timer = 0.0
                if not start_done and timer >= t_lr:
                    trip = (tk_, "51LR - za długi rozruch / zablokowany wirnik")
                elif start_done and timer >= t_stall:
                    trip = (tk_, "48 - utyk w czasie pracy")
        w_arr[k], i_arr[k], th_arr[k] = w, imax, th
        te_arr[k], tl_arr[k], i2_arr[k] = te, tl, i2
    return dict(t=t, w=w_arr, i=i_arr, th=th_arr, te=te_arr, tl=tl_arr, i2=i2_arr,
                trip=trip, t_run=t_run)


def simulate_thermal3(k1=1.0, k2=1.3, t_ov=30.0, v=1.0, unb=0.0, amb=40.0,
                      hours=4.0, tau_relay=900.0, ith=1.05, a_k=3.0):
    """Model cieplny 3-węzłowy (równ. 12.44-12.46), Euler, krok 5 s.

        Cs·dθs/dt = PsL - (θs-θc)/Rsc
        Cc·dθc/dt = PcL + (θs-θc)/Rsc - (θc-θa)/Rca
        Cr·dθr/dt = PrL - (θr-θa)/Rra
    Równolegle działa model przekaźnika cieplnego 49: dθ_p/dt = ((I_eq/I_th)² - θ_p)/τ."""
    cs, cc, cr = 3000.0, 25000.0, 6000.0      # J/K
    rsc, rca, rra = 0.03, 0.07, 0.20          # K/W
    p_s0, p_c0, p_r0 = 600.0, 300.0, 400.0    # W - straty znamionowe
    loss_f = 1 + np.interp(unb, TABLE_12_2["unb"], TABLE_12_2["loss"]) / 100.0
    i2pu = neg_seq_current_ratio(6.0, unb / 100.0)
    dt = 5.0
    n = int(hours * 3600 / dt) + 1
    t = np.arange(n) * dt
    ts, tc, tr = amb, amb, amb
    th_p = 0.0
    out = np.zeros((n, 5))
    trip_t = None
    t_ov0 = 60.0 * 60.0                      # przeciążenie zaczyna się po 60 min
    for i in range(n):
        ti = t[i]
        k = k2 if t_ov0 <= ti < t_ov0 + t_ov * 60 else k1
        if trip_t is not None:
            k = 0.0
        ps = p_s0 * k * k * loss_f
        pr = p_r0 * k * k * loss_f
        pc = p_c0 * v * v if trip_t is None else 0.0
        ts += (ps - (ts - tc) / rsc) / cs * dt
        tc += (pc + (ts - tc) / rsc - (tc - amb) / rca) / cc * dt
        tr += (pr - (tr - amb) / rra) / cr * dt
        i_eq = i_equivalent(k, k * i2pu, a_k)
        th_p += ((i_eq / ith) ** 2 - th_p) / tau_relay * dt
        if trip_t is None and th_p >= 1.0:
            trip_t = ti
        out[i] = (ts, tc, tr, th_p, k)
    return dict(t=t / 60.0, ts=out[:, 0], tc=out[:, 1], tr=out[:, 2], thp=out[:, 3],
                k=out[:, 4], trip=None if trip_t is None else trip_t / 60.0)


# =============================================================================
# GUI - klasa bazowa zakładki
# =============================================================================

def style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)


class Tab:
    """Zakładka: lewa kolumna (suwaki + wyniki), prawa: wykres + wyjaśnienie."""
    title = "?"
    explanation = ""

    def __init__(self, app, nb):
        self.app = app
        self.v = {}
        self._job = None
        self.frame = ttk.Frame(nb)
        nb.add(self.frame, text=self.title)

        left = ttk.Frame(self.frame, width=390)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)
        cv = tk.Canvas(left, bg=BG, highlightthickness=0)
        sb = ttk.Scrollbar(left, orient="vertical", command=cv.yview)
        self.ctrl = ttk.Frame(cv)
        win = cv.create_window((0, 0), window=self.ctrl, anchor="nw")
        self.ctrl.bind("<Configure>", lambda e: cv.configure(scrollregion=cv.bbox("all")))
        cv.bind("<Configure>", lambda e: cv.itemconfigure(win, width=e.width))
        cv.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        cv.pack(side="left", fill="both", expand=True)
        self._bind_wheel(cv)

        pw = ttk.PanedWindow(self.frame, orient="vertical")
        pw.pack(side="left", fill="both", expand=True)
        figf = ttk.Frame(pw)
        pw.add(figf, weight=3)
        self.fig = Figure(figsize=(9, 5.4), dpi=100, layout="constrained")
        self.canvas = FigureCanvasTkAgg(self.fig, master=figf)
        tb = NavigationToolbar2Tk(self.canvas, figf, pack_toolbar=False)
        tb.update()
        tb.pack(side="bottom", fill="x")
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

        txtf = ttk.Frame(pw)
        pw.add(txtf, weight=2)
        self.text = tk.Text(txtf, wrap="word", bg=SURF, fg=FG, relief="flat",
                            font=("DejaVu Sans", 10), padx=12, pady=8, height=12)
        tsb = ttk.Scrollbar(txtf, orient="vertical", command=self.text.yview)
        self.text.configure(yscrollcommand=tsb.set)
        tsb.pack(side="right", fill="y")
        self.text.pack(side="left", fill="both", expand=True)
        self._setup_tags()
        self.write_explanation(self.explanation)

        self.build()
        self.res = tk.Label(self.ctrl, text="", justify="left", anchor="nw", bg=SURF, fg=GREEN,
                            font=("DejaVu Sans Mono", 9), padx=8, pady=6, wraplength=360)
        ttk.Label(self.ctrl, text="Wyniki", style="Sec.TLabel").pack(fill="x", padx=6, pady=(10, 2))
        self.res.pack(fill="x", padx=6, pady=(0, 10))
        self.refresh()

    # ----- przewijanie kółkiem myszy -----
    def _bind_wheel(self, cv):
        def on_wheel(e):
            if getattr(e, "num", None) == 4:
                cv.yview_scroll(-2, "units")
            elif getattr(e, "num", None) == 5:
                cv.yview_scroll(2, "units")
            else:
                cv.yview_scroll(int(-e.delta / 40) or (-1 if e.delta > 0 else 1), "units")

        def enter(_):
            cv.bind_all("<MouseWheel>", on_wheel)
            cv.bind_all("<Button-4>", on_wheel)
            cv.bind_all("<Button-5>", on_wheel)

        def leave(_):
            cv.unbind_all("<MouseWheel>")
            cv.unbind_all("<Button-4>")
            cv.unbind_all("<Button-5>")
        cv.bind("<Enter>", enter)
        cv.bind("<Leave>", leave)

    # ----- tekst wyjaśnienia z prostym formatowaniem -----
    def _setup_tags(self):
        t = self.text
        t.tag_configure("h1", font=("DejaVu Sans", 13, "bold"), foreground=CYAN, spacing1=6, spacing3=4)
        t.tag_configure("h2", font=("DejaVu Sans", 11, "bold"), foreground=MAUVE, spacing1=8, spacing3=2)
        t.tag_configure("eq", font=("DejaVu Sans Mono", 10), foreground=YELLOW, lmargin1=30, lmargin2=30)
        t.tag_configure("tip", foreground=PEACH, lmargin1=10, lmargin2=10, spacing1=3)
        t.tag_configure("ana", foreground=GREEN, font=("DejaVu Sans", 10, "italic"), lmargin1=10, lmargin2=10)
        t.tag_configure("p", foreground=FG, spacing1=2)

    def write_explanation(self, src):
        """Znaczniki: '# ' nagłówek, '## ' podtytuł, '= ' wzór, '! ' rada inżyniera,
        '> ' analogia; reszta - zwykły akapit."""
        t = self.text
        t.configure(state="normal")
        t.delete("1.0", "end")
        for line in src.strip("\n").split("\n"):
            s = line.strip()
            if s.startswith("## "):
                t.insert("end", s[3:] + "\n", "h2")
            elif s.startswith("# "):
                t.insert("end", s[2:] + "\n", "h1")
            elif s.startswith("= "):
                t.insert("end", s[2:] + "\n", "eq")
            elif s.startswith("! "):
                t.insert("end", "🔧 Inżynier radzi: " + s[2:] + "\n", "tip")
            elif s.startswith("> "):
                t.insert("end", "💡 " + s[2:] + "\n", "ana")
            else:
                t.insert("end", s + "\n", "p")
        t.configure(state="disabled")

    # ----- kontrolki -----
    def section(self, text):
        ttk.Label(self.ctrl, text=text, style="Sec.TLabel").pack(fill="x", padx=6, pady=(10, 2))

    def slider(self, key, label, lo, hi, init, unit="", fmt="{:.3g}", integer=False):
        fr = ttk.Frame(self.ctrl)
        fr.pack(fill="x", padx=8, pady=1)
        top = ttk.Frame(fr)
        top.pack(fill="x")
        ttk.Label(top, text=f"{label}" + (f" [{unit}]" if unit else "")).pack(side="left")
        var = tk.DoubleVar(value=init)
        val = ttk.Label(top, text=fmt.format(init), style="Val.TLabel", width=9, anchor="e")
        val.pack(side="right")

        def on_move(_=None):
            x = var.get()
            if integer:
                x = round(x)
                var.set(x)
            val.config(text=fmt.format(x))
            self.schedule()
        ttk.Scale(fr, variable=var, from_=lo, to=hi, orient="horizontal",
                  command=on_move).pack(fill="x")
        self.v[key] = var
        return var

    def check(self, key, label, init=False):
        var = tk.BooleanVar(value=init)
        ttk.Checkbutton(self.ctrl, text=label, variable=var, command=self.schedule).pack(anchor="w", padx=8, pady=2)
        self.v[key] = var
        return var

    def button(self, label, cmd):
        ttk.Button(self.ctrl, text=label, command=cmd).pack(fill="x", padx=8, pady=3)

    def g(self, key):
        return float(self.v[key].get())

    def schedule(self, *_):
        if self._job is not None:
            self.frame.after_cancel(self._job)
        self._job = self.frame.after(90, self.refresh)

    def refresh(self):
        self._job = None
        try:
            self.fig.clear()
            txt = self.draw()
            self.res.config(text=txt or "", fg=GREEN)
            self.canvas.draw_idle()
            self.app.status(f"{self.title}: przeliczono")
        except Exception as exc:  # nie pozwalamy, by błąd rysowania zabił GUI
            self.res.config(text=f"Błąd obliczeń:\n{exc}", fg=RED)
            self.app.status(f"Błąd: {exc}")

    # do nadpisania
    def build(self):
        pass

    def draw(self):
        return ""


# =============================================================================
# 0. PRZEGLĄD
# =============================================================================

class OverviewTab(Tab):
    title = "0 · Przegląd"
    explanation = """
# Po co chronić silnik? (rozdz. 12.1)
Silnik indukcyjny (asynchroniczny) to „koń roboczy” przemysłu – napędza pompy, wentylatory, taśmociągi, sprężarki. Jest prosty i wytrzymały, ale ma jedną słabość: nie lubi się przegrzewać. Ponad 80 % awarii silników da się uniknąć dobrymi zabezpieczeniami.
> Pomyśl o silniku jak o sportowcu. Może chwilowo biec sprintem (rozruch, prąd 5–8 razy większy niż normalnie), ale jeśli będzie sprintował za długo – „zagotuje się”. Zabezpieczenia są jak trener z zegarkiem, który mówi: „stop, wystarczy!”.
## Tor zasilania silnika (rys. 12.1)
Sieć → rozłącznik (odłącza do prac) → bezpiecznik lub wyłącznik (zabezpieczenie zwarciowe) → stycznik (włącza/wyłącza silnik w pracy normalnej) → przekaźnik przeciążeniowy → silnik.
## Kody ANSI zabezpieczeń silnika – ściągawka
= 49      przeciążenie cieplne (model cieplny)          → rozdz. 12.4
= 48/51LR za długi rozruch / zablokowany wirnik / utyk   → rozdz. 12.5
= 50      zwarcie (bezzwłoczne nadprądowe)              → rozdz. 12.6
= 50N/51G doziemienie                                   → rozdz. 12.7
= 46      asymetria / składowa przeciwna prądu          → rozdz. 12.8
= 27/59   podnapięcie / nadnapięcie                     → rozdz. 12.10
= 37      utrata obciążenia (np. pompa pracuje „na sucho”) → rozdz. 12.10.3
## Jak korzystać z aplikacji
Każda zakładka to jeden temat lub przykład z rozdziału. Po lewej przesuwasz suwaki – wykresy i wyniki liczą się od nowa. Pod wykresem jest wyjaśnienie krok po kroku. Pasek narzędzi pod wykresem pozwala powiększać (lupa) i przesuwać (krzyżyk) wykres.
! Zawsze zaczynaj od danych z tabliczki znamionowej silnika: moc, napięcie, prąd, krotność prądu rozruchowego, czas rozruchu i dopuszczalny czas zablokowania wirnika (zimny/gorący). Bez nich nie ustawisz dobrze żadnego zabezpieczenia.
"""

    def draw(self):
        ax = self.fig.add_subplot(1, 2, 1)
        ax.set_xlim(0, 10); ax.set_ylim(0, 12); ax.axis("off")
        ax.set_title("Tor zasilania silnika (rys. 12.1)")
        chain = [("Sieć 3~", BLUE), ("Rozłącznik", SUB), ("Bezpiecznik / wyłącznik\n(zabezp. zwarciowe 50)", PEACH),
                 ("Stycznik", MAUVE), ("Przekaźnik przeciążeniowy\n(49)", YELLOW)]
        for i, (name, col) in enumerate(chain):
            y = 11 - i * 2.0
            ax.add_patch(FancyBboxPatch((1.5, y - 0.6), 5, 1.2, boxstyle="round,pad=0.08",
                                        fc=SURF, ec=col, lw=2))
            ax.text(4, y, name, ha="center", va="center", color=col, fontsize=9)
            ax.annotate("", xy=(4, y - 0.75 - 0.5), xytext=(4, y - 0.7),
                        arrowprops=dict(arrowstyle="->", color=SUB, lw=1.5))
        ax.add_patch(Circle((4, 0.9), 0.75, fc=SURF, ec=GREEN, lw=2.5))
        ax.text(4, 0.9, "M\n3~", ha="center", va="center", color=GREEN, fontsize=11, weight="bold")
        ax.add_patch(Rectangle((7.3, 1.2), 2.5, 9, fc=BG2, ec=CYAN, lw=1.5, ls="--"))
        ax.text(8.55, 10.35, "Przekaźnik cyfrowy", ha="center", va="bottom", color=CYAN, fontsize=9, weight="bold")
        for j, code in enumerate(["49", "48/51LR", "50", "50N", "46", "27/59", "37"]):
            ax.text(8.55, 8.6 - j * 1.05, code, ha="center", va="center", fontsize=9, color=FG,
                    bbox=dict(boxstyle="round", fc=SURF, ec=OVER))

        # podsumowanie liczbowe przykładów (dane z książki)
        mc = motor_currents(13, 230)
        sc5 = sc_setting(mc["Ilr_min"])
        sc6 = sc_setting(mc["Ilr_max"])
        _, t_op = voltage_decay(400, 0, 31, 0.31, 0.26, 50)
        ieq6 = i_equivalent(0.912, 0.92, 3)
        nema, _ = nema_unbalance([371, 374, 395])
        _, v1, v2 = seq_from_magnitudes(371, 374, 395)
        _, i1, i2 = seq_from_magnitudes(18.3, 16, 12)
        ax2 = self.fig.add_subplot(1, 2, 2)
        ax2.axis("off")
        ax2.set_title("Podsumowanie przykładów 12.1-12.8")
        rows = [
            ("12.1", f"In ≈ {mc['In']:.1f} A, I0 ≈ {mc['Inl']:.1f} A, Ilr ≈ {mc['Ilr_min']:.0f}-{mc['Ilr_max']:.0f} A"),
            ("12.2", f"Isc = {sc5[0]:.0f}/{sc6[0]:.0f} A → {sc5[1]:.2f}/{sc6[1]:.2f} A wt."),
            ("12.3", f"50N: 0,25·In = {0.25 * mc['In']:.1f} A; 51LR/48: 2In, 15 s / 6,5 s"),
            ("12.4", f"T_op = {t_op:.3f} s; V(T_op)={400 / math.e:.0f} V; V(0,5 s)={400 * math.exp(-0.5 / t_op):.0f} V"),
            ("12.5", f"V(4·T_op) = {400 * math.exp(-4):.2f} V"),
            ("12.6", f"I_eq = {ieq6:.3f} j.w."),
            ("12.7", f"NEMA {nema:.2f} %, VUF {abs(v2) / abs(v1) * 100:.2f} % (kąty symetr.)"),
            ("12.8", f"I1={abs(i1):.2f} A, I2={abs(i2):.2f} A, I_eq={i_equivalent(abs(i1), abs(i2)):.2f} A"),
        ]
        for i, (k, r) in enumerate(rows):
            y = 0.93 - i * 0.115
            ax2.text(0.0, y, k, color=YELLOW, fontsize=10, weight="bold", transform=ax2.transAxes)
            ax2.text(0.12, y, r, color=FG, fontsize=9, transform=ax2.transAxes)
        return ("Aplikacja: 10 zakładek.\nPrzesuwaj suwaki w kolejnych\nzakładkach - wykresy liczą się\nna żywo."
                "\n\nPrzykłady: zakładki 4-9.\nTeoria i symulacje: 1-3.")


# =============================================================================
# 1. OBWÓD ZASTĘPCZY I MOMENT (12.2-12.3)
# =============================================================================

class CircuitTab(Tab):
    title = "1 · Obwód zastępczy"
    explanation = """
# Obwód zastępczy i moment silnika (rozdz. 12.2–12.3)
## Jednostki względne (j.w., per unit)
Zamiast liczyć w amperach i woltach, dzielimy wszystko przez wartości znamionowe z tabliczki: S_B = √3·U_B·I_B, Z_B = U_B²/S_B. Wtedy prąd 1,0 j.w. = prąd znamionowy, niezależnie czy silnik ma 1 kW czy 10 MW.
> To jak podawanie wyniku w procentach zamiast w punktach – łatwo porównać różne sprawdziany.
## Składowe symetryczne
Dowolny układ trzech napięć da się rozłożyć na: składową zgodną (V1 – kręci pole magnetyczne „do przodu”), przeciwną (V2 – kręci pole „do tyłu”) i zerową (V0). Silnik w gwiazdę bez uziemionego punktu nie przepuszcza składowej zerowej (równ. 12.14), więc zostają V1 i V2:
= S = 3·(V1·I1* + V2·I2*),   P = Re{S}
## Obwód Steinmetza (rys. 12.2)
Dla każdej składowej rysujemy ten sam obwód: Rs, Xs (stojan), Xm (magnesowanie), Xr i rezystancja wirnika „widziana” jako Rr/s. Różnica jest tylko w poślizgu:
= składowa zgodna:   Rr1/s        = Rr1 + Rr1·(1−s)/s     (12.26)
= składowa przeciwna: Rr2/(2−s)   = Rr2 − Rr2·(1−s)/(2−s)  (12.27)
Poślizg s = (ns − n)/ns. Dla pola wirującego do tyłu wirnik „widzi” poślizg 2 − s ≈ 2, więc prąd w wirniku ma częstotliwość ≈ 100 Hz. Przez efekt naskórkowy rezystancja Rr2 jest wtedy kilka razy większa niż Rr1 (K = Rr2/Rr1 ≈ 3).
## Bilans mocy (równ. 12.29–12.36)
= P_ag = |Ir|²·Rr/s               moc przez szczelinę powietrzną
= P_Cu,r = s·P_ag                  straty w miedzi wirnika
= P_dev = (1−s)·P_ag               moc mechaniczna rozwinięta
= T1 = P_ag1/ωs,  T2 = −P_ag2/ωs   (ωs = 1 j.w.)
Moment od składowej przeciwnej jest zawsze ujemny – hamuje silnik (równ. 12.43). Moment wypadkowy T = T1 + T2 jest więc mniejszy przy asymetrii napięć, a prąd i grzanie – większe.
> Wyobraź sobie dwie osoby pchające karuzelę: jedna mocno do przodu (V1), druga słabiej do tyłu (V2). Karuzela jedzie, ale wolniej, a obie osoby się męczą (straty!).
## Co obserwować na wykresach
• Zwiększ asymetrię V2 – czerwona krzywa T2 rośnie w dół, a wypadkowy moment spada.
• Przy s = 1 (postój) impedancje obu składowych są równe (tekst pod równ. 12.78).
• Słupki pokazują, gdzie „ucieka” moc: przy dużym poślizgu większość mocy szczeliny zamienia się w ciepło w wirniku (P_Cu,r = s·P_ag). Dlatego rozruch tak grzeje wirnik!
! Silnik o dużym poślizgu znamionowym (duże Rr) ma większy moment rozruchowy, ale gorszą sprawność. To zawsze kompromis konstruktora.
"""

    def build(self):
        self.section("Parametry obwodu (j.w.)")
        self.slider("rs", "Rs", 0.005, 0.1, 0.03, "j.w.")
        self.slider("xs", "Xs = Xr", 0.03, 0.2, 0.09, "j.w.")
        self.slider("xm", "Xm", 1.0, 5.0, 3.0, "j.w.")
        self.slider("rr0", "Rr0 (przy s→0)", 0.01, 0.08, 0.025, "j.w.")
        self.slider("drr", "ΔRr (efekt naskórkowy)", 0.0, 0.1, 0.04, "j.w.")
        self.section("Zasilanie i punkt pracy")
        self.slider("v1", "V1 (składowa zgodna)", 0.7, 1.1, 1.0, "j.w.")
        self.slider("unb", "V2/V1 (asymetria)", 0.0, 15.0, 4.0, "%")
        self.slider("s", "Poślizg s punktu pracy", 0.005, 1.0, 0.03, "", "{:.3f}")
        self.slider("pcon", "P_con straty mech.", 0.0, 0.08, 0.05, "j.w.", "{:.3f}")

    def draw(self):
        m = MotorPU(self.g("rs"), self.g("xs"), self.g("xm"), self.g("rr0"), self.g("drr"), self.g("xs"))
        v1 = self.g("v1"); unb = self.g("unb") / 100
        sp = np.linspace(0.0, 0.999, 400)
        res = [m.operate(1 - w, v1, v1 * unb) for w in sp]
        t1 = np.array([r["T1"] for r in res]); t2 = np.array([r["T2"] for r in res])
        i1 = np.array([abs(r["Is1"]) for r in res]); i2 = np.array([abs(r["Is2"]) for r in res])
        s = self.g("s")
        op = m.operate(s, v1, v1 * unb)

        ax = self.fig.add_subplot(2, 2, 1)
        ax.plot(sp, t1, label="T1 (zgodna)")
        ax.plot(sp, t2, color=RED, label="T2 (przeciwna)")
        ax.plot(sp, t1 + t2, color=GREEN, lw=2.4, label="T = T1 + T2")
        ax.axvline(1 - s, color=YELLOW, ls="--", lw=1)
        ax.axhline(0, color=SUB, lw=0.8)
        style_ax(ax, "Moment a prędkość", "prędkość ω/ωs [j.w.]", "moment [j.w.]")
        ax.legend()

        ax = self.fig.add_subplot(2, 2, 2)
        ax.plot(sp, i1, label="|I1| stojana")
        ax.plot(sp, i2, color=RED, label="|I2| stojana")
        ax.axvline(1 - s, color=YELLOW, ls="--", lw=1)
        style_ax(ax, "Prąd składowych a prędkość", "prędkość ω/ωs [j.w.]", "prąd [j.w.]")
        ax.legend()

        pin = (v1 * op["Is1"].conjugate()).real + (v1 * unb * op["Is2"].conjugate()).real
        pscu = m.rs * (abs(op["Is1"]) ** 2 + abs(op["Is2"]) ** 2)
        pag = op["Pag1"]
        prcu = s * pag + abs(op["Ir2"]) ** 2 * m.rr(2 - s)
        pdev = (1 - s) * pag - (1 - s) * op["Pag2"]
        prot = self.g("pcon") * (1 - s) ** 2
        pshaft = pdev - prot
        ax = self.fig.add_subplot(2, 2, 3)
        names = ["P_we", "P_Cu,s", "P_ag1", "P_Cu,r", "P_dev", "P_mech", "P_wał"]
        vals = [pin, pscu, pag, prcu, pdev, prot, pshaft]
        cols = [BLUE, PEACH, CYAN, RED, GREEN, SUB, YELLOW]
        ax.bar(names, vals, color=cols)
        for i, vv in enumerate(vals):
            ax.text(i, vv, f"{vv:.3f}", ha="center", va="bottom", fontsize=7)
        style_ax(ax, f"Bilans mocy przy s = {s:.3f}", "", "moc [j.w.]")
        ax.tick_params(axis="x", labelsize=7)

        ax = self.fig.add_subplot(2, 2, 4)
        self._draw_circuit(ax, m, s)

        eff = pshaft / pin * 100 if pin > 0 else 0
        tmax = (t1 + t2).max()
        return (f"Punkt pracy s = {s:.3f}\n"
                f"T1 = {op['T1']:.3f}  T2 = {op['T2']:.4f}\n"
                f"T  = {op['T']:.3f} j.w.\n"
                f"|I1| = {abs(op['Is1']):.3f}  |I2| = {abs(op['Is2']):.3f}\n"
                f"I2/I1 = {abs(op['Is2']) / max(abs(op['Is1']), 1e-9) * 100:.1f} %\n"
                f"P_we = {pin:.3f}  P_wał = {pshaft:.3f}\n"
                f"η ≈ {eff:.1f} %\n"
                f"Moment rozruchowy T(s=1) = {(t1 + t2)[0]:.2f}\n"
                f"Moment maksymalny = {tmax:.2f}\n"
                f"Prąd rozruchowy |I1(s=1)| = {i1[0]:.2f}")

    def _draw_circuit(self, ax, m, s):
        """Uproszczony rysunek obwodu Steinmetza z wartościami."""
        ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
        ax.set_title("Obwód zastępczy Steinmetza (składowa zgodna)")
        ax.plot([0.5, 9.5], [5, 5], color=SUB); ax.plot([0.5, 9.5], [1, 1], color=SUB)

        def box(x, y, w, h, txt, col):
            ax.add_patch(Rectangle((x, y), w, h, fc=BG2, ec=col, lw=2))
            ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", color=col, fontsize=8)
        box(1.0, 4.6, 1.5, 0.8, f"Rs\n{m.rs:.3f}", PEACH)
        box(2.8, 4.6, 1.5, 0.8, f"jXs\n{m.xs:.3f}", CYAN)
        ax.plot([5, 5], [1, 5], color=SUB)
        box(4.4, 2.5, 1.2, 1.0, f"jXm\n{m.xm:.2f}", MAUVE)
        box(6.0, 4.6, 1.5, 0.8, f"jXr\n{m.xr:.3f}", CYAN)
        ax.plot([8.5, 8.5], [1, 5], color=SUB)
        box(7.8, 2.3, 1.5, 1.4, f"Rr/s\n{m.rr(s) / s:.3f}", RED)
        ax.text(0.3, 3, "V1", color=GREEN, fontsize=11, weight="bold")
        ax.text(5, 0.3, f"Rr(s)=Rr0+ΔRr·s = {m.rr(s):.3f}   (s={s:.3f})", ha="center", color=SUB, fontsize=8)


# =============================================================================
# 2. ROZRUCH I UTYK (12.2.3 + 12.5)
# =============================================================================

class StartTab(Tab):
    title = "2 · Rozruch i utyk"
    explanation = """
# Rozruch, zablokowany wirnik i utyk (rozdz. 12.2.3 i 12.5)
## Równanie ruchu (równ. 12.17)
Wał silnika przyspiesza tylko wtedy, gdy moment silnika T_e jest większy od momentu obciążenia T_L. W jednostkach względnych:
= 2H · dω/dt = T_e(ω) − T_L(ω)
H to stała bezwładności [s] – ile sekund silnik mógłby oddawać moc znamionową z samej energii kinetycznej wirującej masy. Duży wentylator = duże H = długi rozruch.
> Rozpędzanie ciężkiej karuzeli: im jest cięższa (H) i im słabiej pchasz (mały zapas momentu T_e − T_L), tym dłużej trwa rozpędzanie.
## Dlaczego rozruch jest groźny?
Przy postoju (s = 1) silnik pobiera 4–8 razy prąd znamionowy. Straty to I²R, więc rosną 25–36 razy (przykład 12.1)! Stojący wirnik nie ma też wentylacji. Typowy silnik wytrzyma zablokowany wirnik tylko ok. 20 s (rozdz. 12.5.1).
## Model cieplny wirnika (Zocholl, rozdz. 12.4.4)
Grzanie wirnika q = I1²·Rr(s) + I2²·Rr(2−s). Pojemność cieplną C dobieramy tak, by przy zablokowanym wirniku (U = 1, stan zimny) „termometr” θ doszedł do 100 % dokładnie po czasie wytrzymałości t_sw:
= C · dθ/dt = q − θ/R    →    θ = 100 %  ⇒  zadziałanie 49
Zauważ, że Rr zależy od poślizgu – przy rozruchu jest większe (wypieranie prądu), więc wirnik grzeje się szybciej niż „wynikałoby z prądu”.
## Rozróżnienie: normalny rozruch czy utyk? (równ. 12.74–12.75)
= t_st < t_sw  → wystarczy zabezpieczenie czasowe niezależne (51LR)
= t_st ≥ t_sw  → potrzebny czujnik prędkości (bo zwykły przekaźnik nie odróżni)
Nastawy w aplikacji (jak w przykładzie 12.3):
• 51LR – prąd > 2·In dłużej niż 15 s w czasie rozruchu → wyłącz (czas > czas rozruchu, < czas wytrzymałości na zimno).
• 48 – utyk po zakończonym rozruchu: prąd > 2·In dłużej niż 6,5 s → wyłącz (czas > czas rozruchu, < czas wytrzymałości na gorąco).
• 49 – model cieplny, zawsze czuwa.
## Kalkulator rozruchu (równ. 12.73, jednostki amerykańskie)
= I_st = 1000 · (kVA/KM) · KM / (√3 · U)
= t_st = WK² · Δn / (308 · T_śr)      [s; WK² w lb·ft², T w lb·ft]
Litera kodu NEMA na tabliczce podaje kVA/KM (np. G: 5,6–6,3).
## Eksperymenty do wykonania
1. Ustaw U = 0,75 j.w. – rozruch wydłuża się (moment ∝ U²!); przy U ≈ 0,62 j.w. rozruch trwa ponad 15 s i zadziała 51LR.
2. Zaznacz „zablokowany wirnik” – 51LR wyłączy po 15 s, zanim wirnik się spali (θ < 100 %).
3. Ustaw zakleszczenie przy t = 10 s – po rozruchu zadziała 48 (utyk) po 6,5 s.
4. Stan początkowy 70 % (gorący silnik) + zablokowany wirnik – zdąży zadziałać 49?
5. Asymetria 5 % – zobacz, jak rośnie grzanie wirnika.
! Nastawa czasu 51LR musi leżeć w „oknie”: dłużej niż najdłuższy normalny rozruch (przy najniższym napięciu!), krócej niż czas wytrzymałości wirnika. Jeśli okna nie ma (t_st ≥ t_sw) – stosuje się czujnik obrotów.
"""

    def build(self):
        self.section("Zasilanie")
        self.slider("v", "Napięcie U", 0.6, 1.1, 1.0, "j.w.")
        self.slider("unb", "Asymetria V2/V1", 0.0, 10.0, 0.0, "%")
        self.section("Obciążenie i mechanika")
        self.slider("tl0", "Moment obciążenia T_L(ω=1)", 0.1, 1.2, 0.8, "j.w.")
        self.check("fan", "Obciążenie wentylatorowe (T_L ∝ ω²)", True)
        self.slider("h", "Stała bezwładności H", 0.2, 8.0, 4.0, "s")
        self.slider("jam", "Zakleszczenie przy t (0 = brak)", 0.0, 20.0, 0.0, "s", "{:.1f}")
        self.check("locked", "Zablokowany wirnik od początku", False)
        self.slider("th0", "Początkowy stan cieplny", 0.0, 90.0, 0.0, "%", "{:.0f}")
        self.section("Nastawy zabezpieczeń")
        self.slider("tsw", "Wytrzymałość zablok. wirnika (zimny)", 5.0, 40.0, 20.0, "s", "{:.1f}")
        self.slider("iset", "Prąd rozruchu 51LR/48 (×In)", 1.5, 4.0, 2.0, "j.w.")
        self.slider("tlr", "Czas 51LR", 2.0, 30.0, 15.0, "s", "{:.1f}")
        self.slider("tst", "Czas 48 (utyk)", 1.0, 15.0, 6.5, "s", "{:.1f}")
        self.section("Kalkulator 12.73")
        self.slider("hp", "Moc", 1.0, 200.0, 13.0, "KM", "{:.0f}")
        self.slider("vv", "Napięcie", 200.0, 690.0, 230.0, "V", "{:.0f}")
        self.slider("kva", "kVA/KM (kod NEMA)", 3.0, 10.0, 6.0, "")
        self.slider("wk2", "WK² silnik + obciążenie", 1.0, 200.0, 50.0, "lb·ft²", "{:.0f}")
        self.slider("rpm", "Prędkość", 700.0, 3600.0, 1750.0, "obr/min", "{:.0f}")
        self.slider("tacc", "Średni moment przyspieszający", 5.0, 400.0, 58.0, "lb·ft", "{:.0f}")

    def draw(self):
        m = MotorPU()
        r = simulate_start(m, v=self.g("v"), unb=self.g("unb"), tl0=self.g("tl0"),
                           fan=bool(self.v["fan"].get()), h=self.g("h"), jam_t=self.g("jam"),
                           locked=bool(self.v["locked"].get()), theta0=self.g("th0") / 100,
                           t_sw=self.g("tsw"), i_set=self.g("iset"), t_lr=self.g("tlr"),
                           t_stall=self.g("tst"), i_sc=1.25 * m.operate(1, 1.1)["Imax"])
        t = r["t"]
        trip = r["trip"]
        ax1 = self.fig.add_subplot(2, 2, 1)
        ax1.plot(t, r["w"], color=CYAN, label="ω / ωs")
        style_ax(ax1, "Prędkość wirnika", "czas [s]", "ω [j.w.]")
        ax1.set_ylim(-0.05, 1.05)
        ax2 = self.fig.add_subplot(2, 2, 2, sharex=ax1)
        ax2.plot(t, r["i"], color=PEACH, label="prąd max faz")
        if self.g("unb") > 0:
            ax2.plot(t, r["i2"], color=RED, lw=1, label="I2")
        ax2.axhline(self.g("iset"), color=YELLOW, ls="--", lw=1, label="próg 51LR/48")
        style_ax(ax2, "Prąd stojana", "czas [s]", "I [j.w.]")
        ax3 = self.fig.add_subplot(2, 2, 3, sharex=ax1)
        ax3.plot(t, r["th"] * 100, color=MAUVE, label="θ wirnika")
        ax3.axhline(100, color=RED, ls="--", lw=1, label="próg 49")
        style_ax(ax3, "Wykorzystanie pojemności cieplnej wirnika", "czas [s]", "θ [%]")
        for ax in (ax1, ax2, ax3):
            if trip:
                ax.axvline(trip[0], color=RED, lw=1.5)
            if r["t_run"]:
                ax.axvline(r["t_run"], color=GREEN, lw=1, ls=":")
        ax2.legend(loc="upper right"); ax3.legend(loc="upper left")
        if trip:
            ax1.text(trip[0], 0.5, "  WYŁĄCZENIE\n  " + trip[1], color=RED, fontsize=8)

        ax4 = self.fig.add_subplot(2, 2, 4)
        sp = np.linspace(0, 0.999, 300)
        te = [m.operate(1 - w, self.g("v"), self.g("v") * self.g("unb") / 100)["T"] for w in sp]
        tl = self.g("tl0") * (0.1 + 0.9 * sp ** 2) if self.v["fan"].get() else np.full_like(sp, self.g("tl0"))
        ax4.plot(sp, te, color=GREEN, label="T_e silnika")
        ax4.plot(sp, tl, color=RED, label="T_L obciążenia")
        ax4.fill_between(sp, tl, te, where=np.array(te) > tl, color=GREEN, alpha=0.12, label="zapas → przyspieszanie")
        style_ax(ax4, "Moment silnika i obciążenia", "ω [j.w.]", "T [j.w.]")
        ax4.legend(fontsize=7)

        # kalkulator 12.73
        hp, vv = self.g("hp"), self.g("vv")
        i_st = 1000 * self.g("kva") * hp / (SQ3 * vv)
        t_st = self.g("wk2") * self.g("rpm") / (308 * self.g("tacc"))
        t_rated_lbft = 5252 * hp / self.g("rpm")
        verdict = "t_st < t_sw → wystarczy 51LR" if t_st < self.g("tsw") else "t_st ≥ t_sw → potrzebny czujnik obrotów!"
        txt = []
        if r["t_run"]:
            txt.append(f"Rozruch zakończony po {r['t_run']:.2f} s")
        else:
            txt.append("Rozruch NIE zakończył się")
        txt.append(f"Prąd rozruchowy: {r['i'][1]:.2f} j.w.")
        txt.append(f"θ max wirnika: {r['th'].max() * 100:.1f} %")
        txt.append("ZADZIAŁAŁO: " + (f"{trip[1]}\n  po {trip[0]:.2f} s" if trip else "nic – praca OK"))
        txt.append("")
        txt.append("Kalkulator 12.73:")
        txt.append(f"  I_st = {i_st:.1f} A")
        txt.append(f"  T_zn = {t_rated_lbft:.1f} lb·ft")
        txt.append(f"  t_st = {t_st:.2f} s")
        txt.append(f"  {verdict}")
        return "\n".join(txt)


# =============================================================================
# 3. MODEL CIEPLNY (12.4)
# =============================================================================

class ThermalTab(Tab):
    title = "3 · Model cieplny"
    explanation = """
# Zabezpieczenie przeciążeniowe i model cieplny (rozdz. 12.4)
## Analogia elektryczno-cieplna (rys. 12.5)
Ciepło płynie jak prąd, a temperatura zachowuje się jak napięcie:
= moc strat P [W]         ↔ prąd źródła I [A]
= temperatura θ [°C]      ↔ napięcie U [V]
= pojemność cieplna C [J/K] ↔ kondensator C [F]   (C = m·c_p, równ. 12.51)
= opór cieplny R [K/W]     ↔ rezystor R [Ω]
> Garnek z wodą na kuchence: moc palnika to P, masa wody to pojemność C, a to, jak szybko garnek oddaje ciepło do kuchni, to opór R. Po długim czasie temperatura ustala się na θ = θ_ot + P·R.
## Model 3-węzłowy (równ. 12.44–12.46)
= Cs·dθs/dt = PsL − (θs−θc)/Rsc                  uzwojenie stojana
= Cc·dθc/dt = PcL + (θs−θc)/Rsc − (θc−θa)/Rca    rdzeń (żelazo)
= Cr·dθr/dt = PrL − (θr−θa)/Rra                   wirnik (odsprzężony)
Straty w miedzi rosną z kwadratem prądu (I²R), straty w żelazie z kwadratem napięcia.
Przyjęto: silnik ~10 kW, straty znamionowe Cu stojana 600 W, żelaza 300 W, Cu wirnika 400 W; Cs=3000, Cc=25000, Cr=6000 J/K; Rsc=0,03, Rca=0,07, Rra=0,2 K/W.
## Przekaźnik cieplny 49 (równ. 12.67–12.71)
Przekaźnik liczy „wirtualną temperaturę” z prądu. Uwzględnia też składową przeciwną, która grzeje wirnik K ≈ 3 razy mocniej:
= θ(t) = θmax·(1 − e^(−t/τ))
= I_eq = √(I1² + K·I2²)                          (12.70)
= t = τ·ln[(k² − A)/(k² − 1)],   k = I_eq/I_th     (12.71)
A = 0 dla silnika zimnego, A ≈ (I_obc/I_th)² dla ciepłego – ciepły silnik wyłącza się szybciej, bo ma mniejszy „zapas”.
## Żywotność izolacji (rys. 12.6, 12.7)
Reguła 10 stopni (Montsingera): każde +10 °C ponad dopuszczalną temperaturę klasy izolacji skraca życie uzwojenia o połowę. Już 5 % za dużo prądu daje ok. +10 °C!
= L/L0 = 2^((θ_klasy − θ)/10)       klasa B: 130 °C, F: 155 °C, H: 180 °C
## Eksperymenty
1. Obciążenie 1,0 – temperatury ustalają się poniżej 155 °C.
2. Przeciążenie 1,3 przez 60 min – przekaźnik 49 wyłączy silnik (pionowa czerwona linia), a temperatura zacznie spadać.
3. Asymetria 3,5 % – straty rosną o 12,5 % (tabela 12.2), temperatura rośnie, przekaźnik reaguje szybciej, bo I_eq > I1.
! Stała czasowa τ przekaźnika powinna odpowiadać stałej czasowej uzwojenia z karty katalogowej. Za duża τ = przekaźnik „nie nadąża” za grzaniem; za mała = zbędne wyłączenia przy rozruchach.
"""

    def build(self):
        self.section("Cykl obciążenia")
        self.slider("k1", "Obciążenie bazowe I/In", 0.2, 1.2, 1.0, "j.w.", "{:.2f}")
        self.slider("k2", "Przeciążenie I/In (od 60 min)", 0.5, 2.0, 1.3, "j.w.", "{:.2f}")
        self.slider("tov", "Czas przeciążenia", 0.0, 150.0, 60.0, "min", "{:.0f}")
        self.slider("v", "Napięcie", 0.85, 1.15, 1.0, "j.w.", "{:.2f}")
        self.slider("unb", "Asymetria napięć", 0.0, 5.0, 0.0, "%", "{:.1f}")
        self.slider("amb", "Temperatura otoczenia", 0.0, 50.0, 40.0, "°C", "{:.0f}")
        self.section("Przekaźnik 49")
        self.slider("tau", "Stała czasowa τ", 300.0, 3000.0, 900.0, "s", "{:.0f}")
        self.slider("ith", "I_th (nastawa cieplna)", 1.0, 1.3, 1.05, "×In", "{:.2f}")
        self.slider("K", "K = Rr2/Rr1", 1.0, 8.0, 3.0, "", "{:.1f}")
        self.slider("A", "Stan początkowy A (gorący)", 0.0, 0.95, 0.7, "", "{:.2f}")

    def draw(self):
        r = simulate_thermal3(self.g("k1"), self.g("k2"), self.g("tov"), self.g("v"), self.g("unb"),
                              self.g("amb"), 4.0, self.g("tau"), self.g("ith"), self.g("K"))
        ax = self.fig.add_subplot(2, 2, 1)
        ax.plot(r["t"], r["ts"], label="θs uzwojenie")
        ax.plot(r["t"], r["tc"], label="θc rdzeń")
        ax.plot(r["t"], r["tr"], label="θr wirnik")
        ax.axhline(155, color=RED, ls="--", lw=1, label="klasa F 155 °C")
        ax.axhline(130, color=YELLOW, ls=":", lw=1, label="klasa B 130 °C")
        if r["trip"]:
            ax.axvline(r["trip"], color=RED)
        style_ax(ax, "Temperatury (model 3-węzłowy)", "czas [min]", "θ [°C]")
        ax.legend(fontsize=7, loc="lower right")

        ax = self.fig.add_subplot(2, 2, 2)
        ax.plot(r["t"], r["thp"] * 100, color=MAUVE, label="stan cieplny 49")
        ax.plot(r["t"], r["k"] * 100, color=SUB, lw=1, label="obciążenie I/In ×100")
        ax.axhline(100, color=RED, ls="--", lw=1, label="próg zadziałania")
        if r["trip"]:
            ax.axvline(r["trip"], color=RED)
            ax.text(r["trip"], 50, f" TRIP {r['trip']:.1f} min", color=RED)
        style_ax(ax, "Przekaźnik cieplny 49", "czas [min]", "[%]")
        ax.legend(fontsize=7)

        ax = self.fig.add_subplot(2, 2, 3)
        k = np.logspace(np.log10(1.02), np.log10(10), 300)
        tau, a = self.g("tau"), self.g("A")
        ax.loglog(k, thermal_trip_time(k, tau, 0.0), label="stan zimny (A=0)")
        ax.loglog(k, thermal_trip_time(k, tau, a), color=RED, label=f"stan gorący (A={a:.2f})")
        k_now = self.g("k2") / self.g("ith")
        if k_now > 1:
            tt = float(thermal_trip_time(k_now, tau, 0))
            ax.plot([k_now], [tt], "o", color=YELLOW, ms=8, label=f"I/I_th={k_now:.2f}: {tt:.0f} s")
        style_ax(ax, "Charakterystyka t(I) przekaźnika 49 (12.71)", "k = I_eq / I_th", "czas [s]")
        ax.legend(fontsize=7)

        ax = self.fig.add_subplot(2, 2, 4)
        temps = np.linspace(100, 200, 200)
        ax.semilogy(temps, relative_life(temps, 155) * 100, color=GREEN, label="klasa F")
        ax.semilogy(temps, relative_life(temps, 130) * 100, color=YELLOW, label="klasa B")
        tmax = r["ts"].max()
        ax.plot([tmax], [relative_life(tmax, 155) * 100], "o", color=RED, ms=8,
                label=f"θs max={tmax:.0f} °C")
        style_ax(ax, "Żywotność izolacji (reguła 10 °C)", "temperatura uzwojenia [°C]", "żywotność [% nominalnej]")
        ax.legend(fontsize=7)

        life = relative_life(tmax, 155) * 100
        return (f"θs max = {r['ts'].max():.1f} °C\n"
                f"θc max = {r['tc'].max():.1f} °C\n"
                f"θr max = {r['tr'].max():.1f} °C\n"
                f"Żywotność przy θs max: {life:.0f} % (kl. F)\n"
                f"Przekaźnik 49: " + (f"WYŁĄCZENIE po {r['trip']:.1f} min" if r["trip"] else "nie zadziałał") +
                f"\nt(I) zimny przy {self.g('k2'):.2f}·In: "
                f"{float(thermal_trip_time(self.g('k2') / self.g('ith'), tau)):.0f} s\n"
                f"t(I) gorący: {float(thermal_trip_time(self.g('k2') / self.g('ith'), tau, a)):.0f} s")


# =============================================================================
# 4. PRZYKŁADY 12.1 - 12.3
# =============================================================================

class Ex123Tab(Tab):
    title = "4 · Przykł. 12.1–12.3"
    explanation = """
# Przykład 12.1 – prądy silnika 13 KM, 230 V
Dane: P = 13 KM, U = 230 V, 3 fazy. Szukamy prądu znamionowego, jałowego i rozruchowego.
## Krok 1: prąd znamionowy (wzór Wildiego, równ. 12.88)
= I_n ≈ 600 · P[KM] / U[V] = 600 · 13 / 230 ≈ 33,9 A
Skąd 600? Z definicji: I = P/(√3·U·η·cosφ) = 746·P/(√3·U·η·cosφ). Dla typowego małego silnika η·cosφ ≈ 0,72, więc 746/(1,732·0,72) ≈ 600.
> Sprawdzenie „na chłopski rozum”: 13 KM ≈ 9,7 kW. Przy 230 V w trzech fazach to √3·230 ≈ 400 V·A na każdy amper. 9700 W / 400 ≈ 24 A – ale silnik ma straty (η) i pobiera moc bierną (cosφ), więc prąd jest większy: ≈ 34 A. Pasuje!
## Krok 2: prąd jałowy i rozruchowy (tabela 12.3, mały silnik < 11 kW)
= I_0  = 0,5 · I_n        ≈ 17,0 A
= I_lr = (5…6) · I_n       ≈ 170 … 203 A
Na biegu jałowym silnik pobiera dużo prądu (połowę znamionowego!) – to prąd magnesujący, potrzebny do wytworzenia pola. Przy rozruchu prąd jest 5–6 razy większy, a straty I²R aż 25–36 razy większe.
# Przykład 12.2 – nastawa zabezpieczenia zwarciowego (50)
Zabezpieczenie zwarciowe nie może zadziałać przy rozruchu, więc nastawiamy je z zapasem 25 % powyżej prądu rozruchowego (rozdz. 12.6):
= I_sc = 1,25 · I_st
= dla 5·In:  I_sc = 1,25 · 169,6 ≈ 212 A  →  wtórnie 212/40 ≈ 5,30 A
= dla 6·In:  I_sc = 1,25 · 203,5 ≈ 254 A  →  wtórnie 254/40 ≈ 6,36 A
Przekładnik prądowy (CT) 40/1 zamienia 40 A w obwodzie silnika na 1 A dla przekaźnika – przekaźnik „widzi” zmniejszoną kopię prądu.
Opóźnienie: 100 ms dla prądów ≤ 120 % nastawy (aby nie reagować na chwilowe przeskoki i nasycenie CT) oraz 40 ms powyżej.
# Przykład 12.3 – doziemienie, zablokowany wirnik, utyk
## Doziemienie 50N (25 % prądu znamionowego)
= I_0> = 0,25 · I_n = 0,25 · 33,9 ≈ 8,5 A  →  wtórnie ≈ 0,21 A
## Zablokowany wirnik / za długi rozruch (51LR) i utyk (48)
Prąd nastawczy musi być mniejszy od prądu rozruchowego (żeby „widzieć” rozruch), ale większy od prądu pracy: w książce 2·I_n.
! Uwaga na niespójność w książce: podaje 2·I_n = 80 A, czyli przyjmuje I_n = 40 A (to prąd pierwotny CT 40/1). Z przykładu 12.1 wychodzi I_n ≈ 33,9 A, więc 2·I_n ≈ 68 A. Oba ustawienia są poprawne co do zasady – ważne, by próg leżał między prądem roboczym a rozruchowym (170 A).
Czasy (typowy układ):
= t_rozruchu (~5 s)  <  t_51LR = 15 s  <  t_wytrzymałości na zimno (~20 s)
= t_rozruchu (~5 s)  <  t_48  = 6,5 s  <  t_wytrzymałości na gorąco
Zabezpieczenie przed utykiem (48) działa dopiero PO udanym rozruchu, dlatego może mieć krótszy czas – musi zdążyć przed czasem wytrzymałości silnika gorącego.
## Wykres TCC (czas–prąd)
Na wykresie log–log widzisz: krzywą rozruchu silnika (prąd rozruchowy przez czas rozruchu, potem prąd znamionowy), krzywe przekaźnika cieplnego 49 (zimny/gorący), nastawy 51LR, 48, 50 i 50N oraz punkty wytrzymałości zablokowanego wirnika. Poprawna koordynacja: krzywa rozruchu leży POD i NA LEWO od wszystkich charakterystyk wyłączających, a punkty wytrzymałości – NAD charakterystykami zabezpieczeń.
## Rezystor stabilizujący w obwodzie 50N (równ. 12.76)
Przy rozruchu przekładniki mogą się nasycić nierówno, a przekaźnik doziemny „zobaczy” fałszywy prąd. Rezystor szeregowy temu zapobiega:
= R_stab = I_st·(R_ct + k1·R_l) / I_0     (k1 = 1 gwiazda przy CT, 2 przy przekaźniku)
"""

    def build(self):
        self.section("Silnik (przykład 12.1)")
        self.slider("hp", "Moc", 1.0, 50.0, 13.0, "KM", "{:.0f}")
        self.slider("v", "Napięcie", 200.0, 690.0, 230.0, "V", "{:.0f}")
        self.slider("klr", "Krotność prądu rozruch. I_lr/In", 4.0, 8.0, 5.0, "", "{:.2f}")
        self.slider("tst", "Czas rozruchu", 1.0, 15.0, 5.0, "s", "{:.1f}")
        self.slider("tcold", "Wytrzymałość zablok. (zimny)", 5.0, 40.0, 20.0, "s", "{:.1f}")
        self.slider("thot", "Wytrzymałość zablok. (gorący)", 3.0, 30.0, 8.0, "s", "{:.1f}")
        self.section("Zabezpieczenia (12.2-12.3)")
        self.slider("ct", "Przekładnia CT (x/1)", 20.0, 200.0, 40.0, "A", "{:.0f}")
        self.slider("msc", "Zapas członu 50", 1.1, 2.0, 1.25, "×I_st", "{:.2f}")
        self.slider("ef", "Nastawa 50N", 5.0, 50.0, 25.0, "% In", "{:.0f}")
        self.slider("ilr", "Prąd 51LR/48", 1.5, 4.0, 2.0, "×In", "{:.2f}")
        self.slider("tlr", "Czas 51LR", 2.0, 30.0, 15.0, "s", "{:.1f}")
        self.slider("tstall", "Czas 48", 1.0, 20.0, 6.5, "s", "{:.1f}")
        self.slider("tau", "τ przekaźnika 49", 200.0, 3000.0, 900.0, "s", "{:.0f}")
        self.slider("A", "Stan gorący A", 0.0, 0.95, 0.5, "", "{:.2f}")
        self.section("Rezystor stabilizujący (12.76)")
        self.slider("rct", "R_ct uzwojenia CT", 0.1, 5.0, 0.5, "Ω", "{:.2f}")
        self.slider("rl", "R_l przewodu", 0.05, 3.0, 0.3, "Ω", "{:.2f}")
        self.check("k1", "Punkt gwiazdy przy przekaźniku (k1 = 2)", False)

    def draw(self):
        mc = motor_currents(self.g("hp"), self.g("v"), 0.5, self.g("klr"), self.g("klr"))
        i_n, i_lr = mc["In"], mc["Ilr_min"]
        ct = self.g("ct")
        i_sc, i_sc2 = sc_setting(i_lr, self.g("msc"), ct)
        i_ef = self.g("ef") / 100 * i_n
        i_lrset = self.g("ilr") * i_n
        tst, tlr, tstall = self.g("tst"), self.g("tlr"), self.g("tstall")
        tcold, thot = self.g("tcold"), self.g("thot")
        tau, a = self.g("tau"), self.g("A")
        ith = 1.05 * i_n

        gs = self.fig.add_gridspec(2, 2)
        ax = self.fig.add_subplot(gs[:, 0])
        # krzywa rozruchu silnika
        ax.loglog([i_n * 0.6, i_n, i_n, i_lr, i_lr], [1e4, 1e4, tst, tst, 0.01], color=GREEN, lw=2.5,
                  label=f"rozruch silnika ({i_lr:.0f} A, {tst:.1f} s)")
        cur = np.logspace(np.log10(ith * 1.01), np.log10(i_sc * 3), 300)
        ax.loglog(cur, thermal_trip_time(cur / ith, tau, 0), color=CYAN, label="49 zimny")
        ax.loglog(cur, thermal_trip_time(cur / ith, tau, a), color=BLUE, ls="--", label="49 gorący")
        ax.loglog([i_lrset, i_lrset, i_sc * 3], [1e4, tlr, tlr], color=YELLOW, label=f"51LR {i_lrset:.0f} A / {tlr:.1f} s")
        ax.loglog([i_lrset, i_lrset, i_sc * 3], [1e4, tstall, tstall], color=PEACH, ls="--",
                  label=f"48 utyk {tstall:.1f} s (po rozruchu)")
        ax.loglog([i_sc, i_sc, i_sc * 1.2, i_sc * 1.2, i_sc * 3], [1e4, 0.1, 0.1, 0.04, 0.04], color=RED,
                  lw=2.2, label=f"50 zwarciowe {i_sc:.0f} A")
        ax.loglog([i_ef, i_ef], [1e4, 0.01], color=MAUVE, ls=":", label=f"50N doziemne {i_ef:.1f} A")
        ax.plot([i_lr], [tcold], "s", color=PINK, ms=8, label=f"wytrzym. zimny {tcold:.0f} s")
        ax.plot([i_lr], [thot], "^", color=PINK, ms=8, label=f"wytrzym. gorący {thot:.0f} s")
        ax.set_xlim(i_ef * 0.6, i_sc * 3)
        ax.set_ylim(0.01, 1e4)
        style_ax(ax, "Koordynacja czas–prąd (TCC)", "prąd pierwotny [A]", "czas [s]")
        ax.legend(fontsize=6.5, loc="upper right")

        ax = self.fig.add_subplot(gs[0, 1])
        names = ["I0", "In", "2In", "Ilr 5×", "Ilr 6×", "Isc 5×", "Isc 6×"]
        vals = [0.5 * i_n, i_n, 2 * i_n, 5 * i_n, 6 * i_n, self.g("msc") * 5 * i_n, self.g("msc") * 6 * i_n]
        cols = [SUB, GREEN, YELLOW, PEACH, PEACH, RED, RED]
        ax.bar(names, vals, color=cols)
        for i, vv in enumerate(vals):
            ax.text(i, vv, f"{vv:.0f}", ha="center", va="bottom", fontsize=7)
        style_ax(ax, "Prądy silnika i nastawy [A]", "", "A")
        ax.tick_params(axis="x", labelsize=7)

        # kontrola koordynacji
        checks = [
            ("51LR: t_lr > t_rozruchu", tlr > tst),
            ("51LR: t_lr < wytrzym. zimny", tlr < tcold),
            ("48: t_48 > t_rozruchu", tstall > tst),
            ("48: t_48 < wytrzym. gorący", tstall < thot),
            ("51LR: In < I_set < I_lr", i_n < i_lrset < i_lr),
            ("50: I_sc > I_lr", i_sc > i_lr),
            ("49 zimny nie zadziała w rozruchu", float(thermal_trip_time(i_lr / ith, tau, 0)) > tst),
            ("49 gorący nie zadziała w rozruchu", float(thermal_trip_time(i_lr / ith, tau, a)) > tst),
        ]
        ax = self.fig.add_subplot(gs[1, 1])
        ax.axis("off")
        ax.set_title("Kontrola nastaw")
        for i, (name, ok) in enumerate(checks):
            ax.text(0.02, 0.92 - i * 0.12, ("✔ " if ok else "✘ ") + name, color=GREEN if ok else RED,
                    fontsize=9, transform=ax.transAxes)

        k1 = 2 if self.v["k1"].get() else 1
        i_st_sec = i_lr / ct
        i0_sec = i_ef / ct
        r_stab = i_st_sec * (self.g("rct") + k1 * self.g("rl")) / i0_sec
        return (f"12.1  In = 600·P/U = {i_n:.2f} A\n"
                f"      I0 = 0,5·In = {0.5 * i_n:.2f} A\n"
                f"      Ilr = 5..6·In = {5 * i_n:.1f}..{6 * i_n:.1f} A\n"
                f"12.2  Isc(5×) = {self.g('msc') * 5 * i_n:.1f} A → {self.g('msc') * 5 * i_n / ct:.3f} A wt.\n"
                f"      Isc(6×) = {self.g('msc') * 6 * i_n:.1f} A → {self.g('msc') * 6 * i_n / ct:.3f} A wt.\n"
                f"      (dla bieżącej krotności: {i_sc:.1f} A → {i_sc2:.3f} A)\n"
                f"12.3  50N = {i_ef:.2f} A → {i_ef / ct:.3f} A wt.\n"
                f"      51LR/48 = {i_lrset:.1f} A → {i_lrset / ct:.2f} A wt.\n"
                f"      t_51LR = {tlr:.1f} s, t_48 = {tstall:.1f} s\n"
                f"49: t(Ilr) zimny = {float(thermal_trip_time(i_lr / ith, tau)):.1f} s,"
                f" gorący = {float(thermal_trip_time(i_lr / ith, tau, a)):.1f} s\n"
                f"R_stab = {r_stab:.1f} Ω (12.76)\n"
                f"Koordynacja: {sum(ok for _, ok in checks)}/{len(checks)} OK")


# =============================================================================
# 5. PRZYKŁADY 12.4 - 12.5
# =============================================================================

class DecayTab(Tab):
    title = "5 · Przykł. 12.4–12.5"
    explanation = """
# Przykład 12.4 – zanik napięcia po odłączeniu zasilania
Dane: U = 400 V, f = 50 Hz, t = 500 ms, reaktancja wirnika x2 = 0,31 Ω, rezystancja wirnika r2 = 0,26 Ω, reaktancja magnesowania xm = 31 Ω.
## Dlaczego napięcie nie znika od razu?
Gdy na chwilę zniknie zasilanie (np. zapad w sieci, przełączenie), wirnik dalej się kręci, a w jego klatce płyną prądy, które podtrzymują pole magnetyczne. Silnik przez chwilę działa jak generator – na zaciskach jest „napięcie szczątkowe”, które zanika wykładniczo.
> Jak gasnący dzwon: po uderzeniu dźwięk nie urywa się, tylko cichnie coraz bardziej. Szybkość „cichnięcia” opisuje stała czasowa.
## Krok 1: stała czasowa obwodu otwartego (równ. 12.101)
Obwód wirnika to indukcyjność (L = x/ω) i rezystancja r2. Stała czasowa obwodu RL: τ = L/R.
= T_op = (xm + x2) / (2π·f·r2) = (31 + 0,31) / (2π·50·0,26)
= T_op = 31,31 / 81,68 ≈ 0,383 s
## Krok 2: napięcie przy t = T_op (równ. 12.100)
= V(t) = V0 · e^(−t/T_op)
= V(T_op) = 400 · e^(−1) = 400 · 0,368 ≈ 147,2 V
## Krok 3: napięcie przy t = 0,5 s
= V(0,5) = 400 · e^(−0,5/0,383) = 400 · e^(−1,304) ≈ 400 · 0,271 ≈ 108,5 V
# Przykład 12.5 – napięcie po 4 stałych czasowych
= V(4·T_op) = 400 · e^(−4) = 400 · 0,0183 ≈ 7,33 V
Po 4 stałych czasowych zostaje < 2 % – w praktyce napięcie „zniknęło”. To ogólna zasada dla każdego zjawiska wykładniczego: po 3τ zostaje 5 %, po 5τ < 1 %.
## Po co to inżynierowi?
Jeśli zasilanie wróci, gdy napięcie szczątkowe jest jeszcze duże i przesunięte w fazie, to sieć i silnik „zderzą się” – wielki udar prądu i momentu (nawet 2× moment zwarciowy!), może uszkodzić wał lub sprzęgło. Dlatego:
• automatyka SZR / ponownego załączenia czeka, aż napięcie szczątkowe spadnie poniżej ok. 25–33 % (lub stosuje kontrolę zgodności faz),
• zabezpieczenie podnapięciowe (27) ma opóźnienie 0,5–2 s (rozdz. 12.10.1).
Drugi wykres pokazuje przebieg chwilowy: amplituda maleje, a częstotliwość też spada, bo wirnik zwalnia (to ilustracja – tempo hamowania ustawiasz suwakiem).
! Duża stała T_op mają duże silniki (niska rezystancja wirnika). Dla silników MW napięcie szczątkowe może trwać kilka sekund – trzeba to uwzględnić w automatyce przełączania zasilania.
"""

    def build(self):
        self.section("Dane (przykład 12.4)")
        self.slider("v0", "Napięcie zasilania V0", 100.0, 11000.0, 400.0, "V", "{:.0f}")
        self.slider("f", "Częstotliwość f", 50.0, 60.0, 50.0, "Hz", "{:.0f}")
        self.slider("xm", "Reaktancja magnesowania xm", 5.0, 100.0, 31.0, "Ω", "{:.1f}")
        self.slider("x2", "Reaktancja wirnika x2", 0.05, 2.0, 0.31, "Ω", "{:.2f}")
        self.slider("r2", "Rezystancja wirnika r2", 0.02, 1.0, 0.26, "Ω", "{:.3f}")
        self.slider("t", "Czas t", 0.0, 3.0, 0.5, "s", "{:.2f}")
        self.section("Ponowne załączenie")
        self.slider("safe", "Bezpieczne napięcie", 10.0, 50.0, 25.0, "% V0", "{:.0f}")
        self.slider("dec", "Hamowanie wirnika", 0.0, 50.0, 15.0, "%/s", "{:.0f}")

    def draw(self):
        v0, f = self.g("v0"), self.g("f")
        t = np.linspace(0, 5 * max(0.05, (self.g("xm") + self.g("x2")) / (2 * math.pi * f * self.g("r2"))), 800)
        v, t_op = voltage_decay(v0, t, self.g("xm"), self.g("x2"), self.g("r2"), f)
        ts = self.g("t")
        v_ts = v0 * math.exp(-ts / t_op)
        safe = self.g("safe") / 100
        t_safe = -t_op * math.log(safe)
        ax = self.fig.add_subplot(2, 1, 1)
        ax.plot(t, v, color=CYAN, lw=2.4, label="V(t) = V0·e^(−t/T_op)")
        for mult, col in [(1, GREEN), (4, MAUVE)]:
            ax.plot([mult * t_op], [v0 * math.exp(-mult)], "o", color=col, ms=8,
                    label=f"t = {mult}·T_op = {mult * t_op:.3f} s → {v0 * math.exp(-mult):.2f} V")
        ax.plot([ts], [v_ts], "D", color=YELLOW, ms=8, label=f"t = {ts:.2f} s → {v_ts:.1f} V")
        ax.axhline(safe * v0, color=RED, ls="--", lw=1)
        ax.axvline(t_safe, color=RED, ls=":", lw=1)
        ax.text(t_safe, safe * v0 * 1.1, f" bezpieczne ponowne załączenie\n po {t_safe:.2f} s", color=RED, fontsize=8)
        style_ax(ax, "Zanik napięcia szczątkowego (12.4–12.5)", "czas [s]", "napięcie [V]")
        ax.legend(fontsize=8)

        ax = self.fig.add_subplot(2, 1, 2)
        tw = np.linspace(0, min(t[-1], 4 * t_op), 4000)
        fr = f * np.clip(1 - self.g("dec") / 100 * tw, 0.05, 1)
        phase = 2 * math.pi * np.cumsum(fr) * (tw[1] - tw[0])
        env = v0 * math.sqrt(2) / SQ3 * np.exp(-tw / t_op)
        ax.plot(tw, env * np.sin(phase), color=GREEN, lw=0.9, label="napięcie fazowe chwilowe")
        ax.plot(tw, env, color=PEACH, ls="--", lw=1, label="obwiednia")
        ax.plot(tw, -env, color=PEACH, ls="--", lw=1)
        style_ax(ax, "Przebieg chwilowy (ilustracja – wirnik zwalnia)", "czas [s]", "u(t) [V]")
        ax.legend(fontsize=8, loc="upper right")
        return (f"T_op = (xm+x2)/(2πf·r2)\n     = {t_op:.4f} s\n"
                f"V(T_op)   = {v0 / math.e:.2f} V\n"
                f"V({ts:.2f} s) = {v_ts:.2f} V  ({v_ts / v0 * 100:.1f} %)\n"
                f"V(4·T_op) = {v0 * math.exp(-4):.3f} V\n"
                f"Spadek do {self.g('safe'):.0f} % po {t_safe:.3f} s")


# =============================================================================
# 6. PRZYKŁAD 12.6 - PRĄD RÓWNOWAŻNY
# =============================================================================

class IeqTab(Tab):
    title = "6 · Przykł. 12.6"
    explanation = """
# Przykład 12.6 – prąd równoważny cieplnie
Dane (j.w.): składowa przeciwna I2 = 0,92, składowa zgodna I1 = 0,912, K = Rr2/Rr1 = 3.
## Idea
Przekaźnik cieplny musi wiedzieć, ile ciepła powstaje w silniku. Składowa zgodna grzeje „normalnie”. Składowa przeciwna ma w wirniku częstotliwość ≈ 2f = 100 Hz, prąd wypierany jest na powierzchnię prętów klatki (efekt naskórkowy), rezystancja rośnie K ≈ 3 razy – więc ten sam prąd grzeje 3 razy mocniej.
= I_eq = √(I1² + K·I2²)                  (równ. 12.70)
> Porównanie: bieg po płaskim (I1) i bieg pod górę (I2). Ten sam dystans, ale pod górę męczysz się K razy bardziej. I_eq to „dystans przeliczony na płaski teren”.
## Rozwiązanie
= I1² = 0,912² = 0,8317
= K·I2² = 3 · 0,92² = 3 · 0,8464 = 2,5392
= I_eq = √(0,8317 + 2,5392) = √3,3709 ≈ 1,836 j.w.
Silnik grzeje się jak przy prądzie 1,84 × znamionowy, chociaż amperomierz pokazałby „tylko” ok. 0,9–1,3 j.w. w poszczególnych fazach. Przekaźnik bez uwzględniania I2 zupełnie by to przeoczył!
! Tak duże I2 (≈ I1) oznacza w praktyce praca na dwóch fazach (przerwa fazy). Takie dane w zadaniu służą pokazaniu mechanizmu – przy przerwie fazy przekaźnik 46/49 musi wyłączyć silnik w ciągu sekund.
## Co pokazuje wykres
• Lewy: I_eq w funkcji I2 dla różnych K – im większe K, tym szybciej rośnie I_eq.
• Prawy górny: udział ciepła od I1 i od I2.
• Prawy dolny: jak szybko zadziała przekaźnik 49 (τ = 900 s, zimny) dla danego I_eq.
"""

    def build(self):
        self.section("Dane (j.w.)")
        self.slider("i1", "I1 składowa zgodna", 0.0, 2.0, 0.912, "j.w.", "{:.3f}")
        self.slider("i2", "I2 składowa przeciwna", 0.0, 1.5, 0.92, "j.w.", "{:.3f}")
        self.slider("K", "K = Rr2/Rr1", 1.0, 8.0, 3.0, "", "{:.2f}")
        self.slider("tau", "τ przekaźnika 49", 200.0, 3000.0, 900.0, "s", "{:.0f}")
        self.slider("ith", "I_th", 1.0, 1.3, 1.05, "j.w.", "{:.2f}")

    def draw(self):
        i1, i2, kk = self.g("i1"), self.g("i2"), self.g("K")
        ieq = i_equivalent(i1, i2, kk)
        ax = self.fig.add_subplot(1, 2, 1)
        x = np.linspace(0, 1.5, 200)
        for kv, c in [(1, SUB), (3, CYAN), (6, MAUVE)]:
            ax.plot(x, np.sqrt(i1 ** 2 + kv * x ** 2), color=c, label=f"K = {kv}")
        ax.plot(x, np.sqrt(i1 ** 2 + kk * x ** 2), color=YELLOW, ls="--", label=f"K = {kk:.2f} (bieżące)")
        ax.plot([i2], [ieq], "o", color=RED, ms=9, label=f"I_eq = {ieq:.3f}")
        ax.axhline(1.0, color=GREEN, ls=":", lw=1, label="prąd znamionowy")
        style_ax(ax, f"I_eq = √(I1² + K·I2²),  I1 = {i1:.3f}", "I2 [j.w.]", "I_eq [j.w.]")
        ax.legend(fontsize=8)
        ax = self.fig.add_subplot(2, 2, 2)
        ax.barh(["I1²", "K·I2²", "I_eq²"], [i1 ** 2, kk * i2 ** 2, ieq ** 2], color=[CYAN, RED, YELLOW])
        for i, vv in enumerate([i1 ** 2, kk * i2 ** 2, ieq ** 2]):
            ax.text(vv, i, f" {vv:.3f}", va="center", fontsize=8)
        style_ax(ax, "Udział w grzaniu (∝ I²)", "[j.w.²]", "")
        ax = self.fig.add_subplot(2, 2, 4)
        k = np.linspace(1.02, 4, 300)
        ax.semilogy(k, thermal_trip_time(k, self.g("tau")), color=CYAN)
        kn = ieq / self.g("ith")
        tt = float(thermal_trip_time(kn, self.g("tau")))
        if np.isfinite(tt):
            ax.plot([kn], [tt], "o", color=RED, ms=8)
            ax.text(kn, tt, f"  {tt:.0f} s", color=RED)
        style_ax(ax, "Czas zadziałania 49 (stan zimny)", "I_eq / I_th", "t [s]")
        return (f"I1² = {i1 ** 2:.4f}\nK·I2² = {kk * i2 ** 2:.4f}\n"
                f"I_eq = √({i1 ** 2 + kk * i2 ** 2:.4f})\n     = {ieq:.4f} j.w.\n"
                f"I_eq / I1 = {ieq / max(i1, 1e-9):.2f}\n"
                f"t_49 (zimny) = " + (f"{tt:.0f} s" if np.isfinite(tt) else "brak zadziałania"))


# =============================================================================
# 7. PRZYKŁADY 12.7 - 12.8 - ASYMETRIA
# =============================================================================

class UnbalanceTab(Tab):
    title = "7 · Przykł. 12.7–12.8"
    explanation = """
# Przykład 12.7 – asymetria napięć
Dane: U_AB = 371 V, U_BC = 374 V, U_CA = 395 V; prądy 18,3 A, 16 A, 12 A; kąty fazowe przyjmujemy symetryczne (co 120°).
## Krok 1: średnia i procentowa asymetria (NEMA, równ. 12.105–12.108)
= U_śr = (371 + 374 + 395) / 3 = 380 V
= max odchyłka = |395 − 380| = 15 V
= asymetria % = 15 / 380 · 100 ≈ 3,95 %
## Krok 2: składowe symetryczne (równ. 12.7, operator a = 1∠120°)
= V1 = (V_AB + a·V_BC + a²·V_CA)/3
= V2 = (V_AB + a²·V_BC + a·V_CA)/3
Gdy kąty są dokładnie co 120°, V1 to po prostu średnia: V1 = 380 V, a V2 ≈ 7,55 V.
= VUF = |V2| / |V1| · 100 ≈ 1,99 %
## Ważna uwaga inżynierska
Trzy napięcia międzyfazowe zawsze tworzą zamknięty trójkąt (V_AB + V_BC + V_CA = 0). Jeśli moduły są różne, to kąty NIE mogą być dokładnie co 120° – założenie z zadania jest uproszczeniem. Dokładny VUF liczony z samych modułów (wzór IEC z parametrem β) wynosi ≈ 4,01 % – dwa razy więcej! Aplikacja pokazuje obie wartości, a lewy wykres rysuje rzeczywisty (zamknięty) trójkąt napięć.
= β = (U_AB⁴ + U_BC⁴ + U_CA⁴) / (U_AB² + U_BC² + U_CA²)²
= VUF = √[(1 − √(3 − 6β)) / (1 + √(3 − 6β))]
> Trzy patyki różnej długości zawsze da się złożyć w trójkąt – ale wtedy kąty nie mogą być równe. To samo z napięciami!
# Przykład 12.8 – prąd równoważny dla prądów z 12.7
= I1 = (18,3 + 16 + 12)/3 ≈ 15,43 A
= I2 = |18,3 + 16∠120° + 12∠240°| / 3 ≈ 1,84 A
= I_eq = √(I1² + 3·I2²) = √(238,2 + 10,2) ≈ 15,76 A
Stosunek I2/I1 ≈ 11,9 % jest ok. 6 razy większy niż VUF ≈ 2 % – bo impedancja silnika dla składowej przeciwnej jest mała, bliska impedancji rozruchowej (patrz zakładka 8). Asymetria prądów liczona metodą NEMA wynosi aż ≈ 22 %.
## Tabela 12.2 i reguła NEMA
Już 3,5 % asymetrii zwiększa straty o 12,5 % i grzanie do 114 %. NEMA MG-1 zaleca obniżenie mocy silnika (derating) przy asymetrii > 1 % i zakazuje pracy > 5 %. Przybliżenie: przyrost temperatury ≈ 2·(asymetria %)² %.
! Źródła asymetrii: nierówno rozłożone odbiory jednofazowe, przepalony bezpiecznik jednej fazy, luźny zacisk, uszkodzona bateria kondensatorów. Zawsze mierz napięcia na zaciskach silnika, nie w rozdzielni.
"""

    def build(self):
        self.section("Napięcia międzyfazowe (12.7)")
        self.slider("vab", "U_AB", 300.0, 450.0, 371.0, "V", "{:.0f}")
        self.slider("vbc", "U_BC", 300.0, 450.0, 374.0, "V", "{:.0f}")
        self.slider("vca", "U_CA", 300.0, 450.0, 395.0, "V", "{:.0f}")
        self.section("Prądy fazowe (12.8)")
        self.slider("ia", "I_A", 0.0, 40.0, 18.3, "A", "{:.1f}")
        self.slider("ib", "I_B", 0.0, 40.0, 16.0, "A", "{:.1f}")
        self.slider("ic", "I_C", 0.0, 40.0, 12.0, "A", "{:.1f}")
        self.slider("K", "K = Rr2/Rr1", 1.0, 8.0, 3.0, "", "{:.1f}")

    def draw(self):
        vab, vbc, vca = self.g("vab"), self.g("vbc"), self.g("vca")
        nema, avg = nema_unbalance([vab, vbc, vca])
        _, v1, v2 = seq_from_magnitudes(vab, vbc, vca)
        vuf_sym = abs(v2) / abs(v1) * 100
        try:
            vuf_ex = vuf_exact(vab, vbc, vca)
            tri = closed_triangle(vab, vbc, vca)
        except (ValueError, ZeroDivisionError):
            vuf_ex, tri = float("nan"), None
        ia, ib, ic = self.g("ia"), self.g("ib"), self.g("ic")
        _, i1, i2 = seq_from_magnitudes(ia, ib, ic)
        ieq = i_equivalent(abs(i1), abs(i2), self.g("K"))
        iunb, iavg = nema_unbalance([ia, ib, ic]) if (ia + ib + ic) > 0 else (0.0, 0.0)

        ax = self.fig.add_subplot(1, 2, 1)
        ax.set_aspect("equal")
        cols = [CYAN, GREEN, PEACH]
        names = ["U_AB", "U_BC", "U_CA"]
        if tri:
            pts = [0j, tri[0], tri[0] + tri[1]]
            for k in range(3):
                p0 = pts[k]
                p1 = p0 + tri[k]
                ax.annotate("", xy=(p1.real, p1.imag), xytext=(p0.real, p0.imag),
                            arrowprops=dict(arrowstyle="-|>", color=cols[k], lw=2.2))
                mid = (p0 + p1) / 2
                ax.text(mid.real, mid.imag, f" {names[k]}={abs(tri[k]):.0f} V\n ∠{math.degrees(cmath.phase(tri[k])):.1f}°",
                        color=cols[k], fontsize=8)
        # trójkąt idealny (symetryczny) dla porównania
        ideal = [avg, avg * A_OP ** 2, avg * A_OP]
        pts = [0j, ideal[0], ideal[0] + ideal[1]]
        xs = [p.real for p in pts] + [0]
        ys = [p.imag for p in pts] + [0]
        ax.plot(xs, ys, color=SUB, ls=":", lw=1, label=f"symetryczny {avg:.0f} V")
        if tri:
            vx = [0, tri[0].real, (tri[0] + tri[1]).real]
            vy = [0, tri[0].imag, (tri[0] + tri[1]).imag]
            ax.plot(vx, vy, "o", color=FG, ms=4)
            for (px, py), nm in zip(zip(vx, vy), "ABC"):
                ax.text(px, py, f"  {nm}", color=FG, fontsize=10, weight="bold")
        style_ax(ax, "Zamknięty trójkąt napięć międzyfazowych", "Re [V]", "Im [V]")
        ax.legend(fontsize=8, loc="lower right")
        ax.margins(0.2)

        ax = self.fig.add_subplot(2, 2, 2)
        labels = ["NEMA %", "VUF sym. %", "VUF dokł. %", "asym. prądów %"]
        vals = [nema, vuf_sym, vuf_ex, iunb]
        ax.bar(labels, vals, color=[CYAN, YELLOW, PEACH, MAUVE])
        for i, vv in enumerate(vals):
            ax.text(i, vv, f"{vv:.2f}", ha="center", va="bottom", fontsize=8)
        ax.axhline(1, color=GREEN, ls=":", lw=1)
        ax.axhline(5, color=RED, ls="--", lw=1)
        ax.text(3.4, 5, "limit 5 %", color=RED, fontsize=7, ha="right", va="bottom")
        style_ax(ax, "Miary asymetrii", "", "%")
        ax.tick_params(axis="x", labelsize=7)

        ax = self.fig.add_subplot(2, 2, 4)
        u = np.linspace(0, 5, 100)
        ax.plot(TABLE_12_2["unb"], TABLE_12_2["heat"], "o-", color=RED, label="grzanie % (tab. 12.2)")
        ax.plot(TABLE_12_2["unb"], [100 + x for x in TABLE_12_2["loss"]], "s-", color=PEACH, label="straty % (tab. 12.2)")
        ax.plot(u, 100 + 2 * u ** 2, color=SUB, ls=":", label="100 + 2·u² (NEMA)")
        h_now = np.interp(min(nema, 5), TABLE_12_2["unb"], TABLE_12_2["heat"])
        ax.plot([nema], [h_now], "D", color=YELLOW, ms=8, label=f"bieżące: {h_now:.0f} %")
        style_ax(ax, "Skutki asymetrii (tabela 12.2)", "asymetria napięć [%]", "[%]")
        ax.legend(fontsize=7)
        return (f"12.7  U_śr = {avg:.2f} V\n"
                f"      asym. NEMA = {nema:.3f} %\n"
                f"      |V1| = {abs(v1):.2f} V, |V2| = {abs(v2):.3f} V\n"
                f"      VUF (kąty symetr.) = {vuf_sym:.3f} %\n"
                f"      VUF (dokładny IEC) = {vuf_ex:.3f} %\n"
                f"12.8  |I1| = {abs(i1):.3f} A\n"
                f"      |I2| = {abs(i2):.3f} A\n"
                f"      I2/I1 = {abs(i2) / max(abs(i1), 1e-9) * 100:.2f} %\n"
                f"      I_eq (K={self.g('K'):.1f}) = {ieq:.3f} A\n"
                f"      asym. prądów NEMA = {iunb:.2f} %")


# =============================================================================
# 8. SKŁADOWA PRZECIWNA (12.8-12.9)
# =============================================================================

class NegSeqTab(Tab):
    title = "8 · Składowa przeciwna"
    explanation = """
# Zabezpieczenie od składowej przeciwnej (46) – rozdz. 12.8–12.9
## Dlaczego mała asymetria napięcia daje dużą asymetrię prądu?
Dla składowej przeciwnej wirnik ma poślizg 2 − s ≈ 2 i „wygląda” jak silnik zablokowany. Jego impedancja jest więc mniej więcej równa impedancji rozruchowej – bardzo mała (równ. 12.77–12.80):
= Z1 (praca) / Z2 ≈ I_st / I_run ≈ 5…8
= I2/I1 = (V2/Z2)/(V1/Z1) = (I_st/I_run) · (V2/V1)      (12.81)
## Przykład z tekstu (s. 444)
= I_st/I_run = 7,   V2/V1 = 0,04 (4 %)
= I2/I1 = 7 · 0,04 = 0,28  →  28 % !
Zaledwie 4 % asymetrii napięć wywołuje 28 % składowej przeciwnej prądu.
> Jak lekko przekrzywione koło w rowerze: z zewnątrz ledwo widać, ale przy każdym obrocie coś „szarpie” i hamulce się grzeją.
## Grzanie wirnika (równ. 12.83–12.87)
= P_straty = I1²·(Rs + Rr1) + I2²·(Rs + g·Rr1),   g = Rr2/Rr1 > 1
Prąd przeciwny ma w wirniku częstotliwość 2f = 100 Hz. Z powodu efektu naskórkowego Rr2 jest kilka razy większe niż Rr1 – ten sam amper grzeje mocniej. Moment od I2 jest mały i ujemny (hamuje) – równ. 12.82.
## Uzwojenia wirnika pierścieniowego (rozdz. 12.9)
Zabezpieczenie różnicowe stojana nie widzi zwarć w wirniku. Stosuje się człon nadprądowy bezzwłoczny z opóźnieniem ~30 ms, nastawiony na 3·I_n (prąd rozruchowy ograniczony rezystorem do 2·I_n).
## Co zobaczysz na wykresach
• Lewy górny: I2/I1 rośnie liniowo z asymetrią napięć – im większe I_st/I_run, tym stromiej.
• Prawy górny: moment wypadkowy T = T1 + T2 i straty w wirniku w funkcji asymetrii (model z zakładki 1).
• Dolny: ile razy rosną straty w wirniku przy danej asymetrii dla różnych g.
! Typowa nastawa 46: alarm przy I2 ≈ 10–15 % In, wyłączenie z charakterystyką zależną I2²·t = K (np. K = 30–40 s dla silników). Przerwa jednej fazy (I2 ≈ 58 % przy pełnym obciążeniu) musi być wyłączona w kilka sekund.
"""

    def build(self):
        self.section("Parametry")
        self.slider("ratio", "I_st / I_run", 3.0, 9.0, 7.0, "", "{:.2f}")
        self.slider("v2", "V2/V1", 0.0, 10.0, 4.0, "%", "{:.2f}")
        self.slider("g", "g = Rr2/Rr1", 1.0, 8.0, 3.0, "", "{:.2f}")
        self.slider("rs", "Rs/Rr1", 0.2, 3.0, 1.0, "", "{:.2f}")
        self.slider("s", "Poślizg pracy", 0.01, 0.08, 0.03, "", "{:.3f}")

    def draw(self):
        ratio, v2, g = self.g("ratio"), self.g("v2") / 100, self.g("g")
        r21 = neg_seq_current_ratio(ratio, v2)
        ax = self.fig.add_subplot(2, 2, 1)
        x = np.linspace(0, 10, 100)
        for rr, c in [(5, GREEN), (6, CYAN), (7, MAUVE), (8, PEACH)]:
            ax.plot(x, rr * x, color=c, label=f"I_st/I_run = {rr}")
        ax.plot([v2 * 100], [r21 * 100], "o", color=RED, ms=9, label=f"bieżące: {r21 * 100:.1f} %")
        ax.plot([4], [28], "*", color=YELLOW, ms=14, label="przykład: 28 %")
        style_ax(ax, "I2/I1 = (I_st/I_run)·(V2/V1)", "V2/V1 [%]", "I2/I1 [%]")
        ax.legend(fontsize=7)

        m = MotorPU()
        s = self.g("s")
        u = np.linspace(0, 0.10, 60)
        ops = [m.operate(s, 1.0, uu) for uu in u]
        ax = self.fig.add_subplot(2, 2, 2)
        ax.plot(u * 100, [o["T"] for o in ops], color=GREEN, label="moment T (j.w.)")
        ax.plot(u * 100, [o["q"] / ops[0]["q"] for o in ops], color=RED, label="straty wirnika / bez asym.")
        ax.plot(u * 100, [abs(o["Is2"]) / abs(o["Is1"]) for o in ops], color=CYAN, label="I2/I1 z modelu")
        ax.axvline(v2 * 100, color=YELLOW, ls="--", lw=1)
        style_ax(ax, f"Model silnika (s = {s:.3f})", "V2/V1 [%]", "[j.w.]")
        ax.legend(fontsize=7)

        ax = self.fig.add_subplot(2, 1, 2)
        rs = self.g("rs")
        for gg, c in [(1, SUB), (3, CYAN), (5, MAUVE), (g, YELLOW)]:
            r = neg_seq_current_ratio(ratio, x / 100)
            fac = (1 * (rs + 1) + r ** 2 * (rs + gg)) / (rs + 1)
            ax.plot(x, (fac - 1) * 100, color=c, ls="--" if gg == g else "-", label=f"g = {gg:.2f}")
        fac_now = (rs + 1 + r21 ** 2 * (rs + g)) / (rs + 1)
        ax.plot([v2 * 100], [(fac_now - 1) * 100], "o", color=RED, ms=8)
        style_ax(ax, "Wzrost strat (12.87): [I1²(Rs+Rr1) + I2²(Rs+g·Rr1)] / [I1²(Rs+Rr1)] − 1",
                 "V2/V1 [%]", "wzrost strat [%]")
        ax.legend(fontsize=7)
        op_now = m.operate(s, 1.0, v2)
        return (f"I2/I1 = {ratio:.2f} · {v2:.3f}\n      = {r21:.4f} ({r21 * 100:.1f} %)\n"
                f"Wzrost strat (12.87): {(fac_now - 1) * 100:.1f} %\n"
                f"Model: T = {op_now['T']:.3f} j.w.\n"
                f"       T2 = {op_now['T2']:.4f} j.w.\n"
                f"       I2/I1 = {abs(op_now['Is2']) / abs(op_now['Is1']) * 100:.1f} %\n"
                f"Wirnik pierścieniowy:\n  człon bezzwłoczny 3·In, 30 ms")


# =============================================================================
# 9. ZADANIA NIEROZWIĄZANE 12.9 - 12.10
# =============================================================================

class UnsolvedTab(Tab):
    title = "9 · Zadania 12.9–12.10"
    explanation = """
# Zadanie 12.9 – udział składowej zgodnej w mocy pozornej (gwiazda)
## Rozwiązanie ogólne
Moc pozorna w składowych symetrycznych (równ. 12.9–12.14):
= S = 3·(V0·I0* + V1·I1* + V2·I2*)
W gwieździe bez uziemionego punktu neutralnego prąd I0 nie ma którędy płynąć (I_A + I_B + I_C = 0), więc V0·I0* = 0:
= S = 3·V1·I1* + 3·V2·I2*,     udział zgodny:  S1 = 3·V1·I1*
V1 – napięcie fazowe składowej zgodnej, I1* – sprzężenie prądu. P1 = Re{S1} = 3·|V1|·|I1|·cosφ1.
## Przykład liczbowy (dane z 12.7–12.8)
= |V1| (fazowe) = 380/√3 ≈ 219,4 V,  |I1| ≈ 15,43 A
= |S1| = 3 · 219,4 · 15,43 ≈ 10,16 kVA
= |S2| = 3 · (7,55/√3) · 1,84 ≈ 0,024 kVA   → udział zgodny ≈ 99,8 %
Składowa przeciwna praktycznie nie przenosi mocy użytecznej – za to grzeje wirnik!
# Zadanie 12.10 – przeciążenie i straty w rdzeniu przy przepięciu
## Model (rozdz. 12.10.2)
Książka podaje: 10 % przepięcia → ok. 10 % przeciążenia i 20–30 % większe straty w rdzeniu. Straty w żelazie (histereza + prądy wirowe) rosną z kwadratem strumienia, a strumień ∝ U. Przy nasyceniu rosną jeszcze szybciej:
= ΔP_Fe % = [(1 + ΔU)^n − 1] · 100,   n = 2 (bez nasycenia) … ≈ 2,6 (nasycenie)
= przeciążenie % ≈ ΔU %
## Wyniki (n = 2 … 2,6)
= ΔU = 5 %:   przeciążenie ≈ 5 %,   ΔP_Fe ≈ 10,3 … 13,5 %
= ΔU = 10 %:  przeciążenie ≈ 10 %,  ΔP_Fe ≈ 21,0 … 28,1 %   (zgodne z książką 20–30 %)
= ΔU = 15 %:  przeciążenie ≈ 15 %,  ΔP_Fe ≈ 32,3 … 43,8 %
= ΔU = 20 %:  przeciążenie ≈ 20 %,  ΔP_Fe ≈ 44,0 … 60,7 %
= ΔU = 25 %:  przeciążenie ≈ 25 %,  ΔP_Fe ≈ 56,3 … 78,7 %
> Przepięcie działa jak za mocno napompowana dętka: pole magnetyczne w żelazie „wypycha” się poza wygodny zakres (nasycenie), prąd magnesujący gwałtownie rośnie i rdzeń się grzeje.
! Przepięcie 12 % zabezpieczenie 59 może wykryć w 8,3 ms (pół okresu), ale wyłączenie powinno być skoordynowane z zabezpieczeniem cieplnym – krótkie przepięcia (sekundy) nie grzeją silnika groźnie. Normy dopuszczają pracę w zakresie ±10 % U_n.
"""

    def build(self):
        self.section("Zadanie 12.9")
        self.slider("v1", "|V1| międzyfazowe", 300.0, 450.0, 380.0, "V", "{:.1f}")
        self.slider("i1", "|I1|", 0.0, 40.0, 15.43, "A", "{:.2f}")
        self.slider("v2", "|V2| międzyfazowe", 0.0, 40.0, 7.55, "V", "{:.2f}")
        self.slider("i2", "|I2|", 0.0, 10.0, 1.84, "A", "{:.2f}")
        self.slider("phi1", "Kąt φ1 (cosφ)", 0.0, 90.0, 36.9, "°", "{:.1f}")
        self.slider("phi2", "Kąt φ2", 0.0, 90.0, 80.0, "°", "{:.1f}")
        self.section("Zadanie 12.10")
        self.slider("n", "Wykładnik strat w rdzeniu n", 1.6, 3.0, 2.0, "", "{:.2f}")
        self.slider("dv", "Przepięcie ΔU", 0.0, 30.0, 10.0, "%", "{:.1f}")

    def draw(self):
        v1f = self.g("v1") / SQ3
        v2f = self.g("v2") / SQ3
        s1 = 3 * v1f * self.g("i1") * cmath.exp(1j * math.radians(self.g("phi1")))
        s2 = 3 * v2f * self.g("i2") * cmath.exp(1j * math.radians(self.g("phi2")))
        stot = s1 + s2
        ax = self.fig.add_subplot(1, 2, 1)
        ax.bar(["P1", "Q1", "|S1|", "P2", "Q2", "|S2|"],
               [s1.real / 1e3, s1.imag / 1e3, abs(s1) / 1e3, s2.real / 1e3, s2.imag / 1e3, abs(s2) / 1e3],
               color=[CYAN, BLUE, GREEN, RED, PEACH, MAUVE])
        share = abs(s1) / max(abs(s1) + abs(s2), 1e-9) * 100
        style_ax(ax, f"12.9: S = 3V1I1* + 3V2I2*  (udział zgodny {share:.2f} %)", "", "kW / kvar / kVA")

        ax = self.fig.add_subplot(1, 2, 2)
        dvs = np.array([5, 10, 15, 20, 25])
        n = self.g("n")
        ol = dvs.astype(float)
        c2 = np.array([overvoltage_effects(d, 2.0)[1] for d in dvs])
        cn = np.array([overvoltage_effects(d, n)[1] for d in dvs])
        c26 = np.array([overvoltage_effects(d, 2.6)[1] for d in dvs])
        w = 1.2
        ax.bar(dvs - w, ol, width=w, color=YELLOW, label="przeciążenie ≈ ΔU")
        ax.bar(dvs, cn, width=w, color=RED, label=f"ΔP_Fe (n = {n:.2f})")
        ax.bar(dvs + w, c26 - c2, bottom=c2, width=w, color=PEACH, alpha=0.6, label="zakres n = 2…2,6")
        for d, val in zip(dvs, cn):
            ax.text(d, val, f"{val:.1f}", ha="center", va="bottom", fontsize=7)
        dv = self.g("dv")
        ax.axvline(dv, color=CYAN, ls="--", lw=1)
        style_ax(ax, "12.10: skutki przepięcia", "przepięcie ΔU [%]", "[%]")
        ax.legend(fontsize=7)
        lines = [f"12.9  |S1| = {abs(s1) / 1e3:.3f} kVA",
                 f"      P1 = {s1.real / 1e3:.3f} kW",
                 f"      |S2| = {abs(s2) / 1e3:.4f} kVA",
                 f"      |S| = {abs(stot) / 1e3:.3f} kVA",
                 f"      udział S1 = {share:.2f} %", "",
                 f"12.10 (n = {n:.2f}):"]
        for d in dvs:
            o, c = overvoltage_effects(d, n)
            lines.append(f"  ΔU={d:>2}%: przeciąż.≈{o:.0f}%, ΔP_Fe={c:.1f}%")
        o, c = overvoltage_effects(dv, n)
        lines.append(f"  bieżące ΔU={dv:.1f}%: ΔP_Fe={c:.1f}%")
        return "\n".join(lines)


# =============================================================================
# APLIKACJA
# =============================================================================

class App:
    def __init__(self, root):
        self.root = root
        root.title("Zabezpieczenia silnika indukcyjnego – rozdz. 12 (interaktywnie)")
        root.geometry("1500x940")
        root.configure(bg=BG)
        self._style()
        head = tk.Label(root, text="⚡ Zabezpieczenia silnika indukcyjnego  ·  przykłady 12.1–12.10  ·  "
                                   "suwaki → obliczenia → wykresy → wyjaśnienia",
                        bg=BG, fg=CYAN, font=("DejaVu Sans", 12, "bold"), anchor="w", padx=10, pady=6)
        head.pack(side="top", fill="x")
        self.status_var = tk.StringVar(value="Gotowe")
        tk.Label(root, textvariable=self.status_var, bg=SURF, fg=SUB, anchor="w", padx=10,
                 font=("DejaVu Sans", 9)).pack(side="bottom", fill="x")
        nb = ttk.Notebook(root)
        nb.pack(side="top", fill="both", expand=True)
        self.tabs = []
        for cls in (OverviewTab, CircuitTab, StartTab, ThermalTab, Ex123Tab, DecayTab,
                    IeqTab, UnbalanceTab, NegSeqTab, UnsolvedTab):
            self.tabs.append(cls(self, nb))
        self.status("Gotowe – wybierz zakładkę i przesuwaj suwaki")

    def status(self, msg):
        self.status_var.set(msg)

    def _style(self):
        st = ttk.Style(self.root)
        try:
            st.theme_use("clam")
        except tk.TclError:
            pass
        st.configure(".", background=BG, foreground=FG, fieldbackground=SURF, bordercolor=OVER)
        st.configure("TFrame", background=BG)
        st.configure("TLabel", background=BG, foreground=FG, font=("DejaVu Sans", 9))
        st.configure("Sec.TLabel", background=SURF, foreground=MAUVE, font=("DejaVu Sans", 10, "bold"), padding=4)
        st.configure("Val.TLabel", background=BG, foreground=YELLOW, font=("DejaVu Sans Mono", 9, "bold"))
        st.configure("TNotebook", background=BG, borderwidth=0)
        st.configure("TNotebook.Tab", background=SURF, foreground=SUB, padding=(10, 4), font=("DejaVu Sans", 9, "bold"))
        st.map("TNotebook.Tab", background=[("selected", OVER)], foreground=[("selected", CYAN)])
        st.configure("TCheckbutton", background=BG, foreground=FG)
        st.map("TCheckbutton", background=[("active", BG)])
        st.configure("TButton", background=SURF, foreground=FG)
        st.configure("Horizontal.TScale", background=BG, troughcolor=SURF)
        st.configure("TPanedwindow", background=BG)


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
