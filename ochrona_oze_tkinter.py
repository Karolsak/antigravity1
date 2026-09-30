"""
Ochrona systemów z rozproszonymi źródłami odnawialnymi (RDG / OZE)
===================================================================
Interaktywna aplikacja Tkinter + Matplotlib do rozdziału 18
"Protection of Renewable Distributed Generation System"
(Power System Protection in Smart Grid Environment).

Rozdział jest opisowy (brak przykładów liczbowych), więc KAŻDE zagadnienie,
rysunek (18.1-18.11), tabelę (18.1, 18.2) i pytanie kontrolne (18.8)
zamieniono na interaktywny przykład liczbowy: suwaki -> obliczenia ->
wykresy -> objaśnienie językiem ucznia liceum (ale z inżynierskimi wzorami).

Zakładki:
  00  Przegląd: mikrosieć (rys. 18.1) i mapa rozdziału
  01  Profil napięcia i straty vs. moc/miejsce OZE        (18.1.2, 18.2.2)
  02  Prąd zwarciowy, "oślepienie" i fałszywe wyłączenia   (18.2.4, 18.4.1)
  03  Koordynacja reklozer - bezpiecznik z OZE             (18.4.1)
  04  Uziemienie i zwarcie doziemne, 173 % napięcia        (18.2.5, rys. 18.5)
  05  Praca wyspowa - strefa niewykrywalności NDZ          (18.2.6, 18.3.3, 18.3.6)
  06  ROCOF i odciążanie częstotliwościowe (SCO)           (18.2.8, 18.3.3)
  07  Aktywne metody antywyspowe: AFD, AFDPF, SMS          (18.3.4, 18.3.5)
  08  Mikrosieć: tryb sieciowy vs wyspowy, adaptacja, FCL  (18.4.1-18.4.4, 18.4.9)
  09  Zabezpieczenie odległościowe i różnicowe             (18.4.6, 18.4.7)
  10  SPZ (auto-reclosing) przy pracującym OZE             (18.4.1, 18.2.7)
  11  Elektrownia wiatrowa stałoobrotowa SCIG, soft-start  (18.5.1, rys. 18.6)
  12  Aerodynamika turbiny, MPPT, stała vs zmienna prędk.  (18.5.2, rys. 18.7, tab. 18.1)
  13  DFIG - przepływ mocy pod/nadsynchronicznie          (18.5.3, rys. 18.8, tab. 18.2)
  14  DFIG - zapad napięcia i crowbar (symulacja RK4)      (18.5.4, rys. 18.9)
  15  Fotowoltaika: I-U, bezpieczniki stringów, przepięcia (18.6, rys. 18.10)
  16  Sieci przyszłości: moc zwarciowa, wyłącznik, DNO     (18.7, rys. 18.11)
  17  Pytania kontrolne 18.8 - odpowiedzi

Uruchomienie:  python ochrona_oze_tkinter.py
Wymagania:     numpy, matplotlib (tkinter jest w standardowym Pythonie)
"""
import math
import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch, Circle
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from cycler import cycler

PI = math.pi
SQ3 = math.sqrt(3.0)
A_OP = complex(-0.5, SQ3 / 2)          # operator a = 1∠120°
F0 = 50.0                              # częstotliwość znamionowa [Hz]
W0 = 2 * PI * F0                       # pulsacja [rad/s]

# ---- paleta (Catppuccin Mocha) -------------------------------------------
C = dict(bg="#1e1e2e", ax="#181825", panel="#313244", text="#cdd6f4", sub="#a6adc8",
         grid="#45475a", blue="#89b4fa", green="#a6e3a1", red="#f38ba8",
         yellow="#f9e2af", peach="#fab387", mauve="#cba6f7", teal="#94e2d5",
         sky="#89dceb", pink="#f5c2e7", gray="#7f849c")

matplotlib.rcParams.update({
    "figure.facecolor": C["bg"], "axes.facecolor": C["ax"], "savefig.facecolor": C["bg"],
    "axes.edgecolor": C["gray"], "axes.labelcolor": C["text"], "text.color": C["text"],
    "xtick.color": C["sub"], "ytick.color": C["sub"], "grid.color": C["grid"],
    "grid.alpha": 0.6, "axes.grid": True, "legend.facecolor": C["panel"],
    "legend.edgecolor": C["grid"], "legend.fontsize": 8, "axes.titlesize": 10,
    "axes.labelsize": 9, "font.size": 9,
    "axes.prop_cycle": cycler(color=[C["sky"], C["green"], C["peach"], C["mauve"],
                                     C["red"], C["yellow"], C["teal"], C["pink"]]),
})


# =============================================================================
# RDZEŃ OBLICZENIOWY
# =============================================================================

# ---------- charakterystyki czasowo-prądowe --------------------------------
IEC = {"SI": (0.14, 0.02), "VI": (13.5, 1.0), "EI": (80.0, 2.0), "LTI": (120.0, 1.0)}


def iec_time(I, Ip, tms, curve="SI"):
    """Czas zadziałania przekaźnika nadprądowego wg IEC 60255:
    t = TMS * k / ((I/Ip)^a - 1). Dla I <= Ip przekaźnik nie działa (inf)."""
    k, a = IEC[curve]
    M = np.asarray(I, float) / Ip
    with np.errstate(all="ignore"):
        t = tms * k / (M ** a - 1.0)
    return np.where(M > 1.0001, t, np.inf)


def fuse_melt(I, In):
    """Uproszczona krzywa minimalnego topienia bezpiecznika (typ ~K).
    Asymptota przy 2*In, przy dużych prądach ~ I^-2.2 (stałe I^2t)."""
    M = np.asarray(I, float) / (2.0 * In)
    with np.errstate(all="ignore"):
        t = 12.0 / (M ** 2.2 - 1.0) + 0.008
    return np.where(M > 1.0001, t, np.inf)


def fuse_clear(I, In):
    """Całkowity czas wyłączenia bezpiecznika (topienie + łuk)."""
    return 1.25 * fuse_melt(I, In) + 0.015


# ---------- 01: rozpływ mocy w linii promieniowej (backward-forward sweep) ---
def feeder_loadflow(n, L_km, r, x, P_load, pf, Vn_kV, dg_node, P_dg, Q_dg,
                    v_sub=1.0, iters=40):
    """Rozpływ mocy w promieniowej linii SN z n węzłami (obciążenie rozłożone
    równomiernie) i jednym źródłem OZE w węźle dg_node.
    Metoda: iteracyjny 'backward-forward sweep'.
    Jednostki: MW, Mvar, kV, kA, Ω.
    Zwraca: |V| w p.u. (węzły 0..n), moc czynną w gałęziach [MW], straty [MW]."""
    z = (r + 1j * x) * L_km / n
    tanphi = math.tan(math.acos(pf))
    S = np.full(n + 1, (P_load / n) * (1 + 1j * tanphi), complex)
    S[0] = 0
    if 1 <= dg_node <= n:
        S[dg_node] -= (P_dg + 1j * Q_dg)
    Vb = Vn_kV / SQ3
    V = np.full(n + 1, v_sub * Vb, complex)
    Ib = np.zeros(n + 1, complex)
    for _ in range(iters):
        Iinj = np.conj(S / (3 * V))                # prąd pobierany w węźle [kA]
        Iinj[0] = 0
        Ib = np.cumsum(Iinj[::-1])[::-1]           # Ib[k] = prąd gałęzi (k-1 -> k)
        V[1:] = V[0] - z * np.cumsum(Ib[1:])
    Pbr = 3 * np.real(V[:-1] * np.conj(Ib[1:]))
    loss = 3 * np.sum(np.abs(Ib[1:]) ** 2) * z.real
    return np.abs(V) / Vb, Pbr, loss


# ---------- 02: zwarcie w linii promieniowej z OZE --------------------------
def radial_fault(x_km, d_km, Vn, Sk, r, x, dg_type, S_dg, xd, klim, with_dg=True):
    """Zwarcie 3-fazowe w odległości x_km od stacji; OZE w odległości d_km.
    dg_type: 'sync' (maszyna wirująca, X''d) lub 'inv' (falownik, prąd
    ograniczony do klim*In). Zwraca (I_przekaźnika, I_OZE) zespolone [kA]."""
    E = Vn / SQ3
    Zs = (Vn ** 2 / Sk) * (0.1 + 1j)
    z = r + 1j * x
    x_km = max(x_km, 1e-3)
    if (not with_dg) or S_dg <= 0:
        return E / (Zs + z * x_km), 0j
    if dg_type == "sync":
        Zdg = 1j * (xd + 0.06) * Vn ** 2 / S_dg    # generator + transformator 6 %
        if x_km <= d_km:
            return E / (Zs + z * x_km), E / (Zdg + z * (d_km - x_km))
        Zup = Zs + z * d_km
        Zth = Zup * Zdg / (Zup + Zdg)
        If = E / (Zth + z * (x_km - d_km))
        Vn_node = If * z * (x_km - d_km)
        return (E - Vn_node) / Zup, (E - Vn_node) / Zdg
    # falownik: źródło prądowe o module klim*In, w fazie z prądem zwarciowym
    Idg_mag = klim * S_dg / (SQ3 * Vn)
    Ir0 = E / (Zs + z * x_km)
    Idg = Idg_mag * Ir0 / abs(Ir0)
    if x_km <= d_km:
        return Ir0, Idg
    return (E - z * (x_km - d_km) * Idg) / (Zs + z * x_km), Idg


def sympathetic_current(y_km, d_km, Vn, Sk, r, x, dg_type, S_dg, xd, klim):
    """Prąd wsteczny płynący z OZE (linia 1) przez przekaźnik linii 1 przy
    zwarciu na SĄSIEDNIEJ linii 2 w odległości y_km od szyn stacji [kA]."""
    if S_dg <= 0:
        return 0.0
    E = Vn / SQ3
    Zs = (Vn ** 2 / Sk) * (0.1 + 1j)
    z = r + 1j * x
    zf = z * max(y_km, 0.01)
    if dg_type == "sync":
        Zd = 1j * (xd + 0.06) * Vn ** 2 / S_dg + z * d_km
        Vb = E * (1 / Zs + 1 / Zd) / (1 / Zs + 1 / Zd + 1 / zf)
        return abs((E - Vb) / Zd)
    return klim * S_dg / (SQ3 * Vn)


# ---------- 04: zwarcie doziemne a sposób uziemienia -------------------------
def slg_fault(option, Vn, Sk, C0_uF, RN, detune, Xzz, isl_frac, Rf):
    """Zwarcie jednofazowe (faza a) metodą składowych symetrycznych.
    option: 'RN' | 'izol' | 'Pet' | 'zz' | 'wyspa'.
    Zwraca: If [A], Va, Vb, Vc (zespolone, w p.u. napięcia fazowego)."""
    E = Vn * 1e3 / SQ3
    X1 = (Vn * 1e3) ** 2 / (Sk * 1e6)
    Z1 = Z2 = 1j * X1
    C0 = C0_uF * 1e-6
    if option in ("zz", "wyspa"):
        C0 = C0 * isl_frac                     # wyspa = tylko fragment sieci
    Xc0 = 1.0 / (W0 * max(C0, 1e-12))
    Zc = -1j * Xc0

    def par(a, b):
        return a * b / (a + b)

    if option == "RN":
        Z0 = par(1j * X1 + 3 * RN, Zc)
    elif option == "izol":
        Z0 = Zc
    elif option == "Pet":
        XL3 = Xc0 / (1.0 + detune)             # 3*X_L; detune = rozstrojenie
        Z0 = par(1j * XL3 + 0.03 * XL3, Zc)
    elif option == "zz":
        Z0 = par(1j * Xzz, Zc)
    else:                                      # wyspa bez uziemienia
        Z0 = Zc
    I1 = E / (Z1 + Z2 + Z0 + 3 * Rf)
    V1 = E - Z1 * I1
    V2 = -Z2 * I1
    V0 = -Z0 * I1
    Va = V0 + V1 + V2
    Vb = V0 + A_OP ** 2 * V1 + A_OP * V2
    Vc = V0 + A_OP * V1 + A_OP ** 2 * V2
    return abs(3 * I1), Va / E, Vb / E, Vc / E


GROUND_OPTS = [("RN", "Uziemienie bezpośrednie / przez R_N"),
               ("izol", "Sieć izolowana"),
               ("Pet", "Kompensowana (cewka Petersena)"),
               ("zz", "Wyspa + transformator zig-zag"),
               ("wyspa", "Wyspa bez uziemienia")]


# ---------- 05: wyspa z obciążeniem RLC --------------------------------------
def island_rlc(dp, dq, Qf, f0=F0):
    """Napięcie i częstotliwość wyspy po utracie sieci.
    OZE (falownik, cos φ = 1) daje P_dg = 1 p.u.; obciążenie RLC:
    P_L = 1 + dp, Q_L - Q_C = dq (p.u., przy f0), dobroć Qf.
    V'^2 = P_dg / P_L ;  f' = f0 * sqrt(Q_L / Q_C)."""
    PL = 1.0 + dp
    QL = (dq + np.sqrt(dq ** 2 + 4 * (Qf * PL) ** 2)) / 2
    QC = QL - dq
    V = np.sqrt(1.0 / PL)
    f = f0 * np.sqrt(QL / QC)
    return V, f


# ---------- 06: częstotliwość wyspy i odciążanie -----------------------------
def island_frequency(deficit, H, D, R, Tg, reserve, mode, stage, rocof_th,
                     t_end=10.0, dt=0.002, t_isl=0.5):
    """Równanie ruchu (wahadłowe) wyspy:  2H dΔf/dt = Pm - Pe,
    Pe = P_obc * (1 + D Δf), regulator z statyzmem R, stałą Tg i rezerwą.
    mode: 'brak' | 'stat' (stopnie SCO) | 'adapt' (SCO wg ROCOF).
    Zwraca t, f[Hz], rocof[Hz/s], P_obc [p.u.], czas wykrycia ROCOF."""
    n = int(t_end / dt) + 1
    t = np.arange(n) * dt
    f = np.full(n, F0)
    load = np.ones(n)
    rocof = np.zeros(n)
    Pm0 = 1.0 - deficit
    Pm, df, shed = Pm0, 0.0, 0.0
    stages = [49.0, 48.8, 48.6, 48.4, 48.2]
    armed = [None] * len(stages)
    done = [False] * len(stages)
    adapt_t, adapt_amt = None, 0.0
    nwin = max(1, int(0.1 / dt))
    t_det = None
    for i in range(1, n):
        if t[i] >= t_isl:
            Pe = (1.0 - shed) * (1 + D * df)
            Pt = min(max(Pm0 - df / R, 0.0), Pm0 + reserve)
            Pm += dt * (Pt - Pm) / Tg
            df += dt * (Pm - Pe) / (2 * H)
        f[i] = F0 * (1 + df)
        j = max(0, i - nwin)
        rocof[i] = (f[i] - f[j]) / ((i - j) * dt)
        if t_det is None and t[i] > t_isl and abs(rocof[i]) > rocof_th:
            t_det = t[i] - t_isl
        if mode in ("stat", "adapt"):
            for k, fs in enumerate(stages):
                if not done[k] and f[i] < fs:
                    if armed[k] is None:
                        armed[k] = t[i]
                    elif t[i] - armed[k] >= 0.1:
                        shed += stage
                        done[k] = True
        if mode == "adapt" and adapt_t is None and f[i] < 49.8:
            est = 2 * H * abs(rocof[i]) / F0
            adapt_amt = math.ceil(est / 0.05) * 0.05
            adapt_t = t[i]
        if adapt_t is not None and adapt_amt > 0 and t[i] - adapt_t >= 0.1:
            shed += adapt_amt
            adapt_amt = 0.0
        shed = min(shed, 0.95)
        load[i] = 1.0 - shed
    return t, f, rocof, load, t_det


# ---------- 07: aktywne metody antywyspowe -----------------------------------
def inverter_phase(method, f, cf0, K, th_m, fm_off):
    """Kąt fazowy prądu falownika [rad] narzucany przez metodę aktywną."""
    if method == "pasywna":
        return 0.0
    if method == "AFD":
        return PI * cf0 / 2
    if method == "AFDPF":
        cf = cf0 + K * (f - F0)
        return PI * float(np.clip(cf, -0.5, 0.5)) / 2
    arg = float(np.clip((f - F0) / fm_off, -1, 1))       # SMS
    return th_m * math.sin(PI / 2 * arg)


def active_island_sim(method, Qf, fr, cf0, K, th_m, fm_off, n_cyc=150, alpha=0.5,
                      f_lo=49.5, f_hi=50.5):
    """Po wyspieniu częstotliwość ustala się tam, gdzie kąt obciążenia RLC
    atan(Qf (f/fr - fr/f)) = kąt falownika. Iteracja cykl po cyklu."""
    f = F0 + 0.02                      # drobne zaburzenie (szum pomiaru / niedopasowanie)
    fs = [f]
    t_trip = None
    for k in range(1, n_cyc):
        th = inverter_phase(method, f, cf0, K, th_m, fm_off)
        u = math.tan(th) / Qf
        f_new = fr * (u + math.sqrt(u * u + 4)) / 2
        f = f + alpha * (f_new - f)
        fs.append(f)
        if t_trip is None and (f > f_hi or f < f_lo):
            t_trip = k / F0
    return np.arange(n_cyc) / F0, np.array(fs), t_trip


# ---------- 11: maszyna indukcyjna klatkowa (SCIG) ---------------------------
SCIG = dict(Rs=0.01, Xs=0.10, Xm=3.0, Rr=0.012, Xr=0.12)


def scig_point(s, V=1.0, p=SCIG):
    """Schemat zastępczy (p.u.). Zwraca Is, moment Te, P, Q (konw. odbiornika)."""
    s = np.where(np.abs(s) < 1e-5, 1e-5, s)
    Zr = p["Rr"] / s + 1j * p["Xr"]
    Zm = 1j * p["Xm"]
    Zin = p["Rs"] + 1j * p["Xs"] + Zm * Zr / (Zm + Zr)
    Is = V / Zin
    Ir = Is * Zm / (Zm + Zr)
    Te = np.abs(Ir) ** 2 * p["Rr"] / s
    S = V * np.conj(Is)
    return Is, Te, S.real, S.imag


def scig_start(H, V0, t_ramp, soft, I_lim=3.0, Tl_k=0.1, t_end=15.0, dt=2e-3):
    """Rozruch (quasi-statyczny): 2H dω/dt = Te(s,V) - Tl_k ω^2.
    soft=True -> napięcie narasta liniowo od V0 do 1 w czasie t_ramp,
    ale nie więcej, niż pozwala ograniczenie prądu I_lim (Is ~ V)."""
    n = int(t_end / dt)
    t = np.arange(n) * dt
    w = 0.0
    I = np.zeros(n)
    W = np.zeros(n)
    for i in range(n):
        if soft:
            Is1, _, _, _ = scig_point(1.0 - w, 1.0)
            V = min(1.0, V0 + (1 - V0) * t[i] / t_ramp, I_lim / abs(Is1))
        else:
            V = 1.0
        Is, Te, _, _ = scig_point(1.0 - w, V)
        I[i] = abs(Is)
        W[i] = w
        w += dt * (float(Te) - Tl_k * w * w) / (2 * H)
    return t, I, W


# ---------- 12: aerodynamika turbiny -----------------------------------------
def cp_heier(lam, beta=0.0):
    """Współczynnik mocy Cp(λ, β) - przybliżenie Heiera."""
    lam = np.maximum(np.asarray(lam, float), 0.1)
    inv = 1.0 / (lam + 0.08 * beta) - 0.035 / (beta ** 3 + 1)
    li = 1.0 / inv
    cp = 0.5176 * (116 / li - 0.4 * beta - 5) * np.exp(-21 / li) + 0.0068 * lam
    return np.maximum(cp, 0.0)


LAM_OPT = 8.1
CP_MAX = float(cp_heier(LAM_OPT))


def turbine_power(v, R, w_rad, rho=1.225):
    """Moc aerodynamiczna P = 0.5 ρ π R² Cp(λ) v³ [W], λ = ωR/v."""
    v = np.maximum(v, 0.1)
    return 0.5 * rho * PI * R ** 2 * cp_heier(w_rad * R / v) * v ** 3


def power_curve(v, R, P_rat, w_max, w_fix, variable, rho=1.225):
    """Krzywa mocy [W] turbiny: zmienna prędkość (MPPT) lub stała prędkość."""
    v = np.asarray(v, float)
    if variable:
        w = np.minimum(LAM_OPT * v / R, w_max)
    else:
        w = np.full_like(v, w_fix)
    P = turbine_power(v, R, w, rho)
    P = np.minimum(P, P_rat)
    return np.where((v >= 3.0) & (v <= 25.0), P, 0.0)


# ---------- 14: DFIG - zapad napięcia i crowbar ------------------------------
DFIG = dict(Rs=0.01, Rr=0.01, Lls=0.10, Llr=0.10, Lm=3.5)


def dfig_fault_sim(slip, P0, dip, t_fault, crowbar, Rcb, I_th, T_cb, Vr_max,
                   chopper, t_end=0.6, dt=1e-4, t_f0=0.1):
    """Symulacja DFIG w układzie stojana (wektory przestrzenne, p.u.):
        dψs/dt = ωb (vs - Rs is)
        dψr/dt = ωb (vr - Rr ir + j ωr ψr)
    Przekształtnik RSC = regulator PI prądu wirnika z ograniczeniem |vr|.
    Crowbar: gdy |ir| > I_th -> vr = -Rcb ir (RSC zablokowany) przez T_cb.
    Obwód DC: C dVdc/dt = P_RSC - P_GSC - P_chopper."""
    p = DFIG
    wb = W0
    Ls, Lr, Lm = p["Lls"] + p["Lm"], p["Llr"] + p["Lm"], p["Lm"]
    Dt = Ls * Lr - Lm * Lm
    wr = 1.0 - slip
    # stan ustalony (układ synchroniczny), stojan oddaje P0 przy cos φ = 1
    vs0 = 1.0 + 0j
    is0 = -P0 + 0j
    psis0 = (vs0 - p["Rs"] * is0) / 1j
    ir0 = (psis0 - Ls * is0) / Lm
    psir0 = Lm * is0 + Lr * ir0
    vr_ff = p["Rr"] * ir0 + 1j * slip * psir0
    Kp, Ki = 0.5, 20.0
    Cdc, Kdc, Pg_max, Rch = 0.01, 5.0, 0.35, 0.5
    Pin0 = -(vr_ff * ir0.conjugate()).real
    Vdc = 1.0 + Pin0 / Kdc
    integ = 0j
    ps, pr = psis0, psir0
    n = int(t_end / dt)
    keep = 5
    out = {k: [] for k in ("t", "Vs", "Ir", "Irsc", "Is", "Te", "Vdc", "cb")}
    cb_on, cb_t = False, -1.0

    def cur(ps_, pr_):
        return (Lr * ps_ - Lm * pr_) / Dt, (Ls * pr_ - Lm * ps_) / Dt

    for i in range(n):
        t = i * dt
        Vmag = (1.0 - dip) if (t_f0 <= t < t_f0 + t_fault) else 1.0
        th = wb * t
        rot = complex(math.cos(th), math.sin(th))
        is_, ir_ = cur(ps, pr)
        # --- logika crowbara ---
        if crowbar and not cb_on and abs(ir_) > I_th:
            cb_on, cb_t = True, t
        if cb_on and t - cb_t >= T_cb and abs(ir_) < I_th:
            cb_on = False
            integ = 0j
        if cb_on:
            vr_mode = "cb"
            vr_sync = 0j
        else:
            vr_mode = "rsc"
            e = ir0 - ir_ / rot
            vr_sync = vr_ff + Kp * e + integ
            if abs(vr_sync) > Vr_max:
                vr_sync *= Vr_max / abs(vr_sync)
            else:
                integ += Ki * e * dt
        vs = Vmag * rot

        def deriv(ps_, pr_):
            is2, ir2 = cur(ps_, pr_)
            vr = -Rcb * ir2 if vr_mode == "cb" else vr_sync * rot
            return (wb * (vs - p["Rs"] * is2),
                    wb * (vr - p["Rr"] * ir2 + 1j * wr * pr_))

        k1 = deriv(ps, pr)
        k2 = deriv(ps + 0.5 * dt * k1[0], pr + 0.5 * dt * k1[1])
        k3 = deriv(ps + 0.5 * dt * k2[0], pr + 0.5 * dt * k2[1])
        k4 = deriv(ps + dt * k3[0], pr + dt * k3[1])
        ps += dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        pr += dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        # --- obwód DC ---
        if vr_mode == "cb":
            P_in = 0.0
            I_rsc = 0.0
        else:
            P_in = -((vr_sync * rot) * ir_.conjugate()).real
            I_rsc = abs(ir_)
        Pg = float(np.clip(Kdc * (Vdc - 1.0), -Pg_max * Vmag, Pg_max * Vmag))
        Pch = (Vdc ** 2 / Rch) if (chopper and Vdc > 1.10) else 0.0
        Vdc += dt * (P_in - Pg - Pch) / (Cdc * max(Vdc, 0.2))
        Vdc = max(Vdc, 0.2)
        if i % keep == 0:
            out["t"].append(t)
            out["Vs"].append(Vmag)
            out["Ir"].append(abs(ir_))
            out["Irsc"].append(I_rsc)
            out["Is"].append(abs(is_))
            out["Te"].append((ps.conjugate() * is_).imag)
            out["Vdc"].append(Vdc)
            out["cb"].append(1.0 if cb_on else 0.0)
    return {k: np.array(v) for k, v in out.items()}


# ---------- 15: fotowoltaika --------------------------------------------------
PV_MOD = dict(Isc=11.0, Voc=49.5, Nc=72, n=1.3, Rs=0.35, a_isc=0.0005, b_voc=-0.0028)


def pv_iv(G, Tc, Ns=1, Np=1, m=PV_MOD, npts=300):
    """Model jednodiodowy (jawny wzór na U(I)):
    U = n Nc Vt ln((Iph - I)/I0 + 1) - I Rs. Zwraca U [V], I [A] całego pola."""
    k, q = 1.380649e-23, 1.602176634e-19
    T = Tc + 273.15
    Vt = k * T / q
    Iph = m["Isc"] * G / 1000 * (1 + m["a_isc"] * (Tc - 25))
    Voc_T = m["Voc"] * (1 + m["b_voc"] * (Tc - 25))
    nVt = m["n"] * m["Nc"] * Vt
    I0 = m["Isc"] * (1 + m["a_isc"] * (Tc - 25)) / (math.exp(Voc_T / nVt) - 1)
    I = np.linspace(0, Iph * 0.99999, npts)
    U = nVt * np.log((Iph - I) / I0 + 1) - I * m["Rs"]
    ok = U >= 0
    return U[ok] * Ns, I[ok] * Np


# =============================================================================
# OBJAŚNIENIA (dla ucznia liceum - ale z wzorami inżynierskimi)
# =============================================================================
EXPL = {}

EXPL["T00"] = """PRZEGLĄD ROZDZIAŁU 18 - czym jest rozproszona generacja odnawialna (RDG) i czemu komplikuje ochronę?

WYOBRAŹ SOBIE: dawniej sieć była jak rzeka płynąca z gór (elektrownia) do morza (odbiorcy) - woda płynęła
zawsze w jedną stronę. Zabezpieczenia (bezpieczniki, przekaźniki) projektowano jak tamy, które zakładają,
że woda płynie tylko "w dół". Dziś na dachach są panele PV, na polach wiatraki, w piwnicach baterie -
czyli "dopływy" w środku rzeki. Prąd może płynąć W OBIE STRONY, a to myli klasyczne zabezpieczenia.

RYSUNEK 18.1 (po lewej): typowa mikrosieć - sieć zasilająca, transformator, punkt przyłączenia (PCC),
wyłącznik odcinający mikrosieć, źródła: PV (przez falownik), turbina wiatrowa, magazyn energii, odbiory.

NAJWAŻNIEJSZE POJĘCIA:
 • DG / RDG - generacja rozproszona (odnawialna) - małe źródła blisko odbiorców.
 • PCC - punkt wspólnego przyłączenia (granica "moje/operatora").
 • Penetracja - udział mocy OZE w obciążeniu sieci (np. 50 %).
 • Wyspa (islanding) - fragment sieci z OZE pracuje sam, odcięty od systemu.
 • IEEE 1547 / IEC 61727 - normy przyłączeniowe: m.in. odłączenie OZE w ciągu 2 s od powstania wyspy.

TRZY RODZAJE OCHRONY (rys. 18.2-18.4):
 1) ochrona przyłączeniowa (interconnection) - chroni SIEĆ przed OZE (wyspa, prądy zwarciowe, przepięcia),
    po stronie pierwotnej lub wtórnej transformatora przyłączeniowego;
 2) ochrona generatora - chroni OZE przed jego własnymi awariami (moc zwrotna, utrata wzbudzenia,
    niesymetria, przewzbudzenie, zwarcia wewnętrzne);
 3) ochrona OZE przed siecią - np. przed niezsynchronizowanym SPZ (automatycznym ponownym załączeniem).

JAK UŻYWAĆ PROGRAMU: każda zakładka = jedno zagadnienie z rozdziału. Po lewej suwaki, na środku wykresy
(można przybliżać lupą z paska narzędzi), po prawej wyniki liczbowe, na dole - takie objaśnienie.
Sekcja "SPRÓBUJ" podpowiada eksperyment, który pokazuje sedno zjawiska.
"""

EXPL["T01"] = """01. PROFIL NAPIĘCIA I STRATY - wpływ mocy i miejsca przyłączenia OZE (18.1.2, 18.2.2)

INTUICJA: prąd płynący przez przewód "zużywa" część napięcia (spadek napięcia), jak ciśnienie wody spada
w długim wężu. W linii promieniowej bez OZE napięcie maleje od stacji do końca linii. Gdy na końcu linii
postawimy farmę PV, prąd płynie częściowo W DRUGĄ STRONĘ - napięcie może wtedy ROSNĄĆ ponad normę.

WZÓR (spadek napięcia na odcinku o rezystancji R i reaktancji X):
      ΔU ≈ (R·P + X·Q) / U
 P, Q - moc płynąca odcinkiem. Gdy OZE oddaje więcej niż lokalne obciążenie, P < 0 -> ΔU < 0 -> WZROST napięcia.
 Straty mocy:  ΔP = 3·I²·R  - maleją, gdy źródło jest blisko odbiorów (prąd nie musi płynąć daleko),
 ale ROSNĄ ponownie, gdy OZE jest za duże (prąd zwrotny też grzeje przewody) -> krzywa w kształcie "U".

MODEL: linia 15 kV podzielona na 20 odcinków, obciążenie rozłożone równo, OZE w wybranym węźle.
Rozpływ liczony metodą "backward-forward sweep" (dokładna, iteracyjna - tak liczą programy operatorów).

WYKRESY: (1) napięcie wzdłuż linii bez i z OZE, granice ±5 % / +10 %; (2) moc płynąca w każdej gałęzi
(wartość ujemna = przepływ zwrotny do stacji); (3) straty i maks. napięcie w funkcji mocy OZE -
"hosting capacity", czyli ile OZE zmieści się w linii bez przekroczenia 1,05 p.u.

PRZYKŁAD: 4 MW obciążenia, 10 km, OZE 6 MW w węźle 15 -> napięcie max ≈ 1,05 p.u., moc wraca do stacji.
SPRÓBUJ: (a) przesuń OZE na początek linii - napięcie prawie się nie zmienia (stacja "trzyma" napięcie);
(b) ustaw Q_OZE ujemne (falownik pobiera moc bierną) - napięcie spada: tak działa funkcja Q(U) falowników;
(c) zmniejsz napięcie stacji (regulator zaczepów LTC) - tą metodą operator też walczy z przepięciami.
"""

EXPL["T02"] = """02. PRĄD ZWARCIOWY, "OŚLEPIENIE" ZABEZPIECZENIA I FAŁSZYWE WYŁĄCZENIA (18.2.4, 18.4.1)

INTUICJA: przekaźnik na początku linii "liczy" prąd. Przy zwarciu prąd jest duży - przekaźnik wyłącza.
Gdy między stacją a miejscem zwarcia pracuje generator (OZE), to ON TEŻ dostarcza prąd do zwarcia.
Zwarcie "dostaje" więcej prądu, ale przekaźnik w stacji widzi MNIEJ (bo część prądu płynie z OZE,
a napięcie w węźle OZE jest podtrzymane). To tzw. OŚLEPIENIE ZABEZPIECZENIA (protection blinding):
przekaźnik zadziała później albo wcale.

WZORY (źródła o jednakowej SEM E):
  bez OZE:   I_p = E / (Z_s + z·x)
  z OZE (maszyna synchroniczna w odległości d < x):
      Z_th = (Z_s + z·d) ∥ Z_dg,   I_zw = E / (Z_th + z·(x-d)),   I_p = (E - U_d) / (Z_s + z·d)
  OZE falownikowe: źródło prądowe ograniczone do k·I_n (typowo 1,1-1,5 I_n) - wpływ mały!
Czas zadziałania przekaźnika (IEC, charakterystyka normalnie odwrotna SI):
      t = TMS · 0,14 / ((I/I_r)^0,02 - 1)

FAŁSZYWE (SYMPATYCZNE) WYŁĄCZENIE: zwarcie na SĄSIEDNIEJ linii 2. OZE z linii 1 "karmi" to zwarcie,
jego prąd płynie przez przekaźnik linii 1 w złą stronę. Jeśli przekroczy nastawę - linia 1 zostanie
wyłączona niepotrzebnie. Lekarstwo: przekaźnik kierunkowy (67).

WYKRESY: (1) prąd widziany przez przekaźnik w funkcji miejsca zwarcia (z OZE i bez), linia nastawy;
(2) charakterystyka czasowo-prądowa i czasy zadziałania dla zwarcia na końcu linii;
(3) prąd wsteczny przy zwarciu na linii sąsiedniej w funkcji mocy OZE (maszyna vs falownik).

SPRÓBUJ: ustaw OZE typu "maszyna" 8 MVA w połowie linii - zasięg zabezpieczenia skraca się, czas rośnie.
Przełącz na "falownik" - efekt znika, bo falownik daje mały prąd zwarciowy (to z kolei problem w wyspie!).
"""

EXPL["T03"] = """03. KOORDYNACJA REKLOZER - BEZPIECZNIK PRZY OBECNOŚCI OZE (18.4.1)

INTUICJA: ~80 % zwarć w liniach napowietrznych jest przemijających (gałąź dotknęła przewodu, ptak).
Dlatego stosuje się filozofię "ratowania bezpiecznika" (fuse saving):
  1) reklozer (wyłącznik z SPZ) otwiera się SZYBKO (krzywa szybka) - zanim bezpiecznik zdąży się stopić,
  2) po przerwie beznapięciowej załącza ponownie - jeśli zwarcie zniknęło, nikt nie stracił zasilania na długo,
  3) jeśli zwarcie trwa - reklozer przechodzi na krzywą WOLNĄ, a bezpiecznik ma czas przepalić się
     i odłączyć tylko uszkodzone odgałęzienie.
Warunek: dla każdego prądu zwarcia  t_reklozer_szybka(I_R) < 0,75·t_topienia_bezp(I_B)
         oraz  t_reklozer_wolna(I_R) > t_wyłączenia_bezp(I_B).

PROBLEM Z OZE: jeśli OZE jest między reklozerem a bezpiecznikiem, to prąd bezpiecznika
      I_B = I_R + I_OZE  > I_R.
Bezpiecznik "widzi" większy prąd niż reklozer - topi się SZYBCIEJ niż reklozer zdąży zareagować.
Koordynacja jest stracona: każde przemijające zwarcie kończy się przepalonym bezpiecznikiem i przyjazdem
ekipy. Im więcej OZE, tym węższy zakres prądów, w którym "fuse saving" działa.

WYKRESY: (1) krzywe czas-prąd (log-log): bezpiecznik (topienie i wyłączenie), reklozer (szybka, wolna),
punkty pracy dla bieżącego zwarcia; (2) zapas koordynacji t_topienia / t_szybka w funkcji prądu OZE -
poniżej 1/0,75 = 1,33 koordynacja jest stracona.

SPRÓBUJ: zwiększaj I_OZE i obserwuj, kiedy wskaźnik zmieni się na "UTRACONA". Potem zwiększ prąd
znamionowy bezpiecznika albo zmniejsz TMS krzywej szybkiej - to typowe środki zaradcze
(oraz ograniczniki prądu FCL albo szybkie odłączenie OZE przed pierwszym SPZ).
"""

EXPL["T04"] = """04. UZIEMIENIE PUNKTU NEUTRALNEGO I ZWARCIE DOZIEMNE - skąd 173 % napięcia? (18.2.5, rys. 18.5)

INTUICJA: zwarcie jednej fazy z ziemią (SLG) to najczęstsze zwarcie (~70-80 %). Jego prąd płynie przez
ziemię i wraca do punktu neutralnego transformatora. Jak łatwo wraca, zależy od sposobu uziemienia:
 • bezpośrednie (R_N = 0) - duży prąd (łatwo wykryć), napięcie faz zdrowych prawie się nie zmienia;
   już rezystor R_N rzędu kilku-kilkudziesięciu Ω ogranicza prąd do setek A, ale napięcie faz zdrowych
   rośnie wtedy w stronę 173 % (bo 3R_N >> X1) - to cena za mniejsze uszkodzenia w miejscu zwarcia;
 • sieć izolowana - prąd płynie tylko przez pojemności przewodów względem ziemi -> mały prąd (dziesiątki A),
   ale napięcie faz zdrowych rośnie do √3 = 173 % (punkt neutralny "przesuwa się" do potencjału fazy zwartej);
 • cewka Petersena - indukcyjność kompensuje prąd pojemnościowy -> prąd bardzo mały, łuk sam gaśnie;
 • WYSPA bez uziemienia - OZE przyłączone przez trafo z uzwojeniem w trójkąt nie ma drogi dla prądu
   zerowego. Po oddzieleniu od sieci wyspa staje się izolowana: odbiorcy na fazach zdrowych dostają 173 %!
 • transformator zig-zag (rys. 18.5) - tworzy "sztuczny punkt neutralny" dla wyspy -> problem znika.

METODA SKŁADOWYCH SYMETRYCZNYCH (dla fazy a):
     I1 = I2 = I0 = E / (Z1 + Z2 + Z0 + 3·R_f),     I_zw = 3·I0
     U0 = -Z0·I0,  U1 = E - Z1·I1,  U2 = -Z2·I2
     U_b = U0 + a²U1 + aU2,   U_c = U0 + aU1 + a²U2       (a = 1∠120°)
Z0 obejmuje uziemienie (3·R_N lub 3·X_L cewki) równolegle z reaktancją pojemnościową sieci X_C0 = 1/(ωC0).
Gdy Z0 >> Z1 (sieć izolowana), U0 ≈ -E, czyli U_b = E(a² - 1) -> |U_b| = √3·E.

WYKRESY: (1) wskazy napięć fazowych przed (szare) i po zwarciu (kolorowe); (2) porównanie prądu zwarcia
i maks. napięcia faz zdrowych dla 5 sposobów uziemienia; (3) napięcie faz zdrowych w funkcji rezystancji
przejścia R_f.
SPRÓBUJ: wybierz "Wyspa bez uziemienia", a potem "Wyspa + zig-zag" - różnica 173 % -> ~100 %.
Zwiększ R_N - zobaczysz, że już kilkadziesiąt omów "zamienia" sieć w prawie izolowaną.
"""

EXPL["T05"] = """05. PRACA WYSPOWA - STREFA NIEWYKRYWALNOŚCI (NDZ) METOD PASYWNYCH (18.2.6, 18.3.3, 18.3.6)

INTUICJA: gdy sieć zniknie (wyłącznik otworzył się), falownik PV dalej "pompuje" moc do lokalnych odbiorów.
Jeśli produkcja ≈ zużycie, NIC się nie zmienia - napięcie i częstotliwość zostają w normie, a falownik
nie wie, że jest sam. To niebezpieczne: monter myśli, że linia jest bez napięcia! (wyspa niezamierzona)
Metody PASYWNE (U<, U>, f<, f>) wykrywają wyspę tylko wtedy, gdy niedopasowanie mocy jest wystarczające.

MODEL (standard badań IEEE 1547 / UL 1741): obciążenie równoległe RLC o dobroci Qf, falownik cos φ = 1.
  Niedopasowanie mocy czynnej ΔP/P i biernej ΔQ/P przed wyspieniem.
  Po wyspieniu rezystancja R musi "zjeść" dokładnie moc falownika:   U'² = U²·P_OZE / P_obc
  Moc bierna musi się zbilansować w samym obciążeniu (Q_L = Q_C):  f' = f0·√(Q_L/Q_C)
  Dobroć: Qf = R·√(C/L) = √(Q_L·Q_C)/P  (norma testowa: Qf = 1 lub 2,5).
Falownik wyłącza się, jeśli U' ∉ [U_min, U_max] lub f' ∉ [f_min, f_max].
NDZ = zbiór (ΔP, ΔQ), w którym NIE wyłączy się -> wyspa trwa.

WYKRESY: (1) mapa ΔP-ΔQ: zielony prostokąt = NDZ, kropka = bieżący punkt; (2) U' i f' wzdłuż ΔP
dla bieżącego ΔQ z progami.
SPRÓBUJ: zwiększ Qf (obciążenie "bardziej rezonansowe", np. dużo kondensatorów i silników) - NDZ w osi ΔQ
rośnie. Zawęź progi f - NDZ maleje, ale rośnie ryzyko fałszywych wyłączeń przy zwykłych wahaniach sieci.
Dlatego pasywne metody same nie wystarczą -> zakładki 06 (ROCOF) i 07 (metody aktywne).
"""

EXPL["T06"] = """06. ROCOF I ODCIĄŻANIE CZĘSTOTLIWOŚCIOWE (SCO) W WYSPIE (18.2.8, 18.3.3)

INTUICJA: częstotliwość sieci to "tętno" wirujących maszyn. Gdy brakuje mocy, wirniki zwalniają (oddają
energię kinetyczną), gdy jej za dużo - przyspieszają. Wyspa ma małą bezwładność (małe generatory, dużo
falowników), więc częstotliwość spada BARDZO szybko.

RÓWNANIE RUCHU (równanie wahadłowe, w p.u.):
      2H · dΔf/dt = P_m - P_e,     P_e = P_obc·(1 + D·Δf)
  H - stała bezwładności [s] (ile sekund maszyna mogłaby oddawać moc znamionową z samej energii wirowania),
  D - samoregulacja odbiorów (silniki przy niższej f pobierają mniej).
  Tuż po wyspieniu:  ROCOF = df/dt ≈ -f0 · ΔP / (2H)     -> np. 50·0,2/(2·3) = 1,67 Hz/s.
Regulator (statyzm R, stała Tg) dodaje moc, ale OZE pracujące w MPPT nie mają rezerwy (produkują maksimum)!

ODCIĄŻANIE (Under-Frequency Load Shedding):
 • statyczne - stopnie: 49,0 / 48,8 / 48,6 / 48,4 / 48,2 Hz, każdy odłącza stałą część odbiorów (po 0,1 s);
 • adaptacyjne (dynamiczne) - z ROCOF szacuje deficyt  ΔP ≈ 2H·|df/dt|/f0  i od razu odłącza właściwą ilość.
Przekaźnik ROCOF (df/dt) jest też pasywną metodą wykrywania wyspy - duży ROCOF = utrata sieci.

WYKRESY: (1) f(t) bez odciążania, statycznie i adaptacyjnie (granica 47,5 Hz = wyłączenie generatorów);
(2) ROCOF i próg przekaźnika; (3) pozostałe obciążenie.
SPRÓBUJ: zmniejsz H do 1 s (sieć "falownikowa") - częstotliwość spada 3x szybciej; odciążanie statyczne
nie nadąża, adaptacyjne ratuje wyspę. Zwiększ rezerwę - potrzeba mniej odciążania.
"""

EXPL["T07"] = """07. AKTYWNE METODY ANTYWYSPOWE: AFD, AFDPF, SMS (18.3.4, 18.3.5)

INTUICJA: skoro wyspa może "udawać" normalną sieć (NDZ), falownik sam lekko "szturcha" system.
Gdy sieć jest - szturchnięcie nic nie daje (sieć jest sztywna). W wyspie - szturchnięcie się kumuluje
i częstotliwość "ucieka" poza próg -> wykrycie.

WARUNEK FAZY: w wyspie prąd falownika płynie przez obciążenie RLC, więc kąt fazowy obciążenia musi być
równy kątowi narzuconemu przez falownik:
      atan( Qf·(f/f_r - f_r/f) ) = θ_falownika(f)
 • AFD (active frequency drift) - falownik skraca każdą półfalę prądu o ułamek cf ("chopping fraction"):
      θ = π·cf/2  (stała) -> częstotliwość przesuwa się o stałą wartość. Przy dużym Qf może nie wyjść poza próg.
 • AFDPF - AFD z dodatnim sprzężeniem: cf = cf0 + K·(f - f0) -> im dalej od 50 Hz, tym mocniej pcha -> ucieka.
 • SMS (slip-mode frequency shift) - θ = θm·sin(π/2·(f - f0)/(fm - f0)).
      Wyspa jest niestabilna (wykryta), gdy nachylenie krzywej falownika > nachylenia krzywej obciążenia:
      θm·π/(2(fm-f0)) > 2Qf/f0.  To dlatego duże Qf tworzy NDZ także dla metod aktywnych!
Cena: celowe zniekształcenie prądu (THD) i ryzyko "kłótni" wielu falowników - ograniczenie przy dużej penetracji.
METODY HYBRYDOWE (18.3.5) łączą pasywne (tanie, bez zakłóceń) z aktywnymi (włączanymi dopiero,
gdy pasywna coś podejrzewa).

WYKRESY: (1) częstotliwość po wyspieniu cykl po cyklu; (2) krzywe fazowe: obciążenie vs metody;
(3) czas wykrycia w funkcji Qf (granica 2 s wg IEEE 1547).
SPRÓBUJ: ustaw Qf = 2,5 i f_r = 50 Hz: SMS przestaje wykrywać (NDZ). Zwiększ θm - znów działa.
"""

EXPL["T08"] = """08. MIKROSIEĆ: TRYB SIECIOWY vs WYSPOWY, ZABEZPIECZENIA ADAPTACYJNE, FCL (18.4.1-18.4.4, 18.4.8, 18.4.9)

INTUICJA: w trybie sieciowym zwarcie "karmi" potężna sieć przez transformator - prąd zwarcia jest np.
20x większy od prądu obciążenia, zwykły przekaźnik nadprądowy łatwo go odróżnia. W trybie wyspowym
zostają głównie falowniki, które ze względu na tranzystory ograniczają prąd do ~1,2-1,5 I_n.
Prąd zwarcia bywa wtedy PODOBNY do prądu obciążenia - przekaźnik nastawiony "na sieć" nie zadziała!

ROZWIĄZANIA z rozdziału:
 • zabezpieczenie adaptacyjne (18.4.4) - dwie grupy nastaw: A (sieć) i B (wyspa), przełączane sygnałem
   o stanie wyłącznika PCC (potrzebna komunikacja i znajomość wszystkich konfiguracji);
 • urządzenia zewnętrzne (18.4.9): magazyn/koło zamachowe o dużej przeciążalności PODNOSI prąd zwarcia
   w wyspie; ogranicznik prądu FCL w PCC OBNIŻA udział sieci -> oba tryby stają się podobne;
 • metody oparte na napięciu, składowych symetrycznych, różnicowe, odległościowe (zakładka 09).

WZORY (moduły, szacunek górny):
  I_sieć = E / |Z_T + Z_FCL + Z_l|,  I_SG = E / |Z''_d + Z_l|,  I_fal = k_fal·I_n,fal,  I_mag = k_mag·I_n,mag
  Z_T = u_k·U²/S_T,  E = U/√3.
WYKRESY: (1) słupki prądów zwarciowych w obu trybach vs prąd obciążenia i nastawy;
(2) charakterystyki grup A i B z punktami zwarć; (3) prąd zwarcia w wyspie w funkcji udziału falowników.
SPRÓBUJ: udział falowników 100 %, magazyn 0 -> w wyspie grupa A nie widzi zwarcia ("∞").
Dodaj magazyn z k = 3 lub przełącz na adaptację - zabezpieczenie znowu działa.
"""

EXPL["T09"] = """09. ZABEZPIECZENIE ODLEGŁOŚCIOWE I RÓŻNICOWE W SIECI Z OZE (18.4.6, 18.4.7)

ODLEGŁOŚCIOWE (21): przekaźnik liczy impedancję  Z = U / I. Linia ma impedancję proporcjonalną do długości,
więc Z mówi "jak daleko" jest zwarcie. Strefa 1 (natychmiast) obejmuje ~80 % linii, strefa 2 (z opóźnieniem)
~120 %. Charakterystyka MHO to okrąg na płaszczyźnie R-X.
Problemy z OZE (wg rozdziału):
  • DOPŁYW (infeed): OZE między przekaźnikiem a zwarciem -> przekaźnik widzi impedancję WIĘKSZĄ:
        Z_pozorna = z·d + (I_OZE/I_p)·z·(d - m) + R_f·(1 + I_OZE/I_p)
    -> zwarcie "wydaje się" dalej, strefa 1 skraca się (niedozasięg), czas wyłączenia rośnie;
  • rezystancja przejścia R_f przesuwa punkt w prawo, a z dopływem jest dodatkowo "mnożona".

RÓŻNICOWE (87): porównuje prądy na obu końcach strefy. Zdrowa strefa: co wpłynęło, to wypłynęło,
I_d = |I1 + I2| ≈ 0. Zwarcie wewnątrz: I_d duży. Stabilizacja: zadziałanie, gdy
      I_d > max(I_min, k·I_s),  I_s = (|I1| + |I2|)/2 (prąd hamujący).
Zalety: nie zależy od kierunku przepływu mocy ani od poziomu prądu zwarciowego -> dobre dla mikrosieci.
Wady (rozdział): wymaga łącza komunikacyjnego (awaria = brak ochrony), synchronizacji pomiarów,
błędy przekładników przy zwarciach zewnętrznych, wyższy koszt.

WYKRESY: (1) płaszczyzna R-X, strefy MHO, trajektoria impedancji pozornej przy przesuwaniu miejsca zwarcia;
(2) charakterystyka stabilizowana różnicowa z punktami: zwarcie zewnętrzne, wewnętrzne (sieć), wewnętrzne (wyspa).
SPRÓBUJ: zwiększ I_OZE/I_p do 2 - zwarcie w 70 % linii "wychodzi" poza strefę 1. Zwiększ błąd przekładników
do 20 % i obniż nachylenie k - zwarcie zewnętrzne zaczyna powodować fałszywe zadziałanie.
"""

EXPL["T10"] = """10. SPZ (SAMOCZYNNE PONOWNE ZAŁĄCZENIE) PRZY PRACUJĄCYM OZE (18.4.1, 18.2.7)

INTUICJA: po zwarciu reklozer otwiera linię na krótką "przerwę beznapięciową" (0,3-1 s) i załącza ponownie.
Jeśli OZE NIE zostało w tym czasie odłączone:
 1) podtrzymuje napięcie -> łuk w miejscu zwarcia nie gaśnie -> SPZ nieudane (zwarcie przemijające staje się trwałe);
 2) wyspa z OZE "odpływa" częstotliwością od sieci (niedopasowanie mocy) -> kąt między napięciem wyspy
    i sieci rośnie. Załączenie przy dużym kącie = zderzenie dwóch układów niesynchronicznych:
    ogromny prąd wyrównawczy i udar momentu na wale generatora (może uszkodzić wał, sprzęgło, przekładnię).

WZORY:
  częstotliwość wyspy:  2H dΔf/dt = -ΔP - D·Δf   ->  Δf(t) = -(ΔP/D)·(1 - e^(-D t / 2H))
  kąt:                  δ(t) = 360°·f0·∫Δf dt
  prąd przy załączeniu: I ≈ 2·E·sin(δ/2) / (X''_d + X_s)     (dla δ = 180° -> 2x prąd zwarcia 3-fazowego!)
Dlatego: (a) zabezpieczenie antywyspowe OZE musi zadziałać SZYBCIEJ niż przerwa SPZ, lub
         (b) SPZ z kontrolą synchronizmu (25): załączenie tylko przy |δ| < ~20°, |Δf| < ~0,2 Hz, |ΔU| < 10 %.

WYKRESY: (1) kąt δ(t) w wyspie z zaznaczonym momentem SPZ i granicą synchronizmu; (2) prąd załączenia
w funkcji czasu przerwy dla kilku niedopasowań; (3) oś czasu: zwarcie - otwarcie - odłączenie OZE - SPZ.
SPRÓBUJ: ustaw czas zadziałania antywyspowego większy niż przerwa SPZ - wynik: "SPZ NIEUDANE".
"""

EXPL["T11"] = """11. ELEKTROWNIA WIATROWA STAŁOOBROTOWA Z GENERATOREM KLATKOWYM (SCIG) (18.5.1, rys. 18.6)

INTUICJA: silnik indukcyjny klatkowy, napędzany wiatrem SZYBCIEJ niż pole wirujące (poślizg s < 0),
staje się generatorem. Pracuje w wąskim zakresie prędkości (s ≈ -1 %), dlatego "stałoobrotowy".
Zalety: prosty, tani, niezawodny, nie wymaga synchronizacji. Wady: ZAWSZE pobiera moc bierną
(magnesowanie) -> potrzebna bateria kondensatorów; duży prąd rozruchowy -> soft-starter (rys. 18.6b);
podmuchy wiatru przenoszą się na moment i przekładnię.

SCHEMAT ZASTĘPCZY (p.u.):  Z = R_s + jX_s + jX_m ∥ (R_r/s + jX_r)
  I_s = U/Z,   moment  T = |I_r|²·R_r/s,   S = U·I_s*  (P<0 = oddawanie mocy, Q>0 = pobór biernej).
Kondensatory: Q_C = U²·B_C (maleje z kwadratem napięcia - przy zapadzie nie pomagają!)
ROZRUCH: 2H dω/dt = T_e(s,U) - T_obc. Przy rozruchu bezpośrednim prąd ~ 4-7 I_n.
Soft-starter (tyrystory) obniża napięcie -> prąd ~ U, ale moment ~ U² -> rozruch dłuższy.

WYKRESY: (1) moment w funkcji poślizgu (silnik / generator), zaznaczone momenty krytyczne;
(2) P i Q w okolicy synchronizmu z kompensacją; (3) prąd i prędkość przy rozruchu bezpośrednim i z soft-starterem.
SPRÓBUJ: ustaw ograniczenie prądu soft-startera 2 p.u. i dużą bezwładność H - rozruch trwa bardzo długo
(albo utyka - moment < opór!). To kompromis: mniejszy udar dla sieci vs czas i grzanie uzwojeń.
"""

EXPL["T12"] = """12. AERODYNAMIKA TURBINY, MPPT, STAŁA vs ZMIENNA PRĘDKOŚĆ (18.5.2, rys. 18.7, tabela 18.1)

INTUICJA: moc wiatru rośnie z SZEŚCIANEM prędkości (2x szybszy wiatr = 8x więcej mocy).
Łopaty mogą przechwycić maks. 59,3 % (granica Betza), realnie Cp ≈ 0,45-0,50.
      P = ½·ρ·π·R²·Cp(λ, β)·v³,    λ = ω·R / v  (wyróżnik szybkobieżności - prędkość końca łopaty / wiatru)
Cp ma maksimum dla jednego λ_opt (tu ≈ 8,1). Żeby zawsze pracować w maksimum, prędkość wirnika musi rosnąć
proporcjonalnie do wiatru: ω = λ_opt·v/R - to MPPT (śledzenie punktu mocy maksymalnej): P_opt = k·ω³.
Turbina STAŁOOBROTOWA (SCIG) ma jedno ω -> Cp optymalny tylko przy jednej prędkości wiatru.
Turbina ZMIENNOOBROTOWA potrzebuje przekształtnika (pełnego - rys. 18.7, lub częściowego - DFIG).

TABELA 18.1 (skrót): DC - prosty, ale regulacja napięcia DC; PMSG - bez przekładni, mało strat, drogie magnesy,
ryzyko rozmagnesowania; EESG - bez poboru Q, potrzebne wzbudzenie; SCIG + pełny przekształtnik - tani i
solidny, ale drogi przekształtnik 100 %; DFIG - przekształtnik tylko 25-30 %, ale pierścienie ślizgowe;
BDFIG - bezszczotkowy (offshore); SRG - prosty, ale tętnienia momentu i hałas.

ENERGIA ROCZNA: rozkład wiatru Weibulla (k = 2, Rayleigh):  E = 8760 h · Σ P(v)·f(v)·Δv.
WYKRESY: (1) moc turbiny w funkcji prędkości wirnika dla różnych v + krzywa MPPT + linia stałej prędkości;
(2) Cp(λ) dla różnych kątów łopat β; (3) krzywe mocy i rozkład wiatru -> zysk energii ze zmiennej prędkości.
SPRÓBUJ: przesuń prędkość stałoobrotowej tak, by jej Cp było maks. przy 7 m/s - zysk MPPT wciąż kilka-kilkanaście %.
"""

EXPL["T13"] = """13. DFIG - PRZEPŁYW MOCY W PRACY POD- I NADSYNCHRONICZNEJ (18.5.3, rys. 18.8, tabela 18.2)

INTUICJA: DFIG (maszyna dwustronnie zasilana) ma stojan przyłączony WPROST do sieci, a wirnik przez
przekształtnik "plecy w plecy". Przekształtnik wstrzykuje do wirnika prąd o częstotliwości poślizgu
f_r = s·f_s, dzięki czemu maszyna może pracować w zakresie ±30 % wokół prędkości synchronicznej.

BILANS MOCY (bez strat):
      P_s = P_m / (1 - s),     P_r = -s·P_s,     P_m = P_s + P_r
 • praca PODSYNCHRONICZNA (s > 0, wolniej niż pole): P_r < 0 - wirnik POBIERA moc z sieci przez przekształtnik;
 • praca NADSYNCHRONICZNA (s < 0): P_r > 0 - moc płynie z wirnika DO sieci (obie drogi oddają energię).
Przekształtnik przenosi tylko |s|·P_s -> dla s_max = 30 % ma moc ~30 % mocy turbiny - to główna zaleta
(koszt!). Napięcie wirnika: U_r ≈ |s|·U_s / ϑ (ϑ - przekładnia stojan/wirnik) - małe przy małym poślizgu.

TABELA 18.2: mostek diodowy + falownik tyrystorowy - tylko nadsynchronicznie; SCR/SCR - obustronnie,
ale komutacja i harmoniczne; back-to-back PWM (standard dla MW) - mało harmonicznych, potrzebny kondensator DC;
przekształtnik macierzowy - bez obwodu DC, ale dużo łączników.

WYKRESY: (1) P_m, P_s, P_r i poślizg w funkcji prędkości wiatru (MPPT + ograniczenie prędkości);
(2) P_s i P_r w funkcji prędkości generatora dla stałej mocy mechanicznej, obszar pracy przekształtnika.
SPRÓBUJ: zmniejsz s_max do 10 % - przekształtnik tańszy, ale turbina szybciej traci MPPT przy słabym wietrze.
"""

EXPL["T14"] = """14. DFIG - ZAPAD NAPIĘCIA I CROWBAR (symulacja dynamiczna RK4) (18.5.4, rys. 18.9)

CO SIĘ DZIEJE: przy zwarciu w sieci napięcie na zaciskach stojana gwałtownie spada. Strumień stojana nie
może zmienić się skokowo (jak prąd w cewce) - pojawia się "naturalna", nieruchoma składowa strumienia.
Wirnik obraca się względem niej z prędkością ~ω -> indukuje się w nim duże napięcie (dla całego zapadu
nawet ~ (1-s)·U, zamiast normalnego s·U!). Przekształtnik RSC ma za małe napięcie, by to zrównoważyć -
TRACI KONTROLĘ nad prądem wirnika. Prąd wirnika i napięcie obwodu DC skaczą -> zniszczenie tranzystorów.

MODEL (wektory przestrzenne w układzie stojana, p.u., ω_b = 2π·50):
      dψs/dt = ω_b (u_s - R_s i_s)
      dψr/dt = ω_b (u_r - R_r i_r + j·ω_r·ψr)
      ψs = L_s i_s + L_m i_r,   ψr = L_m i_s + L_r i_r
RSC: regulator PI prądu wirnika z ograniczeniem |u_r| ≤ U_r,max. Obwód DC: C·dU_dc/dt = P_RSC - P_GSC - P_chopper.

OCHRONA SPRZĘTOWA (rys. 18.9a):
 • CROWBAR na wirniku: gdy |i_r| > próg, tyrystory/IGBT zwierają wirnik przez rezystor R_cb, RSC zostaje
   zablokowany (prąd płynie obok przekształtnika); aktywny crowbar (IGBT) można wyłączyć po ustaniu stanu przejściowego;
 • CHOPPER (rezystor hamujący) na obwodzie DC - "spala" nadmiar energii, gdy U_dc > 1,1 p.u.;
 • alternatywy: magazyn energii w obwodzie DC, łącznik w stojanie, szeregowy przekształtnik podtrzymujący napięcie.
W czasie działania crowbara DFIG zachowuje się jak zwykły silnik klatkowy - pobiera moc bierną.

WYKRESY: (1) napięcie sieci; (2) prąd wirnika bez i z crowbarem + prąd przekształtnika RSC;
(3) napięcie obwodu DC; (4) moment elektromagnetyczny (udar na przekładnię).
SPRÓBUJ: zwiększ R_cb - prąd wirnika szybciej zanika, ale przy zbyt dużym R_cb rośnie napięcie wirnika
(ryzyko przepięcia RSC). Zmniejsz głębokość zapadu do 30 % - crowbar może w ogóle nie zadziałać.
"""

EXPL["T15"] = """15. FOTOWOLTAIKA: CHARAKTERYSTYKA I-U, BEZPIECZNIKI STRINGÓW, PRZEPIĘCIA (18.6, rys. 18.10)

BUDOWA: ogniwo -> moduł (ogniwa szeregowo) -> string (moduły szeregowo, wysokie napięcie) ->
pole (stringi równolegle w skrzynce przyłączeniowej) -> falownik -> sieć.
MODEL JEDNODIODOWY:  I = I_ph - I_0·(e^((U + I·R_s)/(n·N_c·U_T)) - 1),   I_ph ~ nasłonecznienie G,
U_oc maleje o ~0,28 %/K (zimno = WYŻSZE napięcie!).

OCHRONA DC (rys. 18.10a):
 • BEZPIECZNIKI STRINGÓW: przy zwarciu w jednym stringu pozostałe (N_p - 1) stringów "wpychają" w niego prąd
   wsteczny ~ (N_p - 1)·1,25·I_sc. Jeśli przekroczy dopuszczalny prąd wsteczny modułu I_R -> pożar.
   Dlatego przy N_p ≥ 3 zwykle potrzebne bezpieczniki gPV o prądzie ~1,5-2,4·I_sc (i ≤ I_R).
 • NAPIĘCIE MAKSYMALNE: U_oc przy najniższej temperaturze (np. -25 °C) × liczba modułów ≤ 1000/1500 V
   falownika i odgranicznika przepięć (SPD).
 • PRZEPIĘCIA OD PIORUNA (IEC 62305-2): nawet uderzenie OBOK instalacji indukuje w pętli okablowania napięcie
      U = (μ0/2π)·b·ln((d + a)/d)·di/dt,   di/dt pioruna ~ 25-100 kA/µs.
   -> minimalizuj powierzchnię pętli (+ i - prowadzone razem), stosuj SPD typu 1/2 po obu stronach falownika.
OCHRONA AC (rys. 18.10b): wyłącznik nadprądowy, RCD, SPD, zabezpieczenie antywyspowe falownika.

WYKRESY: (1) I-U i P-U pola z punktem MPP; (2) prąd wsteczny w stringu vs I_R; (3) napięcie stringu
w zimie vs limit falownika; (4) napięcie indukowane w pętli vs odległość uderzenia.
SPRÓBUJ: temperatura minimalna -25 °C i 24 moduły -> przekroczenie 1000 V! Zmniejsz szerokość pętli do 0,1 m.
"""

EXPL["T16"] = """16. SIECI PRZYSZŁOŚCI: MOC ZWARCIOWA, ZDOLNOŚĆ WYŁĄCZALNA, KOMUNIKACJA (18.7, rys. 18.11)

PROBLEM: każde źródło wirujące (np. biogazownia, mała elektrownia wodna) dokłada swój prąd zwarciowy.
Łączny prąd w stacji może przekroczyć ZDOLNOŚĆ WYŁĄCZALNĄ wyłącznika (np. 16 kA) - wyłącznik
nie przerwie zwarcia i eksploduje. Rozdział wymienia 3 środki:
 1) zmiana "okien" (nastaw) zabezpieczeń - najrozsądniejsza, wymaga komunikacji (np. PLC - power line carrier);
 2) przyłączenie OZE w innym miejscu sieci (dalej od szyn = większa impedancja = mniejszy wkład);
 3) dodatkowa reaktancja szeregowa (dławik) ograniczająca wkład OZE do prądu zwarcia.
WZÓR: I_k = E/|Z_s| + N·E/|Z_OZE + Z_dławika + z·l|,   Z_OZE = jX''_d·U²/S.

KOMUNIKACJA (rys. 18.11): operator systemu przesyłowego (TSO) <-> operator sieci dystrybucyjnej (DNO/OSD)
<-> handlowiec (trader) <-> klienci/prosumenci i OZE. DNO pilnuje ograniczeń mocy czynnej i biernej OZE,
koordynuje napięcia, zabezpieczenia antywyspowe (sygnał "transfer trip") i nastawy adaptacyjne.

WYKRESY: (1) prąd zwarcia na szynach w funkcji liczby przyłączonych OZE, z dławikiem i bez, vs zdolność
wyłączalna; (2) schemat przepływu informacji z rys. 18.11.
SPRÓBUJ: 10 źródeł po 4 MVA - przekroczenie; dołóż 20 % reaktancji dławika albo przenieś OZE 5 km od stacji.
"""

EXPL["T17"] = """PYTANIA KONTROLNE 18.8 - ODPOWIEDZI (skrót ucznia-inżyniera)

1. MIKROSIEĆ - lokalny zespół źródeł rozproszonych (PV, wiatr, mikroturbiny, ogniwa paliwowe), magazynów
   (baterie, koła zamachowe, superkondensatory) i odbiorów, sterowany jako całość, mogący pracować
   z siecią lub w wyspie. Schemat - zakładka 00 (rys. 18.1): sieć - transformator - PCC/wyłącznik - szyny
   mikrosieci - źródła przez falowniki / generatory - odbiory krytyczne i niekrytyczne.

2. JAKOŚĆ ENERGII: harmoniczne (falowniki, nieliniowości maszyn), migotanie (flicker - wahania wiatru, rozruchy
   dużych turbin w słabej sieci), niesymetria (jednofazowe mikroinstalacje), wahania napięcia i przepięcia
   (przepływ zwrotny - zakładka 01), zapady przy przyłączaniu SCIG (zakładka 11).

3. WYSPA ZAMIERZONA - planowana praca mikrosieci po odłączeniu (poprawa niezawodności, sterowanie U i f,
   odciążanie, zmiana nastaw, resynchronizacja). NIEZAMIERZONA - OZE nie wykryło utraty sieci i dalej zasila
   fragment sieci: zagrożenie dla monterów, złe napięcie/częstotliwość, SPZ w przeciwfazie, brak uziemienia.

4. WYKRYWANIE WYSPY: (a) zdalne/komunikacyjne - PLC, transfer trip, SCADA (niezawodne, drogie);
   (b) lokalne pasywne - U, f, ROCOF, dP/dt, THD, niesymetria, falki, AI (proste, ale NDZ - zakładka 05, 06);
   (c) lokalne aktywne - pomiar impedancji, AFD, AFDPF, SMS, ALPS, APS, wstrzykiwanie składowej przeciwnej
   (mała NDZ, ale pogarszają jakość energii - zakładka 07); (d) hybrydowe - pasywna uruchamia aktywną.
   Porównanie: NDZ, koszt, fałszywe wyłączenia, wpływ na jakość energii; wymóg: odłączenie ≤ 2 s.

5. KLASYFIKACJA WECS: wg prędkości - stałoobrotowe (SCIG, rys. 18.6) i zmiennoobrotowe; zmiennoobrotowe
   wg przekształtnika - pełny (SCIG, DC, EESG, wielobiegunowe SG, PMSG - rys. 18.7) i częściowy (DFIG, rys. 18.8).

6. RÓŻNE GENERATORY AC: SCIG + kondensatory + soft-start (stała prędkość, pobór Q); SCIG + pełny przekształtnik
   (pełna kontrola, straty 100 % mocy w przekształtniku); SG z diodowym prostownikiem (bez MPPT) lub PWM;
   PMSG wielobiegunowy bez przekładni; DFIG z przekształtnikiem w wirniku (zakładki 11-13).

7. CROWBAR - zwieracz wirnika DFIG (tyrystory lub IGBT + rezystor) przejmujący prąd wirnika przy zapadzie
   napięcia, by chronić przekształtnik RSC. Aktywny crowbar (IGBT) może być wyłączony zaraz po stanie
   przejściowym - spełnia wymagania LVRT kodeksów sieciowych. Schemat 18.9b: pomiar i_r i U_dc -> komparatory
   -> załączenie IGBT crowbara i choppera DC -> blokada impulsów RSC -> po zaniku stanu przejściowego powrót
   (zakładka 14).

8. OBWODY SPRZĘTOWE OCHRONY WECS: crowbar wirnika (pasywny/aktywny), chopper (rezystor) w obwodzie DC,
   magazyn energii w obwodzie DC, łącznik elektroniczny w stojanie, szeregowy przekształtnik (dynamiczny
   stabilizator napięcia) na zaciskach, ograniczniki przepięć.

9. SCHEMAT PV Z OCHRONĄ: moduły -> stringi (bezpieczniki gPV, diody) -> skrzynka przyłączeniowa (SPD DC,
   rozłącznik DC) -> falownik (MPPT, antywyspowe, monitoring izolacji) -> SPD AC, wyłącznik nadprądowy, RCD
   -> licznik -> sieć; plus instalacja odgromowa wg IEC 62305 i połączenia wyrównawcze (zakładka 15).

10. ROLA DNO (operatora sieci dystrybucyjnej): bilansuje wymianę mocy w sieci dystrybucyjnej, nakłada
   ograniczenia na moc czynną i bierną OZE, współpracuje z TSO i handlowcem, koordynuje sterowanie napięciem,
   ochronę antywyspową i adaptacyjne nastawy zabezpieczeń dzięki dwukierunkowej komunikacji (np. PLC) (zakładka 16).
"""


# =============================================================================
# GUI - klasa bazowa zakładki
# =============================================================================

def style_setup(root):
    """Ciemny motyw ttk (Catppuccin)."""
    st = ttk.Style(root)
    try:
        st.theme_use("clam")
    except tk.TclError:
        pass
    st.configure(".", background=C["bg"], foreground=C["text"], fieldbackground=C["panel"])
    st.configure("TFrame", background=C["bg"])
    st.configure("TLabel", background=C["bg"], foreground=C["text"])
    st.configure("Head.TLabel", foreground=C["sky"], font=("Segoe UI", 11, "bold"))
    st.configure("TButton", background=C["panel"], foreground=C["text"])
    st.map("TButton", background=[("active", C["grid"])])
    st.configure("TNotebook", background=C["bg"])
    st.configure("TNotebook.Tab", background=C["panel"], foreground=C["sub"], padding=(8, 3))
    st.map("TNotebook.Tab", background=[("selected", C["ax"])],
           foreground=[("selected", C["green"])])
    st.configure("TCombobox", fieldbackground=C["panel"], background=C["panel"],
                 foreground=C["text"], arrowcolor=C["text"])
    st.configure("TCheckbutton", background=C["bg"], foreground=C["text"])
    st.configure("TPanedwindow", background=C["bg"])
    root.option_add("*TCombobox*Listbox.background", C["panel"])
    root.option_add("*TCombobox*Listbox.foreground", C["text"])


class ExampleTab(ttk.Frame):
    """Zakładka: lewy panel (suwaki, wybory, wyniki), prawy - wykresy
    i objaśnienie. Podklasy definiują PARAMS, CHOICES, CHECKS, key, update_plot()."""
    PARAMS = []    # (nazwa, etykieta, min, max, domyślna, krok)
    CHOICES = []   # (nazwa, etykieta, [opcje], domyślna)
    CHECKS = []    # (nazwa, etykieta, domyślna bool)
    key = ""

    def __init__(self, master):
        super().__init__(master)
        self.vars = {}
        self._pending = None
        pw = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        pw.pack(fill=tk.BOTH, expand=True)
        left_outer = ttk.Frame(pw, width=320)
        right = ttk.Frame(pw)
        pw.add(left_outer, weight=0)
        pw.add(right, weight=1)

        # przewijany panel parametrów
        cv = tk.Canvas(left_outer, bg=C["bg"], highlightthickness=0, width=315)
        sb = ttk.Scrollbar(left_outer, orient=tk.VERTICAL, command=cv.yview)
        left = ttk.Frame(cv)
        left.bind("<Configure>", lambda e: cv.configure(scrollregion=cv.bbox("all")))
        cv.create_window((0, 0), window=left, anchor="nw")
        cv.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        cv.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        cv.bind("<Enter>", lambda e: cv.bind_all("<MouseWheel>",
                lambda ev: cv.yview_scroll(int(-ev.delta / 120), "units")))
        cv.bind("<Leave>", lambda e: cv.unbind_all("<MouseWheel>"))

        ttk.Label(left, text="Parametry", style="Head.TLabel").pack(anchor="w", padx=6, pady=(6, 2))
        for name, label, opts, val in self.CHOICES:
            fr = ttk.Frame(left)
            fr.pack(fill=tk.X, padx=6, pady=2)
            ttk.Label(fr, text=label).pack(anchor="w")
            var = tk.StringVar(value=val)
            self.vars[name] = var
            cb = ttk.Combobox(fr, textvariable=var, values=opts, state="readonly", width=34)
            cb.pack(fill=tk.X)
            cb.bind("<<ComboboxSelected>>", lambda _e: self.schedule())
        for name, label, val in self.CHECKS:
            var = tk.BooleanVar(value=val)
            self.vars[name] = var
            ttk.Checkbutton(left, text=label, variable=var,
                            command=self.schedule).pack(anchor="w", padx=6, pady=1)
        for name, label, lo, hi, val, res in self.PARAMS:
            fr = ttk.Frame(left)
            fr.pack(fill=tk.X, padx=6, pady=0)
            var = tk.DoubleVar(value=val)
            self.vars[name] = var
            ttk.Label(fr, text=label).pack(anchor="w")
            sc = tk.Scale(fr, from_=lo, to=hi, resolution=res, orient=tk.HORIZONTAL,
                          variable=var, showvalue=True, length=280, bg=C["bg"],
                          fg=C["text"], troughcolor=C["panel"], highlightthickness=0,
                          activebackground=C["sky"], bd=0, font=("Segoe UI", 8),
                          command=lambda _e: self.schedule())
            sc.pack(fill=tk.X)
        ttk.Button(left, text="Przywróć domyślne", command=self.reset).pack(fill=tk.X, padx=6, pady=6)
        ttk.Label(left, text="Wyniki", style="Head.TLabel").pack(anchor="w", padx=6)
        self.result = tk.Text(left, height=16, width=40, font=("Consolas", 9), bg=C["ax"],
                              fg=C["green"], insertbackground=C["text"], bd=0)
        self.result.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))

        vpw = ttk.PanedWindow(right, orient=tk.VERTICAL)
        vpw.pack(fill=tk.BOTH, expand=True)
        figfr = ttk.Frame(vpw)
        self.fig = Figure(figsize=(10, 6), dpi=90)
        self.canvas = FigureCanvasTkAgg(self.fig, master=figfr)
        tb = NavigationToolbar2Tk(self.canvas, figfr)
        tb.update()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        txtfr = ttk.Frame(vpw)
        self.expl = ScrolledText(txtfr, height=12, wrap=tk.WORD, font=("Segoe UI", 10),
                                 bg=C["ax"], fg=C["text"], bd=0, padx=8, pady=6)
        self.expl.pack(fill=tk.BOTH, expand=True)
        self.expl.insert("1.0", EXPL.get(self.key, ""))
        self.expl.configure(state=tk.DISABLED)
        vpw.add(figfr, weight=3)
        vpw.add(txtfr, weight=1)
        self.canvas.get_tk_widget().bind("<Configure>", self._on_resize, add="+")
        self.refresh()

    def _on_resize(self, _e=None):
        try:
            self.fig.tight_layout()
        except Exception:
            pass

    def reset(self):
        for name, _l, _lo, _hi, val, _r in self.PARAMS:
            self.vars[name].set(val)
        for name, _l, _o, val in self.CHOICES:
            self.vars[name].set(val)
        for name, _l, val in self.CHECKS:
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
        except Exception as exc:          # nie wywracaj GUI przy nietypowych nastawach
            lines = ["Błąd obliczeń:", repr(exc)]
        try:
            self.fig.tight_layout()
        except Exception:
            pass
        self.result.delete("1.0", tk.END)
        self.result.insert("1.0", "\n".join(lines))
        self.canvas.draw_idle()

    def update_plot(self):
        raise NotImplementedError


def ok_txt(cond, good="OK", bad="PRZEKROCZENIE"):
    return good if cond else bad


def fmt_t(t):
    return "nie działa (∞)" if not np.isfinite(t) else f"{t:.3f} s"


# =============================================================================
# ZAKŁADKI
# =============================================================================

class Tab00(ExampleTab):
    key = "T00"
    PARAMS = [("pen", "Penetracja OZE [% obciążenia]", 0, 150, 60, 5)]

    def update_plot(self):
        pen = self.p("pen") / 100
        ax = self.fig.add_subplot(1, 2, 1)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")
        ax.set_title("Rys. 18.1 - mikrosieć z OZE przyłączona do sieci")

        def box(x, y, w, h, txt, col):
            ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                                        fc=C["panel"], ec=col, lw=1.6))
            ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=8, color=col)

        box(3.8, 8.6, 2.4, 1.0, "Sieć zasilająca\n(utility grid)", C["sky"])
        ax.plot([5, 5], [8.6, 7.7], color=C["sub"])
        ax.add_patch(Circle((5, 7.4), 0.32, fill=False, ec=C["yellow"], lw=1.5))
        ax.add_patch(Circle((5, 6.95), 0.32, fill=False, ec=C["yellow"], lw=1.5))
        ax.text(5.5, 7.15, "Transformator\nprzyłączeniowy", fontsize=7, color=C["yellow"], va="center")
        ax.plot([5, 5], [6.63, 6.0], color=C["sub"])
        box(4.55, 5.55, 0.9, 0.45, "CB", C["red"])
        ax.text(5.6, 5.77, "PCC - punkt wspólnego\nprzyłączenia (ochrona 18.2/18.3)",
                fontsize=7, color=C["red"], va="center")
        ax.plot([5, 5], [5.55, 5.0], color=C["sub"])
        ax.plot([0.8, 9.2], [5.0, 5.0], color=C["text"], lw=3)
        ax.text(0.8, 5.2, "szyny mikrosieci", fontsize=7, color=C["sub"])
        items = [(0.3, "PV\n+ falownik", C["yellow"]), (2.1, "Turbina\nwiatrowa", C["teal"]),
                 (3.9, "Magazyn\nenergii", C["mauve"]), (5.7, "Mikroturbina\n/ CHP", C["peach"]),
                 (7.5, "Odbiory\nkrytyczne", C["green"])]
        for x, t, col in items:
            ax.plot([x + 0.8, x + 0.8], [5.0, 3.9], color=C["sub"])
            box(x + 0.55, 3.6, 0.5, 0.3, "", C["red"])
            ax.plot([x + 0.8, x + 0.8], [3.6, 2.9], color=C["sub"])
            box(x, 1.7, 1.6, 1.2, t, col)
        ax.text(5, 0.8, "Każde źródło ma własną ochronę generatora (rys. 18.4)", ha="center",
                fontsize=8, color=C["sub"])

        ax2 = self.fig.add_subplot(1, 2, 2)
        cats = ["Przepływ\ndwukierunkowy", "Wzrost napięcia", "Zmiana prądów\nzwarciowych",
                "Ryzyko wyspy", "Jakość energii"]
        score = np.clip(np.array([1.0, 0.9, 0.8, 1.1, 0.6]) * pen, 0, 1.5)
        cols = [C["green"] if s < 0.5 else C["yellow"] if s < 1.0 else C["red"] for s in score]
        ax2.barh(cats, score, color=cols)
        ax2.axvline(0.5, color=C["yellow"], ls="--", lw=1)
        ax2.axvline(1.0, color=C["red"], ls="--", lw=1)
        ax2.set_xlim(0, 1.5)
        ax2.set_xlabel("względne nasilenie problemu (poglądowo)")
        ax2.set_title(f"Wpływ penetracji OZE = {pen*100:.0f} % na ochronę (18.1)")
        return ["Mapa rozdziału -> zakładki:",
                " 18.1/18.2.2  -> 01", " 18.2.4/18.4.1 -> 02, 03", " 18.2.5 -> 04",
                " 18.2.6/18.3  -> 05, 06, 07", " 18.2.8 -> 06", " 18.4 -> 08, 09, 10",
                " 18.5 -> 11, 12, 13, 14", " 18.6 -> 15", " 18.7 -> 16", " 18.8 -> 17", "",
                f"Penetracja: {pen*100:.0f} %",
                "Wykres po prawej jest poglądowy:", "im większy udział OZE, tym więcej",
                "zjawisk z kolejnych zakładek staje", "się istotne dla zabezpieczeń."]


class Tab01(ExampleTab):
    key = "T01"
    PARAMS = [("L", "Długość linii [km]", 2, 30, 10, 0.5),
              ("Pl", "Obciążenie całkowite [MW]", 0.5, 10, 4, 0.1),
              ("pf", "cos φ obciążenia", 0.8, 1.0, 0.95, 0.01),
              ("Pdg", "Moc OZE P [MW]", 0, 15, 6, 0.1),
              ("Qdg", "Moc bierna OZE Q [Mvar] (+ oddaje)", -4, 4, 0, 0.1),
              ("node", "Węzeł przyłączenia OZE (1..20)", 1, 20, 15, 1),
              ("vs", "Napięcie stacji (LTC) [p.u.]", 0.95, 1.07, 1.03, 0.005)]

    def update_plot(self):
        L, Pl, pf, Pdg, Qdg = self.p("L"), self.p("Pl"), self.p("pf"), self.p("Pdg"), self.p("Qdg")
        node, vs = int(self.p("node")), self.p("vs")
        n, r, x, Vn = 20, 0.30, 0.35, 15.0
        V0, P0, l0 = feeder_loadflow(n, L, r, x, Pl, pf, Vn, node, 0, 0, vs)
        V1, P1, l1 = feeder_loadflow(n, L, r, x, Pl, pf, Vn, node, Pdg, Qdg, vs)
        dist = np.linspace(0, L, n + 1)
        ax = self.fig.add_subplot(2, 2, 1)
        ax.plot(dist, V0, "o-", ms=3, label="bez OZE")
        ax.plot(dist, V1, "s-", ms=3, label="z OZE")
        ax.axhline(1.05, color=C["yellow"], ls="--", lw=1, label="1,05 p.u.")
        ax.axhline(1.10, color=C["red"], ls="--", lw=1, label="1,10 p.u.")
        ax.axhline(0.95, color=C["yellow"], ls=":", lw=1)
        ax.axvline(dist[node], color=C["mauve"], lw=1, alpha=0.7)
        ax.text(dist[node], ax.get_ylim()[1], " OZE", color=C["mauve"], va="top", fontsize=8)
        ax.set_xlabel("odległość od stacji [km]")
        ax.set_ylabel("napięcie [p.u.]")
        ax.set_title("Profil napięcia wzdłuż linii 15 kV")
        ax.legend(loc="best")
        ax = self.fig.add_subplot(2, 2, 2)
        mid = (dist[:-1] + dist[1:]) / 2
        w = L / n * 0.4
        ax.bar(mid - w / 2, P0, width=w, label="bez OZE")
        ax.bar(mid + w / 2, P1, width=w, label="z OZE")
        ax.axhline(0, color=C["text"], lw=0.8)
        ax.set_xlabel("odległość [km]")
        ax.set_ylabel("P w gałęzi [MW]")
        ax.set_title("Przepływ mocy (ujemny = w stronę stacji)")
        ax.legend()
        ax = self.fig.add_subplot(2, 1, 2)
        Ps = np.linspace(0, max(3 * Pl, Pdg * 1.2, 1), 50)
        Vm, Ls_ = [], []
        for pp in Ps:
            v, _, ll = feeder_loadflow(n, L, r, x, Pl, pf, Vn, node, pp, Qdg * (pp / Pdg if Pdg > 0 else 0), vs, 25)
            Vm.append(v.max())
            Ls_.append(ll * 1000)
        Vm, Ls_ = np.array(Vm), np.array(Ls_)
        ax.plot(Ps, Ls_, color=C["peach"], label="straty [kW]")
        ax.set_xlabel("moc OZE [MW]")
        ax.set_ylabel("straty [kW]", color=C["peach"])
        ax.axvline(Pdg, color=C["mauve"], lw=1)
        ax2 = ax.twinx()
        ax2.plot(Ps, Vm, color=C["sky"], label="U max")
        ax2.axhline(1.05, color=C["yellow"], ls="--", lw=1)
        ax2.set_ylabel("U max [p.u.]", color=C["sky"])
        ax2.grid(False)
        hc = Ps[np.argmax(Vm > 1.05)] if np.any(Vm > 1.05) else None
        ax.set_title("Straty i napięcie maks. vs moc OZE (hosting capacity)")
        Popt = Ps[np.argmin(Ls_)]
        rev = np.any(P1 < -1e-3)
        return [f"Bez OZE: Umin = {V0.min():.4f} p.u.",
                f"         straty = {l0*1000:.1f} kW",
                f"Z OZE:   Umin = {V1.min():.4f}, Umax = {V1.max():.4f}",
                f"         straty = {l1*1000:.1f} kW ({(l1-l0)/max(l0,1e-9)*100:+.0f} %)",
                f"Przepływ zwrotny do stacji: {'TAK' if rev else 'nie'}",
                f"Moc do stacji: {P1[0]:.2f} MW",
                f"Napięcie ≤ 1,05: {ok_txt(V1.max() <= 1.05)}",
                "",
                f"Minimum strat przy P_OZE ≈ {Popt:.2f} MW",
                f"Hosting capacity (1,05 p.u.): " + (f"{hc:.2f} MW" if hc is not None else f"> {Ps[-1]:.1f} MW"),
                "", "ΔU ≈ (R·P + X·Q)/U"]


class Tab02(ExampleTab):
    key = "T02"
    CHOICES = [("typ", "Typ OZE", ["maszyna (synchroniczna)", "falownik (PV/wiatr PE)"],
                "maszyna (synchroniczna)"),
               ("curve", "Charakterystyka przekaźnika", ["SI", "VI", "EI"], "SI")]
    PARAMS = [("Sk", "Moc zwarciowa sieci Sk'' [MVA]", 50, 1000, 250, 10),
              ("L", "Długość linii [km]", 2, 30, 15, 0.5),
              ("d", "Położenie OZE od stacji [km]", 0, 30, 5, 0.5),
              ("Sdg", "Moc OZE [MVA]", 0, 20, 8, 0.5),
              ("xd", "X''d OZE [p.u.]", 0.1, 0.4, 0.18, 0.01),
              ("klim", "Ogranicznik prądu falownika [x In]", 1.0, 2.0, 1.2, 0.05),
              ("Ip", "Nastawa przekaźnika I> [A]", 100, 1500, 400, 10),
              ("tms", "TMS", 0.05, 1.0, 0.2, 0.01),
              ("y", "Zwarcie na linii sąsiedniej [km od szyn]", 0, 10, 0.5, 0.1)]

    def update_plot(self):
        typ = "sync" if self.p("typ").startswith("maszyna") else "inv"
        Sk, L, d, Sdg = self.p("Sk"), self.p("L"), min(self.p("d"), self.p("L")), self.p("Sdg")
        xd, klim, Ip, tms, y = self.p("xd"), self.p("klim"), self.p("Ip"), self.p("tms"), self.p("y")
        curve = self.p("curve")
        Vn, r, x = 15.0, 0.30, 0.35
        xs = np.linspace(0.05, L, 150)
        Ir0 = np.array([abs(radial_fault(xx, d, Vn, Sk, r, x, typ, Sdg, xd, klim, False)[0]) for xx in xs]) * 1e3
        res = [radial_fault(xx, d, Vn, Sk, r, x, typ, Sdg, xd, klim, True) for xx in xs]
        Ir1 = np.array([abs(a) for a, _ in res]) * 1e3
        Idg = np.array([abs(b) for _, b in res]) * 1e3
        ax = self.fig.add_subplot(2, 2, 1)
        ax.plot(xs, Ir0, label="przekaźnik - bez OZE")
        ax.plot(xs, Ir1, label="przekaźnik - z OZE")
        ax.plot(xs, Idg, ls="--", label="prąd OZE")
        ax.axhline(Ip, color=C["red"], ls=":", label="nastawa I>")
        ax.axvline(d, color=C["mauve"], lw=1)
        ax.set_yscale("log")
        ax.set_xlabel("miejsce zwarcia [km]")
        ax.set_ylabel("prąd [A]")
        ax.set_title("Oślepienie zabezpieczenia (protection blinding)")
        ax.legend(fontsize=7)
        I0e, I1e = Ir0[-1], Ir1[-1]
        t0, t1 = float(iec_time(I0e, Ip, tms, curve)), float(iec_time(I1e, Ip, tms, curve))
        ax = self.fig.add_subplot(2, 2, 2)
        Ic = np.logspace(math.log10(Ip * 1.05), math.log10(max(Ir0.max(), Ip * 20)), 200)
        ax.loglog(Ic, iec_time(Ic, Ip, tms, curve), color=C["sky"], label=f"IEC {curve}, TMS={tms:.2f}")
        for I_, t_, col, lab in [(I0e, t0, C["green"], "koniec linii, bez OZE"),
                                 (I1e, t1, C["red"], "koniec linii, z OZE")]:
            if np.isfinite(t_):
                ax.plot(I_, t_, "o", color=col, ms=8, label=f"{lab}: {t_:.2f} s")
        ax.set_xlabel("prąd [A]")
        ax.set_ylabel("czas [s]")
        ax.set_title("Charakterystyka czasowo-prądowa")
        ax.legend(fontsize=7)
        ax = self.fig.add_subplot(2, 1, 2)
        Ss = np.linspace(0.5, 20, 60)
        Isy = [sympathetic_current(y, d, Vn, Sk, r, x, "sync", s, xd, klim) * 1e3 for s in Ss]
        Iiv = [sympathetic_current(y, d, Vn, Sk, r, x, "inv", s, xd, klim) * 1e3 for s in Ss]
        ax.plot(Ss, Isy, label="OZE maszyna synchroniczna")
        ax.plot(Ss, Iiv, label="OZE falownikowe")
        ax.axhline(Ip, color=C["red"], ls=":", label="nastawa I> linii 1")
        ax.axvline(Sdg, color=C["mauve"], lw=1)
        ax.set_xlabel("moc OZE na linii 1 [MVA]")
        ax.set_ylabel("prąd wsteczny [A]")
        ax.set_title(f"Fałszywe wyłączenie: zwarcie na linii sąsiedniej ({y:.1f} km od szyn)")
        ax.legend(fontsize=7)
        Isym = sympathetic_current(y, d, Vn, Sk, r, x, typ, Sdg, xd, klim) * 1e3
        reach0 = xs[Ir0 > Ip].max() if np.any(Ir0 > Ip) else 0
        reach1 = xs[Ir1 > Ip].max() if np.any(Ir1 > Ip) else 0
        return [f"Typ OZE: {typ}, S = {Sdg:.1f} MVA w {d:.1f} km",
                f"Zwarcie na końcu linii ({L:.1f} km):",
                f"  I przekaźnika bez OZE: {I0e:7.0f} A",
                f"  I przekaźnika z OZE:   {I1e:7.0f} A ({(I1e/I0e-1)*100:+.1f} %)",
                f"  I z OZE:               {Idg[-1]:7.0f} A",
                f"  t bez OZE: {fmt_t(t0)}", f"  t z OZE:   {fmt_t(t1)}",
                f"Zasięg I> bez OZE: {reach0:.1f} km", f"Zasięg I> z OZE:   {reach1:.1f} km",
                "", f"Zwarcie na linii sąsiedniej:",
                f"  prąd wsteczny: {Isym:.0f} A vs I> {Ip:.0f} A",
                f"  -> {'FAŁSZYWE WYŁĄCZENIE (potrzebny 67)' if Isym > Ip else 'brak zadziałania'}"]


class Tab03(ExampleTab):
    key = "T03"
    PARAMS = [("Is", "Prąd zwarcia przez reklozer I_R [A]", 200, 4000, 1200, 10),
              ("Idg", "Prąd dopływu z OZE [A]", 0, 3000, 600, 10),
              ("In", "Prąd znamionowy bezpiecznika [A]", 10, 200, 65, 1),
              ("Ipr", "Nastawa reklozera [A]", 50, 800, 200, 10),
              ("tf", "TMS krzywej szybkiej (VI)", 0.005, 0.2, 0.02, 0.005),
              ("ts", "TMS krzywej wolnej (EI)", 0.1, 2.0, 0.6, 0.05)]

    def update_plot(self):
        Is, Idg, In, Ipr, tf, ts = (self.p(k) for k in ("Is", "Idg", "In", "Ipr", "tf", "ts"))
        Ifu = Is + Idg
        I = np.logspace(math.log10(Ipr * 1.05), math.log10(8000), 300)
        ax = self.fig.add_subplot(1, 2, 1)
        ax.loglog(I, fuse_melt(I, In), color=C["yellow"], label=f"bezpiecznik {In:.0f} A - topienie")
        ax.loglog(I, fuse_clear(I, In), color=C["peach"], ls="--", label="bezpiecznik - wyłączenie")
        ax.loglog(I, iec_time(I, Ipr, tf, "VI"), color=C["sky"], label="reklozer - szybka")
        ax.loglog(I, iec_time(I, Ipr, ts, "EI"), color=C["mauve"], label="reklozer - wolna")
        t_f = float(iec_time(Is, Ipr, tf, "VI"))
        t_s = float(iec_time(Is, Ipr, ts, "EI"))
        t_m = float(fuse_melt(Ifu, In))
        t_c = float(fuse_clear(Ifu, In))
        ax.plot(Is, t_f, "o", color=C["sky"], ms=8, label=f"reklozer szybka @ {Is:.0f} A")
        ax.plot(Ifu, t_m, "s", color=C["yellow"], ms=8, label=f"bezpiecznik @ {Ifu:.0f} A")
        ax.axvline(Is, color=C["sky"], lw=0.8, ls=":")
        ax.axvline(Ifu, color=C["yellow"], lw=0.8, ls=":")
        ax.set_ylim(0.005, 200)
        ax.set_xlabel("prąd [A]")
        ax.set_ylabel("czas [s]")
        ax.set_title("Koordynacja reklozer - bezpiecznik (fuse saving)")
        ax.legend(fontsize=7, loc="upper right")
        ax = self.fig.add_subplot(1, 2, 2)
        Id = np.linspace(0, 3000, 200)
        margin = fuse_melt(Is + Id, In) / iec_time(Is, Ipr, tf, "VI")
        ax.plot(Id, margin, color=C["green"])
        ax.axhline(1 / 0.75, color=C["red"], ls="--", label="granica 1/0,75 = 1,33")
        ax.axvline(Idg, color=C["mauve"], lw=1, label="bieżący I_OZE")
        ax.set_yscale("log")
        ax.set_xlabel("prąd dopływu z OZE [A]")
        ax.set_ylabel("t_topienia / t_reklozera_szybka")
        ax.set_title("Zapas koordynacji vs moc OZE")
        ax.legend(fontsize=8)
        lim = Id[np.argmax(margin < 1 / 0.75)] if np.any(margin < 1 / 0.75) else None
        ok1 = t_f < 0.75 * t_m
        ok2 = t_s > t_c
        return [f"I_R (reklozer) = {Is:.0f} A", f"I_B (bezpiecznik) = I_R + I_OZE = {Ifu:.0f} A", "",
                f"t reklozer szybka: {fmt_t(t_f)}", f"t reklozer wolna:  {fmt_t(t_s)}",
                f"t topienia bezp.:  {fmt_t(t_m)}", f"t wyłączenia bezp: {fmt_t(t_c)}", "",
                f"Fuse saving (szybka < 0,75·topienie):", f"  -> {ok_txt(ok1, 'ZACHOWANA', 'UTRACONA')}",
                f"Selektywność (wolna > wyłącz. bezp.):", f"  -> {ok_txt(ok2, 'ZACHOWANA', 'UTRACONA')}", "",
                "Maks. prąd OZE z zachowaniem",
                "  fuse saving: " + (f"{lim:.0f} A" if lim is not None else "> 3000 A")]


class Tab04(ExampleTab):
    key = "T04"
    CHOICES = [("opt", "Sposób uziemienia (wskazy)", [o[1] for o in GROUND_OPTS], GROUND_OPTS[4][1])]
    PARAMS = [("Sk", "Moc zwarciowa [MVA]", 50, 1000, 250, 10),
              ("C0", "Pojemność doziemna sieci C0 [µF/faza]", 0.5, 40, 10, 0.5),
              ("RN", "Rezystor uziemiający R_N [Ω] (0 = bezpośrednie)", 0, 100, 0, 0.5),
              ("det", "Rozstrojenie cewki Petersena [%]", -30, 30, 5, 1),
              ("Xzz", "Reaktancja zerowa zig-zag [Ω]", 0.5, 60, 3, 0.5),
              ("isl", "Część sieci w wyspie [%]", 1, 100, 10, 1),
              ("Rf", "Rezystancja przejścia R_f [Ω]", 0, 500, 0, 1)]

    def update_plot(self):
        Sk, C0, RN, det, Xzz, isl, Rf = (self.p(k) for k in ("Sk", "C0", "RN", "det", "Xzz", "isl", "Rf"))
        sel = [o[0] for o in GROUND_OPTS if o[1] == self.p("opt")][0]
        args = (15.0, Sk, C0, RN, det / 100, Xzz, isl / 100)
        If, Va, Vb, Vc = slg_fault(sel, *args, Rf)
        ax = self.fig.add_subplot(1, 2, 1)
        ax.set_aspect("equal")
        for k, (v0, v1, col, nm) in enumerate([(1, Va, C["red"], "a"), (A_OP ** 2, Vb, C["green"], "b"),
                                              (A_OP, Vc, C["sky"], "c")]):
            ax.annotate("", xy=(v0.real, v0.imag), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="->", color=C["gray"], lw=1))
            ax.annotate("", xy=(v1.real, v1.imag), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=2.2))
            ax.text(v1.real * 1.08, v1.imag * 1.08, f"U{nm}={abs(v1)*100:.0f}%", color=col, fontsize=9)
        V0 = (Va + Vb + Vc) / 3
        ax.plot(V0.real, V0.imag, "o", color=C["yellow"], ms=7)
        ax.text(V0.real, V0.imag - 0.15, "U0 (przesunięcie\npunktu neutralnego)", color=C["yellow"],
                fontsize=7, ha="center", va="top")
        ax.add_patch(Circle((0, 0), SQ3, fill=False, ec=C["red"], ls=":", lw=0.8))
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_title(f"Wskazy napięć fazowych (względem ziemi)\n{self.p('opt')}")
        ax.set_xlabel("Re [p.u.]")
        ax.set_ylabel("Im [p.u.]")
        ax = self.fig.add_subplot(2, 2, 2)
        names, Ifs, Vmx = [], [], []
        for code, nm in GROUND_OPTS:
            i_, a_, b_, c_ = slg_fault(code, *args, Rf)
            names.append(nm.replace(" (", "\n(").replace("Wyspa + ", "Wyspa +\n").replace("Uziemienie bezpośrednie / przez R_N", "Bezpośr./R_N"))
            Ifs.append(i_)
            Vmx.append(max(abs(b_), abs(c_)) * 100)
        xx = np.arange(len(names))
        ax.bar(xx - 0.2, Ifs, width=0.4, color=C["peach"], label="prąd zwarcia [A]")
        ax.set_yscale("log")
        ax.set_ylabel("I zwarcia [A]", color=C["peach"])
        ax2 = ax.twinx()
        ax2.bar(xx + 0.2, Vmx, width=0.4, color=C["sky"], label="U faz zdrowych [%]")
        ax2.axhline(173, color=C["red"], ls=":", lw=1)
        ax2.set_ylabel("max U faz zdrowych [%]", color=C["sky"])
        ax2.set_ylim(0, 200)
        ax2.grid(False)
        ax.set_xticks(xx)
        ax.set_xticklabels(names, fontsize=6.5)
        ax.set_title("Porównanie sposobów uziemienia")
        ax = self.fig.add_subplot(2, 2, 4)
        Rs = np.linspace(0, 500, 80)
        for code, nm in GROUND_OPTS:
            vv = [max(abs(slg_fault(code, *args, rr)[2]), abs(slg_fault(code, *args, rr)[3])) * 100 for rr in Rs]
            ax.plot(Rs, vv, label=nm)
        ax.axvline(Rf, color=C["mauve"], lw=1)
        ax.set_xlabel("R_f [Ω]")
        ax.set_ylabel("max U faz zdrowych [%]")
        ax.set_title("Wpływ rezystancji przejścia")
        ax.legend(fontsize=6)
        Xc0 = 1 / (W0 * C0 * 1e-6)
        Ic = 3 * 15e3 / SQ3 / Xc0
        return [f"Wybrano: {self.p('opt')}",
                f"Prąd zwarcia doziemnego: {If:.1f} A",
                f"|Ua| = {abs(Va)*100:.1f} %", f"|Ub| = {abs(Vb)*100:.1f} %", f"|Uc| = {abs(Vc)*100:.1f} %",
                f"|U0| = {abs(V0)*100:.1f} % (przesunięcie N)", "",
                f"Prąd pojemnościowy całej sieci:", f"  3·E/Xc0 = {Ic:.1f} A",
                f"Cewka Petersena (3X_L): {Xc0/(1+det/100):.0f} Ω", "",
                "Napięcie > 140 % długotrwale:", f"  -> {ok_txt(max(abs(Vb),abs(Vc)) < 1.4, 'nie', 'ZAGROŻENIE izolacji/odbiorów')}"]


class Tab05(ExampleTab):
    key = "T05"
    PARAMS = [("dp", "Niedopasowanie ΔP = P_obc - P_OZE [%]", -40, 40, 5, 0.5),
              ("dq", "Niedopasowanie ΔQ [% P]", -15, 15, 1, 0.1),
              ("Qf", "Dobroć obciążenia Qf", 0.2, 5, 2.5, 0.1),
              ("Vmin", "U< [p.u.]", 0.80, 0.95, 0.88, 0.01),
              ("Vmax", "U> [p.u.]", 1.05, 1.20, 1.10, 0.01),
              ("fmin", "f< [Hz]", 47.0, 49.9, 49.5, 0.1),
              ("fmax", "f> [Hz]", 50.1, 52.0, 50.5, 0.1)]

    def update_plot(self):
        dp, dq, Qf = self.p("dp") / 100, self.p("dq") / 100, self.p("Qf")
        Vmin, Vmax, fmin, fmax = self.p("Vmin"), self.p("Vmax"), self.p("fmin"), self.p("fmax")
        DP, DQ = np.meshgrid(np.linspace(-0.4, 0.4, 241), np.linspace(-0.15, 0.15, 241))
        V, f = island_rlc(DP, DQ, Qf)
        ndz = (V >= Vmin) & (V <= Vmax) & (f >= fmin) & (f <= fmax)
        ax = self.fig.add_subplot(1, 2, 1)
        ax.contourf(DP * 100, DQ * 100, ndz.astype(float), levels=[0.5, 1.5], colors=[C["green"]], alpha=0.45)
        cs = ax.contour(DP * 100, DQ * 100, f, levels=[fmin, fmax], colors=[C["sky"], C["red"]], linewidths=1)
        ax.clabel(cs, fmt="%.1f Hz", fontsize=7)
        cs2 = ax.contour(DP * 100, DQ * 100, V, levels=[Vmin, Vmax], colors=[C["yellow"], C["peach"]], linewidths=1)
        ax.clabel(cs2, fmt="%.2f p.u.", fontsize=7)
        v1, f1 = island_rlc(dp, dq, Qf)
        inside = Vmin <= v1 <= Vmax and fmin <= f1 <= fmax
        ax.plot(dp * 100, dq * 100, "o", ms=10, color=C["red"] if inside else C["sky"],
                mec="white")
        ax.set_xlabel("ΔP / P_OZE [%]")
        ax.set_ylabel("ΔQ / P_OZE [%]")
        ax.set_title(f"Strefa niewykrywalności NDZ (zielona), Qf = {Qf:.1f}")
        ax = self.fig.add_subplot(2, 2, 2)
        dps = np.linspace(-0.4, 0.4, 200)
        Vv, ff = island_rlc(dps, dq, Qf)
        ax.plot(dps * 100, Vv, color=C["yellow"])
        ax.axhline(Vmin, color=C["red"], ls=":")
        ax.axhline(Vmax, color=C["red"], ls=":")
        ax.axvline(dp * 100, color=C["mauve"])
        ax.set_ylabel("U po wyspieniu [p.u.]")
        ax.set_title("Napięcie wyspy vs ΔP")
        ax = self.fig.add_subplot(2, 2, 4)
        dqs = np.linspace(-0.15, 0.15, 200)
        _, ff = island_rlc(dp, dqs, Qf)
        ax.plot(dqs * 100, ff, color=C["sky"])
        ax.axhline(fmin, color=C["red"], ls=":")
        ax.axhline(fmax, color=C["red"], ls=":")
        ax.axvline(dq * 100, color=C["mauve"])
        ax.set_xlabel("ΔQ / P [%]")
        ax.set_ylabel("f po wyspieniu [Hz]")
        ax.set_title("Częstotliwość wyspy vs ΔQ")
        dq_lim = (Qf * (fmin / F0 - F0 / fmin) * 100, Qf * (fmax / F0 - F0 / fmax) * 100)
        return [f"Po wyspieniu:", f"  U' = {v1:.4f} p.u.", f"  f' = {f1:.3f} Hz", "",
                f"Wynik: {'WYSPA NIEWYKRYTA (w NDZ)!' if inside else 'wyspa wykryta - odłączenie'}", "",
                "Granice NDZ (wzory analityczne):",
                f"  ΔP/P: {((1/Vmax**2)-1)*100:+.1f} % ... {((1/Vmin**2)-1)*100:+.1f} %",
                f"  ΔQ/P: {min(dq_lim):+.2f} % ... {max(dq_lim):+.2f} %",
                "", "(ΔP/P = (Umax)^-2 - 1 ... (Umin)^-2 - 1,",
                " ΔQ/P = Qf·(f/f0 - f0/f), dla ΔP = 0)",
                f"Pole NDZ na mapie: {ndz.mean()*100:.1f} %"]


class Tab06(ExampleTab):
    key = "T06"
    PARAMS = [("def", "Deficyt mocy w wyspie [%]", 1, 60, 20, 1),
              ("H", "Stała bezwładności H [s]", 0.3, 8, 3, 0.1),
              ("D", "Samoregulacja odbiorów D [p.u.]", 0, 4, 1.5, 0.1),
              ("R", "Statyzm regulatora R [%]", 2, 10, 5, 0.5),
              ("Tg", "Stała regulatora Tg [s]", 0.2, 10, 2, 0.1),
              ("res", "Rezerwa mocy [%]", 0, 40, 5, 1),
              ("stage", "Wielkość stopnia SCO [%]", 2, 25, 10, 1),
              ("roc", "Próg przekaźnika ROCOF [Hz/s]", 0.1, 3, 0.5, 0.05)]

    def update_plot(self):
        dfc, H, D, R = self.p("def") / 100, self.p("H"), self.p("D"), self.p("R") / 100
        Tg, res, stage, roc = self.p("Tg"), self.p("res") / 100, self.p("stage") / 100, self.p("roc")
        runs = {}
        for mode in ("brak", "stat", "adapt"):
            runs[mode] = island_frequency(dfc, H, D, R, Tg, res, mode, stage, roc)
        names = {"brak": "bez odciążania", "stat": "SCO statyczne", "adapt": "SCO adaptacyjne (ROCOF)"}
        ax = self.fig.add_subplot(2, 1, 1)
        for mode, (t, f, rc, ld, td) in runs.items():
            ax.plot(t, f, label=names[mode])
        for fs in [49.0, 48.8, 48.6, 48.4, 48.2]:
            ax.axhline(fs, color=C["grid"], lw=0.7, ls=":")
        ax.axhline(47.5, color=C["red"], ls="--", label="47,5 Hz - wyłączenie generacji")
        ax.axvline(0.5, color=C["gray"], lw=1)
        ax.set_ylabel("f [Hz]")
        ax.set_title("Częstotliwość wyspy po utracie sieci (t = 0,5 s)")
        ax.set_ylim(max(44, min(r[1].min() for r in runs.values()) - 0.3), 50.3)
        ax.legend(fontsize=7, loc="lower right")
        ax = self.fig.add_subplot(2, 2, 3)
        t, f, rc, ld, td = runs["brak"]
        ax.plot(t, rc, color=C["peach"])
        ax.axhline(-roc, color=C["red"], ls="--", label="próg ROCOF")
        ax.axhline(roc, color=C["red"], ls="--")
        ax.set_xlabel("t [s]")
        ax.set_ylabel("df/dt [Hz/s]")
        ax.set_title("ROCOF (okno 100 ms)")
        ax.legend(fontsize=7)
        ax = self.fig.add_subplot(2, 2, 4)
        for mode in ("stat", "adapt"):
            ax.plot(runs[mode][0], runs[mode][3] * 100, label=names[mode])
        ax.set_xlabel("t [s]")
        ax.set_ylabel("obciążenie [%]")
        ax.set_title("Pozostałe obciążenie")
        ax.legend(fontsize=7)
        roc0 = F0 * dfc / (2 * H)
        lines = [f"ROCOF teoretyczny (t=0+):", f"  f0·ΔP/(2H) = {roc0:.2f} Hz/s",
                 f"Wykrycie wyspy ROCOF po: " + (f"{td*1000:.0f} ms" if td is not None else "BRAK"), ""]
        for mode, (t, f, rc, ld, _) in runs.items():
            lines += [f"{names[mode]}:", f"  f_min = {f.min():.2f} Hz, f_końc = {f[-1]:.2f} Hz",
                      f"  odciążono {100 - ld[-1]*100:.0f} %  -> {ok_txt(f.min() > 47.5, 'wyspa przetrwała', 'BLACKOUT wyspy')}"]
        return lines


class Tab07(ExampleTab):
    key = "T07"
    PARAMS = [("Qf", "Dobroć obciążenia Qf", 0.3, 5, 2.5, 0.1),
              ("fr", "Częstotl. rezonansowa obciążenia f_r [Hz]", 49.0, 51.0, 50.0, 0.05),
              ("cf", "AFD: chopping fraction cf0 [%]", 0, 10, 3, 0.25),
              ("K", "AFDPF: wzmocnienie K [1/Hz]", 0, 0.5, 0.1, 0.01),
              ("thm", "SMS: θm [°]", 0, 30, 10, 0.5),
              ("fm", "SMS: fm - f0 [Hz]", 1, 6, 3, 0.25)]

    def update_plot(self):
        Qf, fr, cf, K = self.p("Qf"), self.p("fr"), self.p("cf") / 100, self.p("K")
        thm, fm = math.radians(self.p("thm")), self.p("fm")
        methods = ["pasywna", "AFD", "AFDPF", "SMS"]
        ax = self.fig.add_subplot(2, 1, 1)
        lines = []
        for m in methods:
            t, f, tt = active_island_sim(m, Qf, fr, cf, K, thm, fm)
            ax.plot(t * 1000, f, label=m + (f" - wykryto po {tt*1000:.0f} ms" if tt else " - NIE wykryto"))
            lines.append(f"{m:8s}: " + (f"wykryto po {tt*1000:.0f} ms" if tt else "NIE WYKRYTO (NDZ)"))
        ax.axhline(50.5, color=C["red"], ls="--", lw=1)
        ax.axhline(49.5, color=C["red"], ls="--", lw=1)
        ax.set_xlabel("czas od wyspienia [ms]")
        ax.set_ylabel("f [Hz]")
        ax.set_ylim(48.5, 51.5)
        ax.set_title("Częstotliwość wyspy cykl po cyklu (progi 49,5 / 50,5 Hz)")
        ax.legend(fontsize=7, loc="upper left")
        ax = self.fig.add_subplot(2, 2, 3)
        fs = np.linspace(47, 53, 300)
        load = np.degrees(np.arctan(Qf * (fs / fr - fr / fs)))
        ax.plot(fs, load, color=C["text"], lw=2, label=f"obciążenie RLC (Qf={Qf:.1f})")
        for m, col in zip(methods[1:], [C["green"], C["peach"], C["mauve"]]):
            ax.plot(fs, [math.degrees(inverter_phase(m, ff, cf, K, thm, fm)) for ff in fs], color=col, label=m)
        ax.set_xlabel("f [Hz]")
        ax.set_ylabel("kąt fazowy [°]")
        ax.set_ylim(-35, 35)
        ax.set_title("Krzywe fazowe: przecięcie = punkt pracy wyspy")
        ax.legend(fontsize=6)
        ax = self.fig.add_subplot(2, 2, 4)
        Qs = np.linspace(0.3, 5, 40)
        for m in methods[1:]:
            tts = []
            for q in Qs:
                tt = active_island_sim(m, q, fr, cf, K, thm, fm, n_cyc=120)[2]
                tts.append(tt if tt else np.nan)
            ax.plot(Qs, tts, "o-", ms=3, label=m)
        ax.axhline(2.0, color=C["red"], ls="--", label="2 s (IEEE 1547)")
        ax.axvline(Qf, color=C["gray"], lw=1)
        ax.set_xlabel("Qf")
        ax.set_ylabel("czas wykrycia [s]")
        ax.set_title("Czas wykrycia vs Qf (brak punktu = NDZ)")
        ax.legend(fontsize=6)
        sl_load = 2 * Qf / F0
        sl_sms = thm * PI / (2 * fm)
        return lines + ["", "Warunek niestabilności SMS:",
                        f"  nachylenie SMS  = {sl_sms*1000:.1f} mrad/Hz",
                        f"  nachylenie obc. = {sl_load*1000:.1f} mrad/Hz",
                        f"  -> {'SMS wykrywa' if sl_sms > sl_load else 'SMS w NDZ'}", "",
                        f"Kąt AFD: {math.degrees(PI*cf/2):.2f}°",
                        f"THD od AFD ≈ {cf*100:.1f}·(...) % - rośnie z cf"]


class Tab08(ExampleTab):
    key = "T08"
    CHECKS = [("adapt", "Zabezpieczenie adaptacyjne (grupa B w wyspie)", True)]
    PARAMS = [("ST", "Transformator S_T [kVA]", 160, 1600, 630, 10),
              ("uk", "u_k transformatora [%]", 4, 8, 6, 0.5),
              ("Ssg", "Generator synchroniczny (CHP) [kVA]", 0, 500, 100, 10),
              ("Sinv", "Źródła falownikowe (PV) [kVA]", 0, 800, 300, 10),
              ("kinv", "Przeciążalność falowników [x In]", 1.0, 2.0, 1.2, 0.05),
              ("Sst", "Magazyn / koło zamachowe [kVA]", 0, 500, 0, 10),
              ("kst", "Przeciążalność magazynu [x In]", 1, 6, 3, 0.1),
              ("Zfcl", "Ogranicznik FCL w PCC [mΩ]", 0, 100, 0, 1),
              ("lf", "Odległość zwarcia [m]", 10, 500, 150, 10),
              ("Iload", "Prąd obciążenia linii [A]", 50, 800, 250, 10),
              ("IpA", "Nastawa I> grupy A [A]", 100, 3000, 500, 10)]

    def update_plot(self):
        g = {k: self.p(k) for k in ("ST", "uk", "Ssg", "Sinv", "kinv", "Sst", "kst", "Zfcl", "lf", "Iload", "IpA")}
        adapt = self.p("adapt")
        U = 400.0
        E = U / SQ3
        ZT = (g["uk"] / 100) * U ** 2 / (g["ST"] * 1e3) * (0.2 + 1j)
        Zl = (0.2 + 0.08j) * g["lf"] / 1000
        Zfcl = g["Zfcl"] / 1000
        Isg_fn = lambda S: 0 if S <= 0 else E / abs(1j * 0.15 * U ** 2 / (S * 1e3) + Zl)
        Igrid = E / abs(ZT + Zfcl + Zl)
        Isg = Isg_fn(g["Ssg"])
        Iinv = g["kinv"] * g["Sinv"] * 1e3 / (SQ3 * U)
        Ist = g["kst"] * g["Sst"] * 1e3 / (SQ3 * U)
        If_grid = Igrid + Isg + Iinv + Ist
        If_isl = Isg + Iinv + Ist
        IpA = g["IpA"]
        IpB = max(1.3 * g["Iload"], 0.5 * If_isl)
        Ip_isl = IpB if adapt else IpA
        tA_g = float(iec_time(If_grid, IpA, 0.1, "VI"))
        tA_i = float(iec_time(If_isl, Ip_isl, 0.1, "VI"))
        ax = self.fig.add_subplot(2, 2, 1)
        lbl = ["sieć", "SG", "falowniki", "magazyn"]
        vals_g = [Igrid, Isg, Iinv, Ist]
        vals_i = [0, Isg, Iinv, Ist]
        bottom_g = bottom_i = 0
        cols = [C["sky"], C["peach"], C["yellow"], C["mauve"]]
        for lb, vg, vi, col in zip(lbl, vals_g, vals_i, cols):
            ax.bar(0, vg, bottom=bottom_g, color=col, label=lb)
            ax.bar(1, vi, bottom=bottom_i, color=col)
            bottom_g += vg
            bottom_i += vi
        ax.axhline(g["Iload"], color=C["green"], ls="--", label="prąd obciążenia")
        ax.axhline(IpA, color=C["red"], ls=":", label="nastawa A")
        if adapt:
            ax.axhline(IpB, color=C["pink"], ls=":", label="nastawa B (wyspa)")
        ax.set_yscale("log")
        ax.set_ylim(max(10, min(g["Iload"], If_isl if If_isl > 0 else 10) / 3), If_grid * 2)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["tryb sieciowy", "tryb wyspowy"])
        ax.set_ylabel("prąd zwarcia [A]")
        ax.set_title("Wkłady do prądu zwarcia")
        ax.legend(fontsize=6, loc="upper right")
        ax = self.fig.add_subplot(2, 2, 2)
        I = np.logspace(1.5, math.log10(max(If_grid * 2, 1e3)), 300)
        ax.loglog(I, iec_time(I, IpA, 0.1, "VI"), color=C["red"], label="grupa A (sieć)")
        if adapt:
            ax.loglog(I, iec_time(I, IpB, 0.1, "VI"), color=C["pink"], label="grupa B (wyspa)")
        if np.isfinite(tA_g):
            ax.plot(If_grid, tA_g, "o", color=C["sky"], ms=8, label=f"zwarcie - sieć: {tA_g:.2f} s")
        if np.isfinite(tA_i):
            ax.plot(If_isl, tA_i, "s", color=C["yellow"], ms=8, label=f"zwarcie - wyspa: {tA_i:.2f} s")
        else:
            ax.axvline(max(If_isl, 1), color=C["yellow"], ls="--", label="zwarcie - wyspa: NIE DZIAŁA")
        ax.set_ylim(0.01, 100)
        ax.set_xlabel("prąd [A]")
        ax.set_ylabel("czas [s]")
        ax.set_title("Charakterystyki VI (TMS = 0,1)")
        ax.legend(fontsize=6)
        ax = self.fig.add_subplot(2, 1, 2)
        share = np.linspace(0, 1, 101)
        Stot = g["Ssg"] + g["Sinv"]
        Ii = [Isg_fn(Stot * (1 - s)) + g["kinv"] * Stot * s * 1e3 / (SQ3 * U) + Ist for s in share]
        ax.plot(share * 100, Ii, color=C["yellow"], label="prąd zwarcia w wyspie")
        ax.axhline(g["Iload"], color=C["green"], ls="--", label="prąd obciążenia")
        ax.axhline(IpA, color=C["red"], ls=":", label="nastawa A")
        cur = g["Sinv"] / Stot * 100 if Stot > 0 else 0
        ax.axvline(cur, color=C["mauve"], lw=1)
        ax.set_xlabel("udział źródeł falownikowych w mocy DG [%]")
        ax.set_ylabel("I zwarcia w wyspie [A]")
        ax.set_title(f"Wyspa: przy stałej mocy DG = {Stot:.0f} kVA, im więcej falowników, tym mniejszy prąd zwarcia")
        ax.legend(fontsize=7)
        return [f"Tryb sieciowy: I_zw = {If_grid:8.0f} A", f"   (sieć {Igrid:.0f} A, FCL {g['Zfcl']:.0f} mΩ)",
                f"Tryb wyspowy:  I_zw = {If_isl:8.0f} A", f"Stosunek sieć/wyspa: {If_grid/max(If_isl,1):.1f}x", "",
                f"Nastawa A = {IpA:.0f} A,  B = {IpB:.0f} A",
                f"t zadziałania - sieć:  {fmt_t(tA_g)}",
                f"t zadziałania - wyspa: {fmt_t(tA_i)}" + (" (grupa B)" if adapt else " (grupa A)"),
                "", f"Nastawa A > 1,3·I_obc: {ok_txt(IpA > 1.3*g['Iload'], 'OK', 'ryzyko zbędnych wyłączeń')}",
                f"Ochrona w wyspie: {ok_txt(np.isfinite(tA_i), 'DZIAŁA', 'NIE DZIAŁA!')}"]


class Tab09(ExampleTab):
    key = "T09"
    PARAMS = [("L", "Długość linii [km]", 2, 40, 10, 0.5),
              ("d", "Miejsce zwarcia [% linii]", 5, 150, 70, 1),
              ("m", "Miejsce dopływu OZE [% linii]", 0, 100, 30, 1),
              ("k", "Stosunek I_OZE / I_przekaźnika", 0, 3, 1.0, 0.05),
              ("Rf", "Rezystancja przejścia R_f [Ω]", 0, 10, 1, 0.1),
              ("z1", "Zasięg strefy 1 [%]", 50, 95, 80, 1),
              ("ct", "Błąd przekładników [%]", 0, 30, 10, 1),
              ("slope", "Nachylenie char. różnicowej k [%]", 10, 80, 30, 1),
              ("Imin", "Próg I_d min [p.u.]", 0.05, 1, 0.2, 0.05),
              ("Iext", "Prąd zwarcia zewnętrznego [p.u.]", 1, 20, 10, 0.5),
              ("Iisl", "Prąd zwarcia wewn. w wyspie [p.u.]", 0.1, 3, 1.2, 0.05)]

    def update_plot(self):
        L, d, m, k = self.p("L"), self.p("d") / 100, self.p("m") / 100, self.p("k")
        Rf, z1 = self.p("Rf"), self.p("z1") / 100
        z = (0.30 + 0.35j) * L
        ax = self.fig.add_subplot(1, 2, 1)
        ax.set_aspect("equal")

        def mho(reach, col, lab):
            c = z * reach / 2
            th = np.linspace(0, 2 * PI, 200)
            ax.plot(c.real + abs(c) * np.cos(th), c.imag + abs(c) * np.sin(th), color=col, label=lab)

        mho(z1, C["green"], f"strefa 1 ({z1*100:.0f} %)")
        mho(1.2, C["yellow"], "strefa 2 (120 %)")
        ax.plot([0, z.real], [0, z.imag], color=C["text"], lw=2, label="impedancja linii")

        def zapp(dd, kk):
            extra = kk * z * max(dd - m, 0.0)
            return z * dd + extra + Rf * (1 + (kk if dd > m else 0.0))

        ds = np.linspace(0.02, 1.5, 120)
        Z0 = np.array([zapp(dd, 0.0) for dd in ds])
        Z1 = np.array([zapp(dd, k) for dd in ds])
        ax.plot(Z0.real, Z0.imag, "--", color=C["sky"], label="Z pozorna - bez OZE")
        ax.plot(Z1.real, Z1.imag, "--", color=C["red"], label="Z pozorna - z OZE")
        za0, za1 = zapp(d, 0), zapp(d, k)
        ax.plot(za0.real, za0.imag, "o", color=C["sky"], ms=8)
        ax.plot(za1.real, za1.imag, "s", color=C["red"], ms=8)
        lim = max(abs(z) * 1.3, abs(za1) * 1.1)
        ax.set_xlim(-0.4 * lim, lim)
        ax.set_ylim(-0.3 * lim, lim)
        ax.set_xlabel("R [Ω]")
        ax.set_ylabel("X [Ω]")
        ax.set_title("Zabezpieczenie odległościowe - płaszczyzna R-X")
        ax.legend(fontsize=6, loc="upper left")

        def in_mho(Z, reach):
            c = z * reach / 2
            return abs(Z - c) <= abs(c)

        zone = lambda Z: "strefa 1 (natychm.)" if in_mho(Z, z1) else ("strefa 2 (~0,3 s)" if in_mho(Z, 1.2) else "poza strefami")
        ct, slope, Imin = self.p("ct") / 100, self.p("slope") / 100, self.p("Imin")
        Iext, Iisl = self.p("Iext"), self.p("Iisl")
        ax = self.fig.add_subplot(1, 2, 2)
        Ib = np.linspace(0, 20, 300)
        Ith = np.maximum(Imin, slope * Ib)
        ax.fill_between(Ib, Ith, 25, color=C["red"], alpha=0.15, label="obszar zadziałania")
        ax.plot(Ib, Ith, color=C["red"])
        pts = [("zwarcie zewn. (błąd CT)", Iext, ct * Iext, C["sky"]),
               ("zwarcie wewn. - sieć", 4.0, 8.0, C["green"]),
               ("zwarcie wewn. - wyspa", Iisl / 2, Iisl, C["yellow"]),
               ("stan normalny", 1.0, 0.02, C["gray"])]
        res = []
        for nm, ib, idd, col in pts:
            trip = idd > max(Imin, slope * ib)
            ax.plot(ib, idd, "o", color=col, ms=8, label=f"{nm}: {'ZADZIAŁA' if trip else 'stabilne'}")
            res.append(f"  {nm}: {'ZADZIAŁA' if trip else 'nie działa'}")
        ax.set_xlim(0, 20)
        ax.set_ylim(0, 12)
        ax.set_xlabel("prąd stabilizujący I_s = (|I1|+|I2|)/2 [p.u.]")
        ax.set_ylabel("prąd różnicowy I_d = |I1+I2| [p.u.]")
        ax.set_title("Zabezpieczenie różnicowe stabilizowane")
        ax.legend(fontsize=6, loc="upper left")
        return [f"Z linii = {abs(z):.2f} Ω ∠{math.degrees(np.angle(z)):.0f}°",
                f"Zwarcie w {d*100:.0f} % linii:",
                f"  bez OZE: Z = {za0.real:.2f}{za0.imag:+.2f}j Ω", f"   -> {zone(za0)}",
                f"  z OZE:   Z = {za1.real:.2f}{za1.imag:+.2f}j Ω", f"   -> {zone(za1)}",
                f"Pozorna odległość z OZE: {abs(za1)/abs(z)*100:.0f} % linii", "",
                "Różnicowe:"] + res + [
                "", "Wniosek: różnicowe nie zależy od",
                "kierunku mocy, ale wymaga łącza."]


class Tab10(ExampleTab):
    key = "T10"
    PARAMS = [("dP", "Niedopasowanie mocy w wyspie [%]", -40, 40, 3, 0.5),
              ("H", "Stała bezwładności OZE H [s]", 0.2, 6, 2.0, 0.1),
              ("D", "Samoregulacja D [p.u.]", 0.5, 4, 1.5, 0.1),
              ("td", "Przerwa beznapięciowa SPZ [s]", 0.1, 3, 0.5, 0.05),
              ("tai", "Czas odłączenia antywyspowego OZE [s]", 0.05, 3, 0.8, 0.05),
              ("X", "X''d + X_s [p.u.]", 0.1, 0.8, 0.3, 0.01),
              ("dlim", "Granica synchronizmu δ [°]", 5, 60, 20, 1)]

    def update_plot(self):
        dP, H, D = self.p("dP") / 100, self.p("H"), self.p("D")
        td, tai, X, dlim = self.p("td"), self.p("tai"), self.p("X"), self.p("dlim")

        def delta_deg(t, dp):
            tau = 2 * H / D
            integ = -(dp / D) * (t - tau * (1 - np.exp(-t / tau)))
            return 360.0 * F0 * integ

        t = np.linspace(0, 3, 1500)
        dl = (delta_deg(t, dP) + 180.0) % 360.0 - 180.0     # kąt zawinięty do ±180°
        dl[np.abs(np.diff(dl, prepend=dl[0])) > 180] = np.nan
        ax = self.fig.add_subplot(2, 2, 1)
        ax.plot(t, dl, color=C["sky"])
        ax.set_ylim(-185, 185)
        ax.axhspan(-dlim, dlim, color=C["green"], alpha=0.15, label=f"|δ| < {dlim:.0f}° (synchro-check)")
        ax.axvline(td, color=C["red"], ls="--", label=f"SPZ po {td:.2f} s")
        ax.set_xlabel("t od otwarcia reklozera [s]")
        ax.set_ylabel("kąt wyspa-sieć δ [°]")
        ax.set_title("Odpływ fazy wyspy")
        ax.legend(fontsize=7)
        ax = self.fig.add_subplot(2, 2, 2)
        tds = np.linspace(0.05, 3, 200)
        for dp in [0.05, 0.1, 0.2, 0.4]:
            dd = np.radians(delta_deg(tds, dp))
            ax.plot(tds, 2 * np.abs(np.sin(dd / 2)) / X, label=f"ΔP = {dp*100:.0f} %")
        ax.axhline(1 / X, color=C["red"], ls=":", label="prąd zwarcia 3f")
        ax.axvline(td, color=C["gray"], lw=1)
        ax.set_xlabel("przerwa beznapięciowa [s]")
        ax.set_ylabel("prąd przy załączeniu [p.u.]")
        ax.set_title("Prąd wyrównawczy SPZ bez synchronizacji")
        ax.legend(fontsize=6)
        ax = self.fig.add_subplot(2, 1, 2)
        ev = [(0, "zwarcie"), (0.08, "reklozer otwiera"), (0.08 + tai, "OZE odłączone (antywysp.)"),
              (0.08 + td, "SPZ - załączenie")]
        ax.barh(0, 0.08, left=0, color=C["red"], label="zwarcie zasilane z sieci")
        ax.barh(0, min(tai, td), left=0.08, color=C["peach"], label="wyspa z OZE podtrzymuje łuk")
        if tai < td:
            ax.barh(0, td - tai, left=0.08 + tai, color=C["green"], label="linia bez napięcia - łuk gaśnie")
        for tt, nm in ev:
            ax.axvline(tt, color=C["text"], lw=0.8)
            ax.text(tt, 0.45, nm, rotation=20, fontsize=7, color=C["text"])
        ax.set_ylim(-0.5, 1.1)
        ax.set_yticks([])
        ax.set_xlim(0, max(td, tai) + 0.4)
        ax.set_xlabel("czas [s]")
        ax.set_title("Oś czasu cyklu SPZ")
        ax.legend(fontsize=7, loc="lower right")
        d_td = (float(delta_deg(td, dP)) + 180.0) % 360.0 - 180.0
        Irec = 2 * abs(math.sin(math.radians(d_td) / 2)) / X
        ok_arc = tai < td
        return [f"Kąt w chwili SPZ: δ = {d_td:.1f}°",
                f"Δf w chwili SPZ: {-(dP/D)*(1-math.exp(-td*D/(2*H)))*F0:.2f} Hz",
                f"Prąd wyrównawczy: {Irec:.2f} p.u.", f"  ({Irec*X*100:.0f} % prądu zwarcia 3f)", "",
                f"Synchro-check: {ok_txt(abs(d_td) < dlim, 'załączenie dozwolone', 'BLOKADA / udar')}",
                f"OZE odłączone przed SPZ: {'TAK' if ok_arc else 'NIE'}",
                f"Wynik SPZ: {'udane (łuk zgasł)' if ok_arc else 'NIEUDANE - OZE podtrzymało łuk'}", "",
                "Wymóg: t_antywyspowe + zapas < t_SPZ"]


class Tab11(ExampleTab):
    key = "T11"
    PARAMS = [("Qc", "Bateria kondensatorów [p.u.]", 0, 1.0, 0.4, 0.02),
              ("V", "Napięcie sieci [p.u.]", 0.7, 1.1, 1.0, 0.01),
              ("H", "Bezwładność H [s]", 0.3, 4, 1.0, 0.1),
              ("Ilim", "Ograniczenie prądu soft-startera [p.u.]", 1.5, 5, 2.5, 0.1),
              ("V0", "Napięcie początkowe soft-startera [p.u.]", 0.2, 0.8, 0.35, 0.05),
              ("tr", "Czas narastania rampy [s]", 0.5, 10, 3, 0.5)]

    def update_plot(self):
        Qc, V, H, Ilim, V0, tr = (self.p(k) for k in ("Qc", "V", "H", "Ilim", "V0", "tr"))
        s = np.linspace(-1, 1, 801)
        _, Te, _, _ = scig_point(s, V)
        ax = self.fig.add_subplot(2, 2, 1)
        ax.plot(1 - s, Te, color=C["sky"])
        ax.axhline(0, color=C["text"], lw=0.7)
        ax.axvline(1, color=C["gray"], ls=":")
        ax.fill_betweenx([-4, 4], 1, 2, color=C["green"], alpha=0.08)
        ax.text(1.5, Te.max() * 0.8, "GENERATOR\n(s < 0)", ha="center", color=C["green"], fontsize=8)
        ax.text(0.5, Te.max() * 0.8, "SILNIK\n(0 < s < 1)", ha="center", color=C["sky"], fontsize=8)
        ax.set_xlabel("prędkość ω / ω_s")
        ax.set_ylabel("moment T_e [p.u.]")
        ax.set_title(f"Charakterystyka momentu SCIG (U = {V:.2f})")
        ax.set_ylim(Te.min() * 1.1, Te.max() * 1.2)
        ax = self.fig.add_subplot(2, 2, 2)
        sg = np.linspace(-0.03, 0.005, 200)
        _, _, P, Q = scig_point(sg, V)
        Qnet = Q - Qc * V ** 2
        ax.plot(sg * 100, P, label="P (ujemna = oddawana)")
        ax.plot(sg * 100, Q, label="Q pobierana przez SCIG")
        ax.plot(sg * 100, Qnet, label="Q z sieci po kompensacji")
        ax.axhline(0, color=C["text"], lw=0.7)
        ax.set_xlabel("poślizg s [%]")
        ax.set_ylabel("[p.u.]")
        ax.set_title("Moc czynna i bierna generatora (praca stałoobrotowa)")
        ax.legend(fontsize=7)
        ax = self.fig.add_subplot(2, 1, 2)
        t, I1, W1 = scig_start(H, V0, tr, False, Ilim, t_end=20)
        t, I2, W2 = scig_start(H, V0, tr, True, Ilim, t_end=20)
        ax.plot(t, I1, color=C["red"], label="prąd - rozruch bezpośredni")
        ax.plot(t, I2, color=C["green"], label="prąd - soft-starter")
        ax.set_xlabel("t [s]")
        ax.set_ylabel("prąd stojana [p.u.]")
        ax2 = ax.twinx()
        ax2.plot(t, W1, "--", color=C["red"], alpha=0.6, label="prędkość - bezpośredni")
        ax2.plot(t, W2, "--", color=C["green"], alpha=0.6, label="prędkość - soft-start")
        ax2.set_ylabel("ω / ω_s")
        ax2.grid(False)
        ax.set_title("Rozruch / przyłączenie: bezpośredni vs soft-starter (rys. 18.6b)")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="center right")
        i_s = np.argmin(np.abs(sg + 0.01))
        Tmax_g = -Te.min()
        tst1 = t[np.argmax(W1 > 0.95)] if np.any(W1 > 0.95) else None
        tst2 = t[np.argmax(W2 > 0.95)] if np.any(W2 > 0.95) else None
        return [f"Przy s = -1 %:", f"  P = {P[i_s]:.3f} p.u. (oddawana)", f"  Q_SCIG = {Q[i_s]:.3f} p.u. (pobierana)",
                f"  Q z sieci po kompensacji = {Qnet[i_s]:.3f}", f"  cos φ = {abs(P[i_s])/math.hypot(P[i_s], Qnet[i_s]):.3f}", "",
                f"Moment krytyczny gen.: {Tmax_g:.2f} p.u.", f"Moment rozruchowy: {float(scig_point(1.0, V)[1]):.2f} p.u.", "",
                f"Rozruch bezpośredni: Imax = {I1.max():.2f} p.u.", "  t(95 %) = " + (f"{tst1:.1f} s" if tst1 else "> 20 s"),
                f"Soft-starter: Imax = {I2.max():.2f} p.u.", "  t(95 %) = " + (f"{tst2:.1f} s" if tst2 else "> 20 s (utknął!)")]


class Tab12(ExampleTab):
    key = "T12"
    PARAMS = [("R", "Promień wirnika R [m]", 20, 80, 40, 1),
              ("Prat", "Moc znamionowa [MW]", 0.5, 6, 2.0, 0.1),
              ("nmax", "Maks. prędkość wirnika [obr/min]", 8, 30, 19, 0.5),
              ("nfix", "Prędkość turbiny stałoobrotowej [obr/min]", 8, 30, 15, 0.5),
              ("vm", "Średnia prędkość wiatru [m/s]", 4, 11, 7, 0.1),
              ("beta", "Kąt łopat β do wykresu Cp [°]", 0, 20, 0, 1)]

    def update_plot(self):
        R, Prat, nmax, nfix = self.p("R"), self.p("Prat") * 1e6, self.p("nmax"), self.p("nfix")
        vm, beta = self.p("vm"), self.p("beta")
        wmax, wfix = nmax * 2 * PI / 60, nfix * 2 * PI / 60
        ax = self.fig.add_subplot(2, 2, 1)
        nn = np.linspace(2, 30, 300)
        w = nn * 2 * PI / 60
        for v in [5, 6, 7, 8, 9, 10, 11]:
            P = turbine_power(v, R, w) / 1e6
            ax.plot(nn, P, lw=1, label=f"{v} m/s")
        vv = np.linspace(3, 25, 200)
        wopt = np.minimum(LAM_OPT * vv / R, wmax)
        Popt = np.minimum(turbine_power(vv, R, wopt), Prat) / 1e6
        ax.plot(wopt * 60 / (2 * PI), Popt, color=C["text"], lw=2.5, label="MPPT (zmienna prędkość)")
        ax.axvline(nfix, color=C["red"], ls="--", lw=1.5, label="stała prędkość")
        ax.set_ylim(0, Prat / 1e6 * 1.4)
        ax.set_xlabel("prędkość wirnika [obr/min]")
        ax.set_ylabel("P [MW]")
        ax.set_title("Moc turbiny vs prędkość wirnika")
        ax.legend(fontsize=6, ncol=2)
        ax = self.fig.add_subplot(2, 2, 2)
        lam = np.linspace(0.5, 16, 300)
        for b in sorted({0, 2, 5, 10, 15, int(beta)}):
            ax.plot(lam, cp_heier(lam, b), lw=2.2 if b == int(beta) else 1, label=f"β = {b}°")
        ax.axhline(16 / 27, color=C["red"], ls=":", label="granica Betza 0,593")
        ax.set_xlabel("λ = ωR/v")
        ax.set_ylabel("Cp")
        ax.set_ylim(0, 0.62)
        ax.set_title("Współczynnik mocy Cp(λ, β)")
        ax.legend(fontsize=6)
        ax = self.fig.add_subplot(2, 1, 2)
        v = np.linspace(0, 26, 521)
        Pv = power_curve(v, R, Prat, wmax, wfix, True)
        Pf = power_curve(v, R, Prat, wmax, wfix, False)
        c = vm / 0.8862
        pdf = (2 / c) * (v / c) * np.exp(-(v / c) ** 2)
        ax.plot(v, Pv / 1e6, color=C["green"], label="zmienna prędkość (MPPT)")
        ax.plot(v, Pf / 1e6, color=C["red"], label="stała prędkość")
        ax.set_xlabel("prędkość wiatru [m/s]")
        ax.set_ylabel("P [MW]")
        ax2 = ax.twinx()
        ax2.fill_between(v, pdf, color=C["sky"], alpha=0.2, label="rozkład Weibulla (k=2)")
        ax2.set_ylabel("gęstość prawdop.")
        ax2.grid(False)
        dv = v[1] - v[0]
        Ev = 8760 * np.sum(Pv * pdf) * dv / 1e9
        Ef = 8760 * np.sum(Pf * pdf) * dv / 1e9
        ax.set_title(f"Krzywe mocy i energia roczna: zmienna {Ev:.2f} GWh vs stała {Ef:.2f} GWh")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper right")
        v_opt_fix = wfix * R / LAM_OPT
        return [f"Cp max = {CP_MAX:.3f} przy λ = {LAM_OPT}", f"Betz: 16/27 = {16/27:.3f}", "",
                f"Stała prędkość {nfix:.1f} obr/min:", f"  Cp max tylko przy v = {v_opt_fix:.1f} m/s",
                f"Prędkość wiatru znamionowa (MPPT):", f"  ≈ {vv[np.argmax(Popt >= Prat/1e6*0.999)]:.1f} m/s", "",
                f"Energia roczna (v_śr = {vm:.1f} m/s):", f"  zmienna: {Ev*1000:.0f} MWh", f"  stała:   {Ef*1000:.0f} MWh",
                f"  zysk MPPT: {(Ev/Ef-1)*100 if Ef > 0 else 0:+.1f} %",
                f"Współczynnik wykorzystania: {Ev*1e9/(Prat*8760)*100:.1f} %"]


class Tab13(ExampleTab):
    key = "T13"
    PARAMS = [("P", "Moc znamionowa turbiny [MW]", 0.5, 6, 2.0, 0.1),
              ("smax", "Zakres poślizgu ± s_max [%]", 5, 40, 30, 1),
              ("R", "Promień wirnika [m]", 20, 80, 40, 1),
              ("G", "Przełożenie przekładni", 50, 150, 100, 1),
              ("v", "Bieżąca prędkość wiatru [m/s]", 3, 20, 9, 0.1),
              ("ratio", "Przekładnia napięć stojan/wirnik", 0.2, 3, 0.4, 0.05)]

    def update_plot(self):
        Pr_, smax, R, G = self.p("P") * 1e6, self.p("smax") / 100, self.p("R"), self.p("G")
        v0, ratio = self.p("v"), self.p("ratio")
        ns = 1500.0

        def op(v):
            w = LAM_OPT * v / R                        # MPPT [rad/s]
            n = w * 60 / (2 * PI) * G
            n = np.clip(n, ns * (1 - smax), ns * (1 + smax))
            w = n / G * 2 * PI / 60
            Pm = np.minimum(turbine_power(v, R, w), Pr_)
            s = (ns - n) / ns
            Ps = Pm / (1 - s)
            return Pm, Ps, -s * Ps, s, n

        vv = np.linspace(3, 20, 300)
        Pm, Ps, Pr, s, n = op(vv)
        ax = self.fig.add_subplot(2, 1, 1)
        ax.plot(vv, Pm / 1e6, color=C["text"], lw=2, label="P_m (mechaniczna)")
        ax.plot(vv, Ps / 1e6, color=C["sky"], label="P_s (stojan)")
        ax.plot(vv, Pr / 1e6, color=C["peach"], label="P_r (wirnik/przekształtnik)")
        ax.axhline(0, color=C["gray"], lw=0.8)
        ax.axvline(v0, color=C["mauve"], lw=1)
        ax2 = ax.twinx()
        ax2.plot(vv, s * 100, ":", color=C["green"], label="poślizg s [%]")
        ax2.set_ylabel("s [%]", color=C["green"])
        ax2.grid(False)
        ax.set_xlabel("prędkość wiatru [m/s]")
        ax.set_ylabel("moc [MW]")
        ax.set_title("DFIG: rozdział mocy stojan / wirnik (P_r > 0 = do sieci)")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left")
        ax = self.fig.add_subplot(2, 2, 3)
        nn = np.linspace(ns * 0.6, ns * 1.4, 200)
        ss = (ns - nn) / ns
        Pm0 = Pr_
        ax.plot(nn, Pm0 / (1 - ss) / 1e6, label="P_s")
        ax.plot(nn, -ss * Pm0 / (1 - ss) / 1e6, label="P_r")
        ax.axvspan(ns * (1 - smax), ns * (1 + smax), color=C["green"], alpha=0.12, label="zakres pracy")
        ax.axvline(ns, color=C["gray"], ls=":")
        ax.text(ns * 0.8, Pm0 / 1e6 * 0.3, "podsynchr.\nP_r < 0\n(sieć -> wirnik)", fontsize=7, ha="center")
        ax.text(ns * 1.2, Pm0 / 1e6 * 0.3, "nadsynchr.\nP_r > 0\n(wirnik -> sieć)", fontsize=7, ha="center")
        ax.set_xlabel("prędkość generatora [obr/min]")
        ax.set_ylabel("[MW]")
        ax.set_title("Rys. 18.8b przy P_m = P_n")
        ax.legend(fontsize=6, loc="upper right")
        ax = self.fig.add_subplot(2, 2, 4)
        sr = np.linspace(-0.4, 0.4, 100)
        ax.plot(sr * 100, np.abs(sr) / ratio, color=C["yellow"])
        ax.axvspan(-smax * 100, smax * 100, color=C["green"], alpha=0.12)
        ax.set_xlabel("poślizg [%]")
        ax.set_ylabel("U_r / U_s (strona wirnika)")
        ax.set_title("Napięcie wirnika ≈ |s|·U_s / ϑ")
        pm, ps, pr, s0, n0 = (float(a) for a in op(v0))
        return [f"Wiatr {v0:.1f} m/s:", f"  n = {n0:.0f} obr/min, s = {s0*100:+.1f} %",
                f"  tryb: {'NADsynchroniczny' if s0 < 0 else 'PODsynchroniczny'}",
                f"  P_m = {pm/1e6:.3f} MW", f"  P_s = {ps/1e6:.3f} MW", f"  P_r = {pr/1e6:+.3f} MW",
                f"  kierunek P_r: {'wirnik -> sieć' if pr > 0 else 'sieć -> wirnik'}", "",
                f"Moc przekształtnika ≈ s_max·P_s:", f"  ≈ {smax/(1-smax)*Pr_/1e6:.2f} MW ({smax/(1-smax)*100:.0f} % P_n)",
                f"U_r max = {smax/ratio:.2f}·U_s"]


class Tab14(ExampleTab):
    key = "T14"
    CHECKS = [("chop", "Chopper DC (rezystor hamujący)", True)]
    PARAMS = [("dip", "Głębokość zapadu napięcia [%]", 10, 95, 80, 1),
              ("tf", "Czas trwania zwarcia [ms]", 50, 500, 150, 10),
              ("slip", "Poślizg s [%] (ujemny = nadsynchr.)", -30, 30, -20, 1),
              ("P0", "Moc stojana przed zwarciem [p.u.]", 0.1, 1.0, 0.8, 0.05),
              ("Rcb", "Rezystancja crowbara R_cb [p.u.]", 0.02, 1.0, 0.2, 0.01),
              ("Ith", "Próg crowbara |i_r| [p.u.]", 1.2, 4, 2.0, 0.1),
              ("Tcb", "Min. czas załączenia crowbara [ms]", 20, 300, 100, 10),
              ("Vr", "Maks. napięcie RSC [p.u.]", 0.15, 0.6, 0.35, 0.01)]

    def update_plot(self):
        dip, tf, slip, P0 = self.p("dip") / 100, self.p("tf") / 1000, self.p("slip") / 100, self.p("P0")
        Rcb, Ith, Tcb, Vr, chop = self.p("Rcb"), self.p("Ith"), self.p("Tcb") / 1000, self.p("Vr"), self.p("chop")
        a = dfig_fault_sim(slip, P0, dip, tf, False, Rcb, Ith, Tcb, Vr, False)
        b = dfig_fault_sim(slip, P0, dip, tf, True, Rcb, Ith, Tcb, Vr, chop)
        t = a["t"] * 1000
        ax = self.fig.add_subplot(2, 2, 1)
        ax.plot(t, a["Vs"], color=C["yellow"])
        ax.fill_between(t, 0, b["cb"] * 1.1, color=C["red"], alpha=0.15, label="crowbar załączony")
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("|U_s| [p.u.]")
        ax.set_title("Napięcie sieci (zapad) i stan crowbara")
        ax.legend(fontsize=7)
        ax = self.fig.add_subplot(2, 2, 2)
        ax.plot(t, a["Ir"], color=C["red"], label="|i_r| bez crowbara (= prąd RSC!)")
        ax.plot(t, b["Ir"], color=C["sky"], label="|i_r| z crowbarem")
        ax.plot(t, b["Irsc"], color=C["green"], lw=1.8, label="prąd RSC z crowbarem")
        ax.axhline(Ith, color=C["yellow"], ls="--", label="próg crowbara")
        ax.set_ylabel("[p.u.]")
        ax.set_title("Prąd wirnika i przekształtnika RSC")
        ax.legend(fontsize=6)
        ax = self.fig.add_subplot(2, 2, 3)
        ax.plot(t, a["Vdc"], color=C["red"], label="U_dc bez ochrony")
        ax.plot(t, b["Vdc"], color=C["green"], label="U_dc z crowbarem" + (" + chopper" if chop else ""))
        ax.axhline(1.2, color=C["yellow"], ls="--", label="granica 1,2 p.u.")
        ax.set_ylim(0, min(max(a["Vdc"].max(), 1.3) * 1.1, 6))
        ax.set_xlabel("t [ms]")
        ax.set_ylabel("U_dc [p.u.]")
        ax.set_title("Napięcie obwodu pośredniczącego DC")
        ax.legend(fontsize=6)
        ax = self.fig.add_subplot(2, 2, 4)
        ax.plot(t, -a["Te"], color=C["red"], label="bez crowbara")
        ax.plot(t, -b["Te"], color=C["sky"], label="z crowbarem")
        ax.set_xlabel("t [ms]")
        ax.set_ylabel("moment generatora [p.u.]")
        ax.set_title("Moment elektromagnetyczny (udar na przekładnię)")
        ax.legend(fontsize=7)
        return [f"Zapad: {dip*100:.0f} % przez {tf*1000:.0f} ms, s = {slip*100:+.0f} %", "",
                "BEZ crowbara:", f"  |i_r| max = {a['Ir'].max():.2f} p.u.",
                f"  U_dc max  = {a['Vdc'].max():.2f} p.u.", f"  moment max = {np.abs(a['Te']).max():.2f} p.u.",
                f"  RSC: {ok_txt(a['Ir'].max() < 2.0, 'bezpieczny', 'ZNISZCZENIE (>2 p.u.)')}", "",
                "Z crowbarem:", f"  |i_r| max = {b['Ir'].max():.2f} p.u.",
                f"  prąd RSC max = {b['Irsc'].max():.2f} p.u.", f"  U_dc max  = {b['Vdc'].max():.2f} p.u.",
                f"  crowbar aktywny: {b['cb'].sum()*(b['t'][1]-b['t'][0])*1000:.0f} ms",
                f"  RSC: {ok_txt(b['Irsc'].max() < 2.0 and b['Vdc'].max() < 1.3, 'chroniony', 'zagrożony')}", "",
                "Szac. napięcie indukowane w wirniku:",
                f"  ≈ (1-s)·ΔU = {(1-slip)*dip:.2f} p.u. vs RSC {Vr:.2f}"]


class Tab15(ExampleTab):
    key = "T15"
    PARAMS = [("G", "Nasłonecznienie G [W/m²]", 100, 1200, 1000, 10),
              ("Tc", "Temperatura ogniw [°C]", -20, 80, 45, 1),
              ("Ns", "Moduły w stringu N_s", 5, 30, 20, 1),
              ("Np", "Stringi równolegle N_p", 1, 12, 4, 1),
              ("IR", "Dopuszcz. prąd wsteczny modułu I_R [A]", 10, 40, 20, 1),
              ("Tmin", "Temperatura minimalna [°C]", -40, 10, -25, 1),
              ("Umax", "Maks. napięcie falownika [V]", 600, 1500, 1000, 50),
              ("a", "Szerokość pętli okablowania a [m]", 0.05, 5, 1.0, 0.05),
              ("b", "Długość pętli b [m]", 2, 60, 20, 1),
              ("didt", "Stromość prądu pioruna [kA/µs]", 10, 100, 25, 1)]

    def update_plot(self):
        G, Tc, Ns, Np = self.p("G"), self.p("Tc"), int(self.p("Ns")), int(self.p("Np"))
        IR, Tmin, Umax = self.p("IR"), self.p("Tmin"), self.p("Umax")
        a, b, didt = self.p("a"), self.p("b"), self.p("didt") * 1e9
        ax = self.fig.add_subplot(2, 2, 1)
        U, I = pv_iv(G, Tc, Ns, Np)
        P = U * I
        k = np.argmax(P)
        U25, I25 = pv_iv(1000, 25, Ns, Np)
        ax.plot(U25, I25, color=C["gray"], ls=":", label="STC 1000 W/m², 25 °C")
        ax.plot(U, I, color=C["sky"], label="I-U")
        ax.plot(U[k], I[k], "o", color=C["red"], ms=8, label=f"MPP {P[k]/1e3:.1f} kW")
        ax.set_xlabel("U [V]")
        ax.set_ylabel("I [A]")
        ax2 = ax.twinx()
        ax2.plot(U, P / 1e3, color=C["yellow"], label="P-U")
        ax2.set_ylabel("P [kW]", color=C["yellow"])
        ax2.grid(False)
        ax.set_title(f"Pole PV {Ns}x{Np} modułów")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=6, loc="lower left")
        ax = self.fig.add_subplot(2, 2, 2)
        Isc = PV_MOD["Isc"] * 1.25
        nps = np.arange(1, 13)
        Irev = (nps - 1) * Isc
        cols = [C["red"] if x > IR else C["green"] for x in Irev]
        ax.bar(nps, Irev, color=cols)
        ax.axhline(IR, color=C["yellow"], ls="--", label=f"I_R modułu = {IR:.0f} A")
        ax.axvline(Np, color=C["mauve"], lw=1.5, label="bieżące N_p")
        ax.set_xlabel("liczba stringów równoległych N_p")
        ax.set_ylabel("prąd wsteczny w stringu [A]")
        ax.set_title("Zwarcie w stringu: (N_p - 1)·1,25·I_sc")
        ax.legend(fontsize=7)
        ax = self.fig.add_subplot(2, 2, 3)
        Ts = np.linspace(-40, 30, 100)
        Uoc = PV_MOD["Voc"] * (1 + PV_MOD["b_voc"] * (Ts - 25)) * Ns
        ax.plot(Ts, Uoc, color=C["sky"])
        ax.axhline(Umax, color=C["red"], ls="--", label=f"limit {Umax:.0f} V")
        ax.axvline(Tmin, color=C["mauve"], lw=1)
        ax.set_xlabel("temperatura ogniw [°C]")
        ax.set_ylabel("U_oc stringu [V]")
        ax.set_title("Napięcie jałowe stringu vs temperatura")
        ax.legend(fontsize=7)
        ax = self.fig.add_subplot(2, 2, 4)
        dd = np.logspace(0, 3, 200)
        mu0 = 4e-7 * PI
        Uind = mu0 / (2 * PI) * b * np.log((dd + a) / dd) * didt / 1e3
        ax.loglog(dd, Uind, color=C["peach"], label=f"pętla {a:.2f} x {b:.0f} m")
        Uind_s = mu0 / (2 * PI) * b * np.log((dd + 0.05) / dd) * didt / 1e3
        ax.loglog(dd, Uind_s, color=C["green"], ls="--", label="pętla 0,05 m (przewody razem)")
        ax.axhline(4, color=C["red"], ls=":", label="typ. wytrzymałość falownika ~4 kV")
        ax.set_xlabel("odległość uderzenia pioruna [m]")
        ax.set_ylabel("napięcie indukowane [kV]")
        ax.set_title("Przepięcie indukowane w pętli DC")
        ax.legend(fontsize=6)
        Uoc_min = PV_MOD["Voc"] * (1 + PV_MOD["b_voc"] * (Tmin - 25)) * Ns
        Irv = (Np - 1) * Isc
        U100 = mu0 / (2 * PI) * b * math.log((100 + a) / 100) * didt / 1e3
        return [f"MPP: {P[k]/1e3:.2f} kW, U = {U[k]:.0f} V, I = {I[k]:.1f} A",
                f"Sprawność względna do STC: {P[k]/(U25*I25).max()*100:.0f} %", "",
                f"Prąd wsteczny przy zwarciu: {Irv:.1f} A",
                f"  bezpiecznik stringu: {'WYMAGANY' if Irv > IR else 'niewymagany'}",
                f"  dobór gPV: {1.5*PV_MOD['Isc']:.0f}...{min(2.4*PV_MOD['Isc'], IR):.0f} A", "",
                f"U_oc przy {Tmin:.0f} °C: {Uoc_min:.0f} V",
                f"  vs limit {Umax:.0f} V: {ok_txt(Uoc_min <= Umax)}",
                f"  maks. N_s: {int(Umax // (PV_MOD['Voc']*(1+PV_MOD['b_voc']*(Tmin-25))))}", "",
                f"Przepięcie od pioruna 100 m obok:", f"  {U100:.2f} kV -> SPD {'konieczny' if U100 > 1 else 'zalecany'}"]


class Tab16(ExampleTab):
    key = "T16"
    PARAMS = [("Sk", "Moc zwarciowa sieci na szynach [MVA]", 100, 600, 350, 10),
              ("N", "Liczba przyłączonych OZE (wirujących)", 0, 20, 10, 1),
              ("S", "Moc jednego OZE [MVA]", 0.5, 10, 4, 0.5),
              ("xd", "X''d + trafo [p.u.]", 0.1, 0.4, 0.2, 0.01),
              ("xr", "Dławik szeregowy [% na bazie OZE]", 0, 50, 0, 1),
              ("l", "Odległość OZE od szyn [km]", 0, 20, 0, 0.5),
              ("Icb", "Zdolność wyłączalna wyłącznika [kA]", 8, 40, 16, 0.5)]

    def update_plot(self):
        Sk, N, S, xd, xr, l, Icb = (self.p(k) for k in ("Sk", "N", "S", "xd", "xr", "l", "Icb"))
        Vn = 15.0
        E = Vn / SQ3
        z = (0.3 + 0.35j) * l
        Ig = E / (Vn ** 2 / Sk)

        def Ik(n, xr_):
            Zdg = 1j * (xd + xr_ / 100) * Vn ** 2 / S + z
            return Ig + n * E / abs(Zdg)

        ns = np.arange(0, 21)
        ax = self.fig.add_subplot(1, 2, 1)
        ax.plot(ns, [Ik(n, 0) for n in ns], "o-", ms=4, color=C["red"], label="bez dławika")
        ax.plot(ns, [Ik(n, xr) for n in ns], "s-", ms=4, color=C["green"], label=f"dławik {xr:.0f} % / odległość {l:.1f} km")
        ax.axhline(Icb, color=C["yellow"], ls="--", label=f"zdolność wyłączalna {Icb:.1f} kA")
        ax.axvline(N, color=C["mauve"], lw=1)
        ax.set_xlabel("liczba źródeł OZE")
        ax.set_ylabel("prąd zwarcia na szynach [kA]")
        ax.set_title("Wzrost mocy zwarciowej (18.7)")
        ax.legend(fontsize=7)
        ax = self.fig.add_subplot(1, 2, 2)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")
        ax.set_title("Rys. 18.11 - wymiana informacji w sieci przyszłości")
        nodes = {"TSO\n(operator przesyłowy)": (5, 9, C["sky"]), "DNO / OSD\n(operator dystrybucyjny)": (5, 6, C["green"]),
                 "Trader\n(handlowiec)": (1.6, 6, C["yellow"]), "Regulator\n(prawo, taryfy)": (8.4, 6, C["gray"]),
                 "Odbiorcy /\nprosumenci": (2, 2, C["peach"]), "OZE / DG\n(PV, wiatr, CHP)": (5, 2, C["mauve"]),
                 "Magazyny\ni zabezpieczenia": (8, 2, C["red"])}
        for txt, (x, y, col) in nodes.items():
            ax.add_patch(FancyBboxPatch((x - 1.3, y - 0.6), 2.6, 1.2, boxstyle="round,pad=0.05",
                                        fc=C["panel"], ec=col, lw=1.5))
            ax.text(x, y, txt, ha="center", va="center", fontsize=7.5, color=col)
        links = [((5, 8.4), (5, 6.6)), ((3.7, 6), (2.9, 6)), ((6.3, 6), (7.1, 6)),
                 ((4.5, 5.4), (2.4, 2.6)), ((5, 5.4), (5, 2.6)), ((5.5, 5.4), (7.6, 2.6)), ((1.8, 5.4), (2, 2.6))]
        for (x1, y1), (x2, y2) in links:
            ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                        arrowprops=dict(arrowstyle="<->", color=C["sub"], lw=1.2))
        ax.text(5, 0.6, "PLC / łącza cyfrowe: nastawy adaptacyjne, transfer trip, sterowanie U i Q",
                ha="center", fontsize=7.5, color=C["sub"])
        I0, I1 = Ik(N, 0), Ik(N, xr)
        return [f"Wkład sieci: {Ig:.2f} kA", f"Wkład 1 OZE (bez dławika): {E/abs(1j*xd*Vn**2/S + z):.3f} kA", "",
                f"N = {int(N)} OZE:", f"  bez dławika: {I0:.2f} kA -> {ok_txt(I0 <= Icb)}",
                f"  z dławikiem: {I1:.2f} kA -> {ok_txt(I1 <= Icb)}", "",
                f"Maks. liczba OZE bez dławika:", f"  {int(np.sum([Ik(n,0) <= Icb for n in range(0,200)]))-1 if Ig <= Icb else 0}",
                f"Maks. liczba OZE z dławikiem:", f"  {int(np.sum([Ik(n,xr) <= Icb for n in range(0,200)]))-1 if Ig <= Icb else 0}"]


class Tab17(ExampleTab):
    key = "T17"
    PARAMS = []

    def update_plot(self):
        ax = self.fig.add_subplot(1, 1, 1)
        ax.axis("off")
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_title("Klasyfikacja metod wykrywania wyspy (18.3) - odpowiedź na pytanie 4")
        tree = [("Wykrywanie wyspy", 5, 9.2, C["text"]),
                ("Zdalne / komunikacyjne", 1.8, 7.0, C["sky"]), ("Lokalne", 6.8, 7.0, C["green"]),
                ("Pasywne", 4.0, 4.6, C["yellow"]), ("Aktywne", 6.8, 4.6, C["peach"]), ("Hybrydowe", 9.0, 4.6, C["mauve"])]
        for txt, x, y, col in tree:
            ax.add_patch(FancyBboxPatch((x - 1.05, y - 0.4), 2.1, 0.8, boxstyle="round,pad=0.05",
                                        fc=C["panel"], ec=col, lw=1.5))
            ax.text(x, y, txt, ha="center", va="center", fontsize=8.5, color=col)
        for (x1, y1), (x2, y2) in [((5, 8.8), (1.8, 7.4)), ((5, 8.8), (6.8, 7.4)), ((6.8, 6.6), (4.0, 5.0)),
                                   ((6.8, 6.6), (6.8, 5.0)), ((6.8, 6.6), (9.0, 5.0))]:
            ax.plot([x1, x2], [y1, y2], color=C["sub"])
        det = [(1.8, 6.3, "PLC (power line\nsignalling)\ntransfer trip\nSCADA\nwstawianie impedancji\n\n+ brak NDZ\n- drogie, łącze", C["sky"]),
               (4.0, 3.9, "U<, U>, f<, f>\nROCOF, dP/dt\nTHD, niesymetria\nfalki, AI\n\n+ proste, tanie\n- duża NDZ", C["yellow"]),
               (6.8, 3.9, "pomiar impedancji\nAFD, AFDPF, SMS\nALPS, APS\nskł. przeciwna\n\n+ mała NDZ\n- jakość energii", C["peach"]),
               (9.0, 3.9, "pasywna wyzwala\naktywną\n(np. ΔU + ΔP,\nU/f + przełącz.\nobciążenia)\n\n+ najlepszy\n  kompromis", C["mauve"])]
        for x, y, txt, col in det:
            ax.text(x, y, txt, ha="center", va="top", fontsize=7.5, color=col)
        return ["Odpowiedzi do 10 pytań", "kontrolnych z 18.8 są", "w polu objaśnień poniżej.", "",
                "Wymóg normowy:", " IEEE 1547 / IEC 61727:", " odłączenie ≤ 2 s", "",
                "Ocena metod:", " - strefa NDZ", " - fałszywe wyłączenia", " - koszt", " - wpływ na jakość energii"]


# =============================================================================
# OKNO GŁÓWNE
# =============================================================================

TABS = [("00 Przegląd", Tab00), ("01 Napięcie/straty", Tab01), ("02 Zwarcia/oślepienie", Tab02),
        ("03 Reklozer-bezp.", Tab03), ("04 Uziemienie SLG", Tab04), ("05 Wyspa NDZ", Tab05),
        ("06 ROCOF/SCO", Tab06), ("07 AFD/SMS", Tab07), ("08 Mikrosieć", Tab08),
        ("09 Odległ./różnic.", Tab09), ("10 SPZ", Tab10), ("11 Wiatr SCIG", Tab11),
        ("12 Cp/MPPT", Tab12), ("13 DFIG moc", Tab13), ("14 DFIG crowbar", Tab14),
        ("15 Fotowoltaika", Tab15), ("16 Sieci przyszłości", Tab16), ("17 Pytania 18.8", Tab17)]


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Ochrona rozproszonych źródeł odnawialnych (RDG) - rozdział 18 - interaktywne przykłady")
        self.geometry("1500x940")
        self.configure(bg=C["bg"])
        style_setup(self)
        nb = ttk.Notebook(self)
        nb.pack(fill=tk.BOTH, expand=True)
        self.nb = nb
        self.built = {}
        self.classes = {}
        for name, cls in TABS:
            fr = ttk.Frame(nb)
            nb.add(fr, text=name)
            self.classes[str(fr)] = (fr, cls)
        nb.bind("<<NotebookTabChanged>>", self._lazy)

    def _lazy(self, _e=None):
        """Zakładki budowane dopiero przy pierwszym otwarciu - szybszy start."""
        sel = self.nb.select()
        if sel in self.classes and sel not in self.built:
            fr, cls = self.classes[sel]
            tab = cls(fr)
            tab.pack(fill=tk.BOTH, expand=True)
            self.built[sel] = tab


if __name__ == "__main__":
    App().mainloop()
