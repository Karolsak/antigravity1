"""
Projekt i sterowanie silnikiem pojazdu elektrycznego (EV) - rozdział 12
=======================================================================
Interaktywna aplikacja Tkinter + Matplotlib ze WSZYSTKIMI przykładami,
ćwiczeniami (Exercise 12.1-12.5) i zadaniami (Problems 12.1-12.10) z rozdziału
"EV Motor Design and Control" (IPMSM do napędu samochodu elektrycznego).

Każda zakładka = jeden temat: suwaki (parametry), wykresy (wyniki), okienko
z wynikami liczbowymi oraz objaśnienie napisane językiem zrozumiałym dla
ucznia liceum (z komentarzem "jak to wygląda w praktyce inżynierskiej").

Zakładki:
  E1  Wymagania EV - silnik elektryczny vs silnik spalinowy z biegami
  E2  Liczba biegunów i żłobków, częstotliwość, straty w żelazie
  E3  Siła elektromotoryczna (SEM) i temperatura magnesu
  E4  Wektory napięcia i prądu (Ćw. 12.1, Zad. 12.2)
  E5  Krzywa moment-prędkość metodą "arkusza Excel" (Ćw. 12.2, 12.3, Zad. 12.3)
  E6  Zmienne indukcyjności (nasycenie) - autobus EV (Zad. 12.8)
  E7  Tętnienia momentu i skos wirnika (MES, 12.4.1)
  E8  Straty (miedź, histereza, wiroprądowe) i mapa sprawności
  E9  Rozmagnesowanie magnesu NdFeB (N39UH)
  E10 Naprężenia mechaniczne w mostkach wirnika
  E11 Uzwojenie: wypełnienie żłobka, rezystancja, okładzina (Zad. 12.1, 12.9)
  E12 Pomiar stałej SEM (Ćw. 12.4, Zad. 12.4, 12.8a, 12.10 - BLDC)
  E13 Pomiar indukcyjności Ld, Lq (rys. 12.35-12.36, Zad. 12.7b)
  E14 Doświadczalne szukanie optymalnych prądów (rys. 12.40-12.41, Zad. 12.5)
  E15 Tablica prądów LUT i kalibracja prędkości od Vdc (tab. 12.9, Ćw. 12.5, Zad. 12.6)
  E16 Sterowanie momentem z napięciowym anti-windup (rys. 12.47) - symulacja

Konwencje (jak w książce):
  P  - liczba BIEGUNÓW (np. 8), p = P/2 - liczba par biegunów
  prądy i napięcia fazowe w wartościach SZCZYTOWYCH (amplitudach)
  kąt prądu β mierzony od osi q:  id = -Is*sin(β),  iq = Is*cos(β)
  moment  Te = (3P/4) * [ψm*iq + (Ld - Lq)*id*iq]
  napięcia vd = rs*id - ωe*Lq*iq,  vq = rs*iq + ωe*(Ld*id + ψm)
  ograniczenie napięcia (SVPWM, zakres liniowy):  Vs + ΔV <= Vdc/√3

Uruchomienie:  python ev_motor_design_control_tkinter.py
Wymagania:     numpy, matplotlib (tkinter jest w standardowym Pythonie)
"""
import csv
import math
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

PI = math.pi
SQ3 = math.sqrt(3.0)
MU0 = 4e-7 * PI            # przenikalność magnetyczna próżni [H/m]
RHO_CU20 = 1.724e-8        # rezystywność miedzi w 20 °C [Ω·m]
ALPHA_CU = 0.0039          # współczynnik temperaturowy rezystancji miedzi [1/K]

# --- kolory (motyw ciemny w stylu Catppuccin Mocha) ---------------------------
C = dict(bg="#1e1e2e", mantle="#181825", surf="#313244", surf2="#45475a",
         text="#cdd6f4", sub="#a6adc8", cyan="#89dceb", green="#a6e3a1",
         blue="#89b4fa", peach="#fab387", red="#f38ba8", mauve="#cba6f7",
         yellow="#f9e2af", teal="#94e2d5", pink="#f5c2e7")
SERIES = [C["cyan"], C["green"], C["peach"], C["mauve"], C["yellow"], C["red"], C["blue"], C["teal"]]

matplotlib.rcParams.update({
    "figure.facecolor": C["bg"], "axes.facecolor": C["mantle"], "savefig.facecolor": C["bg"],
    "axes.edgecolor": C["surf2"], "axes.labelcolor": C["text"], "text.color": C["text"],
    "xtick.color": C["sub"], "ytick.color": C["sub"], "grid.color": C["surf2"],
    "grid.alpha": 0.5, "axes.grid": True, "legend.facecolor": C["surf"],
    "legend.edgecolor": C["surf2"], "legend.fontsize": 8, "axes.titlesize": 10,
    "axes.labelsize": 9, "axes.prop_cycle": matplotlib.cycler(color=SERIES),
})


# =============================================================================
# RDZEŃ OBLICZENIOWY - model IPMSM w układzie dq (stan ustalony)
# =============================================================================

def rpm_to_we(n_rpm, P):
    """Prędkość elektryczna ωe [rad/s] z prędkości obrotowej n [obr/min] i liczby biegunów P."""
    return np.asarray(n_rpm) / 60.0 * 2 * PI * (P / 2.0)


def dq_from_beta(Is, beta_deg):
    """Składowe prądu (id, iq) z amplitudy Is i kąta β mierzonego od osi q."""
    b = np.radians(beta_deg)
    return -Is * np.sin(b), Is * np.cos(b)


def torque(id_, iq, P, psi, Ld, Lq):
    """Moment elektromagnetyczny [Nm]: część magnetyczna + reluktancyjna."""
    return 0.75 * P * (psi * iq + (Ld - Lq) * id_ * iq)


def voltages(id_, iq, we, rs, psi, Ld, Lq):
    """Napięcia w osiach d, q i amplituda napięcia fazowego (stan ustalony)."""
    vd = rs * id_ - we * Lq * iq
    vq = rs * iq + we * (Ld * id_ + psi)
    return vd, vq, np.hypot(vd, vq)


def power_factor(id_, iq, vd, vq):
    """Współczynnik mocy = cos(kąt między wektorem napięcia a wektorem prądu)."""
    num = vd * id_ + vq * iq
    den = np.hypot(vd, vq) * np.hypot(id_, iq)
    return np.where(den > 0, num / np.where(den > 0, den, 1), 1.0)


def operating_point(n, Is, beta, prm):
    """Pełny punkt pracy (jak jeden wiersz arkusza Excel z rys. 12.16)."""
    we = float(rpm_to_we(n, prm["P"]))
    id_, iq = dq_from_beta(Is, beta)
    vd, vq, Vs = voltages(id_, iq, we, prm["rs"], prm["psi"], prm["Ld"], prm["Lq"])
    Te = torque(id_, iq, prm["P"], prm["psi"], prm["Ld"], prm["Lq"])
    Pm = Te * we / (prm["P"] / 2)
    delta = math.degrees(math.atan2(-vd, vq))           # kąt napięcia od osi q
    pf = math.cos(math.radians(beta - delta))
    return dict(we=we, id=float(id_), iq=float(iq), vd=float(vd), vq=float(vq),
                Vs=float(Vs), T=float(Te), P=float(Pm), delta=delta, pf=pf)


def max_torque_envelope(speeds, prm, Imax, Vlim, n_is=121, n_beta=721, L_of_beta=None):
    """Maksymalny moment przy każdej prędkości pod ograniczeniem prądu Is<=Imax
    i napięcia Vs<=Vlim (automatyczna wersja ręcznego szukania w Excelu).
    L_of_beta - opcjonalnie funkcja beta->(Ld, Lq) (indukcyjności zależne od kąta)."""
    Is = np.linspace(Imax / n_is, Imax, n_is)[:, None]
    beta = np.linspace(0.0, 89.5, n_beta)[None, :]
    id_, iq = dq_from_beta(Is, beta)
    if L_of_beta is not None:
        Ld, Lq = L_of_beta(beta)
    else:
        Ld, Lq = prm["Ld"], prm["Lq"]
    Te = torque(id_, iq, prm["P"], prm["psi"], Ld, Lq)
    out = {k: [] for k in ("n", "T", "P", "beta", "Is", "id", "iq", "Vs", "pf")}
    for n in speeds:
        we = float(rpm_to_we(n, prm["P"]))
        vd, vq, Vs = voltages(id_, iq, we, prm["rs"], prm["psi"], Ld, Lq)
        Tm = np.where(Vs <= Vlim, Te, -np.inf)
        k = np.unravel_index(np.argmax(Tm), Tm.shape)
        if not np.isfinite(Tm[k]):
            vals = (0.0, 0.0, np.nan, 0.0, 0.0, 0.0, np.nan, np.nan)
        else:
            i, j = k
            pf = power_factor(id_[k], iq[k], vd[k], vq[k])
            vals = (Te[k], Te[k] * we / (prm["P"] / 2), beta[0, j], Is[i, 0], id_[k], iq[k], Vs[k], float(pf))
        out["n"].append(n)
        for key, v in zip(("T", "P", "beta", "Is", "id", "iq", "Vs", "pf"), vals):
            out[key].append(float(v))
    return {k: np.array(v) for k, v in out.items()}


def current_command(T_ref, lam_max, prm, Imax, n_grid=240):
    """Optymalny rozkaz prądu dla zadanego momentu i dopuszczalnego strumienia:
    najmniejszy prąd |I| dający moment T_ref przy |λ| <= lam_max i |I| <= Imax.
    Jeżeli moment jest nieosiągalny - zwraca punkt maksymalnego osiągalnego momentu.
    (To jest to, co w sterowniku EV zapisuje się w tablicy LUT.)"""
    P, psi, Ld, Lq = prm["P"], prm["psi"], prm["Ld"], prm["Lq"]
    k = 0.75 * P
    idg = np.linspace(0.0, -Imax, n_grid)
    denom = k * (psi + (Ld - Lq) * idg)
    iq = T_ref / np.where(np.abs(denom) > 1e-9, denom, 1e-9)
    lam = np.hypot(psi + Ld * idg, Lq * iq)
    I = np.hypot(idg, iq)
    ok = (lam <= lam_max) & (I <= Imax) & (iq >= 0)
    if T_ref <= 0:
        return 0.0, 0.0
    if np.any(ok):
        j = np.argmin(np.where(ok, I, np.inf))
        return float(idg[j]), float(iq[j])
    # moment nieosiągalny -> maksymalny moment pod ograniczeniami
    rad = lam_max ** 2 - (psi + Ld * idg) ** 2
    iq_v = np.where(rad > 0, np.sqrt(np.clip(rad, 0, None)) / Lq, 0.0)
    iq_i = np.sqrt(np.clip(Imax ** 2 - idg ** 2, 0, None))
    iq_m = np.minimum(iq_v, iq_i)
    Tm = k * (psi + (Ld - Lq) * idg) * iq_m
    j = int(np.argmax(Tm))
    return float(idg[j]), float(iq_m[j])


def mtpa_curve(prm, Imax, n=60):
    """Krzywa MTPA (Maximum Torque Per Ampere) w płaszczyźnie (id, iq)."""
    Is = np.linspace(0, Imax, n)
    dL = prm["Lq"] - prm["Ld"]
    psi = prm["psi"]
    if dL <= 1e-12:
        return np.zeros_like(Is), Is
    idm = psi / (4 * dL) - np.sqrt(psi ** 2 / (16 * dL ** 2) + Is ** 2 / 2)
    return idm, np.sqrt(np.clip(Is ** 2 - idm ** 2, 0, None))


# --- straty w żelazie (model Steinmetza dwuskładnikowy) -----------------------

def core_loss_coeffs(W15_50=2.03, W10_400=13.3, t_mm=0.27):
    """Współczynniki kh, ke z danych katalogowych blachy (tu 27PNF1500):
       W = kh*f*B^2 + ke*f^2*B^2   [W/kg];  dwa punkty -> dwa równania."""
    A = np.array([[50 * 1.5 ** 2, 50 ** 2 * 1.5 ** 2], [400 * 1.0 ** 2, 400 ** 2 * 1.0 ** 2]])
    kh, ke = np.linalg.solve(A, np.array([W15_50, W10_400]))
    return float(kh), float(ke), t_mm


KH, KE, T_REF_LAM = core_loss_coeffs()


def core_loss_per_kg(f, B, t_mm=0.27):
    """Straty w żelazie [W/kg]: histereza ~ f*B^2, prądy wirowe ~ (t*f*B)^2."""
    ph = KH * f * B ** 2
    pe = KE * (t_mm / T_REF_LAM) ** 2 * f ** 2 * B ** 2
    return ph, pe


# --- pojazd / silnik spalinowy ------------------------------------------------

def ice_torque(n, T_peak, n_max):
    """Przybliżona krzywa momentu silnika spalinowego (parabola, max przy 0,55 n_max)."""
    n0 = 0.55 * n_max
    T = T_peak * (1 - 0.55 * ((n - n0) / n0) ** 2)
    return np.where((n >= 800) & (n <= n_max), np.clip(T, 0, None), np.nan)


def ev_torque(n, T_max, P_max_kw, n_max):
    """Krzywa napędu EV: stały moment do prędkości bazowej, potem stała moc."""
    w = np.maximum(n, 1e-6) * 2 * PI / 60
    T = np.minimum(T_max, P_max_kw * 1e3 / w)
    return np.where(n <= n_max, T, np.nan)


# --- dane z książki -----------------------------------------------------------

TABLE_12_1 = {   # pojazd: (P_max kW, bieguny/żłobki, kW/kg, T_max Nm, n_max rpm)
    "Leaf 2012": (80, "8/48", 1.4, 280, 10390),
    "Prius 2010": (60, "8/48", 1.6, 207, 13500),
    "Lexus 2008": (110, "8/48", 2.5, 300, 10230),
    "Camry 2007": (70, "8/48", 1.7, 270, 14000),
    "BMW i3 2016": (125, "12/72", 2.5, 250, 11400),
    "Volt Gen2 2015": (87, "8/72", 2.25, 280, 11000),
}

MOTOR_12_3 = dict(P=8, psi=0.0927, rs=0.013, Ld=0.234e-3, Lq=0.562e-3)   # tab. 12.3

# Zad. 12.8 - indukcyjności autobusu EV w funkcji kąta prądu (Is = 420 A)
TAB_12_8 = np.array([[30, 0.69, 1.59], [35, 0.70, 1.63], [40, 0.71, 1.67], [45, 0.72, 1.72],
                     [50, 0.73, 1.77], [55, 0.74, 1.82], [60, 0.75, 1.88], [65, 0.77, 1.95],
                     [70, 0.79, 2.02], [75, 0.83, 2.11], [80, 0.88, 2.27]])

# Rys. 12.43 - tablica optymalnych prądów (id, iq) dla Vdc = 360 V
LUT_FLUX = [0.180435, 0.170397, 0.160359, 0.150321, 0.140283, 0.130244, 0.120206, 0.110168,
            0.10013, 0.090092, 0.080054, 0.070016, 0.059978, 0.04994, 0.039902, 0.029864]
LUT_RPM = [2750, 2913, 3095, 3301, 3538, 3810, 4128, 4504, 4956, 5508, 6199, 7087, 8273, 9936, 12436, 16616]
LUT_RAW = """
0,0 -16.2,49.8 -39.7,89.1 -69.3,120.0 -94.4,151.1 -128.8,177.3 -163.8,202.2 -200.4,230.5 -234.7,260.6 -278.0,287.9 -320.0,320.0
0,0 -16.2,49.8 -39.4,88.5 -68.6,118.8 -96.3,148.3 -127.2,175.1 -162.9,201.1 -197.6,227.3 -231.8,257.5 -274.6,284.3 -346.7,290.9
0,0 -16.2,49.8 -38.5,86.6 -65.1,117.5 -94.0,144.7 -124.7,171.6 -155.0,198.4 -190.1,222.6 -239.6,240.4 -302.7,248.6 -370.7,259.6
0,0 -15.7,48.4 -37.4,84.0 -63.1,113.8 -88.4,141.5 -116.8,166.8 -151.5,188.5 -201.0,201.0 -254.2,211.8 -315.9,222.0 -387.9,233.1
0,0 -14.1,46.0 -34.3,80.7 -56.5,110.9 -82.3,137.0 -124.2,152.3 -168.2,165.8 -215.0,177.2 -268.4,188.7 -329.4,198.7 -401.4,209.0
0,0 -12.5,43.5 -32.0,75.5 -52.6,103.3 -91.5,120.5 -132.8,133.2 -176.8,145.7 -221.3,157.3 -272.2,167.5 -332.5,176.0 -413.4,184.1
0,0 -11.3,39.4 -28.6,70.8 -60.6,90.5 -98.6,105.3 -139.5,117.5 -181.2,128.3 -223.6,138.6 -272.4,147.3 -333.2,155.4 -423.1,160.7
0,0 -10.5,36.7 -33.5,62.3 -68.9,78.7 -105.3,92.2 -143.4,103.4 -180.7,113.8 -224.3,123.3 -269.7,131.0 -328.8,137.5 -430.2,140.6
0,0 -20.0,30.8 -48.1,51.8 -78.6,66.9 -111.8,79.5 -145.3,90.4 -180.6,100.1 -219.3,108.4 -261.4,115.3 -317.1,121.1 -420.8,123.0
-28.3,0 -44.4,24.8 -65.0,42.7 -92.7,57.1 -121.9,68.9 -151.8,79.0 -182.2,87.7 -217.6,95.5 -256.2,102.0 -305.6,107.3 -395.2,109.6
-55.2,0 -65.9,21.4 -83.2,35.7 -104.1,47.9 -128.8,58.4 -155.7,67.4 -184.5,75.7 -215.2,82.6 -250.6,88.7 -295.2,93.6 -377.0,95.4
-82.0,0 -93.2,16.9 -106.4,28.9 -121.0,39.3 -141.7,48.8 -164.4,56.9 -187.4,63.8 -215.1,70.3 -244.5,75.9 -283.0,80.1 -365.9,81.1
-111.7,0 -119.3,14.9 -127.7,24.8 -139.1,32.6 -154.6,40.6 -173.2,47.4 -193.5,53.7 -216.9,59.3 -243.5,63.9 -278.9,68.0 -346.6,69.9
-138.6,0 -147.9,12.7 -157.2,19.3 -164.8,26.1 -178.1,32.4 -191.4,38.2 -207.7,43.0 -226.9,47.8 -247.6,52.6 -275.8,56.1 -327.2,58.3
-173.9,0 -177.8,11.5 -178.8,16.6 -186.8,21.6 -192.0,26.3 -199.9,30.6 -213.5,35.0 -230.1,38.5 -246.8,41.7 -270.7,44.6 -307.6,46.5
-198.0,0 -199.1,11.8 -203.1,15.3 -205.7,18.4 -212.5,21.6 -219.3,24.2 -228.9,27.3 -237.2,29.5 -248.2,32.2 -263.6,34.7 -294.8,35.7
"""
LUT = np.array([[[float(v) for v in cell.split(",")] for cell in line.split()]
                for line in LUT_RAW.strip().splitlines()])          # kształt (16, 11, 2)
LUT_THROTTLE = np.arange(0, 101, 10)
# Rys. 12.42 - maksymalny moment zmierzony przy Vdc = 360 V (prędkość -> moment)
FIG_12_42_TMAX = np.array([[2750, 309], [3300, 288], [3810, 253], [4500, 208],
                           [6200, 139], [10000, 72], [12000, 55], [16616, 40]])


def lut_tmax(n_fict):
    """Maksymalny moment przy danej (fikcyjnej) prędkości wg rys. 12.42."""
    n0 = FIG_12_42_TMAX[:, 0]
    T0 = FIG_12_42_TMAX[:, 1]
    return float(np.interp(n_fict, n0, T0, left=T0[0], right=T0[-1]))


def lut_read(T_ref, n, Vdc, P=8):
    """Odczyt rozkazu prądu z tablicy LUT z kalibracją prędkości od napięcia Vdc:
       1) strumień dopuszczalny  λ = Vdc / (√3 ωe)
       2) prędkość fikcyjna      n' = n * 360 / Vdc   (ten sam strumień przy 360 V)
       3) przepustnica [%]       = 100 * T / Tmax(n')
       4) interpolacja dwuliniowa w tablicy (wiersz = strumień, kolumna = %)."""
    we = float(rpm_to_we(max(n, 1.0), P))
    lam = Vdc / (SQ3 * we)
    n_f = n * 360.0 / Vdc
    thr = float(np.clip(100.0 * T_ref / lut_tmax(n_f), 0, 100))
    fl = np.array(LUT_FLUX)
    # indeks wiersza (strumień maleje wraz z numerem wiersza)
    if lam >= fl[0]:
        r0, r1, a = 0, 0, 0.0
    elif lam <= fl[-1]:
        r0, r1, a = len(fl) - 1, len(fl) - 1, 0.0
    else:
        r1 = int(np.argmax(fl < lam))
        r0 = r1 - 1
        a = (fl[r0] - lam) / (fl[r0] - fl[r1])
    c = thr / 10.0
    c0 = int(min(math.floor(c), 9))
    b = c - c0

    def cell(r):
        return (1 - b) * LUT[r, c0] + b * LUT[r, c0 + 1]
    idq = (1 - a) * cell(r0) + a * cell(r1)
    return dict(lam=lam, n_f=n_f, thr=thr, row=r0 + a, id=float(idq[0]), iq=float(idq[1]))


# --- magnes NdFeB N39UH (tab. 12.8) ------------------------------------------

def magnet_curves(T_c, Br20=1.25, Hcj20=1989e3, aBr=-0.12, aHcj=-0.55, H=None):
    """Krzywe odmagnesowania magnesu NdFeB w temperaturze T_c [°C].
    J(H) - polaryzacja wewnętrzna (krzywa "intrinsic"), B(H) = J + μ0*H.
    aBr, aHcj - współczynniki temperaturowe [%/K]. Kolano ~ blisko Hcj."""
    Br = Br20 * (1 + aBr / 100 * (T_c - 20))
    Hcj = max(Hcj20 * (1 + aHcj / 100 * (T_c - 20)), 50e3)
    if H is None:
        H = np.linspace(-2.2e6, 0, 1200)
    h0 = 0.07 * Hcj
    J = Br * (1 - np.exp(-(Hcj + H) / h0)) / (1 - math.exp(-Hcj / h0))
    J = np.where(H > -Hcj, J, np.nan)
    B = J + MU0 * H
    return H, B, J, Br, Hcj


def magnet_operating_point(T_c, Pc, Ha, **kw):
    """Punkt pracy magnesu: przecięcie krzywej B(H) z prostą obciążenia
    B = -μ0*Pc*(H + Ha); Ha - pole rozmagnesowujące od prądu -id [A/m]."""
    H, B, J, Br, Hcj = magnet_curves(T_c, **kw)
    line = -MU0 * Pc * (H + Ha)
    diff = np.where(np.isfinite(B), B - line, np.nan)
    ok = np.isfinite(diff)
    Hs, ds = H[ok], diff[ok]
    s = np.where(np.sign(ds[:-1]) != np.sign(ds[1:]))[0]
    if len(s) == 0:
        return dict(H=-Hcj, B=0.0, J=0.0, Br=Br, Hcj=Hcj, ratio=0.0)
    i = s[-1]
    t = ds[i] / (ds[i] - ds[i + 1])
    Hop = Hs[i] + t * (Hs[i + 1] - Hs[i])
    Hop_, Bop, Jop, _, _ = magnet_curves(T_c, H=np.array([Hop]), **kw)
    return dict(H=float(Hop), B=float(Bop[0]), J=float(Jop[0]), Br=Br, Hcj=Hcj,
                ratio=float(Jop[0] / Br))


# --- nieliniowy (nasycający się) model indukcyjności - silnik z tab. 12.3 -----

def Ld_sat(id_):
    """Ld(id) wg charakteru rys. 12.36a: lekko maleje ze wzrostem |id| [H]."""
    return (0.2665 - 0.055e-3 * np.abs(id_)) * 1e-3


def Lq_sat(iq):
    """Lq(iq) wg rys. 12.36b: najpierw rośnie, potem maleje (nasycenie) [H]."""
    x = np.abs(iq) / 90.0
    return (0.50 + 0.25 * x * np.exp(1 - x)) * 1e-3


# --- BLDC: trapezowa SEM (Zad. 12.10) -----------------------------------------

def trapezoid_emf(theta, E, flat_deg=120.0):
    """Idealna trapezowa SEM: płaski wierzchołek flat_deg (elektr.), liniowe zbocza."""
    a = math.radians((180.0 - flat_deg) / 2.0)          # połowa szerokości zbocza
    th = np.mod(theta + PI, 2 * PI) - PI                # -π..π
    x = np.abs(th)
    s = np.sign(th)
    y = np.where(x < a, x / a, np.where(x <= PI - a, 1.0, (PI - x) / a))
    return E * s * y


def trapezoid_coeffs(E, nmax=15, flat_deg=120.0):
    """Współczynniki szeregu Fouriera (tylko sinusy, nieparzyste n):
       b_n = 4E/(π n^2 a) * sin(n a),   a = połowa zbocza w radianach."""
    a = math.radians((180.0 - flat_deg) / 2.0)
    n = np.arange(1, nmax + 1)
    b = np.where(n % 2 == 1, 4 * E / (PI * n ** 2 * a) * np.sin(n * a), 0.0)
    return n, b


# --- symulacja sterowania momentem z napięciowym anti-windup (rys. 12.47) -----

def simulate_torque_control(prm, Imax, T_ref, n0, n1, t_end, Vdc0, Vdc1, t_sag,
                            psi_err_pct, dV_margin, anti_windup=True, Kaw=0.004,
                            f_bw=400.0, dt=1e-5, lut_every=10):
    """Symulacja dynamiczna IPMSM z regulatorami PI prądu (z odsprzęganiem),
    ograniczeniem napięcia falownika i pętlą anti-windup zmniejszającą strumień.
    Prędkość wymusza pojazd (rampa n0->n1). Sterownik zna ψm z błędem psi_err_pct."""
    P, rs, psi, Ld, Lq = prm["P"], prm["rs"], prm["psi"], prm["Ld"], prm["Lq"]
    ctrl = dict(prm)
    ctrl["psi"] = psi * (1 + psi_err_pct / 100.0)          # model "w głowie" sterownika
    wc = 2 * PI * f_bw
    Kpd, Kpq, Ki = wc * Ld, wc * Lq, wc * rs
    N = int(t_end / dt)
    rec_every = max(N // 1500, 1)
    id_ = iq = 0.0
    xd = xq = 0.0              # całki regulatorów PI
    dlam = 0.0                 # korekta strumienia z anti-windup
    idr = iqr = 0.0
    out = {k: [] for k in ("t", "n", "Tref", "T", "id", "iq", "idr", "iqr", "Vref", "Vlim", "dlam")}
    for k in range(N):
        t = k * dt
        n = n0 + (n1 - n0) * min(t / t_end, 1.0)
        we = n / 60 * 2 * PI * P / 2
        Vdc = Vdc0 if t < t_sag else Vdc1
        Vlim = Vdc / SQ3
        Tr = T_ref if t >= 0.01 else 0.0
        if k % lut_every == 0:
            lam_ff = (Vlim - dV_margin) / max(we, 1.0)
            lam_cmd = max(lam_ff - dlam, 0.005)
            idr, iqr = current_command(Tr, lam_cmd, ctrl, Imax, n_grid=160)
        ed, eq = idr - id_, iqr - iq
        vd_ff = -we * ctrl["Lq"] * iq
        vq_ff = we * (ctrl["Ld"] * id_ + ctrl["psi"])
        vd_r = Kpd * ed + xd + vd_ff
        vq_r = Kpq * eq + xq + vq_ff
        Vref = math.hypot(vd_r, vq_r)
        sat = Vref > Vlim
        if sat:
            vd, vq = vd_r * Vlim / Vref, vq_r * Vlim / Vref
        else:
            vd, vq = vd_r, vq_r
        if not sat:                                         # całkowanie warunkowe
            xd += Ki * ed * dt
            xq += Ki * eq * dt
        if anti_windup:                                     # pętla anti-windup strumienia
            dlam = max(dlam + Kaw * (Vref - (Vlim - 0.5 * dV_margin)) * dt * 50, 0.0)
        # obiekt - równania elektryczne IPMSM, metoda Heuna (RK2)
        def f(a, b):
            return ((vd - rs * a + we * Lq * b) / Ld,
                    (vq - rs * b - we * (Ld * a + psi)) / Lq)
        k1 = f(id_, iq)
        k2 = f(id_ + dt * k1[0], iq + dt * k1[1])
        id_ += 0.5 * dt * (k1[0] + k2[0])
        iq += 0.5 * dt * (k1[1] + k2[1])
        if k % rec_every == 0:
            for key, v in (("t", t), ("n", n), ("Tref", Tr), ("T", torque(id_, iq, P, psi, Ld, Lq)),
                           ("id", id_), ("iq", iq), ("idr", idr), ("iqr", iqr), ("Vref", Vref),
                           ("Vlim", Vlim), ("dlam", dlam)):
                out[key].append(v)
    return {k: np.array(v) for k, v in out.items()}


# =============================================================================
# OBJAŚNIENIA (dla ucznia liceum, z komentarzem inżyniera)
# =============================================================================

EXPL = {
"E1": """E1 - CZEGO WYMAGAMY OD SILNIKA SAMOCHODU ELEKTRYCZNEGO? (rys. 12.1, tab. 12.1, 12.2)

INTUICJA: silnik spalinowy (ICE) jest jak biegacz, który dobrze biega tylko w wąskim zakresie
tempa: nie daje żadnego momentu przy zerowej prędkości (musi mieć sprzęgło) i ma dobrą
sprawność tylko w "środku" zakresu obrotów. Dlatego potrzebuje skrzyni biegów - każdy bieg
to osobna krzywa (górny wykres: 1., 2., ... 5. bieg). Silnik elektryczny daje PEŁNY moment
od zera i pracuje w bardzo szerokim zakresie prędkości, więc wystarcza mu JEDNO przełożenie.

KRZYWA SILNIKA EV ma dwa obszary:
  - stały moment T_max od 0 do prędkości bazowej n_b (ogranicza nas prąd = grzanie),
  - stała moc P_max powyżej n_b (ogranicza nas napięcie akumulatora):  T = P_max / ω.
Siła na kole:  F = T * G * η / r ,   prędkość auta: v = (n * 2π/60) * r / G.

SPRAWDZENIE DANYCH NISSANA LEAF 2018 (tab. 12.2):
  P = T * ω = 320 Nm * (3283 * 2π/60) rad/s = 320 * 343,8 = 110 000 W = 110 kW   ✔
  czyli moment maksymalny i moc szczytowa "spotykają się" dokładnie w n_b = 3283 obr/min.
  Zakres stałej mocy: 3283...9795 obr/min, tzn. CPSR = 9795/3283 ≈ 3 (książka: typowo 3,5-4).

WYMAGANIA (rys. 12.2): gęstość mocy > 5,7 kW/litr i > 2 kW/kg, koszt < 4,7 $/kW,
moment szczytowy ≈ 2 x znamionowy przez 20-40 s (ogranicza temperatura uzwojenia),
magnes NIE MOŻE się rozmagnesować w żadnym stanie pracy, SEM przy n_max nie może
zniszczyć tranzystorów IGBT, gdy falownik straci kontrolę.

Dolny wykres porównuje silniki z tab. 12.1. Zauważ: wszystkie mają 8 lub 12 biegunów,
a gęstość mocy rośnie z rokiem produkcji (1,4 -> 2,5 kW/kg).

KOMENTARZ INŻYNIERA: silnik projektuje się tak, by był najsprawniejszy tam, gdzie auto
jeździ NAJCZĘŚCIEJ (cykl jazdy WLTP/JC08 - umiarkowany moment, średnia prędkość),
a nie w punkcie mocy maksymalnej.
SPRÓBUJ: zmniejsz P_max - punkt przejścia n_b przesuwa się w lewo; zwiększ przełożenie G -
większa siła na starcie, ale niższa prędkość maksymalna auta.
""",
"E2": """E2 - LICZBA BIEGUNÓW I ŻŁOBKÓW (p. 12.2.1-12.2.2)

Stojan ma Ns żłobków (rowków na uzwojenie), wirnik P biegunów magnetycznych (N-S-N-S...).
Najpopularniejsze pary w EV:  48 żłobków / 8 biegunów (Prius, Leaf, Accord)
                              72 żłobki / 12 biegunów (BMW i3, Chevrolet Volt Gen2).
Liczba żłobków na biegun i fazę:   q = Ns / (3 * P) = 48/(3*8) = 2   (w obu przypadkach!)
Większe q -> mniej tętnień momentu i harmonicznych, ale trudniejsze nawijanie.

CZĘSTOTLIWOŚĆ ELEKTRYCZNA:  f = n/60 * P/2
  BMW i3: f = 11400/60 * 6 = 1140 Hz  (!),   silnik 8-biegunowy przy 12000 obr/min: 800 Hz.
Dla porównania - gniazdko w domu ma 50 Hz.

DLACZEGO WIĘCEJ BIEGUNÓW? Strumień jednego bieguna jest mniejszy, więc jarzmo (grzbiet)
stojana może być cieńsze -> lżejszy silnik (wykres "względna grubość jarzma" ~ 1/P).
CENA: wyższa częstotliwość -> większe straty w żelazie (środkowy wykres):
     histereza ~ f * B²            (przemagnesowywanie domen - jak zginanie drutu)
     prądy wirowe ~ (t * f * B)²    (t - grubość blachy!)
Dlatego BMW stosuje bardzo cienką blachę 0,2 mm (zamiast 0,27-0,35 mm). Jest droga
(trudne walcowanie), a odpad przy wykrawaniu ogranicza się segmentami z "jaskółczym ogonem".

Współczynniki kh i ke wyliczamy z katalogu blachy 27PNF1500: W15/50 = 2,03 W/kg
(1,5 T, 50 Hz) i W10/400 = 13,3 W/kg (1 T, 400 Hz) - dwa punkty = dwa równania z dwiema
niewiadomymi (to zwykły układ równań z liceum!).

WARSTWY BARIER STRUMIENIA (p. 12.2.2): dla małych tętnień zaleca się liczbę warstw
Ns/P ± 2 -> dla 48/8: 6 ± 2 = 4 lub 8 warstw. Rodzaje wirników: V (Prius), podwójne V
(Volt, Bolt), delta ∇ (Leaf, Lexus LS), podwójne I (BMW i3) - wszystkie działają podobnie,
V jest najtańszy i najodporniejszy na rozmagnesowanie.
""",
"E3": """E3 - SIŁA ELEKTROMOTORYCZNA (SEM) I TEMPERATURA MAGNESU (p. 12.3.1, rys. 12.10)

Gdy wirnik z magnesami się obraca, w uzwojeniach stojana indukuje się napięcie (prawo
Faradaya) - jak w prądnicy rowerowej (dynamie). Amplituda tego napięcia:
        E = ωe * ψm        (ψm - strumień skojarzony magnesu, "siła" magnesu w Wb)
Z symulacji MES przy 3600 obr/min i 140 °C składowa podstawowa SEM = 143,8 V, więc
        ωe = 3600/60 * 2π * 4 = 1508 rad/s,      ψm = 143,8 / 1508 = 0,095 Wb.

MAGNES SŁABNIE Z TEMPERATURĄ: remanencja magnesu Nd-Fe-B zmienia się o -0,12 %/K.
  ψm(T) = ψm(140°C) * [1 - 0,0012 * (T - 140)]
  np. przy 20 °C:  0,095 * (1 + 0,0012*120) = 0,109 Wb  (o 14 % więcej!)
Książka: 0,095 Wb to NAJMNIEJSZA wartość, bo magnes jest utrzymywany poniżej 140 °C.
Projektant musi sprawdzić DWA skrajne przypadki:
  - gorący magnes (mały ψm) -> najmniejszy moment - czy auto nadal przyspiesza jak trzeba?
  - zimny magnes (duży ψm) -> największa SEM przy n_max - czy nie przebije tranzystorów?

HARMONICZNE: SEM nie jest idealną sinusoidą (żłobki, kształt magnesu). Środkowy wykres
pokazuje widmo - w układzie gwiazdy harmoniczne rzędu 3, 9, ... znikają z napięć
międzyprzewodowych, ale 5, 7, 11, 13 zostają i powodują tętnienia momentu 6. i 12. rzędu.

SPRÓBUJ: zwiększ temperaturę do 180 °C i porównaj SEM przy n_max = 12000 obr/min
z napięciem akumulatora Vdc/√3 (prawy wykres) - gdy SEM > Vdc/√3 a falownik się
wyłączy, silnik ładuje baterię niekontrolowanie (tzw. "uncontrolled generation").
""",
"E4": """E4 - WEKTORY NAPIĘCIA I PRĄDU (Ćwiczenie 12.1 i Zadanie 12.2)

NAJWIĘKSZE NAPIĘCIE FAZOWE: gdy falownik przyłącza pełne Vdc między dwa zaciski (rys. 12.11),
szczytowe napięcie międzyprzewodowe = Vdc, więc fazowe:  Vmax = Vdc/√3 = 360/1,732 = 207,9 V.

ROZWIĄZANIE ĆW. 12.1 a) - 2500 obr/min, Is = 453 A, β = 43°  (ψm = 0,0984 Wb, rs = 0,0147 Ω):
  1) prędkość elektryczna:  ωe = 2500/60 * 2π * 4 = 1047,2 rad/s
  2) SEM:                   ωe*ψm = 1047,2 * 0,0984 = 103 V
  3) prądy:   id = -453*sin43° = -308,9 A,   iq = 453*cos43° = 331,3 A
  4) spadki na rezystancji:  rs*id = -4,5 V,  rs*iq = 4,9 V
  5) vd = rs*id - ωe*Lq*iq = -4,5 - 1047,2*0,562e-3*331,3 = -199,5 V
     vq = rs*iq + ωe*Ld*id + ωe*ψm = 4,9 - 75,7 + 103 = 32,2 V
  6) Vs = √(vd² + vq²) = 202 V  < 207,9 V  ✔
  7) kąt napięcia od osi q:  δ = atan(199,5/32,2) = 80,8°
     PF = cos(β - δ) = cos(-37,8°) = 0,79
Presety b) 7200 i c) 12000 obr/min liczą to samo. UWAGA (wynik uczciwy): w c) dla dokładnie
Is = 453 A i β = 80,75° wychodzi Vs ≈ 214 V > 207,9 V - trzeba lekko zmniejszyć prąd
lub zwiększyć β (spróbuj suwakiem!) - dokładnie tak jak robi to krok 8 procedury Excela.

LEWY WYKRES: układ dq - strzałka prądu Is (skala x0,4) oraz "łańcuch" napięć:
SEM ωe*ψm (wzdłuż osi q) + spadek ωe*Ld*id + spadek -ωe*Lq*iq + rs*I = Vs.
Czerwony okrąg = granica napięcia falownika. Wektor napięcia MUSI zmieścić się w okręgu.

PRAWY WYKRES: jak Vs i moment zależą od β przy stałym Is i prędkości. Im większe β
(bardziej ujemny id), tym bardziej "osłabiamy" magnes -> niższe napięcie, ale mniejszy moment.
Program sam znajduje NAJMNIEJSZE β spełniające warunek napięcia (to Zadanie 12.2:
10000 obr/min, Is = 350 A - wybierz preset i przeczytaj β_min w wynikach).
""",
"E5": """E5 - KRZYWA MOMENT-PRĘDKOŚĆ METODĄ "ARKUSZA EXCEL" (p. 12.3.4, rys. 12.16, Ćw. 12.2, 12.3, Zad. 12.3)

Książka pokazuje prostą metodę inżynierską: arkusz, w którym dla każdej prędkości ręcznie
dobieramy amplitudę prądu Is i kąt β tak, by moment był NAJWIĘKSZY, a napięcie się mieściło:
          Vs + ΔV <= Vdc/√3       (ΔV ≈ 6-10 V - zapas dla regulatora prądu PI)
Kolumny arkusza: n -> ωe -> Is, β -> id, iq -> T, P -> vd, vq, Vs -> PF.
Ten program robi to samo AUTOMATYCZNIE (przeszukuje siatkę Is x β) - przycisk
"Tabela (arkusz)" pokazuje wynik w postaci tabeli, jak w Excelu, z eksportem CSV.

TRZY OBSZARY PRACY (środkowy wykres):
  1) STAŁY MOMENT (0...~2500 obr/min): pełny prąd Is, β = β_MTPA ≈ 34-43° (MTPA -
     maksymalny moment na amper). Napięcie jeszcze nie przeszkadza.
  2) OSŁABIANIE POLA / STAŁA MOC: napięcie doszło do granicy, więc zwiększamy β
     (bardziej ujemny id osłabia strumień magnesu). Moment spada ~1/n, moc ≈ stała.
  3) MTPV (Maximum Torque Per Volt): gdy β ≈ 75-81°, samo β już nie wystarcza -
     trzeba zmniejszać także Is.
Wynik dla tab. 12.3 (Is = 453 A, ΔV = 5,9 V): moment rozruchowy ≈ 409 Nm
(książka: 406 Nm) - nierealny, bo model nie uwzględnia NASYCENIA rdzenia (patrz E6).

ĆW. 12.2: ψm = 0,103 Wb, Is = 280 A, ΔV = 6 V, do 10000 obr/min - wybierz preset.
ĆW. 12.3: Is = 212 A, Vdc = 325 V, do 8000 obr/min - sprawdź, czy moc > 60 kW w obszarze
osłabiania pola (Ld, Lq przyjęte z tab. 12.3, bo tabela z rys. 12.19 jest nieczytelna).
ZAD. 12.3: ψm = 0,09 Wb, rs = 0,01 Ω, Ld = 280 µH, Lq = 500 µH, Vdc = 325 V, Is = 280 A.

KOMENTARZ INŻYNIERA: PF na dolnym wykresie rośnie w osłabianiu pola - przy wysokich
prędkościach silnik "lepiej wykorzystuje" falownik. Porównanie z pomiarem (rys. 12.18):
błąd < 7 %, pochodzi z błędu indukcyjności i nieuwzględnionych strat mechanicznych.
""",
"E6": """E6 - ZMIENNE INDUKCYJNOŚCI (NASYCENIE) - SILNIK AUTOBUSU EV (p. 12.3.3, rys. 12.15, 12.17, Zad. 12.8)

Indukcyjność mówi, jak łatwo prąd wytwarza strumień magnetyczny. Żelazo przewodzi strumień
świetnie... do czasu! Przy dużej indukcji (~1,8-1,9 T w zębach) "nasyca się" jak gąbka pełna
wody - dalszy wzrost prądu daje coraz mniej strumienia. Dlatego Ld i Lq NIE są stałe:
zależą od prądu i jego kąta β (rys. 12.15). Lq zmienia się mocniej niż Ld, bo strumień osi q
płynie całym żelazem, a strumień osi d musi przejść przez magnes (który działa jak powietrze).

ZADANIE 12.8 (silnik 12-biegunowy autobusu, Is = 420 A, bateria 600 V):
 a) SEM fazowa 67,482 V przy 300 obr/min:
    ωe = 300/60 * 2π * 6 = 188,5 rad/s    ->   ψm = 67,482 / 188,5 = 0,358 Wb
    (przyjęto, że 67,482 V to amplituda - jak w całym rozdziale).
 b) Moment w funkcji β z TABELI Ld(β), Lq(β) (górny wykres, kropki = punkty tabeli).
    Wynik: maksimum przy β ≈ 35°, T ≈ 1800 Nm (autobus potrzebuje ogromnego momentu!).
    Sprawdzenie dla β = 45°: id = -297 A, iq = 297 A,
      T = 3·12/4·[0,358·297 + (0,72-1,72)·10⁻³·(-297)·297] = 9·(106,3 + 88,2) = 1750 Nm.
    Linia przerywana: gdyby L były stałe (wartości przy 45°) - wyszłoby ok. 1860 Nm przy 32°,
    czyli model bez nasycenia PRZECENIA moment. Dlatego projektant MUSI uwzględnić nasycenie.
 c) Moment i moc przy prędkościach 300...3200 obr/min (Vmax = 600/√3 = 346 V),
    dolny wykres. Wybieramy β (30...80°) i ewentualnie mniejszy prąd tak, by napięcie
    się zmieściło. Kropki = prędkości z treści zadania.

W PRAKTYCE (rys. 12.17): arkusz ma dodatkowe kolumny Ld, Lq odczytywane z tablicy (LUT)
z MES lub z pomiarów. To proces iteracyjny: zmieniam (Is, β) -> zmieniają się L -> zmienia
się napięcie -> poprawiam (Is, β)... Wartości pomiędzy punktami tablicy - interpolacja.
SPRÓBUJ: suwak "skala Lq" symuluje silniejsze/słabsze nasycenie.
""",
"E7": """E7 - TĘTNIENIA MOMENTU I SKOS WIRNIKA (p. 12.4.1, rys. 12.20)

Moment silnika nie jest idealnie stały - "faluje" podczas obrotu. Powód: zęby stojana
i magnesy wirnika przyciągają się nierówno (jak magnes przesuwany po grzebieniu), a SEM
ma harmoniczne. Tętnienia = drgania, hałas, "buczenie" auta. Definicja:
        tętnienia [%] = (T_max - T_min) / T_śr * 100%
W silniku z książki: 28,5 % przed skosem, 10 % po skosie (przy 320 Nm, 3600 obr/min).
Dominuje harmoniczna 6. rzędu (6 okresów na obrót elektryczny) i 12. (od żłobków: 48/4 = 12).

SKOS (skew): wirnik dzieli się osiowo na N segmentów (4-6 w EV), każdy obrócony o kąt
θ/N. Tętnienia z różnych segmentów są przesunięte w fazie i CZĘŚCIOWO SIĘ ZNOSZĄ
(jak fale na wodzie, które się wygaszają). Współczynnik skosu dla harmonicznej h:
      k_h = sin(h*θ/2) / (N * sin(h*θ/(2N)))        (N -> ∞: skos ciągły, sin(x)/x)
Amplituda h-tej harmonicznej po skosie = k_h * amplituda przed skosem.
Harmoniczną h całkowicie usuwa skos θ = 360°/h (elektrycznych):  h=12 -> 30°, h=6 -> 60°.

W KSIĄŻCE: podziałka żłobkowa 48-żłobkowego stojana = 7,5° mech. = 30° el.;
skos = pół podziałki -> 3,75° mech. = 15° el. (silnik 8-biegunowy: kąt el. = 4 x mech.).
CENA SKOSU: moment średni maleje o czynnik k_1 (zwykle < 1 %) i rośnie koszt montażu.
W tym prostym modelu skos 15° el daje ok. 18 % tętnień; aby zejść do ~10-11 % jak w książce,
potrzeba ok. 30° el (cała podziałka żłobkowa - usuwa 12. harmoniczną). Sprawdź suwakiem!

KOMENTARZ INŻYNIERA: prosty model pokazuje trend; w rzeczywistości składowe tętnień zależą
od prądu (przy małym prądzie są małe - rys. 12.20 c, f), a pełną odpowiedź daje MES 3D.
""",
"E8": """E8 - STRATY I MAPA SPRAWNOŚCI (p. 12.4.2, rys. 12.22, 12.46)

Żaden silnik nie zamienia 100 % energii elektrycznej na mechaniczną. Straty to ciepło:
  1) STRATY W MIEDZI:   P_Cu = 3/2 * rs * Is²        (Is - amplituda; = 3 * rs * I_rms²)
     Dominują przy MAŁEJ prędkości i DUŻYM momencie (ruszanie, podjazd).
  2) HISTEREZA w żelazie: P_h ~ f * B²  - rośnie liniowo z prędkością.
  3) PRĄDY WIROWE:        P_e ~ f² * B² - rośnie z KWADRATEM prędkości!
     Przy 12000 obr/min (800 Hz) książka podaje ok. 2 kW samych strat wiroprądowych.
  4) Mechaniczne (łożyska, wentylacja) ~ n + n².
W osłabianiu pola prąd maleje (mniej P_Cu), a strumień maleje (B ~ |λ|/ψm) - dlatego
straty w żelazie nie rosną tak szybko, jak sugerowałby wzór f².

Indukcję w zębach przyjmujemy B = B0 * |λ|/ψm, gdzie λ = √((ψm+Ld id)² + (Lq iq)²).
Współczynnik "dodatkowe straty" uwzględnia harmoniczne pola od żłobków i PWM, których
prosty model nie widzi (MES pokazuje, że najwięcej strat jest na krawędziach zębów).

MAPA SPRAWNOŚCI (dolny wykres): dla każdego punktu (n, T) szukamy prądu (id, iq) dającego
NAJMNIEJSZE straty przy spełnieniu ograniczeń prądu i napięcia. η = P_mech/(P_mech + straty).
Wniosek jak w książce: najniższa sprawność przy dużym momencie i małej prędkości
(miedź) oraz małym momencie i dużej prędkości (żelazo + mechanika). Maksimum ≈ 96-97,5 %
leży "w środku" mapy - tam gdzie auto jeździ najczęściej.
""",
"E9": """E9 - ROZMAGNESOWANIE MAGNESU (p. 12.4.3, tab. 12.8, rys. 12.23)

Magnes N39UH (Nd-Fe-B): Br = 1,22-1,28 T, iHc = 1989 kA/m, bHc = 915 kA/m przy 20 °C.
Magnes ma "odporność" na przeciwne pole - koercję iHc. Gdy pole przeciwne jest za duże
(lub magnes za gorący), domeny magnetyczne "przewracają się" NA STAŁE - silnik traci moment
bezpowrotnie. To najgorsza awaria silnika PMSM!

WYKRES B-H: krzywe odmagnesowania w różnych temperaturach. Każda ma "KOLANO" - punkt,
za którym B gwałtownie spada. Punkt pracy magnesu = przecięcie krzywej z PROSTĄ OBCIĄŻENIA:
      B = -μ0 * Pc * (H + Ha)
  Pc - współczynnik permeancji (geometria: grubość magnesu / szczelina), typowo 2-6,
  Ha - pole przeciwne od prądu osi d (ujemne id "wpycha" pole przeciwne w magnes).
Jeśli punkt pracy jest PRZED kolanem - rozmagnesowanie ODWRACALNE (po ostygnięciu wraca).
Jeśli ZA kolanem - NIEODWRACALNE.

TEMPERATURA: Br spada o 0,12 %/K, a iHc aż o ~0,45 %/K (przyjęte) - na gorąco kolano
przesuwa się w prawo i magnes staje się wrażliwy. Dolny wykres: cykl temperatury
60 °C -> 180 °C -> 60 °C przy 320 Arms, β = 43° (jak na rys. 12.23). Książka: przy 180 °C
widać chwilowe osłabienie, ale po powrocie do 60 °C magnes wraca w 100 % - brak
trwałego rozmagnesowania. Sprawdź, przy jakim prądzie (suwak) pojawia się trwała strata!

KOMENTARZ INŻYNIERA: magnesy V-kształtne leżą głębiej (dalej od szczeliny) -> mniejsze Ha,
dlatego wirnik V jest najodporniejszy (p. 12.2.2). Magnesy "UH" zawierają dysproz (Dy),
który podnosi iHc, ale jest drogi - dlatego walczy się o każdy stopień chłodzenia.
""",
"E10": """E10 - NAPRĘŻENIA MECHANICZNE W WIRNIKU (p. 12.4.4, tab. 12.5, rys. 12.24-12.25)

Kręcący się wirnik to jak karuzela: każda masa "chce uciec" na zewnątrz z siłą odśrodkową
      F = m * r * ω²          (rośnie z KWADRATEM prędkości!)
Biegun wirnika (żelazo + magnes na zewnątrz bariery) trzymają tylko cienkie MOSTKI
(bridges). Naprężenie w mostku:
      σ = K_t * F / (n_m * w * L)
  n_m - liczba mostków na biegun, w - szerokość mostka, L - długość pakietu,
  K_t - współczynnik koncentracji naprężeń w narożu (zaokrąglenie promieniem go zmniejsza).
Warunek: σ przy prędkości maksymalnej + 20 % zapasu < granica plastyczności blachy
(27PNF1500: 410 MPa). Reguła kciuka: przemieszczenie < 10 µm.

DYLEMAT PROJEKTANTA (prawy wykres): szerszy mostek = mniejsze naprężenie, ALE przez mostek
"ucieka" strumień magnesu (mostek nasyca się przy ~2 T i działa jak zwarcie magnetyczne).
Wąski mostek = więcej momentu, ale ryzyko pęknięcia. Wybieramy kompromis.

SPOSOBY NA NAPRĘŻENIA: 1) szerszy mostek, 2) większy promień zaokrąglenia naroża,
3) zaokrąglenia (fillety) mostków, 4) wypełnienie wnęk ŻYWICĄ (klej 3M 2214, E = 5,17 GPa)
- żywica rozkłada siłę na większą powierzchnię (rys. 12.25), 5) wycięcie zbędnej masy.
Dla porównania pokazujemy naprężenie obwodowe pełnej tarczy: σ = (3+ν)/8 * ρ ω² R².

UWAGA: to oszacowanie "na kartce"; dokładne wartości daje MES mechaniczny (rys. 12.24:
550 MPa w mostkach przy 12000 obr/min bez żywicy).
""",
"E11": """E11 - UZWOJENIE: WYPEŁNIENIE ŻŁOBKA, REZYSTANCJA, OKŁADZINA (p. 12.5.1, 12.6.1, Zad. 12.1, 12.9)

Uzwojenie fazy to wiele równoległych cienkich drutów (żył/strands) - cienkie druty łatwiej
się wkłada i mają mniejsze straty od prądów wirowych niż jeden gruby.

PRZYKŁAD Z KSIĄŻKI: 22 żyły AWG18 (d = 1,02 mm), 3 wiązki w żłobku o polu 109 mm²:
  pole miedzi = π*0,51² * 22 * 3 = 53,9 mm²   (książka podaje 54,3 mm²)
  współczynnik wypełnienia = 53,9/109 = 49,5 %  (książka: 49,8 %; tylko goła miedź - to dużo!)
  gęstość prądu przy 320 A:  J = 320 / (π*0,5²*22) = 18,5 A/mm² (książka liczy z d = 1,0 mm;
  z d = 1,02 mm wychodzi 17,8 A/mm²). Bardzo dużo - wymaga chłodzenia wodą;
  w domowej instalacji ~5 A/mm².

DŁUGOŚĆ ZWOJU (rys. 12.27):  l_b = 1,3*τp + 3*hq + 2*la    (czoło zwoju)
  l_b = 1,3*67,15 + 3*24,45 + 2*3 = 166,7 mm
  l_zwoju = 2*(l_b + L_pakietu) = 2*(166,7 + 120) = 573 mm
  l_fazy = N * l_zwoju = 24 * 0,573 = 13,8 m
  R = (0,021 Ω/m / 22) * 13,8 = 0,013 Ω   ✔ (tab. 12.3)

REZYSTANCJA A TEMPERATURA:  R(ϑ) = R20 * [1 + α(ϑ - 20)],  α = 0,0039 1/K
  przy 150 °C: 1 + 0,0039*130 = 1,51 -> rezystancja większa o 51 %! Straty w miedzi też.

ZADANIE 12.1 (okładzina prądowa): 48 żłobków, 3 przewody w żłobku, 350 A, R = 78 mm:
  A = Ns * z * I / (π * D) = 48*3*350 / (π*15,6 cm) = 1028 A/cm
  (tab. 12.3 podaje 854 A/cm dla 293 A - sprawdź suwakiem prądu!)
ZADANIE 12.9 - wybierz preset: Nph = 16, L = 150 mm, D = 160 mm, żłobek 25 x 8 mm,
drut 1,0 mm -> liczba żył dla wypełnienia 0,43-0,45, długość, R20, R140, straty przy 160 Arms.

Klasy izolacji: N = 200 °C (drut z powłoką poliamidoimidową). Próby: 1 kV przez 1 min
(< 5 mA) i rezystancja izolacji przy 500 V DC > 100 MΩ (tab. 12.7).
""",
"E12": """E12 - POMIAR STAŁEJ SEM ψm (p. 12.6.2, Ćw. 12.4, Zad. 12.4, 12.8a, 12.10)

PROCEDURA POMIARU (bez prądu w silniku!):
 a) silnik na hamowni, b) hamownia kręci wałem ze stałą prędkością,
 c) mierzymy napięcie międzyprzewodowe (5-10 okresów), d) analiza Fouriera (FFT) ->
 składowa podstawowa V_LL(1), e) ψm = V_faz,szczyt / ωe.

Dlaczego FFT? Zmierzone napięcie ma harmoniczne (górny wykres), a model dq wymaga tylko
podstawowej. Okno FFT = CAŁKOWITA liczba okresów, inaczej widmo się "rozmywa" (przeciek).

ĆWICZENIE 12.4 (8 biegunów, 1000 obr/min, V_LL(1) = 51,8 V rms):
  V_faz,szczyt = 51,8 * √2 / √3 = 42,3 V
  ωe = 1000/60 * 2π * 4 = 418,9 rad/s
  ψm = 42,3 / 418,9 = 0,101 Wb      (w widmie widać też 1,08 V i 1,78 V harmonicznych)
ZADANIE 12.4 (6 biegunów, V_LL szczyt = 100 V przy 3600 obr/min):
  V_faz = 100/√3 = 57,7 V,  ωe = 3600/60*2π*3 = 1131 rad/s,  ψm = 0,051 Wb
ZADANIE 12.8a: 67,482 V fazowe przy 300 obr/min, 12 biegunów -> ψm = 0,358 Wb.

ZADANIE 12.10 - silnik BLDC (tryb "trapez"): SEM trapezowa z płaskim wierzchołkiem 120°.
 a) Szereg Fouriera (tylko nieparzyste sinusy, a = 30° = π/6 - połowa zbocza):
       b_n = 4E/(π n² a) * sin(n a) = 24E/(π² n²) * sin(nπ/6)
    E = 100 V:  b1 = 121,6 V, b3 = 27,0 V, b5 = 4,9 V, b7 = -2,5 V ...
 b) Okres T = π/2 (kąt mechaniczny) -> na obrót przypada 4 okresy -> 4 pary = 8 biegunów.
 d) Przy 3000 obr/min: ωe = 3000/60*2π*4 = 1256,6 rad/s, ψm = b1/ωe = 0,0968 Wb.
""",
"E13": """E13 - POMIAR INDUKCYJNOŚCI Ld I Lq (p. 12.6.3, rys. 12.35-12.37, Zad. 12.7b)

POMYSŁ: w stanie ustalonym równania napięć są ALGEBRAICZNE (bez pochodnych), więc z
zmierzonych napięć i prądów da się "wyciągnąć" indukcyjności:
      Lq = (rs*id - vd) / (ωe * iq)                 (pomiar z id ≈ 0, iq ≠ 0)
      Ld = (vq - rs*iq - ωe*ψm) / (ωe * id)         (pomiar z iq ≈ 0, id < 0)
PROCEDURA: falownik w trybie regulacji prądu (id*, iq*) = stałe, hamownia kręci wałem ze
stałą prędkością (np. 500 obr/min), odczytujemy vd, vq z regulatorów, liczymy L.

ARKUSZ Z RYS. 12.35 (sprawdzenie, 500 obr/min, ωe = 209,4 rad/s):
  Lq: id = -0,1 A, iq = 48,8 A, vd = -6,5 V ->  Lq = (0,0147*(-0,1)+6,5)/(209,4*48,8) = 0,635 mH ✔
  Ld: id = -49,3 A, iq = 0,2 A, vq = 17,0 V, ψm = 0,0927 ->
      Ld = (17,0 - 0,003 - 19,41)/(209,4*(-49,3)) = 0,234 mH  ✔ (tab. 12.3)

WYKRESY: "prawdziwy" silnik (model z nasyceniem jak rys. 12.36: Ld maleje lekko z |id|,
Lq najpierw rośnie, potem maleje) i "zmierzone" punkty. Suwakami dodaj szum pomiaru napięcia
i błąd rs lub ψm - zobacz, że przy małym prądzie błąd rośnie ogromnie (dzielimy przez mały
prąd!). Dlatego pomiar robi się przy możliwie dużym prądzie i niskiej prędkości.

ZADANIE 12.7b (rs pominięte, 8 biegunów, 3000 obr/min): Vs = 86 V pod kątem 45° od osi q,
Is = 86 A opóźniony o 20° względem napięcia -> β = 45° - 20° = 25°.
  vd = -86 sin45° = -60,8 V,  vq = 86 cos45° = 60,8 V,  id = -86 sin25° = -36,3 A, iq = 77,9 A
  Lq = -vd/(ωe iq),  Ld = (vq - ωe ψm)/(ωe id);  ψm z napięcia jałowego (część a - z
  oscylogramu; suwak "V0 jałowe"). UWAGA: treść zadania jest niejednoznaczna co do kąta
  (20° może też oznaczać β) - suwak pozwala sprawdzić oba warianty.
""",
"E14": """E14 - DOŚWIADCZALNE SZUKANIE OPTYMALNYCH PRĄDÓW (p. 12.6.4, rys. 12.40-12.41, Zad. 12.5)

W prawdziwym silniku L zmienia się z prądem, więc zamiast liczyć - MIERZYMY na hamowni.
Cel: dla każdej prędkości i momentu znaleźć prąd o NAJMNIEJSZEJ amplitudzie (mniej strat).
ALGORYTM (trzy pętle, rys. 12.40):
  pętla i - prędkość ωr (hamownia trzyma stałą prędkość),
  pętla j - amplituda prądu I = 50, 100, ... A,
  pętla k - kąt β: zaczynamy od DUŻEGO β (małe napięcie - bezpiecznie!) i zmniejszamy
            o Δβ, mierząc moment, dopóki √(vd²+vq²) + ΔV <= Vdc/√3.
  Dla każdego I zapamiętujemy β z NAJWIĘKSZYM zmierzonym momentem.

GÓRNY WYKRES (jak rys. 12.41): moment w funkcji β dla kolejnych I. Linia ciągła - napięcie
OK, przerywana - zabronione. Kropka = optimum.
 - przy MAŁEJ prędkości optimum leży na szczycie krzywej = punkt MTPA,
 - przy DUŻEJ prędkości szczyt jest w obszarze zabronionym, więc optimum leży na granicy
   napięcia (przecięcie z elipsą napięcia).
DOLNY WYKRES: płaszczyzna (id, iq): okręgi prądu, elipsa napięcia (środek w -ψm/Ld),
hiperbole stałego momentu, krzywa MTPA i znalezione punkty optymalne.

ZADANIE 12.5 (preset domyślny): ψm = 0,09 Wb, rs = 0,01 Ω, Ld = 280 µH, Lq = 500 µH,
4500 obr/min, bateria 320 V, I = 50...300 A -> tabela wyników w okienku "Wyniki".
Uwaga o konwencji: w książce raz pisze się id = -I cos β, raz -I sin β; tu β zawsze od osi q.
""",
"E15": """E15 - TABLICA PRĄDÓW (LUT) I KALIBRACJA PRĘDKOŚCI OD NAPIĘCIA BATERII (tab. 12.9, rys. 12.42-12.45)

Sterownik EV nie liczy optymalnego prądu na bieżąco - odczytuje go z TABLICY (LUT)
zmierzonej na hamowni (rys. 12.43; wiersz = strumień, kolumna = % "przepustnicy" momentu).
PROBLEM: napięcie baterii zmienia się z naładowaniem (260...430 V). Czy trzeba robić
osobną tablicę dla każdego napięcia? NIE - wystarczy sprytna zamiana prędkości.

POMYSŁ: przy pominięciu rs strumień, który "mieści się" w napięciu, wynosi
      λ = Vdc / (√3 * ωe)
Zatem liczy się STOSUNEK Vdc/ωe, a nie osobno Vdc i ωe. Przy Vdc = 260 V i 3600 obr/min
strumień jest taki sam jak przy 360 V i prędkości FIKCYJNEJ n' = 3600*360/260 = 4985 obr/min.
Czytamy więc tablicę 360 V dla n' - i gotowe!

TABELA 12.9:  λmax = 360/(√3 * 2750/60*2π*4) = 0,1804 Wb   (360 V, 2750 obr/min)
              λmin = 260/(√3 * 12000/60*2π*4) = 0,0299 Wb  (260 V, 12000 obr/min)
16 poziomów strumienia co (λmax - λmin)/15; dla każdego prędkość n = Vdc/(√3 λ) (w obr/min).

PRZYKŁAD Z KSIĄŻKI: 100 Nm, 3600 obr/min, 260 V -> λ ≈ 0,100 Wb -> wiersz 8 (4956 obr/min
przy 360 V) -> odczyt (id, iq) ≈ (-140, 105) A (książka, z wykresu) - program interpoluje
dwuliniowo i daje ≈ (-157, 93) A.
ĆW. 12.5 (300 V, 4200 obr/min, 110 Nm) i ZAD. 12.6 (185 Nm, 4500 obr/min, 360 i 320 V) -
presety. W zad. 12.6c porównujemy LUT z rozwiązaniem modelowym (stałe L, ψm = 0,0927):
najmniejszy prąd dający 185 Nm przy Vs <= Vdc/√3 (romb na wykresie).

% przepustnicy = 100 * T / T_max(n'), gdzie T_max(n') - maksymalny moment z rys. 12.42.
""",
"E16": """E16 - STEROWANIE MOMENTEM Z NAPIĘCIOWYM ANTI-WINDUP (p. 12.6.5, rys. 12.47) - SYMULACJA

ŁAŃCUCH STEROWANIA W AUCIE: pedał gazu -> sterownik pojazdu VCU -> rozkaz momentu T* ->
blok rozkazów prądu (LUT z kalibracją Vdc, E15) -> (id*, iq*) -> regulatory PI prądu
z odsprzęganiem -> napięcia (vd*, vq*) -> SVPWM -> falownik -> silnik.

PROBLEM: falownik ma ograniczone napięcie (okrąg Vdc/√3). Jeśli regulator PI "chce" więcej
napięcia niż jest dostępne, napięcie zostaje obcięte, prąd przestaje nadążać za rozkazem,
a całka regulatora rośnie bez końca (WINDUP - "nakręcanie się" jak sprężyna). Skutek:
spadek i oscylacje momentu, przetężenia. Tak się dzieje, gdy tablica LUT jest trochę
niedokładna (np. ψm w rzeczywistości inne - magnes gorący/zimny) albo bateria się "ugina".

ROZWIĄZANIE (anti-windup napięciowy): liczymy |V*| = √(vd*² + vq*²) i porównujemy z Vdc/√3.
Jeśli |V*| jest za duże, integrator zmniejsza zadany strumień o Δλ -> blok LUT podaje
bardziej ujemny id* (głębsze osłabienie pola) -> napięcie wraca poniżej granicy.

SYMULACJA: prędkość rośnie liniowo (auto przyspiesza), T* = stała, w chwili t_sag bateria
spada z Vdc0 do Vdc1 (np. mocne przyspieszanie przy słabej baterii). Sterownik zna ψm
z błędem (suwak). Porównujemy DWIE symulacje: z anti-windup (kolor) i bez (szary).
Obserwuj: bez anti-windup id nie nadąża za id*, |V*| wisi na ograniczeniu, moment faluje.
Równania obiektu (całkowane metodą Heuna, krok 10 µs):
   Ld did/dt = vd - rs id + ωe Lq iq,    Lq diq/dt = vq - rs iq - ωe (Ld id + ψm)
Regulatory PI dobrane metodą kompensacji bieguna: Kp = 2π f_bw L,  Ki = 2π f_bw rs.
""",
}


# =============================================================================
# GUI - klasa bazowa zakładki
# =============================================================================

def show_table(master, title, headers, rows):
    """Okno z tabelą (jak arkusz Excel) i eksportem do CSV."""
    win = tk.Toplevel(master)
    win.title(title)
    win.configure(bg=C["bg"])
    win.geometry("1000x480")
    fr = ttk.Frame(win)
    fr.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
    tv = ttk.Treeview(fr, columns=headers, show="headings", height=18)
    for h in headers:
        tv.heading(h, text=h)
        tv.column(h, width=max(60, 9 * len(h)), anchor="e")
    for r in rows:
        tv.insert("", tk.END, values=[f"{v:.4g}" if isinstance(v, (float, np.floating)) else v for v in r])
    sb = ttk.Scrollbar(fr, orient=tk.VERTICAL, command=tv.yview)
    tv.configure(yscrollcommand=sb.set)
    tv.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    sb.pack(side=tk.RIGHT, fill=tk.Y)

    def save():
        path = filedialog.asksaveasfilename(parent=win, defaultextension=".csv",
                                            filetypes=[("CSV", "*.csv")])
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter=";")
            w.writerow(headers)
            w.writerows(rows)
        messagebox.showinfo("Zapisano", f"Zapisano {len(rows)} wierszy do\n{path}", parent=win)
    ttk.Button(win, text="Zapisz CSV", command=save).pack(pady=(0, 6))


class ExampleTab(ttk.Frame):
    """Zakładka: lewy panel (presety, suwaki, wyniki), prawy - wykresy,
    dół - objaśnienie. Podklasy definiują PARAMS, PRESETS, key, update_plot()."""
    PARAMS = []    # (nazwa, etykieta, min, max, domyślna, krok)
    PRESETS = {}   # nazwa presetu -> {parametr: wartość}
    key = ""
    has_table = False

    def __init__(self, master):
        super().__init__(master)
        self.vars = {}
        self._pending = None
        self._busy = False
        pw = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        pw.pack(fill=tk.BOTH, expand=True)
        # lewy panel przewijany (dużo suwaków)
        leftwrap = ttk.Frame(pw, width=360)
        right = ttk.Frame(pw)
        pw.add(leftwrap, weight=0)
        pw.add(right, weight=1)
        cv = tk.Canvas(leftwrap, bg=C["bg"], highlightthickness=0, width=345)
        sb = ttk.Scrollbar(leftwrap, orient=tk.VERTICAL, command=cv.yview)
        left = ttk.Frame(cv)
        left.bind("<Configure>", lambda e: cv.configure(scrollregion=cv.bbox("all")))
        win_id = cv.create_window((0, 0), window=left, anchor="nw")
        cv.bind("<Configure>", lambda e: cv.itemconfigure(win_id, width=e.width))
        cv.configure(yscrollcommand=sb.set)
        cv.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        cv.bind("<Enter>", lambda e: cv.bind_all("<MouseWheel>", lambda ev: cv.yview_scroll(int(-ev.delta / 120), "units")))
        cv.bind("<Leave>", lambda e: cv.unbind_all("<MouseWheel>"))

        if self.PRESETS:
            ttk.Label(left, text="Przykład / zadanie z książki", style="H.TLabel").pack(anchor="w", padx=6, pady=(6, 2))
            self.preset = tk.StringVar(value=list(self.PRESETS)[0])
            cb = ttk.Combobox(left, textvariable=self.preset, values=list(self.PRESETS),
                              state="readonly", width=30)
            cb.pack(fill=tk.X, padx=6)
            cb.bind("<<ComboboxSelected>>", lambda e: self.apply_preset())
        ttk.Label(left, text="Parametry", style="H.TLabel").pack(anchor="w", padx=6, pady=(6, 2))
        for name, label, lo, hi, val, res in self.PARAMS:
            fr = ttk.Frame(left)
            fr.pack(fill=tk.X, padx=6, pady=1)
            var = tk.DoubleVar(value=val)
            self.vars[name] = var
            top = ttk.Frame(fr)
            top.pack(fill=tk.X)
            ttk.Label(top, text=label).pack(side=tk.LEFT)
            vl = ttk.Label(top, text="", style="V.TLabel", width=9, anchor="e")
            vl.pack(side=tk.RIGHT)
            sc = ttk.Scale(fr, from_=lo, to=hi, variable=var, orient=tk.HORIZONTAL, length=200,
                           command=lambda _v, n=name, lo=lo, r=res: self._on_slide(n, lo, r))
            sc.pack(fill=tk.X)
            var.trace_add("write", lambda *_a, v=var, l=vl, r=res: l.config(text=self._fmt(v.get(), r)))
            vl.config(text=self._fmt(val, res))
        self.extra_controls(left)
        bfr = ttk.Frame(left)
        bfr.pack(fill=tk.X, padx=6, pady=6)
        ttk.Button(bfr, text="Przywróć", command=self.reset).pack(side=tk.LEFT, expand=True, fill=tk.X)
        if self.has_table:
            ttk.Button(bfr, text="Tabela (arkusz)", command=self.open_table).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(4, 0))
        ttk.Label(left, text="Wyniki", style="H.TLabel").pack(anchor="w", padx=6)
        self.result = tk.Text(left, height=22, width=40, font=("Consolas", 9), bg=C["mantle"],
                              fg=C["green"], insertbackground=C["text"], relief=tk.FLAT)
        self.result.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))

        vpw = ttk.PanedWindow(right, orient=tk.VERTICAL)
        vpw.pack(fill=tk.BOTH, expand=True)
        figfr = ttk.Frame(vpw)
        self.fig = Figure(figsize=(9, 6), dpi=90, tight_layout=True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=figfr)
        tb = NavigationToolbar2Tk(self.canvas, figfr)
        tb.update()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        txtfr = ttk.Frame(vpw)
        self.expl = ScrolledText(txtfr, height=12, wrap=tk.WORD, font=("Segoe UI", 10),
                                 bg=C["mantle"], fg=C["text"], relief=tk.FLAT, padx=8, pady=6)
        self.expl.pack(fill=tk.BOTH, expand=True)
        self.expl.insert("1.0", EXPL.get(self.key, ""))
        self.expl.configure(state=tk.DISABLED)
        vpw.add(figfr, weight=3)
        vpw.add(txtfr, weight=1)
        if self.PRESETS:
            self.apply_preset(refresh=False)
        self.refresh()

    # --- mechanika ---------------------------------------------------------------
    @staticmethod
    def _fmt(v, res):
        dec = max(0, -int(math.floor(math.log10(res)))) if res < 1 else 0
        return f"{v:.{dec}f}"

    def _on_slide(self, name, lo, res):
        """Przyciąganie suwaka do kroku 'res' (ttk.Scale nie ma rozdzielczości)."""
        var = self.vars[name]
        v = lo + round((var.get() - lo) / res) * res
        if abs(v - var.get()) > 1e-12:
            var.set(round(v, 10))
        self.schedule()

    def extra_controls(self, parent):
        pass

    def apply_preset(self, refresh=True):
        for k, v in self.PRESETS[self.preset.get()].items():
            if k in self.vars:
                self.vars[k].set(v)
            else:
                self.set_special(k, v)
        if refresh:
            self.refresh()

    def set_special(self, key, value):
        pass

    def reset(self):
        for name, _l, _lo, _hi, val, _r in self.PARAMS:
            self.vars[name].set(val)
        if self.PRESETS:
            self.apply_preset(refresh=False)
        self.refresh()

    def p(self, name):
        return float(self.vars[name].get())

    def schedule(self):
        if self._pending:
            self.after_cancel(self._pending)
        self._pending = self.after(150, self.refresh)

    def refresh(self):
        self._pending = None
        self.fig.clear()
        try:
            lines = self.update_plot() or []
        except Exception as exc:          # nie pozwalamy, by błąd obliczeń zamknął program
            lines = ["BŁĄD OBLICZEŃ:", repr(exc)]
        self.result.delete("1.0", tk.END)
        self.result.insert("1.0", "\n".join(lines))
        self.canvas.draw_idle()

    def update_plot(self):
        raise NotImplementedError

    def table(self):
        return [], []

    def open_table(self):
        h, rows = self.table()
        show_table(self, f"{self.key} - tabela wyników", h, rows)


def finish(ax, xlabel=None, ylabel=None, title=None, legend=True, loc="best"):
    """Wspólne formatowanie osi."""
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if legend and ax.get_legend_handles_labels()[0]:
        ax.legend(loc=loc)


def twin_legend(ax, ax2, loc="best"):
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc=loc)


def prm_from(tab, P=None):
    """Parametry silnika z suwaków (psi [Wb], rs [mΩ], Ld/Lq [µH])."""
    return dict(P=P if P is not None else int(tab.p("P")), psi=tab.p("psi"),
                rs=tab.p("rs") * 1e-3, Ld=tab.p("Ld") * 1e-6, Lq=tab.p("Lq") * 1e-6)


# =============================================================================
# ZAKŁADKI
# =============================================================================

class TabE1(ExampleTab):
    key = "E1"
    PARAMS = [("Tev", "EV: moment maks. T_max [Nm]", 100, 500, 320, 5),
              ("Pev", "EV: moc szczytowa P_max [kW]", 30, 250, 110, 1),
              ("nmax", "EV: prędkość maks. [obr/min]", 6000, 20000, 10390, 10),
              ("G", "EV: przełożenie G [-]", 4, 12, 8.19, 0.01),
              ("r", "promień koła r [m]", 0.25, 0.40, 0.316, 0.001),
              ("m", "masa auta [kg]", 1000, 2500, 1795, 5),
              ("Tice", "ICE: moment maks. [Nm]", 100, 500, 250, 5),
              ("nice", "ICE: obroty maks. [obr/min]", 4000, 8000, 6500, 50),
              ("fd", "ICE: przekładnia główna [-]", 2.5, 5.0, 3.9, 0.05)]
    PRESETS = {"Nissan Leaf 2018 (tab. 12.2)": dict(Tev=320, Pev=110, nmax=10390, G=8.19, m=1795)}

    def update_plot(self):
        Tev, Pev, nmax, G, r, m = (self.p(k) for k in ("Tev", "Pev", "nmax", "G", "r", "m"))
        Tice, nice, fd = self.p("Tice"), self.p("nice"), self.p("fd")
        eta = 0.95
        gs = self.fig.add_gridspec(2, 2)
        a1 = self.fig.add_subplot(gs[0, :])
        n_ice = np.linspace(800, nice, 200)
        for i, g in enumerate([3.0, 2.0, 1.0, 0.8, 0.7]):
            v = n_ice * 2 * PI / 60 * r / (g * fd) * 3.6
            F = ice_torque(n_ice, Tice, nice) * g * fd * 0.9 / r / 1e3
            a1.plot(v, F, lw=1.6, color=SERIES[(i + 1) % 8], label=f"ICE bieg {i + 1} ({g}:1)")
        n_ev = np.linspace(1, nmax, 400)
        v_ev = n_ev * 2 * PI / 60 * r / G * 3.6
        F_ev = ev_torque(n_ev, Tev, Pev, nmax) * G * eta / r / 1e3
        a1.plot(v_ev, F_ev, "--", color=C["cyan"], lw=2.6, label="silnik EV (1 przełożenie)")
        vv = np.linspace(0, v_ev[-1], 100)
        Fres = (m * 9.81 * 0.01 + 0.5 * 1.2 * 0.7 * (vv / 3.6) ** 2) / 1e3
        a1.plot(vv, Fres, ":", color=C["red"], label="opory ruchu (Crr=0,01, CdA=0,7 m²)")
        finish(a1, "prędkość pojazdu [km/h]", "siła na kołach [kN]",
               "Rys. 12.1: siła napędowa - silnik spalinowy z biegami vs silnik elektryczny", loc="upper right")
        # porównanie silników z tab. 12.1
        a2 = self.fig.add_subplot(gs[1, 0])
        names = list(TABLE_12_1)
        x = np.arange(len(names))
        dens = [TABLE_12_1[k][2] for k in names]
        a2.bar(x, dens, color=C["mauve"], width=0.6, label="gęstość mocy [kW/kg]")
        a2b = a2.twinx()
        a2b.plot(x, [TABLE_12_1[k][3] for k in names], "o-", color=C["peach"], label="T_max [Nm]")
        a2b.grid(False)
        a2.set_xticks(x)
        a2.set_xticklabels([k.replace(" ", "\n", 1) for k in names], fontsize=7)
        a2.set_ylabel("kW/kg")
        a2b.set_ylabel("Nm")
        a2.set_title("Tab. 12.1: silniki EV/HEV")
        twin_legend(a2, a2b, loc="upper left")
        # przyspieszanie 0-100 km/h
        a3 = self.fig.add_subplot(gs[1, 1])
        dt, v, t, ts, vs, t100 = 0.01, 0.0, 0.0, [], [], None
        while t < 25:
            n = v / r * G * 60 / (2 * PI)
            T = float(ev_torque(np.array([max(n, 1)]), Tev, Pev, nmax)[0]) if n < nmax else 0.0
            F = T * G * eta / r - m * 9.81 * 0.01 - 0.5 * 1.2 * 0.7 * v ** 2
            v = max(v + F / (1.05 * m) * dt, 0)
            t += dt
            ts.append(t); vs.append(v * 3.6)
            if t100 is None and v * 3.6 >= 100:
                t100 = t
        a3.plot(ts, vs, color=C["green"], lw=2, label="EV")
        a3.axhline(100, color=C["red"], ls=":", lw=1)
        finish(a3, "czas [s]", "v [km/h]", "Przyspieszanie (model uproszczony)")
        nb = Pev * 1e3 / Tev * 60 / (2 * PI)
        v_max_motor = nmax * 2 * PI / 60 * r / G * 3.6
        return [f"Prędkość bazowa n_b = P/T = {nb:.0f} obr/min",
                f"  (Leaf: 110 kW / 320 Nm -> 3283 obr/min)",
                f"CPSR = n_max / n_b      = {nmax / nb:.2f}",
                f"Moc przy n_b: {Tev * nb * 2 * PI / 60 / 1e3:.1f} kW",
                f"Siła rozruchowa na kołach: {Tev * G * eta / r / 1e3:.2f} kN",
                f"Prędkość przy n_max: {v_max_motor:.0f} km/h",
                f"Czas 0-100 km/h: {t100:.1f} s" if t100 else "0-100 km/h: nie osiągnięto w 25 s",
                "  (Leaf 2018 wg tab. 12.2: 8,5 s)",
                "",
                "Gęstość mocy (tab. 12.1):",
                *[f"  {k:15s} {TABLE_12_1[k][0]:4d} kW  {TABLE_12_1[k][3]} Nm  {TABLE_12_1[k][1]}"
                  for k in names]]


class TabE2(ExampleTab):
    key = "E2"
    PARAMS = [("P", "liczba biegunów P", 2, 16, 8, 2),
              ("Ns", "liczba żłobków Ns", 12, 96, 48, 6),
              ("n", "prędkość maks. [obr/min]", 1000, 20000, 12000, 100),
              ("t", "grubość blachy t [mm]", 0.10, 0.50, 0.27, 0.01),
              ("B", "indukcja w rdzeniu B [T]", 0.5, 2.0, 1.5, 0.05),
              ("m", "masa rdzenia stojana dla P=8 [kg]", 5, 30, 15, 0.5)]
    PRESETS = {"Prius / Leaf: 48 żłobków, 8 biegunów": dict(P=8, Ns=48, n=12000, t=0.27),
               "BMW i3: 72 żłobki, 12 biegunów, 0,2 mm": dict(P=12, Ns=72, n=11400, t=0.20)}

    def update_plot(self):
        P, Ns, n, t, B, m8 = int(self.p("P")), int(self.p("Ns")), self.p("n"), self.p("t"), self.p("B"), self.p("m")
        f = n / 60 * P / 2
        a1 = self.fig.add_subplot(2, 2, 1)
        nn = np.linspace(0, 20000, 100)
        for PP in (4, 8, 12, 16):
            a1.plot(nn, nn / 60 * PP / 2, lw=2.4 if PP == P else 1.0, label=f"P = {PP}")
        a1.plot([n], [f], "o", color=C["red"], ms=8)
        a1.axhline(50, color=C["sub"], ls=":", lw=1)
        finish(a1, "n [obr/min]", "f [Hz]", "Częstotliwość elektryczna f = n/60·P/2", loc="upper left")
        a2 = self.fig.add_subplot(2, 2, 2)
        ff = np.linspace(10, 1500, 200)
        for tt in (0.20, 0.27, 0.35, 0.50):
            ph, pe = core_loss_per_kg(ff, B, tt)
            a2.plot(ff, ph + pe, lw=2.4 if abs(tt - t) < 1e-3 else 1.2, label=f"t = {tt} mm")
        ph, pe = core_loss_per_kg(ff, B, t)
        a2.plot(ff, ph, ":", color=C["sub"], label="sama histereza")
        ph0, pe0 = core_loss_per_kg(f, B, t)
        a2.plot([f], [ph0 + pe0], "o", color=C["red"], ms=8)
        finish(a2, "f [Hz]", "straty [W/kg]", f"Straty w żelazie przy B = {B:.2f} T", loc="upper left")
        a3 = self.fig.add_subplot(2, 1, 2)
        Ps = np.arange(2, 18, 2)
        # rdzeń: zęby (stała masa) + jarzmo (~1/P); przyjęto 55 % jarzmo dla P = 8
        mass = m8 * (0.45 + 0.55 * 8 / Ps)
        loss = []
        for PP, mm in zip(Ps, mass):
            ph, pe = core_loss_per_kg(n / 60 * PP / 2, B, t)
            loss.append((ph + pe) * mm / 1e3)
        a3.bar(Ps, mass, color=[C["cyan"] if PP == P else C["surf2"] for PP in Ps], width=1.2, label="masa rdzenia [kg]")
        a3b = a3.twinx()
        a3b.plot(Ps, loss, "o-", color=C["peach"], label=f"straty żelaza przy {n:.0f} obr/min [kW]")
        a3b.grid(False)
        a3.set_xticks(Ps)
        a3.set_xlabel("liczba biegunów P")
        a3.set_ylabel("kg")
        a3b.set_ylabel("kW")
        a3.set_title("Kompromis: więcej biegunów = cieńsze jarzmo (lżej), ale wyższe f (większe straty)")
        twin_legend(a3, a3b, loc="upper center")
        q = Ns / (3 * P)
        tau_s = 360 / Ns
        return [f"f przy {n:.0f} obr/min  = {f:.0f} Hz",
                f"q = Ns/(3P) = {Ns}/(3·{P}) = {q:.3g}",
                f"żłobki na biegun Ns/P = {Ns / P:.3g}",
                f"zalecane warstwy barier: {Ns / P - 2:.3g} lub {Ns / P + 2:.3g}",
                f"podziałka żłobkowa: {tau_s:.2f}° mech = {tau_s * P / 2:.1f}° el",
                "",
                f"Blacha 27PNF1500 (Steinmetz):",
                f"  kh = {KH:.5f} W/(kg·Hz·T²)",
                f"  ke = {KE:.3e} W/(kg·Hz²·T²) (t=0,27)",
                f"Przy f = {f:.0f} Hz, B = {B} T, t = {t} mm:",
                f"  histereza  = {ph0:.1f} W/kg",
                f"  wirowe     = {pe0:.1f} W/kg",
                f"  razem      = {ph0 + pe0:.1f} W/kg",
                "",
                "BMW i3: 11400/60·6 = 1140 Hz",
                "-> blacha 0,2 mm zmniejsza straty",
                f"   wirowe o (0,2/0,27)² = {(0.2 / 0.27) ** 2:.2f} razy"]


class TabE3(ExampleTab):
    key = "E3"
    PARAMS = [("E140", "SEM podst. przy 3600 obr/min, 140°C [V]", 100, 200, 143.8, 0.1),
              ("T", "temperatura magnesu [°C]", 20, 200, 140, 1),
              ("alpha", "wsp. temp. Br [%/K]", -0.20, -0.05, -0.12, 0.01),
              ("n", "prędkość [obr/min]", 500, 15000, 3600, 50),
              ("nmax", "prędkość maks. [obr/min]", 6000, 16000, 12000, 100),
              ("Vdc", "napięcie baterii Vdc [V]", 200, 800, 360, 5),
              ("h3", "3. harmoniczna [%]", 0, 20, 8, 0.5),
              ("h5", "5. harmoniczna [%]", 0, 15, 3, 0.5),
              ("h7", "7. harmoniczna [%]", 0, 15, 1.5, 0.5),
              ("h11", "11. harmoniczna [%]", 0, 10, 1, 0.5),
              ("h13", "13. harmoniczna [%]", 0, 10, 0.8, 0.5)]
    P = 8

    def update_plot(self):
        E140, T, al, n, nmax, Vdc = (self.p(k) for k in ("E140", "T", "alpha", "n", "nmax", "Vdc"))
        psi140 = E140 / float(rpm_to_we(3600, self.P))
        kT = lambda TT: 1 + al / 100 * (np.asarray(TT) - 140)
        psi = psi140 * kT(T)
        we = float(rpm_to_we(n, self.P))
        H = {1: 1.0, 3: self.p("h3") / 100, 5: self.p("h5") / 100, 7: self.p("h7") / 100,
             11: self.p("h11") / 100, 13: self.p("h13") / 100}
        f = we / (2 * PI)
        t = np.linspace(0, 2 / f, 1000, endpoint=False)
        ea = sum(a * np.sin(h * we * t) for h, a in H.items()) * we * psi
        eb = sum(a * np.sin(h * (we * t - 2 * PI / 3)) for h, a in H.items()) * we * psi
        a1 = self.fig.add_subplot(2, 2, (1, 2))
        a1.plot(t * 1e3, ea, color=C["cyan"], label="SEM fazowa e_a")
        a1.plot(t * 1e3, ea - eb, color=C["peach"], label="SEM międzyprzewodowa e_ab")
        a1.plot(t * 1e3, we * psi * np.sin(we * t), ":", color=C["sub"], label="podstawowa")
        finish(a1, "t [ms]", "napięcie [V]", f"Rys. 12.10a: SEM przy {n:.0f} obr/min i {T:.0f} °C (ψm = {psi:.4f} Wb)", loc="upper right")
        a2 = self.fig.add_subplot(2, 2, 3)
        hs = np.arange(1, 20)
        ph = np.array([H.get(h, 0) for h in hs]) * we * psi
        ll = np.array([0 if h % 3 == 0 else H.get(h, 0) for h in hs]) * we * psi * SQ3
        a2.bar(hs - 0.2, ph, width=0.4, color=C["cyan"], label="fazowa")
        a2.bar(hs + 0.2, ll, width=0.4, color=C["peach"], label="międzyprzewodowa")
        a2.set_xticks(hs[::2])
        finish(a2, "rząd harmonicznej", "amplituda [V]", "Rys. 12.10b: widmo SEM")
        a3 = self.fig.add_subplot(2, 2, 4)
        TT = np.linspace(20, 200, 100)
        a3.plot(TT, psi140 * kT(TT), color=C["green"], label="ψm(T) [Wb]")
        a3.plot([T], [psi], "o", color=C["red"])
        a3.set_ylabel("ψm [Wb]")
        a3b = a3.twinx()
        a3b.grid(False)
        Emax = float(rpm_to_we(nmax, self.P)) * psi140 * kT(TT)
        a3b.plot(TT, Emax, color=C["mauve"], label=f"SEM przy {nmax:.0f} obr/min [V]")
        a3b.axhline(Vdc / SQ3, color=C["red"], ls="--", label="Vdc/√3")
        a3b.set_ylabel("V")
        a3.set_xlabel("temperatura magnesu [°C]")
        a3.set_title("Magnes słabnie z temperaturą")
        twin_legend(a3, a3b, loc="upper right")
        E_nmax = float(rpm_to_we(nmax, self.P)) * psi
        E_nmax20 = float(rpm_to_we(nmax, self.P)) * psi140 * kT(20)
        return [f"ωe(3600 obr/min) = 60·4·2π = {rpm_to_we(3600, 8):.1f} rad/s",
                f"ψm(140°C) = {E140}/{rpm_to_we(3600, 8):.1f} = {psi140:.4f} Wb",
                f"ψm({T:.0f}°C) = {psi:.4f} Wb",
                f"ψm(20°C)  = {psi140 * kT(20):.4f} Wb",
                "",
                f"SEM przy {n:.0f} obr/min: {we * psi:.1f} V szczyt",
                f"SEM przy n_max ({T:.0f}°C): {E_nmax:.0f} V",
                f"SEM przy n_max (20°C): {E_nmax20:.0f} V",
                f"Vdc/√3 = {Vdc / SQ3:.1f} V",
                ("UWAGA: zimny magnes przy n_max daje SEM > Vdc/√3" if E_nmax20 > Vdc / SQ3 else "SEM < Vdc/√3 - bezpiecznie"),
                "  -> po utracie sterowania falownik prostuje",
                "     napięcie i ładuje baterię (ryzyko dla IGBT)",
                "",
                f"THD SEM fazowej = {100 * math.sqrt(sum(a * a for h, a in H.items() if h > 1)):.1f} %",
                f"THD SEM międzyprzew. = {100 * math.sqrt(sum(a * a for h, a in H.items() if h > 1 and h % 3)):.1f} %"]


class TabE4(ExampleTab):
    key = "E4"
    PARAMS = [("n", "prędkość n [obr/min]", 0, 14000, 2500, 50),
              ("Is", "amplituda prądu Is [A]", 0, 600, 453, 1),
              ("beta", "kąt prądu β od osi q [°]", 0, 90, 43, 0.25),
              ("psi", "ψm [Wb]", 0.05, 0.15, 0.0984, 0.0001),
              ("rs", "rs [mΩ]", 0, 30, 14.7, 0.1),
              ("Ld", "Ld [µH]", 100, 600, 234, 1),
              ("Lq", "Lq [µH]", 200, 1200, 562, 1),
              ("Vdc", "Vdc [V]", 200, 800, 360, 5),
              ("dV", "zapas napięcia ΔV [V]", 0, 20, 0, 0.1),
              ("P", "liczba biegunów P", 4, 16, 8, 2)]
    PRESETS = {"Ćw. 12.1 a) 2500 obr/min, (453 A, 43°)": dict(n=2500, Is=453, beta=43, psi=0.0984, rs=14.7, Ld=234, Lq=562, dV=0),
               "Ćw. 12.1 b) 7200 obr/min, (453 A, 75,25°)": dict(n=7200, Is=453, beta=75.25, psi=0.0984, rs=14.7, Ld=234, Lq=562, dV=0),
               "Ćw. 12.1 c) 12000 obr/min, (453 A, 80,75°)": dict(n=12000, Is=453, beta=80.75, psi=0.0984, rs=14.7, Ld=234, Lq=562, dV=0),
               "Zad. 12.2: 10000 obr/min, (350 A, 55°), limit 202 V": dict(n=10000, Is=350, beta=55, psi=0.0927, rs=13, Ld=234, Lq=562, dV=5.9)}

    def update_plot(self):
        prm = prm_from(self)
        n, Is, beta, Vdc, dV = self.p("n"), self.p("Is"), self.p("beta"), self.p("Vdc"), self.p("dV")
        Vmax = Vdc / SQ3
        Vlim = Vmax - dV
        op = operating_point(n, Is, beta, prm)
        we = op["we"]
        a1 = self.fig.add_subplot(1, 2, 1)
        th = np.linspace(0, 2 * PI, 300)
        a1.plot(Vmax * np.cos(th), Vmax * np.sin(th), color=C["red"], lw=1.4, label=f"Vdc/√3 = {Vmax:.1f} V")
        if dV > 0:
            a1.plot(Vlim * np.cos(th), Vlim * np.sin(th), ":", color=C["red"], lw=1, label=f"Vdc/√3 - ΔV = {Vlim:.1f} V")
        # łańcuch napięć: SEM -> ωe·Ld·id -> -ωe·Lq·iq -> rs·I
        pts = [(0, 0), (0, we * prm["psi"]), (0, we * (prm["psi"] + prm["Ld"] * op["id"])),
               (-we * prm["Lq"] * op["iq"], we * (prm["psi"] + prm["Ld"] * op["id"])), (op["vd"], op["vq"])]
        labs = ["ωe·ψm (SEM)", "ωe·Ld·id", "-ωe·Lq·iq", "rs·I"]
        cols = [C["mauve"], C["teal"], C["yellow"], C["pink"]]
        for (x0, y0), (x1, y1), lab, col in zip(pts[:-1], pts[1:], labs, cols):
            if abs(x1 - x0) + abs(y1 - y0) > 1e-6:
                a1.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", color=col, lw=1.8))
                a1.plot([], [], color=col, label=lab)
        a1.annotate("", xy=(op["vd"], op["vq"]), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=C["cyan"], lw=2.6))
        a1.plot([], [], color=C["cyan"], lw=2.6, label=f"Vs = {op['Vs']:.1f} V")
        k = 0.45 * Vmax / max(Is, 1)
        a1.annotate("", xy=(op["id"] * k, op["iq"] * k), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=C["green"], lw=2.6))
        a1.plot([], [], color=C["green"], lw=2.6, label=f"Is = {Is:.0f} A (skala)")
        a1.axhline(0, color=C["sub"], lw=0.6)
        a1.axvline(0, color=C["sub"], lw=0.6)
        R = max(Vmax, op["Vs"], we * prm["psi"]) * 1.12
        a1.set_xlim(-R, R * 0.6)
        a1.set_ylim(-R * 0.6, R)
        a1.set_aspect("equal", adjustable="box")
        finish(a1, "oś d [V]", "oś q [V]", f"Rys. 12.12: wektory przy {n:.0f} obr/min", loc="lower left")
        a2 = self.fig.add_subplot(1, 2, 2)
        bb = np.linspace(0, 90, 1801)
        idb, iqb = dq_from_beta(Is, bb)
        _, _, Vsb = voltages(idb, iqb, we, prm["rs"], prm["psi"], prm["Ld"], prm["Lq"])
        Tb = torque(idb, iqb, prm["P"], prm["psi"], prm["Ld"], prm["Lq"])
        a2.plot(bb, Vsb, color=C["cyan"], label="Vs(β) [V]")
        a2.axhline(Vlim, color=C["red"], ls="--", label="granica napięcia")
        a2.axvline(beta, color=C["sub"], ls=":")
        a2b = a2.twinx()
        a2b.grid(False)
        a2b.plot(bb, Tb, color=C["green"], label="T(β) [Nm]")
        ok = Vsb <= Vlim
        bmin = bb[np.argmax(ok)] if ok.any() else None
        if bmin is not None:
            a2.axvline(bmin, color=C["peach"], lw=1.4, label=f"β_min = {bmin:.2f}°")
        a2.set_ylabel("Vs [V]")
        a2b.set_ylabel("T [Nm]")
        a2.set_xlabel("β [°]")
        a2.set_title(f"Napięcie i moment vs β (Is = {Is:.0f} A)")
        twin_legend(a2, a2b, loc="center left")
        ok_now = op["Vs"] <= Vlim
        lines = [f"ωe = {n:.0f}/60·2π·{prm['P'] // 2} = {we:.1f} rad/s",
                 f"SEM = ωe·ψm = {we * prm['psi']:.1f} V",
                 f"id = -Is·sinβ = {op['id']:.1f} A",
                 f"iq =  Is·cosβ = {op['iq']:.1f} A",
                 f"rs·id = {prm['rs'] * op['id']:.2f} V,  rs·iq = {prm['rs'] * op['iq']:.2f} V",
                 f"vd = {op['vd']:.1f} V",
                 f"vq = {op['vq']:.1f} V",
                 f"Vs = {op['Vs']:.1f} V  (limit {Vlim:.1f} V) {'OK' if ok_now else 'ZA DUŻO!'}",
                 f"δ (kąt napięcia od q) = {op['delta']:.1f}°",
                 f"PF = cos(β-δ) = {op['pf']:.3f}",
                 f"T = {op['T']:.1f} Nm",
                 f"P = {op['P'] / 1e3:.1f} kW",
                 ""]
        if bmin is not None:
            opm = operating_point(n, Is, bmin, prm)
            lines += [f"Najmniejsze β spełniające limit:",
                      f"  β_min = {bmin:.2f}°  -> T = {opm['T']:.1f} Nm",
                      f"  PF = {opm['pf']:.3f}, P = {opm['P'] / 1e3:.1f} kW"]
        else:
            lines += ["Żadne β nie spełnia limitu napięcia", " -> trzeba zmniejszyć Is (obszar MTPV)"]
        return lines


class TabE5(ExampleTab):
    key = "E5"
    has_table = True
    PARAMS = [("psi", "ψm [Wb]", 0.05, 0.15, 0.0984, 0.0001),
              ("rs", "rs [mΩ]", 0, 30, 14.7, 0.1),
              ("Ld", "Ld [µH]", 100, 600, 234, 1),
              ("Lq", "Lq [µH]", 200, 1200, 562, 1),
              ("Imax", "prąd maks. Is [A szczyt]", 50, 600, 453, 1),
              ("Vdc", "Vdc [V]", 200, 800, 360, 5),
              ("dV", "zapas napięcia ΔV [V]", 0, 20, 5.9, 0.1),
              ("P", "liczba biegunów P", 4, 16, 8, 2),
              ("nmax", "prędkość maks. [obr/min]", 2000, 16000, 12000, 100)]
    PRESETS = {"Rys. 12.16 (tab. 12.3, Is = 453 A)": dict(psi=0.0984, rs=14.7, Ld=234, Lq=562, Imax=453, Vdc=360, dV=5.9, P=8, nmax=12000),
               "Ćw. 12.2 (ψm = 0,103, Is = 280 A)": dict(psi=0.103, rs=14.7, Ld=234, Lq=562, Imax=280, Vdc=360, dV=6, P=8, nmax=10000),
               "Ćw. 12.3 (Is = 212 A, 325 V, do 8000)": dict(psi=0.0984, rs=15, Ld=234, Lq=562, Imax=212, Vdc=325, dV=0, P=8, nmax=8000),
               "Zad. 12.3 (ψm = 0,09, 280/500 µH, 325 V)": dict(psi=0.09, rs=10, Ld=280, Lq=500, Imax=280, Vdc=325, dV=6, P=8, nmax=10000)}

    def compute(self, step=None):
        prm = prm_from(self)
        nmax = self.p("nmax")
        speeds = np.arange(step, nmax + 1, step) if step else np.linspace(100, nmax, 60)
        Vlim = self.p("Vdc") / SQ3 - self.p("dV")
        return prm, max_torque_envelope(speeds, prm, self.p("Imax"), Vlim), Vlim

    def update_plot(self):
        prm, e, Vlim = self.compute()
        n = e["n"]
        a1 = self.fig.add_subplot(3, 1, 1)
        a1.plot(n, e["T"], color=C["cyan"], lw=2, label="moment T [Nm]")
        a1b = a1.twinx()
        a1b.grid(False)
        a1b.plot(n, e["P"] / 1e3, color=C["peach"], lw=2, label="moc P [kW]")
        a1.set_ylabel("Nm")
        a1b.set_ylabel("kW")
        a1.set_title("Rys. 12.16b: maksymalny moment i moc")
        twin_legend(a1, a1b, loc="center right")
        a2 = self.fig.add_subplot(3, 1, 2, sharex=a1)
        a2.plot(n, e["beta"], color=C["green"], lw=2, label="kąt prądu β [°]")
        a2b = a2.twinx()
        a2b.grid(False)
        a2b.plot(n, e["Is"], color=C["mauve"], lw=2, label="prąd Is [A]")
        a2b.set_ylim(0, 1.1 * self.p("Imax"))
        a2.set_ylim(0, 90)
        a2.set_ylabel("°")
        a2b.set_ylabel("A")
        twin_legend(a2, a2b, loc="center right")
        a3 = self.fig.add_subplot(3, 1, 3, sharex=a1)
        a3.plot(n, e["Vs"], color=C["cyan"], lw=2, label="Vs [V]")
        a3.axhline(Vlim, color=C["red"], ls="--", label="Vdc/√3 - ΔV")
        a3b = a3.twinx()
        a3b.grid(False)
        a3b.plot(n, e["pf"], color=C["yellow"], lw=2, label="PF")
        a3b.set_ylim(0, 1.05)
        a3.set_ylabel("V")
        a3.set_xlabel("prędkość [obr/min]")
        twin_legend(a3, a3b, loc="center right")
        # prędkość bazowa: pierwszy punkt, gdzie napięcie dochodzi do granicy
        hit = e["Vs"] >= Vlim - 1.0
        nb = n[np.argmax(hit)] if hit.any() else n[-1]
        fw = n >= nb
        return [f"Vlim = Vdc/√3 - ΔV = {Vlim:.1f} V",
                f"Moment rozruchowy (MTPA): {e['T'][0]:.0f} Nm",
                f"  β_MTPA = {e['beta'][0]:.1f}°",
                f"Koniec stałego momentu: ≈ {nb:.0f} obr/min",
                f"Moc maks.: {e['P'].max() / 1e3:.1f} kW",
                f"Moc przy n_max: {e['P'][-1] / 1e3:.1f} kW",
                f"Moc min. w osłabianiu pola: {e['P'][fw].min() / 1e3:.1f} kW",
                f"T przy n_max: {e['T'][-1]:.0f} Nm, β = {e['beta'][-1]:.1f}°",
                f"Is przy n_max: {e['Is'][-1]:.0f} A",
                "",
                "Punkt ψm/Ld (środek elipsy) = "
                f"{prm['psi'] / prm['Ld']:.0f} A",
                ("  < Imax -> zakres prędkości teoretycznie ∞" if prm['psi'] / prm['Ld'] < self.p("Imax")
                 else "  > Imax -> prędkość ograniczona"),
                "",
                "Przycisk 'Tabela (arkusz)' = kolumny",
                "jak w Excelu z rys. 12.16a (co 500 obr/min)."]

    def table(self):
        prm, e, Vlim = self.compute(step=500)
        rows = []
        for i in range(len(e["n"])):
            n = e["n"][i]
            if not np.isfinite(e["Vs"][i]):
                continue
            op = operating_point(n, e["Is"][i], e["beta"][i], prm)
            rows.append([int(n), round(op["we"], 1), round(e["Is"][i], 1), round(e["beta"][i], 2),
                         round(op["id"], 1), round(op["iq"], 1), round(op["T"], 1), round(op["P"] / 1e3, 2),
                         round(op["vd"], 1), round(op["vq"], 1), round(op["Vs"], 1), round(op["pf"], 3)])
        return (["n [rpm]", "ωe [rad/s]", "Is [A]", "β [°]", "id [A]", "iq [A]", "T [Nm]", "P [kW]",
                 "vd [V]", "vq [V]", "Vs [V]", "PF"], rows)


class TabE6(ExampleTab):
    key = "E6"
    has_table = True
    PARAMS = [("Is", "prąd Is [A szczyt]", 100, 600, 420, 5),
              ("Vdc", "napięcie baterii [V]", 300, 800, 600, 5),
              ("E", "zmierzona SEM fazowa [V]", 30, 120, 67.482, 0.001),
              ("nE", "przy prędkości [obr/min]", 100, 1000, 300, 10),
              ("P", "liczba biegunów P", 4, 16, 12, 2),
              ("kL", "skala Lq (nasycenie) [-]", 0.7, 1.3, 1.0, 0.01),
              ("dV", "zapas napięcia ΔV [V]", 0, 20, 0, 0.5),
              ("nmax", "prędkość maks. [obr/min]", 1000, 5000, 3500, 50)]
    SPEEDS = [300, 600, 900, 1000, 1180, 1300, 1500, 1800, 2300, 3200]

    def L_of_beta(self, beta):
        b = np.clip(beta, 30, 80)
        return (np.interp(b, TAB_12_8[:, 0], TAB_12_8[:, 1]) * 1e-3,
                np.interp(b, TAB_12_8[:, 0], TAB_12_8[:, 2]) * 1e-3 * self.p("kL"))

    def compute(self, speeds):
        P = int(self.p("P"))
        psi = self.p("E") / float(rpm_to_we(self.p("nE"), P))
        prm = dict(P=P, psi=psi, rs=0.0, Ld=0.72e-3, Lq=1.72e-3 * self.p("kL"))
        Vlim = self.p("Vdc") / SQ3 - self.p("dV")
        Is = np.linspace(self.p("Is") / 60, self.p("Is"), 60)[:, None]
        beta = np.linspace(30, 80, 201)[None, :]
        id_, iq = dq_from_beta(Is, beta)
        Ld, Lq = self.L_of_beta(beta)
        Te = torque(id_, iq, P, psi, Ld, Lq)
        res = []
        for n in speeds:
            we = float(rpm_to_we(n, P))
            _, _, Vs = voltages(id_, iq, we, 0.0, psi, Ld, Lq)
            Tm = np.where(Vs <= Vlim, Te, -np.inf)
            k = np.unravel_index(np.argmax(Tm), Tm.shape)
            if np.isfinite(Tm[k]):
                res.append((n, Te[k], Te[k] * we / (P / 2) / 1e3, beta[0, k[1]], Is[k[0], 0], Vs[k]))
            else:
                res.append((n, 0, 0, np.nan, 0, np.nan))
        return prm, np.array(res, dtype=float)

    def update_plot(self):
        P, Is = int(self.p("P")), self.p("Is")
        prm, _ = self.compute([])
        psi = prm["psi"]
        a1 = self.fig.add_subplot(2, 1, 1)
        bb = np.linspace(30, 80, 201)
        idb, iqb = dq_from_beta(Is, bb)
        Ld, Lq = self.L_of_beta(bb)
        Tv = torque(idb, iqb, P, psi, Ld, Lq)
        Tc = torque(idb, iqb, P, psi, 0.72e-3, 1.72e-3 * self.p("kL"))
        a1.plot(bb, Tv, color=C["cyan"], lw=2, label="L(β) z tabeli (nasycenie)")
        a1.plot(bb, Tc, "--", color=C["sub"], label="L stałe (wartości przy 45°)")
        idt, iqt = dq_from_beta(Is, TAB_12_8[:, 0])
        a1.plot(TAB_12_8[:, 0], torque(idt, iqt, P, psi, TAB_12_8[:, 1] * 1e-3, TAB_12_8[:, 2] * 1e-3 * self.p("kL")),
                "o", color=C["peach"], label="punkty tabeli")
        j = int(np.argmax(Tv))
        a1.plot(bb[j], Tv[j], "*", color=C["red"], ms=14, label=f"max: β = {bb[j]:.1f}°, T = {Tv[j]:.0f} Nm")
        finish(a1, "β [°]", "T [Nm]", f"Zad. 12.8b: moment vs kąt prądu (Is = {Is:.0f} A, ψm = {psi:.3f} Wb)", loc="lower center")
        a2 = self.fig.add_subplot(2, 1, 2)
        nn = np.linspace(100, self.p("nmax"), 60)
        _, cont = self.compute(nn)
        _, pts = self.compute(self.SPEEDS)
        env_c = max_torque_envelope(nn, prm, Is, self.p("Vdc") / SQ3 - self.p("dV"))
        a2.plot(cont[:, 0], cont[:, 1], color=C["cyan"], lw=2, label="T, L(β)")
        a2.plot(env_c["n"], env_c["T"], "--", color=C["sub"], label="T, L stałe")
        a2.plot(pts[:, 0], pts[:, 1], "o", color=C["cyan"])
        a2b = a2.twinx()
        a2b.grid(False)
        a2b.plot(cont[:, 0], cont[:, 2], color=C["peach"], lw=2, label="P [kW], L(β)")
        a2b.plot(pts[:, 0], pts[:, 2], "o", color=C["peach"])
        a2.set_xlabel("n [obr/min]")
        a2.set_ylabel("T [Nm]")
        a2b.set_ylabel("P [kW]")
        a2.set_title(f"Zad. 12.8c: moment i moc vs prędkość (Vmax = {self.p('Vdc') / SQ3:.0f} V)")
        twin_legend(a2, a2b, loc="center right")
        lines = [f"a) ωe = {self.p('nE'):.0f}/60·2π·{P // 2} = {rpm_to_we(self.p('nE'), P):.1f} rad/s",
                 f"   ψm = {self.p('E'):.3f}/{rpm_to_we(self.p('nE'), P):.1f} = {psi:.4f} Wb",
                 f"b) β_opt = {bb[j]:.1f}°, T_max = {Tv[j]:.0f} Nm",
                 f"   (L stałe dałyby max {Tc.max():.0f} Nm przy {bb[np.argmax(Tc)]:.1f}°)",
                 "c)  n     T[Nm]  P[kW]   β[°]   Is[A]"]
        for r in pts:
            lines.append(f"  {r[0]:5.0f} {r[1]:7.0f} {r[2]:6.1f} {r[3]:6.1f} {r[4]:6.0f}")
        return lines

    def table(self):
        _, pts = self.compute(self.SPEEDS)
        return (["n [rpm]", "T [Nm]", "P [kW]", "β [°]", "Is [A]", "Vs [V]"],
                [[int(r[0]), round(r[1], 1), round(r[2], 2), round(r[3], 1), round(r[4], 1), round(r[5], 1)] for r in pts])


class TabE7(ExampleTab):
    key = "E7"
    PARAMS = [("T", "moment średni [Nm]", 50, 400, 320, 5),
              ("r6", "amplituda 6. harm. [% T]", 0, 20, 9, 0.5),
              ("r12", "amplituda 12. harm. [% T]", 0, 20, 6, 0.5),
              ("r18", "amplituda 18. harm. [% T]", 0, 10, 1.5, 0.5),
              ("ph12", "faza 12. harm. [°]", 0, 360, 0, 5),
              ("skew", "całkowity kąt skosu [° el]", 0, 60, 15, 0.5),
              ("N", "liczba segmentów (1 = brak)", 1, 8, 4, 1),
              ("P", "liczba biegunów P", 4, 16, 8, 2),
              ("Ns", "liczba żłobków Ns", 24, 96, 48, 6)]

    @staticmethod
    def kskew(h, sk_deg, N):
        x = np.radians(sk_deg) * h
        if N <= 1:
            return np.ones_like(np.asarray(x, dtype=float))
        with np.errstate(divide="ignore", invalid="ignore"):
            k = np.sin(x / 2) / (N * np.sin(x / (2 * N)))
        return np.where(np.abs(x) < 1e-9, 1.0, k)

    def update_plot(self):
        T0, sk, N = self.p("T"), self.p("skew"), int(self.p("N"))
        comps = [(6, self.p("r6") / 100, 0.0), (12, self.p("r12") / 100, np.radians(self.p("ph12"))),
                 (18, self.p("r18") / 100, 0.3)]
        th = np.linspace(0, 2 * PI, 1441)
        before = T0 * (1 + sum(a * np.cos(h * th + ph) for h, a, ph in comps))
        k1 = float(self.kskew(1, sk, N))
        after = T0 * (k1 + sum(float(self.kskew(h, sk, N)) * a * np.cos(h * th + ph) for h, a, ph in comps))
        rip = lambda y: (y.max() - y.min()) / y.mean() * 100
        a1 = self.fig.add_subplot(2, 1, 1)
        a1.plot(np.degrees(th), before, color=C["sub"], lw=1.4, label=f"bez skosu: tętnienia {rip(before):.1f} %")
        a1.plot(np.degrees(th), after, color=C["cyan"], lw=2, label=f"ze skosem {sk:.1f}° el, N = {N}: {rip(after):.1f} %")
        a1.set_xlim(0, 360)
        finish(a1, "kąt elektryczny [°]", "moment [Nm]", "Rys. 12.20: moment w czasie obrotu - przed i po skosie", loc="lower right")
        a2 = self.fig.add_subplot(2, 1, 2)
        ss = np.linspace(0, 60, 400)
        for i, h in enumerate((6, 12, 18, 24)):
            a2.plot(ss, np.abs(self.kskew(h, ss, N)), color=SERIES[i], lw=1.8, label=f"|k_{h}| (N = {N})")
            a2.plot(ss, np.abs(self.kskew(h, ss, 400)), ":", color=SERIES[i], lw=1)
        a2.axvline(sk, color=C["red"], ls="--", lw=1)
        a2.plot([], [], ":", color=C["sub"], label="skos ciągły")
        finish(a2, "kąt skosu [° el]", "współczynnik skosu |k_h|", "Jak bardzo skos tłumi daną harmoniczną", loc="upper right")
        P, Ns = int(self.p("P")), int(self.p("Ns"))
        slot_m = 360 / Ns
        return [f"Tętnienia przed skosem: {rip(before):.1f} %",
                f"Tętnienia po skosie:    {rip(after):.1f} %",
                "  (książka: 28,5 % -> 10 %)",
                f"k1 (strata momentu śr.) = {k1:.4f}",
                f"  -> moment śr. {T0 * k1:.1f} Nm ({100 * (1 - k1):.2f} % mniej)",
                f"k6  = {float(self.kskew(6, sk, N)):.3f}",
                f"k12 = {float(self.kskew(12, sk, N)):.3f}",
                f"k18 = {float(self.kskew(18, sk, N)):.3f}",
                "",
                f"Podziałka żłobkowa: 360/{Ns} = {slot_m:.2f}° mech",
                f"  = {slot_m * P / 2:.1f}° el (× P/2 = {P // 2})",
                f"Pół podziałki = {slot_m / 2:.3f}° mech = {slot_m * P / 4:.1f}° el",
                f"Skos {sk:.1f}° el = {sk / (P / 2):.2f}° mech",
                f"Segment obrócony co {sk / max(N, 1):.2f}° el",
                "",
                "Całkowite usunięcie harmonicznej h:",
                "  skos ciągły 360°/h el: h=6 -> 60°, h=12 -> 30°"]


class TabE8(ExampleTab):
    key = "E8"
    PARAMS = [("Imax", "prąd maks. [A szczyt] (293 Arms)", 100, 600, 414, 1),
              ("Vdc", "Vdc [V]", 200, 800, 360, 5),
              ("dV", "zapas ΔV [V]", 0, 20, 6, 0.5),
              ("rs", "rs (gorące uzwojenie) [mΩ]", 5, 30, 15, 0.1),
              ("m", "masa rdzenia stojana [kg]", 5, 30, 15, 0.5),
              ("B0", "indukcja w zębach przy λ = ψm [T]", 0.8, 2.0, 1.6, 0.05),
              ("kadd", "wsp. strat dodatkowych [-]", 1, 8, 3, 0.1),
              ("t", "grubość blachy [mm]", 0.1, 0.5, 0.27, 0.01),
              ("Pm", "straty mech. przy 12000 obr/min [kW]", 0, 2, 0.5, 0.05),
              ("nmax", "prędkość maks. [obr/min]", 4000, 16000, 12000, 100)]

    def losses(self, id_, iq, n, lam):
        """Straty [W]: miedź, histereza, prądy wirowe, mechaniczne."""
        f = n / 60 * 4
        B = np.minimum(self.p("B0") * lam / MOTOR_12_3["psi"], 1.9)   # zęby nasycają się ~1,9 T
        ph, pe = core_loss_per_kg(f, B, self.p("t"))
        k = self.p("m") * self.p("kadd")
        pcu = 1.5 * self.p("rs") * 1e-3 * (id_ ** 2 + iq ** 2)
        pm = self.p("Pm") * 1e3 * (0.3 * n / 12000 + 0.7 * (n / 12000) ** 2)
        return pcu, ph * k, pe * k, pm

    def update_plot(self):
        prm = dict(MOTOR_12_3)
        prm["rs"] = self.p("rs") * 1e-3
        Imax, nmax = self.p("Imax"), self.p("nmax")
        Vlim = self.p("Vdc") / SQ3 - self.p("dV")
        nn = np.linspace(500, nmax, 40)
        e = max_torque_envelope(nn, prm, Imax, Vlim)
        lam = np.hypot(prm["psi"] + prm["Ld"] * e["id"], prm["Lq"] * e["iq"])
        pcu, ph, pe, pm = self.losses(e["id"], e["iq"], nn, lam)
        a1 = self.fig.add_subplot(2, 1, 1)
        a1.plot(nn, e["T"], color=C["cyan"], lw=2, label="moment maks. [Nm]")
        a1.plot(nn, e["Is"] / math.sqrt(2), color=C["mauve"], lw=2, label="prąd [Arms]")
        a1.set_ylabel("Nm, Arms")
        a1b = a1.twinx()
        a1b.grid(False)
        a1b.stackplot(nn, pcu / 1e3, ph / 1e3, pe / 1e3, pm / 1e3, alpha=0.45,
                      colors=[C["peach"], C["green"], C["yellow"], C["sub"]],
                      labels=["miedź", "histereza", "prądy wirowe", "mechaniczne"])
        a1b.set_ylabel("straty [kW]")
        a1.set_title("Rys. 12.22: straty wzdłuż krzywej maksymalnego momentu")
        twin_legend(a1, a1b, loc="lower left")
        a1.set_xlabel("n [obr/min]")
        # mapa sprawności - minimalne straty dla każdego (n, T)
        a2 = self.fig.add_subplot(2, 1, 2)
        g = np.linspace(-Imax, 0, 61)
        ID, IQ = np.meshgrid(g, np.linspace(0, Imax, 61))
        inside = ID ** 2 + IQ ** 2 <= Imax ** 2
        ID, IQ = ID[inside], IQ[inside]
        TT = torque(ID, IQ, prm["P"], prm["psi"], prm["Ld"], prm["Lq"])
        LAM = np.hypot(prm["psi"] + prm["Ld"] * ID, prm["Lq"] * IQ)
        Tmax = e["T"].max()
        tl = np.linspace(Tmax / 25, Tmax, 25)
        ns = np.linspace(300, nmax, 32)
        eta = np.full((len(tl), len(ns)), np.nan)
        tol = Tmax / 40
        for j, n in enumerate(ns):
            we = float(rpm_to_we(n, prm["P"]))
            _, _, Vs = voltages(ID, IQ, we, prm["rs"], prm["psi"], prm["Ld"], prm["Lq"])
            ok = Vs <= Vlim
            c, h_, ed, m_ = self.losses(ID, IQ, n, LAM)
            L = c + h_ + ed + m_
            for i, T in enumerate(tl):
                sel = ok & (np.abs(TT - T) < tol)
                if sel.any():
                    Pout = T * n * 2 * PI / 60
                    eta[i, j] = 100 * Pout / (Pout + L[sel].min())
        cf = a2.contourf(ns, tl, eta, levels=[70, 80, 85, 88, 90, 92, 94, 95, 96, 97, 98], cmap="viridis")
        cs = a2.contour(ns, tl, eta, levels=[90, 94, 96, 97], colors=C["text"], linewidths=0.6)
        a2.clabel(cs, fmt="%d%%", fontsize=7)
        a2.plot(nn, e["T"], color=C["red"], lw=1.6, label="obwiednia momentu")
        self.fig.colorbar(cf, ax=a2, label="η [%]")
        k = np.unravel_index(np.nanargmax(eta), eta.shape)
        a2.plot(ns[k[1]], tl[k[0]], "*", color=C["red"], ms=12)
        finish(a2, "n [obr/min]", "T [Nm]", "Rys. 12.46: mapa sprawności (sterowanie minimalizujące straty)", loc="upper right")
        i12 = -1
        return [f"Przy {nn[0]:.0f} obr/min (maks. moment):",
                f"  Cu = {pcu[0]:.0f} W, histereza = {ph[0]:.0f} W",
                f"  wirowe = {pe[0]:.0f} W",
                f"Przy {nn[i12]:.0f} obr/min ({nn[i12] / 60 * 4:.0f} Hz):",
                f"  Cu = {pcu[i12]:.0f} W, histereza = {ph[i12]:.0f} W",
                f"  wirowe = {pe[i12]:.0f} W, mech = {pm[i12]:.0f} W",
                f"  |λ| = {lam[i12]:.4f} Wb (ψm = {prm['psi']} Wb)",
                "",
                f"Maks. sprawność: {np.nanmax(eta):.1f} %",
                f"  przy n = {ns[k[1]]:.0f} obr/min, T = {tl[k[0]]:.0f} Nm",
                "  (książka, pomiar: 97,5 %)",
                "",
                "Najniższa sprawność: duży T / mała n",
                "(miedź) oraz mały T / duża n (żelazo,",
                "mechanika) - jak w książce."]


class TabE9(ExampleTab):
    key = "E9"
    PARAMS = [("T", "temperatura magnesu [°C]", 20, 220, 180, 1),
              ("I", "prąd [Arms]", 0, 500, 320, 5),
              ("beta", "kąt prądu β [°]", 0, 90, 43, 1),
              ("Pc", "wsp. permeancji Pc [-]", 1, 10, 4, 0.1),
              ("kar", "pole od prądu [kA/m na 100 A (-id)]", 10, 150, 50, 1),
              ("aH", "wsp. temp. iHc [%/K]", -0.7, -0.3, -0.45, 0.01),
              ("Br", "Br przy 20°C [T]", 1.22, 1.28, 1.25, 0.01),
              ("Tpk", "szczyt cyklu temperatury [°C]", 100, 220, 180, 1)]

    def op(self, T, I=None):
        I = self.p("I") if I is None else I
        id_ = -math.sqrt(2) * I * math.sin(math.radians(self.p("beta")))
        Ha = self.p("kar") * 1e3 * (-id_) / 100
        kw = dict(Br20=self.p("Br"), aHcj=self.p("aH"))
        return magnet_operating_point(T, self.p("Pc"), Ha, **kw), Ha, kw

    def update_plot(self):
        T = self.p("T")
        o, Ha, kw = self.op(T)
        a1 = self.fig.add_subplot(1, 2, 1)
        for i, TT in enumerate(sorted({20, 60, 140, 180, T})):
            H, B, J, Br, Hcj = magnet_curves(TT, **kw)
            bold = abs(TT - T) < 1e-6
            a1.plot(H / 1e3, B, color=SERIES[i % 8], lw=2.4 if bold else 1.1, label=f"B(H) {TT:.0f}°C")
            if bold:
                a1.plot(H / 1e3, J, "--", color=SERIES[i % 8], lw=1.2, label=f"J(H) {TT:.0f}°C")
        Hl = np.linspace(-2200e3, 0, 50)
        a1.plot(Hl / 1e3, -MU0 * self.p("Pc") * Hl, ":", color=C["sub"], label="prosta obciążenia (I = 0)")
        a1.plot(Hl / 1e3, -MU0 * self.p("Pc") * (Hl + Ha), "-.", color=C["red"], label=f"prosta z prądem (Ha = {Ha / 1e3:.0f} kA/m)")
        a1.plot(o["H"] / 1e3, o["B"], "o", color=C["red"], ms=9)
        a1.set_xlim(-2200, 0)
        a1.set_ylim(-0.3, 1.45)
        finish(a1, "H [kA/m]", "B, J [T]", "Krzywe odmagnesowania N39UH i punkt pracy", loc="upper left")
        # cykl temperatury 60 -> Tpk -> 60 °C (rys. 12.23)
        a2 = self.fig.add_subplot(1, 2, 2)
        tt = np.linspace(0, 30, 121)
        Tpk = self.p("Tpk")
        prof = np.interp(tt, [0, 10, 20, 30], [60, Tpk, Tpk, 60])
        o60, _, _ = self.op(60)
        loss_perm, ratio = 0.0, []
        for TT in prof:
            oo, _, _ = self.op(TT)
            rel = oo["J"] / oo["Br"]
            if rel < 0.99:                       # za kolanem -> strata trwała
                loss_perm = max(loss_perm, 0.99 - rel)
            ratio.append(oo["B"] * (1 - loss_perm) / o60["B"] * 100)
        a2.plot(tt, ratio, color=C["cyan"], lw=2, label="B pracy / B(60°C) [%]")
        a2.set_ylabel("%")
        a2b = a2.twinx()
        a2b.grid(False)
        a2b.plot(tt, prof, color=C["peach"], label="temperatura [°C]")
        a2b.set_ylabel("°C")
        a2.set_xlabel("czas [min]")
        a2.set_title(f"Rys. 12.23: cykl 60 -> {Tpk:.0f} -> 60 °C przy {self.p('I'):.0f} Arms")
        twin_legend(a2, a2b, loc="lower center")
        rel = o["J"] / o["Br"]
        status = "ODWRACALNE (przed kolanem)" if rel >= 0.99 else "NIEODWRACALNE (za kolanem!)"
        return [f"T = {T:.0f} °C:",
                f"  Br(T)  = {o['Br']:.3f} T",
                f"  iHc(T) = {o['Hcj'] / 1e3:.0f} kA/m",
                f"id = {-math.sqrt(2) * self.p('I') * math.sin(math.radians(self.p('beta'))):.0f} A",
                f"Ha = {Ha / 1e3:.0f} kA/m",
                f"Punkt pracy: H = {o['H'] / 1e3:.0f} kA/m",
                f"             B = {o['B']:.3f} T",
                f"J/Br = {rel * 100:.1f} % -> {status}",
                "",
                f"Po cyklu do {Tpk:.0f} °C:",
                f"  trwała strata strumienia = {loss_perm * 100:.1f} %",
                "  (książka: brak trwałego rozmagnesowania",
                "   przy 320 Arms i 180 °C)",
                "",
                "Dane N39UH (tab. 12.8): Br 1,22-1,28 T,",
                "iHc 1989 kA/m, bHc 915 kA/m,",
                "(BH)max 286-318 kJ/m³, ρ = 7600 kg/m³"]


class TabE10(ExampleTab):
    key = "E10"
    PARAMS = [("nmax", "prędkość maks. [obr/min]", 4000, 20000, 12000, 100),
              ("over", "zapas prędkości [%]", 0, 40, 20, 1),
              ("R", "promień wirnika R [mm]", 50, 120, 80, 1),
              ("rcg", "promień środka masy bieguna [mm]", 40, 110, 70, 1),
              ("mp", "masa bieguna (za barierą) [kg]", 0.05, 1.0, 0.30, 0.01),
              ("L", "długość pakietu L [mm]", 50, 250, 120, 1),
              ("w", "szerokość mostka w [mm]", 0.5, 4.0, 1.5, 0.05),
              ("nb", "liczba mostków na biegun", 1, 4, 2, 1),
              ("Kt", "koncentracja naprężeń K_t [-]", 1.0, 4.0, 2.5, 0.05),
              ("resin", "odciążenie przez żywicę [%]", 0, 60, 0, 1),
              ("lb", "długość mostka [mm]", 1, 10, 4, 0.5),
              ("Bg", "indukcja w szczelinie [T]", 0.5, 1.2, 0.9, 0.05)]
    YIELD = 410.0    # MPa, 27PNF1500
    E_ST = 165e9     # Pa

    def stress(self, n, w):
        om = np.asarray(n) * 2 * PI / 60
        F = self.p("mp") * self.p("rcg") * 1e-3 * om ** 2
        A = self.p("nb") * w * 1e-3 * self.p("L") * 1e-3
        s_avg = F / A * (1 - self.p("resin") / 100)
        return s_avg * self.p("Kt") / 1e6, s_avg / 1e6

    def update_plot(self):
        nmax, over, w = self.p("nmax"), self.p("over"), self.p("w")
        n_os = nmax * (1 + over / 100)
        nn = np.linspace(0, 1.4 * nmax, 200)
        a1 = self.fig.add_subplot(1, 2, 1)
        for i, ww in enumerate((1.0, 1.5, 2.0, 2.5)):
            a1.plot(nn, self.stress(nn, ww)[0], color=SERIES[i + 1], lw=1, label=f"w = {ww} mm")
        a1.plot(nn, self.stress(nn, w)[0], color=C["cyan"], lw=2.6, label=f"w = {w} mm (wybrane)")
        R = self.p("R") * 1e-3
        a1.plot(nn, (3 + 0.3) / 8 * 7600 * (nn * 2 * PI / 60) ** 2 * R ** 2 / 1e6, ":", color=C["sub"], label="pełna tarcza (odniesienie)")
        a1.axhline(self.YIELD, color=C["red"], ls="--", label="granica plast. 410 MPa")
        a1.axvline(n_os, color=C["peach"], ls=":", label=f"n_max + {over:.0f} %")
        a1.set_ylim(0, 1.6 * self.YIELD)
        finish(a1, "n [obr/min]", "σ_max w mostku [MPa]", "Naprężenie od siły odśrodkowej", loc="upper left")
        a2 = self.fig.add_subplot(1, 2, 2)
        ww = np.linspace(0.5, 4, 100)
        a2.plot(ww, self.stress(n_os, ww)[0], color=C["cyan"], lw=2, label=f"σ przy {n_os:.0f} obr/min [MPa]")
        a2.axhline(self.YIELD, color=C["red"], ls="--")
        a2.set_ylim(0, 3 * self.YIELD)
        a2.set_ylabel("MPa")
        a2b = a2.twinx()
        a2b.grid(False)
        tau_p = 2 * PI * R / 8
        leak = self.p("nb") * ww * 1e-3 * 2.0 / (self.p("Bg") * tau_p) * 100
        a2b.plot(ww, leak, color=C["peach"], lw=2, label="strumień rozproszenia [%]")
        a2b.set_ylabel("%")
        a2.axvline(w, color=C["sub"], ls=":")
        a2.set_xlabel("szerokość mostka w [mm]")
        a2.set_title("Kompromis: wytrzymałość vs strumień magnesu")
        twin_legend(a2, a2b, loc="upper right")
        smax, savg = self.stress(n_os, w)
        smax, savg = float(smax), float(savg)
        om = n_os * 2 * PI / 60
        F = self.p("mp") * self.p("rcg") * 1e-3 * om ** 2
        disp = savg * 1e6 / self.E_ST * self.p("lb") * 1e-3 * 1e6
        ok = smax < self.YIELD
        # najmniejsza szerokość bezpieczna
        wmin = ww[np.argmax(self.stress(n_os, ww)[0] < self.YIELD)] if (self.stress(n_os, ww)[0] < self.YIELD).any() else None
        return [f"n sprawdzeniowe = {n_os:.0f} obr/min",
                f"ω = {om:.0f} rad/s",
                f"Siła odśrodkowa bieguna F = m·r·ω²",
                f"  = {self.p('mp')}·{self.p('rcg') / 1e3}·{om:.0f}² = {F / 1e3:.1f} kN",
                f"Pole mostków = {self.p('nb'):.0f}·{w}·{self.p('L'):.0f} = {self.p('nb') * w * self.p('L'):.0f} mm²",
                f"σ średnie = {savg:.0f} MPa",
                f"σ max (×K_t) = {smax:.0f} MPa  {'OK' if ok else 'ZA DUŻO!'}",
                f"Współczynnik bezpieczeństwa = {self.YIELD / smax:.2f}",
                f"Przemieszczenie ≈ σ/E·l = {disp:.1f} µm (reguła < 10 µm)",
                f"Tarcza pełna: {(3.3 / 8) * 7600 * om ** 2 * (self.p('R') * 1e-3) ** 2 / 1e6:.0f} MPa",
                "",
                f"Min. szerokość mostka: {wmin:.2f} mm" if wmin else "Brak bezpiecznej szerokości < 4 mm",
                f"Rozproszenie strumienia: {float(self.p('nb') * w * 1e-3 * 2.0 / (self.p('Bg') * 2 * PI * self.p('R') * 1e-3 / 8) * 100):.1f} %",
                "",
                "Tab. 12.5: blacha 27PNF1500: Rm = 410 MPa,",
                "E = 165 GPa, ν = 0,3; żywica 3M 2214:",
                "E = 5,17 GPa, ν = 0,35"]


class TabE11(ExampleTab):
    key = "E11"
    PARAMS = [("Ns", "liczba żłobków Ns", 24, 96, 48, 6),
              ("P", "liczba biegunów P", 4, 16, 8, 2),
              ("Nph", "zwoje szeregowe na fazę Nph", 8, 48, 24, 1),
              ("d", "średnica drutu d [mm]", 0.5, 1.5, 1.02, 0.01),
              ("ns", "liczba żył równoległych", 1, 80, 22, 1),
              ("As", "pole żłobka [mm²]", 50, 250, 109, 1),
              ("Dsi", "średnica wewn. stojana [mm]", 100, 250, 171, 0.5),
              ("hq", "wysokość żłobka hq [mm]", 10, 40, 24.45, 0.05),
              ("la", "wysięg la [mm]", 0, 10, 3, 0.5),
              ("Lst", "długość pakietu [mm]", 50, 250, 120, 1),
              ("Ij", "prąd do gęstości J [A]", 50, 500, 320, 1),
              ("Irms", "prąd skuteczny [Arms]", 10, 500, 160, 1),
              ("T", "temperatura uzwojenia [°C]", 20, 200, 150, 1)]
    PRESETS = {"Silnik z książki (tab. 12.4, p. 12.5.1)": dict(Ns=48, P=8, Nph=24, d=1.02, ns=22, As=109, Dsi=171, hq=24.45, la=3, Lst=120, Ij=320, Irms=160, T=150),
               "Zad. 12.9 (Nph = 16, D = 160, żłobek 25×8)": dict(Ns=48, P=8, Nph=16, d=1.0, ns=56, As=200, Dsi=160, hq=25, la=3, Lst=150, Ij=226, Irms=160, T=140),
               "Zad. 12.1 (okładzina, R = 78 mm, 350 A)": dict(Ns=48, P=8, Nph=24, d=1.02, ns=22, As=109, Dsi=156, hq=24.45, la=3, Lst=120, Ij=320, Irms=350, T=150)}

    def update_plot(self):
        Ns, P, Nph, d, ns, As = int(self.p("Ns")), int(self.p("P")), self.p("Nph"), self.p("d"), self.p("ns"), self.p("As")
        Dsi, hq, la, Lst, T = self.p("Dsi"), self.p("hq"), self.p("la"), self.p("Lst"), self.p("T")
        z = 2 * 3 * Nph / Ns                       # przewody (boki cewek) w żłobku
        a_str = PI * d ** 2 / 4
        fill = z * ns * a_str / As
        J = self.p("Ij") / (ns * a_str)
        tau = PI * Dsi / P
        lb = 1.3 * tau + 3 * hq + 2 * la
        lturn = 2 * (lb + Lst)
        lph = Nph * lturn / 1e3
        R20 = RHO_CU20 * lph / (ns * a_str * 1e-6)
        RT = lambda t: R20 * (1 + ALPHA_CU * (np.asarray(t) - 20))
        Pcu = 3 * self.p("Irms") ** 2 * float(RT(T))
        Aload = Ns * z * self.p("Irms") / (PI * Dsi / 10)
        ns_opt = 0.44 * As / (z * a_str)
        a1 = self.fig.add_subplot(2, 2, 1)
        tt = np.linspace(0, 200, 100)
        a1.plot(tt, RT(tt) * 1e3, color=C["cyan"], lw=2, label="R(ϑ) = R20[1 + α(ϑ-20)]")
        for tp in (20, 100, 140, 150):
            a1.plot(tp, float(RT(tp)) * 1e3, "o", color=C["peach"])
            a1.annotate(f"{tp}°C: {float(RT(tp)) * 1e3:.2f} mΩ", (tp, float(RT(tp)) * 1e3), fontsize=7,
                        xytext=(4, -10), textcoords="offset points")
        finish(a1, "temperatura [°C]", "R fazy [mΩ]", "Rezystancja rośnie z temperaturą (α = 0,0039 1/K)", loc="upper left")
        a2 = self.fig.add_subplot(2, 2, 2)
        nn = np.arange(1, 81)
        a2.plot(nn, z * nn * a_str / As * 100, color=C["green"], lw=2, label="wypełnienie")
        a2.axhspan(43, 45, color=C["yellow"], alpha=0.3, label="43-45 % (Zad. 12.9)")
        a2.set_ylim(0, 100)
        a2.plot(ns, fill * 100, "o", color=C["red"], ms=8)
        finish(a2, "liczba żył", "wypełnienie miedzią [%]", "Współczynnik wypełnienia żłobka", loc="upper left")
        a3 = self.fig.add_subplot(2, 1, 2)
        parts = [2 * Lst, 2 * 1.3 * tau, 2 * 3 * hq, 2 * 2 * la]
        labels = ["część w żłobkach 2·L", "czoła: 2·1,3·τp", "czoła: 2·3·hq", "czoła: 2·2·la"]
        left = 0
        for i, (v, lab) in enumerate(zip(parts, labels)):
            a3.barh([0], [v], left=left, color=SERIES[i], label=f"{lab} = {v:.0f} mm")
            left += v
        a3.set_yticks([])
        finish(a3, "długość jednego zwoju [mm]", None, f"Rys. 12.27: zwój = {lturn:.0f} mm (czoła = {100 * (lturn - 2 * Lst) / lturn:.0f} %)",
               loc="upper right")
        a3.set_ylim(-0.8, 1.2)
        return [f"Przewody w żłobku z = 2·3·Nph/Ns = {z:.2f}",
                f"Pole żyły = π·{d}²/4 = {a_str:.4f} mm²",
                f"Miedź w żłobku = {z * ns * a_str:.1f} mm²",
                f"Wypełnienie = {fill * 100:.1f} %",
                f"  (dla 44 % potrzeba {ns_opt:.1f} żył)",
                f"J = {self.p('Ij'):.0f}/({ns:.0f}·{a_str:.4f}) = {J:.2f} A/mm²",
                f"τp = π·D/P = {tau:.2f} mm",
                f"l_b = 1,3τp + 3hq + 2la = {lb:.1f} mm",
                f"zwój = 2(l_b + L) = {lturn:.1f} mm",
                f"długość fazy = {lph:.2f} m",
                f"R20 = {R20 * 1e3:.2f} mΩ",
                f"R({T:.0f}°C) = {float(RT(T)) * 1e3:.2f} mΩ (+{ALPHA_CU * (T - 20) * 100:.0f} %)",
                f"P_Cu = 3·I²·R = 3·{self.p('Irms'):.0f}²·R = {Pcu:.0f} W",
                "",
                f"Okładzina prądowa (Zad. 12.1):",
                f"A = Ns·z·I/(πD) = {Ns}·{z:.0f}·{self.p('Irms'):.0f}/(π·{Dsi / 10:.1f} cm)",
                f"  = {Aload:.0f} A/cm"]


class TabE12(ExampleTab):
    key = "E12"
    PARAMS = [("P", "liczba biegunów P", 2, 16, 8, 2),
              ("n", "prędkość hamowni [obr/min]", 100, 6000, 1000, 10),
              ("V", "podstawowa składowa napięcia [V]", 1, 400, 51.8, 0.001),
              ("h5", "5. harm. LL [V rms]", 0, 10, 1.08, 0.01),
              ("h7", "7. harm. LL [V rms]", 0, 10, 1.78, 0.01),
              ("noise", "szum pomiaru [V]", 0, 5, 0.5, 0.1),
              ("per", "liczba okresów w oknie FFT", 2, 10, 8, 1),
              ("E", "BLDC: amplituda trapezu E [V]", 10, 300, 100, 1),
              ("flat", "BLDC: płaski wierzchołek [° el]", 60, 170, 120, 1)]
    VTYPES = ["V_LL rms (międzyprzewodowe skuteczne)", "V_LL szczyt (międzyprzewodowe)", "V_faz szczyt (fazowe)"]
    MODES = ["Pomiar napięcia (sinus + harmoniczne)", "BLDC - trapezowa SEM (Zad. 12.10)"]
    PRESETS = {"Ćw. 12.4: 8 bieg., 1000 obr/min, 51,8 V rms LL": dict(P=8, n=1000, V=51.8, h5=1.08, h7=1.78, _vt=0, _mode=0),
               "Zad. 12.4: 6 bieg., 3600 obr/min, 100 V szczyt LL": dict(P=6, n=3600, V=100, h5=0, h7=0, _vt=1, _mode=0),
               "Zad. 12.8a: 12 bieg., 300 obr/min, 67,482 V faz.": dict(P=12, n=300, V=67.482, h5=0, h7=0, _vt=2, _mode=0),
               "Zad. 12.10: BLDC, E = 100 V, 120°, 3000 obr/min": dict(P=8, n=3000, E=100, flat=120, _mode=1)}

    def extra_controls(self, parent):
        ttk.Label(parent, text="Tryb").pack(anchor="w", padx=6)
        self.mode = tk.StringVar(value=self.MODES[0])
        cb = ttk.Combobox(parent, textvariable=self.mode, values=self.MODES, state="readonly")
        cb.pack(fill=tk.X, padx=6)
        cb.bind("<<ComboboxSelected>>", lambda e: self.refresh())
        ttk.Label(parent, text="Rodzaj podanego napięcia").pack(anchor="w", padx=6)
        self.vtype = tk.StringVar(value=self.VTYPES[0])
        cb2 = ttk.Combobox(parent, textvariable=self.vtype, values=self.VTYPES, state="readonly")
        cb2.pack(fill=tk.X, padx=6)
        cb2.bind("<<ComboboxSelected>>", lambda e: self.refresh())

    def set_special(self, key, value):
        if key == "_vt":
            self.vtype.set(self.VTYPES[value])
        elif key == "_mode":
            self.mode.set(self.MODES[value])

    def update_plot(self):
        P, n = int(self.p("P")), self.p("n")
        we = float(rpm_to_we(n, P))
        if self.mode.get() == self.MODES[1]:
            return self.plot_bldc(P, n, we)
        vt = self.VTYPES.index(self.vtype.get())
        V = self.p("V")
        Vll_pk = [V * math.sqrt(2), V, V * SQ3][vt]
        f = we / (2 * PI)
        per = int(self.p("per"))
        N = 256 * per
        t = np.arange(N) / (256 * f)
        rng = np.random.default_rng(1)
        v = (Vll_pk * np.sin(we * t) + math.sqrt(2) * self.p("h5") * np.sin(5 * we * t + 0.4)
             + math.sqrt(2) * self.p("h7") * np.sin(7 * we * t - 0.8) + self.p("noise") * rng.standard_normal(N))
        spec = np.abs(np.fft.rfft(v)) / N * 2
        h = np.arange(1, 10)
        amp = spec[h * per] / math.sqrt(2)          # wartości skuteczne harmonicznych
        a1 = self.fig.add_subplot(2, 1, 1)
        a1.plot(t * 1e3, v, color=C["cyan"], lw=1, label="zmierzone v_ab(t)")
        a1.plot(t * 1e3, spec[per] * np.sin(we * t), "--", color=C["peach"], lw=1.2, label="składowa podstawowa (FFT)")
        finish(a1, "t [ms]", "V [V]", f"Rys. 12.34a: napięcie jałowe międzyprzewodowe przy {n:.0f} obr/min ({per} okresów)", loc="upper right")
        a2 = self.fig.add_subplot(2, 1, 2)
        bars = a2.bar(h, amp, color=C["mauve"], width=0.6)
        for b, a in zip(bars, amp):
            a2.annotate(f"{a:.2f}", (b.get_x() + b.get_width() / 2, a), ha="center", va="bottom", fontsize=8)
        a2.set_xticks(h)
        finish(a2, "rząd harmonicznej", "V_LL rms [V]", "Rys. 12.34b: widmo (okno = całkowita liczba okresów)")
        V1rms = amp[0]
        Vph_pk = V1rms * math.sqrt(2) / SQ3
        psi = Vph_pk / we
        return [f"Podano: {V:g} V ({self.VTYPES[vt].split(' (')[0]})",
                f"FFT: V_LL(1) = {V1rms:.2f} V rms",
                f"V_faz szczyt = V_LL·√2/√3 = {Vph_pk:.2f} V",
                f"ωe = {n:.0f}/60·2π·{P // 2} = {we:.1f} rad/s",
                f"ψm = V_faz,szczyt/ωe = {psi:.4f} Wb",
                "",
                "Sprawdzenie z książką:",
                "  Ćw. 12.4:   ψm = 42,3/418,9 = 0,101 Wb",
                "  Zad. 12.4:  ψm = 57,7/1131 = 0,051 Wb",
                "  Zad. 12.8a: ψm = 67,48/188,5 = 0,358 Wb",
                "",
                f"Częstotliwość: {f:.1f} Hz, okno {per / f * 1e3:.1f} ms"]

    def plot_bldc(self, P, n, we):
        E, flat = self.p("E"), self.p("flat")
        pp = P // 2
        thm = np.linspace(0, 2 * PI, 2000)
        e = trapezoid_emf(thm * pp, E, flat)
        nn, b = trapezoid_coeffs(E, 15, flat)
        a1 = self.fig.add_subplot(2, 1, 1)
        a1.plot(np.degrees(thm), e, color=C["cyan"], lw=2, label="trapezowa SEM e(θ)")
        a1.plot(np.degrees(thm), b[0] * np.sin(thm * pp), "--", color=C["peach"], label=f"1. harmoniczna b1 = {b[0]:.1f} V")
        a1.axvline(360 / pp, color=C["sub"], ls=":", label=f"okres T = 2π/{pp} = {360 / pp:.0f}° mech")
        finish(a1, "kąt mechaniczny [°]", "e [V]", f"Zad. 12.10: SEM BLDC (płaski wierzchołek {flat:.0f}° el)", loc="upper right")
        a2 = self.fig.add_subplot(2, 1, 2)
        thf = np.linspace(0, 2 * PI, 4096, endpoint=False)
        fft = np.fft.rfft(trapezoid_emf(thf, E, flat)) / 4096 * 2
        a2.bar(nn, np.abs(b), color=C["mauve"], width=0.6, label="|b_n| wzór")
        a2.plot(nn, np.abs(fft[1:16]), "o", color=C["green"], label="FFT (sprawdzenie)")
        a2.set_xticks(nn)
        finish(a2, "n", "|b_n| [V]", "Zad. 12.10c: widmo SEM trapezowej")
        psi = b[0] / we
        lines = [f"a) b_n = 4E/(π n² a)·sin(n a), a = {(180 - flat) / 2:.0f}°",
                 f"   (n parzyste: b_n = 0)",
                 f"b) okres T = π/2 mech -> {int(round(360 / 90))} okresy",
                 f"   na obrót -> 4 pary -> P = 8 biegunów",
                 "c) n   b_n [V]"]
        lines += [f"   {k:2d}  {v:8.3f}" for k, v in zip(nn, b) if k % 2 == 1]
        lines += [f"d) ωe({n:.0f} obr/min, P={P}) = {we:.1f} rad/s",
                  f"   ψm = b1/ωe = {b[0]:.2f}/{we:.1f} = {psi:.4f} Wb",
                  f"   THD trapezu = {100 * math.sqrt(np.sum(b[1:] ** 2)) / abs(b[0]):.1f} % (do n=15)"]
        return lines


class TabE13(ExampleTab):
    key = "E13"
    PARAMS = [("n", "prędkość hamowni [obr/min]", 100, 3000, 500, 10),
              ("idm", "punkt pomiaru id* [A]", -300, -5, -50, 1),
              ("iqm", "punkt pomiaru iq* [A]", 5, 300, 50, 1),
              ("noise", "szum pomiaru napięcia [V]", 0, 1, 0.05, 0.01),
              ("rerr", "błąd znajomości rs [%]", -30, 30, 0, 1),
              ("perr", "błąd znajomości ψm [%]", -10, 10, 0, 0.5),
              ("V0", "Zad. 12.7: napięcie jałowe (faz. szczyt) [V]", 60, 200, 120, 1),
              ("lag", "Zad. 12.7: opóźnienie prądu [°]", 0, 45, 20, 1)]
    TRUE = dict(P=8, psi=0.0927, rs=0.0147)

    def measure(self, id_, iq, n, rng):
        """'Pomiar' napięć na prawdziwym (nasycającym się) silniku + szum."""
        we = float(rpm_to_we(n, 8))
        vd = self.TRUE["rs"] * id_ - we * Lq_sat(iq) * iq
        vq = self.TRUE["rs"] * iq + we * (Ld_sat(id_) * id_ + self.TRUE["psi"])
        s = self.p("noise")
        return vd + s * rng.standard_normal(np.shape(vd)), vq + s * rng.standard_normal(np.shape(vq)), we

    def estimate(self, id_, iq, vd, vq, we):
        rs = self.TRUE["rs"] * (1 + self.p("rerr") / 100)
        psi = self.TRUE["psi"] * (1 + self.p("perr") / 100)
        Ld = (vq - rs * iq - we * psi) / (we * id_)
        Lq = (rs * id_ - vd) / (we * iq)
        return Ld, Lq

    def update_plot(self):
        n = self.p("n")
        rng = np.random.default_rng(7)
        idg = np.linspace(-300, -10, 30)
        vd, vq, we = self.measure(idg, 0.2 * np.ones_like(idg), n, rng)
        Ld_m, _ = self.estimate(idg, 0.2, vd, vq, we)
        iqg = np.linspace(10, 300, 30)
        vd2, vq2, _ = self.measure(-0.1 * np.ones_like(iqg), iqg, n, rng)
        _, Lq_m = self.estimate(-0.1, iqg, vd2, vq2, we)
        a1 = self.fig.add_subplot(1, 2, 1)
        xx = np.linspace(-300, -5, 200)
        a1.plot(xx, Ld_sat(xx) * 1e3, color=C["cyan"], lw=2, label="Ld prawdziwe (model nasycenia)")
        a1.plot(idg, Ld_m * 1e3, "o", color=C["peach"], ms=5, label="Ld 'zmierzone' (iq = 0,2 A)")
        a1.axvline(self.p("idm"), color=C["sub"], ls=":")
        a1.set_ylim(0.15, 0.35)
        finish(a1, "id [A]", "Ld [mH]", "Rys. 12.36a: Ld w funkcji id", loc="lower left")
        a2 = self.fig.add_subplot(1, 2, 2)
        yy = np.linspace(5, 300, 200)
        a2.plot(yy, Lq_sat(yy) * 1e3, color=C["cyan"], lw=2, label="Lq prawdziwe (model nasycenia)")
        a2.plot(iqg, Lq_m * 1e3, "o", color=C["peach"], ms=5, label="Lq 'zmierzone' (id = -0,1 A)")
        a2.axvline(self.p("iqm"), color=C["sub"], ls=":")
        a2.set_ylim(0.3, 0.9)
        finish(a2, "iq [A]", "Lq [mH]", "Rys. 12.36b: Lq w funkcji iq", loc="lower right")
        # wybrany punkt pomiaru
        rng = np.random.default_rng(3)
        vdA, vqA, _ = self.measure(np.array(self.p("idm")), np.array(0.2), n, rng)
        LdA, _ = self.estimate(self.p("idm"), 0.2, vdA, vqA, we)
        vdB, vqB, _ = self.measure(np.array(-0.1), np.array(self.p("iqm")), n, rng)
        _, LqB = self.estimate(-0.1, self.p("iqm"), vdB, vqB, we)
        # sprawdzenie arkusza z rys. 12.35
        w500 = float(rpm_to_we(499.99, 8))
        Lq_b = (0.0147 * (-0.1) - (-6.5)) / (w500 * 48.8) * 1e3
        Ld_b = (17.0 - w500 * 0.0927 - 0.0147 * 0.2) / (w500 * (-49.3)) * 1e3
        # Zadanie 12.7b
        w3 = float(rpm_to_we(3000, 8))
        psi7 = self.p("V0") / w3
        out = []
        for lab, b in (("β = 45° - opóźnienie", 45 - self.p("lag")), ("β = opóźnienie", self.p("lag"))):
            vd7, vq7 = -86 * math.sin(math.radians(45)), 86 * math.cos(math.radians(45))
            id7, iq7 = -86 * math.sin(math.radians(b)), 86 * math.cos(math.radians(b))
            Lq7 = -vd7 / (w3 * iq7)
            Ld7 = (vq7 - w3 * psi7) / (w3 * id7) if abs(id7) > 1e-6 else float("nan")
            out += [f"  {lab}: β = {b:.0f}°",
                    f"   vd = {vd7:.1f}, vq = {vq7:.1f} V, id = {id7:.1f}, iq = {iq7:.1f} A",
                    f"   Lq = {Lq7 * 1e3:.3f} mH, Ld = {Ld7 * 1e3:.3f} mH"]
        return [f"ωe = {we:.1f} rad/s (n = {n:.0f} obr/min)",
                f"Punkt id* = {self.p('idm'):.0f} A:",
                f"  vq = {float(vqA):.2f} V -> Ld = {float(LdA) * 1e3:.4f} mH",
                f"  (prawdziwe {float(Ld_sat(self.p('idm'))) * 1e3:.4f} mH)",
                f"Punkt iq* = {self.p('iqm'):.0f} A:",
                f"  vd = {float(vdB):.2f} V -> Lq = {float(LqB) * 1e3:.4f} mH",
                f"  (prawdziwe {float(Lq_sat(self.p('iqm'))) * 1e3:.4f} mH)",
                "",
                "Arkusz z rys. 12.35 (500 obr/min):",
                f"  Lq = (rs·id - vd)/(ωe·iq) = {Lq_b:.3f} mH",
                f"  Ld = (vq - ωeψm - rs·iq)/(ωe·id) = {Ld_b:.3f} mH",
                "",
                f"Zad. 12.7b (3000 obr/min, ωe = {w3:.1f}):",
                f"  ψm = V0/ωe = {self.p('V0'):.0f}/{w3:.1f} = {psi7:.4f} Wb",
                *out]


class TabE14(ExampleTab):
    key = "E14"
    has_table = True
    PARAMS = [("n", "prędkość [obr/min]", 500, 12000, 4500, 50),
              ("Vdc", "napięcie baterii [V]", 200, 450, 320, 5),
              ("psi", "ψm [Wb]", 0.05, 0.15, 0.09, 0.001),
              ("rs", "rs [mΩ]", 0, 30, 10, 0.5),
              ("Ld", "Ld [µH]", 100, 600, 280, 1),
              ("Lq", "Lq [µH]", 200, 1200, 500, 1),
              ("dV", "zapas ΔV [V]", 0, 20, 6, 0.5),
              ("Imax", "prąd maks. [A]", 50, 500, 300, 10),
              ("Istep", "krok prądu ΔI [A]", 10, 100, 50, 10),
              ("db", "krok kąta Δβ [°]", 0.5, 5, 1, 0.5),
              ("P", "liczba biegunów P", 4, 16, 8, 2)]
    PRESETS = {"Zad. 12.5: 4500 obr/min, 320 V, ψm = 0,09": dict(n=4500, Vdc=320, psi=0.09, rs=10, Ld=280, Lq=500, Imax=300, Istep=50),
               "Jak rys. 12.41a: 2750 obr/min, 360 V (tab. 12.3)": dict(n=2750, Vdc=360, psi=0.0927, rs=13, Ld=234, Lq=562, Imax=400, Istep=50),
               "Jak rys. 12.41b: 10000 obr/min, 360 V (tab. 12.3)": dict(n=10000, Vdc=360, psi=0.0927, rs=13, Ld=234, Lq=562, Imax=400, Istep=50)}

    def search(self):
        """Algorytm z rys. 12.40: dla każdego I zmniejszaj β od 89° aż do granicy napięcia."""
        prm = prm_from(self)
        we = float(rpm_to_we(self.p("n"), prm["P"]))
        Vlim = self.p("Vdc") / SQ3 - self.p("dV")
        res = []
        I = self.p("Istep")
        while I <= self.p("Imax") + 1e-9:
            best, b = None, 89.0
            while b >= 0:
                id_, iq = dq_from_beta(I, b)
                _, _, Vs = voltages(id_, iq, we, prm["rs"], prm["psi"], prm["Ld"], prm["Lq"])
                if Vs > Vlim:
                    break
                T = torque(id_, iq, prm["P"], prm["psi"], prm["Ld"], prm["Lq"])
                if best is None or T > best[2]:
                    best = (I, b, T, id_, iq, Vs)
                b -= self.p("db")
            res.append(best if best else (I, np.nan, np.nan, np.nan, np.nan, np.nan))
            I += self.p("Istep")
        return prm, we, Vlim, res

    def update_plot(self):
        prm, we, Vlim, res = self.search()
        a1 = self.fig.add_subplot(1, 2, 1)
        bb = np.linspace(0, 89, 400)
        for i, r in enumerate(res):
            I = r[0]
            id_, iq = dq_from_beta(I, bb)
            _, _, Vs = voltages(id_, iq, we, prm["rs"], prm["psi"], prm["Ld"], prm["Lq"])
            T = torque(id_, iq, prm["P"], prm["psi"], prm["Ld"], prm["Lq"])
            col = SERIES[i % 8]
            a1.plot(bb, np.where(Vs <= Vlim, np.nan, T), ":", color=col, lw=1)
            a1.plot(bb, np.where(Vs <= Vlim, T, np.nan), color=col, lw=2, label=f"I = {I:.0f} A")
            if np.isfinite(r[1]):
                a1.plot(r[1], r[2], "o", color=col, ms=8, mec=C["text"])
        finish(a1, "β [°] (od osi q)", "T [Nm]", f"Rys. 12.41: pomiar momentu przy {self.p('n'):.0f} obr/min", loc="upper right")
        a2 = self.fig.add_subplot(1, 2, 2)
        Im = self.p("Imax")
        g1, g2 = np.meshgrid(np.linspace(-1.3 * Im, 0, 300), np.linspace(0, 1.3 * Im, 300))
        TT = torque(g1, g2, prm["P"], prm["psi"], prm["Ld"], prm["Lq"])
        cs = a2.contour(g1, g2, TT, levels=8, colors=C["surf2"], linewidths=0.8)
        a2.clabel(cs, fmt="%.0f Nm", fontsize=7)
        _, _, VV = voltages(g1, g2, we, prm["rs"], prm["psi"], prm["Ld"], prm["Lq"])
        a2.contour(g1, g2, VV, levels=[Vlim], colors=C["red"], linewidths=2)
        a2.plot([], [], color=C["red"], lw=2, label="granica napięcia (elipsa)")
        th = np.linspace(PI / 2, PI, 100)
        for r in res:
            a2.plot(r[0] * np.cos(th), r[0] * np.sin(th), color=C["surf2"], lw=0.8)
        im, qm = mtpa_curve(prm, 1.3 * Im)
        a2.plot(im, qm, "--", color=C["yellow"], label="MTPA")
        pts = np.array([[r[3], r[4]] for r in res if np.isfinite(r[1])])
        if len(pts):
            a2.plot(pts[:, 0], pts[:, 1], "o-", color=C["cyan"], ms=7, lw=2, label="znalezione optima")
        a2.axvline(-prm["psi"] / prm["Ld"], color=C["mauve"], ls=":", lw=1, label="-ψm/Ld (środek elipsy)")
        a2.set_xlim(-1.3 * Im, 0)
        a2.set_ylim(0, 1.3 * Im)
        a2.set_aspect("equal", adjustable="box")
        finish(a2, "id [A]", "iq [A]", "Płaszczyzna prądu", loc="upper left")
        lines = [f"ωe = {we:.1f} rad/s, Vlim = {Vlim:.1f} V",
                 "  I[A]   β*[°]   id*[A]   iq*[A]   T[Nm]"]
        for r in res:
            if np.isfinite(r[1]):
                lines.append(f"  {r[0]:4.0f}  {r[1]:6.1f}  {r[3]:7.1f}  {r[4]:7.1f}  {r[2]:6.1f}")
            else:
                lines.append(f"  {r[0]:4.0f}   brak - napięcie za duże")
        mt = np.degrees(np.arcsin(-mtpa_curve(prm, 200, 3)[0][-1] / 200)) if prm["Lq"] > prm["Ld"] else 0
        lines += ["", f"β_MTPA (przy 200 A) = {mt:.1f}°",
                  "Gdy β* > β_MTPA - optimum leży na",
                  "granicy napięcia, nie na szczycie krzywej."]
        return lines

    def table(self):
        _, _, _, res = self.search()
        return (["I [A]", "β* [°]", "T [Nm]", "id* [A]", "iq* [A]", "Vs [V]"],
                [[round(float(v), 2) for v in r] for r in res])


class TabE15(ExampleTab):
    key = "E15"
    has_table = True
    PARAMS = [("T", "zadany moment T* [Nm]", 0, 320, 100, 1),
              ("n", "prędkość [obr/min]", 500, 12000, 3600, 10),
              ("Vdc", "napięcie baterii Vdc [V]", 240, 440, 260, 1),
              ("dV", "zapas ΔV (rozw. modelowe) [V]", 0, 20, 0, 0.5)]
    PRESETS = {"Przykład: 100 Nm, 3600 obr/min, 260 V": dict(T=100, n=3600, Vdc=260),
               "Ćw. 12.5: 110 Nm, 4200 obr/min, 300 V": dict(T=110, n=4200, Vdc=300),
               "Zad. 12.6a: 185 Nm, 4500 obr/min, 360 V": dict(T=185, n=4500, Vdc=360),
               "Zad. 12.6b,c: 185 Nm, 4500 obr/min, 320 V": dict(T=185, n=4500, Vdc=320)}

    def update_plot(self):
        T, n, Vdc = self.p("T"), self.p("n"), self.p("Vdc")
        r = lut_read(T, n, Vdc)
        prm = dict(MOTOR_12_3)
        we = float(rpm_to_we(n, 8))
        lam_lim = (Vdc / SQ3 - self.p("dV")) / we
        idm, iqm = current_command(T, lam_lim, prm, 453, n_grid=2000)
        Tm = float(torque(idm, iqm, 8, prm["psi"], prm["Ld"], prm["Lq"]))
        a1 = self.fig.add_subplot(1, 2, 1)
        cmap = matplotlib.colormaps["cool"]
        for k in range(16):
            col = cmap(k / 15)
            a1.plot(LUT[k, :, 0], LUT[k, :, 1], "-", color=col, lw=1.2, marker=".", ms=4,
                    label=f"{LUT_RPM[k]} obr/min" if k % 3 == 0 else None)
        g1, g2 = np.meshgrid(np.linspace(-450, 0, 200), np.linspace(0, 350, 200))
        TT = torque(g1, g2, 8, prm["psi"], prm["Ld"], prm["Lq"])
        a1.contour(g1, g2, TT, levels=[T], colors=C["yellow"], linewidths=1.2, linestyles="--")
        a1.plot([], [], "--", color=C["yellow"], label=f"T = {T:.0f} Nm (model)")
        LAM = np.hypot(prm["psi"] + prm["Ld"] * g1, prm["Lq"] * g2)
        a1.contour(g1, g2, LAM, levels=[lam_lim], colors=C["red"], linewidths=1.6)
        a1.plot([], [], color=C["red"], label=f"granica strumienia λ = {lam_lim:.4f} Wb")
        a1.plot(r["id"], r["iq"], "*", color=C["green"], ms=16, mec=C["bg"], label=f"LUT: ({r['id']:.0f}, {r['iq']:.0f}) A")
        a1.plot(idm, iqm, "D", color=C["peach"], ms=9, mec=C["bg"], label=f"model: ({idm:.0f}, {iqm:.0f}) A")
        a1.set_xlim(-450, 0)
        a1.set_ylim(0, 350)
        finish(a1, "id [A]", "iq [A]", "Rys. 12.42/12.43: optymalne prądy (Vdc = 360 V)", loc="upper left")
        a2 = self.fig.add_subplot(1, 2, 2)
        fl = np.array(LUT_FLUX)
        for V, col in ((260, C["peach"]), (360, C["cyan"]), (430, C["green"]), (Vdc, C["red"])):
            if col != C["red"] and abs(V - Vdc) < 0.5:
                continue
            nf = V / (SQ3 * fl) * 60 / (2 * PI * 4)
            a2.plot(nf, fl, "o-" if V != Vdc else "-", color=col, ms=3, lw=2.2 if V == Vdc else 1.2, label=f"Vdc = {V:.0f} V")
        a2.plot(n, r["lam"], "s", color=C["red"], ms=9)
        a2.plot(r["n_f"], r["lam"], "*", color=C["green"], ms=14)
        a2.annotate("", xy=(r["n_f"], r["lam"]), xytext=(n, r["lam"]),
                    arrowprops=dict(arrowstyle="->", color=C["text"]))
        a2.set_xlim(0, 17000)
        finish(a2, "prędkość [obr/min]", "strumień λ [Wb]", "Tab. 12.9: kalibracja prędkości (n' = n·360/Vdc)", loc="upper right")
        return [f"1) λ = Vdc/(√3·ωe) = {Vdc:.0f}/(√3·{we:.1f})",
                f"     = {r['lam']:.4f} Wb",
                f"2) prędkość fikcyjna n' = {n:.0f}·360/{Vdc:.0f}",
                f"     = {r['n_f']:.0f} obr/min",
                f"   wiersz tablicy ≈ {r['row']:.2f}",
                f"3) T_max(n') (rys. 12.42) = {lut_tmax(r['n_f']):.0f} Nm",
                f"   przepustnica = {r['thr']:.1f} %",
                f"4) LUT: id* = {r['id']:.1f} A, iq* = {r['iq']:.1f} A",
                f"   |I| = {math.hypot(r['id'], r['iq']):.1f} A",
                f"   moment modelowy tego prądu: {float(torque(r['id'], r['iq'], 8, prm['psi'], prm['Ld'], prm['Lq'])):.0f} Nm",
                "",
                "Rozwiązanie modelowe (Zad. 12.6c):",
                "  Ld=0,234 mH, Lq=0,562 mH, ψm=0,0927",
                f"  id = {idm:.1f} A, iq = {iqm:.1f} A",
                f"  |I| = {math.hypot(idm, iqm):.1f} A, T = {Tm:.1f} Nm",
                ("  (moment nieosiągalny - pokazano maks.)" if Tm < T - 0.5 else ""),
                "",
                "Różnice LUT vs model: LUT zmierzono na",
                "prawdziwym silniku (nasycenie!), model",
                "zakłada stałe L."]

    def table(self):
        Vdc = self.p("Vdc")
        rows = []
        for k in range(16):
            rows.append([k, LUT_RPM[k], LUT_FLUX[k],
                         round(260 / (SQ3 * LUT_FLUX[k]) * 60 / (2 * PI * 4)),
                         round(430 / (SQ3 * LUT_FLUX[k]) * 60 / (2 * PI * 4)),
                         round(Vdc / (SQ3 * LUT_FLUX[k]) * 60 / (2 * PI * 4))])
        return (["krok", "n przy 360 V", "λ [Wb]", "n przy 260 V", "n przy 430 V", f"n przy {Vdc:.0f} V"], rows)


class TabE16(ExampleTab):
    key = "E16"
    PARAMS = [("T", "rozkaz momentu T* [Nm]", 20, 320, 150, 5),
              ("n0", "prędkość początkowa [obr/min]", 500, 8000, 3000, 100),
              ("n1", "prędkość końcowa [obr/min]", 2000, 12000, 9000, 100),
              ("tend", "czas symulacji [s]", 0.1, 0.6, 0.3, 0.01),
              ("Vdc0", "Vdc przed spadkiem [V]", 260, 430, 360, 5),
              ("Vdc1", "Vdc po spadku [V]", 240, 430, 300, 5),
              ("tsag", "chwila spadku napięcia [s]", 0.0, 0.6, 0.2, 0.01),
              ("perr", "błąd ψm w sterowniku [%]", -15, 15, 8, 0.5),
              ("dV", "zapas napięcia ΔV [V]", 0, 20, 8, 0.5),
              ("fbw", "pasmo regulatora prądu [Hz]", 100, 1000, 400, 10),
              ("Kaw", "wzmocnienie anti-windup", 0.001, 0.02, 0.004, 0.001)]

    def update_plot(self):
        prm = dict(MOTOR_12_3)
        args = dict(prm=prm, Imax=414, T_ref=self.p("T"), n0=self.p("n0"), n1=self.p("n1"), t_end=self.p("tend"),
                    Vdc0=self.p("Vdc0"), Vdc1=self.p("Vdc1"), t_sag=self.p("tsag"), psi_err_pct=self.p("perr"),
                    dV_margin=self.p("dV"), Kaw=self.p("Kaw"), f_bw=self.p("fbw"))
        A = simulate_torque_control(anti_windup=True, **args)
        B = simulate_torque_control(anti_windup=False, **args)
        t = A["t"] * 1e3
        a1 = self.fig.add_subplot(4, 1, 1)
        a1.plot(t, A["Tref"], ":", color=C["text"], label="T* rozkaz")
        a1.plot(B["t"] * 1e3, B["T"], color=C["sub"], lw=1.2, label="T bez anti-windup")
        a1.plot(t, A["T"], color=C["cyan"], lw=1.8, label="T z anti-windup")
        a1b = a1.twinx()
        a1b.grid(False)
        a1b.plot(t, A["n"], "--", color=C["peach"], lw=1, label="n [obr/min]")
        a1.set_ylabel("T [Nm]")
        a1b.set_ylabel("obr/min")
        a1.set_title("Rys. 12.47: sterowanie momentem EV z napięciowym anti-windup")
        twin_legend(a1, a1b, loc="lower left")
        a2 = self.fig.add_subplot(4, 1, 2, sharex=a1)
        a2.plot(B["t"] * 1e3, B["id"], color=C["sub"], lw=1, label="id bez AW")
        a2.plot(B["t"] * 1e3, B["idr"], ":", color=C["sub"], lw=1, label="id* bez AW")
        a2.plot(t, A["id"], color=C["green"], lw=1.6, label="id z AW")
        a2.plot(t, A["idr"], ":", color=C["green"], lw=1.2, label="id* z AW")
        a2.plot(t, A["iq"], color=C["mauve"], lw=1.6, label="iq z AW")
        a2.plot(B["t"] * 1e3, B["iq"], color=C["surf2"], lw=1, label="iq bez AW")
        finish(a2, None, "prąd [A]", legend=True, loc="lower left")
        a2.legend(loc="lower left", ncol=3)
        a3 = self.fig.add_subplot(4, 1, 3, sharex=a1)
        a3.plot(B["t"] * 1e3, B["Vref"], color=C["sub"], lw=1, label="|V*| bez AW")
        a3.plot(t, A["Vref"], color=C["cyan"], lw=1.6, label="|V*| z AW")
        a3.plot(t, A["Vlim"], "--", color=C["red"], label="Vdc/√3")
        a3.set_ylim(0, 1.4 * A["Vlim"].max())
        finish(a3, None, "napięcie [V]", loc="lower right")
        a4 = self.fig.add_subplot(4, 1, 4, sharex=a1)
        a4.plot(t, A["dlam"] * 1e3, color=C["yellow"], lw=1.8, label="Δλ z pętli anti-windup")
        finish(a4, "t [ms]", "Δλ [mWb]", loc="upper left")
        m = A["t"] > 0.02
        errA = np.sqrt(np.mean((A["T"][m] - A["Tref"][m]) ** 2))
        errB = np.sqrt(np.mean((B["T"][m] - B["Tref"][m]) ** 2))
        idA = np.sqrt(np.mean((A["id"][m] - A["idr"][m]) ** 2))
        idB = np.sqrt(np.mean((B["id"][m] - B["idr"][m]) ** 2))
        satA = np.mean(A["Vref"][m] > A["Vlim"][m]) * 100
        satB = np.mean(B["Vref"][m] > B["Vlim"][m]) * 100
        return [f"Silnik: tab. 12.3 (Ld/Lq = 0,234/0,562 mH)",
                f"Kp_d = {2 * PI * self.p('fbw') * prm['Ld']:.3f}, Kp_q = {2 * PI * self.p('fbw') * prm['Lq']:.3f} V/A",
                f"Ki = {2 * PI * self.p('fbw') * prm['rs']:.1f} V/(A·s)",
                "",
                "                     z AW    bez AW",
                f"RMS błąd momentu  {errA:7.1f}  {errB:7.1f} Nm",
                f"RMS błąd id       {idA:7.1f}  {idB:7.1f} A",
                f"czas w nasyceniu  {satA:7.1f}  {satB:7.1f} %",
                f"max Δλ            {A['dlam'].max() * 1e3:7.2f}        - mWb",
                "",
                "Uwaga: moment może być < T* także z AW -",
                "przy dużej prędkości fizycznie brakuje",
                "napięcia (obszar stałej mocy). AW nie",
                "tworzy napięcia - zapobiega utracie",
                "kontroli prądu."]


# =============================================================================
# OKNO GŁÓWNE
# =============================================================================

TABS = [("E1 Wymagania EV", TabE1), ("E2 Bieguny", TabE2), ("E3 SEM", TabE3),
        ("E4 Ćw.12.1", TabE4), ("E5 T-n (Excel)", TabE5), ("E6 Zmienne L", TabE6),
        ("E7 Skos", TabE7), ("E8 Straty/η", TabE8), ("E9 Magnes", TabE9),
        ("E10 Mostki", TabE10), ("E11 Uzwojenie", TabE11), ("E12 Pomiar SEM", TabE12),
        ("E13 Pomiar L", TabE13), ("E14 Optimum", TabE14), ("E15 LUT i Vdc", TabE15),
        ("E16 Anti-windup", TabE16)]


def style_app(root):
    """Ciemny motyw dla widżetów ttk."""
    st = ttk.Style(root)
    st.theme_use("clam")
    st.configure(".", background=C["bg"], foreground=C["text"], fieldbackground=C["surf"],
                 bordercolor=C["surf2"], lightcolor=C["surf"], darkcolor=C["mantle"], font=("Segoe UI", 9))
    st.configure("TFrame", background=C["bg"])
    st.configure("TLabel", background=C["bg"], foreground=C["text"])
    st.configure("H.TLabel", background=C["bg"], foreground=C["cyan"], font=("Segoe UI", 11, "bold"))
    st.configure("V.TLabel", background=C["bg"], foreground=C["cyan"], font=("Consolas", 9, "bold"))
    st.configure("Horizontal.TScale", background=C["cyan"], troughcolor=C["surf"], bordercolor=C["surf2"],
                 lightcolor=C["cyan"], darkcolor=C["teal"])
    st.configure("TButton", background=C["surf"], foreground=C["text"], padding=4)
    st.map("TButton", background=[("active", C["surf2"])])
    st.configure("TNotebook", background=C["mantle"], borderwidth=0)
    st.configure("TNotebook.Tab", background=C["surf"], foreground=C["sub"], padding=(3, 3), font=("Segoe UI", 8))
    st.map("TNotebook.Tab", background=[("selected", C["bg"])], foreground=[("selected", C["cyan"])])
    st.configure("TCombobox", fieldbackground=C["surf"], background=C["surf"], foreground=C["text"],
                 arrowcolor=C["cyan"])
    st.map("TCombobox", fieldbackground=[("readonly", C["surf"])], foreground=[("readonly", C["text"])])
    root.option_add("*TCombobox*Listbox.background", C["surf"])
    root.option_add("*TCombobox*Listbox.foreground", C["text"])
    st.configure("Treeview", background=C["mantle"], fieldbackground=C["mantle"], foreground=C["text"], rowheight=20)
    st.configure("Treeview.Heading", background=C["surf"], foreground=C["cyan"])
    st.configure("TPanedwindow", background=C["surf2"])
    st.configure("Vertical.TScrollbar", background=C["surf"], troughcolor=C["mantle"], arrowcolor=C["cyan"])


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Projekt i sterowanie silnikiem EV (IPMSM) - rozdział 12: przykłady i zadania")
        self.geometry("1560x960")
        self.configure(bg=C["bg"])
        style_app(self)
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
