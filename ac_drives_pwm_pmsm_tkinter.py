"""
Sterowanie silnikami AC i pojazdy elektryczne - PWM oraz sterowanie PMSM
========================================================================
Interaktywna aplikacja Tkinter + Matplotlib z przykładami z dwóch rozdziałów:
  * "AC motor control and electrical vehicle - PWM"
  * "AC motor control and electrical vehicle - PMSM control"

Każda zakładka = jeden przykład: suwaki (parametry), wykresy (wyniki) oraz
objaśnienie napisane językiem zrozumiałym dla ucznia liceum.

Zakładki PWM:
  P1. Sinusoidalne PWM jednej gałęzi (nośna, napięcie, widmo)
  P2. Trójfazowy falownik SPWM + prąd obciążenia RL-E
  P3. Wstrzykiwanie 3. harmonicznej / min-max (+15,5 % napięcia)
  P4. Modulacja wektorowa SVPWM (sześciokąt, czasy T1, T2, T0)
  P5. Przemodulowanie i praca blokowa (six-step)
  P6. Czas martwy (dead-time) - błąd napięcia

Zakładki PMSM:
  M1. Model dq w stanie ustalonym (napięcia, moc, wykres wskazowy)
  M2. Moment vs kąt prądu, MTPA (SPMSM i IPMSM)
  M3. Okręgi prądu i elipsy napięcia - osłabianie pola
  M4. Charakterystyka moment/moc - prędkość (obwiednia napędu EV)
  M5. Sterowanie polowo-zorientowane FOC - symulacja dynamiczna
  M6. Pojazd elektryczny: opory ruchu, przyspieszanie 0-100 km/h

Uruchomienie:  python ac_drives_pwm_pmsm_tkinter.py
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
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

PI = math.pi
SQ3 = math.sqrt(3.0)


# =============================================================================
# RDZEŃ OBLICZENIOWY - PWM
# =============================================================================

def triangle(t, fc):
    """Nośna trójkątna -1..+1 o częstotliwości fc."""
    x = (t * fc) % 1.0
    return 4.0 * np.abs(x - 0.5) - 1.0


def spwm_leg(ma, mf, vdc, f1=50.0, n=20000, ref=None):
    """Jedna gałąź falownika: porównanie sinusoidy z nośną.
    Zwraca t, ref, carrier, napięcie biegunowe vAo (względem środka DC)."""
    t = np.linspace(0.0, 1.0 / f1, n, endpoint=False)
    fc = mf * f1
    if ref is None:
        ref = ma * np.sin(2 * PI * f1 * t)
    car = triangle(t, fc)
    vao = np.where(ref >= car, vdc / 2, -vdc / 2)
    return t, ref, car, vao


def spectrum(v):
    """Amplitudy harmonicznych (szczytowe) sygnału z jednego okresu."""
    n = len(v)
    c = np.fft.rfft(v) / n * 2.0
    c[0] /= 2.0
    return np.abs(c)


def thd(amp, nmax=None):
    a = amp[1:nmax] if nmax else amp[1:]
    return math.sqrt(max(np.sum(a[1:] ** 2), 0.0)) / a[0] if a[0] > 0 else 0.0


def three_phase_pwm(ma, mf, vdc, f1=50.0, n=20000, mode="spwm"):
    """Trzy gałęzie; mode: 'spwm', 'thi' (1/6 3. harm.), 'minmax' (≈SVPWM)."""
    t = np.linspace(0.0, 1.0 / f1, n, endpoint=False)
    w = 2 * PI * f1 * t
    refs = np.array([ma * np.sin(w - k * 2 * PI / 3) for k in range(3)])
    if mode == "thi":
        refs = refs + ma / 6.0 * np.sin(3 * w)
    elif mode == "minmax":
        refs = refs - 0.5 * (refs.max(axis=0) + refs.min(axis=0))
    car = triangle(t, mf * f1)
    poles = np.where(refs >= car, vdc / 2, -vdc / 2)
    return t, refs, car, poles


def rle_current(v_phase, t, R, L, E_amp, f1):
    """Prąd fazowy obciążenia R-L-E (E - sinusoidalna SEM, np. silnik)
    całkowany metodą dokładnej dyskretyzacji, 3 okresy aby dojść do stanu ustalonego."""
    dt = t[1] - t[0]
    e = E_amp * np.sin(2 * PI * f1 * t - 0.3)
    a = math.exp(-R * dt / L)
    b = (1 - a) / R
    i = 0.0
    out = np.zeros_like(v_phase)
    for _ in range(3):
        for k in range(len(t)):
            i = a * i + b * (v_phase[k] - e[k])
            out[k] = i
    return out


def fundamental_vs_ma(ma_arr, mf=21, vdc=1.0):
    """Amplituda 1. harmonicznej napięcia biegunowego (odniesiona do Vdc/2)."""
    res = []
    for ma in ma_arr:
        _, _, _, v = spwm_leg(ma, mf, vdc, n=8000)
        res.append(spectrum(v)[1] / (vdc / 2))
    return np.array(res)


def svpwm_times(vref, theta_deg, vdc, ts):
    """Czasy przełączeń SVPWM. vref - amplituda napięcia fazowego."""
    th = math.radians(theta_deg % 360.0)
    sector = int(th // (PI / 3)) + 1
    a = th - (sector - 1) * PI / 3
    k = SQ3 * ts * vref / vdc
    t1 = k * math.sin(PI / 3 - a)
    t2 = k * math.sin(a)
    over = (t1 + t2) > ts
    if over:  # przycięcie do boku sześciokąta
        s = t1 + t2
        t1, t2 = t1 * ts / s, t2 * ts / s
    t0 = max(ts - t1 - t2, 0.0)
    return sector, math.degrees(a), t1, t2, t0, over


# =============================================================================
# RDZEŃ OBLICZENIOWY - PMSM
# =============================================================================

def pmsm_steady(id_, iq, we, Rs, Ld, Lq, psi):
    vd = Rs * id_ - we * Lq * iq
    vq = Rs * iq + we * (Ld * id_ + psi)
    return vd, vq


def pmsm_torque(id_, iq, p, Ld, Lq, psi):
    return 1.5 * p * (psi * iq + (Ld - Lq) * id_ * iq)


def mtpa_id(Is, Ld, Lq, psi):
    """Prąd Id dla MTPA przy amplitudzie prądu Is."""
    dL = Lq - Ld
    if abs(dL) < 1e-12:
        return 0.0 * Is
    return psi / (4 * dL) - np.sqrt(psi ** 2 / (16 * dL ** 2) + Is ** 2 / 2)


def torque_envelope(speeds_rpm, p, Ld, Lq, psi, Imax, Vmax, ngrid=161):
    """Maksymalny moment przy ograniczeniu prądu i napięcia (Rs pominięte).
    Przeszukiwanie siatki punktów (Id, Iq) wewnątrz okręgu prądu."""
    ids = np.linspace(-Imax, 0, ngrid)
    iqs = np.linspace(0, Imax, ngrid)
    ID, IQ = np.meshgrid(ids, iqs)
    ok_i = ID ** 2 + IQ ** 2 <= Imax ** 2
    T = pmsm_torque(ID, IQ, p, Ld, Lq, psi)
    flux2 = (psi + Ld * ID) ** 2 + (Lq * IQ) ** 2
    tmax, idopt, iqopt = [], [], []
    for n in speeds_rpm:
        we = n * 2 * PI / 60 * p
        ok = ok_i & (we ** 2 * flux2 <= Vmax ** 2)
        if not ok.any():
            tmax.append(0.0); idopt.append(np.nan); iqopt.append(np.nan)
            continue
        Tm = np.where(ok, T, -np.inf)
        k = np.unravel_index(np.argmax(Tm), Tm.shape)
        tmax.append(max(Tm[k], 0.0)); idopt.append(ID[k]); iqopt.append(IQ[k])
    return np.array(tmax), np.array(idopt), np.array(iqopt)


def foc_simulate(P, t_end=0.6, dt=2e-5):
    """Symulacja FOC: regulator prędkości PI -> iq*, id*=0 (lub MTPA),
    regulatory prądu PI z odsprzęganiem, ograniczenie napięcia |v|<=Vdc/√3."""
    p, Rs, Ld, Lq, psi, J, B = (P[k] for k in ("p", "Rs", "Ld", "Lq", "psi", "J", "B"))
    Vmax = P["Vdc"] / SQ3
    wc = 2 * PI * P["bw_i"]           # pasmo pętli prądu
    kpd, kid = wc * Ld, wc * Rs
    kpq, kiq = wc * Lq, wc * Rs
    ws = 2 * PI * P["bw_w"]           # pasmo pętli prędkości
    kt = 1.5 * p * psi
    kpw = ws * J / kt
    kiw = kpw * ws / 5
    n_steps = int(t_end / dt)
    id_ = iq = wm = 0.0
    xd = xq = xw = 0.0
    dec = max(1, n_steps // 3000)
    rec = {k: [] for k in ("t", "wref", "wm", "id", "iq", "iqref", "Te", "TL", "vd", "vq")}
    for k in range(n_steps):
        t = k * dt
        wref = P["n_ref"] * 2 * PI / 60 if t >= 0.02 else 0.0
        TL = P["TL"] if t >= P["t_load"] else 0.0
        # pętla prędkości
        ew = wref - wm
        iq_ref = kpw * ew + xw
        iq_lim = P["Imax"]
        if abs(iq_ref) > iq_lim:
            iq_ref = math.copysign(iq_lim, iq_ref)
        else:
            xw += kiw * ew * dt
        id_ref = 0.0
        we = p * wm
        # pętle prądu + odsprzęganie
        ed, eq = id_ref - id_, iq_ref - iq
        vd = kpd * ed + xd - we * Lq * iq
        vq = kpq * eq + xq + we * (Ld * id_ + psi)
        vm = math.hypot(vd, vq)
        if vm > Vmax:
            vd, vq = vd * Vmax / vm, vq * Vmax / vm
        else:
            xd += kid * ed * dt
            xq += kiq * eq * dt
        # model silnika (Euler)
        did = (vd - Rs * id_ + we * Lq * iq) / Ld
        diq = (vq - Rs * iq - we * (Ld * id_ + psi)) / Lq
        Te = 1.5 * p * (psi * iq + (Ld - Lq) * id_ * iq)
        dwm = (Te - TL - B * wm) / J
        id_ += did * dt
        iq += diq * dt
        wm += dwm * dt
        if k % dec == 0:
            for key, val in (("t", t), ("wref", wref), ("wm", wm), ("id", id_), ("iq", iq),
                             ("iqref", iq_ref), ("Te", Te), ("TL", TL), ("vd", vd), ("vq", vq)):
                rec[key].append(val)
    return {k: np.array(v) for k, v in rec.items()}


def ev_forces(v, m, Crr, Cd, A, grade_pct, rho=1.2, g=9.81):
    a = math.atan(grade_pct / 100.0)
    F_roll = m * g * Crr * math.cos(a) * np.ones_like(v)
    F_aero = 0.5 * rho * Cd * A * v ** 2
    F_grade = m * g * math.sin(a) * np.ones_like(v)
    return F_roll, F_aero, F_grade


def ev_motor_curve(n_rpm, T_peak, P_peak_kw, n_max):
    """Idealna obwiednia: stały moment do prędkości bazowej, potem stała moc."""
    w = np.maximum(n_rpm * 2 * PI / 60, 1e-6)
    T = np.minimum(T_peak, P_peak_kw * 1e3 / w)
    return np.where(n_rpm <= n_max, T, 0.0)


def ev_accelerate(m, Crr, Cd, A, grade, r, G, eta, T_peak, P_kw, n_max, t_end=30.0, dt=0.01):
    v = 0.0
    ts, vs, Fs = [], [], []
    t = 0.0
    t100 = None
    m_eff = m * 1.05  # 5% na bezwładność elementów wirujących
    while t < t_end:
        n_mot = v / r * G * 60 / (2 * PI)
        Tm = float(ev_motor_curve(np.array([n_mot]), T_peak, P_kw, n_max)[0])
        F_tr = Tm * G * eta / r
        Fr, Fa, Fg = ev_forces(np.array([v]), m, Crr, Cd, A, grade)
        acc = (F_tr - Fr[0] - Fa[0] - Fg[0]) / m_eff
        v = max(v + acc * dt, 0.0)
        t += dt
        if t100 is None and v >= 100 / 3.6:
            t100 = t
        ts.append(t); vs.append(v); Fs.append(F_tr)
    return np.array(ts), np.array(vs), np.array(Fs), t100


# =============================================================================
# OBJAŚNIENIA (dla ucznia liceum)
# =============================================================================

EXPL = {
"P1": """PRZYKŁAD P1 - Sinusoidalne PWM jednej gałęzi falownika

O CO CHODZI?
Akumulator w samochodzie elektrycznym daje napięcie STAŁE (np. 350 V), a silnik
potrzebuje napięcia PRZEMIENNEGO (sinusoidy). Falownik to "szybki przełącznik":
tranzystor (IGBT/MOSFET) łączy wyjście albo z +Vdc/2, albo z -Vdc/2. Nie potrafi dać
"trochę napięcia" - tylko wszystko albo nic.

SZTUCZKA PWM (Pulse Width Modulation - modulacja szerokości impulsów):
Porównujemy sinusoidę odniesienia (niebieska) z szybkim trójkątem - nośną (szara).
  - sinusoida > trójkąt  -> tranzystor górny ON  -> +Vdc/2
  - sinusoida < trójkąt  -> tranzystor dolny ON  -> -Vdc/2
Gdy sinusoida jest wysoko, impulsy dodatnie są szerokie; gdy nisko - wąskie.
ŚREDNIE napięcie w każdym okresie nośnej odtwarza sinusoidę! Silnik (indukcyjność)
działa jak filtr - "uśrednia" impulsy, więc prąd jest prawie sinusoidalny.

DWA WAŻNE WSPÓŁCZYNNIKI:
  ma = Û_ref / Û_nośnej  - współczynnik modulacji amplitudy (jak "głośność")
  mf = f_nośnej / f_1     - współczynnik modulacji częstotliwości

WZÓR (zakres liniowy ma <= 1):   V_1(szczyt) = ma * Vdc/2
Przykład: Vdc = 350 V, ma = 0,8 -> V_1 = 0,8*175 = 140 V (szczytowe).

WIDMO (dolny wykres): harmoniczne pojawiają się w "pakietach" wokół mf, 2mf, 3mf...
Im wyższe mf, tym dalej od podstawowej są te śmieci - łatwiej je odfiltrować,
ale tranzystory częściej przełączają (większe straty łączeniowe). To klasyczny
kompromis inżynierski! Wybieraj mf nieparzyste i podzielne przez 3 (np. 15, 21)
- wtedy przebieg jest symetryczny i harmoniczne rzędu 3mf znikają w napięciu międzyprzewodowym.

SPRÓBUJ: zwiększ ma powyżej 1 - sinusoida "wychodzi" poza trójkąt, pojawiają się
pominięte impulsy (przemodulowanie) i niskie harmoniczne (5, 7...).
""",
"P2": """PRZYKŁAD P2 - Trójfazowy falownik SPWM i prąd silnika

Falownik trójfazowy = 3 gałęzie z P1, sinusoidy przesunięte o 120°.
Napięcie między przewodami (np. A-B) = vAo - vBo. Ma TRZY poziomy: +Vdc, 0, -Vdc.

WZÓR na wartość skuteczną napięcia międzyprzewodowego (podstawowa harmoniczna):
     V_LL1(rms) = (√3/(2√2)) * ma * Vdc ≈ 0,612 * ma * Vdc
Przykład: Vdc = 350 V, ma = 1 -> V_LL1 = 214 V rms.

PRĄD: podłączamy model silnika - rezystancja R, indukcyjność L i SEM E
(silnik obracając się wytwarza własne napięcie). Indukcyjność "wygładza" prąd -
widać tylko mały "ząbek" (tętnienia) o częstotliwości nośnej.

Tętnienia prądu maleją, gdy:
  - rośnie częstotliwość nośnej fc (krótsze impulsy),
  - rośnie indukcyjność L.
Dlatego w EV stosuje się fc = 8...20 kHz (ponad pasmem słyszalności - silnik nie "piszczy").

THD (Total Harmonic Distortion) - współczynnik zniekształceń: ile "śmieci" jest
w stosunku do przydatnej sinusoidy. Prąd ma dużo mniejsze THD niż napięcie!
""",
"P3": """PRZYKŁAD P3 - Wstrzykiwanie 3. harmonicznej (THI) i metoda min-max

PROBLEM: w zwykłym SPWM (ma <= 1) napięcie międzyprzewodowe wynosi maks.
0,612*Vdc (rms), a teoretycznie można uzyskać 0,707*Vdc. Marnujemy ~15% akumulatora!

SPRYTNY POMYSŁ: do KAŻDEJ z trzech faz dodajemy TEN SAM sygnał (tzw. składowa
wspólna - np. 1/6 trzeciej harmonicznej). Silnik ma gwiazdę bez przewodu neutralnego,
więc widzi tylko RÓŻNICE napięć faz - a różnica tego samego sygnału = 0!
Sygnał spłaszcza "garb" sinusoidy (wykres - kształt siodła), dzięki czemu można
podnieść ma do 2/√3 = 1,155 bez wychodzenia poza trójkąt nośnej.

ZYSK: +15,5% napięcia -> wyższa prędkość maksymalna silnika przy tej samej baterii.

Metoda "min-max" (odjęcie średniej z maksimum i minimum trzech faz) daje
dokładnie to samo co modulacja wektorowa SVPWM (przykład P4).

SPRÓBUJ: ustaw ma = 1,15. W SPWM widać przemodulowanie (przycięcia), a w THI
i min-max - czysta sinusoida napięcia międzyprzewodowego.
""",
"P4": """PRZYKŁAD P4 - Modulacja wektora przestrzennego (SVPWM)

Falownik trójfazowy ma 3 przełączniki -> 2^3 = 8 stanów. Sześć z nich (V1...V6)
to "strzałki" (wektory) o długości 2/3*Vdc w kierunkach co 60°, a dwa (V0, V7)
to wektory zerowe (wszystkie fazy do + lub do -). Końce strzałek tworzą SZEŚCIOKĄT.

CEL: wytworzyć wirującą strzałkę Vref (jak wskazówka zegara) - to ona tworzy wirujące
pole magnetyczne w silniku.

JAK? W każdym krótkim okresie Ts "mieszamy" dwa sąsiednie wektory i wektor zerowy,
tak jak mieszamy farby:   Vref * Ts = V1 * T1 + V2 * T2 (+ zero * T0)
Wzory (α - kąt w sektorze):
   T1 = √3 * Ts * Vref/Vdc * sin(60° - α)
   T2 = √3 * Ts * Vref/Vdc * sin(α)
   T0 = Ts - T1 - T2
Przykład: Vdc=350 V, Vref=150 V, α=20°, Ts=100 µs:
   T1 = 1,732*100*0,4286*sin40° = 47,7 µs,  T2 = 25,4 µs,  T0 = 26,9 µs.

GRANICA liniowa = okrąg wpisany w sześciokąt: Vref_max = Vdc/√3 (to te same +15,5% co w P3).
Gdy Vref wychodzi poza okrąg - T0 < 0, czasy trzeba przyciąć (czerwony alarm).

Prawy wykres pokazuje symetryczny wzór przełączeń w okresie Ts (V0-V1-V2-V7-V2-V1-V0):
każda zmiana stanu przełącza tylko JEDEN tranzystor -> mniejsze straty.
""",
"P5": """PRZYKŁAD P5 - Przemodulowanie i praca blokowa (six-step)

Co się stanie, gdy ma > 1? Sinusoida odniesienia jest wyższa niż trójkąt i przez
część okresu tranzystor jest cały czas włączony. Napięcie rośnie, ale NIELINIOWO
(krzywa wygina się) i pojawiają się niskie harmoniczne 5, 7, 11, 13 - silnik
się grzeje i drga (pulsacje momentu 6*f).

Przy bardzo dużym ma dostajemy falę prostokątną (six-step, 6 stanów na okres):
     V_1(szczyt, fazowe) = (4/π) * Vdc/2 = 1,273 * Vdc/2  (MAKSIMUM możliwe!)
  czyli V_LL1(rms) = (√6/π) * Vdc ≈ 0,78 * Vdc.

Zakresy (wykres górny):
  - ma <= 1        liniowy SPWM (0,612 Vdc)
  - ma <= 1,155    liniowy z THI/SVPWM (0,707 Vdc)
  - dalej          przemodulowanie -> six-step (0,78 Vdc)

W samochodach elektrycznych przemodulowanie stosuje się przy najwyższych
prędkościach (autostrada), by wycisnąć maksimum napięcia z baterii.
""",
"P6": """PRZYKŁAD P6 - Czas martwy (dead-time)

Tranzystory górny i dolny w gałęzi NIGDY nie mogą przewodzić jednocześnie -
byłoby to zwarcie akumulatora (shoot-through) i wybuch tranzystora! Tranzystor
wyłącza się z opóźnieniem, więc sterownik wprowadza "czas martwy" td (np. 1-3 µs),
kiedy OBA są wyłączone.

Skutek: w czasie td napięcie wyjściowe zależy od KIERUNKU PRĄDU (przewodzi dioda):
  - prąd dodatni -> dioda dolna -> tracimy kawałek impulsu dodatniego,
  - prąd ujemny  -> dioda górna -> zyskujemy.
Średni błąd napięcia:   ΔV = td * fc * Vdc * sign(i)
Przykład: td = 2 µs, fc = 10 kHz, Vdc = 350 V ->  ΔV = 2e-6*1e4*350 = 7 V.

Błąd jest prostokątny (zmienia znak z prądem) -> zniekształca prąd przy przejściu
przez zero i dodaje harmoniczne 5, 7. Najgorzej przy MAŁYCH prędkościach (małe napięcie
użyteczne, a błąd stały). Dlatego sterowniki EV mają "kompensację czasu martwego".
""",
"M1": """PRZYKŁAD M1 - Model PMSM w układzie dq (stan ustalony)

PMSM = silnik synchroniczny z magnesami trwałymi (Tesla Model 3, Toyota, większość EV).
Wirnik ma magnesy, stojan - trzy uzwojenia zasilane z falownika.

UKŁAD dq: zamiast oglądać 3 sinusoidalne prądy, "wsiadamy na wirnik" i obracamy się
razem z nim. Z tej perspektywy prądy są STAŁE (jak z karuzeli - drugi wagonik stoi
względem nas). Oś d - wzdłuż magnesu, oś q - prostopadle.
   Id - prąd "strumieniotwórczy" (osłabia lub wzmacnia magnes)
   Iq - prąd "momentotwórczy" (ciągnie wirnik - daje moment)

RÓWNANIA (stan ustalony, ωe = p * ωm - prędkość elektryczna):
   Vd = Rs*Id - ωe*Lq*Iq
   Vq = Rs*Iq + ωe*(Ld*Id + ψf)       <- ωe*ψf to SEM (napięcie indukowane)
   Te = 1,5*p*[ψf*Iq + (Ld-Lq)*Id*Iq]
   Pmech = Te*ωm ,   Pel = 1,5*(Vd*Id + Vq*Iq)

PRZYKŁAD LICZBOWY (wartości domyślne): p=4, ψf=0,08 Wb, Iq=200 A, Id=0, n=3000 obr/min:
  ωm = 314 rad/s, ωe = 1257 rad/s, Te = 1,5*4*0,08*200 = 96 Nm, Pmech ≈ 30 kW.

Wykres wskazowy pokazuje wektory prądu I, strumienia ψ i napięcia V. Zobacz, jak
napięcie rośnie z prędkością - aż dojdzie do granicy falownika (czerwony okrąg Vdc/√3)!
""",
"M2": """PRZYKŁAD M2 - Moment w funkcji kąta prądu i MTPA

Prąd stojana ma stałą amplitudę Is (ogranicza ją falownik i nagrzewanie), ale możemy
wybrać jego KIERUNEK (kąt β od osi q):  Id = -Is*sin β,  Iq = Is*cos β.

Moment ma dwa składniki:
  1) moment od magnesu:    1,5 p ψf Iq         (max przy β = 0)
  2) moment reluktancyjny: 1,5 p (Ld-Lq) Id Iq (w IPMSM Lq > Ld, więc przy Id<0 jest DODATNI)
Reluktancja = wirnik "chce" ustawić się tak, by strumień płynął najłatwiejszą drogą
(jak spinacz przyciągany do magnesu).

SPMSM (magnesy na powierzchni, Ld = Lq): tylko składnik 1 -> najlepiej Id = 0.
IPMSM (magnesy wewnątrz, Lq > Ld): opłaca się dodać trochę UJEMNEGO Id -
suma obu składników jest większa!

MTPA = Maximum Torque Per Ampere - najwięcej momentu z każdego ampera.
     Id_MTPA = ψf/(4(Lq-Ld)) - sqrt( ψf²/(16(Lq-Ld)²) + Is²/2 )
Korzyść: mniejszy prąd dla tego samego momentu -> mniej strat w miedzi -> większy zasięg auta.
Zmieniaj Lq/Ld suwakiem i patrz, jak przesuwa się optimum (zielona kropka).
""",
"M3": """PRZYKŁAD M3 - Ograniczenia prądu i napięcia, osłabianie pola

W płaszczyźnie (Id, Iq) każdy punkt pracy to jedna kropka. Mamy dwa ograniczenia:
  1) PRĄD:     Id² + Iq² <= Imax²          -> okrąg (czerwony)
  2) NAPIĘCIE: (ψf + Ld Id)² + (Lq Iq)² <= (Vmax/ωe)²  -> elipsa (niebieska)
Elipsa ma środek w punkcie (-ψf/Ld, 0) i KURCZY SIĘ, gdy prędkość rośnie!

Punkt pracy musi leżeć w części wspólnej okręgu i elipsy. Czarne linie to linie
stałego momentu (hiperbole).

OSŁABIANIE POLA (field weakening): przy dużych prędkościach SEM ωe*ψf jest większa
niż napięcie baterii. Rozwiązanie - ujemny Id "osłabia" magnes (Ld*Id odejmuje się od ψf),
więc napięcie spada i silnik może kręcić się szybciej. Cena: mniejszy moment.

Punkt ψf/Ld (punkt charakterystyczny) - jeśli leży wewnątrz okręgu prądu, silnik
teoretycznie może osiągnąć nieskończoną prędkość (świetne dla EV).
Zmieniaj prędkość i obserwuj, jak elipsa "zjada" dostępne punkty pracy.
""",
"M4": """PRZYKŁAD M4 - Charakterystyka moment-prędkość napędu EV

Liczymy numerycznie: dla każdej prędkości szukamy punktu (Id, Iq), który daje
NAJWIĘKSZY moment, spełniając oba ograniczenia z M3.

Wynik - dwa obszary:
  1) STAŁY MOMENT (0 ... n_bazowa): ogranicza tylko prąd. Silnik daje pełny moment
     - świetne ruszanie spod świateł.
  2) STAŁA MOC / osłabianie pola (> n_bazowa): ogranicza napięcie. Moment spada ~1/n,
     a moc P = T*ω pozostaje prawie stała.

Prędkość bazowa zależy od napięcia baterii - dlatego nowe auta (Porsche Taycan,
Hyundai Ioniq 5) przechodzą na 800 V: wyższa prędkość bazowa, mniejszy prąd,
cieńsze kable, szybsze ładowanie.

Na dolnym wykresie - jak zmieniają się Id i Iq: powyżej n_bazowej Id robi się coraz
bardziej ujemny (osłabianie pola).
""",
"M5": """PRZYKŁAD M5 - Sterowanie polowo-zorientowane (FOC) - symulacja

FOC to "mózg" falownika. Struktura kaskadowa (jak kierowca i jego nogi):
  PĘTLA ZEWNĘTRZNA (wolna) - regulator PI prędkości: porównuje prędkość zadaną z
     rzeczywistą i mówi, ile momentu potrzeba -> prąd zadany Iq*.
  PĘTLE WEWNĘTRZNE (szybkie) - regulatory PI prądu Id i Iq: liczą napięcia Vd, Vq.
     Id* = 0 (dla SPMSM). Dodane "odsprzęganie" (-ωLqIq, +ω(LdId+ψf)) usuwa
     wzajemny wpływ osi d i q.
  Transformacje Parka/Clarke i SVPWM zamieniają Vd, Vq na impulsy tranzystorów.

Regulator PI: P - reaguje na bieżący błąd, I - "pamięta" błąd i usuwa go do zera.
Nastawy dobrane metodą "kompensacji bieguna": Kp = ωc*L, Ki = ωc*R.

SCENARIUSZ: t = 0,02 s rozkaz rozpędzenia, później skokowe obciążenie TL
(np. wjazd pod górkę). Obserwuj:
  - Iq dochodzi do ograniczenia Imax podczas rozpędzania (pełny moment),
  - po skoku obciążenia prędkość chwilowo spada, a regulator ją przywraca,
  - Id pozostaje ~0 - dzięki odsprzęganiu.
Zwiększ pasmo prędkości - szybsza reakcja, ale ryzyko przeregulowania.
""",
"M6": """PRZYKŁAD M6 - Pojazd elektryczny: siły oporu i przyspieszanie

Aby samochód jechał, siła napędowa musi pokonać:
  F_toczenia = m g Crr cos α     (ugniatanie opon, ~stała)
  F_aero     = ½ ρ Cd A v²       (opór powietrza - rośnie z KWADRATEM prędkości!)
  F_wzniesienia = m g sin α      (jazda pod górę)

Siła na kołach z silnika:  F = T_silnika * G * η / r   (G - przełożenie, r - promień koła)
Prędkość silnika:          n = v/r * G * 60/(2π)

PRZYKŁAD: m = 1800 kg, Crr = 0,01, Cd = 0,23, A = 2,2 m², v = 120 km/h (33,3 m/s):
  F_toczenia = 177 N,  F_aero = 0,5*1,2*0,23*2,2*33,3² = 337 N  -> razem 514 N
  Moc = F*v = 17 kW. Przy 50 km/h tylko ~4 kW - dlatego EV mają największy zasięg w mieście!

Przyspieszenie: a = (F_napędowa - F_opory) / m_eff  (m_eff = 1,05 m - wirujące części).
Symulujemy krok po kroku (metoda Eulera) i liczymy czas 0-100 km/h.
Punkt przecięcia krzywej napędu z krzywą oporów = prędkość maksymalna.

SPRÓBUJ: zwiększ przełożenie G - lepsze przyspieszenie, ale niższa prędkość maksymalna
(silnik szybciej dochodzi do n_max). Klasyczny kompromis - stąd jednobiegowa przekładnia ~9:1.
""",
}


# =============================================================================
# GUI - klasa bazowa zakładki
# =============================================================================

class ExampleTab(ttk.Frame):
    """Zakładka: lewy panel suwaków + wyniki liczbowe, prawy - wykresy,
    dół - objaśnienie. Podklasy definiują PARAMS, key, build_axes(), update_plot()."""
    PARAMS = []   # (nazwa, etykieta, min, max, domyślna, krok)
    key = ""
    nrows, ncols = 2, 1

    def __init__(self, master):
        super().__init__(master)
        self.vars = {}
        self._pending = None
        pw = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        pw.pack(fill=tk.BOTH, expand=True)
        left = ttk.Frame(pw, width=300)
        right = ttk.Frame(pw)
        pw.add(left, weight=0)
        pw.add(right, weight=1)

        ttk.Label(left, text="Parametry", font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=6, pady=(6, 2))
        for name, label, lo, hi, val, res in self.PARAMS:
            fr = ttk.Frame(left)
            fr.pack(fill=tk.X, padx=6, pady=1)
            var = tk.DoubleVar(value=val)
            self.vars[name] = var
            lab = ttk.Label(fr, text=label, width=24)
            lab.pack(anchor="w")
            row = ttk.Frame(fr)
            row.pack(fill=tk.X)
            sc = tk.Scale(row, from_=lo, to=hi, resolution=res, orient=tk.HORIZONTAL,
                          variable=var, showvalue=True, length=200,
                          command=lambda _e: self.schedule())
            sc.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.extra_controls(left)
        ttk.Button(left, text="Przywróć domyślne", command=self.reset).pack(fill=tk.X, padx=6, pady=6)
        ttk.Label(left, text="Wyniki", font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=6)
        self.result = tk.Text(left, height=14, width=36, font=("Consolas", 9), bg="#f4f6fa")
        self.result.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))

        vpw = ttk.PanedWindow(right, orient=tk.VERTICAL)
        vpw.pack(fill=tk.BOTH, expand=True)
        figfr = ttk.Frame(vpw)
        self.fig = Figure(figsize=(9, 6), dpi=90, tight_layout=True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=figfr)
        NavigationToolbar2Tk(self.canvas, figfr).update()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        txtfr = ttk.Frame(vpw)
        self.expl = ScrolledText(txtfr, height=11, wrap=tk.WORD, font=("Segoe UI", 10), bg="#fffdf3")
        self.expl.pack(fill=tk.BOTH, expand=True)
        self.expl.insert("1.0", EXPL.get(self.key, ""))
        self.expl.configure(state=tk.DISABLED)
        vpw.add(figfr, weight=3)
        vpw.add(txtfr, weight=1)
        self.refresh()

    def extra_controls(self, parent):
        pass

    def reset(self):
        for name, _l, _lo, _hi, val, _r in self.PARAMS:
            self.vars[name].set(val)
        self.refresh()

    def p(self, name):
        return self.vars[name].get()

    def schedule(self):
        if self._pending:
            self.after_cancel(self._pending)
        self._pending = self.after(120, self.refresh)

    def refresh(self):
        self._pending = None
        self.fig.clear()
        lines = self.update_plot() or []
        self.result.delete("1.0", tk.END)
        self.result.insert("1.0", "\n".join(lines))
        self.canvas.draw_idle()

    def update_plot(self):
        raise NotImplementedError


# =============================================================================
# ZAKŁADKI PWM
# =============================================================================

class TabP1(ExampleTab):
    key = "P1"
    PARAMS = [("ma", "ma (modulacja amplitudy)", 0.0, 2.0, 0.8, 0.01),
              ("mf", "mf (f_nośnej / f_1)", 3, 45, 15, 1),
              ("vdc", "Vdc [V]", 50, 800, 350, 10),
              ("f1", "f_1 [Hz]", 10, 200, 50, 1)]

    def update_plot(self):
        ma, mf, vdc, f1 = self.p("ma"), int(self.p("mf")), self.p("vdc"), self.p("f1")
        t, ref, car, vao = spwm_leg(ma, mf, vdc, f1)
        amp = spectrum(vao)
        a1 = self.fig.add_subplot(3, 1, 1)
        a1.plot(t * 1e3, car, color="0.6", lw=0.8, label="nośna")
        a1.plot(t * 1e3, ref, "b", lw=1.5, label="odniesienie")
        a1.set_ylabel("sygnały [p.u.]"); a1.legend(loc="upper right", fontsize=8); a1.grid(alpha=.3)
        a1.set_title("Porównanie sinusoidy z nośną trójkątną")
        a2 = self.fig.add_subplot(3, 1, 2, sharex=a1)
        a2.plot(t * 1e3, vao, "k", lw=0.8, label="vAo (impulsy)")
        a2.plot(t * 1e3, amp[1] * np.sin(2 * PI * f1 * t + np.angle(np.fft.rfft(vao)[1]) + PI / 2),
                "r", lw=1.6, label="1. harmoniczna")
        a2.set_ylabel("vAo [V]"); a2.set_xlabel("t [ms]"); a2.legend(loc="upper right", fontsize=8); a2.grid(alpha=.3)
        a3 = self.fig.add_subplot(3, 1, 3)
        h = np.arange(min(len(amp), 4 * mf + 10))
        a3.bar(h, amp[:len(h)] / (vdc / 2), color="tab:purple", width=0.7)
        a3.set_xlabel("rząd harmonicznej h"); a3.set_ylabel("V_h / (Vdc/2)"); a3.grid(alpha=.3)
        a3.set_title("Widmo napięcia biegunowego")
        return [f"f_nośnej   = {mf * f1:.0f} Hz",
                f"V1 (FFT)   = {amp[1]:.1f} V szczyt.",
                f"V1 teoria  = {ma * vdc / 2:.1f} V (ma<=1)",
                f"V1/(Vdc/2) = {amp[1] / (vdc / 2):.3f}",
                f"THD vAo    = {100 * thd(amp):.1f} %",
                "",
                "ma>1 -> przemodulowanie" if ma > 1 else "zakres liniowy"]


class TabP2(ExampleTab):
    key = "P2"
    PARAMS = [("ma", "ma", 0.0, 1.2, 0.9, 0.01),
              ("mf", "mf", 3, 99, 21, 1),
              ("vdc", "Vdc [V]", 50, 800, 350, 10),
              ("R", "R [Ω]", 0.05, 5, 0.5, 0.05),
              ("L", "L [mH]", 0.5, 20, 5, 0.5),
              ("E", "SEM E (szczyt) [V]", 0, 250, 100, 5)]

    def update_plot(self):
        ma, mf, vdc = self.p("ma"), int(self.p("mf")), self.p("vdc")
        f1 = 50.0
        t, refs, car, poles = three_phase_pwm(ma, mf, vdc, f1, n=6000)
        vab = poles[0] - poles[1]
        vn = poles.mean(axis=0)
        van = poles[0] - vn
        ia = rle_current(van, t, self.p("R"), self.p("L") * 1e-3, self.p("E"), f1)
        a1 = self.fig.add_subplot(3, 1, 1)
        a1.plot(t * 1e3, vab, "k", lw=0.7)
        a1.set_ylabel("vAB [V]"); a1.set_title("Napięcie międzyprzewodowe (3 poziomy)"); a1.grid(alpha=.3)
        a2 = self.fig.add_subplot(3, 1, 2, sharex=a1)
        a2.plot(t * 1e3, van, color="tab:blue", lw=0.6, label="vAN (fazowe)")
        a2b = a2.twinx()
        a2b.plot(t * 1e3, ia, "r", lw=1.5, label="iA")
        a2.set_ylabel("vAN [V]"); a2b.set_ylabel("iA [A]", color="r"); a2.grid(alpha=.3)
        a2.set_xlabel("t [ms]")
        a3 = self.fig.add_subplot(3, 1, 3)
        amp_v = spectrum(vab); amp_i = spectrum(ia)
        h = np.arange(min(len(amp_v), 3 * mf + 10))
        a3.bar(h - 0.2, amp_v[:len(h)] / max(amp_v[1], 1e-9), width=0.4, label="vAB")
        a3.bar(h + 0.2, amp_i[:len(h)] / max(amp_i[1], 1e-9), width=0.4, label="iA", color="r")
        a3.set_ylabel("względem 1. harm."); a3.set_xlabel("h"); a3.legend(fontsize=8); a3.grid(alpha=.3)
        vll_rms = amp_v[1] / math.sqrt(2)
        return [f"V_LL1 rms (FFT)   = {vll_rms:.1f} V",
                f"0,612*ma*Vdc      = {0.612 * ma * vdc:.1f} V",
                f"THD napięcia vAB  = {100 * thd(amp_v):.1f} %",
                f"THD prądu iA      = {100 * thd(amp_i):.1f} %",
                f"I1 szczytowy      = {amp_i[1]:.1f} A",
                f"tętnienia i (p-p) ≈ {np.ptp(ia - amp_i[1] * np.sin(2*PI*f1*t + np.angle(np.fft.rfft(ia)[1]) + PI/2)):.2f} A"]


class TabP3(ExampleTab):
    key = "P3"
    PARAMS = [("ma", "ma", 0.0, 1.3, 1.1, 0.01),
              ("mf", "mf", 9, 99, 33, 1),
              ("vdc", "Vdc [V]", 50, 800, 350, 10)]

    def update_plot(self):
        ma, mf, vdc = self.p("ma"), int(self.p("mf")), self.p("vdc")
        res = []
        a1 = self.fig.add_subplot(2, 1, 1)
        a2 = self.fig.add_subplot(2, 1, 2)
        for mode, col, name in (("spwm", "tab:blue", "SPWM"), ("thi", "tab:green", "THI (1/6)"),
                                ("minmax", "tab:red", "min-max (SVPWM)")):
            t, refs, car, poles = three_phase_pwm(ma, mf, vdc, n=8000, mode=mode)
            a1.plot(t * 1e3, refs[0], color=col, lw=1.5, label=name)
            vab = poles[0] - poles[1]
            amp = spectrum(vab)
            res.append((name, amp[1] / math.sqrt(2), thd(amp, 20)))
            # napięcie AB po filtracji (tylko harmoniczne < 20) - pokazuje zniekształcenia niskie
            c = np.fft.rfft(vab); c[20:] = 0
            a2.plot(t * 1e3, np.fft.irfft(c, len(vab)), color=col, lw=1.4, label=name)
        a1.axhline(1, color="k", ls="--", lw=0.8); a1.axhline(-1, color="k", ls="--", lw=0.8)
        a1.set_title("Sygnał odniesienia fazy A (±1 = szczyt nośnej)"); a1.legend(fontsize=8); a1.grid(alpha=.3)
        a2.set_title("vAB - tylko harmoniczne niskiego rzędu (h<20)")
        a2.set_xlabel("t [ms]"); a2.set_ylabel("[V]"); a2.legend(fontsize=8); a2.grid(alpha=.3)
        out = [f"{'metoda':<16}{'V_LL1 rms':>10}{'THD<20':>8}"]
        out += [f"{n:<16}{v:>10.1f}{100 * d:>7.1f}%" for n, v, d in res]
        out += ["", f"granica SPWM: {0.612 * vdc:.0f} V",
                f"granica SVPWM: {vdc / math.sqrt(2):.0f} V (+15,5%)"]
        return out


class TabP4(ExampleTab):
    key = "P4"
    PARAMS = [("vdc", "Vdc [V]", 100, 800, 350, 10),
              ("vref", "|Vref| (szczyt faz.) [V]", 0, 300, 150, 1),
              ("theta", "kąt θ [°]", 0, 359, 80, 1),
              ("ts", "Ts [µs]", 20, 500, 100, 5)]

    def update_plot(self):
        vdc, vref, th, ts = self.p("vdc"), self.p("vref"), self.p("theta"), self.p("ts")
        sec, alpha, t1, t2, t0, over = svpwm_times(vref, th, vdc, ts)
        a1 = self.fig.add_subplot(1, 2, 1)
        R = 2 / 3 * vdc
        ang = np.radians(np.arange(0, 361, 60))
        a1.plot(R * np.cos(ang), R * np.sin(ang), "k", lw=1.5)
        for k in range(6):
            a1.annotate("", xy=(R * math.cos(k * PI / 3), R * math.sin(k * PI / 3)), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="->", color="0.5"))
            a1.text(1.1 * R * math.cos(k * PI / 3), 1.1 * R * math.sin(k * PI / 3), f"V{k + 1}",
                    ha="center", va="center")
            a1.text(0.6 * R * math.cos((k + .5) * PI / 3), 0.6 * R * math.sin((k + .5) * PI / 3),
                    f"S{k + 1}", color="tab:blue", ha="center", fontsize=8)
        c = np.linspace(0, 2 * PI, 200)
        a1.plot(vdc / SQ3 * np.cos(c), vdc / SQ3 * np.sin(c), "g--", lw=1, label="granica liniowa Vdc/√3")
        a1.plot(vref * np.cos(c), vref * np.sin(c), ":", color="tab:orange", lw=1)
        thr = math.radians(th)
        a1.annotate("", xy=(vref * math.cos(thr), vref * math.sin(thr)), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color="red" if over else "tab:red", lw=2.5))
        # składowe T1/Ts*V1 + T2/Ts*V2
        b1 = (sec - 1) * PI / 3; b2 = sec * PI / 3
        p1 = (R * t1 / ts * math.cos(b1), R * t1 / ts * math.sin(b1))
        p2 = (p1[0] + R * t2 / ts * math.cos(b2), p1[1] + R * t2 / ts * math.sin(b2))
        a1.plot([0, p1[0], p2[0]], [0, p1[1], p2[1]], "m-", lw=2, label="T1/Ts·Vk + T2/Ts·Vk+1")
        a1.set_aspect("equal"); a1.grid(alpha=.3); a1.legend(fontsize=7, loc="lower left")
        a1.set_title("Sześciokąt wektorów napięcia" + ("  - PRZEMODULOWANIE!" if over else ""))
        # wzór przełączeń
        a2 = self.fig.add_subplot(1, 2, 2)
        states = {0: (0, 0, 0), 1: (1, 0, 0), 2: (1, 1, 0), 3: (0, 1, 0), 4: (0, 1, 1), 5: (0, 0, 1),
                  6: (1, 0, 1), 7: (1, 1, 1)}
        va, vb = sec, sec % 6 + 1
        seq = [(0, t0 / 4), (va, t1 / 2), (vb, t2 / 2), (7, t0 / 2), (vb, t2 / 2), (va, t1 / 2), (0, t0 / 4)]
        tt = 0.0
        for st, dur in seq:
            for ph in range(3):
                lvl = states[st][ph]
                a2.fill_between([tt, tt + dur], 2 * (2 - ph), 2 * (2 - ph) + 1.5 * lvl,
                                color=["tab:red", "tab:green", "tab:blue"][ph], alpha=.8, step="pre")
            a2.text(tt + dur / 2, 6.3, f"V{st}", ha="center", fontsize=8)
            a2.axvline(tt, color="0.7", lw=0.5)
            tt += dur
        a2.set_yticks([0.75, 2.75, 4.75]); a2.set_yticklabels(["C", "B", "A"])
        a2.set_xlabel("t [µs]"); a2.set_xlim(0, ts); a2.set_ylim(-0.2, 6.8)
        a2.set_title("Wzór przełączeń w okresie Ts")
        m = vref / (vdc / SQ3)
        return [f"sektor          = {sec}",
                f"kąt w sektorze α = {alpha:.1f}°",
                f"T1 = {t1:7.2f} µs", f"T2 = {t2:7.2f} µs", f"T0 = {t0:7.2f} µs",
                f"Vref/(Vdc/√3)   = {m:.3f}",
                "PRZEMODULOWANIE - przycięto" if over else "zakres liniowy",
                f"V_LL rms (1. harm.) = {vref * SQ3 / math.sqrt(2):.1f} V"]


class TabP5(ExampleTab):
    key = "P5"
    PARAMS = [("ma", "ma (punkt pracy)", 0.1, 4.0, 1.5, 0.05),
              ("mf", "mf", 9, 51, 15, 2),
              ("vdc", "Vdc [V]", 50, 800, 350, 10)]

    def update_plot(self):
        ma, mf, vdc = self.p("ma"), int(self.p("mf")), self.p("vdc")
        mas = np.concatenate([np.linspace(0.05, 1, 12), np.linspace(1.05, 4, 25)])
        fund = fundamental_vs_ma(mas, mf=mf)
        a1 = self.fig.add_subplot(2, 1, 1)
        a1.plot(mas, fund * 0.612 / 1.0 * 1.0, "b-o", ms=3, label="SPWM (symulacja)")
        a1.plot([0, 1], [0, 0.612], "g--", lw=1, label="liniowo 0,612·ma")
        a1.axhline(0.78, color="r", ls="--", lw=1, label="six-step 0,78")
        a1.axhline(0.707, color="orange", ls=":", lw=1, label="SVPWM max 0,707")
        _, _, _, v0 = spwm_leg(ma, mf, 1.0, n=8000)
        f0 = spectrum(v0)[1] / 0.5
        a1.plot(ma, f0 * 0.612, "r*", ms=14)
        a1.set_xlabel("ma"); a1.set_ylabel("V_LL1(rms)/Vdc"); a1.grid(alpha=.3); a1.legend(fontsize=8)
        a1.set_title("Napięcie wyjściowe a współczynnik modulacji")
        t, ref, car, vao = spwm_leg(ma, mf, vdc, n=8000)
        amp = spectrum(vao)
        a2 = self.fig.add_subplot(2, 2, 3)
        a2.plot(t * 1e3, car, color="0.7", lw=.7); a2.plot(t * 1e3, ref, "b")
        a2.plot(t * 1e3, vao / (vdc / 2), "k", lw=.8); a2.set_xlabel("t [ms]"); a2.grid(alpha=.3)
        a2.set_title("przebiegi (p.u.)")
        a3 = self.fig.add_subplot(2, 2, 4)
        h = np.arange(1, 26)
        a3.bar(h, amp[h] / (vdc / 2), color="tab:purple"); a3.set_xlabel("h (niskie harmoniczne)")
        a3.grid(alpha=.3); a3.set_title("widmo niskie")
        return [f"V1/(Vdc/2) = {amp[1] / (vdc / 2):.3f}  (max 4/π=1,273)",
                f"V_LL1 rms   = {amp[1] / (vdc / 2) * 0.612 * vdc:.1f} V",
                f"5. harm.    = {100 * amp[5] / amp[1]:.1f} % V1",
                f"7. harm.    = {100 * amp[7] / amp[1]:.1f} % V1",
                f"six-step: V_LL1 = {0.78 * vdc:.0f} V"]


class TabP6(ExampleTab):
    key = "P6"
    PARAMS = [("td", "czas martwy td [µs]", 0, 10, 3, 0.1),
              ("fc", "f_nośnej [kHz]", 1, 20, 5, 0.5),
              ("vdc", "Vdc [V]", 50, 800, 350, 10),
              ("ma", "ma", 0.05, 1, 0.3, 0.01),
              ("phi", "przesunięcie prądu φ [°]", -90, 90, 30, 1)]

    def update_plot(self):
        td, fc, vdc, ma, phi = self.p("td") * 1e-6, self.p("fc") * 1e3, self.p("vdc"), self.p("ma"), self.p("phi")
        f1 = 50.0
        n = int(max(20000, fc / f1 * 200))
        t = np.linspace(0, 1 / f1, n, endpoint=False)
        dt = t[1] - t[0]
        ref = ma * np.sin(2 * PI * f1 * t)
        car = triangle(t, fc)
        gate = ref >= car
        i = np.sin(2 * PI * f1 * t - math.radians(phi))
        # opóźnienie załączenia o td: stan się zmienia dopiero, gdy od zbocza minie td,
        # w tym czasie o napięciu decyduje znak prądu
        nd = int(round(td / dt))
        v_ideal = np.where(gate, vdc / 2, -vdc / 2)
        v_real = v_ideal.copy()
        if nd > 0:
            edges = np.flatnonzero(np.diff(gate.astype(int))) + 1
            for e in edges:
                sl = slice(e, min(e + nd, n))
                v_real[sl] = np.where(i[sl] > 0, -vdc / 2, vdc / 2)
        err = v_real - v_ideal
        # średnia krocząca po okresie nośnej
        w = max(1, int(round(1 / fc / dt)))
        kern = np.ones(w) / w
        err_avg = np.convolve(np.concatenate([err[-w:], err, err[:w]]), kern, "same")[w:-w]
        dv = td * fc * vdc
        a1 = self.fig.add_subplot(2, 1, 1)
        a1.plot(t * 1e3, err_avg, "r", lw=1.2, label="średni błąd napięcia (symulacja)")
        a1.plot(t * 1e3, -dv * np.sign(i), "k--", lw=1, label="-td·fc·Vdc·sign(i)")
        a1b = a1.twinx(); a1b.plot(t * 1e3, i, color="tab:blue", alpha=.5, label="prąd (p.u.)")
        a1.set_ylabel("ΔV [V]"); a1.legend(fontsize=8, loc="upper right"); a1.grid(alpha=.3)
        a1.set_title("Błąd napięcia wywołany czasem martwym")
        a2 = self.fig.add_subplot(2, 1, 2)
        amp_i = spectrum(v_ideal); amp_r = spectrum(v_real)
        v1i = amp_i[1] * np.sin(2 * PI * f1 * t + np.angle(np.fft.rfft(v_ideal)[1]) + PI / 2)
        c = np.fft.rfft(v_real); c[40:] = 0
        a2.plot(t * 1e3, v1i, "g", lw=1.2, label="idealne (1. harm.)")
        a2.plot(t * 1e3, np.fft.irfft(c, n), "r", lw=1.2, label="z czasem martwym (h<40)")
        a2.set_xlabel("t [ms]"); a2.set_ylabel("[V]"); a2.legend(fontsize=8); a2.grid(alpha=.3)
        return [f"ΔV = td·fc·Vdc = {dv:.2f} V",
                f"V1 idealne     = {amp_i[1]:.1f} V",
                f"V1 rzeczyw.    = {amp_r[1]:.1f} V",
                f"błąd względny  = {100 * dv / max(ma * vdc / 2, 1e-9):.1f} % V1",
                f"5. harm. (rz.) = {amp_r[5]:.2f} V",
                f"7. harm. (rz.) = {amp_r[7]:.2f} V"]


# =============================================================================
# ZAKŁADKI PMSM
# =============================================================================

MOTOR = [("p", "pary biegunów p", 1, 8, 4, 1),
         ("psi", "ψf strumień magnesu [mWb]", 20, 200, 80, 1),
         ("Ld", "Ld [µH]", 50, 1000, 200, 10),
         ("Lq", "Lq [µH]", 50, 1500, 450, 10)]


class TabM1(ExampleTab):
    key = "M1"
    PARAMS = MOTOR + [("Rs", "Rs [mΩ]", 1, 100, 20, 1),
                      ("id", "Id [A]", -300, 100, 0, 5),
                      ("iq", "Iq [A]", -300, 300, 200, 5),
                      ("n", "prędkość n [obr/min]", 0, 12000, 3000, 50),
                      ("vdc", "Vdc [V]", 100, 800, 350, 10)]

    def update_plot(self):
        p = int(self.p("p")); psi = self.p("psi") / 1e3; Ld = self.p("Ld") * 1e-6; Lq = self.p("Lq") * 1e-6
        Rs = self.p("Rs") / 1e3; id_, iq = self.p("id"), self.p("iq"); n = self.p("n")
        wm = n * 2 * PI / 60; we = p * wm
        vd, vq = pmsm_steady(id_, iq, we, Rs, Ld, Lq, psi)
        Te = pmsm_torque(id_, iq, p, Ld, Lq, psi)
        Vs = math.hypot(vd, vq); Is = math.hypot(id_, iq)
        Vmax = self.p("vdc") / SQ3
        Pm = Te * wm; Pe = 1.5 * (vd * id_ + vq * iq); Pcu = 1.5 * Rs * Is ** 2
        a1 = self.fig.add_subplot(1, 2, 1)
        sc_i = 1.0; sc_v = max(Is, 1) / max(Vs, Vmax, 1) * 1.0
        def arrow(ax, x, y, col, lab):
            ax.annotate("", xy=(x, y), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=col, lw=2))
            ax.text(x, y, " " + lab, color=col, fontsize=9)
        arrow(a1, id_ * sc_i, iq * sc_i, "tab:red", "I")
        arrow(a1, vd * sc_v, vq * sc_v, "tab:blue", "V")
        fd, fq = psi + Ld * id_, Lq * iq
        sc_f = max(Is, 1) / max(math.hypot(fd, fq), 1e-9) * 0.7
        arrow(a1, fd * sc_f, fq * sc_f, "tab:green", "ψ")
        c = np.linspace(0, 2 * PI, 200)
        a1.plot(Vmax * sc_v * np.cos(c), Vmax * sc_v * np.sin(c), "r--", lw=.8, label="granica V = Vdc/√3")
        L = max(Is, Vs * sc_v, Vmax * sc_v) * 1.2
        a1.set_xlim(-L, L); a1.set_ylim(-L, L); a1.set_aspect("equal"); a1.grid(alpha=.3)
        a1.axhline(0, color="k", lw=.5); a1.axvline(0, color="k", lw=.5)
        a1.set_xlabel("oś d"); a1.set_ylabel("oś q"); a1.legend(fontsize=7, loc="lower left")
        a1.set_title("Wykres wskazowy dq (skalowany)")
        a2 = self.fig.add_subplot(1, 2, 2)
        ns = np.linspace(0, 12000, 200); wes = ns * 2 * PI / 60 * p
        vds, vqs = pmsm_steady(id_, iq, wes, Rs, Ld, Lq, psi)
        a2.plot(ns, np.hypot(vds, vqs), "b", label="|V| przy tych Id, Iq")
        a2.plot(ns, wes * psi, "g--", label="SEM = ωe·ψf")
        a2.axhline(Vmax, color="r", ls="--", label="Vdc/√3")
        a2.plot(n, Vs, "ko")
        a2.set_xlabel("n [obr/min]"); a2.set_ylabel("napięcie fazowe (szczyt) [V]"); a2.grid(alpha=.3)
        a2.legend(fontsize=8); a2.set_title("Napięcie rośnie z prędkością")
        return [f"ωm = {wm:.1f} rad/s, ωe = {we:.1f} rad/s",
                f"f elektr.  = {we / 2 / PI:.1f} Hz",
                f"Vd = {vd:8.2f} V", f"Vq = {vq:8.2f} V",
                f"|V| = {Vs:.1f} V  (max {Vmax:.1f})",
                f"|I| = {Is:.1f} A",
                f"Te   = {Te:.2f} Nm",
                f"Pmech = {Pm / 1e3:.2f} kW",
                f"Pel   = {Pe / 1e3:.2f} kW",
                f"Pcu   = {Pcu:.0f} W",
                f"sprawność ≈ {100 * Pm / Pe:.1f} %" if Pe > 1 else "",
                "!! przekroczone napięcie" if Vs > Vmax else "napięcie OK"]


class TabM2(ExampleTab):
    key = "M2"
    PARAMS = MOTOR + [("Is", "amplituda prądu Is [A]", 10, 500, 250, 5)]

    def update_plot(self):
        p = int(self.p("p")); psi = self.p("psi") / 1e3; Ld = self.p("Ld") * 1e-6; Lq = self.p("Lq") * 1e-6
        Is = self.p("Is")
        beta = np.radians(np.linspace(-90, 90, 361))
        id_ = -Is * np.sin(beta); iq = Is * np.cos(beta)
        Tm = 1.5 * p * psi * iq; Tr = 1.5 * p * (Ld - Lq) * id_ * iq
        a1 = self.fig.add_subplot(1, 2, 1)
        b = np.degrees(beta)
        a1.plot(b, Tm, "b--", label="moment magnesu")
        a1.plot(b, Tr, "g--", label="moment reluktancyjny")
        a1.plot(b, Tm + Tr, "k", lw=2, label="moment całkowity")
        idm = mtpa_id(Is, Ld, Lq, psi); iqm = math.sqrt(max(Is ** 2 - idm ** 2, 0))
        bm = math.degrees(math.atan2(-idm, iqm)); Tmtpa = pmsm_torque(idm, iqm, p, Ld, Lq, psi)
        a1.plot(bm, Tmtpa, "o", color="lime", mec="k", ms=10, label="MTPA")
        a1.set_xlabel("kąt prądu β [°] (β>0 → Id<0)"); a1.set_ylabel("Te [Nm]"); a1.grid(alpha=.3)
        a1.legend(fontsize=8); a1.set_title(f"Moment przy |I| = {Is:.0f} A")
        a2 = self.fig.add_subplot(1, 2, 2)
        Iss = np.linspace(0, 500, 101)
        idt = mtpa_id(Iss, Ld, Lq, psi); iqt = np.sqrt(np.maximum(Iss ** 2 - idt ** 2, 0))
        a2.plot(idt, iqt, "lime", lw=2.5, label="trajektoria MTPA")
        a2.plot([0, 0], [0, 500], "b--", label="Id = 0")
        ID, IQ = np.meshgrid(np.linspace(-500, 50, 200), np.linspace(0, 500, 200))
        cs = a2.contour(ID, IQ, pmsm_torque(ID, IQ, p, Ld, Lq, psi), 10, colors="0.4", linewidths=.7)
        a2.clabel(cs, fontsize=7, fmt="%.0f Nm")
        c = np.linspace(PI / 2, PI, 50)
        a2.plot(Is * np.cos(c), Is * np.sin(c), "r", label="|I| = Is")
        a2.set_aspect("equal"); a2.set_xlabel("Id [A]"); a2.set_ylabel("Iq [A]"); a2.grid(alpha=.3)
        a2.legend(fontsize=8); a2.set_title("Płaszczyzna dq")
        T0 = pmsm_torque(0, Is, p, Ld, Lq, psi)
        return [f"Lq/Ld (wyrazistość) = {Lq / Ld:.2f}",
                f"Te przy Id=0   = {T0:.1f} Nm",
                f"MTPA: Id = {idm:.1f} A", f"      Iq = {iqm:.1f} A",
                f"      β  = {bm:.1f}°",
                f"Te MTPA        = {Tmtpa:.1f} Nm",
                f"zysk MTPA      = {100 * (Tmtpa / T0 - 1) if T0 > 0 else 0:.1f} %"]


class TabM3(ExampleTab):
    key = "M3"
    PARAMS = MOTOR + [("Imax", "Imax [A]", 50, 600, 350, 10),
                      ("vdc", "Vdc [V]", 100, 800, 350, 10),
                      ("n", "prędkość n [obr/min]", 500, 16000, 6000, 100)]

    def update_plot(self):
        p = int(self.p("p")); psi = self.p("psi") / 1e3; Ld = self.p("Ld") * 1e-6; Lq = self.p("Lq") * 1e-6
        Imax, Vmax, n = self.p("Imax"), self.p("vdc") / SQ3, self.p("n")
        we = n * 2 * PI / 60 * p
        a = self.fig.add_subplot(1, 1, 1)
        lim = 1.3 * max(Imax, psi / Ld)
        ID, IQ = np.meshgrid(np.linspace(-lim, 0.3 * Imax, 300), np.linspace(-1.1 * Imax, 1.1 * Imax, 300))
        feas = (ID ** 2 + IQ ** 2 <= Imax ** 2) & (we ** 2 * ((psi + Ld * ID) ** 2 + (Lq * IQ) ** 2) <= Vmax ** 2)
        a.contourf(ID, IQ, feas, levels=[0.5, 1.5], colors=["#c8f0c8"])
        T = pmsm_torque(ID, IQ, p, Ld, Lq, psi)
        cs = a.contour(ID, IQ, T, 12, colors="k", linewidths=.5)
        a.clabel(cs, fontsize=7, fmt="%.0f")
        c = np.linspace(0, 2 * PI, 300)
        a.plot(Imax * np.cos(c), Imax * np.sin(c), "r", lw=2, label="okrąg prądu Imax")
        for nn, st in ((n, "-"), (n * 0.5, ":"), (n * 1.5, "--")):
            w = nn * 2 * PI / 60 * p; r = Vmax / w
            a.plot(-psi / Ld + r / Ld * np.cos(c), r / Lq * np.sin(c), "b", ls=st, lw=1.5 if nn == n else .8,
                   label=f"elipsa napięcia {nn:.0f} obr/min")
        Iss = np.linspace(0, Imax, 60); idt = mtpa_id(Iss, Ld, Lq, psi)
        a.plot(idt, np.sqrt(np.maximum(Iss ** 2 - idt ** 2, 0)), "lime", lw=2, label="MTPA")
        a.plot(-psi / Ld, 0, "kx", ms=10, label="punkt charakt. -ψf/Ld")
        Tm, idop, iqop = torque_envelope([n], p, Ld, Lq, psi, Imax, Vmax, ngrid=301)
        if not np.isnan(idop[0]):
            a.plot(idop[0], iqop[0], "o", color="orange", mec="k", ms=11, label=f"max Te = {Tm[0]:.1f} Nm")
        a.set_xlim(-lim, 0.3 * Imax); a.set_ylim(-1.1 * Imax, 1.1 * Imax)
        a.set_aspect("equal"); a.grid(alpha=.3); a.legend(fontsize=7, loc="lower left")
        a.set_xlabel("Id [A]"); a.set_ylabel("Iq [A]")
        a.set_title("Obszar dopuszczalnych punktów pracy (zielony)")
        nb = 60 * Vmax / (p * 2 * PI * math.hypot(psi + Ld * mtpa_id(Imax, Ld, Lq, psi),
                                                   Lq * math.sqrt(max(Imax ** 2 - mtpa_id(Imax, Ld, Lq, psi) ** 2, 0))))
        return [f"ωe = {we:.0f} rad/s",
                f"Vmax = Vdc/√3 = {Vmax:.1f} V",
                f"ψf/Ld = {psi / Ld:.0f} A",
                ("ψf/Ld < Imax -> prędkość" if psi / Ld < Imax else "ψf/Ld > Imax -> ograniczona"),
                ("  teoretycznie nieograniczona" if psi / Ld < Imax else "  prędkość maksymalna"),
                f"n bazowa (MTPA,Imax) ≈ {nb:.0f} obr/min",
                f"n bez obciąż. (SEM=V) = {60 * Vmax / (p * 2 * PI * psi):.0f} obr/min",
                (f"punkt: Id={idop[0]:.0f}, Iq={iqop[0]:.0f} A" if not np.isnan(idop[0]) else "brak punktu pracy!"),
                f"max Te = {Tm[0]:.1f} Nm"]


class TabM4(ExampleTab):
    key = "M4"
    PARAMS = MOTOR + [("Imax", "Imax [A]", 50, 600, 350, 10),
                      ("vdc", "Vdc [V]", 100, 900, 350, 10),
                      ("nmax", "n max wykresu [obr/min]", 4000, 20000, 14000, 500)]

    def update_plot(self):
        p = int(self.p("p")); psi = self.p("psi") / 1e3; Ld = self.p("Ld") * 1e-6; Lq = self.p("Lq") * 1e-6
        Imax, Vmax = self.p("Imax"), self.p("vdc") / SQ3
        ns = np.linspace(10, self.p("nmax"), 90)
        T, idt, iqt = torque_envelope(ns, p, Ld, Lq, psi, Imax, Vmax)
        P = T * ns * 2 * PI / 60 / 1e3
        # porównanie: 2x wyższe napięcie
        T2, _, _ = torque_envelope(ns, p, Ld, Lq, psi, Imax, 2 * Vmax, ngrid=101)
        a1 = self.fig.add_subplot(2, 1, 1)
        a1.plot(ns, T, "b", lw=2, label="Te max [Nm]")
        a1.plot(ns, T2, "b:", lw=1, label="Te max przy 2·Vdc")
        a1b = a1.twinx(); a1b.plot(ns, P, "r", lw=2, label="P [kW]")
        a1b.set_ylabel("P [kW]", color="r")
        T0 = T[0]; kb = np.argmax(T < 0.98 * T0) if np.any(T < 0.98 * T0) else len(ns) - 1
        a1.axvline(ns[kb], color="k", ls="--", lw=.8)
        a1.text(ns[kb], T0 * 0.5, " n bazowa", fontsize=8)
        a1.set_ylabel("Te [Nm]"); a1.grid(alpha=.3); a1.legend(loc="center right", fontsize=8)
        a1.set_title("Obwiednia moment/moc - prędkość")
        a2 = self.fig.add_subplot(2, 1, 2, sharex=a1)
        a2.plot(ns, idt, "g", label="Id"); a2.plot(ns, iqt, "m", label="Iq")
        a2.plot(ns, np.hypot(idt, iqt), "k--", lw=.8, label="|I|")
        a2.set_xlabel("n [obr/min]"); a2.set_ylabel("[A]"); a2.grid(alpha=.3); a2.legend(fontsize=8)
        return [f"moment szczytowy = {T0:.1f} Nm",
                f"n bazowa   ≈ {ns[kb]:.0f} obr/min",
                f"moc max    = {np.nanmax(P):.1f} kW",
                f"moc przy n max = {P[-1]:.1f} kW",
                f"Vmax faz.  = {Vmax:.1f} V"]


class TabM5(ExampleTab):
    key = "M5"
    PARAMS = MOTOR + [("Rs", "Rs [mΩ]", 5, 100, 20, 1),
                      ("J", "J [g·m²]", 5, 500, 50, 5),
                      ("Imax", "Imax [A]", 50, 500, 250, 10),
                      ("n_ref", "n zadana [obr/min]", 100, 8000, 3000, 100),
                      ("TL", "moment obciążenia [Nm]", 0, 150, 60, 1),
                      ("bw_i", "pasmo pętli prądu [Hz]", 100, 2000, 800, 50),
                      ("bw_w", "pasmo pętli prędkości [Hz]", 1, 50, 10, 1)]

    def update_plot(self):
        P = dict(p=int(self.p("p")), psi=self.p("psi") / 1e3, Ld=self.p("Ld") * 1e-6, Lq=self.p("Lq") * 1e-6,
                 Rs=self.p("Rs") / 1e3, J=self.p("J") / 1e3, B=1e-3, Vdc=350.0, Imax=self.p("Imax"),
                 n_ref=self.p("n_ref"), TL=self.p("TL"), t_load=0.35, bw_i=self.p("bw_i"), bw_w=self.p("bw_w"))
        r = foc_simulate(P, t_end=0.6)
        k = 60 / (2 * PI)
        a1 = self.fig.add_subplot(3, 1, 1)
        a1.plot(r["t"], r["wref"] * k, "k--", label="zadana"); a1.plot(r["t"], r["wm"] * k, "b", label="rzeczywista")
        a1.set_ylabel("n [obr/min]"); a1.legend(fontsize=8); a1.grid(alpha=.3)
        a1.set_title("Pętla prędkości")
        a2 = self.fig.add_subplot(3, 1, 2, sharex=a1)
        a2.plot(r["t"], r["iqref"], "m--", lw=.8, label="Iq*"); a2.plot(r["t"], r["iq"], "m", label="Iq")
        a2.plot(r["t"], r["id"], "g", label="Id")
        a2.set_ylabel("[A]"); a2.legend(fontsize=8); a2.grid(alpha=.3)
        a3 = self.fig.add_subplot(3, 1, 3, sharex=a1)
        a3.plot(r["t"], r["Te"], "r", label="Te"); a3.plot(r["t"], r["TL"], "k--", label="TL")
        a3b = a3.twinx(); a3b.plot(r["t"], np.hypot(r["vd"], r["vq"]), color="tab:blue", alpha=.5)
        a3b.set_ylabel("|V| [V]", color="tab:blue")
        a3.set_xlabel("t [s]"); a3.set_ylabel("[Nm]"); a3.legend(fontsize=8); a3.grid(alpha=.3)
        wr = P["n_ref"] / k
        mask = r["t"] < 0.35
        t_rise = r["t"][mask][np.argmax(r["wm"][mask] >= 0.9 * wr)] - 0.02 if np.any(r["wm"][mask] >= 0.9 * wr) else float("nan")
        after = r["t"] > 0.35
        dip = (wr - r["wm"][after].min()) * k if after.any() else 0
        return [f"Kt = 1,5pψf = {1.5 * P['p'] * P['psi']:.3f} Nm/A",
                f"Te max = {1.5 * P['p'] * P['psi'] * P['Imax']:.1f} Nm",
                f"czas do 90% n* = {t_rise * 1e3:.0f} ms",
                f"przeregulowanie = {max(0, (r['wm'][mask].max() - wr) * k):.0f} obr/min",
                f"spadek po obciąż. = {dip:.0f} obr/min",
                f"n końcowa = {r['wm'][-1] * k:.0f} obr/min"]


class TabM6(ExampleTab):
    key = "M6"
    PARAMS = [("m", "masa pojazdu [kg]", 800, 3000, 1800, 50),
              ("Cd", "Cd (aerodynamika)", 0.15, 0.6, 0.23, 0.01),
              ("A", "pow. czołowa A [m²]", 1.5, 3.5, 2.2, 0.1),
              ("Crr", "Crr (toczenie)", 0.005, 0.03, 0.01, 0.001),
              ("grade", "nachylenie [%]", 0, 30, 0, 1),
              ("G", "przełożenie G", 4, 15, 9, 0.1),
              ("r", "promień koła [m]", 0.25, 0.4, 0.33, 0.01),
              ("T", "moment szczyt. silnika [Nm]", 100, 600, 350, 10),
              ("P", "moc szczyt. [kW]", 50, 400, 200, 5),
              ("nmax", "n max silnika [obr/min]", 8000, 20000, 16000, 500)]

    def update_plot(self):
        m, Cd, A, Crr, gr = self.p("m"), self.p("Cd"), self.p("A"), self.p("Crr"), self.p("grade")
        G, r, T, Pk, nmax = self.p("G"), self.p("r"), self.p("T"), self.p("P"), self.p("nmax")
        eta = 0.95
        vmax_geo = nmax * 2 * PI / 60 / G * r
        v = np.linspace(0, vmax_geo, 300)
        n = v / r * G * 60 / (2 * PI)
        Ftr = ev_motor_curve(n, T, Pk, nmax) * G * eta / r
        Fr, Fa, Fg = ev_forces(v, m, Crr, Cd, A, gr)
        Fres = Fr + Fa + Fg
        a1 = self.fig.add_subplot(2, 2, 1)
        a1.plot(v * 3.6, Ftr, "b", lw=2, label="siła napędowa")
        a1.stackplot(v * 3.6, Fr, Fg, Fa, labels=["toczenie", "wzniesienie", "aero"],
                     colors=["#bbb", "#e7b", "#8cf"], alpha=.8)
        a1.set_xlabel("v [km/h]"); a1.set_ylabel("F [N]"); a1.grid(alpha=.3); a1.legend(fontsize=7)
        a1.set_title("Siły na kołach")
        idx = np.flatnonzero((Ftr - Fres) < 0)
        vtop = v[idx[0]] * 3.6 if len(idx) else vmax_geo * 3.6
        a2 = self.fig.add_subplot(2, 2, 2)
        a2.plot(v * 3.6, Fres * v / 1e3, "k", label="moc oporów")
        a2.plot(v * 3.6, Ftr * v / 1e3, "b", label="moc napędu")
        a2.set_xlabel("v [km/h]"); a2.set_ylabel("P [kW]"); a2.grid(alpha=.3); a2.legend(fontsize=7)
        a2.set_title("Moc potrzebna vs dostępna")
        ts, vs, Fs, t100 = ev_accelerate(m, Crr, Cd, A, gr, r, G, eta, T, Pk, nmax, t_end=25)
        a3 = self.fig.add_subplot(2, 1, 2)
        a3.plot(ts, vs * 3.6, "g", lw=2, label="prędkość")
        a3.axhline(100, color="k", ls=":", lw=.8)
        if t100:
            a3.axvline(t100, color="r", ls="--", lw=.8)
            a3.text(t100, 20, f" 0-100: {t100:.1f} s", color="r")
        a3.set_xlabel("t [s]"); a3.set_ylabel("v [km/h]"); a3.grid(alpha=.3)
        a3.set_title("Przyspieszanie pełną mocą (Euler)")
        Fr1, Fa1, Fg1 = ev_forces(np.array([120 / 3.6]), m, Crr, Cd, A, gr)
        P120 = (Fr1 + Fa1 + Fg1)[0] * 120 / 3.6 / 1e3
        return [f"siła rozruchowa = {Ftr[0] / 1e3:.2f} kN",
                f"przyspieszenie 0 = {(Ftr[0] - Fres[0]) / (1.05 * m):.2f} m/s²",
                f"0-100 km/h = {t100:.2f} s" if t100 else "0-100 km/h: nie osiąga",
                f"v max (ograniczenie) ≈ {vtop:.0f} km/h",
                f"v max (n_max silnika) = {vmax_geo * 3.6:.0f} km/h",
                f"moc przy 120 km/h = {P120:.1f} kW",
                f"energia/100 km @120 ≈ {P120 / 120 * 100 / eta:.1f} kWh"]


# =============================================================================
# APLIKACJA
# =============================================================================

TABS = [("P1 SPWM gałąź", TabP1), ("P2 Falownik 3f", TabP2), ("P3 THI/min-max", TabP3),
        ("P4 SVPWM", TabP4), ("P5 Przemodulowanie", TabP5), ("P6 Czas martwy", TabP6),
        ("M1 Model dq", TabM1), ("M2 MTPA", TabM2), ("M3 Osłabianie pola", TabM3),
        ("M4 Moment-prędkość", TabM4), ("M5 FOC symulacja", TabM5), ("M6 Pojazd EV", TabM6)]


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Sterowanie silnikami AC i pojazdy elektryczne - PWM i PMSM")
        self.geometry("1400x900")
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
