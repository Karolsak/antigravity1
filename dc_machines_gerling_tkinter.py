"""
Maszyny prądu stałego (DC) - interaktywny podręcznik z wykresami
=================================================================
Aplikacja Tkinter + Matplotlib do rozdziału 2 "DC-Machines"
(D. Gerling, "Electrical Machines", Springer 2015).

Każda zakładka = jeden podrozdział / przykład z rozdziału:
  suwaki (parametry)  ->  wykresy (wyniki)  +  objaśnienie "jak dla ucznia liceum".

  0  Budowa maszyny DC (2.1)                        - przekrój, bieguny, komutator
  1  Indukowanie napięcia, komutator (2.2)          - napięcie cewki, prostowanie, K cewek
  2  Komutacja i moment pojedynczej cewki (2.2)     - Fig. 2.5 / 2.6, silnik / prądnica
  3  Uzwojenia pętlicowe i faliste (2.3)            - K = N·u, z = 2·wS·K, 2a, krok y
  4  Równania główne (2.4.1-2.4.4)                  - Ui = kΦn, T = kΦI/2π, bilans mocy
  5  Współczynnik wykorzystania Essona (2.4.5)      - przykład projektowy 100 kW
  6  Przewody w żłobkach (2.5)                      - napięcie i siła "jak w szczelinie"
  7  Maszyna obcowzbudna (2.6)                      - n(I), T(I), T(n), 3 metody regulacji
  8  Tryby pracy - Tabela 2.1                       - silnik / hamulec / prądnica
  9  Rozruch - symulacja dynamiczna (2.4.3, 2.6)    - rozrusznik oporowy, skok obciążenia
  10 Magnesy trwałe (2.7)                           - punkt pracy magnesu, rozmagnesowanie
  11 Maszyna bocznikowa - samowzbudzenie (2.8)      - zasada dynamoelektryczna, obciążenie
  12 Maszyna szeregowa (2.9)                        - charakterystyka "miękka", "rozbieganie"
  13 Porównanie: bocznikowa/szeregowa/szer.-bocz. (2.10)
  14 Zmienne napięcie: Leonard, mostek 3-fazowy (2.11)
  15 Oddziaływanie twornika, uzwojenie kompensacyjne (2.12)
  16 Komutacja rzeczywista, bieguny komutacyjne (2.13)

Uruchomienie:  python dc_machines_gerling_tkinter.py
Wymagania:     numpy, matplotlib (tkinter jest w standardowym Pythonie)
"""
import math
import warnings
import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.patches import Wedge, Circle, Rectangle, FancyArrowPatch

warnings.filterwarnings("ignore", category=UserWarning)

PI = math.pi
MU0 = 4e-7 * PI                     # przenikalność magnetyczna próżni [H/m]
_integ = getattr(np, "trapezoid", None) or np.trapz

# Maszyna wzorcowa (obcowzbudna) używana w kilku zakładkach
REF = dict(UN=440.0,      # napięcie znamionowe [V]
           IAN=50.0,      # prąd znamionowy twornika [A]
           RA=0.6,        # rezystancja twornika [Ω]
           nN=1450.0)     # prędkość znamionowa [obr/min]


# =============================================================================
# RDZEŃ OBLICZENIOWY
# =============================================================================

def b_shape(theta, alpha_i):
    """Kształt indukcji w szczelinie B(θ)/B_max (trapez).
    θ - kąt elektryczny, środek bieguna N w θ = 0, bieguna S w θ = π.
    αi - stosunek łuku bieguna do podziałki biegunowej (plateau ma szerokość αi·π)."""
    th = np.mod(np.asarray(theta, dtype=float) + PI / 2, 2 * PI) - PI / 2   # [-π/2, 3π/2)
    north = th < PI / 2
    d = np.where(north, np.abs(th), np.abs(th - PI))       # odległość od środka bieguna
    edge = (1.0 - alpha_i) * PI / 2 + 1e-9                 # szerokość "zbocza"
    mag = np.clip((PI / 2 - d) / edge, 0.0, 1.0)
    return np.where(north, 1.0, -1.0) * mag


def commutated_voltage(theta, K, alpha_i, e_amp):
    """Napięcie jednej cewki (przemienne) oraz napięcie na szczotkach dla K cewek
    rozłożonych równomiernie na obwodzie (komutator prostuje każdą cewkę: |e|).
    Dla K >= 2 są dwie gałęzie równoległe, każda zawiera połowę cewek."""
    e1 = e_amp * b_shape(theta, alpha_i)
    if K <= 1:
        return e1, np.abs(e1)
    tot = np.zeros_like(theta, dtype=float)
    for k in range(K):
        tot += np.abs(e_amp * b_shape(theta + 2 * PI * k / K, alpha_i))
    return e1, 0.5 * tot


def coil_current(theta, Ic, beta_b):
    """Prąd cewki w funkcji położenia (komutacja liniowa w strefie neutralnej θ = ±90°).
    beta_b - szerokość szczotki w stopniach elektrycznych (czas komutacji)."""
    thw = np.mod(np.asarray(theta, dtype=float) + PI, 2 * PI) - PI
    d = PI / 2 - np.abs(thw)
    half = math.radians(beta_b) / 2 + 1e-9
    return Ic * np.clip(d / half, -1.0, 1.0)


def brush_shift_factor(beta, alpha_i):
    """Ui(β)/Ui(0) - wpływ przesunięcia szczotek o kąt β (el.) na napięcie."""
    th = np.linspace(-PI / 2, PI / 2, 2001)
    f0 = _integ(b_shape(th, alpha_i), th)
    return _integ(b_shape(th + beta, alpha_i), th) / f0


def winding(p, N, u, wS, kind):
    """Dane uzwojenia twornika: K, z, liczba gałęzi 2a, krok komutatorowy y."""
    K = N * u
    z = 2 * wS * K
    y1 = max(1, int(round(N / (2 * p))))           # szerokość cewki ≈ podziałka biegunowa [żłobki]
    if kind == "lap":
        a2, y, ok = 2 * p, 1, True
    else:
        a2, y, ok = 2, None, False
        for cand in (K - 1, K + 1):
            if cand > 0 and cand % p == 0:
                y, ok = cand // p, True
                break
    return dict(K=K, z=z, a2=a2, y=y, ok=ok, y1=y1)


def winding_trace(K, y, n):
    """Kolejne działki komutatora, przez które przechodzi uzwojenie (y - krok)."""
    seq, s = [0], 0
    for _ in range(max(n - 1, 0)):
        s = (s + y) % K
        seq.append(s)
    return seq


def esson(alpha_i, A_acm, B, Pi_kw, n_rpm, p, lam):
    """Projekt wstępny z liczby Essona: Pi = C·D²·l·n,  C = π²·αi·A·B,  l = λ·τp."""
    A = A_acm * 100.0                                # A/cm -> A/m
    C = PI ** 2 * alpha_i * A * B                    # W·s/m³
    n = n_rpm / 60.0
    D = (Pi_kw * 1e3 * 2 * p / (C * n * lam * PI)) ** (1 / 3)
    tau = PI * D / (2 * p)
    l = lam * tau
    T = Pi_kw * 1e3 / (2 * PI * n)
    return dict(C=C, C_kwmin=C / 6e4, D=D, l=l, tau=tau, T=T,
                sigma=alpha_i * A * B, V=PI * D ** 2 / 4 * l)


def kphi_rated(UN, IAN, RA, nN):
    """kΦ_N [V·s] = napięcie indukowane przy 1 obr/s (z danych znamionowych)."""
    return (UN - RA * IAN) / (nN / 60.0)


def dc_start_sim(U, RA, LA, kphi, J, TL, t_load, m, Imax, t_end, Bf=0.0):
    """Symulacja rozruchu silnika obcowzbudnego z m-stopniowym rozrusznikiem.
    Równania:  LA·di/dt = U - R(t)·i - kΦ·ω/2π ;   J·dω/dt = kΦ·i/2π - TL - Bf·ω
    Rozrusznik: R1 = U/Imax, iloraz λ = (R1/RA)^(1/m); przełączenie gdy i spadnie do Imax/λ."""
    if m > 0 and U / Imax > RA:
        R1 = U / Imax
        lam = (R1 / RA) ** (1.0 / m)
        Rs = [R1 / lam ** j - RA for j in range(m + 1)]
        Rs[-1] = 0.0
        I_sw = Imax / lam
    else:
        m, Rs, I_sw = 0, [0.0], 0.0
    tau = LA / (RA + Rs[0])
    dt = min(2e-4, tau / 10)
    nst = int(t_end / dt)
    dec = max(1, nst // 3000)
    i = w = 0.0
    stage = 0
    e_loss = 0.0
    sw_times = []
    rec = {k: [] for k in ("t", "i", "n", "T", "TL")}

    def deriv(i_, w_, R_, TLc):
        return ((U - R_ * i_ - kphi * w_ / (2 * PI)) / LA,
                (kphi * i_ / (2 * PI) - TLc - Bf * w_) / J)

    for k in range(nst):
        t = k * dt
        TLc = TL if t >= t_load else 0.0
        R = RA + Rs[stage]
        k1 = deriv(i, w, R, TLc)
        k2 = deriv(i + dt / 2 * k1[0], w + dt / 2 * k1[1], R, TLc)
        k3 = deriv(i + dt / 2 * k2[0], w + dt / 2 * k2[1], R, TLc)
        k4 = deriv(i + dt * k3[0], w + dt * k3[1], R, TLc)
        i += dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        w += dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        e_loss += Rs[stage] * i * i * dt
        if stage < m and t > 0.01 and i <= I_sw and k1[0] < 0:
            stage += 1
            sw_times.append(t)
        if k % dec == 0:
            rec["t"].append(t); rec["i"].append(i); rec["n"].append(w / (2 * PI) * 60)
            rec["T"].append(kphi * i / (2 * PI)); rec["TL"].append(TLc)
    out = {k: np.array(v) for k, v in rec.items()}
    out.update(Rs=Rs, I_sw=I_sw, sw=sw_times, e_loss=e_loss)
    return out


# Magnesy trwałe - Tabela 2.2 (BR [T], αB [%/K], HcJ [kA/m], αH [%/K])
MAGNETS = {"Ferryt": (0.4, -0.19, 170.0, +0.30),
           "NdFeB": (1.2, -0.09, 1900.0, -0.60),
           "SmCo": (1.1, -0.032, 1800.0, -0.19)}
KNEE = 0.9          # kolano charakterystyki: H_limit ≈ -0,9·HcJ (uproszczenie)


def magnet_at(name, T):
    """BR(T) [T] i HcJ(T) [A/m] z liniowymi współczynnikami temperaturowymi."""
    BR, aB, Hc, aH = MAGNETS[name]
    dT = T - 20.0
    return BR * (1 + aB / 100 * dT), Hc * 1e3 * (1 + aH / 100 * dT)


def magnet_points(BR, mur, hM, delta, Theta):
    """Punkt pracy magnesu: bieg jałowy oraz krawędź nabiegająca/zbiegająca pod obciążeniem.
    Prawo przepływu:  HM·hM + Bδ·δ/μ0 = ±Θ ,  BM = Bδ (przekroje równe),  BM = BR + μ0·μr·HM."""
    den = 1 + mur * delta / hM
    B0 = BR / den
    Bon = (BR + MU0 * mur * Theta / hM) / den
    Boff = (BR - MU0 * mur * Theta / hM) / den
    H = lambda B: (B - BR) / (MU0 * mur)
    return (H(B0), B0), (H(Bon), Bon), (H(Boff), Boff)


def shunt_magnetization(IF, n_ratio, Ur, Us=250.0, I0=1.0):
    """Charakterystyka biegu jałowego Ui(IF) z magnetyzmem szczątkowym Ur i nasyceniem."""
    return n_ratio * (Ur + Us * np.tanh(np.asarray(IF, dtype=float) / I0))


def shunt_noload_point(n_ratio, Ur, Rtot, Us=250.0, I0=1.0):
    """Przecięcie Ui(IF) z prostą Rtot·IF (bisekcja)."""
    g = lambda x: shunt_magnetization(x, n_ratio, Ur, Us, I0) - Rtot * x
    lo, hi = 0.0, 50.0
    if g(hi) > 0:
        return hi
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if g(mid) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def bridge_ud(theta_deg, alpha, ULL):
    """Napięcie wyjściowe sterowanego mostka 6-pulsowego (tyrystory T1..T6, każdy przewodzi 120°)."""
    th = np.asarray(theta_deg, dtype=float)
    Vm = math.sqrt(2) * ULL / math.sqrt(3)
    ph = [Vm * np.sin(np.radians(th - k * 120)) for k in range(3)]   # va, vb, vc
    inw = lambda st: np.mod(th - st, 360.0) < 120.0
    up = np.where(inw(30 + alpha), ph[0], np.where(inw(150 + alpha), ph[1], ph[2]))
    low = np.where(inw(90 + alpha), ph[2], np.where(inw(210 + alpha), ph[0], ph[1]))
    return up - low, ph


def armature_field(alpha, ratio, IA_pu, f, alpha_i, comp, cpole, sat, Bs=1.15):
    """Rozkład przepływów i indukcji w szczelinie (wartości względne B/Bδ,N).
    Środek bieguna N w α = π/2, S w 3π/2, strefy neutralne w α = 0 i π."""
    pole = np.sign(np.sin(alpha))
    gap = np.abs(b_shape(alpha - PI / 2, alpha_i))            # 1 pod biegunem, 0 między biegunami
    aw = np.mod(alpha, 2 * PI)
    tri = 1 - 2 * np.abs(aw - PI) / PI                        # przepływ twornika: trójkąt
    thF = pole / f
    thA = ratio * IA_pu * tri
    mask = gap > 0.999
    thC = np.where(mask, -thA, 0.0) if comp else np.zeros_like(alpha)
    B = gap * (thF + thA + thC)
    thCP = np.zeros_like(alpha)
    if cpole:
        dn = np.minimum(np.abs(np.mod(alpha + PI / 2, PI) - PI / 2), PI)  # odległość od strefy neutralnej
        win = np.clip(1 - dn / (0.08 * PI), 0, 1)
        thCP = -1.3 * thA * (dn < 0.08 * PI)
        B = B + win * (thA + thCP) * (1 - gap)
    if sat:
        B = Bs * np.tanh(B / Bs)
    return thF, thA, thC, thCP, B


def commutation(I, L, Rb, Tc, k_cp, n_steps=2000):
    """Prąd cewki komutowanej (zwartej szczotką) - niejawna metoda Eulera.
    Rezystancje przejścia: krawędź zbiegająca R1 = Rb·Tc/(Tc-t), nabiegająca R2 = Rb·Tc/t.
        L·di/dt = -R1·(I+i) + R2·(I-i) - e_cp ,   e_cp = k_cp·2·L·I/Tc  (bieguny komutacyjne)
    Dla L = 0 daje komutację liniową i = I·(1 - 2t/Tc)."""
    dt = Tc / n_steps
    e_cp = k_cp * 2 * L * I / Tc
    i = np.empty(n_steps + 1)
    i[0] = I
    for k in range(n_steps):
        tn = (k + 1) * dt                                 # niejawnie: rezystancje w chwili t(k+1)
        R1 = Rb * Tc / max(Tc - tn, 1e-9 * Tc)
        R2 = Rb * Tc / tn
        i[k + 1] = (L / dt * i[k] - (R1 - R2) * I - e_cp) / (L / dt + R1 + R2)
    t = np.linspace(0, Tc, n_steps + 1)
    frac = np.clip(1 - t / Tc, 1e-3, 1)
    j1 = np.abs(I + i) / (frac * 2 * I)             # gęstość prądu na krawędzi zbiegającej / średnia
    j1[-1] = j1[-2]                                  # t = Tc: styk zanika (unikamy dzielenia 0/0)
    frac2 = np.clip(t / Tc, 1e-3, 1)
    j2 = np.abs(I - i) / (frac2 * 2 * I)            # krawędź nabiegająca
    j2[0] = j2[1]
    return t, i, j1, j2


# =============================================================================
# OBJAŚNIENIA (dla ucznia liceum)
# =============================================================================

EXPL = {
"T0": """BUDOWA MASZYNY PRĄDU STAŁEGO (rozdz. 2.1)

Wyobraź sobie silniczek z zabawki albo wycieraczek samochodowych - to właśnie maszyna DC.
Ma dwie główne części:
 • STOJAN (część nieruchoma, na zewnątrz) - masywne żelazo + BIEGUNY: czerwone N i niebieskie S.
   Pole magnetyczne robią albo magnesy trwałe, albo cewki (pomarańczowe) zasilane prądem stałym.
 • WIRNIK (twornik, część obracająca się) - pakiet cienkich blach z ŻŁOBKAMI, w których leżą
   miedziane przewody. Między wirnikiem a stojanem jest cieniutka SZCZELINA POWIETRZNA.
 • KOMUTATOR (miedziany walec z działek) + SZCZOTKI (węglowe, czarne) - "przełącznik mechaniczny",
   który doprowadza prąd do obracających się przewodów i ciągle zmienia jego kierunek.

Najważniejsza idea: dzięki komutatorowi prąd w przewodach pod biegunem N płynie ZAWSZE w tę samą
stronę (⊗ = "w kartkę"), a pod biegunem S zawsze w przeciwną (● = "z kartki"). Na przewód z prądem
w polu magnetycznym działa siła F = B·I·l, więc wszystkie siły pchają wirnik w tę samą stronę.
W samych przewodach prąd jest jednak PRZEMIENNY - zmienia kierunek, gdy przewód przechodzi spod N pod S.

SPRÓBUJ: obracaj suwakiem wirnik - zobacz, że znaczki ⊗/● "zostają" pod biegunami, choć przewody jadą dalej.
Zmień p (pary biegunów) - maszyna 4- lub 6-biegunowa ma 2p szczotek (uzwojenie pętlicowe).
Przewody bez znaczka są właśnie w strefie neutralnej - to w nich trwa komutacja (zmiana kierunku prądu).
""",

"T1": """INDUKOWANIE NAPIĘCIA I PROSTOWANIE PRZEZ KOMUTATOR (rozdz. 2.2)

Prawo indukcji (Faraday): przewód o długości l poruszający się z prędkością v w polu B wytwarza napięcie
      u = B·l·v .      Cewka ma dwa boki (pod N i pod S), więc  e = 2·wS·B·l·v  (wS - liczba zwojów).
Prędkość obwodowa  v = 2π·r·n  (n w obr/s).  Pulsacja elektryczna  ω = p·ω_mech  (p - pary biegunów).

WYKRES 1 - rozkład indukcji B wzdłuż obwodu: pod biegunem prawie stała ("płaski dach" o szerokości αi),
            między biegunami spada do zera (strefa neutralna).
WYKRES 2 - napięcie jednej cewki: kształt "kopiuje" rozkład B i jest PRZEMIENNE (+ pod N, - pod S).
            Komutator odwraca końce cewki co pół obrotu -> na szczotkach widzimy |e|: napięcie
            jednokierunkowe, ale mocno "pulsujące".
WYKRES 3 - gdy na obwodzie jest K cewek przesuniętych względem siebie, ich wyprostowane napięcia się sumują.
            Dołki jednych cewek wypełniają szczyty innych -> napięcie coraz bardziej stałe (tętnienia maleją).

SPRÓBUJ: K = 1 -> tętnienia 100 %. K = 4, 8, 16 -> tętnienia maleją do kilku procent.
To dlatego prawdziwy komutator ma dziesiątki działek. Zwiększ n lub B - napięcie rośnie proporcjonalnie.
""",

"T2": """KOMUTACJA I MOMENT POJEDYNCZEJ CEWKI (rozdz. 2.2, rys. 2.5 i 2.6)

Na rysunku: wirnik z jedną cewką (boki pomarańczowy i zielony), komutator z dwóch działek (a, b)
i dwie nieruchome szczotki (czarne). Suwak θ obraca wirnik.
 1) Cewka pod biegunami: prąd płynie szczotka -> działka -> cewka -> druga działka -> szczotka.
    Siła F = I·(l × B) daje moment w kierunku obrotu (strzałki).
 2) Cewka w strefie neutralnej (θ ≈ 90°): każda szczotka dotyka obu działek, cewka jest ZWARTA,
    prąd w niej spada do zera i zmienia znak - to KOMUTACJA. Moment tej cewki = 0, wirnik kręci się
    dalej dzięki bezwładności (jak rower, gdy przestajesz na chwilę pedałować).
 3) Po obrocie o 180° prąd w cewce ma przeciwny kierunek, ale cewka jest też pod przeciwnym biegunem,
    więc moment ma TEN SAM znak co w 1). To jest cała "magia" komutatora!

Przyjmujemy (1. przybliżenie), że prąd w czasie komutacji zmienia się LINIOWO (szerokość szczotki βB).
Silnik: I > 0 -> moment dodatni, energia elektryczna zamienia się w mechaniczną, U = Ui + R·I.
Prądnica: I < 0 (suwak ujemny) -> moment hamujący, maszyna oddaje energię, U = Ui - R·|I|.

SPRÓBUJ: K = 1 i przesuwaj θ - moment "dziurawy". K = 12 - moment prawie stały.
""",

"T3": """UZWOJENIA TWORNIKA: PĘTLICOWE I FALISTE (rozdz. 2.3)

Oznaczenia:  N - liczba żłobków, u - boki cewek obok siebie w jednym żłobku, wS - zwoje w cewce,
             K = N·u - liczba cewek = liczba działek komutatora,  z = 2·wS·K - liczba przewodów.
Uzwojenie dwuwarstwowe: bok "w przód" leży w górnej warstwie (linia ciągła), bok powrotny w dolnej
(linia przerywana) - przesunięty o szerokość cewki y1 ≈ podziałka biegunowa N/(2p).

PĘTLICOWE (lap): koniec cewki łączy się z SĄSIEDNIĄ działką -> krok komutatorowy y = 1.
   Uzwojenie "zawija pętle" pod jedną parą biegunów. Gałęzi równoległych: 2a = 2p.
   Duży prąd dzieli się na wiele gałęzi -> dobre dla maszyn niskonapięciowych, wysokoprądowych.
FALISTE (wave): koniec cewki idzie pod NASTĘPNĄ parę biegunów, "falą" dookoła wirnika.
   Krok y = (K ± 1)/p (musi być liczbą całkowitą!). Zawsze 2a = 2 gałęzie - wystarczą 2 szczotki.
   Więcej przewodów w szeregu -> wyższe napięcie przy tym samym wirniku.

Kolorowa ścieżka na wykresie = jedna gałąź równoległa (K/2a cewek) prześledzona od działki 0.
SPRÓBUJ: p = 2, pętlicowe - ścieżka zostaje pod jedną parą biegunów. Przełącz na faliste - obiega cały
wirnik. Zobacz, dla jakich K i p uzwojenie faliste jest niewykonalne (K±1 niepodzielne przez p).
""",

"T4": """RÓWNANIA GŁÓWNE MASZYNY DC (rozdz. 2.4.1 - 2.4.4)

Trzy równania, które opisują KAŻDĄ maszynę prądu stałego:
  (1) napięcie indukowane   Ui = k·Φ·n        k = p·z/a  - "stała maszyny"  (n w obr/s)
  (2) moment                T  = k·Φ·IA / 2π
  (3) napięcie zacisków     U  = Ui + R·IA + L·dIA/dt     (w stanie ustalonym bez L·di/dt)
Mnożąc (3) przez IA dostajemy BILANS MOCY:
  U·IA  =  Ui·IA  +  R·IA²  +  d(½·L·IA²)/dt
  moc elektryczna = moc wewnętrzna (-> mechaniczna 2π·n·T)  +  straty w miedzi  +  zmiana energii pola.
Sprawdzenie: Ui·IA = kΦn·IA = 2π·n·(kΦ·IA/2π) = 2π·n·T  - zgadza się!

Przesunięcie szczotek o kąt β: szczotki "obejmują" mniejszy strumień, więc napięcie maleje (wykres 3).
Dlatego szczotki ustawia się w strefie neutralnej (β = 0).

SPRÓBUJ: podwój Φ - napięcie przy tej samej prędkości też się podwaja, a do tego samego momentu
wystarcza połowa prądu. Zmień liczbę gałęzi 2a - przy tych samych przewodach zmienia się k.
""",

"T5": """WSPÓŁCZYNNIK WYKORZYSTANIA - LICZBA ESSONA (rozdz. 2.4.5) - PRZYKŁAD Z KSIĄŻKI

Jak duża musi być maszyna? Ograniczają ją dwa materiały:
 • ŻELAZO - indukcja B nie może być za duża (nasycenie), typowo B ≈ 0,8 T,
 • MIEDŹ - okład prądowy A (ampery na centymetr obwodu wirnika) nie może być za duży (grzanie), A ≈ 500 A/cm.
Z nich:  C = π²·αi·A·B   i   moc wewnętrzna  Pi = C·D²·l·n   (D - średnica, l - długość wirnika).

PRZYKŁAD (Gerling): αi = 0,65, A = 500 A/cm, B = 0,8 T  ->  C = 4,28 kW·min/m³.
Projektujemy Pi = 100 kW, n = 2000 obr/min, p = 2, przyjmując l = τp (λ = 1):
   τp = π·D/(2p) = l   ->   D³ = 2p·Pi / (π·C·n)   ->   D ≈ 0,246 m,  l ≈ 0,193 m.
Wniosek: C ~ T / (objętość wirnika)  oraz  σ = F/(pow. wirnika) = αi·A·B  (tu ≈ 26 kN/m² = 0,26 bar!).
Moment zależy od OBJĘTOŚCI, nie od mocy - wolnobieżna maszyna tej samej mocy jest znacznie większa.

SPRÓBUJ: zmniejsz n do 500 obr/min - średnica rośnie ~ ∛4 ≈ 1,6 razy. Zwiększ B lub A (lepsze chłodzenie)
- maszyna maleje. Zmień λ = l/τp - maszyna "długa i chuda" albo "krótka i gruba".
""",

"T6": """PRZEWODY W ŻŁOBKACH - DOKŁADNIEJSZE SPOJRZENIE (rozdz. 2.5)

Zagadka: przewody leżą w ŻŁOBKACH, a strumień "omija" je, płynąc przez żelazne zęby. Pole w żłobku jest
prawie zerowe - czy więc napięcie i siła w ogóle powstają?
 • NAPIĘCIE: liczymy z prawa Faradaya dla całej cewki, u = -dΨ/dt. Strumień objęty cewką o szerokości τp
   zmienia się liniowo przy przesuwaniu cewki -> u = 2·Bδ·l·v, dokładnie jak dla przewodu w szczelinie.
   Indukcja w szczelinie z prawa przepływu:  Bδ = μ0·ΘF/(2δ)  (ΘF - przepływ wzbudzenia, δ - szczelina).
 • SIŁA: prąd w przewodzie osłabia pole w jednym zębie (B1 = Bδ - μ0·I/2δ), a wzmacnia w drugim
   (B2 = Bδ + μ0·I/2δ). Energia pola (B²/2μ0 na m³) zmienia się przy obrocie, a siła = zmiana energii/przesunięcie:
      F = ΔW/Δx = Bδ·l·I  na jeden przewód - znów tak, jakby przewód leżał w szczelinie!
   Siła działa na ZĘBY żelaza, nie na miedź - dzięki temu przewody nie są "wyrywane" ze żłobków.

WYKRESY: 1) B(x) i położenie cewki (zakreskowany strumień), 2) Ψ i u w funkcji położenia cewki,
3) indukcje B1, B2 w zębach oraz siła z metody energetycznej vs wzór B·l·I (pokrywają się).
""",

"T7": """MASZYNA OBCOWZBUDNA - CHARAKTERYSTYKI I REGULACJA PRĘDKOŚCI (rozdz. 2.6)

Wzbudzenie zasilane z osobnego źródła, więc strumień Φ nie zależy od obciążenia. Z trzech równań głównych:
      n = (U - R·IA)/(kΦ)   [obr/s],      T = kΦ·IA/2π
 • bieg jałowy (IA = 0):  n0 = U/(kΦ)          • zwarcie/utyk (n = 0):  I_zw = U/R  - OGROMNY prąd!
Charakterystyka jest "sztywna" - prędkość spada tylko o kilka % od biegu jałowego do obciążenia znamionowego.

TRZY SPOSOBY REGULACJI PRĘDKOŚCI:
 1) Zmniejszenie napięcia U (U/UN < 1): n0 maleje, nachylenie bez zmian. BEZSTRATNE i szybkie.
    Ujemne U -> obroty w drugą stronę.
 2) Osłabienie pola (f = ΦN/Φ > 1): n0 rośnie, prosta bardziej stroma, ten sam prąd daje MNIEJSZY moment.
    Prawie bezstratne, ale wolniejsze (duża stała czasowa uzwojenia wzbudzenia).
 3) Rezystor szeregowy RS: n0 bez zmian, prosta bardziej stroma, prąd zwarcia mniejszy -> stosowany do ROZRUCHU.
    STRATNE: RS·IA² zamienia się w ciepło.
Ograniczenia (wykres T(n)): IA ≤ IA,N (grzanie), 2π·n·T ≤ PN (moc), n ≤ n_max (siły odśrodkowe).

SPRÓBUJ: ustaw moment obciążenia i zmieniaj każdą z trzech metod - zobacz, gdzie przesuwa się punkt pracy.
""",

"T8": """TRYBY PRACY MASZYNY (Tabela 2.1) - SILNIK, HAMULEC, PRĄDNICA

Przyjmujemy "układ odbiornikowy" (dodatni prąd płynie DO maszyny) i stałe napięcie U > 0.
Suwak przesuwa punkt pracy po prostej n(IA):
 • 0 < IA < I_zw:  n > 0, T > 0  -> SILNIK: bierze moc elektryczną, oddaje mechaniczną.
 • IA < 0:         n > n0, T < 0 -> PRĄDNICA: coś (np. zjeżdżający pociąg) napędza maszynę szybciej niż n0,
                   napięcie indukowane Ui > U i prąd płynie z powrotem do sieci (hamowanie odzyskowe!).
 • IA > I_zw:      n < 0, T > 0  -> HAMULEC PRZECIWPRĄDEM: obciążenie kręci maszynę "do tyłu", maszyna bierze
                   moc i elektryczną, i mechaniczną - wszystko zamienia się w ciepło w R.
Dwa wiersze tabeli są NIEMOŻLIWE: maszyna oddawałaby jednocześnie moc elektryczną i mechaniczną = perpetuum mobile.

Wykres słupkowy: P_el = U·IA, P_mech = 2π·n·T, straty R·IA² - zawsze P_el = P_mech + P_Cu (zasada zachowania energii).
""",

"T9": """ROZRUCH SILNIKA - SYMULACJA DYNAMICZNA (równania 2.17 + równanie ruchu)

Tu nie ma już stanu ustalonego - liczymy krok po kroku (metoda Rungego-Kutty 4. rzędu) dwa równania:
     LA·di/dt = U - R·i - kΦ·ω/2π          (obwód elektryczny)
     J·dω/dt  = kΦ·i/2π - T_obc            (ruch: II zasada dynamiki Newtona dla obrotu)
W chwili startu wirnik stoi, więc Ui = 0 i prąd ograniczają tylko R i L: I ≈ U/RA - kilkanaście razy
więcej niż znamionowy! To może spalić komutator i szarpnąć mechanizmem.

ROZRUSZNIK OPOROWY (m stopni): na start dajemy duży opór R1 = U/Imax. Gdy silnik się rozpędzi,
Ui rośnie, prąd spada do I2 = Imax/λ - wtedy wyłączamy kawałek oporu i prąd znów skacze do Imax.
Iloraz λ = (R1/RA)^(1/m) dobrany tak, by po m krokach został tylko RA ("piła" na wykresie prądu).
Cena: energia stracona w rezystorach (wynik po lewej).

SPRÓBUJ: m = 0 - rozruch bezpośredni, zobacz szczyt prądu. m = 4 - prąd ładnie ograniczony.
Zwiększ J (ciężkie koło zamachowe) - rozruch trwa dłużej, straty w rozruszniku rosną.
Obciążenie włącza się w chwili t_obc - prędkość lekko spada, prąd rośnie do nowej wartości.
""",

"T10": """MASZYNA Z MAGNESAMI TRWAŁYMI (rozdz. 2.7) - CZY MAGNES SIĘ NIE ROZMAGNESUJE?

Magnesy zastępują uzwojenie wzbudzenia: mniejsza, lżejsza, tańsza i sprawniejsza maszyna (brak strat
wzbudzenia), ale NIE DA SIĘ osłabiać pola. Magnes opisuje prosta w II ćwiartce:  BM = BR + μ0·μr·HM
(BR - pozostałość magnetyczna). Magnes pracuje z UJEMNYM polem HM (sam siebie "rozmagnesowuje" przez szczelinę).
Punkt pracy (bieg jałowy) z prawa przepływu:  HM·hM + Bδ·δ/μ0 = 0   ->   BM = BR/(1 + μr·δ/hM).

Pod obciążeniem prąd twornika dodaje przepływ Θ = A·αi·τp/2:
  • krawędź NABIEGAJĄCA - pole wzmocnione (punkt w górę),
  • krawędź ZBIEGAJĄCA  - pole osłabione (punkt w dół, w stronę kolana!).
Jeśli HM przekroczy H_limit (kolano), magnes TRWALE traci część magnetyzmu.
Temperatura (Tab. 2.2): ferryt - koercja ROŚNIE z temperaturą, więc niebezpieczny jest MRÓZ;
NdFeB - koercja szybko MALEJE, więc niebezpieczne jest GORĄCO. Środek zaradczy: grubszy magnes hM.

SPRÓBUJ: Ferryt, T = -40 °C, A = 300 A/cm -> krawędź zbiegająca za kolanem (czerwony alarm).
NdFeB przy 150 °C i cienkim magnesie - to samo. Zwiększ hM - margines wraca.
""",

"T11": """MASZYNA BOCZNIKOWA - SAMOWZBUDZENIE PRĄDNICY (rozdz. 2.8) - "ZASADA DYNAMOELEKTRYCZNA"

Uzwojenie wzbudzenia jest połączone RÓWNOLEGLE z twornikiem - maszyna sama zasila swoje wzbudzenie.
Jak prądnica "odpala" bez zewnętrznego źródła? (Werner von Siemens, 1866)
  1. Żelazo ma zawsze trochę magnetyzmu szczątkowego -> przy obrotach indukuje się małe Ur.
  2. Ur wymusza mały prąd wzbudzenia IF, który WZMACNIA pole -> większe Ui -> większy IF ... (lawina)
  3. Proces się kończy, gdy krzywa Ui(IF) przetnie prostą rezystancji (RA + RF)·IF - punkt stabilny.
Rezystancja obwodu wzbudzenia nie może przekroczyć wartości KRYTYCZNEJ (nachylenie krzywej w zerze),
bo wtedy prosta leży nad krzywą i nic się nie wzbudzi. Odwrotne podłączenie wzbudzenia też psuje sprawę -
prąd osłabia magnetyzm szczątkowy.

OBCIĄŻENIE (wykres 3): gdy pobieramy prąd, napięcie spada -> maleje IF -> napięcie spada jeszcze bardziej.
Maksymalny prąd I_max - dalej napięcie się "załamuje", zostaje tylko mały prąd zwarcia od Ur.
SPRÓBUJ: zwiększaj RF aż do krytycznej - samowzbudzenie zanika. Zmniejsz R_obc - punkt pracy zjeżdża.
""",

"T12": """MASZYNA SZEREGOWA (rozdz. 2.9) - SILNIK TRAMWAJU I ROZRUSZNIKA SAMOCHODOWEGO

Wzbudzenie połączone SZEREGOWO z twornikiem: ten sam prąd tworzy pole i moment. Bez nasycenia Φ ~ IA, więc
     T = L'm·IA² / f        n = f·(U - R·IA) / (2π·L'm·IA)       (f - współczynnik osłabienia pola)
 • moment rośnie z KWADRATEM prądu - ogromny moment rozruchowy (tramwaj, dźwig, rozrusznik),
 • charakterystyka "MIĘKKA": przy obciążeniu prędkość mocno spada,
 • bieg jałowy IA -> 0 daje n -> ∞: silnik się ROZBIEGA! Nigdy nie wolno go uruchamiać bez obciążenia
   (np. pasek klinowy, który może spaść - silnik szeregowy łączy się z obciążeniem na sztywno).
 • zmiana biegunowości napięcia NIE odwraca kierunku obrotów (IA i Φ zmieniają znak jednocześnie).

Regulacja: 1) mniejsze U (bezstratnie), 2) rezystor równoległy do wzbudzenia -> osłabienie pola f > 1
(szybciej, ale mniejszy moment), 3) rezystor szeregowy R_AS (stratnie, do rozruchu).
UWAGA: w modelu pomijamy nasycenie, dlatego moment przy utyku wychodzi nierealnie duży - w praktyce
żelazo się nasyca i Φ przestaje rosnąć.
""",

"T13": """PORÓWNANIE: BOCZNIKOWA, SZEREGOWA I SZEREGOWO-BOCZNIKOWA (KOMPAUNDOWA) (rozdz. 2.10)

Wszystkie trzy krzywe przechodzą przez ten sam punkt znamionowy (n = 1, T = 1 w jednostkach względnych).
Strumień:  kΦ(I) = kΦN·[(1 - s) + s·I/IN]   gdzie s - udział wzbudzenia szeregowego.
 • s = 0  BOCZNIKOWA: charakterystyka SZTYWNA, stała prędkość biegu jałowego - obrabiarki, wentylatory.
 • s = 1  SZEREGOWA:  charakterystyka MIĘKKA, duży moment rozruchowy, ryzyko rozbiegania - trakcja.
 • 0 < s < 1  KOMPAUNDOWA: blisko biegu jałowego zachowuje się jak bocznikowa (skończone n0 - nie rozbiega się),
   pod obciążeniem jak szeregowa (duży moment) - "najlepsze z obu światów", np. prasy, walcarki.

SPRÓBUJ: przesuwaj s od 0 do 1 i patrz, jak fioletowa krzywa "przechodzi" od jednej do drugiej.
""",

"T14": """ZMIENNE NAPIĘCIE ZASILANIA: UKŁAD LEONARDA I MOSTEK TYRYSTOROWY (rozdz. 2.11)

Najlepsza regulacja prędkości to zmiana napięcia twornika (bezstratna, szybka). Skąd wziąć zmienne napięcie DC?
 • DAWNIEJ - UKŁAD LEONARDA: silnik prądu przemiennego kręci prądnicę DC, której wzbudzeniem ΦG sterujemy.
   Napięcie prądnicy UG ~ ΦG zasila silnik. Działa we wszystkich 4 ćwiartkach, ale to 3 maszyny!
 • DZIŚ - PROSTOWNIK STEROWANY (mostek 6-pulsowy z tyrystorami S1..S6). Tyrystor włącza się dopiero,
   gdy dostanie impuls - opóźniamy go o KĄT ZAPŁONU α:
         Ud = (3√2/π)·U_LL·cos α ≈ 1,35·U_LL·cos α
   α < 90° - prostownik (Ud > 0), α > 90° - falownik (Ud < 0, energia wraca do sieci).
   Jeden mostek: prąd tylko w jedną stronę -> 2 ćwiartki. Dwa mostki przeciwsobnie -> 4 ćwiartki.

WYKRES 1: sześć napięć międzyfazowych i "wycinane" z nich napięcie ud (sześć "garbów" na okres).
WYKRES 2: kiedy przewodzi który tyrystor (każdy 120°).   WYKRES 3: dostępne ćwiartki Ud-Id.
SPRÓBUJ: α = 0° -> maksymalne napięcie. α = 60° -> połowa. α = 120° -> ujemne średnie napięcie.
""",

"T15": """ODDZIAŁYWANIE TWORNIKA I UZWOJENIE KOMPENSACYJNE (rozdz. 2.12, 2.13, rys. 2.43, 2.45, 2.50)

Na biegu jałowym pole robi tylko wzbudzenie (ΘF, prostokąty pod biegunami). Gdy płynie prąd twornika, on też
tworzy przepływ ΘA - "trójkątny", skierowany PROSTOPADLE do osi biegunów (max w strefie neutralnej).
Suma = pole WYPACZONE: pod jedną krawędzią bieguna silniejsze, pod drugą słabsze.
Skutki:
 • magnetyczna strefa neutralna przesuwa się (silnik - przeciw kierunkowi obrotów, prądnica - zgodnie),
 • napięcie między działkami komutatora lokalnie mocno rośnie (ryzyko przeskoku iskry, "ogień okrężny"),
 • przy NASYCENIU wzmocniona krawędź nie może urosnąć, a osłabiona maleje -> ŚREDNI strumień i moment maleją,
 • przy osłabieniu pola (f > 1) pole pod krawędzią może zmienić ZNAK!
Lekarstwo: UZWOJENIE KOMPENSACYJNE w nabiegunnikach, zasilane prądem twornika w przeciwną stronę ->
znosi ΘA pod biegunami dla KAŻDEGO obciążenia (drogie - tylko duże maszyny).
BIEGUNY KOMUTACYJNE (między biegunami głównymi) wytwarzają małe pole w strefie neutralnej, które pomaga
komutować prąd (zakładka 16).

SPRÓBUJ: włącz nasycenie i zwiększ IA - wynik "strumień względny" spada poniżej 100 %. Włącz kompensację.
""",

"T16": """KOMUTACJA RZECZYWISTA I BIEGUNY KOMUTACYJNE (rozdz. 2.13, rys. 2.46 - 2.49)

W czasie Tc = bB/(π·DC·n) szczotka zwiera cewkę, a jej prąd musi się zmienić z +I na -I (I = IA/2a).
IDEALNIE - komutacja liniowa (szara prosta), wymuszana przez rezystancję styku szczotki.
W RZECZYWISTOŚCI cewka ma indukcyjność L. Szybka zmiana prądu indukuje napięcie reaktancyjne
      e_r = L·di/dt ≈ 2·L·I/Tc  ~  IA·n
które (reguła Lenza) HAMUJE zmianę prądu: prąd "nie nadąża" i musi przeskoczyć na końcu -
na krawędzi zbiegającej szczotki powstaje ISKRZENIE (niszczy szczotki i komutator).
Rozwiązanie: BIEGUNY KOMUTACYJNE między biegunami głównymi, połączone szeregowo z twornikiem.
Ich pole indukuje w komutowanej cewce napięcie e_cp ~ IA·n, czyli rośnie DOKŁADNIE tak jak e_r ->
kompensacja działa przy każdym obciążeniu i prędkości. Dostraja się ją szczeliną pod biegunem δ_CP.
 • k_cp < 1 - niedokompensowanie: komutacja opóźniona (rys. 2.47),
 • k_cp = 1 - komutacja liniowa, brak iskrzenia,
 • k_cp > 1 - przekompensowanie: komutacja przyspieszona, znów iskrzenie (rys. 2.49).
Wykres 2 pokazuje gęstość prądu na krawędziach szczotki (1 = równomierna; duże wartości = iskry).
""",
}


# =============================================================================
# GUI - klasa bazowa zakładki
# =============================================================================

class ExampleTab(ttk.Frame):
    """Zakładka: lewy panel suwaków + wyniki liczbowe, prawy - wykresy,
    dół - objaśnienie. Podklasy definiują PARAMS, key i update_plot()."""
    PARAMS = []   # (nazwa, etykieta, min, max, domyślna, krok)
    key = ""

    def __init__(self, master):
        super().__init__(master)
        self.vars = {}
        self._pending = None
        pw = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        pw.pack(fill=tk.BOTH, expand=True)
        left = ttk.Frame(pw, width=310)
        right = ttk.Frame(pw)
        pw.add(left, weight=0)
        pw.add(right, weight=1)

        ttk.Label(left, text="Parametry", font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=6, pady=(6, 2))
        for name, label, lo, hi, val, res in self.PARAMS:
            fr = ttk.Frame(left)
            fr.pack(fill=tk.X, padx=6, pady=1)
            var = tk.DoubleVar(value=val)
            self.vars[name] = var
            ttk.Label(fr, text=label).pack(anchor="w")
            sc = tk.Scale(fr, from_=lo, to=hi, resolution=res, orient=tk.HORIZONTAL,
                          variable=var, showvalue=True, length=220,
                          command=lambda _e: self.schedule())
            sc.pack(fill=tk.X, expand=True)
        self.extra_controls(left)
        ttk.Button(left, text="Przywróć domyślne", command=self.reset).pack(fill=tk.X, padx=6, pady=6)
        ttk.Label(left, text="Wyniki", font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=6)
        self.result = tk.Text(left, height=14, width=38, font=("Consolas", 9), bg="#f4f6fa")
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

    # --- pomocnicze kontrolki dla podklas ---
    def extra_controls(self, parent):
        pass

    def add_radio(self, parent, title, options, default):
        var = tk.StringVar(value=default)
        box = ttk.LabelFrame(parent, text=title)
        box.pack(fill=tk.X, padx=6, pady=4)
        for val, txt in options:
            ttk.Radiobutton(box, text=txt, value=val, variable=var,
                            command=self.schedule).pack(anchor="w", padx=4)
        return var

    def add_check(self, parent, text, default=False):
        var = tk.BooleanVar(value=default)
        ttk.Checkbutton(parent, text=text, variable=var, command=self.schedule).pack(anchor="w", padx=8)
        return var

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
        try:
            lines = self.update_plot() or []
        except Exception as exc:          # nie pozwól, by błąd obliczeń zamknął aplikację
            lines = [f"Błąd obliczeń: {exc}"]
        self.result.delete("1.0", tk.END)
        self.result.insert("1.0", "\n".join(lines))
        self.canvas.draw_idle()

    def update_plot(self):
        raise NotImplementedError


# =============================================================================
# RYSUNKI POMOCNICZE
# =============================================================================

def draw_machine(ax, p, rot, ai, Ns):
    """Przekrój maszyny DC: jarzmo, 2p biegunów z cewkami, wirnik ze żłobkami, komutator, szczotki."""
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(Circle((0, 0), 1.25, color="#8d96a0"))
    ax.add_patch(Circle((0, 0), 1.06, color="white"))
    pitch = 180.0 / p
    for k in range(2 * p):
        c = 90 + k * pitch
        w = ai * pitch
        col = "#d9534f" if k % 2 == 0 else "#337ab7"
        ax.add_patch(Wedge((0, 0), 1.07, c - w / 2, c + w / 2, width=0.41, color=col, alpha=0.85))
        cr = math.radians(c)
        ax.text(0.86 * math.cos(cr), 0.86 * math.sin(cr), "N" if k % 2 == 0 else "S",
                color="white", ha="center", va="center", fontsize=12, fontweight="bold")
        for side in (-1, 1):
            a0 = c + side * (w / 2 + 2.5)
            ax.add_patch(Wedge((0, 0), 1.03, a0 - 2.5, a0 + 2.5, width=0.25, color="#e69f00"))
        # strumień: N -> wirnik, wirnik -> S
        r_out, r_in = 0.66, 0.30
        if k % 2 == 0:
            xy0, xy1 = (r_out * math.cos(cr), r_out * math.sin(cr)), (r_in * math.cos(cr), r_in * math.sin(cr))
        else:
            xy1, xy0 = (r_out * math.cos(cr), r_out * math.sin(cr)), (r_in * math.cos(cr), r_in * math.sin(cr))
        ax.add_patch(FancyArrowPatch(xy0, xy1, arrowstyle="-|>", mutation_scale=12,
                                     color="#2ca02c", lw=1.3, ls="--", zorder=6))
    ax.add_patch(Circle((0, 0), 0.62, color="#c7ccd1"))
    for j in range(Ns):
        ang = math.radians(rot + j * 360.0 / Ns)
        x, y = 0.53 * math.cos(ang), 0.53 * math.sin(ang)
        ax.add_patch(Circle((x, y), 0.045, color="white", ec="0.3", lw=0.6, zorder=5))
        b = float(b_shape(p * (ang - PI / 2), ai))
        if b > 0.3:
            ax.plot(x, y, marker="x", color="#b22222", ms=6, mew=1.8, zorder=7)
        elif b < -0.3:
            ax.plot(x, y, marker="o", color="#1f4e8c", ms=4, zorder=7)
    ax.add_patch(Circle((0, 0), 0.26, color="#b87333", zorder=6))
    nseg = 12
    for j in range(nseg):
        a = math.radians(rot + j * 360 / nseg)
        ax.plot([0.13 * math.cos(a), 0.26 * math.cos(a)], [0.13 * math.sin(a), 0.26 * math.sin(a)],
                color="#5a3a1a", lw=0.8, zorder=7)
    ax.add_patch(Circle((0, 0), 0.12, color="#333333", zorder=8))
    for k in range(2 * p):
        a = math.radians(90 + pitch / 2 + k * pitch)
        cx, cy = 0.31 * math.cos(a), 0.31 * math.sin(a)
        ax.add_patch(Rectangle((cx - 0.05, cy - 0.05), 0.1, 0.1, angle=0, color="black", zorder=9))
        ax.text(0.40 * math.cos(a), 0.40 * math.sin(a), "+" if k % 2 == 0 else "−",
                ha="center", va="center", fontsize=11, fontweight="bold", zorder=10, color="k")
    ax.add_patch(FancyArrowPatch((1.33 * math.cos(math.radians(60)), 1.33 * math.sin(math.radians(60))),
                                 (1.33 * math.cos(math.radians(20)), 1.33 * math.sin(math.radians(20))),
                                 connectionstyle="arc3,rad=-0.25", arrowstyle="-|>", mutation_scale=15, lw=1.5))
    ax.text(1.28, 0.95, "n", fontsize=12)
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.45, 1.45)


def draw_coil(ax, theta_deg, i_now, torque_now):
    """Wirnik z jedną cewką i dwudziałkowym komutatorem (rys. 2.5)."""
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(Rectangle((-1.6, -0.8), 0.35, 1.6, color="#d9534f"))
    ax.add_patch(Rectangle((1.25, -0.8), 0.35, 1.6, color="#337ab7"))
    ax.text(-1.43, 0, "N", color="white", ha="center", va="center", fontsize=14, fontweight="bold")
    ax.text(1.43, 0, "S", color="white", ha="center", va="center", fontsize=14, fontweight="bold")
    ax.add_patch(Circle((0, 0), 1.0, color="#d0d4d9"))
    phi1 = math.radians(180 - theta_deg)
    sides = [(phi1, "#e69f00", 1), (phi1 + PI, "#2ca02c", -1)]
    for ang, col, sgn in sides:
        x, y = 0.85 * math.cos(ang), 0.85 * math.sin(ang)
        ax.add_patch(Circle((x, y), 0.11, color=col, zorder=5))
        cur = sgn * i_now
        sym = "×" if cur > 1e-6 else ("•" if cur < -1e-6 else "0")
        ax.text(x, y, sym, ha="center", va="center", fontsize=13, fontweight="bold", zorder=6)
        if abs(torque_now) > 1e-9:
            tx, ty = math.sin(ang), -math.cos(ang)            # obrót zgodny z ruchem wskazówek
            s = 0.45 * np.sign(torque_now)
            ax.add_patch(FancyArrowPatch((x, y), (x + s * tx, y + s * ty), arrowstyle="-|>",
                                         mutation_scale=14, color="k", lw=1.6, zorder=7))
    ax.plot([0.85 * math.cos(phi1), 0.85 * math.cos(phi1 + PI)],
            [0.85 * math.sin(phi1), 0.85 * math.sin(phi1 + PI)], color="0.4", lw=0.8, ls=":")
    g = 6.0
    base = math.degrees(phi1)
    ax.add_patch(Wedge((0, 0), 0.38, base - 90 + g, base + 90 - g, width=0.13, color="#e69f00", zorder=4))
    ax.add_patch(Wedge((0, 0), 0.38, base + 90 + g, base + 270 - g, width=0.13, color="#2ca02c", zorder=4))
    am = math.radians(base)
    ax.text(0.16 * math.cos(am), 0.16 * math.sin(am), "a", ha="center", va="center", fontsize=9)
    ax.text(-0.16 * math.cos(am), -0.16 * math.sin(am), "b", ha="center", va="center", fontsize=9)
    ax.add_patch(Rectangle((0.38, -0.07), 0.2, 0.14, color="black", zorder=6))
    ax.add_patch(Rectangle((-0.58, -0.07), 0.2, 0.14, color="black", zorder=6))
    ax.text(-0.66, 0.16, "+", fontsize=12, fontweight="bold")
    ax.text(0.58, 0.16, "−", fontsize=12, fontweight="bold")
    ax.set_xlim(-1.7, 1.7)
    ax.set_ylim(-1.15, 1.15)


# =============================================================================
# ZAKŁADKI
# =============================================================================

class Tab0(ExampleTab):
    key = "T0"
    PARAMS = [("p", "p - liczba par biegunów", 1, 3, 1, 1),
              ("rot", "kąt obrotu wirnika [°]", 0, 360, 0, 1),
              ("ai", "αi - łuk bieguna / podziałka", 0.5, 0.9, 0.7, 0.01),
              ("Ns", "N - liczba żłobków wirnika", 8, 36, 16, 1),
              ("n", "n [obr/min]", 100, 3000, 1500, 50)]

    def update_plot(self):
        p, rot, ai, Ns, n = int(self.p("p")), self.p("rot"), self.p("ai"), int(self.p("Ns")), self.p("n")
        ax = self.fig.add_subplot(1, 2, 1)
        draw_machine(ax, p, rot, ai, Ns)
        ax.set_title(f"Przekrój maszyny DC, 2p = {2 * p}\n× prąd w kartkę   • prąd z kartki", fontsize=10)
        ax2 = self.fig.add_subplot(1, 2, 2)
        phi = np.linspace(0, 360, 1441)
        b = b_shape(p * np.radians(phi - 90), ai)
        ax2.fill_between(phi, b, color="#f0ad4e", alpha=0.35)
        ax2.plot(phi, b, color="#b35900", lw=1.8, label="B(φ)/B_max")
        slots = np.mod(rot + np.arange(Ns) * 360.0 / Ns, 360)
        bs = b_shape(p * np.radians(slots - 90), ai)
        cur = np.where(bs > 0.3, 1, np.where(bs < -0.3, -1, 0))
        ax2.scatter(slots[cur > 0], np.full((cur > 0).sum(), 1.25), marker="x", c="#b22222", s=50, label="prąd ×")
        ax2.scatter(slots[cur < 0], np.full((cur < 0).sum(), -1.25), marker="o", c="#1f4e8c", s=25, label="prąd •")
        ax2.scatter(slots[cur == 0], np.zeros((cur == 0).sum()), marker="s", facecolors="none",
                    edgecolors="k", s=40, label="komutacja")
        ax2.set_xlabel("kąt mechaniczny φ [°]")
        ax2.set_ylabel("B / B_max")
        ax2.set_ylim(-1.6, 1.6)
        ax2.set_xlim(0, 360)
        ax2.grid(alpha=0.3)
        ax2.legend(fontsize=8, loc="lower right")
        ax2.set_title("Rozwinięcie obwodu wirnika: pole i prądy w żłobkach", fontsize=10)
        f = p * n / 60
        return [f"Liczba biegunów 2p   = {2 * p}",
                f"Szczotek (pętlicowe) = {2 * p}",
                f"Podziałka biegunowa  = {360 / (2 * p):.0f}° mech.",
                f"Kąt el. = p·kąt mech.: β = {p}·α",
                f"Częstotliwość prądu w",
                f"  przewodach wirnika f = p·n/60 = {f:.1f} Hz",
                "",
                f"Przewodów pod N: {(cur > 0).sum()}, pod S: {(cur < 0).sum()},",
                f"w komutacji: {(cur == 0).sum()}"]


class Tab1(ExampleTab):
    key = "T1"
    PARAMS = [("B", "B_max [T]", 0.2, 1.2, 0.8, 0.05),
              ("l", "l - długość czynna [m]", 0.05, 0.5, 0.2, 0.01),
              ("r", "r - promień wirnika [m]", 0.02, 0.3, 0.08, 0.005),
              ("n", "n [obr/min]", 0, 3000, 1500, 10),
              ("ws", "wS - zwoje na cewkę", 1, 50, 10, 1),
              ("K", "K - liczba cewek (działek)", 1, 32, 1, 1),
              ("p", "p - pary biegunów", 1, 4, 1, 1),
              ("ai", "αi - łuk bieguna", 0.5, 0.95, 0.7, 0.01)]

    def update_plot(self):
        B, l, r, n = self.p("B"), self.p("l"), self.p("r"), self.p("n")
        ws, K, p, ai = int(self.p("ws")), int(self.p("K")), int(self.p("p")), self.p("ai")
        v = 2 * PI * r * n / 60
        e_amp = 2 * ws * B * l * v
        th = np.linspace(0, 4 * PI, 2000)
        e1, u = commutated_voltage(th, K, ai, e_amp)
        deg = np.degrees(th)
        a1 = self.fig.add_subplot(3, 1, 1)
        a1.plot(deg, B * b_shape(th, ai), color="#b35900", lw=1.6)
        a1.fill_between(deg, B * b_shape(th, ai), color="#f0ad4e", alpha=0.3)
        a1.set_ylabel("B [T]"); a1.grid(alpha=0.3)
        a1.set_title("Rozkład indukcji w szczelinie (kąt elektryczny)")
        a2 = self.fig.add_subplot(3, 1, 2, sharex=a1)
        a2.plot(deg, e1, color="0.5", lw=1.2, label="napięcie cewki e (przemienne)")
        a2.plot(deg, np.abs(e1), color="tab:red", lw=1.6, label="po komutatorze |e|")
        a2.set_ylabel("u [V]"); a2.grid(alpha=0.3); a2.legend(fontsize=8, loc="upper right")
        a3 = self.fig.add_subplot(3, 1, 3, sharex=a1)
        a3.plot(deg, u, color="tab:blue", lw=1.6, label=f"napięcie na szczotkach, K = {K}")
        a3.axhline(u.mean(), color="k", ls="--", lw=1, label=f"średnia = {u.mean():.1f} V")
        a3.set_ylim(0, max(u.max() * 1.15, 1e-3))
        a3.set_xlabel("ωt [° el.]"); a3.set_ylabel("U [V]"); a3.grid(alpha=0.3)
        a3.legend(fontsize=8, loc="lower right")
        rip = (u.max() - u.min()) / u.mean() * 100 if u.mean() > 0 else 0
        return [f"v = 2π·r·n     = {v:.2f} m/s",
                f"e_max = 2wS·B·l·v = {e_amp:.1f} V",
                f"f cewki = p·n/60 = {p * n / 60:.1f} Hz",
                f"ω = p·ω_mech   = {2 * PI * p * n / 60:.1f} rad/s",
                "",
                f"U średnie     = {u.mean():.1f} V",
                f"U max / min   = {u.max():.1f} / {u.min():.1f} V",
                f"Tętnienia     = {rip:.1f} %",
                "",
                "Więcej cewek -> mniejsze tętnienia"]


class Tab2(ExampleTab):
    key = "T2"
    PARAMS = [("th", "położenie cewki θ [° el.]", 0, 360, 30, 1),
              ("I", "prąd twornika I [A] (<0 = prądnica)", -20, 20, 10, 0.5),
              ("bb", "szerokość szczotki βB [° el.]", 0, 60, 20, 1),
              ("K", "K - liczba cewek", 1, 24, 1, 1),
              ("B", "B_max [T]", 0.2, 1.2, 0.8, 0.05),
              ("ws", "wS - zwoje na cewkę", 1, 50, 20, 1),
              ("n", "n [obr/min]", 0, 3000, 1500, 10),
              ("R", "R obwodu twornika [Ω]", 0.1, 5, 1.0, 0.1)]
    L_ACT, R_ROT, AI = 0.15, 0.05, 0.7

    def update_plot(self):
        th0, I, bb, K = self.p("th"), self.p("I"), self.p("bb"), int(self.p("K"))
        B, ws, n, R = self.p("B"), int(self.p("ws")), self.p("n"), self.p("R")
        l, r, ai = self.L_ACT, self.R_ROT, self.AI
        Ic = I if K == 1 else I / 2
        th = np.radians(np.linspace(0, 360, 1441))
        Bth = B * b_shape(th, ai)
        ic = coil_current(th, Ic, bb)
        Tc = 2 * ws * Bth * l * r * ic
        Ttot = np.zeros_like(th)
        for k in range(K):
            s = th + 2 * PI * k / K
            Ttot += 2 * ws * B * b_shape(s, ai) * l * r * coil_current(s, Ic, bb)
        t0 = math.radians(th0)
        i_now = float(coil_current(t0, Ic, bb))
        T_now = float(2 * ws * B * b_shape(t0, ai) * l * r * i_now)
        a0 = self.fig.add_subplot(2, 2, 1)
        draw_coil(a0, th0, i_now, T_now)
        stan = "KOMUTACJA (cewka zwarta)" if abs(i_now) < abs(Ic) * 0.999 else "cewka pod biegunami"
        a0.set_title(f"θ = {th0:.0f}°: {stan}", fontsize=10)
        deg = np.degrees(th)
        a1 = self.fig.add_subplot(2, 2, 2)
        a1.plot(deg, Bth / max(B, 1e-9), color="#b35900", label="B/B_max")
        a1.plot(deg, ic / (abs(Ic) + 1e-9), color="tab:blue", lw=1.8, label="i_cewki / I_c")
        a1.axvline(th0, color="k", ls=":")
        a1.axvspan(90 - bb / 2, 90 + bb / 2, color="0.85")
        a1.axvspan(270 - bb / 2, 270 + bb / 2, color="0.85")
        a1.set_xlim(0, 360); a1.grid(alpha=0.3); a1.legend(fontsize=8, loc="lower left")
        a1.set_title("Pole i prąd cewki (szare = komutacja)", fontsize=10)
        a2 = self.fig.add_subplot(2, 2, 3)
        a2.plot(deg, Tc, color="tab:green", lw=1.8)
        a2.plot([th0], [T_now], "ko")
        a2.set_xlim(0, 360); a2.grid(alpha=0.3)
        a2.set_xlabel("θ [° el.]"); a2.set_ylabel("T_cewki [N·m]")
        a2.set_title("Moment jednej cewki - zawsze tego samego znaku", fontsize=10)
        a3 = self.fig.add_subplot(2, 2, 4)
        a3.plot(deg, Ttot, color="tab:purple", lw=1.8)
        a3.axhline(Ttot.mean(), color="k", ls="--", lw=1)
        lo, hi = min(0, Ttot.min()) * 1.2, max(0, Ttot.max()) * 1.2
        a3.set_ylim(lo - 1e-6, hi + 1e-6)
        a3.set_xlim(0, 360); a3.grid(alpha=0.3); a3.set_xlabel("θ [° el.]"); a3.set_ylabel("T [N·m]")
        a3.set_title(f"Moment całkowity, K = {K} cewek", fontsize=10)
        v = 2 * PI * r * n / 60
        e = 2 * ws * B * l * v
        _, uu = commutated_voltage(th, K, ai, e)
        Ui = uu.mean()
        mode = "SILNIK" if I > 0 else ("PRĄDNICA" if I < 0 else "bieg jałowy")
        U = Ui + R * I
        rip = (Ttot.max() - Ttot.min()) / abs(Ttot.mean()) * 100 if abs(Ttot.mean()) > 1e-9 else 0
        return [f"Tryb: {mode}",
                f"Prąd cewki I_c   = {Ic:.2f} A",
                f"Prąd teraz       = {i_now:.2f} A",
                f"Moment cewki     = {T_now:.3f} N·m",
                f"Moment średni    = {Ttot.mean():.3f} N·m",
                f"Tętnienia momentu= {rip:.0f} %",
                "",
                f"Ui (średnie)     = {Ui:.1f} V",
                f"U = Ui + R·I     = {U:.1f} V",
                f"P_el = U·I       = {U * I:.1f} W",
                f"P_i  = Ui·I      = {Ui * I:.1f} W",
                "(prądnica: U = Ui - R·|I|)"]


class Tab3(ExampleTab):
    key = "T3"
    PARAMS = [("p", "p - pary biegunów", 1, 4, 2, 1),
              ("N", "N - liczba żłobków", 6, 36, 12, 1),
              ("u", "u - boki cewek w żłobku", 1, 3, 1, 1),
              ("ws", "wS - zwoje na cewkę", 1, 10, 2, 1),
              ("IA", "IA - prąd twornika [A]", 10, 400, 100, 10)]

    def extra_controls(self, parent):
        self.kind = self.add_radio(parent, "Rodzaj uzwojenia",
                                   [("lap", "pętlicowe (lap)"), ("wave", "faliste (wave)")], "lap")

    def update_plot(self):
        p, N, u, ws, IA = int(self.p("p")), int(self.p("N")), int(self.p("u")), int(self.p("ws")), self.p("IA")
        kind = self.kind.get()
        W = winding(p, N, u, ws, kind)
        ax = self.fig.add_subplot(2, 1, 1)
        if not W["ok"]:
            ax.axis("off")
            ax.text(0.5, 0.5, f"Uzwojenie faliste NIEWYKONALNE:\n(K±1)/p = ({W['K']}±1)/{p} nie jest liczbą całkowitą.\n"
                    "Zmień N, u lub p.", ha="center", va="center", fontsize=13, color="darkred")
        else:
            K, y, y1 = W["K"], W["y"], W["y1"]
            tau = N / (2 * p)
            xmax = N + max(y1, y / u) + 0.8
            for j in range(int(math.ceil(xmax / tau)) + 1):
                x0 = j * tau + 0.15 * tau
                col = "#d9534f" if j % 2 == 0 else "#337ab7"
                ax.add_patch(Rectangle((x0, 2.35), 0.7 * tau, 0.3, color=col, alpha=0.6))
                ax.text(x0 + 0.35 * tau, 2.5, "N" if j % 2 == 0 else "S", ha="center", va="center",
                        color="white", fontweight="bold")
            ax.axvspan(N - 0.1, xmax, color="0.92", zorder=0)
            ax.text(N + 0.1, -0.15, "powtórzenie obwodu →", fontsize=8, color="0.4")
            for j in range(N):
                ax.add_patch(Rectangle((j + 0.05, 1.0), 0.9, 1.0, fill=False, ec="0.75", lw=0.6))
                ax.text(j + 0.5, 0.9, str(j + 1), fontsize=7, ha="center", va="top", color="0.4")
            for s in range(K):
                ax.add_patch(Rectangle((s / u + 0.05 / u, 0.25), 0.9 / u, 0.2, color="#b87333"))
                if K <= 40:
                    ax.text(s / u + 0.5 / u, 0.35, str(s + 1), fontsize=6, ha="center", va="center", color="white")
            npath = max(1, K // W["a2"])
            seq = winding_trace(K, y, npath)
            cmap = matplotlib.colormaps.get_cmap("viridis")

            def draw(s, col, lw, alpha, z):
                xu = s / u + 0.5 / u
                xl = xu + y1
                xs = xu
                xe = xs + y / u
                ax.plot([xu, xu], [1.05, 1.95], color=col, lw=lw, alpha=alpha, zorder=z)
                ax.plot([xl, xl], [1.05, 1.95], color=col, lw=lw, alpha=alpha, ls="--", zorder=z)
                ax.plot([xu, (xu + xl) / 2, xl], [1.95, 2.25, 1.95], color=col, lw=lw, alpha=alpha, zorder=z)
                ax.plot([xs, xu], [0.45, 1.05], color=col, lw=lw, alpha=alpha, zorder=z)
                ax.plot([xl, xe], [1.05, 0.45], color=col, lw=lw, alpha=alpha, zorder=z)
            for s in range(K):
                draw(s, "0.7", 0.6, 0.6, 1)
            for idx, s in enumerate(seq):
                draw(s, cmap(idx / max(len(seq) - 1, 1)), 2.0, 1.0, 3)
            nb = 2 * p
            for k in range(nb):
                xb = k * tau - y1 / 2 + 0.5 / u     # granica działek zwierających cewkę w strefie neutralnej
                xb = xb % N
                ax.add_patch(Rectangle((xb - 0.3, -0.05), 0.6, 0.22, color="black"))
                ax.text(xb, -0.12, "+" if k % 2 == 0 else "−", ha="center", va="top", fontweight="bold")
            ax.set_xlim(-0.3, xmax)
            ax.set_ylim(-0.45, 2.75)
            ax.set_yticks([])
            ax.set_xlabel("obwód wirnika (rozwinięty) [podziałki żłobkowe]")
            ax.set_title(f"Uzwojenie {'pętlicowe' if kind == 'lap' else 'faliste'}: kolor = jedna gałąź "
                         f"({npath} cewek od działki 1), ciągła - warstwa górna, przerywana - dolna", fontsize=9)
        Wl = winding(p, N, u, ws, "lap")
        Ww = winding(p, N, u, ws, "wave")
        labels = ["pętlicowe", "faliste"]
        vals = [(Wl["a2"], Ww["a2"] if Ww["ok"] else 0),
                (IA / Wl["a2"], IA / Ww["a2"] if Ww["ok"] else 0),
                (Wl["z"] / Wl["a2"], Ww["z"] / Ww["a2"] if Ww["ok"] else 0)]
        titles = ["gałęzie równoległe 2a", "prąd przewodu IA/2a [A]", "przewody w szeregu z/2a (~Ui)"]
        for k in range(3):
            a = self.fig.add_subplot(2, 3, 4 + k)
            a.bar(labels, vals[k], color=["tab:blue", "tab:orange"])
            a.set_title(titles[k], fontsize=9)
            a.grid(alpha=0.3, axis="y")
            if not Ww["ok"]:
                a.text(1, 0, "niewykonalne", ha="center", va="bottom", color="darkred", fontsize=8)
        out = [f"K = N·u      = {W['K']}",
               f"z = 2·wS·K   = {W['z']}",
               f"τp = N/2p    = {N / (2 * p):.2f} żłobka",
               f"y1 (cewka)   = {W['y1']} żłobków"]
        if W["ok"]:
            y = W["y"]
            y2 = W["y1"] * u - y if kind == "lap" else y - W["y1"] * u
            out += [f"y  (komut.)  = {y}" + ("  (= y1 - y2)" if kind == "lap" else "  (= (K∓1)/p)"),
                    f"y2 (połącz.) = {y2} [boki cewek]",
                    f"2a           = {W['a2']}",
                    f"cewek/gałąź  = {W['K'] // W['a2']}",
                    f"I przewodu   = {IA / W['a2']:.1f} A"]
        else:
            out.append("Faliste: niewykonalne dla tych danych")
        return out


class Tab4(ExampleTab):
    key = "T4"
    PARAMS = [("z", "z - liczba przewodów", 100, 2000, 600, 10),
              ("p", "p - pary biegunów", 1, 4, 2, 1),
              ("a2", "2a - gałęzie równoległe", 2, 8, 4, 2),
              ("phi", "Φ - strumień na biegun [mWb]", 2, 40, 15, 0.5),
              ("n", "n [obr/min]", 0, 3000, 1500, 10),
              ("IA", "IA [A]", 0, 200, 50, 1),
              ("R", "R twornika [Ω]", 0.05, 2.0, 0.3, 0.01),
              ("beta", "β - przesunięcie szczotek [° el.]", -90, 90, 0, 1)]

    def update_plot(self):
        z, p, a = self.p("z"), int(self.p("p")), self.p("a2") / 2
        phi, n, IA, R, beta = self.p("phi") * 1e-3, self.p("n") / 60, self.p("IA"), self.p("R"), self.p("beta")
        k = p * z / a
        fb = brush_shift_factor(math.radians(beta), 0.7)
        Ui = k * phi * n * fb
        T = k * phi * fb * IA / (2 * PI)
        U = Ui + R * IA
        Pel, Pi, Pcu = U * IA, Ui * IA, R * IA ** 2
        Pm = 2 * PI * n * T
        eta = Pm / Pel * 100 if Pel > 0 else 0
        a1 = self.fig.add_subplot(2, 2, 1)
        bars = a1.bar(["P_el = U·I", "P_i = Ui·I", "P_Cu = R·I²", "P_mech = 2πnT"],
                      np.array([Pel, Pi, Pcu, Pm]) / 1e3,
                      color=["tab:blue", "tab:green", "tab:red", "tab:purple"])
        for b_ in bars:
            a1.text(b_.get_x() + b_.get_width() / 2, b_.get_height(), f"{b_.get_height():.2f}",
                    ha="center", va="bottom", fontsize=8)
        a1.set_ylabel("moc [kW]"); a1.grid(alpha=0.3, axis="y")
        a1.set_title("Bilans mocy: P_el = P_i + P_Cu,  P_i = P_mech", fontsize=10)
        a1.tick_params(axis="x", labelsize=8)
        a2 = self.fig.add_subplot(2, 2, 2)
        nn = np.linspace(0, 3000, 100)
        for fac, st in ((0.5, ":"), (1.0, "-"), (1.5, "--")):
            a2.plot(nn, k * phi * fac * nn / 60 * fb, st, color="tab:green", label=f"Φ·{fac}")
        a2.plot([n * 60], [Ui], "ko")
        a2.set_xlabel("n [obr/min]"); a2.set_ylabel("Ui [V]"); a2.grid(alpha=0.3); a2.legend(fontsize=8)
        a2.set_title("1. równanie: Ui = k·Φ·n", fontsize=10)
        a3 = self.fig.add_subplot(2, 2, 3)
        ii = np.linspace(0, 200, 100)
        for fac, st in ((0.5, ":"), (1.0, "-"), (1.5, "--")):
            a3.plot(ii, k * phi * fac * ii / (2 * PI) * fb, st, color="tab:purple", label=f"Φ·{fac}")
        a3.plot([IA], [T], "ko")
        a3.set_xlabel("IA [A]"); a3.set_ylabel("T [N·m]"); a3.grid(alpha=0.3); a3.legend(fontsize=8)
        a3.set_title("2. równanie: T = k·Φ·IA/2π", fontsize=10)
        a4 = self.fig.add_subplot(2, 2, 4)
        bb = np.linspace(-90, 90, 181)
        a4.plot(bb, [brush_shift_factor(math.radians(x), 0.7) for x in bb], color="tab:orange")
        a4.plot([beta], [fb], "ko")
        a4.set_xlabel("β [° el.]"); a4.set_ylabel("Ui(β)/Ui(0)"); a4.grid(alpha=0.3)
        a4.set_title("Przesunięcie szczotek zmniejsza napięcie", fontsize=10)
        return [f"k = p·z/a        = {k:.0f}",
                f"kΦ               = {k * phi:.3f} V·s",
                f"Ui = kΦn         = {Ui:.1f} V",
                f"T = kΦ·IA/2π     = {T:.1f} N·m",
                f"U = Ui + R·IA    = {U:.1f} V",
                "",
                f"P_el   = {Pel / 1e3:.2f} kW",
                f"P_i    = {Pi / 1e3:.2f} kW",
                f"P_Cu   = {Pcu / 1e3:.3f} kW",
                f"P_mech = {Pm / 1e3:.2f} kW",
                f"η (bez strat Fe i tarcia) = {eta:.1f} %",
                f"Ui(β)/Ui(0) = {fb:.3f}"]


class Tab5(ExampleTab):
    key = "T5"
    PARAMS = [("ai", "αi", 0.5, 0.8, 0.65, 0.01),
              ("A", "A - okład prądowy [A/cm]", 100, 1000, 500, 10),
              ("B", "B [T]", 0.4, 1.1, 0.8, 0.01),
              ("Pi", "Pi - moc wewnętrzna [kW]", 1, 1000, 100, 1),
              ("n", "n [obr/min]", 100, 5000, 2000, 50),
              ("p", "p - pary biegunów", 1, 6, 2, 1),
              ("lam", "λ = l/τp", 0.3, 3.0, 1.0, 0.05)]

    def update_plot(self):
        ai, A, B, Pi, n, p, lam = (self.p(k) for k in ("ai", "A", "B", "Pi", "n", "p", "lam"))
        p = int(p)
        E = esson(ai, A, B, Pi, n, p, lam)
        a1 = self.fig.add_subplot(2, 2, 1)
        PP = np.linspace(1, 1000, 200)
        DD = np.array([esson(ai, A, B, x, n, p, lam)["D"] for x in PP])
        a1.plot(PP, DD, label="D"); a1.plot(PP, lam * PI * DD / (2 * p), label="l")
        a1.plot([Pi], [E["D"]], "ko"); a1.plot([Pi], [E["l"]], "ks")
        a1.set_xlabel("Pi [kW]"); a1.set_ylabel("wymiar [m]"); a1.grid(alpha=0.3); a1.legend(fontsize=8)
        a1.set_title(f"Wymiary vs moc (n = {n:.0f} obr/min)", fontsize=10)
        a2 = self.fig.add_subplot(2, 2, 2)
        NN = np.linspace(100, 5000, 200)
        VV = np.array([esson(ai, A, B, Pi, x, p, lam)["V"] for x in NN]) * 1e3
        a2.plot(NN, VV, color="tab:red"); a2.plot([n], [E["V"] * 1e3], "ko")
        a2.set_xlabel("n [obr/min]"); a2.set_ylabel("objętość wirnika [dm³]"); a2.grid(alpha=0.3)
        a2.set_title(f"Ta sama moc {Pi:.0f} kW - szybsza maszyna jest mniejsza", fontsize=10)
        a3 = self.fig.add_subplot(2, 2, 3)
        a3.set_aspect("equal")
        a3.add_patch(Rectangle((0, -E["D"] / 2), E["l"], E["D"], color="#c7ccd1", ec="k"))
        a3.add_patch(Rectangle((-0.3 * E["l"], -0.03 * E["D"]), 1.6 * E["l"], 0.06 * E["D"], color="#555"))
        a3.annotate("", (0, -0.62 * E["D"]), (E["l"], -0.62 * E["D"]), arrowprops=dict(arrowstyle="<->"))
        a3.text(E["l"] / 2, -0.72 * E["D"], f"l = {E['l'] * 1e3:.0f} mm", ha="center", va="top", fontsize=9)
        a3.annotate("", (1.45 * E["l"], -E["D"] / 2), (1.45 * E["l"], E["D"] / 2), arrowprops=dict(arrowstyle="<->"))
        a3.text(1.5 * E["l"], 0, f"D = {E['D'] * 1e3:.0f} mm", va="center", fontsize=9)
        a3.set_xlim(-0.4 * E["l"], 2.1 * E["l"] + 0.4 * E["D"])
        a3.set_ylim(-0.95 * E["D"], 0.7 * E["D"])
        a3.axis("off"); a3.set_title("Wirnik w skali (widok z boku)", fontsize=10)
        a4 = self.fig.add_subplot(2, 2, 4)
        bars = a4.bar(["C [kW·min/m³]", "σ [kN/m²]", "T [×100 N·m]"], [E["C_kwmin"], E["sigma"] / 1e3, E["T"] / 100],
               color=["tab:blue", "tab:green", "tab:purple"])
        for b_ in bars:
            a4.text(b_.get_x() + b_.get_width() / 2, b_.get_height(), f"{b_.get_height():.2f}",
                    ha="center", va="bottom", fontsize=8)
        a4.grid(alpha=0.3, axis="y"); a4.set_title("Liczba Essona, naprężenie styczne, moment", fontsize=10)
        return [f"C = π²·αi·A·B",
                f"  = {E['C'] / 1e3:.1f} kW·s/m³",
                f"  = {E['C_kwmin']:.2f} kW·min/m³",
                "",
                f"D  = {E['D'] * 1e3:.0f} mm",
                f"l  = {E['l'] * 1e3:.0f} mm",
                f"τp = πD/2p = {E['tau'] * 1e3:.0f} mm",
                f"V = πD²l/4 = {E['V'] * 1e3:.2f} dm³",
                f"T = Pi/(2πn) = {E['T']:.0f} N·m",
                f"σ = αi·A·B = {E['sigma'] / 1e3:.1f} kN/m²",
                "",
                "Książka: C = 4,28 kW·min/m³",
                "D ≈ 246 mm, l ≈ 193 mm"]


class Tab6(ExampleTab):
    key = "T6"
    PARAMS = [("TF", "ΘF - przepływ wzbudzenia [A]", 200, 5000, 2000, 50),
              ("d", "δ - szczelina [mm]", 0.5, 5.0, 1.5, 0.1),
              ("tau", "τp - podziałka biegunowa [mm]", 30, 300, 100, 5),
              ("l", "l [m]", 0.05, 0.5, 0.2, 0.01),
              ("v", "v - prędkość obwodowa [m/s]", 1, 40, 15, 0.5),
              ("x0", "x0 - położenie cewki [τp]", 0, 2, 0.35, 0.01),
              ("I", "I w przewodzie [A]", 0, 1000, 200, 10),
              ("ai", "αi", 0.5, 0.95, 0.7, 0.01)]

    def update_plot(self):
        TF, d, tau, l, v, x0, I, ai = (self.p(k) for k in ("TF", "d", "tau", "l", "v", "x0", "I", "ai"))
        d, tau = d * 1e-3, tau * 1e-3
        Bd = MU0 * TF / (2 * d)
        xs = np.linspace(0, 3, 1201)                       # w jednostkach τp
        Bx = Bd * b_shape(PI * xs - PI / 2, ai)
        a1 = self.fig.add_subplot(3, 1, 1)
        a1.plot(xs, Bx, color="#b35900")
        m = (xs >= x0) & (xs <= x0 + 1)
        a1.fill_between(xs[m], Bx[m], color="tab:green", alpha=0.35, label="strumień objęty cewką")
        a1.axvline(x0, color="k", lw=2); a1.axvline(x0 + 1, color="k", lw=2, ls="--")
        a1.set_ylabel("B [T]"); a1.set_xlabel("x / τp"); a1.grid(alpha=0.3); a1.legend(fontsize=8)
        a1.set_title(f"Bδ = μ0·ΘF/(2δ) = {Bd:.3f} T; cewka o szerokości τp (bok ciągły i przerywany)", fontsize=10)
        X0 = np.linspace(0, 2, 801)
        Xf = np.linspace(0, 3, 3001)
        cum = np.concatenate([[0], np.cumsum(0.5 * (Bd * b_shape(PI * Xf[1:] - PI / 2, ai)
                                                     + Bd * b_shape(PI * Xf[:-1] - PI / 2, ai)) * np.diff(Xf) * tau)])
        Psi = l * (np.interp(X0 + 1, Xf, cum) - np.interp(X0, Xf, cum))
        u = -v * np.gradient(Psi, X0 * tau)
        a2 = self.fig.add_subplot(3, 1, 2)
        a2.plot(X0, Psi * 1e3, color="tab:green", label="Ψ [mWb] (1 zwój)")
        a2b = a2.twinx()
        a2b.plot(X0, u, color="tab:red", label="u = -dΨ/dt [V]")
        a2b.axhline(2 * Bd * l * v, color="tab:red", ls=":", lw=1)
        a2b.axhline(-2 * Bd * l * v, color="tab:red", ls=":", lw=1)
        a2.axvline(x0, color="k", ls=":")
        a2.set_xlabel("położenie cewki x0 / τp"); a2.set_ylabel("Ψ [mWb]", color="tab:green")
        a2b.set_ylabel("u [V]", color="tab:red"); a2.grid(alpha=0.3)
        a2.set_title("Strumień zmienia się liniowo -> u = ±2·Bδ·l·v (kropkowane)", fontsize=10)
        II = np.linspace(0, 1000, 100)
        B1 = Bd - MU0 * II / (2 * d)
        B2 = Bd + MU0 * II / (2 * d)
        dx = 1e-4
        dW = (B2 ** 2 - B1 ** 2) / (2 * MU0) * (2 * l * d * dx)
        F_en = dW / dx / 2
        a3 = self.fig.add_subplot(3, 1, 3)
        a3.plot(II, B1, color="tab:blue", label="B1 (ząb osłabiony)")
        a3.plot(II, B2, color="tab:orange", label="B2 (ząb wzmocniony)")
        a3.set_xlabel("I [A]"); a3.set_ylabel("B [T]"); a3.grid(alpha=0.3)
        a3b = a3.twinx()
        a3b.plot(II, F_en, "k", lw=3, alpha=0.4, label="F z energii")
        a3b.plot(II, Bd * l * II, "m--", label="F = Bδ·l·I")
        a3b.set_ylabel("F [N]")
        h1, l1 = a3.get_legend_handles_labels(); h2, l2 = a3b.get_legend_handles_labels()
        a3.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
        a3.set_title("Siła z metody energetycznej = siła na przewód w szczelinie", fontsize=10)
        k0 = int(np.argmin(np.abs(X0 - x0)))
        return [f"Bδ = μ0ΘF/2δ  = {Bd:.3f} T",
                f"2·Bδ·l·v      = {2 * Bd * l * v:.2f} V",
                f"u(x0) (1 zwój)= {u[k0]:.2f} V",
                f"Ψ(x0)         = {Psi[k0] * 1e3:.3f} mWb",
                "",
                f"B1 = Bδ - μ0I/2δ = {Bd - MU0 * I / (2 * d):.3f} T",
                f"B2 = Bδ + μ0I/2δ = {Bd + MU0 * I / (2 * d):.3f} T",
                f"F = Bδ·l·I      = {Bd * l * I:.1f} N",
                "",
                "Przewody w żłobkach liczymy",
                "tak, jakby leżały w szczelinie."]


class Tab7(ExampleTab):
    key = "T7"
    PARAMS = [("UN", "UN [V]", 100, 600, REF["UN"], 10),
              ("IAN", "IA,N [A]", 10, 200, REF["IAN"], 1),
              ("RA", "RA [Ω]", 0.05, 2.0, REF["RA"], 0.01),
              ("nN", "nN [obr/min]", 500, 3000, REF["nN"], 10),
              ("uU", "1) U/UN - napięcie", -1.0, 1.0, 1.0, 0.01),
              ("f", "2) f = ΦN/Φ - osłabienie pola", 1.0, 3.0, 1.0, 0.05),
              ("RS", "3) RS - opór szeregowy [Ω]", 0.0, 10.0, 0.0, 0.1),
              ("TL", "moment obciążenia T/TN", -1.5, 1.5, 0.8, 0.05),
              ("nmax", "n_max / nN", 1.0, 3.0, 2.0, 0.05)]

    def update_plot(self):
        UN, IAN, RA, nN = (self.p(k) for k in ("UN", "IAN", "RA", "nN"))
        uU, f, RS, TLr, nmr = (self.p(k) for k in ("uU", "f", "RS", "TL", "nmax"))
        kN = kphi_rated(UN, IAN, RA, nN)
        TN = kN * IAN / (2 * PI)
        PN = 2 * PI * nN / 60 * TN
        U, kp, R = uU * UN, kN / f, RA + RS
        I = np.linspace(-2 * IAN, 4 * IAN, 300)
        n0 = UN / kN * 60
        a1 = self.fig.add_subplot(1, 2, 1)
        a1.plot(I, (UN - RA * I) / kN * 60, color="0.6", ls="--", label="n - podstawowa")
        a1.plot(I, (U - R * I) / kp * 60, color="tab:blue", lw=2, label="n - po zmianie")
        a1.axhline(0, color="k", lw=0.6); a1.axvline(0, color="k", lw=0.6)
        a1.axvline(IAN, color="tab:red", ls=":", lw=1)
        a1.set_xlabel("IA [A]"); a1.set_ylabel("n [obr/min]", color="tab:blue"); a1.grid(alpha=0.3)
        a1b = a1.twinx()
        a1b.plot(I, kN * I / (2 * PI), color="0.6", ls="--")
        a1b.plot(I, kp * I / (2 * PI), color="tab:purple", lw=2, label="T - po zmianie")
        a1b.set_ylabel("T [N·m]", color="tab:purple")
        IL = 2 * PI * TLr * TN / kp
        nL = (U - R * IL) / kp * 60
        a1.plot([IL], [nL], "ko", ms=8)
        h1, l1 = a1.get_legend_handles_labels(); h2, l2 = a1b.get_legend_handles_labels()
        a1.legend(h1 + h2, l1 + l2, fontsize=8, loc="lower left")
        a1.set_title("Charakterystyki n(IA) i T(IA) (rys. 2.16 - 2.20)", fontsize=10)
        a2 = self.fig.add_subplot(1, 2, 2)
        nn = np.linspace(-0.5 * n0, 2.5 * n0, 300)
        Tb = kN / (2 * PI) * (UN - kN * nn / 60) / RA
        Tm = kp / (2 * PI) * (U - kp * nn / 60) / R
        a2.plot(nn, Tb, color="0.6", ls="--", label="podstawowa")
        a2.plot(nn, Tm, color="tab:blue", lw=2, label="po zmianie")
        a2.axhline(TLr * TN, color="tab:green", lw=1.2, label="obciążenie")
        a2.axhline(kp * IAN / (2 * PI), color="tab:red", ls=":", label="IA ≤ IA,N")
        a2.axhline(-kp * IAN / (2 * PI), color="tab:red", ls=":")
        nh = np.linspace(nN * 0.3, 2.5 * n0, 200)
        a2.plot(nh, PN / (2 * PI * nh / 60), color="tab:orange", ls="-.", label="P ≤ PN")
        a2.axvline(nmr * nN, color="k", ls=":", label="n ≤ n_max")
        a2.plot([nL], [TLr * TN], "ko", ms=8)
        a2.set_ylim(-2.5 * TN, 3 * TN); a2.set_xlim(nn[0], nn[-1])
        a2.axhline(0, color="k", lw=0.6); a2.axvline(0, color="k", lw=0.6)
        a2.set_xlabel("n [obr/min]"); a2.set_ylabel("T [N·m]"); a2.grid(alpha=0.3)
        a2.legend(fontsize=8, loc="upper right")
        a2.set_title("Moment w funkcji prędkości T(n) (rys. 2.21) + ograniczenia", fontsize=10)
        Pel = U * IL
        Pm = 2 * PI * nL / 60 * TLr * TN
        warn = []
        if abs(IL) > IAN * 1.001:
            warn.append("! IA > IA,N - przegrzanie")
        if abs(Pm) > PN * 1.001:
            warn.append("! P > PN - przeciążenie")
        if abs(nL) > nmr * nN:
            warn.append("! n > n_max - siły odśrodkowe")
        return [f"kΦN = {kN:.3f} V·s", f"TN = {TN:.1f} N·m, PN = {PN / 1e3:.1f} kW",
                f"n0 (podst.) = {n0:.0f} obr/min",
                f"I_zw (podst.) = UN/RA = {UN / RA:.0f} A",
                f"T_zw (podst.) = {kN * UN / RA / (2 * PI):.0f} N·m",
                "", "Po zmianie:",
                f"n0   = {U / kp * 60:.0f} obr/min",
                f"I_zw = {U / R:.0f} A",
                f"Punkt pracy: IA = {IL:.1f} A",
                f"             n  = {nL:.0f} obr/min",
                f"Straty w RS = {RS * IL ** 2 / 1e3:.2f} kW",
                f"η ≈ {Pm / Pel * 100:.1f} %" if Pel > 0 and Pm > 0 else "η: -"] + warn


class Tab8(ExampleTab):
    key = "T8"
    PARAMS = [("x", "IA / I_zw", -0.4, 1.4, 0.05, 0.005)]
    ROWS = [("> 0", "> 0", "> 0", "> 0", "> 0", "silnik (n > 0)"),
            ("> 0", "> 0", "< 0", "> 0", "< 0", "hamowanie (n > 0)"),
            ("> 0", "< 0", "> 0", "> 0", "< 0", "hamowanie (n < 0)"),
            ("> 0", "< 0", "< 0", "> 0", "> 0", "silnik (n < 0)"),
            ("< 0", "> 0", "> 0", "< 0", "> 0", "niemożliwe"),
            ("< 0", "> 0", "< 0", "< 0", "< 0", "prądnica (n > 0)"),
            ("< 0", "< 0", "> 0", "< 0", "< 0", "prądnica (n < 0)"),
            ("< 0", "< 0", "< 0", "< 0", "> 0", "niemożliwe")]

    def update_plot(self):
        UN, IAN, RA, nN = REF["UN"], REF["IAN"], REF["RA"], REF["nN"]
        kN = kphi_rated(UN, IAN, RA, nN)
        Izw = UN / RA
        IA = self.p("x") * Izw
        n = (UN - RA * IA) / kN
        T = kN * IA / (2 * PI)
        Pel, Pm, Pcu = UN * IA, 2 * PI * n * T, RA * IA ** 2
        a1 = self.fig.add_subplot(2, 2, 1)
        I = np.linspace(-0.4 * Izw, 1.4 * Izw, 200)
        a1.plot(I, (UN - RA * I) / kN * 60, "tab:blue", lw=2)
        a1.axvspan(0, Izw, color="tab:green", alpha=0.12, label="silnik")
        a1.axvspan(I[0], 0, color="tab:orange", alpha=0.12, label="prądnica")
        a1.axvspan(Izw, I[-1], color="tab:red", alpha=0.12, label="hamulec")
        a1.plot([IA], [n * 60], "ko", ms=9)
        a1.axhline(0, color="k", lw=0.6); a1.axvline(0, color="k", lw=0.6)
        a1.set_xlabel("IA [A]"); a1.set_ylabel("n [obr/min]"); a1.grid(alpha=0.3); a1.legend(fontsize=8)
        a1.set_title("Punkt pracy na charakterystyce (U = UN)", fontsize=10)
        a2 = self.fig.add_subplot(2, 2, 2)
        vals = np.array([Pel, Pm, Pcu]) / 1e3
        cols = ["tab:blue" if x >= 0 else "tab:orange" for x in vals[:2]] + ["tab:red"]
        a2.bar(["P_el (pobór z sieci)", "P_mech (na wale)", "straty R·I²"], vals, color=cols)
        a2.axhline(0, color="k", lw=0.8)
        a2.set_ylabel("moc [kW]"); a2.grid(alpha=0.3, axis="y")
        a2.set_title("Moc > 0 = pobierana, < 0 = oddawana", fontsize=10)
        a2.tick_params(axis="x", labelsize=8)
        a3 = self.fig.add_subplot(2, 1, 2)
        a3.axis("off")
        sg = lambda v: "> 0" if v > 0 else "< 0"
        cur = (sg(IA), sg(n), sg(T), sg(Pel), sg(Pm))
        tb = a3.table(cellText=[list(r) for r in self.ROWS],
                      colLabels=["IA", "n", "T", "P_el", "P_mech", "tryb pracy"], loc="center", cellLoc="center")
        tb.auto_set_font_size(False); tb.set_fontsize(9); tb.scale(1, 1.3)
        for r, row in enumerate(self.ROWS, start=1):
            if row[:5] == cur:
                for c in range(6):
                    tb[r, c].set_facecolor("#ffe08a")
            elif row[5] == "niemożliwe":
                for c in range(6):
                    tb[r, c].set_facecolor("#eeeeee")
        a3.set_title("Tabela 2.1 - podświetlony aktualny tryb (U > 0, układ odbiornikowy)", fontsize=10)
        mode = [r[5] for r in self.ROWS if r[:5] == cur]
        return [f"IA = {IA:.1f} A", f"n  = {n * 60:.0f} obr/min", f"T  = {T:.1f} N·m", "",
                f"P_el   = {Pel / 1e3:.2f} kW", f"P_mech = {Pm / 1e3:.2f} kW", f"P_Cu   = {Pcu / 1e3:.2f} kW",
                f"Sprawdzenie: P_el - P_mech - P_Cu = {(Pel - Pm - Pcu):.2e}", "",
                f"TRYB: {mode[0] if mode else '-'}"]


class Tab9(ExampleTab):
    key = "T9"
    PARAMS = [("U", "U [V]", 100, 600, REF["UN"], 10),
              ("J", "J - moment bezwładności [kg·m²]", 0.05, 5.0, 0.8, 0.05),
              ("LA", "LA [mH]", 1, 100, 12, 1),
              ("TL", "T_obc / TN", 0.0, 1.5, 0.8, 0.05),
              ("tl", "t_obc - włączenie obciążenia [s]", 0.0, 5.0, 2.0, 0.1),
              ("m", "m - stopnie rozrusznika (0 = bezpośr.)", 0, 6, 3, 1),
              ("Imax", "Imax / IA,N", 1.2, 3.0, 2.0, 0.1),
              ("tend", "czas symulacji [s]", 1.0, 8.0, 3.5, 0.5)]

    def update_plot(self):
        U, J, LA, TLr, tl, m, Imr, tend = (self.p(k) for k in ("U", "J", "LA", "TL", "tl", "m", "Imax", "tend"))
        kN = kphi_rated(REF["UN"], REF["IAN"], REF["RA"], REF["nN"])
        TN = kN * REF["IAN"] / (2 * PI)
        S = dc_start_sim(U, REF["RA"], LA * 1e-3, kN, J, TLr * TN, tl, int(m), Imr * REF["IAN"], tend)
        t = S["t"]
        a1 = self.fig.add_subplot(3, 1, 1)
        a1.plot(t, S["n"], "tab:blue", lw=1.8)
        a1.axhline(U / kN * 60, color="0.5", ls=":", label="n0 = U/kΦ")
        for ts in S["sw"]:
            a1.axvline(ts, color="tab:orange", lw=0.8, ls="--")
        a1.set_ylabel("n [obr/min]"); a1.grid(alpha=0.3); a1.legend(fontsize=8, loc="lower right")
        a1.set_title("Rozruch silnika obcowzbudnego (RK4); pomarańczowe = przełączenia rozrusznika", fontsize=10)
        a2 = self.fig.add_subplot(3, 1, 2, sharex=a1)
        a2.plot(t, S["i"], "tab:red", lw=1.5)
        a2.axhline(Imr * REF["IAN"], color="k", ls=":", lw=1, label="Imax")
        if S["I_sw"] > 0:
            a2.axhline(S["I_sw"], color="tab:orange", ls=":", lw=1, label="I2 = Imax/λ")
        a2.axhline(REF["IAN"], color="tab:green", ls="--", lw=1, label="IA,N")
        a2.set_ylabel("iA [A]"); a2.grid(alpha=0.3); a2.legend(fontsize=8, loc="upper right")
        a3 = self.fig.add_subplot(3, 1, 3, sharex=a1)
        a3.plot(t, S["T"], "tab:purple", lw=1.5, label="T silnika")
        a3.plot(t, S["TL"], "tab:green", lw=1.5, label="T obciążenia")
        a3.set_xlabel("t [s]"); a3.set_ylabel("T [N·m]"); a3.grid(alpha=0.3); a3.legend(fontsize=8)
        n_end = S["n"][-1]
        k95 = np.argmax(S["n"] >= 0.95 * S["n"].max()) if S["n"].max() > 0 else 0
        out = [f"kΦN = {kN:.3f} V·s,  TN = {TN:.1f} N·m",
               f"τ_el = LA/RA = {LA / REF['RA']:.1f} ms",
               f"I szczyt   = {S['i'].max():.0f} A ({S['i'].max() / REF['IAN']:.1f}×IA,N)",
               f"I_zw bez rozr. = U/RA = {U / REF['RA']:.0f} A",
               f"n końcowe  = {n_end:.0f} obr/min",
               f"t(95% n)   = {t[k95]:.2f} s",
               f"straty w rozruszniku = {S['e_loss'] / 1e3:.1f} kJ"]
        if int(m) > 0:
            out.append("Opory [Ω]: " + ", ".join(f"{r:.2f}" for r in S["Rs"]))
            out.append("Przełączenia [s]: " + ", ".join(f"{x:.2f}" for x in S["sw"]))
        return out


class Tab10(ExampleTab):
    key = "T10"
    PARAMS = [("T", "temperatura magnesu [°C]", -40, 180, 20, 1),
              ("hM", "hM - grubość magnesu [mm]", 1, 20, 5, 0.5),
              ("d", "δ - szczelina [mm]", 0.2, 3.0, 0.6, 0.05),
              ("A", "A - okład prądowy [A/cm]", 0, 400, 100, 5),
              ("tau", "τp - podziałka biegunowa [mm]", 20, 200, 60, 5),
              ("ai", "αi", 0.5, 0.95, 0.7, 0.01),
              ("mur", "μr magnesu", 1.0, 1.2, 1.05, 0.01)]

    def extra_controls(self, parent):
        self.mat = self.add_radio(parent, "Materiał magnesu (Tab. 2.2)",
                                  [(k, k) for k in MAGNETS], "Ferryt")

    def update_plot(self):
        T, hM, d, A, tau, ai, mur = (self.p(k) for k in ("T", "hM", "d", "A", "tau", "ai", "mur"))
        name = self.mat.get()
        hM, d, tau, A = hM * 1e-3, d * 1e-3, tau * 1e-3, A * 100
        Theta = A * ai * tau / 2
        BR20, Hc20 = magnet_at(name, 20.0)
        BR, Hc = magnet_at(name, T)
        Hlim = -KNEE * Hc
        P0, Pon, Poff = magnet_points(BR, mur, hM, d, Theta)
        a1 = self.fig.add_subplot(2, 2, (1, 3))

        def curve(BRx, Hcx, style, lab):
            Hl = -KNEE * Hcx
            H = np.linspace(Hl, 0, 50)
            Bk = BRx + MU0 * mur * Hl
            a1.plot(np.r_[H, 0] / 1e3, np.r_[BRx + MU0 * mur * H, BRx], style, color="tab:blue", label=lab)
            a1.plot(np.array([Hl, -Hcx, -Hcx * 1.03]) / 1e3, [Bk, Bk * 0.35, -0.05], style, color="tab:blue")
        curve(BR20, Hc20, ":", "20 °C")
        curve(BR, Hc, "-", f"{T:.0f} °C")
        Hmin = min(-Hc * 1.05, P0[0] * 1.5, Poff[0] * 1.2)
        HH = np.linspace(Hmin, 0, 50)
        slope = -MU0 * hM / d
        a1.plot(HH / 1e3, slope * HH, color="0.4", lw=1, label="prosta obciążenia (jałowo)")
        for sgn in (1, -1):
            a1.plot(HH / 1e3, slope * (HH - sgn * Theta / hM), color="0.7", lw=0.8, ls="--")
        a1.plot(P0[0] / 1e3, P0[1], "ko", ms=8, label="bieg jałowy")
        a1.plot(Pon[0] / 1e3, Pon[1], "g^", ms=9, label="krawędź nabiegająca")
        danger = Poff[0] < Hlim
        a1.plot(Poff[0] / 1e3, Poff[1], "v", color="red" if danger else "tab:orange", ms=10,
                label="krawędź zbiegająca")
        a1.axvline(Hlim / 1e3, color="red", ls="-.", lw=1, label="H_limit (kolano)")
        a1.set_xlim(Hmin / 1e3, 0.05 * abs(Hmin) / 1e3)
        a1.set_ylim(-0.1, max(BR20, BR) * 1.15)
        a1.set_xlabel("HM [kA/m]"); a1.set_ylabel("BM [T]"); a1.grid(alpha=0.3); a1.legend(fontsize=8, loc="upper left")
        a1.set_title(f"{name}: charakterystyka w II ćwiartce i punkty pracy" + ("  - ROZMAGNESOWANIE!" if danger else ""),
                     fontsize=10, color="red" if danger else "black")
        a2 = self.fig.add_subplot(2, 2, 2)
        TT = np.linspace(-40, 180, 111)
        Hoff = []
        Hl_ = []
        for tt in TT:
            br, hc = magnet_at(name, tt)
            Hoff.append(magnet_points(br, mur, hM, d, Theta)[2][0] / 1e3)
            Hl_.append(-KNEE * hc / 1e3)
        a2.plot(TT, Hoff, color="tab:orange", label="HM krawędź zbiegająca")
        a2.plot(TT, Hl_, color="red", ls="-.", label="H_limit")
        a2.fill_between(TT, Hl_, np.minimum(Hl_, Hoff), color="red", alpha=0.2)
        a2.axvline(T, color="k", ls=":")
        a2.set_xlabel("T [°C]"); a2.set_ylabel("H [kA/m]"); a2.grid(alpha=0.3); a2.legend(fontsize=8)
        a2.set_title("Margines rozmagnesowania vs temperatura (rys. 2.26)", fontsize=10)
        a3 = self.fig.add_subplot(2, 2, 4)
        hist = {"stale": ([1910, 1920, 1935], [2, 6, 8]),
                "AlNiCo": ([1935, 1945, 1960, 1970], [12, 40, 60, 80]),
                "ferryt": ([1950, 1960, 1980, 2000], [8, 20, 32, 40]),
                "SmCo": ([1968, 1975, 1985, 2000], [80, 160, 240, 260]),
                "NdFeB wiązany": ([1990, 2000, 2010], [60, 90, 120]),
                "NdFeB": ([1984, 1990, 2000, 2010], [250, 320, 400, 450])}
        for k, (x, y_) in hist.items():
            a3.plot(x, y_, "o-", ms=3, label=k)
        a3.set_xlabel("rok"); a3.set_ylabel("(BH)max [kJ/m³]"); a3.grid(alpha=0.3); a3.legend(fontsize=7, ncol=2)
        a3.set_title("Rozwój materiałów magnetycznych (rys. 2.24, orientacyjnie)", fontsize=10)
        return [f"BR({T:.0f}°C)  = {BR:.3f} T", f"HcJ({T:.0f}°C) = {Hc / 1e3:.0f} kA/m",
                f"H_limit       = {Hlim / 1e3:.0f} kA/m",
                f"Θ = A·αi·τp/2 = {Theta:.0f} A", "",
                f"Jałowo:   BM = {P0[1]:.3f} T, HM = {P0[0] / 1e3:.0f} kA/m",
                f"Nabieg.:  BM = {Pon[1]:.3f} T, HM = {Pon[0] / 1e3:.0f} kA/m",
                f"Zbieg.:   BM = {Poff[1]:.3f} T, HM = {Poff[0] / 1e3:.0f} kA/m",
                f"Margines: {(Poff[0] - Hlim) / 1e3:.0f} kA/m",
                "", "!! NIEODWRACALNE ROZMAGNESOWANIE" if danger else "OK - magnes bezpieczny"]


class Tab11(ExampleTab):
    key = "T11"
    PARAMS = [("nr", "n / nN", 0.3, 1.3, 1.0, 0.01),
              ("RF", "RF + RF,S - obwód wzbudzenia [Ω]", 20, 400, 150, 1),
              ("Ur", "Ur - napięcie szczątkowe [V]", 0.5, 30, 8, 0.5),
              ("RA", "RA [Ω]", 0.1, 3.0, 0.5, 0.05),
              ("Rl", "R_obc - obciążenie [Ω]", 1, 200, 10, 0.5),
              ("LF", "LF - indukcyjność wzbudzenia [H]", 1, 40, 10, 0.5)]

    def update_plot(self):
        nr, RF, Ur, RA, Rl, LF = (self.p(k) for k in ("nr", "RF", "Ur", "RA", "Rl", "LF"))
        Rtot = RF + RA
        IFn = shunt_noload_point(nr, Ur, Rtot)
        Rcrit = nr * 250.0 / 1.0 - RA
        IF = np.linspace(0, 3.0, 400)
        a1 = self.fig.add_subplot(2, 2, 1)
        a1.plot(IF, shunt_magnetization(IF, nr, Ur), "tab:blue", lw=2, label="Ui(IF) - bieg jałowy")
        a1.plot(IF, Rtot * IF, "tab:red", label="(RA+RF)·IF")
        a1.plot(IF, (Rcrit + RA) * IF, "k:", label="R krytyczna")
        a1.plot([IFn], [Rtot * IFn], "ko", ms=8)
        a1.set_ylim(0, nr * 270 + Ur + 10); a1.set_xlim(0, 3)
        a1.set_xlabel("IF [A]"); a1.set_ylabel("U [V]"); a1.grid(alpha=0.3); a1.legend(fontsize=8)
        a1.set_title("Samowzbudzenie (rys. 2.29)", fontsize=10)
        dt, tend = 2e-3, 8.0
        tt = np.arange(0, tend, dt)
        x = 0.0
        IFt = np.empty_like(tt)
        f = lambda z: (shunt_magnetization(z, nr, Ur) - Rtot * z) / LF
        for k in range(len(tt)):
            IFt[k] = x
            k1 = f(x); k2 = f(x + dt / 2 * k1); k3 = f(x + dt / 2 * k2); k4 = f(x + dt * k3)
            x += dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        a2 = self.fig.add_subplot(2, 2, 2)
        a2.plot(tt, RF * IFt, "tab:green", lw=2)
        a2.set_xlabel("t [s]"); a2.set_ylabel("U [V]"); a2.grid(alpha=0.3)
        a2.set_title("Narastanie napięcia w czasie (lawina od Ur)", fontsize=10)
        IFg = np.linspace(IFn, 0, 800)
        Ug = RF * IFg
        IAg = (shunt_magnetization(IFg, nr, Ur) - Ug) / RA
        Ig = IAg - IFg
        g = Ug - Rl * Ig
        idx = np.where(np.diff(np.sign(g)) != 0)[0]
        a3 = self.fig.add_subplot(2, 1, 2)
        a3.plot(Ig, Ug, "tab:blue", lw=2, label="U(I) - charakterystyka zewnętrzna")
        II = np.linspace(0, max(Ig.max(), 1) * 1.1, 50)
        a3.plot(II, Rl * II, "tab:red", ls="--", label=f"odbiornik R = {Rl:.1f} Ω")
        kmax = int(np.argmax(Ig))
        a3.plot([Ig[kmax]], [Ug[kmax]], "k^", ms=8, label="I_max")
        a3.plot([Ig[-1]], [0], "ks", ms=7, label="I_zw,R (od Ur)")
        if len(idx):
            k = idx[0]
            a3.plot([Ig[k]], [Ug[k]], "o", color="tab:orange", ms=10, label="punkt pracy")
            Iop, Uop = Ig[k], Ug[k]
            collapsed = k > kmax
        else:
            Iop = Uop = 0.0
            collapsed = True
        a3.set_ylim(0, max(Ug.max() * 1.1, 10)); a3.set_xlim(0, II[-1])
        a3.set_xlabel("I obciążenia [A]"); a3.set_ylabel("U [V]"); a3.grid(alpha=0.3); a3.legend(fontsize=8)
        a3.set_title("Obciążanie prądnicy bocznikowej (rys. 2.31)", fontsize=10)
        ok = Rtot < Rcrit + RA
        return [f"R obwodu wzbudz. = {Rtot:.0f} Ω", f"R krytyczna      = {Rcrit + RA:.0f} Ω",
                "SAMOWZBUDZENIE: " + ("TAK" if ok else "NIE (za duże R)"),
                f"IF (jałowo) = {IFn:.3f} A", f"U0 = {RF * IFn:.1f} V", "",
                f"I_max    = {Ig[kmax]:.1f} A (przy U = {Ug[kmax]:.0f} V)",
                f"I_zw,R   = Ur/RA = {nr * Ur / RA:.1f} A",
                f"Punkt pracy: I = {Iop:.1f} A, U = {Uop:.1f} V",
                "Napięcie ZAŁAMANE" if collapsed else "Praca stabilna"]


class Tab12(ExampleTab):
    key = "T12"
    PARAMS = [("uU", "1) U/UN", 0.1, 1.0, 1.0, 0.01),
              ("f", "2) f - osłabienie pola", 1.0, 3.0, 1.0, 0.05),
              ("RS", "3) RA,S - opór szeregowy [Ω]", 0.0, 5.0, 0.0, 0.05),
              ("R", "RA + RF [Ω]", 0.2, 2.0, 0.8, 0.05),
              ("Lm", "L'm [H]", 0.02, 0.1, 0.0527, 0.0001)]

    def update_plot(self):
        uU, f, RS, R, Lm = (self.p(k) for k in ("uU", "f", "RS", "R", "Lm"))
        UN, IAN, nN = REF["UN"], REF["IAN"], REF["nN"]

        def curves(U, f_, Rt):
            Izw = U / Rt
            I = np.linspace(0.03 * IAN, Izw, 400)
            n = f_ * (U - Rt * I) / (2 * PI * Lm * I) * 60
            T = Lm * I ** 2 / f_
            return I, n, T, Izw
        Ib, nb, Tb, Izb = curves(UN, 1.0, R)
        Im, nm, Tm, Izm = curves(uU * UN, f, R + RS)
        a1 = self.fig.add_subplot(1, 2, 1)
        a1.plot(Ib, nb, color="0.6", ls="--", label="n - podstawowa")
        a1.plot(Im, nm, color="tab:blue", lw=2, label="n - po zmianie")
        a1.axvline(IAN, color="tab:red", ls=":", lw=1, label="IA,N")
        a1.set_ylim(0, 4 * nN); a1.set_xlim(0, 4 * IAN)
        a1.set_xlabel("IA [A]"); a1.set_ylabel("n [obr/min]", color="tab:blue"); a1.grid(alpha=0.3)
        a1b = a1.twinx()
        a1b.plot(Ib, Tb, color="0.6", ls=":")
        a1b.plot(Im, Tm, color="tab:purple", lw=2, label="T - po zmianie")
        a1b.set_ylim(0, Lm * (4 * IAN) ** 2); a1b.set_ylabel("T [N·m]", color="tab:purple")
        h1, l1 = a1.get_legend_handles_labels(); h2, l2 = a1b.get_legend_handles_labels()
        a1.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper center")
        a1.set_title("Silnik szeregowy: n(IA), T(IA) (rys. 2.33 - 2.36)", fontsize=10)
        a2 = self.fig.add_subplot(1, 2, 2)
        a2.plot(nb, Tb, color="0.6", ls="--", label="podstawowa")
        a2.plot(nm, Tm, color="tab:blue", lw=2, label="po zmianie")
        a2.set_xlim(0, 4 * nN); a2.set_ylim(0, Lm * (4 * IAN) ** 2)
        a2.set_xlabel("n [obr/min]"); a2.set_ylabel("T [N·m]"); a2.grid(alpha=0.3); a2.legend(fontsize=8)
        a2.set_title("T(n): hiperbola - duży moment przy małej prędkości", fontsize=10)
        U = uU * UN
        nI = f * (U - (R + RS) * IAN) / (2 * PI * Lm * IAN) * 60
        n10 = f * (U - (R + RS) * 0.1 * IAN) / (2 * PI * Lm * 0.1 * IAN) * 60
        return [f"Przy IA = IA,N = {IAN:.0f} A:",
                f"  n = {nI:.0f} obr/min",
                f"  T = {Lm * IAN ** 2 / f:.1f} N·m",
                f"Przy IA = 0,1·IA,N:",
                f"  n = {n10:.0f} obr/min (!)",
                "IA -> 0: n -> ∞ (rozbieganie)", "",
                f"I_zw = U/(R+RS) = {Izm:.0f} A",
                f"T_zw = {Lm * Izm ** 2 / f / 1e3:.1f} kN·m",
                "(bez nasycenia - zawyżone)"]


class Tab13(ExampleTab):
    key = "T13"
    PARAMS = [("s", "s - udział wzbudzenia szeregowego", 0.0, 1.0, 0.4, 0.01),
              ("r", "R [j.w.] (spadek napięcia)", 0.02, 0.2, 0.06, 0.005)]

    def update_plot(self):
        s, r = self.p("s"), self.p("r")
        kN = 1 - r
        a1 = self.fig.add_subplot(1, 2, 1)
        a2 = self.fig.add_subplot(1, 2, 2)
        res = []
        for ss, lab, col in ((0.0, "bocznikowa (s = 0)", "tab:green"), (1.0, "szeregowa (s = 1)", "tab:red"),
                             (s, f"kompaundowa (s = {s:.2f})", "tab:purple")):
            I = np.linspace(0.01, 1 / r, 3000)
            kp = kN * ((1 - ss) + ss * I)
            n = (1 - r * I) / kp
            T = kp * I / kN
            a1.plot(n, T, color=col, lw=2.5 if ss == s else 1.5, label=lab)
            a2.plot(I, n, color=col, lw=2.5 if ss == s else 1.5, label=lab)
            n0 = 1 / (kN * (1 - ss)) if ss < 1 else float("inf")
            T05 = np.interp(0.5, n[::-1], T[::-1])
            res.append((lab, n0, T05))
        a1.plot([1], [1], "ko")
        a1.set_xlim(0, 3); a1.set_ylim(0, 4)
        a1.set_xlabel("n / nN"); a1.set_ylabel("T / TN"); a1.grid(alpha=0.3); a1.legend(fontsize=8)
        a1.set_title("Moment vs prędkość (rys. 2.38)", fontsize=10)
        a2.plot([1], [1], "ko")
        a2.set_xlim(0, 4); a2.set_ylim(0, 3)
        a2.set_xlabel("IA / IA,N"); a2.set_ylabel("n / nN"); a2.grid(alpha=0.3); a2.legend(fontsize=8)
        a2.set_title("Prędkość vs prąd", fontsize=10)
        out = []
        for lab, n0, T05 in res:
            out += [lab, f"  n0 = {'∞ (rozbiega się!)' if n0 == float('inf') else f'{n0:.2f}·nN'}",
                    f"  T przy n = 0,5·nN: {T05:.2f}·TN"]
        return out


class Tab14(ExampleTab):
    key = "T14"
    PARAMS = [("al", "α - kąt zapłonu [°]", 0, 180, 30, 1),
              ("ULL", "U_LL - napięcie sieci [V]", 100, 690, 400, 5),
              ("IA", "Id = IA [A]", 0, 100, 50, 1),
              ("phiG", "Leonard: ΦG/ΦG,N", -1.0, 1.0, 0.8, 0.01)]

    def extra_controls(self, parent):
        self.nb = self.add_radio(parent, "Układ przekształtnika",
                                 [("1", "1 mostek (2 ćwiartki)"), ("2", "2 mostki przeciwsobne (4 ćw.)")], "1")

    def update_plot(self):
        al, ULL, IA, phiG = (self.p(k) for k in ("al", "ULL", "IA", "phiG"))
        th = np.linspace(0, 720, 2881)
        ud, ph = bridge_ud(th, al, ULL)
        Ud = 3 * math.sqrt(2) / PI * ULL * math.cos(math.radians(al))
        a1 = self.fig.add_subplot(3, 1, 1)
        for a_, b_ in ((0, 1), (1, 2), (2, 0)):
            a1.plot(th, ph[a_] - ph[b_], color="0.8", lw=0.8)
            a1.plot(th, ph[b_] - ph[a_], color="0.8", lw=0.8)
        a1.plot(th, ud, "tab:blue", lw=1.8, label="ud(t)")
        a1.axhline(Ud, color="tab:red", ls="--", label=f"Ud = 1,35·U_LL·cos α = {Ud:.0f} V")
        a1.axhline(0, color="k", lw=0.6)
        a1.set_xlim(0, 720); a1.set_ylabel("u [V]"); a1.grid(alpha=0.3); a1.legend(fontsize=8, loc="lower right")
        a1.set_title("Napięcia międzyfazowe (szare) i napięcie wyjściowe mostka (rys. 2.41)", fontsize=10)
        a2 = self.fig.add_subplot(3, 1, 2, sharex=a1)
        for k in range(6):
            st = (30 + al + k * 60) % 360
            segs = []
            for base in (-360, 0, 360, 720):
                x0 = st + base
                lo, hi = max(x0, 0), min(x0 + 120, 720)
                if hi > lo:
                    segs.append((lo, hi - lo))
            a2.broken_barh(segs, (5 - k - 0.35, 0.7), color=f"C{k}")
        a2.set_yticks(range(6)); a2.set_yticklabels([f"S{6 - k}" for k in range(6)])
        a2.set_xlabel("ωt [°]"); a2.grid(alpha=0.3, axis="x")
        a2.set_title("Przewodzenie tyrystorów (każdy 120°, kolejność S1...S6 co 60°)", fontsize=10)
        a3 = self.fig.add_subplot(3, 2, 5)
        Ud0 = 3 * math.sqrt(2) / PI * ULL
        a3.add_patch(Rectangle((0, -Ud0), 1, 2 * Ud0, color="tab:blue", alpha=0.2))
        if self.nb.get() == "2":
            a3.add_patch(Rectangle((-1, -Ud0), 1, 2 * Ud0, color="tab:orange", alpha=0.2))
        a3.plot([IA / 100], [Ud], "ko")
        a3.axhline(0, color="k", lw=0.6); a3.axvline(0, color="k", lw=0.6)
        a3.set_xlim(-1.1, 1.1); a3.set_ylim(-1.1 * Ud0, 1.1 * Ud0)
        a3.set_xlabel("Id [×100 A]"); a3.set_ylabel("Ud [V]")
        a3.set_title("Dostępne ćwiartki pracy", fontsize=9)
        a4 = self.fig.add_subplot(3, 2, 6)
        AL = np.linspace(0, 180, 181)
        a4.plot(AL, 3 * math.sqrt(2) / PI * ULL * np.cos(np.radians(AL)), "tab:red")
        a4.plot([al], [Ud], "ko"); a4.axhline(0, color="k", lw=0.6)
        a4.set_xlabel("α [°]"); a4.set_ylabel("Ud [V]"); a4.grid(alpha=0.3)
        a4.set_title("Ud(α): prostownik α<90°, falownik α>90°", fontsize=9)
        kN = kphi_rated(REF["UN"], REF["IAN"], REF["RA"], REF["nN"])
        n = (Ud - REF["RA"] * IA) / kN * 60
        UG = REF["UN"] * phiG
        nL = (UG - REF["RA"] * IA) / kN * 60
        return [f"Ud0 = 1,35·U_LL = {Ud0:.0f} V", f"Ud(α={al:.0f}°) = {Ud:.0f} V",
                "PROSTOWNIK" if Ud >= 0 else "FALOWNIK (zwrot energii)", "",
                "Silnik wzorcowy (kΦN = %.2f V·s):" % kN,
                f"  n = (Ud - RA·IA)/kΦ = {n:.0f} obr/min", "",
                "Układ Leonarda:",
                f"  UG = UN·ΦG/ΦG,N = {UG:.0f} V",
                f"  n silnika = {nL:.0f} obr/min",
                "  (ΦG < 0 -> obroty wstecz)"]


class Tab15(ExampleTab):
    key = "T15"
    PARAMS = [("IA", "IA / IA,N (<0 = prądnica)", -1.5, 1.5, 1.0, 0.05),
              ("ratio", "ΘA/ΘF przy IA,N", 0.0, 1.2, 0.5, 0.05),
              ("f", "f - osłabienie pola", 1.0, 3.0, 1.0, 0.05),
              ("ai", "αi", 0.5, 0.9, 0.7, 0.01)]

    def extra_controls(self, parent):
        box = ttk.LabelFrame(parent, text="Opcje")
        box.pack(fill=tk.X, padx=6, pady=4)
        self.sat = self.add_check(box, "nasycenie żelaza", True)
        self.comp = self.add_check(box, "uzwojenie kompensacyjne", False)
        self.cp = self.add_check(box, "bieguny komutacyjne", False)

    def update_plot(self):
        IA, ratio, f, ai = (self.p(k) for k in ("IA", "ratio", "f", "ai"))
        al = np.linspace(0, 2 * PI, 2001)
        thF, thA, thC, thCP, B = armature_field(al, ratio, IA, f, ai, self.comp.get(), self.cp.get(), self.sat.get())
        _, _, _, _, B0 = armature_field(al, ratio, 0.0, f, ai, False, False, self.sat.get())
        deg = np.degrees(al)
        a1 = self.fig.add_subplot(2, 1, 1)
        a1.plot(deg, thF * (np.abs(b_shape(al - PI / 2, ai)) > 0.5), "tab:red", label="ΘF (wzbudzenie)")
        a1.plot(deg, thA, "tab:blue", label="ΘA (twornik)")
        if self.comp.get():
            a1.plot(deg, thC, "tab:green", label="ΘC (kompensacja)")
        if self.cp.get():
            a1.plot(deg, thCP, "tab:orange", label="Θ bieguna komutacyjnego")
        a1.axhline(0, color="k", lw=0.6)
        for x in (0, 180, 360):
            a1.axvline(x, color="0.6", ls=":")
        a1.set_xlim(0, 360); a1.set_ylabel("Θ [j.w.]"); a1.grid(alpha=0.3); a1.legend(fontsize=8, ncol=4)
        a1.set_title("Przepływy (rys. 2.43/2.45); kropkowane = geometryczne strefy neutralne", fontsize=10)
        a2 = self.fig.add_subplot(2, 1, 2, sharex=a1)
        a2.plot(deg, B0, color="0.5", ls="--", label="bieg jałowy")
        a2.plot(deg, B, "tab:purple", lw=2, label="obciążenie")
        a2.fill_between(deg, B0, B, color="tab:purple", alpha=0.15)
        a2.axhline(0, color="k", lw=0.6)
        a2.set_xlabel("α [° el.]"); a2.set_ylabel("Bδ / Bδ,N"); a2.grid(alpha=0.3); a2.legend(fontsize=8)
        a2.set_title("Wypadkowa indukcja w szczelinie (rys. 2.50)", fontsize=10)
        m = (al > 0) & (al < PI) & (np.abs(b_shape(al - PI / 2, ai)) > 0.02)
        phi_r = _integ(B[m], al[m]) / _integ(B0[m], al[m]) * 100
        cg = _integ(al[m] * B[m], al[m]) / _integ(B[m], al[m])
        shift = math.degrees(cg - PI / 2)
        peak = B[m].max() / (B0[m].max() + 1e-9)
        core = (al > 0) & (al < PI) & (np.abs(b_shape(al - PI / 2, ai)) > 0.5)   # pod nabiegunnikiem
        neg = B[core].min() < -1e-3
        return [f"Strumień względny = {phi_r:.1f} %",
                f"-> moment spada o {max(0, 100 - phi_r):.1f} %",
                f"Przesunięcie osi pola = {shift:+.1f}° el.",
                f"B_max / B_max,0 = {peak:.2f}",
                "(lokalny wzrost napięcia między działkami)",
                "", "! Pole pod krawędzią ZMIENIA ZNAK" if neg else "Pole pod biegunem bez zmiany znaku",
                "", "Silnik: oś przesuwa się przeciw obrotom,", "prądnica: zgodnie z obrotami."]


class Tab16(ExampleTab):
    key = "T16"
    PARAMS = [("I", "I cewki = IA/2a [A]", 5, 200, 50, 1),
              ("n", "n [obr/min]", 100, 3000, 1500, 10),
              ("L", "L cewki [µH]", 0.0, 100, 25, 0.5),
              ("Rb", "Rb - rezyst. styku szczotki [mΩ]", 2, 100, 20, 1),
              ("bB", "bB - szerokość szczotki [mm]", 5, 30, 10, 0.5),
              ("DC", "DC - średnica komutatora [mm]", 50, 400, 150, 5),
              ("kcp", "k_cp - stopień kompensacji", 0.0, 2.0, 0.0, 0.05)]

    def update_plot(self):
        I, n, L, Rb, bB, DC, kcp = (self.p(k) for k in ("I", "n", "L", "Rb", "bB", "DC", "kcp"))
        L, Rb, bB, DC = L * 1e-6, Rb * 1e-3, bB * 1e-3, DC * 1e-3
        vC = PI * DC * n / 60
        Tc = bB / vC
        t, i, j1, j2 = commutation(I, L, Rb, Tc, kcp)
        _, i0, _, _ = commutation(I, L, Rb, Tc, 0.0)
        a1 = self.fig.add_subplot(2, 2, (1, 2))
        a1.plot(t / Tc, np.linspace(1, -1, len(t)), color="0.6", lw=3, alpha=0.6, label="idealna (liniowa)")
        if kcp != 0:
            a1.plot(t / Tc, i0 / I, "tab:red", ls=":", label="bez biegunów komutacyjnych")
        a1.plot(t / Tc, i / I, "tab:blue", lw=2, label=f"rzeczywista, k_cp = {kcp:.2f}")
        a1.axhline(0, color="k", lw=0.6)
        a1.set_xlabel("t / Tc"); a1.set_ylabel("i_cewki / (IA/2a)"); a1.grid(alpha=0.3); a1.legend(fontsize=8)
        a1.set_title("Prąd komutowanej cewki (rys. 2.46, 2.47, 2.49)", fontsize=10)
        a2 = self.fig.add_subplot(2, 2, 3)
        a2.plot(t / Tc, j1, "tab:orange", lw=1.8, label="krawędź zbiegająca")
        a2.plot(t / Tc, j2, "tab:cyan", lw=1.4, label="krawędź nabiegająca")
        a2.axhline(1, color="0.5", ls="--")
        a2.set_ylim(0, max(4, min(max(j1.max(), j2.max()) * 1.1, 30)))
        a2.set_xlabel("t / Tc"); a2.set_ylabel("J / J średnie"); a2.grid(alpha=0.3); a2.legend(fontsize=8)
        a2.set_title("Gęstość prądu pod krawędziami szczotki", fontsize=10)
        a3 = self.fig.add_subplot(2, 2, 4)
        IAx = np.linspace(0, 200, 50)
        a3.plot(IAx, 2 * L * IAx / Tc, "tab:red", label="e_r = 2LI/Tc ~ I·n")
        a3.plot(IAx, kcp * 2 * L * IAx / Tc, "tab:green", ls="--", label="e_cp (bieguny kom.)")
        a3.plot([I], [2 * L * I / Tc], "ko")
        a3.set_xlabel("I cewki [A]"); a3.set_ylabel("napięcie [V]"); a3.grid(alpha=0.3); a3.legend(fontsize=8)
        a3.set_title(f"Napięcie reaktancyjne przy n = {n:.0f} obr/min", fontsize=10)
        nt = len(j1)
        spark = max(j1[int(0.9 * nt):].max(), j2[:int(0.1 * nt)].max())
        grade = "brak iskrzenia" if spark < 1.5 else ("lekkie iskrzenie" if spark < 3 else "SILNE ISKRZENIE")
        return [f"v_C = π·DC·n = {vC:.1f} m/s",
                f"Tc = bB/v_C  = {Tc * 1e3:.3f} ms",
                f"e_r = 2LI/Tc = {2 * L * I / Tc:.2f} V",
                f"e_cp         = {kcp * 2 * L * I / Tc:.2f} V",
                f"Rb·I         = {Rb * I:.2f} V",
                "",
                f"J_max/J_śr (krawędzie) = {spark:.1f}",
                f"Ocena: {grade}",
                "",
                "k_cp < 1: komutacja opóźniona",
                "k_cp = 1: liniowa (idealna)",
                "k_cp > 1: przyspieszona"]


# =============================================================================
# APLIKACJA
# =============================================================================

TABS = [("0 Budowa", Tab0), ("1 Napięcie", Tab1), ("2 Komutacja", Tab2), ("3 Uzwojenia", Tab3),
        ("4 Równania", Tab4), ("5 Esson", Tab5), ("6 Żłobki", Tab6), ("7 Obcowzbudna", Tab7),
        ("8 Tryby pracy", Tab8), ("9 Rozruch", Tab9), ("10 Magnesy", Tab10), ("11 Bocznikowa", Tab11),
        ("12 Szeregowa", Tab12), ("13 Porównanie", Tab13), ("14 Zasilanie", Tab14),
        ("15 Odd. twornika", Tab15), ("16 Bieguny kom.", Tab16)]


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Maszyny prądu stałego - interaktywny podręcznik (Gerling, rozdz. 2)")
        self.geometry("1450x920")
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
