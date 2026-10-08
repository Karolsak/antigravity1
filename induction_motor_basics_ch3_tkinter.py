"""
Silnik indukcyjny - podstawy (rozdział 3: "AC motor control and electrical vehicle")
=====================================================================================
Interaktywna aplikacja Tkinter + Matplotlib dla WSZYSTKICH przykładów rozdziału 3:
ćwiczenia (Exercise) 3.1-3.13, zagadnienia z tekstu (stabilność, momenty pasożytnicze,
klatka podwójna/NEMA, rozruch, sterowanie napięciem, VVVF) oraz zadania (Problems) 3.1-3.13.

Każda zakładka = jeden przykład: suwaki (parametry), wykresy (wyniki, z zoomem),
wyniki liczbowe krok po kroku oraz objaśnienie napisane językiem ucznia liceum.

Zakładki:
  E3.1  Bilans mocy, diagram Sankeya, tabela strat 3.1
  E3.2  Charakterystyka moment-poślizg: schemat T vs schemat zmodyfikowany (+ prąd)
  E3.3  Wpływ napięcia na poślizg (też zadanie 3.7)
  E3.4  Rezystancja wirnika: moment, prąd, sprawność (też ćw. 3.7)
  E3.5  Wyznaczanie rr z danych znamionowych
  E3.6  Punkt znamionowy: prąd, cos φ, moment, moment krytyczny, wykres wskazowy
  STAB  Obszar stabilny i niestabilny (symulacja skoku obciążenia, utyk silnika)
  PAR   Momenty pasożytnicze 5. i 7. harmonicznej ("pełzanie" silnika)
  E3.8  Indukcyjność rozproszenia żłobka i połączeń czołowych
  KOŁO  Wykres kołowy Heylanda (ćw. 3.9-3.12)
  E3.13 Wypieranie prądu (efekt naskórkowy) w pręcie wirnika
  NEMA  Klatka podwójna i klasy NEMA A/B/C/D
  START Rozruch bezpośredni vs softstart - symulacja dynamiczna modelu αβ
  TYR   Sterownik tyrystorowy (softstart) - kąt załączenia α (też zadanie 3.6)
  NAP   Sterowanie zmianą napięcia (wentylator)
  VVVF  Sterowanie U/f = const, podbicie napięcia, regulator z poślizgiem
  ZAD   Zadania 3.1-3.13 (wybór z listy)

Uruchomienie:  python induction_motor_basics_ch3_tkinter.py
Wymagania:     numpy, matplotlib (tkinter jest w standardowym Pythonie)
"""
import math
import cmath
import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

PI = math.pi
SQ2 = math.sqrt(2.0)
SQ3 = math.sqrt(3.0)
MU0 = 4e-7 * PI          # przenikalność magnetyczna próżni [H/m]
HP = 746.0               # 1 KM (horse power) [W]


# =============================================================================
# RDZEŃ OBLICZENIOWY
# =============================================================================

_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")


def rpm2rad(n):
    return np.asarray(n, dtype=float) * 2 * PI / 60.0


def n_sync(f, P):
    """Prędkość synchroniczna [obr/min] dla P biegunów."""
    return 120.0 * f / P


def _s(s):
    """Unikamy dzielenia przez zero dla s = 0."""
    s = np.asarray(s, dtype=float)
    return np.where(np.abs(s) < 1e-9, 1e-9, s)


def im_T(s, Vph, f, rs, Lls, rr, Llr, Lm, P, zr=None):
    """Pełny schemat zastępczy T (rys. 3.7, bez rc). Vph - napięcie fazowe rms.
    zr - opcjonalna impedancja gałęzi wirnika (zawiera już rr/s), np. klatka podwójna.
    Moment = moc szczeliny / prędkość synchroniczna mechaniczna (3.5)."""
    s = _s(s)
    we = 2 * PI * f
    Zr = rr / s + 1j * we * Llr if zr is None else zr
    Zm = 1j * we * Lm
    Zin = rs + 1j * we * Lls + Zm * Zr / (Zm + Zr)
    Is = Vph / Zin
    Ir = Is * Zm / (Zm + Zr)
    Pag = 3 * np.abs(Ir) ** 2 * np.real(Zr)
    Te = Pag * (P / 2) / we
    return dict(s=s, Is=Is, Ir=Ir, Te=Te, Zin=Zin, Pag=Pag, pf=np.cos(np.angle(Zin)))


def im_mod(s, Vph, f, rs, Lls, rr, Llr, Lm, P):
    """Zmodyfikowany schemat zastępczy (rys. 3.9): Lm przesunięta na zaciski.
    Ir' = Vs / (rs + rr/s + jωe(Lls+Llr))  (3.6)."""
    s = _s(s)
    we = 2 * PI * f
    Ir = Vph / (rs + rr / s + 1j * we * (Lls + Llr))
    Im = Vph / (1j * we * Lm) * np.ones_like(s)
    Is = Ir + Im
    Te = 3 * (P / 2) * np.abs(Ir) ** 2 * rr / (s * we)
    return dict(s=s, Is=Is, Ir=Ir, Im=Im, Te=Te, pf=np.cos(np.angle(Is)))


def te_37(s, Vph, f, rs, Lls, rr, Llr, P):
    """Wzór (3.7): Te = 3P/(2ωe) * Vs² (rr/s) / [(rs + rr/s)² + ωe²(Lls+Llr)²]."""
    s = _s(s)
    we = 2 * PI * f
    return 3 * P / (2 * we) * Vph ** 2 * rr / s / ((rs + rr / s) ** 2 + (we * (Lls + Llr)) ** 2)


def breakdown(Vph, f, rs, Lls, rr, Llr, P):
    """(3.10) sm = rr/sqrt(rs² + X²),  (3.11) Tmax = 3P/(4ωe) Vs²/(rs + sqrt(rs² + X²))."""
    we = 2 * PI * f
    X = we * (Lls + Llr)
    root = math.sqrt(rs ** 2 + X ** 2)
    return rr / root, 3 * P / (4 * we) * Vph ** 2 / (rs + root)


def solve_slip(Tt, func, s_hi, s_lo=1e-7):
    """Poślizg w przedziale (s_lo, s_hi), w którym func(s) rośnie, dający moment Tt
    (bisekcja). Zwraca None, gdy Tt jest większe niż func(s_hi)."""
    lo, hi = s_lo, s_hi
    if float(func(hi)) < Tt:
        return None
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if float(func(mid)) < Tt:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def sigma_exact(Lls, Llr, Lm):
    """Współczynnik rozproszenia σ = 1 - Lm²/(Ls Lr)  (3.18)."""
    return 1 - Lm ** 2 / ((Lls + Lm) * (Llr + Lm))


def sigma_book(Lls, Llr, Lm):
    """Przybliżenie (3.26) użyte w ćw. 3.10:  σ ≈ Ls Lr / Lm² - 1 ≈ Lls/Lm + Llr/Lm."""
    return (Lls + Lm) * (Llr + Lm) / Lm ** 2 - 1


def skin_KrKx(xi):
    """Wypieranie prądu w pręcie prostokątnym:
    Kr = ξ (sh2ξ + sin2ξ)/(ch2ξ - cos2ξ)        (3.36) - wzrost rezystancji
    Kx = 3/(2ξ) (sh2ξ - sin2ξ)/(ch2ξ - cos2ξ)    (3.37) - spadek indukcyjności żłobka."""
    xi = np.clip(np.asarray(xi, dtype=float), 1e-3, 40.0)
    a = 2 * xi
    den = np.cosh(a) - np.cos(a)
    Kr = xi * (np.sinh(a) + np.sin(a)) / den
    Kx = 3 / (2 * xi) * (np.sinh(a) - np.sin(a)) / den
    return Kr, Kx


def skin_depth(gamma, f, mu=MU0):
    """Głębokość wnikania δ = sqrt(2/(μ γ ω)) [m]."""
    return np.sqrt(2.0 / (mu * gamma * 2 * PI * np.maximum(f, 1e-9)))


def thyristor_vrms(alpha):
    """Dwa tyrystory przeciwsobne, obciążenie R: Vrms/V(pełne) w funkcji kąta α [rad]
    Vrms = V/√2 * sqrt(1 - α/π + sin2α/(2π))   (zadanie 3.6)."""
    a = np.asarray(alpha, dtype=float)
    return np.sqrt(np.clip(1 - a / PI + np.sin(2 * a) / (2 * PI), 0, None))


def im_dynamic(prm, t_end, dt=1e-4, TL_func=None, Vamp_func=None):
    """Model dynamiczny silnika indukcyjnego w nieruchomym układzie αβ (wektory
    przestrzenne, strumienie jako zmienne stanu), całkowanie RK4:
        dψs/dt = vs - rs is
        dψr/dt = -rr ir + j p ωm ψr
        J dωm/dt = Te - TL - B ωm,    Te = 1.5 p Im(ψs* is)
    prm: VLL, f, P, rs, rr, Lls, Llr, Lm, J, B.
    Zwraca t, prąd fazy a, ωm [rad/s], Te."""
    rs, rr = prm["rs"], prm["rr"]
    Lm = prm["Lm"]
    Ls = prm["Lls"] + Lm
    Lr = prm["Llr"] + Lm
    D = Ls * Lr - Lm ** 2
    p = prm["P"] / 2
    J, B = prm["J"], prm.get("B", 0.0)
    Vpk = SQ2 * prm["VLL"] / SQ3
    we = 2 * PI * prm["f"]
    TL_func = TL_func or (lambda t, w: 0.0)
    Vamp_func = Vamp_func or (lambda t: 1.0)

    def deriv(t, ps, pr, w):
        vs = Vamp_func(t) * Vpk * cmath.exp(1j * we * t)
        i_s = (Lr * ps - Lm * pr) / D
        i_r = (Ls * pr - Lm * ps) / D
        Te = 1.5 * p * (ps.conjugate() * i_s).imag
        return vs - rs * i_s, -rr * i_r + 1j * p * w * pr, (Te - TL_func(t, w) - B * w) / J, i_s, Te

    n = int(t_end / dt)
    T = np.zeros(n); IA = np.zeros(n); W = np.zeros(n); TE = np.zeros(n)
    ps, pr, w, t = 0j, 0j, 0.0, 0.0
    h = dt / 2
    for k in range(n):
        a1, b1, c1, i_s, Te = deriv(t, ps, pr, w)
        T[k], IA[k], W[k], TE[k] = t, i_s.real, w, Te
        a2, b2, c2, _, _ = deriv(t + h, ps + h * a1, pr + h * b1, w + h * c1)
        a3, b3, c3, _, _ = deriv(t + h, ps + h * a2, pr + h * b2, w + h * c2)
        a4, b4, c4, _, _ = deriv(t + dt, ps + dt * a3, pr + dt * b3, w + dt * c3)
        ps += dt / 6 * (a1 + 2 * a2 + 2 * a3 + a4)
        pr += dt / 6 * (b1 + 2 * b2 + 2 * b3 + b4)
        w += dt / 6 * (c1 + 2 * c2 + 2 * c3 + c4)
        t += dt
    return T, IA, W, TE


def double_cage_zr(s, f, r1, L1, r2, L2, Lc=0.0):
    """Impedancja wirnika z klatką podwójną: górna (rozruchowa) r1, L1 równolegle
    z dolną (pracy) r2, L2, plus wspólna indukcyjność rozproszenia Lc."""
    s = _s(s)
    we = 2 * PI * f
    Z1 = r1 / s + 1j * we * L1
    Z2 = r2 / s + 1j * we * L2
    return 1j * we * Lc + Z1 * Z2 / (Z1 + Z2)


def deep_bar_zr(s, f, rr_dc, Llr_end, Llr_slot, xi1):
    """Pręt głęboki: rr(s) = rr_dc*Kr(ξ), Llr(s) = Llr_end + Llr_slot*Kx(ξ),
    ξ = ξ(s=1)*sqrt(s), bo δ ~ 1/sqrt(f_poślizgu)."""
    s = _s(s)
    Kr, Kx = skin_KrKx(xi1 * np.sqrt(np.abs(s)))
    return rr_dc * Kr / s + 1j * 2 * PI * f * (Llr_end + Llr_slot * Kx)


def te_sweep_parasitic(s, base, A5, A7):
    """Moment podstawowy + pasożytnicze od 5. i 7. harmonicznej MMF.
    s_ν = 1 - ν(1-s), ν = -5, 7 ;  T_ν = A_ν * T1(s_ν) * sign(ν)."""
    T1 = te_37(s, **base)
    s5 = 1 + 5 * (1 - s)
    s7 = 1 - 7 * (1 - s)
    T5 = -A5 * te_37(s5, **base)
    T7 = A7 * te_37(s7, **base)
    return T1, T5, T7


# =============================================================================
# OBJAŚNIENIA (dla ucznia liceum)
# =============================================================================

EXPL = {
"E31": """ĆWICZENIE 3.1 - Bilans mocy silnika indukcyjnego (diagram Sankeya)

JAK DZIAŁA SILNIK INDUKCYJNY? (najpierw intuicja)
Trzy uzwojenia stojana zasilane prądem trójfazowym tworzą WIRUJĄCE pole magnetyczne -
jakby ktoś kręcił magnesem wokół wirnika. Wirnik to "klatka wiewiórki": aluminiowe pręty
zwarte pierścieniami. Wirujące pole "przecina" pręty, indukuje w nich prąd (prawo Faradaya),
a na pręt z prądem w polu magnetycznym działa siła (siła Lorentza, reguła lewej dłoni).
Wirnik zaczyna "gonić" pole - ale NIGDY go nie dogoni: gdyby kręcił się tak samo szybko,
pole nie przecinałoby prętów, prąd by zniknął i moment też. To opóźnienie to POŚLIZG:
        s = (n_s - n) / n_s ,       n_s = 120 f / P   (prędkość synchroniczna, obr/min)
Dla P = 4 bieguny i f = 60 Hz:  n_s = 1800 obr/min.

DROGA ENERGII (jak woda w rurach - diagram Sankeya):
  P_in (z sieci)  ->  - straty w stojanie (miedź + żelazo)  ->  P_ag (moc szczeliny powietrznej)
  P_ag  ->  - straty w miedzi wirnika = s * P_ag   ->  P_m = (1 - s) * P_ag  (moc mechaniczna)
  P_m  ->  - tarcie, wentylacja, straty dodatkowe  ->  P_out (moc na wale)

Najważniejsza zależność (3.4):  P_cu2 = s * P_ag.  Duży poślizg = dużo ciepła w wirniku!

ROZWIĄZANIE (dane z książki):  P_in = 55 kW, straty stojana 4 kW, n = 1740 obr/min, 4 bieguny, 60 Hz
  a) P_ag = 55 - 4 = 51 kW
  b) s = (1800 - 1740)/1800 = 0,0333  ->  P_cu2 = 0,0333 * 51 = 1,7 kW
  c) P_m = 51 - 1,7 = 49,3 kW
  d) η = 49,3/55 = 89,6 %

TABELA 3.1 (prawy wykres): typowy podział strat w silnikach 25, 50 i 100 KM. Najwięcej
traci się w miedzi stojana; w dużych silnikach rośnie udział tarcia/wentylacji i strat
dodatkowych (prądy wirowe, efekt naskórkowy).

SPRÓBUJ: zmniejsz prędkość obrotową - zobacz, jak rośnie strata w wirniku (dolny wykres:
pole pod krzywą mocy mechanicznej maleje liniowo z poślizgiem).
""",
"E32": """ĆWICZENIE 3.2 - Charakterystyka moment-poślizg: schemat "T" a schemat uproszczony

SCHEMAT ZASTĘPCZY: silnik indukcyjny to "transformator z wirującym, zwartym uzwojeniem
wtórnym". Jedną fazę opisujemy obwodem:
  rs, Lls  - rezystancja i indukcyjność rozproszenia stojana
  Lm       - indukcyjność magnesująca (wytwarza pole magnetyczne)
  rr/s, Llr - wirnik; rezystor rr/s "udaje" obciążenie mechaniczne!
Gdy s = 0 (synchronizm): rr/s = nieskończoność -> prąd wirnika 0, moment 0.
Gdy s = 1 (wirnik zablokowany): rr/s = rr -> jak zwarty transformator, ogromny prąd.

SCHEMAT ZMODYFIKOWANY (rys. 3.9): przesuwamy Lm na zaciski. Wtedy prąd wirnika liczy się
jednym wzorem (3.6), a moment wzorem (3.7):
   Te = 3P/(2ωe) * Vs² (rr/s) / [ (rs + rr/s)² + ωe²(Lls+Llr)² ]
Jest to przybliżenie, bo pomijamy spadek napięcia na rs i Lls dla prądu magnesującego.

CO WIDAĆ NA WYKRESIE (rys. 3.10, 3.12):
  - blisko s = 0 moment rośnie LINIOWO z poślizgiem (3.8): Te ≈ 3P Vs² s / (2 ωe rr)
  - maksimum = MOMENT KRYTYCZNY (breakdown) przy s = sm
  - blisko s = 1 moment maleje jak 1/s (3.9)
  - s < 0: praca GENERATOROWA (wirnik szybszy od pola - hamowanie odzyskowe w pojazdach!)
  - s > 1: HAMOWANIE PRZECIWPRĄDEM (plugging) - wirnik kręci się przeciwnie do pola
  - prąd rozruchowy (s = 1) jest kilka razy większy niż znamionowy.

WNIOSEK Z KSIĄŻKI: schemat zmodyfikowany daje nieco większy moment (|Ir'| > |Ir|), ale przy
małych poślizgach różnice są znikome -> w praktyce wystarczy prostszy wzór (3.7).
Dane: rs = 0,1 Ω, Lls = Llr = 3 mH, Lm = 150 mH, rr = 0,25 Ω, 254 V (fazowe), 60 Hz, 4 bieguny.
""",
"E33": """ĆWICZENIE 3.3 (i ZADANIE 3.7) - Jak napięcie zasilania zmienia prędkość?

Silnik pracuje z obciążeniem o STAŁYM momencie (np. podnośnik, przenośnik) przy 1750 obr/min
z sieci 220 V, 60 Hz. Co się stanie, gdy napięcie wzrośnie do 250 V?

KLUCZ: przy małym poślizgu moment jest liniowy (3.8):   Te ≈ k * V² * s
Moment obciążenia się nie zmienia, więc  V1² s1 = V2² s2   ->   s2 = s1 (V1/V2)²

LICZBY:
  s1 = (1800 - 1750)/1800 = 0,02778
  s2 = 0,02778 * (220/250)² = 0,02151   -> spadek obrotów n_s*s2 = 38,7 obr/min
  n2 = 1800 - 38,7 = 1761 obr/min

INTERPRETACJA: większe napięcie = silniejsze pole = ten sam moment przy mniejszym poślizgu.
Prędkość zmienia się jednak BARDZO mało (11 obr/min) - dlatego regulacja prędkości samym
napięciem jest słaba (zob. zakładka NAP).

Na wykresie: dwie krzywe moment-prędkość (dla V1 i V2) i pozioma linia obciążenia. Punkt
pracy to przecięcie. Zielony punkt - wynik dokładny (bisekcja na pełnym wzorze 3.7),
w wynikach porównanie z przybliżeniem liniowym.
""",
"E34": """ĆWICZENIE 3.4 i 3.7 - Wpływ rezystancji wirnika rr (rys. 3.13)

Silnik z Tabeli 3.2 (10 KM, 220 V, 60 Hz, 6 biegunów). Porównujemy rr = 0,16; 0,32; 0,64 Ω.

CO MÓWIĄ WZORY:
  sm = rr / sqrt(rs² + ωe²(Lls+Llr)²)          (3.10) -> rośnie proporcjonalnie do rr
  Tmax = 3P/(4ωe) Vs²/(rs + sqrt(rs² + X²))    (3.11) -> NIE ZALEŻY od rr!
Krzywa "przesuwa się w lewo" - moment maksymalny ten sam, ale przy większym poślizgu.

KORZYŚĆ dużego rr: wyższy moment ROZRUCHOWY (s = 1) - dobre dla dźwigów, wind, młynów.
WADA: dla tego samego momentu (np. 80 Nm) potrzebny jest większy poślizg ->
niższa prędkość -> mniejsza moc mechaniczna P = Te*ω przy tym samym prądzie ->
NIŻSZA SPRAWNOŚĆ. Różnica idzie w ciepło w wirniku: 3*Ir²*rr.

Ciekawostka: dla stałego momentu iloraz rr/s jest stały, więc prąd wirnika Ir
jest IDENTYCZNY dla wszystkich trzech rr (rys. 3.13b) - zmienia się tylko prędkość!

Sprawność wg książki:  η = Te*ωr / (3 |Vs| |Ir|)  (stosunek mocy mechanicznej do mocy
"pozornej" gałęzi wirnika). W wynikach podajemy też sprawność "prawdziwą" Pm/Pin z pełnego
schematu T (bez strat w żelazie i tarcia).

SPRÓBUJ: zmień moment zadany powyżej momentu krytycznego - rozwiązanie w obszarze
stabilnym przestaje istnieć (silnik by utknął).
""",
"E35": """ĆWICZENIE 3.5 - Wyznaczenie rezystancji wirnika z danych tabliczki znamionowej

Dane: 10 KM (7,46 kW) na wale przy 1750 obr/min, 220 V, 60 Hz, 4 bieguny,
rs = 0,2 Ω, Lls = Llr = 2,3 mH, Lm = 34,2 mH.

KROK 1 - moment znamionowy:  Te = P/ω = 7460 / (1750*2π/60) = 40,7 Nm
KROK 2 - poślizg:            s = (1800-1750)/1800 = 0,0278
KROK 3 - z liniowego wzoru (3.8)  Te ≈ 3P Vs² s / (2 ωe rr):
         rr = 3P Vs² s / (2 ωe Te) = 3*4*127²*0,0278 / (2*377*40,7) ≈ 0,175 Ω
KROK 4 - prąd wirnika z mocy szczeliny:  Pag = Te * ωe*2/P = 7673 W,
         Pcu2 = s*Pag = 3 Ir² rr  ->  Ir ≈ 20 A

W wynikach pokazujemy też rr "dokładne" - takie, które z pełnym wzorem (3.7) daje 40,7 Nm
(wzór liniowy pomija rs i reaktancję, więc jest trochę niedokładny).
Wykres: szkic charakterystyki moment-prędkość z zaznaczonym punktem znamionowym i krytycznym.
INŻYNIERSKA UWAGA: to klasyczny sposób "odgadnięcia" parametrów silnika, gdy producent
ich nie podaje - tylko z tabliczki znamionowej.
""",
"E36": """ĆWICZENIE 3.6 - Punkt znamionowy silnika z Tabeli 3.2

1) Poślizg znamionowy:  s = (1200 - 1164)/1200 = 0,03   (6 biegunów -> n_s = 1200 obr/min)
2) Prąd: impedancja gałęzi wirnika rr/s + jωLlr = 5,33 + j0,27 Ω, równolegle z jωLm = j15,46 Ω,
   szeregowo rs + jωLls = 0,29 + j0,52 Ω. Całkowita impedancja ≈ 5,4 Ω pod kątem ≈ 25,7°.
   Napięcie fazowe 220/√3 = 127 V  ->  Is = 127/|Z| ≈ 23 A
3) cos φ = cos 25,7° ≈ 0,90 (indukcyjny - prąd opóźnia się za napięciem)
4) Moment znamionowy z mocy szczeliny: Te = 3 Ir² (rr/s) * (P/2)/ωe ≈ 60 Nm
5) Moment krytyczny (3.11) ≈ 170 Nm, czyli ok. 2,8 razy więcej niż znamionowy
   (typowo 2-4 razy - to "zapas" na chwilowe przeciążenia).

WYKRES WSKAZOWY (lewy): napięcie Vs pionowo; prąd stojana Is = Ir + Im. Prąd magnesujący Im
jest prawie prostopadły do napięcia (tylko "buduje" pole, nie daje mocy czynnej), prąd wirnika
Ir jest prawie w fazie z napięciem (to on daje moc i moment).
Prawy dolny wykres: współczynnik mocy i sprawność w funkcji poślizgu - silnik indukcyjny
najlepiej pracuje w pobliżu obciążenia znamionowego; na biegu jałowym cos φ jest bardzo mały.
""",
"STAB": """3.2.4 - Obszar stabilny i niestabilny

Prędkość ustala się tam, gdzie moment silnika = moment obciążenia (przecięcie krzywych).
Ale nie każde przecięcie jest "dobre"!

OBSZAR STABILNY (0 < s < sm, prawa część krzywej): gdy obciążenie chwilowo wzrośnie,
silnik zwalnia -> poślizg rośnie -> moment silnika ROŚNIE -> silnik wraca do równowagi.
Działa jak sprężyna - sam się stabilizuje (punkt A).

OBSZAR NIESTABILNY (sm < s < 1): gdy silnik zwolni, poślizg rośnie, ale moment MALEJE ->
silnik zwalnia jeszcze bardziej -> ... aż do zatrzymania (UTYK, punkt B).

KRYTERIUM (z nachyleń):  stabilnie, gdy  d(Te - TL)/dω < 0.

SYMULACJA (dolny wykres): równanie ruchu  J dω/dt = Te(ω) - TL(t).
W chwili t_skok obciążenie skacze z TL1 do TL2.
  - TL2 < Tmax: prędkość lekko spada i stabilizuje się (nowy punkt A').
  - TL2 > Tmax: silnik "przewraca się" przez moment krytyczny i staje - w praktyce
    zadziała zabezpieczenie nadprądowe, bo prąd rośnie do wartości rozruchowej.
Typ obciążenia: stały (dźwig) albo wentylatorowy (TL ~ ω²) - wentylator jest
"bezpieczniejszy", bo przy spadku prędkości jego moment też maleje.
""",
"PAR": """3.2.5 - Momenty pasożytnicze (od wyższych harmonicznych pola)

Uzwojenie stojana leży w żłobkach, więc pole w szczelinie nie jest idealną sinusoidą -
ma "schodki". Rozkład na harmoniczne daje fale 5., 7., 11., 13. rzędu (ν = -5, 7, -11, 13 ...).
Harmoniczna ν wiruje |ν| razy WOLNIEJ; znak minus oznacza wirowanie w przeciwną stronę.

Każda harmoniczna działa jak "mały, osobny silnik" ze swoją prędkością synchroniczną.
Jej poślizg:  s_ν = 1 - ν(1 - s)  -> zeruje się przy  s = 1 - 1/ν :
  5. harmoniczna (ν = -5): s = 1,2  (wirnik kręci się do tyłu z 1/5 prędkości)
  7. harmoniczna (ν = 7):  s = 0,857 (1/7 prędkości synchronicznej)

Efekt (rys. 3.15): w charakterystyce pojawia się "dołek" w okolicy s ≈ 0,86.
Jeśli jest głębszy niż moment obciążenia, silnik podczas rozruchu "przyklei się" do
ok. 1/7 prędkości synchronicznej i nie rozpędzi dalej - tzw. PEŁZANIE (crawling).

Dolny wykres: symulacja rozruchu z obciążeniem stałym. Zwiększ A7 lub moment obciążenia
i zobacz, jak silnik zatrzymuje się na ~1/7 n_s.
ROZWIĄZANIA KONSTRUKTORA: właściwy dobór liczby żłobków stojana i wirnika oraz
SKOS żłobków (pręty wirnika lekko skręcone) - "rozmywa" harmoniczne.
""",
"E38": """3.3 / ĆWICZENIE 3.8 - Indukcyjność rozproszenia żłobka

Nie cały strumień stojana dociera do wirnika - część "zawraca" wokół żłobka. To strumień
ROZPROSZENIA: nie wytwarza momentu, tylko pobiera moc bierną. Im większe rozproszenie, tym
mniejszy moment krytyczny (3.11) i gorszy cos φ.

PRAWO AMPÈRE'A w żłobku o szerokości b (żelazo ma μ -> ∞, więc H w żelazie = 0):
Pętla na wysokości y obejmuje część przewodów:  B(y) * b = μ0 * (prąd objęty)
  - w części wypełnionej przewodami (0 < y < h1):  B rośnie LINIOWO z y
  - w pustej części nad przewodami (h1 < y < h1+h2): B = μ0 Nsl I / b = const

ENERGIA pola  W = ∫ B²/(2μ0) dV,  a indukcyjność  L = 2W/I². Wynik (ćw. 3.8):
     L_sl = μ0 Nsl² lst ( h1/(3b) + h2/b )
Część wypełniona liczy się tylko w 1/3 (bo B rośnie od zera), a pusta - w całości!
Wniosek konstrukcyjny: głębokie i wąskie żłobki = duże rozproszenie.

POŁĄCZENIA CZOŁOWE (Hanselman, półokrąg):  L_e ≈ μ0 N² τp/8 * ln(τp² π / (4 As))
  τp - podziałka cewki, As - pole przekroju wiązki przewodów (ang. overhang area).

Wykres górny: przekrój żłobka i rozkład B(y). Dolny: jak indukcyjność rośnie z wysokością
pustej części h2 (np. klin zamykający żłobek).
""",
"CIRC": """3.4 / ĆWICZENIA 3.9-3.12 - Wykres kołowy Heylanda

POMYSŁ: zamiast liczyć prąd dla każdego poślizgu osobno, rysujemy KONIEC wektora prądu
stojana Is na płaszczyźnie. Gdy s zmienia się od -∞ do +∞, koniec wektora zakreśla OKRĄG!
(dla rs = 0 jest to dokładnie okrąg - to matematyczna własność funkcji typu 1/(a+jb s)).
Oś pionowa = składowa czynna (w fazie z napięciem), pozioma = składowa bierna.

CHARAKTERYSTYCZNE PUNKTY:
  N (s = 0)  - prąd jałowy Is0 = Vs/(ωe Ls), prawie czysto bierny
  Q (s = 1)  - zwarcie (rozruch), prąd Is0/σ - ok. 10 razy większy!
  R (s = ∞)  - prąd graniczny
  styczna z początku układu - punkt ZNAMIONOWY (najmniejszy kąt φ = najlepszy cos φ):
      cos φ_min = (1 - σ)/(1 + σ),    s_r = sm * √σ,    |Is(s_r)| = Is0/√σ
  najwyższy punkt okręgu - moment krytyczny, s = sm ≈ 1/(ωe τr σ),  τr = Lr/rr

σ - współczynnik rozproszenia, σ ≈ (Lls + Llr)/Lm (ok. 0,05-0,1). Im mniejsze σ,
tym większy okrąg i lepszy cos φ.

GEOMETRIA MOCY (rys. 3.22): dla punktu pracy P rysujemy pion PB.
  PB  ~ moc szczeliny (moment!),    PD = PB(1-s) ~ moc mechaniczna,    DB = PB*s ~ straty w wirniku
  Linia NQ - linia zerowej mocy mechanicznej, NR - linia zerowego momentu.
Ćw. 3.11: nachylenie DB/NB jest stałe - dlatego punkty D leżą na prostej.

ĆW. 3.10 (Tabela 3.2): σ = 0,052; cos φ_min = 0,90; τr = 0,26 s; s_r = 0,044; Is0 = 7,95 A; Is_r = 34,7 A.
ĆW. 3.12: domyślne suwaki. Prawy wykres porównuje moment odczytany z koła z wzorem.
SPRÓBUJ: rs > 0 - okrąg przesuwa się (praktyczny wykres kołowy, rys. 3.24).
""",
"E313": """3.5 / ĆWICZENIE 3.13 - Wypieranie prądu (efekt naskórkowy) w pręcie wirnika

Prąd przemienny "nie lubi" płynąć środkiem przewodnika. Zmienne pole przechodzące przez pręt
indukuje w nim prądy wirowe, które dodają się do prądu u góry pręta (przy szczelinie)
i odejmują na dole. Efekt: prąd tłoczy się przy powierzchni wirnika.

GŁĘBOKOŚĆ WNIKANIA:  δ = sqrt( 2 / (μ0 γ ω) ),   ω = 2π f s  (częstotliwość prądu w wirniku!)
Wskaźnik ξ = h_pręta / δ.
  Kr = rac/rdc = ξ (sh2ξ + sin2ξ)/(ch2ξ - cos2ξ)     - rezystancja ROŚNIE
  Kx = 3/(2ξ) (sh2ξ - sin2ξ)/(ch2ξ - cos2ξ)          - indukcyjność żłobka MALEJE
Dla dużego ξ:  Kr ≈ ξ,  Kx ≈ 3/(2ξ).

ĆW. 3.13: pręt miedziany h = 3 cm, γ = 5*10^7 S/m, 60 Hz, s = 1 (rozruch):
  δ = sqrt(2/(4π*10^-7 * 5*10^7 * 377)) = 9,19 mm,  ξ = 30/9,19 = 3,26  ->  Kr ≈ 3,26
Rezystancja wirnika przy rozruchu jest ponad 3 razy większa niż w czasie normalnej pracy!

DLACZEGO TO DOBRZE? Przy rozruchu duże rr = duży moment rozruchowy i mniejszy prąd,
a przy pracy (s ≈ 0,03 -> f_wirnika ≈ 2 Hz -> ξ małe) rr wraca do małej wartości = wysoka
sprawność. "Darmowy" automatyczny przełącznik! Wykorzystują to pręty głębokie i klatki
podwójne (zakładka NEMA).
Wykres środkowy: rozkład gęstości prądu w pręcie (góra pręta = przy szczelinie).
""",
"NEMA": """3.5.1 - Klatka podwójna i klasy NEMA

KLATKA PODWÓJNA: każdy pręt dzielimy na dwa:
  - GÓRNY (przy szczelinie): mały przekrój -> duża rezystancja r1, małe rozproszenie L1
    = klatka ROZRUCHOWA
  - DOLNY (głęboko): duży przekrój -> mała rezystancja r2, duże rozproszenie L2
    = klatka PRACY
Przy rozruchu (duża częstotliwość w wirniku) reaktancja ωL2 dolnej klatki jest duża,
więc prąd płynie górą -> duża rezystancja -> DUŻY moment rozruchowy.
Przy pracy (mała częstotliwość) reaktancje znikają, prąd płynie głównie dołem
-> mała rezystancja -> mały poślizg i WYSOKA sprawność. Najlepsze z dwóch światów!

KLASY NEMA (USA), rys. 3.31:
  A - małe rr, pręt owalny: duży moment krytyczny, mały poślizg, duży prąd rozruchowy.
      Dobre do falowników.
  B - pręt głęboki: najpopularniejsze (wentylatory, pompy), prąd rozruchowy <= 6,4 In,
      moment krytyczny 175-300 %.
  C - klatka podwójna: duży moment rozruchowy (~200 %), poślizg < 5 % (sprężarki, przenośniki).
  D - bardzo duże rr (pręty z brązu): ogromny moment rozruchowy, poślizg 5-13 %,
      niska sprawność - praca przerywana (prasy, nożyce).

Wykres górny: krzywe A-D (modele: A,D - pojedyncza klatka; B - pręt głęboki z Kr/Kx;
C - klatka podwójna) oraz TWOJA klatka podwójna (suwaki).
Dolny lewy: efektywna rezystancja wirnika w funkcji poślizgu dla Twojej klatki.
""",
"START": """3.5.2 - Rozruch bezpośredni (DOL) i softstart - symulacja dynamiczna

Przy włączeniu do sieci wirnik stoi (s = 1) -> silnik jest jak zwarty transformator.
Płynie prąd 5-7 razy większy od znamionowego (rys. 3.32: 135 A szczytowo vs 12 A),
który może spowodować spadek napięcia w sieci, wyłączenie bezpiecznika itp.
Dlatego rozruchu bezpośredniego nie stosuje się zwykle powyżej ok. 25 kW.

MODEL (to już "prawdziwa" dynamika, nie stan ustalony): wektory przestrzenne w układzie
nieruchomym αβ, zmienne stanu: strumień stojana ψs, strumień wirnika ψr, prędkość ω:
   dψs/dt = vs - rs is
   dψr/dt = -rr ir + j p ω ψr
   J dω/dt = Te - TL,    Te = 1,5 p Im(ψs* is)
Całkujemy metodą Rungego-Kutty 4. rzędu (krok 0,1 ms).

CO ZOBACZYSZ: na początku prąd ma składową stałą (asymetria - jak w zwarciu
transformatora), moment oscyluje z częstotliwością sieci (udary mechaniczne na wale!),
potem prąd maleje, gdy silnik się rozpędza, aż do prądu jałowego.
Prawy dolny: dynamiczna trajektoria moment-prędkość (z oscylacjami) na tle
charakterystyki statycznej.

SOFTSTART: napięcie zaczyna od V0 (np. 40 %) i rośnie liniowo przez czas T_rampy.
Moment ~ V², prąd ~ V -> mniejszy udar prądu, łagodniejszy rozruch, ale dłuższy.
Ustaw V0 = 100 %, by mieć rozruch bezpośredni.
""",
"TYR": """3.5.2 / ZADANIE 3.6 - Sterownik tyrystorowy (softstart)

Softstart to w każdej fazie para tyrystorów połączonych przeciwsobnie (rys. 3.34).
Tyrystor to "dioda z bramką": nie przewodzi, dopóki nie dostanie impulsu na bramkę,
a wyłącza się sam, gdy prąd spadnie do zera.

KĄT ZAŁĄCZENIA α: impuls podajemy z opóźnieniem α po przejściu napięcia przez zero.
Przez pierwszą część półokresu napięcie "nie dochodzi" do silnika. Im większe α, tym
mniej napięcia (rys. 3.33).

ZADANIE 3.6 - wzór (obciążenie rezystancyjne, v = V sin ωt):
   Vrms² = (1/π) ∫_α^π V² sin²θ dθ = (V²/2) [1 - α/π + sin(2α)/(2π)]
   Vrms = (V/√2) sqrt(1 - α/π + sin2α/(2π))
Sprawdzenie: α = 0 -> V/√2 (pełne napięcie), α = 180° -> 0. α = 90° -> 0,707 * V/√2.

SOFTSTART w praktyce: na początku α duże (np. 120°), potem stopniowo zmniejszane do 0 -
napięcie rośnie łagodnie. Po rozruchu tyrystory są stale załączone (lub zwierane
stycznikiem obejściowym, żeby nie grzały się niepotrzebnie).
WADA: napięcie nie jest sinusoidalne (wykres harmonicznych) - dodatkowe straty i hałas.
(Przy obciążeniu indukcyjnym - silnik - prąd płynie dłużej niż do π, uproszczenie jak w książce.)
""",
"NAP": """3.6.1 - Sterowanie prędkości samym napięciem (częstotliwość stała 60 Hz)

Najprostsza metoda: zmniejszamy napięcie (np. tyrystorami). Moment krytyczny maleje
z kwadratem napięcia (3.11):  Tmax ~ V²,  ale poślizg krytyczny sm NIE zmienia się.

Obciążenie wentylatorowe (pompa, wentylator):  TL = k * ω²
Punkty pracy (kropki) to przecięcia krzywych silnika z krzywą obciążenia.

WNIOSEK (rys. 3.35): zakres regulacji jest MAŁY - punkty pracy przesuwają się tylko
w wąskim przedziale poślizgu. Przy dużym obniżeniu napięcia punkt wchodzi w obszar
niestabilny i rosną straty w wirniku (Pcu2 = s * Pag). Dla obciążenia o stałym momencie
metoda praktycznie nie działa.
Metoda bywa stosowana tylko w małych wentylatorach (np. sufitowych) i tam, gdzie
liczy się cena, a nie sprawność. Lepsza metoda -> VVVF (następna zakładka).
""",
"VVVF": """3.6.2 - Sterowanie VVVF (zmienne napięcie, zmienna częstotliwość), U/f = const

Falownik zmienia CZĘSTOTLIWOŚĆ -> zmienia się prędkość synchroniczna n_s = 120 f / P.
Ale napięcie trzeba zmieniać RAZEM z częstotliwością, bo strumień (3.38):
      |λs| ≈ Vs / ωe
Gdybyśmy obniżyli f przy pełnym napięciu - strumień by wzrósł, żelazo by się nasyciło,
prąd magnesujący wystrzeliłby w górę. Dlatego U/f = const (np. 127 V / 60 Hz).

Wtedy (3.39) moment zależy tylko od prędkości poślizgu ωsl, nie od f -> krzywe mają
ten sam kształt i tylko się PRZESUWAJĄ (rys. 3.36). Punkt "x" (duży moment przy małej
prędkości) jest niestabilny dla 60 Hz, ale stabilny dla 12 Hz - tak rusza dźwig!

PODBICIE NAPIĘCIA (boost): przy małych f spadek na rs jest duży w porównaniu z napięciem
(stosunek rs/(ωe·Ls) rośnie 10 razy przy zejściu z 60 Hz do 6 Hz - w książce 0,048 -> 0,48,
dla danych z Tabeli 3.2 wychodzi 0,018 -> 0,18) -> strumień i moment spadają.
Dlatego dodajemy napięcie V0 przy niskich częstotliwościach (rys. 3.37, środkowy wykres).
Wyłącz boost (V0 = 0) i zobacz, jak "kurczą się" krzywe dla 12 Hz.

REGULATOR Z POŚLIZGIEM (rys. 3.37 dół): regulator PI prędkości daje zadany poślizg ωsl*,
który dodajemy do zmierzonej prędkości: ωe = (P/2)ωr + ωsl. Z tablicy U/f -> napięcie.
Symulacja: rozruch do prędkości zadanej, a potem skok obciążenia.
Zastosowania: sprężarki, wentylatory, dźwigi, lokomotywy - wszędzie, gdzie nie trzeba
bardzo dokładnej regulacji (dokładną daje sterowanie wektorowe - rozdział 6).
""",
}


# =============================================================================
# GUI - klasa bazowa zakładki
# =============================================================================

C1, C2, C3, C4 = "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"


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
        left = ttk.Frame(pw, width=300)
        right = ttk.Frame(pw)
        pw.add(left, weight=0)
        pw.add(right, weight=1)

        self.top_controls(left)
        ttk.Label(left, text="Parametry", font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=6, pady=(6, 2))
        # suwaki w przewijanym obszarze (wiele parametrów nie mieści się na ekranie)
        holder = ttk.Frame(left)
        holder.pack(fill=tk.BOTH, expand=False, padx=2)
        self.pcanvas = tk.Canvas(holder, width=270, height=330, highlightthickness=0)
        sb = ttk.Scrollbar(holder, orient=tk.VERTICAL, command=self.pcanvas.yview)
        self.pframe = ttk.Frame(self.pcanvas)
        self.pframe.bind("<Configure>", lambda _e: self.pcanvas.configure(scrollregion=self.pcanvas.bbox("all")))
        self.pcanvas.create_window((0, 0), window=self.pframe, anchor="nw")
        self.pcanvas.configure(yscrollcommand=sb.set)
        # kółko myszy przewija listę suwaków (Windows/macOS: <MouseWheel>, Linux: Button-4/5)
        wheel = lambda e: self.pcanvas.yview_scroll(-1 if (getattr(e, "delta", 0) > 0 or e.num == 4) else 1, "units")
        holder.bind("<Enter>", lambda _e: [self.pcanvas.bind_all(k, wheel) for k in ("<MouseWheel>", "<Button-4>", "<Button-5>")])
        holder.bind("<Leave>", lambda _e: [self.pcanvas.unbind_all(k) for k in ("<MouseWheel>", "<Button-4>", "<Button-5>")])
        self.pcanvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.build_params(self.PARAMS)

        self.extra_controls(left)
        ttk.Button(left, text="Przywróć domyślne", command=self.reset).pack(fill=tk.X, padx=6, pady=4)
        ttk.Label(left, text="Wyniki", font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=6)
        self.result = tk.Text(left, height=14, width=46, font=("Consolas", 9), bg="#f4f6fa")
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
        self.set_expl(EXPL.get(self.key, ""))
        vpw.add(figfr, weight=3)
        vpw.add(txtfr, weight=1)
        self.refresh()

    # --- budowa suwaków ---------------------------------------------------
    def build_params(self, params):
        for w in self.pframe.winfo_children():
            w.destroy()
        self.vars = {}
        self.cur_params = params
        for name, label, lo, hi, val, res in params:
            fr = ttk.Frame(self.pframe)
            fr.pack(fill=tk.X, padx=4, pady=1)
            var = tk.DoubleVar(value=val)
            self.vars[name] = var
            ttk.Label(fr, text=label).pack(anchor="w")
            sc = tk.Scale(fr, from_=lo, to=hi, resolution=res, orient=tk.HORIZONTAL,
                          variable=var, showvalue=True, length=240,
                          command=lambda _e: self.schedule())
            sc.pack(fill=tk.X)
        self.pcanvas.yview_moveto(0)

    def set_expl(self, text):
        self.expl.configure(state=tk.NORMAL)
        self.expl.delete("1.0", tk.END)
        self.expl.insert("1.0", text)
        self.expl.configure(state=tk.DISABLED)

    def top_controls(self, parent):
        pass

    def extra_controls(self, parent):
        pass

    def reset(self):
        for name, _l, _lo, _hi, val, _r in self.cur_params:
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
        except Exception as exc:          # nie pozwól, by błąd obliczeń zamknął aplikację
            lines = ["BŁĄD OBLICZEŃ:", repr(exc)]
        self.result.delete("1.0", tk.END)
        self.result.insert("1.0", "\n".join(lines))
        self.canvas.draw_idle()

    def update_plot(self):
        raise NotImplementedError


def shade_regions(ax):
    """Zaznacza obszary: hamowanie przeciwprądem (s>1), silnik, generator (s<0)."""
    ax.axvspan(1, 10, color="#f8d7da", alpha=0.5, lw=0)
    ax.axvspan(-10, 0, color="#d4edda", alpha=0.5, lw=0)
    ax.axvline(0, color="k", lw=0.6)
    ax.axvline(1, color="k", lw=0.6)
    ax.axhline(0, color="k", lw=0.6)


# Parametry silnika z Tabeli 3.2 jako suwaki (wielokrotnie używane)
TAB32_PARAMS = [
    ("VLL", "Napięcie przewodowe VLL [V]", 50, 440, 220, 1),
    ("f", "Częstotliwość f [Hz]", 10, 100, 60, 1),
    ("P", "Liczba biegunów P", 2, 12, 6, 2),
    ("rs", "rs [Ω]", 0.0, 2.0, 0.29, 0.01),
    ("Lls", "Lls [mH]", 0.1, 10, 1.38, 0.01),
    ("rr", "rr [Ω]", 0.01, 2.0, 0.16, 0.01),
    ("Llr", "Llr [mH]", 0.1, 10, 0.717, 0.001),
    ("Lm", "Lm [mH]", 5, 300, 41, 0.5),
]


def motor_from(tab, rr=None):
    """Słownik parametrów silnika z suwaków (jednostki SI)."""
    return dict(Vph=tab.p("VLL") / SQ3, f=tab.p("f"), rs=tab.p("rs"), Lls=tab.p("Lls") * 1e-3,
                rr=tab.p("rr") if rr is None else rr, Llr=tab.p("Llr") * 1e-3,
                Lm=tab.p("Lm") * 1e-3, P=int(round(tab.p("P"))))


def no_lm(m):
    d = dict(m)
    d.pop("Lm", None)
    return d


# =============================================================================
# ZAKŁADKI - ĆWICZENIA I ZAGADNIENIA Z ROZDZIAŁU
# =============================================================================

class TabE31(ExampleTab):
    key = "E31"
    PARAMS = [("Pin", "Moc pobierana Pin [kW]", 5, 200, 55, 0.5),
              ("Pst", "Straty w stojanie (Cu+Fe) [kW]", 0, 30, 4, 0.1),
              ("n", "Prędkość wirnika n [obr/min]", 600, 3600, 1740, 1),
              ("P", "Liczba biegunów P", 2, 12, 4, 2),
              ("f", "Częstotliwość f [Hz]", 50, 60, 60, 10),
              ("Pfw", "Tarcie i wentylacja [kW]", 0, 10, 0, 0.1)]

    def update_plot(self):
        Pin, Pst, n, P, f, Pfw = (self.p(k) for k in ("Pin", "Pst", "n", "P", "f", "Pfw"))
        ns = n_sync(f, P)
        s = (ns - n) / ns
        Pag = Pin - Pst
        Pcu2 = s * Pag
        Pm = Pag - Pcu2
        Pout = Pm - Pfw
        eta = Pout / Pin * 100

        gs = self.fig.add_gridspec(2, 2)
        ax = self.fig.add_subplot(gs[0, :])
        labels = ["P_in", "-straty\nstojana", "P_ag\n(szczelina)", "-Cu wirnika\n= s·P_ag", "P_m", "-tarcie\ni went.", "P_out"]
        vals = [Pin, -Pst, Pag, -Pcu2, Pm, -Pfw, Pout]
        bottoms = [0, Pag, 0, Pm, 0, Pout, 0]
        heights = [Pin, Pst, Pag, Pcu2, Pm, Pfw, Pout]
        cols = [C1, C2, C1, C2, C3, C2, C3]
        ax.bar(range(7), heights, bottom=bottoms, color=cols, edgecolor="k")
        for i, v in enumerate(vals):
            ax.text(i, bottoms[i] + heights[i] + Pin * 0.01, f"{v:.2f} kW", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(7))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("Moc [kW]")
        ax.set_title(f"Przepływ mocy (Sankey) - s = {s:.4f}, η = {eta:.1f} %")
        ax.set_ylim(0, Pin * 1.15)

        ax2 = self.fig.add_subplot(gs[1, 0])
        cats = ["Cu stojana", "Cu wirnika", "Żelazo", "Tarcie+went.", "Dodatkowe"]
        data = {"25 KM": [42, 21, 15, 7, 15], "50 KM": [38, 22, 20, 8, 12], "100 KM": [28, 18, 13, 14, 27]}
        x = np.arange(len(cats))
        for i, (lab, d) in enumerate(data.items()):
            ax2.bar(x + (i - 1) * 0.27, d, 0.27, label=lab)
        ax2.set_xticks(x)
        ax2.set_xticklabels(cats, fontsize=7, rotation=15)
        ax2.set_ylabel("Udział w stratach [%]")
        ax2.set_title("Tabela 3.1 - składniki strat")
        ax2.legend(fontsize=7)

        ax3 = self.fig.add_subplot(gs[1, 1])
        nn = np.linspace(0, ns, 200)
        ss = (ns - nn) / ns
        ax3.stackplot(nn, (1 - ss) * Pag, ss * Pag, labels=["P_m = (1-s)P_ag", "P_cu2 = s·P_ag"],
                      colors=[C3, C2], alpha=0.7)
        ax3.axvline(n, color="k", ls="--")
        ax3.set_xlabel("Prędkość n [obr/min]")
        ax3.set_ylabel("[kW]")
        ax3.set_title("Podział P_ag przy stałej mocy szczeliny")
        ax3.legend(fontsize=7, loc="upper left")
        lines = [f"n_s = 120 f / P = {ns:.0f} obr/min",
                 f"s = (n_s - n)/n_s = {s:.4f}",
                 f"a) P_ag = Pin - Pst = {Pag:.2f} kW",
                 f"b) P_cu2 = s·P_ag = {Pcu2:.3f} kW",
                 f"c) P_m = (1-s)·P_ag = {Pm:.2f} kW",
                 f"   P_out = P_m - Pfw = {Pout:.2f} kW",
                 f"d) η = P_out / Pin = {eta:.2f} %",
                 f"Moment: Te = P_ag/ω_s = {Pag * 1e3 / rpm2rad(ns):.1f} Nm"]
        if s < 0:
            lines.append("UWAGA: n > n_s -> praca generatorowa!")
        return lines


class TabE32(ExampleTab):
    key = "E32"
    PARAMS = [("VLL", "Napięcie przewodowe VLL [V]", 100, 600, 440, 1),
              ("f", "Częstotliwość f [Hz]", 10, 100, 60, 1),
              ("P", "Liczba biegunów P", 2, 12, 4, 2),
              ("rs", "rs [Ω]", 0.0, 1.0, 0.1, 0.01),
              ("Lls", "Lls [mH]", 0.5, 10, 3, 0.1),
              ("rr", "rr [Ω]", 0.02, 2.0, 0.25, 0.01),
              ("Llr", "Llr [mH]", 0.5, 10, 3, 0.1),
              ("Lm", "Lm [mH]", 20, 400, 150, 1)]

    def update_plot(self):
        m = motor_from(self)
        s = np.linspace(2.0, -1.0, 1501)
        rT = im_T(s, **m)
        rM = im_mod(s, **m)
        sm, Tmax = breakdown(**no_lm(m))
        ax = self.fig.add_subplot(2, 1, 1)
        shade_regions(ax)
        ax.plot(s, rT["Te"], C1, lw=2, label="(a) pełny schemat T")
        ax.plot(s, rM["Te"], C2, ls="--", lw=2, label="(b) schemat zmodyfikowany (3.7)")
        sl = np.linspace(0.0, sm, 50)
        ax.plot(sl, 3 * m["P"] * m["Vph"] ** 2 * sl / (2 * 2 * PI * m["f"] * m["rr"]), "k:", label="przybliż. liniowe (3.8)")
        ax.plot([sm], [Tmax], "o", color=C2)
        ax.annotate(f"Tmax = {Tmax:.0f} Nm\nsm = {sm:.3f}", (sm, Tmax), xytext=(10, -25), textcoords="offset points", fontsize=8)
        ax.set_xlim(2, -1)
        ymax = max(rT["Te"].max(), rM["Te"].max())
        ax.set_ylim(-1.3 * ymax, 1.3 * ymax)
        ax.text(1.5, 1.1 * ymax, "hamowanie\nprzeciwprądem", ha="center", fontsize=8)
        ax.text(0.5, 1.1 * ymax, "silnik", ha="center", fontsize=8)
        ax.text(-0.5, 1.1 * ymax, "generator", ha="center", fontsize=8)
        ax.set_xlabel("Poślizg s")
        ax.set_ylabel("Moment Te [Nm]")
        ax.set_title("Rys. 3.10a / 3.12 - charakterystyka moment-poślizg")
        ax.legend(fontsize=8, loc="lower left")
        ax.grid(alpha=0.3)
        ax2 = self.fig.add_subplot(2, 1, 2)
        shade_regions(ax2)
        ax2.plot(s, np.abs(rT["Is"]), C1, lw=2, label="|Is| schemat T")
        ax2.plot(s, np.abs(rM["Is"]), C2, ls="--", lw=2, label="|Is| schemat zmodyfikowany")
        ax2.plot(s, np.abs(rT["Ir"]), C3, lw=1, label="|Ir| schemat T")
        ax2.set_xlim(2, -1)
        ax2.set_ylim(0, np.abs(rM["Is"]).max() * 1.1)
        ax2.set_xlabel("Poślizg s")
        ax2.set_ylabel("Prąd [A]")
        ax2.set_title("Rys. 3.10b - prąd stojana w funkcji poślizgu")
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.3)
        k = np.array([1.0, 0.03])
        a, b = im_T(k, **m), im_mod(k, **m)
        iTm = np.argmax(rT["Te"])
        return [f"Prędkość synchr. n_s = {n_sync(m['f'], m['P']):.0f} obr/min",
                f"Vs (fazowe) = {m['Vph']:.1f} V",
                "                    schemat T   zmodyf.",
                f"Te(s=1) rozruch  {a['Te'][0]:9.1f} {b['Te'][0]:9.1f} Nm",
                f"Te(s=0,03)       {a['Te'][1]:9.1f} {b['Te'][1]:9.1f} Nm",
                f"|Is|(s=1)        {abs(a['Is'][0]):9.1f} {abs(b['Is'][0]):9.1f} A",
                f"|Is|(s=0,03)     {abs(a['Is'][1]):9.1f} {abs(b['Is'][1]):9.1f} A",
                f"Tmax             {rT['Te'][iTm]:9.1f} {Tmax:9.1f} Nm",
                f"sm               {rT['s'][iTm]:9.3f} {sm:9.3f}",
                "",
                f"Różnica momentu przy s=0,03: {(b['Te'][1] / a['Te'][1] - 1) * 100:.1f} %",
                f"Różnica przy s=1: {(b['Te'][0] / a['Te'][0] - 1) * 100:.1f} %",
                f"Prąd rozruchowy / prąd przy s=0,03 = {abs(a['Is'][0]) / abs(a['Is'][1]):.1f}"]


class TabE33(ExampleTab):
    key = "E33"
    PARAMS = [("V1", "Napięcie początkowe V1 (przewodowe) [V]", 100, 300, 220, 1),
              ("V2", "Napięcie nowe V2 [V]", 100, 300, 250, 1),
              ("n1", "Prędkość przy V1 [obr/min]", 1650, 1795, 1750, 1),
              ("f", "Częstotliwość f [Hz]", 50, 60, 60, 10),
              ("P", "Liczba biegunów P", 2, 8, 4, 2),
              ("rs", "rs [Ω] (silnik do wykresu)", 0.0, 1.0, 0.2, 0.01),
              ("L", "Lls = Llr [mH]", 0.5, 6, 2.3, 0.1),
              ("rr", "rr [Ω]", 0.05, 1.0, 0.175, 0.005)]

    def update_plot(self):
        V1, V2, n1, f, P = (self.p(k) for k in ("V1", "V2", "n1", "f", "P"))
        P = int(round(P))
        rs, L, rr = self.p("rs"), self.p("L") * 1e-3, self.p("rr")
        ns = n_sync(f, P)
        s1 = (ns - n1) / ns
        s2a = s1 * (V1 / V2) ** 2
        mot = dict(f=f, rs=rs, Lls=L, rr=rr, Llr=L, P=P)
        TL = float(te_37(s1, V1 / SQ3, **mot))
        sm2, Tmax2 = breakdown(V2 / SQ3, **mot)
        s2 = solve_slip(TL, lambda x: te_37(x, V2 / SQ3, **mot), sm2)
        nn = np.linspace(0, ns * 0.9999, 800)
        ss = 1 - nn / ns
        ax = self.fig.add_subplot(1, 2, 1)
        ax2 = self.fig.add_subplot(1, 2, 2)
        for a in (ax, ax2):
            a.plot(nn, te_37(ss, V1 / SQ3, **mot), C1, lw=2, label=f"V1 = {V1:.0f} V")
            a.plot(nn, te_37(ss, V2 / SQ3, **mot), C2, lw=2, label=f"V2 = {V2:.0f} V")
            a.axhline(TL, color="k", ls="--", label=f"obciążenie TL = {TL:.1f} Nm")
            a.plot([n1], [TL], "o", color=C1, ms=8)
            if s2 is not None:
                a.plot([ns * (1 - s2)], [TL], "o", color=C3, ms=8, label="nowy punkt pracy")
            a.set_xlabel("Prędkość n [obr/min]")
            a.set_ylabel("Moment [Nm]")
            a.grid(alpha=0.3)
        ax.set_title("Charakterystyki dla dwóch napięć")
        ax.legend(fontsize=8)
        lo = ns * (1 - 2.0 * s1)
        ax2.set_xlim(lo, ns)
        ax2.set_ylim(0, 2.2 * TL)
        ax2.set_title("Powiększenie - obszar pracy (prawie proste linie!)")
        lines = [f"n_s = {ns:.0f} obr/min",
                 f"s1 = (n_s - n1)/n_s = {s1:.5f}",
                 "Z (3.8): Te ~ V²·s  =>  s2 = s1·(V1/V2)²",
                 f"s2 (przybliż.) = {s2a:.5f}",
                 f"n2 (przybliż.) = {ns * (1 - s2a):.1f} obr/min",
                 f"  (spadek obrotów {ns * s2a:.1f} obr/min)"]
        if s2 is not None:
            lines += [f"s2 (dokładne, wzór 3.7) = {s2:.5f}",
                      f"n2 (dokładne) = {ns * (1 - s2):.1f} obr/min"]
        else:
            lines.append("Brak punktu pracy - obciążenie > Tmax!")
        lines += [f"Moment obciążenia TL = {TL:.2f} Nm", f"Tmax przy V2 = {Tmax2:.1f} Nm"]
        return lines


class TabE34(ExampleTab):
    key = "E34"
    PARAMS = [p for p in TAB32_PARAMS if p[0] != "rr"] + [
        ("rr1", "rr nr 1 [Ω]", 0.05, 1.5, 0.16, 0.01),
        ("rr2", "rr nr 2 [Ω]", 0.05, 1.5, 0.32, 0.01),
        ("rr3", "rr nr 3 [Ω]", 0.05, 1.5, 0.64, 0.01),
        ("T", "Moment zadany Te [Nm]", 5, 250, 80, 1)]

    def update_plot(self):
        base = dict(Vph=self.p("VLL") / SQ3, f=self.p("f"), rs=self.p("rs"), Lls=self.p("Lls") * 1e-3,
                    Llr=self.p("Llr") * 1e-3, P=int(round(self.p("P"))))
        Lm = self.p("Lm") * 1e-3
        Tt = self.p("T")
        ws = 2 * PI * base["f"] / (base["P"] / 2)
        s = np.linspace(1, 0.0005, 1000)
        ax = self.fig.add_subplot(2, 2, (1, 3))
        ax2 = self.fig.add_subplot(2, 2, 2)
        ax3 = self.fig.add_subplot(2, 2, 4)
        lines = [f"Moment zadany Te = {Tt:.0f} Nm", "rr[Ω]   sm    Tmax   s(Te)  |Ir'|  η_ks  η_T"]
        etas = []
        for rr, c in zip((self.p("rr1"), self.p("rr2"), self.p("rr3")), (C1, C3, C2)):
            T = te_37(s, rr=rr, **base)
            Ir = np.abs(im_mod(s, rr=rr, Lm=Lm, **base)["Ir"])
            sm, Tmax = breakdown(rr=rr, **base)
            ax.plot(s, T, color=c, lw=2, label=f"rr = {rr:.2f} Ω")
            ax.plot([sm], [Tmax], "^", color=c)
            ax2.plot(s, Ir, color=c, lw=2, label=f"rr = {rr:.2f} Ω")
            so = solve_slip(Tt, lambda x: te_37(x, rr=rr, **base), min(sm, 1.0))
            if so is None:
                lines.append(f"{rr:5.2f} {sm:6.3f} {Tmax:6.1f}  brak - Te > T(s≤sm)")
                etas.append((0, 0))
                continue
            irr = abs(complex(im_mod(so, rr=rr, Lm=Lm, **base)["Ir"]))
            wr = ws * (1 - so)
            eta_b = Tt * wr / (3 * base["Vph"] * irr)
            rt = im_T(so, rr=rr, Lm=Lm, **base)
            pin = 3 * float(np.real(base["Vph"] * np.conj(rt["Is"])))
            eta_t = float(rt["Te"]) * wr / pin
            etas.append((eta_b * 100, eta_t * 100))
            ax.plot([so], [Tt], "o", color=c, ms=7)
            ax2.plot([so], [irr], "o", color=c, ms=7)
            lines.append(f"{rr:5.2f} {sm:6.3f} {Tmax:6.1f} {so:6.4f} {irr:6.2f} {eta_b * 100:5.1f} {eta_t * 100:5.1f}")
        ax.axhline(Tt, color="k", ls="--", lw=1)
        ax.set_xlim(1, 0)
        ax.set_xlabel("Poślizg s")
        ax.set_ylabel("Moment [Nm]")
        ax.set_title("Rys. 3.13a - Tmax ten sam, sm rośnie z rr")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax2.set_xlim(1, 0)
        ax2.set_xlabel("Poślizg s")
        ax2.set_ylabel("|Ir'| [A]")
        ax2.set_title("Rys. 3.13b - prąd wirnika")
        ax2.grid(alpha=0.3)
        x = np.arange(3)
        ax3.bar(x - 0.2, [e[0] for e in etas], 0.4, label="η wg książki")
        ax3.bar(x + 0.2, [e[1] for e in etas], 0.4, label="η = Pm/Pin (schemat T)")
        ax3.set_xticks(x)
        ax3.set_xticklabels(["rr1", "rr2", "rr3"])
        ax3.set_ylim(0, 100)
        ax3.set_ylabel("Sprawność [%]")
        ax3.set_title(f"Sprawność przy Te = {Tt:.0f} Nm")
        ax3.legend(fontsize=7, loc="lower left")
        lines += ["", "η_ks = Te·ωr/(3|Vs||Ir'|) (wzór z książki)", "η_T  = Pm/Pin z pełnego schematu T",
                  "Zauważ: |Ir'| jednakowe dla wszystkich rr!"]
        return lines


class TabE35(ExampleTab):
    key = "E35"
    PARAMS = [("Php", "Moc na wale [KM]", 1, 50, 10, 0.5),
              ("n", "Prędkość znamionowa [obr/min]", 1650, 1795, 1750, 1),
              ("VLL", "Napięcie przewodowe [V]", 100, 480, 220, 1),
              ("f", "Częstotliwość [Hz]", 50, 60, 60, 10),
              ("P", "Liczba biegunów P", 2, 8, 4, 2),
              ("rs", "rs [Ω]", 0.0, 1.0, 0.2, 0.01),
              ("L", "Lls = Llr [mH]", 0.5, 6, 2.3, 0.1),
              ("Lm", "Lm [mH]", 10, 100, 34.2, 0.1)]

    def update_plot(self):
        Php, n, VLL, f, P = (self.p(k) for k in ("Php", "n", "VLL", "f", "P"))
        P = int(round(P))
        rs, L, Lm = self.p("rs"), self.p("L") * 1e-3, self.p("Lm") * 1e-3
        Vph = VLL / SQ3
        we = 2 * PI * f
        ns = n_sync(f, P)
        Pm = Php * HP
        Te = Pm / rpm2rad(n)
        s = (ns - n) / ns
        rr = 3 * P * Vph ** 2 * s / (2 * we * Te)
        Pag = Te * we * 2 / P
        Ir = math.sqrt(Pag * s / (3 * rr))
        # rr dokładne: Te(rr) maleje z rr dla rr/s > sqrt(rs²+X²)
        X = we * 2 * L
        lo, hi = s * math.sqrt(rs ** 2 + X ** 2), 20.0
        g = lambda r: float(te_37(s, Vph, f, rs, L, r, L, P))
        rr_ex = None
        if g(lo) >= Te:
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                if g(mid) > Te:
                    lo = mid
                else:
                    hi = mid
            rr_ex = 0.5 * (lo + hi)
        nn = np.linspace(0, ns * 0.9999, 800)
        ss = 1 - nn / ns
        ax = self.fig.add_subplot(2, 1, 1)
        T1 = te_37(ss, Vph, f, rs, L, rr, L, P)
        ax.plot(nn, T1, C1, lw=2, label=f"rr = {rr:.3f} Ω (wzór liniowy 3.8)")
        if rr_ex:
            ax.plot(nn, te_37(ss, Vph, f, rs, L, rr_ex, L, P), C2, ls="--", lw=2, label=f"rr = {rr_ex:.3f} Ω (dokładnie z 3.7)")
        sm, Tmax = breakdown(Vph, f, rs, L, rr, L, P)
        ax.plot([n], [Te], "o", color=C3, ms=9, label=f"punkt znamionowy {Te:.1f} Nm")
        ax.plot([ns * (1 - sm)], [Tmax], "^", color=C2, ms=9, label=f"moment krytyczny {Tmax:.1f} Nm")
        ax.set_xlabel("Prędkość [obr/min]")
        ax.set_ylabel("Moment [Nm]")
        ax.set_title("Szkic charakterystyki moment-prędkość (ćw. 3.5)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax2 = self.fig.add_subplot(2, 1, 2)
        rt = im_T(ss, Vph, f, rs, L, rr, L, Lm, P)
        ax2.plot(nn, np.abs(rt["Is"]), C1, lw=2, label="|Is| (schemat T)")
        ax2.plot(nn, np.abs(rt["Ir"]), C3, lw=1.5, label="|Ir|")
        ax2.axvline(n, color="k", ls="--")
        ax2.set_xlabel("Prędkość [obr/min]")
        ax2.set_ylabel("Prąd [A]")
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.3)
        lines = [f"Pm = {Php:.1f}·746 = {Pm:.0f} W",
                 f"ωr = {rpm2rad(n):.2f} rad/s",
                 f"Te = Pm/ωr = {Te:.2f} Nm",
                 f"s = ({ns:.0f}-{n:.0f})/{ns:.0f} = {s:.4f}",
                 f"Vs = {Vph:.1f} V, ωe = {we:.1f} rad/s",
                 f"rr = 3P·Vs²·s/(2ωe·Te) = {rr:.4f} Ω",
                 f"Pag = Te·ωe·2/P = {Pag:.0f} W",
                 f"Ir = sqrt(s·Pag/(3rr)) = {Ir:.2f} A"]
        if rr_ex:
            lines.append(f"rr dokładne (3.7) = {rr_ex:.4f} Ω")
        lines += [f"sm = {sm:.3f},  Tmax = {Tmax:.1f} Nm", f"Tmax/Te = {Tmax / Te:.2f}"]
        return lines


class TabE36(ExampleTab):
    key = "E36"
    PARAMS = TAB32_PARAMS + [("n", "Prędkość znamionowa [obr/min]", 1000, 1199, 1164, 1)]

    def update_plot(self):
        m = motor_from(self)
        ns = n_sync(m["f"], m["P"])
        n = min(self.p("n"), ns - 0.5)
        s = (ns - n) / ns
        r = im_T(s, **m)
        Is, Ir = complex(r["Is"]), complex(r["Ir"])
        Im = Is - Ir
        phi = math.degrees(-cmath.phase(Is))
        Te = float(r["Te"])
        sm, Tmax = breakdown(**no_lm(m))
        ss = np.linspace(1, 0.0005, 1000)
        rT = im_T(ss, **m)
        iTm = np.argmax(rT["Te"])
        ax = self.fig.add_subplot(2, 2, (1, 3))
        sc = abs(Is) * 1.15 / m["Vph"]

        def arr(z, col, lab):
            w = z * 1j     # obrót: napięcie (oś rzeczywista) rysujemy pionowo
            ax.annotate("", xy=(w.real, w.imag), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=2.2))
            ax.text(w.real * 1.04, w.imag * 1.04, lab, color=col, fontsize=9)
        arr(m["Vph"] * sc, "k", f"Vs = {m['Vph']:.0f} V (skala)")
        arr(Is, C1, f"Is = {abs(Is):.1f} A")
        arr(Ir, C3, f"Ir = {abs(Ir):.1f} A")
        arr(Im, C2, f"Im = {abs(Im):.1f} A")
        ax.plot([(Im * 1j).real, (Is * 1j).real], [(Im * 1j).imag, (Is * 1j).imag], ":", color="gray")
        lim = abs(Is) * 1.3
        ax.set_xlim(-0.3 * lim, lim)
        ax.set_ylim(-0.1 * lim, lim)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.set_title(f"Wykres wskazowy: φ = {phi:.1f}°, cos φ = {math.cos(math.radians(phi)):.3f}")
        ax2 = self.fig.add_subplot(2, 2, 2)
        nn = ns * (1 - ss)
        ax2.plot(nn, rT["Te"], C1, lw=2)
        ax2.plot([n], [Te], "o", color=C3, ms=8, label=f"znamionowy {Te:.1f} Nm")
        ax2.plot([nn[iTm]], [rT["Te"][iTm]], "^", color=C2, ms=8, label=f"krytyczny {rT['Te'][iTm]:.0f} Nm")
        ax2.set_xlabel("Prędkość [obr/min]")
        ax2.set_ylabel("Moment [Nm]")
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.3)
        ax3 = self.fig.add_subplot(2, 2, 4)
        sz = np.linspace(0.0005, 0.25, 400)
        rz = im_T(sz, **m)
        pin = 3 * np.real(m["Vph"] * np.conj(rz["Is"]))
        eta = rz["Te"] * rpm2rad(ns) * (1 - sz) / pin
        ax3.plot(sz, rz["pf"], C1, lw=2, label="cos φ")
        ax3.plot(sz, eta, C3, lw=2, label="η (bez strat Fe i tarcia)")
        ax3.axvline(s, color="k", ls="--")
        ax3.set_xlabel("Poślizg s")
        ax3.set_ylim(0, 1.02)
        ax3.legend(fontsize=8)
        ax3.grid(alpha=0.3)
        Z = complex(r["Zin"])
        we = 2 * PI * m["f"]
        return [f"1) s = ({ns:.0f}-{n:.0f})/{ns:.0f} = {s:.4f}",
                f"   rr/s = {m['rr'] / s:.3f} Ω",
                f"   Xls = {we * m['Lls']:.3f}, Xlr = {we * m['Llr']:.3f}, Xm = {we * m['Lm']:.2f} Ω",
                f"   Z = {Z.real:.3f} + j{Z.imag:.3f} = {abs(Z):.3f}∠{math.degrees(cmath.phase(Z)):.1f}° Ω",
                f"2) Vs = {m['Vph']:.1f} V  ->  |Is| = {abs(Is):.2f} A",
                f"   |Ir| = {abs(Ir):.2f} A,  |Im| = {abs(Im):.2f} A",
                f"3) cos φ = {math.cos(math.radians(phi)):.3f} (indukcyjny)",
                f"4) Te = 3|Ir|²(rr/s)·(P/2)/ωe = {Te:.2f} Nm",
                f"   Pm = Te·ωr = {Te * rpm2rad(n) / 1e3:.2f} kW",
                f"5) Tmax (3.11) = {Tmax:.1f} Nm (sm = {sm:.3f})",
                f"   Tmax (schemat T) = {rT['Te'][iTm]:.1f} Nm",
                f"   Tmax/Te = {Tmax / Te:.2f}"]


class TabSTAB(ExampleTab):
    key = "STAB"
    PARAMS = [("TL1", "Obciążenie początkowe TL1 [Nm]", 0, 160, 60, 1),
              ("TL2", "Obciążenie po skoku TL2 [Nm]", 0, 250, 150, 1),
              ("ts", "Chwila skoku [s]", 0.1, 2.0, 0.5, 0.05),
              ("J", "Moment bezwładności J [kg·m²]", 0.02, 2.0, 0.3, 0.01)]

    def extra_controls(self, parent):
        ttk.Label(parent, text="Rodzaj obciążenia").pack(anchor="w", padx=6)
        self.ltype = tk.StringVar(value="stały (dźwig)")
        cb = ttk.Combobox(parent, textvariable=self.ltype, state="readonly",
                          values=["stały (dźwig)", "wentylatorowy (~ω²)"])
        cb.pack(fill=tk.X, padx=6)
        cb.bind("<<ComboboxSelected>>", lambda _e: self.refresh())

    def update_plot(self):
        m = dict(Vph=220 / SQ3, f=60.0, rs=0.29, Lls=1.38e-3, rr=0.16, Llr=0.717e-3, P=6)
        fan = getattr(self, "ltype", None) is not None and self.ltype.get().startswith("went")
        ws = rpm2rad(n_sync(m["f"], m["P"]))
        sm, Tmax = breakdown(**m)
        TL1, TL2, ts, J = self.p("TL1"), self.p("TL2"), self.p("ts"), self.p("J")

        def TL_of(w, TLx):
            return TLx * (w / ws) ** 2 if fan else TLx + 0.0 * w

        def Te_of(w):
            return float(te_37(1 - w / ws, **m))
        # punkt początkowy - przecięcie w obszarze stabilnym
        lo, hi = ws * (1 - sm), ws * 0.99999
        if Te_of(lo) - TL_of(lo, TL1) < 0:
            w0 = 0.0
        else:
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if Te_of(mid) - TL_of(mid, TL1) > 0:
                    lo = mid
                else:
                    hi = mid
            w0 = 0.5 * (lo + hi)
        dt, tend = 1e-3, 3.0
        t = np.arange(0, tend, dt)
        w = np.zeros_like(t)
        TE = np.zeros_like(t)
        TLt = np.zeros_like(t)
        w[0] = w0
        for k in range(len(t) - 1):
            TLx = TL1 if t[k] < ts else TL2
            f1 = lambda x: (Te_of(x) - TL_of(x, TLx)) / J
            a1 = f1(w[k]); a2 = f1(w[k] + dt / 2 * a1); a3 = f1(w[k] + dt / 2 * a2); a4 = f1(w[k] + dt * a3)
            w[k + 1] = max(w[k] + dt / 6 * (a1 + 2 * a2 + 2 * a3 + a4), 0.0)
            TE[k], TLt[k] = Te_of(w[k]), TL_of(w[k], TLx)
        TE[-1], TLt[-1] = TE[-2], TLt[-2]
        s = np.linspace(1, 0.0005, 800)
        ax = self.fig.add_subplot(2, 1, 1)
        wplot = ws * (1 - s)
        ax.plot(s, te_37(s, **m), C1, lw=2, label="silnik Te(s)")
        ax.plot(s, TL_of(wplot, TL1), "k--", label="obciążenie TL1")
        ax.plot(s, TL_of(wplot, TL2), "--", color=C2, label="obciążenie TL2")
        ax.axvspan(1, sm, color="#f8d7da", alpha=0.5, lw=0)
        ax.axvspan(sm, 0, color="#d4edda", alpha=0.5, lw=0)
        ax.axvline(sm, color="gray", ls=":")
        ax.text(min(1, sm + (1 - sm) / 2), Tmax * 1.05, "NIESTABILNY", ha="center", fontsize=9, color=C2)
        ax.text(sm / 2, Tmax * 1.05, "STABILNY", ha="center", fontsize=9, color=C3)
        diff = te_37(s, **m) - TL_of(wplot, TL1)
        idx = np.where(np.diff(np.sign(diff)))[0]
        for i in idx:
            stable = s[i] < sm
            ax.plot([s[i]], [te_37(s[i], **m)], "o", ms=9, color=C3 if stable else C2)
            ax.annotate("A (stabilny)" if stable else "B (niestabilny)", (s[i], te_37(s[i], **m)),
                        xytext=(5, 8), textcoords="offset points", fontsize=8)
        ax.set_xlim(1, 0)
        ax.set_ylim(0, max(Tmax, TL2) * 1.2)
        ax.set_xlabel("Poślizg s")
        ax.set_ylabel("Moment [Nm]")
        ax.set_title(f"Rys. 3.14 - Tmax = {Tmax:.1f} Nm przy sm = {sm:.3f}")
        ax.legend(fontsize=8, loc="center left")
        ax.grid(alpha=0.3)
        ax2 = self.fig.add_subplot(2, 2, 3)
        ax2.plot(t, w * 60 / (2 * PI), C1, lw=2)
        ax2.axvline(ts, color="gray", ls=":")
        ax2.set_xlabel("Czas [s]")
        ax2.set_ylabel("Prędkość [obr/min]")
        ax2.set_title("Symulacja: J dω/dt = Te - TL")
        ax2.set_ylim(0, n_sync(m["f"], m["P"]) * 1.05)
        ax2.grid(alpha=0.3)
        ax3 = self.fig.add_subplot(2, 2, 4)
        ax3.plot(t, TE, C1, label="Te")
        ax3.plot(t, TLt, "k--", label="TL")
        ax3.set_xlabel("Czas [s]")
        ax3.set_ylabel("Moment [Nm]")
        ax3.legend(fontsize=8)
        ax3.grid(alpha=0.3)
        n_end = w[-1] * 60 / (2 * PI)
        lines = ["Silnik z Tabeli 3.2 (220 V, 60 Hz, 6p)",
                 f"Tmax = {Tmax:.1f} Nm, sm = {sm:.3f}",
                 f"Moment rozruchowy Te(s=1) = {float(te_37(1.0, **m)):.1f} Nm",
                 f"Prędkość przed skokiem: {w0 * 60 / (2 * PI):.1f} obr/min",
                 f"Prędkość na końcu: {n_end:.1f} obr/min"]
        if n_end < 1:
            lines.append(">>> UTYK! Obciążenie przekroczyło moment krytyczny.")
        else:
            lines.append(">>> Stabilnie - silnik znalazł nowy punkt pracy.")
        return lines


class TabPAR(ExampleTab):
    key = "PAR"
    PARAMS = [("A5", "Amplituda 5. harmonicznej A5 [%]", 0, 40, 8, 0.5),
              ("A7", "Amplituda 7. harmonicznej A7 [%]", 0, 40, 15, 0.5),
              ("TL", "Obciążenie stałe TL [Nm]", 0, 150, 45, 1),
              ("J", "J [kg·m²]", 0.05, 1.0, 0.2, 0.01)]

    def update_plot(self):
        base = dict(Vph=220 / SQ3, f=60.0, rs=0.29, Lls=1.38e-3, rr=0.16, Llr=0.717e-3, P=6)
        A5, A7, TL, J = self.p("A5") / 100, self.p("A7") / 100, self.p("TL"), self.p("J")
        ns = n_sync(base["f"], base["P"])
        ws = rpm2rad(ns)
        s = np.linspace(1.4, 0.0005, 2000)
        T1, T5, T7 = te_sweep_parasitic(s, base, A5, A7)
        ax = self.fig.add_subplot(2, 1, 1)
        ax.plot(s, T1, C1, lw=1.2, label="podstawowa")
        ax.plot(s, T5, C4, lw=1.2, label="5. harm. (ν=-5)")
        ax.plot(s, T7, C3, lw=1.2, label="7. harm. (ν=7)")
        ax.plot(s, T1 + T5 + T7, "k", lw=2.2, label="suma")
        ax.axhline(TL, color=C2, ls="--", label="obciążenie")
        for sv, lab in ((1.2, "s=1,2"), (6 / 7, "s=0,857")):
            ax.axvline(sv, color="gray", ls=":")
            ax.text(sv, ax.get_ylim()[1] * 0.9, lab, fontsize=8, ha="center")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlim(1.4, 0)
        ax.set_xlabel("Poślizg s")
        ax.set_ylabel("Moment [Nm]")
        ax.set_title("Rys. 3.15 - momenty pasożytnicze")
        ax.legend(fontsize=8, ncol=3)
        ax.grid(alpha=0.3)

        def Tt(w):
            ss = 1 - w / ws
            a, b, c = te_sweep_parasitic(np.array([ss]), base, A5, A7)
            return float(a[0] + b[0] + c[0])
        dt, tend = 2e-3, 4.0
        t = np.arange(0, tend, dt)
        w = np.zeros_like(t)
        for k in range(len(t) - 1):
            acc = (Tt(w[k]) - TL) / J
            w[k + 1] = max(w[k] + dt * acc, 0.0)
        ax2 = self.fig.add_subplot(2, 1, 2)
        ax2.plot(t, w * 60 / (2 * PI), C1, lw=2)
        ax2.axhline(ns / 7, color=C3, ls=":", label=f"n_s/7 = {ns / 7:.0f} obr/min")
        ax2.axhline(ns, color="gray", ls=":", label=f"n_s = {ns:.0f} obr/min")
        ax2.set_xlabel("Czas [s]")
        ax2.set_ylabel("Prędkość [obr/min]")
        ax2.set_title("Rozruch z obciążeniem stałym")
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.3)
        n_end = w[-1] * 60 / (2 * PI)
        tot = T1 + T5 + T7
        msk = (s > 0.75) & (s < 0.86)
        dip = tot[msk].min()
        lines = ["Prędkości synchroniczne harmonicznych:",
                 f"  5. (ν=-5): -n_s/5 = {-ns / 5:.0f} obr/min -> s5 = 1-1/(-5) = 1,2",
                 f"  7. (ν=7):  n_s/7 = {ns / 7:.0f} obr/min -> s7 = 1-1/7 = 0,857",
                 f"Minimum momentu w 'dołku' (s≈0,8): {dip:.1f} Nm",
                 f"Moment obciążenia: {TL:.1f} Nm",
                 f"Prędkość końcowa: {n_end:.0f} obr/min"]
        if n_end < ns * 0.5 and n_end > 1:
            lines.append(">>> PEŁZANIE (crawling) na ~n_s/7!")
        elif n_end <= 1:
            lines.append(">>> Silnik nie ruszył (TL > T rozruchowy)")
        else:
            lines.append(">>> Rozruch poprawny.")
        return lines


class TabE38(ExampleTab):
    key = "E38"
    PARAMS = [("N", "Liczba przewodów w żłobku Nsl", 1, 60, 20, 1),
              ("b", "Szerokość żłobka b [mm]", 2, 20, 8, 0.5),
              ("h1", "Wysokość z przewodami h1 [mm]", 2, 50, 20, 0.5),
              ("h2", "Wysokość pusta h2 [mm]", 0, 30, 3, 0.5),
              ("lst", "Długość pakietu lst [mm]", 20, 400, 150, 5),
              ("I", "Prąd przewodu I [A]", 1, 50, 10, 1),
              ("Ne", "Zwoje cewki (czoła) N", 1, 100, 20, 1),
              ("tp", "Podziałka cewki τp [mm]", 20, 300, 100, 5),
              ("As", "Przekrój wiązki As [mm²]", 20, 2000, 200, 10)]

    def update_plot(self):
        N, b, h1, h2, lst, I = (self.p(k) for k in ("N", "b", "h1", "h2", "lst", "I"))
        b, h1, h2, lst = b * 1e-3, h1 * 1e-3, h2 * 1e-3, lst * 1e-3
        y = np.linspace(0, h1 + h2, 600)
        B = np.where(y < h1, MU0 * N * I * y / (h1 * b), MU0 * N * I / b)
        W = lst * b / (2 * MU0) * _trapz(B ** 2, y)
        L_num = 2 * W / I ** 2
        L_f = MU0 * N ** 2 * lst * (h1 / (3 * b) + h2 / b)
        Ne, tp, As = self.p("Ne"), self.p("tp") * 1e-3, self.p("As") * 1e-6
        arg = tp ** 2 * PI / (4 * As)
        Le = MU0 * Ne ** 2 * tp / 8 * math.log(arg) if arg > 1 else float("nan")

        ax = self.fig.add_subplot(2, 2, 1)
        bm, hm = b * 1e3, (h1 + h2) * 1e3
        ax.fill([-bm, 2 * bm, 2 * bm, bm, bm, 0, 0, -bm], [-3, -3, hm + 3, hm + 3, 0, 0, hm + 3, hm + 3], color="#b0b7c3")
        ax.add_patch(matplotlib.patches.Rectangle((0, 0), bm, h1 * 1e3, color="#f2c48d"))
        ncol = max(1, int(round(math.sqrt(N * bm / max(h1 * 1e3, 1e-3)))))
        nrow = int(math.ceil(N / ncol))
        k = 0
        for r in range(nrow):
            for c in range(ncol):
                if k >= N:
                    break
                ax.plot((c + 0.5) * bm / ncol, (r + 0.5) * h1 * 1e3 / nrow, "o", color="#b5651d",
                        ms=max(2, 60 / max(nrow, ncol)))
                k += 1
        for yy in np.linspace(h1 * 1e3 * 0.2, hm * 0.95, 5):
            ax.annotate("", xy=(bm * 0.98, yy), xytext=(bm * 0.02, yy),
                        arrowprops=dict(arrowstyle="->", color=C1, lw=1))
        ax.text(bm / 2, hm + 4, "szczelina", ha="center", fontsize=8)
        ax.set_aspect("equal")
        ax.set_xlim(-bm, 2 * bm)
        ax.set_ylim(-4, hm + 8)
        ax.set_title("Przekrój żłobka (strzałki = pole rozproszenia)", fontsize=9)
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax2 = self.fig.add_subplot(2, 2, 2)
        ax2.plot(B * 1e3, y * 1e3, C1, lw=2)
        ax2.axhline(h1 * 1e3, color="gray", ls=":")
        ax2.fill_betweenx(y * 1e3, 0, B * 1e3, alpha=0.2)
        ax2.set_xlabel("B(y) [mT]")
        ax2.set_ylabel("y [mm]")
        ax2.set_title("Prawo Ampère'a: B·b = μ0·(objęty prąd)", fontsize=9)
        ax2.grid(alpha=0.3)
        ax3 = self.fig.add_subplot(2, 1, 2)
        hh = np.linspace(0, 30e-3, 100)
        ax3.plot(hh * 1e3, MU0 * N ** 2 * lst * (h1 / (3 * b) + hh / b) * 1e6, C1, lw=2, label="L w funkcji h2 (h1 stałe)")
        ax3.plot(hh * 1e3, MU0 * N ** 2 * lst * (np.maximum(hh, 1e-4) / (3 * b) + h2 / b) * 1e6, C3, lw=2, label="L w funkcji h1 (h2 stałe)")
        ax3.plot([h2 * 1e3], [L_f * 1e6], "o", color=C1)
        ax3.plot([h1 * 1e3], [L_f * 1e6], "o", color=C3)
        ax3.set_xlabel("wysokość [mm]")
        ax3.set_ylabel("L_sl [µH]")
        ax3.set_title("Pusta część żłobka 'kosztuje' 3 razy więcej niż wypełniona")
        ax3.legend(fontsize=8)
        ax3.grid(alpha=0.3)
        return [f"B max = μ0·Nsl·I/b = {MU0 * N * I / b * 1e3:.2f} mT",
                f"Energia W = {W * 1e3:.4f} mJ",
                f"L (numerycznie 2W/I²) = {L_num * 1e6:.3f} µH",
                "L (wzór) = μ0 Nsl² lst (h1/3b + h2/b)",
                f"         = {L_f * 1e6:.3f} µH",
                f"  udział h1/3b = {h1 / (3 * b):.3f}",
                f"  udział h2/b  = {h2 / b:.3f}",
                "",
                "Połączenia czołowe (Hanselman):",
                f"ln(τp²π/(4As)) = {math.log(arg) if arg > 1 else float('nan'):.3f}",
                f"L_e = μ0 N² τp/8 · ln(...) = {Le * 1e6:.2f} µH"]


class TabCIRC(ExampleTab):
    key = "CIRC"
    PARAMS = [("VLL", "Napięcie przewodowe VLL [V]", 50, 600, 220, 1),
              ("f", "Częstotliwość f [Hz]", 10, 100, 60, 1),
              ("P", "Liczba biegunów P", 2, 12, 6, 2),
              ("rs", "rs [Ω] (0 = wykres idealny)", 0.0, 1.0, 0.0, 0.01),
              ("Lls", "Lls [mH]", 0.1, 10, 1.38, 0.01),
              ("rr", "rr [Ω]", 0.01, 2.0, 0.16, 0.01),
              ("Llr", "Llr [mH]", 0.1, 10, 0.72, 0.01),
              ("Lm", "Lm [mH]", 5, 300, 38, 0.5),
              ("sop", "Poślizg punktu pracy P", 0.002, 1.0, 0.05, 0.001)]

    def update_plot(self):
        m = motor_from(self)
        we = 2 * PI * m["f"]
        V = m["Vph"]
        th = np.linspace(-PI / 2 + 1e-4, PI / 2 - 1e-4, 4000)
        sa = np.tan(th) * 3
        Is_all = im_T(sa, **m)["Is"]
        X, Y = -np.imag(Is_all), np.real(Is_all)
        pt = lambda s_: complex(im_T(s_, **m)["Is"])
        xy = lambda z: (-z.imag, z.real)
        N, Q, R = xy(pt(1e-9)), xy(pt(1.0)), xy(pt(1e9))
        sp = np.logspace(-4, 0.5, 3000)
        rp = im_T(sp, **m)
        ang = np.arctan2(-np.imag(rp["Is"]), np.real(rp["Is"]))   # kąt φ od napięcia
        i_r = np.argmin(ang)
        sr_num = sp[i_r]
        i_m = np.argmax(rp["Te"])
        sm_num = sp[i_m]
        sop = self.p("sop")
        Pp = xy(pt(sop))

        def line_y(A, Bp, x):
            return A[1] + (Bp[1] - A[1]) * (x - A[0]) / (Bp[0] - A[0])
        yB = line_y(N, R, Pp[0])
        yD = line_y(N, Q, Pp[0])

        ax = self.fig.add_subplot(1, 2, 1)
        ax.plot(X, Y, color="gray", lw=1)
        ms = (sp <= 1)
        ax.plot(-np.imag(rp["Is"][ms]), np.real(rp["Is"][ms]), C1, lw=2.5, label="0 < s ≤ 1 (silnik)")
        ax.plot([0, 0], [0, max(Y) * 1.1], "k", lw=1)
        ax.text(0, max(Y) * 1.12, "Vs", ha="center")
        for P_, lab in ((N, "N (s=0)"), (Q, "Q (s=1)"), (R, "R (s=∞)")):
            ax.plot(*P_, "ko")
            ax.annotate(lab, P_, xytext=(4, 4), textcoords="offset points", fontsize=8)
        ax.plot([N[0], Q[0]], [N[1], Q[1]], C3, lw=1, label="linia zerowej mocy NQ")
        ax.plot([N[0], R[0]], [N[1], R[1]], C4, lw=1, label="linia zerowego momentu NR")
        Tg = xy(rp["Is"][i_r])
        ax.plot([0, Tg[0] * 1.3], [0, Tg[1] * 1.3], "k--", lw=0.8)
        ax.plot(*Tg, "s", color=C2)
        ax.annotate(f"s_r={sr_num:.3f}", Tg, xytext=(5, -12), textcoords="offset points", fontsize=8, color=C2)
        Mm = xy(rp["Is"][i_m])
        ax.plot(*Mm, "^", color=C2)
        ax.annotate(f"s_m={sm_num:.3f}", Mm, xytext=(5, 5), textcoords="offset points", fontsize=8, color=C2)
        ax.plot(*Pp, "o", color="m", ms=8)
        ax.plot([Pp[0], Pp[0]], [yB, Pp[1]], "m", lw=2)
        ax.plot([Pp[0]], [yD], "mx", ms=8)
        ax.annotate("P", Pp, xytext=(4, 4), textcoords="offset points", color="m")
        ax.annotate("D", (Pp[0], yD), xytext=(4, 0), textcoords="offset points", color="m", fontsize=8)
        ax.annotate("B", (Pp[0], yB), xytext=(4, -10), textcoords="offset points", color="m", fontsize=8)
        ax.set_aspect("equal")
        ax.set_xlabel("składowa bierna (indukcyjna) [A]")
        ax.set_ylabel("składowa czynna [A]")
        ax.set_title("Wykres kołowy Heylanda")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(alpha=0.3)
        ax.set_xlim(min(X.min(), 0) - 5, X.max() * 1.05)
        ax.set_ylim(min(Y.min(), 0) - 5, Y.max() * 1.2)

        # moment z koła dla 0<s<=1
        sc = np.linspace(1, 0.002, 400)
        Ic = im_T(sc, **m)["Is"]
        xc, yc = -np.imag(Ic), np.real(Ic)
        Tc = 3 * V * (yc - line_y(N, R, xc)) * (m["P"] / 2) / we
        ax2 = self.fig.add_subplot(1, 2, 2)
        ax2.plot(sc, Tc, C1, lw=2.5, label="z wykresu kołowego (3·Vs·PB)")
        ax2.plot(sc, im_T(sc, **m)["Te"], "k:", lw=2, label="schemat T (dokładny)")
        ax2.plot(sc, te_37(sc, **no_lm(m)), C2, ls="--", label="wzór (3.7)")
        ax2.set_xlim(1, 0)
        ax2.set_xlabel("Poślizg s")
        ax2.set_ylabel("Moment [Nm]")
        ax2.set_title("Ćw. 3.12c - porównanie")
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.3)

        sig_e = sigma_exact(m["Lls"], m["Llr"], m["Lm"])
        sig_b = sigma_book(m["Lls"], m["Llr"], m["Lm"])
        Ls, Lr = m["Lls"] + m["Lm"], m["Llr"] + m["Lm"]
        taur = Lr / m["rr"]
        Is0 = V / (we * Ls)
        sm_a = 1 / (taur * we * sig_b)
        sr_a = 1 / (taur * we * math.sqrt(sig_b))
        pf = lambda s_: math.cos(cmath.phase(pt(s_)))
        Te_c = 3 * V * (Pp[1] - yB) * (m["P"] / 2) / we
        Pm_c = 3 * V * (Pp[1] - yD)
        # ćw. 3.11: DB/NB stałe
        def slope(s_):
            P_ = xy(pt(s_)); yb = line_y(N, R, P_[0]); yd = line_y(N, Q, P_[0])
            return (yd - yb) / (P_[0] - N[0])
        return [f"σ = 1 - Lm²/(LsLr) = {sig_e:.4f}",
                f"σ ≈ LsLr/Lm² - 1 = {sig_b:.4f} (książka)",
                f"τr = Lr/rr = {taur:.4f} s",
                f"Is0 = Vs/(ωe Ls) = {Is0:.2f} A",
                f"|Is(s=1)| = {abs(pt(1.0)):.1f} A  (Is0/σ = {Is0 / sig_b:.1f})",
                f"cos φ_min = (1-σ)/(1+σ) = {(1 - sig_b) / (1 + sig_b):.3f}",
                f"   numerycznie = {pf(sr_num):.3f}",
                f"s_r ≈ 1/(τr ωe √σ) = {sr_a:.4f} (num {sr_num:.4f})",
                f"s_m ≈ 1/(τr ωe σ) = {sm_a:.4f} (num {sm_num:.4f})",
                f"|Is(s_r)| ≈ Is0/√σ = {Is0 / math.sqrt(sig_b):.2f} A (num {abs(pt(sr_num)):.2f})",
                "Ćw. 3.12b - cos φ:",
                f"  s=0: {pf(1e-9):.3f}  s=1: {pf(1.0):.3f}  s=∞: {pf(1e9):.3f}",
                f"  s=s_m: {pf(sm_num):.3f}  s=s_r: {pf(sr_num):.3f}",
                f"Punkt P (s={sop:.3f}):",
                f"  Te z koła = {Te_c:.1f} Nm, Pm = {Pm_c / 1e3:.2f} kW",
                f"  Te dokładnie = {float(im_T(sop, **m)['Te']):.1f} Nm",
                f"Ćw. 3.11: DB/NB(s=0,02) = {slope(0.02):.4f}",
                f"          DB/NB(s=0,5)  = {slope(0.5):.4f} (stałe!)"]


class TabE313(ExampleTab):
    key = "E313"
    PARAMS = [("h", "Wysokość pręta h [mm]", 2, 60, 30, 0.5),
              ("g", "Przewodność γ [MS/m] (Cu 50-58, Al 30-35)", 5, 60, 50, 0.5),
              ("f", "Częstotliwość sieci f [Hz]", 10, 100, 60, 1),
              ("s", "Poślizg s", 0.001, 1.0, 1.0, 0.001)]

    def update_plot(self):
        h, g, f, s = self.p("h") * 1e-3, self.p("g") * 1e6, self.p("f"), self.p("s")
        fr = f * s
        d = float(skin_depth(g, fr))
        xi = h / d
        Kr, Kx = skin_KrKx(xi)
        Kr, Kx = float(Kr), float(Kx)
        ax = self.fig.add_subplot(2, 2, 1)
        xx = np.linspace(0.01, 6, 400)
        a, b = skin_KrKx(xx)
        ax.plot(xx, a, C2, lw=2, label="Kr = r_ac/r_dc")
        ax.plot(xx, b, C1, lw=2, label="Kx (indukcyjność)")
        ax.plot([xi], [Kr], "o", color=C2)
        ax.plot([xi], [Kx], "o", color=C1)
        ax.set_xlabel("ξ = h/δ")
        ax.set_title("Rys. 3.28 - współczynniki wypierania", fontsize=9)
        ax.set_xlim(0, 6)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax2 = self.fig.add_subplot(2, 2, 2)
        y = np.linspace(0, h, 300)
        k = (1 + 1j) / d
        J = h * k * np.cosh(k * y) / np.sinh(k * h)
        ax2.plot(np.abs(J), y * 1e3, C2, lw=2, label=f"s = {s:.3f}")
        ax2.axvline(1, color="k", ls="--", label="prąd stały (równomierny)")
        ax2.set_xlabel("|J(y)| / J_dc")
        ax2.set_ylabel("y [mm] (góra = szczelina)")
        ax2.set_title("Rys. 3.26b/3.27 - gęstość prądu w pręcie", fontsize=9)
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.3)
        ax3 = self.fig.add_subplot(2, 1, 2)
        sv = np.logspace(-3, 0, 300)
        kr, kx = skin_KrKx(h / skin_depth(g, f * sv))
        ax3.semilogx(sv, kr, C2, lw=2, label="r_ac/r_dc")
        ax3.semilogx(sv, kx, C1, lw=2, label="Kx")
        ax3.axvspan(0.01, 0.05, color="#d4edda", alpha=0.6, label="typowa praca")
        ax3.plot([s], [Kr], "o", color=C2)
        ax3.set_xlabel("Poślizg s (skala log.)")
        ax3.set_title("Rezystancja wirnika 'sama się przełącza' z poślizgiem", fontsize=9)
        ax3.legend(fontsize=8)
        ax3.grid(alpha=0.3, which="both")
        return [f"Częstotliwość w wirniku f2 = s·f = {fr:.3f} Hz",
                f"ω2 = 2π·f2 = {2 * PI * fr:.2f} rad/s",
                f"δ = sqrt(2/(μ0·γ·ω2)) = {d * 1e3:.2f} mm",
                f"ξ = h/δ = {h * 1e3:.1f}/{d * 1e3:.2f} = {xi:.3f}",
                f"Kr = r_ac/r_dc = {Kr:.3f}",
                f"Kx = {Kx:.3f}",
                "",
                f"Przy s = 0,03: Kr = {float(skin_KrKx(h / skin_depth(g, f * 0.03))[0]):.3f}",
                "Dla dużego ξ: Kr ≈ ξ, Kx ≈ 3/(2ξ)"]


class TabNEMA(ExampleTab):
    key = "NEMA"
    PARAMS = [("r1", "Klatka górna r1 [Ω]", 0.05, 2.0, 0.6, 0.01),
              ("L1", "Klatka górna L1 [mH]", 0.0, 3.0, 0.2, 0.05),
              ("r2", "Klatka dolna r2 [Ω]", 0.02, 1.0, 0.09, 0.005),
              ("L2", "Klatka dolna L2 [mH]", 0.2, 8.0, 2.5, 0.05),
              ("Lc", "Wspólne rozproszenie Lc [mH]", 0.0, 2.0, 0.3, 0.05)]

    def update_plot(self):
        st = dict(Vph=220 / SQ3, f=60.0, rs=0.29, Lls=1.38e-3, Lm=41e-3, P=6)
        f = st["f"]
        s = np.linspace(1, 0.001, 600)
        ns = n_sync(f, st["P"])
        sp = (1 - s) * 100
        models = {
            "A (owalny, małe rr)": s * 0 + (0.08 / _s(s) + 1j * 2 * PI * f * 0.55e-3),
            "B (pręt głęboki)": deep_bar_zr(s, f, 0.10, 0.35e-3, 0.9e-3, 2.4),
            "C (klatka podwójna)": double_cage_zr(s, f, 0.75, 0.15e-3, 0.10, 2.6e-3, 0.25e-3),
            "D (duże rr, brąz)": 0.55 / _s(s) + 1j * 2 * PI * f * 0.7e-3,
        }
        zr_user = double_cage_zr(s, f, self.p("r1"), self.p("L1") * 1e-3, self.p("r2"), self.p("L2") * 1e-3, self.p("Lc") * 1e-3)
        ax = self.fig.add_subplot(2, 1, 1)
        ax3 = self.fig.add_subplot(2, 2, 4)
        lines = ["klasa          T_rozr  T_max  I_rozr  s(60Nm)"]
        cols = [C1, C3, C4, C2]
        for (name, zr), c in zip(models.items(), cols):
            r = im_T(s, zr=zr, rr=0, Llr=0, **st)
            ax.plot(sp, r["Te"], color=c, lw=1.6, label=name)
            ax3.plot(sp, np.abs(r["Is"]), color=c, lw=1.2)
            i60 = np.where(r["Te"] >= 60)[0]
            s60 = s[i60[-1]] if len(i60) else float("nan")
            lines.append(f"{name[:14]:14s} {r['Te'][0]:6.0f} {r['Te'].max():6.0f} {abs(r['Is'][0]):6.0f}  {s60:6.3f}")
        ru = im_T(s, zr=zr_user, rr=0, Llr=0, **st)
        ax.plot(sp, ru["Te"], "k", lw=2.6, label="Twoja klatka podwójna")
        ax3.plot(sp, np.abs(ru["Is"]), "k", lw=2)
        ax.set_xlabel("Prędkość [% n_s]")
        ax.set_ylabel("Moment [Nm]")
        ax.set_title("Rys. 3.31 - klasy NEMA (stojan jak w Tabeli 3.2)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax3.set_xlabel("Prędkość [% n_s]")
        ax3.set_ylabel("|Is| [A]")
        ax3.set_title("Prąd stojana", fontsize=9)
        ax3.grid(alpha=0.3)
        ax2 = self.fig.add_subplot(2, 2, 3)
        rr_eff = np.real(zr_user) * s
        L_eff = np.imag(zr_user) / (2 * PI * f) * 1e3
        ax2.plot(s, rr_eff, C2, lw=2, label="rr_eff(s) [Ω]")
        ax2b = ax2.twinx()
        ax2b.plot(s, L_eff, C1, lw=2, label="Llr_eff(s) [mH]")
        ax2.set_xlim(1, 0)
        ax2.set_xlabel("Poślizg s")
        ax2.set_ylabel("rr_eff [Ω]", color=C2)
        ax2b.set_ylabel("Llr_eff [mH]", color=C1)
        ax2.set_title("Twoja klatka: efektywne parametry", fontsize=9)
        ax2.grid(alpha=0.3)
        lines += ["", f"Twoja klatka: T_rozr = {ru['Te'][0]:.0f} Nm, Tmax = {ru['Te'].max():.0f} Nm",
                  f"  rr_eff(s=1) = {rr_eff[0]:.3f} Ω",
                  f"  rr_eff(s→0) = {rr_eff[-1]:.3f} Ω",
                  f"  I_rozr = {abs(ru['Is'][0]):.0f} A",
                  f"n_s = {ns:.0f} obr/min"]
        return lines


class TabSTART(ExampleTab):
    key = "START"
    PARAMS = [("VLL", "Napięcie przewodowe [V]", 100, 440, 220, 1),
              ("f", "Częstotliwość [Hz]", 50, 60, 60, 10),
              ("P", "Liczba biegunów P", 2, 8, 4, 2),
              ("rs", "rs [Ω]", 0.1, 5, 0.9, 0.05),
              ("rr", "rr [Ω]", 0.1, 5, 0.8, 0.05),
              ("Lls", "Lls [mH]", 1, 20, 4.5, 0.1),
              ("Llr", "Llr [mH]", 1, 20, 4.5, 0.1),
              ("Lm", "Lm [mH]", 30, 400, 120, 1),
              ("J", "J [kg·m²]", 0.002, 0.1, 0.012, 0.001),
              ("TL", "Obciążenie wentylator. przy n_s [Nm]", 0, 15, 5, 0.5),
              ("V0", "Softstart: napięcie początkowe [%]", 20, 100, 100, 1),
              ("Tr", "Softstart: czas narastania [s]", 0.05, 2.0, 0.4, 0.05),
              ("tend", "Czas symulacji [s]", 0.2, 2.0, 0.6, 0.05)]

    def update_plot(self):
        prm = dict(VLL=self.p("VLL"), f=self.p("f"), P=int(round(self.p("P"))), rs=self.p("rs"), rr=self.p("rr"),
                   Lls=self.p("Lls") * 1e-3, Llr=self.p("Llr") * 1e-3, Lm=self.p("Lm") * 1e-3, J=self.p("J"), B=0.0)
        ws = rpm2rad(n_sync(prm["f"], prm["P"]))
        TLs = self.p("TL")
        TLf = lambda t, w: TLs * (w / ws) ** 2
        V0, Tr, tend = self.p("V0") / 100, self.p("Tr"), self.p("tend")
        dt = 1e-4 if tend <= 1.0 else 2e-4
        t, ia, w, Te = im_dynamic(prm, tend, dt, TLf)
        soft = V0 < 0.999
        if soft:
            vf = lambda tt: min(1.0, V0 + (1 - V0) * tt / Tr)
            t2, ia2, w2, Te2 = im_dynamic(prm, tend, dt, TLf, vf)
        ax = self.fig.add_subplot(2, 2, (1, 2))
        ax.plot(t, ia, color="gray" if soft else C1, lw=0.8, label="rozruch bezpośredni")
        if soft:
            ax.plot(t2, ia2, C1, lw=0.8, label="softstart")
        ax.set_xlabel("Czas [s]")
        ax.set_ylabel("Prąd fazy a [A]")
        ax.set_title("Rys. 3.32 - prąd przy rozruchu")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax2 = self.fig.add_subplot(2, 2, 3)
        ax2.plot(t, w * 60 / (2 * PI), color="gray" if soft else C1, lw=1.5, label="bezpośredni")
        if soft:
            ax2.plot(t2, w2 * 60 / (2 * PI), C1, lw=1.5, label="softstart")
        ax2.set_xlabel("Czas [s]")
        ax2.set_ylabel("Prędkość [obr/min]")
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.3)
        ax3 = self.fig.add_subplot(2, 2, 4)
        ax3.plot(w * 60 / (2 * PI), Te, color="gray" if soft else C1, lw=0.7, label="dynamiczny (DOL)")
        if soft:
            ax3.plot(w2 * 60 / (2 * PI), Te2, C1, lw=0.7, label="dynamiczny (soft)")
        ss = np.linspace(1, 0.001, 300)
        stat = im_T(ss, prm["VLL"] / SQ3, prm["f"], prm["rs"], prm["Lls"], prm["rr"], prm["Llr"], prm["Lm"], prm["P"])
        ax3.plot((1 - ss) * ws * 60 / (2 * PI), stat["Te"], "k--", lw=1.5, label="statyczna")
        nn = np.linspace(0, ws, 50)
        ax3.plot(nn * 60 / (2 * PI), TLs * (nn / ws) ** 2, C2, lw=1, label="obciążenie")
        ax3.set_xlabel("Prędkość [obr/min]")
        ax3.set_ylabel("Moment [Nm]")
        ax3.legend(fontsize=7)
        ax3.grid(alpha=0.3)
        n_cyc = int(round(prm["f"] * 0.1 / 1)) or 1
        tail = slice(-int(1 / prm["f"] / dt * n_cyc), None)
        ilock = abs(complex(stat["Is"][0]))
        lines = [f"n_s = {ws * 60 / (2 * PI):.0f} obr/min",
                 f"Prąd zwarcia (s=1, ustalony): {ilock:.1f} A rms",
                 f"  = {ilock * SQ2:.1f} A szczytowo",
                 f"DOL: max |ia| = {np.abs(ia).max():.1f} A",
                 f"DOL: Te max = {Te.max():.1f} Nm, min = {Te.min():.1f} Nm",
                 f"DOL: prąd końcowy ≈ {np.abs(ia[tail]).max() / SQ2:.2f} A rms",
                 f"DOL: n końcowe = {w[-1] * 60 / (2 * PI):.0f} obr/min"]
        k95 = np.where(w > 0.9 * w[-1])[0]
        if len(k95):
            lines.append(f"DOL: czas do 90% n_końc = {t[k95[0]]:.3f} s")
        if soft:
            lines += ["", f"Soft: max |ia| = {np.abs(ia2).max():.1f} A",
                      f"   ({np.abs(ia2).max() / np.abs(ia).max() * 100:.0f} % prądu DOL)",
                      f"Soft: n końcowe = {w2[-1] * 60 / (2 * PI):.0f} obr/min"]
            k2 = np.where(w2 > 0.9 * w2[-1])[0]
            if len(k2):
                lines.append(f"Soft: czas do 90% = {t2[k2[0]]:.3f} s")
        return lines


class TabTYR(ExampleTab):
    key = "TYR"
    PARAMS = [("a", "Kąt załączenia α [°]", 0, 180, 60, 1),
              ("V", "Napięcie fazowe sieci [V rms]", 50, 400, 127, 1),
              ("f", "Częstotliwość [Hz]", 50, 60, 60, 10)]

    def update_plot(self):
        a = math.radians(self.p("a"))
        V, f = self.p("V"), self.p("f")
        Vm = SQ2 * V
        N = 4000
        th = np.linspace(0, 4 * PI, N, endpoint=False)   # 2 okresy
        v = Vm * np.sin(th)
        ph = th % PI
        vo = np.where(ph >= a, v, 0.0)
        ax = self.fig.add_subplot(2, 1, 1)
        ax.plot(np.degrees(th), v, color="gray", lw=1, label="napięcie sieci")
        ax.fill_between(np.degrees(th), 0, vo, color=C1, alpha=0.4)
        ax.plot(np.degrees(th), vo, C1, lw=2, label="napięcie na silniku")
        for k in range(4):
            ax.axvline(math.degrees(a + k * PI), color=C2, ls=":", lw=1)
        ax.set_xlabel("kąt ωt [°]")
        ax.set_ylabel("Napięcie [V]")
        ax.set_title(f"Rys. 3.33 - przebieg przy α = {self.p('a'):.0f}° (czerwone - impulsy bramkowe)")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.3)
        ax2 = self.fig.add_subplot(2, 2, 3)
        aa = np.linspace(0, PI, 300)
        ax2.plot(np.degrees(aa), thyristor_vrms(aa) * 100, C1, lw=2)
        ax2.plot([self.p("a")], [thyristor_vrms(a) * 100], "o", color=C2)
        ax2.set_xlabel("α [°]")
        ax2.set_ylabel("Vrms / V_sieci [%]")
        ax2.set_title("Zadanie 3.6: Vrms(α)", fontsize=9)
        ax2.grid(alpha=0.3)
        spec = np.abs(np.fft.rfft(vo)) / N * 2
        ax3 = self.fig.add_subplot(2, 2, 4)
        hk = np.arange(1, 20, 2)
        amps = spec[(2 * hk)]
        ax3.bar(hk, amps / SQ2, color=C1)
        ax3.set_xlabel("Rząd harmonicznej")
        ax3.set_ylabel("V rms")
        ax3.set_xticks(hk)
        ax3.set_title("Widmo napięcia wyjściowego", fontsize=9)
        ax3.grid(alpha=0.3)
        Vrms = V * float(thyristor_vrms(a))
        V1 = amps[0] / SQ2
        thd = math.sqrt(max(Vrms ** 2 - V1 ** 2, 0)) / V1 * 100 if V1 > 1e-6 else float("nan")
        return [f"α = {self.p('a'):.0f}° = {a:.4f} rad",
                f"opóźnienie zapłonu = α/(2π f) = {a / (2 * PI * f) * 1e3:.2f} ms",
                "Vrms = V·sqrt(1 - α/π + sin2α/(2π))",
                f"     = {V:.0f}·sqrt({1 - a / PI:.4f} + {math.sin(2 * a) / (2 * PI):.4f})",
                f"     = {Vrms:.2f} V ({Vrms / V * 100:.1f} %)",
                f"Harmoniczna podstawowa V1 = {V1:.2f} V",
                f"THD napięcia = {thd:.1f} %",
                f"Moment silnika ~ V1² -> {(V1 / V) ** 2 * 100:.1f} % momentu"]


class TabNAP(ExampleTab):
    key = "NAP"
    PARAMS = [("k", "Obciążenie wentylatora przy n_s [Nm]", 10, 200, 90, 1),
              ("Vmin", "Najniższe napięcie [% Vn]", 20, 95, 50, 1),
              ("nc", "Liczba krzywych", 2, 8, 5, 1)]

    def update_plot(self):
        m = dict(Vph=220 / SQ3, f=60.0, rs=0.29, Lls=1.38e-3, rr=0.16, Llr=0.717e-3, P=6)
        k, Vmin, nc = self.p("k"), self.p("Vmin") / 100, int(self.p("nc"))
        ns = n_sync(m["f"], m["P"])
        s = np.linspace(1, 0.0005, 800)
        nn = ns * (1 - s)
        ax = self.fig.add_subplot(1, 2, 1)
        ax.plot(nn, k * (1 - s) ** 2, "k--", lw=2, label="wentylator TL = k·ω²")
        sm, _ = breakdown(**m)
        rows = []
        for i, vr in enumerate(np.linspace(1, Vmin, nc)):
            mm = dict(m, Vph=m["Vph"] * vr)
            T = te_37(s, **mm)
            c = matplotlib.cm.viridis(i / max(nc - 1, 1))
            ax.plot(nn, T, color=c, lw=2, label=f"V = {vr * 100:.0f} %")
            g = lambda x: float(te_37(x, **mm) - k * (1 - x) ** 2)
            lo, hi = 1e-6, 1.0
            if g(hi) < 0:
                rows.append((vr, None, None, None))
                continue
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if g(mid) < 0:
                    lo = mid
                else:
                    hi = mid
            so = 0.5 * (lo + hi)
            To = k * (1 - so) ** 2
            ax.plot([ns * (1 - so)], [To], "o", color=c, ms=8, mec="k")
            Pag = To * rpm2rad(ns)
            rows.append((vr, so, ns * (1 - so), so * Pag))
        ax.set_xlabel("Prędkość [obr/min]")
        ax.set_ylabel("Moment [Nm]")
        ax.set_title("Rys. 3.35 - sterowanie napięciem (f = 60 Hz)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        ax2 = self.fig.add_subplot(2, 2, 2)
        ax3 = self.fig.add_subplot(2, 2, 4)
        ok = [r for r in rows if r[1] is not None]
        ax2.plot([r[0] * 100 for r in ok], [r[2] for r in ok], "o-", color=C1)
        ax2.set_xlabel("Napięcie [%]")
        ax2.set_ylabel("Prędkość [obr/min]")
        ax2.set_title("Prędkość w funkcji napięcia", fontsize=9)
        ax2.grid(alpha=0.3)
        ax3.plot([r[0] * 100 for r in ok], [r[3] for r in ok], "o-", color=C2)
        ax3.set_xlabel("Napięcie [%]")
        ax3.set_ylabel("Straty Cu wirnika [W]")
        ax3.set_title("Pcu2 = s·Pag - ciepło w wirniku", fontsize=9)
        ax3.grid(alpha=0.3)
        lines = ["V[%]   s       n[obr/min]  Pcu2[W]  obszar"]
        for vr, so, n, pc in rows:
            if so is None:
                lines.append(f"{vr * 100:4.0f}   brak punktu pracy")
            else:
                lines.append(f"{vr * 100:4.0f} {so:7.4f} {n:9.1f} {pc:8.0f}  {'stab.' if so < sm else 'NIESTAB.'}")
        if len(ok) > 1:
            lines.append(f"Zakres regulacji: {ok[-1][2]:.0f} - {ok[0][2]:.0f} obr/min")
        lines.append(f"sm = {sm:.3f} (nie zależy od V)")
        return lines


class TabVVVF(ExampleTab):
    key = "VVVF"
    PARAMS = [("V0", "Podbicie napięcia V0 [V]", 0, 40, 8, 0.5),
              ("TL", "Obciążenie stałe [Nm] (punkt x)", 0, 200, 140, 1),
              ("nref", "Prędkość zadana [obr/min]", 50, 1200, 600, 10),
              ("TLs", "Skok obciążenia w t=2 s [Nm]", 0, 150, 60, 1),
              ("Kp", "Kp regulatora prędkości", 0.05, 5, 1.0, 0.05),
              ("Ki", "Ki regulatora prędkości", 0, 30, 5, 0.5)]

    def update_plot(self):
        m = dict(rs=0.29, Lls=1.38e-3, rr=0.16, Llr=0.717e-3, P=6)
        Vn, fn = 220 / SQ3, 60.0
        V0 = self.p("V0")
        Vf = lambda f: np.minimum(V0 + (Vn - V0) * np.asarray(f) / fn, Vn)
        ax = self.fig.add_subplot(2, 2, (1, 2))
        lines = ["f[Hz]  V[V]   Tmax(boost)  Tmax(bez)"]
        for f, c in zip((12, 24, 36, 48, 60), (C1, C3, C4, "m", C2)):
            ns = n_sync(f, m["P"])
            s = np.linspace(1, 0.0005, 500)
            V = float(Vf(f))
            ax.plot(ns * (1 - s), te_37(s, V, f, **m), color=c, lw=2, label=f"{f} Hz, {V:.0f} V")
            ax.plot(ns * (1 - s), te_37(s, Vn * f / fn, f, **m), color=c, lw=1, ls=":")
            lines.append(f"{f:4d} {V:6.1f} {breakdown(V, f, **m)[1]:9.1f} {breakdown(Vn * f / fn, f, **m)[1]:9.1f}")
        TL = self.p("TL")
        ax.axhline(TL, color="k", ls="--", lw=1.5, label="obciążenie stałe (dźwig)")
        ax.set_xlabel("Prędkość [obr/min]")
        ax.set_ylabel("Moment [Nm]")
        ax.set_title("Rys. 3.36 - krzywe VVVF (linia ciągła: z podbiciem, kropki: czyste U/f)")
        ax.legend(fontsize=7, ncol=3)
        ax.set_xlim(0, 1250)
        ax.grid(alpha=0.3)
        ax2 = self.fig.add_subplot(2, 2, 3)
        ff = np.linspace(0, 80, 200)
        ax2.plot(ff, Vf(ff), C1, lw=2, label="z podbiciem (boost)")
        ax2.plot(ff, np.minimum(Vn * ff / fn, Vn), "k--", label="U/f = const")
        ax2.set_xlabel("f [Hz]")
        ax2.set_ylabel("Vs fazowe [V]")
        ax2.set_title("Rys. 3.37 - profil U/f", fontsize=9)
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.3)

        # regulator z poślizgiem (rys. 3.37 dół) - model quasi-statyczny
        p = m["P"] / 2
        J = 0.3
        Kp, Ki = self.p("Kp"), self.p("Ki")
        wref = rpm2rad(self.p("nref"))
        sm60, _ = breakdown(Vn, fn, **m)
        wsl_max = 0.8 * sm60 * 2 * PI * fn
        dt, tend = 1e-3, 4.0
        n = int(tend / dt)
        t = np.arange(n) * dt
        W = np.zeros(n); F = np.zeros(n); TE = np.zeros(n)
        wr, integ = 0.0, 0.0
        for k in range(n):
            e = wref - wr
            u = Kp * e + Ki * integ
            wsl = min(max(u, -wsl_max), wsl_max)
            if wsl == u:
                integ += e * dt                  # anti-windup: całkuj tylko bez nasycenia
            we = max(p * wr + wsl, 0.5)
            f = we / (2 * PI)
            V = float(Vf(f))
            X = we * (m["Lls"] + m["Llr"])
            rrs = m["rr"] * we / max(wsl, 1e-6) if wsl > 1e-6 else 1e9
            Te = 3 * p * V ** 2 * rrs / (we * ((m["rs"] + rrs) ** 2 + X ** 2))
            TLk = self.p("TLs") if t[k] >= 2.0 else 0.0
            wr = max(wr + dt * (Te - TLk) / J, 0.0)
            W[k], F[k], TE[k] = wr, f, Te
        ax3 = self.fig.add_subplot(2, 2, 4)
        ax3.plot(t, W * 60 / (2 * PI), C1, lw=2, label="n [obr/min]")
        ax3.axhline(self.p("nref"), color="gray", ls=":")
        ax3.set_xlabel("Czas [s]")
        ax3.set_ylabel("Prędkość [obr/min]", color=C1)
        ax3b = ax3.twinx()
        ax3b.plot(t, F, C2, lw=1, label="f [Hz]")
        ax3b.set_ylabel("f falownika [Hz]", color=C2)
        ax3.set_title("Regulator z poślizgiem: ωe = (P/2)ωr + ωsl", fontsize=9)
        ax3.grid(alpha=0.3)
        lines += ["", f"V/f = {Vn / fn:.3f} V/Hz (λs = {Vn / (2 * PI * fn):.3f} Wb)",
                  f"rs/(ωe·Ls): 60 Hz -> {0.29 / (2 * PI * 60 * 42.38e-3):.3f}, 6 Hz -> {0.29 / (2 * PI * 6 * 42.38e-3):.3f}",
                  f"Sym.: n końcowe = {W[-1] * 60 / (2 * PI):.1f} obr/min",
                  f"Sym.: f końcowe = {F[-1]:.2f} Hz",
                  f"Sym.: max ωsl = {wsl_max:.1f} rad/s"]
        return lines


# =============================================================================
# ZADANIA (PROBLEMS) 3.1 - 3.13
# =============================================================================

def _bars(ax, labels, vals, title, unit="kW"):
    cols = [C1 if v >= 0 else C2 for v in vals]
    ax.bar(range(len(vals)), np.abs(vals), color=cols, edgecolor="k")
    for i, v in enumerate(vals):
        ax.text(i, abs(v), f"{v:.3g} {unit}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3, axis="y")


def pr31(fig, p):
    ns = n_sync(p["f"], p["P"])
    s = (ns - p["n"]) / ns
    Pm = p["Pout"] + p["Pfw"]
    Pag = Pm / (1 - s)
    Pcu2 = s * Pag
    _bars(fig.add_subplot(1, 1, 1), ["P_ag", "P_cu2 = s·P_ag", "P_m (rozwinięta)", "tarcie+went.", "P_out (wał)"],
          [Pag, -Pcu2, Pm, -p["Pfw"], p["Pout"]], "Zadanie 3.1 - bilans mocy wirnika")
    return [f"n_s = {ns:.0f} obr/min, s = {s:.5f}",
            f"P_m = P_out + P_fw = {Pm:.2f} kW",
            f"P_ag = P_m/(1-s) = {Pag:.3f} kW",
            f"P_cu2 = s·P_ag = {Pcu2:.3f} kW",
            f"Te = P_m/ωr = {Pm * 1e3 / rpm2rad(p['n']):.1f} Nm"]


def pr32(fig, p):
    Pin = p["S"] * p["pf"]
    Pag = Pin - p["Pst"]
    ws = rpm2rad(p["ns"])
    Te = Pag * 1e3 / ws
    s = p["sl"] / 100
    Pm = Pag * (1 - s)
    _bars(fig.add_subplot(1, 2, 1), ["S [kVA]", "P_in", "straty stojana", "P_ag", "P_cu2", "P_m"],
          [p["S"], Pin, -p["Pst"], Pag, -s * Pag, Pm], "Zadanie 3.2 - moce")
    ax = fig.add_subplot(1, 2, 2)
    phi = math.acos(min(p["pf"], 1.0))
    ax.plot([0, Pin], [0, 0], C1, lw=3, label="P (czynna)")
    ax.plot([Pin, Pin], [0, p["S"] * math.sin(phi)], C2, lw=3, label="Q (bierna)")
    ax.plot([0, Pin], [0, p["S"] * math.sin(phi)], "k", lw=3, label="S (pozorna)")
    ax.set_aspect("equal")
    ax.set_title(f"Trójkąt mocy, φ = {math.degrees(phi):.1f}°", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return [f"P_in = S·cos φ = {p['S']:.1f}·{p['pf']:.2f} = {Pin:.2f} kW",
            f"P_ag = P_in - straty = {Pag:.2f} kW",
            f"ω_s = {p['ns']:.0f}·2π/60 = {ws:.3f} rad/s",
            f"Te = P_ag/ω_s = {Te:.1f} Nm",
            f"(n = {p['ns'] * (1 - s):.0f} obr/min, P_m = {Pm:.2f} kW)",
            f"Q = S·sin φ = {p['S'] * math.sin(phi):.2f} kvar"]


def _mot(p):
    Lm = p["Ls"] * 1e-3 - p["Lls"] * 1e-3
    return dict(Vph=p["VLL"] / SQ3, f=p["f"], rs=p["rs"], Lls=p["Lls"] * 1e-3, rr=p["rr"],
                Llr=p["Llr"] * 1e-3, Lm=Lm, P=int(round(p["P"])))


def _torque_plot(ax, m, marks=()):
    s = np.linspace(1, 0.0005, 800)
    ax.plot(s, te_37(s, **no_lm(m)), C1, lw=2, label="wzór (3.7) - schemat zmodyf.")
    ax.plot(s, im_T(s, **m)["Te"], "k:", lw=1.5, label="pełny schemat T")
    for sv, lab, c in marks:
        ax.plot([sv], [float(te_37(sv, **no_lm(m)))], "o", color=c, ms=8, label=lab)
    ax.set_xlim(1, 0)
    ax.set_xlabel("Poślizg s")
    ax.set_ylabel("Moment [Nm]")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)


def pr33(fig, p):
    m = _mot(p)
    sm, Tmax = breakdown(**no_lm(m))
    s = sm / p["k"]
    r = im_mod(s, **m)
    Ir, Im, Is = complex(r["Ir"]), complex(r["Im"]), complex(r["Is"])
    Te = float(r["Te"])
    _torque_plot(fig.add_subplot(1, 2, 1), m, [(sm, f"s_m = {sm:.3f}", C2), (s, f"s_r = s_m/{p['k']:.1f}", C3)])
    ax = fig.add_subplot(1, 2, 2)
    for z, c, lab in ((m["Vph"] * abs(Is) * 1.1 / m["Vph"], "k", "Vs"), (Is, C1, "Is"), (Ir, C3, "Ir'"), (Im, C2, "Im")):
        w = complex(z) * 1j
        ax.annotate("", xy=(w.real, w.imag), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=c, lw=2))
        ax.text(w.real, w.imag, " " + lab, color=c)
    L = abs(Is) * 1.3
    ax.set_xlim(-0.2 * L, L)
    ax.set_ylim(-0.1 * L, L)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_title("Wykres wskazowy (schemat zmodyf.)", fontsize=10)
    we = 2 * PI * m["f"]
    return [f"a) X = ωe(Lls+Llr) = {we * (m['Lls'] + m['Llr']):.3f} Ω",
            f"   s_m = rr/sqrt(rs²+X²) = {sm:.4f}",
            f"   (Tmax = {Tmax:.2f} Nm)",
            f"b) s = s_m/{p['k']:.1f} = {s:.4f}, rr/s = {m['rr'] / s:.3f} Ω",
            f"   |Ir'| = Vs/|rs+rr/s+jX| = {abs(Ir):.2f} A",
            f"   Te = 3(P/2)|Ir'|² rr/(s ωe) = {Te:.2f} Nm",
            f"c) Lm = Ls - Lls = {m['Lm'] * 1e3:.2f} mH",
            f"   |Im| = Vs/(ωe Lm) = {abs(Im):.2f} A",
            f"d) Is = Ir' + Im = {abs(Is):.2f} A",
            f"   cos φ = {float(r['pf']):.3f} (φ = {math.degrees(-cmath.phase(Is)):.1f}°)"]


def pr34(fig, p):
    m = _mot(p)
    s = p["s"]
    Tr, Ts = float(te_37(s, **no_lm(m))), float(te_37(1.0, **no_lm(m)))
    sm, Tmax = breakdown(**no_lm(m))
    _torque_plot(fig.add_subplot(1, 1, 1), m, [(s, f"znamionowy {Tr:.1f} Nm", C3), (1.0, f"rozruchowy {Ts:.1f} Nm", C2),
                                               (sm, f"krytyczny {Tmax:.1f} Nm", C4)])
    return [f"Vs = {m['Vph']:.1f} V, ωe = {2 * PI * m['f']:.1f} rad/s",
            f"Te(s={s:.3f}) = {Tr:.2f} Nm  (znamionowy)",
            f"Te(s=1) = {Ts:.2f} Nm  (rozruchowy)",
            f"T_rozr/T_zn = {Ts / Tr:.2f}",
            f"Tmax = {Tmax:.1f} Nm przy s_m = {sm:.3f}",
            f"Schemat T: Te_zn = {float(im_T(s, **m)['Te']):.2f}, Te_rozr = {float(im_T(1.0, **m)['Te']):.2f} Nm"]


def pr35(fig, p):
    m = _mot(dict(p, rs=0.0))
    we = 2 * PI * m["f"]
    Ls = m["Lls"] + m["Lm"]
    sig = sigma_book(m["Lls"], m["Llr"], m["Lm"])
    sige = sigma_exact(m["Lls"], m["Llr"], m["Lm"])
    Is0 = m["Vph"] / (we * Ls)
    sa = np.tan(np.linspace(-PI / 2 + 1e-4, PI / 2 - 1e-4, 3000)) * 3
    I = im_T(sa, **m)["Is"]
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(-np.imag(I), np.real(I), C1, lw=2, label="miejsce geometryczne Is")
    sp = np.logspace(-4, 0, 2000)
    Ip = im_T(sp, **m)["Is"]
    ang = np.arctan2(-np.imag(Ip), np.real(Ip))
    k = np.argmin(ang)
    T = (-Ip[k].imag, Ip[k].real)
    ax.plot([0, T[0] * 1.2], [0, T[1] * 1.2], "k--", lw=1)
    ax.plot(*T, "o", color=C2, ms=8, label=f"punkt znamionowy s_r = {sp[k]:.4f}")
    ax.plot([Is0], [0], "ko")
    ax.annotate("Is0 (s=0)", (Is0, 0), xytext=(0, -14), textcoords="offset points", fontsize=8)
    ax.set_aspect("equal")
    ax.set_xlabel("składowa bierna [A]")
    ax.set_ylabel("składowa czynna [A]")
    ax.set_title("Zadanie 3.5 - wykres kołowy (rs = 0)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return [f"Ls = {Ls * 1e3:.1f} mH, Lm = {m['Lm'] * 1e3:.1f} mH",
            f"σ ≈ LsLr/Lm² - 1 = {sig:.4f} (dokładnie {sige:.4f})",
            f"Is0 = Vs/(ωe·Ls) = {Is0:.2f} A",
            f"Promień koła = Is0(1-σ)/(2σ) = {Is0 * (1 - sig) / (2 * sig):.1f} A",
            f"b) |Is(s_r)| ≈ Is0/√σ = {Is0 / math.sqrt(sig):.2f} A",
            f"   numerycznie = {abs(Ip[k]):.2f} A",
            f"c) cos φ_min = (1-σ)/(1+σ) = {(1 - sig) / (1 + sig):.4f}",
            f"   φ_min = {math.degrees(math.acos((1 - sig) / (1 + sig))):.2f}°",
            f"   numerycznie φ_min = {math.degrees(ang[k]):.2f}°"]


def pr36(fig, p):
    a = math.radians(p["a"])
    th = np.linspace(0, 2 * PI, 1000)
    v = np.sin(th)
    vo = np.where((th % PI) >= a, v, 0)
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(np.degrees(th), v, color="gray")
    ax.fill_between(np.degrees(th), 0, vo, color=C1, alpha=0.5)
    ax.set_xlabel("ωt [°]")
    ax.set_title("v = V sin ωt, przewodzenie od α do π", fontsize=10)
    ax.grid(alpha=0.3)
    ax2 = fig.add_subplot(1, 2, 2)
    aa = np.linspace(0, PI, 300)
    ax2.plot(np.degrees(aa), thyristor_vrms(aa), C1, lw=2)
    ax2.plot([p["a"]], [float(thyristor_vrms(a))], "o", color=C2)
    ax2.set_xlabel("α [°]")
    ax2.set_ylabel("Vrms / (V/√2)")
    ax2.grid(alpha=0.3)
    return ["Vrms² = (1/π)∫α..π V² sin²θ dθ",
            "      = (V²/2π)[(π - α) + sin2α/2]",
            "      = (V²/2)(1 - α/π + sin2α/(2π))",
            "Vrms = (V/√2)·sqrt(1 - α/π + sin2α/(2π))",
            f"α = {p['a']:.0f}°:  Vrms/(V/√2) = {float(thyristor_vrms(a)):.4f}"]


def pr37(fig, p):
    ns = n_sync(p["f"], p["P"])
    s1 = (ns - p["n1"]) / ns
    s2 = s1 * (p["V1"] / p["V2"]) ** 2
    ax = fig.add_subplot(1, 1, 1)
    nn = np.linspace(ns * (1 - 3 * s1), ns, 100)
    ss = 1 - nn / ns
    TL = 1.0
    ax.plot(nn, TL * ss / s1, C1, lw=2, label=f"V1 = {p['V1']:.0f} V: T ~ V1²·s")
    ax.plot(nn, TL * ss / s1 * (p["V2"] / p["V1"]) ** 2, C2, lw=2, label=f"V2 = {p['V2']:.0f} V")
    ax.axhline(TL, color="k", ls="--", label="obciążenie (stałe)")
    ax.plot([p["n1"]], [TL], "o", color=C1, ms=8)
    ax.plot([ns * (1 - s2)], [TL], "o", color=C2, ms=8)
    ax.set_xlabel("Prędkość [obr/min]")
    ax.set_ylabel("Moment [jednostki względne]")
    ax.set_title("Zadanie 3.7 - część liniowa charakterystyki")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return [f"s1 = {s1:.5f}",
            f"s2 = s1·(V1/V2)² = {s2:.5f}",
            f"n2 = n_s(1-s2) = {ns * (1 - s2):.1f} obr/min"]


def pr38(fig, p):
    Vph = p["VLL"] / SQ3
    S = 3 * Vph * p["I"]
    Pw = p["Pw"]
    phi = math.acos(min(Pw / S, 1.0))
    Q = S * math.sin(phi)
    we = 2 * PI * p["f"]
    X = Q / (3 * p["I"] ** 2)
    R = Pw / (3 * p["I"] ** 2)
    ax = fig.add_subplot(1, 2, 1)
    ax.plot([0, Pw], [0, 0], C1, lw=3, label=f"P = {Pw:.0f} W")
    ax.plot([Pw, Pw], [0, Q], C2, lw=3, label=f"Q = {Q:.0f} var")
    ax.plot([0, Pw], [0, Q], "k", lw=2, label=f"S = {S:.0f} VA")
    ax.set_title(f"Trójkąt mocy biegu jałowego, φ = {math.degrees(phi):.1f}°", fontsize=10)
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.annotate("", xy=(0, 1), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", lw=2))
    ax2.text(0.02, 1.0, "Vs")
    ax2.annotate("", xy=(math.sin(phi), math.cos(phi)), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", lw=2, color=C1))
    ax2.text(math.sin(phi), math.cos(phi), " Is0", color=C1)
    ax2.set_xlim(-0.2, 1.2)
    ax2.set_ylim(-0.1, 1.2)
    ax2.set_aspect("equal")
    ax2.set_title("Prąd jałowy prawie prostopadły do napięcia", fontsize=10)
    return ["UWAGA: w treści 'reactive power = 50 W' - jednostka W",
            "wskazuje na moc CZYNNĄ (3 fazy); tak liczymy.",
            f"Vs = {Vph:.1f} V, S = 3·Vs·I = {S:.1f} VA",
            f"a) cos φ = P/S = {Pw / S:.4f} -> φ = {math.degrees(phi):.2f}°",
            f"   Q = S·sin φ = {Q:.1f} var",
            f"b) X = Q/(3I²) = {X:.2f} Ω  ->  Ls = X/ωe = {X / we * 1e3:.1f} mH",
            f"c) R = P/(3I²) = {R:.3f} Ω",
            "   (R zawiera też straty w żelazie i tarcie,",
            "    więc to tylko oszacowanie rs z góry)",
            f"n = 1750 obr/min -> s = {(1800 - 1750) / 1800:.4f} (prawie 0)"]


def pr39(fig, p):
    Vph = p["VLL"] / SQ3
    I = p["I"]
    S = 3 * Vph * I
    phi = math.acos(min(p["Pw"] / S, 1.0))
    Z = Vph / I
    R = p["Pw"] / (3 * I ** 2)
    X = math.sqrt(max(Z ** 2 - R ** 2, 0))
    rr = R - p["rs"]
    we = 2 * PI * p["f"]
    ax = fig.add_subplot(1, 1, 1)
    ax.plot([0, p["rs"]], [0, 0], C1, lw=4, label=f"rs = {p['rs']:.2f} Ω")
    ax.plot([p["rs"], R], [0, 0], C3, lw=4, label=f"rr = {rr:.2f} Ω")
    ax.plot([R, R], [0, X], C2, lw=4, label=f"X = Xls+Xlr = {X:.2f} Ω")
    ax.plot([0, R], [0, X], "k", lw=2, label=f"|Z| = {Z:.2f} Ω")
    ax.set_aspect("equal")
    ax.set_title(f"Zadanie 3.9 - trójkąt impedancji (próba zwarcia), φ = {math.degrees(phi):.1f}°")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return [f"Vs = {p['VLL']:.0f}/√3 = {Vph:.2f} V",
            f"S = 3·Vs·I = {S:.1f} VA",
            f"a) cos φ = P/S = {p['Pw'] / S:.4f}, φ = {math.degrees(phi):.2f}°",
            f"b) |Z| = Vs/I = {Z:.3f} Ω",
            f"   R = P/(3I²) = {R:.3f} Ω",
            f"c) rr = R - rs = {rr:.3f} Ω",
            f"d) X = sqrt(Z²-R²) = {X:.3f} Ω",
            f"   Xls = Xlr = X/2 = {X / 2:.3f} Ω",
            f"   Lls = Llr = {X / 2 / we * 1e3:.2f} mH",
            "('measured shaft power' = moc pobrana 800 W;",
            " przy zablokowanym wale moc na wale = 0)"]


def pr310(fig, p):
    P = int(round(p["P"]))
    ns = n_sync(p["f"], P)
    s = (ns - p["n"]) / ns
    we = 2 * PI * p["f"]
    Te = p["Pr"] * 1e3 / rpm2rad(p["n"])
    wsl = s * we
    rr, Llr = p["rr"], p["Llr"] * 1e-3
    g = wsl * rr / (rr ** 2 + wsl ** 2 * Llr ** 2)
    lam = math.sqrt(Te / (1.5 * (P / 2) * g))
    Vn = p["VLL"]
    ax = fig.add_subplot(1, 1, 1)
    out = []
    for f, c in ((60, C1), (45, C3), (30, C2)):
        nsf = n_sync(f, P)
        wslv = np.linspace(0, 2 * PI * f, 600)
        T = 1.5 * (P / 2) * lam ** 2 * wslv * rr / (rr ** 2 + wslv ** 2 * Llr ** 2)
        n = nsf - wslv / (P / 2) * 60 / (2 * PI)
        msk = n >= 0
        ax.plot(n[msk], T[msk], color=c, lw=2, label=f"{f} Hz, V ≈ {Vn * f / 60:.0f} V")
        out.append(f"{f} Hz: V_LL = {Vn * f / p['f']:.0f} V, n_s = {nsf:.0f} obr/min")
    ax.axhline(Te, color="k", ls="--", label=f"Te znam. = {Te:.1f} Nm")
    ax.set_xlabel("Prędkość [obr/min]")
    ax.set_ylabel("Moment [Nm]")
    ax.set_title("Zadanie 3.10 - stały strumień: krzywe przesunięte równolegle")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    wmax = rr / Llr
    return [f"a) Te = P/ωr = {p['Pr'] * 1e3:.0f}/{rpm2rad(p['n']):.2f} = {Te:.2f} Nm",
            f"   s = {s:.4f}, ωsl = s·ωe = {wsl:.2f} rad/s",
            "b) Te = (3/2)(P/2) λs² ωsl rr/(rr² + ωsl² Llr²)",
            f"   λs = sqrt(Te/(1,5·{P / 2:.0f}·{g:.3f})) = {lam:.4f} Wb",
            "c) stały strumień -> V ~ f (pomijając rs):"] + ["   " + o for o in out] + [
            f"Moment max przy ωsl = rr/Llr = {wmax:.1f} rad/s",
            f"Tmax = {0.75 * (P / 2) * lam ** 2 / Llr:.1f} Nm (to samo dla każdej f!)"]


def pr311(fig, p):
    base = dict(Vph=p["VLL"] / SQ3, f=p["f"], rs=p["rs"], Lls=p["Lls"] * 1e-3, Llr=p["Llr"] * 1e-3, P=int(round(p["P"])))
    Tt = p["T"]
    ws = rpm2rad(n_sync(base["f"], base["P"]))
    s = np.linspace(1, 0.0005, 800)
    ax = fig.add_subplot(1, 2, 1)
    lines = ["rr    s       |Ir'|   φ[°]  Pin[kW] Pm[kW]  η[%]"]
    etas = []
    for rr, c in ((p["r1"], C1), (p["r2"], C3), (p["r3"], C2)):
        ax.plot(s, te_37(s, rr=rr, **base), color=c, lw=2, label=f"rr = {rr:.2f} Ω")
        sm, _ = breakdown(rr=rr, **base)
        so = solve_slip(Tt, lambda x: te_37(x, rr=rr, **base), min(sm, 1.0))
        if so is None:
            lines.append(f"{rr:4.2f}  brak rozwiązania")
            etas.append(0)
            continue
        Z = base["rs"] + rr / so + 1j * 2 * PI * base["f"] * (base["Lls"] + base["Llr"])
        Ir = base["Vph"] / Z
        phi = math.degrees(cmath.phase(Z))
        Pin = 3 * base["Vph"] * abs(Ir) * math.cos(math.radians(phi))
        Pm = Tt * ws * (1 - so)
        etas.append(Pm / Pin * 100)
        ax.plot([so], [Tt], "o", color=c, ms=8)
        lines.append(f"{rr:4.2f} {so:7.4f} {abs(Ir):6.1f} {phi:5.1f} {Pin / 1e3:7.2f} {Pm / 1e3:6.2f} {Pm / Pin * 100:5.1f}")
    ax.axhline(Tt, color="k", ls="--")
    ax.set_xlim(1, 0)
    ax.set_xlabel("Poślizg s")
    ax.set_ylabel("Moment [Nm]")
    ax.set_title("b) charakterystyki", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.bar(["rr1", "rr2", "rr3"], etas, color=[C1, C3, C2])
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("η [%]")
    ax2.set_title(f"e) sprawność przy Te = {Tt:.0f} Nm", fontsize=10)
    ax2.grid(alpha=0.3, axis="y")
    lines += ["", "Schemat zmodyfikowany bez Lm (nie podano):", "Pin = 3·Vs·|Ir'|·cos φ = 3|Ir'|²(rs + rr/s)",
              "Pm = Te·ωr,  η = Pm/Pin", "Uwaga: |Ir'| prawie jednakowe - rr/s ≈ const"]
    return lines


def pr312(fig, p):
    Vph = p["VLL"] / SQ3
    we = 2 * PI * p["f"]
    P = int(round(p["P"]))
    Ls = p["Ls"] * 1e-3
    sig = p["sig"]
    Lr = Ls
    taur = Lr / p["rr"]
    Is0 = Vph / (we * Ls)
    Rc = Is0 * (1 - sig) / (2 * sig)
    xc = Is0 * (1 + sig) / (2 * sig)
    th = np.linspace(0, 2 * PI, 400)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(xc + Rc * np.cos(th), Rc * np.sin(th), color="gray")
    ax.plot(xc + Rc * np.cos(th[:200]), Rc * np.sin(th[:200]), C1, lw=2, label="koło prądu (rs = 0)")
    cphi = (1 - sig) / (1 + sig)
    Ir_ = Is0 / math.sqrt(sig)
    phi = math.acos(cphi)
    T = (Ir_ * math.sin(phi), Ir_ * math.cos(phi))
    ax.plot([0, T[0] * 1.2], [0, T[1] * 1.2], "k--", lw=1)
    ax.plot(*T, "o", color=C2, ms=8, label="punkt znamionowy (styczna)")
    ax.plot([xc], [Rc], "^", color=C4, ms=8, label="moment krytyczny (szczyt)")
    ax.plot([Is0], [0], "ko")
    ax.plot([Is0 / sig], [0], "ko")
    ax.annotate("Is0", (Is0, 0), xytext=(0, -14), textcoords="offset points")
    ax.annotate("Is0/σ", (Is0 / sig, 0), xytext=(0, -14), textcoords="offset points")
    ax.set_aspect("equal")
    ax.set_xlabel("składowa bierna [A]")
    ax.set_ylabel("składowa czynna [A]")
    ax.set_title("Zadanie 3.12 - wykres kołowy")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    sm = 1 / (taur * we * sig)
    sr = sm * math.sqrt(sig)
    Pag = 3 * Vph * T[1]
    Te = Pag * (P / 2) / we
    Pout = Pag * (1 - sr)
    Tmax = 3 * Vph * Rc * (P / 2) / we
    return [f"Is0 = Vs/(ωe Ls) = {Is0:.2f} A",
            f"Środek koła: {xc:.1f} A, promień: {Rc:.1f} A",
            f"cos φ_max = (1-σ)/(1+σ) = {cphi:.4f}",
            f"|Is(s_r)| = Is0/√σ = {Ir_:.1f} A",
            f"Składowa czynna = {T[1]:.1f} A",
            f"Te_zn = 3·Vs·Is·cosφ·(P/2)/ωe = {Te:.1f} Nm",
            f"Tmax = 3·Vs·R_koła·(P/2)/ωe = {Tmax:.1f} Nm",
            "Moment NIE zależy od rr! Poślizgi już tak:",
            f"(założenie: rr = {p['rr']:.3f} Ω, Lr ≈ Ls)",
            f"τr = Lr/rr = {taur:.3f} s",
            f"s_m = 1/(ωe τr σ) = {sm:.4f}",
            f"s_r = s_m·√σ = {sr:.4f}",
            f"P_out ≈ Pag(1-s_r) = {Pout / 1e3:.1f} kW"]


def pr313(fig, p):
    Vph = p["VLL"] / SQ3
    fn = p["f"]
    we = 2 * PI * fn
    Ls = p["Ls"] * 1e-3
    sig = p["sig"]
    rs = p["rs"]
    lam = Vph / we
    Is0 = Vph / (we * Ls)
    Isr = Is0 / math.sqrt(sig)
    phi = math.acos((1 - sig) / (1 + sig))
    f = np.linspace(0.01, fn, 300)
    E = 2 * PI * f * lam
    Iv = Isr * np.exp(-1j * phi)
    V = np.abs(E + rs * Iv)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(f, V, C1, lw=2, label="V(f) - stały strumień przy prądzie znam.")
    ax.plot(f, Vph * f / fn, "k--", label="czyste U/f = const")
    ax.plot(f, np.abs(E + rs * Is0 * np.exp(-1j * PI / 2)), C3, lw=1, label="przy prądzie jałowym")
    ax.set_xlabel("f [Hz]")
    ax.set_ylabel("Vs fazowe [V]")
    ax.set_title("Zadanie 3.13d - profil napięcia (podbicie przy małych f)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return [f"a) λs = Vs/ωe = {Vph:.1f}/{we:.1f} = {lam:.4f} Wb (rms)",
            f"   Is0 = Vs/(ωe Ls) = {Is0:.2f} A",
            f"b) spadek rs·Is0 = {rs * Is0:.2f} V ({rs * Is0 / Vph * 100:.1f} % Vs)",
            f"c) Is(s_r) = Is0/√σ = {Isr:.1f} A",
            f"   φ_min = {math.degrees(phi):.1f}° względem SEM",
            "d) Vs(f) = |jωe·λs + rs·Is| , Is opóźniony o φ_min",
            f"   V(1 Hz) = {np.interp(1, f, V):.1f} V (U/f dałoby {Vph / fn:.1f} V)",
            f"   V(5 Hz) = {np.interp(5, f, V):.1f} V",
            f"   V(25 Hz) = {np.interp(25, f, V):.1f} V",
            f"   V(50 Hz) = {V[-1]:.1f} V"]


PROBLEMS = [
    ("3.1 Moc szczeliny i straty w wirniku",
     [("Pout", "Moc na wale [kW]", 1, 100, 15, 0.5), ("VLL", "Napięcie [V]", 200, 600, 440, 1),
      ("f", "f [Hz]", 50, 60, 60, 10), ("P", "Bieguny P", 2, 8, 4, 2),
      ("n", "Prędkość [obr/min]", 1600, 1799, 1748, 1), ("Pfw", "Tarcie i wentylacja [kW]", 0, 5, 1, 0.1)],
     pr31,
     """ZADANIE 3.1 - silnik 15 kW, 440 V, 60 Hz, 4 bieguny, 1748 obr/min, tarcie+wentylacja 1 kW.

Idziemy diagramem Sankeya "od końca": moc rozwinięta przez wirnik to moc na wale PLUS to,
co zjada tarcie:  P_m = 15 + 1 = 16 kW.
Poślizg: s = (1800 - 1748)/1800 = 0,0289.
Z zależności P_m = (1 - s)·P_ag:  P_ag = 16/0,9711 = 16,48 kW
Straty w miedzi wirnika: P_cu2 = s·P_ag = 0,476 kW.
Napięcie nie jest tu potrzebne - to typowa "pułapka" w zadaniach!"""),
    ("3.2 Moment z mocy pozornej",
     [("ns", "Prędkość synchr. [obr/min]", 500, 3600, 900, 50), ("S", "Moc pozorna [kVA]", 5, 200, 40, 1),
      ("pf", "cos φ", 0.5, 1.0, 0.9, 0.01), ("Pst", "Straty stojana [kW]", 0, 20, 4, 0.1),
      ("sl", "Poślizg [%]", 0.5, 10, 3, 0.1)],
     pr32,
     """ZADANIE 3.2 - n_s = 900 obr/min, pobór 40 kVA, cos φ = 0,9, straty stojana 4 kW.

Moc pozorna S [kVA] to "iloczyn napięcia i prądu"; moc czynna (zamieniana na pracę) to S·cos φ:
  P_in = 40·0,9 = 36 kW,   P_ag = 36 - 4 = 32 kW.
NAJWAŻNIEJSZE: moment liczymy z mocy SZCZELINY i prędkości SYNCHRONICZNEJ
(pole wiruje z n_s i "przekazuje" moment wirnikowi):
  Te = P_ag/ω_s = 32000/(900·2π/60) = 32000/94,25 = 339,5 Nm.
Poślizg (3 %) nie jest tu potrzebny do momentu - przyda się tylko do mocy mechanicznej."""),
    ("3.3 Poślizg krytyczny, prąd i cos φ",
     [("rs", "rs [Ω]", 0, 5, 1.6, 0.01), ("rr", "rr [Ω]", 0.1, 5, 0.996, 0.001),
      ("Ls", "Ls [mH]", 20, 200, 66, 0.5), ("Lls", "Lls [mH]", 0.5, 10, 3.28, 0.01),
      ("Llr", "Llr [mH]", 0.5, 10, 3.28, 0.01), ("VLL", "Napięcie [V]", 100, 480, 220, 1),
      ("f", "f [Hz]", 50, 60, 60, 10), ("P", "Bieguny P", 2, 8, 4, 2),
      ("k", "s_znam = s_m / k, k =", 1.2, 6, 2.5, 0.1)],
     pr33,
     """ZADANIE 3.3 - rs = 1,6 Ω, rr = 0,996 Ω, Ls = 66 mH, Lls = Llr = 3,28 mH, 220 V, 60 Hz, 4p.

a) Poślizg krytyczny (3.10): X = 377·6,56 mH = 2,47 Ω;  s_m = 0,996/sqrt(1,6² + 2,47²) = 0,338.
b) s = s_m/2,5 = 0,135. Z prawa Ohma dla schematu zmodyfikowanego:
   |Ir'| = 127/|1,6 + 7,37 + j2,47| ≈ 13,65 A, a moment Te = 3(P/2)|Ir'|²rr/(s·ωe) ≈ 21,9 Nm.
c) Prąd magnesujący: Lm = Ls - Lls = 62,7 mH,  Im = 127/(377·0,0627) ≈ 5,37 A.
   Jest opóźniony o 90° względem napięcia - "buduje" pole, nie wykonuje pracy.
d) Prąd stojana to SUMA WEKTOROWA Ir' + Im (wykres wskazowy), cos φ ≈ 0,83."""),
    ("3.4 Moment znamionowy i rozruchowy",
     [("rs", "rs [Ω]", 0, 1, 0.1, 0.01), ("rr", "rr [Ω]", 0.05, 2, 0.25, 0.01),
      ("Ls", "Ls [mH]", 20, 300, 150, 1), ("Lls", "Lls [mH]", 0.5, 10, 3, 0.1),
      ("Llr", "Llr [mH]", 0.5, 10, 3, 0.1), ("VLL", "Napięcie [V]", 100, 600, 440, 1),
      ("f", "f [Hz]", 50, 60, 60, 10), ("P", "Bieguny P", 2, 8, 4, 2),
      ("s", "Poślizg znamionowy", 0.005, 0.1, 0.03, 0.001)],
     pr34,
     """ZADANIE 3.4 - rs = 0,1 Ω, rr = 0,25 Ω, Ls = 150 mH, Lls = Llr = 3 mH, 440 V, 60 Hz, s = 3 %.

Podstawiamy do wzoru (3.7) dwa razy: dla s = 0,03 (praca znamionowa) i s = 1 (rozruch).
  Vs = 440/√3 = 254 V,  X = 377·6 mH = 2,26 Ω
  s = 0,03:  rr/s = 8,33 Ω  ->  Te ≈ 112 Nm
  s = 1:     rr/s = 0,25 Ω  ->  Te ≈ 49 Nm
NIESPODZIANKA: moment rozruchowy to tylko ok. 44 % znamionowego! Przy s = 1 prąd jest
ogromny, ale ograniczony głównie przez reaktancję X = 2,26 Ω, a "użyteczna" rezystancja
rr/s = 0,25 Ω jest mała - większość prądu nie daje mocy czynnej w wirniku.
Taki silnik (małe rr, jak NEMA A) NIE ruszy z pełnym obciążeniem o stałym momencie -
potrzebny byłby falownik (VVVF) albo wirnik z większym rr / klatką podwójną.
(To silnik z ćw. 3.2 - porównaj z tamtą zakładką.)"""),
    ("3.5 Wykres kołowy, prąd znamionowy, kąt φ_min",
     [("rr", "rr [Ω]", 0.05, 2, 0.25, 0.01),
      ("Ls", "Ls [mH]", 20, 300, 150, 1), ("Lls", "Lls [mH]", 0.5, 10, 3, 0.1),
      ("Llr", "Llr [mH]", 0.5, 10, 3, 0.1), ("VLL", "Napięcie [V]", 100, 600, 440, 1),
      ("f", "f [Hz]", 50, 60, 60, 10), ("P", "Bieguny P", 2, 8, 4, 2)],
     pr35,
     """ZADANIE 3.5 - ten sam silnik co 3.4, ale rs = 0 (idealny wykres kołowy).

Is0 = Vs/(ωe·Ls) = 254/(377·0,15) = 4,49 A (prąd jałowy).
σ ≈ (Lls + Llr)/Lm ≈ 6/147 ≈ 0,041 -> koło jest DUŻE (małe rozproszenie = dobry silnik).
Styczna z początku układu wyznacza punkt znamionowy:
   |Is(s_r)| = Is0/√σ ≈ 22 A,   cos φ_min = (1 - σ)/(1 + σ) ≈ 0,92  (φ_min ≈ 23°).
Wykres pokazuje okrąg, styczną i punkt znamionowy obliczone numerycznie z pełnego schematu."""),
    ("3.6 Sterownik tyrystorowy - Vrms(α)",
     [("a", "Kąt załączenia α [°]", 0, 180, 60, 1)],
     pr36,
     """ZADANIE 3.6 - zależność napięcia skutecznego od kąta załączenia α.

Wartość skuteczna = "pierwiastek ze średniej z kwadratu". Napięcie jest obecne tylko od α do π
w każdym półokresie:
   Vrms² = (1/π) ∫ (V sin θ)² dθ  (od α do π)
Korzystamy z tożsamości sin²θ = (1 - cos2θ)/2:
   Vrms² = (V²/2π) [ (π - α) + sin(2α)/2 ]
   Vrms = (V/√2) · sqrt( 1 - α/π + sin(2α)/(2π) )
Szczegóły i widmo - zakładka TYR."""),
    ("3.7 Wzrost napięcia 220 -> 250 V",
     [("V1", "V1 [V]", 100, 400, 220, 1), ("V2", "V2 [V]", 100, 400, 250, 1),
      ("n1", "n1 [obr/min]", 1650, 1799, 1750, 1), ("f", "f [Hz]", 50, 60, 60, 10), ("P", "Bieguny", 2, 8, 4, 2)],
     pr37,
     """ZADANIE 3.7 - identyczne z ćwiczeniem 3.3.
s1 = 50/1800 = 0,0278;  s2 = 0,0278·(220/250)² = 0,0215;  n2 = 1800·(1 - 0,0215) = 1761 obr/min.
Klucz: w części liniowej Te ~ V²·s, a przy stałym momencie obciążenia V²·s = const."""),
    ("3.8 Próba biegu jałowego",
     [("VLL", "Napięcie [V]", 100, 480, 220, 1), ("f", "f [Hz]", 50, 60, 60, 10),
      ("I", "Prąd fazowy [A]", 0.5, 20, 2, 0.1), ("Pw", "Moc czynna (3 fazy) [W]", 5, 1000, 50, 1)],
     pr38,
     """ZADANIE 3.8 - PRÓBA BIEGU JAŁOWEGO (no-load test).

Silnik kręci się bez obciążenia, prawie synchronicznie (1750 vs 1800 obr/min), więc rr/s
jest OGROMNE - gałąź wirnika jakby nie istnieje. Zostaje rs + jωe(Lls + Lm) = rs + jωeLs.
Mierzymy napięcie, prąd i moc. Z nich:
  S = 3·Vs·I,  cos φ = P/S,  X = Q/(3I²) -> Ls = X/ωe,  R = P/(3I²).
W treści moc podano w watach ("reactive power is 50 W") - przyjmujemy, że to moc CZYNNA
pobrana przez silnik (suwak pozwala zmienić). Wynik: φ ≈ 86°, Ls ≈ 168 mH.
Prąd jałowy jest prawie czysto bierny - stąd fatalny cos φ niedociążonych silników!"""),
    ("3.9 Próba zwarcia (wirnik zablokowany)",
     [("VLL", "Napięcie [V]", 20, 200, 90, 1), ("f", "f [Hz]", 50, 60, 60, 10),
      ("I", "Prąd [A]", 1, 50, 7, 0.1), ("rs", "rs [Ω]", 0, 5, 2.1, 0.01),
      ("Pw", "Moc pobrana [W]", 50, 3000, 800, 10)],
     pr39,
     """ZADANIE 3.9 - PRÓBA ZWARCIA (locked-rotor test).

Blokujemy wał (s = 1) i podajemy OBNIŻONE napięcie (90 V zamiast 220 V), żeby prąd był
bliski znamionowemu. Teraz rr/s = rr jest małe, a prąd magnesujący pomijalny w porównaniu
z prądem w gałęzi wirnika - widzimy tylko rs + rr + j(Xls + Xlr).
  |Z| = Vs/I = 51,96/7 = 7,42 Ω
  R = P/(3I²) = 800/147 = 5,44 Ω   ->  rr = R - rs = 3,34 Ω
  X = sqrt(Z² - R²) = 5,05 Ω  ->  Xls = Xlr = 2,52 Ω (6,7 mH)
Próby biegu jałowego i zwarcia to standardowy sposób wyznaczania parametrów schematu
zastępczego w laboratorium - jak "pomiar" transformatora."""),
    ("3.10 Strumień znamionowy i U/f",
     [("Pr", "Moc znamionowa [kW]", 1, 50, 7.5, 0.5), ("VLL", "Napięcie [V]", 200, 600, 440, 1),
      ("f", "f [Hz]", 50, 60, 60, 10), ("P", "Bieguny P", 2, 8, 4, 2),
      ("rr", "rr [Ω]", 0.05, 2, 0.25, 0.01), ("Llr", "Llr [mH]", 0.5, 10, 3.28, 0.01),
      ("n", "Prędkość znamionowa [obr/min]", 1650, 1799, 1746, 1)],
     pr310,
     """ZADANIE 3.10 - 7,5 kW, 440 V, 60 Hz, 4p, rr = 0,25 Ω, Llr = 3,28 mH, 1746 obr/min.

a) Te = 7500/(1746·2π/60) = 41,0 Nm.
b) Przy STAŁYM strumieniu moment zależy tylko od prędkości poślizgu ωsl = s·ωe:
     Te = (3/2)(P/2)·λs²·ωsl·rr/(rr² + ωsl²·Llr²)
   (w treści skrót "3/2 P" - przyjmujemy liczbę par biegunów P/2, zgodnie z rozdz. 4).
   ωsl = 0,03·377 = 11,3 rad/s -> λs ≈ 0,56 Wb.
c) λs ≈ Vs/ωe, więc stały strumień wymaga V ~ f: 45 Hz -> 330 V, 30 Hz -> 220 V.
   Krzywe moment-prędkość mają IDENTYCZNY kształt, przesunięty o różnicę n_s -
   to istota sterowania U/f (zakładka VVVF)."""),
    ("3.11 Trzy rezystancje wirnika przy 100 Nm",
     [("VLL", "Napięcie [V]", 200, 600, 440, 1), ("f", "f [Hz]", 50, 60, 60, 10),
      ("P", "Bieguny P", 2, 8, 4, 2), ("rs", "rs [Ω]", 0, 1, 0.075, 0.005),
      ("Lls", "Lls [mH]", 0.5, 10, 3, 0.1), ("Llr", "Llr [mH]", 0.5, 10, 3, 0.1),
      ("r1", "rr1 [Ω]", 0.02, 2, 0.1, 0.01), ("r2", "rr2 [Ω]", 0.02, 2, 0.4, 0.01),
      ("r3", "rr3 [Ω]", 0.02, 2, 0.8, 0.01), ("T", "Moment obciążenia [Nm]", 10, 300, 100, 1)],
     pr311,
     """ZADANIE 3.11 - 440 V, rs = 0,075 Ω, Lls = Llr = 3 mH, rr = 0,1 / 0,4 / 0,8 Ω, Te = 100 Nm.

a) Dla każdego rr szukamy poślizgu, przy którym wzór (3.7) daje 100 Nm (bisekcja w obszarze
   stabilnym). Poślizg rośnie prawie proporcjonalnie do rr.
c) Prąd |Ir'| = Vs/|rs + rr/s + jX| i kąt φ = arg(impedancji).
d) Moc z sieci Pin = 3·Vs·|Ir'|·cos φ.
e) Moc mechaniczna Pm = Te·ωr i sprawność η = Pm/Pin.
Wniosek jak w ćw. 3.4: większe rr = większy poślizg = niższa prędkość = niższa sprawność,
mimo prawie takiego samego prądu. Różnica mocy zamienia się w ciepło w wirniku."""),
    ("3.12 Wykres kołowy: Ls = 12 mH, σ = 0,08",
     [("VLL", "Napięcie [V]", 200, 600, 440, 1), ("f", "f [Hz]", 50, 60, 60, 10),
      ("P", "Bieguny P", 2, 8, 4, 2), ("Ls", "Ls [mH]", 2, 50, 12, 0.5),
      ("sig", "σ", 0.02, 0.2, 0.08, 0.005), ("rr", "rr [Ω] (nie podano - założenie)", 0.01, 0.5, 0.05, 0.005)],
     pr312,
     """ZADANIE 3.12 - 440 V, 60 Hz, 4p, Ls = 12 mH, σ = 0,08, rs = 0.

a) Koło: zaczyna się w Is0 = Vs/(ωe Ls) = 254/(377·0,012) = 56,2 A, kończy (s = ∞) w Is0/σ.
   Środek: Is0(1+σ)/(2σ), promień Is0(1-σ)/(2σ).
b) Punkt znamionowy = styczna z początku układu: cos φ_max = (1-σ)/(1+σ) = 0,852;
   |Is| = Is0/√σ = 199 A. Moment z mocy czynnej: Te = 3·Vs·Is·cos φ·(P/2)/ωe.
c) Moment krytyczny = najwyższy punkt koła: Tmax = 3·Vs·R_koła·(P/2)/ωe.
CIEKAWOSTKA: momenty NIE zależą od rr! Ale poślizgi s_r i s_m już tak (przez τr = Lr/rr),
a rr nie podano - dlatego jest suwak z założoną wartością (i Lr ≈ Ls)."""),
    ("3.13 Strumień, spadek na rs i profil U/f",
     [("VLL", "Napięcie [V]", 200, 600, 380, 1), ("f", "f [Hz]", 50, 60, 50, 10),
      ("rs", "rs [Ω]", 0, 1, 0.2, 0.01), ("Ls", "Ls [mH]", 2, 50, 12, 0.5),
      ("sig", "σ", 0.02, 0.2, 0.1, 0.005)],
     pr313,
     """ZADANIE 3.13 - 380 V, 50 Hz, 4p, rs = 0,2 Ω, Ls = 12 mH, σ = 0,1.

a) Strumień λs = Vs/ωe = 219,4/314,2 = 0,698 Wb; prąd jałowy Is0 = λs/Ls = 58,2 A.
b) Spadek na rs przy biegu jałowym: 0,2·58,2 = 11,6 V - przy 50 Hz to tylko 5 %, ale przy
   5 Hz (Vs = 22 V) to już ponad połowa napięcia!
c) Prąd znamionowy z wykresu kołowego: Is0/√σ = 184 A.
d) Aby utrzymać stały strumień przy prądzie znamionowym, napięcie musi pokryć SEM ωe·λs
   PLUS spadek na rs:  Vs = |jωe λs + rs·Is|. Przy małych f krzywa (niebieska) leży wyraźnie
   nad prostą U/f - to właśnie "podbicie napięcia" (voltage boost) z rys. 3.37."""),
]


class TabZAD(ExampleTab):
    key = "ZAD"
    PARAMS = PROBLEMS[0][1]

    def top_controls(self, parent):
        ttk.Label(parent, text="Wybierz zadanie", font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=6, pady=(6, 2))
        self.sel = tk.StringVar(value=PROBLEMS[0][0])
        cb = ttk.Combobox(parent, textvariable=self.sel, state="readonly", values=[p[0] for p in PROBLEMS], width=40)
        cb.pack(fill=tk.X, padx=6)
        cb.bind("<<ComboboxSelected>>", self._change)
        self.idx = 0

    def _change(self, _e=None):
        names = [p[0] for p in PROBLEMS]
        self.idx = names.index(self.sel.get())
        self.build_params(PROBLEMS[self.idx][1])
        self.set_expl(PROBLEMS[self.idx][3])
        self.refresh()

    def set_expl(self, text):
        if not text:
            text = PROBLEMS[0][3]
        super().set_expl(text)

    def update_plot(self):
        title, params, func, _ = PROBLEMS[self.idx]
        vals = {k[0]: self.p(k[0]) for k in params}
        return [title, ""] + func(self.fig, vals)


# =============================================================================
# OKNO GŁÓWNE
# =============================================================================

TABS = [("3.1 Moc", TabE31), ("3.2 T(s)", TabE32), ("3.3 U", TabE33), ("3.4 rr", TabE34),
        ("3.5 rr?", TabE35), ("3.6 Znam.", TabE36), ("Stabilność", TabSTAB), ("Harmon.", TabPAR),
        ("3.8 Żłobek", TabE38), ("3.9-12 Koło", TabCIRC), ("3.13 Naskórek", TabE313), ("NEMA", TabNEMA),
        ("Rozruch", TabSTART), ("Tyrystor", TabTYR), ("Napięcie", TabNAP), ("VVVF", TabVVVF),
        ("Zadania 3.1-3.13", TabZAD)]


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Silnik indukcyjny - podstawy (rozdział 3): ćwiczenia i zadania")
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
