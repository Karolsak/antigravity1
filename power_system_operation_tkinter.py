"""
Analiza i sterowanie systemem elektroenergetycznym - interaktywne przykłady
===========================================================================
Aplikacja Tkinter + Matplotlib do rozdziałów:
  11. Ekonomiczna praca systemu (Economic Operation)
  12. Przesył EHV AC i HVDC (HVDC Power Transmission)
  13. Stabilność systemu (Power System Stability)

Zakładki pogrupowane w rozdziały (wszystkie przykłady z załącznika
+ ćwiczenie 11 i kryterium równych pól):
  Start                 - jak korzystać z aplikacji, słowniczek
  Rozdz. 11
    Przykład 11.1       - przyrostowy koszt paliwa (IFC) jednego bloku
    Przykład 11.2       - rozdział obciążenia między 2 turbogeneratory (+ limity)
    Przykład 11.3       - dobowy harmonogram i wyłączanie bloków (unit commitment)
    Przykład 11.4       - przyrostowe straty przesyłowe w systemie dwuobszarowym
    Przykład 11.5       - współczynniki kary (penalty factor)
    Przykład 11.6       - równania koordynacyjne vs metoda współczynnika kary
    Ćwiczenie 11        - iteracja lambda ze stratami (wzór B)
    Przykład 11.7       - elektrownia cieplna + wodna, koszt wody
    Przykład 11.8       - harmonogram hydro-termiczny szczyt / poza szczytem
  Rozdz. 12
    Przykład 12.1       - bipolarne łącze HVDC: moc i napięcie wzdłuż linii
    Przykład 12.2       - kąt wygaszania falownika HVDC
    Przykład 12.3       - napięcia DC i prąd łącza przy kątach alfa / delta
  Rozdz. 13
    Przykład 13.1       - moc przyspieszająca i kąt mocy po 10 okresach
    Przykład 13.2       - energia kinetyczna wirnika i przyspieszenie
    Przykład 13.3       - równanie kołysań + symulacja zwarcia (RK4)
    13.4 - 13.5         - kryterium równych pól (skok mocy mechanicznej)

Interakcja z wykresami:
  * suwaki po lewej - każda zmiana od razu przelicza wynik i wykres,
  * pasek narzędzi Matplotlib - powiększanie, przesuwanie, zapis PNG,
  * ruch myszy nad wykresem - odczyt współrzędnych na pasku statusu,
  * kliknięcie w wykres (tam, gdzie opisano) - ustawia parametr.

Uruchomienie:  python power_system_operation_tkinter.py
Wymagania:     Python 3.8+, numpy, matplotlib, tkinter
"""
import math
import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)

PI = math.pi
SQ2 = math.sqrt(2.0)
K_BRIDGE = 3.0 * SQ2 / PI          # 1.3505 - mostek 6-pulsowy: Vd0 = K * V_LL

C_BLUE, C_RED, C_GREEN = '#1f77b4', '#d62728', '#2ca02c'
C_ORANGE, C_PURPLE, C_GREY = '#ff7f0e', '#9467bd', '#7f7f7f'


# =============================================================================
# Rdzen obliczeniowy (bez GUI - można importować i testować osobno)
# =============================================================================

def quad_cost(a, b, c, p):
    """Koszt paliwa C = a + b*P + c*P^2."""
    return a + b * p + c * p * p


def quad_ic(b, c, p):
    """Przyrostowy koszt dC/dP = b + 2*c*P."""
    return b + 2.0 * c * p


def lambda_dispatch(units, load, b_loss=None, tol=1e-9, max_iter=200):
    """Rozdział ekonomiczny metoda iteracji lambda (bisekcja).

    units  - lista słowników {a, b, c, pmin, pmax}
    load   - zapotrzebowanie odbiorów P_R [MW]
    b_loss - lista diagonalnych wsp. strat B_ii [1/MW]; P_L = sum B_ii P_i^2

    Warunek optymalności dla każdego bloku (równanie koordynacyjne):
        b_i + 2 c_i P_i = lambda * (1 - 2 B_ii P_i)
    => P_i = (lambda - b_i) / (2 c_i + 2 lambda B_ii), obcięte do [pmin, pmax].
    Zwraca słownik z mocami, lambda, stratami i historia iteracji.
    """
    n = len(units)
    bl = b_loss if b_loss is not None else [0.0] * n

    def outputs(lam):
        ps = []
        for u, bii in zip(units, bl):
            den = 2.0 * u['c'] + 2.0 * lam * bii
            p = (lam - u['b']) / den if den > 0 else u['pmax']
            ps.append(min(max(p, u['pmin']), u['pmax']))
        return ps

    def mismatch(lam):
        ps = outputs(lam)
        pl = sum(bii * p * p for bii, p in zip(bl, ps))
        return sum(ps) - pl - load, ps, pl

    lo = min(quad_ic(u['b'], u['c'], u['pmin']) for u in units) * 0.5 - 1.0
    hi = max(quad_ic(u['b'], u['c'], u['pmax']) for u in units) * 3.0 + 10.0
    hist = []
    feasible = True
    if mismatch(hi)[0] < 0 or mismatch(lo)[0] > 0:
        feasible = False
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f, ps, pl = mismatch(mid)
        hist.append((mid, f))
        if abs(f) < tol:
            break
        if f > 0:
            hi = mid
        else:
            lo = mid
    f, ps, pl = mismatch(mid)
    at_limit = [abs(p - u['pmin']) < 1e-6 or abs(p - u['pmax']) < 1e-6
                for p, u in zip(ps, units)]
    cost = sum(quad_cost(u['a'], u['b'], u['c'], p) for u, p in zip(units, ps))
    return dict(P=ps, lam=mid, PL=pl, cost=cost, hist=hist,
                at_limit=at_limit, feasible=feasible)


# ---- 11.3: unit commitment dla krzywej uporządkowanej ------------------------
def uc_period(units, load):
    """Porównanie wariantów pracy dla jednego okresu: oba bloki / tylko 1 / tylko 2."""
    out = {}
    if units[0]['pmin'] + units[1]['pmin'] <= load <= units[0]['pmax'] + units[1]['pmax']:
        r = lambda_dispatch(units, load)
        out['oba'] = (r['cost'], r['P'])
    for k, u in enumerate(units):
        if u['pmin'] <= load <= u['pmax']:
            p = [0.0, 0.0]
            p[k] = load
            out['tylko G%d' % (k + 1)] = (quad_cost(u['a'], u['b'], u['c'], load), p)
    return out


# ---- 11.7 ---------------------------------------------------------------
def hydro_load_from_water(volume, hours, q1, q2):
    """Rozwiazuje: P_h (q1 + q2 P_h) * 3600 * h = V  (równanie kwadratowe)."""
    k = volume / (3600.0 * hours)
    if q2 <= 0:
        return k / q1
    return (-q1 + math.sqrt(q1 * q1 + 4.0 * q2 * k)) / (2.0 * q2)


# ---- 11.8 ---------------------------------------------------------------
def hydrothermal_two_period(b_s, c_s, q1, q2, load_off, load_peak,
                            h_off, h_peak, water):
    """Szczyt / poza szczytem. Układ: b_s + 2c_s P_s = R (q1 + 2 q2 P_h),
    P_s + P_h = P_load, oraz suma zużycia wody = water. Szukamy R bisekcja."""
    def split(R, load):
        ph = (b_s + 2 * c_s * load - R * q1) / (2 * c_s + 2 * q2 * R)
        ph = min(max(ph, 0.0), load)
        return load - ph, ph

    def used(R):
        _, pho = split(R, load_off)
        _, php = split(R, load_peak)
        return 3600.0 * (h_off * (q1 * pho + q2 * pho ** 2)
                         + h_peak * (q1 * php + q2 * php ** 2))

    lo, hi = 1e-4, 50.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if used(mid) > water:
            lo = mid
        else:
            hi = mid
    R = 0.5 * (lo + hi)
    pso, pho = split(R, load_off)
    psp, php = split(R, load_peak)
    return dict(R=R, Pso=pso, Pho=pho, Psp=psp, Php=php, used=used(R),
                lam_o=b_s + 2 * c_s * pso, lam_p=b_s + 2 * c_s * psp,
                used_fn=used, split=split)


# ---- 13.x: równanie kołysań ------------------------------------------------
def swing_simulate(H, f, pm, p_pre, p_fault, p_post, t_clear, d0,
                   damping=0.0, t_end=3.0, dt=1e-3):
    """Równanie kołysań (H/(pi f)) d2delta/dt2 = Pm - Pmax sin(delta) - D*dw,
    całkowane metoda Rungego-Kutty 4. rzędu.
    p_fault obowiązuje dla t < t_clear, potem p_post. delta w radianach."""
    M = H / (PI * f)
    ws = 2.0 * PI * f
    n = int(t_end / dt) + 1
    t = np.linspace(0.0, t_end, n)
    d = np.zeros(n)
    w = np.zeros(n)            # odchylka prędkości [rad/s el.]
    d[0] = d0

    def acc(ti, di, wi):
        pmax = p_fault if ti < t_clear else p_post
        return (pm - pmax * math.sin(di) - damping * wi / ws) / M

    for k in range(n - 1):
        tk_, dk, wk = t[k], d[k], w[k]
        k1d, k1w = wk, acc(tk_, dk, wk)
        k2d, k2w = wk + 0.5 * dt * k1w, acc(tk_ + 0.5 * dt, dk + 0.5 * dt * k1d, wk + 0.5 * dt * k1w)
        k3d, k3w = wk + 0.5 * dt * k2w, acc(tk_ + 0.5 * dt, dk + 0.5 * dt * k2d, wk + 0.5 * dt * k2w)
        k4d, k4w = wk + dt * k3w, acc(tk_ + dt, dk + dt * k3d, wk + dt * k3w)
        d[k + 1] = dk + dt / 6.0 * (k1d + 2 * k2d + 2 * k3d + k4d)
        w[k + 1] = wk + dt / 6.0 * (k1w + 2 * k2w + 2 * k3w + k4w)
        if d[k + 1] > 4 * PI:          # wypadnięcie z synchronizmu - dalej nie liczymy
            d[k + 2:] = np.nan
            w[k + 2:] = np.nan
            break
    stable = bool(np.nanmax(d) < PI)
    return t, d, w, stable


def critical_clearing_angle(pm, p_pre, p_fault, p_post):
    """Krytyczny kąt wyłączenia z kryterium równych pól (wzór ogólny)."""
    if p_post <= pm or p_pre <= pm:
        return None
    d0 = math.asin(pm / p_pre)
    dmax = PI - math.asin(pm / p_post)
    num = pm * (dmax - d0) + p_post * math.cos(dmax) - p_fault * math.cos(d0)
    cosd = num / (p_post - p_fault)
    if cosd < -1 or cosd > 1:
        return None
    return math.acos(cosd)


def critical_clearing_time(H, f, pm, p_pre, p_fault, p_post, d0, damping=0.0):
    """Krytyczny czas wyłączenia zwarcia - bisekcja na symulacji."""
    lo, hi = 0.0, 1.0
    if not swing_simulate(H, f, pm, p_pre, p_fault, p_post, lo, d0, damping, 2.5, 2e-3)[3]:
        return 0.0
    if swing_simulate(H, f, pm, p_pre, p_fault, p_post, hi, d0, damping, 2.5, 2e-3)[3]:
        return None
    for _ in range(18):
        mid = 0.5 * (lo + hi)
        if swing_simulate(H, f, pm, p_pre, p_fault, p_post, mid, d0, damping, 2.5, 2e-3)[3]:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def equal_area_step(pmax, pm0, pm1):
    """Skok mocy mechanicznej Pm0 -> Pm1 na tej samej charakterystyce."""
    d0 = math.asin(min(pm0 / pmax, 1.0))
    if pm1 >= pmax:
        return dict(d0=d0, d1=None, d2=None, d3=None, stable=False, A1=None, A2max=None)
    d1 = math.asin(pm1 / pmax)
    d3 = PI - d1

    def F(d):          # pole netto (przyspieszające - hamujące) od d0 do d
        return pm1 * (d - d0) + pmax * (math.cos(d) - math.cos(d0))

    A1 = F(d1)
    A2max = -(F(d3) - F(d1))
    stable = F(d3) <= 0
    d2 = None
    if stable:
        lo, hi = d1, d3
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if F(mid) > 0:
                lo = mid
            else:
                hi = mid
        d2 = 0.5 * (lo + hi)
    return dict(d0=d0, d1=d1, d2=d2, d3=d3, stable=stable, A1=A1, A2max=A2max)


# =============================================================================
# Szkielet GUI: wspólna klasa zakładki-przykładu
# =============================================================================

class ExampleTab(ttk.Frame):
    """Zakladka: suwaki + wyniki + wyjaśnienie (lewa strona), wykres (prawa)."""
    title = ''
    params = []          # (klucz, etykieta, min, max, domyślna, krok)
    explanation = ''
    click_hint = ''

    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        self.vars = {}
        self._job = None
        self._build()
        self.redraw()

    # ---- budowa ---------------------------------------------------------
    def _build(self):
        pw = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        pw.pack(fill=tk.BOTH, expand=True)
        left = ttk.Frame(pw, width=440)
        right = ttk.Frame(pw)
        pw.add(left, weight=0)
        pw.add(right, weight=1)

        ttk.Label(left, text=self.title, style='Title.TLabel',
                  wraplength=420).pack(anchor='w', padx=6, pady=(6, 2))

        ctrl = ttk.LabelFrame(left, text='Parametry (suwaki)')
        ctrl.pack(fill=tk.X, padx=6, pady=4)
        for key, label, lo, hi, val, res in self.params:
            row = ttk.Frame(ctrl)
            row.pack(fill=tk.X, padx=4, pady=0)
            ttk.Label(row, text=label, width=30, anchor='w').pack(side=tk.LEFT)
            var = tk.DoubleVar(value=val)
            sc = tk.Scale(row, from_=lo, to=hi, resolution=res, orient=tk.HORIZONTAL,
                          variable=var, length=190, showvalue=True,
                          command=lambda _v: self.schedule(),
                          highlightthickness=0, bd=0, font=('TkDefaultFont', 8))
            sc.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self.vars[key] = var
        self.extra_controls(ctrl)
        btns = ttk.Frame(ctrl)
        btns.pack(fill=tk.X, pady=3)
        ttk.Button(btns, text='Wartości z książki (reset)',
                   command=self.reset).pack(side=tk.LEFT, padx=4)
        if self.click_hint:
            ttk.Label(btns, text='Klik: ' + self.click_hint, foreground='#555',
                      wraplength=230).pack(side=tk.LEFT, padx=4)

        nb = ttk.Notebook(left)
        nb.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)
        self.res_txt = ScrolledText(nb, wrap=tk.WORD, height=14, font=('Consolas', 9))
        self.exp_txt = ScrolledText(nb, wrap=tk.WORD, height=14, font=('TkDefaultFont', 10))
        nb.add(self.res_txt, text='Wyniki krok po kroku')
        nb.add(self.exp_txt, text='Wyjaśnienie dla ucznia')
        self.exp_txt.insert('1.0', self.explanation.strip())
        self.exp_txt.configure(state=tk.DISABLED)

        self.fig = Figure(figsize=(8, 6), dpi=95)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        tb = NavigationToolbar2Tk(self.canvas, right, pack_toolbar=False)
        tb.update()
        tb.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect('motion_notify_event', self._on_motion)
        self.canvas.mpl_connect('button_press_event', self._on_click)

    def extra_controls(self, parent):
        """Miejsce na dodatkowe kontrolki (np. lista wyboru) w podklasach."""

    # ---- zdarzenia ------------------------------------------------------
    def v(self, key):
        return float(self.vars[key].get())

    def setv(self, key, value):
        for k, _l, lo, hi, _d, res in self.params:
            if k == key:
                value = min(max(value, lo), hi)
                value = round(value / res) * res
        self.vars[key].set(value)
        self.schedule()

    def reset(self):
        for key, _l, _lo, _hi, val, _r in self.params:
            self.vars[key].set(val)
        self.reset_extra()
        self.schedule()

    def reset_extra(self):
        pass

    def schedule(self):
        if self._job is not None:
            self.after_cancel(self._job)
        self._job = self.after(80, self.redraw)

    def redraw(self):
        self._job = None
        self.fig.clear()
        try:
            text = self.compute_and_plot()
        except Exception as exc:          # nie zamykamy aplikacji przy złych danych
            text = 'Błąd obliczeń dla tych parametrów: %s' % exc
        self.res_txt.configure(state=tk.NORMAL)
        self.res_txt.delete('1.0', tk.END)
        self.res_txt.insert('1.0', text)
        self.res_txt.configure(state=tk.DISABLED)
        try:
            self.fig.tight_layout()
        except Exception:
            pass
        self.canvas.draw_idle()

    def _on_motion(self, ev):
        if ev.inaxes is None or ev.xdata is None:
            self.app.status.set(self.title)
            return
        lx = ev.inaxes.get_xlabel() or 'x'
        ly = ev.inaxes.get_ylabel() or 'y'
        self.app.status.set('%s = %.4g    |    %s = %.4g' % (lx, ev.xdata, ly, ev.ydata))

    def _on_click(self, ev):
        if ev.inaxes is None or ev.xdata is None or ev.button != 1:
            return
        if self.canvas.toolbar is not None and self.canvas.toolbar.mode:
            return                        # aktywne powiększanie/przesuwanie
        self.on_click(ev)

    def on_click(self, ev):
        pass

    def compute_and_plot(self):
        raise NotImplementedError


# =============================================================================
# Zakladka 0: Start
# =============================================================================

START_TEXT = """
WITAJ!  Ta aplikacja pokazuje WSZYSTKIE przykłady liczbowe z załączonych rozdziałów
podręcznika "Power System Analysis, Operation and Control":

  Rozdział 11 - Ekonomiczna praca systemu  (przykłady 11.1 - 11.8 + ćwiczenie 11)
  Rozdział 12 - Przesył HVDC                (przykłady 12.1 - 12.3)
  Rozdział 13 - Stabilność                  (przykłady 13.1 - 13.3 + kryterium równych pól)

JAK KORZYSTAĆ
  1. Wybierz zakładkę z przykładem (góra okna).
  2. Po lewej są SUWAKI - domyślnie ustawione na dane z książki. Przesuń suwak,
     a wynik i wykres przelicza się natychmiast.
  3. "Wyniki krok po kroku" - pełne rozwiązanie z Twoimi liczbami, jak w zeszycie.
  4. "Wyjaśnienie dla ucznia" - o co chodzi, bez skomplikowanej matematyki,
     z analogiami z życia i propozycjami eksperymentów.
  5. Pod wykresem jest pasek narzędzi: lupa (powiększanie), krzyżyk (przesuwanie),
     dyskietka (zapis PNG), domek (powrót do widoku).
  6. Rusz myszką nad wykresem - na dole okna zobaczysz dokładne współrzędne.
  7. W niektórych zakładkach KLIKNIĘCIE w wykres ustawia parametr (opis obok przycisku).

SŁOWNICZEK (najważniejsze pojęcia)
  MW         - megawat, milion watów. Czajnik ma ok. 2000 W = 0,002 MW.
  Koszt C(P) - ile kosztuje godzina pracy elektrowni produkującej moc P.
  IFC / IC   - przyrostowy koszt paliwa = ile kosztuje "jeszcze jeden megawat".
               Matematycznie: pochodna dC/dP (nachylenie krzywej kosztu).
  lambda     - "cena krańcowa" energii w systemie. W optimum wszystkie elektrownie
               maja TAKI SAM przyrostowy koszt równy lambda.
  P_L        - straty mocy w liniach (przewody się grzeją: P = I^2 R).
  Wsp. kary  - (penalty factor) "kara" dla elektrowni daleko od odbiorców,
               bo część ich energii ginie po drodze w liniach.
  HVDC       - przesył energii prądem stałym wysokiego napięcia.
  alfa       - kąt zapłonu prostownika (jak "opóźnienie włącznika").
  delta (12) - kąt wygaszania falownika (margines bezpieczeństwa komutacji).
  delta (13) - kąt mocy generatora: o ile wirnik "wyprzedza" sieć.
  H          - stała bezwładności: ile sekund wirnik mógłby oddawać moc znamionowa
               z samej energii obrotu.
  p.u.       - "per unit", wartości względne (1,0 = wartość znamionowa).

UWAGA INŻYNIERSKA
  W załączniku część wzorów była obrazkami i nie została odczytana. Brakujące dane
  odtworzono z rozwiązań w tekście (np. z "l1 = 3 + 0.015 x 60" wynika IFC1).
  Tam, gdzie nie dało się ich odtworzyć (np. stałe koszty a w 11.2 i 11.3,
  dane do przykładu 11.4), przyjęto rozsądne wartości i oznaczono je jako
  ZAŁOŻENIE. Stałe a NIE wpływają na rozdział mocy - tylko na koszt całkowity.
  Wykryte błędy drukarskie książki opisano w odpowiednich zakładkach.
"""


class StartTab(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master)
        txt = ScrolledText(self, wrap=tk.WORD, font=('Consolas', 10))
        txt.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        txt.insert('1.0', START_TEXT.strip())
        txt.configure(state=tk.DISABLED)


# =============================================================================
# Rozdział 11 - ekonomiczna praca
# =============================================================================

class Ex11_1(ExampleTab):
    title = 'Przykład 11.1 - Przyrostowy koszt paliwa (IFC) bloku 100 MW'
    params = [
        ('a', 'a  [tys. Rs/h]', 0, 1000, 300, 10),
        ('b', 'b  [tys. Rs/MWh]', 10, 150, 75, 1),
        ('c', 'c  [tys. Rs/MW^2h]', 0.0, 0.2, 0.01, 0.001),
        ('pmax', 'Moc znamionowa [MW]', 20, 300, 100, 5),
        ('load', 'Obciążenie [% mocy]', 10, 100, 50, 1),
    ]
    click_hint = 'w wykres ustawia obciążenie'
    explanation = """
O CO CHODZI?
Elektrownia spala węgiel, żeby produkować prąd. Im więcej megawatów (MW) chcemy,
tym więcej paliwa spalamy. Książka podaje wzór na koszt godziny pracy:

      C(P) = (300 + 75 P + 0,01 P^2) x 1000  Rs/h

  * 300 - koszt "stały": elektrownia kosztuje nawet gdy prawie nic nie produkuje
    (rozruch kotła, pompy, ludzie).
  * 75 P - każdy MW kosztuje mniej więcej tyle samo.
  * 0,01 P^2 - przy dużym obciążeniu blok pracuje trochę mniej sprawnie, więc
    każdy kolejny MW jest odrobinę droższy.

PRZYROSTOWY KOSZT (IFC)
To odpowiedź na pytanie: "ile zapłacę za JESZCZE JEDEN megawat?". To nachylenie
krzywej kosztu, czyli pochodna:

      IFC = dC/dP = (75 + 0,02 P) x 1000  Rs/MWh

Analogia: jedziesz taksówką. Opłata startowa to "a", cena za kilometr to "b".
IFC to cena następnego kilometra.

ROZWIĄZANIE Z KSIĄŻKI
  a) 100% (P = 100 MW):  IFC = 75 + 0,02*100 = 77  -> 77 000 Rs/MWh
  b)  50% (P =  50 MW):  IFC = 75 + 0,02*50  = 76  -> 76 000 Rs/MWh
Wniosek: przy zmianie obciążenia o połowę IFC zmienił się tylko o ok. 1,3%,
bo wyraz c = 0,01 jest bardzo mały - krzywa kosztu jest prawie prosta.

CO ZOBACZYSZ NA WYKRESIE
  Góra: krzywa kosztu C(P) i styczna w punkcie pracy - nachylenie stycznej TO jest IFC.
  Dół: IFC(P) - linia prosta, oraz średni koszt C/P (maleje, bo koszt stały
       rozkłada się na więcej MW).

EKSPERYMENT
  * Zwiększ c do 0,1 - zobacz, jak bardzo IFC zaczyna zależeć od obciążenia.
  * Klikaj w wykres - ustawisz obciążenie w miejscu kliknięcia.
"""

    def on_click(self, ev):
        self.setv('load', 100.0 * ev.xdata / self.v('pmax'))

    def compute_and_plot(self):
        a, b, c, pmax, load = (self.v(k) for k in ('a', 'b', 'c', 'pmax', 'load'))
        p_op = pmax * load / 100.0
        P = np.linspace(0, pmax, 300)
        C = quad_cost(a, b, c, P)
        ic = quad_ic(b, c, P)
        ic100 = quad_ic(b, c, pmax)
        ic50 = quad_ic(b, c, pmax / 2)
        icop = quad_ic(b, c, p_op)
        cop = quad_cost(a, b, c, p_op)

        ax1 = self.fig.add_subplot(211)
        ax1.plot(P, C, color=C_BLUE, lw=2, label='C(P)')
        tang = cop + icop * (P - p_op)
        ax1.plot(P, tang, '--', color=C_RED, label='styczna (nachylenie = IFC)')
        ax1.plot([p_op], [cop], 'o', color=C_RED, ms=8)
        ax1.set_ylabel('Koszt C [tys. Rs/h]')
        ax1.set_title('Krzywa kosztu paliwa i styczna w punkcie pracy')
        ax1.grid(alpha=.3)
        ax1.legend(loc='upper left')

        ax2 = self.fig.add_subplot(212)
        ax2.plot(P, ic, color=C_GREEN, lw=2, label='IFC = b + 2cP')
        Pnz = P[P > 1]
        ax2.plot(Pnz, quad_cost(a, b, c, Pnz) / Pnz, color=C_ORANGE, lw=1.5,
                 label='średni koszt C/P')
        for pp, lab in ((pmax, '100%'), (pmax / 2, '50%'), (p_op, 'punkt pracy')):
            ax2.plot([pp], [quad_ic(b, c, pp)], 'o', ms=7)
            ax2.annotate('%s: %.2f' % (lab, quad_ic(b, c, pp)), (pp, quad_ic(b, c, pp)),
                         textcoords='offset points', xytext=(-30, 10), fontsize=8)
        ax2.set_ylim(0, max(ic.max(), b) * 1.6)
        ax2.set_xlabel('Moc P [MW]')
        ax2.set_ylabel('[tys. Rs/MWh]')
        ax2.grid(alpha=.3)
        ax2.legend(loc='upper right')

        return (
            'DANE: C(P) = (%.0f + %.2f P + %.4f P^2) x 10^3 Rs/h,  Pn = %.0f MW\n\n'
            'Krok 1 - pochodna (IFC):\n'
            '  IFC = dC/dP = (%.2f + 2*%.4f*P) x 10^3 = (%.2f + %.4f P) x 10^3 Rs/MWh\n\n'
            'Krok 2 - obciążenie 100%% (P = %.1f MW):\n'
            '  IFC = %.2f + %.4f*%.1f = %.3f  -> %.0f Rs/MWh\n\n'
            'Krok 3 - obciążenie 50%% (P = %.1f MW):\n'
            '  IFC = %.2f + %.4f*%.1f = %.3f  -> %.0f Rs/MWh\n\n'
            'Różnica 100%% vs 50%%: %.3f x 10^3 Rs/MWh (%.2f %%)\n\n'
            'Twój punkt pracy: P = %.1f MW (%.0f %%)\n'
            '  koszt C = %.1f x 10^3 Rs/h,  IFC = %.3f x 10^3 Rs/MWh\n'
            '  średni koszt C/P = %.3f x 10^3 Rs/MWh\n'
            '\nKsiążka: a) 77 000 Rs/MWh,  b) 76 000 Rs/MWh.'
        ) % (a, b, c, pmax, b, c, b, 2 * c, pmax, b, 2 * c, pmax, ic100, ic100 * 1e3,
             pmax / 2, b, 2 * c, pmax / 2, ic50, ic50 * 1e3, ic100 - ic50,
             100 * (ic100 - ic50) / ic100, p_op, load, cop, icop,
             cop / p_op if p_op > 0 else float('nan'))


class Ex11_2(ExampleTab):
    title = 'Przykład 11.2 - Ekonomiczny rozdział obciążenia dwóch turbogeneratorów'
    params = [
        ('load', 'Obciążenie szyny P_R [MW]', 40, 200, 160, 1),
        ('b1', 'G1: b1 (IFC1 = b1 + 2c1 P1)', 1, 6, 3, 0.1),
        ('c1', 'G1: c1', 0.001, 0.02, 0.0075, 0.0005),
        ('b2', 'G2: b2', 1, 6, 2, 0.1),
        ('c2', 'G2: c2', 0.001, 0.02, 0.009, 0.0005),
        ('limits', 'Limity włączone? (0 = nie, 1 = tak)', 0, 1, 0, 1),
        ('p1max', 'P1 max [MW]', 50, 200, 125, 5),
        ('p2max', 'P2 max [MW]', 50, 200, 100, 5),
    ]
    click_hint = 'w dolny wykres wybiera podział P1'
    explanation = """
O CO CHODZI?
Dwa generatory razem zasilają odbiorców mocą 160 MW. Pytanie: ile ma dać każdy,
żeby razem zapłacić za paliwo jak najmniej?

Z rozwiązania w książce wynikają przyrostowe koszty:
     IFC1 = 3 + 0,015 P1        IFC2 = 2 + 0,018 P2
(czyli C1 = a1 + 3P1 + 0,0075 P1^2,  C2 = a2 + 2P2 + 0,009 P2^2;
 stałe a1 = 100, a2 = 80 to ZAŁOŻENIE - na podział mocy nie wpływają).

ZASADA RÓWNYCH PRZYROSTOWYCH KOSZTÓW
Wyobraź sobie, że G1 daje 1 MW więcej, a G2 1 MW mniej. Jeśli IFC1 > IFC2,
to właśnie przepłaciłeś - opłaca się przesunąć moc do tańszego G2. Przesuwamy
tak długo, aż "następny MW" kosztuje w obu tyle samo:

     IFC1 = IFC2 = lambda,     P1 + P2 = 160

To jak dwie kolejki w sklepie - ludzie przechodzą do krótszej, aż obie są równe.

ROZWIĄZANIE a) (bez limitów)
  3 + 0,015 (160 - P2) = 2 + 0,018 P2   ->  3,4 = 0,033 P2
  P2 = 103,03 MW,  P1 = 56,97 MW,  lambda = 3,854

ROZWIĄZANIE b) (P2 max = 100 MW)
  Optimum chciało 103,03 MW, ale G2 nie może więcej niż 100 -> P2 = 100 MW
  (blok "przyklejony" do limitu), reszta: P1 = 60 MW.
  Teraz IFC1 = 3,9 i IFC2 = 3,8 - NIE są równe, bo ograniczenie nie pozwala
  wyrównać. To najlepsze, co MOŻNA zrobić, ale droższe niż w a).

CO ZOBACZYSZ NA WYKRESIE
  Góra: proste IFC1(P1) i IFC2(P2) oraz pozioma linia lambda - punkty pracy
        leżą na niej (gdy brak limitów).
  Dół: koszt całkowity w funkcji podziału P1 (P2 = P_R - P1). Minimum paraboli
       to rozdział ekonomiczny. Kliknięcie pokazuje koszt Twojego podziału.

EKSPERYMENT
  * Ustaw "Limity" = 1 - zobaczysz, jak punkt zostaje "odcięty" na 100 MW.
  * Zmniejsz c2 - tańszy G2 bierze więcej obciążenia.
"""

    def __init__(self, master, app):
        self.p1_click = None
        super().__init__(master, app)

    def reset_extra(self):
        self.p1_click = None

    def on_click(self, ev):
        if ev.inaxes is self.ax_cost:
            self.p1_click = ev.xdata
            self.schedule()

    def compute_and_plot(self):
        load = self.v('load')
        lim = self.v('limits') >= 0.5
        units = [dict(a=100, b=self.v('b1'), c=self.v('c1'), pmin=0.0,
                      pmax=self.v('p1max') if lim else 1e4),
                 dict(a=80, b=self.v('b2'), c=self.v('c2'), pmin=0.0,
                      pmax=self.v('p2max') if lim else 1e4)]
        r = lambda_dispatch(units, load)
        free = lambda_dispatch([dict(u, pmax=1e4) for u in units], load)
        p1, p2 = r['P']
        ic1 = quad_ic(units[0]['b'], units[0]['c'], p1)
        ic2 = quad_ic(units[1]['b'], units[1]['c'], p2)

        ax = self.fig.add_subplot(211)
        P = np.linspace(0, load, 200)
        ax.plot(P, quad_ic(units[0]['b'], units[0]['c'], P), color=C_BLUE, lw=2, label='IFC1(P1)')
        ax.plot(P, quad_ic(units[1]['b'], units[1]['c'], P), color=C_RED, lw=2, label='IFC2(P2)')
        ax.axhline(free['lam'], color=C_GREEN, ls='--', label='lambda = %.3f' % free['lam'])
        ax.plot([p1], [ic1], 'o', color=C_BLUE, ms=9)
        ax.plot([p2], [ic2], 's', color=C_RED, ms=9)
        if lim:
            ax.axvline(units[1]['pmax'], color=C_RED, ls=':', label='P2 max')
            ax.axvline(units[0]['pmax'], color=C_BLUE, ls=':', label='P1 max')
        ax.set_xlabel('Moc bloku [MW]')
        ax.set_ylabel('IFC [Rs/MWh]')
        ax.set_title('Przyrostowe koszty i punkt pracy')
        ax.grid(alpha=.3)
        ax.legend(fontsize=8, loc='upper left')

        ax2 = self.fig.add_subplot(212)
        self.ax_cost = ax2
        P1 = np.linspace(0, load, 300)
        tot = quad_cost(100, units[0]['b'], units[0]['c'], P1) + \
            quad_cost(80, units[1]['b'], units[1]['c'], load - P1)
        ax2.plot(P1, tot, color=C_PURPLE, lw=2, label='C1 + C2')
        ax2.plot([free['P'][0]], [free['cost']], 'v', color=C_GREEN, ms=10,
                 label='minimum bez limitów')
        ax2.plot([p1], [r['cost']], 'o', color=C_ORANGE, ms=8, label='wynik (z limitami)')
        if lim:
            bad = (P1 > units[0]['pmax']) | ((load - P1) > units[1]['pmax'])
            ax2.fill_between(P1, tot.min(), tot.max(), where=bad, color='grey',
                             alpha=.2, label='niedozwolone (limity)')
        extra = ''
        if self.p1_click is not None:
            pc = min(max(self.p1_click, 0), load)
            cc = quad_cost(100, units[0]['b'], units[0]['c'], pc) + \
                quad_cost(80, units[1]['b'], units[1]['c'], load - pc)
            ax2.plot([pc], [cc], 'x', color='k', ms=10, mew=2, label='Twój wybór')
            extra = ('\nTwój podział (klik): P1 = %.1f, P2 = %.1f MW, koszt = %.2f Rs/h,\n'
                     '  czyli o %.2f Rs/h drożej niż optimum.' % (pc, load - pc, cc, cc - free['cost']))
        ax2.set_xlabel('P1 [MW]   (P2 = P_R - P1)')
        ax2.set_ylabel('Koszt całkowity [Rs/h]')
        ax2.grid(alpha=.3)
        ax2.legend(fontsize=8)

        s = ('IFC1 = %.2f + %.4f P1,   IFC2 = %.2f + %.4f P2,   P_R = %.0f MW\n\n'
             'a) BEZ LIMITÓW - równamy IFC1 = IFC2 = lambda:\n'
             '   %.2f + %.4f (P_R - P2) = %.2f + %.4f P2\n'
             '   P1 = %.2f MW,  P2 = %.2f MW,  lambda = %.4f\n'
             '   koszt = %.2f Rs/h (z założonymi a1=100, a2=80)\n'
             % (units[0]['b'], 2 * units[0]['c'], units[1]['b'], 2 * units[1]['c'], load,
                units[0]['b'], 2 * units[0]['c'], units[1]['b'], 2 * units[1]['c'],
                free['P'][0], free['P'][1], free['lam'], free['cost']))
        if lim:
            s += ('\nb) Z LIMITAMI (P1 <= %.0f, P2 <= %.0f):\n'
                  '   P1 = %.2f MW %s,  P2 = %.2f MW %s\n'
                  '   IFC1 = %.3f,  IFC2 = %.3f  %s\n'
                  '   koszt = %.2f Rs/h  (+%.2f Rs/h względem a)\n'
                  % (units[0]['pmax'], units[1]['pmax'], p1,
                     '(NA LIMICIE)' if r['at_limit'][0] else '', p2,
                     '(NA LIMICIE)' if r['at_limit'][1] else '', ic1, ic2,
                     '-> różne, bo limit blokuje wyrównanie' if abs(ic1 - ic2) > 1e-3 else '-> równe',
                     r['cost'], r['cost'] - free['cost']))
        else:
            s += '\n(Ustaw "Limity" = 1, aby zobaczyć część b).)\n'
        s += '\nKsiążka: a) P1 = 56,97; P2 = 103,03 MW; lambda = 3,854\n' \
             '         b) P1 = 60; P2 = 100 MW; IFC1 = 3,9; IFC2 = 3,8' + extra
        return s


class Ex11_3(ExampleTab):
    title = 'Przykład 11.3 - Dobowy harmonogram, wyłączanie bloków (unit commitment)'
    params = [
        ('L1', 'Obciążenie szczytowe [MW]', 900, 1600, 1200, 10),
        ('T1', '   czas trwania [h]', 0, 24, 8, 1),
        ('L2', 'Obciążenie średnie [MW]', 400, 1600, 900, 10),
        ('T2', '   czas trwania [h]', 0, 24, 6, 1),
        ('L3', 'Obciążenie nocne [MW]', 200, 1000, 500, 10),
        ('T3', '   czas trwania [h]', 0, 24, 10, 1),
        ('a1', 'G1: koszt stały a1 [$/h] (ZAŁOŻENIE)', 0, 10000, 4000, 100),
        ('start', 'Koszt wyłączenia + rozruchu [$]', 0, 50000, 1000, 500),
    ]
    explanation = """
O CO CHODZI?
W ciągu doby zapotrzebowanie na prąd się zmienia: w dzień jest duże (1200 MW),
wieczorem średnie (900 MW), w nocy małe (500 MW). Mamy dwa bloki po 200...800 MW.
Pytanie: czy w nocy opłaca się trzymać OBA bloki w ruchu, czy jeden wyłączyć?

DANE (odtworzone z rozwiązania):
   IFC1 = 52,8 + 0,011 P1      ->  C1 = a1 + 52,8 P1 + 0,0055 P1^2
   IFC2 = 15 + 0,1 P2          ->  C2 = 3700 + 15 P2 + 0,05 P2^2   $/h
   Limity: 200 <= P <= 800 MW. Koszt wyłączenia i ponownego uruchomienia: 1000 $.
   a1 - nieczytelne w załączniku -> ZAŁOŻENIE 4000 $/h (suwak).
   Czasy trwania z rysunku E11.1 są nieznane, poza 10 h dla 500 MW - przyjęto
   8 h (1200 MW) i 6 h (900 MW).

KROK PO KROKU
  1200 MW: jeden blok (max 800) nie da rady -> oba. Równamy IFC:
           52,8 + 0,011 P1 = 15 + 0,1 (1200 - P1) -> P1 = 740,5; P2 = 459,5 MW
  900 MW:  też oba:  P1 = 470,3; P2 = 429,7 MW
  500 MW:  tu jest wybór!
     - oba w ruchu: z równania wyszłoby P1 = 109,9 MW, ale to PONIŻEJ minimum
       200 MW, więc P1 = 200, P2 = 300 MW,
     - tylko G2 (500 MW): drożej za MW, ale nie płacimy kosztu stałego G1,
     - doliczamy 1000 $ za wyłączenie i ponowny rozruch G1.
  Wygrywa wariant z najmniejszym kosztem łącznym -> w książce: TYLKO G2.

ANALOGIA
Masz dwa samochody. Na długie trasy z rodzina jedziecie dwoma. Na zakupy
pojedzie jeden - nawet jeśli to akurat ten trochę droższy w spalaniu,
bo drugiego nie trzeba "utrzymywać w ruchu".

BŁĘDY W KSIĄŻCE (spostrzeżenie inżyniera)
  * Dla 1200 MW wydrukowano P2 = 429,73 MW; poprawnie 1200 - 740,54 = 459,46 MW.
  * Dla 500 MW książka nie sprawdza limitu 200 MW (P1 = 109,9 MW jest niedozwolone)
    i liczby kosztów są pomieszane. Aplikacja liczy to poprawnie - wniosek
    (zostawić tylko G2) się nie zmienia.

CO ZOBACZYSZ NA WYKRESIE
  Góra: uporządkowana krzywa obciążenia z podziałem na G1 i G2.
  Dół: koszt każdego wariantu dla okresu nocnego.
To jest zalążek problemu "Unit Commitment" - które bloki włączyć i kiedy.
"""

    def compute_and_plot(self):
        b1, c1 = 52.8, 0.0055
        units = [dict(a=self.v('a1'), b=b1, c=c1, pmin=200, pmax=800),
                 dict(a=3700, b=15, c=0.05, pmin=200, pmax=800)]
        start = self.v('start')
        periods = [(self.v('L1'), self.v('T1')), (self.v('L2'), self.v('T2')),
                   (self.v('L3'), self.v('T3'))]
        periods.sort(key=lambda x: -x[0])
        s = ('C1 = %.0f + 52,8 P1 + 0,0055 P1^2,  C2 = 3700 + 15 P2 + 0,05 P2^2  [$/h]\n'
             'Limity 200..800 MW, koszt wyłącz.+rozruch = %.0f $\n\n' % (units[0]['a'], start))
        sched = []
        total = 0.0
        hours = sum(t for _, t in periods)
        for load, hrs in periods:
            opts = uc_period(units, load)
            s += 'OKRES %.0f MW przez %.0f h:\n' % (load, hrs)
            if not opts:
                s += '  BRAK wykonalnego wariantu (poza zakresem 200..1600 MW)!\n\n'
                sched.append((load, hrs, [0, 0], 'brak', {}))
                continue
            scored = {}
            for name, (cost_h, ps) in opts.items():
                extra = start if name != 'oba' else 0.0
                scored[name] = cost_h * hrs + extra
                s += '  %-8s P1 = %6.1f  P2 = %6.1f MW  koszt %9.0f $/h x %2.0f h%s = %10.0f $\n' % (
                    name, ps[0], ps[1], cost_h, hrs,
                    ' + %.0f' % extra if extra else '', scored[name])
            best = min(scored, key=scored.get)
            total += scored[best]
            s += '  -> NAJTANIEJ: %s\n\n' % best
            sched.append((load, hrs, opts[best][1], best, scored))
        s += 'KOSZT DOBOWY (wybrane warianty): %.0f $\n' % total
        if abs(hours - 24) > 0.5:
            s += 'Uwaga: suma czasów = %.0f h (nie 24 h).\n' % hours
        s += ('\nKsiążka: 1200 MW -> P1 740,54 / P2 459,46 (druk: 429,73 - błąd);\n'
              '900 MW -> 470,27 / 429,73;  500 MW -> tylko G2.')

        ax = self.fig.add_subplot(211)
        t0 = 0.0
        for load, hrs, ps, best, _sc in sched:
            ax.bar(t0, ps[0], width=hrs, align='edge', color=C_BLUE, alpha=.8,
                   label='G1' if t0 == 0 else None, edgecolor='k')
            ax.bar(t0, ps[1], width=hrs, bottom=ps[0], align='edge', color=C_ORANGE,
                   alpha=.8, label='G2' if t0 == 0 else None, edgecolor='k')
            ax.text(t0 + hrs / 2, load + 20, '%s\n%.0f MW' % (best, load),
                    ha='center', fontsize=8)
            t0 += hrs
        ax.set_xlim(0, max(t0, 1))
        ax.set_ylim(0, max(p[0] for p in periods) * 1.25)
        ax.set_xlabel('Czas [h] (krzywa uporządkowana)')
        ax.set_ylabel('Moc [MW]')
        ax.set_title('Harmonogram bloków')
        ax.grid(alpha=.3, axis='y')
        ax.legend(loc='upper right')

        ax2 = self.fig.add_subplot(212)
        names, vals, cols = [], [], []
        for load, hrs, _ps, best, sc in sched:
            for nm, val in sc.items():
                names.append('%.0f MW\n%s' % (load, nm))
                vals.append(val / 1000.0)
                cols.append(C_GREEN if nm == best else C_GREY)
        ax2.bar(range(len(vals)), vals, color=cols)
        ax2.set_xticks(range(len(vals)))
        ax2.set_xticklabels(names, fontsize=7)
        ax2.set_ylabel('Koszt okresu [tys. $]')
        ax2.set_title('Porównanie wariantów (zielony = najtańszy)')
        ax2.grid(alpha=.3, axis='y')
        return s


class Ex11_4(ExampleTab):
    title = 'Przykład 11.4 - Przyrostowe straty w systemie dwuobszarowym A-B'
    params = [
        ('B', 'Wsp. strat B_BB [1/MW] x 1e-4', 0, 10, 4, 0.1),
        ('PR', 'Łączne obciążenie P_R [MW]', 100, 900, 500, 10),
        ('bA', 'Obszar A: IC_A = bA + 2cA P', 2, 20, 8, 0.5),
        ('cA', '   cA', 0.001, 0.02, 0.005, 0.0005),
        ('bB', 'Obszar B: bB', 2, 20, 6, 0.5),
        ('cB', '   cB', 0.001, 0.02, 0.006, 0.0005),
    ]
    explanation = """
O CO CHODZI?
Dwa obszary (A i B) połączone linia. Odbiorcy są przy A. Moc z obszaru B musi
przejść linia, a linia się grzeje - część mocy "ginie". Książka mówi: straty
zależą TYLKO od mocy wytwarzanej w B:   P_L = f(P_gB).

WNIOSEK Z KSIĄŻKI
  * Przyrostowa strata obszaru A:  dP_L/dP_gA = 0  (A jest przy odbiorcach).
  * Przyrostowa strata obszaru B:  dP_L/dP_gB = df/dP_gB  (np. 2 B P_gB).
  * Warunek ekonomiczny:
        IC_A = lambda
        IC_B = lambda (1 - dP_L/dP_gB)   czyli   IC_B * PF_B = lambda
    gdzie PF_B = 1 / (1 - dP_L/dP_gB) to współczynnik kary.

ANALOGIA
Kupujesz jabłka w dwóch sadach. Sad B jest tańszy, ale po drodze część jabłek
się obija (straty). Płacisz więc za jabłko z B tak, jakby było droższe o "karę".

DANE LICZBOWE
Przykład w książce jest jakościowy (bez liczb). Wartości na suwakach to
ZAŁOŻENIE, żeby pokazać zjawisko na wykresie: P_L = B * P_gB^2.

CO ZOBACZYSZ
  Góra: straty P_L, przyrostowa strata ITL_B i wsp. kary PF_B w funkcji P_gB.
  Dół: rozdział obciążenia z uwzględnieniem strat dla różnych B - im większe
       straty, tym mniej opłaca się produkować w B.

EKSPERYMENT
  Ustaw B = 0 - oba obszary podziela się jak w przykładzie 11.2. Potem zwiększaj B.
"""

    def compute_and_plot(self):
        B = self.v('B') * 1e-4
        PR = self.v('PR')
        uA = dict(a=0, b=self.v('bA'), c=self.v('cA'), pmin=0, pmax=1e4)
        uB = dict(a=0, b=self.v('bB'), c=self.v('cB'), pmin=0, pmax=1e4)
        r = lambda_dispatch([uA, uB], PR, [0.0, B])
        r0 = lambda_dispatch([uA, uB], PR, [0.0, 0.0])
        pa, pb = r['P']
        itl = 2 * B * pb
        pf = 1 / (1 - itl) if itl < 1 else float('inf')

        ax = self.fig.add_subplot(211)
        Pg = np.linspace(0, PR, 300)
        ax.plot(Pg, B * Pg ** 2, color=C_RED, lw=2, label='P_L = B P_gB^2 [MW]')
        ax.set_xlabel('Moc obszaru B, P_gB [MW]')
        ax.set_ylabel('Straty [MW]')
        ax.grid(alpha=.3)
        axr = ax.twinx()
        itl_c = 2 * B * Pg
        pf_c = np.where(itl_c < 0.95, 1 / (1 - np.minimum(itl_c, 0.95)), np.nan)
        axr.plot(Pg, itl_c, color=C_BLUE, ls='--', label='ITL_B = dP_L/dP_gB')
        axr.plot(Pg, pf_c, color=C_GREEN, label='PF_B = 1/(1-ITL_B)')
        axr.axhline(0, color=C_ORANGE, ls=':', label='ITL_A = 0')
        axr.set_ylabel('ITL [-],  PF [-]')
        ax.axvline(pb, color='k', ls=':')
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = axr.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=8, loc='upper left')
        ax.set_title('Straty zależą tylko od generacji w B')

        ax2 = self.fig.add_subplot(212)
        Bs = np.linspace(0, 10e-4, 40)
        PAs, PBs, PLs = [], [], []
        for bb in Bs:
            rr = lambda_dispatch([uA, uB], PR, [0.0, bb])
            PAs.append(rr['P'][0])
            PBs.append(rr['P'][1])
            PLs.append(rr['PL'])
        ax2.plot(Bs * 1e4, PAs, color=C_BLUE, lw=2, label='P_gA')
        ax2.plot(Bs * 1e4, PBs, color=C_RED, lw=2, label='P_gB')
        ax2.plot(Bs * 1e4, PLs, color=C_GREY, lw=1.5, label='P_L')
        ax2.axvline(B * 1e4, color='k', ls=':')
        ax2.set_xlabel('B x 1e-4 [1/MW]')
        ax2.set_ylabel('Moc [MW]')
        ax2.set_title('Optymalny rozdział przy rosnących stratach')
        ax2.grid(alpha=.3)
        ax2.legend(fontsize=8)

        return ('Model: P_L = B P_gB^2,  B = %.2e 1/MW,  P_R = %.0f MW\n\n'
                'Przyrostowe straty:\n'
                '  ITL_A = dP_L/dP_gA = 0\n'
                '  ITL_B = dP_L/dP_gB = 2 B P_gB = %.4f\n'
                '  PF_A = 1,  PF_B = 1/(1 - ITL_B) = %.4f\n\n'
                'Warunek: IC_A = lambda,  IC_B = lambda (1 - ITL_B)\n'
                'Wynik (iteracja lambda):\n'
                '  P_gA = %.2f MW,  P_gB = %.2f MW,  P_L = %.2f MW\n'
                '  lambda = %.4f,  IC_A = %.4f,  IC_B*PF_B = %.4f\n\n'
                'Dla porównania bez strat (B = 0):\n'
                '  P_gA = %.2f MW,  P_gB = %.2f MW,  lambda = %.4f\n'
                'Straty "przesuwają" produkcje z B do A o %.2f MW.'
                % (B, PR, itl, pf, pa, pb, r['PL'], r['lam'], quad_ic(uA['b'], uA['c'], pa),
                   quad_ic(uB['b'], uB['c'], pb) * pf, r0['P'][0], r0['P'][1], r0['lam'],
                   r0['P'][1] - pb))


class Ex11_5(ExampleTab):
    title = 'Przykład 11.5 - Współczynniki kary (penalty factors)'
    params = [
        ('P1', 'P1 [MW]', 50, 600, 325, 5),
        ('P2', 'P2 [MW]', 50, 600, 175, 5),
        ('itl2', 'dP_L/dP2 (przyrostowa strata G2)', 0, 0.6, 0.2, 0.01),
        ('b1', 'IC1 = b1 + k1 P1:  b1', 0, 10, 2.5, 0.1),
        ('k1', '   k1', 0.0, 0.1, 0.02, 0.001),
        ('b2', 'IC2 = b2 + k2 P2:  b2', 0, 10, 2.0, 0.1),
        ('k2', '   k2', 0.0, 0.1, 0.04, 0.001),
    ]
    explanation = """
O CO CHODZI?
Dwa bloki pracują w optimum ekonomicznym: P1 = 325 MW, P2 = 175 MW.
Wiemy, że przyrostowa strata G2 wynosi dP_L/dP2 = 0,2 - czyli z każdego
dodatkowego MW z G2 aż 0,2 MW ginie w liniach.
Szukamy współczynników kary PF1 i PF2.

WZORY
   PF_i = 1 / (1 - dP_L/dP_i)
   Warunek optimum ze stratami:  PF1 * IC1 = PF2 * IC2 = lambda
   (IC1 = 2,5 + 0,02 P1 oraz IC2 = 2 + 0,04 P2 - odczytane z rozwiązania)

KROK PO KROKU
  1) PF2 = 1 / (1 - 0,2) = 1,25
  2) IC1 = 2,5 + 0,02*325 = 9,0     IC2 = 2 + 0,04*175 = 9,0
  3) PF1 * 9,0 = 1,25 * 9,0   ->   PF1 = 1,25
  4) lambda = 11,25;  z PF1 wynika też dP_L/dP1 = 1 - 1/PF1 = 0,2

ANALOGIA
Współczynnik kary to jak cena przesyłki: towar z dalekiego magazynu musisz
"przeliczyć" - płacisz za towar i za to, co zginęło po drodze.
PF = 1,25 oznacza: żeby dostarczyć 1 MW, trzeba wyprodukować 1,25 MW... prawie
(dokładnie: przyrostowo każdy MW kosztuje 25% więcej).

CO ZOBACZYSZ
  Lewo: słupki IC i PF*IC - po pomnożeniu przez karę oba są równe lambda.
  Prawo: jak PF rośnie, gdy rosną przyrostowe straty (bardzo szybko przy ITL->1).
"""

    def compute_and_plot(self):
        P1, P2, itl2 = self.v('P1'), self.v('P2'), self.v('itl2')
        ic1 = self.v('b1') + self.v('k1') * P1
        ic2 = self.v('b2') + self.v('k2') * P2
        if ic1 <= 0 or ic2 <= 0:
            return 'Przyrostowe koszty muszą być dodatnie - zwiększ b lub k.'
        pf2 = 1 / (1 - itl2)
        lam = pf2 * ic2
        pf1 = lam / ic1
        itl1 = 1 - 1 / pf1

        ax = self.fig.add_subplot(121)
        x = np.arange(2)
        ax.bar(x - 0.2, [ic1, ic2], 0.4, color=C_BLUE, label='IC (bez kary)')
        ax.bar(x + 0.2, [pf1 * ic1, pf2 * ic2], 0.4, color=C_ORANGE, label='PF * IC')
        ax.axhline(lam, color=C_GREEN, ls='--', label='lambda = %.3f' % lam)
        ax.set_xticks(x)
        ax.set_xticklabels(['G1\nPF=%.3f' % pf1, 'G2\nPF=%.3f' % pf2])
        ax.set_ylabel('[Rs/MWh]')
        ax.set_title('Wyrównanie kosztów z kara')
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(alpha=.3, axis='y')

        ax2 = self.fig.add_subplot(122)
        itl = np.linspace(-0.3, 0.8, 300)
        ax2.plot(itl, 1 / (1 - itl), color=C_PURPLE, lw=2)
        ax2.plot([itl2], [pf2], 'o', color=C_RED, ms=9, label='G2')
        ax2.plot([itl1], [pf1], 's', color=C_BLUE, ms=8, label='G1 (wyliczone)')
        ax2.axvline(0, color='k', lw=.6)
        ax2.set_xlabel('dP_L/dP_i  (przyrostowa strata)')
        ax2.set_ylabel('PF = 1/(1 - dP_L/dP_i)')
        ax2.set_title('Współczynnik kary')
        ax2.grid(alpha=.3)
        ax2.legend(fontsize=8)

        return ('Dane: P1 = %.0f MW, P2 = %.0f MW, dP_L/dP2 = %.3f\n'
                'IC1 = %.2f + %.3f P1,  IC2 = %.2f + %.3f P2\n\n'
                'Krok 1:  PF2 = 1/(1 - %.3f) = %.4f\n'
                'Krok 2:  IC1 = %.3f,  IC2 = %.3f\n'
                'Krok 3:  PF1 * IC1 = PF2 * IC2  ->  PF1 = %.4f * %.3f / %.3f = %.4f\n'
                'Krok 4:  lambda = %.4f Rs/MWh\n'
                '         dP_L/dP1 = 1 - 1/PF1 = %.4f\n\n'
                'Interpretacja: %s\n\n'
                'Książka: PF1 = 1,25 (oraz PF2 = 1,25).'
                % (P1, P2, itl2, self.v('b1'), self.v('k1'), self.v('b2'), self.v('k2'),
                   itl2, pf2, ic1, ic2, pf2, ic2, ic1, pf1, lam, itl1,
                   'PF1 < 1 -> zwiększanie P1 ZMNIEJSZA straty (np. blok przy odbiorcach).'
                   if pf1 < 1 else 'PF1 >= 1 -> zwiększanie P1 zwiększa straty.'))


class Ex11_6(ExampleTab):
    title = 'Przykład 11.6 - Dwie elektrownie i linia: równania koordynacyjne'
    params = [
        ('lam', 'lambda - koszt mocy odebranej [Rs/MWh]', 55, 150, 90, 0.5),
        ('Pref', 'Moc testowa z G1 [MW]', 50, 400, 175, 5),
        ('PLref', 'Straty przy tej mocy [MW]', 1, 60, 15, 0.5),
        ('k1', 'IC1 = k1 P1 + b1:  k1', 0.05, 1, 0.30, 0.01),
        ('b1', '   b1', 0, 80, 50, 1),
        ('k2', 'IC2 = k2 P2 + b2:  k2', 0.05, 1, 0.40, 0.01),
        ('b2', '   b2', 0, 80, 30, 1),
    ]
    click_hint = 'w wykres ustawia lambda'
    explanation = """
O CO CHODZI?
Elektrownia 2 stoi przy odbiorcach, elektrownia 1 jest daleko i łączy ją linia.
Gdy G1 wysyła 175 MW, w linii ginie 15 MW (w książce omyłkowo "175 kW").
Znamy cenę mocy odebranej lambda = 90 Rs/MWh. Szukamy P1, P2 i obciążenia P_R.

Przyrostowe koszty (odczytane z rozwiązania):
   IC1 = 0,30 P1 + 50       IC2 = 0,40 P2 + 30   [Rs/MWh]

KROK 1 - model strat
Straty zależą tylko od P1 (G2 jest przy odbiorcach), więc P_L = B11 P1^2:
   15 = B11 * 175^2   ->   B11 = 0,00049 1/MW

KROK 2a - równania koordynacyjne
   IC1 + lambda * dP_L/dP1 = lambda   ->  0,3 P1 + 50 + 90*2*0,00049 P1 = 90
   -> P1 = 103,1 MW
   IC2 + lambda * 0 = lambda          ->  0,4 P2 + 30 = 90   -> P2 = 150 MW

KROK 2b - metoda współczynnika kary
   PF1 = 1/(1 - 2 B11 P1);   PF1 * IC1 = lambda  -> to samo P1 = 103,1 MW
   (obie metody to te same równania, tylko inaczej zapisane - dlatego wynik ten sam)

KROK 3 - straty i obciążenie
   P_L = 0,00049 * 103,1^2 = 5,2 MW
   P_R = P1 + P2 - P_L = 103,1 + 150 - 5,2 = 247,9 MW

CO ZOBACZYSZ
  Wykres pokazuje jak P1, P2, P_L i P_R zmieniają się z cena lambda.
  Kliknij, aby ustawić lambda. Zauważ: P1 rośnie wolniej niż P2, bo G1 płaci karę
  za straty, które rosną z kwadratem mocy.
"""

    def on_click(self, ev):
        if ev.inaxes is self.ax_main:
            self.setv('lam', ev.xdata)

    def solve(self, lam, B):
        k1, b1, k2, b2 = (self.v(k) for k in ('k1', 'b1', 'k2', 'b2'))
        p1 = max((lam - b1) / (k1 + 2 * lam * B), 0.0)
        p2 = max((lam - b2) / k2, 0.0)
        pl = B * p1 * p1
        return p1, p2, pl, p1 + p2 - pl

    def compute_and_plot(self):
        lam = self.v('lam')
        B = self.v('PLref') / self.v('Pref') ** 2
        p1, p2, pl, pr = self.solve(lam, B)
        pf1 = 1 / (1 - 2 * B * p1)
        ic1 = self.v('k1') * p1 + self.v('b1')

        ax = self.fig.add_subplot(111)
        self.ax_main = ax
        L = np.linspace(max(self.v('b1'), self.v('b2')) + 0.1, 150, 300)
        sol = np.array([self.solve(x, B) for x in L])
        ax.plot(L, sol[:, 0], color=C_BLUE, lw=2, label='P1 (daleka)')
        ax.plot(L, sol[:, 1], color=C_RED, lw=2, label='P2 (przy odbiorcach)')
        ax.plot(L, sol[:, 2], color=C_GREY, lw=1.5, label='P_L straty')
        ax.plot(L, sol[:, 3], color=C_GREEN, lw=2, ls='--', label='P_R = P1 + P2 - P_L')
        ax.axvline(lam, color='k', ls=':')
        for val, col in ((p1, C_BLUE), (p2, C_RED), (pl, C_GREY), (pr, C_GREEN)):
            ax.plot([lam], [val], 'o', color=col)
        ax.set_xlabel('lambda [Rs/MWh]')
        ax.set_ylabel('Moc [MW]')
        ax.set_title('Rozwiązanie równań koordynacyjnych w funkcji lambda')
        ax.grid(alpha=.3)
        ax.legend()

        return ('Krok 1: B11 = P_L/P^2 = %.1f/%.0f^2 = %.6f 1/MW\n\n'
                'Krok 2a (równania koordynacyjne), lambda = %.2f:\n'
                '  %.2f P1 + %.1f + %.2f*2*%.6f P1 = %.2f\n'
                '  -> P1 = %.2f MW\n'
                '  %.2f P2 + %.1f = %.2f  -> P2 = %.2f MW\n\n'
                'Krok 2b (wsp. kary):\n'
                '  PF1 = 1/(1 - 2*B11*P1) = %.4f\n'
                '  PF1 * IC1 = %.4f * %.3f = %.3f = lambda  (OK)\n'
                '  PF2 = 1 (straty nie zależą od P2)\n\n'
                'Krok 3: P_L = B11 P1^2 = %.3f MW\n'
                '        P_R = %.2f + %.2f - %.3f = %.3f MW\n\n'
                'Książka: P1 = 103,09; P2 = 150; P_L = 5,197; P_R = 247,893 MW'
                % (self.v('PLref'), self.v('Pref'), B, lam, self.v('k1'), self.v('b1'), lam, B,
                   lam, p1, self.v('k2'), self.v('b2'), lam, p2, pf1, pf1, ic1, pf1 * ic1,
                   pl, p1, p2, pl, pr))


class ExerciseLoss(ExampleTab):
    title = 'Ćwiczenie 11 - Rozdział ekonomiczny ze stratami (iteracja lambda)'
    params = [
        ('PR', 'Obciążenie P_R [MW]', 100, 800, 400, 5),
        ('B11', 'B11 x 1e-5 [1/MW]', 0, 10, 2, 0.1),
        ('B22', 'B22 x 1e-5 [1/MW]', 0, 10, 4, 0.1),
        ('b1', 'C1 = 200 + b1 P1 + c1 P1^2:  b1', 4, 10, 6.8, 0.1),
        ('c1', '   c1', 0.0005, 0.01, 0.002, 0.0001),
        ('b2', 'C2 = 150 + b2 P2 + c2 P2^2:  b2', 4, 10, 7.2, 0.1),
        ('c2', '   c2', 0.0005, 0.01, 0.003, 0.0001),
    ]
    explanation = """
O CO CHODZI? (ćwiczenie nr 11 z końca rozdziału - z gotowa odpowiedzią)
   C1 = 200 + 6,8 P1 + 0,002 P1^2       C2 = 150 + 7,2 P2 + 0,003 P2^2   [Rs/h]
   Straty: P_L = 0,00002 P1^2 + 0,00004 P2^2 [MW],   obciążenie 400 MW.
   Odpowiedź książki: P1 = 280,15 MW, P2 = 122,01 MW, P_L = 2,16 MW.

JAK TO LICZY INŻYNIER - ITERACJA LAMBDA
Nie da się tu "od ręki" rozwiązać równań, bo straty zależą od mocy, a moce od
strat. Robimy jak w grze "zgadnij liczbę":
  1) Zgadujemy cenę lambda.
  2) Z równania koordynacyjnego liczymy każda moc:
         6,8 + 0,004 P1 = lambda (1 - 0,00004 P1)
         -> P1 = (lambda - 6,8) / (0,004 + 0,00004 lambda)   (podobnie P2)
  3) Sprawdzamy bilans: P1 + P2 - P_L - P_R.
     Za dużo mocy? -> zmniejszamy lambda. Za mało? -> zwiększamy.
  4) Powtarzamy (połowienie przedziału = bisekcja) aż błąd < 0,000000001 MW.

CO ZOBACZYSZ
  Góra: jak lambda zbiega się do wyniku w kolejnych iteracjach.
  Dół: błąd bilansu mocy (skala logarytmiczna) - spada błyskawicznie.
To samo robią komputery w dyspozycji mocy (np. w PSE) co kilka minut!
"""

    def compute_and_plot(self):
        units = [dict(a=200, b=self.v('b1'), c=self.v('c1'), pmin=0, pmax=1e4),
                 dict(a=150, b=self.v('b2'), c=self.v('c2'), pmin=0, pmax=1e4)]
        bl = [self.v('B11') * 1e-5, self.v('B22') * 1e-5]
        r = lambda_dispatch(units, self.v('PR'), bl)
        hist = np.array(r['hist'])
        n = min(len(hist), 40)
        ax = self.fig.add_subplot(211)
        ax.plot(range(1, n + 1), hist[:n, 0], 'o-', color=C_BLUE)
        ax.axhline(r['lam'], color=C_GREEN, ls='--', label='lambda* = %.4f' % r['lam'])
        ax.set_xlabel('Iteracja')
        ax.set_ylabel('lambda [Rs/MWh]')
        ax.set_title('Zbieżność iteracji lambda (bisekcja)')
        ax.grid(alpha=.3)
        ax.legend()
        ax2 = self.fig.add_subplot(212)
        ax2.semilogy(range(1, n + 1), np.abs(hist[:n, 1]) + 1e-12, 's-', color=C_RED)
        ax2.set_xlabel('Iteracja')
        ax2.set_ylabel('|P1+P2-P_L-P_R| [MW]')
        ax2.grid(alpha=.3, which='both')
        p1, p2 = r['P']
        lines = ['Iter   lambda        błąd bilansu [MW]']
        for i, (lm, f) in enumerate(r['hist'][:12]):
            lines.append('%3d  %10.5f   %+14.6f' % (i + 1, lm, f))
        return ('Równania koordynacyjne:  b_i + 2c_i P_i = lambda (1 - 2 B_ii P_i)\n\n'
                + '\n'.join(lines) + '\n ...\n\n'
                'WYNIK po %d iteracjach:\n'
                '  lambda = %.4f Rs/MWh\n'
                '  P1 = %.2f MW,  P2 = %.2f MW,  P_L = %.3f MW\n'
                '  kontrola: P1 + P2 - P_L = %.3f MW\n'
                '  koszt całkowity = %.2f Rs/h\n'
                '  PF1 = %.4f,  PF2 = %.4f\n\n'
                'Książka: P1 = 280,15; P2 = 122,01; P_L = 2,16 MW'
                % (len(r['hist']), r['lam'], p1, p2, r['PL'], p1 + p2 - r['PL'], r['cost'],
                   1 / (1 - 2 * bl[0] * p1), 1 / (1 - 2 * bl[1] * p2)))


class Ex11_7(ExampleTab):
    title = 'Przykład 11.7 - Elektrownia parowa + wodna: obciążenie i koszt wody'
    params = [
        ('Ps', 'Obciążenie el. parowej P_s [MW]', 50, 400, 220, 1),
        ('bs', 'C = 3 + bs P_s + cs P_s^2:  bs', 1, 10, 5, 0.1),
        ('cs', '   cs', 0.01, 0.1, 0.05, 0.001),
        ('V', 'Zużyta woda [mln m^3]', 100, 800, 400, 5),
        ('h', 'Czas pracy hydro [h]', 4, 24, 14, 1),
        ('q1', 'Przepływ q = q1 P_h + q2 P_h^2:  q1', 10, 80, 40, 1),
        ('q2', '   q2', 0.0, 0.1, 0.03, 0.001),
    ]
    explanation = """
O CO CHODZI?
Elektrownia parowa (węglowa) pracuje całą dobę, wodna tylko 14 godzin.
W czasie współpracy parowa daje 220 MW. Przez 14 h zużyto 400 mln m^3 wody.
Pytamy: z jaka mocą pracowała wodna i ile "kosztuje" woda?

DANE
   Koszt parowej:    C = 3 + 5 P_s + 0,05 P_s^2
   Przepływ wody:    q = P_h (40 + 0,03 P_h)  [m^3/s]
   -> przyrostowe zużycie wody IWR = dq/dP_h = 40 + 0,06 P_h

KROK 1 - lambda z elektrowni parowej
   lambda = dC/dP_s = 5 + 0,1 * 220 = 27

KROK 2 - moc elektrowni wodnej z ilości wody
   woda = q * 3600 s * 14 h = 400 * 10^6 m^3
   P_h (40 + 0,03 P_h) * 50400 = 400 000 000
   0,03 P_h^2 + 40 P_h - 7936,5 = 0   ->  (wzór na pierwiastki równania kwadratowego)
   P_h = 175,35 MW

KROK 3 - "cena wody" r
   W optimum: lambda = r * IWR, czyli r = 27 / (40 + 0,06 * 175,35) = 0,534
   (Rs/h za 1 m^3/s przepływu). Woda nie jest darmowa: zużyta teraz, nie będzie
   dostępna później - więc ma "cenę zastępczą".

ANALOGIA
Masz ograniczony bak paliwa na wycieczkę (woda w zbiorniku). Musisz zdecydować
jak szybko go zużywać - cena "r" mówi, ile wart jest litr w Twoim planie.

CO ZOBACZYSZ
  Góra: zużycie wody w funkcji P_h - przecięcie z pozioma linia daje P_h.
  Dół: lambda parowej i r * IWR - w optimum są równe.
"""

    def compute_and_plot(self):
        Ps, bs, cs = self.v('Ps'), self.v('bs'), self.v('cs')
        V, h, q1, q2 = self.v('V') * 1e6, self.v('h'), self.v('q1'), self.v('q2')
        lam = bs + 2 * cs * Ps
        Ph = hydro_load_from_water(V, h, q1, q2)
        iwr = q1 + 2 * q2 * Ph
        r = lam / iwr
        ax = self.fig.add_subplot(211)
        P = np.linspace(0, max(Ph * 2, 50), 300)
        ax.plot(P, (q1 * P + q2 * P ** 2) * 3600 * h / 1e6, color=C_BLUE, lw=2,
                label='woda zużyta w %.0f h' % h)
        ax.axhline(V / 1e6, color=C_RED, ls='--', label='dostępna woda %.0f mln m^3' % (V / 1e6))
        ax.plot([Ph], [V / 1e6], 'o', color='k', ms=8)
        ax.annotate('P_h = %.2f MW' % Ph, (Ph, V / 1e6), xytext=(10, -25),
                    textcoords='offset points')
        ax.set_xlabel('P_h [MW]')
        ax.set_ylabel('Woda [mln m^3]')
        ax.grid(alpha=.3)
        ax.legend(fontsize=8)
        ax2 = self.fig.add_subplot(212)
        ax2.plot(P, r * (q1 + 2 * q2 * P), color=C_GREEN, lw=2, label='r * IWR(P_h)')
        ax2.axhline(lam, color=C_ORANGE, ls='--', label='lambda parowej = %.2f' % lam)
        ax2.plot([Ph], [lam], 'o', color='k')
        ax2.set_xlabel('P_h [MW]')
        ax2.set_ylabel('[Rs/MWh]')
        ax2.grid(alpha=.3)
        ax2.legend(fontsize=8)
        k = V / (3600 * h)
        return ('Krok 1: lambda = %.2f + 2*%.3f*%.0f = %.3f\n\n'
                'Krok 2: P_h (%.1f + %.3f P_h) * 3600 * %.0f = %.3g\n'
                '        %.3f P_h^2 + %.1f P_h - %.2f = 0\n'
                '        P_h = %.4f MW\n\n'
                'Krok 3: IWR = %.1f + %.3f * %.2f = %.3f\n'
                '        r = lambda / IWR = %.3f / %.3f = %.4f\n\n'
                'Łączna moc w czasie współpracy: %.2f MW\n\n'
                'Książka: P_h = 175,3514 MW;  lambda = 27.'
                % (bs, cs, Ps, lam, q1, q2, h, V, q2, q1, k, Ph, q1, 2 * q2, Ph, iwr, lam,
                   iwr, r, Ps + Ph))


class Ex11_8(ExampleTab):
    title = 'Przykład 11.8 - Harmonogram hydro-termiczny: szczyt i poza szczytem'
    params = [
        ('Lp', 'Obciążenie szczytowe [MW]', 300, 1000, 700, 10),
        ('hp', '   czas szczytu [h]', 1, 23, 12, 1),
        ('Lo', 'Obciążenie poza szczytem [MW]', 100, 700, 270, 10),
        ('W', 'Woda na dobę [mln m^3]', 100, 1000, 459.47, 0.01),
        ('bs', 'Termiczna: C = 264 + bs P + cs P^2:  bs', 10, 40, 22, 0.5),
        ('cs', '   cs', 0.002, 0.04, 0.0125, 0.0005),
        ('q1', 'Woda: q = q1 P + q2 P^2:  q1', 5, 20, 11, 0.5),
        ('q2', '   q2', 0.0, 0.02, 0.0075, 0.0005),
    ]
    explanation = """
O CO CHODZI?
Elektrownia cieplna i wodna razem zasilają odbiorców:
   szczyt:          700 MW przez 12 h
   poza szczytem:   270 MW przez 12 h
Woda na całą dobę: 459,47 mln m^3. Jak podzielić moc w każdym okresie?

DANE
   Cieplna: C = 264 + 22 P + 0,0125 P^2   ->  IC = 22 + 0,025 P
   Wodna:   q = 11 P + 0,0075 P^2 [m^3/s] ->  IWR = 11 + 0,015 P

WARUNKI OPTIMUM (3 równania, 3 niewiadome: P_ho, P_hp, R)
  (1) poza szczytem:  22 + 0,025 (270 - P_ho) = R (11 + 0,015 P_ho)
  (2) w szczycie:     22 + 0,025 (700 - P_hp) = R (11 + 0,015 P_hp)
  (3) woda:  12*3600*[q(P_ho) + q(P_hp)] = 459,47 * 10^6
R - "przelicznik" wody na pieniądze (jak r w przykładzie 11.7).

JAK ROZWIĄZAĆ (sposób inżyniera)
Dla zgadniętego R równania (1) i (2) są liniowe - od razu dostajemy P_ho i P_hp.
Potem sprawdzamy (3): jeśli zużyliśmy za dużo wody, woda jest "za tania" -
zwiększamy R. Połowienie przedziału kończy się po kilkudziesięciu krokach.

WYNIK
   R ~ 1,482,   P_ho ~ 263,7 MW (wodna),  P_so ~ 6,3 MW (cieplna)
                P_hp ~ 491,3 MW (wodna),  P_sp ~ 208,7 MW (cieplna)
(W książce rozwiązanie urywa się - liczby wyznacza aplikacja.)
Woda bierze większość obciążenia, a cieplna "dokleja" resztę, głównie w szczycie.

CO ZOBACZYSZ
  Góra: słupki - kto ile produkuje w obu okresach.
  Dół: zużycie wody w funkcji R; przecięcie z linia dostępnej wody wyznacza R.
EKSPERYMENT: zmniejsz ilość wody - cieplna musi pracować więcej, lambda rośnie.
"""

    def compute_and_plot(self):
        bs, cs, q1, q2 = self.v('bs'), self.v('cs'), self.v('q1'), self.v('q2')
        hp = self.v('hp')
        r = hydrothermal_two_period(bs, cs, q1, q2, self.v('Lo'), self.v('Lp'),
                                    24 - hp, hp, self.v('W') * 1e6)
        ax = self.fig.add_subplot(211)
        x = np.arange(2)
        hyd = [r['Pho'], r['Php']]
        th = [r['Pso'], r['Psp']]
        ax.bar(x, hyd, 0.5, color=C_BLUE, label='wodna P_h')
        ax.bar(x, th, 0.5, bottom=hyd, color=C_RED, label='cieplna P_s')
        for i in range(2):
            ax.text(i, hyd[i] / 2, '%.1f' % hyd[i], ha='center', color='w')
            ax.text(i, hyd[i] + th[i] / 2, '%.1f' % th[i], ha='center', color='w')
        ax.set_xticks(x)
        ax.set_xticklabels(['poza szczytem (%.0f h)' % (24 - hp), 'szczyt (%.0f h)' % hp])
        ax.set_ylabel('Moc [MW]')
        ax.set_title('Ekonomiczny podział obciążenia')
        ax.legend()
        ax.grid(alpha=.3, axis='y')
        ax2 = self.fig.add_subplot(212)
        Rs = np.linspace(0.2, 5, 200)
        ax2.plot(Rs, [r['used_fn'](x_) / 1e6 for x_ in Rs], color=C_PURPLE, lw=2,
                 label='woda zużyta przy danym R')
        ax2.axhline(self.v('W'), color=C_GREEN, ls='--', label='woda dostępna')
        ax2.axvline(r['R'], color='k', ls=':')
        ax2.set_xlabel('R [Rs/h na m^3/s]')
        ax2.set_ylabel('Woda [mln m^3]')
        ax2.grid(alpha=.3)
        ax2.legend(fontsize=8)
        return ('IC_s = %.2f + %.4f P_s,   IWR = %.2f + %.4f P_h\n\n'
                'Wynik bisekcji po R:\n'
                '  R = %.4f\n\n'
                'POZA SZCZYTEM (%.0f MW, %.0f h):\n'
                '  P_ho = %.2f MW (wodna),  P_so = %.2f MW (cieplna)\n'
                '  lambda_o = %.3f Rs/MWh\n\n'
                'SZCZYT (%.0f MW, %.0f h):\n'
                '  P_hp = %.2f MW (wodna),  P_sp = %.2f MW (cieplna)\n'
                '  lambda_p = %.3f Rs/MWh\n\n'
                'Kontrola wody: %.2f mln m^3 (zadane %.2f)\n'
                'Koszt paliwa cieplnej: %.0f Rs/doba'
                % (bs, 2 * cs, q1, 2 * q2, r['R'], self.v('Lo'), 24 - hp, r['Pho'], r['Pso'],
                   r['lam_o'], self.v('Lp'), hp, r['Php'], r['Psp'], r['lam_p'],
                   r['used'] / 1e6, self.v('W'),
                   (24 - hp) * quad_cost(264, bs, cs, r['Pso']) + hp * quad_cost(264, bs, cs, r['Psp'])))


# =============================================================================
# Rozdział 12 - HVDC
# =============================================================================

class Ex12_1(ExampleTab):
    title = 'Przykład 12.1 - Bipolarne łącze HVDC +/-500 kV'
    params = [
        ('Pr', 'Moc odebrana P_r [MW]', 100, 3000, 1250, 10),
        ('Vr', 'Napięcie bieguna po str. odbiorczej [kV]', 100, 800, 500, 5),
        ('Ploss', 'Straty w linii DC [MW]', 0, 200, 40, 1),
    ]
    explanation = """
O CO CHODZI?
HVDC to "autostrada" dla energii na duże odległości - prąd stały zamiast
przemiennego. Łącze bipolarne ma dwa przewody: +500 kV i -500 kV
(jak plus i minus baterii) - każdy biegun niesie połowę mocy.
Odbieramy 1250 MW, w linii ginie 40 MW. Szukamy:
 a) mocy wysłanej, b) napięcia po stronie wysyłającej, c) i d) mocy i napięcia
 w połowie linii.

KROK PO KROKU (przyjmujemy +/-500 kV po stronie odbiorczej)
 a) P_s = P_r + straty = 1250 + 40 = 1290 MW
    Prąd łącza:  I_d = P_r / (2 V_r) = 1250 / (2*500) = 1,25 kA
 b) V_s = P_s / (2 I_d) = 1290 / 2,5 = 516 kV   (+/-516 kV)
 c) W połowie linii zginęła połowa strat: P_m = 1250 + 20 = 1270 MW
 d) V_m = P_m / (2 I_d) = 1270 / 2,5 = 508 kV

DLACZEGO NAPIĘCIE SPADA?
Prąd stały płynie przez rezystancje przewodu - tak jak woda w długim wężu traci
ciśnienie. Spadek napięcia R*I jest liniowy wzdłuż linii.
Moc strat: P_strat = 2 * R * I^2 -> można z niej wyznaczyć rezystancje bieguna.

CO ZOBACZYSZ
  Wykres napięcia i mocy wzdłuż linii (0 = wysyłanie, 1 = odbiór).
EKSPERYMENT: podwój napięcie przy tej samej mocy - prąd spada o połowę, a straty
4 razy (P = R I^2)! Dlatego przesyła się energie wysokim napięciem.
"""

    def compute_and_plot(self):
        Pr, Vr, Pl = self.v('Pr'), self.v('Vr'), self.v('Ploss')
        Id = Pr / (2 * Vr)            # kA (MW/kV)
        Ps = Pr + Pl
        Vs = Ps / (2 * Id)
        Pm = Pr + Pl / 2
        Vm = Pm / (2 * Id)
        Rpole = Pl / (2 * Id ** 2)    # ohm (MW / kA^2)
        x = np.linspace(0, 1, 100)
        V = Vs - (Vs - Vr) * x
        P = 2 * V * Id
        ax = self.fig.add_subplot(211)
        ax.plot(x, V, color=C_RED, lw=2, label='+V bieguna')
        ax.plot(x, -V, color=C_BLUE, lw=2, label='-V bieguna')
        for xx, vv, lab in ((0, Vs, 'V_s'), (0.5, Vm, 'V_m'), (1, Vr, 'V_r')):
            ax.plot([xx], [vv], 'o', color='k')
            ax.annotate('%s = %.1f kV' % (lab, vv), (xx, vv), xytext=(5, -18),
                        textcoords='offset points', fontsize=8)
        ax.set_ylabel('Napięcie [kV]')
        ax.set_title('Napięcie wzdłuż łącza bipolarnego')
        ax.grid(alpha=.3)
        ax.legend(fontsize=8, loc='center right')
        ax2 = self.fig.add_subplot(212)
        ax2.plot(x, P, color=C_GREEN, lw=2)
        ax2.plot([0, 0.5, 1], [Ps, Pm, Pr], 'o', color='k')
        ax2.set_xlabel('Położenie wzdłuż linii (0 = nadawcza, 1 = odbiorcza)')
        ax2.set_ylabel('Moc [MW]')
        ax2.grid(alpha=.3)
        return ('a) P_s = %.0f + %.0f = %.0f MW\n'
                '   I_d = P_r/(2 V_r) = %.0f/(2*%.0f) = %.4f kA\n'
                'b) V_s = P_s/(2 I_d) = %.0f/%.3f = %.2f kV  (+/-%.2f kV)\n'
                'c) P_m = P_r + P_loss/2 = %.1f MW\n'
                'd) V_m = P_m/(2 I_d) = %.2f kV\n\n'
                'Dodatkowo:\n'
                '  rezystancja jednego bieguna R = P_loss/(2 I_d^2) = %.2f ohm\n'
                '  sprawność przesyłu = %.2f %%\n\n'
                'Książka: a) 1290 MW (b-d wg tego samego toku: 516 kV, 1270 MW, 508 kV).'
                % (Pr, Pl, Ps, Pr, Vr, Id, Ps, 2 * Id, Vs, Vs, Pm, Vm, Rpole, 100 * Pr / Ps))


class Ex12_2(ExampleTab):
    title = 'Przykład 12.2 - Kąt wygaszania (delta) falownika HVDC'
    params = [
        ('Vdr', 'Napięcie DC prostownika [kV]', 50, 200, 100, 1),
        ('Id', 'Prąd DC [A]', 50, 1500, 500, 10),
        ('RL', 'Rezystancja linii R_L [ohm]', 0, 60, 20, 0.5),
        ('Rc', 'Rez. komutacyjna falownika R_ci [ohm]', 0, 30, 8, 0.5),
        ('Vdi0', 'Napięcie biegu jałowego falownika [kV]', 50, 200, 100, 1),
    ]
    explanation = """
O CO CHODZI?
Na końcu łącza HVDC stoi FALOWNIK - zamienia prąd stały z powrotem na przemienny.
Jego tyrystory muszą "zdążyć się wyłączyć" przed kolejnym przełączeniem.
Kąt wygaszania delta to zapas czasu (w stopniach) na to wyłączenie.
Za mały delta = ryzyko awarii komutacji; za duży = gorsze wykorzystanie napięcia.

DANE
   Strona prostownika: V_d = 100 kV, I_d = 500 A
   Rezystancja linii R_L = 20 ohm, rezystancja komutacyjna R_c = 8 ohm
   Napięcie biegu jałowego falownika (przy delta = 0): 100 kV

KROK PO KROKU
 1) Spadek napięcia na linii: R_L * I_d = 20 * 0,5 kA = 10 kV
    -> na zaciskach falownika: 100 - 10 = 90 kV  (zgadza się z treścią)
 2) Równanie falownika (sterowanego kątem wygaszania):
        V_di = V_di0 * cos(delta) - R_c * I_d
        90 = 100 cos(delta) - 8 * 0,5   ->   cos(delta) = 0,94
 3) delta = arccos(0,94) = 19,95 st. (ok. 20 st.)

ANALOGIA
Delta to jak odstęp między samochodami na autostradzie - musi być wystarczający
na wyhamowanie (wyłączenie tyrystora), ale za duży zmniejsza przepustowość.

CO ZOBACZYSZ
  Góra: charakterystyka falownika V_di(I_d) dla różnych kątów delta.
  Dół: jak delta zależy od prądu przy stałym napięciu prostownika - większy prąd
       to większy spadek napięcia w linii, więc falownik musi pracować z większym
       kątem delta. Czerwona strefa (< 15 st.) to ryzyko błędu komutacji.
"""

    def compute_and_plot(self):
        Vdr, Id, RL, Rc, Vdi0 = (self.v(k) for k in ('Vdr', 'Id', 'RL', 'Rc', 'Vdi0'))
        Ika = Id / 1000.0
        Vdi = Vdr - RL * Ika
        cosd = (Vdi + Rc * Ika) / Vdi0
        ok = -1 <= cosd <= 1
        delta = math.degrees(math.acos(cosd)) if ok else float('nan')
        ax = self.fig.add_subplot(211)
        I = np.linspace(0, 1.5, 200)
        for dd in (10, 15, 20, 30, 45):
            ax.plot(I * 1000, Vdi0 * math.cos(math.radians(dd)) - Rc * I, lw=1,
                    label='delta = %d st.' % dd)
        ax.plot(I * 1000, Vdr - RL * I, 'k--', lw=2, label='V_dr - R_L I_d (z prostownika)')
        ax.plot([Id], [Vdi], 'o', color=C_RED, ms=9)
        ax.set_xlabel('I_d [A]')
        ax.set_ylabel('V_di [kV]')
        ax.set_title('Charakterystyki falownika i punkt pracy')
        ax.grid(alpha=.3)
        ax.legend(fontsize=7, ncol=2)
        ax2 = self.fig.add_subplot(212)
        Iv = np.linspace(0.02, 1.5, 200)
        cd = (Vdr - RL * Iv + Rc * Iv) / Vdi0
        dv = np.where(np.abs(cd) <= 1, np.degrees(np.arccos(np.clip(cd, -1, 1))), np.nan)
        ax2.plot(Iv * 1000, dv, color=C_PURPLE, lw=2)
        if ok:
            ax2.plot([Id], [delta], 'o', color=C_RED, ms=9)
        ax2.axhspan(0, 15, color=C_RED, alpha=.1, label='strefa ryzyka komutacji')
        ax2.set_xlabel('I_d [A]')
        ax2.set_ylabel('delta [st.]')
        ax2.grid(alpha=.3)
        ax2.legend(fontsize=8)
        return ('Krok 1: spadek na linii R_L I_d = %.1f * %.3f = %.2f kV\n'
                '        V_di = %.1f - %.2f = %.2f kV\n\n'
                'Krok 2: V_di = V_di0 cos(delta) - R_c I_d\n'
                '        cos(delta) = (%.2f + %.1f*%.3f)/%.1f = %.4f\n\n'
                'Krok 3: delta = %s\n\n'
                'Moc na wejściu falownika: %.2f MW\n'
                'Książka: R_di = 8 ohm, delta ~ 20 st.'
                % (RL, Ika, RL * Ika, Vdr, RL * Ika, Vdi, Vdi, Rc, Ika, Vdi0, cosd,
                   '%.2f st.' % delta if ok else 'BRAK (|cos| > 1 - niemożliwy punkt pracy)',
                   Vdi * Ika))


class Ex12_3(ExampleTab):
    title = 'Przykład 12.3 - Napięcia DC i prąd łącza (alfa = 12 st., delta = 20 st.)'
    params = [
        ('alpha', 'Kąt zapłonu prostownika alfa [st.]', 0, 90, 12, 0.5),
        ('delta', 'Kąt wygaszania falownika delta [st.]', 0, 90, 20, 0.5),
        ('VLr', 'Napięcie AC po str. prostownika [kV]', 200, 500, 390, 1),
        ('VLi', 'Napięcie AC po str. falownika [kV]', 200, 500, 375, 1),
        ('R', 'Rezystancja obwodu DC [ohm]', 1, 60, 20, 0.5),
    ]
    explanation = """
O CO CHODZI?
Mostek 6-pulsowy (6 tyrystorów) zamienia trójfazowe napięcie AC na DC.
Bez opóźnienia zapłonu (alfa = 0) daje napięcie
      V_d0 = (3*pierwiastek(2)/pi) * V_L = 1,3505 * V_L
Opóźniając zapłon o kąt alfa zmniejszamy napięcie:  V_d = V_d0 * cos(alfa).
To jak ściemniacz światła - im później "włączysz", tym mniej energii przechodzi.

KROK PO KROKU
 1) Napięcia biegu jałowego:
       V_dr0 = 1,3505 * 390 = 526,69 kV       V_di0 = 1,3505 * 375 = 506,43 kV
 2) Bez komutacji (bez nakładania):
       V_dr = 526,69 cos 12 = 515,18 kV       V_di = 506,43 cos 20 = 475,9 kV
       (w książce zaokrąglone do 474 kV)
 3) Prąd łącza - różnica napięć "pcha" prąd przez rezystancje (prawo Ohma):
       I_d = (V_dr - V_di) / R = (515,18 - 475,9)/20 ~ 1,96 kA

DLACZEGO TO WAŻNE?
Bardzo mała zmiana kątów (o 1-2 stopnie) mocno zmienia różnicę napięć, a więc
i prąd! Dlatego łącza HVDC maja szybkie regulatory prądu.

CO ZOBACZYSZ
  Góra: V_d = V_d0 cos(kąt) dla obu stron i punkty pracy.
        Dla alfa > 90 st. prostownik staje się falownikiem (napięcie ujemne).
  Dół: prąd i moc łącza w funkcji alfa - widać ogromna czułość.
"""

    def compute_and_plot(self):
        a, d, VLr, VLi, R = (self.v(k) for k in ('alpha', 'delta', 'VLr', 'VLi', 'R'))
        Vdr0, Vdi0 = K_BRIDGE * VLr, K_BRIDGE * VLi
        Vdr = Vdr0 * math.cos(math.radians(a))
        Vdi = Vdi0 * math.cos(math.radians(d))
        Id = (Vdr - Vdi) / R
        ang = np.linspace(0, 180, 361)
        ax = self.fig.add_subplot(211)
        ax.plot(ang, Vdr0 * np.cos(np.radians(ang)), color=C_BLUE, lw=2, label='V_dr0 cos(alfa)')
        ax.plot(ang, Vdi0 * np.cos(np.radians(ang)), color=C_RED, lw=2, label='V_di0 cos(delta)')
        ax.plot([a], [Vdr], 'o', color=C_BLUE, ms=9)
        ax.plot([d], [Vdi], 's', color=C_RED, ms=9)
        ax.axhline(0, color='k', lw=.6)
        ax.axvline(90, color=C_GREY, ls=':')
        ax.text(95, Vdr0 * 0.6, 'tryb falownikowy', fontsize=8, color=C_GREY)
        ax.set_xlabel('Kąt [st.]')
        ax.set_ylabel('V_d [kV]')
        ax.grid(alpha=.3)
        ax.legend(fontsize=8)
        ax2 = self.fig.add_subplot(212)
        al = np.linspace(0, 40, 200)
        Ia = (Vdr0 * np.cos(np.radians(al)) - Vdi) / R
        Ia = np.where(Ia > 0, Ia, np.nan)       # zawory przewodzą tylko w jedna stronę
        ax2.plot(al, Ia, color=C_GREEN, lw=2, label='I_d [kA]')
        ax2.plot([a], [Id], 'o', color='k')
        ax2.set_xlabel('alfa [st.] (przy stałym delta)')
        ax2.set_ylabel('I_d [kA]')
        ax2.grid(alpha=.3)
        axr = ax2.twinx()
        axr.plot(al, Ia * Vdr0 * np.cos(np.radians(al)), color=C_ORANGE, ls='--',
                 label='P_dr [MW]')
        axr.set_ylabel('P [MW]')
        h1, l1 = ax2.get_legend_handles_labels()
        h2, l2 = axr.get_legend_handles_labels()
        ax2.legend(h1 + h2, l1 + l2, fontsize=8)
        return ('Krok 1: V_d0 = 3*sqrt(2)/pi * V_L = %.4f * V_L\n'
                '        V_dr0 = %.4f * %.0f = %.2f kV\n'
                '        V_di0 = %.4f * %.0f = %.2f kV\n\n'
                'Krok 2 (bez nakładania komutacji):\n'
                '        V_dr = %.2f cos %.1f = %.2f kV\n'
                '        V_di = %.2f cos %.1f = %.2f kV\n\n'
                'Krok 3: I_d = (V_dr - V_di)/R = (%.2f - %.2f)/%.1f = %.4f kA\n'
                '%s\n'
                'Moc wysłana P_dr = %.1f MW, odebrana P_di = %.1f MW, straty %.2f MW\n\n'
                'Książka: 526,69 / 506,43 kV;  V_dr = 515,18 kV; V_di = 474 kV\n'
                '(dokładnie 506,43*cos20 = 475,89 kV).'
                % (K_BRIDGE, K_BRIDGE, VLr, Vdr0, K_BRIDGE, VLi, Vdi0, Vdr0, a, Vdr, Vdi0,
                   d, Vdi, Vdr, Vdi, R, Id,
                   '   UWAGA: I_d < 0 - zawory nie przewodzą wstecz, łącze stoi!' if Id < 0 else '',
                   Vdr * max(Id, 0), Vdi * max(Id, 0), R * max(Id, 0) ** 2))


# =============================================================================
# Rozdział 13 - stabilność
# =============================================================================

class Ex13_1(ExampleTab):
    title = 'Przykład 13.1 - Moc przyspieszająca i kąt mocy po 10 okresach'
    params = [
        ('G', 'Moc znamionowa G [MVA]', 5, 200, 20, 1),
        ('KE', 'Energia kinetyczna wirnika [MJ]', 10, 1000, 60, 5),
        ('Pin', 'Moc wejściowa (turbina) [MW]', 0, 100, 15, 0.5),
        ('Pe', 'Moc elektryczna [MW]', 0, 100, 13.5, 0.5),
        ('f', 'Częstotliwość [Hz]', 50, 60, 50, 10),
        ('cyc', 'Liczba okresów', 1, 50, 10, 1),
        ('d0', 'Kąt początkowy delta0 [st.]', 0, 60, 0, 1),
    ]
    explanation = """
O CO CHODZI?
Generator w elektrowni to ogromny wirujący walec. Turbina "pcha" go mocą 15 MW,
a sieć "hamuje" pobierając 13,5 MW. Różnica 1,5 MW nie znika - przyspiesza wirnik!
Pytamy: o ile wirnik "ucieknie" do przodu (kąt delta) po 10 okresach (0,2 s)?

ANALOGIA
Rower: pedałujesz z siłą 15, opory to 13,5 - nadwyżka 1,5 przyspiesza rower.
Im cięższy rower (bezwładność), tym wolniej przyspiesza.

KROK PO KROKU
 1) Moc przyspieszająca: P_a = 15 - 13,5 = 1,5 MW
 2) Stała bezwładności: H = KE/G = 60/20 = 3 s
 3) Współczynnik M = G*H/(180 f) = 60/(180*50) = 0,00667 MJ*s/st.el.
 4) Równanie kołysań:  M * d2(delta)/dt2 = P_a
       przyspieszenie = 1,5/0,00667 = 225 st.el./s^2  (= 3,927 rad/s^2)
 5) Przy stałym przyspieszeniu (jak spadający kamień: s = a t^2 / 2):
       delta = delta0 + 0,5 * 3,927 * 0,2^2 = delta0 + 0,0785 rad = delta0 + 4,5 st.

CO ZOBACZYSZ
  Góra: kąt delta(t) - parabola (ruch jednostajnie przyspieszony).
  Dół: prędkość wirnika rośnie liniowo (obr/min).
EKSPERYMENT: zwiększ KE (cięższy wirnik) - kąt rośnie wolniej. To dlatego
duża bezwładność pomaga stabilności!
"""

    def compute_and_plot(self):
        G, KE, Pin, Pe, f, cyc, d0 = (self.v(k) for k in ('G', 'KE', 'Pin', 'Pe', 'f', 'cyc', 'd0'))
        H = KE / G
        M = KE / (180 * f)
        Pa = Pin - Pe
        acc_deg = Pa / M
        acc_rad = math.radians(acc_deg)
        T = cyc / f
        t = np.linspace(0, T, 200)
        delta = d0 + 0.5 * acc_deg * t ** 2
        poles = 4
        ns = 120 * f / poles
        n = ns + acc_deg * t / 360.0 * 60 * 2 / poles
        ax = self.fig.add_subplot(211)
        ax.plot(t, delta, color=C_RED, lw=2)
        ax.plot([T], [delta[-1]], 'o', color='k')
        ax.annotate('delta = %.3f st.' % delta[-1], (T, delta[-1]), xytext=(-110, 5),
                    textcoords='offset points')
        ax.set_ylabel('delta [st. el.]')
        ax.set_title('Kąt mocy przy stałej mocy przyspieszającej')
        ax.grid(alpha=.3)
        ax2 = self.fig.add_subplot(212)
        ax2.plot(t, n, color=C_BLUE, lw=2)
        ax2.set_xlabel('t [s]')
        ax2.set_ylabel('n [obr/min] (4 bieguny)')
        ax2.grid(alpha=.3)
        return ('Krok 1: P_a = %.2f - %.2f = %.2f MW\n'
                'Krok 2: H = KE/G = %.0f/%.0f = %.3f s\n'
                'Krok 3: M = KE/(180 f) = %.0f/(180*%.0f) = %.6f MJ s/st.\n'
                'Krok 4: d2delta/dt2 = P_a/M = %.2f st./s^2 = %.4f rad/s^2\n'
                'Krok 5: t = %.0f okresów / %.0f Hz = %.3f s\n'
                '        delta = delta0 + 0,5*%.4f*%.3f^2 = delta0 + %.5f rad\n'
                '              = %.1f + %.3f st. = %.3f st.\n\n'
                'Prędkość po %.3f s: %.3f obr/min (wzrost o %.3f obr/min)\n\n'
                'Książka: delta = 0,07854 rad + delta0 (4,5 st.)'
                % (Pin, Pe, Pa, KE, G, H, KE, f, M, acc_deg, acc_rad, cyc, f, T, acc_rad, T,
                   0.5 * acc_rad * T ** 2, d0, 0.5 * acc_deg * T ** 2, delta[-1], T, n[-1],
                   n[-1] - ns))


class Ex13_2(ExampleTab):
    title = 'Przykład 13.2 - Energia kinetyczna i przyspieszenie turbogeneratora'
    params = [
        ('G', 'Moc znamionowa G [MVA]', 5, 500, 30, 5),
        ('H', 'Stała bezwładności H [s] (kW*s/kVA)', 1, 15, 10, 0.5),
        ('Pin', 'Moc wejściowa [MW]', 0, 300, 25, 1),
        ('Pout', 'Moc wyjściowa [MW]', 0, 300, 20, 1),
        ('f', 'Częstotliwość [Hz]', 50, 60, 50, 10),
        ('p', 'Liczba biegunów', 2, 12, 4, 2),
    ]
    explanation = """
O CO CHODZI?
Turbogenerator 30 MVA, 4 bieguny, 50 Hz, stała bezwładności H = 10 s.
 (i)  Ile energii jest "zmagazynowane" w wirującym wirniku?
 (ii) Turbina daje 25 MW, a sieć bierze 20 MW - jak szybko wirnik przyspiesza?

CO TO JEST H?
H mówi, przez ile sekund wirnik mógłby zasilać sieć mocą znamionowa
wyłącznie z energii swojego obrotu (jak rozpędzony bak/zabawka).

KROK PO KROKU
 (i)  KE = G * H = 30 * 10 = 300 MJ
      (to energia ok. 7 litrów benzyny - ale oddana w ułamku sekundy!)
 (ii) P_a = 25 - 20 = 5 MW
      M = G H / (180 f) = 300 / 9000 = 0,0333 MJ s/st.el.
      przyspieszenie = P_a / M = 5 / 0,0333 = 150 st.el./s^2
      Zamiana na obr/min/s:
        stopnie elektryczne -> mechaniczne: dzielimy przez (p/2) = 2 -> 75 st.mech/s^2
        75/360 obr/s^2 = 0,2083 obr/s^2 = 12,5 obr/min na sekundę

CO ZOBACZYSZ
  Góra: prędkość obrotowa rośnie - gdyby nic nie zrobić, po 4 s byłoby +50 obr/min.
  Dół: energia kinetyczna dla różnych H - im większy H, tym "spokojniejszy" wirnik.
To dlatego turbina musi mieć szybki regulator prędkości!
"""

    def compute_and_plot(self):
        G, H, Pin, Pout, f, p = (self.v(k) for k in ('G', 'H', 'Pin', 'Pout', 'f', 'p'))
        p = max(2, int(round(p / 2)) * 2)
        KE = G * H
        Pa = Pin - Pout
        M = KE / (180 * f)
        acc = Pa / M
        acc_mech = acc / (p / 2)
        rpm_s = acc_mech / 360 * 60
        ns = 120 * f / p
        t = np.linspace(0, 4, 200)
        ax = self.fig.add_subplot(211)
        ax.plot(t, ns + rpm_s * t, color=C_BLUE, lw=2, label='n(t)')
        ax.axhline(ns, color=C_GREY, ls='--', label='n_s = %.0f obr/min' % ns)
        ax.set_xlabel('t [s]')
        ax.set_ylabel('n [obr/min]')
        ax.set_title('Rozpęd wirnika przy P_a = %.1f MW' % Pa)
        ax.grid(alpha=.3)
        ax.legend()
        ax2 = self.fig.add_subplot(212)
        Hs = np.linspace(1, 15, 100)
        ax2.plot(Hs, G * Hs, color=C_PURPLE, lw=2, label='KE = G H [MJ]')
        ax2.plot([H], [KE], 'o', color='k')
        axr = ax2.twinx()
        axr.plot(Hs, Pa / (G * Hs / (180 * f)), color=C_ORANGE, ls='--', label='przyspieszenie')
        axr.set_ylabel('[st.el./s^2]')
        ax2.set_xlabel('H [s]')
        ax2.set_ylabel('KE [MJ]')
        ax2.grid(alpha=.3)
        h1, l1 = ax2.get_legend_handles_labels()
        h2, l2 = axr.get_legend_handles_labels()
        ax2.legend(h1 + h2, l1 + l2, fontsize=8, loc='center right')
        return ('(i)  KE = G H = %.0f * %.1f = %.1f MJ\n\n'
                '(ii) P_a = %.1f - %.1f = %.1f MW\n'
                '     M = KE/(180 f) = %.1f/(180*%.0f) = %.5f MJ s/st.\n'
                '     alfa = P_a/M = %.2f st.el./s^2\n'
                '     liczba par biegunów p/2 = %d -> %.2f st.mech/s^2\n'
                '     = %.2f/360 * 60 = %.3f obr/min/s\n\n'
                'Prędkość synchroniczna n_s = 120 f/p = %.0f obr/min\n\n'
                'Książka: KE = 300 MJ;  150 st./s^2;  12,5 obr/min/s'
                % (G, H, KE, Pin, Pout, Pa, KE, f, M, acc, p // 2, acc_mech, acc_mech, rpm_s, ns))


class Ex13_3(ExampleTab):
    title = 'Przykład 13.3 - Równanie kołysań generatora na szynie sztywnej + zwarcie'
    params = [
        ('P', 'Moc czynna P [p.u.]', 0.1, 1.5, 0.8, 0.01),
        ('Vt', 'Napięcie zacisków V_t [p.u.]', 0.9, 1.2, 1.05, 0.01),
        ('xd', "Reaktancja przejściowa x'd [p.u.]", 0.05, 0.6, 0.2, 0.01),
        ('XT', 'Reaktancja transformatora X_T [p.u.]', 0.0, 0.4, 0.1, 0.01),
        ('XL', 'Reaktancja jednej linii X_L [p.u.]', 0.1, 1.0, 0.4, 0.01),
        ('H', 'Stała bezwładności H [MJ/MVA]', 1, 10, 4, 0.1),
        ('m', 'Miejsce zwarcia na linii (0 = szyna wys.)', 0, 1, 0.0, 0.05),
        ('tc', 'Czas wyłączenia zwarcia t_c [s]', 0.0, 0.6, 0.1, 0.005),
        ('D', 'Tłumienie D [p.u.]', 0, 30, 0, 0.5),
    ]
    click_hint = 'w górny wykres ustawia czas t_c'
    explanation = """
O CO CHODZI?
Generator oddaje 0,8 p.u. przy napięciu 1,05 p.u. do "szyny sztywnej"
(ogromnej sieci o stałym napięciu 1,0 p.u.) przez transformator 0,1 p.u.
i DWIE równoległe linie po 0,4 p.u. Mamy napisać równanie kołysań.

KROK PO KROKU (jak w książce)
 1) Reaktancja od zacisków do sieci: X_T + X_L/2 = 0,1 + 0,2 = 0,3 p.u.
 2) Kąt napięcia zacisków:  P = V_t V sin(theta)/0,3
       sin(theta) = 0,8*0,3/1,05 = 0,2286  ->  theta = 13,21 st.
 3) Prąd: I = (V_t - V)/(j 0,3) = 0,80 p.u. pod kątem -5,28 st.
 4) SEM za reaktancja przejściowa:  E = V_t + j x'd I = 1,11 / 21,06 st.
 5) Reaktancja całkowita X = 0,2 + 0,3 = 0,5  ->  Y12 = 2
       P_e = E V sin(delta)/X = 2,22 sin(delta)
 6) Równanie kołysań (H = 4, f = 50 Hz):
       (H/(180 f)) d2(delta)/dt2 = 0,8 - 2,22 sin(delta)   [delta w stopniach]
       0,000444 d2(delta)/dt2 = 0,8 - 2,22 sin(delta)

ROZSZERZENIE INŻYNIERSKIE - SYMULACJA ZWARCIA
Książka kończy na równaniu. Tu rozwiązujemy je numerycznie (metoda Rungego-Kutty)
dla zwarcia trójfazowego na jednej z linii:
  * przed zwarciem:  P_max1 = 2,22
  * w czasie zwarcia: zależy od miejsca (m = 0 przy szynie -> P_e = 0)
  * po wyłączeniu linii (zostaje jedna): X = 0,2 + 0,1 + 0,4 = 0,7 -> P_max3 = 1,59
Jeśli zwarcie wyłączymy szybciej niż CZAS KRYTYCZNY - generator się "rozkołysze"
i uspokoi. Jeśli za późno - kąt rośnie bez końca: utrata synchronizmu (awaria!).

ANALOGIA
Huśtawka: popchnięcie (zwarcie) rozbuja ją. Jeśli szybko przestaniesz pchać,
wróci. Jeśli pchasz za długo - zrobi "fikołka" przez górę.

CO ZOBACZYSZ
  Góra: krzywa kołysań delta(t) - zielona = stabilnie, czerwona = utrata synchronizmu.
        Kliknij, aby ustawić czas wyłączenia.
  Dół: charakterystyki P(delta) i pola przyspieszające/hamujące (kryterium równych pól).
"""

    def on_click(self, ev):
        if ev.inaxes is self.ax_t:
            self.setv('tc', ev.xdata)

    def network(self):
        P, Vt, xd, XT, XL, m = (self.v(k) for k in ('P', 'Vt', 'xd', 'XT', 'XL', 'm'))
        V = 1.0
        X_ext = XT + XL / 2
        s_th = P * X_ext / (Vt * V)
        th = math.asin(min(s_th, 1.0))
        vt = complex(Vt * math.cos(th), Vt * math.sin(th))
        I = (vt - V) / complex(0, X_ext)
        E = vt + complex(0, xd) * I
        Em = abs(E)
        p_pre = Em * V / (xd + X_ext)
        # zwarcie na jednej linii w odległości m*XL od szyny wysyłającej:
        # gwiazda Xa = xd+XT, Xb = XL || ((1-m)XL), Xc = m XL -> X_transfer = Xa+Xb+Xa*Xb/Xc
        Xa = xd + XT
        Xb = XL
        Xc = m * XL
        p_fault = 0.0 if Xc < 1e-9 else Em * V / (Xa + Xb + Xa * Xb / Xc)
        p_post = Em * V / (xd + XT + XL)
        return dict(theta=th, I=I, E=E, Em=Em, p_pre=p_pre, p_fault=p_fault, p_post=p_post)

    def compute_and_plot(self):
        net = self.network()
        P, H, tc, D = self.v('P'), self.v('H'), self.v('tc'), self.v('D')
        f = 50.0
        d0 = math.asin(min(P / net['p_pre'], 1.0))
        t, d, w, stable = swing_simulate(H, f, P, net['p_pre'], net['p_fault'], net['p_post'],
                                         tc, d0, D, t_end=2.5, dt=1e-3)
        dcr = critical_clearing_angle(P, net['p_pre'], net['p_fault'], net['p_post'])
        tcr = critical_clearing_time(H, f, P, net['p_pre'], net['p_fault'], net['p_post'], d0, D)

        ax = self.fig.add_subplot(211)
        self.ax_t = ax
        ax.plot(t, np.degrees(d), color=C_GREEN if stable else C_RED, lw=2,
                label='delta(t) - %s' % ('STABILNY' if stable else 'UTRATA SYNCHRONIZMU'))
        ax.axvline(tc, color='k', ls=':', label='t_c = %.3f s' % tc)
        if tcr is not None:
            ax.axvline(tcr, color=C_ORANGE, ls='--', label='t_kr = %.3f s' % tcr)
        ax.set_ylim(0, 360)
        ax.set_xlabel('t [s]')
        ax.set_ylabel('delta [st.]')
        ax.set_title('Krzywa kołysań (RK4)')
        ax.grid(alpha=.3)
        ax.legend(fontsize=8, loc='upper right')

        ax2 = self.fig.add_subplot(212)
        dd = np.linspace(0, PI, 300)
        ax2.plot(np.degrees(dd), net['p_pre'] * np.sin(dd), color=C_BLUE, label='przed zwarciem')
        ax2.plot(np.degrees(dd), net['p_fault'] * np.sin(dd), color=C_RED, label='w czasie zwarcia')
        ax2.plot(np.degrees(dd), net['p_post'] * np.sin(dd), color=C_GREEN, label='po wyłączeniu linii')
        ax2.axhline(P, color='k', ls='--', lw=1, label='P_m')
        idx = np.searchsorted(t, tc)
        dc = d[min(idx, len(d) - 1)]
        if not np.isnan(dc):
            m1 = (dd >= d0) & (dd <= dc)
            ax2.fill_between(np.degrees(dd), net['p_fault'] * np.sin(dd), P, where=m1,
                             color=C_RED, alpha=.25, label='A1 (przyspieszanie)')
            dmax = PI - math.asin(min(P / net['p_post'], 1.0))
            m2 = (dd >= dc) & (dd <= dmax)
            ax2.fill_between(np.degrees(dd), P, net['p_post'] * np.sin(dd), where=m2,
                             color=C_GREEN, alpha=.25, label='A2 max (hamowanie)')
        if dcr is not None:
            ax2.axvline(math.degrees(dcr), color=C_ORANGE, ls='--', lw=1)
        ax2.set_xlabel('delta [st.]')
        ax2.set_ylabel('P [p.u.]')
        ax2.grid(alpha=.3)
        ax2.legend(fontsize=7, ncol=2)

        E = net['E']
        M_deg = H / (180 * f)
        return ('Krok 1: X_zew = X_T + X_L/2 = %.3f p.u.\n'
                'Krok 2: theta = asin(P X/V_t V) = %.2f st.\n'
                'Krok 3: I = %.3f / %.2f st. p.u.\n'
                "Krok 4: E' = %.3f / %.2f st. p.u.\n"
                "Krok 5: X = x'd + X_zew = %.3f -> P_e = %.3f sin(delta)\n"
                'Krok 6: równanie kołysań:\n'
                '   %.6f d2delta/dt2 = %.2f - %.3f sin(delta)   [delta w st.]\n'
                '   delta0 = %.2f st.\n\n'
                'SYMULACJA ZWARCIA (m = %.2f):\n'
                '   P_max w czasie zwarcia = %.3f p.u.\n'
                '   P_max po wyłączeniu    = %.3f p.u.\n'
                '   kąt krytyczny (EAC)    = %s\n'
                '   czas krytyczny         = %s\n'
                '   t_c = %.3f s -> %s\n\n'
                "Książka: E' = 1,11/21,06 st.; P_e = 2,22 sin(delta)"
                % (self.v('XT') + self.v('XL') / 2, math.degrees(net['theta']), abs(net['I']),
                   math.degrees(math.atan2(net['I'].imag, net['I'].real)), abs(E),
                   math.degrees(math.atan2(E.imag, E.real)),
                   self.v('xd') + self.v('XT') + self.v('XL') / 2, net['p_pre'], M_deg, P,
                   net['p_pre'], math.degrees(d0), self.v('m'), net['p_fault'], net['p_post'],
                   '%.2f st.' % math.degrees(dcr) if dcr is not None else 'brak',
                   '%.3f s' % tcr if tcr is not None else '> 1 s (zawsze stabilny)',
                   tc, 'STABILNY' if stable else 'NIESTABILNY'))


class EqualArea(ExampleTab):
    title = 'Kryterium równych pól (13.4 - 13.5): skok mocy mechanicznej'
    params = [
        ('Pmax', 'P_max = E V / X [p.u.]', 0.5, 3, 2.22, 0.01),
        ('Pm0', 'Moc początkowa P_m0 [p.u.]', 0, 2, 0.8, 0.01),
        ('Pm1', 'Moc po skoku P_m1 [p.u.]', 0, 3, 1.6, 0.01),
        ('H', 'H [s]', 1, 10, 4, 0.1),
        ('D', 'Tłumienie D [p.u.]', 0, 30, 0, 0.5),
    ]
    click_hint = 'w górny wykres ustawia P_m1'
    explanation = """
O CO CHODZI?
Generator pracuje z mocą P_m0 przy kącie delta0. Nagle turbina daje więcej:
P_m1. Wirnik nie może od razu zmienić położenia (bezwładność!), więc przez
chwilę turbina daje więcej, niż sieć odbiera -> wirnik PRZYSPIESZA.
Dochodzi do nowego punktu delta1, ale "z rozpędu" przelatuje dalej - do delta2,
gdzie sieć hamuje go mocniej. Potem wraca... i tak się kołysze.

ZASADA RÓWNYCH PÓL
Energia nabrana przy przyspieszaniu = pole A1 (między P_m1 a krzywa, od delta0 do delta1).
Energia oddana przy hamowaniu      = pole A2 (od delta1 do delta2).
Wirnik zatrzyma się (względem sieci) gdy A2 = A1.
  * Jeśli da się znaleźć takie A2 przed punktem delta3 = 180 - delta1 -> STABILNY.
  * Jeśli nawet całe dostępne pole hamujące jest za małe -> UTRATA SYNCHRONIZMU.

ANALOGIA
Piłka tocząca się w dołku: nabrana prędkość (A1) musi zostać wytracona na
zboczu (A2). Jeśli zbocze za niskie - piłka wyskoczy z dołka.

LINEARYZACJA (rozdz. 13.3.6)
Dla małych kołysań:  M d2(Ddelta)/dt2 + D d(Ddelta)/dt + P_s Ddelta = 0,
P_s = P_max cos(delta) - współczynnik mocy synchronizującej.
Częstotliwość kołysań  f_n = sqrt(P_s/M)/(2 pi).  P_s > 0 -> stabilnie,
P_s < 0 (delta > 90 st.) -> niestabilnie.

CO ZOBACZYSZ
  Góra: krzywa P(delta), pola A1 (czerwone) i A2 (zielone). Kliknij = nowe P_m1.
  Dół: symulacja delta(t) - potwierdza wynik "na pola".
EKSPERYMENT: znajdź największe P_m1, przy którym układ jest jeszcze stabilny.
"""

    def on_click(self, ev):
        if ev.inaxes is self.ax_p:
            self.setv('Pm1', ev.ydata)

    def compute_and_plot(self):
        Pmax, Pm0, Pm1, H, D = (self.v(k) for k in ('Pmax', 'Pm0', 'Pm1', 'H', 'D'))
        if Pm0 >= Pmax:
            return 'P_m0 musi być mniejsze od P_max - generator nie może tego oddać.'
        r = equal_area_step(Pmax, Pm0, Pm1)
        ax = self.fig.add_subplot(211)
        self.ax_p = ax
        dd = np.linspace(0, PI, 400)
        ax.plot(np.degrees(dd), Pmax * np.sin(dd), color=C_BLUE, lw=2, label='P_e = P_max sin(delta)')
        ax.axhline(Pm0, color=C_GREY, ls=':', label='P_m0')
        ax.axhline(Pm1, color='k', ls='--', label='P_m1')
        d0 = r['d0']
        if r['d1'] is not None:
            m1 = (dd >= d0) & (dd <= r['d1'])
            ax.fill_between(np.degrees(dd), Pmax * np.sin(dd), Pm1, where=m1, color=C_RED,
                            alpha=.3, label='A1 = %.4f' % r['A1'])
            dend = r['d2'] if r['stable'] else r['d3']
            m2 = (dd >= r['d1']) & (dd <= dend)
            ax.fill_between(np.degrees(dd), Pm1, Pmax * np.sin(dd), where=m2, color=C_GREEN,
                            alpha=.3, label='A2 (%s)' % ('= A1' if r['stable'] else 'max %.4f' % r['A2max']))
            for key, nm in (('d0', 'd0'), ('d1', 'd1'), ('d2', 'd2'), ('d3', 'd3')):
                if r[key] is not None:
                    ax.axvline(math.degrees(r[key]), color=C_GREY, lw=.7)
                    ax.text(math.degrees(r[key]), Pmax * 1.03, nm, ha='center', fontsize=8)
        ax.set_ylim(0, Pmax * 1.15)
        ax.set_xlabel('delta [st.]')
        ax.set_ylabel('P [p.u.]')
        ax.grid(alpha=.3)
        ax.legend(fontsize=7, loc='lower center')

        t, d, w, stable = swing_simulate(H, 50.0, Pm1, Pmax, Pmax, Pmax, 0.0, d0, D, 3.0, 1e-3)
        ax2 = self.fig.add_subplot(212)
        ax2.plot(t, np.degrees(d), color=C_GREEN if stable else C_RED, lw=2)
        ax2.set_ylim(0, 360)
        ax2.set_xlabel('t [s]')
        ax2.set_ylabel('delta [st.]')
        ax2.set_title('Symulacja: %s' % ('STABILNY' if stable else 'UTRATA SYNCHRONIZMU'))
        ax2.grid(alpha=.3)

        M = H / (PI * 50.0)
        s = 'delta0 = asin(P_m0/P_max) = %.2f st.\n' % math.degrees(d0)
        if r['d1'] is None:
            return s + 'P_m1 >= P_max: brak punktu równowagi - natychmiastowa utrata synchronizmu.'
        Ps = Pmax * math.cos(r['d1'])
        fn = math.sqrt(Ps / M) / (2 * PI) if Ps > 0 else float('nan')
        s += ('delta1 = asin(P_m1/P_max) = %.2f st.  (nowy punkt równowagi)\n'
              'delta3 = 180 - delta1 = %.2f st.  (granica)\n\n'
              'Pole przyspieszające A1 = %.4f p.u.*rad\n'
              'Maksymalne pole hamujące A2max = %.4f p.u.*rad\n\n'
              % (math.degrees(r['d1']), math.degrees(r['d3']), r['A1'], r['A2max']))
        if r['stable']:
            s += ('A1 <= A2max -> STABILNY\n'
                  'Maksymalny kąt wychylenia delta2 = %.2f st.\n'
                  '(wirnik kołysze się między %.1f a %.1f st.)\n'
                  % (math.degrees(r['d2']), math.degrees(d0), math.degrees(r['d2'])))
        else:
            s += 'A1 > A2max -> NIESTABILNY (utrata synchronizmu)\n'
        s += ('\nLinearyzacja wokół delta1:\n'
              '  P_s = P_max cos(delta1) = %.4f p.u./rad\n'
              '  f_kołysań = sqrt(P_s/M)/(2 pi) = %.3f Hz  (M = H/(pi f) = %.5f)\n'
              'Symulacja RK4 potwierdza: %s'
              % (Ps, fn, M, 'STABILNY' if stable else 'NIESTABILNY'))
        return s


# =============================================================================
# Aplikacja
# =============================================================================

CHAPTERS = [
    ('Start', [('Start', StartTab)]),
    ('Rozdz. 11 - Ekonomiczna praca', [
        ('11.1 IFC', Ex11_1),
        ('11.2 Dwa bloki', Ex11_2),
        ('11.3 Harmonogram', Ex11_3),
        ('11.4 Straty A-B', Ex11_4),
        ('11.5 Wsp. kary', Ex11_5),
        ('11.6 Koordynacja', Ex11_6),
        ('Ćw. 11 Iteracja', ExerciseLoss),
        ('11.7 Hydro + para', Ex11_7),
        ('11.8 Szczyt', Ex11_8),
    ]),
    ('Rozdz. 12 - HVDC', [
        ('12.1 Łącze bipolarne', Ex12_1),
        ('12.2 Falownik', Ex12_2),
        ('12.3 Kąty α i δ', Ex12_3),
    ]),
    ('Rozdz. 13 - Stabilność', [
        ('13.1 Kąt mocy', Ex13_1),
        ('13.2 Energia wirnika', Ex13_2),
        ('13.3 Kołysania + zwarcie', Ex13_3),
        ('13.4-13.5 Równe pola', EqualArea),
    ]),
]


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Analiza systemu elektroenergetycznego - ekonomika, HVDC, stabilność')
        self.geometry('1400x880')
        self.minsize(1000, 650)
        style = ttk.Style(self)
        try:
            style.theme_use('clam')
        except tk.TclError:
            pass
        style.configure('Title.TLabel', font=('TkDefaultFont', 11, 'bold'), foreground='#123')
        style.configure('Chapter.TNotebook.Tab', font=('TkDefaultFont', 10, 'bold'), padding=(10, 4))
        self.status = tk.StringVar(value='Wybierz rozdział i zakładkę z przykładem.')
        ttk.Label(self, textvariable=self.status, relief=tk.SUNKEN,
                  anchor='w').pack(fill=tk.X, side=tk.BOTTOM)
        top = ttk.Notebook(self, style='Chapter.TNotebook')
        top.pack(fill=tk.BOTH, expand=True)
        self.chapters = top
        self.tabs = []                # (indeks rozdziału, wewn. notebook, zakładka)
        for ci, (cname, items) in enumerate(CHAPTERS):
            if len(items) == 1:
                tab = items[0][1](top, self)
                top.add(tab, text=cname)
                self.tabs.append((ci, None, tab))
                continue
            inner = ttk.Notebook(top)
            top.add(inner, text=cname)
            for name, cls in items:
                tab = cls(inner, self)
                inner.add(tab, text=name)
                self.tabs.append((ci, inner, tab))
            inner.bind('<<NotebookTabChanged>>', self._on_tab)
        top.bind('<<NotebookTabChanged>>', self._on_tab)

    def select(self, i):
        """Pokazuje i-tą zakładkę-przykład (licząc po kolei we wszystkich rozdziałach)."""
        ci, inner, tab = self.tabs[i]
        self.chapters.select(ci)
        if inner is not None:
            inner.select(tab)

    def _on_tab(self, _ev):
        page = self.chapters.nametowidget(self.chapters.select())
        if isinstance(page, ttk.Notebook):
            page = page.nametowidget(page.select())
        if isinstance(page, ExampleTab):
            page.schedule()           # odświeżenie rozmiaru wykresu po przełączeniu
            self.status.set(page.title)


if __name__ == '__main__':
    App().mainloop()
