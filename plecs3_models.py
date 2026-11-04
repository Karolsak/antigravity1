"""
plecs3_models.py - rdzen obliczeniowy aplikacji "Silnik indukcyjny: modele i sterowanie"
=======================================================================================
Modele i symulacje do przykladow z rozdzialow 8 i 9 ksiazki
R. W. De Doncker, D. W. J. Pulle, A. Veltman, "Advanced Electrical Drives"
(zalacznik Plecs3: modele UFO, Heyland, maszyna jednofazowa, V/f, FOC/UFO,
granice pracy napedu, oslabianie pola).

Modul NIE zalezy od tkinter - mozna go testowac i uzywac z linii polecen.
Wymaga tylko numpy.

Konwencje:
  * wektory przestrzenne amplitudowo-niezmiennicze (jak w ksiazce),
  * moment  Te = 3/2 * p * Im{ conj(psi_s) * i_s },
  * omega_m - elektryczna predkosc walu  (= p * Omega_mech),
  * modele w ukladzie nieruchomym (alfa-beta) lub synchronicznym (dq).
"""
import math
import cmath

import numpy as np

PI = math.pi
SQ2 = math.sqrt(2.0)
SQ3 = math.sqrt(3.0)
RPM = 60.0 / (2.0 * PI)          # rad/s (mech.) -> obr/min


# =============================================================================
# 1. PARAMETRY MASZYN
# =============================================================================

# Maszyna "dydaktyczna" z tutoriali 8.6.1 - 8.6.5
SMALL = dict(Rs=6.9, Rr=3.0, Lm=0.3409, Ls=0.3469, Lr=0.3469, p=2, J=0.001)

# Maszyna 17,3 kW (w tekscie rozdz. 9 nazywana "22 kW"), polaczenie w trojkat,
# Tabela 8.2 (parametry a = 1) - wynik Listingu 8.1
M17 = dict(Rs=0.600, Rr=0.5408, Lm=0.26000, Ls=0.27205, Lr=0.27205, p=2, J=0.1,
           U_line=415.0, f=50.0, n_nom=1465.0, T_nom=113.0)

# Maszyna jednofazowa z kondensatorem pracy - Tabela 8.4
SINGLE = dict(LsS=0.0128, Rrun=2.02, Raux=7.14, RR=3.87, LM=0.1718, kaux=1.18,
              C=31.0e-6, J=1.46e-2, p=2, U=110.0, f=50.0)


def universal(par, a):
    """Parametry modelu uniwersalnego (rown. 8.12, 8.15) dla wspolczynnika a.
    Zwraca slownik: LsS (L_sigma_S), LM, LsR (L_sigma_R), RR, LR, Ls."""
    Ls, Lr, Lm, Rr = par["Ls"], par["Lr"], par["Lm"], par["Rr"]
    LM = a * Lm
    LsS = Ls - a * Lm
    LsR = a * a * Lr - a * Lm
    RR = a * a * Rr
    return dict(LsS=LsS, LM=LM, LsR=LsR, RR=RR, LR=LM + LsR, Ls=Ls, a=a)


def a_limits(par):
    """Zakres wspolczynnika transformacji Lm/Lr <= a <= Ls/Lm (rown. 8.13)."""
    return par["Lm"] / par["Lr"], par["Ls"] / par["Lm"]


def rotor_model(par):
    """Parametry modelu 'rotor flux based' (a = Lm/Lr) - Tabela 8.3."""
    return universal(par, par["Lm"] / par["Lr"])


# =============================================================================
# 2. NARZEDZIA LTI (zastepuja pakiet control / scipy - tylko numpy)
# =============================================================================

def _trim(c):
    c = np.atleast_1d(np.asarray(c, dtype=complex))
    nz = np.nonzero(np.abs(c) > 1e-14 * max(1.0, np.max(np.abs(c))))[0]
    if len(nz) == 0:
        return np.array([0.0])
    c = c[nz[0]:]
    if np.max(np.abs(c.imag)) < 1e-9 * max(1.0, np.max(np.abs(c.real))):
        c = c.real
    return c


def expm(A):
    """Wykladnik macierzy (skalowanie i podnoszenie do kwadratu + Taylor)."""
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    nrm = np.max(np.sum(np.abs(A), axis=1)) if n else 0.0
    s = max(0, int(math.ceil(math.log2(nrm / 0.25))) if nrm > 0.25 else 0)
    As = A / (2 ** s)
    E = np.eye(n)
    term = np.eye(n)
    for k in range(1, 20):
        term = term @ As / k
        E = E + term
    for _ in range(s):
        E = E @ E
    return E


def pade(T, order=2):
    """Aproksymacja Pade opoznienia e^{-sT}: zwraca (num, den)."""
    if T <= 0:
        return np.array([1.0]), np.array([1.0])
    if order == 1:
        return np.array([-T / 2, 1.0]), np.array([T / 2, 1.0])
    if order == 2:
        return np.array([T * T / 12, -T / 2, 1.0]), np.array([T * T / 12, T / 2, 1.0])
    return np.array([-T ** 3 / 120, T * T / 10, -T / 2, 1.0]), \
        np.array([T ** 3 / 120, T * T / 10, T / 2, 1.0])


class TF:
    """Transmitancja SISO G(s) = num(s)/den(s) * e^{-s*delay}."""

    def __init__(self, num, den, delay=0.0, name="G"):
        num = _trim(num)
        den = _trim(den)
        lead = den[0]

        def _real(c):
            c = np.asarray(c)
            if np.iscomplexobj(c) and np.max(np.abs(c.imag)) <= 1e-6 * max(1e-300, np.max(np.abs(c.real))):
                return c.real.astype(float)
            return c
        self.num = _real(num / lead)
        self.den = _real(den / lead)
        self.delay = float(delay)
        self.name = name

    # --- konstrukcja --------------------------------------------------------
    @staticmethod
    def from_zpk(z, p, k, delay=0.0, name="G"):
        return TF(k * np.poly(z) if len(z) else [k], np.poly(p) if len(p) else [1.0],
                  delay, name)

    @staticmethod
    def from_ss(A, B, C, D=0.0, name="G"):
        """SISO z przestrzeni stanu. Mianownik = det(sI-A); licznik wyznaczany
        przez interpolacje G(s)*den(s) na okregu |s| = rho (dobrze uwarunkowane)."""
        A = np.atleast_2d(np.asarray(A, float))
        B = np.asarray(B, float).reshape(-1, 1)
        C = np.asarray(C, float).reshape(1, -1)
        n = A.shape[0]
        pa = np.real(np.poly(A))
        rho = max(1.0, float(np.max(np.abs(np.linalg.eigvals(A)))))
        N = n + 1
        zk = np.exp(2j * PI * np.arange(N) / N)
        v = np.zeros(N, complex)
        for i, z in enumerate(zk):
            s = rho * z
            g = (C @ np.linalg.solve(s * np.eye(n) - A, B))[0, 0] + D
            v[i] = g * np.polyval(pa, s)
        c = np.fft.fft(v) / N                      # wspolczynniki z^0..z^n
        c = np.real(c)
        cmax = np.max(np.abs(c)) if np.max(np.abs(c)) > 0 else 1.0
        c[np.abs(c) < 1e-9 * cmax] = 0.0
        coef = c / rho ** np.arange(N)             # wspolczynniki s^0..s^n
        return TF(coef[::-1], pa, 0.0, name)

    # --- algebra ------------------------------------------------------------
    def __mul__(self, o):
        if not isinstance(o, TF):
            return TF(self.num * o, self.den, self.delay, self.name)
        return TF(np.polymul(self.num, o.num), np.polymul(self.den, o.den),
                  self.delay + o.delay, self.name)

    __rmul__ = __mul__

    def __add__(self, o):
        if not isinstance(o, TF):
            o = TF([o], [1.0])
        return TF(np.polyadd(np.polymul(self.num, o.den), np.polymul(o.num, self.den)),
                  np.polymul(self.den, o.den), max(self.delay, o.delay), self.name)

    def rational(self, order=2):
        """Wersja wymierna (opoznienie zastapione aproksymacja Pade)."""
        if self.delay <= 0:
            return self
        pn, pd = pade(self.delay, order)
        return TF(np.polymul(self.num, pn), np.polymul(self.den, pd), 0.0, self.name)

    def feedback(self, order=2):
        """Zamkniecie petli ujemnego sprzezenia: T = L/(1+L)."""
        L = self.rational(order)
        return TF(L.num, np.polyadd(L.den, L.num), 0.0, "T").minreal()

    def minreal(self, tol=1e-6):
        """Skrocenie (prawie) wspolnych zer i biegunow."""
        z = list(self.zeros())
        p = list(self.poles())
        k = self.k()
        changed = True
        while changed and z and p:
            changed = False
            for i, zi in enumerate(z):
                d = [abs(zi - pj) / max(1.0, abs(pj)) for pj in p]
                j = int(np.argmin(d))
                if d[j] < tol * 1e3:
                    z.pop(i)
                    p.pop(j)
                    changed = True
                    break
        g = TF.from_zpk(z, p, k, self.delay, self.name)
        return g

    # --- wlasnosci ----------------------------------------------------------
    def poles(self):
        return np.roots(self.den) if len(self.den) > 1 else np.array([])

    def zeros(self):
        return np.roots(self.num) if len(self.num) > 1 else np.array([])

    def k(self):
        return float(np.real(self.num[0] / self.den[0]))

    def proper(self):
        return len(self.num) <= len(self.den)

    def n_integrators(self):
        return int(np.sum(np.abs(self.poles()) < 1e-9))

    def dcgain(self):
        if self.n_integrators() > 0:
            return float("inf")
        d = np.polyval(self.den, 0.0)
        return float(np.real(np.polyval(self.num, 0.0) / d))

    def __call__(self, s):
        s = np.asarray(s, dtype=complex)
        g = np.polyval(self.num, s) / np.polyval(self.den, s)
        if self.delay > 0:
            g = g * np.exp(-s * self.delay)
        return g

    def freqresp(self, w):
        return self(1j * np.asarray(w, float))

    def wrange(self, n=600):
        f = [abs(x) for x in list(self.poles()) + list(self.zeros()) if abs(x) > 1e-9]
        if self.delay > 0:
            f.append(1.0 / self.delay)
        if not f:
            f = [1.0]
        lo = 10 ** math.floor(math.log10(min(f) / 20.0))
        hi = 10 ** math.ceil(math.log10(max(f) * 20.0))
        return np.logspace(math.log10(lo), math.log10(hi), n)

    def t_auto(self):
        p = [x for x in self.poles() if abs(x) > 1e-9]
        if not p:
            return 1.0
        sl = min(abs(x.real) for x in p if abs(x.real) > 1e-9) if any(abs(x.real) > 1e-9 for x in p) else 1.0
        osc = max([abs(x.imag) for x in p] + [1e-9])
        t = 6.0 / sl
        if osc > 1e-6:
            t = max(t, 4 * 2 * PI / osc)
        return min(max(t, 1e-3), 200.0)

    # --- odpowiedz skokowa --------------------------------------------------
    def step(self, t_end=None, n=1500, order=2):
        """Odpowiedz na skok jednostkowy (delay -> przesuniecie w czasie)."""
        if not self.proper():
            return None, None
        g = TF(self.num, self.den, 0.0)
        if t_end is None:
            t_end = self.t_auto() + self.delay
        t = np.linspace(0.0, t_end, n)
        dt = t[1] - t[0]
        nd = len(g.den) - 1
        num = np.real(np.concatenate([np.zeros(nd + 1 - len(g.num)), g.num]))
        D = num[0]
        if nd == 0:
            y = np.full(n, D)
        else:
            a = np.real(g.den[1:])
            b = num[1:] - D * a
            A = np.zeros((nd, nd))
            A[0, :] = -a
            if nd > 1:
                A[1:, :-1] = np.eye(nd - 1)
            B = np.zeros(nd)
            B[0] = 1.0
            C = b
            M = np.zeros((nd + 1, nd + 1))
            M[:nd, :nd] = A * dt
            M[:nd, nd] = B * dt
            E = expm(M)
            Ad = E[:nd, :nd]
            Bd = E[:nd, nd]
            x = np.zeros(nd)
            y = np.zeros(n)
            for i in range(n):
                y[i] = C @ x + D
                x = Ad @ x + Bd
                if not np.all(np.isfinite(x)) or np.max(np.abs(x)) > 1e12:
                    y[i + 1:] = np.nan
                    break
        if self.delay > 0:
            sh = int(round(self.delay / dt))
            y = np.concatenate([np.zeros(sh), y[:n - sh]])
        return t, np.real(y)

    # --- tekst zpk ----------------------------------------------------------
    def zpk_str(self, var="s"):
        def fac(r):
            if abs(r) < 1e-12:
                return var
            if abs(r.imag) < 1e-9:
                v = -r.real
                return f"({var} {'+' if v >= 0 else '-'} {fmt(abs(v))})"
            return None
        def pair(r):
            a = -2 * r.real
            b = abs(r) ** 2
            return f"({var}² {'+' if a >= 0 else '-'} {fmt(abs(a))}{var} + {fmt(b)})"
        def factors(roots):
            out = []
            used = [False] * len(roots)
            for i, r in enumerate(roots):
                if used[i]:
                    continue
                if abs(r.imag) > 1e-9:
                    for j in range(i + 1, len(roots)):
                        if not used[j] and abs(roots[j] - np.conj(r)) < 1e-6 * max(1, abs(r)):
                            used[j] = True
                            break
                    out.append(pair(r))
                else:
                    out.append(fac(complex(r)))
                used[i] = True
            return "·".join(out) if out else "1"
        z = np.sort_complex(self.zeros())
        p = np.sort_complex(self.poles())
        s = f"{fmt(self.k())} · {factors(z)} / [{factors(p)}]"
        if self.delay > 0:
            s += f" · e^(-{fmt(self.delay)}·{var})"
        return s


def fmt(x, d=4):
    """Krotkie formatowanie liczb."""
    if isinstance(x, complex):
        if abs(x.imag) < 1e-12:
            x = x.real
        else:
            return f"{fmt(x.real, d)}{'+' if x.imag >= 0 else '-'}j{fmt(abs(x.imag), d)}"
    if x == 0:
        return "0"
    if not math.isfinite(x):
        return "∞"
    ax = abs(x)
    if ax >= 1e5 or ax < 1e-3:
        return f"{x:.{d - 1}e}"
    return f"{x:.{d}g}"


def margins(L):
    """Zapasy stabilnosci petli otwartej L(jw).
    Zwraca dict: wc (przejscie |L|=1), pm (zapas fazy, st.), w180, gm (dB)."""
    w = np.logspace(-3, 7, 20000)
    G = L.freqresp(w)
    mag = np.abs(G)
    ph = np.unwrap(np.angle(G)) * 180 / PI
    # normalizacja fazy tak, by przy malych w byla w (-360, 0]
    if ph[0] > 0:
        ph -= 360.0 * math.ceil(ph[0] / 360.0)
    res = dict(wc=None, pm=None, w180=None, gm=None)
    idx = np.nonzero((mag[:-1] >= 1.0) & (mag[1:] < 1.0))[0]
    if len(idx):
        i = idx[-1] if len(idx) else idx[0]
        i = idx[0]
        f = (math.log(mag[i]) - 0.0) / (math.log(mag[i]) - math.log(mag[i + 1]))
        wc = math.exp(math.log(w[i]) + f * (math.log(w[i + 1]) - math.log(w[i])))
        phc = ph[i] + f * (ph[i + 1] - ph[i])
        res["wc"] = wc
        res["pm"] = 180.0 + phc
    for k in range(-5, 6):
        lvl = -180.0 + 360.0 * k
        idx = np.nonzero((ph[:-1] - lvl) * (ph[1:] - lvl) < 0)[0]
        if len(idx):
            i = idx[0]
            f = (ph[i] - lvl) / (ph[i] - ph[i + 1])
            w180 = math.exp(math.log(w[i]) + f * (math.log(w[i + 1]) - math.log(w[i])))
            m = mag[i] * (mag[i + 1] / mag[i]) ** f
            if res["w180"] is None or w180 < res["w180"]:
                res["w180"] = w180
                res["gm"] = -20 * math.log10(max(m, 1e-300))
    return res


def step_info(t, y):
    """Parametry odpowiedzi skokowej: wartosc koncowa, czas narastania 10-90 %,
    czas ustalania 2 %, przeregulowanie %."""
    if t is None or y is None or not np.all(np.isfinite(y)):
        return None
    yf = y[-1]
    if abs(yf) < 1e-12:
        return dict(yf=yf, tr=None, ts=None, os=None, peak=float(np.max(np.abs(y))))
    yn = y / yf
    tr = None
    try:
        i10 = np.nonzero(yn >= 0.1)[0][0]
        i90 = np.nonzero(yn >= 0.9)[0][0]
        tr = t[i90] - t[i10]
    except IndexError:
        pass
    out = np.nonzero(np.abs(yn - 1.0) > 0.02)[0]
    ts = t[out[-1] + 1] if len(out) and out[-1] + 1 < len(t) else (t[0] if not len(out) else None)
    os_ = max(0.0, (np.max(yn) - 1.0) * 100.0)
    return dict(yf=yf, tr=tr, ts=ts, os=os_, peak=float(np.max(y)))


def describe_poles(G):
    """Lista opisow biegunow: (biegun, opis)."""
    out = []
    for p in sorted(G.poles(), key=lambda x: (x.real, -x.imag)):
        if p.imag < -1e-9:
            continue
        if abs(p) < 1e-9:
            out.append((p, "biegun w zerze → człon całkujący (sumuje wejście, jak licznik)"))
        elif abs(p.imag) < 1e-9:
            if p.real < 0:
                out.append((p, f"rzeczywisty stabilny, stała czasowa τ = {fmt(-1 / p.real)} s"))
            else:
                out.append((p, f"rzeczywisty NIESTABILNY (prawa półpłaszczyzna), narasta jak e^({fmt(p.real)}t)"))
        else:
            wn = abs(p)
            z = -p.real / wn
            fo = abs(p.imag) / (2 * PI)
            st = "stabilna" if p.real < 0 else "NIESTABILNA"
            out.append((p, f"para zespolona {st}: ωn = {fmt(wn)} rad/s, ζ = {fmt(z, 3)}, "
                           f"oscylacje {fmt(fo)} Hz"))
    return out


def lti_report(G, kind="plant", extra=""):
    """Automatyczny opis wynikow analizy (jezyk dla ucznia liceum).
    kind = 'plant' (obiekt / uklad zamkniety) lub 'loop' (petla otwarta L(s))."""
    lines = []
    lines.append(f"Transmitancja  {G.name}(s)")
    lines.append("Postać zpk (zera-bieguny-wzmocnienie):")
    lines.append("   " + G.zpk_str())
    z = G.zeros()
    lines.append("")
    lines.append("ZERA: " + (", ".join(fmt(complex(x)) for x in z) if len(z) else "brak"))
    if any(x.real > 1e-9 for x in z):
        fast = all(abs(x) > 50 * max([abs(q) for q in G.poles()] + [1.0]) for x in z if x.real > 1e-9)
        lines.append("   ! zero w prawej półpłaszczyźnie (nieminimalnofazowe) – odpowiedź "
                     "może najpierw pójść w złą stronę." + (
                         " Tu leży bardzo daleko (dużo dalej niż bieguny), więc praktycznie nie "
                         "ma znaczenia." if fast else ""))
    lines.append("BIEGUNY:")
    unstable = False
    for p, d in describe_poles(G):
        lines.append(f"   s = {fmt(complex(p))}:  {d}")
        if p.real > 1e-9:
            unstable = True
    if G.delay > 0:
        lines.append(f"OPÓŹNIENIE: {fmt(G.delay * 1e3)} ms (próbkowanie + PWM) – w Bode "
                     f"obraca fazę o −ω·T, nie zmienia amplitudy.")
    lines.append("")
    if kind == "loop":
        m = margins(G)
        T = G.feedback()
        cl_poles = T.poles()
        cl_unstable = any(p.real > 1e-7 for p in cl_poles)
        P = sum(1 for p in G.poles() if p.real > 1e-9)
        lines.append("ANALIZA PĘTLI OTWARTEJ L(s):")
        if m["wc"]:
            lines.append(f"   pulsacja odcięcia ωc = {fmt(m['wc'])} rad/s ({fmt(m['wc'] / 2 / PI)} Hz)"
                         f"  → pasmo regulacji, szybkość układu ≈ 1/ωc = {fmt(1 / m['wc'])} s")
            lines.append(f"   zapas fazy PM = {fmt(m['pm'], 3)}°  "
                         + ("(bardzo dobry, prawie bez przeregulowania)" if m['pm'] > 60 else
                            "(dobry, umiarkowane przeregulowanie)" if m['pm'] > 45 else
                            "(mały – wyraźne oscylacje)" if m['pm'] > 0 else
                            "(UJEMNY – układ zamknięty niestabilny)"))
        else:
            lines.append("   |L| nie przecina 0 dB w badanym zakresie.")
        if m["gm"] is not None:
            k = 10 ** (m['gm'] / 20)
            lines.append(f"   zapas wzmocnienia GM = {fmt(m['gm'], 3)} dB przy ω180 = {fmt(m['w180'])} rad/s"
                         + (f"  → wzmocnienie można zwiększyć {fmt(k, 3)} razy do granicy stabilności"
                            if k >= 1 else
                            f"  → wzmocnienie trzeba ZMNIEJSZYĆ {fmt(1 / k, 3)} razy, by odzyskać stabilność"))
        else:
            lines.append("   faza nie przecina −180° → zapas wzmocnienia nieskończony.")
        lines.append(f"   Nyquist: L ma {P} biegun(y) niestabilne; układ zamknięty "
                     + ("NIESTABILNY (wykres obejmuje punkt −1)." if cl_unstable else
                        "STABILNY (wykres nie obejmuje punktu −1)."))
        lines.append("   Bieguny układu zamkniętego T = L/(1+L):")
        for p, d in describe_poles(T):
            lines.append(f"      s = {fmt(complex(p))}: {d}")
        t, y = T.step()
        unstable = cl_unstable
    else:
        dc = G.dcgain()
        lines.append("WZMOCNIENIE STATYCZNE (dla s→0): " + (
            "nieskończone – człon całkujący, wyjście rośnie bez końca przy stałym wejściu"
            if not math.isfinite(dc) else fmt(dc)))
        lines.append("STABILNOŚĆ: " + ("NIESTABILNY – co najmniej jeden biegun ma Re > 0!" if unstable
                                       else "stabilny (wszystkie bieguny po lewej stronie)"
                                       if G.n_integrators() == 0 else
                                       "na granicy (człon całkujący) – sam się nie zatrzyma"))
        w = G.wrange()
        mag = 20 * np.log10(np.abs(G.freqresp(w)) + 1e-300)
        if math.isfinite(dc) and dc != 0 and not unstable:
            ref = 20 * math.log10(abs(dc))
            idx = np.nonzero(mag < ref - 3.0)[0]
            if len(idx):
                lines.append(f"PASMO (−3 dB): ω_B ≈ {fmt(w[idx[0]])} rad/s ({fmt(w[idx[0]] / 2 / PI)} Hz)"
                             " – do tej częstotliwości układ 'nadąża' za wejściem.")
        t, y = G.step()
    si = step_info(t, y) if (t is not None and not unstable) else None
    lines.append("")
    if unstable and t is not None:
        lines.append("ODPOWIEDŹ SKOKOWA: oscylacje NARASTAJĄ (układ niestabilny) – parametry odpowiedzi "
                     "(przeregulowanie, czas ustalania) nie mają sensu.")
    elif si:
        lines.append("ODPOWIEDŹ SKOKOWA:")
        lines.append(f"   wartość końcowa {fmt(si['yf'])}")
        if si["tr"] is not None:
            lines.append(f"   czas narastania 10→90 %: {fmt(si['tr'])} s")
        if si["ts"] is not None:
            lines.append(f"   czas ustalania (±2 %): {fmt(si['ts'])} s")
        lines.append(f"   przeregulowanie: {fmt(si['os'], 3)} %")
    elif t is None:
        lines.append("ODPOWIEDŹ SKOKOWA: transmitancja niewłaściwa (licznik wyższego stopnia) – "
                     "skok na wejściu daje impuls Diraca na wyjściu.")
    else:
        lines.append("ODPOWIEDŹ SKOKOWA: rośnie bez ograniczeń (niestabilność lub całkowanie).")
    if extra:
        lines.append("")
        lines.append(extra)
    return "\n".join(lines)


def linearize(f, x0, u0, h=1e-6):
    """Linearyzacja numeryczna dx/dt = f(x,u): zwraca A, B."""
    x0 = np.asarray(x0, float)
    u0 = np.asarray(u0, float)
    A = np.zeros((len(x0), len(x0)))
    B = np.zeros((len(x0), len(u0)))
    for i in range(len(x0)):
        dx = np.zeros(len(x0))
        dx[i] = h * max(1.0, abs(x0[i]))
        A[:, i] = (f(x0 + dx, u0) - f(x0 - dx, u0)) / (2 * dx[i])
    for i in range(len(u0)):
        du = np.zeros(len(u0))
        du[i] = h * max(1.0, abs(u0[i]))
        B[:, i] = (f(x0, u0 + du) - f(x0, u0 - du)) / (2 * du[i])
    return A, B


def newton(F, x0, tol=1e-10, it=80):
    x = np.asarray(x0, float).copy()
    for _ in range(it):
        Fx = F(x)
        if np.max(np.abs(Fx)) < tol:
            break
        Jm = np.zeros((len(x), len(x)))
        for i in range(len(x)):
            dx = np.zeros(len(x))
            dx[i] = 1e-7 * max(1.0, abs(x[i]))
            Jm[:, i] = (F(x + dx) - Fx) / dx[i]
        try:
            x = x - np.linalg.solve(Jm, Fx)
        except np.linalg.LinAlgError:
            break
    return x


# =============================================================================
# 3. ROZDZIAL 8 - MODELE MASZYNY
# =============================================================================

def quad_load_k(T_nom, n_nom):
    """Wspolczynnik obciazenia kwadratowego T = k*Omega^2 (wentylator, pompa)."""
    W = n_nom / RPM
    return T_nom / (W * W)


# ---------- Tutorial 8.6.1 - model uproszczony (bez rozproszen, zasilanie strumieniem)
def t81_simplified(psi=1.25, f=50.0, Rs=6.9, Rr=3.0, Lm=0.3409, p=2, J=0.001,
                   k_load=4.0e-5, T_end=1.0, n=4000):
    """Strumien psi_m o stalej amplitudzie wiruje z ws. Bez rozproszen:
       i_r^xy = -(1/Rr) d psi^xy/dt  ->  Te = 3/2 p psi^2 (ws - wm)/Rr (linia prosta!).
       Mechanika: J dOmega/dt = Te - k Omega^2."""
    ws = 2 * PI * f
    kT = 1.5 * p * psi * psi / Rr        # Nm / (rad/s el.)
    t = np.linspace(0, T_end, n)
    dt = t[1] - t[0]
    W = 0.0
    Om = np.zeros(n)
    Te = np.zeros(n)
    Tl = np.zeros(n)

    def acc(W):
        te = kT * (ws - p * W)
        tl = k_load * W * abs(W)
        return (te - tl) / J, te, tl
    for i in range(n):
        a1, te, tl = acc(W)
        Om[i], Te[i], Tl[i] = W, te, tl
        a2 = acc(W + 0.5 * dt * a1)[0]
        a3 = acc(W + 0.5 * dt * a2)[0]
        a4 = acc(W + dt * a3)[0]
        W += dt / 6 * (a1 + 2 * a2 + 2 * a3 + a4)
    # stan ustalony analitycznie: kT (ws - p W) = k W^2
    a, b, c = k_load, kT * p, -kT * ws
    W0 = (-b + math.sqrt(b * b - 4 * a * c)) / (2 * a) if a > 0 else ws / p
    wsl = ws - p * W0
    i_s = psi / Lm + 1j * wsl * psi / Rr        # w ukladzie wirujacym ze strumieniem
    u_s = Rs * i_s + 1j * ws * psi
    S = 1.5 * u_s * i_s.conjugate()
    ss = dict(W=W0, n=W0 * RPM, wsl=wsl, slip=wsl / ws, Te=kT * wsl, i_s=abs(i_s),
              u_s=abs(u_s), P=S.real, Q=S.imag, Pmech=kT * wsl * W0,
              Pcu_r=1.5 * Rr * (wsl * psi / Rr) ** 2, Pcu_s=1.5 * Rs * abs(i_s) ** 2,
              cosphi=S.real / abs(S), kT=kT, i_m=psi / Lm, i_r=wsl * psi / Rr)
    ss["eta"] = ss["Pmech"] / ss["P"] if ss["P"] > 0 else 0.0
    return t, Om * RPM, Te, Tl, ss


# ---------- Tutoriale 8.6.2, 8.6.3, 8.6.7 - model uniwersalny zasilany napieciem
def sim_voltage_im(par, a=1.0, u_amp=392.7, f=50.0, T_end=2.0, dt=1e-4,
                   load=None, n_out=2000, ramp=0.0):
    """Model uniwersalny (Fig. 8.13/8.15) w ukladzie nieruchomym.
    Stany: psi_s, psi_R (zespolone), Omega (mech.).
    load(t, Omega) -> moment obciazenia [Nm].
    Zwraca slownik przebiegow. Wynik (moment, predkosc, prady) NIE zalezy od 'a'."""
    up = universal(par, a)
    Ls, LM, LR, RR = up["Ls"], up["LM"], up["LR"], up["RR"]
    Rs, p, J = par["Rs"], par["p"], par["J"]
    D = Ls * LR - LM * LM
    ws = 2 * PI * f
    Lsig_s = par["Ls"] - par["Lm"]          # rozproszenie 'fizyczne' (a=1)
    if load is None:
        def load(t, W):
            return 0.0

    def deriv(t, ps, pR, W):
        amp = u_amp * (min(1.0, t / ramp) if ramp > 0 else 1.0)
        us = amp * cmath.exp(1j * ws * t)
        i_s = (LR * ps - LM * pR) / D
        i_R = (Ls * pR - LM * ps) / D
        dps = us - Rs * i_s
        dpR = -RR * i_R + 1j * p * W * pR
        Te = 1.5 * p * (ps.real * i_s.imag - ps.imag * i_s.real)
        dW = (Te - load(t, W)) / J
        return dps, dpR, dW, Te, i_s, us

    N = int(round(T_end / dt))
    every = max(1, N // n_out)
    ps = 0j
    pR = 0j
    W = 0.0
    out = {k: [] for k in ("t", "Te", "Tl", "n", "isa", "isb", "isc", "ua",
                           "psis", "psiM", "psiR", "psim", "is_abs", "ang", "isr")}
    for k in range(N + 1):
        t = k * dt
        d1 = deriv(t, ps, pR, W)
        if k % every == 0:
            i_s = d1[4]
            psiM = ps - up["LsS"] * i_s
            psim = ps - Lsig_s * i_s                 # nieprzetransformowany strumien szczeliny
            rot = cmath.exp(-1j * cmath.phase(psim)) if abs(psim) > 1e-9 else 1.0
            out["t"].append(t)
            out["Te"].append(d1[3])
            out["Tl"].append(load(t, W))
            out["n"].append(W * RPM)
            ia = i_s.real
            ib = -0.5 * i_s.real + SQ3 / 2 * i_s.imag
            out["isa"].append(ia)
            out["isb"].append(ib)
            out["isc"].append(-ia - ib)
            out["ua"].append(d1[5].real)
            out["psis"].append(ps * rot)
            out["psiM"].append(psiM * rot)
            out["psiR"].append(pR * rot)
            out["psim"].append(abs(psim))
            out["is_abs"].append(abs(i_s))
            out["isr"].append(i_s * rot)
        h = dt
        d2 = deriv(t + h / 2, ps + h / 2 * d1[0], pR + h / 2 * d1[1], W + h / 2 * d1[2])
        d3 = deriv(t + h / 2, ps + h / 2 * d2[0], pR + h / 2 * d2[1], W + h / 2 * d2[2])
        d4 = deriv(t + h, ps + h * d3[0], pR + h * d3[1], W + h * d3[2])
        ps += h / 6 * (d1[0] + 2 * d2[0] + 2 * d3[0] + d4[0])
        pR += h / 6 * (d1[1] + 2 * d2[1] + 2 * d3[1] + d4[1])
        W += h / 6 * (d1[2] + 2 * d2[2] + 2 * d3[2] + d4[2])
    return {k: np.array(v) for k, v in out.items()}


# ---------- Tutorial 8.6.4 - model pradowy bez rozproszen (zero leakage, UFO)
def t84_zero_leakage(isd=4.0, isq=4.0, t_q=0.8, Rr=3.0, Lm=0.3409, p=2, J=0.1,
                     T_load=0.0, T_end=2.0, n=4000):
    """dpsi/dt = Rr (isd - psi/Lm);  w_sl = Rr isq/psi;  Te = 3/2 p psi isq."""
    t = np.linspace(0, T_end, n)
    dt = t[1] - t[0]
    psi = 0.0
    W = 0.0
    res = np.zeros((n, 6))
    for i in range(n):
        iq = isq if t[i] >= t_q else 0.0
        wsl = Rr * iq / psi if psi > 1e-6 else 0.0
        Te = 1.5 * p * psi * iq
        wm = p * W
        res[i] = (psi, Te, wm, wsl, wm + wsl, iq)
        # dokladne calkowanie liniowego rownania strumienia
        tau = Lm / Rr
        psi = Lm * isd + (psi - Lm * isd) * math.exp(-dt / tau)
        W += dt * (Te - T_load * (1 if W > 0 else 0)) / J
    return t, res


# ---------- Tutorial 8.6.5 - model UFO pradowy z rozproszeniami
def t85_ufo(par, a, isd=4.0, isq=4.0, t_q=0.8, t_ramp=0.01, J=0.1, T_load=0.0,
            T_end=2.0, dt=2e-5, n_out=3000):
    """Generyczny model UFO (Fig. 8.22) z wymuszeniem pradowym isd, isq (rampy).
    (LR/LM) dpsiM/dt = RR (isd - psiM/LM) + LsR disd/dt - w_sl LsR isq
    w_sl = (RR isq + LsR disq/dt) / (psiM LR/LM - LsR isd)
    Te = 3/2 p psiM isq."""
    up = universal(par, a)
    LM, LsR, RR, LR, LsS = up["LM"], up["LsR"], up["RR"], up["LR"], up["LsS"]
    p = par["p"]
    tr = max(t_ramp, 1e-5)

    def cur(t):
        d = isd * min(1.0, t / tr)
        dd = isd / tr if t < tr else 0.0
        if t < t_q:
            q, dq = 0.0, 0.0
        else:
            q = isq * min(1.0, (t - t_q) / tr)
            dq = isq / tr if t - t_q < tr else 0.0
        return d, dd, q, dq

    def f(t, psi):
        d, dd, q, dq = cur(t)
        den = psi * LR / LM - LsR * d
        wsl = (RR * q + LsR * dq) / den if abs(den) > 1e-6 else 0.0
        dpsi = (RR * (d - psi / LM) + LsR * dd - wsl * LsR * q) * LM / LR
        return dpsi, wsl, d, q

    N = int(round(T_end / dt))
    every = max(1, N // n_out)
    psi = 0.0
    W = 0.0
    out = {k: [] for k in ("t", "psiM", "psis", "psiR", "Te", "wm", "wsl", "ws", "isd", "isq")}
    for k in range(N + 1):
        t = k * dt
        d1, wsl, d, q = f(t, psi)
        Te = 1.5 * p * psi * q
        if k % every == 0:
            i_s = complex(d, q)
            iR = psi / LM - i_s
            out["t"].append(t)
            out["psiM"].append(psi)
            out["psis"].append(abs(psi + LsS * i_s))
            out["psiR"].append(abs(psi + LsR * iR))
            out["Te"].append(Te)
            out["wm"].append(p * W)
            out["wsl"].append(wsl)
            out["ws"].append(p * W + wsl)
            out["isd"].append(d)
            out["isq"].append(q)
        d2 = f(t + dt / 2, psi + dt / 2 * d1)[0]
        d3 = f(t + dt / 2, psi + dt / 2 * d2)[0]
        d4 = f(t + dt, psi + dt * d3)[0]
        psi += dt / 6 * (d1 + 2 * d2 + 2 * d3 + d4)
        W += dt * (Te - T_load) / J
    return {k: np.array(v) for k, v in out.items()}, up


# ---------- 8.3.5 - wykres Heylanda w ukladzie synchronicznym (stale psi_M)
def heyland_ufo(par, a, psiM=1.0, wsl_max=200.0, n=801):
    """Stan ustalony modelu UFO przy stalym psi_M:
       isq = wsl psiM / (RR (1 + x^2)),  isd = psiM/LM + x isq,  x = wsl LsR/RR."""
    up = universal(par, a)
    LM, LsR, RR = up["LM"], up["LsR"], up["RR"]
    wsl = np.linspace(-wsl_max, wsl_max, n)
    x = wsl * LsR / RR
    isq = wsl * psiM / (RR * (1 + x * x))
    isd = psiM / LM + x * isq
    Te = 1.5 * par["p"] * psiM * isq
    circ = None
    if LsR > 1e-9:
        circ = dict(cx=psiM / 2 * (1 / LsR + 2 / LM), r=psiM / (2 * LsR))
        circ["wsl_Tmax"] = RR / LsR
        circ["Tmax"] = 1.5 * par["p"] * psiM * circ["r"]
    return wsl, isd, isq, Te, up, circ


# ---------- 8.3.6 / Tutorial 8.6.8 - stan ustalony maszyny zasilanej napieciem
def steady_voltage(par, u_amp, f, n_rpm, delta=True):
    """Schemat zastepczy Fig. 8.31 (Rs, L_sigma_S, L_M || R_R/s) - model 'rotor flux'.
    n_rpm - wektor predkosci [obr/min]. Zwraca slownik charakterystyk."""
    up = rotor_model(par)
    Rs, LsS, LM, RR, p = par["Rs"], up["LsS"], up["LM"], up["RR"], par["p"]
    ws = 2 * PI * f
    n_rpm = np.asarray(n_rpm, float)
    wm = n_rpm / RPM * p
    s = (ws - wm) / ws
    s = np.where(np.abs(s) < 1e-9, 1e-9, s)
    ZM = 1j * ws * LM
    ZR = RR / s
    Zp = ZM * ZR / (ZM + ZR)
    Z = Rs + 1j * ws * LsS + Zp
    i_s = u_amp / Z
    e = i_s * Zp
    i_R = e / ZR
    Te = 1.5 * p * np.abs(i_R) ** 2 * RR / (s * ws)
    psi_s = (u_amp - Rs * i_s) / (1j * ws)
    I_ph = np.abs(i_s) / SQ2
    I_line = I_ph * (SQ3 if delta else 1.0)
    Pin = 1.5 * np.real(u_amp * np.conj(i_s))
    Pout = Te * n_rpm / RPM
    with np.errstate(divide="ignore", invalid="ignore"):
        eta = np.where((Pin > 0) & (Pout > 0), Pout / Pin, np.nan)
    pf = np.cos(np.angle(i_s) * -1)
    # normalizacja jak w ksiazce (Fig. 8.32)
    i_n = i_s / (u_amp / (ws * LsS))
    Te_n = Te / (p * u_amp ** 2 / (2 * ws ** 2 * LsS))
    return dict(n=n_rpm, s=s, i_s=i_s, Te=Te, psi_s=np.abs(psi_s), I_line=I_line,
                Pin=Pin, Pout=Pout, eta=eta, pf=pf, i_n=i_n, Te_n=Te_n, up=up)


# ---------- 8.4 / Tutorial 8.6.6 - estymacja parametrow z tabliczki
def estimate_parameters(U_line=415.0, I_line=30.0, I_noload=8.41, R_line=0.40,
                        P_out=17332.0, n_nom=1465.0, f=50.0, pf=0.80, p=2):
    """Listing 8.1 przepisany na Pythona (maszyna w trojkat). Zwraca liste krokow
    (opis, symbol, wartosc, jednostka) oraz slownik parametrow."""
    st = []

    def add(desc, sym, val, unit):
        st.append((desc, sym, val, unit))
        return val
    ws = add("pulsacja zasilania 2πf", "ωs", 2 * PI * f, "rad/s")
    wm = add("znamionowa elektryczna prędkość wału n·p·2π/60", "ωm", n_nom * p * 2 * PI / 60, "rad/s")
    wsl = add("pulsacja poślizgu ωs − ωm", "ωsl", ws - wm, "rad/s")
    Te = add("moment 'na parę biegunów' P/ωm", "Te", P_out / wm, "Nm")
    Tm = add("moment znamionowy na wale p·Te", "Tm", Te * p, "Nm")
    Rs = add("rezystancja fazy (trójkąt): 3/2·R_l", "Rs", 1.5 * R_line, "Ω")
    isR = add("amplituda prądu fazowego √2·I/√3", "is_nom", SQ2 * I_line / SQ3, "A")
    isN = add("amplituda prądu biegu jałowego", "is_0", SQ2 * I_noload / SQ3, "A")
    usR = add("amplituda napięcia fazowego √2·U (trójkąt)", "us", SQ2 * U_line, "V")
    psi_s = add("strumień stojana bieg jałowy √(us²−(is0·Rs)²)/ωs  (8.34)", "ψs0",
                math.sqrt(usR ** 2 - (isN * Rs) ** 2) / ws, "Vs")
    Ls = add("indukcyjność własna stojana ψs/is0  (8.35)", "Ls", psi_s / isN, "H")
    sinr = math.sqrt(max(0.0, 1 - pf * pf))
    psi_sR = add("strumień stojana przy obciążeniu znamionowym", "ψs_nom",
                 math.sqrt((usR - isR * Rs * pf) ** 2 + (isR * Rs * sinr) ** 2) / ws, "Vs")
    isq = add("prąd q znamionowy 2/3·Te/ψs  (8.37)", "isq_nom", Te / psi_sR * 2 / 3, "A")
    isd = add("prąd d znamionowy √(is²−isq²)  (8.41)", "isd_nom", math.sqrt(max(0.0, isR ** 2 - isq ** 2)), "A")
    is_o = add("prąd magnesujący ψs/Ls", "i_o", psi_sR / Ls, "A")
    ird = add("składowa d prądu wirnika isd − i_o", "iRd", isd - is_o, "A")
    RR = add("rezystancja wirnika (a=Ls/Lm)  (8.38)", "RR(a=Ls/Lm)", wsl * psi_sR * isq / (ird ** 2 + isq ** 2), "Ω")
    LsR = add("indukcyjność rozproszenia (a=Ls/Lm)  (8.40)", "LσR(a=Ls/Lm)", ird / isq * RR / wsl, "H")
    Lr = Ls
    Lm = add("indukcyjność magnesująca √(Ls³/(LσR+Ls))  (8.42)", "Lm", math.sqrt(Ls ** 3 / (LsR + Ls)), "H")
    Lss = add("rozproszenie stojana Ls − Lm", "Lσs = Lσr", Ls - Lm, "H")
    Rr = add("rezystancja wirnika (a=1)  RR/(Ls/Lm)²", "Rr", RR / (Ls / Lm) ** 2, "Ω")
    a = Lm / Lr
    LsS = Lm * (Ls / Lm - a)
    add("--- model 'rotor flux' (a = Lm/Lr) ---", "a", a, "")
    add("indukcyjność magnesująca a·Lm", "LM", a * Lm, "H")
    add("indukcyjność rozproszenia LσS = Ls − a·Lm", "LσS", LsS, "H")
    add("rezystancja wirnika a²·Rr", "RR", a * a * Rr, "Ω")
    psiR = abs(complex(psi_sR - LsS * isd, -LsS * isq))
    add("strumień wirnika |ψs − LσS·is| (fizycznie poprawnie)", "ψR_nom", psiR, "Vs")
    alt = abs(complex(psi_sR - isd * LsS - ird * LsS, -isq * LsS * 2))
    add("ψR wg linii 46 Listingu 8.1 (wartość z Tab. 8.3)", "ψR (książka)", alt, "Vs")
    par = dict(Rs=Rs, Rr=Rr, Lm=Lm, Ls=Ls, Lr=Lr, p=p, Lss=Lss, psi_s=psi_sR, psi_s0=psi_s,
               psiR=psiR, isd=isd, isq=isq, isR=isR, Tm=Tm, wsl=wsl, RR_s=RR, LsR_s=LsR)
    return st, par


# ---------- 8.5 / Tutorial 8.6.9 - maszyna jednofazowa z kondensatorem
def sim_single_phase(P=None, T_load=2.0, t_load=1.0, T_end=2.0, dt=5e-5, n_out=4000,
                     C=None, kT=1.0):
    """Model z Fig. 8.34/8.35/8.49. Os alfa = uzwojenie pomocnicze (przez C),
    os beta = uzwojenie glowne (run). Asymetryczny ITF: k_aux : 1.
    kT - wspolczynnik w momencie Te = kT * p * Im(conj(psi_s) i_s) (maszyna 2-fazowa: 1)."""
    P = dict(SINGLE if P is None else P)
    if C is not None:
        P["C"] = C
    LsS, LM, RR, k, p, J = P["LsS"], P["LM"], P["RR"], P["kaux"], P["p"], P["J"]
    Ra, Rb, Cc = P["Raux"], P["Rrun"], P["C"]
    w = 2 * PI * P["f"]
    U = SQ2 * P["U"]

    def deriv(t, psa, psb, pRa, pRb, uc, W):
        us = U * math.cos(w * t)
        isa = (psa - pRa) / LsS
        isb = (psb - pRb) / LsS
        i_aux = isa / k                         # ITF: i_alfa = k * i_aux
        u_aux = us - uc                          # napiecie na uzwojeniu pomocniczym
        dpsa = (u_aux - Ra * i_aux) / k          # psi_aux = k psi_alfa
        dpsb = us - Rb * isb
        wm = p * W
        dpRa = RR * (isa - pRa / LM) - wm * pRb
        dpRb = RR * (isb - pRb / LM) + wm * pRa
        duc = (i_aux / Cc) if Cc > 0 else 0.0
        Te = kT * p * (psa * isb - psb * isa)
        Tl = T_load if t >= t_load else 0.0
        dW = (Te - Tl) / J
        return (dpsa, dpsb, dpRa, dpRb, duc, dW), Te, Tl, i_aux, isb, u_aux, us, uc
    N = int(round(T_end / dt))
    every = max(1, N // n_out)
    x = [0.0] * 6
    th = 0.0
    out = {k_: [] for k_ in ("t", "Te", "Tl", "n", "i_aux", "i_run", "u_aux", "u_run", "u_c",
                             "pRa", "pRb", "isa", "isb", "th")}
    for kk in range(N + 1):
        t = kk * dt
        d1, Te, Tl, ia, ib, ua, us, uc = deriv(t, *x)
        if kk % every == 0:
            out["t"].append(t)
            out["Te"].append(Te)
            out["Tl"].append(Tl)
            out["n"].append(x[5] * RPM)
            out["i_aux"].append(ia)
            out["i_run"].append(ib)
            out["u_aux"].append(ua)
            out["u_run"].append(us)
            out["u_c"].append(uc)
            out["pRa"].append(x[2])
            out["pRb"].append(x[3])
            out["isa"].append((x[0] - x[2]) / LsS)
            out["isb"].append((x[1] - x[3]) / LsS)
            out["th"].append(th)
        h = dt
        x2 = [x[i] + h / 2 * d1[i] for i in range(6)]
        d2 = deriv(t + h / 2, *x2)[0]
        x3 = [x[i] + h / 2 * d2[i] for i in range(6)]
        d3 = deriv(t + h / 2, *x3)[0]
        x4 = [x[i] + h * d3[i] for i in range(6)]
        d4 = deriv(t + h, *x4)[0]
        Wold = x[5]
        x = [x[i] + h / 6 * (d1[i] + 2 * d2[i] + 2 * d3[i] + d4[i]) for i in range(6)]
        th += p * 0.5 * (Wold + x[5]) * h
    return {k_: np.array(v) for k_, v in out.items()}


def steady_single_phase(P=None, n_rpm=None, C=None, kT=1.0):
    """Analiza stanu ustalonego 8.5.1 metoda skladowych symetrycznych
    (Z_s+ dla poslizgu s, Z_s- dla poslizgu 2-s). Zwraca srednia moment,
    amplitude tetnien (2 f), amplitudy pradow uzwojen."""
    P = dict(SINGLE if P is None else P)
    if C is not None:
        P["C"] = C
    LsS, LM, RR, k, p = P["LsS"], P["LM"], P["RR"], P["kaux"], P["p"]
    Ra, Rb, Cc = P["Raux"], P["Rrun"], P["C"]
    w = 2 * PI * P["f"]
    U = SQ2 * P["U"]
    ns = 60 * P["f"] / p
    if n_rpm is None:
        n_rpm = np.linspace(-ns * 0.999, ns * 0.999, 401)
    res = dict(n=np.asarray(n_rpm, float), Tav=[], Trip=[], Iaux=[], Irun=[], s=[])
    Tm = np.linspace(0, 2 * PI / w, 64, endpoint=False)
    for n in res["n"]:
        wm = n / RPM * p
        s = (w - wm) / w
        sn = 2 - s

        def Zseq(sl):
            ZM = 1j * w * LM
            ZR = RR / sl if abs(sl) > 1e-9 else 1e12
            return 1j * w * LsS + ZM * ZR / (ZM + ZR)
        Zp, Zn = Zseq(s), Zseq(sn)
        # X_alfa = X+ + X-, X_beta = -j X+ + j X-   (X+ = (Xa+jXb)/2, X- = (Xa-jXb)/2)
        Tinv = np.array([[1, 1], [-1j, 1j]])
        T = np.linalg.inv(Tinv)
        Zab = Tinv @ np.diag([Zp, Zn]) @ T
        Zc = (1.0 / (1j * w * Cc)) if Cc > 0 else 0.0
        M = Zab.copy()
        M[0, 0] += Ra / k ** 2 + Zc / k ** 2
        M[1, 1] += Rb
        I = np.linalg.solve(M, np.array([U / k, U]))
        E = Zab @ I                      # napiecia indukowane (alfa, beta) = jw psi
        Psi = E / (1j * w)
        ia = np.real(I[0] * np.exp(1j * w * Tm))
        ib = np.real(I[1] * np.exp(1j * w * Tm))
        pa = np.real(Psi[0] * np.exp(1j * w * Tm))
        pb = np.real(Psi[1] * np.exp(1j * w * Tm))
        Te = kT * p * (pa * ib - pb * ia)
        res["Tav"].append(np.mean(Te))
        res["Trip"].append((np.max(Te) - np.min(Te)) / 2)
        res["Iaux"].append(abs(I[0]) / k)
        res["Irun"].append(abs(I[1]))
        res["s"].append(s)
    for kk in ("Tav", "Trip", "Iaux", "Irun", "s"):
        res[kk] = np.array(res[kk])
    return res


# =============================================================================
# 4. ROZDZIAL 9 - STEROWANIE
# =============================================================================

def stator_model(par):
    """Parametry modelu zorientowanego na strumien stojana (a = Ls/Lm)."""
    return universal(par, par["Ls"] / par["Lm"])


# ---------- 9.1 - wzor Klossa
def kloss_curves(par, psi_s=1.83, ws_list=(100.0, 200.0, 314.16), n=600):
    up = stator_model(par)
    p = par["p"]
    That = 1.5 * p * psi_s ** 2 / (2 * up["LsR"])
    wsl_hat = up["RR"] / up["LsR"]
    curves = []
    for ws in ws_list:
        wm = np.linspace(-0.2 * ws, ws * 1.2, n)
        wsl = ws - wm
        with np.errstate(divide="ignore", invalid="ignore"):
            Te = np.where(np.abs(wsl) > 1e-9, 2 * That / (wsl / wsl_hat + wsl_hat / wsl), 0.0)
        curves.append((ws, wm / p * RPM, Te))
    return curves, That, wsl_hat, up


# ---------- wspolny model napieciowy w ukladzie synchronicznym (dla V/f)
def im_sync_deriv(x, ws, usd, usq, Tl, mp):
    """Model 'rotor flux' (Table 8.3) w ukladzie wirujacym z ws.
    x = [psi_sd, psi_sq, psi_Rd, psi_Rq, Omega]."""
    psd, psq, prd, prq, W = x
    LsS, LM, RR, Rs, p, J = mp
    isd = (psd - prd) / LsS
    isq = (psq - prq) / LsS
    wsl = ws - p * W
    Te = 1.5 * p * (psd * isq - psq * isd)
    return (usd - Rs * isd + ws * psq,
            usq - Rs * isq - ws * psd,
            RR * (isd - prd / LM) + wsl * prq,
            RR * (isq - prq / LM) - wsl * prd,
            (Te - Tl) / J), Te, isd, isq


def vf_machine_params(par, kRR=1.0):
    up = rotor_model(par)
    return (up["LsS"], up["LM"], up["RR"] * kRR, par["Rs"], par["p"], par["J"]), up


def sim_vf(par, mode="speed", psi_ref=1.83, n_ref=1465.0, n_ramp=None, T_ref=113.0,
           kRR=1.0, load_quad=True, T_nom=113.0, n_nom=1465.0, dist_amp=0.0, dist_f=1.0,
           T_end=2.0, dt=2e-4, n_out=3000, ws_rate=None):
    """Symulacja napedu V/f.
    mode='speed'  - Fig. 9.2: wejscie ws* (n_ref lub rampa n_ramp=(n_end, rpm/s)),
                    usd = Rs psi*/Ls, usq = ws* psi*.
    mode='torque' - Fig. 9.5: wejscie T*, pomiar predkosci; ws* = wm + RR isq*/psi*,
                    isq* = 2T*/(3 p psi*), usq = ws* psi* + Rs isq*.
    Model maszyny w ukladzie sterownika (obrot z katem theta_s*)."""
    mp, up = vf_machine_params(par, kRR)
    LsS, LM, RR, Rs, p, J = mp
    Ls = par["Ls"]
    upS = stator_model(par)                    # RR(a=Ls/Lm) dla sterownika momentu
    kq = quad_load_k(T_nom, n_nom) if load_quad else 0.0
    x = [0.0, 0.0, 0.0, 0.0, 0.0]
    th = 0.0
    N = int(round(T_end / dt))
    every = max(1, N // n_out)
    out = {k: [] for k in ("t", "Te", "Tl", "n", "psis", "psiR", "ia", "ws", "nref", "us")}

    def inputs(t, W):
        if mode == "speed":
            if n_ramp is not None:
                n_end, rate = n_ramp
                nr = min(n_end, rate * t)
            else:
                nr = n_ref
            ws = nr / RPM * p
            usd = Rs * psi_ref / Ls
            usq = ws * psi_ref
        else:
            iq = 2 * T_ref / (3 * p * psi_ref)
            ws = p * W + upS["RR"] * iq / psi_ref
            usd = Rs * psi_ref / Ls
            usq = ws * psi_ref + Rs * iq
            nr = ws / p * RPM
        tl = kq * W * abs(W)
        if dist_amp > 0:
            tl += dist_amp * (1.0 if math.sin(2 * PI * dist_f * t) >= 0 else -1.0)
        return ws, usd, usq, tl, nr

    for k in range(N + 1):
        t = k * dt
        ws, usd, usq, tl, nr = inputs(t, x[4])
        d1, Te, isd, isq = im_sync_deriv(x, ws, usd, usq, tl, mp)
        if k % every == 0:
            out["t"].append(t)
            out["Te"].append(Te)
            out["Tl"].append(tl)
            out["n"].append(x[4] * RPM)
            out["psis"].append(math.hypot(x[0], x[1]))
            out["psiR"].append(math.hypot(x[2], x[3]))
            out["ia"].append((complex(isd, isq) * cmath.exp(1j * th)).real)
            out["ws"].append(ws)
            out["nref"].append(nr)
            out["us"].append(math.hypot(usd, usq))
        h = dt
        x2 = [x[i] + h / 2 * d1[i] for i in range(5)]
        ws2, a2, b2, c2, _ = inputs(t + h / 2, x2[4])
        d2 = im_sync_deriv(x2, ws2, a2, b2, c2, mp)[0]
        x3 = [x[i] + h / 2 * d2[i] for i in range(5)]
        ws3, a3, b3, c3, _ = inputs(t + h / 2, x3[4])
        d3 = im_sync_deriv(x3, ws3, a3, b3, c3, mp)[0]
        x4 = [x[i] + h * d3[i] for i in range(5)]
        ws4, a4, b4, c4, _ = inputs(t + h, x4[4])
        d4 = im_sync_deriv(x4, ws4, a4, b4, c4, mp)[0]
        x = [x[i] + h / 6 * (d1[i] + 2 * d2[i] + 2 * d3[i] + d4[i]) for i in range(5)]
        th += h / 6 * (ws + 2 * ws2 + 2 * ws3 + ws4)
    return {k: np.array(v) for k, v in out.items()}


def vf_linear(par, n_op, psi_ref=1.83, kRR=1.0, mode="speed", T_ref=0.0,
              load_quad=False, T_nom=113.0, n_nom=1465.0, T_ext=0.0):
    """Linearyzacja napedu V/f w punkcie pracy.
    mode='speed'  : n_op = zadana predkosc synchroniczna n*_s [obr/min],
                    punkt pracy walu wynika z obciazenia.
    mode='torque' : n_op = predkosc walu, T_ref = zadany moment; obciazenie
                    zewnetrzne dobrane tak, by byla rownowaga.
    Zwraca: A, eig, x0, Te0, n0 oraz transmitancje
      G_n : delta(n*_s lub T*) -> delta n [obr/min]
      G_l : delta T_load [Nm]  -> delta n [obr/min]
      L   : petla elektromechaniczna L(s) = -G_Te,Omega(s) / (J s + B)."""
    mp, up = vf_machine_params(par, kRR)
    LsS, LM, RR, Rs, p, J = mp
    Ls = par["Ls"]
    upS = stator_model(par)
    kq = quad_load_k(T_nom, n_nom) if load_quad else 0.0

    def ctrl(u_ref, W):
        if mode == "speed":
            ws = u_ref / RPM * p
            return ws, Rs * psi_ref / Ls, ws * psi_ref
        iq = 2 * u_ref / (3 * p * psi_ref)
        ws = p * W + upS["RR"] * iq / psi_ref
        return ws, Rs * psi_ref / Ls, ws * psi_ref + Rs * iq

    def f(x, u):
        ws, usd, usq = ctrl(u[0], x[4])
        tl = kq * x[4] * abs(x[4]) + u[1]
        return np.array(im_sync_deriv(x, ws, usd, usq, tl, mp)[0])

    def te_out(x, u):
        ws, usd, usq = ctrl(u[0], x[4])
        return im_sync_deriv(x, ws, usd, usq, 0.0, mp)[1]

    if mode == "speed":
        u0 = np.array([n_op, T_ext])
        Wg = n_op / RPM
        x0 = newton(lambda x: f(x, u0), np.array([psi_ref, 0.0, psi_ref * 0.9, 0.0, Wg * 0.99]))
    else:
        Wg = n_op / RPM
        ue = np.array([T_ref, 0.0])

        def Fe(z):
            return f(np.array([z[0], z[1], z[2], z[3], Wg]), ue)[:4]
        z = newton(Fe, np.array([psi_ref, 0.0, psi_ref * 0.9, 0.0]))
        x0 = np.array([z[0], z[1], z[2], z[3], Wg])
        te = te_out(x0, ue)
        u0 = np.array([T_ref, te - kq * Wg * abs(Wg)])
    A, B = linearize(f, x0, u0)
    C = np.zeros(5)
    C[4] = RPM
    G_n = TF.from_ss(A, B[:, 0], C, 0.0, "Gn").minreal(1e-5)
    G_l = TF.from_ss(A, B[:, 1], C, 0.0, "Gl").minreal(1e-5)
    Ae = A[:4, :4]
    Bw = A[:4, 4]
    h = 1e-6
    Ct = np.zeros(4)
    for i in range(4):
        dx = np.zeros(5)
        dx[i] = h
        Ct[i] = (te_out(x0 + dx, u0) - te_out(x0 - dx, u0)) / (2 * h)
    dx = np.zeros(5)
    dx[4] = h
    Dt = (te_out(x0 + dx, u0) - te_out(x0 - dx, u0)) / (2 * h)
    G_te_w = TF.from_ss(Ae, Bw, Ct, Dt, "GTe")
    Cfull = np.concatenate([Ct, [Dt]])
    G_te = TF.from_ss(A, B[:, 0], Cfull, 0.0, "GT").minreal(1e-5)
    mech = TF([1.0], [J, 2 * kq * abs(x0[4])])
    L = (G_te_w * mech) * (-1.0)
    L = TF(L.num, L.den, 0.0, "L").minreal(1e-5)
    return dict(A=A, eig=np.linalg.eigvals(A), x0=x0, Te0=te_out(x0, u0), n0=x0[4] * RPM,
                G_n=G_n, G_l=G_l, L=L, G_te=G_te, ws0=ctrl(u0[0], x0[4])[0], up=up)


def vf_eig_sweep(par, n_list, psi_ref=1.83, kRR=1.0, T_ext=0.0):
    """Wartosci wlasne napedu V/f (sterowanie predkoscia) w funkcji n*_s."""
    mp, up = vf_machine_params(par, kRR)
    LsS, LM, RR, Rs, p, J = mp
    Ls = par["Ls"]
    res = []
    x = np.array([psi_ref, 0.0, psi_ref * 0.9, 0.0, 0.0])
    for n in n_list:
        ws = n / RPM * p
        u = (ws, Rs * psi_ref / Ls, ws * psi_ref, T_ext)

        def f(xx, uu=u):
            return np.array(im_sync_deriv(xx, uu[0], uu[1], uu[2], uu[3], mp)[0])
        xg = x.copy()
        xg[4] = n / RPM * 0.99
        x = newton(f, xg)
        A, _ = linearize(lambda xx, _u: f(xx), x, np.zeros(1))
        res.append(np.linalg.eigvals(A))
    return np.array(res)


# ---------- 9.2 - UFO: dyskretny sterownik (odwrocony model Fig. 9.10)
class UFOController:
    """Sterownik UFO (Fig. 9.10) w wersji dyskretnej (rozniczkowanie wsteczne).
    Wejscia: T*, psi_M*.  Wyjscia: isd*, isq*, w_sl*.
    Parametry estymowane: Ls, Lr, Lm, Rr (+ wspolczynniki odstrojenia)."""

    def __init__(self, par, a, Ts, kRR=1.0, kLm=1.0):
        q = dict(par)
        q["Rr"] = par["Rr"] * kRR
        q["Lm"] = par["Lm"] * kLm
        q["Ls"] = par["Ls"] - par["Lm"] + q["Lm"]
        q["Lr"] = par["Lr"] - par["Lm"] + q["Lm"]
        self.up = universal(q, a)
        self.p = par["p"]
        self.Ts = Ts
        self.isd = None
        self.isq = 0.0
        self.psi = None

    def step(self, T_ref, psi_ref):
        u = self.up
        LM, LsR, RR, LR = u["LM"], u["LsR"], u["RR"], u["LR"]
        Ts = self.Ts
        psi = max(psi_ref, 1e-3)
        isq = 2.0 * T_ref / (3.0 * self.p * psi)
        if self.isd is None:
            self.isd = psi / LM
            self.psi = psi
            self.isq = isq
        dpsi = (psi - self.psi) / Ts
        disq = (isq - self.isq) / Ts
        Nq = RR * isq + LsR * disq
        isd = self.isd
        wsl = 0.0
        for _ in range(6):
            den = psi * LR / LM - LsR * isd
            wsl = Nq / den
            # (LR/LM) dpsi = RR (isd - psi/LM) + LsR disd - wsl LsR isq
            isd = (LR / LM * dpsi + RR * psi / LM + LsR * self.isd / Ts + wsl * LsR * isq) \
                / (RR + LsR / Ts)
        self.isd, self.isq, self.psi = isd, isq, psi
        return isd, isq, wsl


def sim_foc_current(par, a, psi_ref=1.83, T_step=113.0, t_on=0.5, t_rev=1.5, tauT=0.02,
                    tau_psi=0.0, premag=True, kRR=1.0, kLm=1.0, load_frac=1.0,
                    T_end=2.5, dt=1e-4, n_out=2500):
    """Tutorial 9.7.3: sterownik UFO + CFO (IFO z enkoderem) + model maszyny
    zasilany pradem (Fig. 9.20, 9.27). Maszyna: model 'rotor flux' Tab. 8.3."""
    upm = rotor_model(par)
    LM, RR, p, J = upm["LM"], upm["RR"], par["p"], par["J"]
    dL = a * par["Lm"] - par["Lm"] ** 2 / par["Lr"]      # Delta L (rown. 9.14)
    ctrl = UFOController(par, a, dt, kRR, kLm)
    kq = quad_load_k(par.get("T_nom", 113.0), par.get("n_nom", 1465.0)) * load_frac
    psiR = complex(psi_ref * (LM / (a * par["Lm"])) if premag else 0.0, 0.0)
    if premag:
        # stan ustalony bez momentu: psi_R = LM * isd*, isd* = psi*/L_M(a)
        psiR = complex(LM * psi_ref / ctrl.up["LM"], 0.0)
    W = 0.0
    th_e = 0.0
    th_sl = 0.0
    Tf = 0.0
    psif = psi_ref if (premag or tau_psi <= 0) else 0.0
    N = int(round(T_end / dt))
    every = max(1, N // n_out)
    out = {k: [] for k in ("t", "Te", "Tref", "Tl", "n", "wsl", "isd", "isq", "psiM", "psiR",
                           "psis", "ia")}
    for k in range(N + 1):
        t = k * dt
        Tr = 0.0 if t < t_on else (T_step if t < t_rev else -T_step)
        Tf += dt / tauT * (Tr - Tf) if tauT > 0 else (Tr - Tf)
        if tau_psi > 0 and not premag:
            psif += dt / tau_psi * (psi_ref - psif)
        else:
            psif = psi_ref
        isd, isq, wsl = ctrl.step(Tf, max(psif, 0.02))
        thM = th_e + th_sl
        i_s = complex(isd, isq) * cmath.exp(1j * thM)

        def f(pr, Wx):
            dpr = RR * (i_s - pr / LM) + 1j * p * Wx * pr
            te = 1.5 * p * (pr.real * i_s.imag - pr.imag * i_s.real)
            tl = kq * Wx * abs(Wx)
            return dpr, (te - tl) / J, te, tl
        d1 = f(psiR, W)
        if k % every == 0:
            psiM = psiR + dL * i_s
            out["t"].append(t)
            out["Te"].append(d1[2])
            out["Tref"].append(Tf)
            out["Tl"].append(d1[3])
            out["n"].append(W * RPM)
            out["wsl"].append(wsl)
            out["isd"].append(isd)
            out["isq"].append(isq)
            out["psiM"].append(abs(psiM))
            out["psiR"].append(abs(psiR))
            out["psis"].append(abs(psiR + rotor_model(par)["LsS"] * i_s))
            out["ia"].append(i_s.real)
        d2 = f(psiR + dt / 2 * d1[0], W + dt / 2 * d1[1])
        d3 = f(psiR + dt / 2 * d2[0], W + dt / 2 * d2[1])
        d4 = f(psiR + dt * d3[0], W + dt * d3[1])
        psiR += dt / 6 * (d1[0] + 2 * d2[0] + 2 * d3[0] + d4[0])
        Wn = W + dt / 6 * (d1[1] + 2 * d2[1] + 2 * d3[1] + d4[1])
        th_e += p * 0.5 * (W + Wn) * dt          # enkoder: kat elektryczny walu
        W = Wn
        th_sl += wsl * dt                         # CFO: kat poslizgu
    return {k: np.array(v) for k, v in out.items()}, ctrl.up


def sim_foc_voltage(par, a, psi_ref=1.83, T_step=113.0, t_on=0.5, t_rev=1.5, tauT=0.002,
                    Ts=1e-4, Kp=None, Ki=None, Udc=1200.0, ff=True, load_frac=1.0,
                    T_end=2.5, n_out=2500, kRR=1.0):
    """Tutorial 9.7.4 / Fig. 9.22: UFO + CFO (tachometr) + regulator pradu PI w dq
    z kompensacja psi_M*w_m, model maszyny zasilany napieciem (falownik idealny,
    ograniczenie |u| <= Udc/sqrt3, opoznienie obliczeniowe 1 probka)."""
    upm = rotor_model(par)
    LsS, LM, RR, Rs, p, J = upm["LsS"], upm["LM"], upm["RR"], par["Rs"], par["p"], par["J"]
    Td = 1.5 * Ts
    Reff = Rs + RR
    if Kp is None:
        Kp = LsS / (2 * Td)
    if Ki is None:
        Ki = Reff / (2 * Td)
    umax = Udc / SQ3
    ctrl = UFOController(par, a, Ts, kRR)
    kq = quad_load_k(par.get("T_nom", 113.0), par.get("n_nom", 1465.0)) * load_frac
    # start: maszyna namagnesowana (stan ustalony T=0)
    isd0 = psi_ref / ctrl.up["LM"]
    psiR = complex(LM * isd0, 0.0)
    psis = psiR + LsS * isd0
    W = 0.0
    thM = 0.0
    u_apply = complex(Rs * isd0, 0.0)
    u_next = u_apply
    Id = Rs * isd0
    Iq = 0.0
    Tf = 0.0
    N = int(round(T_end / Ts))
    every = max(1, N // n_out)
    out = {k: [] for k in ("t", "Te", "Tref", "Tl", "n", "wsl", "isd_ref", "isq_ref", "isd",
                           "isq", "psis", "psiR", "u")}
    sub = 2
    h = Ts / sub
    for k in range(N + 1):
        t = k * Ts
        Tr = 0.0 if t < t_on else (T_step if t < t_rev else -T_step)
        Tf += Ts / tauT * (Tr - Tf) if tauT > 0 else (Tr - Tf)
        isd_r, isq_r, wsl = ctrl.step(Tf, psi_ref)
        # pomiar pradu w ukladzie sterownika
        i_s = (psis - psiR) / LsS
        idq = i_s * cmath.exp(-1j * thM)
        ed = isd_r - idq.real
        eq = isq_r - idq.imag
        wm = p * W
        Id += Ki * ed * Ts
        Iq += Ki * eq * Ts
        ud = Kp * ed + Id
        uq = Kp * eq + Iq + (psi_ref * wm if ff else 0.0)
        u = complex(ud, uq)
        if abs(u) > umax:
            u = u / abs(u) * umax
            Id -= Ki * ed * Ts            # anti-windup (zatrzymanie calkowania)
            Iq -= Ki * eq * Ts
        if k % every == 0:
            out["t"].append(t)
            out["Te"].append(1.5 * p * (psis.real * i_s.imag - psis.imag * i_s.real))
            out["Tref"].append(Tf)
            out["Tl"].append(kq * W * abs(W))
            out["n"].append(W * RPM)
            out["wsl"].append(wsl)
            out["isd_ref"].append(isd_r)
            out["isq_ref"].append(isq_r)
            out["isd"].append(idq.real)
            out["isq"].append(idq.imag)
            out["psis"].append(abs(psis))
            out["psiR"].append(abs(psiR))
            out["u"].append(abs(u_apply))
        # napiecie podane na maszyne (opoznienie 1 probki), wektor w ukladzie alfa-beta
        u_ab = u_apply * cmath.exp(1j * (thM + 0.5 * (wm + wsl) * Ts))

        def f(ps, pr, Wx):
            isx = (ps - pr) / LsS
            dps = u_ab - Rs * isx
            dpr = RR * (isx - pr / LM) + 1j * p * Wx * pr
            te = 1.5 * p * (ps.real * isx.imag - ps.imag * isx.real)
            return dps, dpr, (te - kq * Wx * abs(Wx)) / J
        for _ in range(sub):
            d1 = f(psis, psiR, W)
            d2 = f(psis + h / 2 * d1[0], psiR + h / 2 * d1[1], W + h / 2 * d1[2])
            d3 = f(psis + h / 2 * d2[0], psiR + h / 2 * d2[1], W + h / 2 * d2[2])
            d4 = f(psis + h * d3[0], psiR + h * d3[1], W + h * d3[2])
            psis += h / 6 * (d1[0] + 2 * d2[0] + 2 * d3[0] + d4[0])
            psiR += h / 6 * (d1[1] + 2 * d2[1] + 2 * d3[1] + d4[1])
            W += h / 6 * (d1[2] + 2 * d2[2] + 2 * d3[2] + d4[2])
        thM += (p * W + wsl) * Ts                 # CFO z tachometrem: integral(wm + wsl*)
        u_apply = u_next
        u_next = u
    return {k: np.array(v) for k, v in out.items()}, dict(Kp=Kp, Ki=Ki, Td=Td, Reff=Reff,
                                                           LsS=LsS, umax=umax)


# ---------- 9.2.1.2 - odstrojenie (detuning) IFO
def detuning_curves(par, k_list, psi_ref=None, T_ref=113.0):
    """Stan ustalony IFO zorientowanego na strumien wirnika, gdy estymata
    RR_est = k * RR. Sterownik zadaje |is| i w_sl* = RR_est isq*/psi*.
    Zwraca psi_R/psi*, Te/T*, blad kata dtheta [st]."""
    up = rotor_model(par)
    LM, RR, p = up["LM"], up["RR"], par["p"]
    if psi_ref is None:
        psi_ref = 1.593
    isd = psi_ref / LM
    isq = 2 * T_ref / (3 * p * psi_ref)
    tauR = LM / RR
    res = dict(k=np.asarray(k_list, float), psi=[], Te=[], dth=[])
    for k in res["k"]:
        wsl = k * RR * isq / psi_ref
        i_s = complex(isd, isq)
        psiR = LM * i_s / (1 + 1j * wsl * tauR)          # w ukladzie zwiazanym z pradem sterownika
        Te = 1.5 * p * (psiR.real * i_s.imag - psiR.imag * i_s.real)
        res["psi"].append(abs(psiR) / psi_ref)
        res["Te"].append(Te / T_ref)
        res["dth"].append(math.degrees(cmath.phase(psiR)))
    for key in ("psi", "Te", "dth"):
        res[key] = np.array(res[key])
    return res, dict(isd=isd, isq=isq, tauR=tauR)


# ---------- 9.2.7 / 9.2.8 - bledy obserwatorow strumienia
def observer_errors(par, n_rpm, kRs=1.2, kRR=1.2, psi_ref=1.593, T_ref=113.0):
    """Stan ustalony: blad estymaty strumienia wirnika (modul i kat)
    - model napieciowy (9.2.7): uzywa Rs_est = kRs*Rs, calkuje u - Rs i,
    - model pradowy (9.2.8): uzywa RR_est = kRR*RR i predkosci walu."""
    up = rotor_model(par)
    LsS, LM, RR, Rs, p = up["LsS"], up["LM"], up["RR"], par["Rs"], par["p"]
    isd = psi_ref / LM
    isq = 2 * T_ref / (3 * p * psi_ref)
    wsl = RR * isq / psi_ref
    i_s = complex(isd, isq)
    psiR = complex(psi_ref, 0)
    psis = psiR + LsS * i_s
    ev, ec = [], []
    av, ac = [], []
    for n in np.asarray(n_rpm, float):
        ws = n / RPM * p + wsl
        u = Rs * i_s + 1j * ws * psis
        psis_e = (u - kRs * Rs * i_s) / (1j * ws)
        psiR_v = psis_e - LsS * i_s
        RRe = kRR * RR
        psiR_c = LM * i_s / (1 + 1j * wsl * LM / RRe)
        ev.append(abs(psiR_v) / psi_ref - 1)
        av.append(math.degrees(cmath.phase(psiR_v)))
        ec.append(abs(psiR_c) / psi_ref - 1)
        ac.append(math.degrees(cmath.phase(psiR_c)))
    return dict(n=np.asarray(n_rpm, float), ev=np.array(ev) * 100, av=np.array(av),
                ec=np.array(ec) * 100, ac=np.array(ac))


# ---------- 9.3 / 9.4 - granice pracy i oslabianie pola
def op_limits(par, u_max, i_max, psiR_max=1.593, chi_override=None):
    up = rotor_model(par)
    Ls, LsS, LM, p = par["Ls"], up["LsS"], up["LM"], par["p"]
    chi = LsS / Ls if chi_override is None else chi_override
    isd_max = psiR_max / LM
    isq_D = math.sqrt(max(i_max ** 2 - isd_max ** 2, 0.0))
    ws_b = u_max / (Ls * math.sqrt(isd_max ** 2 + chi ** 2 * isq_D ** 2))
    kT = 1.5 * p * LM
    return dict(chi=chi, isd_max=isd_max, isq_D=isq_D, ws_b=ws_b, kT=kT, Ls=Ls, LM=LM,
                Tmax_b=kT * isd_max * isq_D, u_max=u_max, i_max=i_max, LsS=LsS)


def fw_currents(L, ws, T_cmd):
    """Prady isd, isq dla strategii z rozdz. 9.4 (staly moment, potem MTPF)."""
    chi, Ls, kT = L["chi"], L["Ls"], L["kT"]
    imax, isd_max = L["i_max"], L["isd_max"]
    r = L["u_max"] / (ws * Ls)                     # polos elipsy na osi d
    region = "baza"
    isd = isd_max
    if ws > L["ws_b"]:
        val = (r * r - chi * chi * imax * imax) / (1 - chi * chi)
        isd = math.sqrt(val) if val > 0 else 0.0
        isd = min(isd, isd_max)
        region = "osłabianie pola"
    isd_mtpf = r / SQ2
    if isd < isd_mtpf:
        isd = min(isd_mtpf, isd_max)
        region = "MTPF"
    isq_need = T_cmd / (kT * isd) if isd > 0 else 0.0
    isq_ma = math.sqrt(max(imax * imax - isd * isd, 0.0))
    isq_mf = math.sqrt(max(r * r - isd * isd, 0.0)) / chi
    isq = min(isq_need, isq_ma, isq_mf)
    if isq < isq_need - 1e-9 and region == "baza":
        region = "ograniczenie prądu"
    return isd, isq, kT * isd * isq, region


# ---------- analiza LTI petli pradowej i toru momentu
def current_loop_tf(LsS, R, Kp, Ki, Td):
    """L(s) = (Kp + Ki/s) * e^{-s Td} / (LsS s + R)."""
    C = TF([Kp, Ki], [1.0, 0.0], 0.0, "C")
    G = TF([1.0], [LsS, R], 0.0, "G")
    L = C * G
    return TF(L.num, L.den, Td, "L")


if __name__ == "__main__":
    st, pr = estimate_parameters()
    for s in st:
        print(f"{s[1]:>14s} = {s[2]:.5g} {s[3]}   ({s[0]})")
