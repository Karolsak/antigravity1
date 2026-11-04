"""
pe_control.py - minimalny "Control Toolbox" w czystym numpy.

Funkcje odpowiadają poleceniom z MATLAB/python-control:
  tf (klasa TF), zpk (TF.zpk_text), bode (freq_resp), margin (margins),
  nyquist (freq_resp na płaszczyźnie zespolonej), step (step_response),
  feedback (closed_loop), pade (pade).

Opóźnienie e^{-sTd}:
  * w Bode i Nyquiście liczone DOKŁADNIE (mnożymy przez e^{-jωTd}),
  * w odpowiedzi skokowej i mapie biegunów - aproksymacją Padé 2. rzędu
    (bo e^{-sTd} nie jest funkcją wymierną i nie ma skończonej liczby biegunów).
"""
import math
import numpy as np


# =============================================================================
# Transmitancja
# =============================================================================

class TF:
    """Transmitancja G(s) = num(s)/den(s) * exp(-s*delay); wielomiany jako
    tablice współczynników od najwyższej potęgi (konwencja numpy.polyval)."""

    def __init__(self, num, den, delay=0.0):
        self.num = np.trim_zeros(np.atleast_1d(np.asarray(num, dtype=float)), "f")
        self.den = np.trim_zeros(np.atleast_1d(np.asarray(den, dtype=float)), "f")
        if self.num.size == 0:
            self.num = np.array([0.0])
        self.delay = float(delay)

    def __mul__(self, o):
        if np.isscalar(o):
            return TF(self.num * o, self.den, self.delay)
        return TF(np.polymul(self.num, o.num), np.polymul(self.den, o.den),
                  self.delay + o.delay)

    __rmul__ = __mul__

    def __add__(self, o):
        if np.isscalar(o):
            o = TF([o], [1.0])
        if abs(self.delay - o.delay) > 1e-15:
            raise ValueError("suma transmitancji z różnymi opóźnieniami")
        return TF(np.polyadd(np.polymul(self.num, o.den), np.polymul(o.num, self.den)),
                  np.polymul(self.den, o.den), self.delay)

    def __call__(self, s):
        s = np.asarray(s)
        return np.polyval(self.num, s) / np.polyval(self.den, s) * np.exp(-s * self.delay)

    def rational(self, pade_order=2):
        """Wersja wymierna (opóźnienie -> Padé)."""
        if self.delay <= 0:
            return TF(self.num, self.den)
        pn, pd = pade(self.delay, pade_order)
        return TF(np.polymul(self.num, pn), np.polymul(self.den, pd))

    def zeros(self):
        return np.roots(self.num) if self.num.size > 1 else np.array([])

    def poles(self):
        return np.roots(self.den) if self.den.size > 1 else np.array([])

    def gain_k(self):
        return self.num[0] / self.den[0]

    def dc_gain(self):
        d = np.polyval(self.den, 0.0)
        n = np.polyval(self.num, 0.0)
        if abs(d) < 1e-300:
            return math.inf
        return n / d

    def zpk_text(self, name="G", unit_s=True):
        """Postać zero-biegun-wzmocnienie:  G(s) = k (s-z1)(s-z2).../((s-p1)(s-p2)...)."""
        z = self.zeros()
        p = self.poles()
        k = self.gain_k()

        def fac(r):
            if abs(r.imag) < 1e-9 * max(1, abs(r.real)):
                v = r.real
                if abs(v) < 1e-12:
                    return "s"
                return f"(s{-v:+.4g})"
            return f"(s{-r.real:+.4g}{-r.imag:+.4g}j)"

        def group(rs):
            if len(rs) == 0:
                return "1"
            rs = sorted(rs, key=lambda r: (abs(r), r.imag))
            out, used = [], [False] * len(rs)
            for i_, r in enumerate(rs):
                if used[i_]:
                    continue
                n = 0
                for j_ in range(i_, len(rs)):
                    if not used[j_] and abs(rs[j_] - r) <= 1e-6 * max(1.0, abs(r)):
                        used[j_] = True
                        n += 1
                f_ = fac(r)
                out.append(f_ if n == 1 else f"{f_}^{n}")
            return "·".join(out)

        txt = f"{name}(s) = {k:.5g} · {group(z)} / {group(p)}"
        if self.delay > 0:
            txt += f" · e^(-s·{self.delay * 1e6:.4g}µs)"
        return txt


def pade(T, n=2):
    """Aproksymacja Padé e^{-sT} ≈ num(s)/den(s)."""
    if n == 1:
        return np.array([-T / 2, 1.0]), np.array([T / 2, 1.0])
    if n == 2:
        return (np.array([T * T / 12, -T / 2, 1.0]),
                np.array([T * T / 12, T / 2, 1.0]))
    # rząd 3
    return (np.array([-T ** 3 / 120, T * T / 10, -T / 2, 1.0]),
            np.array([T ** 3 / 120, T * T / 10, T / 2, 1.0]))


def pi_tf(Kp, Ki):
    """Regulator PI: C(s) = Kp + Ki/s = (Kp s + Ki)/s."""
    return TF([Kp, Ki], [1.0, 0.0])


def pr_tf(Kp, Kr, w0, wc):
    """Regulator PR (proporcjonalno-rezonansowy):
    C(s) = Kp + Kr · 2 ωc s / (s² + 2 ωc s + ω0²)."""
    res = TF([2 * wc * Kr, 0.0], [1.0, 2 * wc, w0 * w0])
    return res + Kp


def first_order(K, tau):
    return TF([K], [tau, 1.0])


def closed_loop(Lnum_den_tf, C, P):
    """Pętla z ujemnym sprzężeniem zwrotnym (wersja wymierna):
    T(s)  = CP/(1+CP)  - śledzenie wartości zadanej,
    Sd(s) = P/(1+CP)   - wpływ zakłócenia na wejściu obiektu."""
    Cr, Pr = C.rational(), P.rational()
    nC, dC, nP, dP = Cr.num, Cr.den, Pr.num, Pr.den
    den = np.polyadd(np.polymul(dC, dP), np.polymul(nC, nP))
    T = TF(np.polymul(nC, nP), den)
    Sd = TF(np.polymul(nP, dC), den)
    return T, Sd


# =============================================================================
# Charakterystyki częstotliwościowe i zapasy stabilności
# =============================================================================

def freq_resp(G, w):
    """G(jω) - z dokładnym opóźnieniem."""
    return G(1j * w)


def margins(L, w):
    """Zapas fazy PM [°] przy ωc (|L|=1) i zapas wzmocnienia GM [dB] przy ω180
    (faza = -180° - 360°·n). Gdy przecięć jest kilka (np. regulator PR),
    zwracany jest NAJGORSZY (najmniejszy) zapas - podejście konserwatywne.
    Wartości None oznaczają brak przecięcia w badanym zakresie ω."""
    H = freq_resp(L, w)
    mag = np.abs(H)
    ph = np.unwrap(np.angle(H)) * 180 / math.pi
    ph = ph - 360.0 * round(ph[0] / 360.0)        # faza początkowa w [-180, 180]
    res = dict(wc=None, pm=None, w180=None, gm=None, mag=mag, ph=ph,
               all_wc=[], all_w180=[])
    lm = np.log10(mag + 1e-300)
    lw = np.log10(w)
    idx = np.where(np.sign(lm[:-1]) != np.sign(lm[1:]))[0]
    for k in idx:
        x = lm[k] / (lm[k] - lm[k + 1])
        wc = 10 ** (lw[k] + x * (lw[k + 1] - lw[k]))
        phc = ph[k] + x * (ph[k + 1] - ph[k])
        # PM = odległość kątowa od -180°; dla fazy dodatniej (np. zbocze
        # rezonansu PR) odległość liczymy od drugiej strony okręgu
        pm = 180.0 - phc if phc > 0 else 180.0 + phc
        res["all_wc"].append((wc, pm))
        if res["pm"] is None or pm < res["pm"]:
            res["wc"], res["pm"] = wc, pm
    sh = np.floor((ph + 180.0) / 360.0)
    cr = np.where(sh[:-1] != sh[1:])[0]
    for k in cr:
        lvl = max(sh[k], sh[k + 1]) * 360.0 - 180.0
        if ph[k + 1] == ph[k]:
            continue
        x = (lvl - ph[k]) / (ph[k + 1] - ph[k])
        if not (0 <= x <= 1):
            continue
        w180 = 10 ** (lw[k] + x * (lw[k + 1] - lw[k]))
        gm = -20 * (lm[k] + x * (lm[k + 1] - lm[k]))
        res["all_w180"].append((w180, gm))
        if res["gm"] is None or gm < res["gm"]:
            res["w180"], res["gm"] = w180, gm
    return res


def settle_horizon(T, wc=None, lo=1e-5, hi=5.0):
    """Rozsądny horyzont czasu odpowiedzi skokowej: ~8 stałych czasowych
    najwolniejszego bieguna, który NIE jest skrócony przez zero."""
    p = T.poles()
    z = T.zeros()
    slow = None
    for pk in p:
        if z.size and np.min(np.abs(z - pk)) < 0.02 * max(abs(pk), 1e-9):
            continue
        if pk.real >= 0:
            continue
        a = -pk.real
        slow = a if slow is None else min(slow, a)
    cand = []
    if slow:
        cand.append(8.0 / slow)
    if wc:
        cand.append(12.0 / wc)
    if not cand:
        return 1e-2
    return float(np.clip(max(cand) if len(cand) == 1 else min(max(cand), 4 * min(cand)), lo, hi))


# =============================================================================
# Odpowiedź skokowa - model w przestrzeni stanów + wykładnik macierzy
# =============================================================================

def expm(A):
    """exp(A) metodą skalowania i podnoszenia do kwadratu z aproksymacją Padé (6,6)."""
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    nrm = np.linalg.norm(A, 1)
    s = max(0, int(math.ceil(math.log2(nrm))) + 1) if nrm > 0.5 else 0
    As = A / (2 ** s)
    c = [1.0, 0.5, 5 / 44, 1 / 66, 1 / 792, 1 / 15840, 1 / 665280]
    X = np.eye(n)
    Nm = np.eye(n) * c[0]
    Dm = np.eye(n) * c[0]
    for k in range(1, 7):
        X = X @ As
        Nm = Nm + c[k] * X
        Dm = Dm + ((-1) ** k) * c[k] * X
    E = np.linalg.solve(Dm, Nm)
    for _ in range(s):
        E = E @ E
    return E


def tf2ss(G):
    """Postać sterowalna (kanoniczna) transmitancji właściwej."""
    num = np.asarray(G.num, dtype=float)
    den = np.asarray(G.den, dtype=float)
    a0 = den[0]
    den = den / a0
    num = num / a0
    n = den.size - 1
    if n == 0:
        return np.zeros((0, 0)), np.zeros((0, 1)), np.zeros((1, 0)), num[-1]
    num = np.concatenate([np.zeros(n + 1 - num.size), num])
    D = num[0]
    b = num[1:] - D * den[1:]
    A = np.zeros((n, n))
    A[0, :] = -den[1:]
    A[1:, :-1] = np.eye(n - 1)
    B = np.zeros((n, 1))
    B[0, 0] = 1.0
    Cm = b.reshape(1, n)
    return A, B, Cm, D


def step_response(G, t_end, n=1500):
    """Odpowiedź na skok jednostkowy y(t) układu G (wymiernego)."""
    A, B, Cm, D = tf2ss(G)
    t = np.linspace(0, t_end, n)
    if A.shape[0] == 0:
        return t, np.full(n, D)
    dt = t[1] - t[0]
    nst = A.shape[0]
    M = np.zeros((nst + 1, nst + 1))
    M[:nst, :nst] = A * dt
    M[:nst, nst:] = B * dt
    E = expm(M)
    Ad, Bd = E[:nst, :nst], E[:nst, nst:]
    x = np.zeros((nst, 1))
    y = np.empty(n)
    for k in range(n):
        y[k] = float(Cm @ x) + D
        x = Ad @ x + Bd
        if not np.all(np.isfinite(x)):
            y[k + 1:] = np.nan
            break
    return t, y


def step_info(t, y, yfinal=None):
    """Czas narastania 10-90 %, przeregulowanie, czas ustalania 2 %."""
    if not np.all(np.isfinite(y)):
        return dict(tr=None, os=None, ts=None, yss=None)
    yss = y[-1] if yfinal is None else yfinal
    if abs(yss) < 1e-12:
        return dict(tr=None, os=None, ts=None, yss=yss)
    yn = y / yss
    try:
        t10 = t[np.where(yn >= 0.1)[0][0]]
        t90 = t[np.where(yn >= 0.9)[0][0]]
        tr = t90 - t10
    except IndexError:
        tr = None
    os_ = max(0.0, (np.max(yn) - 1) * 100)
    out = np.where(np.abs(yn - 1) > 0.02)[0]
    ts = t[out[-1] + 1] if out.size and out[-1] + 1 < len(t) else (t[0] if not out.size else None)
    return dict(tr=tr, os=os_, ts=ts, yss=yss)


def is_stable(G):
    p = G.poles()
    return bool(np.all(p.real < 0)) if p.size else True


def damping_of(p):
    """Współczynnik tłumienia ζ i pulsacja własna ωn bieguna."""
    wn = abs(p)
    zeta = -p.real / wn if wn > 0 else 1.0
    return zeta, wn


def interpret_margins(pm, gm, wc, fsw=None):
    """Słowna interpretacja zapasów stabilności (dla ucznia)."""
    lines = []
    if pm is None:
        lines.append("Brak przecięcia |L|=1 - wzmocnienie pętli zawsze <1 (lub >1).")
    elif pm <= 0:
        lines.append(f"PM = {pm:.1f}° ≤ 0 -> UKŁAD NIESTABILNY (drgania narastają)!")
    elif pm < 30:
        lines.append(f"PM = {pm:.1f}° - mały zapas: silne oscylacje, duże przeregulowanie.")
    elif pm < 45:
        lines.append(f"PM = {pm:.1f}° - akceptowalny, ale wyraźne przeregulowanie (~20-35 %).")
    elif pm < 65:
        lines.append(f"PM = {pm:.1f}° - DOBRY kompromis szybkość/tłumienie (przeregulowanie zwykle < 20 %).")
    else:
        lines.append(f"PM = {pm:.1f}° - bardzo bezpiecznie, odpowiedź gładka (może być wolna).")
    if gm is None:
        lines.append("GM = ∞ - faza pętli nigdy nie przekracza -180° (brak opóźnienia), "
                     "więc samo zwiększanie wzmocnienia nie zdestabilizuje układu (zmaleje tylko PM).")
    else:
        if gm <= 0:
            lines.append(f"GM = {gm:.1f} dB ≤ 0 -> niestabilny.")
        elif gm < 6:
            lines.append(f"GM = {gm:.1f} dB < 6 dB - mały zapas wzmocnienia (wrażliwy na zmiany L, Vdc).")
        else:
            lines.append(f"GM = {gm:.1f} dB - wzmocnienie można zwiększyć ~{10 ** (gm / 20):.1f}× zanim układ straci stabilność.")
    if wc and fsw:
        fc = wc / (2 * math.pi)
        ratio = fsw / fc
        lines.append(f"fc = {fc:.0f} Hz = fsw/{ratio:.1f} " +
                     ("(OK: zalecane fc ≤ fsw/10)" if ratio >= 10 else
                      "(UWAGA: pasmo zbyt blisko częstotliwości łączeń)"))
    return lines
