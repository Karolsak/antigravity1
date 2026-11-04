"""
pe_core.py - rdzeń obliczeniowy (tylko numpy) dla aplikacji
"Strategie przełączania przekształtników energoelektronicznych".

Materiał źródłowy: S. V. Iyer, M. N. Aalam, "Switching Strategies for Power
Electronic Converters", rozdz. 4 (mostek H), 5 (przekształtniki trójfazowe,
Clarke, SVPWM), 6 (przekształtniki wielopoziomowe NPC).

Konwencje:
  * czas w sekundach, napięcia w V, prądy w A, indukcyjność w H,
  * nośna trójkątna  tri(t) w przedziale -1..+1, piłokształtna saw(t) 0..1,
  * transformacja Clarke'a w wersji "amplitudowo niezmienniczej" (współczynnik 2/3),
    tak jak w książce: wektory przekształtnika mają długość 2/3*Vdc.
"""
import math
import numpy as np

PI = math.pi
SQ3 = math.sqrt(3.0)
A120 = np.exp(1j * 2 * PI / 3)          # operator obrotu o 120°


# =============================================================================
# Nośne i narzędzia sygnałowe
# =============================================================================

def tri(t, f):
    """Nośna trójkątna -1..+1 o częstotliwości f (minimum w t=0)."""
    x = np.mod(t * f, 1.0)
    return 1.0 - 4.0 * np.abs(x - 0.5)


def saw(t, f, shift=0.0):
    """Nośna piłokształtna 0..1 (narasta liniowo, opada skokiem); shift w okresach."""
    return np.mod(t * f - shift, 1.0)


def rl_response(v, dt, R, L, i0=0.0):
    """Prąd obwodu R-L zasilanego napięciem v[k] (dokładna dyskretyzacja:
    napięcie stałe w każdym kroku):  i[k+1] = a·i[k] + b·v[k],
    a = exp(-R dt / L),  b = (1-a)/R.  v może być macierzą (n_faz, N).
    Rekurencja liczona blokowo (wektorowo) - szybko nawet dla 10^5 próbek."""
    v = np.atleast_2d(np.asarray(v, dtype=float))
    a = math.exp(-R * dt / L)
    b = (1.0 - a) / R if R > 1e-12 else dt / L
    nph, N = v.shape
    i = np.empty_like(v)
    x = np.full(nph, float(i0)) if np.isscalar(i0) else np.array(i0, dtype=float)
    # rozmiar bloku tak, by a^-B nie przepełniło zakresu liczb
    B = int(max(1, min(512, 20.0 / max(R * dt / L, 1e-12))))
    for s in range(0, N, B):
        vb = v[:, s:s + B]
        n = vb.shape[1]
        j = np.arange(n)
        apow = a ** j                                  # a^j
        # i[s+j] = a^j * ( x + b * sum_{m<j} v[m] a^-(m+1) )
        w = vb * (a ** -(j + 1.0))
        cs = np.concatenate([np.zeros((nph, 1)), np.cumsum(w, axis=1)[:, :-1]], axis=1)
        i[:, s:s + n] = apow * (x[:, None] + b * cs)
        x = a ** n * (x + b * np.sum(w, axis=1))
    return i if nph > 1 else i[0]


def spectrum(x, n_periods):
    """Widmo jednostronne sygnału zawierającego dokładnie n_periods okresów
    podstawowych. Zwraca (rząd harmonicznej h, amplituda)."""
    N = len(x)
    X = np.fft.rfft(x) / N
    amp = np.abs(X) * 2
    amp[0] /= 2
    h = np.arange(len(X)) / n_periods
    return h, amp


def thd_from_spectrum(h, amp, hmax=None):
    """THD [%] liczony z widma (harmoniczne całkowite 2..hmax)."""
    idx1 = np.argmin(np.abs(h - 1))
    a1 = amp[idx1]
    if a1 < 1e-12:
        return 0.0
    mask = (h > 1.5) & (np.abs(h - np.round(h)) < 1e-9)
    if hmax:
        mask &= h <= hmax
    return 100.0 * math.sqrt(np.sum(amp[mask] ** 2)) / a1


def fundamental(x, n_periods):
    """Amplituda składowej podstawowej."""
    h, a = spectrum(x, n_periods)
    return a[np.argmin(np.abs(h - 1))]


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x) ** 2)))


# =============================================================================
# ROZDZIAŁ 4 - mostek H (full-bridge)
# =============================================================================

def hbridge_state(S1, S2, S3, S4, i_dir, Vdc=1.0):
    """Analiza stanu przewodzenia mostka H (gałąź a: S1 góra / S2 dół,
    gałąź b: S3 góra / S4 dół). i_dir = +1: prąd płynie od zacisku a do b
    przez obciążenie; -1 odwrotnie.
    Zwraca dict: forbidden, vaN, vbN, vo, elementy przewodzące, opis."""
    out = {"forbidden": False, "msg": "", "path": []}
    if (S1 and S2) or (S3 and S4):
        leg = "a (S1+S2)" if (S1 and S2) else "b (S3+S4)"
        out.update(forbidden=True, vo=None, vaN=None, vbN=None,
                   msg=f"STAN ZABRONIONY: zwarcie szyny DC w gałęzi {leg}!")
        return out
    # gałąź a: prąd WYPŁYWA z zacisku a, gdy i_dir>0
    if i_dir > 0:
        if S1:
            a_el, vaN = "S1", Vdc
        else:
            a_el, vaN = "D2", 0.0        # prąd wypływa: dolna dioda przewodzi
        if S4:
            b_el, vbN = "S4", 0.0        # prąd wpływa do zacisku b
        else:
            b_el, vbN = "D3", Vdc
    else:
        if S2:
            a_el, vaN = "S2", 0.0        # prąd wpływa do zacisku a
        else:
            a_el, vaN = "D1", Vdc
        if S3:
            b_el, vbN = "S3", Vdc        # prąd wypływa z zacisku b
        else:
            b_el, vbN = "D4", 0.0
    vo = vaN - vbN
    out.update(vaN=vaN, vbN=vbN, vo=vo, path=[a_el, b_el])
    if abs(vo) < 1e-12:
        role = "napięcie 0 - prąd krąży wewnątrz mostka (wektor zerowy, 'wolne koło')"
    elif vo * i_dir > 0:
        role = "źródło DC NAPĘDZA prąd (energia: DC -> obciążenie)"
    else:
        role = "źródło DC HAMUJE prąd (energia wraca do źródła DC)"
    out["msg"] = f"Przewodzą: {a_el} i {b_el};  vo = {vo:+.3g}·Vdc/Vdc;  {role}"
    return out


def hbridge_bipolar(Vdc, R, L, f1, fsw, ma, n_cyc=3, ppc=20000):
    """Mostek H, bipolarne sinusoidalne PWM (S1,S4 razem; S2,S3 razem).
    Zwraca t, m, c, vo, iL, idc dla ostatniego okresu (stan ustalony)."""
    N = n_cyc * ppc
    t = np.arange(N) / (ppc * f1)
    dt = t[1]
    m = ma * np.sin(2 * PI * f1 * t)
    c = tri(t, fsw)
    s = m > c
    vo = np.where(s, Vdc, -Vdc)
    iL = rl_response(vo, dt, R, L)
    idc = np.where(s, iL, -iL)
    sl = slice(N - ppc, N)
    return dict(t=t[sl] - t[N - ppc], m=m[sl], c=c[sl], vo=vo[sl], iL=iL[sl],
                idc=idc[sl], dt=dt, S1=s[sl])


def hbridge_unipolar_ac(Vdc, R, L, f1, fsw, ma, n_cyc=3, ppc=20000):
    """Mostek H, unipolarne PWM dla dc-ac wg książki:
    c1 = trójkąt 0..1, c2 = -c1.  S1 ON gdy m > c2,  S4 ON gdy m > c1 (S3 gdy m < c1)."""
    N = n_cyc * ppc
    t = np.arange(N) / (ppc * f1)
    dt = t[1]
    m = ma * np.sin(2 * PI * f1 * t)
    c1 = 0.5 * (tri(t, fsw) + 1.0)
    c2 = -c1
    S1 = m > c2
    S3 = m < c1
    vaN = np.where(S1, Vdc, 0.0)
    vbN = np.where(S3, Vdc, 0.0)
    vo = vaN - vbN
    iL = rl_response(vo, dt, R, L)
    idc = iL * (S1.astype(float) - S3.astype(float))
    sl = slice(N - ppc, N)
    return dict(t=t[sl] - t[N - ppc], m=m[sl], c1=c1[sl], c2=c2[sl], vo=vo[sl],
                iL=iL[sl], idc=idc[sl], S1=S1[sl], S3=S3[sl], dt=dt)


def hbridge_dcdc_unipolar(Vdc, d, fsw, n_sw=6, pps=400):
    """Sygnały mostka H w trybie dc-dc (unipolarne PWM, 4 ćwiartki):
    c1 = piła 0..1, c2 = -c1;  S1 ON gdy d > c2;  S3 ON gdy d < c1.
    Zwraca t, c1, c2, S1, S3, vaN, vbN, vo."""
    N = n_sw * pps
    t = np.arange(N) / (pps * fsw)
    c1 = saw(t, fsw)
    c2 = -c1
    S1 = d > c2
    S3 = d < c1
    vaN = np.where(S1, Vdc, 0.0)
    vbN = np.where(S3, Vdc, 0.0)
    return dict(t=t, c1=c1, c2=c2, S1=S1, S3=S3, vaN=vaN, vbN=vbN, vo=vaN - vbN)


def single_dcdc(Vdc, d, fsw, R, L, E=0.0, n_settle=None, n_show=5, pps=400):
    """Pojedynczy mostek H dc-dc zasilający obciążenie R-L(-E)."""
    tau = L / max(R, 1e-6)
    if n_settle is None:
        n_settle = int(min(4000, max(10, 6 * tau * fsw)))
    n = n_settle + n_show
    sig = hbridge_dcdc_unipolar(Vdc, d, fsw, n_sw=n, pps=pps)
    dt = sig["t"][1]
    iL = rl_response(sig["vo"] - E, dt, R, L)
    sl = slice(n_settle * pps, n * pps)
    out = {k: (v[sl] if isinstance(v, np.ndarray) else v) for k, v in sig.items()}
    out["t"] = out["t"] - out["t"][0]
    out["iL"] = iL[sl]
    return out


def dual_dcdc(Vdc1, Vdc2, d1, d2, fsw, R, L, shift=0.0, n_show=5, pps=400):
    """Dwa mostki H (LC i RC) połączone przez dławik L i rezystancję R.
    L di/dt = v_brg1 - v_brg2 - R i  (prąd dodatni płynie z LC do RC).
    shift - przesunięcie nośnej RC względem LC (w ułamku okresu)."""
    tau = L / max(R, 1e-6)
    n_settle = int(min(4000, max(10, 6 * tau * fsw)))
    n = n_settle + n_show
    N = n * pps
    t = np.arange(N) / (pps * fsw)
    dt = t[1]
    c1 = saw(t, fsw)
    c1r = saw(t, fsw, shift)
    v1 = Vdc1 * ((d1 > -c1).astype(float) - (d1 < c1).astype(float))
    v2 = Vdc2 * ((d2 > -c1r).astype(float) - (d2 < c1r).astype(float))
    i = rl_response(v1 - v2, dt, R, L)
    sl = slice(n_settle * pps, N)
    S1 = (d1 > -c1)[sl]
    S3 = (d1 < c1)[sl]
    tt = t[sl] - t[n_settle * pps]
    return dict(t=tt, c1=c1[sl], c2=-c1[sl], v1=v1[sl], v2=v2[sl], i=i[sl],
                S1=S1, S3=S3, vL=(v1 - v2)[sl] - R * i[sl])


def phase_shift_pwm(Vdc, d, fsw, R, L, n_show=4, pps=800):
    """Mostek H z modulacją z przesunięciem fazowym (phase-shift):
    każda gałąź 50 %, gałąź a przesunięta o d okresu względem b.
    c1 = piła(t), c2 = -piła(t - d·T);  S1 ON gdy -0.5 > c2;  S3 ON gdy 0.5 < c1."""
    tau = L / max(R, 1e-6)
    n_settle = int(min(4000, max(10, 6 * tau * fsw)))
    n = n_settle + n_show
    N = n * pps
    t = np.arange(N) / (pps * fsw)
    dt = t[1]
    c1 = saw(t, fsw)
    c2 = -saw(t, fsw, d)
    S1 = -0.5 > c2
    S3 = 0.5 < c1
    vo = Vdc * (S1.astype(float) - S3.astype(float))
    iL = rl_response(vo, dt, R, L)
    sl = slice(n_settle * pps, N)
    return dict(t=t[sl] - t[n_settle * pps], c1=c1[sl], c2=c2[sl], S1=S1[sl],
                S3=S3[sl], vo=vo[sl], iL=iL[sl])


def vector_intervals(vreq, Vdc, Tsw, strategy):
    """Czasy trwania wektorów +Vdc, 0, -Vdc w jednym okresie przełączania
    tak, by  (+Vdc)·T+ + 0·T0 + (-Vdc)·T- = Vreq·Tsw.
    strategy: 'bipolar' (T0=0), 'unipolar' (T+ lub T- = 0)."""
    x = float(np.clip(vreq / Vdc, -1, 1))
    if strategy == "bipolar":
        Tp = 0.5 * (1 + x) * Tsw
        return Tp, 0.0, Tsw - Tp
    if x >= 0:
        return x * Tsw, (1 - x) * Tsw, 0.0
    return 0.0, (1 + x) * Tsw, -x * Tsw


# =============================================================================
# ROZDZIAŁ 5 - przekształtniki trójfazowe
# =============================================================================

def clarke(a, b, c):
    """Transformacja Clarke'a (amplitudowo niezmiennicza, jak w książce):
    v_alfa = 2/3 (va - vb/2 - vc/2),  v_beta = (vb - vc)/sqrt(3),  v0 = (va+vb+vc)/3."""
    al = (2.0 / 3.0) * (a - 0.5 * b - 0.5 * c)
    be = (b - c) / SQ3
    z = (a + b + c) / 3.0
    return al, be, z


def inv_clarke(al, be, z=0.0):
    a = al + z
    b = -0.5 * al + SQ3 / 2 * be + z
    c = -0.5 * al - SQ3 / 2 * be + z
    return a, b, c


# Numeracja wektorów zgodna z książką (Tabela 5.1): (a, b, c)
VEC2 = {0: (0, 0, 0), 1: (1, 0, 0), 2: (0, 1, 0), 3: (0, 0, 1),
        4: (0, 1, 1), 5: (1, 0, 1), 6: (1, 1, 0), 7: (1, 1, 1)}


def two_level_table(Vdc=1.0):
    """Tabela 5.1/5.2: napięcia fazowe, międzyprzewodowe i wektor Clarke'a
    dla 8 stanów przekształtnika trójgałęziowego."""
    rows = []
    for k, (a, b, c) in VEC2.items():
        vaN, vbN, vcN = a * Vdc, b * Vdc, c * Vdc
        vn = (vaN + vbN + vcN) / 3
        van, vbn, vcn = vaN - vn, vbN - vn, vcN - vn
        vab, vbc, vca = vaN - vbN, vbN - vcN, vcN - vaN
        al, be, _ = clarke(van, vbn, vcn)
        al2, be2, _ = clarke(vab, vbc, vca)
        rows.append(dict(k=k, s=(a, b, c), van=van, vbn=vbn, vcn=vcn, vab=vab,
                         vbc=vbc, vca=vca, V=al + 1j * be, VLL=al2 + 1j * be2))
    return rows


def svpwm_sector(val, vbe):
    """Algorytm z rys. 5.15 (tylko porównania, bez arcus tangens):
    zwraca parę wektorów (Vx, Vy) - Vx: wektor z jedną jedynką (bliżej V0),
    jest pierwszy w sekwencji Tabeli 5.3 - oraz numer sektora 1..6."""
    t60 = 1.732
    if val >= 0:
        if vbe >= 0:
            return ((1, 6), 1) if vbe <= val * t60 else ((2, 6), 2)
        return ((1, 5), 6) if vbe >= -val * t60 else ((3, 5), 5)
    if vbe >= 0:
        return ((2, 4), 3) if vbe <= -val * t60 else ((2, 6), 2)
    return ((3, 4), 4) if vbe >= val * t60 else ((3, 5), 5)


def vec2(k, Vdc=1.0):
    a, b, c = VEC2[k]
    return (2.0 / 3.0) * Vdc * (a + b * A120 + c * A120 ** 2)


def svpwm_times(Vr, T, Vdc=1.0):
    """Czasy t1 (Vx), t2 (Vy), t0 (zerowe) z równania  Vr·T = Vx t1 + Vy t2 + V0 t0
    (rozdzielenie na część rzeczywistą i urojoną - wzór 5.32).
    Przy przemodulowaniu (t1+t2 > T) czasy są skalowane do T."""
    (x, y), sec = svpwm_sector(Vr.real, Vr.imag)
    Vx, Vy = vec2(x, Vdc), vec2(y, Vdc)
    det = Vx.real * Vy.imag - Vy.real * Vx.imag
    t1 = T * (Vr.real * Vy.imag - Vy.real * Vr.imag) / det
    t2 = T * (Vx.real * Vr.imag - Vr.real * Vx.imag) / det
    over = (t1 + t2) > T * (1 + 1e-9)
    raw = (t1, t2)
    if over:
        s = T / (t1 + t2)
        t1, t2 = t1 * s, t2 * s
    t0 = T - t1 - t2
    return dict(x=x, y=y, sector=sec, t1=t1, t2=t2, t0=t0, over=over, raw=raw)


def svpwm_leg_duties(Vr, Vdc=1.0):
    """Współczynniki wypełnienia trzech gałęzi przy sekwencji symetrycznej
    V0 - Vx - Vy - V7 - V7 - Vy - Vx - V0 (Tabela 5.3)."""
    r = svpwm_times(Vr, 1.0, Vdc)
    sx, sy = VEC2[r["x"]], VEC2[r["y"]]
    d = np.array([r["t0"] / 2 + r["t1"] * sx[i] + r["t2"] * sy[i] for i in range(3)])
    return d, r


def three_phase_sim(Vdc, R, L, f1, fsw, Vamp, mode="svpwm", n_cyc=3, pps=200):
    """Trójgałęziowy falownik z obciążeniem R-L w gwiazdę z izolowanym punktem
    neutralnym. mode: 'spwm' (sinus-trójkąt, rys. 5.6-5.8) lub 'svpwm' (rozdz. 5.7).
    Vamp - żądana amplituda napięcia fazowego [V].
    Modulacja próbkowana regularnie raz na okres przełączania."""
    n_sw_cyc = int(round(fsw / f1))
    fsw = n_sw_cyc * f1                         # całkowita liczba okresów nośnej
    Nsw = n_sw_cyc * n_cyc
    T = 1.0 / fsw
    N = Nsw * pps
    t = np.arange(N) * T / pps
    dt = T / pps
    k_sw = np.arange(Nsw)
    tc = (k_sw + 0.5) * T                         # chwila próbkowania (środek okresu)
    th = 2 * PI * f1 * tc
    D = np.empty((3, Nsw))
    if mode == "spwm":
        for i in range(3):
            m = (Vamp / (Vdc / 2)) * np.cos(th - i * 2 * PI / 3)
            D[i] = np.clip(0.5 * (1 + m), 0, 1)
    else:
        for k in range(Nsw):
            Vr = Vamp * np.exp(1j * th[k])
            d, _ = svpwm_leg_duties(Vr, Vdc)
            D[:, k] = d
    tau = (np.arange(N) % pps + 0.5) / pps        # pozycja w okresie 0..1
    kk = np.arange(N) // pps
    S = np.abs(tau - 0.5)[None, :] < (D[:, kk] / 2)   # impuls wyśrodkowany
    vN = S.astype(float) * Vdc
    vn = vN.mean(axis=0)
    vph = vN - vn
    i = rl_response(vph, dt, R, L)
    ppc = n_sw_cyc * pps
    sl = slice(N - ppc, N)
    return dict(t=t[sl] - t[N - ppc], S=S[:, sl], vN=vN[:, sl], vph=vph[:, sl],
                i=i[:, sl], vnN=vn[sl], D=D, fsw=fsw)


# =============================================================================
# ROZDZIAŁ 6 - przekształtniki wielopoziomowe (NPC)
# =============================================================================

def npc3_leg(S, i_dir):
    """Gałąź 3-poziomowa NPC (S1..S4 od góry, diody clampujące Dm1, Dm2).
    i_dir = +1: prąd wypływa z zacisku o; -1: wpływa.
    Zwraca dict(forbidden, level (0,1,2 -> 0, Vdc/2, Vdc), path, msg)."""
    S1, S2, S3, S4 = S
    if S1 and S2 and S3 and S4:
        return dict(forbidden=True, level=None, path=["S1", "S2", "S3", "S4"],
                    msg="ZABRONIONE: zwarcie całego źródła DC (S1..S4 ON)")
    if S1 and S2 and S3:
        return dict(forbidden=True, level=None, path=["S1", "S2", "S3", "Dm2"],
                    msg="ZABRONIONE: zwarcie GÓRNEGO kondensatora (S1-S2-S3-Dm2)")
    if S2 and S3 and S4:
        return dict(forbidden=True, level=None, path=["Dm1", "S2", "S3", "S4"],
                    msg="ZABRONIONE: zwarcie DOLNEGO kondensatora (Dm1-S2-S3-S4)")
    if i_dir > 0:          # prąd wypływa: szukamy drogi "z góry"
        if S1 and S2:
            lev, path = 2, ["S1", "S2"]
        elif S2:
            lev, path = 1, ["Dm1", "S2"]
        else:
            lev, path = 0, ["D4", "D3"]
    else:                  # prąd wpływa: droga "w dół"
        if S3 and S4:
            lev, path = 0, ["S3", "S4"]
        elif S3:
            lev, path = 1, ["S3", "Dm2"]
        else:
            lev, path = 2, ["D2", "D1"]
    intentional = (S1 and S2 and not S3 and not S4) or (S2 and S3 and not S1 and not S4) \
        or (S3 and S4 and not S1 and not S2)
    names = {0: "0 (szyna N)", 1: "Vdc/2 (punkt M)", 2: "Vdc (szyna P)"}
    msg = f"v_oN = {names[lev]}; przewodzą: {', '.join(path)}"
    if not intentional:
        msg += "  [stan NIEZAMIERZONY - wynik zależy od kierunku prądu!]"
    return dict(forbidden=False, level=lev, path=path, msg=msg, intentional=intentional)


def multilevel_vectors(k, Vdc=1.0):
    """Wszystkie k^3 stany trójfazowego przekształtnika k-poziomowego:
    indeks = La*k^2 + Lb*k + Lc (numeracja jak Tabela 6.3), wektor Clarke'a
    napięć fazowych (składowa zerowa znika automatycznie)."""
    out = []
    for La in range(k):
        for Lb in range(k):
            for Lc in range(k):
                va, vb, vc = (np.array([La, Lb, Lc]) / (k - 1)) * Vdc
                al, be, _ = clarke(va, vb, vc)
                out.append(((La, Lb, Lc), complex(round(al, 9), round(be, 9))))
    return out


def ml_sequence(u, delta):
    """SVPWM wielopoziomowe (najbliższe trzy wektory) metodą rozkładu
    'część całkowita + ułamkowa'. u - napięcia fazowe w jednostkach poziomu
    (0..k-1), delta - wspólne przesunięcie (wybór wektorów redundantnych).
    Zwraca listę 4 stanów (sekwencja, każdy krok zmienia JEDNĄ fazę o jeden
    poziom) i czasy względne [d0, d1, d2, d3] (suma = 1)."""
    w = np.asarray(u, dtype=float) + delta
    f = np.floor(w + 1e-12)
    r = w - f
    order = np.argsort(-r)
    rs = r[order]
    states = [f.copy()]
    s = f.copy()
    for idx in order:
        s = s.copy()
        s[idx] += 1
        states.append(s)
    d = [1 - rs[0], rs[0] - rs[1], rs[1] - rs[2], rs[2]]
    return [tuple(int(v) for v in st) for st in states], d


def ml_delta_range(u, k):
    """Dopuszczalny zakres przesunięcia delta: 0 <= u+delta  i  u+delta < k-1."""
    lo = -min(u)
    hi = (k - 1) - max(u) - 1e-9
    return lo, hi


def ml_delta_options(u, k):
    """Warianty wyboru sekwencji redundantnej (rozdz. 6.6).
    Przedział dopuszczalnych delta dzieli się na podprzedziały, w których część
    całkowita (stan początkowy) jest stała - każdy podprzedział to INNA para
    redundantnych realizacji wektora Vz. Zwraca środki: najniższego
    ('lo' - sekwencja korzystająca z dolnych poziomów / dolnego kondensatora),
    najwyższego ('hi') oraz środek całego przedziału ('mid')."""
    lo, hi = ml_delta_range(u, k)
    if hi <= lo:
        return dict(lo=lo, hi=lo, mid=lo, n=1, rng=(lo, hi))
    br = sorted({n - ux for ux in u for n in range(-k, 2 * k)
                 if lo + 1e-9 < n - ux < hi - 1e-9})
    edges = [lo] + br + [hi]
    return dict(lo=0.5 * (edges[0] + edges[1]), hi=0.5 * (edges[-2] + edges[-1]),
                mid=0.5 * (lo + hi), n=len(edges) - 1, rng=(lo, hi))


def ml_index(state, k):
    La, Lb, Lc = state
    return La * k * k + Lb * k + Lc


def npc3_sim(Vdc, R, L, C, f1, fsw, Vamp, mode="hyst", band_pct=2.0, Kp=0.5,
             vc0=0.0, n_cyc=6, pps=100):
    """Trójfazowy 3-poziomowy NPC z obciążeniem R-L, SVPWM (najbliższe 3 wektory)
    i modelem kondensatorów DC.
      v_up + v_low = Vdc (idealne źródło),  vc = v_up - v_low,
      dvc/dt = i_M / C   (i_M - prąd wypływający z punktu środkowego M).
    mode: 'none'  - zawsze sekwencja 'dolna' (bez balansowania),
          'hyst'  - histereza z książki (rozdz. 6.6): vc > vc_max -> używaj
                    górnego kondensatora, vc < vc_min -> dolnego,
          'pred'  - wybór na podstawie znaku przewidywanego i_M,
          'pi'    - ciągłe przesunięcie delta = środek - Kp·sgn(Kb)·vc/(Vdc/2).
    Vamp - amplituda napięcia fazowego [V].
    Obliczenia wektorowe w obrębie okresu przełączania (napięcie kondensatora
    aktualizowane raz na okres - błąd pomijalny)."""
    n_sw_cyc = int(round(fsw / f1))
    fsw = n_sw_cyc * f1
    T = 1.0 / fsw
    Nsw = n_sw_cyc * n_cyc
    dt = T / pps
    a = math.exp(-R * dt / L)
    b = (1 - a) / R
    j = np.arange(pps)
    apow = a ** j
    ainv = a ** -(j + 1.0)
    i = np.zeros(3)
    vc = float(vc0)
    band = band_pct / 100.0 * Vdc
    use_upper = False
    N = Nsw * pps
    out_v = np.empty((3, N)); out_lv = np.empty((3, N), dtype=np.int8)
    out_i = np.empty((3, N)); out_vc = np.empty(N); out_iM = np.empty(N)
    tau = (j + 0.5) / pps
    fold = np.where(tau < 0.5, 2 * tau, 2 * (1 - tau))   # 0..1..0 symetrycznie
    m_lvl = Vamp / (Vdc / 2)
    phi = math.atan2(2 * PI * f1 * L, R)
    I_est = Vamp / math.hypot(R, 2 * PI * f1 * L)
    kb_sign = 1.0 if npc_balance_gain(min(m_lvl, 1.15), I_est, phi) >= 0 else -1.0
    ph = np.arange(3) * 2 * PI / 3
    for k in range(Nsw):
        th = 2 * PI * f1 * (k + 0.5) * T
        u = 1.0 + m_lvl * np.cos(th - ph)
        lo, hi = ml_delta_range(u, 3)
        if hi < lo:                      # przemodulowanie - ograniczenie
            u = 1.0 + (u - 1.0) * (2.0 / (u.max() - u.min())) * 0.999
            lo, hi = ml_delta_range(u, 3)
            hi = max(hi, lo)
        mid = 0.5 * (lo + hi)
        opt = ml_delta_options(u, 3)
        if mode == "none":
            delta = opt["lo"]
        elif mode == "hyst":
            if vc > band:
                use_upper = True
            elif vc < -band:
                use_upper = False
            delta = opt["hi"] if use_upper else opt["lo"]
        elif mode == "pred":
            iM_lo = float(np.sum(i * (1 - np.abs(u + opt["lo"] - 1))))
            iM_hi = float(np.sum(i * (1 - np.abs(u + opt["hi"] - 1))))
            delta = opt["lo"] if iM_lo * vc < iM_hi * vc else opt["hi"]
        else:  # 'pi'
            delta = float(np.clip(mid - Kp * kb_sign * vc / (Vdc / 2), lo, hi))
        w = u + delta
        f = np.floor(w + 1e-12)
        r = w - f
        lev = f[:, None] + (r[:, None] > 1 - fold[None, :])      # (3, pps)
        v_low = 0.5 * Vdc - 0.5 * vc
        vN = np.where(lev == 2, Vdc, np.where(lev == 1, v_low, 0.0))
        vph = vN - vN.mean(axis=0)
        wv = vph * ainv
        cs = np.concatenate([np.zeros((3, 1)), np.cumsum(wv, axis=1)[:, :-1]], axis=1)
        ib = apow * (i[:, None] + b * cs)
        iM = np.sum(ib * (lev == 1), axis=0)
        sl = slice(k * pps, (k + 1) * pps)
        out_v[:, sl] = vph; out_lv[:, sl] = lev; out_i[:, sl] = ib
        out_vc[sl] = np.clip(vc + np.concatenate([[0.0], np.cumsum(iM)[:-1]]) * dt / C,
                             -Vdc, Vdc)
        out_iM[sl] = iM
        vc = float(np.clip(out_vc[sl][-1] + iM[-1] * dt / C, -Vdc, Vdc))
        i = a ** pps * (i + b * np.sum(wv, axis=1))
    t = np.arange(N) * dt
    return dict(t=t, vph=out_v, lev=out_lv, i=out_i, vc=out_vc, iM=out_iM,
                band=band, fsw=fsw, ppc=n_sw_cyc * pps)


def npc_balance_gain(m_lvl, I, phi, n=720):
    """Wzmocnienie małosygnałowe Kb = d<i_M>/d(delta) uśrednione po okresie
    podstawowym [A na jednostkę delta], dla prądów sinusoidalnych I∠-phi.
    <i_M> = sum_x i_x * (1 - |u_x + delta - 1|)  =>  d/d delta = -sum i_x sign(u_x+delta-1)."""
    th = np.linspace(0, 2 * PI, n, endpoint=False)
    acc = 0.0
    for t_ in th:
        u = 1.0 + m_lvl * np.cos(t_ - np.arange(3) * 2 * PI / 3)
        lo, hi = ml_delta_range(u, 3)
        mid = 0.5 * (lo + hi)
        ix = I * np.cos(t_ - phi - np.arange(3) * 2 * PI / 3)
        acc += -np.sum(ix * np.sign(u + mid - 1))
    return acc / n
