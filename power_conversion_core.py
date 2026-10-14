"""
Rdzeń obliczeniowy do aplikacji power_conversion_examples_tkinter.py
=====================================================================
Kirschen, "Power Systems: Fundamental Concepts and the Transition to
Sustainability" - rozdziały 5 (Electronic Power Conversion),
6 (Balancing Load and Generation) i 7 (Network Components).

Tylko obliczenia (numpy/scipy) - bez GUI, więc można je testować osobno
(test_power_conversion_core.py).
"""
import math
import cmath
from itertools import product

import numpy as np

PI = math.pi
SQ3 = math.sqrt(3.0)


def pol(z, deg=True):
    """Liczba zespolona -> (moduł, kąt w stopniach)."""
    return abs(z), math.degrees(cmath.phase(z)) if deg else cmath.phase(z)


def fmt_c(z, unit="", nd=4):
    m, a = pol(z)
    return f"{m:.{nd}f}∠{a:+.2f}° {unit}".rstrip()


# =============================================================================
# ROZDZIAŁ 5 - przekształtniki
# =============================================================================

def converter_capability(S_rated, pf):
    """Przykład 5.1: P^2 + Q^2 <= S^2 (ograniczenie prądowe przekształtnika).
    Zwraca (P_max, Q przy P_max)."""
    pf = min(max(pf, 0.0), 1.0)
    P = S_rated * pf
    Q = math.sqrt(max(S_rated ** 2 - P ** 2, 0.0))
    return P, Q


def converter_power_flow(E, V, X, delta_deg):
    """Moc czynna i bierna płynąca od źródła E∠δ do sieci V∠0 przez reaktancję X:
    P = E V sin δ / X,   Q = (E^2 - E V cos δ) / X  (wysyłana przez przekształtnik),
    Q_sieć = (E V cos δ - V^2) / X (odbierana przez sieć)."""
    d = math.radians(delta_deg)
    P = E * V * math.sin(d) / X
    Q_send = (E ** 2 - E * V * math.cos(d)) / X
    Q_recv = (E * V * math.cos(d) - V ** 2) / X
    return P, Q_send, Q_recv


def triangle(t, fc):
    """Nośna trójkątna -1..+1 o częstotliwości fc."""
    x = (t * fc) % 1.0
    return 4.0 * np.abs(x - 0.5) - 1.0


def hbridge_pwm(ma, mf, vdc, f1=50.0, n=8000, bipolar=True):
    """Mostek H z PWM (Rys. 5.4): porównanie sinusoidy z nośną trójkątną.
    Zwraca t, modulator, nośna, Vs(t)."""
    t = np.linspace(0.0, 2.0 / f1, n, endpoint=False)
    mod = ma * np.sin(2 * PI * f1 * t)
    car = triangle(t, mf * f1)
    if bipolar:
        vs = np.where(mod >= car, vdc, -vdc)
    else:  # unipolarny: dwie gałęzie z przesuniętym odniesieniem
        va = np.where(mod >= car, 1, 0)
        vb = np.where(-mod >= car, 1, 0)
        vs = vdc * (va - vb)
    return t, mod, car, vs.astype(float)


def spectrum(v):
    """Amplitudy harmonicznych (jednostronne). Okno = całkowita liczba okresów."""
    N = len(v)
    a = np.abs(np.fft.rfft(v)) / N * 2
    a[0] /= 2
    return a


def pv_curve(v, irr=1000.0, temp=25.0, Voc0=40.0, Isc0=9.0):
    """Prosty model charakterystyki I-V modułu PV (model wykładniczy).
    irr [W/m2], temp [°C]. Zwraca prąd I(v)."""
    Isc = Isc0 * irr / 1000.0 * (1 + 0.0005 * (temp - 25))
    Voc = Voc0 * (1 - 0.0035 * (temp - 25)) + 1.2 * math.log(max(irr, 1) / 1000.0)
    Vt = 1.8 * Voc0 / 40.0  # "napięcie termiczne" całego modułu (60 ogniw)
    I = Isc * (1 - np.exp((v - Voc) / Vt))
    return np.clip(I, 0, None)


def mppt_perturb_observe(irr_profile, dV=0.5, V0=20.0, **kw):
    """Algorytm MPPT "perturb and observe" (rozdz. 5.4.2) krok po kroku."""
    V = V0
    step = dV
    P_old = V * pv_curve(np.array([V]), irr_profile[0], **kw)[0]
    Vs, Ps = [], []
    for irr in irr_profile:
        V_new = V + step
        P_new = V_new * pv_curve(np.array([V_new]), irr, **kw)[0]
        if P_new < P_old:
            step = -step
        V, P_old = V_new, P_new
        Vs.append(V)
        Ps.append(P_new)
    return np.array(Vs), np.array(Ps)


def type4_frequency(p_poles, G, N_rpm):
    """Typ 4: częstotliwość napięcia generatora synchronicznego f = p G N / 120."""
    return p_poles * G * N_rpm / 120.0


def dfig_split(Pm, slip):
    """Typ 3 (DFIG), bez strat: Ps = Pm/(1-s), Pr = -s Ps  (Pm = Ps + Pr).
    Pr > 0 - moc wyciągana z wirnika (praca nadsynchroniczna, s < 0)."""
    Ps = Pm / (1.0 - slip)
    Pr = Pm - Ps
    return Ps, Pr


# =============================================================================
# ROZDZIAŁ 6 - regulacja częstotliwości (statyzm)
# =============================================================================

def droop_frequency(P, Pmax, rho, fnom):
    """Równanie (6.12): f = f_nom (1 - ρ P / P_max)."""
    return fnom * (1.0 - rho * P / Pmax)


def droop_sharing(PT, Pmaxes, rhos, fnom):
    """Kilka generatorów o statyzmie ρ_i: P_i = P_i,max (f_nom - f)/(ρ_i f_nom).
    Suma P_i = P_T -> f = f_nom (1 - P_T / Σ(P_i,max/ρ_i))."""
    k = sum(Pm / r for Pm, r in zip(Pmaxes, rhos))
    f = fnom * (1.0 - PT / k)
    Ps = [Pm / r * (fnom - f) / fnom for Pm, r in zip(Pmaxes, rhos)]
    return f, Ps


def frequency_dynamics(H, Pmax, rho, P0, dP, fnom=60.0, Tg=0.5, D=1.0,
                       t_end=20.0, dt=0.005, t_step=1.0, secondary_Ki=0.0):
    """Symulacja dynamiki częstotliwości jednego generatora (6.3-6.6).
    Równanie ruchu w jednostkach względnych (baza P_max):
        2H dΔω/dt = Pm - Pe - D Δω
        Tg dPm/dt = Pref - Δω/ρ - Pm           (regulator z statyzmem)
        dPref/dt  = -Ki Δω                       (regulacja wtórna, opcjonalna)
    Zwraca t, f(t) [Hz], Pm(t) [pu], Pe(t) [pu]."""
    n = int(t_end / dt) + 1
    t = np.linspace(0, t_end, n)
    w = 0.0
    Pm = P0
    Pref = P0
    f = np.empty(n)
    Pms = np.empty(n)
    Pes = np.empty(n)

    def deriv(w, Pm, Pref, Pe):
        dw = (Pm - Pe - D * w) / (2 * H)
        dPm = (Pref - w / rho - Pm) / Tg
        dPref = -secondary_Ki * w
        return dw, dPm, dPref

    for i, ti in enumerate(t):
        Pe = P0 + (dP if ti >= t_step else 0.0)
        f[i] = fnom * (1 + w)
        Pms[i] = Pm
        Pes[i] = Pe
        # RK4
        k1 = deriv(w, Pm, Pref, Pe)
        k2 = deriv(w + dt / 2 * k1[0], Pm + dt / 2 * k1[1], Pref + dt / 2 * k1[2], Pe)
        k3 = deriv(w + dt / 2 * k2[0], Pm + dt / 2 * k2[1], Pref + dt / 2 * k2[2], Pe)
        k4 = deriv(w + dt * k3[0], Pm + dt * k3[1], Pref + dt * k3[2], Pe)
        w += dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        Pm += dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        Pref += dt / 6 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
    return t, f, Pms, Pes


# =============================================================================
# ROZDZIAŁ 6 - rozdział ekonomiczny (economic dispatch)
# =============================================================================

# Funkcje kosztu z Przykładu 6.3: C_i(P) = a + b P + c P^2  [$/h]
# (odtworzone z tabeli Przykładu 6.7 - zgadzają się co do wartości:
#  C1(200)=7700, C2(300)=14700, C3(420)=22378 ...)
UNITS_Q = [
    dict(name="Jednostka 1", a=2100.0, b=8.0, c=0.10, Pmin=200.0, Pmax=500.0),
    dict(name="Jednostka 2", a=7200.0, b=7.0, c=0.06, Pmin=200.0, Pmax=450.0),
    dict(name="Jednostka 3", a=6250.0, b=9.0, c=0.07, Pmin=200.0, Pmax=500.0),
]


def q_cost(u, P):
    return u["a"] + u["b"] * P + u["c"] * P ** 2


def q_marginal(u, P):
    return u["b"] + 2 * u["c"] * P


def ed_quadratic(units, L, limits=False, max_iter=10):
    """Rozdział ekonomiczny metodą mnożnika Lagrange'a (6.24)-(6.31).
    dC_i/dP_i = b_i + 2 c_i P_i = λ  ->  P_i = (λ - b_i)/(2 c_i).
    Gdy limits=True: jednostki poza granicami są "przypinane" do granicy,
    a pozostałe rozdzielają resztę (procedura z Przykładów 6.5 i 6.6).
    Zwraca dict(P=[...], lam=λ, cost=..., fixed=[...], log=[...])."""
    n = len(units)
    fixed = {}
    log = []
    for _ in range(max_iter):
        free = [i for i in range(n) if i not in fixed]
        Lnet = L - sum(fixed.values())
        s1 = sum(1 / (2 * units[i]["c"]) for i in free)
        s0 = sum(units[i]["b"] / (2 * units[i]["c"]) for i in free)
        lam = (Lnet + s0) / s1 if free else float("nan")
        P = [fixed[i] if i in fixed else (lam - units[i]["b"]) / (2 * units[i]["c"])
             for i in range(n)]
        log.append((lam, list(P), dict(fixed)))
        if not limits:
            break
        viol = False
        for i in free:
            if P[i] > units[i]["Pmax"] + 1e-9:
                fixed[i] = units[i]["Pmax"]
                viol = True
            elif P[i] < units[i]["Pmin"] - 1e-9:
                fixed[i] = units[i]["Pmin"]
                viol = True
        if not viol:
            break
    cost = sum(q_cost(units[i], P[i]) for i in range(n))
    return dict(P=P, lam=lam, cost=cost, fixed=fixed, log=log)


# Tabela z Przykładu 6.7 - odcinkowo-liniowe krzywe kosztu
UNITS_PWL = [
    dict(name="Jednostka 1", Pmin=200, e1=300, e2=400, Pmax=500,
         Cmin=7700, c=[58.0, 78.0, 98.0]),
    dict(name="Jednostka 2", Pmin=200, e1=300, e2=420, Pmax=450,
         Cmin=11000, c=[37.0, 50.2, 59.2]),
    dict(name="Jednostka 3", Pmin=200, e1=300, e2=420, Pmax=500,
         Cmin=10850, c=[44.0, 59.4, 73.4]),
]


def pwl_points(u):
    """Punkty załamania (P, C) krzywej odcinkowo-liniowej."""
    Ps = [u["Pmin"], u["e1"], u["e2"], u["Pmax"]]
    Cs = [u["Cmin"]]
    for k in range(3):
        Cs.append(Cs[-1] + u["c"][k] * (Ps[k + 1] - Ps[k]))
    return Ps, Cs


def pwl_cost(u, P):
    Ps, Cs = pwl_points(u)
    return float(np.interp(P, Ps, Cs))


def pwl_segments(units, idx=None):
    """Lista odcinków posortowana rosnąco wg kosztu krańcowego (krok 2-3 algorytmu)."""
    idx = range(len(units)) if idx is None else idx
    segs = []
    for i in idx:
        u = units[i]
        Ps = [u["Pmin"], u["e1"], u["e2"], u["Pmax"]]
        for k in range(3):
            segs.append(dict(unit=i, k=k, length=Ps[k + 1] - Ps[k], mc=u["c"][k]))
    segs.sort(key=lambda s: s["mc"])
    return segs


def ed_pwl(units, L, idx=None):
    """Algorytm 'merit order' z rozdz. 6.8.3 (kroki 1-8).
    idx - które jednostki są włączone (domyślnie wszystkie).
    Zwraca dict(P, cost, lam, marginal_unit, feasible, trace)."""
    idx = list(range(len(units))) if idx is None else list(idx)
    P = {i: 0.0 for i in range(len(units))}
    for i in idx:
        P[i] = float(units[i]["Pmin"])
    pmin_sum = sum(units[i]["Pmin"] for i in idx)
    pmax_sum = sum(units[i]["Pmax"] for i in idx)
    if L < pmin_sum - 1e-9 or L > pmax_sum + 1e-9 or not idx:
        return dict(P=[P[i] for i in range(len(units))], cost=float("nan"),
                    lam=float("nan"), marginal_unit=None, feasible=False, trace=[])
    resid = L - pmin_sum
    trace = []
    lam, mu = float("nan"), None
    segs = pwl_segments(units, idx)
    for s in segs:
        if resid <= 1e-9:
            # Następny MW dostarczy pierwszy niewykorzystany odcinek
            lam, mu = s["mc"], s["unit"]
            break
        take = min(s["length"], resid)
        P[s["unit"]] += take
        resid -= take
        trace.append((s, take))
        if take < s["length"] - 1e-9:
            lam, mu = s["mc"], s["unit"]
            break
    else:
        lam, mu = (segs[-1]["mc"], segs[-1]["unit"])
    cost = sum(pwl_cost(units[i], P[i]) for i in idx)
    return dict(P=[P[i] for i in range(len(units))], cost=cost, lam=lam,
                marginal_unit=mu, feasible=True, trace=trace)


def combos_table(units, L):
    """Przykład 6.11 - wszystkie kombinacje jednostek dla obciążenia L."""
    rows = []
    n = len(units)
    for mask in product([0, 1], repeat=n):
        idx = [i for i in range(n) if mask[i]]
        if not idx:
            continue
        r = ed_pwl(units, L, idx)
        rows.append((mask, r))
    return rows


# =============================================================================
# ROZDZIAŁ 6 - Unit commitment (MILP, scipy.optimize.milp)
# =============================================================================

def unit_commitment(units, load, reserve=0.0, ramp=False, init_on=None, init_P=None,
                    reserve_within_ramp=None):
    """Rozwiązuje problem doboru jednostek (6.36)-(6.44) jako MILP.

    units: lista dict(name, Pmin, Pmax, mc [$/MWh], st [$ za rozruch],
                      min_up [h], min_down [h], ramp_up [MW/h])
    load: lista obciążeń godzinowych; reserve: wymagana rezerwa (headroom) [MW].
    init_on: stan początkowy (True/False) każdej jednostki.

    Zmienne (na jednostkę i godzinę): P, u (stan 0/1), v (rozruch), w (odstawienie),
    r (wkład w rezerwę). Rezerwa: r <= Pmax u - P, a gdy włączone są ograniczenia
    szybkości zmian mocy, także r <= ΔP_up (rezerwa musi być osiągalna w ciągu godziny).
    """
    if reserve_within_ramp is None:
        reserve_within_ramp = ramp
    from scipy.optimize import milp, LinearConstraint, Bounds

    G, T = len(units), len(load)
    init_on = init_on or [False] * G
    init_P = init_P or [0.0] * G
    nv = 5 * G * T

    def iP(g, t): return 0 * G * T + g * T + t
    def iu(g, t): return 1 * G * T + g * T + t
    def iv(g, t): return 2 * G * T + g * T + t
    def iw(g, t): return 3 * G * T + g * T + t
    def ir(g, t): return 4 * G * T + g * T + t

    c = np.zeros(nv)
    for g, u in enumerate(units):
        for t in range(T):
            c[iP(g, t)] = u["mc"]
            c[iv(g, t)] = u.get("st", 0.0)
            c[iu(g, t)] = u.get("nl", 0.0)
    A, lo, hi = [], [], []

    def add(coefs, l, h):
        row = np.zeros(nv)
        for k, val in coefs:
            row[k] += val
        A.append(row); lo.append(l); hi.append(h)

    for t in range(T):
        # bilans mocy
        add([(iP(g, t), 1.0) for g in range(G)], load[t], load[t])
        if reserve > 0:
            # Σ r >= R,   r <= Pmax u - P,   (r <= ΔP_up u)
            add([(ir(g, t), 1.0) for g in range(G)], reserve, np.inf)
            for g, u in enumerate(units):
                add([(ir(g, t), 1.0), (iu(g, t), -u["Pmax"]), (iP(g, t), 1.0)], -np.inf, 0)
                if reserve_within_ramp and u.get("ramp_up") is not None:
                    add([(ir(g, t), 1.0), (iu(g, t), -u["ramp_up"])], -np.inf, 0)
    for g, u in enumerate(units):
        u0 = 1.0 if init_on[g] else 0.0
        for t in range(T):
            # Pmin u <= P <= Pmax u
            add([(iP(g, t), 1.0), (iu(g, t), -u["Pmin"])], 0, np.inf)
            add([(iP(g, t), 1.0), (iu(g, t), -u["Pmax"])], -np.inf, 0)
            # u_t - u_{t-1} = v_t - w_t
            if t == 0:
                add([(iu(g, t), 1.0), (iv(g, t), -1.0), (iw(g, t), 1.0)], u0, u0)
            else:
                add([(iu(g, t), 1.0), (iu(g, t - 1), -1.0), (iv(g, t), -1.0), (iw(g, t), 1.0)], 0, 0)
            add([(iv(g, t), 1.0), (iw(g, t), 1.0)], -np.inf, 1)
            # minimalny czas pracy: Σ_{τ=t-UT+1..t} v_τ <= u_t
            UT = int(u.get("min_up", 1))
            if UT > 1:
                co = [(iv(g, k), 1.0) for k in range(max(0, t - UT + 1), t + 1)]
                add(co + [(iu(g, t), -1.0)], -np.inf, 0)
            # minimalny czas postoju: Σ w_τ <= 1 - u_t
            DT = int(u.get("min_down", 1))
            if DT > 1:
                co = [(iw(g, k), 1.0) for k in range(max(0, t - DT + 1), t + 1)]
                add(co + [(iu(g, t), 1.0)], -np.inf, 1)
            # ograniczenie szybkości zmian mocy (tylko w górę, jak w Przykładzie 6.14)
            if ramp and u.get("ramp_up") is not None:
                RU = u["ramp_up"]
                if t == 0:
                    add([(iP(g, t), 1.0)], -np.inf, init_P[g] + RU)
                else:
                    add([(iP(g, t), 1.0), (iP(g, t - 1), -1.0)], -np.inf, RU)
    integrality = np.zeros(nv)
    integrality[G * T:4 * G * T] = 1
    lb = np.zeros(nv)
    pmax_rep = np.repeat([u["Pmax"] for u in units], T)
    ub = np.concatenate([pmax_rep, np.ones(3 * G * T), pmax_rep])
    res = milp(c, constraints=LinearConstraint(np.array(A), lo, hi),
               integrality=integrality, bounds=Bounds(lb, ub))
    if not res.success:
        return dict(ok=False, msg=res.message)
    x = res.x
    P = np.array([[x[iP(g, t)] for t in range(T)] for g in range(G)])
    U = np.array([[round(x[iu(g, t)]) for t in range(T)] for g in range(G)])
    V = np.array([[round(x[iv(g, t)]) for t in range(T)] for g in range(G)])
    head = np.array([sum(units[g]["Pmax"] * U[g, t] - P[g, t] for g in range(G)) for t in range(T)])
    energy_cost = float(sum(units[g]["mc"] * P[g].sum() for g in range(G)))
    st_cost = float(sum(units[g].get("st", 0.0) * V[g].sum() for g in range(G)))
    return dict(ok=True, P=P, U=U, V=V, cost=float(res.fun), energy_cost=energy_cost,
                startup_cost=st_cost, headroom=head)


def schedule_cost(units, P, init_on=None):
    """Koszt zadanego harmonogramu P[g][t] (do porównania z tabelą z książki)."""
    G = len(units)
    init_on = init_on or [False] * G
    cost = 0.0
    for g, u in enumerate(units):
        prev = init_on[g]
        for p in P[g]:
            on = p > 1e-9
            cost += u["mc"] * p
            if on and not prev:
                cost += u.get("st", 0.0)
            prev = on
    return cost


# =============================================================================
# ROZDZIAŁ 7 - linie, transformatory
# =============================================================================

def line_pi(r, x, y, length, Vbase_kV, Sbase_MVA):
    """Przykład 7.1: parametry schematu π w jednostkach względnych.
    r, x [Ω/km], y [S/km] (susceptancja poprzeczna)."""
    Zb = Vbase_kV ** 2 / Sbase_MVA
    R = r * length / Zb
    X = x * length / Zb
    B = y * length * Zb
    return dict(Zb=Zb, R=R, X=X, B=B, Z=complex(R, X), Ysh=complex(0, B / 2))


def line_open_end(pi, Vs=1.0 + 0j):
    """Przykład 7.2: linia nieobciążona (efekt Ferrantiego)."""
    Z, Y2 = pi["Z"], pi["Ysh"]
    Zc = 1 / Y2
    VR = Vs * Zc / (Z + Zc)
    I1 = VR * Y2                 # prąd przez Z (i drugą gałąź poprzeczną)
    I2 = Vs * Y2                 # prąd pierwszej gałęzi poprzecznej
    I = I1 + I2
    S = Vs * I.conjugate()
    return dict(VR=VR, I1=I1, I2=I2, I=I, S=S)


def line_with_load(pi, P, Q, VR=1.0 + 0j):
    """Przykład 7.3: obciążenie P+jQ [pu] na końcu linii, VR zadane."""
    Z, Y2 = pi["Z"], pi["Ysh"]
    Iload = (complex(P, Q) / VR).conjugate()
    I1 = Iload + VR * Y2
    Vs = VR + Z * I1
    I2 = Vs * Y2
    I = I1 + I2
    S = Vs * I.conjugate()
    return dict(Vs=Vs, I1=I1, I2=I2, I=I, S=S, Iload=Iload,
                losses=S.real - P)


def ideal_transformer(V1_rated, V2_rated, V1, Z2):
    """Przykład 7.4: transformator idealny."""
    a = V1_rated / V2_rated
    V2 = V1 / a
    I2 = V2 / Z2
    I1 = I2 / a
    Z2p = a ** 2 * Z2
    S = V2 * I2.conjugate()
    return dict(a=a, V2=V2, I2=I2, I1=I1, Z2p=Z2p, S=S)


def transformer_currents(S_VA, V1_LL, V2_LL):
    """Przykład 7.5: I = S / (√3 V_LL)."""
    return S_VA / (SQ3 * V1_LL), S_VA / (SQ3 * V2_LL)


def per_unit_system(Sb=100.0, gen=(75.0, 11.0, 1.83), t1=(75.0, 0.125), line_len=20.0,
                    zline=complex(0.18, 0.40), t2=(45.0, 0.125), load_MVA=10.0, pf=0.8,
                    Vb=(11.0, 132.0, 33.0)):
    """Przykład 7.6: zamiana na wspólną bazę Sb i obliczenie napięć/prądów.
    Z_new = Z_old (S_new/S_old)(V_old/V_new)^2 - tu napięcia bazowe = znamionowe."""
    Xg = gen[2] * Sb / gen[0] * (gen[1] / Vb[0]) ** 2
    Xt1 = t1[1] * Sb / t1[0]
    Zb_line = Vb[1] ** 2 / Sb
    Zl_ohm = line_len * zline
    Zl = Zl_ohm / Zb_line
    Xt2 = t2[1] * Sb / t2[0]
    Spu = load_MVA / Sb
    phi = math.acos(pf)
    V_load = 1.0 + 0j
    I = (cmath.rect(Spu, phi) / V_load).conjugate()
    Z_series = 1j * Xt1 + Zl + 1j * Xt2
    V_gen = V_load + Z_series * I
    E = V_gen + 1j * Xg * I
    Ib = [Sb * 1e6 / (SQ3 * v * 1e3) for v in Vb]
    return dict(Xg=Xg, Xt1=Xt1, Zl=Zl, Zl_ohm=Zl_ohm, Zb_line=Zb_line, Xt2=Xt2, Spu=Spu,
                I=I, V_gen=V_gen, E=E, Ib=Ib, V_load=V_load,
                I_A=[I * ib for ib in Ib])


def tap_changer(t, XT=0.1, RL=1.0, XL=5.0, Vs=1.0 + 0j):
    """Przykład 7.7: transformator z przełącznikiem zaczepów 1:t (po stronie obciążenia).
    Schemat π: Y12 = tY, Y10 = t(t-1)Y, Y20 = (1-t)Y, Y = 1/(jXT).
    Rozwiązujemy dokładny model z transformatorem idealnym (sprawdzenie modelu π)."""
    Y = 1 / (1j * XT)
    YL = 1 / RL + 1 / (1j * XL)
    # węzeł obciążenia: I2 = Y (t Vs - VL) = YL VL
    VL = Y * t * Vs / (Y + YL)
    I2 = YL * VL
    Is = t * I2
    S = Vs * Is.conjugate()
    # model π - te same wyniki
    Y12, Y10, Y20 = t * Y, t * (t - 1) * Y, (1 - t) * Y
    VL_pi = Y12 * Vs / (Y12 + Y20 + YL)
    Is_pi = Vs * Y10 + (Vs - VL_pi) * Y12
    Z10 = 1 / Y10 if abs(Y10) > 1e-12 else complex("inf")
    Z20 = 1 / Y20 if abs(Y20) > 1e-12 else complex("inf")
    return dict(VL=VL, Is=Is, S=S, VL_pi=VL_pi, Is_pi=Is_pi, Z10=Z10, Z20=Z20,
                Y12=Y12, Y10=Y10, Y20=Y20)
