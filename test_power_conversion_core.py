"""Testy rdzenia obliczeniowego - porównanie z wynikami przykładów z książki
(Kirschen, rozdz. 5-7). Uruchomienie:  python -m pytest test_power_conversion_core.py
"""
import math

import pytest

import power_conversion_core as core

approx = pytest.approx


def test_ex_5_1_power_factor():
    P, Q = core.converter_capability(5.0, 0.9)
    assert P == approx(4.5)
    assert Q == approx(2.179, abs=1e-3)
    assert core.converter_capability(5.0, 1.0) == approx((5.0, 0.0))


def test_ex_6_1_droop():
    assert core.droop_frequency(25, 50, 0.04, 60) == approx(58.8)


def test_ex_6_2_two_generators():
    f, (P1, P2) = core.droop_sharing(20, [20, 30], [0.04, 0.04], 60)
    assert (P1, P2) == approx((8.0, 12.0))
    assert f == approx(59.04)


def test_droop_dynamics_settles_on_droop_line():
    t, f, Pm, Pe = core.frequency_dynamics(H=4, Pmax=1, rho=0.04, P0=0.5, dP=0.2,
                                           fnom=60, D=0.0, t_end=60)
    # odchyłka f od punktu pracy przed skokiem = przesunięcie po prostej statyzmu
    df = core.droop_frequency(0.7, 1, 0.04, 60) - core.droop_frequency(0.5, 1, 0.04, 60)
    assert f[-1] - 60 == approx(df, abs=1e-3)
    assert Pm[-1] == approx(0.7, abs=1e-3)


@pytest.mark.parametrize("L, lam, cost", [(800, 47.012, 37523.69), (1200, 66.547, None),
                                          (600, 37.244, 29098.11)])
def test_ex_6_3_6_4_quadratic_dispatch(L, lam, cost):
    r = core.ed_quadratic(core.UNITS_Q, L)
    assert r["lam"] == approx(lam, abs=1e-3)
    assert sum(r["P"]) == approx(L)
    if cost:
        assert r["cost"] == approx(cost, abs=0.01)


def test_ex_6_3_unit_outputs():
    r = core.ed_quadratic(core.UNITS_Q, 1200)
    assert r["P"] == approx([292.73, 496.22, 411.05], abs=0.01)
    r = core.ed_quadratic(core.UNITS_Q, 600)
    assert r["P"] == approx([146.22, 252.03, 201.74], abs=0.01)


def test_ex_6_5_limits():
    r = core.ed_quadratic(core.UNITS_Q, 1200, limits=True)
    assert r["lam"] == approx(70.353, abs=1e-3)
    assert r["P"] == approx([311.76, 450.0, 438.24], abs=0.01)


def test_ex_6_6_limits_all_at_pmin():
    r = core.ed_quadratic(core.UNITS_Q, 600, limits=True)
    assert r["P"] == approx([200.0, 200.0, 200.0])


def test_ex_6_7_pwl_table_consistent_with_quadratic():
    # łamana z Przykładu 6.7 przechodzi przez punkty paraboli z Przykładu 6.3
    for uq, up in zip(core.UNITS_Q, core.UNITS_PWL):
        Ps, Cs = core.pwl_points(up)
        for P, C in zip(Ps, Cs):
            assert core.q_cost(uq, P) == approx(C, abs=0.5)


@pytest.mark.parametrize("L, P, cost, lam", [
    (800, [200, 300, 300], 37650, 50.2),
    (1200, [300, 450, 450], 60580, 73.4),
    (850, [200, 350, 300], 40160, None),
    (1150, [300, 450, 400], 57190, None),
    (1160, [300, 450, 410], 57784, None),
])
def test_ex_6_8_to_6_10_pwl_dispatch(L, P, cost, lam):
    r = core.ed_pwl(core.UNITS_PWL, L)
    assert r["P"] == approx(P)
    assert r["cost"] == approx(cost)
    if lam:
        assert r["lam"] == approx(lam)


def test_ex_6_9_6_10_savings():
    C = lambda L: core.ed_pwl(core.UNITS_PWL, L)["cost"]  # noqa: E731
    assert C(800) + C(1200) == approx(98230)
    assert C(850) + C(1150) == approx(97350)
    assert C(850) + C(1200 - 0.8 * 50) < C(800) + C(1200)


def test_ex_6_11_combinations():
    rows = {mask: r for mask, r in core.combos_table(core.UNITS_PWL, 475)}
    assert rows[(0, 0, 1)]["cost"] == approx(26415)
    assert not rows[(0, 1, 0)]["feasible"]
    assert rows[(1, 0, 0)]["cost"] == approx(28650)
    assert rows[(0, 1, 1)]["cost"] == approx(24625)
    assert rows[(1, 1, 0)]["cost"] == approx(21475)
    assert rows[(1, 0, 1)]["cost"] == approx(21850)
    assert not rows[(1, 1, 1)]["feasible"]


def test_ex_6_12_combinations():
    assert 30 * 24 * math.log10(2) == approx(216.74, abs=0.01)  # 5,5e216


def _hg(up=1, dn=1, st=0.0):
    return [dict(name="Hansel", Pmin=3000, Pmax=10000, mc=20, st=st, min_down=dn),
            dict(name="Gretel", Pmin=500, Pmax=5000, mc=70, min_up=up)]


@pytest.mark.parametrize("args, cost", [((1, 1, 0), 296000), ((3, 1, 0), 346000),
                                        ((3, 2, 0), 521000), ((3, 2, 375000), 896000)])
def test_ex_6_13_unit_commitment(args, cost):
    r = core.unit_commitment(_hg(*args), [4000, 800, 5000, 3000])
    assert r["ok"]
    assert r["cost"] == approx(cost)


def test_ex_6_13_schedules():
    L = [4000, 800, 5000, 3000]
    r = core.unit_commitment(_hg(3, 1, 0), L)
    assert r["P"].ravel().tolist() == approx([3500, 0, 4500, 3000, 500, 800, 500, 0])
    r = core.unit_commitment(_hg(3, 2, 0), L)
    assert r["P"].ravel().tolist() == approx([0, 0, 4500, 3000, 4000, 800, 500, 0])


U614 = [dict(name="Alpha", Pmin=10, Pmax=20, mc=10, st=100, ramp_up=20),
        dict(name="Beta", Pmin=5, Pmax=30, mc=12, st=100, ramp_up=20),
        dict(name="Gamma", Pmin=5, Pmax=50, mc=13, st=100, ramp_up=5),
        dict(name="Delta", Pmin=10, Pmax=20, mc=20, st=0, ramp_up=20)]
L614 = [30, 40, 60, 50]
INI614 = [True, True, False, False]


def test_ex_6_14_base_case():
    r = core.unit_commitment(U614, L614, init_on=INI614)
    assert r["cost"] == approx(2080)
    assert r["headroom"].tolist() == approx([20, 10, 10, 0])


def test_ex_6_14_reserve_and_ramp_not_worse_than_book():
    # Harmonogramy z książki są dopuszczalne, a MILP jest od nich nie gorszy.
    r = core.unit_commitment(U614, L614, reserve=20, init_on=INI614)
    assert min(r["headroom"]) >= 20 - 1e-6
    assert r["cost"] <= 2125 + 1e-6
    book = [[20, 20, 20, 20], [10, 10, 30, 25], [0, 10, 10, 5], [0, 0, 0, 0]]
    assert core.schedule_cost(U614, book, INI614) == approx(2125)
    r = core.unit_commitment(U614, L614, reserve=20, ramp=True, init_on=INI614,
                             init_P=[20, 10, 0, 0])
    assert r["cost"] <= 2225 + 1e-6
    book = [[15, 20, 20, 20], [10, 10, 15, 25], [5, 10, 15, 5], [0, 0, 10, 0]]
    assert core.schedule_cost(U614, book, INI614) == approx(2225)


def test_ex_7_1_line_parameters():
    pi = core.line_pi(0.3, 0.9, 2e-6, 250, 345, 100)
    assert pi["R"] == approx(0.063, abs=5e-4)
    assert pi["X"] == approx(0.189, abs=5e-4)
    assert 1 / (pi["B"] / 2) == approx(3.361, abs=1e-3)


def test_ex_7_2_ferranti():
    o = core.line_open_end(core.line_pi(0.3, 0.9, 2e-6, 250, 345, 100))
    assert abs(o["VR"]) == approx(1.059, abs=1e-3)
    assert o["S"].real * 100 == approx(0.626, abs=1e-3)
    assert o["S"].imag * 100 == approx(-61.27, abs=0.01)


def test_ex_7_3_loaded_line():
    pi = core.line_pi(0.3, 0.9, 2e-6, 250, 345, 100)
    w = core.line_with_load(pi, 1.25, 1.25 * math.tan(math.acos(0.85)))
    assert w["S"].real * 100 == approx(136.3, abs=0.1)
    assert w["S"].imag * 100 == approx(39.6, abs=0.1)


def test_ex_7_4_ideal_transformer():
    r = core.ideal_transformer(240, 12, 200, complex(5, 2))
    assert abs(r["V2"]) == approx(10)
    assert abs(r["I2"]) == approx(1.857, abs=1e-3)
    assert r["S"].real == approx(17.24, abs=0.01)
    assert r["S"].imag == approx(6.90, abs=0.01)
    assert r["I1"] == approx(200 / r["Z2p"])


def test_ex_7_5_rated_currents():
    I1, I2 = core.transformer_currents(12e3, 13.8e3, 416)
    assert I1 == approx(0.502, abs=1e-3)
    assert I2 == approx(16.65, abs=0.01)


def test_ex_7_6_per_unit():
    r = core.per_unit_system()
    assert r["Xg"] == approx(2.44)
    assert r["Xt1"] == approx(0.1667, abs=1e-4)
    assert r["Xt2"] == approx(0.2778, abs=1e-4)
    assert abs(r["V_gen"]) == approx(1.032, abs=1e-3)
    assert math.degrees(math.atan2(r["V_gen"].imag, r["V_gen"].real)) == approx(2.11, abs=0.01)
    assert [abs(i) for i in r["I_A"]] == approx([524.87, 43.74, 174.95], abs=0.05)


@pytest.mark.parametrize("t, VL", [(1.0, 0.976), (1.05, 1.024)])
def test_ex_7_7_tap_changer(t, VL):
    r = core.tap_changer(t)
    assert abs(r["VL"]) == approx(VL, abs=1e-3)
    assert r["VL"] == approx(r["VL_pi"])
    assert r["Is"] == approx(r["Is_pi"])
