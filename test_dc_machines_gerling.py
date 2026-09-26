"""Testy rdzenia obliczeniowego dc_machines_gerling_tkinter.py
Uruchomienie:  python -m pytest test_dc_machines_gerling.py   (lub: python test_dc_machines_gerling.py)
"""
import math

import numpy as np

import dc_machines_gerling_tkinter as M


def test_esson_book_example():
    """Przykład z rozdz. 2.4.5: C = 4,28 kW·min/m³, D ≈ 0,246 m, l = τp ≈ 0,193 m."""
    E = M.esson(0.65, 500, 0.8, 100, 2000, 2, 1.0)
    assert abs(E["C_kwmin"] - 4.28) < 0.01
    assert abs(E["D"] - 0.246) < 0.002
    assert abs(E["l"] - 0.193) < 0.002
    assert abs(E["sigma"] - 26000) < 1


def test_power_balance():
    """Ui·IA = 2π·n·T  (moc wewnętrzna = moc mechaniczna)."""
    kphi, n, IA = 17.0, 25.0, 40.0
    Ui = kphi * n
    T = kphi * IA / (2 * math.pi)
    assert abs(Ui * IA - 2 * math.pi * n * T) < 1e-9


def test_bridge_mean_voltage():
    for alpha in (0, 30, 60, 90, 120, 150):
        th = np.linspace(0, 360, 36000, endpoint=False)
        ud, _ = M.bridge_ud(th, alpha, 400.0)
        ref = 3 * math.sqrt(2) / math.pi * 400 * math.cos(math.radians(alpha))
        assert abs(ud.mean() - ref) < 1.0


def test_commutation_linear_without_inductance():
    t, i, _, _ = M.commutation(50.0, 0.0, 0.02, 1e-3, 0.0)
    assert np.allclose(i, 50.0 * (1 - 2 * t / 1e-3), atol=1e-6)


def test_commutation_pole_compensation():
    """k_cp = 1 kompensuje napięcie reaktancyjne -> komutacja (prawie) liniowa."""
    t, i, _, _ = M.commutation(50.0, 25e-6, 0.02, 0.85e-3, 1.0)
    assert np.max(np.abs(i - 50.0 * (1 - 2 * t / 0.85e-3))) < 0.5


def test_windings():
    assert M.winding(2, 12, 1, 1, "lap")["a2"] == 4
    w = M.winding(2, 13, 1, 1, "wave")
    assert w["ok"] and w["y"] == 6 and w["a2"] == 2
    assert not M.winding(2, 12, 1, 1, "wave")["ok"]
    assert M.winding(3, 12, 3, 2, "lap")["z"] == 2 * 2 * 36


def test_brush_shift():
    assert abs(M.brush_shift_factor(0.0, 0.7) - 1.0) < 1e-9
    assert abs(M.brush_shift_factor(math.pi / 2, 0.7)) < 1e-6


def test_magnet_noload_point():
    (H0, B0), (_, Bon), (_, Boff) = M.magnet_points(0.4, 1.05, 5e-3, 0.5e-3, 200.0)
    assert abs(B0 - 0.4 / (1 + 1.05 * 0.1)) < 1e-12
    assert Bon > B0 > Boff
    # punkt leży na prostej obciążenia: HM·hM + BM·δ/μ0 = 0
    assert abs(H0 * 5e-3 + B0 * 0.5e-3 / M.MU0) < 1e-6


def test_shunt_self_excitation():
    ok = M.shunt_noload_point(1.0, 8.0, 150.0)
    fail = M.shunt_noload_point(1.0, 8.0, 400.0)       # powyżej rezystancji krytycznej 250 Ω
    assert 150.0 * ok > 200 and 400.0 * fail < 30          # tylko napięcie rzędu Ur


def test_start_simulation_steady_state():
    R = M.REF
    kN = M.kphi_rated(R["UN"], R["IAN"], R["RA"], R["nN"])
    TN = kN * R["IAN"] / (2 * math.pi)
    S = M.dc_start_sim(R["UN"], R["RA"], 12e-3, kN, 0.8, TN, 0.0, 3, 2 * R["IAN"], 4.0)
    assert abs(S["n"][-1] - R["nN"]) < 5              # obciążenie znamionowe -> n ≈ nN
    assert abs(S["i"][-1] - R["IAN"]) < 0.5
    assert S["i"].max() < 2.05 * R["IAN"]            # rozrusznik ogranicza prąd
    assert len(S["sw"]) == 3


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("OK ", name)
