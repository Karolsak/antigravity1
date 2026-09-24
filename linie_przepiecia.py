# -*- coding: utf-8 -*-
"""
Prądy zwarciowe w liniach przesyłowych (rozdz. 14) oraz przepięcia i ochrona
odgromowa (rozdz. 15) - interaktywna aplikacja edukacyjna (Tkinter + matplotlib).

Każda zakładka = jeden przykład z tekstu:
  * suwaki po lewej  -> zmieniasz parametry,
  * wykres po prawej -> od razu widzisz skutek,
  * tekst na dole    -> wyjaśnienie "jak dla ucznia liceum".

Uruchomienie:  python linie_przepiecia.py
Wymagania:     python 3.8+, numpy, matplotlib (tkinter jest w standardowym Pythonie)
"""
import math
import numpy as np
import matplotlib
from matplotlib.figure import Figure

# ----------------------------------------------------------------- stałe ---
MU0_2PI = 2e-7          # μ0/(2π) [H/m]  -> F = 2e-7 * I^2 * l / d
G = 9.81                # przyspieszenie ziemskie [m/s^2]

BG, PANEL, FG = "#1e1e2e", "#181825", "#cdd6f4"
CYAN, GREEN, RED, YEL, MAU, PEACH = "#89dceb", "#a6e3a1", "#f38ba8", "#f9e2af", "#cba6f7", "#fab387"

matplotlib.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": PANEL, "axes.edgecolor": "#6c7086",
    "axes.labelcolor": FG, "xtick.color": FG, "ytick.color": FG, "text.color": FG,
    "axes.grid": True, "grid.color": "#313244", "grid.linestyle": "--",
    "legend.facecolor": BG, "legend.edgecolor": "#45475a", "font.size": 9,
})


def rk4(f, y0, t_end, dt):
    """Klasyczny Runge-Kutta 4. rzędu. Zwraca (t, Y)."""
    n = int(t_end / dt) + 1
    t = np.linspace(0, t_end, n)
    Y = np.zeros((n, len(y0)))
    y = np.array(y0, float)
    Y[0] = y
    for i in range(n - 1):
        ti = t[i]
        k1 = f(ti, y)
        k2 = f(ti + dt / 2, y + dt / 2 * k1)
        k3 = f(ti + dt / 2, y + dt / 2 * k2)
        k4 = f(ti + dt, y + dt * k3)
        y = y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        Y[i + 1] = y
    return t, Y


class Example:
    """Bazowa klasa przykładu. params: (klucz, etykieta, od, do, start, jednostka)
    lub ("choice", klucz, etykieta, [opcje], start)."""
    chapter = 14
    title = ""
    params = []
    anim = None          # klucz parametru, który przycisk "Animuj" przesuwa
    explanation = ""

    def compute(self, p, fig):
        raise NotImplementedError


# =========================================================================
#                         ROZDZIAŁ 14 - LINIE
# =========================================================================
class NCI(Example):
    title = "14.2 Izolator kompozytowy"
    params = [
        ("load", "Obciążenie robocze", 10, 90, 45, "% SML"),
        ("drop", "Wytrzymałość w czasie łuku", 50, 100, 80, "% SML"),
        ("recov", "Wytrzymałość po ostygnięciu", 90, 120, 105, "% SML"),
        ("tarc", "Czas trwania łuku", 0.05, 1.0, 0.2, "s"),
    ]
    explanation = """\
CO TO JEST?  Izolator kompozytowy (NCI) to pręt z włókna szklanego (FRP) w "płaszczu" z gumy
silikonowej z talerzykami (kloszami). Na końcach ma metalowe okucia. Trzyma przewód, który wisi
na nim jak ciężarek na lince - więc pracuje głównie na ROZCIĄGANIE.

SML (Specified Mechanical Load) to "tabliczkowa" siła, którą izolator na pewno wytrzyma.

CO SIĘ DZIEJE PRZY PRZESKOKU?  Gdy piorun wywoła przeskok (łuk elektryczny) wzdłuż izolatora,
płynie przez okucia ogromny prąd zwarciowy. Okucia się grzeją i na chwilę wytrzymałość spada nawet do
ok. 80% SML. Po ostygnięciu wraca POWYŻEJ SML.

DLACZEGO TO NIE JEST GROŹNE MECHANICZNIE?  Inżynierowie obciążają izolatory najwyżej ~50% SML.
Zobacz na wykresie: zielona linia (obciążenie) jest daleko pod czerwonym "dołkiem". Zapas = różnica.
Przesuń suwak obciążenia powyżej wytrzymałości w czasie łuku - wtedy izolator by się zerwał!

CO JEST GROŹNE?  Łuk wypala cynk (ocynk) z okuć -> korozja. Nie wiemy też na pewno, czy gorące okucie
nie uszkodziło gumy lub pręta od środka. Dlatego praktyka inżynierska: izolator po przeskoku z łukiem
mocy - WYMIENIĆ, nawet gdy wygląda prawie dobrze. Tu bezpieczeństwo ludzi pod linią jest ważniejsze
niż koszt jednego izolatora."""

    def compute(self, p, fig):
        t = np.linspace(-0.5, 3, 1000)
        ta = p["tarc"]
        s = np.full_like(t, 100.0)
        dur = (t >= 0) & (t < ta)
        s[dur] = 100 - (100 - p["drop"]) * np.minimum(t[dur] / (0.3 * ta), 1)
        aft = t >= ta
        s[aft] = p["recov"] - (p["recov"] - p["drop"]) * np.exp(-(t[aft] - ta) / 0.4)
        ax = fig.add_subplot(111)
        ax.plot(t, s, color=RED, lw=2, label="wytrzymałość izolatora")
        ax.axhline(p["load"], color=GREEN, lw=2, label="obciążenie robocze")
        ax.axhline(100, color=YEL, ls=":", label="SML (100%)")
        ax.axvspan(0, ta, color=PEACH, alpha=0.15, label="łuk (zwarcie)")
        ax.fill_between(t, p["load"], s, where=s > p["load"], color=GREEN, alpha=0.08)
        ax.set_xlabel("czas [s]"); ax.set_ylabel("% SML"); ax.set_ylim(0, 130)
        ax.set_title("Wytrzymałość mechaniczna NCI podczas i po łuku zwarciowym")
        ax.legend(loc="lower right")
        margin = p["drop"] - p["load"]
        ok = "OK - izolator wytrzyma" if margin > 0 else "ZERWANIE! obciążenie > wytrzymałość"
        return (f"Najmniejszy zapas: {margin:.0f}% SML\n{ok}\n"
                f"Współczynnik bezpieczeństwa w czasie łuku: {p['drop']/p['load']:.2f}\n"
                "Zalecenie: izolator po łuku -> WYMIENIĆ")


class Force(Example):
    title = "14.3 Siła między przewodami"
    params = [
        ("I", "Prąd zwarcia (RMS)", 1, 80, 30, "kA"),
        ("d", "Odległość przewodów", 0.2, 10, 3, "m"),
        ("l", "Długość przęsła", 10, 500, 200, "m"),
        ("xr", "X/R sieci", 1, 30, 15, "-"),
    ]
    explanation = """\
WZÓR (14.1):      F = 2·10⁻⁷ · I² · ℓ / d        [N]   (I w amperach, ℓ i d w metrach)

Dwa równoległe przewody z prądem to jak dwa magnesy. Prąd w JEDNĄ stronę -> przyciągają się,
w PRZECIWNE strony -> odpychają się. Przy zwarciu międzyfazowym prądy w dwóch fazach płyną w przeciwne
strony, więc przewody się ODPYCHAJĄ, a po wyłączeniu zwarcia huśtają się z powrotem i mogą się zderzyć.

DWA WAŻNE WNIOSKI (baw się suwakami):
 1) Siła rośnie z KWADRATEM prądu: 2 razy większy prąd = 4 razy większa siła.
 2) Siła rośnie odwrotnie do odległości: przewody 2 razy bliżej = 2 razy większa siła.
    Dlatego problem dotyczy LINII KOMPAKTOWYCH (fazy blisko siebie) i wiązek przewodów EHV.

PRZYKŁAD: 30 kA, d = 3 m -> 2e-7·(30000)²/3 = 60 N na każdy metr. Metr przewodu waży ok. 15 N.
Siła magnetyczna jest 4 razy większa niż ciężar! Przewód dostaje "kopniaka" w bok.

Dolny wykres: siła nie jest stała. Prąd zmienny ma przebieg sinus, a siła ~ i², więc pulsuje z
podwójną częstotliwością (100 Hz). Na początku zwarcia jest składowa stała (asymetria) - pierwsze
szczyty siły są nawet ~3 razy większe od średniej. Im większe X/R, tym dłużej trwa ta asymetria."""

    def compute(self, p, fig):
        I = p["I"] * 1e3
        ax1 = fig.add_subplot(211)
        d = np.linspace(0.2, 10, 300)
        for k, c in zip([0.5, 1, 1.5], [GREEN, CYAN, RED]):
            ax1.plot(d, MU0_2PI * (k * I) ** 2 / d, color=c, label=f"I = {k*p['I']:.0f} kA")
        f_here = MU0_2PI * I ** 2 / p["d"]
        ax1.plot(p["d"], f_here, "o", color=YEL, ms=9)
        ax1.axhline(15, color=MAU, ls=":", label="ciężar 1 m przewodu ≈ 15 N")
        ax1.set_yscale("log"); ax1.set_xlabel("odległość d [m]"); ax1.set_ylabel("siła [N/m]")
        ax1.set_title("Siła na metr przewodu vs odległość"); ax1.legend(fontsize=8)
        ax2 = fig.add_subplot(212)
        w = 2 * np.pi * 50; tau = p["xr"] / w
        t = np.linspace(0, 0.2, 3000)
        i = math.sqrt(2) * I * (np.sin(w * t - np.pi / 2) + np.exp(-t / tau))
        f = MU0_2PI * i ** 2 / p["d"]
        ax2.plot(t * 1e3, f, color=CYAN, lw=1, label="siła chwilowa (z asymetrią)")
        ax2.axhline(f_here, color=YEL, ls="--", label="wartość średnia (I RMS)")
        ax2.set_xlabel("czas [ms]"); ax2.set_ylabel("siła [N/m]"); ax2.legend(fontsize=8)
        ax2.set_title("Siła w czasie - pulsuje 100 Hz, pierwsze szczyty największe")
        return (f"Siła średnia: {f_here:.1f} N/m\nNa całe przęsło: {f_here*p['l']/1e3:.2f} kN\n"
                f"Szczyt siły: {f.max():.0f} N/m ({f.max()/f_here:.1f}× średnia)\n"
                f"Siła / ciężar przewodu: {f_here/15:.1f}×")


def catenary_k(S, y0):
    """Parametr krzywej łańcuchowej k (= H/w) z warunku y(S/2) = y0 (bisekcja)."""
    lo, hi = S / 1e3, S * 1e5
    for _ in range(200):
        k = math.sqrt(lo * hi)
        if k * (math.cosh(S / (2 * k)) - 1) > y0:
            lo = k
        else:
            hi = k
    return k


class Catenary(Example):
    title = "14.5 Kształt przewodu"
    params = [
        ("S", "Rozpiętość przęsła S", 50, 800, 300, "m"),
        ("y0", "Zwis y0", 1, 40, 8, "m"),
        ("w", "Ciężar przewodu w", 3, 40, 15, "N/m"),
    ]
    explanation = """\
Przewód zawieszony między dwoma słupami przyjmuje kształt KRZYWEJ ŁAŃCUCHOWEJ (catenary):
        y = k·[cosh(x/k) − 1],     k = H / w    (H - naciąg poziomy, w - ciężar na metr)
Tak samo wygląda łańcuszek trzymany w dwóch palcach.

Zwis y0 = najniższy punkt przewodu poniżej punktów zawieszenia (w połowie przęsła, x = S/2 od środka).

Dla linii przesyłowych zwis jest mały w porównaniu z rozpiętością (np. 8 m na 300 m), więc prawie
idealnie pasuje prostsza PARABOLA:
        y ≈ 4·y0·x² / S²     oraz     H ≈ w·S² / (8·y0)
Na wykresie obie krzywe praktycznie się pokrywają - błąd pokazano w centymetrach (dolny wykres).
Spróbuj zwiększyć zwis do 40 m przy S = 50 m - dopiero wtedy parabola "odjeżdża".

Ważny fakt do dalszych przykładów: średnie przesunięcie całego przęsła parabolicznego to 2/3
przesunięcia środka. Dlatego model "wahadła" huśta się na ramieniu 2/3·y0 poniżej zawieszeń.
Kąt przewodu przy słupie: tg θ = sinh(S/2k) ≈ 4·y0/S."""

    def compute(self, p, fig):
        S, y0, w = p["S"], p["y0"], p["w"]
        k = catenary_k(S, y0)
        x = np.linspace(-S / 2, S / 2, 400)
        yc = k * (np.cosh(x / k) - 1) - y0
        yp = 4 * y0 * x ** 2 / S ** 2 - y0
        ax = fig.add_subplot(211)
        ax.plot(x, yc, color=CYAN, lw=3, label="krzywa łańcuchowa (dokładna)")
        ax.plot(x, yp, "--", color=PEACH, lw=1.5, label="parabola (przybliżenie)")
        ax.axhline(-y0 / 3, color=MAU, ls=":", label="środek masy (2/3·y0 pod zawieszeniem)")
        ax.plot([-S / 2, S / 2], [0, 0], "s", color=YEL, ms=10)
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.legend(fontsize=8)
        ax.set_title("Przewód między dwoma słupami")
        ax2 = fig.add_subplot(212)
        ax2.plot(x, (yc - yp) * 100, color=RED)
        ax2.set_xlabel("x [m]"); ax2.set_ylabel("różnica [cm]")
        ax2.set_title("Błąd paraboli względem krzywej łańcuchowej")
        H = w * k
        th = math.degrees(math.atan(math.sinh(S / (2 * k))))
        L = 2 * k * math.sinh(S / (2 * k))
        return (f"k = H/w = {k:.1f} m\nNaciąg poziomy H = {H/1e3:.2f} kN\n"
                f"H z paraboli = {w*S**2/(8*y0)/1e3:.2f} kN\nKąt przy słupie θ = {th:.2f}°\n"
                f"Długość przewodu L = {L:.2f} m\n(S + 8y0²/3S = {S+8*y0**2/(3*S):.2f} m)\n"
                f"Maks. błąd paraboli: {np.abs(yc-yp).max()*100:.1f} cm")


class Swing(Example):
    title = "14.4-14.6 Wychylenie (poziomo)"
    params = [
        ("I", "Prąd zwarcia 2-faz. (RMS)", 5, 60, 25, "kA"),
        ("tf", "Czas wyłączenia zwarcia", 20, 500, 100, "ms"),
        ("d0", "Odstęp faz d0", 1, 8, 3, "m"),
        ("S", "Rozpiętość S", 50, 500, 250, "m"),
        ("y0", "Zwis y0", 1, 20, 6, "m"),
        ("w", "Ciężar przewodu", 5, 30, 15, "N/m"),
        ("clr", "Wymagany odstęp izolacyjny", 0, 3, 1.0, "m"),
    ]
    explanation = """\
SCENARIUSZ: zwarcie międzyfazowe gdzieś dalej w sieci ("through-fault"). Przez NASZĄ linię płynie
duży prąd, ale NASZA linia jest zdrowa i ma pracować dalej. Siła z przykładu 14.3 rozpycha fazy.

MODEL (jak u inżyniera - prosto, ale wystarczająco dokładnie):
 * Całe przęsło huśta się w bok jak WAHADŁO. Masa m = w·S/g "wisi" na ramieniu 2/3·y0.
 * Siły: F (elektromagnetyczna, poziomo, tylko dopóki trwa zwarcie) i W = m·g (ciężar, pionowo).
 * Składowa styczna do toru:     F_tang = F·cos θ − W·sin θ
 * Przyspieszenie:               a = F_tang / m      ->   θ'' = F_tang / (m · 2/3·y0)
 * Liczymy krok po kroku w czasie (metoda Rungego-Kutty): prędkość, potem kąt, potem położenie.
 * Przesunięcie środka przęsła:  x = y0·sin θ; odstęp faz w środku: d = d0 + 2·y0·sin θ.

CO WIDAĆ?  W czasie zwarcia przewody się rozchodzą (odstęp rośnie). Po wyłączeniu przez wyłącznik
siła znika, a przewody jak huśtawka wracają, PRZELATUJĄ przez położenie spoczynkowe i zbliżają się
do siebie. Jeśli minimalny odstęp < wymaganego odstępu izolacyjnego -> przeskok, czyli wyłączenie
ZDROWEJ linii (lub zderzenie przewodów).

WAŻNE: liczy się nie tylko prąd, ale też CZAS zwarcia (energia = siła × czas). Szybszy wyłącznik
(przesuń tf w dół) = mniejsze wychylenie. Zwiększ tf z 100 do 300 ms i zobacz różnicę!
Pomijamy tłumienie (opór powietrza) - wynik jest więc "po bezpiecznej stronie"."""

    def compute(self, p, fig):
        I = p["I"] * 1e3; S, y0, d0 = p["S"], p["y0"], p["d0"]
        m = p["w"] * S / G; W = m * G; Lp = 2 / 3 * y0; tf = p["tf"] / 1e3

        def f(t, y):
            th, om = y
            F = 0.0
            if t < tf:
                davg = max(d0 + 2 * Lp * math.sin(th), 0.1)
                F = MU0_2PI * I ** 2 * S / davg
            return np.array([om, (F * math.cos(th) - W * math.sin(th)) / (m * Lp)])
        t, Y = rk4(f, [0.0, 0.0], 8.0, 2e-3)
        th = Y[:, 0]
        x = y0 * np.sin(th); dmid = d0 + 2 * x
        ax = fig.add_subplot(211)
        ax.plot(t, dmid, color=CYAN, lw=2, label="odstęp faz w środku przęsła")
        ax.axhline(d0, color=FG, ls=":", label="odstęp w spoczynku")
        ax.axhline(p["clr"], color=RED, ls="--", label="wymagany odstęp")
        ax.axvspan(0, tf, color=PEACH, alpha=0.25, label="zwarcie trwa")
        ax.set_xlabel("czas [s]"); ax.set_ylabel("odstęp [m]"); ax.legend(fontsize=8)
        ax.set_title("Ruch przewodów po zwarciu międzyfazowym")
        ax2 = fig.add_subplot(212)
        # widok z czoła: dwa przewody (lewa i prawa faza) w chwili max i min odstępu
        for idx, col, lab in [(np.argmax(dmid), GREEN, "maks. rozejście"),
                              (np.argmin(dmid), RED, "maks. zbliżenie")]:
            a = th[idx]
            for sgn in (-1, 1):
                xs = sgn * (d0 / 2 + y0 * math.sin(a)); ys = -y0 * math.cos(a)
                ax2.plot([sgn * d0 / 2, xs], [0, ys], color=col, lw=1)
                ax2.plot(xs, ys, "o", color=col, ms=10, label=lab if sgn == 1 else None)
        for sgn in (-1, 1):
            ax2.plot([sgn * d0 / 2] * 2, [0, -y0], ":", color=FG)
        ax2.set_aspect("equal"); ax2.set_title("Widok wzdłuż linii (środek przęsła)")
        ax2.set_xlabel("[m]"); ax2.legend(fontsize=8)
        mn = dmid.min()
        verdict = "PRZESKOK / ZDERZENIE!" if mn < p["clr"] else "OK - odstęp zachowany"
        return (f"Siła/ciężar = {MU0_2PI*I**2/d0/p['w']:.2f}\n"
                f"Maks. kąt: {math.degrees(th.max()):.1f}°\n"
                f"Maks. odstęp: {dmid.max():.2f} m\nMin. odstęp: {mn:.2f} m\n"
                f"Okres wahań ≈ {2*math.pi*math.sqrt(Lp/G):.1f} s\n{verdict}")


class Vertical(Example):
    title = "14.7-14.10 Układ pionowy"
    params = [
        ("I", "Prąd zwarcia (RMS)", 5, 60, 30, "kA"),
        ("tf", "Czas zwarcia", 20, 500, 150, "ms"),
        ("d0", "Odstęp pionowy d0", 1, 8, 3, "m"),
        ("S", "Rozpiętość S", 50, 500, 200, "m"),
        ("D0", "Zwis D0", 1, 15, 5, "m"),
        ("w", "Ciężar przewodu", 5, 30, 15, "N/m"),
        ("EA", "Sztywność EA", 5, 60, 30, "MN"),
    ]
    explanation = """\
Teraz fazy wiszą JEDNA NAD DRUGĄ. Siła odpychania działa w pionie:
 * GÓRNY przewód jest pchany do góry - siła magnetyczna "odejmuje" mu ciężar:  w_ef = w − f
 * DOLNY przewód jest pchany w dół - "dokłada" mu ciężar:                      w_ef = w + f
   (f = 2·10⁻⁷·I²/d_śr - siła na metr; średni odstęp d_śr = d0 + 2/3·(zmiana zwisów))

Tu nie ma już "wahadła" - przewód zmienia ZWIS. A zwis zależy od naciągu przewodu H:
      równowaga środka przęsła (parabola):   w·S = 8·H·D / S
      długość przewodu:                       L ≈ S + 8·D²/(3·S)
      rozciągnięcie przewodu (prawo Hooke'a): H − H0 = EA · (L − L0) / L0
EA to sztywność przewodu (moduł Younga × przekrój). Przewód jest jak bardzo sztywna sprężyna:
małe wydłużenie = duży wzrost naciągu. Dlatego DOLNY przewód szybko "trafia na sprężynę" (naciąg
rośnie), a GÓRNY, gdy naciąg spada prawie do zera, jest bardzo luźny i łatwo podskakuje.

Równanie ruchu (II zasada Newtona, F = m·a):  m·D'' = w_ef·S − 8·H(D)·D/S

Dolny wykres pokazuje NACIĄG - też ważny dla bezpieczeństwa: skoki naciągu obciążają izolatory,
okucia i słupy. Zmień EA i zobacz, jak rośnie szarpnięcie w dolnym przewodzie."""

    def compute(self, p, fig):
        I = p["I"] * 1e3; S, D0, w, d0 = p["S"], p["D0"], p["w"], p["d0"]
        EA = p["EA"] * 1e6; m = w * S / G; tf = p["tf"] / 1e3
        H0 = w * S ** 2 / (8 * D0); L0 = S + 8 * D0 ** 2 / (3 * S)

        def H(D):
            return max(H0 + EA * (S + 8 * D ** 2 / (3 * S) - L0) / L0, 50.0)

        def f(t, y):
            Dt, vt, Db, vb = y
            fm = 0.0
            if t < tf:
                dav = max(d0 + 2 / 3 * ((Db - D0) - (Dt - D0)), 0.1)
                fm = MU0_2PI * I ** 2 / dav
            at = ((w - fm) * S - 8 * H(Dt) * Dt / S) / m
            ab = ((w + fm) * S - 8 * H(Db) * Db / S) / m
            return np.array([vt, at, vb, ab])
        t, Y = rk4(f, [D0, 0, D0, 0], 5.0, 5e-4)
        Dt, Db = Y[:, 0], Y[:, 2]
        sep = d0 + Db - Dt
        ax = fig.add_subplot(311)
        ax.plot(t, -Dt, color=GREEN, label="środek górnego przewodu")
        ax.plot(t, -Db - d0, color=CYAN, label="środek dolnego przewodu")
        ax.axvspan(0, tf, color=PEACH, alpha=0.25)
        ax.set_ylabel("wysokość [m]"); ax.legend(fontsize=7, loc="upper right")
        ax.set_title("Położenie przewodów (0 = poziom zawieszenia górnego)")
        ax2 = fig.add_subplot(312, sharex=ax)
        ax2.plot(t, sep, color=YEL); ax2.axhline(d0, color=FG, ls=":")
        ax2.set_ylabel("odstęp [m]")
        ax3 = fig.add_subplot(313, sharex=ax)
        ax3.plot(t, [H(v) / 1e3 for v in Dt], color=GREEN, label="naciąg górny")
        ax3.plot(t, [H(v) / 1e3 for v in Db], color=CYAN, label="naciąg dolny")
        ax3.set_ylabel("H [kN]"); ax3.set_xlabel("czas [s]"); ax3.legend(fontsize=7)
        Hb = max(H(v) for v in Db)
        return (f"Naciąg początkowy H0 = {H0/1e3:.2f} kN\nMaks. naciąg dolnego = {Hb/1e3:.2f} kN "
                f"({Hb/H0:.2f}× H0)\nMin. odstęp = {sep.min():.2f} m\nMaks. odstęp = {sep.max():.2f} m\n"
                + ("ZDERZENIE!" if sep.min() <= 0.3 else "OK"))


class Spacer(Example):
    title = "14.11 Rozpórki międzyfazowe"
    params = [
        ("I", "Prąd zwarcia (RMS)", 5, 60, 30, "kA"),
        ("tf", "Czas zwarcia", 20, 400, 100, "ms"),
        ("d", "Odstęp faz", 0.5, 5, 1.5, "m"),
        ("s", "Długość podprzęsła", 10, 150, 60, "m"),
        ("T", "Naciąg przewodu", 5, 50, 20, "kN"),
        ("w", "Ciężar przewodu", 5, 30, 15, "N/m"),
    ]
    explanation = """\
W liniach kompaktowych między fazami wstawia się IZOLACYJNE ROZPÓRKI (spacers) - "patyczki"
trzymające fazy w stałej odległości. Linia dzieli się na krótsze odcinki (podprzęsła) o długości s.

Co przeżywa rozpórka?
 1) W czasie zwarcia fazy się odpychają -> rozpórka jest ROZCIĄGANA (ciągnie fazy do siebie).
 2) Po wyłączeniu zwarcia wybrzuszone odcinki przewodów wahają się z powrotem, przelatują przez
    środek -> rozpórka jest ŚCISKANA.
Podprzęsło jest krótkie i mocno napięte, więc zachowuje się jak napięta STRUNA gitary:
wybrzuszenie x w środku daje siłę powrotną  F = 8·T·x / s  (ta sama równowaga co 8·H·D/S z 14.9).
To układ masa-sprężyna:  m·x'' = F_elmag(t) − (8T/s)·x,   m = w·s/g.
Siła w rozpórce = siła, którą struna "ciągnie" rozpórkę:  F_rozpórki = 8·T·x / s.

Wniosek z tekstu i z wykresu: gdy zwarcie skończy się ZANIM przewód osiągnie maks. wychylenie
(typowy przypadek), maksymalna siła jest TAKA SAMA przy rozciąganiu i ściskaniu - wahadło bez
tłumienia huśta się symetrycznie ±θmax. Rozpórkę dobiera się na tę siłę z zapasem.
Krótsze podprzęsła (więcej rozpórek) = mniejsze siły na jedną rozpórkę."""

    def compute(self, p, fig):
        I = p["I"] * 1e3; s, w, T = p["s"], p["w"], p["T"] * 1e3
        D = w * s ** 2 / (8 * T); m = w * s / G; tf = p["tf"] / 1e3
        fem = MU0_2PI * I ** 2 * s / p["d"]
        K = 8 * T / s          # sztywność poprzeczna napiętej struny [N/m]

        def f(t, y):
            x, vx = y
            F = fem if t < tf else 0.0
            return np.array([vx, (F - K * x) / m])
        per = 2 * math.pi * math.sqrt(m / K)
        t, Y = rk4(f, [0, 0], max(3 * per, 3 * tf), min(per / 400, 1e-3))
        Fs = K * Y[:, 0] / 1e3
        ax = fig.add_subplot(111)
        ax.plot(t, Fs, color=CYAN, lw=2)
        ax.fill_between(t, 0, Fs, where=Fs > 0, color=GREEN, alpha=0.3, label="rozciąganie")
        ax.fill_between(t, 0, Fs, where=Fs < 0, color=RED, alpha=0.3, label="ściskanie")
        ax.axvspan(0, tf, color=PEACH, alpha=0.15, label="zwarcie")
        ax.set_xlabel("czas [s]"); ax.set_ylabel("siła w rozpórce [kN]"); ax.legend()
        ax.set_title("Siła w rozpórce międzyfazowej")
        return (f"Zwis podprzęsła D = {D:.2f} m\nSiła elmag. na podprzęsło = {fem/1e3:.2f} kN\n"
                f"Maks. wybrzuszenie: {Y[:,0].max()*100:.1f} cm\n"
                f"Maks. rozciąganie: {Fs.max():.2f} kN\nMaks. ściskanie: {-Fs.min():.2f} kN\n"
                f"Okres wahań: {per:.2f} s")


class Pinch(Example):
    title = "14.12 Efekt 'pinch' w wiązce"
    params = [
        ("I", "Prąd zwarcia fazy (RMS)", 5, 80, 40, "kA"),
        ("n", "Liczba przewodów w wiązce", 2, 4, 2, "-"),
        ("a", "Odstęp przewodów as", 100, 600, 400, "mm"),
        ("ds", "Średnica przewodu ds", 15, 45, 30, "mm"),
        ("ls", "Odstęp rozpórek ℓs", 20, 100, 60, "m"),
        ("m", "Masa przewodu", 0.5, 3, 1.6, "kg/m"),
    ]
    explanation = """\
Linie najwyższych napięć (EHV, ≥ 345 kV) mają w każdej fazie WIĄZKĘ 2-4 przewodów trzymanych
rozpórkami co ℓs metrów (różne od rozpórek międzyfazowych!). W jednej fazie prąd płynie w każdym
przewodzie w TĘ SAMĄ stronę -> przewody wiązki się PRZYCIĄGAJĄ (zob. 14.3).

Przy zwarciu przewody "zapadają się" do siebie - dotykają się pośrodku między rozpórkami. To
zjawisko nazywa się PINCH (ściśnięcie). Rozpórka jest wtedy mocno ŚCISKANA, a naciąg przewodów
chwilowo wzrasta.

Stary wzór Manuzio dawał siłę ściskającą rozpórkę, ale testy (Lilien i in.) pokazały, że ZANIŻAŁ
wynik aż o ~50%, bo pomijał: (1) wzrost naciągu przy pinch, (2) asymetrię prądu zwarciowego,
(3) długość podprzęsła ℓs. Dziś stosuje się metodę z normy IEC 60865-1 (przewód w kształcie
paraboli między rozpórką a punktem styku, równania rozwiązywane numerycznie) lub metodę elementów
skończonych (MES).

Ta zakładka pokazuje UPROSZCZONĄ fizykę zjawiska (nie zastępuje IEC 60865!):
 * siła na metr między przewodami wiązki:  f = 2·10⁻⁷·(I/n)²·(n−1)/as   (przybliżenie)
 * przyspieszenie a = f/m i czas do zetknięcia t = √(2·(as−ds)/a)
Jeśli przewody zetkną się SZYBCIEJ niż trwa zwarcie (typowo!), dochodzi do pinch."""

    def compute(self, p, fig):
        I = p["I"] * 1e3; n = int(round(p["n"])); a = p["a"] / 1e3; ds = p["ds"] / 1e3
        ax = fig.add_subplot(121)
        aa = np.linspace(0.1, 0.6, 200)
        for In, c in [(0.5 * I, GREEN), (I, CYAN), (1.5 * I, RED)]:
            ax.plot(aa * 1e3, MU0_2PI * (In / n) ** 2 * (n - 1) / aa, color=c,
                    label=f"{In/1e3:.0f} kA")
        f = MU0_2PI * (I / n) ** 2 * (n - 1) / a
        ax.plot(a * 1e3, f, "o", color=YEL, ms=9)
        ax.set_xlabel("odstęp as [mm]"); ax.set_ylabel("siła przyciągania [N/m]")
        ax.set_title("Siła między przewodami wiązki"); ax.legend(fontsize=8)
        acc = f / p["m"]; tc = math.sqrt(2 * (a - ds) / acc)
        # rysunek wiązki w przekroju: przed i po pinch
        ax2 = fig.add_subplot(122)
        R = a / (2 * math.sin(math.pi / n)) if n > 1 else 0
        for k in range(n):
            ang = 2 * math.pi * k / n + math.pi / n
            x, y = R * math.cos(ang), R * math.sin(ang)
            ax2.add_patch(matplotlib.patches.Circle((x, y), ds / 2, color=CYAN, alpha=0.6))
            r2 = ds / (2 * math.sin(math.pi / n)) if n > 2 else ds / 2
            ax2.add_patch(matplotlib.patches.Circle((r2 * math.cos(ang), r2 * math.sin(ang) - 1.4 * R - 0.05),
                                                    ds / 2, color=RED, alpha=0.8))
        ax2.set_xlim(-0.4, 0.4); ax2.set_ylim(-1.4 * R - 0.25, R + 0.15); ax2.set_aspect("equal")
        ax2.set_title("przekrój: niebieski = przed, czerwony = pinch", fontsize=8)
        tot = f * p["ls"]
        return (f"Siła przyciągania: {f:.0f} N/m\nNa podprzęsło ℓs: {tot/1e3:.1f} kN\n"
                f"Przyspieszenie: {acc/G:.1f} g\nCzas do zetknięcia: {tc*1e3:.0f} ms\n"
                "(Manuzio zaniża ~50% -> użyj IEC 60865-1 lub MES)")


# =========================================================================
#                       ROZDZIAŁ 15 - PRZEPIĘCIA
# =========================================================================
class Impulse(Example):
    chapter = 15
    title = "15.1 Udary probiercze"
    params = [
        ("choice", "typ", "Rodzaj udaru",
         ["piorunowy 1,2/50 μs", "ucięty (chopped)", "łączeniowy 250/2500 μs"], "piorunowy 1,2/50 μs"),
        ("BIL", "BIL urządzenia", 45, 1300, 550, "kV"),
        ("tc", "Chwila ucięcia", 1.5, 6, 3, "μs"),
    ]
    explanation = """\
Przepięcia (surges) to bardzo krótkie, bardzo wysokie impulsy napięcia. Źródła:
 * PIORUN: trafienie w przewód fazowy (awaria osłony), w przewód odgromowy/słup -> "przeskok
   odwrotny" (słup ma wyższe napięcie niż przewód!), uderzenie w ziemię obok -> przepięcie indukowane.
 * ŁĄCZENIA: wyłączniki, bezpieczniki, tyrystory. Każda gwałtowna zmiana prądu daje napięcie L·di/dt.

Żeby porównywać urządzenia, normy definiują STANDARDOWE kształty udaru (funkcja impulsowa 15.1):
        v(t) = V0 · (e^(−α1·t) − e^(−α2·t))
 * 1,2/50 μs: narasta do szczytu w 1,2 μs, opada do połowy po 50 μs  (α1=1,47·10⁴, α2=2,47·10⁶ 1/s)
   Szczyt = BIL (Basic Impulse Level) - poziom, który izolacja MUSI wytrzymać. Udaje piorun.
 * UCIĘTY: ten sam kształt, ale obcięty (np. iskiernik) po kilku μs; szczyt 110-115% BIL.
   Nagłe ucięcie najbardziej męczy izolację MIĘDZYZWOJOWĄ w transformatorze.
 * ŁĄCZENIOWY 250/2500 μs (α1=3,17·10², α2=1,60·10⁴ 1/s): wolniejszy, udaje przepięcia od łączeń.

Uwaga na skalę: 1 μs to milionowa część sekundy! Cały udar piorunowy trwa krócej niż
tysięczna część okresu napięcia sieci 50 Hz (20 ms)."""

    def compute(self, p, fig):
        typ = p["typ"]; B = p["BIL"]
        if typ.startswith("łącz"):
            a1, a2, T = 3.17e2, 1.60e4, 5000e-6
        else:
            a1, a2, T = 1.47e4, 2.47e6, 70e-6 if typ.startswith("pior") else 6e-6
        t = np.linspace(0, T, 5000)
        v = np.exp(-a1 * t) - np.exp(-a2 * t)
        v /= v.max()
        crest = B
        if typ.startswith("ucięty"):
            crest = 1.1 * B
            tc = p["tc"] * 1e-6
            v = np.where(t < tc, v, np.maximum(v[np.searchsorted(t, tc)] * (1 - (t - tc) / 0.2e-6), 0))
        ax = fig.add_subplot(111)
        ax.plot(t * 1e6, v * crest, color=CYAN, lw=2)
        ax.axhline(crest, color=YEL, ls=":", label=f"szczyt {crest:.0f} kV")
        ax.axhline(0.5 * crest, color=MAU, ls=":", label="50% szczytu")
        ipk = np.argmax(v)
        ax.plot(t[ipk] * 1e6, crest, "o", color=RED)
        ax.set_xlabel("czas [μs]"); ax.set_ylabel("napięcie [kV]"); ax.legend()
        ax.set_title(f"Udar: {typ}")
        half = t[ipk:][np.argmax(v[ipk:] < 0.5)] * 1e6 if (v[ipk:] < 0.5).any() else float("nan")
        t30 = t[np.argmax(v >= 0.3)]; t90 = t[np.argmax(v >= 0.9)]
        return (f"T1 (norma: 1,67·(t90−t30)): {1.67*(t90-t30)*1e6:.2f} μs\n"
                f"Rzeczywisty szczyt po: {t[ipk]*1e6:.2f} μs\nSpadek do 50%: {half:.1f} μs\n"
                f"Wartość szczytowa: {crest:.0f} kV")


class CLF(Example):
    chapter = 15
    title = "15.1 Bezpiecznik ograniczający"
    params = [
        ("V", "Napięcie źródła (szczyt)", 200, 800, 400, "V"),
        ("Ip", "Spodziewany prąd zwarcia (szczyt)", 5, 60, 25, "kA"),
        ("xr", "X/R", 1, 20, 5, "-"),
        ("i2t", "Całka topienia I²t", 0.01, 2, 0.15, "kA²s"),
        ("tau", "Szybkość narastania oporu łuku", 0.05, 1, 0.25, "ms"),
    ]
    explanation = """\
Bezpiecznik ograniczający prąd (CLF - current-limiting fuse): srebrne paski w piasku kwarcowym.
Przy zwarciu paski topią się w ułamku milisekundy, łuk topi piasek, powstaje "szkło" -> opór
łuku gwałtownie rośnie i prąd zostaje zduszony, ZANIM osiągnie wartość spodziewaną.

 * Prąd spodziewany (prospective) - co płynęłoby bez bezpiecznika (przerywana linia).
 * Prąd przepuszczony (let-through) - co faktycznie przepuszcza bezpiecznik (dużo mniej!).

CENA za to: szybki spadek prądu => duże di/dt => indukcyjność sieci "broni się" napięciem
v = L·di/dt. Na bezpieczniku pojawia się SZPILKA NAPIĘCIA wyższa niż napięcie sieci (górny wykres).
To jest właśnie przepięcie łączeniowe.

MODEL (rys. 15.4): źródło + R + L sieci, bezpiecznik = rezystor zmienny w czasie:
 * dopóki ∫i²dt < I²t topienia - opór ≈ 0,
 * potem R_łuku rośnie wykładniczo (stała τ). Szybszy wzrost (mniejsze τ) = mniejszy prąd,
   ale WYŻSZE przepięcie - klasyczny kompromis inżynierski. Sprawdź suwakiem τ!"""

    def compute(self, p, fig):
        w = 2 * np.pi * 50; Vm = p["V"]; Zs = Vm / (p["Ip"] * 1e3)
        R = Zs / math.sqrt(1 + p["xr"] ** 2); L = R * p["xr"] / w
        phi = math.atan2(w * L, R)
        dt = 2e-6; t = np.arange(0, 0.016, dt)
        vs = Vm * np.sin(w * t)
        ip = Vm / Zs * (np.sin(w * t - phi) + math.sin(phi) * np.exp(-t * R / L))
        i = 0.0; I2t = 0.0; tm = None; iv = np.zeros_like(t); vf = np.zeros_like(t)
        tau = p["tau"] * 1e-3; lim = p["i2t"] * 1e6
        for k in range(len(t)):
            if tm is None and I2t >= lim:
                tm = t[k]
            Rf = 0.0 if tm is None else min(1e-3 * math.exp((t[k] - tm) / tau), 1e4)
            Rt = R + Rf
            iss = vs[k] / Rt
            i = iss + (i - iss) * math.exp(-Rt * dt / L)
            I2t += i * i * dt
            iv[k] = i; vf[k] = i * Rf
        ax = fig.add_subplot(211)
        ax.plot(t * 1e3, vs, color=GREEN, label="napięcie źródła")
        ax.plot(t * 1e3, vf, color=RED, label="napięcie na bezpieczniku")
        ax.set_ylabel("[V]"); ax.legend(fontsize=8); ax.set_title("Przepięcie przy zadziałaniu CLF")
        ax2 = fig.add_subplot(212, sharex=ax)
        ax2.plot(t * 1e3, ip / 1e3, "--", color=FG, label="prąd spodziewany")
        ax2.plot(t * 1e3, iv / 1e3, color=CYAN, lw=2, label="prąd ograniczony")
        ax2.set_xlabel("czas [ms]"); ax2.set_ylabel("[kA]"); ax2.legend(fontsize=8)
        return (f"Szczyt spodziewany: {np.abs(ip).max()/1e3:.1f} kA\n"
                f"Prąd przepuszczony: {np.abs(iv).max()/1e3:.1f} kA\n"
                f"Przepięcie: {vf.max():.0f} V = {vf.max()/Vm:.2f} p.u.\n"
                f"Topienie po: {tm*1e3 if tm else float('nan'):.2f} ms")


class Restrike(Example):
    chapter = 15
    title = "15.1 Ponowny zapłon (kondensator)"
    params = [
        ("Vm", "Napięcie szczytowe", 5, 30, 10, "kV"),
        ("vw0", "Wytrzymałość przerwy tuż po zgaszeniu", 0, 20, 14, "kV"),
        ("rr", "Szybkość odbudowy wytrzymałości", 0.1, 5, 0.6, "kV/ms"),
        ("f0", "Częstotliwość drgań własnych", 1, 10, 3, "kHz"),
        ("zeta", "Tłumienie na pół okresu", 0, 0.5, 0.05, "-"),
    ]
    explanation = """\
Wyłączamy baterię kondensatorów. Prąd kondensatora wyprzedza napięcie o 90°, więc wyłącznik gasi
łuk w zerze prądu = gdy napięcie jest w SZCZYCIE. Kondensator zostaje naładowany do −Vm i trzyma
to napięcie (jak bateria). Tymczasem napięcie sieci dalej się zmienia (sinusoida) - pół okresu
później jest +Vm. Między stykami wyłącznika jest więc aż 2·Vm!

Jeśli styki rozchodzą się zbyt wolno (mała "odbudowa wytrzymałości"), przerwa przebija się
ponownie: RESTRIKE. Powstaje obwód LC - napięcie kondensatora oscyluje z dużą częstotliwością
wokół napięcia sieci i "przestrzeliwuje": nowe napięcie ≈ 2·Vs − Vc = 2·(+Vm) − (−Vm) = 3·Vm!
Po kolejnym restrike może być jeszcze gorzej (teoretycznie 5 p.u.).

Na wykresie: zielony - napięcie sieci, niebieski - napięcie kondensatora, żółty - wytrzymałość
przerwy między stykami. Zwiększ szybkość odbudowy - restrike znika. Dlatego do baterii
kondensatorów stosuje się wyłączniki "restrike-free" (np. próżniowe, SF6 klasy C2)."""

    def compute(self, p, fig):
        f = 50; w = 2 * np.pi * f; Vm = p["Vm"]
        dt = 5e-6; t = np.arange(0, 0.05, dt)
        vs = -Vm * np.cos(w * t)             # szczyt ujemny w t=0 -> przerwanie
        vc = np.empty_like(t); vw = np.zeros_like(t)
        Vc = -Vm; tr = None; w0 = 2 * np.pi * p["f0"] * 1e3; T2 = np.pi / w0
        rst = []; osc = None
        for k, tk in enumerate(t):
            vw[k] = p["vw0"] + p["rr"] * tk * 1e3
            if osc is not None:
                t0, Vs0, V0 = osc
                if tk - t0 < T2:
                    vc[k] = Vs0 + (V0 - Vs0) * math.cos(w0 * (tk - t0)) * math.exp(-p["zeta"] * (tk - t0) / T2)
                    continue
                Vc = Vs0 - (V0 - Vs0) * math.exp(-p["zeta"]); osc = None
            if abs(vs[k] - Vc) > vw[k] and tk > 1e-4 and len(rst) < 4:
                rst.append(tk); osc = (tk, vs[k], Vc); vc[k] = Vc; continue
            vc[k] = Vc
        ax = fig.add_subplot(111)
        ax.plot(t * 1e3, vs, color=GREEN, label="napięcie sieci")
        ax.plot(t * 1e3, vc, color=CYAN, lw=1.5, label="napięcie kondensatora")
        ax.plot(t * 1e3, vw, ":", color=YEL, label="wytrzymałość przerwy |Vs−Vc|")
        for r in rst:
            ax.axvline(r * 1e3, color=RED, alpha=0.4)
        ax.set_ylim(-5.5 * Vm, 5.5 * Vm)
        ax.set_xlabel("czas [ms]"); ax.set_ylabel("[kV]"); ax.legend(fontsize=8)
        ax.set_title("Wyłączanie baterii kondensatorów z ponownymi zapłonami (czerwone linie)")
        return (f"Liczba restrike: {len(rst)}\nMaks. |Vc| = {np.abs(vc).max():.1f} kV\n"
                f"= {np.abs(vc).max()/Vm:.2f} p.u.")


class Travel(Example):
    chapter = 15
    title = "15.2 Fala wędrowna i odbicia"
    params = [
        ("Z1", "Impedancja falowa linii 1", 20, 1000, 400, "Ω"),
        ("Z2", "Impedancja falowa linii 2", 10, 5000, 40, "Ω"),
        ("v1", "Prędkość fali w linii 1", 100, 300, 300, "m/μs"),
        ("v2", "Prędkość fali w linii 2", 100, 300, 150, "m/μs"),
        ("tf", "Czas czoła fali", 0.2, 5, 1.2, "μs"),
        ("t", "Chwila t", 0, 25, 8, "μs"),
    ]
    anim = "t"
    explanation = """\
Dla przepięć linia NIE jest "zwykłym drutem". To łańcuch małych cewek ΔL i kondensatorów ΔC
(rys. 15.9). Energia przelewa się między nimi, a impuls BIEGNIE po linii jak fala po linie.
      prędkość:              v  = 1/√(L·C)      (linia napowietrzna ≈ 300 m/μs = prędkość światła,
                                                  kabel ≈ 150 m/μs)
      impedancja falowa:     Z0 = √(L/C)        (linia ≈ 400 Ω, kabel ≈ 40 Ω) - to NIE jest opór
                                                  ciepła, tylko stosunek napięcia do prądu fali.
Na złączu dwóch linii o różnych Z0 fala częściowo się ODBIJA, a częściowo PRZECHODZI (jak światło
na granicy powietrze-szkło):
      odbita:     Γ = (Z2 − Z1)/(Z2 + Z1)        przepuszczona:   τ = 2·Z2/(Z2 + Z1)  (zawsze > 0)
PRZYKŁADY (ustaw suwaki):
 * linia 400 Ω -> kabel 40 Ω: Γ ujemne, napięcie w kablu MALEJE (kabel "łagodzi" udar).
 * linia -> duża Z2 (koniec otwarty, transformator bez obciążenia): τ -> 2. Napięcie się PODWAJA!
   To dlatego na końcu linii przepięcie jest najgroźniejsze. Nigdy jednak nie przekracza 2×.
Przycisk "Animuj" puszcza falę. Dolny wykres: prąd - odbity prąd ma ZNAK PRZECIWNY do odbitego napięcia."""

    def compute(self, p, fig):
        Z1, Z2, v1, v2, tf, tt = p["Z1"], p["Z2"], p["v1"], p["v2"], p["tf"], p["t"]
        L1 = 3000.0; L2 = 3000.0
        G_ = (Z2 - Z1) / (Z2 + Z1); T_ = 2 * Z2 / (Z2 + Z1)

        def shape(s):
            return np.clip(s / tf, 0, 1) * np.where(s > 0, 1, 0)
        x1 = np.linspace(-L1, 0, 600); x2 = np.linspace(0, L2, 600)
        inc = shape(tt - (x1 + L1) / v1)
        ref = G_ * shape(tt - (L1 - x1) / v1)
        tra = T_ * shape(tt - L1 / v1 - x2 / v2)
        ax = fig.add_subplot(211)
        ax.plot(x1, inc + ref, color=CYAN, lw=2.5, label="napięcie wypadkowe")
        ax.plot(x1, inc, ":", color=GREEN, label="fala padająca")
        ax.plot(x1, ref, ":", color=RED, label=f"fala odbita (Γ={G_:+.2f})")
        ax.plot(x2, tra, color=YEL, lw=2.5, label=f"fala przepuszczona (τ={T_:.2f})")
        ax.axvline(0, color=MAU, lw=3, alpha=0.6)
        ax.set_ylim(-1.1, 2.1); ax.set_ylabel("napięcie [p.u.]"); ax.legend(fontsize=7, loc="upper left")
        ax.set_title(f"t = {tt:.1f} μs    |  złącze: Z1 = {Z1:.0f} Ω  ->  Z2 = {Z2:.0f} Ω")
        ax2 = fig.add_subplot(212, sharex=ax)
        ax2.plot(x1, (inc - ref) / Z1 * 1e3, color=CYAN, lw=2, label="prąd, linia 1")
        ax2.plot(x2, tra / Z2 * 1e3, color=YEL, lw=2, label="prąd, linia 2")
        ax2.axvline(0, color=MAU, lw=3, alpha=0.6)
        ax2.set_xlabel("położenie x [m]"); ax2.set_ylabel("prąd [A na 1 kV fali]"); ax2.legend(fontsize=7)
        Lh = Z1 / v1; C = 1 / (Z1 * v1)
        return (f"Γ (odbicie) = {G_:+.3f}\nτ (przejście) = {T_:.3f}\n"
                f"Linia 1: L = {Lh:.2f} μH/m\n         C = {C*1e6:.2f} pF/m\n"
                "(bo Z=√(L/C), v=1/√(LC))")


class Separation(Example):
    chapter = 15
    title = "15.2 Odległość ogranicznika"
    params = [
        ("Va", "Poziom ochrony ogranicznika LPL", 50, 1000, 300, "kV"),
        ("S", "Stromość czoła fali", 100, 3000, 1000, "kV/μs"),
        ("v", "Prędkość fali", 100, 300, 300, "m/μs"),
        ("BIL", "BIL chronionego urządzenia", 95, 1300, 550, "kV"),
        ("D", "Odległość ogranicznik–urządzenie", 0, 60, 15, "m"),
    ]
    explanation = """\
Ogranicznik przepięć "obcina" napięcie w miejscu, w którym jest zamontowany. Ale urządzenie (np.
transformator) jest kawałek dalej - ΔD metrów. Fala dobiega do transformatora, odbija się od niego
(transformator dla fali to prawie "koniec otwarty", Γ ≈ +1) i wraca do ogranicznika. Zanim
ogranicznik "zobaczy" to odbicie i zareaguje, na transformatorze napięcie zdąży urosnąć ponad LPL:
        V_urz ≈ V_ogr + 2 · S · ΔD / v        (ale najwyżej 2·V_ogr)
gdzie S - stromość czoła [kV/μs], ΔD - odległość, v - prędkość fali.

Z wzoru widać to, co mówi tekst - przyrost napięcia jest WIĘKSZY gdy:
 1) czoło jest bardziej strome (duże S),   2) odległość ΔD jest większa,
 3) fala jest wolniejsza (małe v, np. kabel),   4) impedancja zakończenia jest większa.
ZASADA PRAKTYCZNA: montuj ogranicznik JAK NAJBLIŻEJ chronionego urządzenia (i krótkimi
przewodami do uziemienia!). Wykres pokazuje, gdzie kończy się bezpieczna strefa (zapas 20% do BIL)."""

    def compute(self, p, fig):
        Va, S, v, B = p["Va"], p["S"], p["v"], p["BIL"]
        D = np.linspace(0, 60, 300)
        V = np.minimum(Va + 2 * S * D / v, 2 * Va)
        lim = B / 1.2
        ax = fig.add_subplot(111)
        ax.plot(D, V, color=CYAN, lw=2, label="napięcie na urządzeniu")
        ax.axhline(B, color=RED, ls="--", label="BIL")
        ax.axhline(lim, color=YEL, ls=":", label="BIL/1,2 (zapas 20%)")
        ax.axhline(Va, color=GREEN, ls=":", label="LPL ogranicznika")
        Vh = min(Va + 2 * S * p["D"] / v, 2 * Va)
        ax.plot(p["D"], Vh, "o", color=MAU, ms=10)
        ax.fill_between(D, 0, V, where=V <= lim, color=GREEN, alpha=0.1)
        ax.set_xlabel("odległość ΔD [m]"); ax.set_ylabel("[kV]"); ax.legend(fontsize=8)
        ax.set_title("Wpływ odległości ogranicznika od urządzenia")
        Dmax = max((lim - Va) * v / (2 * S), 0)
        PM = (B / Vh - 1) * 100
        return (f"Napięcie na urządzeniu: {Vh:.0f} kV\nMargines: {PM:.0f}% "
                + ("OK" if PM >= 20 else "ZA MAŁY!") + f"\nMaks. odległość (20%): {Dmax:.1f} m")


class Arrester(Example):
    chapter = 15
    title = "15.4 Charakterystyka ogranicznika"
    params = [
        ("Ur", "Napięcie odniesienia (przy 1 mA)", 5, 300, 100, "kV"),
        ("aZ", "Wykładnik nieliniowości ZnO α", 15, 50, 30, "-"),
        ("aS", "Wykładnik nieliniowości SiC α", 2, 8, 5, "-"),
        ("Is", "Prąd udaru", 0.1, 40, 10, "kA"),
    ]
    explanation = """\
Ogranicznik przepięć to rezystor NIELINIOWY:      I = I_ref · (U/U_ref)^α
 * Przy normalnym napięciu sieci płynie znikomy prąd (mikroampery) - ogranicznik "śpi".
 * Gdy napięcie choć trochę wzrośnie, prąd rośnie OGROMNIE (α = 30: +10% napięcia -> 17× więcej prądu!)
   Ogranicznik odprowadza wtedy energię udaru do ziemi, a napięcie prawie nie rośnie.
Idealny ogranicznik to "zawór": zamknięty do pewnego napięcia, całkiem otwarty powyżej.

HISTORIA: węglik krzemu (SiC, α ≈ 5) - za mało nieliniowy; przy napięciu sieci płynąłby za
duży prąd, więc potrzebował szeregowego ISKIERNIKA (który miał kaprysy). Dzisiejsze tlenek cynku
(ZnO, α ≈ 25-50) - tak nieliniowy, że działa BEZ iskiernika ("gapless").
Zwróć uwagę na wykres (skala logarytmiczna prądu): przy prądzie udaru 10 kA napięcie
ogranicznika ZnO rośnie ledwie o kilkadziesiąt %.

Klasy ograniczników (od najtańszej): rozdzielcze (distribution) -> pośrednie (intermediate) ->
stacyjne (station, najbardziej wytrzymałe, dla urządzeń ≥ 7,5 MVA i ważnych maszyn)."""

    def compute(self, p, fig):
        Ur = p["Ur"]; Iref = 1e-3
        I = np.logspace(-6, 5, 400)
        ax = fig.add_subplot(111)
        for a, c, lab in [(p["aZ"], CYAN, "ZnO (bez iskiernika)"), (p["aS"], PEACH, "SiC")]:
            ax.semilogx(I, Ur * (I / Iref) ** (1 / a), color=c, lw=2, label=f"{lab}, α={a:.0f}")
        ax.semilogx(I, Ur * I / Iref, ":", color=FG, label="zwykły rezystor (α=1)")
        ax.axhline(0.8 * Ur, color=GREEN, ls="--", label="ok. MCOV (praca ciągła)")
        ax.axvline(p["Is"] * 1e3, color=RED, ls=":", label="prąd udaru")
        ax.set_ylim(0, 3 * Ur); ax.set_xlabel("prąd [A] (skala log)"); ax.set_ylabel("napięcie [kV]")
        ax.legend(fontsize=8); ax.set_title("Charakterystyka U-I ogranicznika")
        Is = p["Is"] * 1e3
        Vz = Ur * (Is / Iref) ** (1 / p["aZ"]); Vs = Ur * (Is / Iref) ** (1 / p["aS"])
        i_mcov = Iref * 0.8 ** p["aZ"]
        return (f"Przy {p['Is']:.1f} kA:\n ZnO: {Vz:.0f} kV ({Vz/Ur:.2f}×Uref)\n"
                f" SiC: {Vs:.0f} kV ({Vs/Ur:.2f}×Uref)\nPrąd ZnO przy 0,8·Uref: {i_mcov*1e6:.3f} μA")


BIL_TABLES = {
    "15.1 Wyłączniki / rozdzielnice": ([2.4, 4.16, 7.2, 13.8, 14.4, 23, 34.5, 46, 69, 92, 115, 138, 161, 230, 345],
                                      [45, 60, 75, 95, 110, 150, 200, 250, 350, 450, 550, 650, 750, 900, 1300]),
    "15.2 Transformatory olejowe mocy": ([1.2, 2.5, 5, 8.7, 15, 25, 34.5, 46, 69, 92, 115, 161],
                                        [45, 60, 75, 95, 110, 150, 200, 250, 350, 450, 550, 750]),
    "15.3 Transformatory olejowe rozdz.": ([1.2, 2.5, 5, 8.7, 15], [30, 45, 60, 75, 95]),
    "15.4 Transformatory suche (Δ/Y izol.)": ([1.2, 2.52, 7.2, 8.32, 13.8, 18, 23, 27.6, 34.5],
                                             [10, 20, 30, 45, 60, 95, 110, 125, 150]),
    "15.5 Transformatory suche (Y uziem.)": ([1.2, 4.36, 8.72, 13.8, 22.86, 24.94, 34.5],
                                            [10, 20, 30, 60, 95, 110, 125]),
}


class BILTab(Example):
    chapter = 15
    title = "15.3 Tabele BIL"
    params = [("choice", "a", "Tabela A", list(BIL_TABLES), "15.2 Transformatory olejowe mocy"),
              ("choice", "b", "Tabela B", list(BIL_TABLES), "15.4 Transformatory suche (Δ/Y izol.)"),
              ("U", "Napięcie znamionowe", 1, 345, 15, "kV")]
    explanation = """\
BIL (Basic Impulse insulation Level) - wartość szczytowa udaru 1,2/50 μs, którą izolacja MUSI
wytrzymać podczas próby. Im wyższe napięcie znamionowe, tym wyższy BIL.

Porównaj tabele: TRANSFORMATORY SUCHE mają wyraźnie niższy BIL niż olejowe przy tym samym
napięciu (np. 15 kV: olejowy 95-110 kV, suchy 60 kV). Tekst podkreśla, że to było przyczyną wielu
awarii transformatorów suchych bez porządnej ochrony przepięciowej -> suchy transformator
ZAWSZE wymaga ogranicznika przepięć.

Inne próby izolacji (od najłagodniejszej):
 * próba napięciem 50/60 Hz przez 1 minutę (hipot),
 * udar łączeniowy (poziom niższy od BIL),
 * pełny udar piorunowy = BIL (najbardziej męczy izolację faza-ziemia),
 * udar ucięty: 10-15% ponad BIL (izolacja międzyzwojowa),
 * udar ucięty na czole (front of wave): jeszcze wyżej.
Maszyny wirujące (silniki, generatory) nie mają BIL - ich izolacja jest słabsza (tab. 15.6-15.7:
udar tylko ok. 4,5-9 × szczyt napięcia fazowego), dlatego chroni się je kondensatorami
spłaszczającymi czoło fali + ogranicznikami."""

    def compute(self, p, fig):
        ax = fig.add_subplot(111)
        out = []
        for key, c in [("a", CYAN), ("b", PEACH)]:
            U, B = BIL_TABLES[p[key]]
            ax.plot(U, B, "o-", color=c, lw=2, label=p[key])
            Bi = np.interp(p["U"], U, B, left=np.nan, right=np.nan)
            out.append(f"{p[key][:4]}: BIL({p['U']:.0f} kV) ≈ {Bi:.0f} kV")
        ax.axvline(p["U"], color=YEL, ls=":")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("napięcie znamionowe [kV]"); ax.set_ylabel("BIL [kV]"); ax.legend(fontsize=8)
        ax.set_title("Poziomy BIL wg IEEE (skala log-log)")
        return "\n".join(out) + "\n(interpolacja liniowa)"


class Coord(Example):
    chapter = 15
    title = "15.5 Koordynacja izolacji"
    params = [
        ("U", "Napięcie sieci (międzyfazowe)", 4, 400, 115, "kV"),
        ("kg", "Współczynnik uziemienia", 0.7, 1.0, 0.8, "-"),
        ("BIL", "BIL urządzenia", 45, 1300, 550, "kV"),
        ("CWW", "Wytrzymałość na udar ucięty CWW", 50, 1500, 630, "kV"),
        ("SSL", "Poziom udaru łączeniowego SSL", 30, 1100, 460, "kV"),
        ("LPL", "Poziom ochrony piorun. LPL", 20, 1000, 350, "kV"),
        ("FOW", "Poziom ochrony czoła FOW", 20, 1100, 400, "kV"),
        ("SPL", "Poziom ochrony łączen. SPL", 20, 900, 300, "kV"),
    ]
    explanation = """\
KOORDYNACJA IZOLACJI = dobór ogranicznika tak, by ZAWSZE "zadziałał przed" izolacją urządzenia,
z zapasem (marginesem ochrony PM):
      PM = (wytrzymałość izolacji / napięcie przepuszczone przez ogranicznik − 1) · 100%
Zalecane marginesy:
      PM1 = CWW / FOW − 1  ≥ 20%    (udar ucięty vs ochrona dla stromego czoła)
      PM2 = BIL / LPL − 1  ≥ 20%    (udar piorunowy)
      PM3 = SSL / SPL − 1  ≥ 15%    (udar łączeniowy)
W sieciach rozdzielczych zwykle sprawdza się tylko PM1 i PM2 (łączeniowe są tam łagodniejsze).

DOBÓR NAPIĘCIA OGRANICZNIKA (żeby się nie przegrzał przy normalnej pracy):
 * MCOV (maks. napięcie trwałe) ≥ maks. napięcie fazowe = 1,05·U/√3
 * TOV (napięcie przejściowe) ≥ napięcie zdrowej fazy przy zwarciu 1-fazowym = k_uz · 1,05·U
   (k_uz - współczynnik uziemienia: 0,8 sieć skutecznie uziemiona, 1,0 izolowana)
Trudność: ogranicznik "za nisko" -> przegrzeje się od napięcia sieci; "za wysoko" -> za mały margines.
Pobaw się: zwiększ LPL i zobacz, kiedy słupek PM2 robi się czerwony."""

    def compute(self, p, fig):
        pm = [(p["CWW"] / p["FOW"] - 1) * 100, (p["BIL"] / p["LPL"] - 1) * 100,
              (p["SSL"] / p["SPL"] - 1) * 100]
        req = [20, 20, 15]
        names = ["PM1\nCWW/FOW", "PM2\nBIL/LPL", "PM3\nSSL/SPL"]
        ax = fig.add_subplot(121)
        cols = [GREEN if a >= r else RED for a, r in zip(pm, req)]
        ax.bar(names, pm, color=cols)
        for i, r in enumerate(req):
            ax.plot([i - 0.4, i + 0.4], [r, r], color=YEL, lw=2)
        ax.set_ylabel("margines [%]"); ax.set_title("Marginesy ochrony (żółte = wymagane)")
        ax2 = fig.add_subplot(122)
        lv = [("LPL", p["LPL"], CYAN), ("BIL", p["BIL"], RED), ("FOW", p["FOW"], CYAN),
              ("CWW", p["CWW"], RED), ("SPL", p["SPL"], CYAN), ("SSL", p["SSL"], RED)]
        ax2.barh([a for a, _, _ in lv], [b for _, b, _ in lv], color=[c for *_, c in lv])
        ax2.set_xlabel("[kV]"); ax2.set_title("niebieski = ogranicznik, czerwony = izolacja")
        Vmax = 1.05 * p["U"]
        mcov = Vmax / math.sqrt(3); tov = p["kg"] * Vmax
        return (f"PM1 = {pm[0]:.0f}%  PM2 = {pm[1]:.0f}%  PM3 = {pm[2]:.0f}%\n"
                + ("Koordynacja OK" if all(a >= r for a, r in zip(pm, req)) else "NIE SPEŁNIA!")
                + f"\nWymagane MCOV ≥ {mcov:.1f} kV\nWymagane TOV ≥ {tov:.1f} kV")


EXAMPLES = [NCI, Force, Catenary, Swing, Vertical, Spacer, Pinch,
            Impulse, CLF, Restrike, Travel, Separation, Arrester, BILTab, Coord]


# =========================================================================
#                                  GUI
# =========================================================================
def run_gui():
    import tkinter as tk
    from tkinter import ttk
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

    root = tk.Tk()
    root.title("Zwarcia w liniach i przepięcia - interaktywny podręcznik")
    root.geometry("1400x900"); root.configure(bg=BG)
    st = ttk.Style(); st.theme_use("clam")
    st.configure(".", background=BG, foreground=FG, fieldbackground=PANEL)
    st.configure("TNotebook.Tab", background=PANEL, foreground=FG, padding=(8, 3))
    st.map("TNotebook.Tab", background=[("selected", "#313244")], foreground=[("selected", CYAN)])
    st.configure("TLabelframe.Label", foreground=CYAN)
    st.configure("TButton", background="#313244", foreground=FG)

    top = ttk.Notebook(root); top.pack(fill="both", expand=True)
    books = {}
    for ch, name in [(14, "Rozdz. 14: Prądy zwarciowe a linie"), (15, "Rozdz. 15: Przepięcia i ochrona")]:
        nb = ttk.Notebook(top); top.add(nb, text=name); books[ch] = nb

    class Page:
        def __init__(self, ex):
            self.ex = ex; self.vars = {}; self.job = None; self.animating = False
            frame = ttk.Frame(books[ex.chapter]); books[ex.chapter].add(frame, text=ex.title)
            left = ttk.Frame(frame, width=340); left.pack(side="left", fill="y", padx=4, pady=4)
            right = ttk.Frame(frame); right.pack(side="left", fill="both", expand=True)
            # --- przewijany panel suwaków
            cv = tk.Canvas(left, bg=BG, highlightthickness=0, width=330, height=360)
            sb = ttk.Scrollbar(left, orient="vertical", command=cv.yview)
            inner = ttk.LabelFrame(cv, text="Parametry")
            inner.bind("<Configure>", lambda e: cv.configure(scrollregion=cv.bbox("all")))
            cv.create_window((0, 0), window=inner, anchor="nw"); cv.configure(yscrollcommand=sb.set)
            for prm in ex.params:
                if prm[0] == "choice":
                    _, key, lab, opts, init = prm
                    ttk.Label(inner, text=lab).pack(anchor="w", padx=6)
                    var = tk.StringVar(value=init)
                    cb = ttk.Combobox(inner, textvariable=var, values=opts, state="readonly", width=38)
                    cb.pack(padx=6, pady=2, anchor="w")
                    cb.bind("<<ComboboxSelected>>", lambda e: self.schedule())
                else:
                    key, lab, lo, hi, init, unit = prm
                    ttk.Label(inner, text=f"{lab} [{unit}]").pack(anchor="w", padx=6)
                    row = ttk.Frame(inner); row.pack(fill="x", padx=6)
                    var = tk.DoubleVar(value=init)
                    ttk.Scale(row, variable=var, from_=lo, to=hi, orient="horizontal", length=230,
                              command=lambda v: self.schedule()).pack(side="left")
                    vl = ttk.Label(row, text=f"{init:.3g}", width=7, foreground=YEL); vl.pack(side="left")
                    var.trace_add("write", lambda *_, v=var, l=vl: l.config(text=f"{v.get():.3g}"))
                self.vars[key] = var
            cv.pack(side="top", fill="x"); sb.place(in_=cv, relx=1.0, rely=0, relheight=1.0, anchor="ne")
            bf = ttk.Frame(left); bf.pack(fill="x", pady=4)
            ttk.Button(bf, text="Reset", command=self.reset).pack(side="left", padx=4)
            if ex.anim:
                self.abtn = ttk.Button(bf, text="▶ Animuj", command=self.toggle_anim)
                self.abtn.pack(side="left", padx=4)
            rf = ttk.LabelFrame(left, text="Wyniki"); rf.pack(fill="both", expand=True, pady=4)
            self.res = tk.Label(rf, text="", justify="left", anchor="nw", bg=PANEL, fg=GREEN,
                                font=("Consolas", 10))
            self.res.pack(fill="both", expand=True, padx=4, pady=4)
            # --- prawa strona: wykres + wyjaśnienie
            pw = ttk.PanedWindow(right, orient="vertical"); pw.pack(fill="both", expand=True)
            pf = ttk.Frame(pw); tf = ttk.Frame(pw)
            pw.add(pf, weight=3); pw.add(tf, weight=2)
            self.fig = Figure(figsize=(8, 5), dpi=100)
            self.canvas = FigureCanvasTkAgg(self.fig, master=pf)
            NavigationToolbar2Tk(self.canvas, pf).update()
            self.canvas.get_tk_widget().pack(fill="both", expand=True)
            txt = tk.Text(tf, wrap="word", bg=PANEL, fg=FG, font=("Segoe UI", 10), padx=10, pady=6,
                          relief="flat")
            ts = ttk.Scrollbar(tf, command=txt.yview); txt.configure(yscrollcommand=ts.set)
            ts.pack(side="right", fill="y"); txt.pack(fill="both", expand=True)
            txt.insert("1.0", ex.explanation); txt.configure(state="disabled")
            self.canvas.get_tk_widget().bind("<Configure>", lambda e: self.schedule(), add="+")
            self.drawn = False
            frame.bind("<Map>", lambda e: (None if self.drawn else self.redraw()))

        def values(self):
            out = {}
            for prm in self.ex.params:
                if prm[0] == "choice":
                    out[prm[1]] = self.vars[prm[1]].get()
                else:
                    k, _, lo, hi = prm[:4]
                    out[k] = min(max(float(self.vars[k].get()), lo), hi)
            return out

        def schedule(self):
            if self.job:
                root.after_cancel(self.job)
            self.job = root.after(120, self.redraw)

        def redraw(self):
            self.job = None; self.drawn = True
            self.fig.clear()
            try:
                txt = self.ex.compute(self.values(), self.fig)
                self.fig.tight_layout()
            except Exception as e:          # noqa - pokaż błąd zamiast zawiesić GUI
                txt = f"Błąd obliczeń:\n{e}"
            self.res.config(text=txt)
            self.canvas.draw_idle()

        def reset(self):
            for prm in self.ex.params:
                self.vars[prm[1] if prm[0] == "choice" else prm[0]].set(prm[4])
            self.redraw()

        def toggle_anim(self):
            self.animating = not self.animating
            self.abtn.config(text="■ Stop" if self.animating else "▶ Animuj")
            if self.animating:
                self.step()

        def step(self):
            if not self.animating:
                return
            prm = next(q for q in self.ex.params if q[0] == self.ex.anim)
            v = self.vars[self.ex.anim]; nv = v.get() + (prm[3] - prm[2]) / 150
            v.set(prm[2] if nv > prm[3] else nv)
            self.redraw(); root.after(40, self.step)

    pages = [Page(E()) for E in EXAMPLES]  # noqa: F841 - trzymamy referencje
    root.mainloop()


if __name__ == "__main__":
    run_gui()
