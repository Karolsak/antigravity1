"""
Maszyny prądu stałego - interaktywny zbiór przykładów (Tkinter + Matplotlib)
============================================================================

Aplikacja do rozdziału "UNIT III - D.C. Machines" (Basic Electrical and
Instrumentation Engineering). Każdy przykład z rozdziału ma własną zakładkę:

  * suwaki z danymi zadania (domyślnie dokładnie dane z książki),
  * rozwiązanie krok po kroku, przeliczane na żywo po każdej zmianie,
  * wyjaśnienie "jak dla ucznia liceum" + komentarz inżyniera,
  * interaktywne wykresy (najedź myszą na krzywą - pojawi się odczyt,
    pasek narzędzi pozwala powiększać / przesuwać / zapisywać wykres).

Uruchomienie:
    pip install numpy matplotlib
    python dc_machines_examples_tkinter.py
"""

import math
import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

import numpy as np
import matplotlib

matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# ─────────────────────────────────────────────────────────────────────────────
# Kolory (stała kolejność kategorii - kolor zawsze należy do tej samej serii)
# ─────────────────────────────────────────────────────────────────────────────
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
     "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8985"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"

TWO_PI = 2 * math.pi


def fmt(x, d=2):
    """Liczba w zapisie polskim (przecinek dziesiętny)."""
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "—"
    s = f"{x:.{d}f}"
    return s.replace(".", ",")


def emf(phi, P, N, Z, A):
    """Równanie SEM maszyny prądu stałego: E = φ·P·N·Z / (60·A)."""
    return phi * P * N * Z / (60.0 * A)


def omega(N):
    """Prędkość kątowa [rad/s] z obr/min."""
    return TWO_PI * N / 60.0


# ─────────────────────────────────────────────────────────────────────────────
# Budowanie tekstu rozwiązania
# ─────────────────────────────────────────────────────────────────────────────
class Doc:
    """Lista fragmentów (tekst, tag) wstawianych do widżetu Text."""

    def __init__(self):
        self.parts = []

    def h(self, s):
        self.parts.append(("\n" + s + "\n", "h"))

    def p(self, s):
        self.parts.append((s + "\n", "p"))

    def eq(self, s):
        self.parts.append(("    " + s + "\n", "eq"))

    def res(self, s):
        self.parts.append(("  ▶ " + s + "\n", "res"))

    def book(self, s):
        self.parts.append(("  Odpowiedź w książce (dla danych domyślnych): " + s + "\n", "book"))

    def eng(self, s):
        self.parts.append((s + "\n", "eng"))

    def bullet(self, s):
        self.parts.append(("  • " + s + "\n", "p"))


# ─────────────────────────────────────────────────────────────────────────────
# Pomocnicze funkcje rysowania
# ─────────────────────────────────────────────────────────────────────────────
def style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, fontsize=10.5, color=INK, loc="left", pad=8)
    ax.set_xlabel(xlabel, fontsize=9, color=INK2)
    ax.set_ylabel(ylabel, fontsize=9, color=INK2)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#bdbcb7")
    ax.tick_params(colors=INK2, labelsize=8.5)


def line(ax, x, y, color, label, lw=2.0, ls="-", alpha=1.0):
    return ax.plot(x, y, color=color, lw=lw, ls=ls, label=label, alpha=alpha)[0]


def point(ax, x, y, color, text=None, dx=8, dy=8):
    ax.plot([x], [y], "o", ms=8, color=color, mec="white", mew=2, zorder=5,
            label="_pt")
    if text:
        ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points",
                    fontsize=8.5, color=INK, zorder=6,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#d6d5d0", lw=0.8))


def legend(ax, loc="best"):
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) >= 2:
        leg = ax.legend(loc=loc, fontsize=8, frameon=True, framealpha=0.92,
                        edgecolor="#d6d5d0")
        for t in leg.get_texts():
            t.set_color(INK)


def waterfall(ax, items, title, ylabel="Napięcie [V]"):
    """Wykres 'schodkowy' napięć: items = [(nazwa, wartość, typ)], typ:
    'total' - słupek od zera, 'drop' - spadek (odejmowany od poprzedniego)."""
    level = 0.0
    names = []
    for i, (name, val, kind) in enumerate(items):
        if kind == "total":
            ax.bar(i, val, width=0.62, color=C[0], edgecolor=SURFACE, linewidth=2)
            level = val
            ax.text(i, val, fmt(val, 1) + " V", ha="center", va="bottom",
                    fontsize=8.5, color=INK)
        else:
            ax.bar(i, -val, bottom=level, width=0.62, color=C[1],
                   edgecolor=SURFACE, linewidth=2)
            ax.text(i, level, "−" + fmt(val, 2) + " V", ha="center", va="bottom",
                    fontsize=8.5, color=INK)
            level -= val
        names.append(name)
    ax.set_xticks(range(len(items)))
    ax.set_xticklabels(names, fontsize=8)
    style(ax, title, "", ylabel)
    top = max(v for _, v, k in items if k == "total")
    ax.set_ylim(0, top * 1.12)


# ─────────────────────────────────────────────────────────────────────────────
# Klasa bazowa strony (przykładu)
# ─────────────────────────────────────────────────────────────────────────────
class Page:
    group = ""
    title = ""
    subtitle = ""
    # (klucz, etykieta, min, max, domyślna, krok)
    params = []
    # (klucz, etykieta, [opcje], domyślna)
    choices = []
    layout = (1, 2)

    def __init__(self):
        self.values = {k: d for k, _, _, _, d, _ in self.params}
        self.values.update({k: d for k, _, _, d in self.choices})

    def defaults(self):
        v = {k: d for k, _, _, _, d, _ in self.params}
        v.update({k: d for k, _, _, d in self.choices})
        return v

    def solve(self, p, t):
        raise NotImplementedError

    def draw(self, axs, p):
        raise NotImplementedError


# =============================================================================
# 1. PODSTAWY
# =============================================================================
class GeneratorPrinciple(Page):
    group = "1. Podstawy działania"
    title = "Zasada działania prądnicy (e = B·l·v·sinθ)"
    subtitle = "Rozdz. 3.2–3.5: indukcja w przewodzie, komutator jako 'prostownik mechaniczny'"
    params = [
        ("B", "Indukcja magnetyczna B [T]", 0.1, 1.5, 0.8, 0.05),
        ("l", "Długość przewodu l [m]", 0.05, 1.0, 0.3, 0.05),
        ("v", "Prędkość przewodu v [m/s]", 1.0, 30.0, 10.0, 0.5),
        ("n", "Liczba cewek (wycinków komutatora)", 1, 12, 1, 1),
    ]

    def solve(self, p, t):
        em = p["B"] * p["l"] * p["v"]
        n = int(p["n"])
        th = np.linspace(0, 2 * np.pi, 3601)
        out = self._rectified(th, n)
        ripple = (out.max() - out.min()) / out.max() * 100
        t.h("O co chodzi? (wersja dla licealisty)")
        t.p("Wyobraź sobie dynamo rowerowe. Gdy przewód porusza się w polu magnesu, "
            "'przecina' linie pola i na jego końcach pojawia się napięcie - to jest prawo "
            "Faradaya. Im silniejszy magnes (B), dłuższy przewód (l) i szybszy ruch (v), "
            "tym większe napięcie.")
        t.p("Ale liczy się tylko ta część prędkości, która jest PROSTOPADŁA do linii pola. "
            "Stąd czynnik sinθ: gdy przewód sunie wzdłuż linii pola (θ = 0°), niczego nie "
            "przecina i napięcie jest zerowe; gdy porusza się w poprzek (θ = 90°) - "
            "napięcie jest największe.")
        t.h("Rozwiązanie krok po kroku")
        t.eq("e = B · l · v · sinθ")
        t.eq(f"e_max = {fmt(p['B'])} T · {fmt(p['l'])} m · {fmt(p['v'])} m/s = {fmt(em, 3)} V")
        t.res(f"Amplituda SEM jednego przewodu: {fmt(em, 3)} V")
        t.p("Przewód w obracającej się cewce raz jest pod biegunem N, raz pod S, więc jego "
            "napięcie zmienia znak - jest PRZEMIENNE (lewy wykres).")
        t.p("Komutator (pierścień rozcięty na wycinki) co pół obrotu zamienia końce cewki "
            "dołączone do szczotek. Dzięki temu na zaciskach napięcie ma zawsze ten sam znak "
            "- jest STAŁE (prawy wykres).")
        t.eq(f"Liczba cewek n = {n}  →  tętnienia napięcia ≈ {fmt(ripple, 1)} %")
        t.res(f"Średnie napięcie po komutatorze: {fmt(out.mean() * em, 3)} V "
              f"(szczyt {fmt(em, 3)} V)")
        t.h("Komentarz inżyniera")
        t.eng("Zwiększ liczbę cewek suwakiem - tętnienia szybko maleją (dla 1 cewki 100 %, "
              "dla 12 cewek ok. 1 %). Dlatego w prawdziwej maszynie liczba wycinków komutatora "
              "jest równa liczbie cewek twornika i napięcie jest praktycznie idealnie stałe. "
              "Kierunek SEM wyznaczysz regułą prawej dłoni (Fleminga): palec wskazujący - "
              "pole, kciuk - ruch, środkowy - prąd.")

    @staticmethod
    def _rectified(th, n):
        # n cewek przesuniętych równomiernie o π/n - szczotki zbierają największe napięcie
        shifts = np.arange(n) * np.pi / n
        return np.max(np.abs(np.sin(th[None, :] - shifts[:, None])), axis=0)

    def draw(self, axs, p):
        em = p["B"] * p["l"] * p["v"]
        n = int(p["n"])
        deg = np.linspace(0, 720, 1441)
        th = np.radians(deg)
        a1, a2 = axs
        line(a1, deg, em * np.sin(th), C[0], "SEM w przewodzie e(θ)")
        a1.axhline(0, color="#9a9994", lw=0.8)
        point(a1, 90, em, C[0], f"e_max = {fmt(em, 2)} V (θ = 90°)")
        style(a1, "Napięcie w przewodzie - przemienne", "Kąt obrotu θ [°]", "e [V]")
        a1.set_xticks(range(0, 721, 90))
        out = em * self._rectified(th, n)
        line(a2, deg, out, C[2], f"Napięcie na szczotkach (n = {n})")
        line(a2, deg, np.full_like(deg, out.mean()), C[3], "Wartość średnia", ls="--")
        style(a2, "Po komutatorze - jednokierunkowe", "Kąt obrotu θ [°]", "U [V]")
        a2.set_xticks(range(0, 721, 90))
        a2.set_ylim(0, em * 1.15)
        legend(a2, "lower right")


class MotorPrinciple(Page):
    group = "1. Podstawy działania"
    title = "Zasada działania silnika (F = B·I·l)"
    subtitle = "Rozdz. 3.21: siła na przewodnik z prądem, moment obrotowy cewki"
    params = [
        ("B", "Indukcja magnetyczna B [T]", 0.1, 1.5, 0.8, 0.05),
        ("I", "Prąd w przewodzie I [A]", 0.5, 50.0, 10.0, 0.5),
        ("l", "Długość przewodu l [m]", 0.05, 1.0, 0.3, 0.05),
        ("r", "Promień twornika r [m]", 0.02, 0.5, 0.1, 0.01),
        ("w", "Liczba zwojów cewki", 1, 50, 10, 1),
    ]

    def solve(self, p, t):
        F = p["B"] * p["I"] * p["l"]
        Tm = 2 * F * p["r"] * p["w"]
        t.h("O co chodzi? (wersja dla licealisty)")
        t.p("Przewód, przez który płynie prąd, sam wytwarza wokół siebie pole magnetyczne "
            "(okręgi wokół przewodu). Gdy włożymy go między bieguny magnesu, z jednej strony "
            "przewodu oba pola się dodają (linie 'zagęszczają się'), a z drugiej odejmują. "
            "Linie pola zachowują się jak napięte gumki - wypychają przewód z obszaru gęstego "
            "do rzadkiego. Tak powstaje siła.")
        t.h("Rozwiązanie krok po kroku")
        t.eq("F = B · I · l")
        t.eq(f"F = {fmt(p['B'])} · {fmt(p['I'])} · {fmt(p['l'])} = {fmt(F, 3)} N")
        t.p("Cewka ma dwa boki (jeden pod N, drugi pod S) - siły działają w przeciwne strony "
            "i tworzą parę sił, czyli moment obrotowy:")
        t.eq("T_max = 2 · F · r · (liczba zwojów)")
        t.eq(f"T_max = 2 · {fmt(F, 3)} · {fmt(p['r'])} · {int(p['w'])} = {fmt(Tm, 3)} N·m")
        t.res(f"Siła na jeden przewód: {fmt(F, 3)} N,  maks. moment cewki: {fmt(Tm, 3)} N·m")
        t.p("Moment zależy od położenia cewki (∝ |sinθ|). Bez komutatora po pół obrocie "
            "moment zmieniłby kierunek i silnik by się 'kołysał'. Komutator odwraca prąd w "
            "cewce dokładnie wtedy, gdy trzeba - moment ma zawsze ten sam kierunek.")
        t.h("Komentarz inżyniera")
        t.eng("Kierunek siły - reguła lewej dłoni (Fleminga): palec wskazujący - pole, "
              "środkowy - prąd, kciuk - siła/ruch. Aby zmienić kierunek obrotów silnika, "
              "odwracamy prąd albo w tworniku, albo w uzwojeniu wzbudzenia - ale nie w obu "
              "naraz (wtedy kierunek się nie zmieni).")

    def draw(self, axs, p):
        a1, a2 = axs
        F = p["B"] * p["I"] * p["l"]
        Tm = 2 * F * p["r"] * p["w"]
        I = np.linspace(0, 50, 200)
        for i, B in enumerate([0.4, 0.8, 1.2]):
            line(a1, I, B * I * p["l"], C[i], f"B = {fmt(B, 1)} T")
        point(a1, p["I"], F, INK, f"F = {fmt(F, 2)} N")
        style(a1, "Siła F(I) dla różnych B", "Prąd I [A]", "Siła F [N]")
        legend(a1, "upper left")
        deg = np.linspace(0, 360, 721)
        th = np.radians(deg)
        line(a2, deg, Tm * np.sin(th), C[1], "Bez komutatora", ls="--")
        line(a2, deg, Tm * np.abs(np.sin(th)), C[2], "Z komutatorem")
        a2.axhline(0, color="#9a9994", lw=0.8)
        style(a2, "Moment cewki T(θ)", "Kąt θ [°]", "Moment T [N·m]")
        a2.set_xticks(range(0, 361, 90))
        legend(a2, "lower left")


# =============================================================================
# 2. RÓWNANIE SEM
# =============================================================================
EMF_INTRO = [
    "SEM (siła elektromotoryczna) to napięcie wytwarzane wewnątrz prądnicy. Wzór "
    "E = φ·P·N·Z / (60·A) wygląda groźnie, ale opisuje zdrowy rozsądek:",
    "φ - strumień jednego bieguna [Wb] (jak 'mocny' jest magnes),",
    "P - liczba biegunów, N - prędkość [obr/min] (dzielimy przez 60, żeby mieć obr/s),",
    "Z - liczba wszystkich przewodów twornika,",
    "A - liczba gałęzi równoległych: uzwojenie pętlicowe (lap) A = P, "
    "uzwojenie faliste (wave) A = 2.",
]


def emf_intro(t):
    t.h("O co chodzi? (wersja dla licealisty)")
    t.p(EMF_INTRO[0])
    for s in EMF_INTRO[1:]:
        t.bullet(s)
    t.p("Analogia: przewody twornika to baterie. Pętlicowo łączymy je w wiele (A = P) "
        "krótszych 'łańcuchów' połączonych równolegle - mniejsze napięcie, większy prąd. "
        "Falisto - tylko w 2 długie łańcuchy - większe napięcie, mniejszy prąd.")


def emf_plot(ax, p_phi, P, Z, N_mark, title="E w funkcji prędkości"):
    N = np.linspace(0, max(2000, N_mark * 1.4), 300)
    line(ax, N, emf(p_phi, P, N, Z, P), C[0], f"Pętlicowe (A = {P})")
    line(ax, N, emf(p_phi, P, N, Z, 2), C[1], "Faliste (A = 2)")
    style(ax, title, "Prędkość N [obr/min]", "SEM E [V]")
    legend(ax, "upper left")


def emf_bars(ax, phi, P, N, Z):
    El, Ew = emf(phi, P, N, Z, P), emf(phi, P, N, Z, 2)
    ax.bar([0], [El], color=C[0], width=0.55, edgecolor=SURFACE, linewidth=2)
    ax.bar([1], [Ew], color=C[1], width=0.55, edgecolor=SURFACE, linewidth=2)
    for x, v in ((0, El), (1, Ew)):
        ax.text(x, v, fmt(v, 1) + " V", ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"Pętlicowe\n(gałęzi: {P})", "Faliste\n(gałęzi: 2)"], fontsize=8.5)
    style(ax, f"Porównanie przy N = {fmt(N, 0)} obr/min", "", "SEM E [V]")
    ax.set_ylim(0, max(El, Ew) * 1.15)


class EmfReview2(Page):
    group = "2. Równanie SEM"
    title = "SEM - zadanie: 4 bieguny, φ = 0,07 Wb, 900 obr/min"
    subtitle = "Pytanie kontrolne 2 (rozdz. 3.7): oblicz SEM dla uzwojenia pętlicowego i falistego"
    params = [
        ("phi", "Strumień na biegun φ [Wb]", 0.01, 0.15, 0.07, 0.005),
        ("P", "Liczba biegunów P", 2, 12, 4, 2),
        ("N", "Prędkość N [obr/min]", 100, 2000, 900, 10),
        ("Z", "Liczba przewodów Z", 100, 1500, 440, 10),
    ]

    def solve(self, p, t):
        phi, P, N, Z = p["phi"], int(p["P"]), p["N"], int(p["Z"])
        El, Ew = emf(phi, P, N, Z, P), emf(phi, P, N, Z, 2)
        emf_intro(t)
        t.h("Rozwiązanie krok po kroku")
        t.p("a) Uzwojenie pętlicowe: A = P")
        t.eq(f"E = {fmt(phi, 3)} · {P} · {fmt(N, 0)} · {Z} / (60 · {P}) = {fmt(El, 2)} V")
        t.p("b) Uzwojenie faliste: A = 2")
        t.eq(f"E = {fmt(phi, 3)} · {P} · {fmt(N, 0)} · {Z} / (60 · 2) = {fmt(Ew, 2)} V")
        t.res(f"E (pętlicowe) = {fmt(El, 1)} V,   E (faliste) = {fmt(Ew, 1)} V")
        t.book("462 V oraz 924 V")
        t.h("Komentarz inżyniera")
        t.eng(f"Przy P = {P} uzwojenie faliste daje {fmt(P / 2, 1)}× większe napięcie, ale "
              f"prąd znamionowy jest {fmt(P / 2, 1)}× mniejszy - moc maszyny jest TAKA SAMA. "
              "Dlatego: maszyny wysokonapięciowe, małoprądowe → uzwojenie faliste; "
              "niskonapięciowe, wielkoprądowe (np. do galwanizacji) → pętlicowe.")

    def draw(self, axs, p):
        phi, P, N, Z = p["phi"], int(p["P"]), p["N"], int(p["Z"])
        emf_plot(axs[0], phi, P, Z, N)
        point(axs[0], N, emf(phi, P, N, Z, P), C[0], f"{fmt(emf(phi, P, N, Z, P), 1)} V")
        point(axs[0], N, emf(phi, P, N, Z, 2), C[1], f"{fmt(emf(phi, P, N, Z, 2), 1)} V")
        emf_bars(axs[1], phi, P, N, Z)


class EmfReview3(Page):
    group = "2. Równanie SEM"
    title = "SEM - zadanie: 600 przewodów, 1200 obr/min (720 V)"
    subtitle = "Pytanie kontrolne 3 (rozdz. 3.7): SEM przy uzwojeniu pętlicowym i prędkość dla falistego"
    params = [
        ("phi", "Strumień na biegun φ [Wb]", 0.01, 0.15, 0.06, 0.005),
        ("P", "Liczba biegunów P", 2, 12, 4, 2),
        ("N", "Prędkość N [obr/min]", 100, 2500, 1200, 10),
        ("Z", "Liczba przewodów Z", 100, 1500, 600, 10),
    ]

    def solve(self, p, t):
        phi, P, N, Z = p["phi"], int(p["P"]), p["N"], int(p["Z"])
        El = emf(phi, P, N, Z, P)
        Nw = El * 60 * 2 / (phi * P * Z)
        emf_intro(t)
        t.p("Uwaga: w skanie książki strumień jest nieczytelny ('24 Wb'). Odpowiedź 720 V "
            "odpowiada strumieniowi 0,06 Wb na biegun - taką wartość przyjęto domyślnie.")
        t.h("Rozwiązanie krok po kroku")
        t.p("i) SEM przy uzwojeniu pętlicowym (A = P):")
        t.eq(f"E = {fmt(phi, 3)} · {P} · {fmt(N, 0)} · {Z} / (60 · {P}) = {fmt(El, 2)} V")
        t.p("ii) Jaka prędkość da TO SAMO napięcie przy uzwojeniu falistym (A = 2)?")
        t.eq("N = E · 60 · A / (φ · P · Z)")
        t.eq(f"N = {fmt(El, 1)} · 60 · 2 / ({fmt(phi, 3)} · {P} · {Z}) = {fmt(Nw, 1)} obr/min")
        t.res(f"E = {fmt(El, 1)} V,  wymagana prędkość (faliste) = {fmt(Nw, 1)} obr/min")
        t.book("720 V, 600 obr/min")
        t.h("Komentarz inżyniera")
        t.eng("Uzwojenie faliste ma mniej gałęzi równoległych, więc każda gałąź zawiera więcej "
              "przewodów połączonych szeregowo. To samo napięcie osiągamy przy prędkości "
              "mniejszej o czynnik P/2. Na wykresie widać, że linia 'faliste' jest bardziej "
              "stroma - pozioma linia stałego napięcia przecina ją wcześniej.")

    def draw(self, axs, p):
        phi, P, N, Z = p["phi"], int(p["P"]), p["N"], int(p["Z"])
        El = emf(phi, P, N, Z, P)
        Nw = El * 60 * 2 / (phi * P * Z)
        emf_plot(axs[0], phi, P, Z, N)
        axs[0].axhline(El, color=MUTED, lw=1, ls=":")
        point(axs[0], N, El, C[0], f"pętlicowe: {fmt(N, 0)} obr/min")
        point(axs[0], Nw, El, C[1], f"faliste: {fmt(Nw, 0)} obr/min", dy=-18)
        emf_bars(axs[1], phi, P, N, Z)


class EmfReview4(Page):
    group = "2. Równanie SEM"
    title = "SEM - zadanie: 51 żłobków × 24 przewody, E = 220 V"
    subtitle = "Pytanie kontrolne 4 (rozdz. 3.7): prędkość dla falistego i napięcie dla pętlicowego"
    params = [
        ("phi", "Strumień na biegun φ [Wb]", 0.005, 0.05, 0.01, 0.001),
        ("P", "Liczba biegunów P", 2, 12, 4, 2),
        ("slots", "Liczba żłobków", 10, 100, 51, 1),
        ("cps", "Przewodów w żłobku", 2, 40, 24, 1),
        ("E", "Wymagana SEM E [V]", 50, 600, 220, 5),
    ]

    def solve(self, p, t):
        phi, P = p["phi"], int(p["P"])
        Z = int(p["slots"]) * int(p["cps"])
        N = p["E"] * 60 * 2 / (phi * P * Z)
        El = emf(phi, P, N, Z, P)
        emf_intro(t)
        t.h("Rozwiązanie krok po kroku")
        t.p("Krok 1: liczba przewodów = żłobki × przewody w żłobku")
        t.eq(f"Z = {int(p['slots'])} · {int(p['cps'])} = {Z}")
        t.p("Krok 2: prędkość dla uzwojenia falistego (A = 2) - przekształcamy wzór na N:")
        t.eq(f"N = E·60·A/(φ·P·Z) = {fmt(p['E'], 0)}·60·2/({fmt(phi, 3)}·{P}·{Z}) = {fmt(N, 2)} obr/min")
        t.p("Krok 3: ta sama prędkość, ale uzwojenie pętlicowe (A = P):")
        t.eq(f"E = {fmt(phi, 3)}·{P}·{fmt(N, 2)}·{Z}/(60·{P}) = {fmt(El, 2)} V")
        t.res(f"N = {fmt(N, 1)} obr/min,  E (pętlicowe) = {fmt(El, 1)} V")
        t.book("≈ 539 obr/min oraz 110 V")
        t.h("Komentarz inżyniera")
        t.eng("Zauważ, że przy 4 biegunach przejście z falistego na pętlicowe dokładnie "
              "połowi napięcie. Napięcie zależy od ILOCZYNU φ·N - jeśli strumień jest mały, "
              "trzeba szybciej kręcić. Dlatego prądnice napędzane wolnymi silnikami "
              "spalinowymi mają silne pola i wiele biegunów.")

    def draw(self, axs, p):
        phi, P = p["phi"], int(p["P"])
        Z = int(p["slots"]) * int(p["cps"])
        N = p["E"] * 60 * 2 / (phi * P * Z)
        emf_plot(axs[0], phi, P, Z, N)
        axs[0].axhline(p["E"], color=MUTED, lw=1, ls=":")
        point(axs[0], N, p["E"], C[1], f"N = {fmt(N, 0)} obr/min")
        point(axs[0], N, emf(phi, P, N, Z, P), C[0], f"{fmt(emf(phi, P, N, Z, P), 0)} V", dy=-18)
        emf_bars(axs[1], phi, P, N, Z)


class EmfQ37(Page):
    group = "2. Równanie SEM"
    title = "SEM - Q.37: 45 żłobków × 18 przewodów, 1200 obr/min"
    subtitle = "Pytanie 'Two marks' Q.37: oblicz SEM prądnicy z uzwojeniem falistym"
    params = [
        ("phi", "Strumień na biegun φ [Wb]", 0.005, 0.05, 0.016, 0.001),
        ("P", "Liczba biegunów P", 2, 12, 4, 2),
        ("slots", "Liczba żłobków", 10, 100, 45, 1),
        ("cps", "Przewodów w żłobku", 2, 40, 18, 1),
        ("N", "Prędkość N [obr/min]", 100, 2500, 1200, 10),
    ]
    choices = [("wind", "Uzwojenie", ["faliste (A = 2)", "pętlicowe (A = P)"], "faliste (A = 2)")]

    def solve(self, p, t):
        phi, P, N = p["phi"], int(p["P"]), p["N"]
        Z = int(p["slots"]) * int(p["cps"])
        A = 2 if p["wind"].startswith("faliste") else P
        E = emf(phi, P, N, Z, A)
        emf_intro(t)
        t.h("Rozwiązanie krok po kroku")
        t.eq(f"Z = {int(p['slots'])} · {int(p['cps'])} = {Z} przewodów,   A = {A}")
        t.eq(f"E = {fmt(phi, 3)} · {P} · {fmt(N, 0)} · {Z} / (60 · {A}) = {fmt(E, 2)} V")
        t.res(f"E = {fmt(E, 1)} V")
        t.book("518,4 V (uzwojenie faliste)")
        t.h("Komentarz inżyniera")
        t.eng("Przełącz rodzaj uzwojenia - dla 4 biegunów napięcie spadnie o połowę. "
              "W praktyce strumień φ nie jest stały: zależy od prądu wzbudzenia i nasycenia "
              "żelaza (patrz zakładka 'Charakterystyka magnesowania').")

    def draw(self, axs, p):
        phi, P, N = p["phi"], int(p["P"]), p["N"]
        Z = int(p["slots"]) * int(p["cps"])
        A = 2 if p["wind"].startswith("faliste") else P
        emf_plot(axs[0], phi, P, Z, N)
        point(axs[0], N, emf(phi, P, N, Z, A), C[1] if A == 2 else C[0],
              f"{fmt(emf(phi, P, N, Z, A), 1)} V")
        emf_bars(axs[1], phi, P, N, Z)


# =============================================================================
# 3. PRĄDNICE
# =============================================================================
SHUNT_GEN_INTRO = [
    "Prądnica bocznikowa ma uzwojenie wzbudzenia podłączone RÓWNOLEGLE (bocznikowo) do "
    "twornika. Prąd wytworzony w tworniku dzieli się na dwie części: dużą - do odbiornika "
    "(I_L) i małą - do uzwojenia wzbudzenia (I_sh).",
    "Analogia wodna: pompa (twornik) tłoczy wodę; większość płynie do kranu (odbiornik), "
    "a mała odnoga zasila 'silniczek' utrzymujący pompę w ruchu (wzbudzenie).",
    "Napięcie wewnętrzne E musi być większe od napięcia na zaciskach V_t, bo część napięcia "
    "'gubi się' na rezystancji twornika (spadek I_a·R_a).",
]


class ShuntGenPage(Page):
    group = "3. Prądnice"
    layout = (1, 2)
    book_answer = ""
    ask_power = False
    has_feeder = False

    def solve(self, p, t):
        P, V, Ra, Rsh = p["P"] * 1000, p["V"], p["Ra"], p["Rsh"]
        Rf = p.get("Rf", 0.0)
        IL = P / V
        Vt = V + IL * Rf
        Ish = Vt / Rsh
        Ia = IL + Ish
        E = Vt + Ia * Ra
        Pdev = E * Ia
        t.h("O co chodzi? (wersja dla licealisty)")
        for s in SHUNT_GEN_INTRO:
            t.p(s)
        if self.has_feeder:
            t.p("Tutaj odbiornik jest daleko - prąd płynie przewodami zasilającymi (feederami) "
                "o rezystancji R_f, więc na zaciskach prądnicy musi być WIĘCEJ niż na odbiorniku.")
        t.h("Rozwiązanie krok po kroku")
        t.p("Krok 1: prąd odbiornika z mocy (P = V·I):")
        t.eq(f"I_L = P / V = {fmt(P, 0)} / {fmt(V, 1)} = {fmt(IL, 4)} A")
        step = 2
        if self.has_feeder:
            t.p(f"Krok {step}: napięcie na zaciskach prądnicy = napięcie odbiornika + spadek na feederach:")
            t.eq(f"V_t = V + I_L·R_f = {fmt(V, 1)} + {fmt(IL, 4)}·{fmt(Rf, 3)} = {fmt(Vt, 4)} V")
            step += 1
        t.p(f"Krok {step}: prąd wzbudzenia (bocznik ma na sobie napięcie V_t):")
        t.eq(f"I_sh = V_t / R_sh = {fmt(Vt, 3)} / {fmt(Rsh, 1)} = {fmt(Ish, 4)} A")
        step += 1
        t.p(f"Krok {step}: prąd twornika = prąd odbiornika + prąd wzbudzenia (prawo Kirchhoffa):")
        t.eq(f"I_a = I_L + I_sh = {fmt(IL, 4)} + {fmt(Ish, 4)} = {fmt(Ia, 4)} A")
        step += 1
        t.p(f"Krok {step}: SEM indukowana (napięcie wewnętrzne):")
        t.eq(f"E = V_t + I_a·R_a = {fmt(Vt, 3)} + {fmt(Ia, 4)}·{fmt(Ra, 3)} = {fmt(E, 4)} V")
        t.res(f"Indukowana SEM E = {fmt(E, 3)} V")
        if self.has_feeder:
            t.res(f"Napięcie na zaciskach prądnicy V_t = {fmt(Vt, 3)} V")
        t.p("Bilans mocy:")
        t.eq(f"Moc wytworzona w tworniku  E·I_a = {fmt(Pdev / 1000, 4)} kW")
        t.eq(f"Straty w tworniku   I_a²·R_a = {fmt(Ia ** 2 * Ra, 1)} W")
        t.eq(f"Straty we wzbudzeniu V_t·I_sh = {fmt(Vt * Ish, 1)} W")
        if self.has_feeder:
            t.eq(f"Straty w feederach  I_L²·R_f = {fmt(IL ** 2 * Rf, 1)} W")
        if self.ask_power:
            t.res(f"Moc rozwinięta przez twornik = {fmt(Pdev / 1000, 4)} kW")
        t.book(self.book_answer)
        t.h("Komentarz inżyniera")
        t.eng(f"Sprawność 'elektryczna' (bez strat mechanicznych i w żelazie): "
              f"{fmt(P / Pdev * 100, 1)} %. Prąd wzbudzenia to tylko {fmt(Ish / Ia * 100, 1)} % "
              "prądu twornika - dlatego w wielu zadaniach można go w przybliżeniu pominąć. "
              "Na prawym wykresie widać, że przy większym obciążeniu E musi rosnąć - w realnej "
              "prądnicy bocznikowej (bez regulacji) napięcie zacisków po prostu spada.")

    def draw(self, axs, p):
        P, V, Ra, Rsh = p["P"] * 1000, p["V"], p["Ra"], p["Rsh"]
        Rf = p.get("Rf", 0.0)
        IL = P / V
        Vt = V + IL * Rf
        Ia = IL + Vt / Rsh
        E = Vt + Ia * Ra
        items = [("E", E, "total"), ("I_a·R_a", Ia * Ra, "drop"), ("V_t", Vt, "total")]
        if Rf > 0:
            items += [("I_L·R_f", IL * Rf, "drop"), ("V odb.", V, "total")]
        waterfall(axs[0], items, "Gdzie 'znika' napięcie")
        Ps = np.linspace(0.0, 2 * P, 200)
        ILs = Ps / V
        Vts = V + ILs * Rf
        Es = Vts + (ILs + Vts / Rsh) * Ra
        line(axs[1], Ps / 1000, Es, C[0], "SEM E (wymagana)")
        line(axs[1], Ps / 1000, Vts, C[2], "Napięcie zacisków V_t")
        point(axs[1], P / 1000, E, C[0], f"E = {fmt(E, 2)} V")
        style(axs[1], "Wymagana SEM a moc odbiornika", "Moc odbiornika P [kW]", "Napięcie [V]")
        legend(axs[1], "upper left")


class Ex3121(ShuntGenPage):
    title = "Przykład 3.12.1 - prądnica bocznikowa 7,5 kW, 200 V"
    subtitle = "Oblicz indukowaną SEM; R_a = 0,6 Ω, R_sh = 80 Ω"
    book_answer = "E = 224 V (I_L = 37,5 A, I_sh = 2,5 A, I_a = 40 A)"
    params = [
        ("P", "Moc odbiornika P [kW]", 1, 30, 7.5, 0.5),
        ("V", "Napięcie zacisków V_t [V]", 100, 400, 200, 5),
        ("Ra", "Rezystancja twornika R_a [Ω]", 0.05, 2.0, 0.6, 0.05),
        ("Rsh", "Rezystancja bocznika R_sh [Ω]", 20, 300, 80, 5),
    ]


class Ex3122(ShuntGenPage):
    title = "Przykład 3.12.2 - prądnica bocznikowa 30 kW, 300 V"
    subtitle = "Oblicz moc rozwiniętą przez twornik; R_a = 0,05 Ω, R_sh = 100 Ω"
    book_answer = "E = 305,15 V, E·I_a = 31,4304 kW"
    ask_power = True
    params = [
        ("P", "Moc odbiornika P [kW]", 5, 60, 30, 1),
        ("V", "Napięcie zacisków V_t [V]", 100, 500, 300, 5),
        ("Ra", "Rezystancja twornika R_a [Ω]", 0.01, 0.5, 0.05, 0.01),
        ("Rsh", "Rezystancja bocznika R_sh [Ω]", 20, 300, 100, 5),
    ]


class ExShuntFeeder(ShuntGenPage):
    title = "Zadanie 3.12-2 - prądnica 10 kW, 200 V przez feedery 0,05 Ω"
    subtitle = "Pytanie kontrolne 2 (rozdz. 3.12): napięcie zacisków i SEM"
    book_answer = "V_t = 202,5 V, E = 207,7025 V"
    has_feeder = True
    params = [
        ("P", "Moc odbiornika P [kW]", 1, 30, 10, 0.5),
        ("V", "Napięcie na odbiorniku V [V]", 100, 400, 200, 5),
        ("Rf", "Rezystancja feederów R_f [Ω]", 0.0, 0.5, 0.05, 0.01),
        ("Ra", "Rezystancja twornika R_a [Ω]", 0.01, 1.0, 0.1, 0.01),
        ("Rsh", "Rezystancja bocznika R_sh [Ω]", 20, 300, 100, 5),
    ]


class ExQ36(ShuntGenPage):
    title = "Q.36 - prądnica 10 kW, 220 V przez feedery 0,1 Ω"
    subtitle = "Pytanie 'Two marks' Q.36: oblicz napięcie na zaciskach prądnicy"
    book_answer = "V_t = 224,5454 V"
    has_feeder = True
    params = [
        ("P", "Moc odbiornika P [kW]", 1, 30, 10, 0.5),
        ("V", "Napięcie na odbiorniku V [V]", 100, 400, 220, 5),
        ("Rf", "Rezystancja feederów R_f [Ω]", 0.0, 0.5, 0.1, 0.01),
        ("Ra", "Rezystancja twornika R_a [Ω]", 0.01, 1.0, 0.05, 0.01),
        ("Rsh", "Rezystancja bocznika R_sh [Ω]", 20, 300, 100, 5),
    ]


def compound_gen(Vt, IL, Ra, Rse, Rsh, long_shunt):
    if long_shunt:
        Ish = Vt / Rsh
        Ia = IL + Ish
        Ise = Ia
        E = Vt + Ia * (Ra + Rse)
    else:
        Ish = (Vt + IL * Rse) / Rsh
        Ia = IL + Ish
        Ise = IL
        E = Vt + Ia * Ra + IL * Rse
    return Ish, Ia, Ise, E


COMPOUND_INTRO = [
    "Prądnica szeregowo-bocznikowa (kompound) ma DWA uzwojenia wzbudzenia na tych samych "
    "biegunach: bocznikowe (wiele zwojów cienkiego drutu, równolegle) i szeregowe (kilka "
    "zwojów grubego drutu, w szereg z obciążeniem).",
    "Bocznik daje 'podstawowy' strumień, a uzwojenie szeregowe DOKŁADA strumienia, gdy rośnie "
    "obciążenie - dzięki temu napięcie nie spada (może nawet rosnąć).",
    "Bocznik długi (long shunt): bocznik jest równolegle do (twornik + uzwojenie szeregowe), "
    "więc przez uzwojenie szeregowe płynie I_a.  Bocznik krótki (short shunt): bocznik jest "
    "równolegle tylko do twornika, przez uzwojenie szeregowe płynie I_L.",
]


class CompoundPage(Page):
    group = "3. Prądnice"
    book_answer = ""
    lamps = False

    def _IL(self, p):
        if self.lamps:
            return int(p["n"]) * p["V"] / p["Rl"]
        return p["P"] * 1000 / p["V"]

    def solve(self, p, t):
        V, Ra, Rse, Rsh = p["V"], p["Ra"], p["Rse"], p["Rsh"]
        long_shunt = p["conn"].startswith("długi")
        IL = self._IL(p)
        Ish, Ia, Ise, E = compound_gen(V, IL, Ra, Rse, Rsh, long_shunt)
        t.h("O co chodzi? (wersja dla licealisty)")
        for s in COMPOUND_INTRO:
            t.p(s)
        t.h("Rozwiązanie krok po kroku")
        if self.lamps:
            n = int(p["n"])
            t.p("Krok 1: lampy są połączone równolegle, więc każda ma napięcie V_t:")
            t.eq(f"I_lampy = V_t / R_lampy = {fmt(V, 0)} / {fmt(p['Rl'], 0)} = {fmt(V / p['Rl'], 3)} A")
            t.eq(f"I_L = {n} · {fmt(V / p['Rl'], 3)} = {fmt(IL, 3)} A")
        else:
            t.p("Krok 1: prąd odbiornika z mocy:")
            t.eq(f"I_L = P / V_t = {fmt(p['P'] * 1000, 0)} / {fmt(V, 0)} = {fmt(IL, 3)} A")
        if long_shunt:
            t.p("Krok 2: bocznik długi - na boczniku jest pełne napięcie zacisków:")
            t.eq(f"I_sh = V_t / R_sh = {fmt(V, 0)} / {fmt(Rsh, 1)} = {fmt(Ish, 4)} A")
            t.p("Krok 3: prąd twornika (płynie też przez uzwojenie szeregowe):")
            t.eq(f"I_a = I_se = I_L + I_sh = {fmt(IL, 3)} + {fmt(Ish, 4)} = {fmt(Ia, 4)} A")
            t.p("Krok 4: SEM:")
            t.eq("E = V_t + I_a·(R_a + R_se)")
            t.eq(f"E = {fmt(V, 0)} + {fmt(Ia, 4)}·({fmt(Ra, 3)} + {fmt(Rse, 3)}) = {fmt(E, 3)} V")
        else:
            t.p("Krok 2: bocznik krótki - napięcie na boczniku = V_t + spadek na uzwojeniu szeregowym:")
            t.eq(f"I_sh = (V_t + I_L·R_se)/R_sh = ({fmt(V, 0)} + {fmt(IL, 3)}·{fmt(Rse, 3)})/{fmt(Rsh, 1)} "
                 f"= {fmt(Ish, 4)} A")
            t.p("Krok 3: prąd twornika:")
            t.eq(f"I_a = I_L + I_sh = {fmt(IL, 3)} + {fmt(Ish, 4)} = {fmt(Ia, 4)} A")
            t.p("Krok 4: SEM (przez uzwojenie szeregowe płynie I_L):")
            t.eq("E = V_t + I_a·R_a + I_L·R_se")
            t.eq(f"E = {fmt(V, 0)} + {fmt(Ia, 4)}·{fmt(Ra, 3)} + {fmt(IL, 3)}·{fmt(Rse, 3)} = {fmt(E, 3)} V")
        RL = V / IL
        t.p("Krok 5: rezystancja obciążenia (prawo Ohma):")
        t.eq(f"R_L = V_t / I_L = {fmt(V, 0)} / {fmt(IL, 3)} = {fmt(RL, 4)} Ω")
        t.res(f"E = {fmt(E, 3)} V,  I_a = {fmt(Ia, 3)} A,  R_L = {fmt(RL, 3)} Ω")
        t.book(self.book_answer)
        other = compound_gen(V, IL, Ra, Rse, Rsh, not long_shunt)[3]
        t.h("Komentarz inżyniera")
        t.eng(f"Dla porównania - przy drugim sposobie połączenia ({'krótki' if long_shunt else 'długi'} "
              f"bocznik) wyszłoby E = {fmt(other, 2)} V. Różnica jest niewielka, bo prąd bocznika "
              "jest mały. Kompound zgodny stosuje się do zasilania oświetlenia i przesyłu na "
              "większe odległości - uzwojenie szeregowe kompensuje spadki napięcia na liniach.")

    def draw(self, axs, p):
        V, Ra, Rse, Rsh = p["V"], p["Ra"], p["Rse"], p["Rsh"]
        long_shunt = p["conn"].startswith("długi")
        IL = self._IL(p)
        Ish, Ia, Ise, E = compound_gen(V, IL, Ra, Rse, Rsh, long_shunt)
        if long_shunt:
            items = [("E", E, "total"), ("I_a·R_a", Ia * Ra, "drop"),
                     ("I_a·R_se", Ia * Rse, "drop"), ("V_t", V, "total")]
        else:
            items = [("E", E, "total"), ("I_a·R_a", Ia * Ra, "drop"),
                     ("I_L·R_se", IL * Rse, "drop"), ("V_t", V, "total")]
        waterfall(axs[0], items, "Bilans napięć")
        ILs = np.linspace(0, 2 * IL, 200)
        El = np.array([compound_gen(V, x, Ra, Rse, Rsh, True)[3] for x in ILs])
        Es = np.array([compound_gen(V, x, Ra, Rse, Rsh, False)[3] for x in ILs])
        line(axs[1], ILs, El, C[0], "Bocznik długi")
        line(axs[1], ILs, Es, C[1], "Bocznik krótki", ls="--")
        point(axs[1], IL, E, C[0] if long_shunt else C[1], f"E = {fmt(E, 2)} V")
        style(axs[1], f"Wymagana SEM przy V_t = {fmt(V, 0)} V", "Prąd obciążenia I_L [A]", "SEM E [V]")
        legend(axs[1], "upper left")


class Ex3141(CompoundPage):
    title = "Przykład 3.14.1 - kompound zasila lampy 500 Ω przy 550 V"
    subtitle = "Lampy równolegle, R_sh = 25 Ω, R_a = 0,06 Ω, R_se = 0,04 Ω - oblicz prąd twornika i SEM"
    book_answer = ("skan jest częściowo nieczytelny - liczbę lamp (domyślnie 20, wg rysunku I₁…I₂₀) "
                   "i rodzaj połączenia można zmienić")
    lamps = True
    params = [
        ("n", "Liczba lamp", 1, 100, 20, 1),
        ("Rl", "Rezystancja lampy [Ω]", 100, 1500, 500, 10),
        ("V", "Napięcie zacisków V_t [V]", 200, 700, 550, 5),
        ("Ra", "Rezystancja twornika R_a [Ω]", 0.01, 0.5, 0.06, 0.01),
        ("Rse", "Rezystancja uzw. szeregowego R_se [Ω]", 0.01, 0.3, 0.04, 0.01),
        ("Rsh", "Rezystancja bocznika R_sh [Ω]", 10, 300, 25, 1),
    ]
    choices = [("conn", "Połączenie", ["długi bocznik", "krótki bocznik"], "długi bocznik")]


class Ex3142(CompoundPage):
    title = "Przykład 3.14.2 - kompound z krótkim bocznikiem, 7,5 kW, 230 V"
    subtitle = "R_se = 0,3 Ω, R_a = 0,4 Ω, R_sh = 100 Ω - oblicz SEM i rezystancję obciążenia"
    book_answer = "I_L = 32,608 A, I_sh = 2,3978 A, I_a = 35,006 A, E ≈ 253,8 V, R_L ≈ 7,05 Ω"
    params = [
        ("P", "Moc obciążenia P [kW]", 1, 30, 7.5, 0.5),
        ("V", "Napięcie zacisków V_t [V]", 100, 500, 230, 5),
        ("Ra", "Rezystancja twornika R_a [Ω]", 0.05, 1.0, 0.4, 0.05),
        ("Rse", "Rezystancja uzw. szeregowego R_se [Ω]", 0.01, 1.0, 0.3, 0.01),
        ("Rsh", "Rezystancja bocznika R_sh [Ω]", 20, 300, 100, 5),
    ]
    choices = [("conn", "Połączenie", ["krótki bocznik", "długi bocznik"], "krótki bocznik")]


class Ex3143(CompoundPage):
    title = "Przykład 3.14.3 - kompound z krótkim bocznikiem, 48 kW, 240 V"
    subtitle = "R_se = 0,015 Ω, R_a = 0,03 Ω, R_sh = 120 Ω - oblicz SEM i rezystancję obciążenia"
    book_answer = "I_L = 200 A, I_sh = 2,025 A, I_a = 202,025 A, E = 249,06 V, R_L = 1,2 Ω"
    params = [
        ("P", "Moc obciążenia P [kW]", 5, 100, 48, 1),
        ("V", "Napięcie zacisków V_t [V]", 100, 500, 240, 5),
        ("Ra", "Rezystancja twornika R_a [Ω]", 0.005, 0.2, 0.03, 0.005),
        ("Rse", "Rezystancja uzw. szeregowego R_se [Ω]", 0.005, 0.1, 0.015, 0.005),
        ("Rsh", "Rezystancja bocznika R_sh [Ω]", 20, 300, 120, 5),
    ]
    choices = [("conn", "Połączenie", ["krótki bocznik", "długi bocznik"], "krótki bocznik")]


# ─── Charakterystyka magnesowania (OCC) i samowzbudzenie ─────────────────────
def occ(If, N, Nr=1000.0, Emax=300.0, If0=1.0, Er=8.0):
    """Model Froelicha z magnetyzmem szczątkowym: E = (N/Nr)·(Er + Emax·If/(If0+If))."""
    If = np.maximum(If, 0.0)
    return (N / Nr) * (Er + Emax * If / (If0 + If))


class OCCPage(Page):
    group = "3. Prądnice"
    title = "Charakterystyka magnesowania i samowzbudzenie"
    subtitle = "Rozdz. 3.11, 3.15–3.16: OCC dla różnych prędkości, rezystancja krytyczna, narastanie napięcia"
    params = [
        ("N", "Prędkość N [obr/min]", 300, 1500, 1000, 10),
        ("Rf", "Rezystancja obwodu wzbudzenia R_f [Ω]", 20, 400, 100, 5),
        ("Er", "Napięcie od magnetyzmu szczątkowego [V]", 0.0, 30.0, 8.0, 1.0),
        ("L", "Indukcyjność wzbudzenia L_f [H]", 1, 50, 10, 1),
    ]

    @staticmethod
    def operating(N, Rf, Er):
        If = np.linspace(0, 6, 6001)
        g = occ(If, N, Er=Er) - Rf * If
        idx = np.where(np.sign(g[:-1]) != np.sign(g[1:]))[0]
        if len(idx) == 0:
            return 0.0, float(occ(0.0, N, Er=Er))
        i = idx[-1]
        x = If[i] - g[i] * (If[i + 1] - If[i]) / (g[i + 1] - g[i])
        return x, x * Rf

    def solve(self, p, t):
        N, Rf, Er = p["N"], p["Rf"], p["Er"]
        Rcr = (N / 1000.0) * 300.0 / 1.0  # nachylenie OCC w początku układu
        If_op, V_op = self.operating(N, Rf, Er)
        t.h("O co chodzi? (wersja dla licealisty)")
        t.p("Charakterystyka magnesowania (biegu jałowego, OCC) to wykres napięcia E w funkcji "
            "prądu wzbudzenia I_f przy stałej prędkości i bez obciążenia. Na początku jest prawie "
            "prostą (E ∝ φ ∝ I_f), ale potem żelazo się NASYCA - jak gąbka, która nie wchłonie już "
            "więcej wody - i dalsze zwiększanie prądu prawie nie zwiększa strumienia.")
        t.p("Ponieważ E = k·φ·N, dla większej prędkości cała krzywa leży wyżej (krzywe dla "
            "N₁ < N₂ < N₃ na lewym wykresie).")
        t.p("Samowzbudzenie (prądnica bocznikowa) przypomina kulę śnieżną: magnetyzm szczątkowy "
            "daje małe napięcie → płynie mały prąd wzbudzenia → strumień rośnie → napięcie rośnie… "
            "aż do punktu, w którym krzywa OCC przecina prostą rezystancji wzbudzenia V = R_f·I_f.")
        t.h("Rozwiązanie / analiza")
        t.eq(f"Prosta wzbudzenia: V = R_f · I_f = {fmt(Rf, 0)} · I_f")
        t.eq(f"Rezystancja krytyczna (nachylenie OCC w zerze) R_kr ≈ {fmt(Rcr, 0)} Ω")
        if Er <= 0:
            t.res("Brak magnetyzmu szczątkowego → prądnica NIE wzbudzi się (napięcie = 0 V).")
        elif Rf >= Rcr:
            t.res(f"R_f = {fmt(Rf, 0)} Ω ≥ R_kr → prądnica NIE wzbudzi się, zostaje tylko "
                  f"≈ {fmt(V_op, 1)} V od magnetyzmu szczątkowego.")
        else:
            t.res(f"Punkt pracy: I_f = {fmt(If_op, 3)} A,  E = {fmt(V_op, 1)} V")
        t.h("Dlaczego prądnica może się nie wzbudzić?")
        t.bullet("brak magnetyzmu szczątkowego (suwak 'magnetyzm szczątkowy' = 0),")
        t.bullet("zbyt duża rezystancja obwodu wzbudzenia (R_f > R_kr),")
        t.bullet("odwrotnie podłączone uzwojenie wzbudzenia lub zły kierunek obrotów - "
                 "prąd wzbudzenia KASUJE magnetyzm szczątkowy,")
        t.bullet("zbyt mała prędkość (R_kr maleje razem z prędkością - przesuń suwak N).")
        t.h("Komentarz inżyniera")
        t.eng("Prawy wykres pokazuje narastanie napięcia w czasie: L_f·dI_f/dt = E(I_f) − R_f·I_f. "
              "Im większa indukcyjność, tym wolniej. Blisko rezystancji krytycznej napięcie "
              "narasta bardzo wolno i jest niestabilne - dlatego prądnice pracują w obszarze "
              "kolana charakterystyki, gdzie napięcie jest stabilne.")

    def draw(self, axs, p):
        N, Rf, Er, L = p["N"], p["Rf"], p["Er"], p["L"]
        a1, a2 = axs
        If = np.linspace(0, 4, 400)
        for i, k in enumerate([0.6, 0.8, 1.0]):
            line(a1, If, occ(If, 1000 * k, Er=Er), C[i], f"OCC przy {fmt(1000 * k, 0)} obr/min", lw=1.4)
        line(a1, If, occ(If, N, Er=Er), INK, f"OCC przy N = {fmt(N, 0)} obr/min", lw=2.4)
        line(a1, If, Rf * If, C[7], f"Prosta R_f = {fmt(Rf, 0)} Ω", ls="--")
        Rcr = (N / 1000.0) * 300.0
        line(a1, If, Rcr * If, MUTED, f"R krytyczna ≈ {fmt(Rcr, 0)} Ω", ls=":", lw=1.4)
        If_op, V_op = self.operating(N, Rf, Er)
        if If_op > 0:
            point(a1, If_op, V_op, C[7], f"{fmt(V_op, 0)} V")
        a1.set_ylim(0, 420)
        style(a1, "Charakterystyka magnesowania E(I_f)", "Prąd wzbudzenia I_f [A]", "SEM E [V]")
        legend(a1, "lower right")
        # narastanie napięcia w czasie
        dt, T = 0.005, 20.0
        n = int(T / dt)
        tt = np.arange(n) * dt
        ifv = np.zeros(n)
        for k in range(1, n):
            ifv[k] = ifv[k - 1] + dt * (float(occ(ifv[k - 1], N, Er=Er)) - Rf * ifv[k - 1]) / L
        line(a2, tt, occ(ifv, N, Er=Er), C[0], "Napięcie E(t)")
        style(a2, "Samowzbudzenie w czasie", "Czas t [s]", "E [V]")
        a2.set_ylim(0, 420)


# ─── Charakterystyki zewnętrzne prądnic ──────────────────────────────────────
def mag(F, Emax=300.0, F0=1.0, Er=8.0):
    F = np.maximum(F, 0.0)
    return Er + Emax * F / (F0 + F)


class GenCharPage(Page):
    group = "3. Prądnice"
    title = "Charakterystyki zewnętrzne prądnic V_t(I_L)"
    subtitle = "Rozdz. 3.16–3.18: obcowzbudna, bocznikowa, szeregowa, kompound zgodny i przeciwny"
    layout = (1, 2)
    params = [
        ("Ra", "Rezystancja twornika R_a [Ω]", 0.05, 1.0, 0.25, 0.05),
        ("Rsh", "Rezystancja bocznika R_sh [Ω]", 50, 250, 110, 5),
        ("kar", "Reakcja twornika [A_f na 1 A twornika]", 0.0, 0.01, 0.003, 0.0005),
        ("kc", "Siła uzw. szeregowego w kompoundzie [A_f/A]", 0.0, 0.03, 0.012, 0.001),
    ]
    Rse = 0.05

    def curves(self, p):
        Ra, Rsh, kar, kc = p["Ra"], p["Rsh"], p["kar"], p["kc"]
        Rse = self.Rse
        out = {}
        # obcowzbudna: stały prąd wzbudzenia tak, by na biegu jałowym było ~ jak dla bocznikowej
        Vgrid = np.linspace(400, 0.01, 4000)
        If_sep = 2.0
        Ia = np.linspace(0, 160, 300)
        out["Obcowzbudna"] = (Ia, mag(If_sep - kar * Ia) - Ia * Ra)
        # samowzbudne: dla każdego R_L szukamy największego pierwiastka V_t
        RLs = np.geomspace(60, 0.02, 500)

        def solve_RL(F_fun, R_arm, shunt=True):
            IL_list, V_list = [], []
            for RL in RLs:
                Ia_v = Vgrid / RL + (Vgrid / Rsh if shunt else 0)
                g = mag(F_fun(Vgrid, Ia_v)) - Ia_v * R_arm - Vgrid
                idx = np.where(g > 0)[0]
                if len(idx) == 0:
                    V = 0.0
                else:
                    V = Vgrid[idx[0]]
                IL_list.append(V / RL)
                V_list.append(V)
            return np.array(IL_list), np.array(V_list)

        out["Bocznikowa"] = solve_RL(lambda V, I: V / Rsh - kar * I, Ra)
        out["Szeregowa"] = solve_RL(lambda V, I: 0.02 * I - kar * I, Ra + Rse, shunt=False)
        out["Kompound zgodny"] = solve_RL(lambda V, I: V / Rsh + kc * I - kar * I, Ra + Rse)
        out["Kompound przeciwny"] = solve_RL(lambda V, I: V / Rsh - kc * I - kar * I, Ra + Rse)
        return out

    def solve(self, p, t):
        cur = self.curves(p)
        t.h("O co chodzi? (wersja dla licealisty)")
        t.p("Charakterystyka zewnętrzna mówi, jak zmienia się napięcie na zaciskach prądnicy, "
            "gdy dokładamy odbiorników (rośnie prąd I_L). To jak ciśnienie w kranie, gdy "
            "sąsiedzi odkręcają kolejne krany.")
        t.h("Dlaczego napięcie się zmienia?")
        t.bullet("spadek na rezystancji twornika I_a·R_a (rośnie z prądem),")
        t.bullet("reakcja twornika - pole wytworzone przez prąd twornika osłabia pole główne,")
        t.bullet("w prądnicy bocznikowej dodatkowo: mniejsze napięcie → mniejszy prąd wzbudzenia → "
                 "jeszcze mniejsze napięcie (efekt 'spirali').")
        t.h("Co pokazuje wykres (wyniki modelu)")
        for name, (I, V) in cur.items():
            t.eq(f"{name:<20s}: V(bieg jałowy) ≈ {fmt(V[0], 0)} V,  maks. prąd ≈ {fmt(np.max(I), 0)} A")
        t.p("• Obcowzbudna - napięcie spada łagodnie (prawie prosta).")
        t.p("• Bocznikowa - spada mocniej, a przy przeciążeniu krzywa 'zawraca': zmniejszanie "
            "rezystancji odbiornika powoduje spadek napięcia tak duży, że PRĄD też maleje. "
            "Przy zwarciu płynie tylko mały prąd od magnetyzmu szczątkowego - prądnica bocznikowa "
            "sama 'chroni się' przed zwarciem.")
        t.p("• Szeregowa - napięcie ROŚNIE z obciążeniem (prąd obciążenia jest prądem wzbudzenia); "
            "bez obciążenia ma tylko napięcie szczątkowe. Używana jako 'booster' w liniach.")
        t.p("• Kompound zgodny - uzwojenie szeregowe dokłada strumienia: napięcie prawie stałe "
            "(płaski), rosnące (przekompoundowany) lub lekko spadające (niedokompoundowany) - "
            "zmieniaj suwak siły uzwojenia szeregowego!")
        t.p("• Kompound przeciwny - strumienie się odejmują, napięcie szybko spada - "
            "charakterystyka 'prądowa', np. do spawarek łukowych.")
        t.h("Komentarz inżyniera")
        t.eng("To model jakościowy (krzywa magnesowania Froelicha + liniowa reakcja twornika), "
              "kształtami odpowiada rysunkom z rozdziałów 3.16-3.18. Ustaw reakcję twornika na 0 "
              "i zobacz, ile spadku napięcia daje sama rezystancja.")

    def draw(self, axs, p):
        cur = self.curves(p)
        a1, a2 = axs
        F = np.linspace(0, 4, 300)
        line(a1, F, mag(F), C[0], "E(I_f) przy stałej prędkości")
        style(a1, "Krzywa magnesowania użyta w modelu", "Równoważny prąd wzbudzenia [A]", "E [V]")
        for i, (name, (I, V)) in enumerate(cur.items()):
            line(a2, I, V, C[i], name)
        style(a2, "Charakterystyki zewnętrzne", "Prąd obciążenia I_L [A]", "Napięcie zacisków V_t [V]")
        a2.set_xlim(0, 170)
        a2.set_ylim(0, 300)
        legend(a2, "upper right")


# =============================================================================
# 4. SILNIKI
# =============================================================================
class BackEmfPage(Page):
    group = "4. Silniki"
    title = "SEM wsteczna - Q.26: silnik 220 V, R_a = 0,5 Ω, I_a = 20 A"
    subtitle = "Rozdz. 3.22 i pytanie Q.26: oblicz SEM wsteczną; dlaczego jest tak ważna"
    params = [
        ("V", "Napięcie zasilania V [V]", 50, 500, 220, 5),
        ("Ra", "Rezystancja twornika R_a [Ω]", 0.05, 2.0, 0.5, 0.05),
        ("Ia", "Prąd twornika I_a [A]", 1, 100, 20, 1),
        ("Nn", "Prędkość w tym punkcie N [obr/min]", 200, 3000, 1000, 50),
    ]

    def solve(self, p, t):
        V, Ra, Ia = p["V"], p["Ra"], p["Ia"]
        Eb = V - Ia * Ra
        Ist = V / Ra
        t.h("O co chodzi? (wersja dla licealisty)")
        t.p("Wirujący twornik silnika to… także prądnica! Jego przewody przecinają pole, więc "
            "indukuje się w nich napięcie. Zgodnie z regułą Lenza przeciwdziała ono przyczynie - "
            "jest skierowane PRZECIWNIE do napięcia zasilania. Nazywamy je SEM wsteczną E_b.")
        t.p("Analogia: jedziesz rowerem z górki - im szybciej jedziesz, tym mniej musisz pedałować. "
            "Silnik im szybciej się kręci, tym mniej prądu pobiera.")
        t.h("Rozwiązanie krok po kroku")
        t.eq("V = E_b + I_a·R_a   →   E_b = V − I_a·R_a")
        t.eq(f"E_b = {fmt(V, 0)} − {fmt(Ia, 1)}·{fmt(Ra, 2)} = {fmt(Eb, 2)} V")
        t.res(f"SEM wsteczna E_b = {fmt(Eb, 2)} V")
        t.book("E_b = 210 V")
        t.p("Moc: V·I_a = E_b·I_a + I_a²·R_a, czyli moc pobrana = moc zamieniona na mechaniczną + straty:")
        t.eq(f"{fmt(V * Ia, 0)} W = {fmt(Eb * Ia, 0)} W (mechaniczna) + {fmt(Ia ** 2 * Ra, 0)} W (ciepło)")
        t.h("Komentarz inżyniera - dlaczego E_b jest tak ważna")
        t.eng(f"Przy rozruchu (N = 0) E_b = 0, więc prąd ograniczałaby TYLKO mała rezystancja "
              f"twornika: I = V/R_a = {fmt(Ist, 0)} A, czyli {fmt(Ist / Ia, 0)}× więcej niż teraz! "
              "Dlatego silniki prądu stałego potrzebują rozrusznika. E_b działa też jak automatyczny "
              "regulator: gdy obciążenie rośnie, silnik zwalnia, E_b maleje, prąd rośnie i moment "
              "rośnie aż do zrównoważenia obciążenia.")

    def draw(self, axs, p):
        V, Ra, Ia, Nn = p["V"], p["Ra"], p["Ia"], p["Nn"]
        Eb = V - Ia * Ra
        k = Eb / Nn
        N0 = V / k
        N = np.linspace(0, N0, 300)
        Iarr = (V - k * N) / Ra
        a1, a2 = axs
        line(a1, N, Iarr, C[7], "Prąd twornika I_a(N)")
        point(a1, 0, V / Ra, C[7], f"rozruch: {fmt(V / Ra, 0)} A", dx=10, dy=-14)
        point(a1, Nn, Ia, C[0], f"{fmt(Ia, 0)} A przy {fmt(Nn, 0)} obr/min", dy=12)
        style(a1, "I_a maleje przy rozpędzaniu", "Prędkość N [obr/min]", "I_a [A]")
        line(a2, N, k * N, C[0], "SEM wsteczna E_b")
        line(a2, N, V - k * N, C[1], "Spadek I_a·R_a")
        a2.axhline(V, color=MUTED, lw=1, ls=":")
        point(a2, Nn, Eb, C[0], f"E_b = {fmt(Eb, 1)} V", dx=-90, dy=10)
        style(a2, f"Podział napięcia V = {fmt(V, 0)} V", "Prędkość N [obr/min]", "Napięcie [V]")
        legend(a2, "center right")


class GenMotorPage(Page):
    group = "4. Silniki"
    title = "Ta sama maszyna jako prądnica i silnik (I_f = 2 A, I_L = 20 A, 200 V)"
    subtitle = "Zadanie z rozdz. 3.24: SEM przy pracy prądnicowej i moment przy 1500 obr/min jako silnik"
    params = [
        ("V", "Napięcie sieci V [V]", 100, 400, 200, 5),
        ("Ish", "Prąd wzbudzenia I_sh [A]", 0.5, 5, 2, 0.1),
        ("IL", "Prąd linii I_L [A]", 5, 60, 20, 1),
        ("Ra", "Rezystancja twornika R_a [Ω]", 0.05, 2.0, 0.5, 0.05),
        ("N", "Prędkość jako silnik N [obr/min]", 300, 3000, 1500, 50),
    ]

    def solve(self, p, t):
        V, Ish, IL, Ra, N = p["V"], p["Ish"], p["IL"], p["Ra"], p["N"]
        Ia_g, Ia_m = IL + Ish, IL - Ish
        Eg, Eb = V + Ia_g * Ra, V - Ia_m * Ra
        T = Eb * Ia_m / omega(N)
        t.h("O co chodzi? (wersja dla licealisty)")
        t.p("Maszyna prądu stałego jest 'dwukierunkowa': gdy ją napędzamy - oddaje prąd do sieci "
            "(prądnica), gdy ją zasilamy - kręci się (silnik). Różnica jest tylko w kierunku prądu "
            "twornika względem prądu linii:")
        t.bullet("prądnica: twornik oddaje prąd do linii I tak samo do bocznika → I_a = I_L + I_sh,")
        t.bullet("silnik: z linii płynie prąd do twornika i do bocznika → I_a = I_L − I_sh.")
        t.p("W skanie brakuje R_a; odpowiedź 211 V odpowiada R_a = 0,5 Ω - przyjęto tę wartość.")
        t.h("Rozwiązanie krok po kroku")
        t.p("i) Praca prądnicowa:")
        t.eq(f"I_a = {fmt(IL, 1)} + {fmt(Ish, 1)} = {fmt(Ia_g, 1)} A")
        t.eq(f"E = V + I_a·R_a = {fmt(V, 0)} + {fmt(Ia_g, 1)}·{fmt(Ra, 2)} = {fmt(Eg, 2)} V")
        t.p("ii) Praca silnikowa:")
        t.eq(f"I_a = {fmt(IL, 1)} − {fmt(Ish, 1)} = {fmt(Ia_m, 1)} A")
        t.eq(f"E_b = V − I_a·R_a = {fmt(V, 0)} − {fmt(Ia_m, 1)}·{fmt(Ra, 2)} = {fmt(Eb, 2)} V")
        t.eq(f"ω = 2π·N/60 = 2π·{fmt(N, 0)}/60 = {fmt(omega(N), 3)} rad/s")
        t.eq(f"T = E_b·I_a / ω = {fmt(Eb, 1)}·{fmt(Ia_m, 1)} / {fmt(omega(N), 3)} = {fmt(T, 3)} N·m")
        t.res(f"Prądnica: E = {fmt(Eg, 1)} V;   silnik: T = {fmt(T, 3)} N·m")
        t.book("211 V oraz 21,887 N·m")
        t.h("Komentarz inżyniera")
        t.eng("Moment liczymy z bilansu mocy: moc 'przerobiona' w tworniku E_b·I_a = T·ω. "
              "Przydatny wzór skrócony: T = 9,55·E_b·I_a/N (bo 60/2π ≈ 9,55). "
              "Jako prądnica E > V, jako silnik E_b < V - to E 'decyduje', w którą stronę płynie energia.")

    def draw(self, axs, p):
        V, Ish, IL, Ra, N = p["V"], p["Ish"], p["IL"], p["Ra"], p["N"]
        ILs = np.linspace(Ish, 60, 200)
        a1, a2 = axs
        line(a1, ILs, V + (ILs + Ish) * Ra, C[0], "Prądnica: E = V + I_a·R_a")
        line(a1, ILs, V - (ILs - Ish) * Ra, C[1], "Silnik: E_b = V − I_a·R_a")
        a1.axhline(V, color=MUTED, lw=1, ls=":")
        point(a1, IL, V + (IL + Ish) * Ra, C[0], f"{fmt(V + (IL + Ish) * Ra, 1)} V")
        point(a1, IL, V - (IL - Ish) * Ra, C[1], f"{fmt(V - (IL - Ish) * Ra, 1)} V", dy=-18)
        style(a1, "SEM: prądnica vs silnik", "Prąd linii I_L [A]", "E [V]")
        legend(a1, "upper left")
        Ns = np.linspace(300, 3000, 200)
        Ia_m = IL - Ish
        Eb = V - Ia_m * Ra
        line(a2, Ns, Eb * Ia_m / omega(Ns), C[2], "T = E_b·I_a/ω (ta sama moc)")
        point(a2, N, Eb * Ia_m / omega(N), C[2], f"{fmt(Eb * Ia_m / omega(N), 2)} N·m")
        style(a2, "Moment przy tej samej mocy", "Prędkość N [obr/min]", "Moment T [N·m]")


class Ex3251(Page):
    group = "4. Silniki"
    title = "Przykład 3.25.1 - silnik bocznikowy 200 V, 100 A, 750 obr/min"
    subtitle = "R_sh = 40 Ω, R_a = 0,1 Ω - oblicz moment rozwijany przez twornik"
    params = [
        ("V", "Napięcie zasilania V [V]", 100, 400, 200, 5),
        ("IL", "Prąd linii I_L [A]", 10, 200, 100, 1),
        ("Rsh", "Rezystancja bocznika R_sh [Ω]", 10, 200, 40, 1),
        ("Ra", "Rezystancja twornika R_a [Ω]", 0.02, 0.5, 0.1, 0.01),
        ("N", "Prędkość N [obr/min]", 200, 2000, 750, 10),
    ]

    def solve(self, p, t):
        V, IL, Rsh, Ra, N = p["V"], p["IL"], p["Rsh"], p["Ra"], p["N"]
        Ish = V / Rsh
        Ia = IL - Ish
        Eb = V - Ia * Ra
        T = Eb * Ia / omega(N)
        t.h("O co chodzi? (wersja dla licealisty)")
        t.p("Silnik bocznikowy ma uzwojenie wzbudzenia podłączone wprost do sieci. Prąd z sieci "
            "dzieli się: mała część płynie przez wzbudzenie (tworzy pole), reszta przez twornik "
            "(tworzy moment). Moment to 'siła obrotowa' - jak siła, z jaką kręcisz kluczem.")
        t.h("Rozwiązanie krok po kroku")
        t.eq(f"I_sh = V / R_sh = {fmt(V, 0)} / {fmt(Rsh, 0)} = {fmt(Ish, 3)} A")
        t.eq(f"I_a = I_L − I_sh = {fmt(IL, 0)} − {fmt(Ish, 3)} = {fmt(Ia, 3)} A")
        t.eq(f"E_b = V − I_a·R_a = {fmt(V, 0)} − {fmt(Ia, 3)}·{fmt(Ra, 2)} = {fmt(Eb, 3)} V")
        t.eq(f"ω = 2π·{fmt(N, 0)}/60 = {fmt(omega(N), 3)} rad/s")
        t.eq(f"T_a = E_b·I_a/ω = {fmt(Eb, 2)}·{fmt(Ia, 2)}/{fmt(omega(N), 3)} = {fmt(T, 3)} N·m")
        t.res(f"Moment rozwijany przez twornik T_a = {fmt(T, 2)} N·m")
        t.book("T_a = 230,424 N·m (I_sh = 5 A, I_a = 95 A, E_b = 190,5 V)")
        t.h("Komentarz inżyniera")
        t.eng("To moment ELEKTROMAGNETYCZNY (rozwijany przez twornik). Na wale jest nieco mniej, "
              "bo część zużywa się na tarcie, wentylację i straty w żelazie (moment strat T_f). "
              "W silniku bocznikowym strumień jest stały, więc T ∝ I_a (prosta na lewym wykresie), "
              "a prędkość spada tylko nieznacznie z obciążeniem (prawy wykres) - "
              "'silnik o prawie stałej prędkości': wentylatory, pompy, obrabiarki.")

    def draw(self, axs, p):
        V, IL, Rsh, Ra, N = p["V"], p["IL"], p["Rsh"], p["Ra"], p["N"]
        Ish = V / Rsh
        Ia = IL - Ish
        Eb = V - Ia * Ra
        kphi = Eb / omega(N)
        I = np.linspace(0, 2 * Ia, 200)
        a1, a2 = axs
        line(a1, I, kphi * I, C[0], "T = kφ·I_a")
        point(a1, Ia, kphi * Ia, C[0], f"{fmt(kphi * Ia, 1)} N·m")
        style(a1, "T(I_a) przy φ = const", "Prąd twornika I_a [A]", "Moment T [N·m]")
        Ts = kphi * I
        Ns = (V - I * Ra) / kphi * 60 / TWO_PI
        line(a2, Ts, Ns, C[2], "N(T)")
        point(a2, kphi * Ia, N, C[2], f"{fmt(N, 0)} obr/min", dy=-18)
        a2.set_ylim(0, max(Ns) * 1.1)
        style(a2, "N(T): prawie płaska", "Moment T [N·m]", "Prędkość N [obr/min]")


class Ex3261(Page):
    group = "4. Silniki"
    title = "Przykład 3.26.1 - wpływ reakcji twornika na prędkość"
    subtitle = "Silnik bocznikowy 230 V: bieg jałowy 3,3 A / 1000 obr/min, pełne obciążenie 40 A, strumień −4 %"
    params = [
        ("V", "Napięcie V [V]", 100, 400, 230, 5),
        ("Ia0", "Prąd twornika na biegu jałowym I_a0 [A]", 0.5, 10, 3.3, 0.1),
        ("N0", "Prędkość biegu jałowego N₀ [obr/min]", 300, 2000, 1000, 10),
        ("Ra", "Rezystancja twornika R_a [Ω]", 0.05, 1.0, 0.3, 0.05),
        ("Rsh", "Rezystancja bocznika R_sh [Ω]", 50, 400, 160, 5),
        ("IL1", "Prąd linii przy pełnym obciążeniu I_L1 [A]", 5, 100, 40, 1),
        ("w", "Osłabienie strumienia przez reakcję twornika [%]", 0, 15, 4, 0.5),
    ]

    def calc(self, p):
        V, Ia0, N0, Ra, Rsh, IL1, w = (p["V"], p["Ia0"], p["N0"], p["Ra"], p["Rsh"],
                                       p["IL1"], p["w"] / 100)
        Ish = V / Rsh
        Ia1 = IL1 - Ish
        Eb0 = V - Ia0 * Ra
        Eb1 = V - Ia1 * Ra
        N1 = N0 * (Eb1 / Eb0) / (1 - w)
        return Ish, Ia1, Eb0, Eb1, N1

    def solve(self, p, t):
        Ish, Ia1, Eb0, Eb1, N1 = self.calc(p)
        w = p["w"] / 100
        T1 = Eb1 * Ia1 / omega(N1)
        reg = (p["N0"] - N1) / N1 * 100
        t.h("O co chodzi? (wersja dla licealisty)")
        t.p("Prędkość silnika rośnie z napięciem E_b i MALEJE ze strumieniem: N ∝ E_b/φ. "
            "Czyli: im słabsze pole, tym szybciej silnik musi się kręcić, żeby wytworzyć tę samą "
            "SEM wsteczną.")
        t.p("Reakcja twornika - pole od prądu twornika 'zniekształca' i nieco osłabia pole główne. "
            "Przy obciążeniu strumień maleje (tu o " + fmt(p["w"], 1) + " %), co PODNOSI prędkość "
            "i częściowo kompensuje spadek E_b.")
        t.h("Rozwiązanie krok po kroku")
        t.eq(f"E_b0 = V − I_a0·R_a = {fmt(p['V'], 0)} − {fmt(p['Ia0'], 2)}·{fmt(p['Ra'], 2)} = {fmt(Eb0, 3)} V")
        t.eq(f"I_sh = V/R_sh = {fmt(p['V'], 0)}/{fmt(p['Rsh'], 0)} = {fmt(Ish, 4)} A")
        t.eq(f"I_a1 = I_L1 − I_sh = {fmt(p['IL1'], 0)} − {fmt(Ish, 4)} = {fmt(Ia1, 4)} A")
        t.eq(f"E_b1 = V − I_a1·R_a = {fmt(Eb1, 4)} V")
        t.eq(f"φ₁ = (1 − {fmt(w, 2)})·φ₀ = {fmt(1 - w, 2)}·φ₀")
        t.eq("N₁/N₀ = (E_b1/E_b0)·(φ₀/φ₁)")
        t.eq(f"N₁ = {fmt(p['N0'], 0)}·({fmt(Eb1, 3)}/{fmt(Eb0, 3)})/{fmt(1 - w, 2)} = {fmt(N1, 2)} obr/min")
        t.eq(f"T₁ = E_b1·I_a1/ω₁ = {fmt(T1, 2)} N·m")
        t.eq(f"Regulacja prędkości = (N₀ − N₁)/N₁·100 % = {fmt(reg, 2)} %")
        t.res(f"Prędkość przy pełnym obciążeniu N₁ = {fmt(N1, 1)} obr/min, moment {fmt(T1, 1)} N·m")
        t.book("E_b0 = 229,01 V, I_a1 = 38,5625 A, E_b1 = 218,4312 V, N₁ ≈ 993,5 obr/min")
        t.h("Komentarz inżyniera")
        t.eng("Ustaw osłabienie na 0 % - prędkość spadnie mocniej. Przy ok. 5 % prędkość wcale nie "
              "spada, a przy większym osłabieniu ROŚNIE z obciążeniem - to niebezpieczne "
              "(niestabilność), dlatego duże maszyny mają uzwojenia kompensacyjne i bieguny "
              "komutacyjne ograniczające reakcję twornika.")

    def draw(self, axs, p):
        V, Ia0, N0, Ra = p["V"], p["Ia0"], p["N0"], p["Ra"]
        Ish, Ia1, Eb0, Eb1, N1 = self.calc(p)
        w = p["w"] / 100
        I = np.linspace(Ia0, 1.6 * Ia1, 200)
        frac = (I - Ia0) / max(Ia1 - Ia0, 1e-9)
        Eb = V - I * Ra
        N_no = N0 * Eb / Eb0
        N_ar = N0 * Eb / Eb0 / (1 - w * frac)
        a1, a2 = axs
        line(a1, I, N_no, C[0], "Bez reakcji twornika")
        line(a1, I, N_ar, C[1], f"Z reakcją twornika ({fmt(p['w'], 1)} % przy I_a1)")
        point(a1, Ia1, N1, C[1], f"{fmt(N1, 1)} obr/min")
        point(a1, Ia0, N0, INK, "bieg jałowy", dy=-18)
        style(a1, "Prędkość - prąd twornika", "I_a [A]", "N [obr/min]")
        legend(a1, "lower left")
        T_no = Eb * I / omega(N_no)
        T_ar = Eb * I / omega(N_ar)
        line(a2, T_no, N_no, C[0], "Bez reakcji twornika")
        line(a2, T_ar, N_ar, C[1], "Z reakcją twornika")
        style(a2, "Prędkość - moment", "Moment T [N·m]", "N [obr/min]")
        legend(a2, "lower left")


class MotorCharPage(Page):
    group = "4. Silniki"
    title = "Charakterystyki silników: bocznikowy, szeregowy, kompound"
    subtitle = "Rozdz. 3.27–3.31: T(I_a), N(I_a), N(T) i zastosowania"
    layout = (1, 3)
    params = [
        ("V", "Napięcie V [V]", 110, 440, 220, 10),
        ("In", "Prąd znamionowy twornika I_n [A]", 10, 200, 50, 5),
        ("Nn", "Prędkość znamionowa N_n [obr/min]", 500, 3000, 1000, 50),
        ("R", "Rezystancja obwodu twornika [Ω]", 0.05, 1.0, 0.3, 0.05),
        ("sat", "Nasycenie obwodu magnetycznego", 0.0, 2.0, 0.6, 0.1),
        ("kc", "Udział uzw. szeregowego w kompoundzie", 0.05, 0.45, 0.3, 0.05),
    ]

    def model(self, p):
        V, In, Nn, R, s, kc = p["V"], p["In"], p["Nn"], p["R"], p["sat"], p["kc"]
        kphi_n = (V - In * R) / omega(Nn)          # kφ w punkcie znamionowym
        x = np.linspace(0.03, 2.0, 400)            # I_a / I_n
        sat = lambda F: F / (1 + s * F)
        phis = {
            "Bocznikowy": np.ones_like(x),
            "Szeregowy": sat(x) / sat(1.0),
            "Kompound zgodny": sat(1 + kc * x) / sat(1 + kc),
            "Kompound przeciwny": sat(np.maximum(1 - kc * x, 0.02)) / sat(1 - kc),
        }
        res = {}
        for name, ph in phis.items():
            Ia = x * In
            kphi = kphi_n * ph
            T = kphi * Ia
            N = (V - Ia * R) / kphi * 60 / TWO_PI
            res[name] = (Ia, T, N)
        return res

    def solve(self, p, t):
        t.h("O co chodzi? (wersja dla licealisty)")
        t.p("Dwa wzory tłumaczą wszystko:  T ∝ φ·I_a  (moment)  oraz  N ∝ (V − I_a·R_a)/φ  (prędkość).")
        t.bullet("Bocznikowy: φ = const → moment rośnie liniowo z prądem, prędkość prawie stała.")
        t.bullet("Szeregowy: φ ∝ I_a → moment ∝ I_a² (parabola - ogromny moment rozruchowy), "
                 "a prędkość ∝ 1/I_a - przy małym obciążeniu silnik 'ucieka' do niebezpiecznych "
                 "prędkości. NIGDY nie uruchamiaj silnika szeregowego bez obciążenia i nie łącz go "
                 "z obciążeniem paskiem (pasek może spaść)!")
        t.bullet("Kompound zgodny: kompromis - duży moment rozruchowy, ale bezpieczna prędkość biegu jałowego.")
        t.bullet("Kompound przeciwny: strumień maleje z obciążeniem → prędkość ROŚNIE z obciążeniem - "
                 "niestabilny, praktycznie nieużywany.")
        t.p("Po nasyceniu żelaza strumień silnika szeregowego przestaje rosnąć, więc parabola "
            "T(I_a) przechodzi w prostą (zwiększ suwak 'nasycenie').")
        res = self.model(p)
        t.h("Wyniki modelu przy 2× prądzie znamionowym")
        Tn = res["Bocznikowy"][1][np.argmin(abs(res['Bocznikowy'][0] - p['In']))]
        for name, (Ia, T, N) in res.items():
            t.eq(f"{name:<19s}: T = {fmt(T[-1], 0)} N·m ({fmt(T[-1] / Tn, 2)}·T_n),  N = {fmt(N[-1], 0)} obr/min")
        t.h("Zastosowania (tabela 3.31.1)")
        t.bullet("Bocznikowy - wentylatory, dmuchawy, pompy odśrodkowe, tokarki, frezarki, obrabiarki.")
        t.bullet("Szeregowy - dźwigi, wciągarki, windy, trolejbusy, tramwaje, lokomotywy elektryczne.")
        t.bullet("Kompound zgodny - walcarki, prasy, nożyce, ciężkie strugarki (obciążenia udarowe).")
        t.bullet("Kompound przeciwny - brak praktycznych zastosowań.")
        t.h("Komentarz inżyniera")
        t.eng("Wszystkie krzywe przechodzą przez ten sam punkt znamionowy - tak łatwiej je porównać. "
              "Najedź myszą na krzywą, by odczytać wartości. Na wykresie N(T) od razu widać, który "
              "silnik 'trzyma' prędkość, a który 'ustępuje' pod obciążeniem (jak koń pociągowy).")

    def draw(self, axs, p):
        res = self.model(p)
        a1, a2, a3 = axs
        for i, (name, (Ia, T, N)) in enumerate(res.items()):
            line(a1, Ia, T, C[i], name)
            line(a2, Ia, N, C[i], name)
            line(a3, T, N, C[i], name)
        style(a1, "(a) T w funkcji I_a", "I_a [A]", "T [N·m]")
        style(a2, "(b) N w funkcji I_a", "I_a [A]", "N [obr/min]")
        style(a3, "(c) N w funkcji T", "T [N·m]", "N [obr/min]")
        for a in (a2, a3):
            a.set_ylim(0, 3.0 * p["Nn"])
        legend(a1, "upper left")


# =============================================================================
# 5. ROZRUCH
# =============================================================================
class StarterPage(Page):
    group = "5. Rozruch i regulacja prędkości"
    title = "Rozruch silnika - rozrusznik 3-/4-punktowy"
    subtitle = "Rozdz. 3.32: prąd rozruchowy bez rozrusznika, dobór stopni rezystancji, przebiegi czasowe"
    params = [
        ("V", "Napięcie V [V]", 110, 500, 250, 10),
        ("Ra", "Rezystancja twornika R_a [Ω]", 0.1, 2.0, 0.5, 0.05),
        ("In", "Prąd znamionowy I_n [A]", 5, 100, 40, 1),
        ("kmax", "Prąd maks. przy rozruchu [× I_n]", 1.2, 3.0, 2.0, 0.1),
        ("kmin", "Prąd przełączenia stopnia [× I_n]", 1.0, 2.0, 1.2, 0.05),
        ("J", "Moment bezwładności J [kg·m²]", 0.1, 5.0, 1.0, 0.1),
        ("load", "Obciążenie [× T_n]", 0.0, 1.0, 0.5, 0.05),
    ]

    def design(self, p):
        V, Ra, In = p["V"], p["Ra"], p["In"]
        Imax, Imin = p["kmax"] * In, min(p["kmin"], p["kmax"] - 0.05) * In
        Rs = [V / Imax]
        while Rs[-1] * Imin / Imax > Ra and len(Rs) < 30:
            Rs.append(Rs[-1] * Imin / Imax)
        Rs.append(Ra)
        return Imax, Imin, Rs

    def simulate(self, p, Rs, Imin, tmax=None):
        V, Ra, In, J = p["V"], p["Ra"], p["In"], p["J"]
        Nn = 1000.0
        kphi = (V - In * Ra) / omega(Nn)
        TL = p["load"] * kphi * In
        dt = 5e-4
        tmax = tmax or 20.0
        n = int(tmax / dt)
        w = 0.0
        stage = 0
        ts, Is, Ns = [], [], []
        for k in range(n):
            R = Rs[stage]
            I = (V - kphi * w) / R
            if stage < len(Rs) - 1 and I <= Imin:
                stage += 1
                R = Rs[stage]
                I = (V - kphi * w) / R
            Te = kphi * I
            acc = (Te - TL) / J
            if w <= 0 and acc < 0:      # moment rozruchowy za mały - wał stoi
                acc = 0.0
            w = max(w + dt * acc, 0.0)
            if stage == len(Rs) - 1 and k * dt > 0.5 and abs(acc) < 0.05:
                ts.append(k * dt)
                Is.append(I)
                Ns.append(w * 60 / TWO_PI)
                break                    # stan ustalony osiągnięty
            if k % 4 == 0:
                ts.append(k * dt)
                Is.append(I)
                Ns.append(w * 60 / TWO_PI)
        return np.array(ts), np.array(Is), np.array(Ns)

    def solve(self, p, t):
        V, Ra, In = p["V"], p["Ra"], p["In"]
        Imax, Imin, Rs = self.design(p)
        t.h("O co chodzi? (wersja dla licealisty)")
        t.p("W chwili startu silnik stoi, więc SEM wsteczna E_b = 0. Prąd ogranicza tylko "
            "bardzo mała rezystancja twornika - to prawie zwarcie! Taki prąd spaliłby komutator "
            "i szczotki, a gwałtowne szarpnięcie mogłoby uszkodzić mechanizmy.")
        t.h("Rozwiązanie krok po kroku")
        t.eq(f"I_rozr (bez rozrusznika) = V / R_a = {fmt(V, 0)} / {fmt(Ra, 2)} = {fmt(V / Ra, 0)} A"
             f"  = {fmt(V / Ra / In, 1)} × I_n")
        t.p("Rozrusznik to rezystor włączony szeregowo z twornikiem, odłączany stopniami w miarę "
            "rozpędzania się silnika (rośnie E_b, więc rezystancję można zmniejszać).")
        t.eq(f"Rezystancja całkowita na 1. stopniu: R₁ = V/I_max = {fmt(V, 0)}/{fmt(Imax, 1)} = {fmt(Rs[0], 3)} Ω")
        t.eq(f"Rezystancja zewnętrzna rozrusznika: R₁ − R_a = {fmt(Rs[0] - Ra, 3)} Ω")
        t.p("Gdy prąd spadnie do I_min, przechodzimy na następny styk. Kolejne rezystancje tworzą "
            "ciąg geometryczny o ilorazie I_min/I_max:")
        for i, R in enumerate(Rs):
            label = "praca (sam twornik)" if i == len(Rs) - 1 else f"styk {i + 1}"
            t.eq(f"{label:<20s}: R = {fmt(R, 3)} Ω")
        t.res(f"Potrzeba {len(Rs) - 1} sekcji rezystora; prąd szczytowy {fmt(Imax, 0)} A zamiast {fmt(V / Ra, 0)} A")
        t.h("Rozrusznik 3-punktowy i 4-punktowy")
        t.bullet("NVC (cewka zanikowa, 'hold-on') trzyma rączkę w pozycji RUN. Gdy zaniknie "
                 "napięcie, rączka wraca sprężyną do OFF - silnik nie ruszy sam bez rozrusznika "
                 "po powrocie zasilania.")
        t.bullet("OLR (wyzwalacz nadprądowy) przy przeciążeniu zwiera cewkę NVC i rozłącza silnik.")
        t.bullet("3-punktowy (L, F, A): NVC w szereg z polem - przerwa w obwodzie wzbudzenia "
                 "też wyłącza silnik (ochrona przed 'rozbieganiem'), ale regulacja prędkości "
                 "przez osłabienie pola może przypadkowo zwolnić rączkę.")
        t.bullet("4-punktowy (L, N, F, A): NVC zasilana osobno - można swobodnie osłabiać pole, "
                 "ale brak ochrony przed przerwą w obwodzie wzbudzenia.")
        t.h("Komentarz inżyniera")
        t.eng("Zwiększ I_max - mniej stopni, szybszy rozruch, ale większe udary. Zwiększ J lub "
              "obciążenie - rozruch trwa dłużej. Dziś zamiast rezystorów stosuje się przekształtniki "
              "tyrystorowe/tranzystorowe z ograniczeniem prądu - zasada pozostaje ta sama.")

    def draw(self, axs, p):
        Imax, Imin, Rs = self.design(p)
        V, Ra = p["V"], p["Ra"]
        ts, Is, Ns = self.simulate(p, Rs, Imin)
        ts2, Is2, Ns2 = self.simulate(p, [Ra], 0)
        a1, a2 = axs
        line(a1, ts, Is, C[0], "Z rozrusznikiem")
        line(a1, ts2, Is2, C[7], "Bez rozrusznika (poza skalą)", ls="--")
        a1.axhline(Imax, color=MUTED, lw=1, ls=":")
        a1.axhline(Imin, color=MUTED, lw=1, ls=":")
        a1.text(0.0, Imax, " I_max", ha="left", va="bottom", fontsize=8, color=INK2)
        a1.text(0.0, Imin, " I_min", ha="left", va="bottom", fontsize=8, color=INK2)
        a1.set_ylim(0, Imax * 1.35)
        a1.text(0.98, 0.97, f"szczyt bez rozrusznika: {fmt(V / Ra, 0)} A", transform=a1.transAxes,
                fontsize=8.5, color=C[7], va="top", ha="right")
        # okno czasowe dopasowane do czasu rozruchu (98 % prędkości ustalonej)
        done = np.where(Ns >= 0.98 * Ns[-1])[0]
        t_end = ts[done[0]] if len(done) and Ns[-1] > 0 else ts[-1]
        for a in (a1, a2):
            a.set_xlim(0, min(max(1.6 * t_end, 0.2), ts[-1]))
        style(a1, "Prąd twornika podczas rozruchu", "Czas t [s]", "I_a [A]")
        a1.legend(loc="upper right", bbox_to_anchor=(1.0, 0.92), fontsize=8, edgecolor="#d6d5d0")
        line(a2, ts, Ns, C[0], "Z rozrusznikiem")
        line(a2, ts2, Ns2, C[7], "Bez rozrusznika", ls="--")
        style(a2, "Prędkość podczas rozruchu", "Czas t [s]", "N [obr/min]")
        legend(a2, "lower right")


# =============================================================================
# 6. REGULACJA PRĘDKOŚCI
# =============================================================================
class Ex3341(Page):
    group = "5. Rozruch i regulacja prędkości"
    title = "Przykład 3.34.1 - rezystor w obwodzie twornika (silnik bocznikowy)"
    subtitle = "230 V, R_a = 0,4 Ω, 500 obr/min przy I_a = 30 A; dodano R = 1,1 Ω - prędkość przy T_n i 1,5·T_n"
    params = [
        ("V", "Napięcie V [V]", 100, 400, 230, 5),
        ("Ra", "Rezystancja twornika R_a [Ω]", 0.05, 1.5, 0.4, 0.05),
        ("N1", "Prędkość przy pełnym obciążeniu N₁ [obr/min]", 200, 2000, 500, 10),
        ("Ia1", "Prąd twornika przy pełnym obciążeniu [A]", 5, 100, 30, 1),
        ("Rx", "Dodatkowa rezystancja R_x [Ω]", 0.0, 5.0, 1.1, 0.1),
        ("k", "Moment obciążenia [× T_n]", 0.2, 2.0, 1.5, 0.1),
    ]

    def solve(self, p, t):
        V, Ra, N1, Ia1, Rx, k = p["V"], p["Ra"], p["N1"], p["Ia1"], p["Rx"], p["k"]
        Eb1 = V - Ia1 * Ra
        Eb2 = V - Ia1 * (Ra + Rx)
        N2 = N1 * Eb2 / Eb1
        Ia3 = k * Ia1
        Eb3 = V - Ia3 * (Ra + Rx)
        N3 = N1 * Eb3 / Eb1
        t.h("O co chodzi? (wersja dla licealisty)")
        t.p("Prędkość silnika jest proporcjonalna do napięcia, które 'dociera' do twornika "
            "(dokładniej do E_b). Wstawiając rezystor szeregowo, 'zabieramy' część napięcia - "
            "silnik zwalnia. Jak przyciśnięcie wężyka ogrodowego.")
        t.p("Strumień jest stały (bocznik zasilany wprost z sieci), więc moment T ∝ I_a. "
            "Ten sam moment = ten sam prąd, niezależnie od rezystora!")
        t.h("Rozwiązanie krok po kroku")
        t.eq(f"Bez R_x:  E_b1 = V − I_a·R_a = {fmt(V, 0)} − {fmt(Ia1, 0)}·{fmt(Ra, 2)} = {fmt(Eb1, 2)} V")
        t.p("i) Ten sam moment (pełne obciążenie) → I_a2 = I_a1:")
        t.eq(f"E_b2 = V − I_a·(R_a + R_x) = {fmt(V, 0)} − {fmt(Ia1, 0)}·({fmt(Ra, 2)} + {fmt(Rx, 2)}) = {fmt(Eb2, 2)} V")
        t.eq(f"N₂ = N₁·E_b2/E_b1 = {fmt(N1, 0)}·{fmt(Eb2, 2)}/{fmt(Eb1, 2)} = {fmt(N2, 2)} obr/min")
        t.p(f"ii) Moment {fmt(k, 2)}× większy → prąd {fmt(k, 2)}× większy:")
        t.eq(f"I_a3 = {fmt(k, 2)}·{fmt(Ia1, 0)} = {fmt(Ia3, 2)} A")
        t.eq(f"E_b3 = {fmt(V, 0)} − {fmt(Ia3, 2)}·{fmt(Ra + Rx, 2)} = {fmt(Eb3, 2)} V")
        t.eq(f"N₃ = {fmt(N1, 0)}·{fmt(Eb3, 2)}/{fmt(Eb1, 2)} = {fmt(N3, 2)} obr/min")
        t.res(f"N (pełny moment) = {fmt(N2, 1)} obr/min,  N ({fmt(k, 1)}·T_n) = {fmt(N3, 1)} obr/min")
        t.book("≈ 424,3 obr/min oraz ≈ 372,7 obr/min")
        loss = Ia1 ** 2 * Rx
        t.h("Komentarz inżyniera")
        t.eng(f"Metoda jest prosta, ale rozrzutna: w rezystorze przy pełnym obciążeniu marnuje się "
              f"{fmt(loss, 0)} W, czyli {fmt(loss / (V * Ia1) * 100, 1)} % mocy pobieranej przez twornik "
              "(prawy wykres). Do tego prędkość silnie zależy od obciążenia (bardziej stroma "
              "charakterystyka). Pozwala tylko ZMNIEJSZAĆ prędkość poniżej znamionowej.")

    def draw(self, axs, p):
        V, Ra, N1, Ia1, Rx, k = p["V"], p["Ra"], p["N1"], p["Ia1"], p["Rx"], p["k"]
        Eb1 = V - Ia1 * Ra
        a1, a2 = axs
        x = np.linspace(0, 2.0, 200)  # T/T_n = I_a/I_a1
        for i, R in enumerate([0.0, Rx, 2 * Rx]):
            N = N1 * (V - x * Ia1 * (Ra + R)) / Eb1
            line(a1, x, N, C[i], f"R_x = {fmt(R, 2)} Ω")
        N2 = N1 * (V - Ia1 * (Ra + Rx)) / Eb1
        N3 = N1 * (V - k * Ia1 * (Ra + Rx)) / Eb1
        point(a1, 1.0, N2, C[1], f"{fmt(N2, 1)}")
        point(a1, k, N3, C[1], f"{fmt(N3, 1)}", dy=-18)
        a1.set_ylim(0, None)
        style(a1, "Prędkość - moment dla różnych R_x", "Moment T / T_n", "N [obr/min]")
        legend(a1, "lower left")
        cases = [("bez R_x", 0.0), (f"R_x = {fmt(Rx, 1)} Ω", Rx)]
        for i, (_, R) in enumerate(cases):
            Ia = Ia1
            Pm = (V - Ia * (Ra + R)) * Ia
            La = Ia ** 2 * Ra
            Lx = Ia ** 2 * R
            a2.bar(i, Pm, color=C[0], width=0.55, edgecolor=SURFACE, linewidth=2,
                   label="Moc mechaniczna" if i == 0 else "_")
            a2.bar(i, La, bottom=Pm, color=C[3], width=0.55, edgecolor=SURFACE, linewidth=2,
                   label="Straty w R_a" if i == 0 else "_")
            a2.bar(i, Lx, bottom=Pm + La, color=C[7], width=0.55, edgecolor=SURFACE, linewidth=2,
                   label="Straty w R_x" if i == 0 else "_")
            a2.text(i, Pm / 2, f"{fmt(Pm / (V * Ia) * 100, 0)} %", ha="center", color="white", fontsize=9)
        a2.set_xticks([0, 1])
        a2.set_xticklabels([c[0] for c in cases])
        style(a2, "Bilans mocy przy pełnym momencie", "", "Moc [W]")
        legend(a2, "lower right")


class FieldControlPage(Page):
    group = "5. Rozruch i regulacja prędkości"
    title = "Regulacja strumieniem - rezystor w obwodzie wzbudzenia (800 → 1000 obr/min)"
    subtitle = "Silnik bocznikowy 230 V (rys. w rozdz. 3.34): 800 obr/min przy I_a = 50 A; ile Ω dodać, by mieć 1000 obr/min przy 80 A?"
    params = [
        ("V", "Napięcie V [V]", 100, 400, 230, 5),
        ("Ra", "Rezystancja twornika R_a [Ω]", 0.05, 1.0, 0.15, 0.01),
        ("Rsh", "Rezystancja uzw. bocznikowego R_sh [Ω]", 50, 500, 250, 5),
        ("N1", "Prędkość początkowa N₁ [obr/min]", 300, 2000, 800, 10),
        ("Ia1", "Prąd twornika I_a1 [A]", 5, 150, 50, 1),
        ("N2", "Prędkość żądana N₂ [obr/min]", 300, 2500, 1000, 10),
        ("Ia2", "Prąd twornika I_a2 [A]", 5, 150, 80, 1),
    ]

    def calc(self, p):
        V, Ra, Rsh, N1, Ia1, N2, Ia2 = (p[k] for k in ("V", "Ra", "Rsh", "N1", "Ia1", "N2", "Ia2"))
        Ish1 = V / Rsh
        Eb1, Eb2 = V - Ia1 * Ra, V - Ia2 * Ra
        Ish2 = Ish1 * (Eb2 / Eb1) * (N1 / N2)
        Rtot = V / Ish2
        return Ish1, Eb1, Eb2, Ish2, Rtot, Rtot - Rsh

    def solve(self, p, t):
        Ish1, Eb1, Eb2, Ish2, Rtot, Radd = self.calc(p)
        t.h("O co chodzi? (wersja dla licealisty)")
        t.p("Z wzoru N ∝ E_b/φ wynika, że OSŁABIAJĄC pole, silnik przyspiesza (musi kręcić się "
            "szybciej, by wytworzyć tę samą SEM wsteczną). Pole osłabiamy, dodając rezystor "
            "(reostat polowy) szeregowo z uzwojeniem bocznikowym - zmniejsza się prąd wzbudzenia.")
        t.p("Uwaga: dane tego zadania w skanie są nieczytelne - odtworzono je z rysunku "
            "(230 V, R_sh = 250 Ω, 50 A/800 obr/min → 80 A/1000 obr/min) wg klasycznej wersji "
            "zadania z R_a = 0,15 Ω. Założenie: strumień ∝ prąd wzbudzenia.")
        t.h("Rozwiązanie krok po kroku")
        t.eq(f"I_sh1 = V/R_sh = {fmt(p['V'], 0)}/{fmt(p['Rsh'], 0)} = {fmt(Ish1, 4)} A")
        t.eq(f"E_b1 = {fmt(p['V'], 0)} − {fmt(p['Ia1'], 0)}·{fmt(p['Ra'], 2)} = {fmt(Eb1, 2)} V")
        t.eq(f"E_b2 = {fmt(p['V'], 0)} − {fmt(p['Ia2'], 0)}·{fmt(p['Ra'], 2)} = {fmt(Eb2, 2)} V")
        t.eq("N₂/N₁ = (E_b2/E_b1)·(I_sh1/I_sh2)  →  I_sh2 = I_sh1·(E_b2/E_b1)·(N₁/N₂)")
        t.eq(f"I_sh2 = {fmt(Ish1, 4)}·({fmt(Eb2, 1)}/{fmt(Eb1, 1)})·({fmt(p['N1'], 0)}/{fmt(p['N2'], 0)}) = {fmt(Ish2, 4)} A")
        t.eq(f"R_całk = V/I_sh2 = {fmt(Rtot, 2)} Ω  →  R_dod = {fmt(Rtot, 2)} − {fmt(p['Rsh'], 0)} = {fmt(Radd, 2)} Ω")
        if Radd < 0:
            t.res("Wynik ujemny - żądanej prędkości nie da się osiągnąć osłabianiem pola "
                  "(trzeba by pole WZMOCNIĆ, a to niemożliwe powyżej wartości znamionowej).")
        else:
            t.res(f"Należy dodać ok. {fmt(Radd, 1)} Ω w obwodzie wzbudzenia")
        t.h("Komentarz inżyniera")
        t.eng("Zalety: rezystor polowy przewodzi mały prąd (tu < 1 A), więc straty są znikome - "
              "metoda tania i sprawna. Wady: pozwala tylko ZWIĘKSZAĆ prędkość powyżej znamionowej, "
              "a zbyt słabe pole pogarsza komutację (iskrzenie). Przy przerwie w obwodzie "
              "wzbudzenia silnik może się rozbiegać - stąd ochrona w rozruszniku 3-punktowym.")

    def draw(self, axs, p):
        V, Ra, Rsh = p["V"], p["Ra"], p["Rsh"]
        Ish1, Eb1, Eb2, Ish2, Rtot, Radd = self.calc(p)
        c = p["N1"] * Ish1 / Eb1  # N = c·E_b/I_sh
        a1, a2 = axs
        Ish = np.linspace(0.3 * Ish1, Ish1 * 1.0, 200)
        line(a1, Ish, c * Eb2 / Ish, C[0], f"przy I_a = {fmt(p['Ia2'], 0)} A")
        line(a1, Ish, c * Eb1 / Ish, C[1], f"przy I_a = {fmt(p['Ia1'], 0)} A", ls="--")
        point(a1, Ish1, p["N1"], C[1], f"{fmt(p['N1'], 0)} obr/min", dy=12)
        if Radd >= 0:
            point(a1, Ish2, p["N2"], C[0], f"{fmt(p['N2'], 0)} obr/min (R_dod = {fmt(Radd, 1)} Ω)")
        style(a1, "N w funkcji prądu wzbudzenia", "Prąd wzbudzenia I_sh [A]", "N [obr/min]")
        legend(a1, "upper right")
        Ia = np.linspace(0, 1.5 * max(p["Ia1"], p["Ia2"]), 200)
        for i, R in enumerate([0.0, max(Radd, 0) * 0.5, max(Radd, 0)]):
            Ish_ = V / (Rsh + R)
            line(a2, Ia, c * (V - Ia * Ra) / Ish_, C[i], f"R_dod = {fmt(R, 1)} Ω")
        style(a2, "N(I_a) dla różnych R_dod", "I_a [A]", "N [obr/min]")
        legend(a2, "lower left")


class Ex3351(Page):
    group = "5. Rozruch i regulacja prędkości"
    title = "Przykład 3.35.1 - silnik szeregowy, bocznik uzwojenia wzbudzenia (diverter)"
    subtitle = "250 V, 30 A przy 800 obr/min; pole zbocznikowane rezystancją = R_se, moment +50 %"
    params = [
        ("V", "Napięcie V [V]", 100, 500, 250, 5),
        ("I1", "Prąd początkowy I₁ [A]", 5, 100, 30, 1),
        ("N1", "Prędkość początkowa N₁ [obr/min]", 200, 2000, 800, 10),
        ("Ra", "Rezystancja twornika R_a [Ω]", 0.02, 1.0, 0.15, 0.01),
        ("Rse", "Rezystancja uzw. szeregowego R_se [Ω]", 0.02, 1.0, 0.1, 0.01),
        ("Rd", "Rezystancja bocznika (divertera) R_d [Ω]", 0.02, 1.0, 0.1, 0.01),
        ("k", "Nowy moment [× T₁]", 0.5, 3.0, 1.5, 0.1),
    ]

    def calc(self, p):
        V, I1, N1, Ra, Rse, Rd, k = (p[x] for x in ("V", "I1", "N1", "Ra", "Rse", "Rd", "k"))
        a = Rd / (Rd + Rse)                 # część prądu płynąca przez uzwojenie pola
        Ia2 = I1 * math.sqrt(k / a)
        Ise2 = a * Ia2
        Rp = Rse * Rd / (Rse + Rd)
        Eb1 = V - I1 * (Ra + Rse)
        Eb2 = V - Ia2 * (Ra + Rp)
        N2 = N1 * (Eb2 / Eb1) * (I1 / Ise2)
        return a, Ia2, Ise2, Rp, Eb1, Eb2, N2

    def solve(self, p, t):
        a, Ia2, Ise2, Rp, Eb1, Eb2, N2 = self.calc(p)
        t.h("O co chodzi? (wersja dla licealisty)")
        t.p("W silniku szeregowym ten sam prąd płynie przez twornik i uzwojenie wzbudzenia, więc "
            "φ ∝ I. Jeśli równolegle do uzwojenia pola dołożymy rezystor (diverter), część prądu "
            "'ominie' uzwojenie → pole słabnie → silnik przyspiesza.")
        t.p("Moment T ∝ φ·I_a ∝ I_se·I_a. Gdy rośnie obciążenie, silnik musi pobrać większy prąd.")
        t.h("Rozwiązanie krok po kroku")
        t.eq(f"Udział prądu w polu: I_se = I_a·R_d/(R_d + R_se) = {fmt(a, 3)}·I_a")
        t.eq(f"T₂/T₁ = (I_se2·I_a2)/(I₁·I₁) = {fmt(p['k'], 2)}  →  {fmt(a, 3)}·I_a2² = {fmt(p['k'], 2)}·{fmt(p['I1'], 0)}²")
        t.eq(f"I_a2 = {fmt(p['I1'], 0)}·√({fmt(p['k'], 2)}/{fmt(a, 3)}) = {fmt(Ia2, 4)} A,   I_se2 = {fmt(Ise2, 4)} A")
        t.eq(f"E_b1 = V − I₁·(R_a + R_se) = {fmt(Eb1, 3)} V")
        t.eq(f"R_se ∥ R_d = {fmt(Rp, 4)} Ω;  E_b2 = V − I_a2·(R_a + R_se∥R_d) = {fmt(Eb2, 3)} V")
        t.eq("N₂/N₁ = (E_b2/E_b1)·(φ₁/φ₂) = (E_b2/E_b1)·(I₁/I_se2)")
        t.eq(f"N₂ = {fmt(p['N1'], 0)}·({fmt(Eb2, 3)}/{fmt(Eb1, 3)})·({fmt(p['I1'], 0)}/{fmt(Ise2, 4)}) = {fmt(N2, 2)} obr/min")
        t.res(f"Nowy prąd {fmt(Ia2, 2)} A, nowa prędkość {fmt(N2, 1)} obr/min")
        t.book("I_a2 = 51,9615 A, E_b2 = 239,607 V, N₂ ≈ 912,7 obr/min")
        t.h("Komentarz inżyniera")
        t.eng("Ciekawe: mimo że moment wzrósł o 50 %, silnik przyspieszył - bo osłabiliśmy pole. "
              "Diverter stosuje się, gdy potrzebna jest prędkość wyższa od znamionowej przy "
              "stałym obciążeniu. Ten sam efekt daje uzwojenie z odczepami (tapped field) "
              "lub przełączanie sekcji pola szeregowo/równolegle - popularne w trakcji.")

    def draw(self, axs, p):
        V, I1, N1, Ra, Rse = p["V"], p["I1"], p["N1"], p["Ra"], p["Rse"]
        a, Ia2, Ise2, Rp, Eb1, Eb2, N2 = self.calc(p)
        cN = N1 * I1 / Eb1           # N = cN·E_b/I_se
        T1 = 1.0
        I = np.linspace(0.3 * I1, 2.5 * I1, 200)
        a1, a2 = axs
        Nn = cN * (V - I * (Ra + Rse)) / I
        Nd = cN * (V - I * (Ra + Rp)) / (a * I)
        line(a1, I, Nn, C[0], "Bez divertera")
        line(a1, I, Nd, C[1], f"Z diverterem R_d = {fmt(p['Rd'], 2)} Ω")
        point(a1, I1, N1, C[0], f"{fmt(N1, 0)} obr/min")
        point(a1, Ia2, N2, C[1], f"{fmt(N2, 0)} obr/min")
        style(a1, "Prędkość - prąd", "I_a [A]", "N [obr/min]")
        legend(a1, "upper right")
        Tn = (I / I1) ** 2 * T1
        Td = a * (I / I1) ** 2 * T1
        line(a2, Tn, Nn, C[0], "Bez divertera")
        line(a2, Td, Nd, C[1], "Z diverterem")
        point(a2, 1.0, N1, C[0])
        point(a2, p["k"], N2, C[1], f"{fmt(p['k'], 1)}·T₁, {fmt(N2, 0)} obr/min")
        a2.set_xlim(0, 3.2)
        style(a2, "Prędkość - moment (względny)", "T / T₁", "N [obr/min]")
        legend(a2, "upper right")


class Ex3353(Page):
    group = "5. Rozruch i regulacja prędkości"
    title = "Przykład 3.35.3 - silnik szeregowy 440 V z rezystorem regulacyjnym"
    subtitle = "R_a + R_se = 0,3 Ω; R = 0: 20 A, 1200 obr/min; R = 3 Ω: 15 A, strumień przy 15 A = 80 % strumienia przy 20 A"
    params = [
        ("V", "Napięcie V [V]", 200, 600, 440, 10),
        ("Rm", "Rezystancja twornika + pola [Ω]", 0.05, 1.0, 0.3, 0.05),
        ("I1", "Prąd I₁ (R = 0) [A]", 5, 60, 20, 1),
        ("N1", "Prędkość N₁ [obr/min]", 300, 3000, 1200, 10),
        ("R", "Rezystancja regulacyjna R [Ω]", 0.0, 10.0, 3.0, 0.1),
        ("I2", "Prąd I₂ (drugie obciążenie) [A]", 5, 60, 15, 1),
        ("r", "Strumień φ₂/φ₁ [%]", 50, 100, 80, 1),
    ]

    def calc(self, p):
        V, Rm, I1, N1, R, I2, r = (p[x] for x in ("V", "Rm", "I1", "N1", "R", "I2", "r"))
        r /= 100
        Eb1 = V - I1 * Rm
        Eb2 = V - I2 * (Rm + R)
        N2 = N1 * (Eb2 / Eb1) / r
        P1, P2 = Eb1 * I1, Eb2 * I2
        return Eb1, Eb2, N2, P1, P2

    def solve(self, p, t):
        Eb1, Eb2, N2, P1, P2 = self.calc(p)
        t.h("O co chodzi? (wersja dla licealisty)")
        t.p("Rezystor regulacyjny w szereg z silnikiem szeregowym 'zjada' część napięcia - "
            "zostaje mniej na SEM wsteczną, więc przy tym samym strumieniu silnik zwalnia. "
            "Ale tu jednocześnie zmalało obciążenie (15 A zamiast 20 A), więc pole jest słabsze "
            "(80 %) - a słabsze pole = większa prędkość. Który efekt wygra? Liczymy!")
        t.p("Uwaga: końcówka treści w skanie jest ucięta; dane I₂ = 15 A i φ₂ = 0,8·φ₁ "
            "przyjęto wg klasycznej wersji tego zadania.")
        t.h("Rozwiązanie krok po kroku")
        t.eq(f"E_b1 = V − I₁·R_m = {fmt(p['V'], 0)} − {fmt(p['I1'], 0)}·{fmt(p['Rm'], 2)} = {fmt(Eb1, 2)} V")
        t.eq(f"E_b2 = V − I₂·(R_m + R) = {fmt(p['V'], 0)} − {fmt(p['I2'], 0)}·{fmt(p['Rm'] + p['R'], 2)} = {fmt(Eb2, 2)} V")
        t.eq("N₂ = N₁·(E_b2/E_b1)·(φ₁/φ₂)")
        t.eq(f"N₂ = {fmt(p['N1'], 0)}·({fmt(Eb2, 2)}/{fmt(Eb1, 2)})/{fmt(p['r'] / 100, 2)} = {fmt(N2, 1)} obr/min")
        t.p("Stosunek mocy mechanicznych (rozwiniętych w tworniku) P = E_b·I_a:")
        t.eq(f"P₂/P₁ = ({fmt(Eb2, 1)}·{fmt(p['I2'], 0)})/({fmt(Eb1, 1)}·{fmt(p['I1'], 0)}) = {fmt(P2 / P1, 3)}")
        t.res(f"N₂ = {fmt(N2, 1)} obr/min,  P₂/P₁ = {fmt(P2 / P1, 3)}")
        t.h("Komentarz inżyniera")
        t.eng(f"W rezystorze R wydziela się {fmt(p['I2'] ** 2 * p['R'], 0)} W ciepła - regulacja "
              "rezystancyjna jest prosta, ale nieekonomiczna (stąd w dawnych tramwajach ciepłe "
              "rezystory pod podłogą). Dziś zastępują ją przekształtniki (choppery).")

    def draw(self, axs, p):
        V, Rm, I1, N1, R, I2, r = (p[x] for x in ("V", "Rm", "I1", "N1", "R", "I2", "r"))
        r /= 100
        Eb1, Eb2, N2, P1, P2 = self.calc(p)
        # model strumienia φ = I/(a + I) dopasowany do φ(I₂)/φ(I₁) = r
        den = I2 - r * I1
        aa = I1 * I2 * (r - 1) / den if abs(den) > 1e-9 else -1
        if aa > 0:
            phi = lambda I: I / (aa + I) / (I1 / (aa + I1))
        else:
            phi = lambda I: I / I1
        I = np.linspace(0.25 * I1, 2.5 * I1, 200)
        a1, a2 = axs
        for i, Rr in enumerate([0.0, R]):
            N = N1 * (V - I * (Rm + Rr)) / Eb1 / phi(I)
            line(a1, I, N, C[i], f"R = {fmt(Rr, 1)} Ω")
        point(a1, I1, N1, C[0], f"{fmt(N1, 0)} obr/min")
        point(a1, I2, N2, C[1], f"{fmt(N2, 0)} obr/min")
        a1.set_ylim(0, 3 * N1)
        style(a1, "Prędkość - prąd (silnik szeregowy)", "I [A]", "N [obr/min]")
        legend(a1, "upper right")
        labels = ["Obciążenie 1\n(R = 0)", f"Obciążenie 2\n(R = {fmt(R, 1)} Ω)"]
        Pm = [P1, P2]
        Ls = [I1 ** 2 * Rm, I2 ** 2 * Rm]
        Lr = [0.0, I2 ** 2 * R]
        for i in range(2):
            a2.bar(i, Pm[i] / 1000, color=C[0], width=0.55, edgecolor=SURFACE, linewidth=2,
                   label="Moc mechaniczna" if i == 0 else "_")
            a2.bar(i, Ls[i] / 1000, bottom=Pm[i] / 1000, color=C[3], width=0.55,
                   edgecolor=SURFACE, linewidth=2, label="Straty w silniku" if i == 0 else "_")
            a2.bar(i, Lr[i] / 1000, bottom=(Pm[i] + Ls[i]) / 1000, color=C[7], width=0.55,
                   edgecolor=SURFACE, linewidth=2, label="Straty w R" if i == 0 else "_")
        a2.set_xticks([0, 1])
        a2.set_xticklabels(labels, fontsize=8.5)
        style(a2, "Bilans mocy", "", "Moc [kW]")
        legend(a2, "upper right")


class VoltageControlPage(Page):
    group = "5. Rozruch i regulacja prędkości"
    title = "Regulacja napięciem twornika i osłabianiem pola (pełny zakres)"
    subtitle = "Rozdz. 3.34.2–3.34.3, 3.35: napięcie twornika poniżej, osłabianie pola powyżej prędkości znamionowej"
    params = [
        ("Vn", "Napięcie znamionowe V_n [V]", 110, 500, 220, 10),
        ("Ra", "Rezystancja twornika R_a [Ω]", 0.05, 1.0, 0.2, 0.05),
        ("In", "Prąd znamionowy I_n [A]", 10, 200, 50, 5),
        ("Nn", "Prędkość znamionowa N_n [obr/min]", 500, 3000, 1500, 50),
        ("Va", "Napięcie twornika [% V_n]", 0, 100, 60, 1),
        ("f", "Strumień [% φ_n]", 40, 100, 100, 1),
    ]

    def solve(self, p, t):
        Vn, Ra, In, Nn = p["Vn"], p["Ra"], p["In"], p["Nn"]
        Va = p["Va"] / 100 * Vn
        f = p["f"] / 100
        kphi_n = (Vn - In * Ra) / omega(Nn)
        N0 = Va / (kphi_n * f) * 60 / TWO_PI
        N_load = (Va - In * Ra) / (kphi_n * f) * 60 / TWO_PI
        Tn = kphi_n * In
        t.h("O co chodzi? (wersja dla licealisty)")
        t.p("Silnik obcowzbudny/bocznikowy ma dwa 'pokrętła' prędkości:")
        t.bullet("napięcie twornika - jak pedał gazu: od zera do prędkości znamionowej; "
                 "moment dostępny przy prądzie znamionowym pozostaje STAŁY (strefa stałego momentu),")
        t.bullet("osłabianie pola - jak 'nadbieg': powyżej prędkości znamionowej; prędkość rośnie, "
                 "ale moment maleje (∝ φ), a moc pozostaje stała (strefa stałej mocy).")
        t.p("Książka opisuje też układ potencjometryczny (dzielnik napięcia - pozwala zejść do zera) "
            "i sterowanie wielonapięciowe / Ward-Leonarda (osobna prądnica zasila twornik).")
        t.h("Obliczenia dla ustawień suwaków")
        t.eq(f"kφ_n = (V_n − I_n·R_a)/ω_n = ({fmt(Vn, 0)} − {fmt(In, 0)}·{fmt(Ra, 2)})/{fmt(omega(Nn), 2)} "
             f"= {fmt(kphi_n, 4)} V·s/rad")
        t.eq(f"V_a = {fmt(Va, 1)} V,  φ = {fmt(f, 2)}·φ_n")
        t.eq(f"Bieg jałowy: N₀ = V_a/(kφ) = {fmt(N0, 0)} obr/min")
        t.eq(f"Przy I_n: N = (V_a − I_n·R_a)/(kφ) = {fmt(max(N_load, 0), 0)} obr/min,  T = kφ·I_n = {fmt(Tn * f, 1)} N·m")
        if N_load <= 0:
            t.res("Napięcie zbyt małe, by pokonać spadek I_n·R_a - silnik obciążony znamionowo nie ruszy "
                  "(rys. 3.34.6: potrzebne minimalne napięcie na pokonanie tarcia i spadku I_a·R_a).")
        else:
            t.res(f"Prędkość przy obciążeniu znamionowym: {fmt(N_load, 0)} obr/min")
        t.h("Komentarz inżyniera")
        t.eng("Prawy wykres to 'mapa możliwości' napędu: do N_n regulujemy napięciem (pełny moment), "
              "powyżej - polem (pełna moc). Tak pracują np. napędy walcarek i dawne lokomotywy; "
              "dziś napięcie twornika podaje prostownik tyrystorowy lub chopper.")

    def draw(self, axs, p):
        Vn, Ra, In, Nn = p["Vn"], p["Ra"], p["In"], p["Nn"]
        f = p["f"] / 100
        kphi_n = (Vn - In * Ra) / omega(Nn)
        Tn = kphi_n * In
        T = np.linspace(0, 1.5 * Tn, 100)
        a1, a2 = axs
        for i, frac in enumerate([0.25, 0.5, 0.75, 1.0]):
            Va = frac * Vn
            N = (Va - T / (kphi_n * f) * Ra) / (kphi_n * f) * 60 / TWO_PI
            line(a1, T, N, C[i], f"V_a = {int(frac * 100)} %", lw=1.4)
        Va = p["Va"] / 100 * Vn
        N = (Va - T / (kphi_n * f) * Ra) / (kphi_n * f) * 60 / TWO_PI
        line(a1, T, N, INK, f"Ustawienie: {int(p['Va'])} %, φ = {int(p['f'])} %", lw=2.6)
        a1.set_ylim(0, None)
        style(a1, "N(T) dla różnych napięć", "Moment T [N·m]", "N [obr/min]")
        legend(a1, "upper right")
        Ns = np.linspace(1, 2.5 * Nn, 300)
        Tmax = np.where(Ns <= Nn, 1.0, Nn / Ns)
        Pmax = np.where(Ns <= Nn, Ns / Nn, 1.0)
        line(a2, Ns, Tmax, C[0], "Moment dopuszczalny T/T_n")
        line(a2, Ns, Pmax, C[1], "Moc dopuszczalna P/P_n")
        a2.axvline(Nn, color=MUTED, ls=":", lw=1)
        a2.text(Nn * 0.5, 1.07, "regulacja napięciem", ha="center", fontsize=8.5, color=INK2)
        a2.text(Nn * 1.75, 1.07, "osłabianie pola", ha="center", fontsize=8.5, color=INK2)
        N_op = max((Va - In * Ra) / (kphi_n * f) * 60 / TWO_PI, 0)
        point(a2, N_op, f, INK, "punkt pracy (I_n)")
        a2.set_ylim(0, 1.2)
        style(a2, "Strefy regulacji [p.u.]", "N [obr/min]", "Wartość względna [p.u.]")
        legend(a2, "lower right")


# =============================================================================
# 7. SILNIK UNIWERSALNY
# =============================================================================
class UniversalPage(Page):
    group = "6. Silnik uniwersalny"
    title = "Silnik uniwersalny - praca przy AC i DC"
    subtitle = "Rozdz. 3.36: silnik szeregowy z komutatorem zasilany prądem stałym lub przemiennym"
    params = [
        ("V", "Napięcie (DC lub skuteczne AC) [V]", 50, 300, 230, 5),
        ("R", "Rezystancja (twornik + pole) R [Ω]", 1, 20, 6, 0.5),
        ("X", "Reaktancja przy AC X [Ω]", 0, 60, 25, 1),
        ("k", "Stała maszyny k [V/(A·obr/min)]", 0.002, 0.03, 0.01, 0.001),
    ]

    def curves(self, p):
        V, R, X, k = p["V"], p["R"], p["X"], p["k"]
        I = np.linspace(0.2, V / np.hypot(R, X) * 0.999 if X > 0 else V / R * 0.999, 400)
        N_dc = (V - I * R) / (k * I)
        Eb_ac = np.sqrt(np.maximum(V ** 2 - (I * X) ** 2, 0)) - I * R
        N_ac = Eb_ac / (k * I)
        kt = k * 60 / TWO_PI
        T = kt * I ** 2
        m1, m2 = N_dc > 0, N_ac > 0
        return I, T, N_dc, N_ac, m1, m2

    def solve(self, p, t):
        V, R, X = p["V"], p["R"], p["X"]
        t.h("O co chodzi? (wersja dla licealisty)")
        t.p("Silnik uniwersalny to silnik SZEREGOWY, który działa zarówno na prąd stały, jak i "
            "przemienny. Dlaczego na AC też działa? Bo gdy prąd zmienia kierunek, zmienia się "
            "JEDNOCZEŚNIE prąd w tworniku i pole - a minus razy minus daje plus: moment ma "
            "ciągle ten sam kierunek.")
        t.p("Przy AC pojawia się dodatkowo reaktancja uzwojeń (opór dla prądu zmiennego), która "
            "'zabiera' część napięcia - dlatego przy tym samym momencie silnik na AC kręci się "
            "wolniej niż na DC.")
        t.h("Model (wzory)")
        t.eq("DC:  E_b = V − I·R")
        t.eq("AC:  V² = (E_b + I·R)² + (I·X)²   →   E_b = √(V² − (I·X)²) − I·R")
        t.eq("N = E_b/(k·I),   T ∝ I²  (strumień ∝ prąd)")
        t.eq(f"Prąd zwarcia (rozruchu): DC: {fmt(V / R, 1)} A,  AC: {fmt(V / math.hypot(R, X), 1)} A")
        t.h("Modyfikacje silnika szeregowego do pracy na AC")
        t.bullet("cały obwód magnetyczny (także stojan) z blach - mniejsze straty na prądy wirowe,")
        t.bullet("mniej zwojów pola i szczelina jak najmniejsza - mniejsza reaktancja,")
        t.bullet("uzwojenie kompensacyjne (przewodzące lub indukcyjne) - znosi reaktancję twornika "
                 "(ustaw X → mały - krzywe AC i DC się zbliżają, jak w typie kompensowanym).")
        t.h("Komentarz inżyniera")
        t.eng("Zastosowania: odkurzacze, miksery, roboty kuchenne, suszarki, młynki, golarki, "
              "wiertarki i szlifierki ręczne. Zaleta: bardzo duża prędkość (nawet 20 000+ obr/min) "
              "i duży moment rozruchowy przy małych wymiarach. Wada: zużycie szczotek, hałas, iskrzenie.")

    def draw(self, axs, p):
        I, T, N_dc, N_ac, m1, m2 = self.curves(p)
        a1, a2 = axs
        line(a1, T[m1], N_dc[m1], C[0], "Zasilanie DC")
        line(a1, T[m2], N_ac[m2], C[1], "Zasilanie AC")
        a1.set_ylim(0, min(np.nanmax(N_dc[m1]), 30000))
        style(a1, "Prędkość - moment", "Moment T [N·m]", "N [obr/min]")
        legend(a1, "upper right")
        th = np.linspace(0, 4 * np.pi, 600)
        i = np.sin(th)
        line(a2, np.degrees(th), i, C[2], "Prąd i (= prąd twornika i pola)")
        line(a2, np.degrees(th), i * i, C[6], "Moment ∝ i·φ ∝ i² (zawsze ≥ 0)")
        a2.axhline(0, color="#9a9994", lw=0.8)
        style(a2, "Na AC moment nie zmienia znaku", "Kąt ωt [°]", "Wartość względna")
        a2.set_xticks(range(0, 721, 180))
        legend(a2, "lower right")


# =============================================================================
# Aplikacja
# =============================================================================
PAGES = [
    GeneratorPrinciple, MotorPrinciple,
    EmfReview2, EmfReview3, EmfReview4, EmfQ37,
    Ex3121, Ex3122, ExShuntFeeder, ExQ36, Ex3141, Ex3142, Ex3143, OCCPage, GenCharPage,
    BackEmfPage, GenMotorPage, Ex3251, Ex3261, MotorCharPage,
    StarterPage, Ex3341, FieldControlPage, Ex3351, Ex3353, VoltageControlPage,
    UniversalPage,
]


class HoverReadout:
    """Interaktywny odczyt: po najechaniu na krzywą pokazuje najbliższy punkt."""

    def __init__(self, canvas):
        self.canvas = canvas
        self.annots = {}
        canvas.mpl_connect("motion_notify_event", self.on_move)

    def reset(self):
        self.annots = {}

    def on_move(self, ev):
        fig = self.canvas.figure
        changed = False
        for ax in fig.axes:
            ann = self.annots.get(ax)
            if ev.inaxes is not ax:
                if ann is not None and ann.get_visible():
                    ann.set_visible(False)
                    changed = True
                continue
            best = None
            for ln in ax.get_lines():
                lab = ln.get_label()
                if lab.startswith("_") or len(ln.get_xdata()) < 2:
                    continue
                xy = np.column_stack([ln.get_xdata(), ln.get_ydata()]).astype(float)
                pix = ax.transData.transform(xy)
                d = np.hypot(pix[:, 0] - ev.x, pix[:, 1] - ev.y)
                i = int(np.nanargmin(d))
                if best is None or d[i] < best[0]:
                    best = (d[i], xy[i], lab, ln.get_color())
            if ann is None:
                ann = ax.annotate("", (0, 0), xytext=(12, 12), textcoords="offset points",
                                  fontsize=8.5, zorder=20,
                                  bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#9a9994", lw=0.8))
                ann.set_visible(False)
                self.annots[ax] = ann
            if best is not None and best[0] < 40:
                (x, y), lab, col = best[1], best[2], best[3]
                ann.xy = (x, y)
                ann.set_text(f"{lab}\nx = {fmt(x, 3)}\ny = {fmt(y, 3)}")
                ann.get_bbox_patch().set_edgecolor(col)
                ann.set_visible(True)
                changed = True
            elif ann.get_visible():
                ann.set_visible(False)
                changed = True
        if changed:
            self.canvas.draw_idle()


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Maszyny prądu stałego - interaktywne przykłady (UNIT III: D.C. Machines)")
        self.geometry("1600x920")
        self.minsize(1100, 700)
        self.configure(bg="#f4f3ef")
        st = ttk.Style(self)
        try:
            st.theme_use("clam")
        except tk.TclError:
            pass
        st.configure("Treeview", rowheight=24, font=("Segoe UI", 9))
        st.configure("Title.TLabel", font=("Segoe UI", 13, "bold"), foreground=INK)
        st.configure("Sub.TLabel", font=("Segoe UI", 9), foreground=INK2)

        self.pages = [cls() for cls in PAGES]
        self.current = None
        self._pending = None

        outer = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        outer.pack(fill=tk.BOTH, expand=True)

        # nawigacja
        nav = ttk.Frame(outer, padding=4)
        ttk.Label(nav, text="Przykłady z rozdziału", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(2, 4))
        self.tree = ttk.Treeview(nav, show="tree", selectmode="browse")
        self.tree.column("#0", width=340, stretch=True)
        self.tree.pack(fill=tk.BOTH, expand=True)
        groups = {}
        for i, pg in enumerate(self.pages):
            if pg.group not in groups:
                groups[pg.group] = self.tree.insert("", "end", text=pg.group, open=True)
            self.tree.insert(groups[pg.group], "end", iid=str(i), text=pg.title)
        self.tree.bind("<<TreeviewSelect>>", self.on_select)
        outer.add(nav, weight=0)

        # część główna
        main = ttk.Frame(outer, padding=(8, 6))
        outer.add(main, weight=1)
        self.lbl_title = ttk.Label(main, style="Title.TLabel")
        self.lbl_title.pack(anchor="w")
        self.lbl_sub = ttk.Label(main, style="Sub.TLabel", wraplength=1100)
        self.lbl_sub.pack(anchor="w", pady=(0, 6))

        body = ttk.PanedWindow(main, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(body)
        body.add(left, weight=0)
        self.ctrl = ttk.LabelFrame(left, text=" Dane zadania (suwaki) ", padding=6)
        self.ctrl.pack(fill=tk.X)
        self.text = ScrolledText(left, wrap=tk.WORD, width=62, font=("Segoe UI", 10),
                                 bg="white", relief=tk.FLAT, padx=10, pady=6)
        self.text.pack(fill=tk.BOTH, expand=True, pady=(6, 0))
        self.text.tag_configure("h", font=("Segoe UI", 11, "bold"), foreground=C[0], spacing1=6)
        self.text.tag_configure("p", font=("Segoe UI", 10), foreground=INK, spacing1=2)
        self.text.tag_configure("eq", font=("Consolas", 10), foreground="#1b1b1b",
                                background="#f2f1ec", spacing1=1)
        self.text.tag_configure("res", font=("Segoe UI", 10, "bold"), foreground="#006b00", spacing1=4)
        self.text.tag_configure("book", font=("Segoe UI", 9, "italic"), foreground=INK2)
        self.text.tag_configure("eng", font=("Segoe UI", 10), foreground="#3b2f8f",
                                background="#f3f1fb", lmargin1=6, lmargin2=6)

        right = ttk.Frame(body)
        body.add(right, weight=1)
        self.fig = Figure(figsize=(9, 6), dpi=100, facecolor=SURFACE)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        tb = NavigationToolbar2Tk(self.canvas, right, pack_toolbar=False)
        tb.update()
        tb.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.hover = HoverReadout(self.canvas)

        self.tree.selection_set("0")

    # ── obsługa wyboru strony ──
    def on_select(self, _ev=None):
        sel = self.tree.selection()
        if not sel or not sel[0].isdigit():
            return
        self.show(self.pages[int(sel[0])])

    def show(self, page):
        self.current = page
        self.lbl_title.config(text=page.title)
        self.lbl_sub.config(text=page.subtitle)
        for w in self.ctrl.winfo_children():
            w.destroy()
        self.vars = {}
        row = 0
        for key, label, lo, hi, _d, res in page.params:
            ttk.Label(self.ctrl, text=label).grid(row=row, column=0, sticky="w", padx=(0, 6))
            var = tk.DoubleVar(value=page.values[key])
            sc = tk.Scale(self.ctrl, from_=lo, to=hi, resolution=res, orient=tk.HORIZONTAL,
                          variable=var, length=230, showvalue=True, bg="#f4f3ef",
                          highlightthickness=0, troughcolor="#dcdad3", font=("Segoe UI", 8),
                          command=lambda _v, k=key: self.changed(k))
            sc.grid(row=row, column=1, sticky="ew")
            self.vars[key] = var
            row += 1
        for key, label, opts, _d in page.choices:
            ttk.Label(self.ctrl, text=label).grid(row=row, column=0, sticky="w")
            var = tk.StringVar(value=page.values[key])
            fr = ttk.Frame(self.ctrl)
            fr.grid(row=row, column=1, sticky="w")
            for o in opts:
                ttk.Radiobutton(fr, text=o, value=o, variable=var,
                                command=lambda k=key: self.changed(k)).pack(side=tk.LEFT, padx=2)
            self.vars[key] = var
            row += 1
        ttk.Button(self.ctrl, text="↺ Przywróć dane z książki", command=self.reset).grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(4, 0))
        self.ctrl.columnconfigure(1, weight=1)
        self.refresh()

    def reset(self):
        pg = self.current
        pg.values = pg.defaults()
        for k, v in pg.values.items():
            self.vars[k].set(v)
        self.refresh()

    def changed(self, _key):
        if self._pending is not None:
            self.after_cancel(self._pending)
        self._pending = self.after(80, self.refresh)

    def read_values(self):
        pg = self.current
        for key, *_ in pg.params:
            pg.values[key] = float(self.vars[key].get())
        for key, *_ in pg.choices:
            pg.values[key] = self.vars[key].get()
        return dict(pg.values)

    def refresh(self):
        self._pending = None
        pg = self.current
        p = self.read_values()
        doc = Doc()
        try:
            pg.solve(p, doc)
        except (ZeroDivisionError, ValueError) as e:
            doc.res(f"Nie da się policzyć dla tych danych: {e}")
        y = self.text.yview()[0]
        self.text.config(state=tk.NORMAL)
        self.text.delete("1.0", tk.END)
        for s, tag in doc.parts:
            self.text.insert(tk.END, s, tag)
        self.text.config(state=tk.DISABLED)
        self.text.yview_moveto(y)
        self.fig.clear()
        r, c = pg.layout
        axs = [self.fig.add_subplot(r, c, i + 1) for i in range(r * c)]
        try:
            pg.draw(axs, p)
        except (ZeroDivisionError, ValueError, FloatingPointError):
            pass
        self.fig.tight_layout(pad=1.6)
        self.hover.reset()
        self.canvas.draw_idle()


if __name__ == "__main__":
    np.seterr(all="ignore")
    App().mainloop()
