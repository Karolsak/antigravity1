"""
Maszyny synchroniczne - stan ustalony (Electric machines design: Synchronous - steady state)
===========================================================================================
Interaktywna aplikacja Tkinter + Matplotlib. Każda zakładka = jeden przykład:
suwaki (parametry), wykresy (wyniki, z zoomem), wyniki liczbowe oraz objaśnienie
napisane językiem zrozumiałym dla ucznia liceum - tak, jak tłumaczyłby to
doświadczony inżynier.

Zakładki:
  S1.  Próba biegu jałowego i zwarcia: Xs nienasycona / nasycona, SCR
  S2.  Generator z wirnikiem cylindrycznym: schemat zastępczy, wykres wskazowy
  S3.  Zmienność napięcia (regulacja), charakterystyki zewnętrzne i regulacyjne
  S4.  Charakterystyka kątowa mocy P(δ), Q(δ), granica stabilności
  S5.  Silnik synchroniczny: krzywe V (krzywe Mordey'a), współczynnik mocy
  S6.  Kompensacja mocy biernej w zakładzie silnikiem synchronicznym
  S7.  Maszyna z biegunami jawnymi: teoria dwóch reakcji (Xd, Xq), moc reluktancyjna
  S8.  Wykres kołowy P-Q (capability curve): granice pracy generatora
  S9.  Straty i sprawność w funkcji obciążenia
  S10. Praca równoległa dwóch generatorów: podział mocy czynnej i biernej (statyzm)
  S11. Synchronizacja z siecią: różnica napięć na wyłączniku, dudnienia

Wszystkie obliczenia w jednostkach względnych (p.u.) na fazę; tam gdzie to
pomaga, przeliczamy na wartości fizyczne przy zadanych U_N [kV] i S_N [MVA].

Uruchomienie:  python synchronous_machine_steady_tkinter.py
Wymagania:     numpy, matplotlib (tkinter jest w standardowym Pythonie)
"""
import cmath
import math
import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

PI = math.pi
SQ3 = math.sqrt(3.0)
DEG = PI / 180.0


# =============================================================================
# RDZEŃ OBLICZENIOWY
# =============================================================================

def occ(i_f, s):
    """Charakterystyka biegu jałowego E(If) w p.u. (model Froelicha).
    Styczna w zerze = prosta szczeliny powietrznej E = If; s - stopień nasycenia."""
    return i_f / (1.0 + s * i_f)


def occ_inverse(e, s):
    """Prąd wzbudzenia potrzebny do uzyskania SEM e przy nasyceniu s."""
    return e / (1.0 - s * e)


def phasor_current(ia, phi_deg):
    """Prąd twornika jako wskazówka względem napięcia V (kąt 0).
    phi > 0 - prąd opóźniony (obciążenie indukcyjne), phi < 0 - wyprzedzający."""
    return cmath.rect(ia, -phi_deg * DEG)


def gen_ef(v, ia, phi_deg, ra, xs):
    """Generator cylindryczny: Ef = V + (Ra + jXs) Ia."""
    i = phasor_current(ia, phi_deg)
    return v + complex(ra, xs) * i, i


def terminal_voltage(ef, ia, phi_deg, ra, xs):
    """Napięcie na zaciskach przy stałym Ef i danym prądzie/kącie (charakterystyka zewnętrzna).
    Ef^2 = (V + I a)^2 + (I b)^2  ->  V = -I a + sqrt(Ef^2 - (I b)^2)."""
    c, s = math.cos(phi_deg * DEG), math.sin(phi_deg * DEG)
    a = ra * c + xs * s
    b = xs * c - ra * s
    ia = np.asarray(ia, dtype=float)
    disc = ef ** 2 - (ia * b) ** 2
    v = -ia * a + np.sqrt(np.clip(disc, 0.0, None))
    return np.where((disc >= 0) & (v > 0), v, np.nan)


def power_angle(v, ef, xs, delta):
    """Moc czynna i bierna generatora cylindrycznego (Ra = 0), p.u. trójfazowe."""
    p = v * ef / xs * np.sin(delta)
    q = (v * ef * np.cos(delta) - v ** 2) / xs
    return p, q


def motor_state(v, ef, xs, p):
    """Silnik synchroniczny: V = Ef + jXs Ia. Zwraca (Ia fazor, δ) lub (None, None)
    gdy moc przekracza moc maksymalną (utrata synchronizmu)."""
    sd = p * xs / (v * ef) if ef > 0 else 2.0
    if abs(sd) > 1.0:
        return None, None
    d = math.asin(sd)
    efp = cmath.rect(ef, -d)
    ia = (v - efp) / complex(0.0, xs)
    return ia, d


def salient_gen(v, ia, phi_deg, xd, xq, ra=0.0):
    """Teoria dwóch reakcji dla generatora z biegunami jawnymi.
    Zwraca słownik z δ, Ef, Id, Iq, fazorami."""
    i = phasor_current(ia, phi_deg)
    e_q = v + complex(ra, xq) * i          # wskazówka na osi q
    d = cmath.phase(e_q)
    q_axis = cmath.rect(1.0, d)
    d_axis = cmath.rect(1.0, d - PI / 2)
    iq = (i * q_axis.conjugate()).real     # rzut na oś q
    idd = (i * d_axis.conjugate()).real    # rzut na oś d (dodatni = rozmagnesowuje)
    ef = abs(e_q) + (xd - xq) * idd
    return dict(delta=d, ef=ef, Id=idd * d_axis, Iq=iq * q_axis, I=i, Eq=e_q,
                Efp=ef * q_axis, id=idd, iq=iq)


def salient_power(v, ef, xd, xq, delta):
    p_exc = v * ef / xd * np.sin(delta)
    p_rel = v ** 2 / 2.0 * (1.0 / xq - 1.0 / xd) * np.sin(2 * delta)
    return p_exc, p_rel, p_exc + p_rel


def droop_share(fnl1, fnl2, k1, k2, p_load):
    """Dwa generatory ze statyzmem f = f_nl - k P. Wspólna częstotliwość i podział mocy."""
    f = (fnl1 / k1 + fnl2 / k2 - p_load) / (1.0 / k1 + 1.0 / k2)
    return f, (fnl1 - f) / k1, (fnl2 - f) / k2


# =============================================================================
# OBJAŚNIENIA (dla ucznia liceum, głosem inżyniera)
# =============================================================================

EXPL = {
"S1": """PRZYKŁAD S1 - Próba biegu jałowego i zwarcia. Jak wyznaczyć reaktancję synchroniczną Xs?

CO TO JEST: Maszyna synchroniczna ma wirnik z elektromagnesem (uzwojenie wzbudzenia zasilane prądem stałym If). Wirnik obraca się i jego pole "przecina" uzwojenia stojana, indukując w nich napięcie Ef - tak jak magnes przesuwany przy cewce w pracowni fizycznej.

PRÓBA BIEGU JAŁOWEGO (OCC): kręcimy maszyną z prędkością znamionową, zaciski otwarte, zwiększamy If i mierzymy napięcie. Na początku rośnie ono proporcjonalnie (prosta szczeliny powietrznej - "air-gap line"), potem żelazo się nasyca (jak gąbka, która nie wchłonie więcej wody) i krzywa się wygina.

PRÓBA ZWARCIA (SCC): zwieramy zaciski i znów zwiększamy If. Prąd zwarcia rośnie liniowo - bo przy zwarciu pole w maszynie jest małe i żelazo się nie nasyca.

REAKTANCJA SYNCHRONICZNA: przy tym samym If dzielimy napięcie z OCC przez prąd z SCC:  Xs = E_oc / I_sc.
 • z prostej szczeliny -> Xs nienasycona (większa),
 • z OCC przy napięciu znamionowym -> Xs nasycona (mniejsza, lepiej opisuje rzeczywistą pracę).
SCR (short-circuit ratio) = If dla U_N na biegu jałowym / If dla I_N przy zwarciu ≈ 1 / Xs_nas.
Duże SCR = maszyna "sztywniejsza", stabilniejsza, ale większa i droższa (większa szczelina).

UWAGA INŻYNIERA: w jednostkach względnych (p.u.) napięcie 1,0 = znamionowe, prąd 1,0 = znamionowy. Dzięki temu parametry maszyn 5 kVA i 500 MVA wyglądają podobnie (Xs ≈ 1-2 p.u.).

SPRÓBUJ: zwiększ nasycenie s - Xs nasycona spada, SCR rośnie. Zmień Xs nienasyconą - prosta SCC robi się bardziej płaska.
""",

"S2": """PRZYKŁAD S2 - Generator z wirnikiem cylindrycznym: schemat zastępczy i wykres wskazowy

MODEL: jedna faza generatora to źródło napięcia Ef (SEM od wirnika) połączone szeregowo z rezystancją Ra i reaktancją synchroniczną Xs. Na zaciskach mamy napięcie V i prąd Ia:
        Ef = V + (Ra + j·Xs)·Ia
To zwykłe prawo Kirchhoffa, tylko że wielkości są WSKAZÓWKAMI (strzałkami) - mają długość i kąt, bo prąd przemienny może być przesunięty w czasie względem napięcia.

KĄT φ (fi): przesunięcie prądu względem napięcia. cos φ to współczynnik mocy.
 • obciążenie indukcyjne (silniki, dławiki) - prąd opóźniony, φ > 0,
 • obciążenie pojemnościowe - prąd wyprzedzający, φ < 0.
KĄT MOCY δ (delta): kąt między Ef a V. Im większa moc czynna, tym większe δ - wirnik "wyprzedza" pole sieci jak pies ciągnący smycz.

CO WIDAĆ: przy obciążeniu indukcyjnym trzeba DUŻO większego Ef (więcej prądu wzbudzenia), przy pojemnościowym - mniejszego, czasem nawet Ef < V.

UWAGA INŻYNIERA: Ra jest zwykle bardzo mała (0,005-0,02 p.u.) wobec Xs (1-2 p.u.), więc w obliczeniach często się ją pomija. Moc na fazę: P = V·Ia·cos φ; w układzie 3-fazowym w p.u. to po prostu P = V·Ia·cos φ (bo 1 p.u. = S_N).

SPRÓBUJ: ustaw φ od +37° (cos φ = 0,8 ind.) do -37° (0,8 poj.) i obserwuj długość Ef i kąt δ.
""",

"S3": """PRZYKŁAD S3 - Zmienność napięcia (regulacja napięcia) i charakterystyki zewnętrzne

PYTANIE: jeśli generator pracuje sam (nie z siecią) i nie ruszamy prądu wzbudzenia, to jak zmieni się napięcie, gdy odłączymy obciążenie?
ZMIENNOŚĆ NAPIĘCIA:   ΔU% = (Ef - V_N) / V_N · 100 %
(Ef to napięcie na biegu jałowym przy tym samym If.)

WYKRES 1 - charakterystyka zewnętrzna V(Ia) przy stałym Ef: dla obciążenia indukcyjnego napięcie spada szybko, dla rezystancyjnego wolniej, a dla pojemnościowego może nawet ROSNĄĆ (reakcja twornika wtedy "pomaga" wzbudzeniu).
WYKRES 2 - charakterystyka regulacyjna If(Ia) przy stałym V = 1: tyle prądu wzbudzenia musi dodać regulator napięcia (AVR), by utrzymać stałe napięcie.
WYKRES 3 - ΔU% w funkcji cos φ dla prądu znamionowego.

UWAGA INŻYNIERA: w praktyce Ef jest ograniczone nasyceniem, dlatego do ΔU używa się Xs nasyconej (lub metody Potiera/ASA). Metoda z Xs nienasyconą daje wynik "pesymistyczny" (za duże ΔU).

SPRÓBUJ: zwiększ Xs - krzywe robią się bardziej strome. Zmień kąt φ roboczego punktu - zobacz, dla jakiego cos φ zmienność napięcia jest zerowa.
""",

"S4": """PRZYKŁAD S4 - Charakterystyka kątowa mocy P(δ) i granica stabilności

WZÓR (Ra ≈ 0):   P = V·Ef / Xs · sin δ        Q = (V·Ef·cos δ - V²) / Xs
Moc czynna zależy od sinusa kąta δ. Największa jest dla δ = 90°:  P_max = V·Ef / Xs.

ANALOGIA: wirnik jest połączony z wirującym polem sieci "sprężyną magnetyczną". Turbina ciągnie wirnik do przodu - sprężyna się napina (δ rośnie). Do 90° sprężyna jest coraz silniejsza, ale za 90° SŁABNIE - jeśli turbina da więcej niż P_max, wirnik "wyskakuje" z synchronizmu (poślizg biegunów) - poważna awaria!

ZAPAS STABILNOŚCI: w praktyce pracuje się przy δ = 20-40°, aby zostawić margines na zakłócenia (np. zwarcie w sieci).
  współczynnik zapasu k = P_max / P.
Większe Ef (więcej wzbudzenia) -> większe P_max -> bezpieczniej. Dlatego generator niedowzbudzony (pobierający Q) jest mniej stabilny.

Q(δ): dodatnie Q = generator oddaje moc bierną (przewzbudzony), ujemne = pobiera (niedowzbudzony).

SPRÓBUJ: zwiększaj P_turbiny aż do przekroczenia P_max - program pokaże utratę synchronizmu. Potem zwiększ Ef.
""",

"S5": """PRZYKŁAD S5 - Silnik synchroniczny: krzywe V

SILNIK: równanie odwrotne niż w generatorze:  V = Ef + j·Xs·Ia  (Ra ≈ 0). Silnik obraca się DOKŁADNIE z prędkością synchroniczną n = 60·f / p (p - liczba par biegunów), niezależnie od obciążenia - dlatego nazwa "synchroniczny".

KRZYWE V: przy stałej mocy mechanicznej zmieniamy prąd wzbudzenia (Ef) i mierzymy prąd stojana Ia. Wykres wygląda jak litera "V":
 • niedowzbudzenie (Ef małe) - silnik pobiera z sieci moc bierną jak cewka, cos φ opóźniony,
 • minimum krzywej - cos φ = 1, prąd najmniejszy,
 • przewzbudzenie (Ef duże) - silnik ODDAJE moc bierną do sieci, działa jak kondensator! cos φ wyprzedzający.
Linia przerywana z lewej = granica stabilności (δ = 90°) - przy zbyt małym wzbudzeniu silnik wypada z synchronizmu.

UWAGA INŻYNIERA: to wyjątkowa cecha - silnikiem synchronicznym można "przy okazji" poprawiać cos φ całego zakładu (patrz S6). Silnik pracujący bez obciążenia tylko w tym celu to kompensator synchroniczny.

SPRÓBUJ: ustaw P i przesuwaj Ef - obserwuj kropkę na krzywej V i jak na wykresie wskazowym prąd obraca się z opóźnienia na wyprzedzenie.
""",

"S6": """PRZYKŁAD S6 - Poprawa współczynnika mocy zakładu silnikiem synchronicznym

SYTUACJA: zakład pobiera moc P_z przy cos φ_z = 0,7 (dużo silników indukcyjnych - pobierają moc bierną). Dostawca energii karze za niski cos φ, bo prąd w liniach jest większy niż trzeba (większe straty i spadki napięcia).

ROZWIĄZANIE: dokupujemy silnik synchroniczny (np. do sprężarki) o mocy P_s i PRZEWZBUDZAMY go, żeby oddawał moc bierną Q_s.
  Moc całkowita:   P = P_z + P_s,     Q = Q_z - Q_s,     S = √(P² + Q²),    cos φ = P / S
TRÓJKĄT MOCY: pozioma strzałka = moc czynna (robi pożyteczną pracę), pionowa = bierna ("przelewa się" tam i z powrotem), przeciwprostokątna = pozorna (decyduje o prądzie i grubości kabli).

Program liczy też, jakiego Ef i prądu potrzebuje silnik - i sprawdza, czy nie przekracza on prądu znamionowego (przegrzanie!).

UWAGA INŻYNIERA: zwykle nie kompensuje się do cos φ = 1, tylko do ~0,95 - dalsza kompensacja kosztuje dużo (duży silnik), a zysk jest mały.

SPRÓBUJ: znajdź Q_s, przy którym cos φ zakładu = 0,95; sprawdź, ile wynosi wtedy prąd silnika.
""",

"S7": """PRZYKŁAD S7 - Maszyna z biegunami jawnymi: teoria dwóch reakcji (Blondela)

BIEGUNY JAWNE: wirnik hydrogeneratora (wolnoobrotowy, dużo biegunów) ma wystające bieguny. Szczelina powietrzna nad biegunem (oś d) jest mała, a między biegunami (oś q) duża. Dlatego strumień łatwiej płynie w osi d -> Xd > Xq (typowo Xd ≈ 1,0, Xq ≈ 0,6 p.u.).

METODA: prąd twornika dzielimy na dwie składowe: Id (wzdłuż osi d) i Iq (wzdłuż osi q), każda ma swoją reaktancję:
   Ef = V + j·Xd·Id + j·Xq·Iq
Sztuczka: najpierw liczymy E' = V + j·Xq·Ia - jej kierunek wyznacza oś q i kąt δ. Potem  Ef = |E'| + (Xd - Xq)·Id.
PRZYKŁAD LICZBOWY: V = 1, Ia = 1, cos φ = 0,8 ind., Xd = 1, Xq = 0,6  ->  E' = 1,36 + j0,48 = 1,442∠19,4°,
Id = Ia·sin(δ+φ) = 0,832,  Ef = 1,442 + 0,4·0,832 = 1,775 p.u.

MOC:   P = V·Ef/Xd · sin δ  +  V²/2 · (1/Xq - 1/Xd) · sin 2δ
Drugi składnik to MOC RELUKTANCYJNA - istnieje nawet bez wzbudzenia! Wirnik z "wystającym" żelazem sam ustawia się wzdłuż pola (jak gwóźdź przy magnesie). Maksimum mocy przesuwa się poniżej 90°.

SPRÓBUJ: ustaw Xq = Xd - wróć do maszyny cylindrycznej (moc reluktancyjna znika). Zmniejsz Xq - zielona krzywa rośnie.
""",

"S8": """PRZYKŁAD S8 - Wykres kołowy P-Q (capability curve): gdzie generator może bezpiecznie pracować?

Każdy punkt (P, Q) to stan pracy generatora na sieci sztywnej V = 1. Obszar dozwolony ograniczają:
 1) OKRĄG PRĄDU TWORNIKA (czerwony):  P² + Q² ≤ (V·Ia_max)² - grzanie uzwojenia stojana.
 2) OKRĄG PRĄDU WZBUDZENIA (niebieski): środek w (0, -V²/Xs), promień V·Ef_max/Xs - grzanie wirnika. Ogranicza pracę przy dużym przewzbudzeniu.
 3) MOC TURBINY (pozioma linia): P ≤ P_turbiny.
 4) GRANICA STABILNOŚCI (zielona): przy niedowzbudzeniu kąt δ zbliża się do 90°; z zapasem praktycznym używa się granicy o np. 10 % mocy niższej.
 5) Minimalne wzbudzenie (Ef_min).

Punkt pracy (kropka) - zadajesz P i Q, program liczy prąd, Ef i δ oraz sprawdza każde ograniczenie.

UWAGA INŻYNIERA: operator elektrowni ma taki wykres na ścianie. Dyspozytor sieci często prosi o pobór mocy biernej nocą (gdy linie są mało obciążone i napięcie rośnie) - wtedy pracujemy blisko granicy stabilności.

SPRÓBUJ: przesuń Q mocno w dół (niedowzbudzenie) - wejdziesz w obszar niestabilny.
""",

"S9": """PRZYKŁAD S9 - Straty i sprawność generatora

Nie cała moc z turbiny zamienia się w energię elektryczną. Straty to:
 • mechaniczne (tarcie w łożyskach, wentylacja) - stałe,
 • w żelazie (histereza, prądy wirowe) - prawie stałe przy stałym napięciu,
 • w miedzi twornika  3·I²·Ra - rosną z KWADRATEM prądu,
 • w miedzi wzbudzenia - zależą od If (większe przy obciążeniu indukcyjnym),
 • dodatkowe (rozproszenie) - zwykle ~ I².
Sprawność:  η = P_wyj / (P_wyj + ΣP_strat).

CIEKAWOSTKA: maksimum sprawności wypada tam, gdzie straty zmienne (∝ I²) = straty stałe. Duże generatory mają η ≈ 98,5-99 %, ale przy 500 MW nawet 1 % to 5 MW ciepła - trzeba je odprowadzić (wodór, woda w prętach!).

UWAGA INŻYNIERA: przy niższym cos φ generator oddaje mniej mocy czynnej przy tym samym prądzie, a straty w miedzi pozostają - sprawność spada.

SPRÓBUJ: zwiększ straty stałe - maksimum sprawności przesuwa się w stronę większych obciążeń.
""",

"S10": """PRZYKŁAD S10 - Praca równoległa dwóch generatorów: kto ile daje?

STATYZM (droop): regulator turbiny jest tak zbudowany, że częstotliwość lekko spada, gdy rośnie moc:
     f = f_bj - k·P     (f_bj - częstotliwość biegu jałowego, "punkt nastawczy")
Dwa generatory połączone razem MUSZĄ mieć tę samą częstotliwość. Suma ich mocy = moc odbiorów. Na "wykresie domkowym" (house diagram) prosta G1 idzie w prawo, G2 w lewo - punkt wspólny to podział obciążenia.

MOC BIERNA dzieli się podobnie, ale przez napięcie:  V = V_bj - k_q·Q - zależy od prądu wzbudzenia.

WNIOSKI:
 • podniesienie nastawy f_bj1 -> G1 przejmuje moc od G2 (a f rośnie),
 • generator z mniejszym statyzmem (bardziej płaska prosta) bierze większą część zmian obciążenia,
 • zwiększenie wzbudzenia G1 przesuwa moc BIERNĄ, a nie czynną.

UWAGA INŻYNIERA: w dużym systemie energetycznym (tysiące generatorów) częstotliwość jest wspólna - to "puls" sieci. W Europie: 50 Hz, statyzm typowo 4-5 %.

SPRÓBUJ: zwiększ obciążenie - obie maszyny dokładają mocy, f spada. Potem podnieś f_bj1, by przywrócić 50 Hz.
""",

"S11": """PRZYKŁAD S11 - Synchronizacja generatora z siecią

Zanim zamkniemy wyłącznik łączący generator z siecią, muszą być spełnione WARUNKI SYNCHRONIZACJI:
 1) równe napięcia (wartości skuteczne),
 2) równe częstotliwości,
 3) zgodne fazy (kąt ≈ 0),
 4) ta sama kolejność faz (sprawdzana raz, przy montażu).

Na wykresie widać napięcie generatora, sieci i RÓŻNICĘ między nimi - to napięcie pojawia się na stykach wyłącznika. Przy różnicy częstotliwości różnica "pulsuje" (dudnienia - jak dwie struny gitary lekko rozstrojone). Synchronoskop obraca się z częstotliwością Δf; wyłącznik zamyka się, gdy wskazówka jest "na godzinie 12".

Jeżeli zamkniemy w złym momencie, popłynie prąd wyrównawczy  I ≈ ΔU / (X_gen + X_sieci)  - może być kilka razy większy od znamionowego i daje ogromny udar momentu na wale!

UWAGA INŻYNIERA: w praktyce: ΔU < 5 %, Δf < 0,1-0,2 Hz, Δθ < 10°. Generator lekko "szybszy" od sieci, by po załączeniu od razu oddawał moc (a nie pracował jako silnik).

SPRÓBUJ: ustaw Δf = 0,5 Hz - zobacz dudnienia. Ustaw kąt 180° - najgorszy moment załączenia, ΔU ≈ 2 p.u.
""",
}


# =============================================================================
# GUI - klasa bazowa zakładki
# =============================================================================

class ExampleTab(ttk.Frame):
    """Zakładka: lewy panel suwaków + wyniki liczbowe, prawy - wykresy,
    dół - objaśnienie. Podklasy definiują PARAMS, key, update_plot()."""
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

        ttk.Label(left, text="Parametry", font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=6, pady=(6, 2))
        for name, label, lo, hi, val, res in self.PARAMS:
            fr = ttk.Frame(left)
            fr.pack(fill=tk.X, padx=6, pady=1)
            var = tk.DoubleVar(value=val)
            self.vars[name] = var
            ttk.Label(fr, text=label, width=28).pack(anchor="w")
            sc = tk.Scale(fr, from_=lo, to=hi, resolution=res, orient=tk.HORIZONTAL,
                          variable=var, showvalue=True, length=220,
                          command=lambda _e: self.schedule())
            sc.pack(fill=tk.X, expand=True)
        ttk.Button(left, text="Przywróć domyślne", command=self.reset).pack(fill=tk.X, padx=6, pady=6)
        ttk.Label(left, text="Wyniki", font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=6)
        self.result = tk.Text(left, height=14, width=38, font=("Consolas", 9), bg="#f4f6fa")
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
        self.expl.insert("1.0", EXPL.get(self.key, ""))
        self.expl.configure(state=tk.DISABLED)
        vpw.add(figfr, weight=3)
        vpw.add(txtfr, weight=1)
        self.refresh()

    def reset(self):
        for name, _l, _lo, _hi, val, _r in self.PARAMS:
            self.vars[name].set(val)
        self.refresh()

    def p(self, name):
        return self.vars[name].get()

    def schedule(self):
        if self._pending:
            self.after_cancel(self._pending)
        self._pending = self.after(120, self.refresh)

    def refresh(self):
        self._pending = None
        self.fig.clear()
        lines = self.update_plot() or []
        self.result.delete("1.0", tk.END)
        self.result.insert("1.0", "\n".join(lines))
        self.canvas.draw_idle()

    def update_plot(self):
        raise NotImplementedError


def draw_phasor(ax, start, end, color, label, lw=2.0):
    """Strzałka wskazowa od start do end (liczby zespolone) z opisem."""
    ax.annotate("", xy=(end.real, end.imag), xytext=(start.real, start.imag),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, shrinkA=0, shrinkB=0))
    mid = start + 0.55 * (end - start)
    ax.text(mid.real, mid.imag, " " + label, color=color, fontsize=9, fontweight="bold")


def phasor_axes(ax, points, title):
    pts = [0j] + list(points)
    xs = [z.real for z in pts]
    ys = [z.imag for z in pts]
    pad = 0.15 * max(max(xs) - min(xs), max(ys) - min(ys), 0.5)
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.axhline(0, color="0.8", lw=0.8)
    ax.axvline(0, color="0.8", lw=0.8)
    ax.grid(alpha=.3)
    ax.set_title(title)


def pf_text(phi_deg):
    c = math.cos(phi_deg * DEG)
    if abs(phi_deg) < 0.5:
        return f"cos φ = {c:.3f} (czynne)"
    return f"cos φ = {c:.3f} ({'ind./opóźn.' if phi_deg > 0 else 'poj./wyprz.'})"


# =============================================================================
# ZAKŁADKI
# =============================================================================

class TabS1(ExampleTab):
    key = "S1"
    PARAMS = [("xsu", "Xs nienasycona [p.u.]", 0.5, 3.0, 1.6, 0.05),
              ("sat", "nasycenie s (OCC)", 0.0, 0.5, 0.2, 0.01),
              ("vll", "U_N przewodowe [kV]", 0.4, 30.0, 13.8, 0.1),
              ("sn", "S_N [MVA]", 0.1, 500, 50, 0.1),
              ("ifb", "If bazowy (prosta szczeliny, 1 p.u.) [A]", 10, 2000, 400, 10)]

    def update_plot(self):
        xsu, s = self.p("xsu"), self.p("sat")
        vll, sn, ifb = self.p("vll"), self.p("sn"), self.p("ifb")
        if0 = occ_inverse(1.0, s) if s < 1 else float("inf")   # If dla U_N (bieg jałowy)
        ifsc = xsu                                              # If dla I_N przy zwarciu
        xss = xsu / if0
        scr = if0 / ifsc
        zb = vll ** 2 / sn
        i_f = np.linspace(0, 2.5, 300)
        ax = self.fig.add_subplot(1, 2, 1)
        ax.plot(i_f, occ(i_f, s), "b", lw=2, label="OCC  E(If) - bieg jałowy")
        ax.plot(i_f, i_f, "b--", lw=1, label="prosta szczeliny powietrznej")
        ax.set_ylim(0, 1.6)
        ax.set_xlabel("prąd wzbudzenia If [p.u.]")
        ax.set_ylabel("napięcie E [p.u.]", color="b")
        ax2 = ax.twinx()
        ax2.plot(i_f, i_f / xsu, "r", lw=2, label="SCC  Isc(If) - zwarcie")
        ax2.set_ylabel("prąd zwarcia Isc [p.u.]", color="r")
        ax2.set_ylim(0, 1.6)
        ax.axhline(1.0, color="0.5", ls=":")
        ax.axvline(if0, color="b", ls=":")
        ax.axvline(ifsc, color="r", ls=":")
        ax.text(if0, 0.05, f" If0={if0:.2f}", color="b", fontsize=8)
        ax.text(ifsc, 0.15, f" Ifsc={ifsc:.2f}", color="r", fontsize=8)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)
        ax.grid(alpha=.3)
        ax.set_title("Próby: biegu jałowego i zwarcia")

        a3 = self.fig.add_subplot(1, 2, 2)
        e = occ(i_f[1:], s)
        isc = i_f[1:] / xsu
        a3.plot(i_f[1:], e / isc, "k", lw=2, label="Xs = E_oc / I_sc (z OCC)")
        a3.axhline(xsu, color="b", ls="--", label=f"Xs nienasycona = {xsu:.2f}")
        a3.axhline(xss, color="g", ls="--", label=f"Xs nasycona = {xss:.2f}")
        a3.axvline(if0, color="0.5", ls=":")
        a3.set_xlabel("If [p.u.]"); a3.set_ylabel("Xs [p.u.]")
        a3.set_ylim(0, xsu * 1.15)
        a3.legend(fontsize=8); a3.grid(alpha=.3)
        a3.set_title("Reaktancja synchroniczna a nasycenie")
        return [f"If0 (U_N, bieg jałowy) = {if0:.3f} p.u. = {if0 * ifb:.0f} A",
                f"Ifsc (I_N, zwarcie)    = {ifsc:.3f} p.u. = {ifsc * ifb:.0f} A",
                "",
                f"Xs nienasycona = {xsu:.3f} p.u.",
                f"Xs nasycona    = {xss:.3f} p.u.",
                f"SCR = If0/Ifsc = {scr:.3f}",
                f"1/Xs_nas       = {1 / xss:.3f}",
                "",
                f"Z_bazowa = U²/S = {zb:.3f} Ω",
                f"Xs_nien  = {xsu * zb:.3f} Ω/faza",
                f"Xs_nas   = {xss * zb:.3f} Ω/faza",
                f"I_N      = {sn / (SQ3 * vll) * 1e3:.0f} A"]


class TabS2(ExampleTab):
    key = "S2"
    PARAMS = [("ia", "prąd twornika Ia [p.u.]", 0.0, 1.5, 1.0, 0.01),
              ("phi", "kąt φ [°] (+ind., -poj.)", -90, 90, 36.87, 0.5),
              ("xs", "Xs [p.u.]", 0.2, 3.0, 1.2, 0.05),
              ("ra", "Ra [p.u.]", 0.0, 0.2, 0.02, 0.005),
              ("v", "V na zaciskach [p.u.]", 0.5, 1.2, 1.0, 0.01),
              ("vll", "U_N przewodowe [kV]", 0.4, 30.0, 13.8, 0.1),
              ("sn", "S_N [MVA]", 0.1, 500, 50, 0.1)]

    def update_plot(self):
        ia, phi, xs, ra, v = (self.p(k) for k in ("ia", "phi", "xs", "ra", "v"))
        vll, sn = self.p("vll"), self.p("sn")
        ef, i = gen_ef(v, ia, phi, ra, xs)
        vr = complex(v, 0)
        p_ra = vr + ra * i
        ax = self.fig.add_subplot(1, 2, 1)
        k = 0.8 / max(ia, 1e-6) * max(v, 0.5) * 0.7   # skala prądu na rysunku
        draw_phasor(ax, 0j, vr, "k", "V")
        draw_phasor(ax, 0j, i * k, "r", "Ia")
        draw_phasor(ax, vr, p_ra, "tab:orange", "Ra·Ia", lw=1.5)
        draw_phasor(ax, p_ra, ef, "m", "jXs·Ia", lw=1.5)
        draw_phasor(ax, 0j, ef, "b", "Ef", lw=2.5)
        phasor_axes(ax, [vr, ef, i * k, p_ra], f"Wykres wskazowy (δ = {math.degrees(cmath.phase(ef)):.1f}°)")

        a2 = self.fig.add_subplot(1, 2, 2)
        phis = np.linspace(-90, 90, 181)
        efs = [abs(gen_ef(v, ia, ph, ra, xs)[0]) for ph in phis]
        dls = [math.degrees(cmath.phase(gen_ef(v, ia, ph, ra, xs)[0])) for ph in phis]
        a2.plot(phis, efs, "b", lw=2, label="|Ef| [p.u.]")
        a2.plot(phi, abs(ef), "bo")
        a2.set_xlabel("kąt φ [°]   (← pojemnościowe | indukcyjne →)")
        a2.set_ylabel("|Ef| [p.u.]", color="b")
        a2.grid(alpha=.3)
        a3 = a2.twinx()
        a3.plot(phis, dls, "g--", lw=1.5, label="δ [°]")
        a3.plot(phi, math.degrees(cmath.phase(ef)), "go")
        a3.set_ylabel("δ [°]", color="g")
        a2.set_title("Wymagana SEM i kąt mocy przy stałym Ia")
        p, q = v * ia * math.cos(phi * DEG), v * ia * math.sin(phi * DEG)
        return [f"Ia = {ia:.3f} p.u.   {pf_text(phi)}",
                f"Ef = {abs(ef):.4f} p.u. ∠ {math.degrees(cmath.phase(ef)):.2f}°",
                f"   = {abs(ef) * vll / SQ3:.3f} kV fazowe",
                f"   = {abs(ef) * vll:.3f} kV przewodowe",
                f"δ  = {math.degrees(cmath.phase(ef)):.2f}°",
                "",
                f"P = {p:.3f} p.u. = {p * sn:.2f} MW",
                f"Q = {q:.3f} p.u. = {q * sn:.2f} Mvar",
                f"Straty Cu = Ia²Ra = {ia ** 2 * ra:.4f} p.u.",
                f"P_mech ≈ P + straty Cu = {(p + ia ** 2 * ra) * sn:.2f} MW",
                f"Ia fiz. = {ia * sn / (SQ3 * vll) * 1e3:.0f} A",
                f"Zmienność napięcia ΔU = {100 * (abs(ef) - v) / v:.1f} %"]


class TabS3(ExampleTab):
    key = "S3"
    PARAMS = [("xs", "Xs [p.u.]", 0.2, 3.0, 1.2, 0.05),
              ("ra", "Ra [p.u.]", 0.0, 0.2, 0.02, 0.005),
              ("phi", "φ punktu roboczego [°]", -90, 90, 36.87, 0.5),
              ("imax", "zakres prądu Ia_max [p.u.]", 0.5, 3.0, 1.5, 0.1)]

    def update_plot(self):
        xs, ra, phi, imax = self.p("xs"), self.p("ra"), self.p("phi"), self.p("imax")
        ia = np.linspace(0, imax, 300)
        cases = [(36.87, "cos φ=0,8 ind.", "r"), (0.0, "cos φ=1", "k"),
                 (-36.87, "cos φ=0,8 poj.", "b"), (phi, f"φ={phi:.1f}° (wybrany)", "g")]
        a1 = self.fig.add_subplot(2, 2, 1)
        a2 = self.fig.add_subplot(2, 2, 2)
        for ph, lab, col in cases:
            ef0 = abs(gen_ef(1.0, 1.0, ph, ra, xs)[0])   # Ef dobrane: V=1 przy Ia=1
            a1.plot(ia, terminal_voltage(ef0, ia, ph, ra, xs), col, lw=2 if col == "g" else 1.3, label=lab)
            efr = [abs(gen_ef(1.0, x, ph, ra, xs)[0]) for x in ia]
            a2.plot(ia, efr, col, lw=2 if col == "g" else 1.3, label=lab)
        a1.axhline(1.0, color="0.5", ls=":"); a1.axvline(1.0, color="0.5", ls=":")
        a1.set_xlabel("Ia [p.u.]"); a1.set_ylabel("V [p.u.]"); a1.set_ylim(0, 2.5)
        a1.set_title("Charakterystyki zewnętrzne (If = const)"); a1.grid(alpha=.3); a1.legend(fontsize=7)
        a2.set_xlabel("Ia [p.u.]"); a2.set_ylabel("Ef ~ If [p.u.]")
        a2.set_title("Charakterystyki regulacyjne (V = 1)"); a2.grid(alpha=.3); a2.legend(fontsize=7)

        a3 = self.fig.add_subplot(2, 1, 2)
        phis = np.linspace(-90, 90, 181)
        reg = np.array([100 * (abs(gen_ef(1.0, 1.0, x, ra, xs)[0]) - 1.0) for x in phis])
        a3.plot(phis, reg, "m", lw=2)
        a3.axhline(0, color="k", lw=0.8)
        r0 = 100 * (abs(gen_ef(1.0, 1.0, phi, ra, xs)[0]) - 1.0)
        a3.plot(phi, r0, "go", ms=8)
        a3.set_xlabel("φ [°]  (← poj. | ind. →)"); a3.set_ylabel("ΔU [%]")
        a3.set_title("Zmienność napięcia przy Ia = 1 p.u."); a3.grid(alpha=.3)
        zero = phis[np.argmin(np.abs(reg))]
        return [f"Punkt roboczy: {pf_text(phi)}",
                f"Ef (V=1, Ia=1) = {abs(gen_ef(1.0, 1.0, phi, ra, xs)[0]):.4f} p.u.",
                f"Zmienność ΔU   = {r0:.2f} %",
                "",
                "ΔU dla typowych obciążeń:",
                f"  0,8 ind.: {100 * (abs(gen_ef(1, 1, 36.87, ra, xs)[0]) - 1):7.2f} %",
                f"  1,0     : {100 * (abs(gen_ef(1, 1, 0, ra, xs)[0]) - 1):7.2f} %",
                f"  0,8 poj.: {100 * (abs(gen_ef(1, 1, -36.87, ra, xs)[0]) - 1):7.2f} %",
                "",
                f"ΔU = 0 przy φ ≈ {zero:.1f}° (poj.)",
                f"  (cos φ ≈ {math.cos(zero * DEG):.3f})"]


class TabS4(ExampleTab):
    key = "S4"
    PARAMS = [("ef", "Ef [p.u.]", 0.3, 2.5, 1.6, 0.01),
              ("v", "V sieci [p.u.]", 0.8, 1.2, 1.0, 0.01),
              ("xs", "Xs [p.u.]", 0.3, 3.0, 1.2, 0.05),
              ("pm", "P turbiny [p.u.]", 0.0, 2.5, 0.8, 0.01)]

    def update_plot(self):
        ef, v, xs, pm = self.p("ef"), self.p("v"), self.p("xs"), self.p("pm")
        d = np.linspace(-PI, PI, 721)
        p, q = power_angle(v, ef, xs, d)
        pmax = v * ef / xs
        ax = self.fig.add_subplot(2, 1, 1)
        ax.plot(np.degrees(d), p, "b", lw=2, label="P(δ)")
        ax.plot(np.degrees(d), q, "g", lw=1.5, label="Q(δ)")
        ax.axhline(pm, color="r", ls="--", label="P turbiny")
        ax.axvspan(90, 180, color="red", alpha=0.07)
        ax.axvspan(-180, -90, color="red", alpha=0.07)
        ax.text(120, -pmax * 0.9, "niestabilnie", color="r", fontsize=8)
        ax.axhline(0, color="k", lw=0.6)
        lines = [f"P_max = V·Ef/Xs = {pmax:.3f} p.u.", ""]
        if pm <= pmax:
            d0 = math.asin(pm / pmax)
            p0, q0 = power_angle(v, ef, xs, d0)
            ax.plot(math.degrees(d0), p0, "ro", ms=8)
            ax.plot(math.degrees(PI - d0), pm, "rx", ms=8)
            ia = abs(complex(p0, -q0) / v)
            phi = math.degrees(math.atan2(q0, p0))
            lines += [f"δ stabilny   = {math.degrees(d0):.2f}°",
                      f"δ niestabilny= {180 - math.degrees(d0):.2f}°",
                      f"Q = {q0:.3f} p.u. ({'oddaje' if q0 >= 0 else 'pobiera'})",
                      f"Ia = {ia:.3f} p.u.   {pf_text(phi)}",
                      f"Zapas k = Pmax/P = {pmax / pm:.2f}" if pm > 0 else "P = 0 (bieg jałowy na sieci)",
                      f"dP/dδ (synchr.) = {pmax * math.cos(d0):.3f} p.u./rad"]
        else:
            ax.text(-170, pmax * 1.05, "P_turbiny > P_max  →  UTRATA SYNCHRONIZMU!",
                    color="r", fontsize=11, fontweight="bold")
            lines += ["!!! P > P_max", "Brak punktu równowagi -", "wirnik wypada z synchronizmu.",
                      f"Potrzebne Ef ≥ {pm * xs / v:.3f} p.u."]
        ax.set_xlabel("kąt mocy δ [°]"); ax.set_ylabel("P, Q [p.u.]")
        ax.set_xlim(-180, 180); ax.grid(alpha=.3); ax.legend(fontsize=8, loc="lower right")
        ax.set_title("Charakterystyka kątowa generatora cylindrycznego")

        a2 = self.fig.add_subplot(2, 1, 2)
        for e, c in [(0.8, "c"), (1.2, "tab:blue"), (1.6, "navy"), (2.0, "k")]:
            a2.plot(np.degrees(d[d >= 0]), power_angle(v, e, xs, d[d >= 0])[0], color=c, lw=1.2,
                    label=f"Ef = {e}")
        a2.plot(np.degrees(d[d >= 0]), p[d >= 0], "b", lw=2.5, label=f"Ef = {ef:.2f} (wybrane)")
        a2.axhline(pm, color="r", ls="--")
        a2.set_xlim(0, 180); a2.set_xlabel("δ [°]"); a2.set_ylabel("P [p.u.]")
        a2.grid(alpha=.3); a2.legend(fontsize=8, ncol=5)
        a2.set_title("Wpływ wzbudzenia na moc maksymalną")
        return lines


class TabS5(ExampleTab):
    key = "S5"
    PARAMS = [("p", "moc silnika P [p.u.]", 0.0, 1.2, 0.6, 0.01),
              ("ef", "Ef (wzbudzenie) [p.u.]", 0.2, 2.5, 1.3, 0.01),
              ("xs", "Xs [p.u.]", 0.3, 2.5, 1.0, 0.05),
              ("v", "V sieci [p.u.]", 0.8, 1.2, 1.0, 0.01)]

    def update_plot(self):
        pp, ef, xs, v = self.p("p"), self.p("ef"), self.p("xs"), self.p("v")
        efs = np.linspace(0.05, 2.6, 400)
        ax = self.fig.add_subplot(2, 2, (1, 3))
        a2 = self.fig.add_subplot(2, 2, 2)
        for pk, col in [(0.0, "0.4"), (0.25, "c"), (0.5, "tab:blue"), (0.75, "navy"), (1.0, "k")]:
            ia_c, pf_c = [], []
            for e in efs:
                i, _ = motor_state(v, e, xs, pk)
                ia_c.append(abs(i) if i is not None else np.nan)
                pf_c.append(math.copysign(pk / (v * abs(i)), i.imag) if (i is not None and abs(i) > 1e-9) else np.nan)
            ax.plot(efs, ia_c, color=col, lw=1.3, label=f"P = {pk}")
            a2.plot(efs, pf_c, color=col, lw=1.3)
        # granica stabilności: δ = 90° -> Ef = P Xs / V, Ia = |V - (-j Ef)|/Xs
        pl = np.linspace(0.01, 1.2, 50)
        e_lim = pl * xs / v
        ia_lim = np.abs(v + 1j * e_lim) / xs
        ax.plot(e_lim, ia_lim, "r--", lw=1.2, label="granica δ=90°")
        i0, d0 = motor_state(v, ef, xs, pp)
        lines = []
        if i0 is not None:
            ax.plot(ef, abs(i0), "ro", ms=9)
            pf = pp / (v * abs(i0)) if abs(i0) > 1e-9 else 1.0
            a2.plot(ef, math.copysign(pf, i0.imag), "ro", ms=8)
            kind = "wyprzedzający (przewzbudzony, oddaje Q)" if i0.imag > 1e-6 else \
                   ("opóźniony (niedowzbudzony, pobiera Q)" if i0.imag < -1e-6 else "jednostkowy")
            ef_unity = abs(complex(v, -xs * pp / v))
            lines = [f"Ia = {abs(i0):.3f} p.u. ∠ {math.degrees(cmath.phase(i0)):.1f}°",
                     f"δ  = {math.degrees(d0):.2f}°",
                     f"cos φ = {pf:.3f}",
                     f"  {kind}",
                     f"Q oddawane do sieci = {v * i0.imag:.3f} p.u.",
                     "",
                     f"Ef dla cos φ = 1: {ef_unity:.3f} p.u.",
                     f"Ia_min = P/V = {pp / v:.3f} p.u.",
                     f"P_max przy tym Ef = {v * ef / xs:.3f} p.u."]
            a3 = self.fig.add_subplot(2, 2, 4)
            k = 0.7 / max(abs(i0), 0.1)
            efp = cmath.rect(ef, -d0)
            draw_phasor(a3, 0j, complex(v, 0), "k", "V")
            draw_phasor(a3, 0j, efp, "b", "Ef")
            draw_phasor(a3, efp, complex(v, 0), "m", "jXs·Ia", lw=1.5)
            draw_phasor(a3, 0j, i0 * k, "r", "Ia")
            phasor_axes(a3, [complex(v, 0), efp, i0 * k], "Wykres wskazowy silnika")
        else:
            ax.text(0.1, 2.4, "Ef za małe - silnik wypada z synchronizmu!", color="r", fontweight="bold")
            lines = ["!!! P > V·Ef/Xs", "Silnik traci synchronizm.", f"Minimalne Ef = {pp * xs / v:.3f} p.u."]
        ax.set_xlabel("Ef ~ prąd wzbudzenia [p.u.]"); ax.set_ylabel("Ia [p.u.]")
        ax.set_ylim(0, 2.6); ax.grid(alpha=.3); ax.legend(fontsize=8)
        ax.set_title("Krzywe V silnika synchronicznego")
        a2.axhline(0, color="k", lw=0.6)
        a2.set_ylabel("cos φ (+ wyprz. / - opóźn.)"); a2.set_xlabel("Ef [p.u.]")
        a2.set_ylim(-1.05, 1.05); a2.grid(alpha=.3); a2.set_title("Współczynnik mocy")
        return lines


class TabS6(ExampleTab):
    key = "S6"
    PARAMS = [("pz", "moc zakładu P_z [kW]", 50, 2000, 800, 10),
              ("pfz", "cos φ zakładu (ind.)", 0.4, 1.0, 0.7, 0.01),
              ("ps", "moc silnika synchr. P_s [kW]", 0, 1000, 200, 10),
              ("qs", "moc bierna silnika Q_s [kvar]", -300, 1200, 500, 10),
              ("ssn", "S_N silnika [kVA]", 50, 1500, 600, 10),
              ("xs", "Xs silnika [p.u.]", 0.3, 2.0, 1.0, 0.05),
              ("ull", "U sieci [V]", 230, 11000, 6000, 10)]

    def update_plot(self):
        pz, pfz, ps, qs, ssn, xs, ull = (self.p(k) for k in ("pz", "pfz", "ps", "qs", "ssn", "xs", "ull"))
        qz = pz * math.tan(math.acos(pfz))
        pt, qt = pz + ps, qz - qs
        st = math.hypot(pt, qt)
        pft = pt / st if st > 0 else 1.0
        q95 = qz - pt * math.tan(math.acos(0.95))
        ax = self.fig.add_subplot(1, 2, 1)
        # trójkąty mocy
        ax.annotate("", xy=(pz, qz), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="r", lw=2))
        ax.text(pz / 2, qz / 2 + 20, f"S_z (cos φ={pfz:.2f})", color="r", fontsize=9)
        ax.annotate("", xy=(pz + ps, qz - qs), xytext=(pz, qz),
                    arrowprops=dict(arrowstyle="-|>", color="b", lw=2))
        ax.text(pz + ps / 2, qz - qs / 2, " S_s (silnik)", color="b", fontsize=9)
        ax.annotate("", xy=(pt, qt), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="g", lw=2.5))
        ax.text(pt * 0.55, qt * 0.55 - 40, f"S_całk (cos φ={pft:.3f})", color="g", fontsize=9)
        ax.plot([0, pt], [0, pt * math.tan(math.acos(0.95))], "k:", lw=1, label="cos φ = 0,95")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xlabel("P [kW]"); ax.set_ylabel("Q [kvar]  (+ pobierana indukcyjna)")
        ax.set_aspect("equal", adjustable="datalim"); ax.grid(alpha=.3); ax.legend(fontsize=8)
        ax.set_title("Trójkąty mocy zakładu")

        a2 = self.fig.add_subplot(2, 2, 2)
        qq = np.linspace(-300, 1200, 300)
        pfs = (pz + ps) / np.hypot(pz + ps, qz - qq)
        a2.plot(qq, pfs, "g", lw=2)
        a2.plot(qs, pft, "go", ms=8)
        a2.axhline(0.95, color="k", ls=":")
        a2.set_xlabel("Q_s silnika [kvar]"); a2.set_ylabel("cos φ zakładu"); a2.grid(alpha=.3)
        a2.set_title("cos φ zakładu a moc bierna silnika")

        # stan silnika w p.u. (baza: S_N silnika, U sieci)
        p_pu, q_pu = ps / ssn, qs / ssn
        ia_pu = math.hypot(p_pu, q_pu)
        i_ph = complex(p_pu, q_pu)          # prąd silnika (konwencja odbiornika) - Q oddawane -> wyprzedzający
        efp = 1.0 - 1j * xs * i_ph
        ef_pu = abs(efp)
        a3 = self.fig.add_subplot(2, 2, 4)
        il = st * 1e3 / (SQ3 * ull)
        il0 = math.hypot(pz, qz) * 1e3 / (SQ3 * ull) if ps == 0 else math.hypot(pt, qz) * 1e3 / (SQ3 * ull)
        bars = a3.bar(["prąd linii\nbez kompensacji", "prąd linii\nz kompensacją", "prąd silnika"],
                      [il0, il, ia_pu * ssn * 1e3 / (SQ3 * ull)], color=["r", "g", "b"])
        a3.axhline(ssn * 1e3 / (SQ3 * ull), color="b", ls="--", lw=1)
        a3.bar_label(bars, fmt="%.1f A", fontsize=8)
        a3.set_ylabel("prąd [A]"); a3.set_title("Prądy (linia przer. = I_N silnika)")
        warn = "  !!! PRZECIĄŻONY (Ia > 1)" if ia_pu > 1.0 else ""
        return [f"Zakład:  Q_z = {qz:.0f} kvar",
                f"Razem:   P = {pt:.0f} kW, Q = {qt:.0f} kvar",
                f"         S = {st:.0f} kVA",
                f"cos φ całk. = {pft:.3f} ({'ind.' if qt >= 0 else 'poj.'})",
                f"Q_s dla cos φ=0,95: {q95:.0f} kvar",
                "",
                "Silnik synchroniczny (p.u.):",
                f"  P = {p_pu:.3f}, Q = {q_pu:.3f}",
                f"  Ia = {ia_pu:.3f} p.u.{warn}",
                f"  Ef = {ef_pu:.3f} p.u. (δ = {math.degrees(-cmath.phase(efp)):.1f}°)",
                f"  cos φ silnika = {p_pu / ia_pu if ia_pu > 0 else 1:.3f} wyprz.",
                "",
                f"Prąd linii: {il0:.1f} A -> {il:.1f} A"]


class TabS7(ExampleTab):
    key = "S7"
    PARAMS = [("ia", "prąd twornika Ia [p.u.]", 0.0, 1.5, 1.0, 0.01),
              ("phi", "kąt φ [°] (+ind., -poj.)", -90, 90, 36.87, 0.5),
              ("xd", "Xd [p.u.]", 0.3, 2.0, 1.0, 0.05),
              ("xq", "Xq [p.u.]", 0.2, 2.0, 0.6, 0.05),
              ("v", "V [p.u.]", 0.8, 1.2, 1.0, 0.01)]

    def update_plot(self):
        ia, phi, xd, xq, v = (self.p(k) for k in ("ia", "phi", "xd", "xq", "v"))
        xq = min(xq, xd)
        r = salient_gen(v, ia, phi, xd, xq)
        vr = complex(v, 0)
        ax = self.fig.add_subplot(1, 2, 1)
        k = 0.7 / max(ia, 0.1)
        draw_phasor(ax, 0j, vr, "k", "V")
        draw_phasor(ax, 0j, r["I"] * k, "r", "Ia")
        draw_phasor(ax, 0j, r["Id"] * k, "tab:orange", "Id", lw=1.3)
        draw_phasor(ax, 0j, r["Iq"] * k, "tab:pink", "Iq", lw=1.3)
        p1 = vr + 1j * xq * r["Iq"]
        draw_phasor(ax, vr, p1, "m", "jXq·Iq", lw=1.5)
        draw_phasor(ax, p1, r["Efp"], "c", "jXd·Id", lw=1.5)
        draw_phasor(ax, 0j, r["Efp"], "b", "Ef", lw=2.5)
        ax.plot([0, r["Eq"].real], [0, r["Eq"].imag], "b:", lw=1)
        ax.text(r["Eq"].real, r["Eq"].imag, " E' (oś q)", color="b", fontsize=8)
        phasor_axes(ax, [vr, r["Efp"], r["I"] * k, p1, r["Eq"]], "Wykres wskazowy - dwie reakcje")

        a2 = self.fig.add_subplot(1, 2, 2)
        d = np.linspace(0, PI, 361)
        pe, pr, pt = salient_power(v, r["ef"], xd, xq, d)
        a2.plot(np.degrees(d), pe, "b--", lw=1.3, label="od wzbudzenia")
        a2.plot(np.degrees(d), pr, "g--", lw=1.3, label="reluktancyjna")
        a2.plot(np.degrees(d), pt, "k", lw=2.2, label="całkowita")
        imax = int(np.argmax(pt))
        a2.plot(np.degrees(d[imax]), pt[imax], "k^", ms=8)
        p0 = v * ia * math.cos(phi * DEG)
        a2.plot(math.degrees(r["delta"]), p0, "ro", ms=8, label="punkt pracy")
        a2.axhline(0, color="k", lw=0.6)
        a2.set_xlabel("δ [°]"); a2.set_ylabel("P [p.u.]"); a2.grid(alpha=.3); a2.legend(fontsize=8)
        a2.set_title("Charakterystyka kątowa - bieguny jawne")
        ef_cyl = abs(gen_ef(v, ia, phi, 0.0, xd)[0])
        return [f"δ  = {math.degrees(r['delta']):.2f}°",
                f"|E'| = {abs(r['Eq']):.4f} p.u.",
                f"Id = {r['id']:.4f} p.u.  (>0: rozmagn.)",
                f"Iq = {r['iq']:.4f} p.u.",
                f"Ef = {r['ef']:.4f} p.u.",
                "",
                f"P = {p0:.3f} p.u.",
                f"P_max = {pt[imax]:.3f} p.u. przy δ = {np.degrees(d[imax]):.1f}°",
                f"P_rel,max = {v ** 2 / 2 * (1 / xq - 1 / xd):.3f} p.u.",
                "",
                f"Dla porównania - model",
                f"cylindryczny (Xs=Xd): Ef = {ef_cyl:.4f}"]


class TabS8(ExampleTab):
    key = "S8"
    PARAMS = [("p", "P zadane [p.u.]", 0.0, 1.2, 0.8, 0.01),
              ("q", "Q zadane [p.u.] (+oddawane)", -1.0, 1.0, 0.4, 0.01),
              ("xs", "Xs [p.u.]", 0.5, 2.5, 1.5, 0.05),
              ("iamax", "Ia max [p.u.]", 0.5, 1.3, 1.0, 0.01),
              ("efmax", "Ef max (wzbudzenie) [p.u.]", 1.0, 3.5, 2.3, 0.05),
              ("pt", "P turbiny max [p.u.]", 0.3, 1.2, 0.9, 0.01),
              ("marg", "zapas stabilności [% P_max]", 0, 30, 10, 1)]

    def update_plot(self):
        pp, qq, xs, iam, efm, ptm, marg = (self.p(k) for k in ("p", "q", "xs", "iamax", "efmax", "pt", "marg"))
        v = 1.0
        ax = self.fig.add_subplot(1, 1, 1)
        th = np.linspace(-PI / 2, PI / 2, 400)
        ax.plot(iam * np.cos(th), iam * np.sin(th), "r", lw=2, label="granica prądu twornika")
        c0 = -v ** 2 / xs
        rf = v * efm / xs
        thf = np.linspace(-PI / 2, PI / 2, 400)
        ax.plot(rf * np.cos(thf), c0 + rf * np.sin(thf), "b", lw=2, label="granica prądu wzbudzenia")
        ax.axhline(0, color="k", lw=0.6)
        ax.plot([ptm, ptm], [-1.5, 1.5], "k--", lw=1.5, label="moc turbiny")
        # granica stabilności praktycznej: dla danego Ef punkt o mocy (1-m) P_max
        efs = np.linspace(0.01, efm * 1.2, 300)
        pm_ = v * efs / xs
        pp_s = pm_ * (1 - marg / 100.0)
        dd = np.arcsin(np.clip(pp_s / pm_, -1, 1))
        qs_ = (v * efs * np.cos(dd) - v ** 2) / xs
        ax.plot(pp_s, qs_, "g", lw=2, label=f"praktyczna granica stabilności ({marg:.0f} %)")
        ax.plot([0, 0], [c0, 0], "g:", lw=1)
        ax.plot(0, c0, "g+", ms=12)
        ax.text(0.01, c0, " -V²/Xs (δ = 90° dla Ef→0)", color="g", fontsize=8)
        # obszar dozwolony (siatka)
        P, Q = np.meshgrid(np.linspace(0, 1.4, 200), np.linspace(-1.5, 1.5, 300))
        ok = (P ** 2 + Q ** 2 <= iam ** 2) & (P ** 2 + (Q - c0) ** 2 <= rf ** 2) & (P <= ptm)
        efg = np.hypot(P * xs / v, v + Q * xs / v)
        dg = np.arctan2(P * xs / v, v + Q * xs / v)
        ok &= (dg < PI / 2) & (P <= (v * efg / xs) * (1 - marg / 100.0) + 1e-9)
        ax.contourf(P, Q, ok.astype(float), levels=[0.5, 1.5], colors=["#c8f0c8"], alpha=0.6)
        # punkt pracy
        ef = complex(v + qq * xs / v, pp * xs / v)
        delta = math.degrees(cmath.phase(ef))
        ia = math.hypot(pp, qq) / v
        checks = [("Ia ≤ Ia_max", ia <= iam), ("Ef ≤ Ef_max", abs(ef) <= efm),
                  ("P ≤ P_turbiny", pp <= ptm),
                  ("stabilność (z zapasem)", delta < 90 and pp <= v * abs(ef) / xs * (1 - marg / 100) + 1e-9)]
        allok = all(c for _, c in checks)
        ax.plot(pp, qq, "o", color="g" if allok else "r", ms=11, mec="k")
        ax.set_xlim(-0.1, 1.4); ax.set_ylim(-1.5, 1.4)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("P [p.u.]"); ax.set_ylabel("Q [p.u.]  (+ przewzbudzony / - niedowzbudzony)")
        ax.grid(alpha=.3); ax.legend(fontsize=8, loc="lower left")
        ax.set_title("Wykres kołowy (capability) generatora - zielony obszar = dozwolona praca")
        lines = [f"Punkt: P = {pp:.2f}, Q = {qq:.2f} p.u.",
                 f"Ia = {ia:.3f} p.u.",
                 f"Ef = {abs(ef):.3f} p.u.",
                 f"δ  = {delta:.1f}°",
                 f"cos φ = {pp / ia if ia > 0 else 1:.3f}",
                 ""]
        lines += [f"[{'OK' if c else '!!'}] {n}" for n, c in checks]
        lines += ["", "=> PRACA DOZWOLONA" if allok else "=> POZA OBSZAREM PRACY"]
        return lines


class TabS9(ExampleTab):
    key = "S9"
    PARAMS = [("sn", "S_N [MVA]", 1, 500, 100, 1),
              ("pf", "cos φ (ind.)", 0.5, 1.0, 0.85, 0.01),
              ("load", "obciążenie [% S_N]", 5, 130, 100, 1),
              ("pfe", "straty w żelazie [% S_N]", 0.0, 2.0, 0.4, 0.05),
              ("pmech", "straty mech. [% S_N]", 0.0, 2.0, 0.3, 0.05),
              ("ra", "Ra [p.u.] (straty Cu tw.)", 0.0, 0.02, 0.004, 0.0005),
              ("pfn", "straty wzbudz. przy If=1 [% S_N]", 0.0, 1.0, 0.15, 0.01),
              ("xs", "Xs [p.u.]", 0.5, 2.5, 1.5, 0.05)]

    def losses(self, x, pf):
        sn = self.p("sn")
        pfe, pme, ra, pfn, xs = self.p("pfe") / 100, self.p("pmech") / 100, self.p("ra"), self.p("pfn") / 100, self.p("xs")
        phi = math.degrees(math.acos(pf))
        x = np.atleast_1d(x)
        ef = np.array([abs(gen_ef(1.0, xi, phi, ra, xs)[0]) for xi in x])
        l_cu = x ** 2 * ra
        l_add = 0.1 * l_cu
        l_f = pfn * ef ** 2
        pout = x * pf
        tot = pfe + pme + l_cu + l_add + l_f
        eta = pout / (pout + tot)
        return dict(pout=pout * sn, fe=pfe * sn * np.ones_like(x), me=pme * sn * np.ones_like(x),
                    cu=l_cu * sn, add=l_add * sn, fld=l_f * sn, tot=tot * sn, eta=eta)

    def update_plot(self):
        pf, load = self.p("pf"), self.p("load") / 100
        x = np.linspace(0.02, 1.3, 200)
        ax = self.fig.add_subplot(2, 1, 1)
        for pfk, col in [(1.0, "k"), (0.9, "tab:blue"), (0.8, "navy"), (pf, "r")]:
            ax.plot(x * 100, 100 * self.losses(x, pfk)["eta"], color=col, lw=2.2 if col == "r" else 1.2,
                    label=f"cos φ = {pfk:.2f}")
        L0 = self.losses(load, pf)
        ax.plot(load * 100, 100 * L0["eta"][0], "ro", ms=8)
        ax.set_xlabel("obciążenie [% S_N]"); ax.set_ylabel("η [%]")
        ax.set_ylim(max(80, 100 * float(self.losses(0.25, 0.8)["eta"][0]) - 2), 100)
        ax.grid(alpha=.3); ax.legend(fontsize=8, ncol=4); ax.set_title("Sprawność w funkcji obciążenia")

        a2 = self.fig.add_subplot(2, 2, 3)
        L = self.losses(x, pf)
        a2.stackplot(x * 100, L["fe"], L["me"], L["fld"], L["cu"], L["add"],
                     labels=["żelazo", "mechaniczne", "wzbudzenie", "Cu twornika", "dodatkowe"], alpha=0.8)
        a2.axvline(load * 100, color="r", ls="--")
        a2.set_xlabel("obciążenie [%]"); a2.set_ylabel("straty [MW]"); a2.grid(alpha=.3)
        a2.legend(fontsize=7, loc="upper left"); a2.set_title("Składniki strat")

        a3 = self.fig.add_subplot(2, 2, 4)
        names = ["Fe", "mech", "wzbudz.", "Cu tw.", "dodatk."]
        vals = [L0[k][0] for k in ("fe", "me", "fld", "cu", "add")]
        if sum(vals) > 0:
            a3.pie(vals, labels=names, autopct="%1.0f%%", textprops={"fontsize": 8})
        a3.set_title(f"Podział strat przy {load * 100:.0f} %")
        imax = int(np.argmax(L["eta"]))
        return [f"P_wyj = {L0['pout'][0]:.2f} MW",
                f"Σ strat = {L0['tot'][0]:.3f} MW",
                f"P_turbiny = {L0['pout'][0] + L0['tot'][0]:.2f} MW",
                f"η = {100 * L0['eta'][0]:.3f} %",
                "",
                f"  żelazo      {L0['fe'][0]:.3f} MW",
                f"  mechaniczne {L0['me'][0]:.3f} MW",
                f"  wzbudzenie  {L0['fld'][0]:.3f} MW",
                f"  Cu twornika {L0['cu'][0]:.3f} MW",
                f"  dodatkowe   {L0['add'][0]:.3f} MW",
                "",
                f"η_max = {100 * L['eta'][imax]:.3f} % przy {x[imax] * 100:.0f} % S_N"]


class TabS10(ExampleTab):
    key = "S10"
    PARAMS = [("fnl1", "G1: f biegu jałowego [Hz]", 49.0, 53.0, 51.5, 0.05),
              ("fnl2", "G2: f biegu jałowego [Hz]", 49.0, 53.0, 51.0, 0.05),
              ("s1", "G1: statyzm [%]", 1.0, 10.0, 5.0, 0.1),
              ("s2", "G2: statyzm [%]", 1.0, 10.0, 4.0, 0.1),
              ("pn1", "G1: moc znamionowa [MW]", 10, 500, 100, 5),
              ("pn2", "G2: moc znamionowa [MW]", 10, 500, 80, 5),
              ("pl", "obciążenie P [MW]", 0, 400, 120, 1),
              ("ql", "obciążenie Q [Mvar]", 0, 300, 60, 1),
              ("dv", "G1: wzbudzenie ΔV_bj [%]", -5, 5, 1.0, 0.1)]

    def update_plot(self):
        g = {k: self.p(k) for k in ("fnl1", "fnl2", "s1", "s2", "pn1", "pn2", "pl", "ql", "dv")}
        # statyzm s [%]: spadek f o s% * 50 Hz przy mocy znamionowej
        k1 = g["s1"] / 100 * 50.0 / g["pn1"]
        k2 = g["s2"] / 100 * 50.0 / g["pn2"]
        f, p1, p2 = droop_share(g["fnl1"], g["fnl2"], k1, k2, g["pl"])
        # moc bierna: V = V_bj - kq Q (statyzm napięciowy 4 %)
        kq1, kq2 = 0.04 / g["pn1"], 0.04 / g["pn2"]
        vnl1, vnl2 = 1.0 + 0.02 + g["dv"] / 100, 1.02
        vb, q1, q2 = droop_share(vnl1, vnl2, kq1, kq2, g["ql"])

        ax = self.fig.add_subplot(1, 2, 1)
        pa = np.linspace(0, g["pl"] * 1.4 + 10, 100)
        ax.plot(pa, g["fnl1"] - k1 * pa, "b", lw=2, label="G1")
        ax.plot(g["pl"] - pa, g["fnl2"] - k2 * pa, "r", lw=2, label="G2 (odwrócony)")
        ax.axhline(f, color="g", ls="--")
        ax.axvline(p1, color="0.5", ls=":")
        ax.plot(p1, f, "ko", ms=8)
        ax.annotate("", xy=(0, f - 0.3), xytext=(p1, f - 0.3), arrowprops=dict(arrowstyle="<->", color="b"))
        ax.text(p1 / 2, f - 0.25, f"P1={p1:.1f}", color="b", ha="center", fontsize=9)
        ax.annotate("", xy=(p1, f - 0.6), xytext=(g["pl"], f - 0.6), arrowprops=dict(arrowstyle="<->", color="r"))
        ax.text((p1 + g["pl"]) / 2, f - 0.55, f"P2={p2:.1f}", color="r", ha="center", fontsize=9)
        ax.set_xlim(min(0, p1) - 5, max(g["pl"], p1) + 5)
        ax.set_ylim(min(f - 1.0, 48.5), max(g["fnl1"], g["fnl2"]) + 0.3)
        ax.set_xlabel("P1 → [MW]   (← P2)"); ax.set_ylabel("f [Hz]"); ax.grid(alpha=.3); ax.legend(fontsize=8)
        ax.set_title(f"Podział mocy czynnej - 'wykres domkowy'  f = {f:.3f} Hz")

        a2 = self.fig.add_subplot(1, 2, 2)
        qa = np.linspace(0, g["ql"] * 1.4 + 10, 100)
        a2.plot(qa, vnl1 - kq1 * qa, "b", lw=2, label="G1")
        a2.plot(g["ql"] - qa, vnl2 - kq2 * qa, "r", lw=2, label="G2 (odwrócony)")
        a2.axhline(vb, color="g", ls="--")
        a2.plot(q1, vb, "ko", ms=8)
        a2.set_xlim(min(0, q1) - 5, max(g["ql"], q1) + 5)
        a2.set_xlabel("Q1 → [Mvar]   (← Q2)"); a2.set_ylabel("V [p.u.]"); a2.grid(alpha=.3); a2.legend(fontsize=8)
        a2.set_title(f"Podział mocy biernej  V = {vb:.4f} p.u.")
        warn = []
        if p1 < 0 or p2 < 0:
            warn.append("!!! jeden generator pracuje jako silnik (moc zwrotna) - zabezpieczenie go wyłączy")
        if p1 > g["pn1"] or p2 > g["pn2"]:
            warn.append("!!! przeciążenie generatora")
        return [f"Częstotliwość wspólna f = {f:.3f} Hz",
                f"G1: P1 = {p1:.1f} MW ({100 * p1 / g['pn1']:.0f} %)",
                f"G2: P2 = {p2:.1f} MW ({100 * p2 / g['pn2']:.0f} %)",
                "",
                f"Napięcie szyn V = {vb:.4f} p.u.",
                f"G1: Q1 = {q1:.1f} Mvar",
                f"G2: Q2 = {q2:.1f} Mvar",
                "",
                f"Aby przywrócić 50 Hz, podnieś",
                f"f_bj1 o {(50 - f):+.3f} Hz (G1 przejmie",
                f"całą korektę)",
                ""] + warn


class TabS11(ExampleTab):
    key = "S11"
    PARAMS = [("vg", "U generatora [p.u.]", 0.8, 1.2, 1.03, 0.01),
              ("df", "Δf = f_gen - f_sieci [Hz]", -2.0, 2.0, 0.2, 0.01),
              ("dth", "kąt fazowy Δθ [°]", -180, 180, 20, 1),
              ("xg", "X generatora (przejściowa) [p.u.]", 0.1, 1.0, 0.25, 0.01),
              ("xn", "X sieci [p.u.]", 0.01, 0.5, 0.1, 0.01),
              ("tw", "okno czasowe [s]", 0.1, 5.0, 1.0, 0.1)]

    def update_plot(self):
        vg, df, dth, xg, xn, tw = (self.p(k) for k in ("vg", "df", "dth", "xg", "xn", "tw"))
        f = 50.0
        t = np.linspace(0, tw, 6000)
        vb = np.sqrt(2) * np.sin(2 * PI * f * t)
        vgen = np.sqrt(2) * vg * np.sin(2 * PI * (f + df) * t + dth * DEG)
        ax = self.fig.add_subplot(2, 2, (1, 2))
        ax.plot(t, vb, "k", lw=0.7, label="sieć")
        ax.plot(t, vgen, "b", lw=0.7, alpha=0.7, label="generator")
        ax.plot(t, vgen - vb, "r", lw=0.9, label="różnica (na wyłączniku)")
        ang = dth * DEG + 2 * PI * df * t
        env = np.sqrt(2) * np.abs(vg * np.exp(1j * ang) - 1.0)
        ax.plot(t, env, "m--", lw=1.2, label="obwiednia ΔU")
        ax.set_xlabel("t [s]"); ax.set_ylabel("u [p.u.]"); ax.grid(alpha=.3); ax.legend(fontsize=8, ncol=4)
        ax.set_title("Przebiegi napięć przed synchronizacją")

        a2 = self.fig.add_subplot(2, 2, 3, projection="polar")
        a2.set_theta_zero_location("N"); a2.set_theta_direction(-1)
        a2.annotate("", xy=(dth * DEG, vg), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="b", lw=2.5))
        a2.annotate("", xy=(0, 1.0), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="k", lw=2))
        a2.set_rmax(1.3); a2.set_rticks([0.5, 1.0])
        a2.set_title("Synchroskop (czarny - sieć, niebieski - gen.)", fontsize=9)

        a3 = self.fig.add_subplot(2, 2, 4)
        th = np.linspace(-180, 180, 361)
        du = np.abs(vg * np.exp(1j * th * DEG) - 1.0)
        a3.plot(th, du / (xg + xn), "r", lw=2)
        du0 = abs(vg * cmath.exp(1j * dth * DEG) - 1.0)
        i0 = du0 / (xg + xn)
        a3.plot(dth, i0, "ro", ms=8)
        a3.axhline(1.0, color="0.5", ls=":")
        a3.set_xlabel("Δθ w chwili załączenia [°]"); a3.set_ylabel("prąd wyrównawczy [p.u.]")
        a3.grid(alpha=.3); a3.set_title("Udar prądu przy złym załączeniu")
        conds = [("ΔU ≤ 5 %", abs(vg - 1) <= 0.05), ("|Δf| ≤ 0,2 Hz", abs(df) <= 0.2),
                 ("|Δθ| ≤ 10°", abs(dth) <= 10)]
        ok = all(c for _, c in conds)
        lines = [f"ΔU na wyłączniku = {du0:.3f} p.u.",
                 f"Prąd wyrównawczy ≈ {i0:.2f} p.u.",
                 f"Okres dudnień = {1 / abs(df):.2f} s" if abs(df) > 1e-3 else "Brak dudnień (Δf = 0)",
                 f"Synchroskop: {abs(df) * 60:.1f} obr/min "
                 f"({'szybciej' if df > 0 else 'wolniej' if df < 0 else 'stoi'})",
                 ""]
        lines += [f"[{'OK' if c else '!!'}] {n}" for n, c in conds]
        lines += ["", "=> MOŻNA ZAMKNĄĆ WYŁĄCZNIK" if ok else "=> NIE ZAMYKAĆ!"]
        if ok and df < 0:
            lines.append("(uwaga: gen. wolniejszy - po załączeniu pobierze moc)")
        return lines


# =============================================================================
# APLIKACJA
# =============================================================================

TABS = [("S1 OCC/SCC, Xs", TabS1), ("S2 Wykres wskazowy", TabS2), ("S3 Zmienność napięcia", TabS3),
        ("S4 Moc P(δ)", TabS4), ("S5 Krzywe V silnika", TabS5), ("S6 Kompensacja cos φ", TabS6),
        ("S7 Bieguny jawne", TabS7), ("S8 Wykres P-Q", TabS8), ("S9 Sprawność", TabS9),
        ("S10 Praca równoległa", TabS10), ("S11 Synchronizacja", TabS11)]


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Maszyny synchroniczne - stan ustalony (przykłady interaktywne)")
        self.geometry("1400x900")
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
