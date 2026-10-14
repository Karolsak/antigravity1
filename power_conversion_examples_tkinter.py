"""
Przekształtniki, bilans mocy, rozdział ekonomiczny i elementy sieci
===================================================================
Interaktywna aplikacja Tkinter + Matplotlib z WSZYSTKIMI przykładami z rozdziałów
5-7 książki D. S. Kirschen, "Power Systems: Fundamental Concepts and the
Transition to Sustainability" (Wiley 2024):

  Rozdział 5 - Electronic Power Conversion (przekształtniki energoelektroniczne)
  Rozdział 6 - Balancing Load and Generation (statyzm, economic dispatch,
               unit commitment, rezerwa)
  Rozdział 7 - Network Components (linie, transformatory, jednostki względne,
               przełącznik zaczepów)

Każda zakładka = jeden przykład (albo jedno zagadnienie z teorii rozdziału):
suwaki (parametry), wykresy (wyniki, z zoomem), wyniki liczbowe oraz objaśnienie
napisane językiem zrozumiałym dla ucznia liceum - z pełnym rozwiązaniem liczbowym
z książki. Wartości domyślne suwaków = dane z książki, więc po otwarciu zakładki
widać dokładnie wynik przykładu.

Uruchomienie:  python power_conversion_examples_tkinter.py
Wymagania:     numpy, matplotlib, scipy (tkinter jest w standardowym Pythonie)
Obliczenia:    power_conversion_core.py (testy: test_power_conversion_core.py)
"""
import math
import cmath
import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.patches
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import power_conversion_core as core
from power_conversion_core import PI, SQ3, fmt_c

C_UNITS = ["tab:blue", "tab:orange", "tab:green", "tab:red"]


# =============================================================================
# OBJAŚNIENIA DLA UCZNIA LICEUM
# =============================================================================

EXPL = {
"5.1": """PRZYKŁAD 5.1 - Wpływ współczynnika mocy na moc przekształtnika

O CO CHODZI? Przekształtnik (falownik) łączy baterię albo panele PV z siecią. W środku
pracują tranzystory IGBT - to one "przełączają" prąd. Tranzystor ma ograniczenie PRĄDU
(za duży prąd = przegrzanie = koniec tranzystora). Przy stałym napięciu sieci V
ograniczenie prądu I oznacza ograniczenie MOCY POZORNEJ:  S = V * I.

Moc pozorna ma dwie "składowe", jak boki trójkąta prostokątnego:
   S² = P² + Q²        P - moc czynna (robi pracę: grzeje, kręci, ładuje) [kW]
                       Q - moc bierna (tylko "pływa" tam i z powrotem, potrzebna do
                           utrzymania napięcia i pola magnetycznego) [kvar]
Współczynnik mocy pf = cos φ = P / S.

Dopuszczalne punkty pracy leżą więc wewnątrz KOŁA o promieniu S (wykres po lewej).

ROZWIĄZANIE (dane z książki: 5 kVA, 220 V):
  * pf = 1:   P = 5 * 1 = 5 kW, Q = 0
  * pf = 0,9: P = 5 * 0,9 = 4,5 kW
              Q = sqrt(5² - 4,5²) = sqrt(25 - 20,25) = sqrt(4,75) = 2,179 kvar
  Prąd znamionowy: I = S/V = 5000/220 = 22,7 A - taki sam w obu przypadkach!

WNIOSEK INŻYNIERA: jeśli operator sieci każe falownikowi pomagać w regulacji napięcia
(oddawać/pobierać Q), to falownik "traci" część mocy czynnej. Dlatego falowniki PV
często projektuje się z zapasem: np. 10 kW paneli + falownik 11 kVA.
Przesuń suwak pf i zobacz, jak punkt wędruje po okręgu.""",

"5.3a": """ROZDZ. 5.3 - Mostek H i modulacja szerokości impulsów (PWM)

Mostek H to 4 łączniki (IGBT + dioda zwrotna). Włączamy je PARAMI:
  s1 = s2 = 1  ->  Vs = +Vdc        s3 = s4 = 1  ->  Vs = -Vdc
Nigdy s1 i s4 razem (zwarcie baterii!). Napięcie wyjściowe ma więc tylko DWIE wartości.

Jak z "prostokątów" zrobić sinusoidę? PWM: porównujemy sinusoidę (modulator, niebieska)
z szybką piłą trójkątną (nośna, szara):
   modulator > nośna  ->  Vs = +Vdc        modulator < nośna  ->  Vs = -Vdc
Gdy sinusoida jest wysoko, impulsy dodatnie są szerokie; gdy nisko - wąskie.
ŚREDNIA z impulsów (czerwona linia - 1. harmoniczna) to właśnie sinusoida!

Wzór (zakres liniowy, ma <= 1):  amplituda 1. harmonicznej = ma * Vdc
  ma = amplituda modulatora / amplituda nośnej (współczynnik modulacji amplitudy)
  mf = f_nośnej / f_sieci (współczynnik modulacji częstotliwości)

Widmo (wykres dolny): oprócz 50 Hz pojawiają się harmoniczne wokół mf, 2mf, ...
Im wyższe mf, tym dalej są te "śmieci" i tym łatwiej je odfiltrować dławikiem L.
Spróbuj: ma > 1 (przemodulowanie) - pojawiają się niskie harmoniczne 3, 5, 7...
Tryb UNIPOLARNY (dwie gałęzie z osobnymi odniesieniami) daje 3 poziomy: +Vdc, 0, -Vdc
i przesuwa harmoniczne do 2mf - mniejsze zniekształcenia.""",

"5.3b": """ROZDZ. 5.3.2 - Jak przekształtnik steruje mocą P i Q

Przekształtnik wytwarza napięcie E∠δ, sieć ma napięcie V∠0. Między nimi jest dławik
o reaktancji X = ωL. Tak samo jak generator synchroniczny (rozdz. 4.7):

   P = E V sin δ / X                (moc czynna od przekształtnika do sieci)
   Q = (E² - E V cos δ) / X         (moc bierna wysyłana przez przekształtnik)

STEROWANIE:
  * KĄT δ steruje mocą czynną P. δ > 0 (przekształtnik "wyprzedza" sieć) -> energia
    płynie z baterii do sieci (falownik). δ < 0 -> z sieci do baterii (prostownik).
  * AMPLITUDA E steruje mocą bierną Q. E > V -> przekształtnik oddaje Q (jak
    kondensator), E < V -> pobiera Q (jak dławik).
Pętla PLL "łapie" fazę sieci, a PWM ustawia dokładnie żądane δ i E.

Kondensator w obwodzie DC: P_dc - P_ac = d/dt(½ C V_dc²). Jeśli przekształtnik oddaje
do sieci więcej niż dostaje z baterii/PV - kondensator się rozładowuje i Vdc spada.
Regulator utrzymuje stałe Vdc, więc P_ac "śledzi" P_dc.

Na wykresie wskazowym: V (sieć), E (przekształtnik), jXI (spadek na dławiku) i prąd I.
Zmień δ na ujemne i zobacz, jak prąd obraca się o ~180° (zmiana kierunku mocy).""",

"5.4a": """ROZDZ. 5.4.1-5.4.2 - Bateria i fotowoltaika z MPPT

BATERIA: falownik pracuje w obu kierunkach: P_ref > 0 rozładowanie (falownik),
P_ref < 0 ładowanie (prostownik). Ograniczenia: |P| <= P_max (żywotność ogniw),
P² + Q² <= S² (prąd tranzystorów, patrz Przykład 5.1).

FOTOWOLTAIKA: panel ma charakterystykę prąd-napięcie (górny wykres, niebieska).
  * przy zwarciu (V=0) płynie największy prąd, ale moc P = V*I = 0
  * przy rozwarciu (I=0) napięcie największe, ale moc też 0
  * gdzieś pomiędzy jest MAKSIMUM MOCY (MPP) - czerwona kropka
Problem: MPP przesuwa się, gdy zmienia się nasłonecznienie i temperatura.

ALGORYTM "PERTURB AND OBSERVE" (zaburz i obserwuj) - jak wspinanie się po górze
z zawiązanymi oczami:
  1) zmierz moc P_old
  2) zmień napięcie o ΔV
  3) zmierz P_new
  4) jeśli P_new < P_old - zawróć (ΔV = -ΔV)
  5) powtarzaj
Punkt pracy "wspina się" na szczyt i potem drga wokół niego (dolny wykres).
W chwili kroku 60 nasłonecznienie spada (chmura) - algorytm sam znajduje nowy szczyt.
Większe ΔV = szybsze śledzenie, ale większe drgania wokół szczytu (straty).""",

"5.4b": """ROZDZ. 5.4.3-5.4.4 - Turbiny wiatrowe typu 4 i typu 3 (DFIG)

TYP 4 (pełny przekształtnik): turbina kręci generatorem synchronicznym. Częstotliwość
jego napięcia zależy od prędkości:   f = p * G * N / 120
  (p - liczba biegunów, G - przełożenie przekładni, N - obroty turbiny [obr/min])
Bo wiatr się zmienia, a turbina ma się kręcić z optymalną prędkością, f "pływa".
Rozwiązanie: prostownik (AC->DC) + falownik (DC->AC 50 Hz). Cała moc przechodzi przez
przekształtniki - drogo, ale pełna kontrola. Standard w dużych turbinach morskich.

TYP 3 (DFIG - generator indukcyjny dwustronnie zasilany): stojan połączony prosto
z siecią, a przez pierścienie ślizgowe i przekształtnik - wirnik.
   ω_m = (1 - s) ω_s         s - poślizg
   P_m = P_s + P_r  (bez strat),   P_s = P_m / (1 - s),  P_r = -s P_s
   częstotliwość prądów wirnika:  f_r = s f
  * s > 0 (wolniej niż synchronicznie) - moc WPŁYWA do wirnika (P_r < 0)
  * s < 0 (szybciej) - moc WYPŁYWA z wirnika (P_r > 0)
Przy poślizgu ±30% przekształtnik przenosi tylko ~30% mocy - dlatego typ 3 jest tańszy.

Dolny wykres: podział mocy stojan/wirnik w funkcji poślizgu.""",

"6.1": """PRZYKŁAD 6.1 - Statyzm (droop) jednego generatora

FIZYKA: wirnik generatora + turbina to ciężkie koło zamachowe. Ma energię kinetyczną
E = ½ J ω². Gdy obciążenie nagle rośnie (P_e > P_m), brakująca energia jest
"pożyczana" z wirującej masy - wirnik zwalnia, CZĘSTOTLIWOŚĆ SPADA:
   J ω dω/dt = P_m - P_e      (równanie 6.9)
Regulator (governor) widzi spadek obrotów i dodaje paliwa/pary - to REGULACJA PIERWOTNA.

Regulator ma charakterystykę ze STATYZMEM ρ (opadającą prostą, wykres lewy):
   f = f_nom (1 - ρ P / P_max)     (6.12)
ρ = 4% oznacza: od biegu jałowego do pełnej mocy częstotliwość spada o 4%.

ROZWIĄZANIE (50 kW, 60 Hz, ρ = 4%, obciążenie 25 kW):
   f = 60 * (1 - 0,04 * 25/50) = 60 * (1 - 0,02) = 60 * 0,98 = 58,8 Hz

WYKRES PRAWY - symulacja dynamiczna (RK4) po skokowym wzroście obciążenia:
   2H dΔω/dt = P_m - P_e - DΔω        H - stała bezwładności [s]
   T_g dP_m/dt = P_ref - Δω/ρ - P_m   T_g - stała czasowa turbiny
Widać: szybki spadek (bezwładność), dołek (nadir), ustalenie na nowej, NIŻSZEJ f.
Zaznacz "regulacja wtórna" - regulator całkujący przesuwa P_ref i przywraca 60 Hz
(rozdz. 6.6). Mniejsze H (mało wirujących mas - dużo PV/wiatru) -> głębszy dołek!""",

"6.2": """PRZYKŁAD 6.2 - Dwa generatory pracujące równolegle

Oba generatory widzą TĘ SAMĄ częstotliwość sieci. Każdy z nich, zgodnie ze swoją
prostą statyzmu, oddaje moc:
   P_i = P_i,max * (f_nom - f) / (ρ f_nom)          (6.17, 6.18)
Suma mocy = obciążenie:  P_1 + P_2 = P_T  ->  wyznaczamy f.

ROZWIĄZANIE (P_T = 20 kW, P_1max = 20 kW, P_2max = 30 kW, ρ = 0,04, 60 Hz):
   P_1 = P_T * P_1max / (P_1max + P_2max) = 20 * 20/50 = 8 kW
   P_2 = P_T * P_2max / (P_1max + P_2max) = 20 * 30/50 = 12 kW
   f = f_nom (1 - ρ P_1/P_1max) = 60 (1 - 0,04 * 8/20) = 60 * 0,984 = 59,04 Hz

WNIOSEK: przy jednakowym ρ generatory dzielą się obciążeniem PROPORCJONALNIE do mocy
znamionowej - bez żadnej komunikacji między nimi! Wystarczy, że każdy mierzy f.
Dlaczego nie ρ = 0? Dwa "sztywne" regulatory walczyłyby ze sobą (każdy mierzy
odrobinę inną f) - moc skakałaby w górę i w dół. Statyzm to stabilizuje.

Zmień ρ jednego z generatorów: ten z mniejszym ρ (bardziej "sztywny") bierze
większą część obciążenia. Wykres: trzy panele jak na Rys. 6.4 - wspólna pozioma
linia f łączy punkty pracy.""",

"6.3": """PRZYKŁADY 6.3 i 6.4 - Rozdział ekonomiczny (economic dispatch)

ZADANIE: trzy elektrownie mają pokryć obciążenie L. Jak podzielić moc, żeby całkowity
koszt paliwa był najmniejszy? Koszt godzinowy (odtworzony z danych książki):
   C1 = 2100 + 8,0 P1 + 0,10 P1²   [$/h]
   C2 = 7200 + 7,0 P2 + 0,06 P2²
   C3 = 6250 + 9,0 P3 + 0,07 P3²
KOSZT KRAŃCOWY = pochodna = ile kosztuje 1 MW więcej:  dC1/dP1 = 8 + 0,2 P1 itd.

ZASADA (Lagrange): w optimum WSZYSTKIE jednostki mają TEN SAM koszt krańcowy λ.
Intuicja: gdyby jednostka A miała koszt krańcowy 40, a B - 50 $/MWh, opłaca się
przesunąć 1 MW z B na A (oszczędność 10 $/h). Przesuwamy, aż koszty się wyrównają.

ROZWIĄZANIE: P_i = (λ - b_i)/(2c_i), suma = L:
   (λ-8)/0,2 + (λ-7)/0,12 + (λ-9)/0,14 = L  ->  20,476 λ - 162,62 = L
 L = 800 MW:  λ = 47,012 $/MWh; P1 = 195,06; P2 = 333,43; P3 = 271,51 MW; C = 37 523,69 $/h
 L = 1200 MW: λ = 66,547 $/MWh; P1 = 292,73; P2 = 496,22; P3 = 411,05 MW; C = 60 235,32 $/h

PRZYKŁAD 6.4 (OZE 200 MW): koszt OZE ~0, więc odejmujemy: L_net = 800 - 200 = 600 MW:
   λ = 37,244 $/MWh; P1 = 146,22; P2 = 252,03; P3 = 201,74 MW; C = 29 098,11 $/h
OZE obniża λ - to zjawisko "merit order effect" - ceny na giełdzie spadają w wietrzne dni.

Wykres: poziome przecięcie linii λ z prostymi kosztu krańcowego wyznacza moce P_i.""",

"6.5": """PRZYKŁADY 6.5 i 6.6 - Ograniczenia mocy jednostek

Prawdziwa elektrownia ma P_min (poniżej - niestabilny płomień w kotle) i P_max.
Tu: P_min = 200 MW dla wszystkich; P_max = 500, 450, 500 MW.

PROCEDURA: (1) licz bez ograniczeń, (2) jednostkę poza zakresem "przypnij" do granicy,
(3) resztę obciążenia rozdziel między pozostałe, (4) sprawdź ponownie.

PRZYKŁAD 6.5 (L = 1200 MW): bez ograniczeń P2 = 496,22 > 450 -> P2 = 450.
   Jednostki 1 i 3 dzielą L_net = 750 MW:  λ = 70,353 $/MWh,
   P1 = 311,76 MW; P3 = 438,24 MW; P2 = 450 MW; C = 60 451,47 $/h
   Koszty krańcowe: 70,35 / 61,0 / 70,35 - jednostka 2 jest tańsza "na marginesie",
   ale nie może dać więcej. Koszt wzrósł o 216 $/h względem 6.3 - cena ograniczenia.

PRZYKŁAD 6.6 (L_net = 600 MW): P1 = 146 < 200 -> P1 = 200. Ponowny rozdział 400 MW
między 2 i 3 daje P3 = 176,9 < 200 -> też trzeba przypiąć. Uwaga inżyniera: suma P_min
= 600 MW = L_net, więc JEDYNY dopuszczalny rozdział to 200/200/200 MW (C = 29 550 $/h).
Program pokazuje wszystkie iteracje w polu wyników.

WYKRES DOLNY (Rys. 6.11): koszt ŚREDNI (C/L) ma minimum - dla małych obciążeń
dominuje koszt stały (biegu jałowego) - opłacałoby się wyłączyć jedną jednostkę.
Koszt KRAŃCOWY rośnie monotonicznie.""",

"6.7": """PRZYKŁADY 6.7 i 6.8 - Krzywe kosztu odcinkowo-liniowe ("merit order")

W praktyce operatorzy zapisują koszty jako ŁAMANE (3 odcinki). Na każdym odcinku koszt
krańcowy jest STAŁY (schodki). Tabela z Przykładu 6.7:
            P_min  e1   e2   P_max |  c1     c2     c3 [$/MWh]
  U1:    200  300  400   500   | 58,0   78,0   98,0
  U2:    200  300  420   450   | 37,0   50,2   59,2
  U3:    200  300  420   500   | 44,0   59,4   73,4

ALGORYTM (jak kolejka od najtańszego):
  1) wszystkie jednostki na P_min (600 MW razem), reszta = L - 600
  2) posortuj WSZYSTKIE odcinki wg kosztu krańcowego:
     U2-1 37,0 | U3-1 44,0 | U2-2 50,2 | U1-1 58,0 | U2-3 59,2 | U3-2 59,4 | U3-3 73,4 | U1-2 78 | U1-3 98
  3) "nalewaj" moc po kolei, aż wyczerpiesz resztę obciążenia

L = 800 MW: reszta 200 = 100 (U2) + 100 (U3) -> P = 200/300/300; λ = 50,2 $/MWh
   C = 7700 + 14 700 + 15 250 = 37 650 $/h
L = 1200 MW: reszta 600 = 6 odcinków (570 MW) + 30 MW z U3-3 -> P = 300/450/450
   λ = 73,4 $/MWh (jednostka 3 jest "krańcowa"), C = 60 580 $/h

Jednostka krańcowa wyznacza cenę - tak działa giełda energii (Towarowa Giełda Energii).
Wykres lewy: łamana vs parabola. Prawy: "schody" podaży i pionowa linia L.""",

"6.9": """PRZYKŁADY 6.9 i 6.10 - Elastyczny odbiór i magazyn energii

Mamy 2 godziny: "dolina" 800 MW i "szczyt" 1200 MW. Koszt krańcowy w szczycie (73,4)
jest dużo większy niż w dolinie (50,2). Co jeśli przesunąć zużycie?

PRZYKŁAD 6.9 - odbiorca przesuwa 50 MW ze szczytu do doliny (np. pralnia przemysłowa,
ładowanie aut w nocy):
   C(850) = 40 160 $,  C(1150) = 57 190 $  ->  razem 97 350 $
   bez przesunięcia: C(800) + C(1200) = 37 650 + 60 580 = 98 230 $
   OSZCZĘDNOŚĆ = 880 $ (w ciągu 2 godzin, z jednego odbiorcy!)

PRZYKŁAD 6.10 - bateria: ładujemy 50 MWh w dolinie (obciążenie 850 MW), ale sprawność
"w obie strony" to 80%, więc w szczycie oddajemy tylko 0,8 * 50 = 40 MWh:
   C(850) + C(1160) = 40 160 + 57 784 = 97 944 $ < 98 230 $
   Zysk = 286 $ - magazyn opłaca się, ale mniej niż przesunięcie (straty 20%).
To jest ARBITRAŻ: kup tanio, sprzedaj drogo. Zysk rośnie z różnicą cen.

Suwaki: ilość przesuniętej energii, sprawność baterii, obciążenia w obu godzinach.
Sprawdź: przy jakiej sprawności magazyn przestaje się opłacać?""",

"6.11": """PRZYKŁAD 6.11 - Ile jednostek uruchomić?

Dla L = 475 MW sprawdzamy WSZYSTKIE kombinacje (2³ - 1 = 7):
   tylko U3:        475 MW          26 415 $   λ = 73,4
   tylko U2:        niewykonalne (P2max = 450 < 475)
   tylko U1:        475 MW          28 650 $   λ = 98,0
   U2 + U3:         275 + 200       24 625 $   λ = 37,0
   U1 + U2:         200 + 275       21 475 $   λ = 37,0   <- NAJTAŃSZE
   U1 + U3:         200 + 275       21 850 $   λ = 44,0
   wszystkie trzy:  niewykonalne (P_min razem = 600 > 475)

DLACZEGO? Jednostka 1 ma mały koszt stały (biegu jałowego), ale drogie dodatkowe MW.
Przy małym obciążeniu liczy się koszt stały, przy dużym - krańcowy.

WYKRES (Rys. 6.12): koszt całkowity każdej kombinacji w funkcji obciążenia.
Dolna obwiednia = najlepszy wybór dla danego L. Zwróć uwagę, że w zakresie
600-700 MW dwie jednostki są tańsze niż trzy! Pionowa linia = wybrane L.""",

"6.12": """PRZYKŁAD 6.12 - Eksplozja kombinatoryczna

Unit commitment = decyzja, które jednostki włączyć w każdej godzinie doby.
Każda jednostka: ON albo OFF -> 2^N stanów w jednej godzinie.
Przez T godzin: (2^N)^T = 2^(N*T) kombinacji.

ROZWIĄZANIE (N = 30, T = 24):  2^720 = 5,5 * 10^216 kombinacji.
Komputer sprawdzający miliard (10^9) kombinacji na sekundę potrzebowałby
   5,5*10^216 / 10^9 / 3,15*10^7 s/rok = 1,7 * 10^200 lat.
Wiek Wszechświata to "tylko" 1,4 * 10^10 lat!

JAK TO SIĘ ROBI NAPRAWDĘ? Programowanie mieszane całkowitoliczbowe (MILP) i algorytm
"branch-and-cut" (podziału i ograniczeń) - odcina całe gałęzie drzewa decyzji, które
na pewno nie dadzą lepszego wyniku. Operatorzy (np. PSE) rozwiązują tak problem dla
setek jednostek co dzień. Zakładki 6.13 i 6.14 używają takiego solvera (scipy.milp/HiGHS).

Wykres: log10 liczby kombinacji vs liczba jednostek (dla wybranego T) oraz czas.""",

"6.13": """PRZYKŁAD 6.13 - Unit commitment: Jaś (Hansel) i Małgosia (Gretel)

Dwie jednostki, 4 godziny, obciążenie 4000 / 800 / 5000 / 3000 MW.
   Hansel: P_min = 3000, P_max = 10 000 MW, 20 $/MWh   (duża, tania, "ciężka")
   Gretel: P_min =  500, P_max =  5000 MW, 70 $/MWh    (mała, droga, elastyczna)

KROK 1 - bez dodatkowych ograniczeń: Hansel wszystko, ale w godz. 2 obciążenie
(800) < P_min Hansela (3000), więc Hansel OFF, Gretel 800.          Koszt 296 000 $
KROK 2 - Gretel musi pracować min. 3 h (min uptime): skoro włączona w godz. 2, to musi
pracować też w 3 (lub 1-3). Wynik: H = 3500/0/4500/3000, G = 500/800/500/0. 346 000 $
KROK 3 - + Hansel po wyłączeniu stoi min. 2 h (min downtime): opłaca się włączyć
Hansela dopiero od godz. 3: H = 0/0/4500/3000, G = 4000/800/500/0.     521 000 $
KROK 4 - + koszt rozruchu Hansela 375 000 $: Gretel robi wszystko.    896 000 $
  (UWAGA INŻYNIERA: przy 375 000 $ oba warianty kosztują DOKŁADNIE 896 000 $ - remis!
   521 000 + 375 000 = 896 000 = 70 * 12 800. Przesuń suwak na 374 000 lub 376 000
   i zobacz, jak rozwiązanie "przeskakuje".)

KAŻDE DODATKOWE OGRANICZENIE PODNOSI KOSZT - realne elektrownie są "ociężałe".
(W książce tabele kroku 3 mają zamienione wiersze - poprawne liczby powyżej
 zgadzają się z kosztem 521 000 $.)
Solver: MILP (scipy.optimize.milp). Zmienne: P (moc), u (stan 0/1), v (rozruch), w (odstawienie).""",

"6.14": """PRZYKŁAD 6.14 - Unit commitment z REZERWĄ i ograniczeniem szybkości zmian mocy

Cztery jednostki (koszt krańcowy, koszt rozruchu, stan początkowy):
   Alpha 10-20 MW, 10 $/MWh, 100 $, ON      Beta  5-30 MW, 12 $/MWh, 100 $, ON
   Gamma 5-50 MW, 13 $/MWh, 100 $, OFF      Delta 10-20 MW, 20 $/MWh, 0 $, OFF
Obciążenie: 30 / 40 / 60 / 50 MW.

1) BEZ REZERWY: Alpha 20 (najtańsza, na maks.), Beta resztę, w godz. 3 Delta 10 MW
   (droga, ale bez kosztu rozruchu).  Koszt 2080 $.  Zapas (headroom) = 20/10/10/0 MW.
2) REZERWA 20 MW w każdej godzinie (= "co jeśli wypadnie największa jednostka?"):
   trzeba uruchomić Gammę (ma duży zapas do 50 MW). Książka podaje 2125 $.
3) + szybkość zmian mocy (Gamma 5 MW/h, pozostałe 20 MW/h): Gamma nie zdąży urosnąć,
   Delta znów potrzebna. Książka podaje 2225 $.

UWAGA INŻYNIERA (uczciwie): nasz solver MILP (HiGHS, rozwiązanie dokładne) znajduje
dla punktów 2 i 3 harmonogramy o nieco NIŻSZYM koszcie (2120 $ i 2210 $) przy
spełnieniu wszystkich ograniczeń. Tabela z książki (pypsa + glpk) to rozwiązanie
"prawie optymalne" - książka sama pisze, że algorytmy branch-and-cut zwykle podają
rozwiązanie bliskie optimum. W polu wyników podajemy koszt harmonogramu z książki
dla porównania. W wariancie 3 przyjęto (jak w pypsa), że rezerwa danej jednostki nie
może przekraczać jej szybkości zmian mocy (musi być osiągalna w ciągu godziny).

WNIOSEK: rezerwa i ograniczenia dynamiki = wyższy koszt. Magazyny energii mogą dać
rezerwę taniej (rozdz. 6.10).""",

"7.1": """PRZYKŁADY 7.1, 7.2 i 7.3 - Linia napowietrzna 345 kV, 250 km

MODEL: każdy kilometr linii ma rezystancję r (grzanie przewodów), indukcyjność
(pole magnetyczne wokół przewodu -> reaktancja x) i pojemność względem ziemi
(pole elektryczne -> susceptancja y). Dla linii < kilkuset km - schemat "π":
   R = r*ℓ,  X = x*ℓ,  B = y*ℓ  - połowa B na każdym końcu.

PRZYKŁAD 7.1 (r = 0,3 Ω/km, x = 0,9 Ω/km, y = 2 µS/km, S_b = 100 MVA):
   Z_b = 345² / 100 = 1190,25 Ω
   R = 75/1190,25 = 0,063 pu;  X = 225/1190,25 = 0,189 pu;  B/2 = 0,2976 pu
   (czyli impedancja gałęzi poprzecznej -j3,361 pu)

PRZYKŁAD 7.2 - linia NIEOBCIĄŻONA (otwarty koniec), V_S = 1,0 pu:
   dzielnik napięcia: V_R = V_S * (-j3,361) / (0,063 + j0,189 - j3,361) = 1,059∠-1,1° pu
   NAPIĘCIE NA KOŃCU JEST WYŻSZE NIŻ NA POCZĄTKU! - to EFEKT FERRANTIEGO.
   Pojemność linii "produkuje" moc bierną, która płynąc przez indukcyjność podnosi napięcie.
   Generator: P = 0,63 MW (straty), Q = -61,3 Mvar (musi POCHŁANIAĆ moc bierną!).
   Dlatego na długich liniach w nocy włącza się dławiki kompensacyjne.

PRZYKŁAD 7.3 - obciążenie 125 MW, pf = 0,85 ind., V_R = 1∠0°:
   P = 1,25 pu, Q = 1,25*tg(acos 0,85) = 0,775 pu -> R_L = 0,8 pu, X_L = 1,29 pu
   V_S = 1,187∠10,0° pu; generator: 136,3 MW (125 + 11,3 MW strat), 39,6 Mvar

Wykresy: V_R(ℓ) dla linii bez obciążenia; V_S(P) dla obciążenia; wykres wskazowy.""",

"7.4": """PRZYKŁAD 7.4 - Transformator idealny

Dwa uzwojenia na wspólnym rdzeniu. Zmienny strumień Φ indukuje w każdym zwoju to
samo napięcie, więc napięcia są proporcjonalne do liczby zwojów:
   E1/E2 = N1/N2 = a          I1/I2 = N2/N1 = 1/a        (moc się zachowuje)
Impedancja "widziana" z pierwotnej strony:  Z2' = a² Z2.

ROZWIĄZANIE (240/12 V, zasilanie 200 V, obciążenie 5 + j2 Ω):
   a = 240/12 = 20
   V2 = 200/20 = 10 V
   I2 = V2/Z2 = 10/(5 + j2) = 10/5,385∠21,8° = 1,857∠-21,8° A
   I1 = I2/a = 0,0929∠-21,8° A  (92,9 mA)
   lub: Z2' = 400*(5 + j2) = 2000 + j800 Ω -> I1 = 200/(2000 + j800) - ten sam wynik
   S = V2 I2* = 10 * 1,857∠21,8° = 17,24 W + j6,90 var

Prawdziwy transformator dodaje: R1, R2 (straty w miedzi), X1, X2 (strumień
rozproszenia), Rm (straty w żelazie), Xm (prąd magnesujący). W jednostkach względnych
idealny transformator "znika" (rys. 7.9-7.10)!

Wykres: przebiegi czasowe v1, v2, i1*a, i2 i wykres wskazowy. Prąd opóźnia się za
napięciem o kąt φ = arctg(X/R).""",

"7.5": """PRZYKŁAD 7.5 - Prądy znamionowe transformatora trójfazowego

W układzie trójfazowym moc pozorna:  S = √3 * V_LL * I_L
   (V_LL - napięcie międzyprzewodowe, I_L - prąd przewodowy)
MVA transformatora mówi, ile prądu uzwojenia wytrzymają bez przegrzania izolacji.

ROZWIĄZANIE (Y-Y, 13,8 kV / 416 V, 12 kVA):
   I_1max = 12 000 / (√3 * 13 800) = 0,502 A
   I_2max = 12 000 / (√3 * 416)   = 16,65 A
Stosunek prądów = 16,65/0,502 = 33,2 = 13 800/416 - odwrotnie do napięć.

PRAKTYKA: dlatego przesyłamy energię wysokim napięciem - ta sama moc, 33 razy mniejszy
prąd, a straty I²R aż 1100 razy mniejsze! Grube kable po stronie nn, cienkie po sn.""",

"7.6": """PRZYKŁAD 7.6 - Mały system w jednostkach względnych (per unit)

Układ: generator 75 MVA 11 kV (X_S = 1,83) -> T1 11/132 kV 75 MVA (X = 0,125) ->
linia 20 km (0,18 + j0,40 Ω/km) -> T2 132/33 kV 45 MVA (X = 0,125) -> obciążenie
10 MVA, pf 0,8 ind., napięcie znamionowe.

KROK 1 - wspólna baza S_b = 100 MVA; napięcia bazowe 11 / 132 / 33 kV (strefy).
KROK 2 - przeliczenie:  Z_nowa = Z_stara * (S_b,nowa / S_b,stara)
   X_G  = 1,83 * 100/75  = 2,44 pu
   X_T1 = 0,125 * 100/75 = 0,1667 pu
   Z_b,linii = 132²/100 = 174,24 Ω;  Z_linii = 3,6 + j8,0 Ω = 0,0207 + j0,0459 pu
   X_T2 = 0,125 * 100/45 = 0,2778 pu
   S_obc = 10/100 = 0,1 pu
KROK 3 - prąd (napięcie obciążenia 1∠0°): I = 0,1∠-36,87° pu
   V_gen = 1 + (j0,1667 + 0,0207 + j0,0459 + j0,2778) * I = 1,032∠2,11° pu
   E = V_gen + j2,44 * I = 1,200∠11,2° pu
KROK 4 - prądy w amperach (różne bazy prądu w każdej strefie):
   I_b = S_b/(√3 V_b): 5248,7 A (11 kV), 437,4 A (132 kV), 1749,5 A (33 kV)
   I_gen = 524,9 A,  I_linii = 43,7 A,  I_obc = 175,0 A

MAGIA PER UNIT: transformatory "znikają", cały system to jeden prosty obwód szeregowy!""",

"7.7": """PRZYKŁAD 7.7 - Transformator z przełącznikiem zaczepów (OLTC)

Przełącznik zaczepów zmienia liczbę zwojów -> przekładnia 1:t. t > 1 podbija napięcie
po stronie obciążenia, t < 1 - obniża. Model π (rys. 7.18):
   gałąź podłużna:  tY       gałąź po stronie źródła: t(t-1)Y     po stronie odbioru: (1-t)Y
   Y = 1/(jX_T)
Dla t > 1: po stronie źródła "dławik", po stronie odbioru "kondensator" - zmiana
zaczepu działa jak przeniesienie mocy biernej z jednej strony na drugą.

DANE: V_S = 1,0 pu, X_T = 0,1 pu, obciążenie R_L = 1,0 pu || X_L = 5,0 pu.

t = 1,00:  V_L = 0,976∠-5,6° pu;  I_S = 0,995∠-16,9° pu;  S = 0,952 + j0,289 pu
t = 1,05:  Z10 = jX/(t(t-1)) = j1,905 pu;  Z20 = jX/(1-t) = -j2,0 pu
           V_L = 1,024∠-5,6° pu;  I_S = 1,097∠-16,9° pu;  S = 1,050 + j0,319 pu

WNIOSEK: wyższe zaczepy podnoszą napięcie odbioru (+5%), a obciążenie (stała
impedancja) pobiera więcej mocy: P ~ V². Tak działa automatyczna regulacja napięcia
w stacjach WN/SN - co kilkadziesiąt sekund regulator sprawdza napięcie i w razie
potrzeby przestawia zaczep o ~1,25-1,5%.

Program liczy dokładny model z transformatorem idealnym ORAZ model π - wyniki
muszą być identyczne (sprawdzenie w polu wyników).""",
}


# =============================================================================
# GUI - klasa bazowa zakładki
# =============================================================================

class ExampleTab(ttk.Frame):
    """Zakładka: lewy panel suwaków + wyniki liczbowe, prawy - wykresy,
    dół - objaśnienie. Podklasy definiują PARAMS, CHECKS, key, update_plot()."""
    PARAMS = []   # (nazwa, etykieta, min, max, domyślna, krok)
    CHECKS = []   # (nazwa, etykieta, domyślna)
    key = ""

    def __init__(self, master):
        super().__init__(master)
        self.vars = {}
        self.bvars = {}
        self._pending = None
        pw = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        pw.pack(fill=tk.BOTH, expand=True)
        left_outer = ttk.Frame(pw, width=330)
        right = ttk.Frame(pw)
        pw.add(left_outer, weight=0)
        pw.add(right, weight=1)

        # przewijany lewy panel (dużo suwaków w niektórych zakładkach)
        cv = tk.Canvas(left_outer, width=320, highlightthickness=0)
        sb = ttk.Scrollbar(left_outer, orient=tk.VERTICAL, command=cv.yview)
        left = ttk.Frame(cv)
        left.bind("<Configure>", lambda _e: cv.configure(scrollregion=cv.bbox("all")))
        cv.create_window((0, 0), window=left, anchor="nw")
        cv.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        cv.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        ttk.Label(left, text="Parametry", font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=6, pady=(6, 2))
        for name, label, lo, hi, val, res in self.PARAMS:
            fr = ttk.Frame(left)
            fr.pack(fill=tk.X, padx=6, pady=1)
            var = tk.DoubleVar(value=val)
            self.vars[name] = var
            ttk.Label(fr, text=label).pack(anchor="w")
            sc = tk.Scale(fr, from_=lo, to=hi, resolution=res, orient=tk.HORIZONTAL,
                          variable=var, showvalue=True, length=280,
                          command=lambda _e: self.schedule())
            sc.pack(fill=tk.X)
        for name, label, val in self.CHECKS:
            var = tk.BooleanVar(value=val)
            self.bvars[name] = var
            ttk.Checkbutton(left, text=label, variable=var,
                            command=self.schedule).pack(anchor="w", padx=6, pady=1)
        self.extra_controls(left)
        ttk.Button(left, text="Przywróć dane z książki", command=self.reset).pack(fill=tk.X, padx=6, pady=6)
        ttk.Label(left, text="Wyniki", font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=6)
        self.result = tk.Text(left, height=22, width=44, font=("Consolas", 9), bg="#f4f6fa")
        self.result.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))

        vpw = ttk.PanedWindow(right, orient=tk.VERTICAL)
        vpw.pack(fill=tk.BOTH, expand=True)
        figfr = ttk.Frame(vpw)
        self.fig = Figure(figsize=(9, 6), dpi=90, tight_layout=True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=figfr)
        NavigationToolbar2Tk(self.canvas, figfr).update()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        txtfr = ttk.Frame(vpw)
        self.expl = ScrolledText(txtfr, height=12, wrap=tk.WORD, font=("Segoe UI", 10), bg="#fffdf3")
        self.expl.pack(fill=tk.BOTH, expand=True)
        self.expl.insert("1.0", EXPL.get(self.key, ""))
        self.expl.configure(state=tk.DISABLED)
        vpw.add(figfr, weight=3)
        vpw.add(txtfr, weight=2)
        self.refresh()

    def extra_controls(self, parent):
        pass

    def reset(self):
        for name, _l, _lo, _hi, val, _r in self.PARAMS:
            self.vars[name].set(val)
        for name, _l, val in self.CHECKS:
            self.bvars[name].set(val)
        self.refresh()

    def p(self, name):
        return self.vars[name].get()

    def b(self, name):
        return self.bvars[name].get()

    def schedule(self):
        if self._pending:
            self.after_cancel(self._pending)
        self._pending = self.after(120, self.refresh)

    def refresh(self):
        self._pending = None
        self.fig.clear()
        try:
            lines = self.update_plot() or []
        except Exception as exc:  # błąd obliczeń nie może zamknąć aplikacji
            lines = [f"Błąd obliczeń: {exc}"]
        self.result.delete("1.0", tk.END)
        self.result.insert("1.0", "\n".join(lines))
        self.canvas.draw_idle()

    def update_plot(self):
        raise NotImplementedError


def phasor(ax, z, color, label, origin=0j, lw=2.0):
    """Rysuje wskaz (strzałkę) od origin do origin+z."""
    ax.annotate("", xy=(origin.real + z.real, origin.imag + z.imag),
                xytext=(origin.real, origin.imag),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw))
    ax.plot([], [], color=color, lw=lw, label=label)


def finish_phasor_axes(ax, pts, title):
    pts = list(pts) + [0j]
    xs = [p.real for p in pts]
    ys = [p.imag for p in pts]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1e-6) * 0.6 + 1e-9
    cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
    ax.set_xlim(cx - span * 1.1, cx + span * 1.1)
    ax.set_ylim(cy - span * 1.1, cy + span * 1.1)
    ax.set_aspect("equal", adjustable="box")
    ax.axhline(0, color="0.7", lw=0.6)
    ax.axvline(0, color="0.7", lw=0.6)
    ax.grid(alpha=.3)
    ax.legend(loc="best", fontsize=8)
    ax.set_title(title)


# =============================================================================
# ROZDZIAŁ 5
# =============================================================================

class Tab51(ExampleTab):
    key = "5.1"
    PARAMS = [("S", "S znamionowa przekształtnika [kVA]", 1, 20, 5, 0.5),
              ("pf", "współczynnik mocy pf = cos φ", 0.0, 1.0, 0.9, 0.01),
              ("V", "napięcie sieci V [V]", 110, 400, 220, 10)]
    CHECKS = [("cap", "Q pojemnościowe (oddawane)", True)]

    def update_plot(self):
        S, pf, V = self.p("S"), self.p("pf"), self.p("V")
        P, Q = core.converter_capability(S, pf)
        if not self.b("cap"):
            Q = -Q
        I = S * 1e3 / V
        a1 = self.fig.add_subplot(1, 2, 1)
        th = np.linspace(0, 2 * PI, 400)
        a1.fill(S * np.cos(th), S * np.sin(th), color="tab:green", alpha=0.12, label="obszar dopuszczalny")
        a1.plot(S * np.cos(th), S * np.sin(th), "g", lw=1.5)
        a1.plot([0, P], [0, Q], "k-", lw=1.5)
        a1.plot([0, P], [0, 0], "tab:blue", lw=3, label=f"P = {P:.2f} kW")
        a1.plot([P, P], [0, Q], "tab:red", lw=3, label=f"Q = {Q:.3f} kvar")
        a1.plot(P, Q, "ko", ms=8)
        a1.annotate(f"S = {S:.1f} kVA\nφ = {math.degrees(math.acos(pf)):.1f}°", (P / 2, Q / 2),
                    xytext=(-60, 20 if Q >= 0 else -30), textcoords="offset points")
        a1.text(-S * 0.95, S * 0.85, "P<0: ładowanie\nbaterii", fontsize=8, color="0.4")
        a1.text(S * 0.45, S * 0.85, "P>0: oddawanie\ndo sieci", fontsize=8, color="0.4")
        a1.set_aspect("equal")
        a1.set_xlabel("P [kW]"); a1.set_ylabel("Q [kvar]")
        a1.grid(alpha=.3); a1.legend(loc="lower left", fontsize=8)
        a1.set_title("Koło możliwości P-Q (P² + Q² ≤ S²)")
        a2 = self.fig.add_subplot(1, 2, 2)
        pfs = np.linspace(0, 1, 201)
        a2.plot(pfs, S * pfs, "tab:blue", lw=2, label="P_max(pf)")
        a2.plot(pfs, S * np.sqrt(1 - pfs ** 2), "tab:red", lw=2, label="Q przy P_max")
        a2.axvline(pf, color="k", ls="--", lw=1)
        a2.plot([pf, pf], [S * pf, abs(Q)], "ko")
        a2.set_xlabel("współczynnik mocy pf"); a2.set_ylabel("kW / kvar")
        a2.grid(alpha=.3); a2.legend(fontsize=8)
        a2.set_title("Ile mocy czynnej zostaje przy danym pf?")
        return [f"S        = {S:.2f} kVA",
                f"pf       = {pf:.3f}  (φ = {math.degrees(math.acos(pf)):.2f}°)",
                f"P_max    = S·pf = {P:.3f} kW",
                f"Q        = √(S²-P²) = {abs(Q):.3f} kvar",
                f"I_znam   = S/V = {I:.2f} A",
                "",
                f"Utrata mocy czynnej: {S - P:.3f} kW ({100 * (1 - pf):.1f} %)",
                "",
                "Książka: pf=1 -> 5 kW; pf=0,9 -> 4,5 kW",
                "         i 2,179 kvar"]


class Tab53PWM(ExampleTab):
    key = "5.3a"
    PARAMS = [("ma", "ma (modulacja amplitudy)", 0.0, 1.6, 0.8, 0.01),
              ("mf", "mf = f_nośnej / f_sieci", 3, 41, 15, 1),
              ("vdc", "Vdc [V]", 50, 800, 400, 10),
              ("f1", "f sieci [Hz]", 40, 70, 50, 1)]
    CHECKS = [("uni", "PWM unipolarny (3 poziomy)", False)]

    def update_plot(self):
        ma, mf, vdc, f1 = self.p("ma"), int(self.p("mf")), self.p("vdc"), self.p("f1")
        t, mod, car, vs = core.hbridge_pwm(ma, mf, vdc, f1, n=12000, bipolar=not self.b("uni"))
        amp = core.spectrum(vs)
        # okno = 2 okresy -> harmoniczna h jest w prążku 2h
        h_amp = amp[::2]
        V1 = h_amp[1]
        ph = np.angle(np.fft.rfft(vs)[2])
        a1 = self.fig.add_subplot(3, 1, 1)
        a1.plot(t * 1e3, car, color="0.6", lw=0.8, label="nośna")
        a1.plot(t * 1e3, mod, "b", lw=1.6, label="modulator")
        if self.b("uni"):
            a1.plot(t * 1e3, -mod, "c--", lw=1.0, label="-modulator")
        a1.set_ylabel("p.u."); a1.grid(alpha=.3); a1.legend(loc="upper right", fontsize=8)
        a1.set_title("Porównanie modulatora z nośną (Rys. 5.4a)")
        a2 = self.fig.add_subplot(3, 1, 2, sharex=a1)
        a2.plot(t * 1e3, vs, "k", lw=0.7, label="Vs(t) - impulsy")
        a2.plot(t * 1e3, V1 * np.cos(2 * PI * f1 * t + ph), "r", lw=1.8, label="1. harmoniczna")
        a2.set_ylabel("Vs [V]"); a2.set_xlabel("t [ms]"); a2.grid(alpha=.3)
        a2.legend(loc="upper right", fontsize=8)
        a3 = self.fig.add_subplot(3, 1, 3)
        hmax = min(len(h_amp), 3 * mf + 8)
        a3.bar(np.arange(hmax), h_amp[:hmax] / vdc, color="tab:purple", width=0.7)
        a3.set_xlabel("rząd harmonicznej h (f = h·f_sieci)"); a3.set_ylabel("V_h / Vdc")
        a3.grid(alpha=.3); a3.set_title("Widmo napięcia wyjściowego")
        thd = math.sqrt(np.sum(h_amp[2:200] ** 2)) / max(V1, 1e-9)
        return [f"f nośnej       = {mf * f1:.0f} Hz",
                f"V1 (FFT)       = {V1:.1f} V (amplituda)",
                f"V1 teoria ma·Vdc = {ma * vdc:.1f} V (ma ≤ 1)",
                f"V1 RMS         = {V1 / math.sqrt(2):.1f} V",
                f"THD napięcia   = {100 * thd:.1f} %",
                "",
                "PRZEMODULOWANIE (ma > 1)!" if ma > 1 else "zakres liniowy (ma ≤ 1)",
                "",
                "Harmoniczne skupione wokół:",
                f"  {'2' if self.b('uni') else ''}mf = {(2 if self.b('uni') else 1) * mf}"]


class Tab53PQ(ExampleTab):
    key = "5.3b"
    PARAMS = [("E", "E - napięcie przekształtnika [pu]", 0.8, 1.2, 1.05, 0.005),
              ("V", "V - napięcie sieci [pu]", 0.9, 1.1, 1.0, 0.005),
              ("X", "X = ωL dławika [pu]", 0.05, 0.5, 0.15, 0.005),
              ("d", "kąt δ [°]", -30, 30, 8, 0.5)]

    def update_plot(self):
        E, V, X, d = self.p("E"), self.p("V"), self.p("X"), self.p("d")
        P, Qs, Qr = core.converter_power_flow(E, V, X, d)
        Ev = cmath.rect(E, math.radians(d))
        Vv = complex(V, 0)
        I = (Ev - Vv) / (1j * X)
        a1 = self.fig.add_subplot(1, 2, 1)
        dd = np.linspace(-90, 90, 361)
        Pp = E * V * np.sin(np.radians(dd)) / X
        Qq = (E ** 2 - E * V * np.cos(np.radians(dd))) / X
        a1.plot(dd, Pp, "tab:blue", lw=2, label="P(δ) = EV sinδ / X")
        a1.plot(dd, Qq, "tab:red", lw=2, label="Q(δ) wysyłane")
        a1.axvline(d, color="k", ls="--", lw=1)
        a1.plot(d, P, "bo"); a1.plot(d, Qs, "ro")
        a1.axhline(0, color="0.5", lw=0.6)
        a1.set_xlabel("δ [°]"); a1.set_ylabel("P, Q [pu]")
        a1.grid(alpha=.3); a1.legend(fontsize=8)
        a1.set_title("Moc w funkcji kąta δ")
        a2 = self.fig.add_subplot(1, 2, 2)
        phasor(a2, Vv, "tab:green", "V (sieć)")
        phasor(a2, Ev, "tab:blue", "E (przekształtnik)")
        phasor(a2, Ev - Vv, "tab:orange", "jX·I", origin=Vv, lw=1.5)
        Isc = I / max(abs(I), 1e-9) * 0.6
        phasor(a2, Isc, "tab:red", "I (kierunek)", lw=1.5)
        finish_phasor_axes(a2, [Vv, Ev, Isc], "Wykres wskazowy")
        return [f"P = E·V·sinδ/X     = {P:+.3f} pu",
                f"Q wysyłane         = {Qs:+.3f} pu",
                f"Q odbierane (sieć) = {Qr:+.3f} pu",
                f"|I| = {abs(I):.3f} pu, ∠I = {math.degrees(cmath.phase(I)):+.1f}°",
                f"straty Q w dławiku = X·I² = {X * abs(I) ** 2:.4f} pu",
                "",
                ("P > 0: bateria -> sieć (falownik)" if P > 0 else
                 "P < 0: sieć -> bateria (prostownik)" if P < 0 else "P = 0"),
                ("Q > 0: przekształtnik jak kondensator" if Qs > 0 else
                 "Q < 0: przekształtnik jak dławik"),
                f"P_max (δ = 90°)    = {E * V / X:.2f} pu"]


class Tab54PV(ExampleTab):
    key = "5.4a"
    PARAMS = [("G1", "nasłonecznienie przed chmurą [W/m²]", 200, 1200, 1000, 10),
              ("G2", "nasłonecznienie po kroku 60 [W/m²]", 100, 1200, 400, 10),
              ("T", "temperatura ogniw [°C]", -10, 70, 25, 1),
              ("dV", "krok ΔV algorytmu [V]", 0.1, 3.0, 0.5, 0.1),
              ("V0", "napięcie startowe [V]", 5, 38, 20, 0.5)]

    def update_plot(self):
        G1, G2, T, dV, V0 = self.p("G1"), self.p("G2"), self.p("T"), self.p("dV"), self.p("V0")
        prof = np.r_[np.full(60, G1), np.full(80, G2)]
        Vs, Ps = core.mppt_perturb_observe(prof, dV, V0, temp=T)
        v = np.linspace(0, 45, 600)
        a1 = self.fig.add_subplot(2, 2, 1)
        a2 = self.fig.add_subplot(2, 2, 2)
        res = []
        for G, col in ((G1, "tab:orange"), (G2, "tab:blue")):
            i = core.pv_curve(v, G, T)
            p = v * i
            k = p.argmax()
            a1.plot(v, i, color=col, lw=1.8, label=f"I(V), {G:.0f} W/m²")
            a2.plot(v, p, color=col, lw=1.8, label=f"P(V), {G:.0f} W/m²")
            a2.plot(v[k], p[k], "o", color="r")
            res.append((G, v[k], p[k]))
        a2.plot(Vs[:60], Ps[:60], ".", color="k", ms=4, label="trajektoria MPPT")
        a2.plot(Vs[60:], Ps[60:], ".", color="0.5", ms=4)
        a1.set_xlabel("V [V]"); a1.set_ylabel("I [A]"); a1.grid(alpha=.3); a1.legend(fontsize=8)
        a1.set_title("Charakterystyka prądowo-napięciowa modułu")
        a2.set_xlabel("V [V]"); a2.set_ylabel("P [W]"); a2.grid(alpha=.3); a2.legend(fontsize=8)
        a2.set_title("Moc i punkt mocy maksymalnej (MPP)")
        a3 = self.fig.add_subplot(2, 1, 2)
        k = np.arange(len(Vs))
        pmpp = np.where(k < 60, res[0][2], res[1][2])
        a3.plot(k, Ps, "k", lw=1.5, label="P z MPPT")
        a3.plot(k, pmpp, "r--", lw=1.2, label="P_max możliwa")
        a3b = a3.twinx()
        a3b.plot(k, Vs, "tab:green", lw=1, alpha=0.7, label="V_dc")
        a3b.set_ylabel("V_dc [V]", color="tab:green")
        a3.axvline(60, color="0.5", ls=":")
        a3.set_xlabel("krok algorytmu"); a3.set_ylabel("P [W]"); a3.grid(alpha=.3)
        a3.legend(loc="lower left", fontsize=8)
        a3.set_title("Perturb & Observe w czasie (chmura w kroku 60)")
        eff = Ps.sum() / pmpp.sum() * 100
        return [f"MPP przy {res[0][0]:.0f} W/m²: V* = {res[0][1]:.1f} V, P = {res[0][2]:.1f} W",
                f"MPP przy {res[1][0]:.0f} W/m²: V* = {res[1][1]:.1f} V, P = {res[1][2]:.1f} W",
                "",
                f"Sprawność śledzenia MPPT = {eff:.2f} %",
                "(energia zebrana / możliwa)",
                "",
                "Mały ΔV: wolny start, mało drgań",
                "Duży ΔV: szybko, ale drga wokół MPP"]


class Tab54Wind(ExampleTab):
    key = "5.4b"
    PARAMS = [("N", "obroty turbiny N [obr/min]", 5, 25, 15, 0.5),
              ("p", "liczba biegunów p (typ 4)", 2, 120, 4, 2),
              ("G", "przełożenie przekładni G", 1, 120, 100, 1),
              ("Pm", "moc mechaniczna P_m [MW]", 0.5, 6, 3, 0.1),
              ("s", "poślizg DFIG s [-]", -0.3, 0.3, -0.2, 0.01)]

    def update_plot(self):
        N, p, G, Pm, s = self.p("N"), int(self.p("p")), self.p("G"), self.p("Pm"), self.p("s")
        f = core.type4_frequency(p, G, N)
        Ps, Pr = core.dfig_split(Pm, s)
        a1 = self.fig.add_subplot(1, 2, 1)
        Nn = np.linspace(5, 25, 100)
        a1.plot(Nn, core.type4_frequency(p, G, Nn), "tab:blue", lw=2, label="f generatora (typ 4)")
        a1.axhline(50, color="r", ls="--", label="sieć 50 Hz")
        a1.plot(N, f, "ko")
        a1.set_xlabel("N turbiny [obr/min]"); a1.set_ylabel("f [Hz]")
        a1.grid(alpha=.3); a1.legend(fontsize=8)
        a1.set_title("Typ 4: f = p·G·N/120 - zmienna!")
        a2 = self.fig.add_subplot(1, 2, 2)
        ss = np.linspace(-0.3, 0.3, 121)
        ps = Pm / (1 - ss)
        a2.plot(ss, ps, "tab:green", lw=2, label="P_s (stojan)")
        a2.plot(ss, Pm - ps, "tab:orange", lw=2, label="P_r (wirnik, >0 = oddaje)")
        a2.axhline(Pm, color="k", ls=":", label="P_m = P_s + P_r")
        a2.axvline(s, color="k", ls="--", lw=1)
        a2.plot([s, s], [Ps, Pr], "ko")
        a2.set_xlabel("poślizg s"); a2.set_ylabel("moc [MW]")
        a2.grid(alpha=.3); a2.legend(fontsize=8)
        a2.set_title("Typ 3 (DFIG): podział mocy")
        return ["TYP 4:",
                f"  f gen = {p}·{G:.0f}·{N:.1f}/120 = {f:.1f} Hz",
                f"  przekształtniki: 100 % mocy = {Pm:.2f} MW",
                "",
                "TYP 3 (DFIG):",
                f"  ω_m/ω_s = 1 - s = {1 - s:.2f}",
                f"  P_s = P_m/(1-s) = {Ps:.3f} MW",
                f"  P_r = -s·P_s   = {Pr:+.3f} MW",
                f"  f wirnika = |s|·50 = {abs(s) * 50:.1f} Hz",
                f"  przekształtnik: {abs(Pr) / Pm * 100:.1f} % mocy",
                "  " + ("nadsynchronicznie: wirnik ODDAJE moc" if s < 0 else
                        "podsynchronicznie: wirnik POBIERA moc" if s > 0 else "synchronicznie")]


# =============================================================================
# ROZDZIAŁ 6
# =============================================================================

class Tab61(ExampleTab):
    key = "6.1"
    PARAMS = [("Pmax", "P_max generatora [kW]", 10, 100, 50, 1),
              ("rho", "statyzm ρ [%]", 1, 10, 4, 0.5),
              ("P", "obciążenie P [kW]", 0, 100, 25, 0.5),
              ("fn", "f_nom [Hz]", 50, 60, 60, 10),
              ("dP", "skok obciążenia ΔP [kW]", -20, 30, 10, 0.5),
              ("H", "stała bezwładności H [s]", 0.5, 10, 4, 0.1),
              ("Tg", "stała czasowa turbiny T_g [s]", 0.1, 5, 0.8, 0.1)]
    CHECKS = [("sec", "regulacja wtórna (przywraca f_nom)", False)]

    def update_plot(self):
        Pmax, rho, P, fn = self.p("Pmax"), self.p("rho") / 100, self.p("P"), self.p("fn")
        dP, H, Tg = self.p("dP"), self.p("H"), self.p("Tg")
        P = min(P, Pmax)
        f = core.droop_frequency(P, Pmax, rho, fn)
        f2 = core.droop_frequency(min(P + dP, Pmax), Pmax, rho, fn)
        a1 = self.fig.add_subplot(1, 2, 1)
        pp = np.linspace(0, Pmax, 50)
        a1.plot(pp, core.droop_frequency(pp, Pmax, rho, fn), "tab:blue", lw=2, label="charakterystyka statyzmu")
        a1.plot(P, f, "ro", ms=8, label=f"punkt pracy {f:.3f} Hz")
        a1.plot(min(P + dP, Pmax), f2, "s", color="tab:orange", ms=7, label=f"po skoku {f2:.3f} Hz")
        a1.axhline(fn, color="0.5", ls=":")
        a1.set_xlabel("P [kW]"); a1.set_ylabel("f [Hz]")
        a1.grid(alpha=.3); a1.legend(fontsize=8)
        a1.set_title("f = f_nom (1 - ρ P/P_max)")
        t, fs, Pm, Pe = core.frequency_dynamics(H, Pmax, rho, P / Pmax, dP / Pmax, fnom=fn,
                                                Tg=Tg, D=1.0, t_end=30, dt=0.01,
                                                secondary_Ki=0.05 if self.b("sec") else 0.0)
        # częstotliwość w modelu dynamicznym liczona względem punktu pracy przed skokiem
        fs = fs - fn + f
        a2 = self.fig.add_subplot(2, 2, 2)
        a2.plot(t, fs, "tab:blue", lw=1.8)
        a2.axhline(f2, color="tab:orange", ls="--", lw=1, label="statyzm (stan ustalony)")
        nad = fs.min() if dP >= 0 else fs.max()
        a2.set_ylabel("f [Hz]"); a2.grid(alpha=.3); a2.legend(fontsize=8)
        a2.set_title("Dynamika po skoku obciążenia (RK4)")
        a3 = self.fig.add_subplot(2, 2, 4, sharex=a2)
        a3.plot(t, Pm * Pmax, "tab:green", lw=1.8, label="P_m (turbina)")
        a3.plot(t, Pe * Pmax, "tab:red", lw=1.2, ls="--", label="P_e (obciążenie)")
        a3.set_xlabel("t [s]"); a3.set_ylabel("P [kW]"); a3.grid(alpha=.3); a3.legend(fontsize=8)
        rocof = np.max(np.abs(np.diff(fs) / np.diff(t)))
        return [f"f = {fn:.0f}·(1 - {rho:.3f}·{P:.1f}/{Pmax:.0f})",
                f"  = {f:.3f} Hz",
                "",
                f"Po skoku ΔP = {dP:+.1f} kW:",
                f"  f ustalona (statyzm) = {f2:.3f} Hz",
                f"  nadir/szczyt         = {nad:.3f} Hz",
                f"  RoCoF max            = {rocof:.3f} Hz/s",
                f"  f końcowa (t=30 s)   = {fs[-1]:.3f} Hz",
                "",
                "Książka: 50 kW, ρ=4 %, 25 kW -> 58,8 Hz"]


class Tab62(ExampleTab):
    key = "6.2"
    PARAMS = [("PT", "obciążenie P_T [kW]", 0, 50, 20, 0.5),
              ("P1", "P_1max [kW]", 5, 50, 20, 1),
              ("P2", "P_2max [kW]", 5, 50, 30, 1),
              ("r1", "ρ_1 [%]", 1, 10, 4, 0.5),
              ("r2", "ρ_2 [%]", 1, 10, 4, 0.5),
              ("fn", "f_nom [Hz]", 50, 60, 60, 10)]

    def update_plot(self):
        PT, P1m, P2m = self.p("PT"), self.p("P1"), self.p("P2")
        r1, r2, fn = self.p("r1") / 100, self.p("r2") / 100, self.p("fn")
        f, (P1, P2) = core.droop_sharing(PT, [P1m, P2m], [r1, r2], fn)
        fmin = fn * (1 - max(r1, r2) * 1.3)
        axs = [self.fig.add_subplot(1, 3, k + 1) for k in range(3)]
        k_sys = P1m / r1 + P2m / r2
        ff = np.linspace(fmin, fn, 50)
        axs[0].plot(k_sys * (fn - ff) / fn, ff, "k", lw=2)
        axs[0].plot(PT, f, "ro")
        axs[0].set_title("System (suma)")
        for ax, Pm, r, Pi, col, nm in ((axs[1], P1m, r1, P1, "tab:blue", "Generator 1"),
                                       (axs[2], P2m, r2, P2, "tab:orange", "Generator 2")):
            pp = np.linspace(0, Pm, 40)
            ax.plot(pp, fn * (1 - r * pp / Pm), color=col, lw=2)
            ax.plot(Pi, f, "ro")
            ax.set_title(f"{nm} ({Pm:.0f} kW, ρ={100 * r:.1f}%)")
            if Pi > Pm:
                ax.text(0.05, 0.05, "PRZECIĄŻENIE!", color="r", transform=ax.transAxes)
        for ax in axs:
            ax.axhline(f, color="r", ls="--", lw=1)
            ax.axhline(fn, color="0.6", ls=":")
            ax.set_xlabel("P [kW]"); ax.grid(alpha=.3)
            ax.set_ylim(fmin, fn + (fn - fmin) * 0.1)
        axs[0].set_ylabel("f [Hz]")
        return [f"f  = {f:.4f} Hz",
                f"P1 = {P1:.3f} kW ({100 * P1 / max(PT, 1e-9):.1f} % obciążenia)",
                f"P2 = {P2:.3f} kW ({100 * P2 / max(PT, 1e-9):.1f} % obciążenia)",
                f"P1/P1max = {P1 / P1m:.3f},  P2/P2max = {P2 / P2m:.3f}",
                "",
                "Książka (20 kW; 20 i 30 kW; ρ=0,04):",
                "  P1 = 8 kW, P2 = 12 kW, f = 59,04 Hz"]


class Tab63(ExampleTab):
    key = "6.3"
    PARAMS = [("L", "obciążenie L [MW]", 600, 1500, 800, 5),
              ("R", "produkcja OZE [MW] (Przykład 6.4: 200)", 0, 500, 0, 5)]

    def update_plot(self):
        L, R = self.p("L"), self.p("R")
        Ln = L - R
        r = core.ed_quadratic(core.UNITS_Q, Ln)
        lam = r["lam"]
        a1 = self.fig.add_subplot(1, 2, 1)
        pp = np.linspace(0, 700, 200)
        for i, u in enumerate(core.UNITS_Q):
            a1.plot(pp, core.q_marginal(u, pp), color=C_UNITS[i], lw=2, label=f"dC{i + 1}/dP{i + 1}")
            a1.plot(r["P"][i], lam, "o", color=C_UNITS[i], ms=8)
            a1.vlines(r["P"][i], 0, lam, color=C_UNITS[i], ls=":")
        a1.axhline(lam, color="k", ls="--", label=f"λ = {lam:.3f} $/MWh")
        a1.set_xlabel("P [MW]"); a1.set_ylabel("koszt krańcowy [$/MWh]")
        a1.set_ylim(0, max(120, lam * 1.2)); a1.grid(alpha=.3); a1.legend(fontsize=8)
        a1.set_title("Równe koszty krańcowe = optimum")
        a2 = self.fig.add_subplot(2, 2, 2)
        for i, u in enumerate(core.UNITS_Q):
            a2.plot(pp, core.q_cost(u, pp) / 1e3, color=C_UNITS[i], lw=2, label=f"C{i + 1}(P)")
            a2.plot(r["P"][i], core.q_cost(u, r["P"][i]) / 1e3, "o", color=C_UNITS[i])
        a2.set_xlabel("P [MW]"); a2.set_ylabel("C [k$/h]"); a2.grid(alpha=.3); a2.legend(fontsize=8)
        a2.set_title("Krzywe kosztu (paraboliczne)")
        a3 = self.fig.add_subplot(2, 2, 4)
        Ls = np.linspace(300, 1500, 100)
        lams = [core.ed_quadratic(core.UNITS_Q, x)["lam"] for x in Ls]
        a3.plot(Ls, lams, "k", lw=2)
        a3.plot(Ln, lam, "ro")
        a3.set_xlabel("obciążenie netto [MW]"); a3.set_ylabel("λ [$/MWh]"); a3.grid(alpha=.3)
        a3.set_title("Koszt krańcowy systemu rośnie z obciążeniem")
        lines = [f"L = {L:.0f} MW, OZE = {R:.0f} MW, L_net = {Ln:.0f} MW",
                 f"λ = {lam:.3f} $/MWh", ""]
        for i, u in enumerate(core.UNITS_Q):
            lines.append(f"P{i + 1} = {r['P'][i]:8.2f} MW  dC/dP = {core.q_marginal(u, r['P'][i]):.3f}")
        lines += [f"Suma = {sum(r['P']):.2f} MW", f"C = {r['cost']:,.2f} $/h".replace(",", " "), "",
                  "(bez ograniczeń P_min/P_max -", " patrz zakładka 6.5-6.6)", "",
                  "Książka: L=800 -> λ=47,012; C=37 523,69",
                  "         L=1200 -> λ=66,547",
                  "         L_net=600 -> λ=37,244; C=29 098,11"]
        return lines


class Tab65(ExampleTab):
    key = "6.5"
    PARAMS = [("L", "obciążenie L [MW]", 600, 1450, 1200, 5),
              ("R", "produkcja OZE [MW] (Przykład 6.6: 200)", 0, 400, 0, 5)]
    CHECKS = [("lim", "uwzględnij P_min i P_max", True)]

    def update_plot(self):
        L, R = self.p("L"), self.p("R")
        Ln = L - R
        lim = self.b("lim")
        r = core.ed_quadratic(core.UNITS_Q, Ln, limits=lim)
        r0 = core.ed_quadratic(core.UNITS_Q, Ln, limits=False)
        a1 = self.fig.add_subplot(2, 1, 1)
        x = np.arange(3)
        a1.bar(x - 0.2, r0["P"], 0.4, color="0.7", label="bez ograniczeń")
        a1.bar(x + 0.2, r["P"], 0.4, color=C_UNITS[:3], label="z ograniczeniami" if lim else "obliczone")
        for i, u in enumerate(core.UNITS_Q):
            a1.hlines([u["Pmin"], u["Pmax"]], i - 0.45, i + 0.45, colors=["k", "r"], linestyles=["--", "-"])
        a1.set_xticks(x, [u["name"] for u in core.UNITS_Q])
        a1.set_ylabel("P [MW]"); a1.grid(alpha=.3, axis="y"); a1.legend(fontsize=8)
        a1.set_title("Rozdział mocy (czarna kreska = P_min, czerwona = P_max)")
        a2 = self.fig.add_subplot(2, 1, 2)
        Ls = np.linspace(600, 1450, 120)
        avg, mc = [], []
        for x_ in Ls:
            rr = core.ed_quadratic(core.UNITS_Q, x_, limits=True)
            avg.append(rr["cost"] / x_)
            free = [i for i in range(3) if i not in rr["fixed"]]
            mc.append(rr["lam"] if free else np.nan)
        a2.plot(Ls, avg, "k", lw=2, label="koszt średni C/L")
        a2.plot(Ls, mc, "k--", lw=1.5, label="koszt krańcowy λ")
        a2.axvline(Ln, color="r", lw=1)
        a2.set_xlabel("obciążenie [MW]"); a2.set_ylabel("$/MWh")
        a2.grid(alpha=.3); a2.legend(fontsize=8)
        a2.set_title("Rys. 6.11 - koszt średni i krańcowy (z ograniczeniami)")
        lines = [f"L_net = {Ln:.0f} MW", ""]
        for k, (lam, P, fixed) in enumerate(r["log"]):
            lines.append(f"Iteracja {k + 1}: λ = {lam:.3f}")
            lines.append("  P = " + " / ".join(f"{p:.2f}" for p in P))
            if fixed:
                lines.append("  przypięte: " + ", ".join(f"U{i + 1}={v:.0f}" for i, v in fixed.items()))
        lines += ["", "WYNIK:"]
        for i, u in enumerate(core.UNITS_Q):
            st = "  (granica)" if i in r["fixed"] else ""
            lines.append(f"P{i + 1} = {r['P'][i]:7.2f} MW  dC/dP = {core.q_marginal(u, r['P'][i]):.2f}{st}")
        lines.append(f"C = {r['cost']:,.2f} $/h".replace(",", " "))
        lines.append(f"(bez ograniczeń: {r0['cost']:,.2f} $/h)".replace(",", " "))
        if Ln < 600 or Ln > 1450:
            lines.append("UWAGA: L_net poza zakresem Σ P_min..Σ P_max!")
        return lines


class Tab67(ExampleTab):
    key = "6.7"
    PARAMS = [("L", "obciążenie L [MW]", 600, 1450, 800, 5)]

    def update_plot(self):
        L = self.p("L")
        r = core.ed_pwl(core.UNITS_PWL, L)
        a1 = self.fig.add_subplot(1, 2, 1)
        for i, u in enumerate(core.UNITS_PWL):
            Ps, Cs = core.pwl_points(u)
            pp = np.linspace(u["Pmin"], u["Pmax"], 80)
            a1.plot(pp, core.q_cost(core.UNITS_Q[i], pp) / 1e3, color=C_UNITS[i], ls=":", lw=1.2)
            a1.plot(Ps, np.array(Cs) / 1e3, "o-", color=C_UNITS[i], lw=2, label=f"U{i + 1} łamana")
            a1.plot(r["P"][i], core.pwl_cost(u, r["P"][i]) / 1e3, "k*", ms=12)
        a1.set_xlabel("P [MW]"); a1.set_ylabel("C [k$/h]"); a1.grid(alpha=.3); a1.legend(fontsize=8)
        a1.set_title("Łamana (ciągła) vs parabola (kropkowana)")
        a2 = self.fig.add_subplot(1, 2, 2)
        segs = core.pwl_segments(core.UNITS_PWL)
        x0 = 600.0
        a2.axvspan(0, 600, color="0.85", label="Σ P_min = 600 MW")
        for s in segs:
            a2.bar(x0 + s["length"] / 2, s["mc"], width=s["length"], color=C_UNITS[s["unit"]],
                   edgecolor="k", alpha=0.85)
            a2.text(x0 + s["length"] / 2, s["mc"] + 1.5, f"U{s['unit'] + 1}", ha="center", fontsize=7)
            x0 += s["length"]
        a2.axvline(L, color="r", lw=2, label=f"L = {L:.0f} MW")
        a2.axhline(r["lam"], color="k", ls="--", lw=1, label=f"λ = {r['lam']:.1f} $/MWh")
        for i in range(3):
            a2.bar(0, 0, color=C_UNITS[i], label=f"Jednostka {i + 1}")
        a2.set_xlim(0, 1500); a2.set_ylim(0, 110)
        a2.set_xlabel("łączna moc [MW]"); a2.set_ylabel("koszt krańcowy [$/MWh]")
        a2.grid(alpha=.3); a2.legend(fontsize=7, loc="upper left")
        a2.set_title("Krzywa podaży 'merit order'")
        lines = [f"L = {L:.0f} MW, reszta po P_min = {L - 600:.0f} MW", "",
                 "Kolejność odcinków (jedn., MW, $/MWh):"]
        for k, (s, take) in enumerate(r["trace"]):
            lines.append(f" {k + 1}. U{s['unit'] + 1}  +{take:5.0f}/{s['length']:3.0f}  {s['mc']:5.1f}")
        lines.append("")
        for i in range(3):
            lines.append(f"P{i + 1} = {r['P'][i]:.2f} MW")
        mu = r["marginal_unit"]
        lines += [f"λ = {r['lam']:.1f} $/MWh (jednostka krańcowa U{mu + 1 if mu is not None else '-'})",
                  f"C = {r['cost']:,.0f} $/h".replace(",", " "), "",
                  "Książka: 800 -> 200/300/300, λ=50,2",
                  "         1200 -> 300/450/450, λ=73,4"]
        return lines


class Tab69(ExampleTab):
    key = "6.9"
    PARAMS = [("Llow", "obciążenie w 'dolinie' [MW]", 600, 1000, 800, 10),
              ("Lhigh", "obciążenie w 'szczycie' [MW]", 900, 1450, 1200, 10),
              ("E", "przesunięta / zmagazynowana energia [MWh]", 0, 200, 50, 5),
              ("eta", "sprawność baterii (round-trip) [%]", 30, 100, 80, 1)]

    def update_plot(self):
        Ll, Lh, E, eta = self.p("Llow"), self.p("Lhigh"), self.p("E"), self.p("eta") / 100
        U = core.UNITS_PWL

        def C(L):
            return core.ed_pwl(U, L)["cost"]
        base = C(Ll) + C(Lh)
        flex = C(Ll + E) + C(Lh - E)
        stor = C(Ll + E) + C(Lh - eta * E)
        a1 = self.fig.add_subplot(1, 2, 1)
        names = ["bez zmian", "elastyczny\nodbiór (6.9)", "magazyn\n(6.10)"]
        lows = [C(Ll), C(Ll + E), C(Ll + E)]
        highs = [C(Lh), C(Lh - E), C(Lh - eta * E)]
        x = np.arange(3)
        a1.bar(x, np.array(lows) / 1e3, color="tab:blue", label="godzina doliny")
        a1.bar(x, np.array(highs) / 1e3, bottom=np.array(lows) / 1e3, color="tab:red", label="godzina szczytu")
        for k, tot in enumerate([base, flex, stor]):
            a1.text(k, tot / 1e3 + 1, f"{tot:,.0f} $".replace(",", " "), ha="center", fontsize=9)
        a1.set_xticks(x, names); a1.set_ylabel("koszt [k$]")
        a1.set_ylim(0, base / 1e3 * 1.15); a1.grid(alpha=.3, axis="y"); a1.legend(fontsize=8)
        a1.set_title("Koszt dwóch godzin")
        a2 = self.fig.add_subplot(1, 2, 2)
        Es = np.arange(0, 205, 5)
        etas = [0.6, 0.7, 0.8, 0.9, 1.0]
        a2.plot(Es, [base - (C(Ll + e) + C(Lh - e)) for e in Es], "k", lw=2.5, label="elastyczny odbiór")
        for et in etas:
            a2.plot(Es, [base - (C(Ll + e) + C(Lh - et * e)) for e in Es], lw=1.3, label=f"magazyn η={et:.0%}")
        a2.axhline(0, color="0.5", lw=0.8)
        a2.axvline(E, color="r", ls="--", lw=1)
        a2.set_xlabel("energia [MWh]"); a2.set_ylabel("oszczędność [$]")
        a2.grid(alpha=.3); a2.legend(fontsize=7)
        a2.set_title("Oszczędność vs ilość energii")
        rl, rh = core.ed_pwl(U, Ll), core.ed_pwl(U, Lh)
        return [f"λ dolina = {rl['lam']:.1f} $/MWh, λ szczyt = {rh['lam']:.1f} $/MWh", "",
                f"C({Ll:.0f}) + C({Lh:.0f}) = {base:,.0f} $".replace(",", " "),
                "", "ELASTYCZNY ODBIÓR:",
                f"C({Ll + E:.0f}) + C({Lh - E:.0f}) = {flex:,.0f} $".replace(",", " "),
                f"oszczędność = {base - flex:,.0f} $".replace(",", " "),
                "", f"MAGAZYN (η = {eta:.0%}): oddaje {eta * E:.1f} MWh",
                f"C({Ll + E:.0f}) + C({Lh - eta * E:.1f}) = {stor:,.0f} $".replace(",", " "),
                f"zysk arbitrażu = {base - stor:,.0f} $".replace(",", " "),
                "  " + ("OPŁACA SIĘ" if base - stor > 0 else "NIE OPŁACA SIĘ"), "",
                "Książka: 98 230 -> 97 350 $ (6.9)",
                "         -> 97 944 $ (6.10, η=80 %)"]


class Tab611(ExampleTab):
    key = "6.11"
    PARAMS = [("L", "obciążenie L [MW]", 200, 1450, 475, 5)]

    def update_plot(self):
        L = self.p("L")
        U = core.UNITS_PWL
        rows = core.combos_table(U, L)
        a1 = self.fig.add_subplot(1, 1, 1)
        Ls = np.linspace(200, 1450, 251)
        best = np.full(len(Ls), np.nan)
        styles = ["-", "--", "-.", ":", "-", "--", "-"]
        for k, (mask, _r) in enumerate(rows):
            idx = [i for i in range(3) if mask[i]]
            cost = np.array([core.ed_pwl(U, x, idx)["cost"] for x in Ls])
            best = np.fmin(best, cost)
            lab = "Jednostki " + " & ".join(str(i + 1) for i in idx)
            a1.plot(Ls, cost / 1e3, ls=styles[k], lw=1.8, label=lab)
        a1.plot(Ls, best / 1e3, color="k", lw=4, alpha=0.25, label="najtańszy wybór")
        a1.axvline(L, color="r", lw=1.5)
        a1.set_xlabel("obciążenie [MW]"); a1.set_ylabel("koszt całkowity [k$/h]")
        a1.grid(alpha=.3); a1.legend(fontsize=8, ncol=2)
        a1.set_title("Rys. 6.12 - koszt dla różnych kombinacji jednostek")
        lines = [f"L = {L:.0f} MW", "", " U1    U2    U3   koszt[$]  λ"]
        feas = []
        for mask, r in rows:
            cells = [f"{r['P'][i]:5.0f}" if mask[i] else "   - " for i in range(3)]
            if r["feasible"]:
                feas.append((r["cost"], mask))
                lines.append(" ".join(cells) + f" {r['cost']:9.0f} {r['lam']:5.1f}")
            else:
                lines.append(" ".join(c if c.strip() == "-" else "  X  " for c in cells) + "  niewykonalne")
        if feas:
            c, m = min(feas)
            lines += ["", "NAJTAŃSZE: jednostki " + ", ".join(str(i + 1) for i in range(3) if m[i]),
                      f"           {c:,.0f} $/h".replace(",", " ")]
        return lines


class Tab612(ExampleTab):
    key = "6.12"
    PARAMS = [("N", "liczba jednostek N", 1, 100, 30, 1),
              ("T", "liczba okresów T [h]", 1, 168, 24, 1),
              ("speed", "log10(kombinacji/s) komputera", 3, 18, 9, 1)]

    def update_plot(self):
        N, T, sp = int(self.p("N")), int(self.p("T")), self.p("speed")
        log_c = N * T * math.log10(2)
        log_years = log_c - sp - math.log10(3.156e7)
        a1 = self.fig.add_subplot(1, 2, 1)
        Ns = np.arange(1, 101)
        for TT, col in ((1, "tab:blue"), (24, "tab:orange"), (168, "tab:green")):
            a1.plot(Ns, Ns * TT * math.log10(2), color=col, lw=2, label=f"T = {TT} h")
        a1.plot(N, log_c, "ro", ms=8)
        a1.axhline(80, color="0.5", ls=":", lw=1)
        a1.text(2, 85, "≈ liczba atomów we Wszechświecie (10^80)", fontsize=8, color="0.4")
        a1.set_xlabel("N jednostek"); a1.set_ylabel("log10(liczba kombinacji)")
        a1.grid(alpha=.3); a1.legend(fontsize=8)
        a1.set_title("(2^N)^T kombinacji")
        a2 = self.fig.add_subplot(1, 2, 2)
        a2.barh(["Twoje zadanie", "wiek Wszechświata", "życie człowieka", "1 dzień"],
                [log_years, math.log10(1.38e10), math.log10(80), math.log10(1 / 365)],
                color=["tab:red", "tab:purple", "tab:green", "tab:blue"])
        a2.set_xlabel("log10(lat)"); a2.grid(alpha=.3, axis="x")
        a2.set_title("Ile lat liczyłby komputer metodą 'brute force'")
        m, e = 10 ** (log_c % 1), int(log_c // 1)
        my, ey = 10 ** (log_years % 1), int(log_years // 1)
        return [f"N = {N}, T = {T}",
                f"(2^{N})^{T} = 2^{N * T}",
                f"         = {m:.2f} × 10^{e}", "",
                f"Przy 10^{sp:.0f} kombinacji/s:",
                f"czas = {my:.2f} × 10^{ey} lat", "",
                "Książka: N=30, T=24 -> 5,5×10^216",
                "         1,7×10^200 lat przy 10^9/s"]


class Tab613(ExampleTab):
    key = "6.13"
    PARAMS = [("upG", "min. czas pracy Gretel [h]", 1, 4, 3, 1),
              ("dnH", "min. czas postoju Hansela [h]", 1, 4, 2, 1),
              ("stH", "koszt rozruchu Hansela [k$]", 0, 600, 375, 1),
              ("l1", "obciążenie godz. 1 [MW]", 500, 10000, 4000, 100),
              ("l2", "obciążenie godz. 2 [MW]", 500, 10000, 800, 100),
              ("l3", "obciążenie godz. 3 [MW]", 500, 10000, 5000, 100),
              ("l4", "obciążenie godz. 4 [MW]", 500, 10000, 3000, 100)]
    CHECKS = [("cup", "krok 2: min. czas pracy Gretel", True),
              ("cdn", "krok 3: min. czas postoju Hansela", True),
              ("cst", "krok 4: koszt rozruchu Hansela", True)]

    def units(self, up, dn, st):
        return [dict(name="Hansel", Pmin=3000, Pmax=10000, mc=20, st=st, min_down=dn),
                dict(name="Gretel", Pmin=500, Pmax=5000, mc=70, min_up=up)]

    def update_plot(self):
        load = [self.p(f"l{k}") for k in range(1, 5)]
        up = int(self.p("upG")) if self.b("cup") else 1
        dn = int(self.p("dnH")) if self.b("cdn") else 1
        st = self.p("stH") * 1e3 if self.b("cst") else 0.0
        r = core.unit_commitment(self.units(up, dn, st), load)
        # koszty kolejnych kroków (zawsze z bieżącym profilem obciążenia)
        steps = [("bazowy", self.units(1, 1, 0)),
                 ("+min up", self.units(int(self.p("upG")), 1, 0)),
                 ("+min down", self.units(int(self.p("upG")), int(self.p("dnH")), 0)),
                 ("+rozruch", self.units(int(self.p("upG")), int(self.p("dnH")), self.p("stH") * 1e3))]
        costs = []
        for nm, uu in steps:
            rr = core.unit_commitment(uu, load)
            costs.append(rr["cost"] if rr["ok"] else np.nan)
        a1 = self.fig.add_subplot(1, 2, 1)
        h = np.arange(1, 5)
        if r["ok"]:
            a1.bar(h, r["P"][0], color="tab:blue", label="Hansel (20 $/MWh)")
            a1.bar(h, r["P"][1], bottom=r["P"][0], color="tab:pink", label="Gretel (70 $/MWh)")
        a1.hlines(load, h - 0.42, h + 0.42, colors="k", lw=3, label="obciążenie")
        a1.set_xticks(h); a1.set_xlabel("godzina"); a1.set_ylabel("MW")
        a1.grid(alpha=.3, axis="y"); a1.legend(fontsize=8)
        a1.set_title("Optymalny harmonogram (MILP)")
        a2 = self.fig.add_subplot(1, 2, 2)
        a2.bar(range(4), np.array(costs) / 1e3, color=["tab:green", "tab:olive", "tab:orange", "tab:red"])
        for k, c in enumerate(costs):
            a2.text(k, c / 1e3 + 10, f"{c / 1e3:.0f}k", ha="center")
        a2.set_xticks(range(4), [s[0] for s in steps]); a2.set_ylabel("koszt [k$]")
        a2.grid(alpha=.3, axis="y")
        a2.set_title("Każde ograniczenie podnosi koszt")
        if not r["ok"]:
            return ["Brak rozwiązania dopuszczalnego!", str(r["msg"])]
        lines = ["godz.     1      2      3      4",
                 "Hansel " + " ".join(f"{v:6.0f}" for v in r["P"][0]),
                 "Gretel " + " ".join(f"{v:6.0f}" for v in r["P"][1]), "",
                 f"koszt energii   = {r['energy_cost']:>10,.0f} $".replace(",", " "),
                 f"koszt rozruchów = {r['startup_cost']:>10,.0f} $".replace(",", " "),
                 f"RAZEM           = {r['cost']:>10,.0f} $".replace(",", " "), "",
                 "Kroki (bieżące obciążenie):"]
        for (nm, _), c in zip(steps, costs):
            lines.append(f"  {nm:10s} {c:>10,.0f} $".replace(",", " "))
        lines += ["", "Książka: 296 000 / 346 000 /", "         521 000 / 896 000 $"]
        if self.b("cst") and abs(self.p("stH") - 375) < 0.5 and load == [4000, 800, 5000, 3000]:
            lines += ["", "375 k$ = punkt REMISU: Hansel h3-4", "i 'tylko Gretel' kosztują po 896 000 $"]
        return lines


BOOK_614 = {
    "base": [[20, 20, 20, 20], [10, 20, 30, 30], [0, 0, 0, 0], [0, 0, 10, 0]],
    "res": [[20, 20, 20, 20], [10, 10, 30, 25], [0, 10, 10, 5], [0, 0, 0, 0]],
    "ramp": [[15, 20, 20, 20], [10, 10, 15, 25], [5, 10, 15, 5], [0, 0, 10, 0]],
}


class Tab614(ExampleTab):
    key = "6.14"
    PARAMS = [("R", "wymagana rezerwa R [MW]", 0, 40, 20, 1),
              ("rg", "szybkość zmian Gammy [MW/h]", 1, 50, 5, 1),
              ("ro", "szybkość zmian pozostałych [MW/h]", 1, 50, 20, 1),
              ("l1", "obciążenie godz. 1 [MW]", 15, 100, 30, 1),
              ("l2", "obciążenie godz. 2 [MW]", 15, 100, 40, 1),
              ("l3", "obciążenie godz. 3 [MW]", 15, 100, 60, 1),
              ("l4", "obciążenie godz. 4 [MW]", 15, 100, 50, 1)]
    CHECKS = [("res", "ograniczenie rezerwy", True),
              ("ramp", "ograniczenie szybkości zmian mocy", True)]
    NAMES = ["Alpha", "Beta", "Gamma", "Delta"]

    def units(self):
        ro, rg = self.p("ro"), self.p("rg")
        return [dict(name="Alpha", Pmin=10, Pmax=20, mc=10, st=100, ramp_up=ro),
                dict(name="Beta", Pmin=5, Pmax=30, mc=12, st=100, ramp_up=ro),
                dict(name="Gamma", Pmin=5, Pmax=50, mc=13, st=100, ramp_up=rg),
                dict(name="Delta", Pmin=10, Pmax=20, mc=20, st=0, ramp_up=ro)]

    def update_plot(self):
        load = [self.p(f"l{k}") for k in range(1, 5)]
        U = self.units()
        ini = [True, True, False, False]
        res = self.p("R") if self.b("res") else 0.0
        ramp = self.b("ramp")
        # stan początkowy Alpha/Beta: pracują z mocą pokrywającą obciążenie godz. 1
        r = core.unit_commitment(U, load, reserve=res, ramp=ramp, init_on=ini,
                                 init_P=[20, max(load[0] - 20, 5), 0, 0])
        a1 = self.fig.add_subplot(1, 2, 1)
        h = np.arange(1, 5)
        if r["ok"]:
            bottom = np.zeros(4)
            for g in range(4):
                a1.bar(h, r["P"][g], bottom=bottom, color=C_UNITS[g], label=self.NAMES[g])
                bottom += r["P"][g]
            cap = np.array([sum(U[g]["Pmax"] * r["U"][g][t] for g in range(4)) for t in range(4)])
            a1.plot(h, cap, "k^--", lw=1, label="Σ P_max włączonych")
        a1.hlines(load, h - 0.42, h + 0.42, colors="k", lw=3, label="obciążenie")
        a1.set_xticks(h); a1.set_xlabel("godzina"); a1.set_ylabel("MW")
        a1.grid(alpha=.3, axis="y"); a1.legend(fontsize=8)
        a1.set_title("Harmonogram MILP (słupki = moc, trójkąty = zdolność)")
        a2 = self.fig.add_subplot(2, 2, 2)
        if r["ok"]:
            a2.bar(h, r["headroom"], color="tab:cyan")
        a2.axhline(res, color="r", ls="--", label=f"wymagana rezerwa {res:.0f} MW")
        a2.set_xticks(h); a2.set_ylabel("zapas [MW]"); a2.grid(alpha=.3, axis="y"); a2.legend(fontsize=8)
        a2.set_title("Zapas mocy (headroom) = Σ(P_max·u - P)")
        a3 = self.fig.add_subplot(2, 2, 4)
        variants = [("bazowy", 0.0, False), ("+rezerwa", self.p("R"), False), ("+rampy", self.p("R"), True)]
        mine = []
        for _nm, rr_, rp in variants:
            x = core.unit_commitment(U, load, reserve=rr_, ramp=rp, init_on=ini,
                                     init_P=[20, max(load[0] - 20, 5), 0, 0])
            mine.append(x["cost"] if x["ok"] else np.nan)
        book = [2080, 2125, 2225]
        xx = np.arange(3)
        a3.bar(xx - 0.2, mine, 0.4, color="tab:green", label="MILP (ta aplikacja)")
        a3.bar(xx + 0.2, book, 0.4, color="0.6", label="książka")
        a3.set_xticks(xx, [v[0] for v in variants]); a3.set_ylabel("koszt [$]")
        a3.set_ylim(min(book + mine) * 0.9, max(book + mine) * 1.05)
        a3.grid(alpha=.3, axis="y"); a3.legend(fontsize=8)
        if not r["ok"]:
            return ["Brak rozwiązania dopuszczalnego!", "(zmniejsz rezerwę lub obciążenie)"]
        lines = ["godz.     1     2     3     4"]
        for g in range(4):
            lines.append(f"{self.NAMES[g]:6s} " + " ".join(f"{v:5.0f}" for v in r["P"][g]))
        lines += ["zapas  " + " ".join(f"{v:5.0f}" for v in r["headroom"]), "",
                  f"energia  = {r['energy_cost']:.0f} $", f"rozruchy = {r['startup_cost']:.0f} $",
                  f"RAZEM    = {r['cost']:.0f} $", "",
                  "Porównanie (dane z książki):",
                  f"  bazowy  MILP {mine[0]:6.0f}  książka 2080",
                  f"  rezerwa MILP {mine[1]:6.0f}  książka 2125",
                  f"  rampy   MILP {mine[2]:6.0f}  książka 2225"]
        if load == [30, 40, 60, 50]:
            lines += ["", "Koszt harmonogramów z książki",
                      "policzony tą samą funkcją:",
                      f"  {core.schedule_cost(U, BOOK_614['base'], ini):.0f} / "
                      f"{core.schedule_cost(U, BOOK_614['res'], ini):.0f} / "
                      f"{core.schedule_cost(U, BOOK_614['ramp'], ini):.0f} $",
                      "-> MILP znalazł tańsze, też dopuszczalne"]
        return lines


# =============================================================================
# ROZDZIAŁ 7
# =============================================================================

class Tab71(ExampleTab):
    key = "7.1"
    PARAMS = [("len", "długość linii ℓ [km]", 10, 600, 250, 5),
              ("r", "r [Ω/km]", 0.01, 0.5, 0.3, 0.01),
              ("x", "x [Ω/km]", 0.2, 1.2, 0.9, 0.01),
              ("y", "y [µS/km]", 0.5, 6, 2.0, 0.1),
              ("Vb", "napięcie bazowe [kV]", 110, 765, 345, 5),
              ("P", "obciążenie P [MW] (Przykład 7.3)", 0, 400, 125, 5),
              ("pf", "pf obciążenia (ind.)", 0.5, 1.0, 0.85, 0.01)]

    def update_plot(self):
        ln, r, x, y, Vb = self.p("len"), self.p("r"), self.p("x"), self.p("y") * 1e-6, self.p("Vb")
        P, pf = self.p("P") / 100, self.p("pf")
        Q = P * math.tan(math.acos(pf))
        pi = core.line_pi(r, x, y, ln, Vb, 100)
        o = core.line_open_end(pi)
        w = core.line_with_load(pi, P, Q)
        a1 = self.fig.add_subplot(2, 2, 1)
        Ls = np.linspace(5, 600, 120)
        VR = [abs(core.line_open_end(core.line_pi(r, x, y, L_, Vb, 100))["VR"]) for L_ in Ls]
        a1.plot(Ls, VR, "tab:red", lw=2)
        a1.plot(ln, abs(o["VR"]), "ko")
        a1.axhline(1.0, color="0.5", ls=":")
        a1.set_xlabel("długość [km]"); a1.set_ylabel("|V_R| [pu]")
        a1.grid(alpha=.3); a1.set_title("Efekt Ferrantiego (linia bez obciążenia)")
        a2 = self.fig.add_subplot(2, 2, 3)
        Pp = np.linspace(0, 4, 100)
        tg = math.tan(math.acos(pf))
        a2.plot(Pp * 100, [abs(core.line_with_load(pi, p_, p_ * tg)["Vs"]) for p_ in Pp], "tab:blue",
                lw=2, label=f"pf = {pf:.2f} ind.")
        a2.plot(Pp * 100, [abs(core.line_with_load(pi, p_, 0)["Vs"]) for p_ in Pp], "tab:green",
                lw=1.5, ls="--", label="pf = 1")
        a2.plot(P * 100, abs(w["Vs"]), "ko")
        a2.set_xlabel("P obciążenia [MW]"); a2.set_ylabel("|V_S| potrzebne [pu]")
        a2.grid(alpha=.3); a2.legend(fontsize=8)
        a2.set_title("Ile napięcia trzeba na początku (V_R = 1 pu)")
        a3 = self.fig.add_subplot(1, 2, 2)
        VRv = 1 + 0j
        phasor(a3, VRv, "tab:green", "V_R")
        phasor(a3, w["Vs"], "tab:blue", "V_S")
        phasor(a3, pi["Z"] * w["I1"], "tab:orange", "Z·I1", origin=VRv, lw=1.5)
        sc = 0.5 / max(abs(w["I1"]), 1e-9)
        phasor(a3, w["I1"] * sc, "tab:red", "I1 (skala)", lw=1.5)
        phasor(a3, w["Iload"] * sc, "tab:purple", "I_obc (skala)", lw=1.2)
        finish_phasor_axes(a3, [VRv, w["Vs"], w["I1"] * sc], "Przykład 7.3 - wykres wskazowy")
        return [f"Z_b = {pi['Zb']:.2f} Ω",
                f"R = {pi['R']:.4f} pu, X = {pi['X']:.4f} pu",
                f"B/2 = {pi['B'] / 2:.4f} pu  (-j{2 / pi['B']:.3f} pu)", "",
                "7.2 linia nieobciążona, V_S = 1:",
                f"  V_R = {fmt_c(o['VR'], 'pu')}",
                f"  P = {o['S'].real * 100:.3f} MW, Q = {o['S'].imag * 100:.2f} Mvar", "",
                f"7.3 obciążenie {P * 100:.0f} MW + j{Q * 100:.1f} Mvar:",
                f"  V_S = {fmt_c(w['Vs'], 'pu')}",
                f"  P_gen = {w['S'].real * 100:.2f} MW",
                f"  Q_gen = {w['S'].imag * 100:.2f} Mvar",
                f"  straty = {w['losses'] * 100:.2f} MW", "",
                "Książka: V_R = 1,059 pu; 626 kW; -61,27 Mvar",
                "         136 MW (11 MW strat); 39,6 Mvar"]


class Tab74(ExampleTab):
    key = "7.4"
    PARAMS = [("V1r", "napięcie znamionowe pierwotne [V]", 100, 400, 240, 1),
              ("V2r", "napięcie znamionowe wtórne [V]", 6, 48, 12, 1),
              ("V1", "napięcie zasilania V1 [V]", 50, 260, 200, 1),
              ("R", "R obciążenia [Ω]", 0.5, 20, 5, 0.1),
              ("X", "X obciążenia [Ω] (+ ind., - poj.)", -10, 10, 2, 0.1)]

    def update_plot(self):
        V1r, V2r, V1, R, X = self.p("V1r"), self.p("V2r"), self.p("V1"), self.p("R"), self.p("X")
        r = core.ideal_transformer(V1r, V2r, V1, complex(R, X))
        a = r["a"]
        a1 = self.fig.add_subplot(1, 2, 1)
        t = np.linspace(0, 0.04, 800)
        w = 2 * PI * 50
        v1 = math.sqrt(2) * V1 * np.sin(w * t)
        v2 = math.sqrt(2) * abs(r["V2"]) * np.sin(w * t)
        i2 = math.sqrt(2) * abs(r["I2"]) * np.sin(w * t + cmath.phase(r["I2"]))
        a1.plot(t * 1e3, v1 / a, "tab:blue", lw=2, label="v1 / a [V]")
        a1.plot(t * 1e3, v2, "c--", lw=1.5, label="v2 [V]")
        a1.plot(t * 1e3, i2, "tab:red", lw=2, label="i2 [A]")
        a1.plot(t * 1e3, math.sqrt(2) * abs(r["I1"]) * a * np.sin(w * t + cmath.phase(r["I1"])),
                "m:", lw=2, label="i1·a [A]")
        a1.set_xlabel("t [ms]"); a1.grid(alpha=.3); a1.legend(fontsize=8)
        a1.set_title("Przebiegi czasowe (po sprowadzeniu na stronę wtórną)")
        a2 = self.fig.add_subplot(1, 2, 2)
        V2 = r["V2"]
        sc = abs(V2) / max(abs(r["I2"]), 1e-9) * 0.7
        phasor(a2, V2, "tab:blue", "V2")
        phasor(a2, r["I2"] * sc, "tab:red", "I2 (skala)")
        phasor(a2, R * r["I2"] * 1, "tab:green", "R·I2", lw=1.3)
        phasor(a2, 1j * X * r["I2"], "tab:orange", "jX·I2", origin=R * r["I2"], lw=1.3)
        finish_phasor_axes(a2, [V2, r["I2"] * sc, R * r["I2"]], "Wykres wskazowy strony wtórnej")
        return [f"a = {V1r:.0f}/{V2r:.0f} = {a:.3f}",
                f"V2 = V1/a = {abs(V2):.3f} V",
                f"I2 = V2/Z2 = {fmt_c(r['I2'], 'A')}",
                f"I1 = I2/a  = {fmt_c(r['I1'], 'A')}",
                f"Z2' = a²Z2 = {r['Z2p'].real:.1f} {r['Z2p'].imag:+.1f}j Ω",
                f"I1 = V1/Z2' = {fmt_c(V1 / r['Z2p'], 'A')}", "",
                f"P = {r['S'].real:.3f} W",
                f"Q = {r['S'].imag:.3f} var",
                f"|S| = {abs(r['S']):.3f} VA, pf = {r['S'].real / abs(r['S']):.3f}", "",
                "Książka: V2 = 10 V, I2 = 1,857∠-21,8° A,",
                "         17,24 W, 6,90 var"]


class Tab75(ExampleTab):
    key = "7.5"
    PARAMS = [("S", "moc znamionowa S [kVA]", 1, 100, 12, 1),
              ("V1", "napięcie pierwotne V_LL [kV]", 0.4, 30, 13.8, 0.1),
              ("V2", "napięcie wtórne V_LL [V]", 100, 1000, 416, 1)]

    def update_plot(self):
        S, V1, V2 = self.p("S") * 1e3, self.p("V1") * 1e3, self.p("V2")
        I1, I2 = core.transformer_currents(S, V1, V2)
        a1 = self.fig.add_subplot(1, 2, 1)
        b = a1.bar(["I1 (strona WN)", "I2 (strona nn)"], [I1, I2], color=["tab:blue", "tab:red"])
        a1.bar_label(b, fmt="%.3f A")
        a1.set_ylabel("prąd znamionowy [A]"); a1.set_yscale("log"); a1.grid(alpha=.3, axis="y")
        a1.set_title("Prądy znamionowe (skala log)")
        a2 = self.fig.add_subplot(1, 2, 2)
        ss = np.linspace(1, 100, 100) * 1e3
        a2.plot(ss / 1e3, ss / (SQ3 * V2), "tab:red", lw=2, label="I2 (nn)")
        a2.plot(ss / 1e3, ss / (SQ3 * V1) * 10, "tab:blue", lw=2, label="I1 × 10 (WN)")
        a2.axvline(S / 1e3, color="k", ls="--")
        a2.set_xlabel("S [kVA]"); a2.set_ylabel("I [A]"); a2.grid(alpha=.3); a2.legend(fontsize=8)
        a2.set_title("I = S / (√3 V_LL)")
        return [f"I1 = {S:.0f}/(√3·{V1:.0f}) = {I1:.4f} A",
                f"I2 = {S:.0f}/(√3·{V2:.0f}) = {I2:.3f} A", "",
                f"I2/I1 = {I2 / I1:.2f} = V1/V2 = {V1 / V2:.2f}",
                "straty I²R po stronie nn byłyby", f"{(I2 / I1) ** 2:.0f}× większe przy tym", "samym oporze przewodu", "",
                "Książka: 0,502 A i 16,65 A"]


class Tab76(ExampleTab):
    key = "7.6"
    PARAMS = [("S", "obciążenie [MVA]", 1, 40, 10, 0.5),
              ("pf", "pf obciążenia (ind.)", 0.5, 1.0, 0.8, 0.01),
              ("len", "długość linii [km]", 1, 200, 20, 1),
              ("Sb", "moc bazowa S_b [MVA]", 10, 200, 100, 5)]

    def update_plot(self):
        S, pf, ln, Sb = self.p("S"), self.p("pf"), self.p("len"), self.p("Sb")
        r = core.per_unit_system(Sb=Sb, line_len=ln, load_MVA=S, pf=pf)
        a1 = self.fig.add_subplot(2, 1, 1)
        a1.set_xlim(-0.3, 10.4); a1.set_ylim(-1.2, 1.6); a1.axis("off")
        a1.set_title("Schemat impedancyjny w jednostkach względnych (Rys. 7.14b)")
        boxes = [(1.5, f"jX_G\nj{r['Xg']:.3f}"), (3.6, f"jX_T1\nj{r['Xt1']:.4f}"),
                 (5.7, f"Z_linii\n{r['Zl'].real:.4f}\n+j{r['Zl'].imag:.4f}"), (7.8, f"jX_T2\nj{r['Xt2']:.4f}")]
        a1.plot([0.3, 9.7], [1, 1], "k", lw=1.5)
        a1.plot([0.3, 9.7], [-1, -1], "k", lw=1.5)
        for xc, txt in boxes:
            a1.add_patch(matplotlib.patches.Rectangle((xc - 0.6, 0.75), 1.2, 0.5, fc="#dbe9ff", ec="k"))
            a1.text(xc, 1.0, txt, ha="center", va="center", fontsize=8)
        a1.add_patch(matplotlib.patches.Circle((0.3, 0), 0.35, fc="#ffe9c9", ec="k"))
        a1.text(0.3, 0, "E", ha="center", va="center", fontsize=10)
        a1.plot([0.3, 0.3], [0.35, 1], "k"); a1.plot([0.3, 0.3], [-0.35, -1], "k")
        a1.add_patch(matplotlib.patches.Rectangle((9.4, -0.4), 0.6, 0.8, fc="#e5ffd9", ec="k"))
        a1.text(9.7, 0, "obc.", ha="center", va="center", fontsize=8)
        a1.plot([9.7, 9.7], [0.4, 1], "k"); a1.plot([9.7, 9.7], [-0.4, -1], "k")
        a1.text(2.55, 1.35, "V_gen", ha="center", fontsize=8)
        a1.text(9.7, 1.35, "V_obc = 1∠0°", ha="center", fontsize=8)
        a1.text(5, -0.5, f"I = {fmt_c(r['I'], 'pu')}", ha="center", fontsize=9, color="tab:red")
        a2 = self.fig.add_subplot(2, 2, 3)
        phasor(a2, r["V_load"], "tab:green", "V_obc")
        phasor(a2, r["V_gen"], "tab:blue", "V_gen")
        phasor(a2, r["E"], "tab:purple", "E")
        sc = 0.5 / max(abs(r["I"]), 1e-9)
        phasor(a2, r["I"] * sc, "tab:red", "I (skala)", lw=1.3)
        finish_phasor_axes(a2, [r["E"], r["V_gen"], r["I"] * sc], "Wykres wskazowy")
        a3 = self.fig.add_subplot(2, 2, 4)
        b = a3.bar(["generator\n11 kV", "linia\n132 kV", "obciążenie\n33 kV"], [abs(i) for i in r["I_A"]],
                   color=["tab:purple", "tab:orange", "tab:green"])
        a3.bar_label(b, fmt="%.1f A")
        a3.set_ylabel("prąd [A]"); a3.grid(alpha=.3, axis="y")
        a3.set_title("Ten sam prąd w pu, różne ampery")
        return [f"S_b = {Sb:.0f} MVA",
                f"X_G  = {r['Xg']:.4f} pu", f"X_T1 = {r['Xt1']:.4f} pu",
                f"Z_b linii = {r['Zb_line']:.2f} Ω",
                f"Z_linii = {r['Zl_ohm'].real:.2f}+j{r['Zl_ohm'].imag:.2f} Ω",
                f"        = {r['Zl'].real:.4f}+j{r['Zl'].imag:.4f} pu",
                f"X_T2 = {r['Xt2']:.4f} pu", f"S_obc = {r['Spu']:.3f} pu", "",
                f"I     = {fmt_c(r['I'], 'pu')}",
                f"V_gen = {fmt_c(r['V_gen'], 'pu')}",
                f"E     = {fmt_c(r['E'], 'pu')}", "",
                f"I_b: {r['Ib'][0]:.1f} / {r['Ib'][1]:.1f} / {r['Ib'][2]:.1f} A",
                f"I_gen = {abs(r['I_A'][0]):.2f} A",
                f"I_lin = {abs(r['I_A'][1]):.2f} A",
                f"I_obc = {abs(r['I_A'][2]):.2f} A", "",
                "Książka: V_gen = 1,032∠2,11° pu",
                "  524,87 / 43,74 / 174,95 A"]


class Tab77(ExampleTab):
    key = "7.7"
    PARAMS = [("t", "przekładnia zaczepów t", 0.85, 1.15, 1.05, 0.0125),
              ("XT", "X_T transformatora [pu]", 0.02, 0.3, 0.1, 0.005),
              ("RL", "R_L obciążenia [pu]", 0.3, 3.0, 1.0, 0.05),
              ("XL", "X_L obciążenia [pu]", 0.5, 20, 5.0, 0.1)]

    def update_plot(self):
        t, XT, RL, XL = self.p("t"), self.p("XT"), self.p("RL"), self.p("XL")
        r = core.tap_changer(t, XT, RL, XL)
        r1 = core.tap_changer(1.0, XT, RL, XL)
        ts = np.linspace(0.85, 1.15, 121)
        rr = [core.tap_changer(x, XT, RL, XL) for x in ts]
        a1 = self.fig.add_subplot(2, 2, 1)
        a1.plot(ts, [abs(x["VL"]) for x in rr], "tab:blue", lw=2)
        a1.plot(t, abs(r["VL"]), "ro")
        a1.axhline(1.0, color="0.5", ls=":")
        a1.set_xlabel("t"); a1.set_ylabel("|V_L| [pu]"); a1.grid(alpha=.3)
        a1.set_title("Napięcie odbioru vs zaczep")
        a2 = self.fig.add_subplot(2, 2, 3)
        a2.plot(ts, [x["S"].real for x in rr], "tab:green", lw=2, label="P źródła")
        a2.plot(ts, [x["S"].imag for x in rr], "tab:red", lw=2, label="Q źródła")
        a2.axvline(t, color="k", ls="--", lw=1)
        a2.set_xlabel("t"); a2.set_ylabel("[pu]"); a2.grid(alpha=.3); a2.legend(fontsize=8)
        a2.set_title("Moc pobierana ze źródła")
        a3 = self.fig.add_subplot(1, 2, 2)
        a3.set_xlim(0, 10); a3.set_ylim(-2, 3); a3.axis("off")
        a3.set_title(f"Schemat π dla t = {t:.4f} (Rys. 7.18b)")
        a3.plot([0.5, 9.5], [2, 2], "k"); a3.plot([0.5, 9.5], [-1, -1], "k")
        a3.add_patch(matplotlib.patches.Rectangle((3.0, 1.7), 4.0, 0.6, fc="#dbe9ff", ec="k"))
        a3.text(5, 2, f"Z12 = jX/t = j{XT / t:.4f}", ha="center", va="center", fontsize=8)

        def shunt(x, z, lab):
            inductive = np.isfinite(abs(z)) and z.imag > 0
            a3.add_patch(matplotlib.patches.Rectangle((x - 0.5, -0.2), 1.0, 1.4,
                                                      fc="#ffe9c9" if inductive else "#e5ffd9", ec="k"))
            a3.plot([x, x], [1.2, 2], "k"); a3.plot([x, x], [-0.2, -1], "k")
            txt = "∞ (brak)" if not np.isfinite(abs(z)) else f"{z.imag:+.3f}j"
            kind = "" if not np.isfinite(abs(z)) else ("dławik" if z.imag > 0 else "kondensator")
            a3.text(x, 0.5, f"{lab}\n{txt}\n{kind}", ha="center", va="center", fontsize=7)
        shunt(2.2, r["Z10"], "Z10")
        shunt(7.8, r["Z20"], "Z20")
        a3.text(0.5, 2.3, "V_S = 1", fontsize=8)
        a3.text(8.6, 2.3, f"V_L = {abs(r['VL']):.4f}", fontsize=8)
        a3.text(5, -1.6, "t > 1: Z10 dławik, Z20 kondensator\nt < 1: odwrotnie",
                ha="center", fontsize=8, color="0.3")
        return [f"t = {t:.4f}",
                "Z10 = jX/(t(t-1)) = " + ("∞" if not np.isfinite(abs(r["Z10"])) else f"{r['Z10'].imag:+.4f}j pu"),
                "Z20 = jX/(1-t)    = " + ("∞" if not np.isfinite(abs(r["Z20"])) else f"{r['Z20'].imag:+.4f}j pu"), "",
                f"V_L = {fmt_c(r['VL'], 'pu')}",
                f"I_S = {fmt_c(r['Is'], 'pu')}",
                f"S   = {r['S'].real:.4f} + j{r['S'].imag:.4f} pu", "",
                "Sprawdzenie modelem π:",
                f"V_L = {fmt_c(r['VL_pi'], 'pu')}",
                f"I_S = {fmt_c(r['Is_pi'], 'pu')}", "",
                f"Przy t = 1: V_L = {abs(r1['VL']):.4f} pu,",
                f"            S = {r1['S'].real:.4f}+j{r1['S'].imag:.4f}",
                f"Zmiana |V_L|: {100 * (abs(r['VL']) / abs(r1['VL']) - 1):+.2f} %", "",
                "Książka: t=1 -> V_L = 0,976 pu",
                "         t=1,05 -> V_L = 1,024 pu"]


# =============================================================================
# GŁÓWNE OKNO
# =============================================================================

CHAPTERS = [
    ("Rozdz. 5: Przekształtniki", [
        ("5.1 Moc P-Q", Tab51), ("5.3 PWM mostek H", Tab53PWM), ("5.3 Sterowanie P i Q", Tab53PQ),
        ("5.4 PV i MPPT", Tab54PV), ("5.4 Wiatr typ 3/4", Tab54Wind)]),
    ("Rozdz. 6: Bilans mocy i ekonomia", [
        ("6.1 Statyzm", Tab61), ("6.2 Dwa generatory", Tab62), ("6.3-6.4 Dispatch", Tab63),
        ("6.5-6.6 Limity", Tab65), ("6.7-6.8 Łamane", Tab67), ("6.9-6.10 Elastyczność", Tab69),
        ("6.11 Ile jednostek", Tab611), ("6.12 Kombinacje", Tab612), ("6.13 Jaś i Małgosia", Tab613),
        ("6.14 Rezerwa", Tab614)]),
    ("Rozdz. 7: Elementy sieci", [
        ("7.1-7.3 Linia", Tab71), ("7.4 Transformator", Tab74), ("7.5 Prądy znam.", Tab75),
        ("7.6 Jednostki wzgl.", Tab76), ("7.7 Zaczepy", Tab77)]),
]

INTRO = """Witaj! Ta aplikacja zawiera WSZYSTKIE przykłady z rozdziałów 5-7 książki
D. S. Kirschen, "Power Systems: Fundamental Concepts and the Transition to Sustainability".

JAK KORZYSTAĆ
 * Wybierz rozdział (górny pasek), potem przykład (drugi pasek).
 * Po lewej - suwaki. Wartości startowe = dane z książki, więc od razu widzisz wynik przykładu.
 * Po prawej - wykresy (pasek narzędzi pozwala powiększać i zapisywać obrazki).
 * Na dole - objaśnienie z pełnym rozwiązaniem "krok po kroku", pisane dla ucznia liceum.
 * Pole "Wyniki" pokazuje obliczenia dla bieżących suwaków i porównanie z książką.
 * Przycisk "Przywróć dane z książki" wraca do przykładu.

O CZYM SĄ TE ROZDZIAŁY?
 Rozdz. 5 - Energoelektronika: jak tranzystory zamieniają prąd stały (bateria, panel PV)
            na przemienny (sieć) i jak sterujemy mocą czynną i bierną. Turbiny wiatrowe, HVDC.
 Rozdz. 6 - Bilans mocy: w każdej sekundzie produkcja musi równać się zużyciu. Pilnuje tego
            częstotliwość (50 Hz). Potem ekonomia: jak rozdzielić moc między elektrownie
            najtaniej (economic dispatch), które elektrownie włączyć (unit commitment),
            ile rezerwy trzymać.
 Rozdz. 7 - Elementy sieci: linie (model π, efekt Ferrantiego), transformatory (idealny,
            trójfazowy, jednostki względne, przełącznik zaczepów).

UWAGI INŻYNIERA
 * Funkcje kosztu z Przykładu 6.3 nie wydrukowały się w załączniku - zostały ODTWORZONE
   z tabeli Przykładu 6.7 (C1 = 2100 + 8P + 0,1P² itd.) i dają dokładnie liczby z książki
   (λ = 47,012 i 66,547 $/MWh, C = 37 523,69 $/h).
 * Unit commitment (6.13, 6.14) liczy prawdziwy solver MILP (scipy.optimize.milp / HiGHS).
   Przykład 6.13 zgadza się co do dolara; w 6.14 solver znajduje harmonogramy o kilka
   dolarów tańsze niż tabela z książki - wyjaśnienie w zakładce.
 * Zakładki 5.3, 5.4 nie są numerowanymi przykładami - to interaktywne ilustracje teorii
   z rozdziału 5 (PWM, sterowanie P/Q, MPPT, turbiny wiatrowe), bo jedyny przykład
   liczbowy rozdziału 5 to Przykład 5.1.
"""


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Przekształtniki, bilans mocy i elementy sieci - przykłady z rozdz. 5-7 (Kirschen)")
        self.geometry("1450x930")
        style = ttk.Style(self)
        style.configure("TNotebook.Tab", padding=(8, 3))
        top = ttk.Notebook(self)
        top.pack(fill=tk.BOTH, expand=True)
        intro = ttk.Frame(top)
        txt = ScrolledText(intro, wrap=tk.WORD, font=("Segoe UI", 11), bg="#fffdf3")
        txt.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        txt.insert("1.0", INTRO)
        txt.configure(state=tk.DISABLED)
        top.add(intro, text="Start")
        self.classes = {}
        self.built = {}
        for chap_name, tabs in CHAPTERS:
            inner = ttk.Notebook(top)
            top.add(inner, text=chap_name)
            for name, cls in tabs:
                fr = ttk.Frame(inner)
                inner.add(fr, text=name)
                self.classes[str(fr)] = (fr, cls)
            inner.bind("<<NotebookTabChanged>>", self._lazy)

    def _lazy(self, event):
        """Zakładki budowane dopiero przy pierwszym otwarciu - szybszy start."""
        sel = event.widget.select()
        if sel in self.classes and sel not in self.built:
            fr, cls = self.classes[sel]
            tab = cls(fr)
            tab.pack(fill=tk.BOTH, expand=True)
            self.built[sel] = tab


if __name__ == "__main__":
    App().mainloop()
