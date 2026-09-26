# Maszyny prądu stałego (DC) – interaktywny podręcznik w Tkinter

Plik `dc_machines_gerling_tkinter.py` to aplikacja z **17 zakładkami**. Obejmuje wszystkie przykłady z rozdziału 2 „DC-Machines” (D. Gerling, *Electrical Machines*, Springer 2015).
W każdej zakładce są:

* **suwaki** z parametrami maszyny (wykres przelicza się od razu, na bieżąco),
* **wykresy Matplotlib** (pasek narzędzi daje zoom, przesuwanie i zapis do PNG),
* **wyniki liczbowe** w lewym dolnym panelu,
* **objaśnienie** na dole okna, napisane tak, żeby zrozumiał je uczeń liceum.

```
pip install numpy matplotlib
python dc_machines_gerling_tkinter.py
python test_dc_machines_gerling.py      # testy obliczeń (albo: python -m pytest)
```

| Zakładka | Podrozdział | Co pokazuje |
|---|---|---|
| 0 Budowa | 2.1 | Przekrój maszyny: bieguny, wirnik, żłobki, komutator, szczotki; kierunki prądu pod biegunami |
| 1 Napięcie | 2.2 | Napięcie indukowane w cewce, prostowanie przez komutator, tętnienia dla K cewek |
| 2 Komutacja | 2.2, rys. 2.5–2.6 | Obracana cewka z komutatorem, prąd i moment cewki, silnik/prądnica |
| 3 Uzwojenia | 2.3, rys. 2.7–2.9 | Uzwojenie pętlicowe i faliste: K, z, 2a, kroki y, y1, y2, rysunek rozwinięty |
| 4 Równania | 2.4.1–2.4.4 | Ui = kΦn, T = kΦI/2π, U = Ui + RI, bilans mocy, przesunięcie szczotek |
| 5 Esson | 2.4.5 | **Przykład z książki**: C = 4,28 kW·min/m³, maszyna 100 kW → D ≈ 246 mm, l ≈ 193 mm |
| 6 Żłobki | 2.5 | Dlaczego przewody w żłobkach liczymy „jak w szczelinie” (napięcie i siła) |
| 7 Obcowzbudna | 2.6, rys. 2.16–2.21 | n(I), T(I), T(n), trzy metody regulacji prędkości, ograniczenia |
| 8 Tryby pracy | Tabela 2.1 | Silnik / hamulec / prądnica, znaki mocy, podświetlona tabela |
| 9 Rozruch | równ. 2.17 + ruch | Symulacja RK4 rozruchu z rozrusznikiem m-stopniowym i skokiem obciążenia |
| 10 Magnesy | 2.7, tab. 2.2, rys. 2.23–2.26 | Punkt pracy magnesu, krawędź nabiegająca i zbiegająca, rozmagnesowanie a temperatura |
| 11 Bocznikowa | 2.8, rys. 2.29–2.31 | Samowzbudzenie, rezystancja krytyczna, narastanie napięcia, charakterystyka obciążenia |
| 12 Szeregowa | 2.9, rys. 2.33–2.36 | Charakterystyka „miękka”, rozbieganie, trzy metody regulacji |
| 13 Porównanie | 2.10, rys. 2.38 | Bocznikowa, szeregowa i kompaundowa (suwak udziału wzbudzenia szeregowego s) |
| 14 Zasilanie | 2.11, rys. 2.39–2.41 | Układ Leonarda, mostek tyrystorowy 6-pulsowy, Ud(α), ćwiartki pracy |
| 15 Odd. twornika | 2.12, rys. 2.43–2.45, 2.50 | Wypaczenie pola, nasycenie, uzwojenie kompensacyjne, bieguny komutacyjne |
| 16 Bieguny kom. | 2.13, rys. 2.46–2.49 | Komutacja rzeczywista, napięcie reaktancyjne, iskrzenie, stopień kompensacji |

---

## Wyjaśnienie przykładów (tak, jakby tłumaczył inżynier uczniowi liceum)

### 0. Budowa (2.1)
Maszyna DC ma nieruchomy **stojan** z biegunami N/S (magnesy albo cewki z prądem stałym) oraz obracający się **wirnik (twornik)**. Wirnik to pakiet cienkich blach, w jego żłobkach leżą miedziane przewody.
Prąd trafia do wirujących przewodów przez **szczotki** węglowe, które ślizgają się po **komutatorze**.
Komutator to „mechaniczny przełącznik”: dzięki niemu pod biegunem N prąd płynie zawsze „w kartkę”, a pod S zawsze „z kartki”. Siły F = B·I·l na wszystkich przewodach pchają więc wirnik w tę samą stronę.
W samym przewodzie prąd jest przemienny. Przy n = 1500 obr/min i p = 1 ma częstotliwość f = p·n/60 = 25 Hz.

### 1. Indukowanie napięcia (2.2)
Przewód o długości l, który przecina pole B z prędkością v, wytwarza napięcie u = B·l·v. Cewka ma dwa boki, więc e = 2·wS·B·l·v.
Dla B = 0,8 T, l = 0,2 m, r = 8 cm, n = 1500 obr/min i wS = 10 wychodzi v = 2πrn = 12,6 m/s i e_max ≈ 40 V.
Napięcie jednej cewki „kopiuje” trapezowy rozkład B i zmienia znak. Komutator odwraca końce cewki, więc na szczotkach jest |e|. Tętnienia są jednak ogromne (> 100 %).
Po dodaniu K cewek przesuniętych na obwodzie ich garby się nakładają. Przy K = 16 tętnienia spadają do kilku procent, dlatego prawdziwy komutator ma wiele działek.

### 2. Komutacja i moment cewki (2.2, rys. 2.5–2.6)
Suwak θ obraca wirnik z jedną cewką.
1. Cewka jest pod biegunami: płynie prąd i powstaje moment.
2. Cewka jest w strefie neutralnej (θ ≈ 90°): szczotka zwiera obie działki, prąd zmienia kierunek (to jest **komutacja**), a moment chwilowo spada do zera. Wirnik kręci się dalej dzięki bezwładności.
3. Po 180° prąd w cewce ma przeciwny znak, ale cewka jest też pod przeciwnym biegunem. Moment ma więc ten sam znak co w punkcie 1.

Ujemny prąd oznacza pracę prądnicową: moment hamuje, a U = Ui − R·|I|.

### 3. Uzwojenia (2.3)
Oznaczenia: K = N·u (liczba cewek = liczba działek komutatora), z = 2·wS·K (liczba przewodów).
* **Pętlicowe:** krok komutatorowy y = 1, liczba gałęzi 2a = 2p. Przykład: p = 2, N = 12 daje 2a = 4 i prąd w przewodzie IA/4.
* **Faliste:** y = (K ± 1)/p musi być liczbą całkowitą. Zawsze 2a = 2, więc wystarczą 2 szczotki.
  Dla p = 2 i N = 13 wychodzi y = 6. Dla N = 12 uzwojenie faliste jest niewykonalne (aplikacja to pokazuje).

Kolorowa ścieżka na rysunku to jedna gałąź równoległa. W uzwojeniu pętlicowym zostaje pod jedną parą biegunów, a w falistym obiega cały wirnik.

### 4. Równania główne (2.4)
Trzy równania opisują każdą maszynę DC:

* Ui = k·Φ·n, gdzie k = p·z/a,
* T = k·Φ·IA/2π,
* U = Ui + R·IA (+ L·di/dt w stanach nieustalonych).

Przykład z domyślnymi suwakami: z = 600, p = 2, 2a = 4, Φ = 15 mWb, n = 1500 obr/min, IA = 50 A, R = 0,3 Ω. Wyniki:

* k = 600, Ui = 225 V, T = 71,6 N·m, U = 240 V,
* P_el = 12 kW = P_i (11,25 kW) + P_Cu (0,75 kW),
* sprawdzenie: P_mech = 2π·n·T = 11,25 kW = P_i.

Przesunięcie szczotek o β zmniejsza napięcie, bo szczotki obejmują mniej strumienia.

### 5. Liczba Essona – **przykład z książki** (2.4.5)
Wymiary maszyny ograniczają dwa materiały: żelazo (B ≈ 0,8 T, nasycenie) i miedź (A ≈ 500 A/cm, grzanie).

C = π²·αi·A·B = π²·0,65·50 000·0,8 = 256,6 kW·s/m³ = **4,28 kW·min/m³**.

Projekt: Pi = 100 kW, n = 2000 obr/min, p = 2, l = τp = πD/(2p).

D³ = 2p·Pi/(π·C·n) = 4·100 000 / (π·256 600·33,3) ≈ 0,0149 m³, stąd **D ≈ 246 mm, l ≈ 193 mm**.
Naprężenie styczne σ = αi·A·B = 26 kN/m², a moment T = Pi/(2πn) = 477 N·m.
Wniosek: wielkość maszyny wyznacza **moment**, a nie moc. Wolnoobrotowa maszyna tej samej mocy jest dużo większa.

### 6. Przewody w żłobkach (2.5)
Pole omija przewody w żłobkach, bo płynie zębami. Mimo to:
* **Napięcie** liczymy z Faradaya dla całej cewki. Strumień objęty cewką zmienia się liniowo, więc u = 2·Bδ·l·v.
  Przy ΘF = 2000 A i δ = 1,5 mm mamy Bδ = μ0ΘF/(2δ) = 0,84 T.
* **Siła:** prąd osłabia pole w jednym zębie (B1 = Bδ − μ0I/2δ), a wzmacnia w drugim (B2 = Bδ + μ0I/2δ). Z bilansu energii pola wychodzi F = Bδ·l·I, dokładnie jak dla przewodu w szczelinie.
  Siła działa na żelazne zęby, a nie na miedź.

### 7. Maszyna obcowzbudna (2.6)
Maszyna wzorcowa: UN = 440 V, IA,N = 50 A, RA = 0,6 Ω, nN = 1450 obr/min. Z tych danych:

* kΦN = (440 − 30)/(1450/60) = 16,97 V·s,
* TN = 135 N·m,
* n0 = 1556 obr/min,
* prąd zwarcia I_zw = U/RA = 733 A (prawie 15 × IA,N!).

Trzy sposoby regulacji prędkości:
1. **Niższe napięcie U:** n0 maleje, nachylenie charakterystyki bez zmian. Metoda bezstratna i szybka.
2. **Osłabienie pola f:** n0 rośnie, charakterystyka jest bardziej stroma, a z tego samego prądu mamy mniej momentu.
3. **Rezystor RS:** n0 bez zmian, charakterystyka bardziej stroma. Metoda stratna, stosowana do rozruchu.

Wykres T(n) pokazuje też ograniczenia: IA ≤ IA,N, P ≤ PN oraz n ≤ n_max.

### 8. Tryby pracy (Tabela 2.1)
Suwak przesuwa punkt pracy po prostej n(IA) przy U = UN.
* 0 < IA < I_zw: **silnik**.
* IA < 0: **prądnica**, np. pociąg zjeżdża z góry i oddaje energię do sieci.
* IA > I_zw: **hamowanie przeciwprądem**, cała energia idzie w ciepło.

Dwa wiersze tabeli są niemożliwe, bo oznaczałyby perpetuum mobile. Słupki potwierdzają, że zawsze P_el = P_mech + R·I².

### 9. Rozruch – symulacja (równanie 2.17 + II zasada dynamiki)
Symulacja rozwiązuje metodą Rungego-Kutty 4. rzędu dwa równania:

* LA·di/dt = U − R·i − kΦ·ω/2π,
* J·dω/dt = kΦ·i/2π − T_obc.

Rozruch bezpośredni dałby prąd rzędu 733 A. Rozrusznik 3-stopniowy ogranicza go do Imax = 100 A:
* R1 = U/Imax,
* iloraz λ = (R1/RA)^(1/m),
* przełączenie następuje, gdy prąd spadnie do Imax/λ.

Wyniki: opory 3,80 / 1,66 / 0,57 Ω, przełączenia po 0,32 / 0,49 / 0,58 s. W rozruszniku traci się ok. 8,5 kJ.

### 10. Magnesy trwałe (2.7)
Magnes opisuje prosta BM = BR + μ0·μr·HM. Na biegu jałowym BM = BR/(1 + μr·δ/hM).
Pod obciążeniem dochodzi przepływ Θ = A·αi·τp/2:
* krawędź nabiegająca: pole rośnie,
* krawędź zbiegająca: pole maleje i punkt pracy zbliża się do kolana.

Jeśli HM < H_limit, magnes jest **nieodwracalnie rozmagnesowany**.
* **Ferryt:** koercja rośnie z temperaturą, więc groźny jest mróz (−40 °C).
* **NdFeB:** koercja maleje z temperaturą, więc groźne jest gorąco. Przykład: 160 °C, hM = 3 mm, A = 300 A/cm daje rozmagnesowanie.

Środek zaradczy to grubszy magnes.

### 11. Maszyna bocznikowa – samowzbudzenie (2.8)
Magnetyzm szczątkowy daje Ur. Ur wymusza mały prąd IF, który wzmacnia pole, więc rośnie Ui i znowu IF, jak lawina. Proces kończy się tam, gdzie Ui(IF) przecina prostą (RA+RF)·IF.
Dla RF = 150 Ω wychodzi U0 ≈ 237 V. Powyżej rezystancji krytycznej (tu 250 Ω) samowzbudzenie nie zachodzi.

Pod obciążeniem napięcie spada. Przy I_max ≈ 108 A charakterystyka „zawraca” i napięcie się załamuje. Zostaje tylko prąd zwarcia od Ur (I_zw,R = Ur/RA ≈ 16 A).

### 12. Maszyna szeregowa (2.9)
Tu Φ ~ IA, więc T = L'm·IA²/f (moment rośnie z kwadratem prądu), a n = f·(U − R·IA)/(2π·L'm·IA).
* Przy IA,N: n = 1450 obr/min.
* Przy 0,1·IA,N: n ≈ 15 800 obr/min, czyli **rozbieganie**. Dlatego silnika szeregowego nie wolno uruchamiać bez obciążenia.
* Odwrócenie polaryzacji napięcia nie zmienia kierunku obrotów.

(Model pomija nasycenie, więc moment przy utyku wychodzi zawyżony.)

### 13. Porównanie (2.10)
kΦ(I) = kΦN·[(1 − s) + s·I/IN]:
* s = 0 – bocznikowa: charakterystyka sztywna,
* s = 1 – szeregowa: charakterystyka miękka, n0 = ∞,
* 0 < s < 1 – kompaundowa: skończone n0 (np. 1,77·nN przy s = 0,4) i duży moment przy obciążeniu.

### 14. Zasilanie o zmiennym napięciu (2.11)
**Układ Leonarda:** silnik AC napędza prądnicę DC, a napięcie tej prądnicy (UG ~ ΦG) zasila silnik. Działa w 4 ćwiartkach, ale wymaga trzech maszyn.
Dziś stosuje się **mostek tyrystorowy**: Ud = 1,35·U_LL·cos α. Dla U_LL = 400 V i α = 30° wychodzi Ud = 468 V.
* α > 90°: praca falownikowa (Ud < 0).
* Jeden mostek pracuje w 2 ćwiartkach, dwa mostki przeciwsobne w 4.

Wykres przewodzenia pokazuje tyrystory S1…S6. Każdy przewodzi 120°, a kolejne włączają się co 60°.

### 15. Oddziaływanie twornika (2.12)
Prąd twornika tworzy „trójkątny” przepływ ΘA prostopadły do osi biegunów. W efekcie pole się **wypacza**: pod jedną krawędzią jest silniejsze, pod drugą słabsze. Skutki:
* strefa neutralna się przesuwa,
* lokalnie rośnie napięcie między działkami,
* przy **nasyceniu** maleje średni strumień, a więc i moment,
* przy silnym osłabieniu pola pole pod krawędzią może zmienić znak.

**Uzwojenie kompensacyjne** w nabiegunnikach znosi ΘA dla każdego obciążenia.

### 16. Bieguny komutacyjne (2.13)
Czas komutacji Tc = bB/(π·DC·n) wynosi tu 0,85 ms. Indukcyjność cewki daje napięcie reaktancyjne e_r = 2LI/Tc ~ IA·n, które opóźnia zmianę prądu. Pod krawędzią zbiegającą powstaje wtedy **iskrzenie**: przy k_cp = 0 gęstość prądu jest ok. 20 razy większa od średniej.

Bieguny komutacyjne indukują e_cp ~ IA·n, czyli rosnące dokładnie jak e_r:
* k_cp = 1: komutacja liniowa, gęstość prądu 1,0,
* k_cp > 1: przekompensowanie, znów iskrzenie (rys. 2.49).

---

## Model i założenia
* Stan ustalony liczymy bez nasycenia (poza zakładkami 11 i 15) i bez strat w żelazie oraz tarcia.
* Rozkład pola pod biegunem jest trapezowy, a αi to szerokość „płaskiego dachu”.
* Magnesy mają liniową charakterystykę z kolanem H_limit ≈ −0,9·HcJ(T). Współczynniki temperaturowe pochodzą z Tabeli 2.2.
* Komutacja: prąd wymusza rezystancja styku szczotki, którą zmienia pole styku z działkami. Obliczamy go niejawną metodą Eulera. Przy L = 0 wynik jest dokładnie liniowy (sprawdza to test).
* Wykres (BH)max w funkcji roku (rys. 2.24) jest orientacyjny.

## Testy
`test_dc_machines_gerling.py` sprawdza:
* przykład Essona z książki,
* bilans mocy,
* średnie napięcie mostka 1,35·U_LL·cos α,
* komutację liniową i kompensację,
* kroki uzwojeń,
* punkt pracy magnesu,
* samowzbudzenie,
* stan ustalony po rozruchu.
