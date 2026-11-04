"""
pe_texts.py - objaśnienia wszystkich przykładów "jak dla ucznia liceum".
Klucz słownika = identyfikator zakładki.
"""

EXPL = {}

# =============================================================================
# ROZDZIAŁ 4 - MOSTEK H
# =============================================================================

EXPL["H1"] = """PRZYKŁAD 4.2 - Mostek H (full-bridge): topologia i stany przewodzenia (rys. 4.1, 4.3, 4.4)

CO TO JEST?  Mostek H to dwie "gałęzie" (nogi) połączone równolegle do źródła napięcia stałego Vdc.
Gałąź a: tranzystor górny S1 i dolny S2.  Gałąź b: górny S3 i dolny S4.  Każdy tranzystor ma
równolegle "odwrotną" diodę (D1..D4). Obciążenie (R + L, np. uzwojenie silnika) wpinamy MIĘDZY
środkami gałęzi - stąd litera "H" na schemacie.

ANALOGIA:  Wyobraź sobie dwie zwrotnice kolejowe. Każda może skierować "tor" (zacisk obciążenia)
albo do bieguna +, albo do bieguna -. Mając DWIE zwrotnice możemy przyłożyć do obciążenia:
   +Vdc  (a do +, b do -)       S1 + S4
   -Vdc  (a do -, b do +)       S2 + S3
    0    (oba do + lub oba do -) S1 + S3  lub  S2 + S4    <- NOWOŚĆ względem półmostka!

REGUŁA BEZPIECZEŃSTWA:  w jednej gałęzi NIGDY nie włączamy obu tranzystorów naraz (S1+S2 lub
S3+S4) - to zwarcie źródła DC ("shoot-through"), prąd ograniczony tylko rezystancją przewodów.

DLACZEGO DIODY?  Indukcyjność "nie lubi" zmian prądu - prąd w cewce nie może skoczyć do zera.
Gdy wyłączymy tranzystor, prąd znajduje drogę przez diodę po przeciwnej stronie gałęzi.
Dlatego to, KTÓRY element faktycznie przewodzi, zależy od dwóch rzeczy:
   1) które tranzystory są włączone,   2) w którą stronę płynie prąd obciążenia.
Tranzystor przewodzi tylko "w swoją stronę"; w przeciwną stronę prąd płynie jego diodą.

ENERGIA:  jeśli vo i prąd mają ten sam znak - źródło DC oddaje energię (napędza prąd).
Jeśli przeciwne znaki - energia wraca do źródła (cewka "oddaje" zgromadzoną energię).
Przy vo = 0 prąd krąży w mostku ("wolne koło") i powoli zanika na rezystancji.

JAK UŻYWAĆ ZAKŁADKI:  zaznacz tranzystory S1..S4 i kierunek prądu. Schemat pokaże elementy
przewodzące (czerwone), napięcie wyjściowe oraz ostrzeżenie o stanie zabronionym.
SPRÓBUJ:  włącz tylko S1 przy prądzie "b -> a" - zobaczysz, że przewodzi dioda D1, nie S1!
Policz:  2 stany na gałąź, 2 gałęzie -> 2·2 = 4 dozwolone kombinacje (z 16 możliwych).
"""

EXPL["H2"] = """PRZYKŁAD 4.2 (symulacja) - Mostek H dc-ac z BIPOLARNYM PWM (rys. 4.5, 4.6, 4.7)

IDEA PWM (modulacja szerokości impulsu):  tranzystory umieją tylko "włącz/wyłącz", a my chcemy
płynnej sinusoidy. Robimy więc bardzo szybkie impulsy, których SZEROKOŚĆ zmienia się w rytm
sinusoidy. Cewka obciążenia "uśrednia" impulsy (jak bezwładność koła zamachowego) i prąd
wychodzi prawie sinusoidalny.

ZASADA PORÓWNANIA (sinus-trójkąt):
   m(t) = ma·sin(2π f1 t)  - sygnał modulujący (kształt, który chcemy uzyskać),
   c(t)                    - nośna trójkątna -1..+1 o częstotliwości fsw (np. 2500 Hz),
   m > c  ->  S1, S4 ON  ->  vo = +Vdc;      m < c  ->  S2, S3 ON  ->  vo = -Vdc.
"Bipolarne", bo wyjście ma TYLKO dwa poziomy: +Vdc i -Vdc (stany zerowe nieużywane).

WZORY:  amplituda 1. harmonicznej napięcia  V1 = ma · Vdc  (dla ma ≤ 1)
        prąd:  L di/dt = vo - R i   ->   I1 = V1 / sqrt(R² + (2π f1 L)²)
        tętnienie prądu ~ Vdc / (L · fsw)  - większe L lub fsw -> gładszy prąd.

PRZYKŁAD (wartości z książki: Vdc = 24 V, R = 10 Ω, L = 1 mH, fsw = 2500 Hz, ma = 0,8):
   V1 = 0,8·24 = 19,2 V;  X_L = 2π·50·0,001 = 0,314 Ω;  I1 ≈ 19,2/10,005 ≈ 1,92 A.

PRĄD SZYNY DC (rys. 4.7):  idc = +iL gdy S1S4, -iL gdy S2S3. Wygląda jak sygnał z modulacją
amplitudy, a jego wartość średnia NIE jest zerowa: z zasady zachowania energii
   Vdc · Idc_śr = P_R = I_rms² · R  (+ straty).   Sprawdź to w polu "Wyniki"!

CO OBSERWOWAĆ:  napięcie skacze aż o 2·Vdc przy każdym przełączeniu (duże tętnienia),
w widmie harmoniczne są skupione wokół fsw/f1 = 50 (np. 48, 50, 52...).
SPRÓBUJ:  zmniejsz L do 0,2 mH - tętnienie rośnie; podnieś fsw - maleje.
"""

EXPL["H3"] = """PRZYKŁAD 4.3 - Wektorowa reprezentacja napięcia mostka H (rys. 4.8, 4.9)

Mostek H może wytworzyć tylko trzy napięcia: +Vdc, 0, -Vdc. Nazywamy je "WEKTORAMI" napięcia
(na razie leżą na jednej prostej: -1, 0, +1). Wektor zerowy ma DWIE realizacje: S1+S3 lub S2+S4.

PODSTAWOWE RÓWNANIE (średnia w jednym okresie przełączania Tsw):
       (+Vdc)·T+  +  0·T0  +  (-Vdc)·T-  =  Vreq · Tsw,        T+ + T0 + T- = Tsw
To jak średnia ocen ważona czasem: chcemy, by średnie napięcie w okresie było równe zadanemu.

TRZY STRATEGIE (rys. 4.9):
 * BIPOLARNE   - T0 = 0, używamy +Vdc i -Vdc:  T+ = (1 + Vreq/Vdc)/2 · Tsw.
 * UNIPOLARNE  - używamy tylko wektora o znaku Vreq i zera (T- = 0 gdy Vreq > 0):
                 T+ = Vreq/Vdc · Tsw, T0 dzielimy na dwie połówki T0/2 (symetria).
 * PHASE-SHIFT - T+ = T- (średnia ZERO!), zero dzielone na trzy przedziały. Stosowane z
                 transformatorem wysokiej częstotliwości (nie można mu podać składowej stałej).

DLACZEGO UNIPOLARNE JEST LEPSZE?  Przy każdym przełączeniu napięcie zmienia się tylko o Vdc
(a nie o 2·Vdc), a przy przejściu do stanu zerowego zmienia stan tylko JEDNA gałąź - mniej strat
łączeniowych i dwa razy mniejsze tętnienie prądu.

STEROWANIE CYFROWE:  mikrokontroler co okres Tsw dostaje z regulatora nowe Vreq, liczy czasy
T+, T0, T- i ustawia liczniki PWM. Równanie u góry to "serce" każdej modulacji wektorowej -
w rozdziale 5 zobaczymy je w wersji dwuwymiarowej (SVPWM).

PRZYKŁAD:  Vdc = 24 V, Vreq = 6 V, Tsw = 200 µs:
   bipolarne:   T+ = (1+0,25)/2·200 = 125 µs, T- = 75 µs  (125·24 - 75·24)/200 = 6 V ✓
   unipolarne:  T+ = 0,25·200 = 50 µs, T0 = 150 µs        (50·24)/200 = 6 V ✓
SPRÓBUJ:  ustaw Vreq ujemne i porównaj wzory impulsów.
"""

EXPL["H4"] = """PRZYKŁAD 4.4 - Mostek H jako przekształtnik dc-dc z UNIPOLARNYM PWM (4 ćwiartki)

ZADANIE:  z napięcia Vdc otrzymać regulowane napięcie stałe dowolnego znaku (np. dla silnika DC
- jazda w przód i w tył, napędzanie i hamowanie = 4 ćwiartki).

NOŚNE (piłokształtne, bo dla dc symetria nie jest potrzebna):
   c1(t) - piła 0..1,   c2(t) = -c1(t) - piła 0..-1,    d - współczynnik wypełnienia -1..+1
   d > c2  ->  S1 ON (inaczej S2)          d < c1  ->  S3 ON (inaczej S4)
Wynik (sprawdź na wykresie!):
   d > 0:  S1 stale ON, S3/S4 przełączają  ->  impulsy +Vdc i 0,  średnio  Vo = d·Vdc
   d < 0:  S3 stale ON, S1/S2 przełączają  ->  impulsy -Vdc i 0,  średnio  Vo = d·Vdc
Jedna reguła obsługuje oba znaki - to elegancja metody z książki: zaczynamy od pożądanego
przebiegu napięcia, rysujemy stany przewodzenia, a potem dobieramy nośne.

OBCIĄŻENIE R-L-E (np. silnik z SEM E):
   L di/dt = vo - R i - E,     średnio  I = (d·Vdc - E)/R.
Tętnienie prądu (d > 0):  ΔI ≈ (Vdc - E)·d·Tsw / L  (dla L/R >> Tsw).

PRZYKŁAD:  Vdc = 24 V, d = 0,5, E = 0, R = 2 Ω, L = 2 mH, fsw = 5 kHz:
   Vo = 12 V,  I = 6 A,  ΔI ≈ 12·0,5·200e-6/2e-3 ≈ 0,6 A (szczyt-szczyt).
SPRÓBUJ:  ustaw E = 15 V i d = 0,5: Vo < E, więc prąd jest UJEMNY - silnik oddaje energię
(hamowanie odzyskowe). To ćwiartka II: napięcie +, prąd -.
"""

EXPL["H5"] = """PRZYKŁAD 4.4 (symulacja) - Dwa mostki H: LC i RC połączone dławikiem (rys. 4.10-4.18)

UKŁAD:  lewy przekształtnik (LC, np. przy baterii) i prawy (RC, np. przy silniku), oba z
Vdc = 24 V, połączone indukcyjnością L i rezystancją R. Każdy ma własne d (d1, d2).
   L di/dt = v_brg1 - v_brg2 - R i,     średnio  I = (d1·Vdc1 - d2·Vdc2)/R.
Prąd dodatni = energia płynie z LC do RC.

PRZYPADEK 1 (książka: d1 = 0,5, d2 = 0,3):  obie średnie dodatnie, LC "silniejszy":
   I = (12 - 7,2)/R  > 0 - LC zasila RC. W RC tranzystory są "włączone", ale przewodzą DIODY
   (prąd płynie w przeciwną stronę niż "lubią" tranzystory) - RC działa jak prostownik.
PRZYPADEK 2 (d1 < 0, np. -0,5):  LC daje ujemne, RC dodatnie napięcie - na dławiku pojawia się
   aż -2·Vdc w części okresu, prąd ma DUŻĄ wartość ujemną (płynie z RC do LC). Tak wygląda
   gwałtowne hamowanie odzyskowe: silnik oddaje energię do baterii. W praktyce regulator
   musi ograniczyć ten prąd!

TRZY NACHYLENIA PRĄDU:  w przypadku 2 napięcie na dławiku przyjmuje wartości -2Vdc, -Vdc i 0,
więc w jednym okresie prąd ma trzy różne nachylenia di/dt (sprawdź na powiększeniu).
"OFFSET" NAPIĘĆ:  zmierzone napięcia mostków zawierają spadek na R - stąd przesunięcie.

ANALOGIA:  dwa zbiorniki wody połączone rurą - woda płynie od wyższego "ciśnienia" (średniego
napięcia) do niższego, a rura (R) ogranicza przepływ.
SPRÓBUJ:  przesuń nośną RC (shift) - średni prąd się nie zmienia, ale tętnienie tak!
Ustaw d1 = d2 - prąd średni 0, płyną tylko tętnienia.
"""

EXPL["H6"] = """PRZYKŁAD 4.5 - Mostek H dc-ac z UNIPOLARNYM PWM (3 poziomy, rys. 4.19-4.21)

NOŚNE:  c1(t) - trójkąt 0..1,  c2(t) = -c1(t) - trójkąt -1..0.
   m > c2 -> S1 ON (inaczej S2);    m > c1 -> S4 ON,  m < c1 -> S3 ON.
Dodatni półokres:  S1 stale ON, S3/S4 przełączają -> +Vdc (S1-S4) lub 0 (S1-S3).
Ujemny półokres:   S3 stale ON, S1/S2 przełączają -> -Vdc (S2-S3) lub 0 (S1-S3).
Wyjście ma TRZY poziomy: +Vdc, 0, -Vdc  -> "trójpoziomowe sinusoidalne PWM".

PORÓWNANIE Z BIPOLARNYM (dolny wykres!):
 * skok napięcia przy przełączeniu: Vdc zamiast 2·Vdc,
 * tętnienie prądu około 2× mniejsze przy tej samej fsw,
 * w widmie znikają harmoniczne wokół fsw - pierwsze grupy są dopiero przy 2·fsw
   ("efektywnie podwojona częstotliwość"),
 * źródło DC tylko NAPĘDZA prąd (w stanie 0 prąd krąży w mostku), mniej strat.

DLACZEGO TO WAŻNE?  Im bliżej przebieg napięcia jest sinusoidy, tym mniejszy (tańszy, lżejszy)
filtr L lub LC jest potrzebny. To główna myśl rozdziału 4.6 książki: wykorzystać WSZYSTKIE
dostępne poziomy napięcia przekształtnika.

PRZYKŁAD:  Vdc = 24 V, ma = 0,8, R = 10 Ω, L = 1 mH, fsw = 2500 Hz:
   V1 = 19,2 V w obu metodach, ale tętnienie prądu i THD napięcia wyraźnie mniejsze dla unipolarnej.
SPRÓBUJ:  porównaj THD w polu "Wyniki" przy ma = 0,3 i ma = 0,95.
"""

EXPL["H7"] = """PRZYKŁAD 4.5 - Modulacja z przesunięciem fazowym (phase-shift PWM, rys. 4.22-4.24)

GDZIE?  Przekształtniki dc-dc z transformatorem wysokiej częstotliwości (zasilacze, ładowarki,
przekształtniki DAB). Transformator NIE MOŻE dostać napięcia o niezerowej średniej - rdzeń by się
nasycił (prąd magnesujący rośnie bez końca, jak przy podaniu napięcia stałego na uzwojenie).

ZASADA:  każda gałąź pracuje z wypełnieniem dokładnie 50 % (d1 = d2 = 0,5). Regulujemy tylko
PRZESUNIĘCIE w czasie (fazę) między gałęziami. Nośne:
   c1(t) - piła 0..1,   c2(t) = -piła przesunięta o d·Tsw
   -0,5 > c2 -> S1 ON;   +0,5 < c1 -> S3 ON.
Napięcie mostka: w każdym okresie jeden impuls +Vdc i jeden -Vdc, każdy o długości d·Tsw,
rozdzielone wektorami zerowymi. Średnia = 0 zawsze!  (wymagane d ≤ 0,5)

WZORY:
   wartość skuteczna   Vrms = Vdc·sqrt(2d)
   1. harmoniczna (fsw)  V1 = (4/π)·Vdc·sin(π d)
   moc przenoszona przez transformator rośnie z d.

PRZYKŁAD (książka: d = 0,2, Vdc = 24 V):
   Vrms = 24·sqrt(0,4) = 15,2 V;  V1 = 1,273·24·sin(36°) = 17,96 V.
SPRÓBUJ:  zwiększaj d od 0 do 0,5 - przy d = 0,5 dostajemy przebieg prostokątny ±Vdc
(maksymalna moc). Zwróć uwagę, że średnia prądu też jest zerowa.
"""

# =============================================================================
# ROZDZIAŁ 5 - UKŁADY TRÓJFAZOWE
# =============================================================================

EXPL["T1"] = """PRZYKŁAD 5.2 - Układy trójfazowe, gwiazda i trójkąt (rys. 5.1-5.3)

SKĄD TRZY FAZY?  W prądnicy wirujący magnes mija trzy uzwojenia rozmieszczone co 120°. W każdym
indukuje się sinusoida, ale z opóźnieniem o 1/3 okresu:
   va = Vm cos(ωt),  vb = Vm cos(ωt - 120°),  vc = Vm cos(ωt + 120°).
Suma va + vb + vc = 0 w KAŻDEJ chwili (dla układu symetrycznego) - dlatego w gwieździe bez
przewodu neutralnego nie potrzebujemy "powrotu" prądu.

WEKTORY WIRUJĄCE (rys. 5.3):  każde napięcie to rzut na oś poziomą strzałki o długości Vm,
obracającej się z prędkością kątową ω = 2π f.  Trzy strzałki tworzą "gwiazdkę" co 120°.

NAPIĘCIA MIĘDZYPRZEWODOWE:  vab = va - vb itd. Są sqrt(3) razy większe i wyprzedzają o 30°:
   V_LL = sqrt(3)·V_ph     (230 V fazowe -> 400 V międzyprzewodowe w Europie).

POŁĄCZENIA (rys. 5.2):
   gwiazda z izolowanym punktem neutralnym - 3 przewody, I_linii = I_fazy,
   trójkąt - napięcie fazy = napięcie międzyprzewodowe, I_linii = sqrt(3)·I_fazy,
   gwiazda z przewodem neutralnym - 4 przewody, pozwala zasilać odbiorniki jednofazowe.

NIESYMETRIA:  gdy amplitudy lub kąty się różnią, suma ≠ 0 i w przewodzie neutralnym płynie prąd.
SPRÓBUJ:  zmień kąt ωt suwakiem i zobacz, jak rzuty strzałek dają chwilowe wartości napięć.
Wprowadź niesymetrię amplitudy fazy c i obserwuj prąd/napięcie punktu neutralnego.
"""

EXPL["T2"] = """PRZYKŁAD 5.3 - Falownik trójgałęziowy z sinusoidalnym PWM (rys. 5.4-5.8)

UKŁAD:  trzy gałęzie (jak w mostku H, ale trzy), obciążenie R-L w gwiazdę z IZOLOWANYM punktem
neutralnym n. Książka: Vdc = 24 V, R = 10 Ω, L = 1 mH, fsw = 2500 Hz.
Każda gałąź ma 2 stany -> 2³ = 8 kombinacji (rys. 5.5). Zabronione: oba tranzystory w gałęzi.

PWM:  jedna nośna trójkątna, trzy sygnały modulujące przesunięte o 120°:
   ma = ma·cos(ωt), mb = ma·cos(ωt-120°), mc = ma·cos(ωt+120°);   m_x > c -> górny tranzystor ON.

NAPIĘCIE PUNKTU NEUTRALNEGO (równania pętli z rozdz. 5.3):
   v_nN = (v_aN + v_bN + v_cN)/3,        v_an = v_aN - v_nN   itd.
Dlatego napięcie fazowe obciążenia ma 5 poziomów: 0, ±Vdc/3, ±2Vdc/3 (rys. 5.8) - choć każda
gałąź ma tylko 0 lub Vdc!

WZORY (ma ≤ 1):  amplituda napięcia fazowego  V1 = ma·Vdc/2
                 maksimum liniowe SPWM:  V1max = Vdc/2 = 12 V dla 24 V.
PRZYKŁAD:  ma = 0,8:  V1 = 9,6 V,  |Z| = sqrt(10² + 0,314²) ≈ 10 Ω  ->  I1 ≈ 0,96 A.

CO OBSERWOWAĆ:  prądy są sinusoidalne z "ząbkami" od przełączeń (L = 1 mH to mało: L/R = 0,1 ms
< Tsw = 0,4 ms). Suma trzech prądów = 0.
SPRÓBUJ:  zwiększ L do 10 mH - prąd staje się gładki. Ustaw ma > 1 - przemodulowanie, pojawiają
się harmoniczne niskiego rzędu (5., 7.).
"""

EXPL["T3"] = """PRZYKŁAD 5.4 - Transformacja Clarke'a (rys. 5.9-5.11)

PROBLEM:  trzy sinusoidy to sporo do śledzenia. Clarke zamienia je na DWIE składowe
prostopadłe (α, β) + składową zerową (0):
   vα = 2/3·(va - vb/2 - vc/2)
   vβ = (vb - vc)/sqrt(3)
   v0 = (va + vb + vc)/3
Dla układu symetrycznego: vα = Vm cos ωt, vβ = Vm sin ωt, v0 = 0.
Łączymy je w jedną liczbę zespoloną  V = vα + j·vβ  - jeden WEKTOR wirujący po okręgu o
promieniu Vm (współczynnik 2/3 zachowuje amplitudę - wersja "amplitudowo niezmiennicza").

SENS FIZYCZNY (rys. 5.11):  trzy uzwojenia przesunięte o 120° wytwarzają to samo wirujące pole
magnetyczne co DWA uzwojenia prostopadłe zasilane cos i sin. Silnik "czuje" tylko wektor wypadkowy.

PRZYKŁAD (książka):  Vm = 0,3, ωt = 30°:  V = 0,3·(cos30° + j sin30°) = 0,26 + j0,15;
                     ωt = 260°:  V = -0,052 - j0,295.

CO OBSERWOWAĆ:
 * symetria -> trajektoria to OKRĄG,
 * niesymetria amplitud lub kątów -> ELIPSA (składowa przeciwna wiruje w drugą stronę),
 * 5. harmoniczna -> "kwiatek" z 6 płatkami (5. harmoniczna wiruje przeciwnie),
 * składowa wspólna (offset) -> tylko v0, okrąg się NIE zmienia - dlatego w gwieździe bez
   przewodu neutralnego składowa zerowa nie wpływa na prądy.
SPRÓBUJ:  dodaj offset - zobaczysz, że α i β się nie zmieniają (to podstawa SVPWM!).
"""

EXPL["T4"] = """PRZYKŁAD 5.5 - Wektory napięcia przekształtnika trójgałęziowego (Tabele 5.1, 5.2, rys. 5.12-5.13)

Każdy z 8 stanów (a, b, c) - 1 = górny tranzystor ON - daje inne napięcia. Liczymy:
   v_nN = Vdc·(a + b + c)/3,   v_an = Vdc·a - v_nN  itd.
i robimy transformację Clarke'a. Wynik: 6 wektorów aktywnych o długości 2/3·Vdc tworzących
SZEŚCIOKĄT foremny oraz 2 wektory zerowe (V0 = 000 i V7 = 111) w środku.

NUMERACJA (jak w książce):  V1 = 100 (0°), V6 = 110 (60°), V2 = 010 (120°), V4 = 011 (180°),
V3 = 001 (240°), V5 = 101 (300°).  Sąsiednie wektory różnią się stanem JEDNEJ gałęzi.
Wektory napięć międzyprzewodowych tworzą sześciokąt obrócony o 30° i sqrt(3) razy większy
(długość 2/sqrt(3)·Vdc ≈ 1,155·Vdc).

PRZYKŁAD (Vdc = 1):  stan V1 = (1,0,0):  v_nN = 1/3,  v_an = 2/3, v_bn = v_cn = -1/3
   vα = 2/3·(2/3 + 1/6 + 1/6) = 2/3,  vβ = 0   ->  V1 = 0,667∠0°.
Stan V6 = (1,1,0): v_an = v_bn = 1/3, v_cn = -2/3 -> V6 = 0,667∠60°.

WNIOSEK:  przekształtnik umie wytworzyć tylko 7 różnych "punktów" na płaszczyźnie. Każdy inny
wektor uzyskamy ŚREDNIO w czasie, przeplatając sąsiednie wektory - to idea SVPWM.
SPRÓBUJ:  wybierz stany po kolei 1 -> 6 -> 2 -> 4 -> 3 -> 5 - wektor obiega sześciokąt, a przy
każdym kroku zmienia się tylko jedna gałąź.
"""

EXPL["T5"] = """PRZYKŁAD 5.6 - Algorytm SVPWM (modulacja wektorowa, rys. 5.14-5.15, Tabela 5.3)

KROK 1 - wektor zadany:  z napięć zadanych (Clarke) dostajemy Vr = vrα + j·vrβ.
KROK 2 - sektor (rys. 5.15) - bez liczenia arcus tangensa, tylko porównania z tan60° = 1,732:
   vrα > 0, vrβ > 0:  vrβ ≤ 1,732·vrα -> (V1, V6), inaczej (V6, V2)
   vrα > 0, vrβ < 0:  vrβ ≥ -1,732·vrα -> (V5, V1), inaczej (V3, V5)   ... itd.
KROK 3 - czasy:   Vr·T = Vx·t1 + Vy·t2 + V0·t0      (średnia ważona czasem, jak w rozdz. 4!)
   rozdzielamy na część rzeczywistą i urojoną -> 2 równania, 2 niewiadome (t1, t2):
   t0 = T - t1 - t2   (dzielone po równo między V0 i V7).
KROK 4 - sekwencja (Tabela 5.3):  V0 - Vx - Vy - V7 - V7 - Vy - Vx - V0  z czasami
   t0/4, t1/2, t2/2, t0/4 | t0/4, t2/2, t1/2, t0/4.  Każdy krok = zmiana JEDNEJ gałęzi
   (minimum strat), a przebieg jest symetryczny (mniej harmonicznych).
   Vx = wektor z jedną "jedynką" (bliżej V0), np. dla (V6, V2) najpierw V2 (010), potem V6 (110).

PRZYKŁAD Z KSIĄŻKI:  |Vr| = 0,3, ωt = 30°  ->  Vr = 0,26 + j0,15, para (V1, V6):
   t1 = t2 = 0,26·T,   t0 = 0,48·T.
PRZYKŁAD GRANICZNY:  |Vr| = 2/3 (okrąg przez wierzchołki), 30°:  t1 = t2 = 0,577·T,
   t1 + t2 = 1,155·T > T  ->  NIEMOŻLIWE (przemodulowanie, rys. 5.19).
MAKSIMUM liniowe = promień okręgu wpisanego w sześciokąt = 2/3·cos30° = 1/sqrt(3) ≈ 0,577·Vdc
(o 15,5 % więcej niż SPWM: 0,5·Vdc).
SPRÓBUJ:  ustaw |Vr| = 0,6 - pojawi się ostrzeżenie o przemodulowaniu.
"""

EXPL["T6"] = """PRZYKŁAD 5.7 - Symulacja SVPWM i granice przekształtnika (rys. 5.16-5.19)

Ten sam falownik co w 5.3 (24 V, 10 Ω, 1 mH, 2500 Hz), ale impulsy liczone algorytmem SVPWM:
w każdym okresie przełączania (próbkowanie regularne) wyznaczamy sektor, t1, t2, t0 i sekwencję
z Tabeli 5.3. Wynik dla gałęzi: wyśrodkowany impuls o szerokości d_x·T.

PORÓWNANIE Z SPWM:
 * SPWM:   max amplituda napięcia fazowego = Vdc/2 = 0,5·Vdc
 * SVPWM:  max = Vdc/sqrt(3) = 0,577·Vdc   (+15,5 %)
   bo SVPWM "dokłada" do napięć gałęzi wspólną składową (zerową), której obciążenie w gwiazdzie
   z izolowanym punktem neutralnym nie widzi (patrz Clarke!) - sygnały modulujące gałęzi mają
   kształt "siodła".
 * maksymalne napięcie skuteczne fazowe: 0,577/sqrt(2) = 0,408·Vdc (9,8 V dla 24 V),
   międzyprzewodowe: 0,707·Vdc.  Do sieci 400 V potrzeba więc Vdc ≥ 400/0,707 ≈ 566 V.

CO OBSERWOWAĆ:  prądy, impulsy gałęzi, 5-poziomowe napięcie fazowe, trajektoria prądu α-β
(prawie okrąg) i wypełnienia gałęzi d_a (kształt "siodła").
SPRÓBUJ:  ustaw amplitudę 0,55·Vdc: w SPWM pojawia się przemodulowanie (zniekształcenie,
wzrost THD), a w SVPWM wszystko jest jeszcze liniowe. Powyżej 0,577 obie metody się nasycają.
"""

# =============================================================================
# ROZDZIAŁ 6 - WIELOPOZIOMOWE
# =============================================================================

EXPL["N1"] = """PRZYKŁAD 6.3-6.4 - Gałąź 3-poziomowa NPC (Neutral Point Clamped, rys. 6.2, Tabele 6.1-6.2)

PO CO?  W dużych mocach (MW, kilka kV) pojedynczy tranzystor nie wytrzyma pełnego napięcia.
Dzielimy szynę DC dwoma kondensatorami na dwie połówki Vdc/2 i łączymy w szereg 4 tranzystory
S1..S4. Diody clampujące Dm1, Dm2 łączą środek gałęzi z punktem środkowym M kondensatorów.
Każdy tranzystor "widzi" tylko Vdc/2!

POZIOMY WYJŚCIA (Tabela 6.1):
   S1 S2 ON          -> v_oN = Vdc     (zacisk do P)
   S2 S3 ON          -> v_oN = Vdc/2   (zacisk do M przez Dm1 lub Dm2)
   S3 S4 ON          -> v_oN = 0       (zacisk do N)
ZAWSZE włączone są DWA SĄSIEDNIE tranzystory (S1 i S3 działają komplementarnie, tak samo S2 i S4).

STANY ZABRONIONE:
   S1 S2 S3 ON  -> zwarcie górnego kondensatora (droga S1-S2-S3-Dm2),
   S2 S3 S4 ON  -> zwarcie dolnego kondensatora (Dm1-S2-S3-S4),
   S1..S4 ON    -> zwarcie całego źródła.
Wniosek: więcej niż 2 włączone tranzystory = zwarcie.
STANY NIEZAMIERZONE:  gdy włączony jest tylko jeden tranzystor (lub żaden), wynik zależy od
kierunku prądu - prąd "sam" znajdzie drogę przez diody do P lub N.

4 POZIOMY (Tabela 6.2):  3 kondensatory, 6 tranzystorów, zawsze 3 sąsiednie ON: 0, Vdc/3,
2Vdc/3, Vdc. Ogólnie k poziomów: k-1 kondensatorów, 2(k-1) tranzystorów.
SPRÓBUJ:  włącz S2 i S3, zmieniaj kierunek prądu - zobacz, że raz przewodzi Dm1+S2, a raz S3+Dm2,
ale wyjście jest zawsze Vdc/2. Potem włącz S1, S2, S3 - zobaczysz ostrzeżenie o zwarciu.
"""

EXPL["N2"] = """PRZYKŁAD 6.4 - Mapa wektorów przekształtnika 3- i 4-poziomowego (Tabela 6.3, rys. 6.5-6.6)

LICZBA STANÓW:  każda gałąź ma k poziomów, gałęzie są niezależne  ->  k³ stanów:
   2 poziomy: 8,   3 poziomy: 27,   4 poziomy: 64.
Numeracja jak w Tabeli 6.3:  numer = 9·La + 3·Lb + Lc  (L = 0, 1, 2 - poziom fazy).

WEKTORY (Clarke napięć fazowych, Vdc = 1):
 3 poziomy:  19 różnych punktów:
   - 1 zerowy (realizowany 3 sposobami: V0, V13, V26),
   - 6 MAŁYCH (wewnętrzny sześciokąt, długość 1/3) - każdy na 2 sposoby (redundancja!),
   - 6 ŚREDNICH (długość 1/sqrt(3) = 0,577), 6 DUŻYCH (długość 2/3).
 4 poziomy:  37 punktów, wektory wewnętrzne mają 3 lub 2 realizacje.
PRZYKŁAD:  V12 = (1,1,0) i V25 = (2,2,1) to TEN SAM wektor 0,1667 + j0,2887
           (różnią się tylko składową wspólną, której obciążenie nie widzi).

DLACZEGO TO WAŻNE?
 * więcej punktów = wektor zadany można przybliżać bliższymi wektorami -> mniejsze tętnienia,
 * redundancja (kilka stanów na jeden wektor) daje swobodę wyboru, który kondensator
   obciążamy - klucz do balansowania napięć kondensatorów (rozdz. 6.6),
 * zdolność napięciowa: maks. sinusoidalny wektor = 0,577·Vdc - TAKA SAMA jak dla 2 poziomów
   (o napięciu decyduje Vdc), ale jakość przebiegu jest dużo lepsza.
SPRÓBUJ:  wybierz wektor suwakiem i zobacz listę jego redundantnych realizacji.
"""

EXPL["N3"] = """PRZYKŁAD 6.5-6.6 - SVPWM dla przekształtnika 3-poziomowego (rys. 6.7-6.11)

ZASADA:  mapę wektorów dzielimy na małe TRÓJKĄTY RÓWNOBOCZNE. Wektor zadany Vr leży w jednym
z nich - używamy jego trzech wierzchołków (najbliższe trzy wektory). Dlaczego równoboczne?
Ich wierzchołki różnią się JEDNYM przełączeniem jednej fazy - minimum strat.

CZASY - sztuczka z "przesunięciem" (rozdz. 6.6):  w zewnętrznych trójkątach nie ma wektora
zerowego. Równanie:   Vr·T = Vz·t0 + Vx·t1 + Vy·t2,   t0 + t1 + t2 = T.
Definiujemy  Vr' = Vr - Vz,  Vx' = Vx - Vz,  Vy' = Vy - Vz  (przesuwamy trójkąt do początku
układu) i dostajemy  Vr'·T = Vx'·t1 + Vy'·t2  - DOKŁADNIE te same wzory co w rozdz. 5!

SEKWENCJA:  pierwszy i ostatni stan to dwie realizacje TEGO SAMEGO wektora Vz (redundancja);
kolejne stany różnią się zmianą jednej fazy o jeden poziom. Przykład z książki (trójkąt
V9/V22 - V12/V25 - V21):
   V9 - V12 - V21 - V22   (obciąża bardziej DOLNY kondensator: V9 = (1,0,0) korzysta tylko z niego)
   V12 - V21 - V22 - V25  (czytany wstecz: V25-V22-V21-V12 - obciąża bardziej GÓRNY)
Oba dają to samo napięcie wyjściowe - wybór decyduje o ładowaniu/rozładowaniu kondensatorów.

JAK TO LICZYMY W PROGRAMIE?  Napięcia faz w "jednostkach poziomu" u_x ∈ [0, 2], przesunięte
o wspólne δ. Część całkowita = stan początkowy, części ułamkowe (posortowane) = czasy.
Zmiana δ = wybór innej redundantnej sekwencji (to samo napięcie fazowe obciążenia!).
PRZYKŁAD:  |Vr| = 0,2 - wewnętrzny sześciokąt (jak 2 poziomy); |Vr| = 0,5 - zewnętrzne trójkąty.
SPRÓBUJ:  przełącz "dolny/górny kondensator" i zobacz, jak zmienia się sekwencja i przebieg
poziomów, a czasy t1, t2 zostają te same.
"""

EXPL["N4"] = """PRZYKŁAD 6.6-6.7 - Balansowanie kondensatorów i symulacja 3-poziomowego NPC (rys. 6.12-6.14)

PROBLEM:  gdy faza jest podłączona do punktu M, jej prąd wpływa/wypływa z punktu środkowego
kondensatorów. Model (źródło Vdc idealne):
   v_góra + v_dół = Vdc,    vc = v_góra - v_dół,    dvc/dt = i_M / C
(i_M - suma prądów faz podłączonych do M). Jeśli zawsze wybieramy sekwencje "dolne", dolny
kondensator się rozładowuje i vc ucieka - napięcie wyjściowe się psuje (tryb "brak").

ALGORYTM Z KSIĄŻKI (histereza):  pasmo ±vc_max (np. 2 % Vdc).
   vc > +vc_max  (dolny rozładowany) -> wybieraj sekwencje obciążające GÓRNY kondensator,
   vc < -vc_max                      -> wybieraj sekwencje obciążające DOLNY.
Jak termostat: grzałka włącza się poniżej progu, wyłącza powyżej.
INNE METODY (dla porównania):
   "pred" - przewiduje i_M dla obu sekwencji (ze zmierzonych prądów) i wybiera tę, która
            zmniejsza |vc|  (lepiej - mniejsze tętnienie vc),
   "PI"   - płynna zmiana przesunięcia δ proporcjonalnie do vc (patrz zakładka C4).

WYNIKI SYMULACJI:  napięcie fazowe ma 9 poziomów (gałąź: 3 poziomy -> 0, Vdc/2, Vdc),
prądy mają DUŻO mniejsze tętnienie niż w falowniku 2-poziomowym przy tej samej fsw
(skoki napięcia są o połowę mniejsze), vc pozostaje w paśmie histerezy.
PRZYKŁAD:  Vdc = 24 V, C = 2200 µF, i_M ≈ 1 A:  dvc/dt ≈ 455 V/s -> w okresie 0,4 ms vc zmienia
się o ~0,18 V, więc pasmo ±0,48 V (2 %) jest utrzymywane.
SPRÓBUJ:  ustaw tryb "brak" - vc rośnie aż do Vdc (dolny kondensator pusty!). Zmniejsz C -
tętnienie vc rośnie. Zwiększ początkową niesymetrię vc0 i obserwuj, jak regulator ją usuwa.
"""

# =============================================================================
# UKŁADY STEROWANIA
# =============================================================================

CONTROL_INTRO = """JAK CZYTAĆ WYKRESY ANALIZY STEROWANIA (wspólne dla zakładek C1-C4)

L(s) = C(s)·P(s) - transmitancja pętli otwartej (regulator × obiekt). Pętla zamknięta:
   T(s) = L/(1+L) - jak wyjście śledzi wartość zadaną;  Sd(s) = P/(1+L) - wpływ zakłócenia.

ZPK (zera-bieguny-wzmocnienie):  G(s) = k·(s - z1)(s - z2).../((s - p1)(s - p2)...)
   BIEGUNY = "naturalne rytmy" układu. Część rzeczywista < 0 -> drgania gasną (STABILNY);
   > 0 -> rosną (niestabilny). Część urojona ≠ 0 -> oscylacje; im bliżej osi urojonej, tym słabiej
   tłumione. ZERA kształtują odpowiedź (zero w prawej półpłaszczyźnie - od opóźnienia Padé -
   powoduje chwilowy "start w złą stronę").
BODE:  |L(jω)| w dB i faza w funkcji częstotliwości.
   fc (ωc) - częstotliwość odcięcia, gdzie |L| = 1 (0 dB) ~ PASMO regulatora (szybkość).
   PM - zapas fazy = 180° + faza(L) przy fc.  Ile fazy "brakuje" do niestabilności.
   GM - zapas wzmocnienia = -|L| [dB] tam, gdzie faza = -180°. Ile razy wolno zwiększyć wzmocnienie.
NYQUIST:  ta sama L(jω), ale narysowana na płaszczyźnie zespolonej. Krzywa NIE może okrążać
   punktu -1 (dla obiektu stabilnego). Odległość od -1 = "margines bezpieczeństwa".
STEP:  odpowiedź na skok wartości zadanej:  czas narastania tr (10-90 %), przeregulowanie OS,
   czas ustalania ts (±2 %). Drugi wykres - skok zakłócenia (jak szybko regulator je "zjada").
Reguła kciuka:  PM 45-65° -> OS ≈ 5-20 %;  PM < 30° -> silne oscylacje;  fc ≤ fsw/10.
PRZEMIATANIE:  zakładka "Przemiatanie parametru" pokazuje, jak PM, GM, fc i przeregulowanie
   zmieniają się w całym zakresie WYBRANEGO parametru - analiza wrażliwości.
"""

EXPL["C1"] = """UKŁAD STEROWANIA C1 - Regulator PI prądu mostka H (obciążenie R-L)

PO CO?  W rozdziale 4 książka zakłada, że "regulator w pętli zamkniętej" dostarcza co okres
napięcie zadane Vreq. Tu projektujemy ten regulator: mierzymy prąd i, porównujemy z zadanym i*,
a regulator PI wylicza napięcie, które mostek ma wytworzyć (modulator PWM ma wzmocnienie ≈ 1).

MODEL OBIEKTU:   P(s) = I(s)/V(s) = 1/(L s + R) · e^(-s·Td)
   biegun obiektu  p = -R/L  (stała czasowa τ = L/R),
   Td - opóźnienie cyfrowe: obliczenia (1 okres) + PWM (pół okresu) ≈ 1,5·Tsw.
REGULATOR:  C(s) = Kp + Ki/s = Kp (s + Ki/Kp)/s   - zero w -Ki/Kp, biegun w 0 (całkowanie
   usuwa uchyb ustalony: prąd dokładnie równa się zadanemu).
STROJENIE (przycisk "Dobierz"):  "kompensacja bieguna" - zero regulatora skraca biegun obiektu:
   Ki/Kp = R/L  ->  Kp = ωc·L,  Ki = ωc·R   =>  L(s) = ωc/s · e^(-sTd)
   Wtedy  PM = 90° - ωc·Td·180°/π.  Np. fc = 1 kHz, fsw = 10 kHz (Td = 150 µs):
   PM = 90 - 360·1000·150e-6 = 90 - 54 = 36°  (opóźnienie zjada fazę!).
   Dla PM = 60° potrzeba fc ≈ 555 Hz  ->  prawie fsw/18.

WNIOSKI DLA UCZNIA:
 * Opóźnienie jest głównym wrogiem szybkiej regulacji: każdy µs opóźnienia przy częstotliwości f
   odbiera 360°·f·Td fazy.
 * Za duże Kp -> fc rośnie -> PM spada -> przeregulowanie, a potem niestabilność (GM < 0).
 * Ki za duże względem R/L -> wolny, "pełzający" ogon lub oscylacje; za małe -> powolne
   dochodzenie do wartości zadanej.
 * Podwójne próbkowanie (Td = 0,75·Tsw) pozwala podwoić pasmo przy tym samym PM.
SPRÓBUJ:  zwiększaj Kp i obserwuj, jak krzywa Nyquista zbliża się do -1, a w odpowiedzi skokowej
rośnie przeregulowanie. W "Przemiataniu" wybierz fsw - zobaczysz, jak PM rośnie z fsw.
"""

EXPL["C2"] = """UKŁAD STEROWANIA C2 - Regulator PR prądu sinusoidalnego (mostek H podłączony do sieci)

PROBLEM:  w trybie dc-ac (rozdz. 4.5) zadany prąd jest SINUSOIDĄ 50 Hz. Zwykły PI ma nieskończone
wzmocnienie tylko dla prądu stałego (ω = 0), więc sinusoidę śledzi z błędem amplitudy i fazy.
Sieć (napięcie e) działa jak zakłócenie.

REGULATOR PR (proporcjonalno-rezonansowy):
   C(s) = Kp + Kr · 2ωc·s / (s² + 2ωc·s + ω0²),     ω0 = 2π·50 Hz
   Człon rezonansowy to "całkowanie dla sinusoidy": bardzo duże wzmocnienie dokładnie przy 50 Hz
   (na Bode ostry szczyt). ωc - szerokość rezonansu (odporność na zmiany częstotliwości sieci).
OBIEKT:  dławik sieciowy  P(s) = 1/(L s + R) · e^(-s·Td).

ZPK regulatora:  zera i bieguny blisko ±j·ω0 - para biegunów słabo tłumionych w ±j2π50 (to
"pamięć" sinusoidy, nie niestabilność obiektu). W pętli zamkniętej pojawia się para biegunów
blisko nich i zero, które ją prawie skraca.

WNIOSKI:
 * Kp ustala pasmo (fc ≈ Kp/(2πL)) i zapas fazy, tak jak w C1.
 * Kr ustala, jak szybko znika błąd sinusoidy (obwiednia błędu ~ e^(-Kr·t/L)); za duże Kr psuje
   zapas fazy przy fc.
 * Na wykresie symulacji czasowej porównujemy PR z PI o tym samym Kp - PI zostawia błąd.
Nyquist dla PR ma dużą pętlę (rezonans) - liczy się, czy ominie -1.
SPRÓBUJ:  ustaw Kr = 0 (czysty P) - błąd śledzenia jest duży. Zmień f sieci na 51 Hz przy małym
ωc - błąd rośnie (rezonans "nie trafia"); zwiększ ωc - poprawa.
"""

EXPL["C3"] = """UKŁAD STEROWANIA C3 - Kaskada: regulator napięcia DC + wewnętrzna pętla prądu
(np. przekształtnik LC z rys. 4.10 utrzymujący napięcie kondensatora szyny DC, z której RC zasila silnik)

STRUKTURA KASKADOWA (jak w każdym napędzie):
   pętla zewnętrzna (wolna):  napięcie kondensatora Vdc  -> regulator PI -> prąd zadany i*
   pętla wewnętrzna (szybka): prąd -> PI z zakładki C1 -> napięcie mostka.
Wewnętrzną pętlę zastępujemy członem inercyjnym  Gi(s) = 1/(τi s + 1),  τi = 1/(2π·fci).
OBIEKT ZEWNĘTRZNY:  kondensator   C dv/dt = i - i_obc   ->   Pv(s) = 1/(C s) · Gi(s)
   (całkowanie: kondensator "zbiera" różnicę prądów).

STROJENIE "OPTIMUM SYMETRYCZNE" (przycisk):  parametr a (zwykle 2..4):
   Kp = C/(a·τi),   Ti = a²·τi  (Ki = Kp/Ti)
   -> fc = 1/(a·τi·2π),  maksimum fazy dokładnie przy fc,  PM = arcsin((a²-1)/(a²+1)):
   a = 2 -> 36,9°,  a = 3 -> 53,1°,  a = 4 -> 61,9°.
REGUŁA KASKADY:  pętla zewnętrzna co najmniej 5-10× wolniejsza od wewnętrznej.

ZAKŁÓCENIE:  skok prądu obciążenia (silnik nagle pobiera więcej) - napięcie chwilowo spada
(wykres "zakłócenie"), regulator PI je przywraca. Spadek ~ ΔI/(C·ωc).
WNIOSKI:
 * duże a -> większy zapas fazy, ale wolniej i większy spadek napięcia przy zakłóceniu,
 * małe C -> szybciej, ale większe tętnienia napięcia w praktyce,
 * odpowiedź na skok zadanej w optimum symetrycznym ma duże przeregulowanie (~43 % dla a = 2) -
   w praktyce dodaje się filtr wartości zadanej.
SPRÓBUJ:  zmniejszaj fci (wolniejsza pętla prądu) przy stałych Kp, Ki - PM spada!
"""

EXPL["C4"] = """UKŁAD STEROWANIA C4 - Regulator balansu kondensatorów 3-poziomowego NPC (rozdz. 6.6)

Książka stosuje prostą histerezę. Tu pokazujemy, jak to samo zadanie zapisać jako LINIOWĄ pętlę
regulacji, którą można zbadać metodami Bode/Nyquist/step.

MODEL:  dvc/dt = <i_M>/C.  Średni prąd punktu środkowego zależy od przesunięcia δ (wyboru
redundantnych sekwencji):   <i_M> ≈ Kb · δ.
   Kb [A] liczymy numerycznie z modelu modulatora:  Kb = d<i_M>/dδ = -<Σ i_x·sgn(u_x + δ - 1)>,
   uśrednione po okresie 50 Hz - zależy od amplitudy prądu I i współczynnika mocy cos φ
   (przy cos φ = 0 Kb ≈ 0 - kondensatorów NIE da się balansować samym δ!).
OBIEKT:  P(s) = |Kb| / (C s) · 1/(τf s + 1) · e^(-s·Ts)
   całkowanie (kondensator) + filtr pomiaru vc (usuwa tętnienie 150 Hz) + opóźnienie cyfrowe.
REGULATOR:  P lub PI:  δ = -(Kp·vc + Ki∫vc)/(Vdc/2)  (znormalizowane).

WNIOSKI:
 * Obiekt całkujący + regulator P -> uchyb ustalony zerowy dla skoku zadanej, ale stałe
   zakłócenie (np. niesymetria obciążenia) zostawia błąd - wtedy dodaj Ki.
 * Filtr pomiaru wprowadza opóźnienie fazy - zbyt duże Kp przy dużym τf -> oscylacje vc.
 * Małe I lub cos φ -> małe Kb -> pętla zwalnia (fc ~ Kp·|Kb|/(C·Vdc/2·2π)) - to dlatego
   balansowanie NPC jest trudne przy biegu jałowym i przy obciążeniu biernym.
Symulacja czasowa (3. zakładka) używa PEŁNEGO modelu przełączanego NPC z zakładki 6.4 z
regulatorem P - porównaj ją z przewidywaniem liniowym.
SPRÓBUJ:  zmniejsz cos φ (zwiększ L) - fc spada; zwiększ Kp aż PM < 30° i obserwuj odpowiedź.
"""
