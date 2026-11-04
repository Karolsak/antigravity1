"""Teksty objasnien do aplikacji plecs3_im_drives_tkinter.py (jezyk: polski,
poziom: uczen liceum + wnioski inzynierskie)."""

INTRO = """SILNIK INDUKCYJNY – MODELE I STEROWANIE
Interaktywny przewodnik po rozdziałach 8 i 9 książki „Advanced Electrical Drives” (De Doncker, Pulle, Veltman) – załącznik Plecs3.

JAK KORZYSTAĆ
• Wybierz u góry „Rozdział 8 – Modele maszyny” albo „Rozdział 9 – Sterowanie napędem”, a potem konkretny przykład.
• Po lewej są suwaki (możesz też wpisać dokładną liczbę w polu obok i nacisnąć Enter). Każda zmiana automatycznie przelicza symulację.
• Pod suwakami są „Wyniki liczbowe”: najważniejsze liczby, które warto porównać z książką.
• Po prawej:
   – „Wykresy”: pasek narzędzi na dole pozwala powiększać (lupa), przesuwać (krzyżyk), wracać (domek) i zapisać obraz PNG (dyskietka). Pozycja kursora jest pokazywana w pasku.
   – „Analiza LTI” (tylko w układach sterowania): charakterystyki Bodego, postać zpk, mapa zer i biegunów, wykres Nyquista i odpowiedź skokowa wraz z automatycznym opisem wyników.
   – „Wyjaśnienie”: opis przykładu napisany prostym językiem.

KRÓTKI SŁOWNICZEK
• Silnik indukcyjny (asynchroniczny) – najpopularniejszy silnik świata: pompy, wentylatory, tramwaje, Tesla Model S. Stojan wytwarza wirujące pole magnetyczne, które „ciągnie” wirnik. Wirnik zawsze kręci się trochę wolniej niż pole: ta różnica to POŚLIZG. Bez poślizgu w wirniku nie indukowałby się prąd i nie byłoby momentu.
• Strumień ψ [Vs] – miara „ilości pola magnetycznego” sprzężonego z uzwojeniem.
• Moment Te [Nm] – siła obrotowa (jak siła na ramieniu klucza).
• Wektor przestrzenny – sposób zapisania trzech prądów fazowych jako jednej strzałki, która się obraca. Zamiast trzech sinusoid mamy jedną wirującą strzałkę.
• Układ dq – „obracające się okulary”: patrzymy na strzałki z układu, który kręci się razem z polem, więc strzałki wydają się stać w miejscu (wartości stałe, łatwe do regulacji).
• Współczynnik a (model UFO) – „suwak”, który wybiera, z którym strumieniem związujemy oś d: wirnika (a = Lm/Lr), szczeliny (a = 1) albo stojana (a = Ls/Lm). Fizyka maszyny się nie zmienia, zmienia się tylko sposób opisu.
• V/f – najprostsze sterowanie: napięcie rośnie proporcjonalnie do częstotliwości.
• FOC/UFO – sterowanie polowo zorientowane: osobno sterujemy strumieniem (prąd i_d) i momentem (prąd i_q), jak w silniku prądu stałego.

ANALIZA UKŁADÓW STEROWANIA (zakładki „Analiza LTI”)
• Transmitancja G(s) – „przepis”, jak układ zamienia wejście na wyjście. s to „zmienna częstotliwości”.
• zpk – zapis przez ZERA (z), BIEGUNY (p) i WZMOCNIENIE (k). Bieguny mówią o naturze ruchu: biegun rzeczywisty ujemny = łagodne dochodzenie do celu (stała czasowa τ = −1/p); para zespolona = drgania; biegun z dodatnią częścią rzeczywistą = NIESTABILNOŚĆ (drgania rosną).
• Bode – jak układ reaguje na sinusoidę o różnej częstotliwości: wzmocnienie [dB] i przesunięcie fazy [°].
• Nyquist – ten sam wynik narysowany jako krzywa na płaszczyźnie zespolonej. Dla pętli regulacji: jeśli krzywa „obejmuje” punkt −1, układ zamknięty jest niestabilny.
• Odpowiedź skokowa – co się dzieje po nagłej zmianie wejścia (np. włączeniu).
• Zapas fazy PM i zapas wzmocnienia GM – „margines bezpieczeństwa” regulatora. Dobra praktyka: PM ≥ 45…60°, GM ≥ 6 dB.

PARAMETRY UŻYTE W PRZYKŁADACH
• Maszyna dydaktyczna (Tutoriale 8.6.1–8.6.5): Rs = 6,9 Ω, Rr = 3,0 Ω, Lm = 340,9 mH, Ls = Lr = 346,9 mH, p = 2.
• Maszyna 17,3 kW (w tekście rozdz. 9 nazwana „22 kW”), trójkąt, 415 V, 30 A, 1465 obr/min: parametry wyznaczone w Tutorialu 8.6.6 (Tab. 8.2 i 8.3), J = 0,1 kg·m².
• Silnik jednofazowy 150 W z kondensatorem 31 µF (Tab. 8.4), 110 V, 50 Hz.

WERYFIKACJA Z KSIĄŻKĄ (wyniki tej aplikacji)
• Estymacja parametrów: Ls = 272 mH, Lm = 260 mH, Lσ = 12 mH, Rr = 0,541 Ω, LM = 248,5 mH, LσS = 23,6 mH, RR = 0,494 Ω, ψs = 1,83 Vs. Zgodne z Tab. 8.2 i 8.3.
• Model uniwersalny przy 1 Nm: n = 1496,8 obr/min (w książce 1496,57).
• Sieć 415 V, 113 Nm: n = 1464,9 obr/min, I = 29,96 A (tabliczka: 1465 obr/min, 30 A); bieg jałowy 8,5 A (pomiar 8,41 A).
• V/f z czujnikiem prędkości: Te = 105,2 Nm, |ψs| = 1,871 Vs (w książce 105,1 Nm i 1,871 Vs).
• V/f bez czujnika: analiza biegunów pokazuje obszar niestabilności ok. 400–550 obr/min (w książce ok. 580 obr/min).
• UFO: a = 0,9557 (wirnik) i 1,0464 (stojan), tak jak na Fig. 9.28 i 9.29.
"""

TEXTS = {}

TEXTS["t81"] = """TUTORIAL 8.6.1 – UPROSZCZONY MODEL SILNIKA INDUKCYJNEGO

CO TO JEST?
To najprostszy możliwy opis silnika indukcyjnego. Udajemy, że całe pole magnetyczne przechodzi przez szczelinę powietrzną do wirnika i nigdzie „nie ucieka” (brak tzw. indukcyjności rozproszenia). Zamiast podawać napięcie, podajemy od razu wirujący strumień ψm o stałej amplitudzie 1,25 Vs i częstotliwości 50 Hz (ωs = 100π rad/s). Autorzy robią tak celowo: to strumień, a nie napięcie, jest „sercem” działania maszyny.

ANALOGIA
Wyobraź sobie karuzelę magnesów (pole), która kręci się dookoła klatki z aluminiowych prętów (wirnik). Gdy wirnik stoi, pręty „widzą” szybko przelatujące pole, indukuje się w nich duży prąd i powstaje duża siła. Gdy wirnik dogoni pole, nic już nie przelatuje: prąd i siła spadają do zera. Dlatego silnik indukcyjny zawsze kręci się odrobinę wolniej niż pole. Ta różnica to poślizg.

WZORY (bez strachu)
• Prąd wirnika jest proporcjonalny do tempa zmian strumienia widzianego przez wirnik: i_r = −(1/Rr)·dψ/dt.
• Moment: Te = 3/2 · p · ψ² · ωsl / Rr, gdzie ωsl = ωs − ωm (pulsacja poślizgu), a p = 2 to liczba par biegunów.
• Wniosek: w tym modelu moment rośnie LINIOWO z poślizgiem. Wykres „Charakterystyka mechaniczna” to prosta.

CO POKAZUJĄ WYKRESY
1. Rozruch: moment silnika (niebieski) jest na początku największy i maleje, gdy prędkość rośnie. Obciążenie wentylatorowe (pomarańczowy) rośnie z kwadratem prędkości. Silnik przyspiesza, dopóki moment silnika jest większy od momentu obciążenia.
2. Punkt pracy (czerwona kropka) to przecięcie prostej silnika z parabolą wentylatora.
3. Bilans mocy: moc pobrana dzieli się na moc mechaniczną i straty w miedzi stojana i wirnika. Moc bierna Q (na wykresie podzielona przez 10) jest duża, bo silnik musi „magnesować się” prądem i_m.
4. Wykres wektorowy: prąd stojana i_s to suma prądu magnesującego (wzdłuż strumienia) i prądu „momentowego” (prostopadle). To zapowiedź sterowania polowego z rozdziału 9!

EKSPERYMENTY
• Zwiększ moment obciążenia: poślizg rośnie, rośnie prąd wirnika i straty, a sprawność spada.
• Zmniejsz Rr: prosta staje się bardziej stroma („sztywniejszy” silnik), więc poślizg przy tym samym obciążeniu maleje.
• Zmniejsz ψm o połowę: moment przy danym poślizgu spada czterokrotnie (ψ²).
• Zwiększ J: rozruch trwa dłużej, ale punkt pracy jest ten sam.

WNIOSKI INŻYNIERA
• Silnik indukcyjny bez rozproszeń zachowuje się jak „hamulec wiroprądowy z napędem”: moment ∝ poślizg. Prawdziwy silnik ma moment krytyczny i ten model go nie pokazuje (zobacz zakładkę „Stan ustalony”).
• Niska wartość cos φ w tym przykładzie wynika z małego obciążenia: prąd magnesujący dominuje.
• W oryginalnym modelu PLECS prąd wirnika liczony jest przez różniczkowanie. Autorzy ostrzegają, że różniczkowanie w symulacji jest numerycznie ryzykowne. Tutaj liczymy go analitycznie.
"""

TEXTS["t82"] = """TUTORIALE 8.6.2 I 8.6.3 – UNIWERSALNY MODEL IRTF I WSPÓŁCZYNNIK a

CO TO JEST?
Prawdziwy silnik ma indukcyjności rozproszenia: część pola stojana nie dociera do wirnika i odwrotnie. Mamy więc cztery indukcyjności: Lσs, Lσr, Lm (i Ls = Lσs + Lm, Lr = Lσr + Lm). Okazuje się, że ten sam silnik można opisać na nieskończenie wiele równoważnych sposobów, wybierając liczbę a (współczynnik transformacji). Model nazywa się „uniwersalnym” (IRTF – Ideal Rotating Transformer, idealny wirujący transformator).

ANALOGIA
To jak mierzenie wysokości: od poziomu morza, od podłogi albo od stołu. Liczby są różne, ale budynek ten sam. Współczynnik a zmienia „poziom odniesienia” dla strumienia wirnika: przesuwa część rozproszenia ze strony wirnika na stronę stojana albo odwrotnie.

TRZY WAŻNE WYBORY
• a = Lm/Lr ≈ 0,9827: całe rozproszenie po stronie stojana (LσR = 0). Strumień ψM = ψR (strumień wirnika). Na tym opiera się klasyczne sterowanie polowe.
• a = 1: klasyczny schemat T z Lσs i Lσr.
• a = Ls/Lm ≈ 1,0177: całe rozproszenie po stronie wirnika (LσS = 0). Strumień ψM = ψs (strumień stojana). Na tym opiera się V/f i DTC.
Dozwolony zakres: Lm/Lr ≤ a ≤ Ls/Lm (wzór 8.13).

PRZELICZENIA (wzory 8.12, 8.15)
LM = a·Lm,  LσS = Ls − a·Lm,  LσR = a²·Lr − a·Lm,  RR = a²·Rr,  ψR = a·ψr,  iR = ir/a.

CO POKAZUJĄ WYKRESY
1. Moment i prędkość dla trzech wartości a (wybranej i dwóch skrajnych). Linie się pokrywają! To dowód, że a nie zmienia fizyki, czyli tego, co mierzymy na zaciskach i na wale.
2. Moduły strumieni |ψM| i |ψR| zależą od a, a |ψs| nie (to prawdziwy strumień stojana).
3. Wykres wektorowy (oś d związana ze strumieniem szczeliny ψm): grube szare/fioletowe linie to miejsca, po których „jeżdżą” końce wektorów ψM i ψR, gdy przesuwasz a. Wektor ψM leży zawsze na odcinku między ψR(a_min) a ψs (Fig. 8.11b).
4. Prądy fazowe – trzy sinusoidy przesunięte o 120°.

EKSPERYMENTY
• Przesuwaj a i obserwuj, że Te i n w „Wynikach liczbowych” nie zmieniają się (różnice są na poziomie błędów zaokrągleń).
• Zmień obciążenie z 1 Nm na 10 Nm: wektory się rozsuwają, bo rośnie prąd, a z nim spadki napięcia na rozproszeniach.
• Zasilanie U/f: książka podaje napięcie „220 V”, a generator U/f w modelu PLECS daje û = ωs·ψ*. Dla ψ* = 1,25 Vs dostajemy n ≈ 1496,8 obr/min przy 1 Nm, czyli prawie to samo co w książce (1496,57 obr/min, Fig. 8.42).

WNIOSKI INŻYNIERA
• Model uniwersalny ma 4 parametry zamiast 5 (Rs, LσS, LM, RR + a). Z pomiarów na zaciskach nie da się rozróżnić Lσs od Lσr. Wybór a usuwa ten nadmiar.
• Wybierając a, wybieramy, który strumień będzie „łatwy” w sterowaniu. To podstawa sterownika UFO z rozdz. 9.
"""

TEXTS["t84"] = """TUTORIAL 8.6.4 – MODEL PRĄDOWY BEZ ROZPROSZEŃ (FUNDAMENT STEROWANIA POLOWEGO)

CO TO JEST?
Tym razem nie podajemy napięcia, tylko od razu PRĄDY w obracającym się układzie dq, zgodnym ze strumieniem:
• i_d (oś d) – prąd „magnesujący”: buduje strumień,
• i_q (oś q) – prąd „momentowy”: wytwarza moment.
Tak właśnie działa falownik z regulatorem prądu: silnik dostaje prąd, jakiego chcemy.

ANALOGIA
Silnik prądu stałego ma dwa osobne pokrętła: prąd wzbudzenia (pole) i prąd twornika (moment). Tutaj, dzięki obracającemu się układowi dq, silnik indukcyjny też zyskuje dwa osobne pokrętła: i_d i i_q.

RÓWNANIA (Fig. 8.6 / 8.43)
• dψ/dt = Rr·(i_d − ψ/Lm). Strumień narasta jak temperatura herbaty w kubku: wykładniczo, ze stałą czasową τ = Lm/Rr ≈ 114 ms.
• ωsl = Rr·i_q / ψ – poślizg potrzebny do wytworzenia prądu wirnika.
• Te = 3/2·p·ψ·i_q – moment.
• J·dΩ/dt = Te − Tl – druga zasada dynamiki dla ruchu obrotowego.

CO POKAZUJĄ WYKRESY (Fig. 8.44)
• Strumień rośnie łagodnie (człon inercyjny I rzędu) do wartości Lm·i_d = 1,36 Vs.
• W chwili skoku i_q moment skacze natychmiast, poślizg też skacze natychmiast, a strumień się NIE zmienia. To jest właśnie odsprzężenie: moment zmieniamy bez ruszania strumienia.
• Prędkość rośnie liniowo, bo moment jest stały, a obciążenia brak. ωs = ωm + ωsl też rośnie liniowo.

ANALIZA LTI (zakładka obok)
• Gψ(s) = Lm/(τs + 1): jeden biegun rzeczywisty w s = −1/τ. Wykres Bodego: płasko do ω ≈ 1/τ, potem spadek −20 dB/dekadę. Strumień „nie nadąża” za szybkimi zmianami i_d.
• Gω(s) = K/(J·s): człon całkujący (biegun w zerze). Stały prąd i_q daje prędkość rosnącą bez końca, czyli liniową rampę.
• Pętla prędkości z regulatorem PI (rozszerzenie inżynierskie): dzięki temu, że moment reaguje natychmiast, pętlę prędkości projektuje się jak dla prostego całkowania. Zobacz zapas fazy i odpowiedź skokową; zmieniaj Kp i Ki.

EKSPERYMENTY
• Zwiększ Rr: τ maleje, strumień szybciej się buduje, ale poślizg przy danym i_q rośnie.
• Daj i_q przed ustaleniem się strumienia (t_q = 0,1 s): moment rośnie razem ze strumieniem. Dlatego napęd FOC najpierw „magnesuje” silnik, a dopiero potem daje moment.

WNIOSKI INŻYNIERA
To najważniejsza idea rozdziału: sterując prądami w układzie związanym ze strumieniem wirnika, dostajemy silnik indukcyjny, który zachowuje się jak obcowzbudny silnik prądu stałego.
"""

TEXTS["t85"] = """TUTORIAL 8.6.5 – MODEL UFO Z ROZPROSZENIAMI (ROTOR- I STATOR-FLUX ORIENTED)

CO TO JEST?
Uogólnienie poprzedniego przykładu: model prądowy z rozproszeniami, w którym oś d wiąże się ze strumieniem ψM wybranym parametrem a (UFO = Universal Field Oriented). Równania modelu (wyprowadzone z Fig. 8.22):
• (L_R/L_M)·dψM/dt = R_R·(i_d − ψM/L_M) + LσR·di_d/dt − ωsl·LσR·i_q
• ωsl = (R_R·i_q + LσR·di_q/dt) / (ψM·L_R/L_M − LσR·i_d)
• Te = 3/2·p·ψM·i_q

DWA SKRAJNE PRZYPADKI
• a = Lm/Lr (orientacja na strumień WIRNIKA): LσR = 0, więc znikają wszystkie człony z LσR. Model staje się identyczny jak model bez rozproszeń (Fig. 8.25 = Fig. 8.6). Pełne odsprzężenie: zmiana i_q nie rusza strumienia (Fig. 8.26).
• a = Ls/Lm (orientacja na strumień STOJANA): LσR ≠ 0. Pojawia się różniczkowanie i_q, więc skok momentu daje IMPULS (prawie deltę Diraca) w poślizgu. Strumień ψM = ψs lekko reaguje na zmiany i_q, czyli osie są sprzężone (Fig. 8.29).

ANALOGIA
Jedziesz rowerem z przyczepką. Jeśli trzymasz za przyczepkę (strumień wirnika), to szarpnięcie nie przenosi się dalej. Jeśli trzymasz za rower (strumień stojana), a przyczepka jest połączona sprężyną (rozproszenie LσR), każde szarpnięcie wprawia ją w ruch.

ANALIZA LTI
• Gψ(s) = (L_M/L_R)·(LσR·s + R_R)/(s + R_R/L_R). Przy a = Lm/Lr zero znika (LσR = 0) i zostaje czysty człon inercyjny. Przy a = Ls/Lm pojawia się zero w s = −R_R/LσR (daleko w lewo), które daje natychmiastowy „skok” strumienia o L_M·LσR/L_R·Δi_d.
• Gsl(s): poślizg w funkcji i_q z rampą (filtr 1/(Tr·s+1) zastępuje ogranicznik narastania). Przy orientacji na stojan licznik ma człon LσR·s, czyli różniczkowanie. Im krótsza rampa Tr, tym wyższy impuls poślizgu. Dlatego w książce zamiast skoków użyto ramp.

EKSPERYMENTY
• Ustaw a na minimum i maksimum i porównaj wykres ωsl. Przy maksimum pojawia się szpilka.
• Skróć czas rampy do 1 ms: szpilka rośnie. Wydłuż do 50 ms: znika.
• Zauważ w wynikach: zmiana ψM po skoku i_q to 0 % dla a_min i kilka promili dla a_max.

WNIOSKI INŻYNIERA
Orientacja na strumień wirnika daje najprostszy sterownik (dlatego ją stosowano jako pierwszą, w czasach analogowych). Orientacja na strumień stojana jest trudniejsza do „odwrócenia” w sterowniku, ale ma zalety (mniejsza wrażliwość na parametry wirnika, łatwy pomiar strumienia).
"""

TEXTS["heyl"] = """8.3.5 – WYKRES HEYLANDA W UKŁADZIE SYNCHRONICZNYM

CO TO JEST?
Pytamy: gdzie leży koniec wektora prądu stojana i_s (w układzie dq), gdy strumień ψM utrzymujemy stały, a zmieniamy obciążenie, czyli poślizg ωsl? Ze stanu ustalonego modelu UFO wychodzi:
• i_sq = ωsl·ψM / (R_R·(1 + x²)),  i_sd = ψM/L_M + x·i_sq,  gdzie x = ωsl·LσR/R_R.
Wszystkie te punkty leżą na OKRĘGU o środku (ψM/2·(1/LσR + 2/L_M), 0) i promieniu ψM/(2·LσR). To okrąg Heylanda (Fig. 8.30).

DLACZEGO TO WAŻNE?
• Najwyższy punkt okręgu to maksymalne i_q, czyli maksymalny moment Te_max = 3/2·p·ψM·r przy ωsl = R_R/LσR. Jest to moment krytyczny dla danego strumienia.
• Im mniejsze rozproszenie LσR, tym większy okrąg. Przy a = Lm/Lr (LσR = 0) promień staje się nieskończony i okrąg zamienia się w PROSTĄ pionową i_d = ψM/L_M (Fig. 8.30b). Wtedy i_d nie zależy od obciążenia, a moment nie ma teoretycznego maksimum.

ANALOGIA
To jak koło z szprychą: zmieniając poślizg, „kręcisz” końcem szprychy po obwodzie. Duże koło (małe rozproszenie) to łagodny łuk; nieskończenie duże koło to linia prosta.

CO POKAZUJĄ WYKRESY
• Lewy: okręgi dla kilku wartości a (czarny – wybrana) i zielony wektor i_s dla zaznaczonego poślizgu.
• Prawy: Te(ωsl) przy stałym ψM. Dla dużego a wykres ma maksimum (moment krytyczny), dla a_min jest prostą.

EKSPERYMENTY
• Ustaw a na maksimum (orientacja na stojan) i przesuń poślizg do ±RR/LσR (wynik w okienku) – zobaczysz szczyt okręgu.
• Zmniejszaj a: okrąg puchnie, a moment krytyczny ucieka do nieskończoności.

WNIOSKI INŻYNIERA
Przy orientacji na stojan prąd i_d MUSI rosnąć z momentem (punkt wędruje po okręgu w prawo). Ten fakt wykorzystuje się w 8.4 do wyznaczenia LσR z danych znamionowych. Wtedy i_d(nom) > i_d(bieg jałowy).
"""

TEXTS["steady"] = """8.3.6 + TUTORIAL 8.6.8 – STAN USTALONY SILNIKA ZASILANEGO Z SIECI

CO TO JEST?
Silnik 17,3 kW (trójkąt, 415 V, 50 Hz) zasilany prosto z sieci. W stanie ustalonym wszystkie przebiegi są sinusoidalne, więc zamiast równań różniczkowych używamy liczb zespolonych (fazorów) i schematu zastępczego z Fig. 8.31: Rs, LσS, a równolegle LM i R_R/s (s – poślizg).
Prąd: i_s = û_s / (Rs + jωs·LσS + (jωs·LM ∥ R_R/s)).
Moment: Te = 3/2·p·|i_R|²·R_R/(s·ωs) – moc „przepływająca przez szczelinę” podzielona przez prędkość pola.

CO POKAZUJĄ WYKRESY
1. Wykres Heylanda: koniec wektora prądu stojana zatacza okrąg. Niebieski – praca silnikowa (0 < s ≤ 1), zielony – prądnicowa (s < 0, wirnik szybszy niż pole, oddajemy energię do sieci), czerwony – hamowanie przeciwprądem (s > 1).
2. Charakterystyka mechaniczna Te(n): duży moment krytyczny (ok. 2,5× Tn), stromy odcinek roboczy przy małym poślizgu i ujemny moment powyżej 1500 obr/min (prądnica).
3. Prąd przewodowy I_l: przy rozruchu (n = 0) prąd jest kilka razy większy od znamionowego 30 A! Dlatego duże silniki uruchamia się przez przełącznik gwiazda-trójkąt, softstart albo falownik.
4. Moc mechaniczna i sprawność: najwyższa sprawność jest blisko punktu znamionowego.

PORÓWNANIE Z TABLICZKĄ
Dla n = 1465 obr/min wychodzi Te ≈ 113 Nm i I_l ≈ 30 A, czyli dokładnie jak na tabliczce (Tab. 8.1). Na biegu jałowym I_l ≈ 8,5 A (pomiar: 8,41 A). Model jest dobrze dopasowany.

EKSPERYMENTY
• Zmniejsz napięcie do 70 %: moment krytyczny spada do ok. 50 % (moment ∝ U²).
• Zwiększ Rr 2×: moment krytyczny pozostaje ten sam, ale przesuwa się w stronę mniejszych prędkości. Rośnie moment rozruchowy (tak działały silniki pierścieniowe z rezystorami rozruchowymi).
• Zmień częstotliwość na 25 Hz przy 415 V: strumień rośnie 2×. W rzeczywistości silnik by się nasycił, dlatego przy zmianie f zmienia się też U (sterowanie V/f).

WNIOSKI INŻYNIERA
Wykres znormalizowany (Fig. 8.32) jest uniwersalny: prąd dzieli się przez û/(ωs·LσS), a moment przez p·û²/(2ωs²·LσS). Widać wtedy, że kształt zależy tylko od stosunku rezystancji do reaktancji rozproszenia.
"""

TEXTS["estim"] = """8.4 + TUTORIAL 8.6.6 – PARAMETRY SILNIKA Z TABLICZKI ZNAMIONOWEJ

PROBLEM
Do sterowania polowego potrzebujemy parametrów modelu (Ls, Lm, Lσ, Rr, Rs), a producent podaje tylko tabliczkę: moc, napięcie, prąd, prędkość, cos φ. Autorzy pokazują przepis (Listing 8.1), który z tabliczki i dwóch prostych pomiarów (prąd biegu jałowego i rezystancja między zaciskami) wylicza wszystkie parametry.

PRZEPIS KROK PO KROKU (lista po lewej)
1. Trójkąt: rezystancja między dwoma zaciskami to jedna faza równolegle z dwiema szeregowymi, R_l = (2/3)·Rs, więc Rs = 3/2·R_l.
2. Amplitudy wektorów: napięcie fazowe (trójkąt) û = √2·U, prąd fazowy î = √2·I/√3.
3. Bieg jałowy (i_q = 0, ωm = ωs): ψs = √(û² − (î0·Rs)²)/ωs, a Ls = ψs/î0.
4. Moment znamionowy z mocy w szczelinie: Te = P/ωm. Prąd momentowy: i_sq = 2/3·Te/ψs.
5. Prąd i_sd = √(î² − i_sq²). Jest większy niż prąd biegu jałowego, bo punkt wędruje po okręgu Heylanda (zakładka „Heyland UFO”).
6. Z okręgu Heylanda: RR i LσR (dla a = Ls/Lm).
7. Przejście do a = 1 przy założeniu Lσs = Lσr: Lm = √(Ls³/(LσR + Ls)), Rr = RR/(Ls/Lm)².
8. Przejście do modelu „rotor flux” (a = Lm/Lr): LM = a·Lm, LσS = Ls − a·Lm, RR = a²·Rr. Te wartości trafiają do symulacji w rozdz. 9 (Tab. 8.3).

WYNIK DLA SILNIKA Z KSIĄŻKI
Ls = 272 mH, Lm = 260 mH, Lσ = 12 mH, Rr = 0,54 Ω, Rs = 0,6 Ω; LM = 248,5 mH, LσS = 23,6 mH, RR = 0,494 Ω; ψs = 1,83 Vs. Wykres słupkowy porównuje wynik z tabelami w książce.

UWAGA ZNALEZIONA PRZEZ „INŻYNIERA”
Tab. 8.3 podaje ψR = 1,67 Vs. Ta liczba pochodzi z linii 46 Listingu 8.1, w której składowa q jest mnożona przez 2 i odejmowany jest dodatkowy człon i_rd·LσS. Fizycznie poprawny wzór ψR = |ψs − LσS·i_s| daje ≈ 1,59 Vs. Aplikacja pokazuje obie wartości. Różnica ma znaczenie przy ustawianiu strumienia zadanego w FOC.

EKSPERYMENTY
• Zmień cos φ z 0,80 na 0,85: zmienia się podział prądu na i_sd/i_sq, a więc LσR i Lm.
• Zmień prąd biegu jałowego: bezpośrednio zmienia się Ls (silnik „łatwiej się magnesuje”).

WNIOSKI INŻYNIERA
Metoda jest szybka i wystarczająco dokładna do pierwszego uruchomienia napędu. Autorzy podkreślają, że strumień przy obciążeniu znamionowym jest o kilka % mniejszy niż na biegu jałowym. Pominięcie tej różnicy psuje zgodność momentu i poślizgu w symulacji.
"""

TEXTS["grid"] = """TUTORIAL 8.6.7 – SILNIK 17,3 kW ZAŁĄCZONY BEZPOŚREDNIO DO SIECI

CO TO JEST?
Pełny model dynamiczny (Fig. 8.15/8.46) z parametrami z Tab. 8.3. Silnik stoi, w chwili t = 0 włączamy napięcie 415 V, a po rozruchu (t = 1 s) dokładamy znamionowe obciążenie 113 Nm.

CO POKAZUJĄ WYKRESY (Fig. 8.47)
1. Moment na początku silnie oscyluje (raz w przód, raz w tył, także ujemnie!). To efekt nagłego „wpuszczenia” pola do maszyny: strumień musi się zbudować, a w uzwojeniach płyną przebiegi przejściowe o częstotliwości sieci. Po skoku obciążenia moment też chwilę drga, zanim się ustali.
2. Trajektoria dynamiczna na tle charakterystyki statycznej: w czasie szybkiego rozruchu silnik wcale nie porusza się po charakterystyce z katalogu! Dopiero na końcu „ląduje” na niej.
3. Prąd fazowy: prąd rozruchowy jest wielokrotnością znamionowego (≈ 35 A amplitudy). Tak powstają spadki napięcia w sieci przy rozruchu dużych silników.
4. Strumienie ustalają się po kilkudziesięciu ms.

ANALIZA LTI
Linearyzujemy model (5 stanów: ψsd, ψsq, ψRd, ψRq, Ω) w punkcie pracy po skoku obciążenia:
• G_l(s) = Δn/ΔT_load: jak prędkość „ugina się” po zmianie obciążenia. Wzmocnienie statyczne ≈ −0,4 obr/min/Nm to nachylenie charakterystyki mechanicznej. Para biegunów zespolonych (≈ 13 Hz) to „sprężyna” pola magnetycznego połączona z masą wirnika.
• L(s) – pętla elektromechaniczna: wirnik (masa) jest połączony z wirującym polem (napęd) jakby przez sprężynę z tłumikiem. Analiza Nyquista pokazuje, że dla zasilania sieciowego ta pętla jest stabilna (dodatni zapas fazy).

EKSPERYMENTY
• Zwiększ J 5×: rozruch trwa dłużej, oscylacje elektromechaniczne mają niższą częstotliwość.
• Obniż napięcie do 300 V: rozruch się wydłuża, a przy 113 Nm rośnie poślizg.
• Zmieniaj a: wynik się nie zmienia (niezmienniczość modelu uniwersalnego).

WNIOSKI INŻYNIERA
Duże udary momentu przy bezpośrednim rozruchu męczą sprzęgło i przekładnię. Jest to jeden z powodów stosowania softstartów i falowników.
"""

TEXTS["single1"] = """8.5 + TUTORIAL 8.6.9 – SILNIK JEDNOFAZOWY Z KONDENSATOREM PRACY (ROZRUCH)

CO TO JEST?
W domu mamy tylko jedną fazę (230 V, tu: 110 V). Jedno uzwojenie daje pole PULSUJĄCE, a nie wirujące, więc silnik sam nie ruszy. Rozwiązanie: drugie uzwojenie (pomocnicze, aux) przesunięte w przestrzeni o 90°, zasilane przez kondensator, który przesuwa fazę prądu. Dwa przesunięte prądy w dwóch prostopadłych uzwojeniach dają (prawie) wirujące pole. Tak działa wentylator, pompa obiegowa czy pralka.

MODEL (Fig. 8.34, 8.35, 8.49)
• Oś α – uzwojenie pomocnicze (przez kondensator C), oś β – uzwojenie główne (run), wprost z sieci.
• Uzwojenia mają różne liczby zwojów: asymetryczny transformator ITF o przekładni k_aux : 1 = 1,18.
• Po stronie wirnika ten sam model IRTF co dla silnika trójfazowego (LσS, LM, RR).
• Kondensator: du_C/dt = i_aux/C, u_aux = u_sieci − u_C.

CO POKAZUJĄ WYKRESY (Fig. 8.50, 8.51)
1. Moment ma OGROMNE tętnienia (100 Hz), a jego średnia równa się obciążeniu 2 Nm. To typowa cecha silników jednofazowych: słychać „buczenie”.
2. Prądy uzwojeń są przesunięte w fazie, ale nie dokładnie o 90° i mają różne amplitudy, więc pole jest ELIPTYCZNE.
3. Napięcie na kondensatorze jest większe niż napięcie sieci! (rezonans z indukcyjnością uzwojenia). Kondensator musi mieć odpowiednie napięcie znamionowe.
4. Wykres wektorowy w układzie wirnika (Fig. 8.51): kąt między strumieniem wirnika a prądem stojana ciągle się zmienia, więc moment nie jest stały.

EKSPERYMENTY
• Ustaw C = 0 (silnik ze zwartym uzwojeniem pomocniczym… a raczej rozwartym): silnik nie wytwarza momentu rozruchowego. Pole jest tylko pulsujące.
• Zmień C na 15 i 60 µF: zmienia się szybkość rozruchu i tętnienia.
• Zwiększ obciążenie do 3 Nm: silnik może nie ruszyć albo „utknąć” przy małej prędkości.

WNIOSKI INŻYNIERA
Model uogólniony łatwo przerobić: silnik z kondensatorem rozruchowym to wyłącznik odłączający C przy pewnej prędkości; silnik z fazą rozruchową rezystancyjną (split-phase) to C → ∞ (1/C = 0). Wynik symulacji w stanie ustalonym zgadza się z analizą składowych symetrycznych (następna zakładka) z dokładnością do ułamków procenta. To dobry test poprawności obu metod.
"""

TEXTS["single2"] = """8.5.1 – STAN USTALONY SILNIKA JEDNOFAZOWEGO: SKŁADOWA ZGODNA I PRZECIWNA

IDEA
Pole eliptyczne można rozłożyć na DWA pola kołowe wirujące w przeciwne strony:
• składowa zgodna (+) – wiruje w kierunku wirnika, poślizg s,
• składowa przeciwna (−) – wiruje w przeciwną stronę, poślizg 2 − s.
Każda składowa „widzi” zwykły schemat silnika trójfazowego (Z_s+ i Z_s−, Fig. 8.37). Składowa zgodna ciągnie wirnik do przodu, a przeciwna hamuje. Ich iloczyn daje tętnienia momentu o częstotliwości 2f.

MATEMATYKA (w skrócie)
• X+ = (Xα + jXβ)/2,  X− = (Xα − jXβ)/2  (transformacja A, wzór 8.59).
• Z_s± = jωs·LσS + (jωs·LM ∥ R_R/s±).
• Uzwojenie pomocnicze ma po swojej stronie Raux i kondensator; po przeliczeniu przez ITF dzielimy je przez k_aux².
• Rozwiązujemy układ 2 równań liniowych z liczbami zespolonymi i liczymy moment średni oraz amplitudę tętnień.

CO POKAZUJĄ WYKRESY
1. Moment średni i tętnienia w funkcji n/n_s (Fig. 8.38): moment rozruchowy jest niewielki, maksimum występuje blisko prędkości synchronicznej, a tętnienia są duże w całym zakresie.
2. Amplitudy prądów uzwojeń.
3. Dobór kondensatora: dla wybranej prędkości szukamy C, przy którym tętnienia są najmniejsze (pole najbliższe kołowemu). To zadanie projektowe!
4. Wskazy prądów: idealnie I_aux wyprzedza I_run o 90°, a amplitudy odpowiadają przekładni zwojów.

EKSPERYMENTY
• Zmieniaj C: przy optymalnym C tętnienia spadają kilkukrotnie, a kąt zbliża się do 90°.
• Zmień k_aux: optymalne C się przesuwa.
• Zauważ, że optymalne C dla pracy (małe tętnienia przy 1400 obr/min) jest inne niż dla rozruchu (duży moment przy n = 0). Dlatego produkuje się silniki z dwoma kondensatorami.

WNIOSKI INŻYNIERA
Metoda składowych symetrycznych to standardowe narzędzie do maszyn i sieci niesymetrycznych. Pozwala policzyć charakterystyki bez długiej symulacji.
"""

TEXTS["kloss"] = """9.1 – IDEA STEROWANIA V/f I WZÓR KLOSSA

CO TO JEST?
Najprostszy sposób regulacji prędkości silnika indukcyjnego: falownik zmienia częstotliwość f (czyli prędkość pola), a napięcie zmienia proporcjonalnie (U/f = const). Dzięki temu strumień stojana ψs ≈ U/ω pozostaje stały i silnik się nie nasyca.

WZÓR KLOSSA (9.4, 9.5)
Przy stałym ψs moment zależy tylko od pulsacji poślizgu ωsl = ωs − ωm:
   Te = 2·T̂e / (ωsl/ω̂sl + ω̂sl/ωsl),   T̂e = 3/2·p·ψs²/(2·LσR),   ω̂sl = R_R/LσR
(parametry modelu „stator flux”, a = Ls/Lm).
• T̂e – moment krytyczny (maksymalny), NIE zależy od częstotliwości!
• ω̂sl – poślizg krytyczny.

CO POKAZUJĄ WYKRESY (Fig. 9.1)
• Lewy: charakterystyki dla trzech częstotliwości. Mają ten sam kształt, tylko są przesunięte w poziomie. Zmieniając f, przesuwamy całą charakterystykę. Punkt pracy to przecięcie z charakterystyką obciążenia.
• Prawy: uniwersalna krzywa Klossa. Obszar |ωsl| < ω̂sl to praca stabilna (gdy obciążenie rośnie, silnik zwalnia i daje WIĘKSZY moment). Za szczytem silnik przy wzroście obciążenia daje MNIEJSZY moment i „utyka”.

ANALOGIA
Rowerzysta na stałym przełożeniu: zmiana biegu (częstotliwości) przesuwa zakres prędkości, w którym jedzie mu się wygodnie, ale jego maksymalna siła (moment krytyczny) zostaje ta sama.

EKSPERYMENTY
• Zmniejsz ψs: moment krytyczny maleje z kwadratem.
• Zwiększ obciążenie ponad T̂e: punkt pracy znika (brak przecięcia). Silnik by stanął.

WNIOSKI INŻYNIERA
V/f jest proste i nie potrzebuje czujnika prędkości, ale prędkość wału różni się od zadanej o poślizg, a dynamika jest słaba (brak regulacji prądu). Następna zakładka pokazuje, że taki napęd może być nawet niestabilny.
"""

TEXTS["vf1"] = """9.1.1 + TUTORIAL 9.7.1 – PROSTY NAPĘD V/f (BEZ CZUJNIKA) I JEGO NIESTABILNOŚĆ

STEROWNIK (Fig. 9.2, 9.24)
Wejście: prędkość zadana n* (czyli częstotliwość ωs*) i strumień ψs*. Sterownik liczy napięcie w układzie wirującym:
   u_sd = Rs·ψs*/Ls   (mała kompensacja spadku na Rs),   u_sq = ωs*·ψs*
i wektor napięcia obraca kątem θs* = ∫ωs* dt. Nie ma żadnej pętli prądu ani pomiaru prędkości: to sterowanie w układzie otwartym.

CZĘŚĆ 1: ROZRUCH (Fig. 9.25)
Skok n* = 1465 obr/min przy obciążeniu wentylatorowym 113 Nm. Strumień dochodzi do ψs* po dużych przebiegach przejściowych, prąd rozruchowy jest bardzo duży (brak ograniczenia prądu!). Prędkość ustala się NIŻEJ niż zadana, bo sterownik zadaje prędkość pola, a wirnik ma poślizg. W praktyce rozruch robi się rampą, żeby ograniczyć prąd.

CZĘŚĆ 2: RAMPA I NIESTABILNOŚĆ (Fig. 9.4)
Powoli zwiększamy prędkość od 0 do 900 obr/min (30 obr/min/s) bez obciążenia, ale z małym prostokątnym zakłóceniem momentu (1 Nm, 1 Hz). Rr jest o 10 % mniejsze od nominalnego (jak w książce). W pewnym zakresie prędkości (ok. 450–600 obr/min) prędkość i prąd zaczynają GWAŁTOWNIE oscylować, a potem, przy wyższej prędkości, układ znowu się uspokaja.

DLACZEGO? (to wyjaśnia analiza LTI)
Wirnik (masa) połączony jest z wirującym polem jak przez sprężynę. W pętli „pole ↔ wirnik” tłumienie zależy od prędkości. W pewnym zakresie elektryczne zjawiska w maszynie dają UJEMNE tłumienie: para biegunów (ok. 12 Hz) przechodzi na prawą półpłaszczyznę (Re > 0). Prawy dolny wykres pokazuje największą część rzeczywistą biegunów w funkcji prędkości. Obszar powyżej zera (czerwony) to strefa niestabilności. Zakłócenie tylko „wyzwala” drgania. Bez niego niestabilność też by w końcu wystąpiła, choć z opóźnieniem (w praktyce wyzwalają ją harmoniczne żłobkowe).

ANALIZA LTI (punkt pracy z suwaka „n*_s punktu linearyzacji”)
• G_n(s) i G_l(s): przy niestabilnym punkcie zobaczysz biegun z Re > 0, rosnącą odpowiedź skokową i ostrzeżenie „NIESTABILNY”.
• L(s) – pętla elektromechaniczna: Nyquist dla 450 obr/min obejmuje punkt −1 (ujemny zapas fazy), a dla 300 lub 800 obr/min już nie.

EKSPERYMENTY
• Ustaw mnożnik Rr = 1,0: obszar niestabilny się kurczy (książka: „instability is substantially reduced”).
• Ustaw zakłócenie 0 Nm: drgania pojawiają się później.
• Zmieniaj punkt linearyzacji i obserwuj, jak bieguny „przechodzą” przez oś urojoną.

WNIOSKI INŻYNIERA
Producenci falowników V/f stosują „tłumienie aktywne” (np. korektę częstotliwości od prądu czynnego), żeby ominąć ten problem. Prawdziwe rozwiązanie to sterowanie polowe z regulacją prądu.
"""

TEXTS["vf2"] = """9.1.2 + TUTORIAL 9.7.2 – NAPĘD V/f STEROWANY MOMENTEM (Z CZUJNIKIEM PRĘDKOŚCI)

STEROWNIK (Fig. 9.5, 9.26)
Wejścia: moment zadany T* i strumień ψs*. Mamy pomiar prędkości wału ωm. Sterownik liczy:
• i_sq* = 2·T*/(3·p·ψs*) – potrzebny prąd momentowy,
• ωsl* = R_R·i_sq*/ψs* – potrzebny poślizg (wzór 9.7, pomija LσR),
• ωs* = ωm + ωsl* – częstotliwość = prędkość wału + poślizg,
• u_sd = Rs·ψs*/Ls,  u_sq = ωs*·ψs* + Rs·i_sq* („boost” napięcia przy małych prędkościach).

CO POKAZUJĄ WYKRESY (Fig. 9.6)
• Moment szybko dochodzi do ok. 105 Nm, a nie do 113 Nm! Prędkość rośnie, aż obciążenie wentylatorowe zrówna się z momentem silnika.
• Strumień |ψs| ≈ 1,871 Vs ≈ ψs*. W stanach przejściowych widać oscylacje, bo V/f NIE odsprzęga strumienia i momentu: energia „przelewa się” między polem a masą wirnika.
• ωs* rośnie razem z prędkością wału: sterownik zawsze utrzymuje stały poślizg.

DLACZEGO 105 Nm ZAMIAST 113 Nm?
Wzór na poślizg pomija rozproszenie wirnika LσR. W rzeczywistości dla wytworzenia 113 Nm potrzebny jest trochę większy poślizg (przypomnij sobie okrąg Heylanda: przy orientacji na stojan część prądu idzie w i_sd). Książka podaje Te = 105,1 Nm, a ta aplikacja 105,2 Nm. Dokładniejszy model poślizgu usunąłby błąd.

ANALIZA LTI
• G_T(s) = ΔTe/ΔT*: wzmocnienie statyczne ≈ 0,8–0,93 (a nie 1) to ten sam błąd modelu poślizgu. Para słabo tłumionych biegunów ≈ 50 Hz to oscylacje elektromagnetyczne widoczne na wykresie strumienia.
• G_n(s): prędkość po zmianie momentu – powolna, bo decyduje bezwładność i nachylenie obciążenia.
• L(s): pętla elektromechaniczna. Dzięki sprzężeniu od prędkości (ωs* = ωm + ωsl*) poślizg jest pilnowany i pętla jest stabilna w całym zakresie.

EKSPERYMENTY
• Zmień mnożnik Rr maszyny na 1,3 (nagrzany wirnik): moment spada, bo sterownik liczy poślizg ze złym RR.
• Wyłącz obciążenie: silnik przyspiesza bez końca (sterujemy momentem, nie prędkością).

WNIOSKI INŻYNIERA
To krok w stronę FOC: mamy już osobne wejścia „moment” i „strumień”, ale bez regulacji prądu i bez dokładnego modelu dynamiki. Dlatego odsprzężenie działa tylko w stanie ustalonym.
"""

TEXTS["detune"] = """9.2.1 – ZASADA FOC, WYZNACZANIE KĄTA POLA (CFO) I ODSTROJENIE (DETUNING)

ZASADA
Sterowanie polowo zorientowane (FOC) wymaga, by układ dq sterownika był DOKŁADNIE zgodny z prawdziwym strumieniem maszyny. Wtedy:
   i_sd → strumień (powoli, stała czasowa LM/RR),   i_sq → moment (natychmiast).
Kąt pola wyznacza moduł CFO (Calculation of Field Orientation): bezpośrednio (DFO – czujniki lub obserwator) albo pośrednio (IFO – kąt wału + obliczony kąt poślizgu).

BŁĄD KĄTA Δθ (Fig. 9.8)
Jeśli układ sterownika (niebieski) jest obrócony względem prawdziwego (czerwony), to zadany prąd rozkłada się w maszynie inaczej:
   i_sd(maszyna) = i_sd*·cosΔθ − i_sq*·sinΔθ  (dla dodatniego Δθ – mniej; i_sq* jest kilka razy większe od i_sd*, więc już kilka stopni błędu mocno osłabia pole),
więc strumień ψR = LM·i_sd jest mniejszy niż zadany. Choć i_sq jest trochę większe, moment Te = 3/2·p·ψR·i_sq wychodzi MNIEJSZY od zadanego.

ODSTROJENIE IFO: BŁĘDNA REZYSTANCJA WIRNIKA
W IFO poślizg liczy się z RR_est. Rezystancja wirnika rośnie z temperaturą nawet o 30–50 %! Gdy RR_est ≠ RR, sterownik zadaje zły poślizg, układ dq się „rozjeżdża” i w stanie ustalonym:
• strumień i moment odbiegają od zadanych (wykres prawy górny),
• pojawia się błąd kąta (lewy dolny).
Wykres prawy dolny pokazuje symulację dynamiczną (UFO + model prądowy): dostrojony sterownik vs odstrojony.

ANALOGIA
Kierowca, który widzi drogę przez krzywo założone lusterko: kręci kierownicą „dobrze” według swojego obrazu, ale samochód jedzie inaczej.

EKSPERYMENTY
• k = 1,3 (wirnik zimny, sterownik myśli, że gorący): strumień spada, a moment też jest za mały.
• k = 0,7: strumień ROŚNIE ponad zadany. Grozi to nasyceniem rdzenia.
• Zmieniaj Δθ: przy ±30° błąd momentu sięga kilkudziesięciu procent.

WNIOSKI INŻYNIERA
IFO jest proste i działa przy zerowej prędkości, ale wymaga dobrej estymaty RR (adaptacja on-line, pomiar temperatury). DFO z pętlą regulacji strumienia usuwa błąd w stanie ustalonym, o ile sam pomiar strumienia nie zależy od parametrów.
"""

TEXTS["cfo"] = """9.2.4 – 9.2.8 METODY WYZNACZANIA KĄTA POLA (CFO)

SZEŚĆ METOD Z KSIĄŻKI (tabela w dolnej części wykresu)
• IFO z tachometrem (Fig. 9.11): θM = ∫(ωm + ωsl*)dt. Błąd offsetu prędkości ma znaczenie przy małych prędkościach.
• IFO z enkoderem (Fig. 9.12): θM = θm + ∫ωsl*dt. Działa nawet przy f = 0. Najczęściej stosowane w serwonapędach.
• DFO z hallotronami w szczelinie (Fig. 9.13): bezpośredni pomiar pola. Brak zależności od parametrów, ale silnik musi być specjalny, a hallotrony nie znoszą temperatury powyżej ok. 120 °C.
• DFO z cewkami pomiarowymi (Fig. 9.14): całkowanie indukowanego napięcia daje ψs. Nie działa przy bardzo małej częstotliwości (całkowanie).
• DFO z pomiarem napięcia i prądu – model napięciowy (Fig. 9.15): ψs = ∫(u_s − Rs·i_s)dt, ψM = ψs − LσS·i_s. Bez czujnika na wale, ale wrażliwe na błąd Rs przy małych prędkościach.
• DFO z pomiarem prądu i prędkości – model prądowy (Fig. 9.16): to model wirnika „uruchomiony” w sterowniku. Wrażliwy na RR, ale dobry przy małych prędkościach.

CO POKAZUJĄ WYKRESY
Policzyliśmy w stanie ustalonym, przy obciążeniu znamionowym, jak bardzo każdy z dwóch obserwatorów myli się co do strumienia wirnika, gdy jego parametr jest błędny:
• Model napięciowy: błąd ∝ ΔRs·i_s/ωs. Przy małej prędkości ωs → 0, więc błąd rośnie do ogromnych wartości. Przy dużych prędkościach znika (napięcie jest wtedy dużo większe od spadku Rs·i_s).
• Model prądowy: błąd wynika z błędnej stałej czasowej wirnika. Prawie nie zależy od prędkości (przy stałym momencie poślizg jest stały).
• Zielona linia: obserwator łączony (prądowy przy małych prędkościach, napięciowy przy dużych). Prędkość przełączenia wyznaczana jest automatycznie tam, gdzie błędy obu modeli się zrównują. Tak robią nowoczesne napędy bezczujnikowe.
• Oś prędkości jest logarytmiczna, żeby było widać zakres bardzo małych prędkości.

EKSPERYMENTY
• Ustaw Rs_est = 1,0: model napięciowy staje się idealny w całym zakresie (poza f = 0).
• Ustaw RR_est = 1,0: model prądowy staje się idealny.
• Ustaw Rs_est = 1,5 i RR_est = 1,05: punkt przełączenia przesuwa się w stronę większych prędkości.
• Ustaw moment 0 Nm: przy n → 0 błąd modelu napięciowego rośnie bez ograniczeń (ωs → 0).

WNIOSKI INŻYNIERA
Nie ma jednej idealnej metody. Inżynier wybiera ją zależnie od wymagań: zakresu prędkości, kosztu czujnika, odporności (kabel od enkodera to słaby punkt), temperatury i wymaganej dokładności momentu.
"""

TEXTS["limits"] = """9.3 – 9.4 GRANICE PRACY NAPĘDU FOC I OSŁABIANIE POLA

PŁASZCZYZNA PRĄDU dq (orientacja na strumień wirnika)
Każdy punkt (i_sd, i_sq) to możliwy stan pracy. Ograniczają go:
• MA – okrąg maksymalnego prądu |i_s| ≤ î_max (falownik, nagrzewanie uzwojeń),
• ψR_max – pionowa prosta i_sd ≤ ψR_max/LM (nasycenie rdzenia),
• MF – elipsy maksymalnego napięcia: i_sd² + χi²·i_sq² ≤ (û_max/(ωs·Ls))², gdzie χi = LσS/Ls. Im większa prędkość, tym mniejsza elipsa (napięcie falownika się kończy).
• Hiperbole stałego momentu: Te = 3/2·p·LM·i_sd·i_sq.
• MTPF (max moment na strumień): i_sd = χi·i_sq – punkty styczności hiperbol z elipsami.
• MTPA (max moment na amper): i_sd = i_sq – w silniku indukcyjnym rzadko stosowane (zmiana strumienia jest powolna).

TRZY ZAKRESY PRĘDKOŚCI (Fig. 9.18, 9.19)
1. Zakres podstawowy (do ωs_b): pełny strumień, i_sd = i_sd,max, moment do T_max. Prędkość bazowa wynika z punktu D (przecięcie MA, ψR_max i elipsy).
2. Osłabianie pola: napięcie się skończyło, więc zmniejszamy i_sd (strumień). Moment przy danym prądzie maleje, ale moc może pozostać prawie stała.
3. MTPF: dalsze zwiększanie prędkości tylko wzdłuż linii MTPF; moment szybko spada.

ANALOGIA
Samochód: na pierwszym biegu masz pełną siłę (zakres podstawowy). Na wyższych biegach jedziesz szybciej, ale z mniejszą siłą (osłabianie pola), a moc silnika pozostaje mniej więcej stała.

CO POKAZUJĄ WYKRESY
• Lewy: wszystkie ograniczenia i zielona trajektoria prądu przy wzroście prędkości dla zadanego momentu (czerwona kropka – zaznaczona prędkość).
• Środkowy: i_sd, i_sq, |i_s| w funkcji ωs/ωs_b.
• Prawy: moment i moc. Szara obwiednia to maksimum możliwe przy danej prędkości.

EKSPERYMENTY
• Włącz „dydaktyczne χi = 0,3”: elipsy stają się „grube” i dobrze widać MTPF (tak jak w książce, gdzie χi wybrano duże celowo).
• Zwiększ û_max (wyższe U_DC): rośnie prędkość bazowa.
• Zwiększ î_max: rośnie moment, ale prędkość bazowa maleje.

WNIOSKI INŻYNIERA
Dla rzeczywistego silnika χi ≈ 0,087 jest małe, więc elipsy są bardzo wydłużone i MTPF zaczyna się dopiero przy kilkukrotnej prędkości bazowej. Silniki trakcyjne (tramwaje, EV) pracują długo w obszarze stałej mocy.
"""

TEXTS["foc1"] = """9.5 + TUTORIAL 9.7.3 – STEROWANIE UFO Z MODELEM PRĄDOWYM I ENKODEREM

SCHEMAT (Fig. 9.20, 9.27)
• Sterownik UFO (Fig. 9.10) to ODWRÓCONY model maszyny z Fig. 8.22. Z zadanego momentu T* i strumienia ψM* liczy prądy i_sd*, i_sq* i poślizg ωsl*. Tutaj zaimplementowany cyfrowo (różniczkowanie wsteczne, krok 0,1 ms).
• CFO = IFO z enkoderem (Fig. 9.12): θM* = θ_wału·p + ∫ωsl* dt.
• Maszyna to model zasilany PRĄDEM (idealny regulator prądu), parametry z Tab. 8.3.
• Scenariusz: T* = 0 → 113 Nm przy t = 0,5 s, zmiana na −113 Nm przy t = 1,5 s, filtr momentu 20 ms, obciążenie wentylatorowe, ψM* = 1,83 Vs.

CO POKAZUJĄ WYKRESY (Fig. 9.28, 9.29)
• Moment silnika idealnie pokrywa się z zadanym (po filtrze): sterownik dokładnie „odwraca” maszynę.
• Strumień ψM pozostaje stały przy skokach momentu. Jest to pełne odsprzężenie toru momentu od toru strumienia, czyli to, czego V/f nie potrafi.
• Prądy: i_sq* skacze proporcjonalnie do momentu. Przy orientacji na stojan (a max) i_sd* też się zmienia, bo okrąg Heylanda wymaga większego i_d przy obciążeniu.
• Poślizg: przy a = Ls/Lm widać impulsy (pochodna i_sq) – Fig. 9.29.
• Prędkość: dochodzi do ok. 1465 obr/min, a po zmianie znaku momentu silnik hamuje i rozpędza się w drugą stronę.

ANALIZA LTI
• G_T(s) = 1/(τT·s + 1): przy idealnym dostrojeniu cały „silnik + sterownik” to tylko filtr momentu. Biegun w −1/τT, pasmo ≈ 1/τT.
• G_n(s): moment → prędkość, czyli masa i obciążenie wentylatorowe zlinearyzowane wokół 1465 obr/min (tłumienie B = 2kΩ0).
• G_ψR(s) = LM/(τR·s + 1), τR = LM/RR ≈ 0,5 s: strumień wirnika jest BARDZO wolny. Dlatego FOC trzyma strumień stały, a moment zmienia przez i_q.
• L(s): projekt pętli prędkości z regulatorem PI (tego nie ma w książce; to typowy następny krok inżyniera). Zmieniaj Kp, Ki i patrz na zapas fazy oraz przeregulowanie.

EKSPERYMENTY
• Przesuwaj a od wirnika do stojana: moment i strumień ψM się nie zmieniają (sterownik i maszyna liczone są spójnie), zmieniają się prądy i poślizg.
• Odstrój Rr_est = 1,3: moment i strumień odbiegają od zadanych (detuning).
• Skróć filtr momentu do 1 ms przy orientacji na stojan: rośnie impuls poślizgu.
• Ustaw obciążenie 0: prędkość rośnie bardzo szybko (113 Nm / 0,1 kg·m²).

WNIOSKI INŻYNIERA
Dzięki regulacji prądu i odwróceniu modelu silnik indukcyjny reaguje na zmianę momentu w milisekundach. Porównaj z oscylacjami V/f (Fig. 9.6).
"""

TEXTS["foc2"] = """9.6 + TUTORIAL 9.7.4 – UFO Z REGULATOREM PRĄDU I MASZYNĄ ZASILANĄ NAPIĘCIEM

CO NOWEGO?
W rzeczywistości falownik nie daje prądu, tylko NAPIĘCIE. Prąd musi więc ustawić regulator prądu (Fig. 9.22):
• Pomiar prądów → transformacja do układu dq sterownika (kąt θM* z CFO, tu: tachometr ∫(ωm + ωsl*)dt).
• Dwa regulatory PI (oś d i q), próbkowanie Ts = 100 µs, jedna próbka opóźnienia obliczeniowego.
• Kompensacja SEM (feed-forward): do u_q dodajemy ψM*·ωm. Jak w książce używamy ωm (zmierzone) zamiast ωs; resztę błędu „wyłapuje” całkowanie regulatora.
• Ograniczenie napięcia |u| ≤ U_DC/√3 (falownik) + anti-windup.
• Maszyna: pełny model napięciowy (Fig. 8.15), parametry z Tab. 8.3. Domyślnie orientacja na strumień stojana (a = Ls/Lm), ψM* = ψs* = 1,83 Vs, filtr momentu 2 ms (Fig. 9.23).

DOBÓR REGULATORA (projekt inżynierski)
Obiekt dla szybkich zmian prądu: G(s) = 1/(LσS·s + R), gdzie LσS = 23,6 mH, a R ≈ Rs + RR (strumień wirnika „przepuszcza” szybkie zmiany przez RR). Opóźnienie Td = 1,5·Ts (obliczenia + PWM).
Kryterium modułowe („optimum techniczne”): Ti = LσS/R (zero PI skraca biegun obiektu), Kp = LσS/(2·Td), Ki = R/(2·Td). Daje to zapas fazy ≈ 60–65° i pasmo ≈ 1/(2Td) ≈ 3300 rad/s.

CO POKAZUJĄ WYKRESY
• Moment śledzi T* w ciągu kilku ms. Prądy i_sd, i_sq pokrywają się z zadanymi (regulator działa).
• |ψs| jest stały (orientacja na stojan, ψM = ψs), a |ψR| lekko się zmienia z momentem.
• Napięcie: rośnie z prędkością (SEM). Gdy dotknie granicy U_DC/√3, regulator traci zapas i prąd przestaje nadążać.

ANALIZA LTI
• L(s) pętli prądowej: Bode z zapasami PM/GM, Nyquist i odpowiedź skokowa zamkniętej pętli prądu. Opóźnienie Td „obraca” fazę o −ω·Td, co ogranicza możliwe pasmo.
• G_i(s) – sam obiekt (uzwojenie): jeden biegun −R/LσS ≈ −46 1/s (τ ≈ 22 ms).
• G_z(s) = G/(1+L): jak mocno regulator tłumi zakłócenie (np. błąd kompensacji SEM). Im wyższe Kp, Ki, tym mniejszy wpływ.

EKSPERYMENTY
• Kp/Kp_opt = 2: zapas fazy spada, pojawia się przeregulowanie prądu (zobacz Analizę LTI).
• Kp/Kp_opt = 2,5 i Ts = 300 µs: pętla traci stabilność (ujemny PM), prądy oscylują.
• Wyłącz feed-forward: przy rosnącej prędkości pojawia się chwilowy błąd prądu q, który usuwa dopiero całkowanie.
• Zmniejsz U_DC do 700 V: przy wysokiej prędkości napięcie się kończy i moment spada.

WNIOSKI INŻYNIERA
Realny napęd FOC to kaskada: szybka pętla prądu (≈ kHz), nad nią sterownik UFO (moment/strumień), nad nim ewentualnie pętla prędkości (≈ 10 Hz). Każda wewnętrzna pętla musi być kilka razy szybsza od zewnętrznej.
"""

# ---------------------------------------------------------------------------
# objasnienia do poszczegolnych transmitancji (Analiza LTI)
# ---------------------------------------------------------------------------
LTI_INFO = {
    "t84_psi": """• Jeden biegun rzeczywisty s = −Rr/Lm: tor strumienia to człon inercyjny I rzędu. Po skoku i_d strumień osiąga 63 % wartości po czasie τ i ≈ 98 % po 4τ.
• Bode: płasko do ω ≈ 1/τ (pasmo), potem −20 dB/dekadę, faza od 0° do −90°. Szybkie zmiany i_d są „tłumione”, więc strumienia nie da się zmienić szybko.
• Nyquist: półokrąg w prawej półpłaszczyźnie od Lm (ω=0) do 0 (ω→∞). Typowy obraz członu I rzędu.
• To jest obiekt, a nie pętla: jest stabilny sam z siebie (biegun po lewej).""",
    "t84_w": """• Biegun w zerze: człon całkujący. Prędkość jest „sumą” momentu w czasie (2. zasada dynamiki).
• Bode: prosta −20 dB/dekadę, stała faza −90°. Brak wzmocnienia statycznego (∞).
• Nyquist: półprosta wzdłuż ujemnej osi urojonej (dla ω→0 wartości dążą do −j∞).
• Odpowiedź skokowa: rampa (prędkość rośnie liniowo), dokładnie jak na wykresie ωm po skoku i_q.
• Dlatego do sterowania prędkością potrzebny jest regulator (sprzężenie zwrotne). Zobacz kolejną transmitancję L(s).""",
    "t84_loop": """• L(s) = (Kp s + Ki)/s · K/(J s): dwa całkowania (podwójny biegun w zerze) i zero PI w −Ki/Kp.
• Zero PI musi leżeć wyraźnie poniżej pulsacji odcięcia ωc, wtedy podnosi fazę i daje dodatni zapas fazy. Jeśli Ki jest za duże (zero blisko ωc lub wyżej), zapas fazy maleje i pojawiają się oscylacje prędkości.
• Reguła „optimum symetrycznego”: ωc ≈ √(Ki/Kp)·(kilka razy). Spróbuj ustawić Ki/Kp ≈ ωc/4.
• Odpowiedź układu zamkniętego ma przeregulowanie spowodowane zerem PI. W praktyce stosuje się filtr wartości zadanej.""",
    "t85_psi": """• a = Lm/Lr (min): LσR = 0, zero znika, zostaje czysty człon I rzędu (jak w modelu bez rozproszeń). Pełne odsprzężenie.
• a > Lm/Lr: pojawia się zero s = −R_R/LσR (bardzo daleko w lewo) i biegun −R_R/L_R. Zero sprawia, że strumień ψM reaguje NATYCHMIAST na skok i_d częścią L_M·LσR/L_R (skok na początku odpowiedzi skokowej), a dalej dochodzi wykładniczo.
• Interpretacja fizyczna: przy orientacji na stojan część zmiany prądu od razu zmienia strumień stojana przez indukcyjność rozproszenia.
• Wzmocnienie statyczne = L_M: w stanie ustalonym ψM = L_M·i_d, niezależnie od a.""",
    "t85_sl": """• Licznik (LσR·s + R_R) zawiera człon różniczkujący dla a > Lm/Lr. Skok i_q daje impuls poślizgu, którego wysokość ogranicza tylko rampa (filtr 1/(Tr·s+1)).
• Przy a = Lm/Lr licznik jest stały: poślizg skacze do nowej wartości bez impulsu.
• Bode: przy dużych ω amplituda rośnie (zero), więc układ wzmacnia szybkie zmiany. Dlatego w sterowniku UFO stosuje się ogranicznik narastania i_q*.
• Zero s = −R_R/LσR leży na tej samej pulsacji co poślizg krytyczny (okrąg Heylanda), co nie jest przypadkiem.""",
    "grid_l": """• Wzmocnienie statyczne (ujemne) to nachylenie charakterystyki mechanicznej: o ile obr/min spada prędkość na każdy dodatkowy Nm obciążenia.
• Dominująca para biegunów zespolonych (ok. 13 Hz) to drgania elektromechaniczne: masa wirnika na „sprężynie” pola magnetycznego. To one powodują kołysanie momentu po skoku obciążenia (Fig. 8.47).
• Szybka para ok. 50 Hz (≈ ωs) to przebiegi elektromagnetyczne w uzwojeniach stojana, widoczne zaraz po włączeniu do sieci.
• Odpowiedź skokowa: po dołożeniu 1 Nm prędkość spada z lekkim przeregulowaniem i ustala się po ok. 0,3 s.""",
    "grid_L": """• Pętlę tworzy: Δprędkość → zmiana poślizgu → zmiana momentu (dynamika pola G_Te,Ω) → przyspieszenie masy 1/(J·s) → Δprędkość. Znak minus: wzrost prędkości zmniejsza moment (ujemne sprzężenie), co stabilizuje układ.
• Przy zasilaniu z sieci (stałe U i f) zapas fazy jest dodatni, więc układ zamknięty (prawdziwy silnik) jest stabilny. Bieguny T odpowiadają biegunom pełnego modelu.
• Ta sama pętla przy V/f i małej częstotliwości może mieć ujemny zapas fazy (zakładka „V/f prosty”).""",
    "vf_n": """• G_n(s) mówi, jak prędkość wału podąża za zmianą prędkości zadanej (częstotliwości). Wzmocnienie statyczne ≈ 1: w stanie ustalonym bez obciążenia wał „dogania” pole.
• Najważniejsza jest para biegunów ok. 10–14 Hz. Jej część rzeczywista decyduje o stabilności. Dla n*_s ≈ 400–550 obr/min (Rr × 0,9) jest DODATNIA, więc odpowiedź skokowa narasta oscylacyjnie: niestabilność z Fig. 9.4.
• Przesuń suwak punktu linearyzacji na 300 i 800 obr/min: bieguny wracają na lewą stronę.""",
    "vf_l": """• Wpływ zakłócenia momentu obciążenia na prędkość. W niestabilnym punkcie nawet malutkie zakłócenie (np. 1 Nm prostokąt, harmoniczne żłobkowe) wzbudza rosnące drgania.
• Bode: rezonansowy „pik” przy częstotliwości drgań elektromechanicznych; im wyższy pik, tym słabsze tłumienie.""",
    "vf_L": """• Kryterium Nyquista: L(s) nie ma niestabilnych biegunów (P = 0), więc układ zamknięty jest stabilny wtedy i tylko wtedy, gdy krzywa NIE obejmuje punktu −1.
• W strefie niestabilności (np. 450 obr/min) zapas fazy jest ujemny: krzywa przechodzi po złej stronie punktu −1.
• Fizycznie: przy niskich częstotliwościach opóźnienie, z jakim strumień wirnika reaguje na zmianę poślizgu, zamienia tłumienie na „anty-tłumienie”. Moment zaczyna wtedy pchać wirnik w rytm jego własnych drgań.""",
    "vf2_T": """• Wzmocnienie statyczne < 1 to ten sam błąd, który daje 105 Nm zamiast 113 Nm: wzór na poślizg pomija LσR.
• Słabo tłumiona para biegunów ≈ 47–50 Hz (ζ ≈ 0,08): oscylacje strumienia i momentu po zmianie T*, widoczne na wykresie |ψs|. V/f nie odsprzęga strumienia i momentu w stanach przejściowych.
• Porównaj z G_T sterownika UFO (zakładka „UFO + model prądowy”): tam jest czysty filtr I rzędu, bez drgań.""",
    "vf2_n": """• Prędkość po zmianie momentu: dominuje wolny biegun mechaniczny (bezwładność + nachylenie obciążenia wentylatorowego). Stała czasowa rzędu sekund.
• Szybkie bieguny elektryczne są prawie niewidoczne w odpowiedzi prędkości, bo masa je „filtruje”.""",
    "vf2_L": """• Sterownik ze sprzężeniem od prędkości (ωs* = ωm + ωsl*) utrzymuje stały poślizg, więc pętla elektromechaniczna jest bardzo dobrze tłumiona i stabilna w całym zakresie prędkości.
• W odróżnieniu od prostego V/f nie ma tu strefy niestabilności: to zaleta pomiaru prędkości.""",
    "foc_T": """• Idealny sterownik UFO „skraca się” z modelem maszyny: zostaje tylko filtr momentu zadanego 1/(τT·s+1).
• Biegun s = −1/τT: dla τT = 20 ms moment osiąga 63 % po 20 ms i 98 % po 80 ms, bez przeregulowania.
• To idealna sytuacja: regulator prądu jest nieskończenie szybki, a parametry dokładnie znane. Odstrojenie (Rr_est ≠ Rr) zmieniłoby wzmocnienie statyczne (zobacz zakładkę „Odstrojenie FOC”).""",
    "foc_n": """• Moment → prędkość: filtr momentu i masa z tłumieniem B = 2·k·Ω0 (linearyzacja wentylatora T = k·Ω²).
• Wolny biegun −B/J to mechaniczna stała czasowa J/B (rzędu 1 s). Szybki biegun −1/τT to filtr momentu.
• Przy obciążeniu = 0 biegun mechaniczny przechodzi do zera (całkowanie). Prędkość rośnie bez końca, jak w symulacji bez obciążenia.""",
    "foc_psi": """• τR = LM/RR ≈ 0,5 s: strumień wirnika jest około 25 razy wolniejszy od filtra momentu. Gdyby moment regulować przez strumień (jak w MTPA), napęd byłby bardzo powolny.
• Dlatego FOC utrzymuje strumień stały (lub zmienia go powoli przy osłabianiu pola), a moment reguluje szybkim prądem i_sq.
• Sterownik UFO może przyspieszyć magnesowanie, chwilowo „przeforsowując” i_d (wyraz LσR·di_d/dt i (L_R/L_M)dψ/dt). Kosztem jest duży prąd chwilowy.""",
    "foc_L": """• Pętla prędkości: PI · filtr momentu · mechanika. Typowy projekt: pasmo pętli prędkości 5–10 razy mniejsze niż pasmo toru momentu (1/τT).
• Spróbuj: Kp = 0,5 Nm/(obr/min), Ki = 5 daje dobry zapas fazy. Zwiększ Kp 5× i zmniejsz τT: pętla przyspiesza; zwiększ Ki 10×: pojawiają się oscylacje.
• Ograniczenie: realny moment jest ograniczony prądem (MA) – przy dużych skokach prędkości regulator nasyca się i analiza liniowa przestaje obowiązywać.""",
    "cur_L": """• L(s) = (Kp + Ki/s)·e^(−s·Td)/(LσS·s + R). Przy Ki/Kp = R/LσS zero PI skraca biegun obiektu i L ≈ (1/(2Td·s))·e^(−s·Td).
• Wtedy ωc ≈ 1/(2Td) i PM ≈ 90° − ωc·Td·(180/π) ≈ 61°. Odpowiedź skokowa ma ok. 5–10 % przeregulowania, a czas narastania wynosi ok. 3–5·Td.
• Opóźnienie e^(−sTd) nie zmienia |L|, ale obraca fazę liniowo z ω. To ono ogranicza możliwą szybkość regulatora (GM ≈ 10 dB dla nastaw optymalnych).
• Odpowiedź skokowa liczona z aproksymacją Padé 2. rzędu opóźnienia (dokładna w paśmie regulacji).""",
    "cur_G": """• Obiekt: uzwojenie (indukcyjność przejściowa LσS w szereg z R ≈ Rs + RR). Biegun −R/LσS: τ ≈ 22 ms.
• Bez regulatora prąd po skoku napięcia narastałby 22 ms. Regulator PI skraca to do ułamka milisekundy.""",
    "cur_z": """• G_z(s) = G/(1+L): jak zakłócenie napięciowe (np. niedokładnie skompensowana SEM ψ·ω) przenosi się na błąd prądu.
• Przy małych ω wynik jest bliski 0 (całkowanie w PI usuwa stały błąd). Maksimum leży w paśmie regulatora: tam zakłócenia przechodzą najsilniej.
• Kompensacja feed-forward ψM·ωm usuwa większość zakłócenia, zanim w ogóle trafi do regulatora.""",
}
