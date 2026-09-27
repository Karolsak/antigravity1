# Modele zaawansowane maszyn i stany nieustalone maszyn DC – wyjaśnienia

_Wygenerowano z aplikacji `advanced_models_dc_transients_tkinter.py` (opcja `--export-md`). Te same teksty są widoczne w zakładkach aplikacji._


---

# Zakładka: Start – mapa materiału


## Jak korzystać z aplikacji


Aplikacja ilustruje rozdziały 7 („Zaawansowane modele maszyn elektrycznych”) i 8 („Stany nieustalone maszyn prądu stałego z komutatorem”). Każda zakładka ma trzy części:
- po lewej – suwaki z parametrami (możesz też wpisać liczbę i nacisnąć Enter) oraz wyniki liczbowe,
- u góry po prawej – wykresy (pasek narzędzi pozwala powiększać i zapisywać rysunek),
- na dole – wyjaśnienie krok po kroku, najpierw „po ludzku”, potem wzory i komentarz inżyniera (⚙).



## O co w ogóle chodzi? (wersja dla licealisty)


Silnik elektryczny w podręczniku do fizyki działa „w stanie ustalonym”: stałe napięcie, stała prędkość, stały prąd. W prawdziwym świecie silnik jest włączany, hamowany, obciążany, zasilany z falownika, który przełącza napięcie tysiące razy na sekundę. Wtedy prądy i prędkość się ZMIENIAJĄ – to są stany nieustalone (przejściowe).

Żeby je policzyć, inżynier potrzebuje modelu – zestawu równań różniczkowych („jak szybko zmienia się prąd, jeśli…”). Rozdział 7 pokazuje, jak zbudować taki model dla każdej maszyny, a rozdział 8 liczy go w praktyce dla najprostszej maszyny – silnika prądu stałego.



### Trzy najważniejsze idee

- Stała czasowa τ – „jak długo coś się rozpędza”. Obwód z cewką L i opornikiem R: τ = L/R. Po czasie τ zmiana osiąga ok. 63%, po 3τ – 95%, po 5τ – 99%.
- Transformacja dq (Parka) – patrzymy na maszynę z obracającego się „karuzelowego” układu współrzędnych. Wtedy prądy przemienne stają się stałymi liczbami, a równania mają stałe współczynniki. To jak zdjęcie karuzeli zrobione z samej karuzeli – konie stoją w miejscu.
- Wartości własne (bieguny) – liczby, które mówią, czy odpowiedź układu będzie gładka (liczby rzeczywiste ujemne), czy z oscylacjami (liczby zespolone).



## Wykresy na tej stronie


Lewy wykres: porównanie stałych czasowych z przykładów (skala logarytmiczna!). Widać, że w jednej maszynie żyją zjawiska różniące się szybkością nawet 2000 razy – obwód twornika reaguje w ułamku milisekundy, a obwód wzbudzenia w pół sekundy.

Prawy wykres: zakresy częstotliwości, w których stosuje się różne modele (Rozdz. 7.11): do ok. 400 Hz model R-L-E, od 20 kHz model pojemnościowy (wysokiej częstotliwości), pomiędzy – model uniwersalny.

> ⚙ Inżynier: dobór modelu to zawsze kompromis „dokładność kontra czas obliczeń”. Model dq wystarcza do sterowania i większości stanów przejściowych; MES (metoda elementów skończonych) do projektowania szczegółów; model HF do kompatybilności elektromagnetycznej (EMC) i prądów łożyskowych.


---

# Zakładka: 7.2–7.4 Model dq i SEM


## Po co model dq? (7.1–7.2)


Wyobraź sobie karuzelę. Stojąc obok, widzisz konia, który ciągle zmienia położenie – jego współrzędne x i y to sinusoidy. Gdy wskoczysz na karuzelę, koń stoi nieruchomo obok ciebie. Model dq to właśnie „wskoczenie na karuzelę”: patrzymy na maszynę z układu osi d (oś bieguna, „direct”) i q (oś poprzeczna, „quadrature”, 90° dalej), który obraca się z prędkością ωb.

Autor proponuje model FIZYCZNY: maszynę z uzwojeniami komutatorowymi, których szczotki leżą w osiach d i q i obracają się z prędkością ωb. Dzięki temu pole każdego uzwojenia zawsze stoi w osi szczotek, a indukcyjności NIE zależą od kąta wirnika. Równania mają stałe współczynniki – ogromne uproszczenie.

### Jaką prędkość osi wybrać? (Tabela 7.1)

- maszyna synchroniczna (SM): ωb = ωr – osie przyklejone do wirnika, bo wirnik ma wystające bieguny (asymetrię magnetyczną),
- maszyna indukcyjna (IM): szczelina równomierna, więc wolno wybrać dowolne ωb; najczęściej 0, ωr lub ω1,
- maszyna DC: ωb = 0 – asymetria (bieguny, magnes) jest w stojanie, szczotki stoją.

Kliknij ▶ Start – na lewym wykresie osie d-q obracają się zgodnie z wybraną opcją, wirnik (pomarańczowa szprycha) i pole wirujące (zielona strzałka) też się kręcą.



## Dwa rodzaje napięcia indukowanego (7.3)


Z prawa Faradaya: SEM = −dΨ/dt. Strumień skojarzony z cewką Ψ(θ, t) zależy od czasu I od położenia, więc pochodna ma dwie części:
    e = −∂Ψ/∂t − (∂Ψ/∂θ)·(dθ/dt) = e_p + e_m
- e_p – napięcie pulsacyjne (transformatorowe): strumień „pompuje” się w czasie, cewka stoi. Tak działa transformator.
- e_m – napięcie rotacji (ruchu): strumień jest stały, ale cewka się przez niego przesuwa z prędkością ω. Tak działa prądnica rowerowa.

Dla Ψ = Ψm(t)·cos θ dostajemy:
    e_p = −(dΨm/dt)·cos θ         e_m = +Ψm·ω·sin θ

Ważny wniosek (Równ. 7.3): SEM rotacji w osi q pochodzi od strumienia osi d i odwrotnie – osie „rozmawiają ze sobą” tylko przez ruch. I tylko SEM rotacji wytwarza moment!

Na wykresach: górny prawy – strumień Ψ(t); dolny lewy – e_p, e_m, suma oraz (kropki) −dΨ/dt policzone numerycznie. Kropki leżą dokładnie na sumie – to dowód, że rozkład jest poprawny. Dolny prawy – wartości skuteczne obu składników.

> 💡 Spróbuj: ustaw n = 0 – zostaje tylko e_p (transformator). Ustaw m = 0 – zostaje tylko e_m (prądnica). Zwiększ fp – rośnie e_p.




## Silnik DC z magnesem jako model dq (7.4)


Wystarczy zostawić jedno uzwojenie wirnika w osi q (twornik) i magnes w osi d stojana. Równanie:
    V_qr = R_a·I_qr + L_a·dI_qr/dt + ω_r·Ψ_dr,    T_e = p1·Ψ_dr·I_qr

Ψ_dr to strumień od magnesu, „widziany” przez fikcyjne uzwojenie w osi d. Stąd klasyczne wzory E = k·Φ·n i T = k·Φ·I. „Maszyna DC to uproszczony model dq!”

> ⚙ Inżynier: w maszynie DC komutator robi fizycznie to, co w napędzie z falownikiem robi procesor (transformacja Parka w czasie rzeczywistym). Dlatego sterowanie wektorowe silników AC nazywa się „robieniem z silnika AC silnika DC”.


---

# Zakładka: 7.5 SM/PMSM (Zad. 7.3)


## Maszyna synchroniczna w osiach wirnika (7.5)


W maszynie synchronicznej wirnik kręci się dokładnie z prędkością pola. Przyklejamy więc osie dq do wirnika (ωb = ωr): oś d – w stronę bieguna (magnesu), oś q – 90° elektrycznych dalej. W takim układzie w stanie ustalonym wszystkie wielkości są STAŁE (prąd „stały” zamiast sinusoidy) – idealnie do sterowania.

Zadanie 7.3 każe usunąć z wirnika klatkę i uzwojenie wzbudzenia i wstawić magnesy w osi d. Zamiast L_dm·I_F piszemy Ψ_PM. Równania upraszczają się do:
    V_d = R_s·I_d + L_d·dI_d/dt − ω_r·L_q·I_q
    V_q = R_s·I_q + L_q·dI_q/dt + ω_r·(L_d·I_d + Ψ_PM)
    Ψ_d = L_d·I_d + Ψ_PM,     Ψ_q = L_q·I_q
    T_e = 3/2·p1·(Ψ_d·I_q − Ψ_q·I_d) = 3/2·p1·[Ψ_PM·I_q + (L_d − L_q)·I_d·I_q]

Czynnik 3/2 wynika z transformacji Parka z niezmiennikiem amplitudy (patrz zakładka 7.9–7.10).

### Dwa „silniki w jednym”

- moment magnesu 3/2·p1·Ψ_PM·I_q – jak siła na przewód w polu magnesu (siła Lorentza),
- moment reluktancyjny 3/2·p1·(L_d−L_q)·I_d·I_q – żelazo „chce” ustawić się tak, by strumień miał najłatwiejszą drogę (jak spinacz przyciągany przez magnes).

Gdy L_q > L_d (magnesy zagłębione – IPMSM, zadania 7.2 i 7.4), ujemne I_d (γ > 0) dodaje moment reluktancyjny. Istnieje kąt γ dający największy moment przy danym prądzie – MTPA (Maximum Torque Per Ampere) – zaznaczony na wykresie.

### Wykresy

- lewy górny: moment w funkcji kąta prądu γ – składowa magnesu, reluktancyjna i suma,
- prawy górny: wykres wektorowy w osiach dq – prąd I, strumień Ψ_s, napięcie V (przeskalowane),
- dolne: stan nieustalony po nagłym przyłożeniu napięć V_d, V_q (przy stałej prędkości). Prądy dążą do wartości zadanych z oscylacjami o częstotliwości zbliżonej do ω_r – to sprzężenie osi przez człony ω_r·L·I (SEM rotacji z zakładki 7.2–7.4!).

> 💡 Spróbuj: ustaw L_d = L_q (magnesy powierzchniowe) – moment reluktancyjny znika, MTPA to γ = 0. Zmniejsz R_s – oscylacje zanikają wolniej.


> ⚙ Inżynier: w praktyce regulator prądu „odsprzęga” osie, odejmując człony ω_r·L·I (tzw. feed-forward), dlatego w napędach nie widać tych oscylacji. Napięcie |V| rośnie z prędkością – przy granicy napięcia falownika trzeba zwiększyć −I_d (osłabianie pola).


---

# Zakładka: 7.6 IM w osiach dq


## Silnik indukcyjny w osiach dq (7.6)


Silnik indukcyjny („klatkowy”) ma symetryczny stojan i symetryczny wirnik, a szczelina jest równa dookoła. Dlatego – jak pisze autor – prędkość osi ωb możemy wybrać DOWOLNIE, a indukcyjności i tak będą stałe. Równania (7.21–7.22) w zapisie strumieniowym:
    dΨ_ds/dt = V_ds − R_s·I_ds + ω_b·Ψ_qs         dΨ_qs/dt = V_qs − R_s·I_qs − ω_b·Ψ_ds
    dΨ_dr/dt = −R_r·I_dr + (ω_b − ω_r)·Ψ_qr     dΨ_qr/dt = −R_r·I_qr − (ω_b − ω_r)·Ψ_dr
    Ψ_s = L_sl·I_s + L_m·(I_s + I_r),   Ψ_r = L_rl·I_r + L_m·(I_s + I_r)
    T_e = 3/2·p1·(Ψ_ds·I_qs − Ψ_qs·I_ds),     (J/p1)·dω_r/dt = T_e − T_L

Człony ω_b·Ψ to właśnie SEM rotacji: stojan „porusza się” względem osi z prędkością −ω_b, a wirnik z prędkością ω_r − ω_b.

### Co pokazuje symulacja?


Silnik 4 kW, 400 V, 50 Hz załączamy bezpośrednio do sieci (rozruch bezpośredni), a w chwili t_L dokładamy obciążenie.
- Lewy górny: prądy i_d, i_q w wybranych osiach. W osiach synchronicznych (ω_b = ω1) po rozruchu są STAŁE – to „prąd stały” na karuzeli. W osiach stojana (ω_b = 0) są sinusoidami 50 Hz, w osiach wirnika – sinusoidami o częstotliwości poślizgu.
- Prawy górny: prąd fazy A – taki sam niezależnie od wybranych osi (bo to prawdziwy prąd w kablu!). Widać 5–7-krotny prąd rozruchowy.
- Lewy dolny: moment – linia ciągła dla wybranych osi, przerywana dla osi synchronicznych. Pokrywają się – moment jest wielkością fizyczną, nie zależy od „punktu widzenia”. Oscylacje na początku to efekt składowej stałej strumienia po załączeniu.
- Prawy dolny: prędkość – rozpędzanie, a potem spadek po obciążeniu (poślizg rośnie).

> 💡 Zmień „osie” w liście – zmienia się tylko lewy górny wykres! Zwiększ J – rozruch trwa dłużej. Zmniejsz V – maleje moment (∝ V²).


> ⚙ Inżynier: osie ω_b = 0 stosuje się do badania softstartów i falowników (Podsumowanie 7.12), ω_b = ω1 do sterowania polowo-zorientowanego (wszystko jest DC → proste regulatory PI), ω_b = ω_r dla maszyn pierścieniowych zasilanych od strony wirnika (np. generator DFIG w elektrowni wiatrowej).


---

# Zakładka: 7.7 Nasycenie


## Nasycenie – żelazo „się zapycha” (7.7)


Żelazo wzmacnia pole magnetyczne tysiące razy, ale tylko do pewnego momentu. Gdy wszystkie „domeny magnetyczne” są już ustawione, dalsze zwiększanie prądu prawie nie zwiększa strumienia – jak gąbka, która nasiąkła wodą. Krzywa Ψ(i) się wygina (lewy górny wykres).

Model z książki: każda oś ma własną krzywą magnesowania Ψ*_dm(i_m), Ψ*_qm(i_m), ale ZALEŻY ONA TYLKO od wypadkowego prądu magnesującego:
    i_m = √(i_dm² + i_qm²),   i_dm = i_d + i_dr + i_F,   i_qm = i_q + i_qr
    L_dm(i_m) = Ψ*_dm(i_m)/i_m     (indukcyjność „cięciwowa”, do stanu ustalonego)

### Stan nieustalony – trzy różne indukcyjności


Przy zmianie prądów potrzebna jest pochodna strumienia (Równ. 7.27–7.31):
    dΨ_dm/dt = L_dmt·di_dm/dt + L_dqm·di_qm/dt
    dΨ_qm/dt = L_qdm·di_dm/dt + L_qmt·di_qm/dt
    L_dmt = L_dm + (i_dm²/i_m)·dL_dm/di_m        L_qmt = L_qm + (i_qm²/i_m)·dL_qm/di_m
    L_dqm = (i_dm·i_qm/i_m)·dL_dm/di_m
- L_dmt, L_qmt – indukcyjności „przejściowe” (styczne w kierunku danej osi),
- L_dqm – sprzężenie skrośne: zmiana prądu w osi q zmienia strumień w osi d, choć osie są prostopadłe! Pojawia się TYLKO przy nasyceniu (dL/di ≠ 0) i gdy oba prądy są niezerowe.

### Wykresy

- lewy górny: krzywa magnesowania, punkt pracy, cięciwa (L_dm) i styczna (L_dyn = dΨ/di),
- prawy górny: L_dm i L_dyn w funkcji prądu – obie spadają, styczna szybciej,
- lewy dolny: L_dmt, L_qmt, L_dqm w funkcji kąta wektora i_m (przy stałym |i_m|) – przy 0° i 90° sprzężenie znika, największe (co do modułu) jest przy 45°. L_dqm wychodzi UJEMNE, bo przy nasyceniu dL_dm/di_m < 0,
- prawy dolny: mapa sprzężenia skrośnego L_dqm(i_dm, i_qm).

> 💡 Ustaw małe prądy (np. 1 A) – nasycenia prawie nie ma, L_dqm ≈ 0. Zwiększ prądy do 15 A – sprzężenie rośnie.


> ⚙ Inżynier: nowoczesne maszyny (np. silniki samochodów elektrycznych) pracują „na granicy” nasycenia, więc pominięcie go daje błędy momentu rzędu 10–30%. W sterownikach używa się tablic L(i_d, i_q) z pomiarów lub MES. Przy małych sygnałach AC (testy postojowe) zamiast L_dmt występują indukcyjności przyrostowe z lokalnej pętli histerezy, μ_i ≈ (120–150)·μ0. Uwaga: dla kq ≠ 1 wzory dają L_dqm ≠ L_qdm – książka zakłada tu wzajemność, która ściśle zachodzi dla jednej wspólnej krzywej (kq = 1).


---

# Zakładka: 7.8 Naskórkowość


## Efekt naskórkowości (7.8) – prąd ucieka na powierzchnię


Prąd przemienny w grubym przewodzie nie płynie równomiernie. Zmienne pole magnetyczne indukuje w przewodzie prądy wirowe, które „wypychają” prąd w stronę powierzchni (w pręcie klatki – w stronę szczeliny, do góry żłobka). Im wyższa częstotliwość, tym cieńsza „skórka”, w której płynie prąd. Grubość tej skórki to głębokość wnikania:
    δ = √(2/(ω·μ0·σ))       (dla miedzi przy 50 Hz: ok. 1 cm)

Skutek dla pręta klatki silnika:
- rezystancja ROŚNIE: R_r(ω) = k_R·R_dc, k_R > 1 (prąd ma węższą drogę),
- indukcyjność rozproszenia MALEJE: L_rl(ω) = k_X·L_dc, k_X < 1.

Przy rozruchu (poślizg s = 1, częstotliwość w wirniku 50 Hz) rezystancja wirnika jest duża → duży moment rozruchowy i mniejszy prąd. Przy pracy normalnej (1–3 Hz w wirniku) rezystancja jest mała → wysoka sprawność. Konstruktorzy robią to celowo – silniki z głębokim żłobkiem lub z podwójną klatką.

### Jak to wstawić do modelu dq?


Model dq lubi stałe parametry, a tu R i L zależą od częstotliwości. Sztuczka z Rys. 7.6: zastępujemy jeden obwód o zmiennych parametrach kilkoma (2–3) obwodami R-L o STAŁYCH parametrach połączonymi równolegle. Dobieramy je (regresją, jak tu – metodą najmniejszych kwadratów), aby ich impedancja pasowała do rzeczywistej w całym zakresie częstotliwości.
    Z_pręta/R_dc = (1+j)ξ·coth((1+j)ξ),   ξ = h/δ
    Z_zast = 1 / Σ_k 1/(R_k + jωL_k)

### Wykresy

- górne: k_R(f) i k_X(f) – dokładne (biała linia) i przybliżone 1, 2, 3 obwodami,
- lewy dolny: rozkład gęstości prądu wzdłuż wysokości pręta (0 = dno żłobka, 1 = strona szczeliny) przy wybranej częstotliwości,
- prawy dolny: błąd dopasowania |Z| w % – jeden obwód jest bardzo zły, trzy wystarczają (co potwierdza Podsumowanie 7.12: „trzy obwody w praktyce wystarczają dla wszystkich SM i IM”).

> 💡 Zmień materiał na aluminium – δ rośnie (gorzej przewodzi), efekt słabnie. Zwiększ h – efekt rośnie jak h².


> ⚙ Inżynier: parametry obwodów wyznacza się MES lub z próby postojowej odpowiedzi częstotliwościowej (SSFR). W dużych turbogeneratorach litej stali wirnika potrzeba nawet 3 obwodów w osi d i q. Straty w żelazie modeluje się podobnie – dodatkowymi zwartymi uzwojeniami dq (Boldea & Nasar 1987).


---

# Zakładka: 7.9–7.10 Park i wektor


## Równoważność maszyny 3-fazowej i modelu dq (7.9)


Trzy fazy A, B, C przesunięte o 120° wytwarzają razem jedno pole wirujące. To samo pole można wytworzyć DWOMA prostopadłymi uzwojeniami d i q. Warunek: przepływy (MMF, „siła magnesująca” = prąd × zwoje) muszą dawać ten sam wynik – rzutujemy prądy faz na osie d i q (Rys. 7.7). Wychodzi transformacja Parka (Równ. 7.36–7.37), tutaj w wersji z niezmiennikiem amplitudy (czynnik 2/3):
    i_d = 2/3·[i_A·cos θ + i_B·cos(θ − 2π/3) + i_C·cos(θ + 2π/3)]
    i_q = −2/3·[i_A·sin θ + i_B·sin(θ − 2π/3) + i_C·sin(θ + 2π/3)]
    i_0 = (i_A + i_B + i_C)/3          (składowa zerowa)

Dwie zmienne d, q nie wystarczą do opisu trzech faz – trzecią jest składowa zerowa i_0. Nie wytwarza ona pola wirującego, płynie tylko przez rezystancję i indukcyjność rozproszenia. Przy połączeniu w gwiazdę bez przewodu neutralnego i_0 = 0.

Moc: przy współczynniku 2/3 moc w modelu dq trzeba pomnożyć przez 3/2 (Równ. 7.42):
    p_abc = v_A·i_A + v_B·i_B + v_C·i_C = 3/2·(v_d·i_d + v_q·i_q) + 3·v_0·i_0
    Q = 3/2·(v_q·i_d − v_d·i_q)   (moc bierna – mimo że w stanie ustalonym wszystko jest DC!)



## Wektor przestrzenny (7.10)


Zamiast pary liczb (i_d, i_q) piszemy jedną liczbę zespoloną I_s = i_d + j·i_q. Równanie napięć stojana skraca się do (Równ. 7.51):
    V_s = R_s·I_s + dΨ_s/dt + j·ω_b·Ψ_s

W osiach stojana (ω_b = 0) wektor prądu kręci się po okręgu z prędkością ω1 (Rys. 7.8a). W osiach synchronicznych (ω_b = ω1) ten sam wektor STOI w miejscu (Rys. 7.8b).

### Wykresy (kliknij ▶ Start)

- lewy górny: prądy faz – kursor pokazuje chwilę animacji,
- prawy górny: i_d, i_q, i_0 w wybranych osiach – dla ω1 i symetrii to proste linie!
- lewy dolny: trajektoria wektora prądu w osiach stojana (αβ, szara) i w osiach wybranych (kolorowa); strzałka i obracająca się oś d,
- prawy dolny: moc chwilowa policzona z faz (linia) i z dq0 (kropki) – idealnie się pokrywają.

> 💡 Dodaj 5. harmoniczną – w osiach synchronicznych pojawiają się oscylacje 6·f1 (5. harm. wiruje w przeciwną stronę: −5 − 1 = −6). Dodaj asymetrię – oscylacje 2·f1 (składowa przeciwna). Dodaj i_0 – pojawia się tylko w i_0, nie w d, q.


> ⚙ Inżynier: wariant √(2/3) zachowuje moc bez czynnika 3/2 (transformacja ortonormalna), wariant 2/3 zachowuje amplitudy i dominuje w sterowaniu napędami. Dla maszyn 6-, 9-, 12-fazowych stosuje się 2, 3, 4 pary osi dq plus składowe zerowe.


---

# Zakładka: 7.11 Model HF / łożyska


## Gdy liczą się mikrosekundy (7.11)


Tranzystor IGBT w falowniku przełącza napięcie setek woltów w czasie 0,5–2 µs. Dla tak szybkich zmian cewki uzwojeń stanowią „ścianę” (duża reaktancja ωL), a do głosu dochodzą malutkie pojemności pasożytnicze: między zwojami, między uzwojeniem a obudową, między stojanem a wirnikiem. To jak w telefonie „z puszek”: wolne zmiany idą sznurkiem (drutem), a szybkie drgania przeskakują przez powietrze.
- do ok. 400 Hz wystarczy model R-L-E (dq),
- powyżej 20 kHz – model pojemnościowy (Rys. 7.9: rozłożone L i C wzdłuż uzwojenia),
- 400 Hz – 20 kHz – model uniwersalny (Rys. 7.10), razem z kablem.

### Napięcie wspólne (common mode)


Falownik 3-fazowy łączy każdą fazę do +V_dc/2 albo −V_dc/2. Suma trzech napięć NIGDY nie jest zerem – średnia skacze schodkami ±V_dc/6, ±V_dc/2:
    v_cm = (v_A0 + v_B0 + v_C0)/3

Punkt neutralny silnika „podskakuje” względem ziemi – to źródło V_nsg z Rys. 7.10.

### Napięcie na wale i prądy łożyskowe


Wirnik jest połączony z uzwojeniem przez C_sr, a z obudową przez C_rf i łożyska C_b. To dzielnik pojemnościowy:
    v_wału = BVR·v_cm,     BVR = C_sr/(C_sr + C_rf + C_b)   (zwykle 2–10%)

Film olejowy w łożysku jest izolatorem tylko do pewnego napięcia. Gdy v_wału przekroczy próg, następuje wyładowanie (EDM – jak mini-spawarka) i mikroskopijny krater na bieżni. Po milionach takich iskier łożysko ma „tarkę” (fluting) i hałasuje. To jedna z głównych przyczyn awarii silników zasilanych z falowników!

Prąd przez pojemność uzwojenie–obudowa: i_cm = C_sf·dv/dt ≈ C_sf·ΔV/t_r – krótkie impulsy o amplitudzie amperów.

### Wykresy

- lewy górny: 2 okresy nośnej – napięcia gałęzi (cienkie), v_cm (grube) i impulsy prądu i_cm,
- prawy górny: napięcie wału w całym okresie podstawowym z zaznaczonymi wyładowaniami EDM (czerwone ×),
- lewy dolny: impedancja fazy |Z(f)| – indukcyjna do rezonansu, potem pojemnościowa; tło pokazuje zakresy modeli,
- prawy dolny: widmo v_cm – prążki wokół wielokrotności f_sw.

> 💡 Skróć t_r → rosną impulsy i_cm (dv/dt!). Zmniejsz C_rf lub zwiększ C_sr → rośnie BVR i liczba wyładowań. Obniż próg V_th (zużyty smar) → więcej EDM.


> ⚙ Inżynier: środki zaradcze – filtr sinus/dU-dt lub filtr common-mode, szczotka uziemiająca wał, łożysko izolowane (ceramiczne kulki) od strony N, ekranowany kabel symetryczny, klatka Faradaya (ekran elektrostatyczny) w szczelinie zmniejszająca C_sr.


---

# Zakładka: Zadania 7.1–7.7


## Zadania 7.1–7.7 – rozwiązania z komentarzem


### 7.1 Czy obcowzbudny silnik DC pasuje do modelu dq0?


TAK. Uzwojenie wzbudzenia leży w osi d stojana, twornik (ze szczotkami w strefie neutralnej) – w osi q wirnika, ω_b = 0. W uzwojeniu wzbudzenia nie ma SEM rotacji (ono stoi, a jego oś nie ma „prostopadłego partnera” wytwarzającego strumień w ruchu), a prostopadłość osi wyklucza sprzężenie transformatorowe. Dwa równania:
    V_F = R_F·I_F + L_F·dI_F/dt
    V_a = R_a·I_a + L_a·dI_a/dt + ω_r·L_dm·I_F

### 7.2 Wirnik z barierą strumienia


Model dq jest ścisły dla STAŁEJ szczeliny. Wirnik jawnobiegunowy zastępujemy cylindrycznym z cienką średnicową barierą (szczeliną „nadprzewodzącą” dla strumienia). Dla L_dm > L_qm bariera leży WZDŁUŻ osi d – przecina drogę strumienia osi q, więc L_qm maleje. Dla L_dm < L_qm bariera leży wzdłuż osi q i wypełniamy ją magnesami magnesowanymi w osi d (tak powstaje wirnik IPMSM). Zasada: bariera stoi w poprzek drogi strumienia, który ma być „trudny”.

### 7.3 PMSM – patrz zakładka „7.5 SM/PMSM”


Zamiast L_dm·I_F piszemy Ψ_PM, usuwamy równania klatki i wzbudzenia; T_e = 3/2·p1·[Ψ_PM·I_q + (L_d − L_q)·I_d·I_q].

### 7.4 Dwufazowa maszyna PM o różnych liczbach zwojów


Uzwojenia o RÓŻNEJ liczbie zwojów, ale tej samej masie miedzi, są po sprowadzeniu do wspólnej liczby zwojów symetryczne (rezystancja i indukcyjność skalują się z kwadratem przekładni, a przepływ ∝ N·I jest taki sam). Model dq w osiach wirnika (ω_b = ω_r) jest więc poprawny. Składowej zerowej NIE ma – dwie prostopadłe fazy to dokładnie osie α, β (składowa zerowa pojawia się dopiero przy 3 fazach).

Po usunięciu jednego uzwojenia stojana asymetria jest i w stojanie, i w wirniku (jawne bieguny) – żaden układ osi nie da stałych indukcyjności, potrzebny jest model fazowy. Natomiast dla wersji z symetryczną klatką (Rys. 7.11b) jedno uzwojenie stojana nie przeszkadza: model dq działa w osiach stojana (ω_b = 0) – tak liczy się silnik jednofazowy. Reguła: osie przyklejamy do części NIESYMETRYCZNEJ, druga część musi być symetryczna.

### 7.5 Dwufazowy silnik indukcyjny


TAK – to wprost model dq bez składowej zerowej. Dowolne ω_b (0, ω_r, ω1), bo obie strony są symetryczne.

### 7.6 PMSM 6 żłobków / 4 bieguny z uzwojeniem skupionym


TAK, w osiach wirnika (ω_b = ω_r), bo strumień magnesu skojarzony z cewkami jest sinusoidalny w funkcji kąta. Harmoniczna przepływu rzędu 2p1 = 4 tworzy moment; pozostałe harmoniczne nie dają momentu średniego i trafiają do indukcyjności rozproszenia.

### 7.7 Silnik reluktancyjny przełączalny (SRM) 6/4 – wykresy obok


Wzajemne indukcyjności ≈ 0, własne zmieniają się z kątem. Jeśli zmiany są SINUSOIDALNE, SRM jest w istocie synchroniczną maszyną reluktancyjną → model dq w osiach wirnika działa. Moment z energii:
    T = Σ ½·i_k²·dL_k/dθ

Z prądami sinusoidalnymi (3-fazowymi) moment jest STAŁY – brak tętnień (zielona linia, prawy dolny wykres). Przy indukcyjności trapezowej (liniowe narastanie + płaski odcinek) składniki się już nie znoszą → tętnienia momentu, a model dq NIE jest ścisły (indukcyjności nie są sinusoidalne) – trzeba stosować model we współrzędnych fazowych.

> 💡 Suwak „udział płaskich odcinków”: 0 = trójkąt, 0,45 = prawie prostokąt. Obserwuj, jak rosną tętnienia. Kąt φ zmienia moment średni (maksimum przy 45°, jak w maszynie reluktancyjnej sin 2φ).


> ⚙ Inżynier: prawdziwe SRM zasila się impulsami prądu (nie sinusoidą) i sterownik „profiluje” prąd, żeby zmniejszyć tętnienia i hałas – to temat na osobną zakładkę sterowania.


---

# Zakładka: Przykład 8.1


## Przykład 8.1 – prądnica prądu stałego: skok wzbudzenia i zwarcie


Dane: R_a = 0,1 Ω, R_F = 1 Ω, L_a = 0,5 mH, L_F = 0,5 H, V_an = 200 V, I_an = 100 A (w książce −100 A, bo przyjęto konwencję silnikową – minus oznacza pracę prądnicową), I_Fn = 5 A, n = 1500 obr/min, obciążenie rezystancyjne.

### Po ludzku


Prądnica to „pompa elektryczna”: wirnik kręcony z zewnątrz przecina pole magnetyczne wytworzone przez prąd wzbudzenia I_F i indukuje SEM E. Prędkość jest stała (ciężki napęd, np. turbina), więc liczymy tylko zjawiska elektromagnetyczne – to są SZYBKIE stany nieustalone (8.3).

Co się dzieje, gdy podniesiemy napięcie wzbudzenia o 20%? Prąd wzbudzenia nie skoczy od razu – cewka L_F = 0,5 H działa jak bezwładność („koło zamachowe prądu”). Rośnie powoli ze stałą czasową T_F = L_F/R_F = 0,5 s. Za nim rośnie SEM i napięcie wyjściowe.

### Krok 1: rezystancja obciążenia i SEM

    R_load = V_an/|I_an| = 200/100 = 2 Ω
    E = V_an + R_a·|I_an| = 200 + 0,1·100 = 210 V
    ω_r·L_dm = E/I_Fn = 210/5 = 42 V/A       („ile woltów na amper wzbudzenia”)
    V_F0 = R_F·I_Fn = 5 V

### Krok 2: transmitancja (Równ. 8.9)

    V_a(s)/V_F(s) = ω_r·L_dm·R_load / [(R_F + s·L_F)·(R_a + R_load + s·L_a)]
    = 42·2 / [(1 + 0,5·s)·(2,1 + 0,0005·s)]

Dwa bieguny rzeczywiste ujemne: s1 = −R_F/L_F = −2 1/s oraz s2 = −(R_a+R_load)/L_a = −4200 1/s. Oba ujemne i rzeczywiste → odpowiedź stabilna i bez oscylacji (aperiodyczna).

### Krok 3: odpowiedź na skok ΔV_F = 0,2·5 = 1 V

    ΔV_a(∞) = 42·2/(1·2,1)·1 = 40 V   →   V_a: 200 → 240 V
    ΔI_a(∞) = 40/2 = 20 A              →   I_a: 100 → 120 A
    Δi_a(t) = 20·[1 − (T_F·e^(−t/T_F) − T_a·e^(−t/T_a))/(T_F − T_a)],  T_a = L_a/(R_a+R_load) = 0,238 ms

Ponieważ T_a jest 2000 razy mniejsze od T_F, praktycznie Δi_a(t) ≈ 20·(1 − e^(−t/0,5 s)). Po 3·T_F = 1,5 s osiągamy 95% zmiany.

### Krok 4: nagłe zwarcie zacisków (V_a = 0, I_F = I_Fn)


SEM zostaje (strumień wzbudzenia się nie zmienia), a prąd ogranicza TYLKO mała rezystancja twornika:
    L_a·di/dt + R_a·i = E    →    i(t) = E/R_a + (I_n − E/R_a)·e^(−t/T_e)
    i(∞) = 210/0,1 = 2100 A = 21·I_n (!),   T_e = L_a/R_a = 5 ms

W 15 ms (3·T_e) prąd rośnie do ponad 2000 A – dlatego zwarcie jest niebezpieczne i wymaga zabezpieczeń działających w milisekundach (wyłączniki szybkie).

### Wykresy

- lewy górny: prąd wzbudzenia i_F(t) – powolny wykładniczy wzrost,
- prawy górny: napięcie zacisków V_a(t),  • lewy dolny: prąd twornika i_a(t),
- prawy dolny: prąd zwarcia w milisekundach.

> 💡 Zwiększ R_F przy stałym L_F – T_F maleje, odpowiedź przyspiesza. Dlatego w praktyce stosuje się „forsowanie wzbudzenia” – chwilowo dużo wyższe V_F.


> ⚙ Inżynier: ta sama logika (wolny obwód strumienia, szybki obwód momentu) doprowadziła do sterowania polowo-zorientowanego maszyn AC: strumień trzymamy stały, a momentem sterujemy szybkim prądem.


---

# Zakładka: Przykład 8.2


## Przykład 8.2 – silnik DC z magnesami: nagłe obniżenie napięcia


Dane: P_n = 50 W, V_n = 12 V, η_n = 0,9, R_a = 0,12 Ω, T_e = 2 ms, p1 = 1, n_n = 1500 obr/min; dwie bezwładności J = 2·10⁻⁴ i J' = 10⁻³ kg·m²; napięcie spada z 12 do 10 V przy stałym momencie obciążenia.

### Po ludzku


Mały silniczek (np. wentylator w samochodzie). Obniżamy napięcie – silnik zwolni. Pytanie: czy zwolni „gładko”, czy z przeregulowaniem (spadnie poniżej nowej prędkości i wróci, jak samochód hamujący z zawieszeniem na sprężynach)? Odpowiedź zależy od stosunku dwóch stałych czasowych.

### Krok 1: stan znamionowy (d/dt = 0)

    I_an = P_n/(η_n·V_n) = 50/(0,9·12) = 4,63 A
    E = V_n − R_a·I_an = 12 − 0,12·4,63 = 11,44 V
    ω_r = 2π·1500/60 = 157,08 rad/s
    Ψ_PM = E/ω_r = 0,0729 Wb      T_e = p1·Ψ_PM·I_an = 0,337 N·m
    L_a = T_e·R_a = 0,002·0,12 = 0,24 mH

### Krok 2: model (Równ. 8.16–8.21)

    L_a·di_a/dt = V_a − R_a·i_a − ω_r·Ψ_PM
    (J/p1)·dω_r/dt = p1·Ψ_PM·i_a − T_L

Po wyeliminowaniu prądu dostajemy równanie 2. rzędu z dwiema stałymi czasowymi:
    T_e·T_em·d²ω/dt² + T_em·dω/dt + ω = (V_a − R_a·T_L/(p1Ψ))/Ψ
    T_e = L_a/R_a (elektryczna),   T_em = J·R_a/(p1·Ψ_PM)² (elektromechaniczna)
    bieguny: s1,2 = [−1 ± √(1 − 4T_e/T_em)]/(2T_e)

### Krok 3: dwie bezwładności

- J = 2·10⁻⁴: T_em = 4,52 ms < 4·T_e = 8 ms → pierwiastek z liczby ujemnej → bieguny ZESPOLONE → oscylacje tłumione (przeregulowanie prędkości i prądu),
- J' = 10⁻³: T_em = 22,6 ms > 8 ms → bieguny rzeczywiste → odpowiedź aperiodyczna (gładka, wolniejsza).

### Krok 4: stan końcowy


Moment obciążenia stały → prąd końcowy taki sam jak początkowy: i_a(∞) = 4,63 A.
    ω_rf = (V_a − R_a·I_a)/Ψ_PM = (10 − 0,556)/0,0729 = 129,6 rad/s    (książka: 129,55)

Czyli prędkość spada o ok. 17,5% (z 157 do 130 rad/s) – prawie proporcjonalnie do napięcia.

Warunki początkowe (Równ. 8.22): ω(0) = 157 rad/s, dω/dt(0) = 0 (prąd nie zmienia się skokowo, bo jest cewka). W chwili skoku prąd zaczyna gwałtownie spadać (a nawet zmienia znak – silnik chwilowo hamuje prądnicowo!).

### Wykresy

- lewy górny: ω_r(t) dla J i J' (Rys. 8.4a),  • prawy górny: i_a(t) (Rys. 8.4b),
- lewy dolny: bieguny na płaszczyźnie zespolonej – na osi rzeczywistej brak oscylacji,
- prawy dolny: współczynnik tłumienia ζ = ½·√(T_em/T_e) w funkcji J; ζ < 1 → oscylacje.

> 💡 Scenariusz 2 (uwaga w książce): skok momentu przy stałym napięciu. Teraz di_a/dt(0) = 0, a nie dω/dt(0) = 0. Prędkość spada o R_a·ΔT_L/(p1Ψ)² – mało, bo silnik PM ma „sztywną” charakterystykę.


> ⚙ Inżynier: silnik PM ma wbudowane sprzężenie zwrotne (SEM ∝ prędkości) – dlatego zawsze jest stabilny w pętli otwartej. Małe J (np. twornik bezżłobkowy) daje najszybszą odpowiedź momentu, ale z przeregulowaniem – tę rolę przejmuje regulator (zakładka 8.5).


---

# Zakładka: 8.4.2 Zmienny strumień


## 8.4.2 Zmiana strumienia – osłabianie pola


Chcemy, by silnik kręcił się SZYBCIEJ niż znamionowo, ale napięcie twornika jest już maksymalne. Sztuczka: osłabiamy pole (zmniejszamy prąd wzbudzenia I_F). Wtedy do wytworzenia tej samej SEM potrzebna jest większa prędkość: E = ω_r·L_dm·I_F.

### Po ludzku


To jak jazda na rowerze na lżejszym przełożeniu: łatwiej kręcić szybko, ale „siła” (moment na amper) jest mniejsza.

### Model (Równ. 8.27–8.28) – układ nieliniowy 3. rzędu

    L_F·dI_F/dt = V_F − R_F·I_F
    L_a·dI_a/dt = V_a − R_a·I_a − ω_r·L_dm·I_F
    (J/p1)·dω_r/dt = p1·L_dm·I_F·I_a − T_L

Iloczyny zmiennych (I_F·I_a, ω_r·I_F) czynią układ NIELINIOWYM. Inżynier linearyzuje go wokół punktu pracy (teoria małych odchyleń, Równ. 8.29–8.31): x = x0 + Δx i odrzuca iloczyny małych Δ.

### Wartości własne (Równ. 8.32)

    γ3 = −R_F/L_F         (obwód wzbudzenia – oddzielnie, bo osie d i q są prostopadłe!)
    γ1,2 – jak dla stałego strumienia (Przykład 8.2), ale policzone dla I_F0

Obwód wzbudzenia ma dużą stałą czasową (tu 0,2 s), więc zmiany strumienia są wolne.

### Wykresy

- lewy górny: I_F(t) – wolny wykładniczy spadek,
- prawy górny: I_a(t) – ciekawe! Gdy strumień maleje, SEM spada, więc prąd twornika gwałtownie ROŚNIE (niebezpieczny udar prądu), bo V_a − E rośnie. Linia przerywana – model zlinearyzowany,
- lewy dolny: prędkość rośnie do nowej, wyższej wartości,
- prawy dolny: wartości własne na płaszczyźnie zespolonej.

> 💡 Zwiększ skok do −50%: model liniowy (przerywany) zaczyna się wyraźnie różnić od nieliniowego – linearyzacja jest dobra tylko dla MAŁYCH zmian. Zmniejsz L_F – strumień zmienia się szybciej, udar prądu większy.


> ⚙ Inżynier: w praktyce osłabianie pola robi się powoli z ograniczeniem prądu twornika. Analogicznie w silnikach AC ze sterowaniem wektorowym: prąd i_d (strumień) zmieniamy wolno, a i_q (moment) szybko. To podobieństwo dało początek sterowaniu polowo-zorientowanemu, które zrewolucjonizowało napędy.


---

# Zakładka: 8.4.3 Szeregowy (Zad. 8.4)


## 8.4.3 Silnik szeregowy – Zadanie 8.4


Dane: R_a = 2·R_F = 1 Ω (czyli R_Fs = 0,5 Ω), L_a = 10 mH, L_Fs = 0,5 H, p1 = 2, n_n = 1500 obr/min, V = 500 V, I_a = 100 A. Straty poza miedzią pomijamy. Moment bezwładności nie jest podany – przyjmujemy J = 2 kg·m² (suwak).

### Po ludzku


W silniku szeregowym TEN SAM prąd płynie przez twornik i przez wzbudzenie. Im większe obciążenie, tym większy prąd, tym silniejsze pole i tym większy moment (∝ I²!). To idealny silnik trakcyjny – tramwaj ruszający pod górę dostaje ogromny moment. Wada: bez obciążenia silnik „ucieka” (rozbiegnie się), bo słabe pole → bardzo duża prędkość.

### Krok 1: stan znamionowy

    E = V − (R_a + R_Fs)·I_a = 500 − 1,5·100 = 350 V
    ω_r = p1·2π·n/60 = 2·157,08 = 314,16 rad/s (elektryczne)
    L_dm = E/(ω_r·I_a) = 350/(314,16·100) = 11,14 mH
    T_e = p1·L_dm·I_a² = 2·0,01114·100² = 222,8 N·m     (sprawdzenie: E·I/Ω_mech = 35000/157,08 ✓)

### Krok 2: linearyzacja (Równ. 8.33–8.36)

    (L_a + L_Fs)·di/dt = V − (R_a + R_Fs)·i − ω_r·L_dm·i
    (J/p1)·dω_r/dt = p1·L_dm·i² − T_L

Małe odchylenia wokół (I0, ω0):
    [(L_a+L_Fs)·s + R_a + R_Fs + ω0·L_dm]·Δi + L_dm·I0·Δω = ΔV
    (J/p1)·s·Δω = 2·p1·L_dm·I0·Δi − ΔT_L

Równanie charakterystyczne:
    (L·s + R + ω0·L_dm)·(J/p1)·s + 2·p1·L_dm²·I0² = 0

Zastępcza elektryczna stała czasowa (Równ. 8.38):
    T_es = (L_a + L_Fs)/(R_a + R_Fs + ω0·L_dm)

Zauważ: SEM rotacji działa jak dodatkowa rezystancja ω0·L_dm – dlatego T_es < (L_a+L_Fs)/(R_a+R_Fs), a odpowiedź jest szybsza niż „prosta” stała L/R.

### Krok 3: 1500 i 750 obr/min


Przy 750 obr/min (V = 500 V, bez nasycenia) prąd jest większy: I0 = V/(R + ω0·L_dm) = 500/(1,5 + 1,75) = 153,8 A. Wartości własne zmieniają się z prędkością (prawy dolny wykres).

### Krok 4: wzrost obciążenia o 20%


Nowy prąd: I = √(T_L/(p1·L_dm)) = √(1,2)·100 = 109,5 A, prędkość spada. Symulacja nieliniowa (ciągła) vs zlinearyzowana (przerywana).

### Wykresy

- lewy górny: prąd,  • prawy górny: prędkość,
- lewy dolny: charakterystyka mechaniczna T(n) – „hiperbola” silnika szeregowego i linie obciążenia,
- prawy dolny: części rzeczywiste wartości własnych i T_es w funkcji prędkości.

> ⚙ Inżynier: w rzeczywistości L_dm(i) zależy od prądu (nasycenie jest „nieuniknione”, bo prąd twornika to prąd wzbudzenia) – wtedy T_es zmienia się z prędkością jeszcze silniej. Silnik szeregowy nie może pracować bez obciążenia (np. przy zerwanym pasku) – zabezpieczenie nadobrotowe jest obowiązkowe.


---

# Zakładka: 8.5 Regulacja PI


## 8.5 Podstawowa regulacja zamknięta – kaskada PI


Silnik z Przykładu 8.2 (12 V, Ψ = 0,0729 Wb) chcemy rozpędzić do zadanej prędkości i utrzymać ją mimo zmian obciążenia. Układ z Rys. 8.7 ma DWIE pętle:
- wewnętrzna – szybka pętla PRĄDU (moment), regulator PI_i porównuje prąd zadany i*_a z mierzonym,
- zewnętrzna – wolniejsza pętla PRĘDKOŚCI, regulator PI_ω na podstawie błędu prędkości wylicza, jaki prąd (moment) jest potrzebny.

### Po ludzku


Kierowca (pętla prędkości) mówi: „potrzebuję więcej mocy”. Pedał gazu (pętla prądu) szybko i dokładnie ją dostarcza, ale nigdy ponad limit (I_max) – żeby nie spalić silnika. Bez regulatora, przy bezpośrednim podaniu 11,4 V na stojący silnik, prąd rozruchu wynosi V/R_a ≈ 95 A (20 × znamionowy!) – patrz linia przerywana.

### Regulator PI

    u(t) = K_p·e(t) + K_i·∫e(t)dt

Część P reaguje na bieżący błąd, część I „pamięta” błąd z przeszłości i usuwa uchyb ustalony (np. po obciążeniu).

### Dobór nastaw (kompensacja biegunów)


Pętla prądu: obiekt 1/(R_a + s·L_a); wybieramy zero regulatora w biegunie obiektu:
    K_pi = L_a·ω_ci,   K_ii = R_a·ω_ci   →   pętla otwarta = ω_ci/s   (idealny integrator)

Pętla prędkości: obiekt k_t/(J·s), k_t = p1·Ψ:
    K_pω = J·ω_cω/k_t,   K_iω = K_pω·ω_cω/4

Zasada kaskady: pętla zewnętrzna co najmniej 5–10 razy wolniejsza od wewnętrznej.

Anti-windup: gdy prąd jest ograniczony, integrator prędkości zostaje „zamrożony” – inaczej po rozruchu pojawiłoby się duże przeregulowanie.

Przekształtnik PWM modelujemy jako stałe wzmocnienie z ograniczeniem ±V_dc (dopuszczalne przy prądzie ciągłym – patrz zakładka 8.6).

### Wykresy

- lewy górny: prędkość – zadana, z regulacją i w pętli otwartej,
- prawy górny: prąd – z regulacją ograniczony do I_max; w pętli otwartej ogromny udar,
- lewy dolny: napięcie sterujące V_a,
- prawy dolny: charakterystyka Bodego pętli prędkości z zapasem fazy (PM > 45° = dobre tłumienie).

> 💡 Zmniejsz ω_cω – odpowiedź wolniejsza i łagodniejsza. Zwiększ ją bardzo – zapas fazy spada, pojawiają się oscylacje. Zmniejsz I_max – rozruch trwa dłużej (mniejszy moment).


> ⚙ Inżynier: w rzeczywistych napędach dochodzą filtry pomiarowe, próbkowanie (opóźnienie ~1,5 okresu PWM) i kompensacja SEM (feed-forward). Nastawy weryfikuje się w laboratorium odpowiedzią skokową i sprawdza przy maksymalnym oraz minimalnym J.


---

# Zakładka: 8.6 DC-DC (Zad. 8.5)


## 8.6 Silnik DC zasilany z przekształtnika DC-DC – Zadanie 8.5


Dane: R_a = 1 Ω, L_a/R_a = 5 ms, p1 = 1, SEM 0,05 V/(rad/s), ω_r = 120 rad/s, T_s = 1 ms, V = 12 V. Prędkość stała (mechanika jest dużo wolniejsza niż przełączenia). Współczynnika wypełnienia α w treści nie podano – ustaw go suwakiem.

### Po ludzku


Przekształtnik to bardzo szybki włącznik (tranzystor IGBT): przez czas α·T_s silnik jest podłączony do 12 V, przez resztę okresu – odłączony, a prąd płynie dalej przez diodę zwrotną (bo cewka „nie lubi” przerw w prądzie). Średnio silnik „widzi” napięcie V_av ≈ α·V. Tak działa ściemniacz LED czy regulator hulajnogi.

### Równania (8.39–8.40)

    IGBT włączony (0 < t < αT_s):   V = R_a·i + L_a·di/dt + E
    Dioda przewodzi (αT_s < t < λT_s):   0 = R_a·i + L_a·di/dt + E

Rozwiązania – wykładnicze dążenie do (V−E)/R_a, a potem do −E/R_a:
    i(t) = (V−E)/R_a + [i(0) − (V−E)/R_a]·e^(−t/T_e)
    i(t) = −E/R_a + [i(αT_s) + E/R_a]·e^(−(t−αT_s)/T_e)

### Prąd ciągły czy przerywany?


Tu E = 0,05·120 = 6 V. Jeśli w czasie wyłączenia prąd spadnie do zera przed końcem okresu (dioda nie przepuści ujemnego prądu), zaczyna się przerwa – prąd PRZERYWANY, λ < 1. W przerwie na zaciskach jest samo E (silnik „pokazuje” swoją SEM).
    V_av = α·V + (1 − λ)·E    (prąd przerywany)       V_av = α·V   (prąd ciągły, λ = 1)
    I_av = (V_av − E)/R_a,     T_av = k_E·I_av

Wniosek książki: przy prądzie przerywanym średnie napięcie jest WIĘKSZE niż α·V – „wzmocnienie” przekształtnika się zmienia, a regulator dostaje nieliniowy obiekt → sterowanie staje się ospałe (szczególnie przy małych prędkościach). Rozwiązanie: wykryć tryb przerywany i dodać Δα z tablicy.

### Wykresy

- lewy górny: prąd i_a(t) w 3 okresach (stan ustalony),  • prawy górny: napięcie na zaciskach,
- lewy dolny: V_av(α) – rzeczywiste vs „idealne” α·V; szary obszar = prąd przerywany,
- prawy dolny: I_av(α) i tętnienia prądu (p-p).

> 💡 Domyślne α = 0,5 daje α·V = 6 V = E → według „idealnego” wzoru prąd średni byłby zero, a w rzeczywistości płyną krótkie impulsy prądu. Zwiększ T_s (niższa częstotliwość) → większe tętnienia, łatwiej o przerywanie.


> ⚙ Inżynier: typowe częstotliwości łączeń to kilka–kilkadziesiąt kHz. Tętnienia prądu ≈ V·α(1−α)·T_s/L_a (dla T_s << T_e) – aby je zmniejszyć, podnosimy częstotliwość albo dodajemy dławik szeregowy.


---

# Zakładka: 8.7 Próby (Lab 8.1)


## 8.7 Jak zmierzyć parametry modelu? (Lab 8.1)


Model jest tyle wart, ile jego parametry. Książka opisuje próby POSTOJOWE (wirnik stoi) – nie trzeba drugiej maszyny, a zużycie energii jest małe.

### Próba zaniku prądu (Rys. 8.9)


1. Przez przekształtnik DC-DC ustalamy w tworniku prąd i_0. Napięcie V_0 i prąd mierzymy → R_a = V_0/i_0 (w stanie ustalonym cewka nie ma spadku napięcia).

2. Wyłączamy IGBT. Prąd płynie dalej przez diodę zwrotną i zanika. Rejestrujemy i(t) i napięcie diody V_d.

3. Równanie obwodu (Równ. 8.46):  0 = R_a·i + L_a·di/dt + V_d

4. Całkujemy od 0 do t_da (gdy i ≈ 0,01·i_0) (Równ. 8.47):
    L_a·i_0 = R_a·∫i·dt + V_d·t_da      →      L_a = (R_a·∫i dt + V_d·t_da)/i_0

Po ludzku: cewka zgromadziła energię (prąd „rozpędzony”); ile „pracy” wykonał zanik prądu na rezystancji i diodzie, tyle wynosiła jej „bezwładność” L_a·i_0.

Pomiar dla kilku i_0 pokazuje, czy L_a zależy od prądu (nasycenie). Pominięcie V_d daje błąd – szczególnie przy małych napięciach i dużych prądach (prawy górny wykres). Przy nasyceniu próba daje indukcyjność „uśrednioną energetycznie” po całym zaniku (od i_0 do 0), a nie wartość dokładnie przy i_0 – stąd różnica zielonej i białej krzywej. Ustaw nasycenie = 0, a zgodność będzie bardzo dobra.

### Próba wybiegu – moment bezwładności J


Rozpędzamy silnik, wyłączamy zasilanie (I_F = 0 – brak strat w żelazie) i mierzymy spadek prędkości. Straty mechaniczne p_mec(Ω) znamy z prób biegu jałowego. Z bilansu energii (Równ. 8.49):
    p_mec = −J·Ω·dΩ/dt     →     J = −p_mec/(Ω·dΩ/dt)

Po ludzku: bąk zwalnia tym wolniej, im jest „cięższy” (większe J) przy tych samych oporach.

Nachylenie dΩ/dt wyznaczamy z danych pomiarowych (regresja liniowa lokalnie) – na wykresie styczna.

### Wykresy

- lewy górny: zanik prądu (z szumem) i napięcie diody,
- prawy górny: L_a zmierzone dla różnych i_0 (z V_d i bez) vs prawdziwe,
- lewy dolny: wybieg z zaznaczoną styczną,
- prawy dolny: błędy wyznaczenia parametrów w %.

> 💡 Ustaw V_d = 0 – obie metody się zgadzają. Zwiększ szum – rośnie błąd J (różniczkowanie wzmacnia szum!), a L_a prawie nie (całkowanie uśrednia szum). To ważna lekcja: inżynier woli całkować niż różniczkować.


> ⚙ Inżynier: w maszynie z magnesami (PM) próba wybiegu nie oddzieli strat mechanicznych od strat w żelazie (magnesu nie da się wyłączyć) – wtedy J mierzy się metodą wahadła. Pełne procedury opisują normy IEEE, IEC, NEMA.


---

# Zakładka: Zad. 8.1


## Zadanie 8.1 – narastanie prądu prądnicy po załączeniu wzbudzenia


Dane: obciążenie R_l = 1 Ω, L = 1 H; R_a = 0,1 Ω, L_a = 0; wzbudzenie R_F = 50 Ω, L_F = 5 H nagle przyłączone do 120 V; E/i_F = ω_r·L_dm = 40 V/A; prędkość stała.

### Po ludzku


Dwa „zbiorniki” napełniane jeden po drugim: najpierw prąd wzbudzenia napełnia się ze stałą czasową T_F, a SEM (proporcjonalna do i_F) napełnia drugi zbiornik – obwód twornika z obciążeniem – ze stałą czasową T_a. Wynik to krzywa „S” (na początku prąd rośnie bardzo wolno).

### Krok 1: obwód wzbudzenia

    i_F(t) = (V_F/R_F)·(1 − e^(−t/T_F)),   V_F/R_F = 2,4 A,   T_F = L_F/R_F = 0,1 s

### Krok 2: obwód twornika + obciążenia

    (R_a + R_l)·i_a + L·di_a/dt = E(t) = 40·i_F(t)
    T_a = L/(R_a + R_l) = 1/1,1 = 0,909 s,   I_a(∞) = 40·2,4/1,1 = 87,27 A

### Krok 3: rozwiązanie (dwie stałe czasowe)

    i_a(t) = I_a∞·[1 − (T_a·e^(−t/T_a) − T_F·e^(−t/T_F))/(T_a − T_F)]

Na starcie di_a/dt = 0 (krzywa „S”), po ok. 5·T_a ≈ 4,5 s prąd osiąga wartość ustaloną. Napięcie na zaciskach: V = E − R_a·i_a.

> 💡 Jak w Przykładzie 8.1 – ale teraz wolniejszy jest obwód twornika (duże L obciążenia), a nie wzbudzenia. Zamień wartości tak, by T_a ≈ T_F – krzywa nadal jest poprawna (przypadek graniczny t·e^(−t/T)).


---

# Zakładka: Zad. 8.2


## Zadanie 8.2 – rozruch silnika PM pod obciążeniem


Dane: R_a = 0,12 Ω, L_a = 0, 2p1 = 2, p1·Ψ_PM = 0,073 N·m/A, J = 2·10⁻⁴ kg·m², obciążenie T_L = 0,2 + 10⁻³·ω_r, silnik włączony wprost na 12 V.

### Po ludzku


W chwili włączenia silnik stoi – nie ma SEM, więc prąd ogranicza tylko R_a: 12/0,12 = 100 A (!). Daje to ogromny moment 7,3 N·m i silnik gwałtownie przyspiesza. W miarę rozpędzania SEM rośnie, prąd maleje, aż moment silnika zrówna się z obciążeniem.

### Krok 1: równania (L_a = 0 → Równ. 8.21 z T_e = 0)

    i_a = (V − k·ω)/R_a,    k = p1·Ψ_PM = 0,073
    J·dω/dt = k·i_a − T0 − B·ω = k·V/R_a − T0 − (k²/R_a + B)·ω

To równanie 1. rzędu: ω dąży wykładniczo do wartości końcowej.

### Krok 2: stała czasowa i prędkość końcowa

    τ_m = J/(k²/R_a + B) = 2·10⁻⁴/(0,04441 + 0,001) = 4,40 ms
    ω_f = (k·V/R_a − T0)/(k²/R_a + B) = (7,3 − 0,2)/0,04541 = 156,3 rad/s ≈ 1493 obr/min

### Krok 3: przebiegi

    ω(t) = ω_f·(1 − e^(−t/τ_m)),    i_a(t) = (V − k·ω)/R_a,    T_e = k·i_a

Prąd startuje od 100 A i spada do ok. 4,9 A. Po 5·τ_m ≈ 22 ms rozruch zakończony.

> 💡 Ustaw L_a = 0,24 mH (T_e = 2 ms jak w Przykładzie 8.2) – prąd nie skacze już do 100 A natychmiast, a przebieg ma drugi rząd (lekkie oscylacje). Zwiększ J – rozruch dłuższy, ale prąd rozruchowy ten sam.


> ⚙ Inżynier: 20-krotny prąd rozruchowy grzeje szczotki i może rozmagnesować magnesy – w praktyce rozruch robi się przez przekształtnik z ograniczeniem prądu (zakładka 8.5).


---

# Zakładka: Zad. 8.3


## Zadanie 8.3 – zrzut obciążenia o 20%


Silnik z Zadania 8.2 pracuje w stanie ustalonym przy 1500 obr/min. Moment obciążenia maleje skokowo o 20%. Szukamy nowego prądu ustalonego oraz ω_r(t), T_e(t).

### Krok 1: stan początkowy

    ω0 = 2π·1500/60 = 157,08 rad/s
    T_L0 = 0,2 + 10⁻³·157,08 = 0,357 N·m     →   I_a0 = T_L0/k = 4,89 A
    napięcie potrzebne do 1500 obr/min: V = R_a·I_a0 + k·ω0 = 0,587 + 11,467 = 12,05 V (≈ 12 V z Zad. 8.2)

### Krok 2: nowy stan ustalony


Obciążenie: T_L = 0,8·(0,2 + 10⁻³·ω). Z równań V = R_a·I + k·ω oraz k·I = T_L:
    ω_f = (k·V/R_a − 0,8·0,2)/(k²/R_a + 0,8·10⁻³),   I_af = (V − k·ω_f)/R_a

Wynik: prąd spada z 4,89 do ok. 3,93 A, a prędkość rośnie tylko o ok. 15 obr/min (do ok. 1515 obr/min, +1%) – silnik PM ma „sztywną” charakterystykę.

### Krok 3: warunek początkowy (wskazówka książki)


Teraz skok dotyczy MOMENTU, a nie napięcia. Prąd w chwili t = 0⁺ jest ciągły i jego pochodna jest zerowa: (di_a/dt)(0) = 0, bo napięcie i SEM się nie zmieniły. Zmienia się natomiast od razu przyspieszenie: dω/dt(0) = (k·I_a0 − T_L)/J ≠ 0.

Równanie 2. rzędu jak w Przykładzie 8.2 – z L_a = 0,24 mH i J = 2·10⁻⁴ mamy T_em < 4T_e, więc odpowiedź jest lekko oscylacyjna.

> 💡 Porównaj z Przykładem 8.2: tam skakało napięcie (i prąd reagował gwałtownie), tu skacze moment (prąd reaguje łagodnie, bo po drodze jest mechanika).

