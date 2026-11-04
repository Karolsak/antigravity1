# Silnik indukcyjny: modele i sterowanie (Plecs3), aplikacja Tkinter

Interaktywna aplikacja do **wszystkich przykładów** z załącznika *Plecs3*, czyli rozdziałów 8 i 9 książki
R. W. De Doncker, D. W. J. Pulle, A. Veltman, *Advanced Electrical Drives*:

- rozdział 8 *Induction Machine Modeling Concepts*,
- rozdział 9 *Control of Induction Machine Drives*.

Każda zakładka zawiera:

- **suwaki** z polem do wpisania dokładnej wartości (Enter). Wykresy przeliczają się automatycznie.
- **wykresy Matplotlib** z paskiem narzędzi: zoom, przesuwanie, zapis PNG, odczyt współrzędnych kursora.
- **wyniki liczbowe** do porównania z książką.
- **„Wyjaśnienie (dla ucznia)”**: opis zjawiska prostym językiem, analogie, proponowane eksperymenty i wnioski inżyniera.
- **„Analiza LTI”** dla układów sterowania (7 zakładek, 20 transmitancji):
  - charakterystyki **Bodego** (amplituda i faza, z zaznaczonymi zapasami PM i GM),
  - postać **zpk** oraz **mapa zer i biegunów** (dla pętli także bieguny układu zamkniętego),
  - wykres **Nyquista** (dla pętli z punktem −1),
  - **odpowiedź skokowa** (obiektu albo zamkniętej pętli),
  - automatyczny raport: stałe czasowe, ωn, ζ, pasmo, czas narastania i ustalania, przeregulowanie, stabilność. Do tego interpretacja inżynierska konkretnej transmitancji.

## Uruchomienie

```bash
pip install numpy matplotlib
```

```bash
python plecs3_im_drives_tkinter.py
```

Test bez okna, który przelicza wszystkie zakładki i zapisuje wykresy do PNG:

```bash
python plecs3_im_drives_tkinter.py --selftest wykresy_png
```

Pliki:

| Plik | Zawartość |
|---|---|
| `plecs3_im_drives_tkinter.py` | GUI: 18 zakładek przykładów, panel analizy LTI |
| `plecs3_models.py` | modele i symulacje (numpy, bez tkinter), narzędzia LTI: TF, zpk, Bode, Nyquist, step, zapasy, linearyzacja |
| `plecs3_teksty.py` | wyjaśnienia dla ucznia i interpretacje wyników analiz |

## Zakładki

### Rozdział 8: modele maszyny

| Zakładka | Przykład z książki | Co można badać |
|---|---|---|
| 8.6.1 Uproszczony | Tutorial 1, Fig. 8.39 | model bez rozproszeń zasilany strumieniem, rozruch, charakterystyka prosta, bilans mocy, wykres wektorowy |
| 8.6.2 Uniwersalny (a) | Tutoriale 2 i 3, Fig. 8.40–8.42 | model IRTF z parametrem a; dowód, że Te, n i i_s nie zależą od a; miejsca końców ψM i ψR |
| 8.6.4 Prądowy i_d/i_q | Tutorial 4, Fig. 8.43–8.44 | zasada sterowania polowego: i_d → strumień, i_q → moment. **LTI:** Gψ, Gω, pętla prędkości PI |
| 8.6.5 UFO | Tutorial 5, Fig. 8.22, 8.26, 8.29, 8.45 | orientacja na wirnik i na stojan, impuls poślizgu. **LTI:** Gψ (zero zależne od a), Gsl, Gω |
| 8.3.5 Heyland | Fig. 8.30 | okrąg prądu przy stałym ψM, moment krytyczny, przejście w prostą dla a = Lm/Lr |
| 8.3.6 Stan ustalony | Fig. 8.31, 8.32, 8.48 (Tutorial 8) | wykres Heylanda, Te(n), I(n), ψs(n), P, η dla maszyny 17,3 kW |
| 8.6.6 Parametry | 8.4, Tutorial 6, Listing 8.1, Tab. 8.1–8.3 | estymacja parametrów z tabliczki krok po kroku, schematy zastępcze |
| 8.6.7 Rozruch z sieci | Tutorial 7, Fig. 8.46–8.47 | rozruch bezpośredni, skok 113 Nm, trajektoria na tle charakterystyki. **LTI:** linearyzacja (ΔT_l→Δn, pętla elektromechaniczna) |
| 8.6.9 1-faz. rozruch | 8.5, Tutorial 9, Fig. 8.36, 8.49–8.51 | silnik z kondensatorem: tętnienia momentu, prądy, napięcie na C, wektory w układzie wirnika |
| 8.5.1 1-faz. ustalony | 8.5.1, Fig. 8.37–8.38 | składowa zgodna i przeciwna, moment średni i tętnienia, dobór kondensatora |

### Rozdział 9: sterowanie

| Zakładka | Przykład z książki | Co można badać |
|---|---|---|
| 9.1 Kloss | 9.1, Fig. 9.1 | wzór Klossa, rodzina charakterystyk V/f, punkty pracy |
| 9.7.1 V/f prosty | 9.1.1, Tutorial 9.7.1, Fig. 9.2–9.4, 9.24–9.25 | rozruch i rampa z zakłóceniem; **niestabilność** i mapa max Re(λ) w funkcji prędkości. **LTI:** G_n, G_l, pętla elektromechaniczna (Nyquist) |
| 9.7.2 V/f + czujnik | 9.1.2, Tutorial 9.7.2, Fig. 9.5–9.6, 9.26 | sterowanie momentem, Te = 105 Nm zamiast 113 Nm i jego przyczyna. **LTI:** G_T, G_n, G_l, L |
| 9.2.1 Odstrojenie | 9.2.1, Fig. 9.8 | błąd kąta orientacji, odstrojenie RR w IFO (statyka i dynamika) |
| 9.2.4 Metody CFO | 9.2.4–9.2.8, Fig. 9.11–9.16 | tabela metod; błędy obserwatora napięciowego i prądowego, obserwator łączony |
| 9.3 Granice / osł. pola | 9.3–9.4, Fig. 9.17–9.19 | okrąg MA, ψR_max, elipsy MF, MTPF i MTPA, trajektoria prądu, prędkość bazowa, moment i moc w funkcji prędkości |
| 9.7.3 UFO prądowy | 9.5, Tutorial 9.7.3, Fig. 9.20, 9.27–9.29 | sterownik UFO (cyfrowy) + IFO z enkoderem + maszyna zasilana prądem, a od wirnika do stojana, odstrojenie. **LTI:** G_T, G_n, G_ψR, pętla prędkości PI |
| 9.7.4 UFO + reg. prądu | 9.6, Tutorial 9.7.4, Fig. 9.22–9.23 | regulator prądu PI w dq z kompensacją SEM, ograniczenie napięcia, opóźnienie 1,5·Ts. **LTI:** pętla prądu (PM, GM), obiekt, tłumienie zakłóceń |

## Weryfikacja z książką

| Wielkość | Aplikacja | Książka |
|---|---|---|
| Ls, Lm, Lσ, Rr (Listing 8.1) | 272 mH, 260 mH, 12,1 mH, 0,541 Ω | Tab. 8.2: 272, 260, 12, 0,54 |
| LM, LσS, RR (a = Lm/Lr) | 248,5 mH, 23,6 mH, 0,494 Ω | Tab. 8.3: 248,5, 23,6, 0,494 |
| ψs znamionowe | 1,831 Vs | 1,83 Vs |
| a dla orientacji na wirnik i stojan | 0,9557 i 1,0464 | 0,955 i 1,046 (Fig. 9.28, 9.29) |
| Model uniwersalny, 1 Nm | 1496,8 obr/min | 1496,57 obr/min (Fig. 8.42) |
| Sieć 415 V, 113 Nm | 1464,9 obr/min, 29,96 A | 1465 obr/min, 30 A |
| Bieg jałowy | 8,5 A | 8,41 A (pomiar) |
| V/f z czujnikiem, T* = 113 Nm | Te = 105,2 Nm, ψs = 1,871 Vs | 105,1 Nm, 1,871 Vs |
| V/f bez czujnika, Rr −10 % | niestabilny ok. 390–550 obr/min | ok. 580 obr/min |
| Silnik 1-faz.: symulacja vs metoda składowych | 2,000 / 2,007 Nm, tętnienia 0,63 / 0,64 Nm | zgodność obu metod |

Uwaga: w Tab. 8.3 podano ψR = 1,67 Vs. Ta wartość pochodzi z linii 46 Listingu 8.1, która używa nietypowego wzoru.
Fizycznie ψR = |ψs − LσS·is| ≈ 1,59 Vs. Aplikacja pokazuje obie wartości.

## Uwagi modelowe

- Wektory przestrzenne amplitudowo niezmiennicze, Te = 3/2·p·Im{ψs*·is}; silnik jednofazowy (dwufazowy) Te = p·Im{ψs*·is}.
- Symulacje: RK4 ze stałym krokiem (50–500 µs); kod napisany w czystym Pythonie i numpy, bez scipy i python-control.
- Sterownik UFO jest zaimplementowany dyskretnie (różniczkowanie wsteczne, rozwiązanie pętli algebraicznej iteracją).
- Regulator prądu w 9.7.4 to PI w osiach dq dobrany kryterium modułowym (Kp = LσS/(2·Td), Ki = R/(2·Td), Td = 1,5·Ts)
  z kompensacją ψM·ωm. Równania „model based current control” z rozdz. 3 nie są częścią załącznika.
- Linearyzacja napędów V/f i rozruchu z sieci: numeryczna (różnice centralne) w układzie synchronicznym, 5 stanów.
  Licznik transmitancji wyznaczany przez interpolację na okręgu, co jest dobrze uwarunkowane numerycznie.
