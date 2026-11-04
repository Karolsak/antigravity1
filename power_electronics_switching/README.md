# Strategie przełączania przekształtników energoelektronicznych — interaktywna aplikacja Tkinter

Aplikacja do rozdziałów 4–6 książki *S. V. Iyer, M. N. Aalam, „Switching Strategies for Power Electronic Converters”* (mostek H, przekształtniki trójfazowe i SVPWM, przekształtniki wielopoziomowe NPC). Dochodzi do tego analiza **układów sterowania**: Bode, ZPK, Nyquist, odpowiedź skokowa, bieguny i zera oraz przemiatanie każdego parametru.

Każda zakładka ma suwaki parametrów, interaktywne wykresy Matplotlib (zoom, przesuwanie, zapis PNG), pole **Wyniki** z liczbami i porównaniem z teorią oraz **objaśnienie dla ucznia liceum**. Objaśnienie obejmuje analogie, wzory, przykład liczbowy i rubrykę „SPRÓBUJ”.

## Uruchomienie

```bash
pip install numpy matplotlib
```

```bash
cd power_electronics_switching
```

```bash
python pe_app.py
```

Aplikacja nie wymaga SciPy ani python-control. Funkcje Bode, zapasy stabilności, ZPK, Nyquist, odpowiedź skokowa (wykładnik macierzy) i aproksymację Padé zaimplementowano w `pe_control.py` na czystym numpy.

| Plik | Zawartość |
|---|---|
| `pe_app.py` | GUI: 21 zakładek w 4 grupach (ładowane przy pierwszym otwarciu) |
| `pe_core.py` | modele: PWM, symulacje R-L, Clarke, tablice wektorów, SVPWM 2- i 3-poziomowe, model kondensatorów NPC |
| `pe_control.py` | mini „control toolbox”: TF, ZPK, Bode, marginesy, Nyquist, step, Padé, PI, PR |
| `pe_texts.py` | objaśnienia wszystkich przykładów (po polsku) |

## Rozdział 4 — Mostek H (full-bridge)

| Zakładka | Przykład z książki | Co pokazuje |
|---|---|---|
| 4.2 Stany przewodzenia | rys. 4.1, 4.3, 4.4 | Interaktywny schemat: wybierasz S1–S4 i kierunek prądu, a aplikacja pokazuje przewodzące tranzystory i diody oraz vo. Tabela wszystkich 16 kombinacji: zamierzone, zależne od prądu i zabronione (zwarcie gałęzi). |
| 4.2 PWM bipolarne dc-ac | rys. 4.5–4.7 | Porównanie sinus–trójkąt, napięcie ±Vdc, prąd R-L, prąd szyny DC i widmo. **Bilans energii** Vdc·Idc = I²R jest sprawdzany liczbowo. |
| 4.3 Wektory T+ T0 T− | rys. 4.8–4.9 | Równanie (+Vdc)T+ + 0·T0 + (−Vdc)T− = Vreq·Tsw dla strategii bipolarnej, unipolarnej i phase-shift |
| 4.4 dc-dc unipolarne | rys. 4.11–4.12 | Nośne c1 i c2 = −c1, praca w 4 ćwiartkach, obciążenie R-L-E (silnik DC), automatycznie rozpoznawana ćwiartka pracy |
| 4.4 Dwa mostki LC–RC | rys. 4.10, 4.13–4.18 | Przypadek 1 (d1 = 0,5, d2 = 0,3) i przypadek 2 (d1 < 0, duży prąd zwrotny). Trzy nachylenia di/dt, kierunek mocy, przesunięcie nośnej RC |
| 4.5 PWM unipolarne dc-ac | rys. 4.19–4.21 | Trzy poziomy napięcia, porównanie z bipolarnym: tętnienie mniejsze ok. 2×, harmoniczne przesunięte do 2·mf, THD |
| 4.5 Phase-shift PWM | rys. 4.22–4.24 | Każda gałąź pracuje z wypełnieniem 50 %, regulowane jest przesunięcie fazowe d; średnia napięcia = 0 (transformator), Vrms = Vdc·√(2d), V1 = (4/π)·Vdc·sin(πd) |

## Rozdział 5 — Przekształtniki trójfazowe i SVPWM

| Zakładka | Przykład | Co pokazuje |
|---|---|---|
| 5.2 Układy 3-fazowe | rys. 5.1–5.3 | Wektory wirujące i ich rzuty, napięcia międzyprzewodowe (√3, +30°), gwiazda i trójkąt, niesymetria i przesunięcie punktu neutralnego |
| 5.3 Falownik sine-PWM | rys. 5.4–5.8 (24 V, 10 Ω, 1 mH, 2500 Hz) | Prądy, impulsy gałęzi, napięcie fazowe o 5 poziomach (v_nN = suma/3), widmo i trajektoria α-β |
| 5.4 Clarke | rys. 5.9–5.11 | α, β, 0. Okrąg → elipsa przy niesymetrii, „kwiatek” przy 5. harmonicznej; offset trafia tylko do v0. Przykład z książki: 0,3∠30° = 0,26 + j0,15 |
| 5.5 Wektory przekształtnika | Tabele 5.1–5.2, rys. 5.12–5.13 | Pełna tabela 8 stanów, sześciokąty fazowy (2/3·Vdc) i międzyprzewodowy (2/√3·Vdc), wybór stanu |
| 5.6 Algorytm SVPWM | rys. 5.14–5.15, Tabela 5.3 | Sektor wyznaczany bez arcus tangensa (porównania z tan 60°), t1, t2, t0, symetryczna sekwencja V0–Vx–Vy–V7–V7–Vy–Vx–V0, wykrywanie przemodulowania |
| 5.7 Symulacja SVPWM | rys. 5.16–5.19 | Pełna symulacja, wypełnienia w kształcie „siodła”, charakterystyka V1 = f(żądane): granica **0,5·Vdc (SPWM)** i **0,577·Vdc (SVPWM)** |

**Weryfikacja przykładu z książki:** dla |Vr| = 0,3 i ωt = 30° aplikacja liczy t1 = t2 = 0,2598·T i t0 = 0,4804·T. Dla |Vr| = 2/3 przy 30° wychodzi t1 + t2 = 1,155·T > T, czyli przemodulowanie. Oba wyniki są zgodne z rozdz. 5.6–5.7.

## Rozdział 6 — Przekształtniki wielopoziomowe NPC

| Zakładka | Przykład | Co pokazuje |
|---|---|---|
| 6.3 Gałąź NPC 3/4-poz. | rys. 6.2–6.3, Tabele 6.1–6.2 | Schemat gałęzi z diodami Dm1 i Dm2, droga prądu, stany zabronione (zwarcie górnego lub dolnego kondensatora) i niezamierzone, tabela 4-poziomowa |
| 6.4 Mapa wektorów | Tabela 6.3, rys. 6.5–6.6 | 8, 27 i 64 stany, 7, 19 i 37 różnych wektorów, redundancje (np. V12 = V25 = 0,1667 + j0,2887), granica 0,577·Vdc |
| 6.5 SVPWM 3-poziomowe | rys. 6.7–6.11, rozdz. 6.6 | Trójkąt najbliższych wektorów, sztuczka z przesunięciem Vr′ = Vr − Vz (porównanie ze wzorem 5.32), sekwencje redundantne: dolny i górny kondensator, np. V9–V12–V21–V22 albo V12–V21–V22–V25 |
| 6.6–6.7 Balans + symulacja | rys. 6.12–6.14 | Pełna symulacja 3-poziomowego NPC z modelem kondensatorów (dvc/dt = i_M/C). Tryby: **histereza (książka)**, predykcyjny, regulator P, brak balansowania. Porównanie harmonicznych z falownikiem 2-poziomowym |

## Układy sterowania — analiza Bode / ZPK / Nyquist / Step

Każda z 4 zakładek sterowania ma trzy części.

1. **Bode / Nyquist / Step / Bieguny**:
   - Bode amplitudy i fazy dla L = C·P, a także P, C i T, z zaznaczonymi PM i GM
   - Nyquist z punktem −1
   - odpowiedź na skok wartości zadanej z tr, OS i ts
   - odpowiedź na skok zakłócenia
   - mapa biegunów i zer z liniami stałego tłumienia ζ
2. **Przemiatanie parametru**: PM, GM, fc, przeregulowanie i ts w funkcji dowolnego wybranego parametru, czyli analiza wrażliwości „dla wszystkich parametrów”.
3. Pole wyników: **ZPK** dla P, C, L i T, bieguny pętli zamkniętej z ζ i fn, zapasy stabilności oraz **słowna interpretacja** wyników.

Opóźnienie e^(−sTd) w Bode i Nyquiście jest liczone dokładnie. W odpowiedzi skokowej i mapie biegunów zastępuje je aproksymacja Padé 2. rzędu.

| Zakładka | Obiekt i regulator | Wyniki dla nastaw domyślnych |
|---|---|---|
| **C1 PI prądu mostka H** | P = 1/(Ls+R)·e^(−1,5·s·Ts), C = Kp + Ki/s; przycisk strojenia „kompensacja bieguna” (Kp = ωc·L, Ki = ωc·R) | fc = 478 Hz, PM = 64,2°, GM = 10,9 dB, OS = 1,4 %, ts = 0,64 ms. Po strojeniu na 500 Hz: PM = 62,7°. Teoria: PM = 90° − 360°·fc·Td |
| **C2 PR prądu sieci** | Dławik 5 mH / 0,2 Ω, PR: Kp + Kr·2ωc·s/(s² + 2ωc·s + ω0²) | fc = 479 Hz, PM = 61,1°, GM = 10,8 dB, \|C(j2π50)\| = 315. Symulacja: PR usuwa uchyb sinusoidy, PI (Ki = 2ωc·Kr) zostawia uchyb ok. 1 A |
| **C3 Kaskada napięcia DC** | P = 1/(Cs)·1/(τi·s + 1), PI; przycisk „optimum symetryczne” | Domyślnie fc = 100 Hz, PM = 71,6°, OS = 9,2 %. Optimum symetryczne a = 3: PM = **53,1°** (teoria arcsin((a²−1)/(a²+1)) = 53,1°), fc = 166,5 Hz |
| **C4 Balans kondensatorów NPC** | P = \|Kb\|/(Vdc/2)/(Cs)·filtr·e^(−sTs); Kb = d⟨i_M⟩/dδ liczone numerycznie z modulatora | Kb = −1,92 A (cos φ ≈ 1), fc = 5,8 Hz, PM = 88°, GM = 37,6 dB. **Model liniowy pokrywa się z pełną symulacją przełączaną** (wykres w zakładce „Symulacja czasowa”) |

### Najważniejsze wnioski z analizy sterowania

- **Opóźnienie cyfrowe** (obliczenia + PWM ≈ 1,5·Ts) ogranicza pasmo regulatora prądu. Przy fc = 1 kHz i fsw = 10 kHz zabiera 54° fazy. W praktyce stosuje się fc ≤ fsw/10–fsw/20.
- **PI nie śledzi sinusoidy bez uchybu.** Regulator PR ma „całkowanie dla 50 Hz”: bardzo duże wzmocnienie dokładnie przy częstotliwości sieci.
- **W kaskadzie** pętla zewnętrzna musi być 5–10× wolniejsza od wewnętrznej. W optimum symetrycznym stosunek pasm wynosi dokładnie a.
- **Balansowanie NPC** działa tylko przy przepływie mocy czynnej. Wzmocnienie Kb maleje razem z prądem i cos φ, dlatego balansowanie jest trudne przy obciążeniu biernym.
