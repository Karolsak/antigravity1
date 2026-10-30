# Analiza systemu elektroenergetycznego – ekonomika, HVDC, stabilność

Interaktywna aplikacja **Python + Tkinter + Matplotlib** (`power_system_operation_tkinter.py`)
dla rozdziałów 11–13 podręcznika *Power System Analysis, Operation and Control*.
Każdy przykład liczbowy z rozdziałów ma swoją zakładkę z suwakami, wykresami
i wyjaśnieniem napisanym dla ucznia liceum.

## Uruchomienie

```bash
pip install numpy matplotlib
```

```bash
python power_system_operation_tkinter.py
```

Wymagany jest Python 3.8+ z modułem `tkinter` (na Linuksie: pakiet `python3-tk`).

## Zawartość

| Rozdział | Zakładka | Co liczy | Wynik z książki (sprawdzony) |
|---|---|---|---|
| 11 | 11.1 IFC | przyrostowy koszt paliwa przy 100% i 50% | 77 000 / 76 000 Rs/MWh |
| 11 | 11.2 Dwa bloki | równe przyrostowe koszty, limity generacji | P1 = 56,97, P2 = 103,03 MW, λ = 3,854; z limitem 60/100 MW |
| 11 | 11.3 Harmonogram | rozdział dla 1200/900/500 MW + decyzja o wyłączeniu bloku | 740,54/459,46 MW; 470,27/429,73 MW; 500 MW → tylko G2 |
| 11 | 11.4 Straty A-B | przyrostowe straty i współczynnik kary obszaru B | ITL_A = 0, ITL_B = dP_L/dP_gB |
| 11 | 11.5 Wsp. kary | PF1 z warunku PF1·IC1 = PF2·IC2 | PF1 = PF2 = 1,25 |
| 11 | 11.6 Koordynacja | równania koordynacyjne vs metoda wsp. kary | P1 = 103,09, P2 = 150, P_R = 247,9 MW |
| 11 | Ćw. 11 Iteracja | iteracja λ (bisekcja) ze wzorem strat B | P1 = 280,15, P2 = 122,01, P_L = 2,16 MW |
| 11 | 11.7 Hydro + para | moc elektrowni wodnej z ilości wody, „cena wody” | P_h = 175,35 MW, λ = 27 |
| 11 | 11.8 Szczyt | harmonogram hydro-termiczny (3 równania, bisekcja po R) | R ≈ 1,482; P_h = 263,7 / 491,3 MW |
| 12 | 12.1 Łącze bipolarne | moc i napięcie na początku, w środku i na końcu linii DC | 1290 MW; 516 kV; 1270 MW; 508 kV |
| 12 | 12.2 Falownik | kąt wygaszania δ | δ ≈ 19,95° |
| 12 | 12.3 Kąty α i δ | V_d0 = 1,3505·V_L, napięcia DC i prąd łącza | 526,69 / 506,43 kV; 515,18 kV |
| 13 | 13.1 Kąt mocy | moc przyspieszająca, δ po 10 okresach | δ = δ0 + 0,07854 rad (4,5°) |
| 13 | 13.2 Energia wirnika | energia kinetyczna, przyspieszenie | 300 MJ; 150 °/s²; 12,5 obr/min/s |
| 13 | 13.3 Kołysania + zwarcie | E′, równanie kołysań, symulacja zwarcia RK4, czas krytyczny | E′ = 1,11∠21,06°, P_e = 2,22 sin δ |
| 13 | 13.4–13.5 Równe pola | kryterium równych pól dla skoku mocy + linearyzacja | – |

## Interakcja

- **Suwaki** – zmiana od razu przelicza wynik i wykres; przycisk *Wartości z książki* przywraca dane.
- **Wyniki krok po kroku** – pełne rozwiązanie dla bieżących wartości, jak w zeszycie.
- **Wyjaśnienie dla ucznia** – idea, analogia z życia, rozwiązanie z książki, propozycje eksperymentów.
- **Pasek narzędzi Matplotlib** – powiększanie, przesuwanie, zapis PNG.
- **Ruch myszy** nad wykresem – współrzędne na pasku statusu.
- **Kliknięcie w wykres** (11.1, 11.2, 11.6, 13.3, równe pola) – ustawia parametr, np. czas wyłączenia zwarcia.

## Uwagi inżynierskie

- W załączniku część wzorów była obrazkami. Brakujące współczynniki odtworzono z rozwiązań
  (np. z „λ1 = 3 + 0,015 × 60” wynika IFC1 w przykładzie 11.2).
- Tam, gdzie danych nie da się odtworzyć, przyjęto wartości oznaczone w aplikacji jako **ZAŁOŻENIE**:
  stałe koszty a1, a2 (11.2) i a1 (11.3), czasy trwania 1200 MW i 900 MW (11.3) oraz liczby w 11.4,
  który w książce jest przykładem jakościowym. Stałe koszty nie wpływają na rozdział mocy.
- Wykryte błędy druku w książce:
  - 11.3: dla 1200 MW wydrukowano P2 = 429,73 MW; poprawnie jest 459,46 MW.
    Dla 500 MW książka pomija limit 200 MW i podaje niespójne koszty, ale wniosek (pracuje tylko G2) jest poprawny.
  - 11.6: „175 kW” powinno brzmieć 175 MW.
  - 12.3: 506,43·cos 20° = 475,9 kV, a nie 474 kV.
- 11.8: rozwiązanie w książce urywa się; aplikacja rozwiązuje układ do końca.
- 13.3: aplikacja wykracza poza książkę – dodaje symulację zwarcia (RK4), krytyczny kąt i czas wyłączenia.
