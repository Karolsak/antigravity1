# Przekształtniki, bilans mocy, ekonomia i elementy sieci (Tkinter)

Interaktywna aplikacja z **wszystkimi przykładami z rozdziałów 5–7** książki
D. S. Kirschen, *Power Systems: Fundamental Concepts and the Transition to Sustainability* (Wiley 2024).

Każda zakładka ma suwaki (domyślnie ustawione na dane z książki), wykresy Matplotlib (z zoomem i zapisem),
pole wyników z porównaniem z książką oraz objaśnienie krok po kroku napisane dla ucznia liceum.

| Zakładka | Przykład / temat |
|---|---|
| 5.1 Moc P-Q | Przykład 5.1: koło możliwości przekształtnika, P_max przy danym pf |
| 5.3 PWM mostek H | Mostek H, PWM bipolarny/unipolarny, widmo harmonicznych |
| 5.3 Sterowanie P i Q | P = EV sinδ / X, Q(E), wykres wskazowy |
| 5.4 PV i MPPT | Charakterystyka panelu, algorytm perturb & observe (chmura) |
| 5.4 Wiatr typ 3/4 | f = pGN/120 (typ 4), podział mocy stojan/wirnik w DFIG (typ 3) |
| 6.1 Statyzm | Przykład 6.1 + symulacja RK4 częstotliwości po skoku obciążenia, regulacja wtórna |
| 6.2 Dwa generatory | Przykład 6.2: podział obciążenia przez statyzm |
| 6.3-6.4 Dispatch | Rozdział ekonomiczny (Lagrange), wpływ OZE na λ |
| 6.5-6.6 Limity | Ograniczenia P_min/P_max (iteracje), koszt średni i krańcowy (Rys. 6.11) |
| 6.7-6.8 Łamane | Krzywe odcinkowo-liniowe, algorytm merit order, jednostka krańcowa |
| 6.9-6.10 Elastyczność | Przesunięcie obciążenia i arbitraż magazynem (sprawność) |
| 6.11 Ile jednostek | Wszystkie kombinacje jednostek, Rys. 6.12 |
| 6.12 Kombinacje | Eksplozja kombinatoryczna (2^N)^T |
| 6.13 Jaś i Małgosia | Unit commitment MILP: min. czas pracy/postoju, koszt rozruchu |
| 6.14 Rezerwa | Unit commitment z rezerwą i ograniczeniem szybkości zmian mocy |
| 7.1-7.3 Linia | Schemat π, efekt Ferrantiego, linia obciążona, wykres wskazowy |
| 7.4 Transformator | Transformator idealny, przebiegi i wskazy |
| 7.5 Prądy znam. | I = S / (√3 V_LL) |
| 7.6 Jednostki wzgl. | System generator–T1–linia–T2–odbiór w per unit |
| 7.7 Zaczepy | Przełącznik zaczepów, model π, V_L(t), P i Q źródła |

## Uruchomienie
```
pip install numpy matplotlib scipy
python power_conversion_examples_tkinter.py
```
Testy (34 porównania z wynikami z książki):
```
pip install pytest
python -m pytest test_power_conversion_core.py
```

## Pliki
- `power_conversion_examples_tkinter.py` – GUI (zakładki, wykresy, objaśnienia)
- `power_conversion_core.py` – obliczenia (bez GUI)
- `test_power_conversion_core.py` – testy względem liczb z książki

## Uwagi
- W załączonym tekście nie wydrukowały się funkcje kosztu z Przykładu 6.3. Odtworzono je z tabeli
  Przykładu 6.7: C1 = 2100 + 8P + 0,10P², C2 = 7200 + 7P + 0,06P², C3 = 6250 + 9P + 0,07P². Dają
  dokładnie wyniki z książki (λ = 47,012 / 66,547 / 37,244 $/MWh, C = 37 523,69 $/h).
- Unit commitment liczy solver MILP `scipy.optimize.milp` (HiGHS). Przykład 6.13 zgadza się co do dolara
  (296 000 / 346 000 / 521 000 / 896 000 $). W tekście książki wiersze tabeli dla kroku z minimalnym
  czasem postoju są zamienione miejscami; poprawny harmonogram daje 521 000 $.
- W Przykładzie 6.14 solver znajduje harmonogramy z rezerwą (2120 $ zamiast 2125 $) i z rampami
  (2210 $ zamiast 2225 $), które spełniają wszystkie ograniczenia i są tańsze. Harmonogramy z książki też
  są dopuszczalne – aplikacja liczy ich koszt dla porównania. W wariancie z rampami przyjęto, że rezerwa
  jednostki nie przekracza jej szybkości zmian mocy.
- Linia z Przykładu 7.1: r = 0,3 Ω/km, x = 0,9 Ω/km, y = 2 µS/km (wartości odtworzone z wyników
  0,063 + j0,189 pu i −j3,361 pu).
