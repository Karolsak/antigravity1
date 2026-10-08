# Maszyny prądu stałego – interaktywne przykłady (Tkinter)

Aplikacja `dc_machines_examples_tkinter.py` przerabia wszystkie przykłady z rozdziału
**UNIT III – D.C. Machines** (*Basic Electrical and Instrumentation Engineering*), które da się
odczytać ze skanu. Każdy przykład ma:

- **suwaki z danymi zadania** – domyślnie dokładnie dane z książki (przycisk *Przywróć dane z książki*),
- **rozwiązanie krok po kroku**, przeliczane na żywo po każdej zmianie,
- **wyjaśnienie dla ucznia liceum** (analogie: dynamo rowerowe, wąż ogrodowy, kula śnieżna…)
  i **komentarz inżyniera** (praktyka, zastosowania, pułapki),
- **interaktywne wykresy** – po najechaniu myszą na krzywą pokazuje się odczyt; pasek narzędzi
  matplotlib pozwala powiększać, przesuwać i zapisywać wykres.

## Uruchomienie

```bash
pip install numpy matplotlib
python dc_machines_examples_tkinter.py
```

(Tkinter jest w standardowej instalacji Pythona; na Linuksie może być potrzebny pakiet `python3-tk`).

## Zawartość i weryfikacja z odpowiedziami z książki

| Grupa | Przykład | Wynik aplikacji | Odpowiedź w książce |
|---|---|---|---|
| Podstawy | Zasada działania prądnicy (e = B·l·v·sinθ, komutator) | interaktywna | – |
| Podstawy | Zasada działania silnika (F = B·I·l, moment cewki) | interaktywna | – |
| SEM | Zad. 2: 4 bieguny, φ = 0,07 Wb, 900 obr/min, Z = 440 | 462 V / 924 V | 462 V / 924 V |
| SEM | Zad. 3: Z = 600, 1200 obr/min | 720 V, 600 obr/min | 720 V, 600 obr/min |
| SEM | Zad. 4: 51 × 24 przewody, E = 220 V | 539,2 obr/min, 110 V | ≈ 539 obr/min, 110 V |
| SEM | Q.37: 45 × 18 przewodów, 1200 obr/min | 518,4 V | 518,4 V |
| Prądnice | Przykład 3.12.1 (7,5 kW, 200 V) | E = 224 V | 224 V |
| Prądnice | Przykład 3.12.2 (30 kW, 300 V) | 31,4304 kW | 31,4304 kW |
| Prądnice | Zad. 3.12-2 (10 kW, feedery 0,05 Ω) | 202,5 V / 207,7025 V | 202,5 V / 207,7025 V |
| Prądnice | Q.36 (10 kW, 220 V, feedery 0,1 Ω) | V_t = 224,545 V | 224,5454 V |
| Prądnice | Przykład 3.14.1 (kompound, lampy 500 Ω, 550 V) | E = 554,4 V | skan nieczytelny* |
| Prądnice | Przykład 3.14.2 (krótki bocznik, 7,5 kW, 230 V) | E = 253,785 V, R_L = 7,053 Ω | I_a = 35,006 A … |
| Prądnice | Przykład 3.14.3 (krótki bocznik, 48 kW, 240 V) | E = 249,06 V, R_L = 1,2 Ω | 249,06 V, 1,2 Ω |
| Prądnice | Charakterystyka magnesowania, samowzbudzenie, R krytyczna | interaktywna | – |
| Prądnice | Charakterystyki zewnętrzne (obcowzbudna, bocznikowa, szeregowa, kompound) | interaktywna | – |
| Silniki | Q.26: SEM wsteczna (220 V, 0,5 Ω, 20 A) | 210 V | 210 V |
| Silniki | Maszyna jako prądnica i silnik (2 A, 20 A, 200 V) | 211 V, 21,887 N·m | 211 V, 21,887 N·m |
| Silniki | Przykład 3.25.1 (200 V, 100 A, 750 obr/min) | 230,42 N·m | 230,424 N·m |
| Silniki | Przykład 3.26.1 (reakcja twornika −4 %) | 993,5 obr/min | ≈ 993,5 obr/min |
| Silniki | Charakterystyki T(I_a), N(I_a), N(T) + zastosowania | interaktywna | – |
| Rozruch | Rozrusznik 3-/4-punktowy, dobór stopni, symulacja rozruchu | interaktywna | – |
| Regulacja | Przykład 3.34.1 (rezystor w tworniku 1,1 Ω) | 424,3 / 372,7 obr/min | 424,3 / 372,7 obr/min |
| Regulacja | Regulacja strumieniem 800 → 1000 obr/min | R_dod ≈ 69 Ω | skan nieczytelny* |
| Regulacja | Przykład 3.35.1 (silnik szeregowy, diverter) | 51,96 A, 912,7 obr/min | 51,9615 A, E_b2 = 239,607 V |
| Regulacja | Przykład 3.35.3 (silnik szeregowy 440 V, R = 3 Ω) | 1349,7 obr/min, P₂/P₁ = 0,675 | treść ucięta* |
| Regulacja | Regulacja napięciem twornika + osłabianie pola | interaktywna | – |
| Uniwersalny | Silnik uniwersalny AC/DC | interaktywna | – |

\* Tam, gdzie skan jest uszkodzony, brakujące dane uzupełniono (np. liczba lamp z rysunku,
klasyczne wersje zadań 3.34/3.35.3). Każde takie założenie jest opisane w aplikacji przy danym
przykładzie, a wszystkie wartości można zmienić suwakami.

Niektóre zadania kontrolne ze skanu (np. prądnica szeregowa 1500 obr/min, silnik 200 V z
odpowiedzią 244,67 obr/min) mają zbyt mało czytelnych danych, by je odtworzyć – zostały pominięte.
