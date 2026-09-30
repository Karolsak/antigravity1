# Zabezpieczenia silnika indukcyjnego – interaktywna aplikacja (Tkinter)

Plik `induction_motor_protection_tkinter.py` to aplikacja z 10 zakładkami do rozdziału 12 „Induction Motor Protection”
(*Power System Protection in Smart Grid Environment*). W każdej zakładce są suwaki, wykresy Matplotlib (z zoomem
i przesuwaniem), wyniki liczone na żywo oraz wyjaśnienie krok po kroku, napisane zrozumiale dla ucznia liceum.

| Zakładka | Temat |
|---|---|
| 0 · Przegląd | Tor zasilania silnika (rys. 12.1), kody ANSI 49/48/51LR/50/50N/46/27/59/37, podsumowanie wyników |
| 1 · Obwód zastępczy | Obwód Steinmetza dla składowej zgodnej i przeciwnej, momenty T1, T2, bilans mocy (12.2–12.3) |
| 2 · Rozruch i utyk | Symulacja ODE `2H·dω/dt = Te − TL` z cieplnym modelem wirnika i zabezpieczeniami 49/51LR/48, kalkulator 12.73 |
| 3 · Model cieplny | Model 3-węzłowy (12.44–12.46), przekaźnik 49 (12.67–12.71), żywotność izolacji według reguły 10 °C |
| 4 · Przykł. 12.1–12.3 | Prądy silnika 13 KM / 230 V, nastawy 50, 50N, 51LR, 48, krzywe TCC, rezystor stabilizujący (12.76) |
| 5 · Przykł. 12.4–12.5 | Zanik napięcia szczątkowego po odłączeniu zasilania, bezpieczne ponowne załączenie |
| 6 · Przykł. 12.6 | Prąd równoważny cieplnie `I_eq = √(I1² + K·I2²)` |
| 7 · Przykł. 12.7–12.8 | Asymetria napięć (NEMA, VUF), zamknięty trójkąt napięć, składowe prądów, tabela 12.2 |
| 8 · Składowa przeciwna | `I2/I1 = (Ist/Irun)·(V2/V1)`, grzanie wirnika (12.83–12.87), wirnik pierścieniowy |
| 9 · Zadania 12.9–12.10 | Moc składowej zgodnej (gwiazda), skutki przepięć 5–25 % (zadania nierozwiązane w książce) |

## Wyniki przykładów

| Przykład | Wynik |
|---|---|
| 12.1 | In ≈ 600·13/230 = **33,9 A**, I0 ≈ **17,0 A**, Ilr ≈ **170–203 A** |
| 12.2 | Isc = 1,25·Ist = **212 A / 254 A** → wtórnie (CT 40/1) **5,30 A / 6,36 A**; 100 ms (≤120 %), 40 ms powyżej |
| 12.3 | 50N: 0,25·In ≈ **8,5 A** (0,21 A wt.); 51LR/48: 2·In, **15 s / 6,5 s** |
| 12.4 | T_op = (31+0,31)/(2π·50·0,26) = **0,383 s**; V(T_op) = **147,2 V**; V(0,5 s) = **108,5 V** |
| 12.5 | V(4·T_op) = 400·e⁻⁴ = **7,33 V** |
| 12.6 | I_eq = √(0,912² + 3·0,92²) = **1,836 j.w.** |
| 12.7 | U_śr = 380 V, asymetria NEMA **3,95 %**, VUF (przy symetrycznych kątach) **1,99 %**, VUF dokładny (IEC) **4,01 %** |
| 12.8 | I1 = **15,43 A**, I2 = **1,84 A**, I_eq = **15,76 A** |
| 12.9 | S1 = 3·V1·I1* (I0 = 0 w gwieździe), liczbowo z danych 12.7–12.8: \|S1\| ≈ **10,16 kVA** (≈ 99,8 % mocy) |
| 12.10 | Przeciążenie ≈ ΔU; ΔP_Fe = (1+ΔU)ⁿ − 1, gdzie n = 2…2,6: 5 % → 10–14 %, 10 % → 21–28 %, 15 % → 32–44 %, 20 % → 44–61 %, 25 % → 56–79 % |

**Uwagi inżynierskie:**
- W przykładzie 12.3 książka podaje 2·In = 80 A, czyli przyjmuje In = 40 A (prąd pierwotny przekładnika 40/1).
  Z przykładu 12.1 wynika In ≈ 33,9 A, więc 2·In ≈ 68 A. Aplikacja liczy z In z przykładu 12.1 i zaznacza tę rozbieżność.
- W przykładzie 12.7 przyjęto, że kąty napięć są dokładnie co 120°. Przy różnych modułach trójkąt napięć
  międzyfazowych nie mógłby się wtedy zamknąć. Dokładny VUF wynosi około 4 %, a nie 2 %. Aplikacja pokazuje obie wartości.

## Uruchomienie
```
pip install numpy matplotlib
python induction_motor_protection_tkinter.py
```
Na Linuksie może być potrzebny pakiet `python3-tk`.
