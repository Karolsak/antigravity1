# Sterowanie silnikami AC i pojazdy elektryczne: PWM i PMSM (Tkinter)

Plik `ac_drives_pwm_pmsm_tkinter.py` to interaktywna aplikacja z 12 zakładkami. Każda zakładka ma suwaki, wykresy Matplotlib (z zoomem), wyniki liczbowe i objaśnienie dla ucznia liceum.

| Zakładka | Temat |
|---|---|
| P1 | Sinusoidalne PWM jednej gałęzi: nośna, impulsy, widmo (ma, mf) |
| P2 | Trójfazowy falownik SPWM: napięcie międzyprzewodowe, prąd obciążenia R-L-E, THD |
| P3 | Wstrzykiwanie 3. harmonicznej i metoda min-max (+15,5 % napięcia) |
| P4 | SVPWM: sześciokąt wektorów, sektory, czasy T1/T2/T0, wzór przełączeń |
| P5 | Przemodulowanie i praca six-step (0,612 → 0,707 → 0,78 Vdc) |
| P6 | Czas martwy: błąd napięcia ΔV = td·fc·Vdc, zniekształcenia |
| M1 | Model PMSM w dq: Vd, Vq, moment, moc, wykres wskazowy |
| M2 | Moment a kąt prądu, moment reluktancyjny, MTPA |
| M3 | Okrąg prądu, elipsy napięcia, osłabianie pola |
| M4 | Obwiednia moment/moc–prędkość napędu EV |
| M5 | Symulacja FOC: kaskada regulatorów PI prędkości i prądu, skok obciążenia |
| M6 | Pojazd EV: siły oporu, prędkość maksymalna, czas 0–100 km/h |

Uruchomienie:
```
pip install numpy matplotlib
python ac_drives_pwm_pmsm_tkinter.py
```
