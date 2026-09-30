# Ochrona rozproszonych źródeł odnawialnych (RDG) — interaktywne przykłady (Tkinter)

Plik `ochrona_oze_tkinter.py` to aplikacja do rozdziału 18 *„Protection of Renewable Distributed Generation System”*.
Rozdział jest opisowy, bez przykładów liczbowych. Dlatego każde zagadnienie, rysunek (18.1–18.11), tabelę (18.1, 18.2)
i pytanie kontrolne (18.8) zamieniłem na **interaktywny przykład obliczeniowy**. Każda zakładka ma:

- suwaki i listy wyboru z parametrami,
- wykresy Matplotlib w ciemnym motywie; pasek narzędzi pozwala je przybliżać i zapisywać,
- panel „Wyniki” z liczbami i oceną OK / przekroczenie,
- objaśnienie dla ucznia liceum: intuicja, wzory, przykład liczbowy i sekcja **SPRÓBUJ** z eksperymentem.

## Uruchomienie

```
pip install numpy matplotlib
python ochrona_oze_tkinter.py
```
(`tkinter` jest częścią standardowego Pythona; na Linuksie może być potrzebny pakiet `python3-tk`.)

## Zakładki

| Zakładka | Temat (sekcja rozdziału) | Model / co liczy |
|---|---|---|
| 00 | Przegląd, mikrosieć (rys. 18.1, 18.2–18.4) | Schemat mikrosieci i mapa rozdziału |
| 01 | Profil napięcia i straty (18.1.2, 18.2.2) | Rozpływ mocy metodą *backward–forward sweep* w linii 15 kV; przepływ zwrotny, hosting capacity, krzywa strat w kształcie „U” |
| 02 | Prąd zwarciowy, oślepienie, fałszywe wyłączenia (18.2.4, 18.4.1) | Zwarcie 3f z dopływem OZE (maszyna X''d albo falownik z ograniczeniem prądu), IEC 60255 SI/VI/EI, prąd wsteczny przy zwarciu na linii sąsiedniej |
| 03 | Koordynacja reklozer–bezpiecznik (18.4.1) | Krzywe TCC, warunek *fuse saving* t_R < 0,75·t_topienia, zapas koordynacji w funkcji prądu OZE |
| 04 | Uziemienie i zwarcie SLG, 173 % (18.2.5, rys. 18.5) | Składowe symetryczne dla 5 sposobów uziemienia (bezpośrednie/R_N, izolowana, Petersen, wyspa, zig-zag), wskazy napięć |
| 05 | Wyspa, strefa NDZ (18.2.6, 18.3.3, 18.3.6) | Obciążenie RLC o dobroci Qf: U' = √(P_OZE/P_obc), f' = f0·√(Q_L/Q_C); mapa NDZ w płaszczyźnie ΔP–ΔQ |
| 06 | ROCOF i odciążanie SCO (18.2.8) | Równanie wahadłowe 2H·dΔf/dt = P_m − P_e, regulator ze statyzmem, SCO statyczne i adaptacyjne (z ROCOF) |
| 07 | Metody aktywne AFD / AFDPF / SMS (18.3.4–18.3.5) | Warunek fazy atan(Qf(f/f_r − f_r/f)) = θ_falownika, iteracja cykl po cyklu, czas wykrycia w funkcji Qf |
| 08 | Mikrosieć: tryb sieciowy i wyspowy (18.4.1–18.4.4, 18.4.9) | Wkłady do prądu zwarcia (sieć, SG, falowniki, magazyn), FCL, zabezpieczenie adaptacyjne z grupami nastaw A/B |
| 09 | Zabezpieczenie odległościowe i różnicowe (18.4.6–18.4.7) | Strefy MHO, impedancja pozorna z dopływem i R_f; charakterystyka różnicowa stabilizowana |
| 10 | SPZ przy pracującym OZE (18.4.1) | Odpływ kąta wyspy δ(t), prąd wyrównawczy 2·sin(δ/2)/X, oś czasu cyklu SPZ (łuk podtrzymany przez OZE) |
| 11 | Turbina stałoobrotowa SCIG (18.5.1, rys. 18.6) | Schemat zastępczy: T(s), P/Q, kompensacja kondensatorami; rozruch bezpośredni i z soft-starterem |
| 12 | Cp(λ,β), MPPT, stała i zmienna prędkość (18.5.2, rys. 18.7, tab. 18.1) | Model Cp Heiera, krzywe mocy, energia roczna przy rozkładzie Weibulla |
| 13 | DFIG — przepływ mocy (18.5.3, rys. 18.8, tab. 18.2) | P_s = P_m/(1−s), P_r = −s·P_s, praca pod- i nadsynchroniczna, moc przekształtnika ≈ s_max |
| 14 | DFIG — zapad napięcia i crowbar (18.5.4, rys. 18.9) | Symulacja RK4 modelu wektorowego DFIG z regulatorem PI w RSC, crowbar, chopper i obwód DC |
| 15 | Fotowoltaika (18.6, rys. 18.10) | Model jednodiodowy I–U; prąd wsteczny w stringu i bezpieczniki gPV; U_oc przy mrozie; przepięcie indukowane przez piorun |
| 16 | Sieci przyszłości (18.7, rys. 18.11) | Prąd zwarcia na szynach a zdolność wyłączalna wyłącznika, dławik szeregowy, schemat komunikacji TSO–DNO–trader |
| 17 | Pytania kontrolne 18.8 | Odpowiedzi na 10 pytań i drzewo klasyfikacji metod wykrywania wyspy |

## Najważniejsze wnioski

1. **OZE zmienia kierunek przepływu mocy.** Napięcie na końcu linii może przekroczyć 1,05–1,10 p.u. Straty najpierw maleją,
   a potem rosną (zakładka 01).
2. **Źródła wirujące zwiększają prąd zwarcia, a falownikowe ledwo go zmieniają.** Pierwsze powodują oślepienie zabezpieczeń,
   fałszywe wyłączenia i utratę koordynacji reklozer–bezpiecznik. Drugie sprawiają, że w wyspie prąd zwarcia jest za mały
   dla klasycznych zabezpieczeń nadprądowych (zakładki 02, 03, 08).
3. **Wyspa bez uziemienia daje 173 % napięcia w fazach zdrowych** przy zwarciu doziemnym. Pomaga transformator zig-zag
   (zakładka 04).
4. **Metody pasywne mają strefę NDZ.** Metody aktywne ją zmniejszają, ale przy dużym Qf też zawodzą. Norma wymaga
   odłączenia w ciągu 2 s (zakładki 05–07).
5. **SPZ przy pracującym OZE** oznacza podtrzymany łuk i załączenie w przeciwfazie. Pomagają kontrola synchronizmu
   i szybkie zabezpieczenie antywyspowe (zakładka 10).
6. **DFIG przy zapadzie napięcia traci kontrolę nad prądem wirnika.** Crowbar i chopper chronią przekształtnik RSC
   i obwód DC (zakładka 14).
7. **W instalacjach PV** potrzebne są bezpieczniki stringów przy N_p ≥ 3, sprawdzenie U_oc przy mrozie, ochronniki SPD
   i krótkie pętle okablowania (zakładka 15).
