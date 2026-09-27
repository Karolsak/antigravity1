# Modele zaawansowane maszyn i stany nieustalone maszyn DC (Tkinter)

Plik `advanced_models_dc_transients_tkinter.py` to interaktywna aplikacja do rozdziałów 7 i 8
książki I. Boldea, *Electric Machines: Steady State, Transients, and Design with MATLAB*.
Każda z 19 zakładek ma suwaki (albo pole do wpisania liczby), wyniki liczbowe, cztery wykresy
Matplotlib (z zoomem i zapisem do pliku) oraz wyjaśnienie krok po kroku dla ucznia liceum,
z uwagami inżyniera praktyka. Wszystkie wyjaśnienia są też w pliku
[`WYJASNIENIA_MODELE_ZAAWANSOWANE_DC.md`](WYJASNIENIA_MODELE_ZAAWANSOWANE_DC.md).

## Rozdział 7 – modele zaawansowane

| Zakładka | Temat | Co pokazuje |
|---|---|---|
| 7.2–7.4 | Model fizyczny dq, SEM pulsacyjna i rotacji | animacja osi dq dla ωb = 0 / ωr / ω1; e = e_p + e_m sprawdzone numerycznie |
| 7.5 (Zad. 7.3) | SM/PMSM w osiach wirnika | moment od magnesu i reluktancyjny, MTPA, wykres wektorowy, stan nieustalony i_d, i_q |
| 7.6 | Silnik indukcyjny w osiach dq | rozruch bezpośredni 4 kW; wybór osi zmienia i_d, i_q, a moment pozostaje ten sam |
| 7.7 | Nasycenie magnetyczne | krzywa magnesowania, indukcyjności L_dm, L_dmt, L_qmt, mapa sprzężenia skrośnego L_dqm |
| 7.8 | Efekt naskórkowości | k_R(f), k_X(f) głębokiego pręta, dopasowanie 1/2/3 obwodów R-L, rozkład gęstości prądu |
| 7.9–7.10 | Transformacja Parka i wektor przestrzenny | abc → dq0, składowa zerowa, harmoniczne, asymetria, równoważność mocy p i q |
| 7.11 | Model wysokiej częstotliwości | napięcie wspólne PWM, napięcie wału (BVR), wyładowania EDM, impedancja w funkcji częstotliwości |
| Zadania 7.1–7.7 | Odpowiedzi + SRM 6/4 | moment bez tętnień dla indukcyjności sinusoidalnej, tętnienia dla trapezowej |

## Rozdział 8 – stany nieustalone maszyn DC

| Zakładka | Temat | Najważniejsze wyniki (dane domyślne) |
|---|---|---|
| Przykład 8.1 | Prądnica: skok V_F o 20% i zwarcie | R_load = 2 Ω, ω_r·L_dm = 42 V/A, ΔV_a = 40 V, T_F = 0,5 s, I_zw = 2100 A = 21·I_n, T_e = 5 ms |
| Przykład 8.2 | Silnik PM: 12 V → 10 V | I_a = 4,63 A, Ψ_PM = 0,0729 Wb, ω_rf = 129,6 rad/s; J = 2·10⁻⁴ daje odpowiedź oscylacyjną, J = 10⁻³ aperiodyczną |
| 8.4.2 | Zmiana strumienia (osłabianie pola) | model nieliniowy i zlinearyzowany, γ3 = −R_F/L_F, udar prądu twornika |
| 8.4.3 (Zad. 8.4) | Silnik szeregowy | L_dm = 11,14 mH, T_e = 222,8 N·m, wartości własne przy 1500 i 750 obr/min, skok obciążenia +20% |
| 8.5 | Kaskada PI (prąd + prędkość) | nastawy z kompensacji biegunów, anti-windup, ograniczenie prądu, Bode i zapas fazy |
| 8.6 (Zad. 8.5) | Przekształtnik DC-DC | prąd ciągły i przerywany, λ, V_av = αV + (1−λ)E |
| 8.7 (Lab 8.1) | Próby wyznaczania parametrów | zanik prądu → L_a (z V_d i bez V_d), wybieg → J |
| Zad. 8.1 | Narastanie prądu prądnicy | T_F = 0,1 s, T_a = 0,909 s, I_a∞ = 87,27 A |
| Zad. 8.2 | Rozruch silnika PM | I_rozr = 100 A, τ_m = 4,4 ms, ω_f = 156,4 rad/s (1493 obr/min) |
| Zad. 8.3 | Zmniejszenie obciążenia o 20% | I_a: 4,89 → 3,93 A, n: 1500 → ok. 1515 obr/min |

## Uruchomienie

```
pip install numpy scipy matplotlib
python advanced_models_dc_transients_tkinter.py
```

Na Linuksie może być potrzebny pakiet `python3-tk`. Obliczenia działają w osobnym wątku, więc
okno nie zamarza podczas przesuwania suwaków. Wyjaśnienia można ponownie wyeksportować do
Markdown poleceniem:

```
python advanced_models_dc_transients_tkinter.py --export-md WYJASNIENIA_MODELE_ZAAWANSOWANE_DC.md
```
