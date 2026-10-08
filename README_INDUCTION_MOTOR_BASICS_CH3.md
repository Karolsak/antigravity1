# Silnik indukcyjny – podstawy (rozdział 3) – aplikacja Tkinter

Plik `induction_motor_basics_ch3_tkinter.py` to interaktywna aplikacja z 17 zakładkami. Obejmuje wszystkie przykłady z rozdziału 3 „Induction Motor Basics” (AC motor control and electrical vehicle applications). Każda zakładka ma suwaki, wykresy Matplotlib (z zoomem i zapisem do pliku), wyniki liczbowe krok po kroku i objaśnienie dla ucznia liceum.

| Zakładka | Temat |
|---|---|
| 3.1 Moc | Ćw. 3.1: bilans mocy (diagram Sankeya), `P_cu2 = s·P_ag`, Tabela 3.1 strat |
| 3.2 T(s) | Ćw. 3.2: moment i prąd w funkcji poślizgu, schemat T a schemat zmodyfikowany, praca silnikowa, generatorowa i hamowanie przeciwprądem |
| 3.3 U | Ćw. 3.3 / zad. 3.7: wpływ napięcia na poślizg przy stałym momencie |
| 3.4 rr | Ćw. 3.4 i 3.7: trzy rezystancje wirnika, moment krytyczny, prąd, sprawność |
| 3.5 rr? | Ćw. 3.5: wyznaczanie `rr` z tabliczki znamionowej |
| 3.6 Znam. | Ćw. 3.6: prąd, cos φ, moment znamionowy i krytyczny, wykres wskazowy |
| Stabilność | 3.2.4: obszar stabilny i niestabilny, symulacja skoku obciążenia (utyk) |
| Harmon. | 3.2.5: momenty pasożytnicze 5. i 7. harmonicznej, pełzanie przy n_s/7 |
| 3.8 Żłobek | Ćw. 3.8: indukcyjność rozproszenia żłobka (prawo Ampère'a, energia pola), połączenia czołowe |
| 3.9-12 Koło | Ćw. 3.9–3.12: wykres kołowy Heylanda, linie zerowej mocy i zerowego momentu, σ, s_r, s_m |
| 3.13 Naskórek | Ćw. 3.13: wypieranie prądu, współczynniki Kr i Kx, rozkład gęstości prądu w pręcie |
| NEMA | 3.5.1: klatka podwójna, pręt głęboki, klasy NEMA A/B/C/D |
| Rozruch | 3.5.2: rozruch bezpośredni i softstart, dynamiczny model αβ (RK4) |
| Tyrystor | Zad. 3.6: sterownik tyrystorowy, Vrms(α), widmo napięcia |
| Napięcie | 3.6.1: regulacja prędkości samym napięciem (wentylator) |
| VVVF | 3.6.2: U/f = const, podbicie napięcia, regulator prędkości z poślizgiem |
| Zadania 3.1-3.13 | Wszystkie zadania końcowe (lista rozwijana): rozwiązania, wykresy i objaśnienia |

Uruchomienie:
```
pip install numpy matplotlib
python induction_motor_basics_ch3_tkinter.py
```

Uwagi do treści zadań:
- **Zad. 3.8**: w treści podano „reactive power = 50 W”. Jednostka W wskazuje na moc czynną, więc tak to liczymy. Wartość można zmienić suwakiem.
- **Zad. 3.9**: „measured shaft power” traktujemy jako moc pobraną. Przy zablokowanym wale moc na wale wynosi 0.
- **Zad. 3.10**: wzór na moment liczymy z liczbą par biegunów P/2, tak jak w rozdziale 4.
- **Zad. 3.12**: nie podano `rr`. Momenty z wykresu kołowego od niego nie zależą, a poślizgi tak. Dlatego `rr` jest suwakiem z założoną wartością.
