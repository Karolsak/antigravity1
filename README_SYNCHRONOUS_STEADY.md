# Maszyny synchroniczne – stan ustalony (Tkinter)

Plik `synchronous_machine_steady_tkinter.py` to interaktywna aplikacja z 11 zakładkami. Każda zakładka ma suwaki, wykresy Matplotlib (z zoomem), wyniki liczbowe i objaśnienie dla ucznia liceum, napisane z perspektywy doświadczonego inżyniera.

| Zakładka | Temat |
|---|---|
| S1 | Próba biegu jałowego (OCC) i zwarcia (SCC): Xs nienasycona/nasycona, SCR, wartości w Ω |
| S2 | Generator cylindryczny: Ef = V + (Ra + jXs)·Ia, wykres wskazowy, kąt mocy δ |
| S3 | Zmienność napięcia ΔU %, charakterystyki zewnętrzne V(Ia) i regulacyjne If(Ia) |
| S4 | Charakterystyka kątowa P(δ), Q(δ), moc maksymalna, utrata synchronizmu |
| S5 | Silnik synchroniczny: krzywe V, współczynnik mocy, granica stabilności |
| S6 | Poprawa cos φ zakładu przewzbudzonym silnikiem synchronicznym (trójkąty mocy) |
| S7 | Bieguny jawne: teoria dwóch reakcji (Xd, Xq), moc reluktancyjna |
| S8 | Wykres kołowy P-Q (capability): granice prądu twornika, wzbudzenia, turbiny, stabilności |
| S9 | Straty i sprawność w funkcji obciążenia i cos φ |
| S10 | Praca równoległa dwóch generatorów: statyzm f(P) i V(Q), „wykres domkowy” |
| S11 | Synchronizacja z siecią: dudnienia, synchroskop, prąd wyrównawczy |

Uruchomienie:
```
pip install numpy matplotlib
python synchronous_machine_steady_tkinter.py
```
