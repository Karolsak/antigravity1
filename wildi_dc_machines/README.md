# Maszyny prądu stałego – interaktywne przykłady (Wildi)

Aplikacja Tkinter do wszystkich 9 przykładów rachunkowych z rozdziałów
„Direct-Current Generators” (G1, G2) i „Direct-Current Motors” (M1–M7),
plus zakładka analizy sterowania (Bode, zpk, Nyquist, step) dla układów dynamicznych.

## Instalacja i uruchomienie
```bash
pip install numpy scipy matplotlib pillow
```
```bash
python maszyny_dc_wildi.py
```
(Na Linuksie może być potrzebny pakiet `python3-tk`.)

## Zakładki
| Zakładka | Treść |
|---|---|
| Start | opis, wzory, tabela weryfikacji: wynik z książki vs aplikacja (34 wielkości) |
| G1 · E₀ | napięcie indukowane, rola komutatora i liczby cewek |
| G2 · szczotki | prądy szczotek/cewek, napięcie na cewkę w maszynie 12-biegunowej |
| M1 · rozruch | prąd rozruchowy, SEM, symulacja ODE rozruchu |
| M2 · moment | prąd, przewodniki/żłobek, moment, strumień, kontrola B i okładziny prądowej |
| M3 · Ward-Leonard | praca silnikowa i hamowanie odzyskowe, płaszczyzna T–n, symulacja ODE |
| M4 · bocznikowy | prądy, SEM, bilans mocy, charakterystyka n(T) |
| M5 · szeregowy | charakterystyki w pu, sprawność |
| M6 · ham. dynam. | stała T₀, czasy hamowania, porównanie z wybiegiem |
| M7 · przeciwprąd | hamowanie przeciwprądem, porównanie z dynamicznym |
| Bode · zpk · Nyquist · step | M1, M4, M3, M6, M7 – pętla otwarta i zamknięta z PI, automatyczna interpretacja |

Parametry oznaczone „(zał.)” (L, J, B, τ_f, K_g) nie występują w książce – przyjęto
typowe wartości; można je zmieniać suwakami.
