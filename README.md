# Tehnička dokumentacija
## Vizuelizacija bazičnih operacija konvolucionih mreža

---

## 1. Pregled projekta

CNN Vizualizator je edukativni softver otvorenog koda razvijen u programskom jeziku Python, namenjen za vizuelni prikaz osnovnih operacija dubokih konvolucionih neuronskih mreža. Aplikacija omogućava korisnicima da isprobaju različite parametre konvolucije, tipove sloja za sažimanje, kao i da posmatraju detekciju obrazaca u interaktivnom okruženju.

---

## 2. Korišćene tehnologije

| Kategorija         | Tehnologija / Biblioteka                          |
|--------------------|---------------------------------------------------|
| Programski jezik   | Python 3.12.6                                     |
| GUI okviri         | PyQt5 (korisnički interfejs), Matplotlib (3D renderovanje) |
| Numerička obrada   | NumPy (matrične operacije)                        |
| Obrada videa       | OpenCV                                            |
| Distribucija       | PyInstaller                                       |

---

## 3. Arhitektura sistema

Aplikacija je podeljena u četiri logička sloja:

1. **Core (Logički sloj):** Moduli `convolution.py`, `pooling.py`, `pattern.py`. Ovde se unapred vrše svi matematički proračuni.
2. **Visualization (Sloj vizuelizacije):** `renderer_3d.py` (crta kocke u 3D prostoru) i `step_animator.py` (upravlja stanjima animacije).
3. **GUI (Korisnički interfejs):** `app.py` (glavni prozor) i `controls.py` (paneli sa dugmićima i unosima).
4. **Utilities (Pomoćni alati):** `exporter.py` (snimanje videa) i `random_gen.py` (generisanje podataka).

---

## 4. Opis modula

### `core/` — Matematika

- **`ConvolutionEngine`** — Izračunava konvoluciju. Podržava više kanala (RGB), padding, stride (korak) i bias. Svi koraci su unapred izračunati, kako bi korisnik imao mogućnost kretanja unapred i unazad.
- **`PoolingEngine`** — Implementira četiri tipa pooling-a:
  - `Max` — najveća vrednost
  - `Average` — prosek
  - `L2` — norma
  - `Weighted Average` — težinski prosek
- **`PatternEngine`** — Generiše ulaznu mapu dimenzija 12×12 koja sadrži pozitivne i negativne vrednosti filtera (šablone), čime se demonstrira mogućnost filtera da prepoznaju oblike.

### `visualization/` — Grafika

- **`renderer_3d.py`** — Koristi klasu `Poly3DCollection` za crtanje kockica koje predstavljaju neurone. Dinamički ispisuje matematičku formulu iznad 3D prikaza kako bi korisnik video postupak računanja u realnom vremenu.
- **`step_animator.py`** — Povezuje matematičku logiku (klase unutar `core/` paketa) i grafiku (unutar `Renderer` klase).

## 5. Korisničko uputstvo

Korisničko uputstvo nalazi se u fajlu [TUTORIAL.md](./TUTORIAL.md)
