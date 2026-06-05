# Korisničko uputstvo za upotrebu
## Vizuelizacija bazičnih operacija konvolucionih mreža

---

## 1. Pokretanje i preuzimanje aplikacije

Postoje dva načina za pokretanje aplikacije:

### Opcija 1: Preuzimanje gotove verzije *(Preporučeno za krajnje korisnike)*

Stabilne, kompajlirane verzije aplikacije (eng. *Releases*) nalaze se na **GitHub** repozitorijumu projekta. Preuzimanjem gotovog `.exe` fajla sa GitHub-a izbegavate potrebu za ručnom instalacijom Pythona i biblioteka.

### Opcija 2: Pokretanje iz izvornog koda

Ako želite da pokrenete aplikaciju direktno kroz Python okruženje, instalirajte zavisnosti koristeći sledeću komandu:

```bash
pip install numpy matplotlib PyQt5 opencv-python
```

Pokretanje se vrši pozivom glavnog skripta iz korenskog direktorijuma projekta:

```bash
python main.py
```

---

## 2. Rad sa grafičkim interfejsom

- **Podešavanje parametara (Levi panel):** Izaberite karticu (Konvolucija, Pooling ili Detekcija) i unesite parametre poput veličine filtera, koraka (stride) i broja kanala. Kliknite na dugme `"Generiši"` da inicijalizujete matricu.

- **Manipulacija 3D prikazom (Središnji panel):** Kliknite i držite levi taster miša na 3D prikazu da biste rotirali strukture.

- **Zumiranje i pomeranje prikaza:** Za promenu perspektive i uvećavanje specifičnih delova 3D modela koriste se sledeće ikonice:
  - **Ikona sa lupom (Zoom-to-rectangle):** Aktivira režim za zumiranje. Klikom na ovu alatku, a zatim klikom i prevlačenjem miša preko željene regije na grafikonu, iscrtavate pravougaonik koji definiše prostor koji će se uvećati. Pogodno je za detaljan pregled pojedinačnih ćelija i parcijalnih proizvoda konvolucije.
  - **Ikona sa krstićem (Pan/Zoom):** Kada je ova opcija aktivna, držanjem **levog tastera miša** možete pomerati ceo grafikon u svim pravcima, dok držanjem **desnog tastera miša** i pomeranjem gore-dole vršite glatko zumiranje celokupnog prikaza.
  - **Ikona sa kućicom (Home):** Resetuje perspektivu, zum i sve izmene na podrazumevani početni položaj.

- **Čuvanje prikaza kao slike:**
  - **Ikona sa disketom (Save the figure):** Poslednje dugme na desnoj strani trake — koristi se za trajno čuvanje trenutnog stanja vizuelizacije. Klikom se otvara sistemski dijalog u kom možete odabrati lokaciju i format fajla (`.png`, `.jpg` ili `.pdf`). Na ovaj način čuvate statičku sliku trenutnog koraka proračuna sa svim obojenim regijama i ispisanim formulama, što može biti korisno za pripremu prezentacija ili izveštaja.

- **Kontrola toka (Donji panel):**
  - `"Sledeći"` i `"Prethodni"` — manuelni prolaz korak-po-korak.
  - `"Auto"` — pokreće automatsku prezentaciju u realnom vremenu.
  - `"Video"` — izvoz cele operacije u `.mp4` format na vaš računar.

---

## 3. Kompajliranje i kreiranje samostalne aplikacije

Ukoliko menjate kod i želite da napravite sopstveni izvršni fajl (`.exe`) koji radi bez instaliranog Pythona, pratite sledeće korake:

### 3.1. Instalacija PyInstallera

Otvorite terminal (ili Command Prompt) i instalirajte alat za pakovanje:

```bash
pip install pyinstaller
```

### 3.2. Pravljenje aplikacije

U terminalu se pozicionirajte u glavni (korenski) folder projekta i pokrenite:

```bash
pyinstaller --noconsole --onefile --icon=ikona.ico main.py
```

Nakon završetka procesa, gotov izvršni fajl nalaziće se u novokreiranom direktorijumu `dist/`.