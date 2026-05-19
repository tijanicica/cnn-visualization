# core/convolution.py
import numpy as np
from utils.random_gen import random_int_array


class ConvolutionEngine:
    """
    Enkapsulira sve što je potrebno za jednu konvolucijsku operaciju:
      - čuvanje parametara
      - generisanje ulaza i filtera
      - dodavanje paddinga
      - računanje svih koraka (svaki korak = jedna pozicija filtera)
      - pristup rezultatima korak-po-korak
    """

    def __init__(
        self,
        input_size: int = 5,    # prostorna dimenzija ulaza (H = W)
        channels: int = 1,      # broj kanala ulazne mape
        filter_size: int = 3,   # prostorna dimenzija filtera (F × F)
        stride: int = 1,        # korak pomjeranja filtera
        padding: bool = False,  # da li dodajemo nule na ivice
        bias: int = 0,          # bias koji se dodaje na svaki izlazni element (0 = ne koristi se)
    ):
        self.input_size = input_size
        self.channels = channels
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.bias = bias

        # Generišemo vrijednosti odmah pri inicijalizaciji
        self._generate()

    # ------------------------------------------------------------------
    # Generisanje
    # ------------------------------------------------------------------

    def _generate(self):
        """
        Generiše slučajni ulaz i filter, primjenjuje padding ako je uključen,
        i unaprijed izračunava SVE korake konvolucije.

        Zašto unaprijed?  GUI treba "prethodni / sljedeći" — lakše je imati
        listu gotovih stanja nego računati svaki put nanovo.
        """
        # Ulazna mapa: shape (input_size, input_size, channels)
        self.input_map = random_int_array(
            (self.input_size, self.input_size, self.channels), low=0, high=3
        )

        # Filter: shape (filter_size, filter_size, channels)
        # Jedan filter — za multi-filter (zadatak C) koristimo drugi modul
        self.filter_weights = random_int_array(
            (self.filter_size, self.filter_size, self.channels), low=0, high=3
        )

        # Padding: okružimo ulaz nulama debljine 1 sa svake strane
        # np.pad prima listu (before, after) za svaku dimenziju
        # Dimenzije su (H, W, C) → paddujemo samo H i W, ne C
        if self.padding:
            self.padded_input = np.pad(
                self.input_map,
                pad_width=((1, 1), (1, 1), (0, 0)),  # (H, W, C)
                mode='constant',
                constant_values=0
            )
        else:
            self.padded_input = self.input_map

        # Izračunaj dimenziju izlaza
        padded_size = self.padded_input.shape[0]
        self.output_size = (padded_size - self.filter_size) // self.stride + 1

        # Inicijalizuj izlaznu mapu (puna nulama — popunjavamo korak po korak)
        self.output_map = np.full((self.output_size, self.output_size), np.nan, dtype=float)
        # Izračunaj sve korake unaprijed
        self.steps = self._compute_all_steps()
        self.current_step = 0  # indeks trenutnog koraka

        if self.steps:
            step0 = self.steps[0]
            self.output_map[step0["out_row"], step0["out_col"]] = step0["output_val"]

    def _compute_all_steps(self) -> list[dict]:
        """
        Prolazi kroz sve pozicije filtera i za svaku poziciju čuva:
          - out_row, out_col  → gdje u izlaznoj mapi ide rezultat
          - in_row, in_col    → gornji lijevi ugao regiona u paddovanom ulazu
          - region            → isječak ulaza koji se množi s filterom (shape = filter shape)
          - products          → element-wise množenje (region * filter_weights), po kanalu
          - conv_sum          → suma svih proizvoda (skalar, sve kanale zajedno)
          - output_val        → conv_sum + bias
        """
        steps = []

        for i in range(self.output_size):        # red izlaza
            for j in range(self.output_size):    # kolona izlaza

                # Pozicija u paddovanom ulazu (gornji lijevi ugao regiona)
                row_start = i * self.stride
                col_start = j * self.stride

                # Isječak ulaza iste veličine kao filter
                # shape: (filter_size, filter_size, channels)
                region = self.padded_input[
                    row_start : row_start + self.filter_size,
                    col_start : col_start + self.filter_size,
                    :
                ]

                # Element-wise množenje — svaki element regiona s odgovarajućim težinom
                products = region * self.filter_weights   # shape isto kao region

                # Suma svih proizvoda (po svim prostornim pozicijama i kanalima)
                conv_sum = int(np.sum(products))

                # Dodaj bias
                output_val = conv_sum + self.bias

                steps.append({
                    "out_row": i,
                    "out_col": j,
                    "in_row": row_start,
                    "in_col": col_start,
                    "region": region.copy(),        # .copy() jer numpy slices su view-ovi
                    "products": products.copy(),
                    "conv_sum": conv_sum,
                    "output_val": output_val,
                })

        return steps

    # ------------------------------------------------------------------
    # Navigacija korak-po-korak
    # ------------------------------------------------------------------

    def get_current_step(self) -> dict:
        return self.steps[self.current_step]

    def next_step(self) -> bool:
        """Pomjeri na sljedeći korak. Vraća False ako smo na kraju."""
        if self.current_step < len(self.steps) - 1:
            self.current_step += 1
            # Upiši rezultat u izlaznu mapu
            step = self.steps[self.current_step]
            self.output_map[step["out_row"], step["out_col"]] = step["output_val"]
            return True
        return False

    def prev_step(self) -> bool:
        if self.current_step > 0:
            step = self.steps[self.current_step]
            self.output_map[step["out_row"], step["out_col"]] = np.nan # Umesto 0, stavljamo NaN
            self.current_step -= 1
            return True
        return False

    def reset(self):
        self.output_map = np.full((self.output_size, self.output_size), np.nan, dtype=float)
        self.current_step = 0
        if self.steps:
            step0 = self.steps[0]
            self.output_map[step0["out_row"], step0["out_col"]] = step0["output_val"]

    def is_finished(self) -> bool:
        return self.current_step == len(self.steps) - 1

    # ------------------------------------------------------------------
    # Korisne informacije za GUI info panel
    # ------------------------------------------------------------------

    def get_info(self) -> dict:
        """Vraća string-ove za prikaz u info panelu."""
        p = self.padded_input.shape[0]
        return {
            "input_shape": f"{self.input_size}×{self.input_size}×{self.channels}",
            "padded_shape": f"{p}×{p}×{self.channels}" if self.padding else "N/A",
            "filter_shape": f"{self.filter_size}×{self.filter_size}×{self.channels}",
            "output_shape": f"{self.output_size}×{self.output_size}×1",
            "stride": str(self.stride),
            "padding": "Da" if self.padding else "Ne",
            "bias": str(self.bias),
            "total_steps": len(self.steps),
        }