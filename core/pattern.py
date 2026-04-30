# core/pattern.py
import numpy as np
from utils.random_gen import random_int_array


class PatternEngine:
    """
    Generiše 12×12 ulaznu mapu s ugrađenim obrascima i
    izvodi konvoluciju s 3 filtera 3×3, korak po korak.

    Specijalni regioni u mapi:
      - Za svaki filter: jedan region identičan filteru (pozitivno poklapanje)
      - Za svaki filter: jedan region = -1 × filter (negativno poklapanje)
      - Ostatak: slučajne vrijednosti iz [-3, 3]

    Izlaz: za svaki filter posebna 10×10 mapa (stride=1, bez paddinga).
    """

    MAP_SIZE = 12       # fiksno prema zadatku
    FILTER_SIZE = 3     # fiksno prema zadatku
    NUM_FILTERS = 3     # fiksno prema zadatku
    STRIDE = 1          # fiksno

    def __init__(self):
        self._generate()

    # ------------------------------------------------------------------
    # Generisanje
    # ------------------------------------------------------------------

    def _generate(self):
        """
        Korak 1: Generiši 3 slučajna filtera.
        Korak 2: Generiši slučajnu baznu mapu iz [-3, 3].
        Korak 3: Smjesti specijalne regione u mapu.
        Korak 4: Izračunaj sve korake konvolucije za sva 3 filtera.
        """

        # Korak 1: Filteri — vrijednosti iz [-3, 3] prema zadatku
        # Shape: (NUM_FILTERS, FILTER_SIZE, FILTER_SIZE)
        self.filters = random_int_array(
            (self.NUM_FILTERS, self.FILTER_SIZE, self.FILTER_SIZE),
            low=-3, high=3
        )

        # Korak 2: Bazna mapa — popuni sve slučajno
        self.input_map = random_int_array(
            (self.MAP_SIZE, self.MAP_SIZE),
            low=-3, high=3
        )

        # Korak 3: Smjesti specijalne regione
        self.special_regions = self._place_special_regions()

        # Korak 4: Unaprijed izračunaj korake za svaki filter
        # steps_per_filter[f] = lista koraka za filter f
        self.output_size = (
            (self.MAP_SIZE - self.FILTER_SIZE) // self.STRIDE + 1
        )  # = 10

        self.steps_per_filter = [
            self._compute_steps_for_filter(f)
            for f in range(self.NUM_FILTERS)
        ]

        # Izlazne mape — jedna po filteru, pune se korak po korak
        self.output_maps = [
            np.zeros((self.output_size, self.output_size), dtype=float)
            for _ in range(self.NUM_FILTERS)
        ]

        # Navigacija
        self.current_filter = 0   # koji filter trenutno vizualizujemo
        self.current_step = 0

    def _place_special_regions(self) -> list[dict]:
        """
        Smješta 6 specijalnih regiona (3 pozitivna + 3 negativna) u mapu.

        Strategija blokova:
          Mapa 12×12 → 9 blokova 4×4 (indeksi blokova: 0–8)
          U bloku (br, bc) region počinje na (br*4, bc*4)
          Biramo 6 različitih blokova nasumično.

        Vraća listu dict-ova s informacijama o svakom specijalnom regionu
        (za vizualizaciju — da možemo istaknuti ta mjesta).
        """
        # Sve moguće pozicije blokova
        block_positions = [
            (br * 4, bc * 4)
            for br in range(3)
            for bc in range(3)
        ]  # 9 blokova

        # Odaberi 6 različitih blokova bez ponavljanja
        chosen_indices = np.random.choice(len(block_positions), size=6, replace=False)
        chosen_positions = [block_positions[i] for i in chosen_indices]

        special_regions = []

        for filter_idx in range(self.NUM_FILTERS):
            # Pozitivno poklapanje — region = filter
            pos_row, pos_col = chosen_positions[filter_idx * 2]
            self.input_map[
                pos_row : pos_row + self.FILTER_SIZE,
                pos_col : pos_col + self.FILTER_SIZE
            ] = self.filters[filter_idx]

            special_regions.append({
                "filter_idx": filter_idx,
                "type": "positive",        # identičan filteru
                "row": pos_row,
                "col": pos_col,
            })

            # Negativno poklapanje — region = -1 * filter
            neg_row, neg_col = chosen_positions[filter_idx * 2 + 1]
            self.input_map[
                neg_row : neg_row + self.FILTER_SIZE,
                neg_col : neg_col + self.FILTER_SIZE
            ] = -self.filters[filter_idx]

            special_regions.append({
                "filter_idx": filter_idx,
                "type": "negative",        # negativan filter
                "row": neg_row,
                "col": neg_col,
            })

        return special_regions

    def _compute_steps_for_filter(self, filter_idx: int) -> list[dict]:
        """
        Ista logika kao ConvolutionEngine._compute_all_steps,
        ali za jedan specifičan filter i jednokanalni ulaz.

        Dodatno bilježimo je li trenutna pozicija specijalni region
        (za vizualizaciju — isticanje pozitivnih/negativnih poklapanja).
        """
        f = self.filters[filter_idx]  # shape (3, 3)
        steps = []

        # Skup specijalnih pozicija za ovaj filter (za brzo lookup)
        special_map = {}
        for sr in self.special_regions:
            if sr["filter_idx"] == filter_idx:
                # Ključ: (out_row, out_col) u izlaznoj mapi
                # Specijalni region počinje na (sr["row"], sr["col"]) u ulazu
                # To odgovara out poziciji (sr["row"]/stride, sr["col"]/stride)
                out_r = sr["row"] // self.STRIDE
                out_c = sr["col"] // self.STRIDE
                special_map[(out_r, out_c)] = sr["type"]

        for i in range(self.output_size):
            for j in range(self.output_size):

                row_start = i * self.STRIDE
                col_start = j * self.STRIDE

                region = self.input_map[
                    row_start : row_start + self.FILTER_SIZE,
                    col_start : col_start + self.FILTER_SIZE
                ].copy()

                products = region * f
                conv_sum = float(np.sum(products))

                steps.append({
                    "out_row": i,
                    "out_col": j,
                    "in_row": row_start,
                    "in_col": col_start,
                    "region": region,
                    "products": products,
                    "conv_sum": conv_sum,
                    # None ako nije specijalni, "positive"/"negative" ako jeste
                    "special": special_map.get((i, j), None),
                })

        return steps

    # ------------------------------------------------------------------
    # Navigacija
    # ------------------------------------------------------------------

    def get_current_step(self) -> dict:
        return self.steps_per_filter[self.current_filter][self.current_step]

    def next_step(self) -> bool:
        steps = self.steps_per_filter[self.current_filter]
        if self.current_step < len(steps) - 1:
            self.current_step += 1
            step = steps[self.current_step]
            self.output_maps[self.current_filter][
                step["out_row"], step["out_col"]
            ] = step["conv_sum"]
            return True
        return False

    def prev_step(self) -> bool:
        if self.current_step > 0:
            step = self.steps_per_filter[self.current_filter][self.current_step]
            self.output_maps[self.current_filter][
                step["out_row"], step["out_col"]
            ] = 0.0
            self.current_step -= 1
            return True
        return False

    def set_filter(self, filter_idx: int):
        """
        Prebaci vizualizaciju na drugi filter.
        Reset koraka — svaki filter počinje od nule.
        """
        if 0 <= filter_idx < self.NUM_FILTERS:
            self.current_filter = filter_idx
            self.current_step = 0
            # Resetuj izlaznu mapu za taj filter
            self.output_maps[filter_idx] = np.zeros(
                (self.output_size, self.output_size), dtype=float
            )

    def reset(self):
        self.current_step = 0
        self.output_maps = [
            np.zeros((self.output_size, self.output_size), dtype=float)
            for _ in range(self.NUM_FILTERS)
        ]

    def is_finished(self) -> bool:
        steps = self.steps_per_filter[self.current_filter]
        return self.current_step == len(steps) - 1

    # ------------------------------------------------------------------
    # Analiza — za info panel
    # ------------------------------------------------------------------

    def get_max_response(self, filter_idx: int) -> dict:
        """
        Vraća poziciju i vrijednost maksimalnog outputa za dati filter.
        Ovo je "dokaz" da je filter najjače reagovao na pozitivno poklapanje.

        Može se pozvati tek kad je izlazna mapa popunjena (is_finished).
        """
        out_map = self.output_maps[filter_idx]
        flat_idx = np.argmax(out_map)
        row, col = np.unravel_index(flat_idx, out_map.shape)
        return {
            "row": int(row),
            "col": int(col),
            "value": float(out_map[row, col]),
        }

    def get_min_response(self, filter_idx: int) -> dict:
        """Analogno — najmanji output odgovara negativnom poklapanju."""
        out_map = self.output_maps[filter_idx]
        flat_idx = np.argmin(out_map)
        row, col = np.unravel_index(flat_idx, out_map.shape)
        return {
            "row": int(row),
            "col": int(col),
            "value": float(out_map[row, col]),
        }

    def get_info(self) -> dict:
        return {
            "input_shape": f"{self.MAP_SIZE}×{self.MAP_SIZE}×1",
            "filter_shape": f"{self.FILTER_SIZE}×{self.FILTER_SIZE}",
            "num_filters": str(self.NUM_FILTERS),
            "output_shape": f"{self.output_size}×{self.output_size}",
            "current_filter": str(self.current_filter + 1),
            "total_steps": str(len(self.steps_per_filter[self.current_filter])),
        }