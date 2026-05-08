# core/pattern.py
import numpy as np
from utils.random_gen import random_int_array


class PatternEngine:
    """
    Generiše 12×12 ulaznu mapu s ugrađenim obrascima i
    izvodi konvoluciju s 3 filtera 3×3, korak po korak.

    Specijalni regioni u mapi:
      - Za svaki filter: jedan region identičan filteru     (pozitivno poklapanje)
      - Za svaki filter: jedan region = -1 × filter         (negativno poklapanje)
      - Ostatak: slučajne vrijednosti iz [-3, 3]

    Step dict koji get_current_step() vraća:
    {
        "filter_idx":      int,          # koji filter (0/1/2)
        "filter_row":      int,          # gornji lijevi red u ulazu
        "filter_col":      int,          # gornji lijevi col u ulazu
        "out_row":         int,
        "out_col":         int,
        "region":          np.ndarray,   # (3,3) isječak ulaza
        "kernel":          np.ndarray,   # (3,3) trenutni filter
        "products":        np.ndarray,   # (3,3) element-wise množenje
        "output_value":    float,        # suma proizvoda
        "match_type":      str,          # "positive" | "negative" | "neutral"
        "step_idx":        int,
        "total_steps":     int,
        "input_map":       np.ndarray,   # cijela 12×12 mapa (referenca)
        "all_output_maps": list,         # lista 3 izlazne mape (10×10 svaka)
    }
    """

    MAP_SIZE    = 12
    FILTER_SIZE = 3
    NUM_FILTERS = 3
    STRIDE      = 1

    def __init__(self):
        self._generate()

    # ------------------------------------------------------------------
    # Generisanje
    # ------------------------------------------------------------------

    def _generate(self):
        # 3 filtera, vrijednosti iz [-3, 3]
        self.filters = random_int_array(
            (self.NUM_FILTERS, self.FILTER_SIZE, self.FILTER_SIZE),
            low=-3, high=3
        )

        # Bazna mapa — popuni slučajno
        self.input_map = random_int_array(
            (self.MAP_SIZE, self.MAP_SIZE),
            low=-3, high=3
        )

        # Smjesti specijalne regione i zapamti pozicije
        self.special_regions = self._place_special_regions()

        # Izlazna dimenzija: (12 - 3) / 1 + 1 = 10
        self.output_size = (
            (self.MAP_SIZE - self.FILTER_SIZE) // self.STRIDE + 1
        )

        # Izlazne mape — moraju biti kreirane PRIJE _compute_steps_for_filter
        # jer steps drže referencu na output_maps
        # NaN = još nije izračunato (renderer ih prikazuje kao "čeka")
        self.output_maps = [
            np.full((self.output_size, self.output_size), np.nan)
            for _ in range(self.NUM_FILTERS)
        ]

        # Unaprijed izračunaj korake za svaki filter
        self.steps_per_filter = [
            self._compute_steps_for_filter(f)
            for f in range(self.NUM_FILTERS)
        ]

        self.current_filter = 0
        self.current_step   = 0

    def _place_special_regions(self) -> list[dict]:
        """
        Smješta 6 specijalnih regiona (3 pozitivna + 3 negativna) u mapu.
        Koristi strategiju 4×4 blokova — 9 blokova ukupno, biramo 6.
        Vraća listu dict-ova s pozicijama (koriste se u rendereru).
        """
        block_positions = [
            (br * 4, bc * 4)
            for br in range(3)
            for bc in range(3)
        ]

        chosen_idx = np.random.choice(len(block_positions), size=6, replace=False)
        chosen     = [block_positions[i] for i in chosen_idx]

        special_regions = []

        for fi in range(self.NUM_FILTERS):
            # Pozitivno poklapanje
            pr, pc = chosen[fi * 2]
            self.input_map[
                pr : pr + self.FILTER_SIZE,
                pc : pc + self.FILTER_SIZE
            ] = self.filters[fi]

            special_regions.append({
                "filter_idx": fi,
                "type": "positive",
                "row":  pr,
                "col":  pc,
            })

            # Negativno poklapanje
            nr, nc = chosen[fi * 2 + 1]
            self.input_map[
                nr : nr + self.FILTER_SIZE,
                nc : nc + self.FILTER_SIZE
            ] = -self.filters[fi]

            special_regions.append({
                "filter_idx": fi,
                "type": "negative",
                "row":  nr,
                "col":  nc,
            })

        return special_regions

    def _compute_steps_for_filter(self, filter_idx: int) -> list[dict]:
        """
        Prolazi kroz sve pozicije filtera i za svaku gradi step dict.
        Uključuje match_type ("positive"/"negative"/"neutral") koji
        renderer koristi za bojanje i naslov.
        """
        f = self.filters[filter_idx]

        # Brzi lookup: (out_row, out_col) → match_type za ovaj filter
        match_lookup = {}
        for sr in self.special_regions:
            if sr["filter_idx"] == filter_idx:
                # Specijalni region počinje na (sr["row"], sr["col"]) u ulazu
                # što odgovara out poziciji (sr["row"]//stride, sr["col"]//stride)
                out_r = sr["row"] // self.STRIDE
                out_c = sr["col"] // self.STRIDE
                match_lookup[(out_r, out_c)] = sr["type"]

        steps = []
        total = self.output_size * self.output_size

        for i in range(self.output_size):
            for j in range(self.output_size):
                rs = i * self.STRIDE
                cs = j * self.STRIDE

                region   = self.input_map[rs:rs+self.FILTER_SIZE,
                                          cs:cs+self.FILTER_SIZE].copy()
                products = (region * f).astype(float)
                conv_sum = float(np.sum(products))

                steps.append({
                    "filter_idx":      filter_idx,
                    "filter_row":      rs,
                    "filter_col":      cs,
                    "out_row":         i,
                    "out_col":         j,
                    "region":          region,
                    "kernel":          f.copy(),
                    "products":        products,
                    "output_value":    conv_sum,
                    "match_type":      match_lookup.get((i, j), "neutral"),
                    "step_idx":        len(steps),
                    "total_steps":     total,
                    # Reference — renderer ih čita direktno
                    "input_map":       self.input_map,
                    "all_output_maps": self.output_maps,
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
            ] = step["output_value"]
            return True
        return False

    def prev_step(self) -> bool:
        if self.current_step > 0:
            step = self.steps_per_filter[self.current_filter][self.current_step]
            self.output_maps[self.current_filter][
                step["out_row"], step["out_col"]
            ] = np.nan
            self.current_step -= 1
            return True
        return False

    def set_filter(self, filter_idx: int):
        """Prebaci vizualizaciju na drugi filter i resetuj korake."""
        if 0 <= filter_idx < self.NUM_FILTERS:
            self.current_filter = filter_idx
            self.current_step   = 0
            self.output_maps[filter_idx] = np.full(
                (self.output_size, self.output_size), np.nan
            )

    def reset(self):
        self.current_step = 0
        self.output_maps = [
            np.full((self.output_size, self.output_size), np.nan)
            for _ in range(self.NUM_FILTERS)
        ]

    def is_finished(self) -> bool:
        steps = self.steps_per_filter[self.current_filter]
        return self.current_step == len(steps) - 1

    # ------------------------------------------------------------------
    # Helpers za renderer — zamjena za stare konstante
    # ------------------------------------------------------------------

    def get_positive_position(self, filter_idx: int) -> tuple[int, int]:
        """Vraća (row, col) pozitivnog poklapanja za dati filter."""
        for sr in self.special_regions:
            if sr["filter_idx"] == filter_idx and sr["type"] == "positive":
                return sr["row"], sr["col"]
        return (0, 0)

    def get_negative_position(self, filter_idx: int) -> tuple[int, int]:
        """Vraća (row, col) negativnog poklapanja za dati filter."""
        for sr in self.special_regions:
            if sr["filter_idx"] == filter_idx and sr["type"] == "negative":
                return sr["row"], sr["col"]
        return (0, 0)

    def get_info(self) -> dict:
        return {
            "input_shape":   f"{self.MAP_SIZE}×{self.MAP_SIZE}×1",
            "filter_shape":  f"{self.FILTER_SIZE}×{self.FILTER_SIZE}",
            "num_filters":   str(self.NUM_FILTERS),
            "output_shape":  f"{self.output_size}×{self.output_size}",
            "current_filter": str(self.current_filter + 1),
            "total_steps":   str(len(self.steps_per_filter[self.current_filter])),
        }
