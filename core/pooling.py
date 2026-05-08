# core/pooling.py
import numpy as np
from utils.random_gen import random_int_array


# Fiksne težine za weighted average pooling.
# Čuvamo ih kao konstantu modula — korisnik ih ne mijenja,
# ali vizualizacija ih prikazuje da bi bilo jasno šta se množi.
# Shape: (5, 5) — maksimalna veličina filtera prema zadatku.
# Kad je filter manji, uzimamo gornji lijevi ugao.
_FIXED_WEIGHTS = np.array([
    [1, 2, 1, 2, 1],
    [2, 4, 2, 4, 2],
    [1, 2, 1, 2, 1],
    [2, 4, 2, 4, 2],
    [1, 2, 1, 2, 1],
], dtype=float)


class PoolingEngine:
    """
    Enkapsulira logiku pooling sloja.
    Analogno ConvolutionEngine — čuva parametre, generiše ulaz,
    unaprijed računa sve korake, podržava navigaciju.
    """

    # Podržani tipovi — string koji GUI prikazuje : interni ključ
    POOL_TYPES = ["max", "avg", "l2", "weighted"]

    def __init__(
        self,
        input_size: int = 5,
        channels: int = 1,
        filter_size: int = 2,
        stride: int = 1,
        pool_type: str = "max",
    ):
        self.input_size = input_size
        self.channels = channels
        self.filter_size = filter_size
        self.stride = stride
        self.pool_type = pool_type  # "max" | "avg" | "l2" | "weighted"

        self._generate()

    # ------------------------------------------------------------------
    # Generisanje
    # ------------------------------------------------------------------

    def _generate(self):
        """
        Generiše ulaznu mapu i izračunava sve korake unaprijed.
        Ulaz: shape (input_size, input_size, channels), vrijednosti 0–3.
        """
        self.input_map = random_int_array(
            (self.input_size, self.input_size, self.channels),
            low=0, high=3
        )

        # Izračunaj dimenziju izlaza
        self.output_size = (self.input_size - self.filter_size) // self.stride + 1

        # Izlazna mapa: čuvamo float jer avg/l2/weighted daju decimale
        # Shape: (output_size, output_size, channels)
        self.output_map = np.full(
            (self.output_size, self.output_size, self.channels),
            np.nan
        )

        # Isječak fiksnih težina za trenutnu veličinu filtera
        # Ako je filter 3×3, uzimamo _FIXED_WEIGHTS[:3, :3]
        self.weights = _FIXED_WEIGHTS[:self.filter_size, :self.filter_size]

        self.steps = self._compute_all_steps()
        self.current_step = 0

    def _pool_region(self, region: np.ndarray) -> float:
        """
        Prima region oblika (filter_size, filter_size) — jedan kanal.
        Vraća jednu vrijednost prema odabranom tipu poolinga.

        Razdvajamo ovo u posebnu metodu jer je ista logika potrebna
        i u _compute_all_steps i potencijalno u testovima.
        """
        if self.pool_type == "max":
            return float(np.max(region))

        elif self.pool_type == "avg":
            return float(np.mean(region))

        elif self.pool_type == "l2":
            # sqrt(sum(x²)) — L2 norma vektora svih elemenata regiona
            return float(np.sqrt(np.sum(region ** 2)))

        elif self.pool_type == "weighted":
            # sum(w * x) / sum(w)
            # self.weights i region su isti shape
            w = self.weights
            return float(np.sum(w * region) / np.sum(w))

        else:
            raise ValueError(f"Nepoznat tip poolinga: {self.pool_type}")

    def _compute_all_steps(self) -> list[dict]:
        """
        Za svaku poziciju filtera, za svaki kanal, čuva:
          - out_row, out_col       → pozicija u izlaznoj mapi
          - in_row, in_col         → gornji lijevi ugao u ulaznoj mapi
          - regions                → lista regiona po kanalima (svaki F×F)
          - output_vals            → lista izlaznih vrijednosti po kanalima
          - weights_used           → težine (samo za weighted avg, inače None)

        Zašto lista po kanalima?
        Pooling radi nezavisno po kanalima — svaki kanal daje svoju
        izlaznu vrijednost na istoj poziciji.
        """
        steps = []

        for i in range(self.output_size):
            for j in range(self.output_size):

                row_start = i * self.stride
                col_start = j * self.stride

                regions = []
                output_vals = []

                for c in range(self.channels):
                    # Isječak ulaza za kanal c
                    region = self.input_map[
                        row_start : row_start + self.filter_size,
                        col_start : col_start + self.filter_size,
                        c
                    ].copy()  # (filter_size, filter_size)

                    val = self._pool_region(region)
                    regions.append(region)
                    output_vals.append(val)

                steps.append({
                    "out_row": i,
                    "out_col": j,
                    "in_row": row_start,
                    "in_col": col_start,
                    "regions": regions,           # lista dužine channels
                    "output_vals": output_vals,   # lista dužine channels
                    "weights_used": (
                        self.weights.copy()
                        if self.pool_type == "weighted"
                        else None
                    ),
                    "pool_type": self.pool_type,
                })

        return steps

    # ------------------------------------------------------------------
    # Navigacija — isti interfejs kao ConvolutionEngine
    # ------------------------------------------------------------------

    def get_current_step(self) -> dict:
        return self.steps[self.current_step]

    def next_step(self) -> bool:
        if self.current_step < len(self.steps) - 1:
            self.current_step += 1
            step = self.steps[self.current_step]
            # Upiši rezultate svih kanala u izlaznu mapu
            for c, val in enumerate(step["output_vals"]):
                self.output_map[step["out_row"], step["out_col"], c] = val
            return True
        return False

    def prev_step(self) -> bool:
        if self.current_step > 0:
            step = self.steps[self.current_step]
            # Poništi
            for c in range(self.channels):
                self.output_map[step["out_row"], step["out_col"], c] = np.nan
            self.current_step -= 1
            return True
        return False

    def reset(self):
        self.output_map = np.full(
            (self.output_size, self.output_size, self.channels),
            np.nan
        )
        self.current_step = 0

    def is_finished(self) -> bool:
        return self.current_step == len(self.steps) - 1

    # ------------------------------------------------------------------
    # Info panel
    # ------------------------------------------------------------------

    def get_info(self) -> dict:
        return {
            "input_shape": f"{self.input_size}×{self.input_size}×{self.channels}",
            "filter_shape": f"{self.filter_size}×{self.filter_size}",
            "output_shape": (
                f"{self.output_size}×{self.output_size}×{self.channels}"
            ),
            "stride": str(self.stride),
            "pool_type": self.pool_type.upper(),
            "total_steps": len(self.steps),
        }