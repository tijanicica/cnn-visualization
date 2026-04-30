# utils/random_gen.py
import numpy as np

def random_int_array(shape: tuple, low: int, high: int) -> np.ndarray:
    """
    Generiše NumPy niz slučajnih CIJELIH brojeva u opsegu [low, high].

    shape  — dimenzije niza, npr. (5, 5) ili (5, 5, 3)
    low    — minimalna vrijednost (uključivo)
    high   — maksimalna vrijednost (uključivo)

    np.random.randint(low, high+1) → +1 jer randint isključuje gornju granicu
    """
    return np.random.randint(low, high + 1, size=shape)