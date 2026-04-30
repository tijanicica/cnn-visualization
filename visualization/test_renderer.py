"""
Brzi test renderera — pokreni direktno da vidiš izgled.
Nema potrebe za core modulima, generišemo dummy podatke.

Pokretanje:
    cd cnn_visualizer
    python test_renderer.py
"""

import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PyQt5.QtCore import Qt

# Dodaj parent folder u path da se mogu importovati moduli
sys.path.insert(0, ".")
from visualization.renderer import CNNRenderWidget


class FakeConvEngine:
    """Simulira ConvolutionEngine bez ikakve logike — samo dummy podaci."""
    def __init__(self):
        self.input_size   = 5
        self.filter_size  = 3
        self.output_size  = 3
        self.channels     = 1
        self.padding      = False
        self.bias         = 1
        self.current_step = 3   # simuliramo korak 3

        # Dummy ulaz i filter
        self.padded_input   = np.array([
            [2, 0, 1, 3, 1],
            [0, 3, 2, 1, 0],
            [1, 1, 0, 2, 3],
            [3, 2, 1, 0, 1],
            [0, 1, 3, 2, 2],
        ], dtype=int)[:, :, np.newaxis]   # dodaj channel dimenziju

        self.filter_weights = np.array([
            [1, 0, 1],
            [0, 1, 0],
            [1, 0, 1],
        ], dtype=int)[:, :, np.newaxis]

        # Izlaz — djelimično popunjen do koraka 3
        self.output_map = np.array([
            [8, 9, 7],
            [8, 0, 0],
            [0, 0, 0],
        ], dtype=int)

        self.steps = list(range(9))   # 9 koraka ukupno


# Generiši fake step dict za korak (1, 0)
fake_step = {
    "out_row": 1,
    "out_col": 0,
    "in_row":  1,
    "in_col":  0,
    "region":  np.array([[0, 3, 2], [1, 1, 0], [3, 2, 1]]),
    "products": np.array([[0, 0, 2], [0, 1, 0], [3, 0, 1]]),
    "conv_sum": 7,
    "output_val": 8,
}


class TestWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CNN Renderer Test")
        self.resize(1100, 600)
        self.setStyleSheet("background: #12121A;")

        # Centralni widget
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)

        # Render widget
        self.renderer = CNNRenderWidget()
        layout.addWidget(self.renderer)

        # Postavi dummy state
        engine = FakeConvEngine()
        self.renderer.set_convolution_state(engine, fake_step)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = TestWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()