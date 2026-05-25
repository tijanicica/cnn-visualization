# gui/app.py
import sys
import matplotlib

matplotlib.use('Qt5Agg')

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QPushButton, QLabel, QProgressBar
)
from PyQt5.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

# Importujemo vaše core module
from core.convolution import ConvolutionEngine
from core.pooling import PoolingEngine
from visualization.step_animator import StepAnimator

# IMPORT NOVIH KONTROLA
from gui.controls import ConvControlsTab, PoolControlsTab


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CNN Visualizer")
        self.resize(1200, 700)

        self.fig = Figure(figsize=(8, 4), facecolor='#1e2128')
        self.canvas = FigureCanvas(self.fig)

        self.engine = None
        self.animator = None

        self._init_ui()
        self.generate_convolution()

    def _init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # --- LEVI PANEL: Tabovi i Kontrole ---
        left_panel = QWidget()
        left_panel.setFixedWidth(600)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.tabs = QTabWidget()
        self.tabs.tabBar().setExpanding(True)
        self.tabs.setUsesScrollButtons(False)
        self.tabs.setStyleSheet("""
            QTabBar::tab {
                width: 196px;       
                height: 40px;       
                font-size: 25px;    
            }
        """)

        # Tab 1: Konvolucija (Povezivanje sa klasom iz controls.py)
        self.tab_conv = ConvControlsTab()
        self.tab_conv.btn_generate.clicked.connect(self.generate_convolution)
        self.tabs.addTab(self.tab_conv, "Konvolucija")

        # Tab 2: Pooling (Povezivanje sa klasom iz controls.py)
        self.tab_pool = PoolControlsTab()
        self.tab_pool.btn_generate.clicked.connect(self.generate_pooling)
        self.tabs.addTab(self.tab_pool, "Pooling")

        # Tab 3: Pattern (Ostaće prazan za sada)
        self.tab_pattern = QWidget()
        self.tabs.addTab(self.tab_pattern, "Detekcija")

        left_layout.addWidget(self.tabs)

        self.info_label = QLabel("Info panel:\nČekam podatke...")
        self.info_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.info_label.setStyleSheet("""
            color: #98c379; 
            font-size: 25px; 
            padding: 15px; 
            border: 2px solid #3e4451; 
            border-radius: 6px;
            background-color: #21252b;
        """)
        left_layout.addWidget(self.info_label)

        main_layout.addWidget(left_panel)

        # --- SREDNJI PANEL: Vizualizacija i Playback ---
        center_panel = QWidget()
        center_layout = QVBoxLayout(center_panel)

        center_layout.addWidget(self.canvas)
        self.toolbar = NavigationToolbar(self.canvas, self)
        center_layout.addWidget(self.toolbar)

        playback_layout = QHBoxLayout()

        self.btn_prev = QPushButton("◄ Prethodni")
        self.btn_prev.setStyleSheet("font-size: 20px")
        self.btn_prev.clicked.connect(self.on_prev)

        self.btn_auto = QPushButton("Auto ▶")
        self.btn_auto.setStyleSheet("background-color: #235a39; color: white; font-weight: bold; font-size: 20px")
        self.btn_auto.clicked.connect(self.on_auto)

        self.btn_next = QPushButton("Sledeći ►")
        self.btn_next.setStyleSheet("font-size: 20px")
        self.btn_next.clicked.connect(self.on_next)

        self.btn_reset = QPushButton("↻ Reset")
        self.btn_reset.setStyleSheet("font-size: 20px")
        self.btn_reset.clicked.connect(self.on_reset)

        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)

        playback_layout.addWidget(self.btn_prev)
        playback_layout.addWidget(self.btn_auto)
        playback_layout.addWidget(self.btn_next)
        playback_layout.addWidget(self.btn_reset)

        center_layout.addLayout(playback_layout)
        center_layout.addWidget(self.progress_bar)

        main_layout.addWidget(center_panel)
        self.tabs.currentChanged.connect(self.on_tab_changed)

    # ---------------------------------------------------------
    # GENERISANJE LOGIKE
    # ---------------------------------------------------------
    def generate_convolution(self):
        if self.animator:
            self.animator.stop_auto()

        # Učitavanje parametara iz controls.py
        p = self.tab_conv.get_params()

        actual_input_size = p["size"] + 2 if p["padding"] else p["size"]
        if p["f_size"] > actual_input_size:
            self.info_label.setText("Greška: Filter je veći od ulaza!")
            return

        self.engine = ConvolutionEngine(
            input_size=p["size"], channels=p["channels"], filter_size=p["f_size"],
            stride=p["stride"], padding=p["padding"], bias=p["bias"]
        )

        self.animator = StepAnimator(self.engine, self.fig, "conv", self.update_ui_state, 800)
        self.animator.draw_current()
        self.canvas.draw()
        self.update_ui_state()

    def generate_pooling(self):
        if self.animator:
            self.animator.stop_auto()

        # Učitavanje parametara iz controls.py
        p = self.tab_pool.get_params()

        if p["f_size"] > p["size"]:
            self.info_label.setText("Greška: Filter je veći od ulaza!")
            return

        self.engine = PoolingEngine(
            input_size=p["size"], channels=p["channels"], filter_size=p["f_size"],
            stride=p["stride"], pool_type=p["pool_type"]
        )

        self.animator = StepAnimator(self.engine, self.fig, "pool", self.update_ui_state, 800)
        self.animator.draw_current()
        self.canvas.draw()
        self.update_ui_state()

    # ---------------------------------------------------------
    # PLAYBACK FUNKCIJE
    # ---------------------------------------------------------
    def on_prev(self):
        if self.animator:
            self.animator.prev()
            self.canvas.draw()

    def on_next(self):
        if self.animator:
            self.animator.next()
            self.canvas.draw()

    def on_auto(self):
        if not self.animator: return
        if self.animator.toggle_auto():
            self.btn_auto.setText("Pauza ⏸")
            self.btn_auto.setStyleSheet("background-color: #8c2626; color: white; font-weight: bold; font-size: 20px")
        else:
            self.btn_auto.setText("Auto ▶")
            self.btn_auto.setStyleSheet("background-color: #235a39; color: white; font-weight: bold; font-size: 20px")

    def on_reset(self):
        if self.animator:
            self.animator.reset()
            self.canvas.draw()
            self.update_ui_state()

    # ---------------------------------------------------------
    # OSVEŽAVANJE GUI-a
    # ---------------------------------------------------------
    def update_ui_state(self):
        curr, total = self.animator.get_progress()
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(curr)
        self.progress_bar.setFormat(f"Korak {curr} / {total}")

        self.btn_prev.setEnabled(not self.animator.is_at_start())
        self.btn_next.setEnabled(not self.animator.is_at_end())

        if self.animator.is_at_end():
            self.btn_auto.setText("Auto ▶")
            self.btn_auto.setStyleSheet("background-color: #235a39; color: white; font-weight: bold; font-size: 20px")

        if hasattr(self.engine, "get_info"):
            info = self.engine.get_info()
            info_text = "Trenutni parametri:\n\n"
            for k, v in info.items():
                info_text += f"• {k}: {v}\n"
            self.info_label.setText(info_text)

        self.canvas.draw()

    def on_tab_changed(self, index):
        if self.animator:
            self.animator.stop_auto()

        if index == 0:
            self.generate_convolution()
        elif index == 1:
            self.generate_pooling()
        elif index == 2:
            self.info_label.setText("Detekcija obrazaca uskoro...")