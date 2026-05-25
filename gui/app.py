# gui/app.py
import sys
import matplotlib

matplotlib.use('Qt5Agg')  # Govorimo matplotlibu da koristi Qt render

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QTabWidget, QPushButton, QSpinBox, QCheckBox, QLabel, QProgressBar, QComboBox
)
from PyQt5.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Importujemo module
from core.convolution import ConvolutionEngine
from core.pooling import PoolingEngine  # <-- OTKOMENTARISANO!
# from core.pattern import PatternEngine
from visualization.step_animator import StepAnimator
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CNN Visualizer")
        self.resize(1200, 700)

        # Matplotlib Figure koji će se prikazivati u GUI-u
        self.fig = Figure(figsize=(8, 4), facecolor='#1e2128')  # Boja pozadine ista kao tamna tema
        self.canvas = FigureCanvas(self.fig)

        # Glavni state (trenutni engine i animator)
        self.engine = None
        self.animator = None

        self._init_ui()

        # Pokrećemo inicijalni mod (Konvolucija)
        self.generate_convolution()

    def _init_ui(self):
        """Kreira raspored (layout) aplikacije"""
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
                        width: 196px;       /* Širina svakog taba */
                        height: 40px;       /* Opciono: visina taba da izgleda masivnije */
                        font-size: 25px;    /* Opciono: povećaš font naslova taba */
                    }
                """)

        # Tab 1: Konvolucija
        self.tab_conv = QWidget()
        self.setup_conv_controls(self.tab_conv)
        self.tabs.addTab(self.tab_conv, "Konvolucija")

        # Tab 2: Pooling
        self.tab_pool = QWidget()
        self.setup_pool_controls(self.tab_pool)  # <-- DODATO
        self.tabs.addTab(self.tab_pool, "Pooling")

        # Tab 3: Pattern
        self.tab_pattern = QWidget()
        self.tabs.addTab(self.tab_pattern, "Detekcija")

        left_layout.addWidget(self.tabs)

        # Info panel ispod kontrola (ispisuje status)
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

        # Gore: Canvas (3D Kocke)
        center_layout.addWidget(self.canvas)

        self.toolbar = NavigationToolbar(self.canvas, self)
        center_layout.addWidget(self.toolbar)

        # Dole: Kontrole za puštanje animacije
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
    # KONTROLE ZA KONVOLUCIJU
    # ---------------------------------------------------------
    def setup_conv_controls(self, parent_widget):
        main_vbox = QVBoxLayout(parent_widget)
        main_vbox.setContentsMargins(20, 20, 20, 20)
        main_vbox.addStretch()

        form_layout = QFormLayout()
        form_layout.setVerticalSpacing(20)
        form_layout.setHorizontalSpacing(15)

        self.sp_conv_size = QSpinBox()
        self.sp_conv_size.setRange(5, 10)
        self.sp_conv_size.setValue(5)

        self.sp_conv_chan = QSpinBox()
        self.sp_conv_chan.setRange(1, 3)
        self.sp_conv_chan.setValue(1)

        self.sp_conv_filt = QSpinBox()
        self.sp_conv_filt.setRange(1, 5)
        self.sp_conv_filt.setValue(3)

        self.sp_conv_stride = QSpinBox()
        self.sp_conv_stride.setRange(1, 5)
        self.sp_conv_stride.setValue(1)

        self.chk_conv_pad = QCheckBox()

        self.sp_conv_bias = QSpinBox()
        self.sp_conv_bias.setRange(0, 3)
        self.sp_conv_bias.setValue(0)

        form_layout.addRow("Dimenzija ulaza:", self.sp_conv_size)
        form_layout.addRow("Broj kanala:", self.sp_conv_chan)
        form_layout.addRow("Dimenzija filtera:", self.sp_conv_filt)
        form_layout.addRow("Korak (stride):", self.sp_conv_stride)
        form_layout.addRow("Bias:", self.sp_conv_bias)
        form_layout.addRow("Padding:", self.chk_conv_pad)

        main_vbox.addLayout(form_layout)
        main_vbox.addSpacing(20)

        btn_generate = QPushButton("Generiši")
        btn_generate.setMinimumHeight(60)
        btn_generate.setStyleSheet("""
            background-color: #528bff; 
            color: white; 
            font-size: 20px; 
            font-weight: bold; 
            border-radius: 5px;
        """)
        btn_generate.clicked.connect(self.generate_convolution)

        main_vbox.addWidget(btn_generate)
        main_vbox.addStretch()

    def generate_convolution(self):
        if self.animator:
            self.animator.stop_auto()

        size = self.sp_conv_size.value()
        channels = self.sp_conv_chan.value()
        f_size = self.sp_conv_filt.value()
        stride = self.sp_conv_stride.value()
        padding = self.chk_conv_pad.isChecked()
        bias = self.sp_conv_bias.value()

        actual_input_size = size + 2 if padding else size
        if f_size > actual_input_size:
            self.info_label.setText("Greška: Filter je veći od ulaza!")
            return

        self.engine = ConvolutionEngine(
            input_size=size, channels=channels, filter_size=f_size,
            stride=stride, padding=padding, bias=bias
        )

        self.animator = StepAnimator(
            engine=self.engine, fig=self.fig, mode="conv",
            on_step=self.update_ui_state, interval=800
        )

        self.animator.draw_current()
        self.canvas.draw()
        self.update_ui_state()

    # ---------------------------------------------------------
    # KONTROLE ZA POOLING (NOVO)
    # ---------------------------------------------------------
    def setup_pool_controls(self, parent_widget):
        main_vbox = QVBoxLayout(parent_widget)
        main_vbox.setContentsMargins(20, 20, 20, 20)
        main_vbox.addStretch()

        form_layout = QFormLayout()
        form_layout.setVerticalSpacing(20)
        form_layout.setHorizontalSpacing(15)

        self.sp_pool_size = QSpinBox()
        self.sp_pool_size.setRange(5, 10)
        self.sp_pool_size.setValue(5)

        self.sp_pool_chan = QSpinBox()
        self.sp_pool_chan.setRange(1, 3)
        self.sp_pool_chan.setValue(1)

        self.sp_pool_filt = QSpinBox()
        self.sp_pool_filt.setRange(2, 5)
        self.sp_pool_filt.setValue(2)

        self.sp_pool_stride = QSpinBox()
        self.sp_pool_stride.setRange(1, 5)
        self.sp_pool_stride.setValue(2)

        self.cb_pool_type = QComboBox()
        self.cb_pool_type.addItem("Max Pooling", "max")
        self.cb_pool_type.addItem("Average Pooling", "avg")
        self.cb_pool_type.addItem("L2 Pooling", "l2")
        self.cb_pool_type.addItem("Weighted Average", "weighted")

        form_layout.addRow("Dimenzija ulaza:", self.sp_pool_size)
        form_layout.addRow("Broj kanala:", self.sp_pool_chan)
        form_layout.addRow("Dimenzija filtera:", self.sp_pool_filt)
        form_layout.addRow("Korak (stride):", self.sp_pool_stride)
        form_layout.addRow("Tip pooling-a:", self.cb_pool_type)

        main_vbox.addLayout(form_layout)
        main_vbox.addSpacing(20)

        btn_generate = QPushButton("Generiši")
        btn_generate.setMinimumHeight(60)
        btn_generate.setStyleSheet("""
            background-color: #528bff; 
            color: white; 
            font-size: 20px; 
            font-weight: bold; 
            border-radius: 5px;
        """)
        btn_generate.clicked.connect(self.generate_pooling)

        main_vbox.addWidget(btn_generate)
        main_vbox.addStretch()

    def generate_pooling(self):
        if self.animator:
            self.animator.stop_auto()

        size = self.sp_pool_size.value()
        channels = self.sp_pool_chan.value()
        f_size = self.sp_pool_filt.value()
        stride = self.sp_pool_stride.value()
        pool_type = self.cb_pool_type.currentData()

        if f_size > size:
            self.info_label.setText("Greška: Filter je veći od ulaza!")
            return

        self.engine = PoolingEngine(
            input_size=size, channels=channels, filter_size=f_size,
            stride=stride, pool_type=pool_type
        )

        self.animator = StepAnimator(
            engine=self.engine, fig=self.fig, mode="pool",
            on_step=self.update_ui_state, interval=800
        )

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
        if not self.animator:
            return
        is_running = self.animator.toggle_auto()
        if is_running:
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
    # OSVEŽAVANJE GUI-a (Progress bar i dugmići)
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

    # ---------------------------------------------------------
    # PROMENA MODA (Tabova)
    # ---------------------------------------------------------
    def on_tab_changed(self, index):
        if self.animator:
            self.animator.stop_auto()

        if index == 0:
            self.generate_convolution()
        elif index == 1:
            self.generate_pooling()  # <-- SADA SE POKREĆE POOLING LOGIKA
        elif index == 2:
            self.info_label.setText("Detekcija obrazaca uskoro...")