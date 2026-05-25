# gui/controls.py
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout,
    QSpinBox, QCheckBox, QPushButton, QComboBox
)
from PyQt5.QtWidgets import QListView


class ConvControlsTab(QWidget):
    """GUI kontrole za tab Konvolucija"""

    def __init__(self):
        super().__init__()
        self._init_ui()

    def _init_ui(self):
        main_vbox = QVBoxLayout(self)
        main_vbox.setContentsMargins(20, 20, 20, 20)
        main_vbox.addStretch()

        form_layout = QFormLayout()
        form_layout.setVerticalSpacing(20)
        form_layout.setHorizontalSpacing(15)

        self.sp_size = QSpinBox()
        self.sp_size.setRange(5, 10)
        self.sp_size.setValue(5)

        self.sp_chan = QSpinBox()
        self.sp_chan.setRange(1, 3)
        self.sp_chan.setValue(1)

        self.sp_filt = QSpinBox()
        self.sp_filt.setRange(1, 5)
        self.sp_filt.setValue(3)

        self.sp_stride = QSpinBox()
        self.sp_stride.setRange(1, 5)
        self.sp_stride.setValue(1)

        self.chk_pad = QCheckBox()

        self.sp_bias = QSpinBox()
        self.sp_bias.setRange(0, 3)
        self.sp_bias.setValue(0)

        form_layout.addRow("Dimenzija ulaza:", self.sp_size)
        form_layout.addRow("Broj kanala:", self.sp_chan)
        form_layout.addRow("Dimenzija filtera:", self.sp_filt)
        form_layout.addRow("Korak (stride):", self.sp_stride)
        form_layout.addRow("Bias:", self.sp_bias)
        form_layout.addRow("Padding:", self.chk_pad)

        main_vbox.addLayout(form_layout)
        main_vbox.addSpacing(20)

        self.btn_generate = QPushButton("Generiši")
        self.btn_generate.setMinimumHeight(60)
        self.btn_generate.setStyleSheet("""
            background-color: #528bff; 
            color: white; 
            font-size: 20px; 
            font-weight: bold; 
            border-radius: 5px;
        """)

        main_vbox.addWidget(self.btn_generate)
        main_vbox.addStretch()

    def get_params(self):
        """Vraća rečnik sa svim izabranim parametrima"""
        return {
            "size": self.sp_size.value(),
            "channels": self.sp_chan.value(),
            "f_size": self.sp_filt.value(),
            "stride": self.sp_stride.value(),
            "padding": self.chk_pad.isChecked(),
            "bias": self.sp_bias.value()
        }


class PoolControlsTab(QWidget):
    """GUI kontrole za tab Pooling"""

    def __init__(self):
        super().__init__()
        self._init_ui()

    def _init_ui(self):
        main_vbox = QVBoxLayout(self)
        main_vbox.setContentsMargins(20, 20, 20, 20)
        main_vbox.addStretch()

        form_layout = QFormLayout()
        form_layout.setVerticalSpacing(20)
        form_layout.setHorizontalSpacing(15)

        self.sp_size = QSpinBox()
        self.sp_size.setRange(5, 10)
        self.sp_size.setValue(5)

        self.sp_chan = QSpinBox()
        self.sp_chan.setRange(1, 3)
        self.sp_chan.setValue(1)

        self.sp_filt = QSpinBox()
        self.sp_filt.setRange(2, 5)
        self.sp_filt.setValue(2)

        self.sp_stride = QSpinBox()
        self.sp_stride.setRange(1, 5)
        self.sp_stride.setValue(2)

        self.cb_type = QComboBox()
        self.cb_type.setView(QListView())
        self.cb_type.setStyleSheet("""
                    QComboBox {
                        background-color: #282c34;
                        color: #dcdfe4;
                        border: 1px solid #3e4451;
                        padding: 4px;
                    }
                    QComboBox QAbstractItemView {
                        background-color: #282c34;
                        selection-background-color: #3e4451; /* Boja "trake" na hoveru */
                        selection-color: #61afef;           /* Boja slova na hoveru */
                        border: 1px solid #181a1f;
                    }
                """)
        self.cb_type.addItem("Max Pooling", "max")
        self.cb_type.addItem("Average Pooling", "avg")
        self.cb_type.addItem("L2 Pooling", "l2")
        self.cb_type.addItem("Weighted Average", "weighted")

        form_layout.addRow("Dimenzija ulaza:", self.sp_size)
        form_layout.addRow("Broj kanala:", self.sp_chan)
        form_layout.addRow("Dimenzija filtera:", self.sp_filt)
        form_layout.addRow("Korak (stride):", self.sp_stride)
        form_layout.addRow("Tip pooling-a:", self.cb_type)

        main_vbox.addLayout(form_layout)
        main_vbox.addSpacing(20)

        self.btn_generate = QPushButton("Generiši")
        self.btn_generate.setMinimumHeight(60)
        self.btn_generate.setStyleSheet("""
            background-color: #528bff; 
            color: white; 
            font-size: 20px; 
            font-weight: bold; 
            border-radius: 5px;
        """)

        main_vbox.addWidget(self.btn_generate)
        main_vbox.addStretch()

    def get_params(self):
        """Vraća rečnik sa svim izabranim parametrima"""
        return {
            "size": self.sp_size.value(),
            "channels": self.sp_chan.value(),
            "f_size": self.sp_filt.value(),
            "stride": self.sp_stride.value(),
            "pool_type": self.cb_type.currentData()
        }