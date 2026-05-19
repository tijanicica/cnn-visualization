# main.py
import sys
from PyQt5.QtWidgets import QApplication
from gui.app import MainWindow


def main():
    app = QApplication(sys.argv)

    app.setStyle("Fusion")
    dark_stylesheet = """
        QWidget { background-color: #1e2128; color: #d3d3d3; font-family: 'Segoe UI', Arial; font-size: 13px; }
        QPushButton { background-color: #2c313c; border: 1px solid #3e4451; padding: 6px; border-radius: 4px; }
        QPushButton:hover { background-color: #3e4451; border: 1px solid #528bff; }
        QComboBox, QSpinBox { background-color: #2c313c; border: 1px solid #3e4451; padding: 5px; }

        /* TABOVI - Siri i krupniji */
        QTabWidget::pane { border: 1px solid #3e4451; }
        QTabBar::tab { background: #2c313c; padding: 12px 25px; margin-right: 2px; border: 1px solid #3e4451; min-width: 200px; font-size: 14px; font-weight: bold; }
        QTabBar::tab:selected { background: #1e2128; border-bottom: 2px solid #528bff; color: #528bff; }

        /* CHECKBOX - Jasan okvir */
        QCheckBox::indicator { width: 18px; height: 18px; border: 2px solid #528bff; border-radius: 4px; background-color: #2c313c; }
        QCheckBox::indicator:checked { background-color: #528bff; image: url(check.png); } /* Qt interno generiše checkmark ako nema slike */
        QCheckBox { font-size: 14px; }

        QProgressBar { text-align: center; border: 1px solid #3e4451; background-color: #2c313c; font-weight: bold;}
        QProgressBar::chunk { background-color: #528bff; }
    """
    app.setStyleSheet(dark_stylesheet)

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()