# main.py
import sys
from PyQt5.QtWidgets import QApplication
from gui.app import MainWindow


def main():
    app = QApplication(sys.argv)

    app.setStyle("Fusion")
    dark_stylesheet = """
        QWidget { background-color: #1e2128; color: #d3d3d3; font-family: 'Segoe UI', Arial; }

        /* --- OGROMAN FONT ZA KONTROLE (22px - 25px) --- */
        QLabel { font-size: 22px; } 
        QSpinBox { 
            background-color: #2c313c; 
            border: 1px solid #3e4451; 
            padding: 10px; 
            font-size: 25px; /* Vaš traženi font za brojeve */
            font-weight: bold;
        }

        /* CHECKBOX JE SADA TAKOĐE VEĆI DA PRATI TEKST OD 25px */
        QCheckBox { font-size: 22px; }
        QCheckBox::indicator { width: 25px; height: 25px; border: 2px solid #528bff; border-radius: 4px; background-color: #2c313c; }
        QCheckBox::indicator:checked { background-color: #528bff; image: url(check.png); }

        /* DUGMIĆI */
        QPushButton { background-color: #2c313c; border: 1px solid #3e4451; padding: 6px; border-radius: 4px; font-size: 16px;}
        QPushButton:hover { background-color: #021b75; border: 1px solid #528bff; }

        /* --- TABOVI (Sređen razmak da stanu sva 3 taba na ekran) --- */
        QTabWidget::pane { border: 1px solid #3e4451; }
        QTabBar::tab { 
            background: #2c313c; 
            padding: 15px 5px;   /* Smanjen horizontalni prostor po tabu */
            border: 1px solid #3e4451; 
            font-size: 16px; 
            font-weight: bold; 
        }
        QTabBar::tab:selected { background: #1e2128; border-bottom: 3px solid #528bff; color: #528bff; }

        QProgressBar { text-align: center; border: 1px solid #3e4451; background-color: #2c313c; font-weight: bold; font-size: 14px;}
        QProgressBar::chunk { background-color: #528bff; }
    """
    app.setStyleSheet(dark_stylesheet)

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()