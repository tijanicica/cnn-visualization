# visualization/exporter.py
import cv2
import numpy as np
import os
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QProgressDialog, QApplication
from PyQt5.QtCore import Qt


class VideoExporter:
    def __init__(self, fig, animator):
        self.fig = fig
        self.animator = animator

    def export(self, parent_widget):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            parent_widget, "Sačuvaj Video", "cnn_vizualizacija.mp4",
            "Video fajlovi (*.mp4);;Sve datoteke (*)", options=options
        )

        if not file_path:
            return

        engine = self.animator.engine
        if self.animator.mode == "pattern":
            total_steps = engine.output_size * engine.output_size
        else:
            total_steps = len(engine.steps)

        # 1. Kreiranje i konfiguracija popup prozora
        progress = QProgressDialog("Snimanje u toku...", "Otkaži", 0, total_steps, parent_widget)
        progress.setWindowModality(Qt.WindowModal)
        progress.setWindowTitle("Video Export")
        progress.setMinimumWidth(300)  # Forsiramo širinu da ne bude kvadratić
        progress.show()

        # OVO JE KLJUČNO: Tera Qt da odmah nacrta prozor pre nego što krene teška obrada
        QApplication.processEvents()

        saved_step = engine.current_step
        engine.reset()

        fps = 1.5
        width, height = self.fig.canvas.get_width_height()
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(file_path, fourcc, fps, (width, height))

        try:
            img_bgr = None
            for i in range(total_steps):
                if progress.wasCanceled():
                    break

                # Pomeranje animacije
                self.animator.draw_current()
                self.fig.canvas.draw()

                # Snimanje frejma
                img_rgba = np.array(self.fig.canvas.buffer_rgba())
                img_bgr = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2BGR)
                out.write(img_bgr)

                # Pomeranje logike
                engine.next_step()

                # 2. AŽURIRANJE POPUP-A
                progress.setValue(i + 1)

                # OVO REŠAVA VAŠ PROBLEM: Dopušta Windowsu da osveži prozorčić
                QApplication.processEvents()

            # Pauza na kraju
            if not progress.wasCanceled() and img_bgr is not None:
                for _ in range(int(fps * 3)):
                    out.write(img_bgr)
                    QApplication.processEvents()  # Da se ne zamrzne ni na kraju

            out.release()
            if not progress.wasCanceled():
                QMessageBox.information(parent_widget, "Uspeh", f"Video je uspešno snimljen!")

        except Exception as e:
            if 'out' in locals(): out.release()
            QMessageBox.critical(parent_widget, "Greška", f"Greška pri snimanju:\n{str(e)}")

        finally:
            engine.current_step = saved_step
            self.animator.draw_current()
            QApplication.processEvents()