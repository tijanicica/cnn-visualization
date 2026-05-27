# visualization/exporter.py
import cv2
import numpy as np
import os
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QProgressDialog
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

        progress = QProgressDialog("Priprema za snimanje...", "Otkaži", 0, total_steps, parent_widget)
        progress.setWindowModality(Qt.WindowModal)
        progress.show()

        saved_step = engine.current_step
        engine.reset()

        # Brzina: 1.5 frejma u sekundi (veoma sporo i pregledno)
        fps = 1.5
        width, height = self.fig.canvas.get_width_height()
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(file_path, fourcc, fps, (width, height))

        try:
            img_bgr = None
            for i in range(total_steps):
                if progress.wasCanceled():
                    break

                # 1. Nacrtaj trenutno stanje
                self.animator.draw_current()

                # 2. DODAVANJE NATPISA "SNIMANJE U TOKU" DIREKTNO NA GRAFIK
                # Postavljamo tekst u gornji levi ugao (koordinate 0.02, 0.95 u odnosu na sliku)
                # status_text = "SNIMANJE U TOKU..."
                # label = self.fig.text(0.02, 0.96, status_text,
                #                       color='red', fontsize=12, fontweight='bold',
                #                       bbox=dict(facecolor='white', edgecolor='red', alpha=0.9))

                self.fig.canvas.draw()

                # 3. Pretvori u OpenCV format
                img_rgba = np.array(self.fig.canvas.buffer_rgba())
                img_bgr = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2BGR)

                # 4. Upiši frejm
                out.write(img_bgr)

                # 5. Ukloni labelu da se ne bi duplirala u sledećem koraku rendera
                #label.remove()

                engine.next_step()
                progress.setLabelText("Snimanje u toku")
                progress.setValue(i + 1)

            # --- DODAVANJE PAUZE NA KRAJU (Završni ekran stoji 4 sekunde) ---
            if not progress.wasCanceled() and img_bgr is not None:
                # Ponovo nacrtaj poslednje stanje bez natpisa "Snimanje u toku" za čist kraj
                self.animator.draw_current()
                self.fig.canvas.draw()
                final_rgba = np.array(self.fig.canvas.buffer_rgba())
                final_bgr = cv2.cvtColor(final_rgba, cv2.COLOR_RGBA2BGR)

                for _ in range(int(fps * 1.5)):
                    out.write(final_bgr)

            out.release()
            if not progress.wasCanceled():
                QMessageBox.information(parent_widget, "Uspeh", f"Video je uspešno snimljen!\nNa putanju: {file_path}")

        except Exception as e:
            if 'out' in locals(): out.release()
            QMessageBox.critical(parent_widget, "Greška", f"Greška pri snimanju:\n{str(e)}")

        finally:
            # Vrati korisnika tamo gde je bio
            engine.current_step = saved_step
            self.animator.draw_current()