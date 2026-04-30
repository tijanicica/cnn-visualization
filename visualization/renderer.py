# visualization/renderer.py
"""
Izometrijska 2.5D vizualizacija CNN operacija pomoću PyQt5 QPainter-a.

Koordinatni sistem:
  - World prostor: (col, row, channel) — prirodne koordinate mape
  - Screen prostor: (x, y) — piksel koordinate na widgetu

Izometrijska projekcija:
  Svaka kocka = 3 poligona (gornje, desno, lijevo lice)
  Boje lica daju iluziju dubine bez pravog 3D renderinga.
"""

from __future__ import annotations
from typing import Optional
import numpy as np

from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import Qt, QPoint, QRect
from PyQt5.QtGui import (
    QPainter, QColor, QPolygon, QPen, QFont, QFontMetrics
)


# ------------------------------------------------------------------
# Konstante projekcije
# ------------------------------------------------------------------

CELL  = 32      # veličina jedne ćelije u world jedinicama
DX    = CELL    # horizontalni pomak po koloni (screen x)
DY    = CELL // 2  # vertikalni pomak po koloni (screen y) — "nagib"
DEPTH = CELL    # visina kocke (Z osa, ide prema dolje na screenu)
GAP   = 2       # razmak između kocki u pikselima


# ------------------------------------------------------------------
# Paleta boja
# ------------------------------------------------------------------

class Palette:
    """
    Boje po tipu bloka.
    Svaki blok ima 3 nijanse: top (najsvjetlija), right (srednja), left (najtamnija).
    """

    INPUT = {
        "top":   QColor(52,  152, 219),   # plava
        "right": QColor(31,  97,  141),
        "left":  QColor(21,  67,  96),
    }
    FILTER = {
        "top":   QColor(230, 126, 34),    # narandžasta
        "right": QColor(154, 84,  22),
        "left":  QColor(100, 55,  15),
    }
    OUTPUT = {
        "top":   QColor(142, 68,  173),   # ljubičasta
        "right": QColor(97,  46,  119),
        "left":  QColor(62,  30,  77),
    }
    ACTIVE = {
        "top":   QColor(39,  174, 96),    # zelena — aktivan region
        "right": QColor(27,  120, 66),
        "left":  QColor(18,  80,  44),
    }
    POSITIVE = {
        "top":   QColor(46,  213, 115),   # jarko zelena — pozitivno poklapanje
        "right": QColor(30,  145, 78),
        "left":  QColor(20,  97,  52),
    }
    NEGATIVE = {
        "top":   QColor(231, 76,  60),    # crvena — negativno poklapanje
        "right": QColor(155, 51,  40),
        "left":  QColor(103, 34,  27),
    }
    EMPTY = {
        "top":   QColor(80,  80,  80),    # siva — prazna ćelija (izlaz još nije popunjen)
        "right": QColor(50,  50,  50),
        "left":  QColor(30,  30,  30),
    }

    TEXT_LIGHT = QColor(255, 255, 255)
    TEXT_DARK  = QColor(20,  20,  20)
    BG         = QColor(18,  18,  24)     # tamna pozadina


# ------------------------------------------------------------------
# Projekcija: world → screen
# ------------------------------------------------------------------

def _world_to_screen(
    col: float,
    row: float,
    channel: float,
    origin_x: int,
    origin_y: int,
) -> tuple[int, int]:
    """
    Pretvara world koordinate (col, row, channel) u screen (x, y).

    Izometrijska formula:
      x = origin_x + col * DX - row * DX
      y = origin_y + col * DY + row * DY - channel * DEPTH

    channel (Z osa) pomjera kocku prema gore na screenu —
    kad imamo više kanala, svaki kanal je "viši" sloj.
    """
    x = origin_x + int(col * DX - row * DX)
    y = origin_y + int(col * DY + row * DY - channel * DEPTH)
    return x, y


def _cube_polygons(
    col: int,
    row: int,
    channel: int,
    origin_x: int,
    origin_y: int,
) -> dict[str, QPolygon]:
    """
    Vraća 3 QPolygon-a za jednu kocku na poziciji (col, row, channel).

    Svaki polygon je lista 4 tačke koja definira jedno lice kocke.
    Tačke su poredane u smjeru kazaljke na satu (konveksni polygon).

    Gornje lice (top):     romb oblika
    Desno lice (right):    paralelogram, desna strana
    Lijevo lice (left):    paralelogram, lijeva strana
    """
    g = GAP  # skraćenica

    # Kutovi kocke u world koordinatama (sa gap-om za estetiku)
    # Gornje lice — 4 tačke romba
    # Idemo u smjeru: lijevo → gore-desno → desno → dole-lijevo

    def pt(c, r, ch):
        sx, sy = _world_to_screen(c, r, ch, origin_x, origin_y)
        return QPoint(sx, sy)

    top = QPolygon([
        pt(col     + g/CELL, row + 1 - g/CELL, channel + 1),  # lijevo
        pt(col     + g/CELL, row     + g/CELL, channel + 1),  # gore-lijevo
        pt(col + 1 - g/CELL, row     + g/CELL, channel + 1),  # gore-desno
        pt(col + 1 - g/CELL, row + 1 - g/CELL, channel + 1),  # desno
    ])

    # Desno lice — između gornjeg-desnog i donjeg-desnog ugla
    right = QPolygon([
        pt(col + 1 - g/CELL, row     + g/CELL, channel + 1),  # gore
        pt(col + 1 - g/CELL, row     + g/CELL, channel    ),  # dole-gore
        pt(col + 1 - g/CELL, row + 1 - g/CELL, channel    ),  # dole
        pt(col + 1 - g/CELL, row + 1 - g/CELL, channel + 1),  # gore-dole
    ])

    # Lijevo lice — između donjeg-lijevog i donjeg-desnog ugla
    left = QPolygon([
        pt(col     + g/CELL, row + 1 - g/CELL, channel + 1),  # gore
        pt(col + 1 - g/CELL, row + 1 - g/CELL, channel + 1),  # gore-desno
        pt(col + 1 - g/CELL, row + 1 - g/CELL, channel    ),  # dole-desno
        pt(col     + g/CELL, row + 1 - g/CELL, channel    ),  # dole
    ])

    return {"top": top, "right": right, "left": left}


# ------------------------------------------------------------------
# Core funkcija za crtanje jednog bloka
# ------------------------------------------------------------------

def draw_block(
    painter: QPainter,
    data: np.ndarray,
    origin_x: int,
    origin_y: int,
    palette: dict,
    highlight_mask: Optional[np.ndarray] = None,
    highlight_palette: Optional[dict] = None,
    special_mask: Optional[dict] = None,   # {(r,c): "positive"/"negative"}
    show_values: bool = True,
    empty_mask: Optional[np.ndarray] = None,  # True = ćelija još nije popunjena
):
    """
    Crta izometrijski blok kocki za datu mapu podataka.

    data          — numpy niz shape (rows, cols) ili (rows, cols, channels)
    origin_x/y    — gornji lijevi ugao bloka na screenu
    palette       — dict s ključevima "top", "right", "left" → QColor
    highlight_mask — bool niz (rows, cols), True = iscrtaj u highlight_palette
    special_mask  — dict (r,c) → "positive"/"negative" za pattern modul
    show_values   — upiši numeričku vrijednost u centar gornjeg lica
    empty_mask    — bool niz, True = kocka je "prazna" (izlaz nije popunjen)
    """

    # Normalizuj na (rows, cols, channels)
    if data.ndim == 2:
        data = data[:, :, np.newaxis]

    rows, cols, channels = data.shape

    font = QFont("Monospace", max(6, CELL // 5))
    font.setBold(True)
    painter.setFont(font)
    fm = QFontMetrics(font)

    # Crtamo od nazad prema naprijed (painter's algorithm)
    # Redoslijed: veći channel, veći row → crtaj prvi (iza)
    # Manji channel, manji row → crtaj zadnji (ispred)

    for ch in range(channels - 1, -1, -1):
        for r in range(rows - 1, -1, -1):
            for c in range(cols - 1, -1, -1):

                # Odredi paletu za ovu kocku
                if special_mask and (r, c) in special_mask:
                    sp = special_mask[(r, c)]
                    if sp == "positive":
                        pal = Palette.POSITIVE
                    else:
                        pal = Palette.NEGATIVE
                elif highlight_mask is not None and highlight_mask[r, c]:
                    pal = highlight_palette or Palette.ACTIVE
                elif empty_mask is not None and empty_mask[r, c]:
                    pal = Palette.EMPTY
                else:
                    pal = palette

                polys = _cube_polygons(c, r, ch, origin_x, origin_y)

                # Crtaj 3 lica
                painter.setPen(QPen(Qt.black, 0.5))

                painter.setBrush(pal["top"])
                painter.drawPolygon(polys["top"])

                painter.setBrush(pal["right"])
                painter.drawPolygon(polys["right"])

                painter.setBrush(pal["left"])
                painter.drawPolygon(polys["left"])

                # Upiši vrijednost u centar gornjeg lica (samo za top channel)
                if show_values and ch == channels - 1:
                    val = data[r, c, ch]

                    # Centar gornjeg lica = prosječna tačka romba
                    top_poly = polys["top"]
                    cx = sum(top_poly.point(i).x() for i in range(4)) // 4
                    cy = sum(top_poly.point(i).y() for i in range(4)) // 4

                    # Format: int ako je cijeli broj, inače 1 decimala
                    if isinstance(val, (int, np.integer)):
                        text = str(int(val))
                    elif val == int(val):
                        text = str(int(val))
                    else:
                        text = f"{val:.1f}"

                    # Centriraj tekst
                    tw = fm.horizontalAdvance(text)
                    th = fm.height()

                    painter.setPen(QPen(Palette.TEXT_LIGHT))
                    painter.drawText(cx - tw // 2, cy + th // 4, text)


# ------------------------------------------------------------------
# Pomoćna: izračunaj dimenzije bloka na screenu
# ------------------------------------------------------------------

def block_screen_size(rows: int, cols: int, channels: int = 1) -> tuple[int, int]:
    """
    Vraća (width, height) koji blok zauzima na screenu.
    Korisno za pozicioniranje blokova jedan pored drugog.
    """
    # Najšira tačka: gornji desni ugao (col=cols, row=0)
    x_max, _ = _world_to_screen(cols, 0, channels, 0, 0)
    # Najniža tačka: donji lijevi ugao na ch=0
    _, y_max = _world_to_screen(0, rows, 0, 0, 0)
    # Najviša tačka: gornji lijevi ugao na ch=channels
    _, y_min = _world_to_screen(0, 0, channels, 0, 0)

    return x_max, y_max - y_min


# ------------------------------------------------------------------
# Glavni widget
# ------------------------------------------------------------------

class CNNRenderWidget(QWidget):
    """
    QWidget koji crta izometrijsku vizualizaciju CNN operacija.

    Pozovi set_convolution_state() / set_pooling_state() / set_pattern_state()
    da ažuriraš prikaz, pa update() da pokrneš repaint.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(900, 500)
        self._state = None      # dict koji opisuje šta se crta
        self._mode  = None      # "conv" | "pool" | "pattern"
        self.setStyleSheet(f"background-color: rgb(18, 18, 24);")

    # ------------------------------------------------------------------
    # Javni API — postavljanje stanja
    # ------------------------------------------------------------------

    def set_convolution_state(self, engine, step: dict):
        """
        Pripremi sve podatke za crtanje konvolucije.
        engine — ConvolutionEngine
        step   — trenutni korak (iz engine.get_current_step())
        """
        self._mode = "conv"
        self._state = {
            "padded":  engine.padded_input,
            "filter":  engine.filter_weights,
            "output":  engine.output_map,
            "out_size": engine.output_size,
            "filter_size": engine.filter_size,
            "in_row":  step["in_row"],
            "in_col":  step["in_col"],
            "out_row": step["out_row"],
            "out_col": step["out_col"],
            "formula": _build_conv_formula(step, engine.bias),
            "padding": engine.padding,
            "bias":    engine.bias,
            "step_idx": engine.current_step,
            "total_steps": len(engine.steps),
        }
        self.update()   # trigeruje paintEvent

    def set_pooling_state(self, engine, step: dict):
        """engine — PoolingEngine"""
        self._mode = "pool"
        self._state = {
            "input":   engine.input_map,
            "output":  engine.output_map[:, :, 0],  # prikazujemo kanal 0
            "filter_size": engine.filter_size,
            "out_size": engine.output_size,
            "in_row":  step["in_row"],
            "in_col":  step["in_col"],
            "out_row": step["out_row"],
            "out_col": step["out_col"],
            "formula": _build_pool_formula(step, engine.pool_type),
            "weights": engine.weights if engine.pool_type == "weighted" else None,
            "pool_type": engine.pool_type,
            "step_idx": engine.current_step,
            "total_steps": len(engine.steps),
        }
        self.update()

    def set_pattern_state(self, engine, step: dict, filter_idx: int):
        """engine — PatternEngine"""
        self._mode = "pattern"

        # Napravi special_mask za ovaj filter
        special_mask = {}
        F = engine.FILTER_SIZE
        for sr in engine.special_regions:
            if sr["filter_idx"] == filter_idx:
                for dr in range(F):
                    for dc in range(F):
                        special_mask[(sr["row"] + dr, sr["col"] + dc)] = sr["type"]

        self._state = {
            "input":    engine.input_map,
            "filter":   engine.filters[filter_idx],
            "output":   engine.output_maps[filter_idx],
            "out_size": engine.output_size,
            "filter_size": engine.FILTER_SIZE,
            "in_row":   step["in_row"],
            "in_col":   step["in_col"],
            "out_row":  step["out_row"],
            "out_col":  step["out_col"],
            "special_mask": special_mask,
            "step_special": step["special"],
            "conv_sum": step["conv_sum"],
            "filter_idx": filter_idx,
            "step_idx": engine.current_step,
            "total_steps": len(engine.steps_per_filter[filter_idx]),
        }
        self.update()

    # ------------------------------------------------------------------
    # paintEvent — sve crtanje ide ovdje
    # ------------------------------------------------------------------

    def paintEvent(self, event):
        if self._state is None:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Tamna pozadina
        painter.fillRect(self.rect(), Palette.BG)

        if self._mode == "conv":
            self._paint_convolution(painter)
        elif self._mode == "pool":
            self._paint_pooling(painter)
        elif self._mode == "pattern":
            self._paint_pattern(painter)

        painter.end()

    # ------------------------------------------------------------------
    # Crtanje konvolucije
    # ------------------------------------------------------------------

    def _paint_convolution(self, painter: QPainter):
        s = self._state
        padded = s["padded"]   # (H, W, C)
        filt   = s["filter"]   # (F, F, C)
        out    = s["output"]   # (out, out)
        F      = s["filter_size"]
        rows_p, cols_p, channels = padded.shape

        w = self.width()
        h = self.height()

        # --- Izračunaj gdje se svaki blok crta ---
        # Rasporedimo ih horizontalno s razmakom
        margin = 30

        inp_w, inp_h = block_screen_size(rows_p, cols_p, channels)
        fil_w, fil_h = block_screen_size(F, F, channels)
        out_w, out_h = block_screen_size(s["out_size"], s["out_size"], 1)

        total_w = inp_w + fil_w + out_w + margin * 4
        start_x = (w - total_w) // 2
        center_y = h // 2 + 40   # malo ispod sredine da ima mjesta za naslov

        # Početne Y koordinate za svaki blok (poravnaj po dnu)
        inp_ox = start_x + margin
        inp_oy = center_y - inp_h // 2

        fil_ox = inp_ox + inp_w + margin * 2
        fil_oy = center_y - fil_h // 2

        out_ox = fil_ox + fil_w + margin * 2
        out_oy = center_y - out_h // 2

        # --- Highlight maska za ulaz ---
        hl_mask = np.zeros((rows_p, cols_p), dtype=bool)
        r0, c0 = s["in_row"], s["in_col"]
        hl_mask[r0 : r0 + F, c0 : c0 + F] = True

        # --- Empty maska za izlaz (ćelije koje još nisu popunjene) ---
        out_size = s["out_size"]
        empty = np.ones((out_size, out_size), dtype=bool)
        # Sve do trenutnog koraka (uključujući) su popunjene
        step_idx = s["step_idx"]
        for idx in range(step_idx + 1):
            ri = idx // out_size
            ci = idx % out_size
            if ri < out_size and ci < out_size:
                empty[ri, ci] = False

        # --- Crtaj blokove ---
        draw_block(painter, padded, inp_ox, inp_oy,
                   Palette.INPUT,
                   highlight_mask=hl_mask,
                   highlight_palette=Palette.ACTIVE)

        draw_block(painter, filt, fil_ox, fil_oy,
                   Palette.FILTER)

        draw_block(painter, out, out_ox, out_oy,
                   Palette.OUTPUT,
                   empty_mask=empty)

        # --- Labele ispod blokova ---
        self._draw_label(painter, inp_ox + inp_w // 2, center_y + 80,
                         f"Ulaz {'(padded)' if s['padding'] else ''} "
                         f"{rows_p}×{cols_p}×{channels}",
                         QColor(100, 180, 255))

        self._draw_label(painter, fil_ox + fil_w // 2, center_y + 80,
                         f"Filter {F}×{F}×{channels}",
                         QColor(255, 160, 80))

        self._draw_label(painter, out_ox + out_w // 2, center_y + 80,
                         f"Izlaz {out_size}×{out_size}",
                         QColor(180, 120, 255))

        # --- Formula u vrhu ---
        self._draw_formula(painter, s["formula"],
                           s["step_idx"] + 1, s["total_steps"])

        # --- Progress bar ---
        self._draw_progress(painter,
                            s["step_idx"] + 1, s["total_steps"])

    # ------------------------------------------------------------------
    # Crtanje poolinga
    # ------------------------------------------------------------------

    def _paint_pooling(self, painter: QPainter):
        s = self._state
        inp    = s["input"]    # (H, W, C)
        out    = s["output"]   # (out, out) — kanal 0
        F      = s["filter_size"]
        rows_i = s["input"].shape[0]

        w = self.width()
        h = self.height()
        margin = 40
        center_y = h // 2 + 40

        inp_w, inp_h = block_screen_size(rows_i, rows_i, inp.shape[2])
        out_size = s["out_size"]
        out_w, out_h = block_screen_size(out_size, out_size, 1)

        has_weights = s["weights"] is not None
        if has_weights:
            wgt_w, wgt_h = block_screen_size(F, F, 1)
            total_w = inp_w + wgt_w + out_w + margin * 4
        else:
            total_w = inp_w + out_w + margin * 3

        start_x = (w - total_w) // 2
        inp_ox = start_x + margin
        inp_oy = center_y - inp_h // 2

        # Highlight maska
        hl_mask = np.zeros((rows_i, rows_i), dtype=bool)
        r0, c0 = s["in_row"], s["in_col"]
        hl_mask[r0 : r0 + F, c0 : c0 + F] = True

        draw_block(painter, inp, inp_ox, inp_oy,
                   Palette.INPUT, highlight_mask=hl_mask)

        next_x = inp_ox + inp_w + margin * 2

        if has_weights:
            wgt_oy = center_y - wgt_h // 2
            draw_block(painter, s["weights"], next_x, wgt_oy, Palette.FILTER)
            self._draw_label(painter, next_x + wgt_w // 2, center_y + 80,
                             "Težine", QColor(255, 160, 80))
            next_x += wgt_w + margin * 2

        # Empty maska za izlaz
        empty = np.ones((out_size, out_size), dtype=bool)
        step_idx = s["step_idx"]
        for idx in range(step_idx + 1):
            ri = idx // out_size
            ci = idx % out_size
            if ri < out_size and ci < out_size:
                empty[ri, ci] = False

        out_oy = center_y - out_h // 2
        draw_block(painter, out, next_x, out_oy,
                   Palette.OUTPUT, empty_mask=empty,
                   show_values=True)

        self._draw_label(painter, inp_ox + inp_w // 2, center_y + 80,
                         f"Ulaz {rows_i}×{rows_i}×{inp.shape[2]}",
                         QColor(100, 180, 255))
        self._draw_label(painter, next_x + out_w // 2, center_y + 80,
                         f"Izlaz {out_size}×{out_size}",
                         QColor(180, 120, 255))

        self._draw_formula(painter, s["formula"],
                           s["step_idx"] + 1, s["total_steps"])
        self._draw_progress(painter, s["step_idx"] + 1, s["total_steps"])

    # ------------------------------------------------------------------
    # Crtanje detekcije obrazaca
    # ------------------------------------------------------------------

    def _paint_pattern(self, painter: QPainter):
        s = self._state
        inp  = s["input"]    # (12, 12)
        filt = s["filter"]   # (3, 3)
        out  = s["output"]   # (10, 10)
        F    = s["filter_size"]
        out_size = s["out_size"]

        w = self.width()
        h = self.height()
        margin = 30
        center_y = h // 2 + 40

        inp_w, inp_h = block_screen_size(12, 12, 1)
        fil_w, fil_h = block_screen_size(F, F, 1)
        out_w, out_h = block_screen_size(out_size, out_size, 1)

        total_w = inp_w + fil_w + out_w + margin * 4
        start_x = (w - total_w) // 2

        inp_ox = start_x + margin
        inp_oy = center_y - inp_h // 2

        fil_ox = inp_ox + inp_w + margin * 2
        fil_oy = center_y - fil_h // 2

        out_ox = fil_ox + fil_w + margin * 2
        out_oy = center_y - out_h // 2

        # Highlight maska za aktivan region
        hl_mask = np.zeros((12, 12), dtype=bool)
        r0, c0 = s["in_row"], s["in_col"]
        hl_mask[r0 : r0 + F, c0 : c0 + F] = True

        # Special maska: positive/negative poklapanja
        # Pretvori iz dict u format koji draw_block razumije
        special_mask = s["special_mask"]

        draw_block(painter, inp, inp_ox, inp_oy,
                   Palette.INPUT,
                   highlight_mask=hl_mask,
                   highlight_palette=Palette.ACTIVE,
                   special_mask=special_mask,
                   show_values=False)   # 12x12 je premalo za vrijednosti

        draw_block(painter, filt, fil_ox, fil_oy, Palette.FILTER)

        # Empty maska za izlaz
        empty = np.ones((out_size, out_size), dtype=bool)
        for idx in range(s["step_idx"] + 1):
            ri = idx // out_size
            ci = idx % out_size
            if ri < out_size and ci < out_size:
                empty[ri, ci] = False

        draw_block(painter, out, out_ox, out_oy,
                   Palette.OUTPUT, empty_mask=empty,
                   show_values=False)

        # Labele
        self._draw_label(painter, inp_ox + inp_w // 2, center_y + 80,
                         "Ulaz 12×12", QColor(100, 180, 255))
        self._draw_label(painter, fil_ox + fil_w // 2, center_y + 80,
                         f"Filter {s['filter_idx']+1} (3×3)",
                         QColor(255, 160, 80))
        self._draw_label(painter, out_ox + out_w // 2, center_y + 80,
                         "Izlaz 10×10", QColor(180, 120, 255))

        # Formula s informacijom o specijalnom regionu
        special = s["step_special"]
        conv_sum = s["conv_sum"]
        if special == "positive":
            formula = f"Σ = {conv_sum:.1f}  ← MAKSIMUM (savršeno poklapanje!)"
            fcolor = QColor(46, 213, 115)
        elif special == "negative":
            formula = f"Σ = {conv_sum:.1f}  ← MINIMUM (negativno poklapanje!)"
            fcolor = QColor(231, 76, 60)
        else:
            formula = f"Σ = {conv_sum:.1f}"
            fcolor = QColor(220, 220, 220)

        self._draw_formula(painter, formula,
                           s["step_idx"] + 1, s["total_steps"],
                           color=fcolor)
        self._draw_progress(painter, s["step_idx"] + 1, s["total_steps"])

    # ------------------------------------------------------------------
    # Pomoćne metode za crtanje UI elemenata
    # ------------------------------------------------------------------

    def _draw_label(self, painter: QPainter, cx: int, y: int,
                    text: str, color: QColor):
        """Crta label centrirano ispod bloka."""
        font = QFont("Sans", 10)
        font.setBold(True)
        painter.setFont(font)
        fm = QFontMetrics(font)
        tw = fm.horizontalAdvance(text)
        painter.setPen(QPen(color))
        painter.drawText(cx - tw // 2, y, text)

    def _draw_formula(self, painter: QPainter, formula: str,
                      step: int, total: int,
                      color: QColor = QColor(220, 220, 220)):
        """Crta formulu i info o koraku u vrhu widgeta."""
        font = QFont("Monospace", 11)
        painter.setFont(font)
        fm = QFontMetrics(font)

        # Info o koraku — gornji desni ugao
        step_text = f"Korak {step}/{total}"
        painter.setPen(QPen(QColor(150, 150, 150)))
        painter.drawText(self.width() - fm.horizontalAdvance(step_text) - 20,
                         30, step_text)

        # Formula — gornji centar
        painter.setPen(QPen(color))
        tw = fm.horizontalAdvance(formula)
        painter.drawText(self.width() // 2 - tw // 2, 30, formula)

    def _draw_progress(self, painter: QPainter, step: int, total: int):
        """Crta progress bar na dnu widgeta."""
        bar_h = 4
        bar_y = self.height() - bar_h - 8
        bar_w = self.width() - 40

        # Pozadina
        painter.setBrush(QColor(50, 50, 60))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(20, bar_y, bar_w, bar_h, 2, 2)

        # Napredak
        progress_w = int(bar_w * step / max(total, 1))
        painter.setBrush(QColor(52, 152, 219))
        painter.drawRoundedRect(20, bar_y, progress_w, bar_h, 2, 2)


# ------------------------------------------------------------------
# Pomoćne funkcije za formatiranje formula
# ------------------------------------------------------------------

def _build_conv_formula(step: dict, bias: int) -> str:
    """Gradi string formule za trenutni korak konvolucije."""
    conv_sum = step["conv_sum"]
    out_val  = step["output_val"]
    r, c     = step["out_row"], step["out_col"]

    if bias:
        return (f"Σ(ulaz × filter) = {conv_sum}  "
                f"+ bias({bias}) = {out_val}  "
                f"→ izlaz[{r},{c}]")
    else:
        return f"Σ(ulaz × filter) = {conv_sum}  →  izlaz[{r},{c}]"


def _build_pool_formula(step: dict, pool_type: str) -> str:
    """Gradi string formule za trenutni korak poolinga."""
    vals   = step["regions"][0].flatten().astype(int)
    result = step["output_vals"][0]
    r, c   = step["out_row"], step["out_col"]
    vstr   = ", ".join(map(str, vals))

    if pool_type == "max":
        return f"max({vstr}) = {result:.2f}  →  izlaz[{r},{c}]"
    elif pool_type == "avg":
        return f"avg({vstr}) = {result:.2f}  →  izlaz[{r},{c}]"
    elif pool_type == "l2":
        return f"L2({vstr}) = {result:.2f}  →  izlaz[{r},{c}]"
    else:
        return f"weighted_avg({vstr}) = {result:.2f}  →  izlaz[{r},{c}]"