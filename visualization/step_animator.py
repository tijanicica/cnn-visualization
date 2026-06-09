# visualization/step_animator.py
"""
StepAnimator — posrednik između engine-a i renderer-a.

GUI ne zna ništa o tome kako se računa konvolucija niti kako se crta.
On samo poziva:
    animator.next()
    animator.prev()
    animator.start_auto()
    animator.stop_auto()

Animator se brine za sve ostalo.
"""

from PyQt5.QtCore import QTimer
from matplotlib.figure import Figure

# Importujemo sve tri render funkcije
from visualization.renderer_3d import (
    render_convolution,
    render_pooling,
    render_pattern,
)


class StepAnimator:
    """
    Upravlja korak-po-korak izvršavanjem i auto-play animacijom
    za jedan od tri moda: konvolucija, pooling, pattern.

    Parametri:
        engine   — ConvolutionEngine | PoolingEngine | PatternEngine
        fig      — Matplotlib Figure koji se prikazuje u GUI-u
        mode     — "conv" | "pool" | "pattern"
        on_step  — callback koji GUI poziva nakon svakog koraka
                   (npr. da ažurira progress bar ili info panel)
        interval — millisekundi između auto-play koraka (default 800ms)

    Tipičan tok:
        1. GUI kreira engine s parametrima
        2. GUI kreira StepAnimator(engine, fig, mode, on_step=gui.refresh)
        3. animator.draw_current() — crtaj početno stanje
        4. Korisnik klikne "Sljedeći" → animator.next()
           ili "Auto" → animator.start_auto()
    """

    def __init__(self, engine, fig: Figure, mode: str,
                 on_step=None, interval: int = 800):
        self.engine   = engine
        self.fig      = fig
        self.mode     = mode          # "conv" | "pool" | "pattern"
        self.on_step  = on_step       # callback: fn() → None
        self.interval = interval

        # QTimer za auto-play — ne starta odmah
        self._timer = QTimer()
        self._timer.setInterval(interval)
        self._timer.timeout.connect(self._auto_tick)

        # Za pattern mod: koji filter trenutno prikazujemo
        # (engine.current_filter je autoritativan, ovo je samo cache)
        self._filter_idx = 0

    # ------------------------------------------------------------------
    # Crtanje
    # ------------------------------------------------------------------

    def draw_current(self):
        """
        Crta trenutni korak bez pomjeranja.
        Poziva se: pri inicijalizaciji, nakon next/prev, i nakon
        promjene parametara (regeneracija).
        """
        step = self.engine.get_current_step()

        if self.mode == "conv":
            render_convolution(self.fig, self.engine, step)

        elif self.mode == "pool":
            render_pooling(self.fig, self.engine, step)

        elif self.mode == "pattern":
            render_pattern(
                self.fig,
                self.engine,
                step,
                filter_idx=self.engine.current_filter,
            )

        # Notify GUI da osvježi canvas
        if self.on_step:
            self.on_step()

    # ------------------------------------------------------------------
    # Navigacija
    # ------------------------------------------------------------------

    def next(self) -> bool:
        """
        Ide na sljedeći korak.
        Vraća False ako smo već na kraju (GUI može deaktivirati dugme).
        """
        moved = self.engine.next_step()
        if moved:
            self.draw_current()
        else:
            # Na kraju — zaustavi auto-play ako je aktivan
            self.stop_auto()
        return moved

    def prev(self) -> bool:
        """
        Ide na prethodni korak.
        Vraća False ako smo na početku.
        """
        moved = self.engine.prev_step()
        if moved:
            self.draw_current()
        return moved

    def reset(self):
        """Vrati na početak i nacrtaj inicijalno stanje."""
        self.stop_auto()
        self.engine.reset()
        self.draw_current()

    # ------------------------------------------------------------------
    # Auto-play
    # ------------------------------------------------------------------

    def start_auto(self):
        """Pokreni automatsko izvršavanje korak po korak."""
        if not self._timer.isActive():
            self._timer.start()

    def stop_auto(self):
        """Zaustavi automatsko izvršavanje."""
        if self._timer.isActive():
            self._timer.stop()

    def toggle_auto(self) -> bool:
        """
        Starta ili zaustavlja auto-play.
        Vraća True ako je auto-play sada aktivan.
        Korisno za GUI dugme koje mijenja label "Auto ▶" / "Pauza ⏸".
        """
        if self._timer.isActive():
            self.stop_auto()
            return False
        else:
            self.start_auto()
            return True

    def is_auto_running(self) -> bool:
        return self._timer.isActive()

    def set_interval(self, ms: int):
        """Promijeni brzinu auto-play-a u toku rada."""
        self.interval = ms
        self._timer.setInterval(ms)

    def _auto_tick(self):
        """Interni slot — poziva se iz QTimer-a svaki interval."""
        finished = not self.next()
        if finished:
            # next() je već pozvao stop_auto() — samo notify GUI
            if self.on_step:
                self.on_step()

    # ------------------------------------------------------------------
    # Pattern-specifično: mijenjanje aktivnog filtera
    # ------------------------------------------------------------------

    def set_filter(self, filter_idx: int):
        """
        Samo za pattern mod — prebaci na drugi filter.
        Resetuje korake za taj filter i crta inicijalno stanje.
        """
        if self.mode != "pattern":
            return
        self.stop_auto()
        self.engine.set_filter(filter_idx)
        self._filter_idx = filter_idx
        self.draw_current()

    # ------------------------------------------------------------------
    # Stanje za GUI (info panel, progress bar)
    # ------------------------------------------------------------------

    def get_progress(self) -> tuple[int, int]:
        """
        Vraća (trenutni_korak, ukupno_koraka) za progress bar.
        Indeksiranje od 1 (korak 1/9, ne 0/9).
        """
        step = self.engine.get_current_step()

        if self.mode == "pattern":
            idx   = step["step_idx"]
            total = step["total_steps"]
        else:
            # ConvolutionEngine i PoolingEngine čuvaju current_step direktno
            idx   = self.engine.current_step
            total = len(self.engine.steps)

        return idx + 1, total

    def is_at_start(self) -> bool:
        return self.engine.current_step == 0

    def is_at_end(self) -> bool:
        return self.engine.is_finished()