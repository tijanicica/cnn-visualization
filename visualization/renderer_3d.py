"""
visualization/renderer_3d.py
-----------------------------
Hibridni layout:
  - Gornji red:  3D blokovi (Matplotlib mpl_toolkits) — ulaz, filter, izlaz
  - Donji panel: 2D prikaz računanja trenutnog koraka (patch × filter = produkti → Σ)

Svaka "kocka" u 3D prikazu = gornje lice (popunjen quad) + 4 wireframe linije prema dole.
Brojevi se ne ispisuju u 3D blokovima (prenatrpano) — sve brojeve vidiš u 2D panelu.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import Optional

matplotlib.rcParams['font.family'] = 'monospace'

# ─── Boje ────────────────────────────────────────────────────────────────────

C_BG            = "#0a0f1e"
C_INPUT_FACE    = "#1a3a5c"
C_INPUT_EDGE    = "#2196F3"
C_FILTER_FACE   = "#3a1a1a"
C_FILTER_EDGE   = "#FF5722"
C_ACTIVE_FACE   = "#1a3a1a"
C_ACTIVE_EDGE   = "#00E676"
C_OUTPUT_FACE   = "#2a1a3a"
C_OUTPUT_EDGE   = "#7C4DFF"
C_OUTPUT_ACTIVE = "#9C27B0"
C_EMPTY_FACE    = "#111118"
C_EMPTY_EDGE    = "#2a2a3a"
C_POSITIVE_FACE = "#0a2a0a"
C_POSITIVE_EDGE = "#00C853"
C_NEGATIVE_FACE = "#2a0a0a"
C_NEGATIVE_EDGE = "#FF1744"
C_TEXT_PRIMARY  = "#E0F7FA"
C_TEXT_DIM      = "#546E7A"
C_PANEL_BG      = "#0d1426"
C_GRID_LINE     = "#1a2a3a"
C_ACCENT        = "#00E5FF"


# ─── 3D helpers ──────────────────────────────────────────────────────────────

def _top_face(x: float, y: float, z: float, dx=0.82, dy=0.82) -> list:
    """
    Gornje lice kocke — jedan popunjen paralelogram (quad).
    x, y, z  — donji lijevi ugao kocke
    dx, dy   — veličina (< 1.0 ostavlja gap između kocki)
    """
    return [(x,    y,    z),
            (x+dx, y,    z),
            (x+dx, y+dy, z),
            (x,    y+dy, z)]


def _draw_cell(ax: Axes3D,
               col: int, row: int, ch: int,
               face_color: str, edge_color: str,
               alpha: float = 0.82,
               wire_alpha: float = 0.45,
               dz: float = 0.75):
    """
    Crta jednu ćeliju kao:
      - popunjeno gornje lice na visini z=ch+dz
      - 4 tanke wireframe linije od gornjeg lica prema dole (z=ch)

    col, row  — pozicija u 2D mapi
    ch        — kanal (Z osa — kanali se slažu jedan iza drugog)
    dz        — visina kocke (samo za wireframe, gornje lice je uvijek na vrhu)
    """
    dx = dy = 0.82
    x, y, z_top = col, row, ch + dz

    # Gornje lice
    face = _top_face(x, y, z_top, dx, dy)
    poly = Poly3DCollection([face], alpha=alpha, zorder=ch)
    poly.set_facecolor(face_color)
    poly.set_edgecolor(edge_color)
    poly.set_linewidth(0.9)
    ax.add_collection3d(poly)

    # Wireframe — 4 vertikalne linije
    corners = [(x, y), (x+dx, y), (x+dx, y+dy), (x, y+dy)]
    for cx, cy in corners:
        ax.plot([cx, cx], [cy, cy], [ch, z_top],
                color=edge_color, linewidth=0.55,
                alpha=wire_alpha, zorder=ch - 1)


def _draw_block(ax: Axes3D,
                data: np.ndarray,
                origin: tuple = (0, 0, 0),
                face_color: str = C_INPUT_FACE,
                edge_color: str = C_INPUT_EDGE,
                active_mask: Optional[np.ndarray] = None,
                empty_mask: Optional[np.ndarray] = None,
                match_mask: Optional[dict] = None,
                alpha: float = 0.82):
    """
    Crta cijeli 3D blok kocki za datu mapu.

    data        — (H, W) ili (H, W, C)
    origin      — (x0, y0, z0) polazna tačka u 3D prostoru
    active_mask — bool niz (H, W); True = zeleni highlight
    empty_mask  — bool niz (H, W); True = tamna "čeka" kocka
    match_mask  — dict {(r,c): 'positive'|'negative'} za pattern mod
    """
    if data.ndim == 2:
        data = data[:, :, np.newaxis]

    H, W, C = data.shape
    x0, y0, z0 = origin

    for ch in range(C):
        for r in range(H):
            for c in range(W):
                x = x0 + c
                # Redovi: r=0 je "gore" — u 3D prostoru veći Y ide "nazad"
                y = y0 + (H - 1 - r)
                z = z0 + ch

                # Odredi boje
                is_active = active_mask is not None and active_mask[r, c]
                is_empty  = empty_mask  is not None and empty_mask[r, c]
                match     = match_mask.get((r, c)) if match_mask else None

                if match == 'positive':
                    fc, ec = C_POSITIVE_FACE, C_POSITIVE_EDGE
                elif match == 'negative':
                    fc, ec = C_NEGATIVE_FACE, C_NEGATIVE_EDGE
                elif is_active:
                    fc, ec = C_ACTIVE_FACE, C_ACTIVE_EDGE
                elif is_empty:
                    fc, ec = C_EMPTY_FACE, C_EMPTY_EDGE
                    alpha  = 0.4
                else:
                    fc, ec = face_color, edge_color

                _draw_cell(ax, x, y, z, fc, ec, alpha=alpha)


def _style_ax(ax: Axes3D, title: str,
              xlim: tuple, ylim: tuple, zlim: tuple,
              elev: float = 22, azim: float = -52):
    """Stilizuje 3D axis — tamna pozadina, bez gridova, kut gledanja."""
    ax.set_facecolor(C_BG)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor(C_GRID_LINE)
    ax.grid(False)
    ax.set_title(title, color=C_TEXT_PRIMARY,
                 fontsize=8.5, pad=4, fontweight='bold')
    ax.view_init(elev=elev, azim=azim)


# ─── 2D panel helpers ────────────────────────────────────────────────────────

def _draw_2d_grid(ax,
                  data: np.ndarray,
                  x0: float, y0: float,
                  cell_w: float = 0.28, cell_h: float = 0.28,
                  face_color: str = C_INPUT_FACE,
                  edge_color: str = C_INPUT_EDGE,
                  highlight_rc: Optional[tuple] = None,
                  highlight_color: str = C_ACTIVE_EDGE,
                  fmt: str = "{}",
                  fontsize: float = 9.5):
    """
    Crta 2D grid u axes koordinatama (0..1 prostor).
    data        — 2D numpy niz
    x0, y0      — gornji lijevi ugao u axes koordinatama
    cell_w/h    — veličina ćelije
    highlight_rc — (r, c) ćelija koja se ističe drugom bojom ivice
    fmt         — format string za vrijednost, npr. "{:.1f}"
    """
    H, W = data.shape
    for r in range(H):
        for c in range(W):
            px = x0 + c * cell_w
            py = y0 + r * cell_h

            is_hl = highlight_rc == (r, c)
            ec = highlight_color if is_hl else edge_color
            lw = 1.5 if is_hl else 0.7

            rect = mpatches.FancyBboxPatch(
                (px, py), cell_w * 0.92, cell_h * 0.92,
                boxstyle="round,pad=0.01",
                facecolor=face_color,
                edgecolor=ec,
                linewidth=lw,
                transform=ax.transAxes,
                clip_on=False,
                zorder=3,
            )
            ax.add_patch(rect)

            val = data[r, c]
            if isinstance(val, (float, np.floating)) and np.isnan(val):
                text = "?"
                tc = C_TEXT_DIM
            else:
                text = fmt.format(val)
                tc = C_ACCENT if is_hl else C_TEXT_PRIMARY

            ax.text(px + cell_w * 0.46, py + cell_h * 0.46, text,
                    ha='center', va='center',
                    fontsize=fontsize, color=tc, fontweight='bold',
                    transform=ax.transAxes, zorder=4)


def _operator_text(ax, x: float, y: float, symbol: str, fontsize=16):
    """Crta operator simbol (×, =, +, Σ) u 2D panelu."""
    ax.text(x, y, symbol,
            ha='center', va='center',
            fontsize=fontsize, color=C_TEXT_DIM,
            fontweight='bold', transform=ax.transAxes, zorder=4)


def _draw_calc_panel(ax,
                     patch: np.ndarray,
                     kernel: np.ndarray,
                     products: np.ndarray,
                     conv_sum: float,
                     output_val: float,
                     out_row: int, out_col: int,
                     bias: int = 0,
                     pool_type: Optional[str] = None,
                     weights: Optional[np.ndarray] = None):
    """
    Crta donji 2D panel s računanjem trenutnog koraka.

    Layout (za 3×3):
      [patch] ×  [kernel/weights]  =  [products]   Σ=X  (+bias)  →  rezultat

    Za pooling: umjesto kernel-a prikazujemo prozor s bojama za max/avg/l2,
    ili težine za weighted avg.
    """
    ax.set_facecolor(C_PANEL_BG)
    ax.axis('off')

    H, W = patch.shape
    cw = ch_ = min(0.072, 0.72 / max(W, H))  # veličina ćelije, skalira sa grijom

    # Pozicije blokova (u axes koordinatama 0..1)
    margin_top = 0.08
    patch_x0   = 0.02
    kern_x0    = patch_x0 + W * cw + 0.10   # operator × + razmak
    prod_x0    = kern_x0  + W * cw + 0.10
    sigma_x    = prod_x0  + W * cw + 0.06
    result_x   = sigma_x  + 0.12

    # Labele iznad blokova
    lbl_y = 1.0 - margin_top + 0.01
    for lbl, lx in [("patch iz ulaza", patch_x0 + W*cw/2),
                    ("filter" if pool_type is None else
                     ("težine" if pool_type == "weighted" else "prozor"),
                     kern_x0 + W*cw/2),
                    ("element×element", prod_x0 + W*cw/2)]:
        ax.text(lx, lbl_y, lbl,
                ha='center', va='bottom', fontsize=7,
                color=C_TEXT_DIM, transform=ax.transAxes)

    grid_y0 = 1.0 - margin_top - H * ch_ - 0.02   # gornji lijevi ugao gridova

    # Patch
    _draw_2d_grid(ax, patch, patch_x0, grid_y0,
                  cell_w=cw, cell_h=ch_,
                  face_color=C_INPUT_FACE, edge_color=C_INPUT_EDGE)

    # Operator ×
    _operator_text(ax, kern_x0 - 0.055, grid_y0 + H*ch_/2, "×")

    # Kernel ili težine
    if pool_type == "weighted" and weights is not None:
        kern_display = weights.astype(float)
        kfc, kec = "#2a1a3a", "#AA00FF"
        kfmt = "{:.0f}"
    elif pool_type is not None:
        # Za max/avg/l2 prikazujemo patch i tip operacije
        kern_display = patch
        kfc, kec = C_FILTER_FACE, C_FILTER_EDGE
        kfmt = "{}"
    else:
        kern_display = kernel
        kfc, kec = C_FILTER_FACE, C_FILTER_EDGE
        kfmt = "{}"

    _draw_2d_grid(ax, kern_display, kern_x0, grid_y0,
                  cell_w=cw, cell_h=ch_,
                  face_color=kfc, edge_color=kec, fmt=kfmt)

    # Operator =
    _operator_text(ax, prod_x0 - 0.055, grid_y0 + H*ch_/2, "=")

    # Produkti (ili skip za pooling)
    if pool_type is None:
        _draw_2d_grid(ax, products, prod_x0, grid_y0,
                      cell_w=cw, cell_h=ch_,
                      face_color="#111a11", edge_color="#2a3a2a",
                      fmt="{}")

    # Σ i rezultat
    sigma_y = grid_y0 + H*ch_/2

    if pool_type is None:
        # Konvolucija: Σ + opcioni bias
        ax.text(sigma_x, sigma_y + 0.08, "Σ =",
                ha='left', va='center', fontsize=8,
                color=C_TEXT_DIM, transform=ax.transAxes)
        ax.text(sigma_x + 0.01, sigma_y - 0.02,
                f"{int(conv_sum)}",
                ha='left', va='center', fontsize=14,
                color=C_TEXT_PRIMARY, fontweight='bold',
                transform=ax.transAxes)
        if bias != 0:
            ax.text(sigma_x, sigma_y - 0.12,
                    f"+ bias({bias})",
                    ha='left', va='center', fontsize=7.5,
                    color=C_TEXT_DIM, transform=ax.transAxes)

        # Linija i rezultat
        ax.plot([result_x - 0.01, result_x + 0.08], [sigma_y - 0.16]*2,
                color=C_GRID_LINE, linewidth=0.8,
                transform=ax.transAxes)
        ax.text(result_x + 0.035, sigma_y - 0.26,
                f"= {int(output_val)}",
                ha='center', va='center', fontsize=14,
                color=C_OUTPUT_ACTIVE, fontweight='bold',
                transform=ax.transAxes)
        ax.text(result_x + 0.035, sigma_y - 0.38,
                f"izlaz[{out_row},{out_col}]",
                ha='center', va='center', fontsize=7.5,
                color=C_TEXT_DIM, transform=ax.transAxes)

    else:
        # Pooling: formula tekst
        vals = patch.flatten()
        if pool_type == "max":
            op_str = f"max({', '.join(map(str, vals))}) = {output_val:.2f}"
        elif pool_type == "avg":
            op_str = f"avg({', '.join(map(str, vals))}) = {output_val:.2f}"
        elif pool_type == "l2":
            op_str = f"L2 = √Σx² = {output_val:.2f}"
        else:
            op_str = f"Σ(w·x)/Σw = {output_val:.2f}"

        ax.text(sigma_x, sigma_y + 0.05, op_str,
                ha='left', va='center', fontsize=8.5,
                color=C_TEXT_PRIMARY, fontweight='bold',
                transform=ax.transAxes)
        ax.text(sigma_x, sigma_y - 0.12,
                f"→  izlaz[{out_row},{out_col}]",
                ha='left', va='center', fontsize=8,
                color=C_OUTPUT_ACTIVE,
                transform=ax.transAxes)


# ─── Konvolucija ─────────────────────────────────────────────────────────────

def render_conv_step(step, config, fig=None):
    """
    Crta jedan korak konvolucije.

    step   — ConvStep dataclass (vidi core/convolution.py)
    config — ConvConfig dataclass
    fig    — postojeći Figure (ako None, kreira novi)

    Layout:
      [3D ulaz] [3D filter] [3D izlaz]   ← gornji red (visina 3)
      [     2D računanje panel         ]  ← donji red (visina 2)
    """
    if fig is None:
        fig = plt.figure(figsize=(14, 7), facecolor=C_BG)
    else:
        fig.clear()
        fig.patch.set_facecolor(C_BG)

    gs = GridSpec(2, 3, figure=fig,
                  height_ratios=[3, 2],
                  hspace=0.35, wspace=0.15,
                  left=0.02, right=0.98,
                  top=0.93, bottom=0.04)

    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    ax2 = fig.add_subplot(gs[0, 1], projection='3d')
    ax3 = fig.add_subplot(gs[0, 2], projection='3d')
    ax_calc = fig.add_subplot(gs[1, :])

    H, W, C = step.input_padded.shape
    k = config.kernel_size
    out_size = step.output_map.shape[0]

    # Active mask — region koji filter trenutno pokriva
    active = np.zeros((H, W), dtype=bool)
    for dr in range(k):
        for dc in range(k):
            active[step.filter_row + dr, step.filter_col + dc] = True

    # Empty mask — izlazne ćelije koje još nisu popunjene
    empty = np.ones((out_size, out_size), dtype=bool)
    for idx in range(step.step_idx + 1):
        ri, ci = divmod(idx, out_size)
        if ri < out_size:
            empty[ri, ci] = False

    # ── 3D blokovi ────────────────────────────────────────────────────
    _draw_block(ax1, step.input_padded,
                face_color=C_INPUT_FACE, edge_color=C_INPUT_EDGE,
                active_mask=active)
    pad_str = " (padded)" if config.padding else ""
    _style_ax(ax1, f"Ulaz {config.input_size}×{config.input_size}×{C}{pad_str}",
              xlim=(-0.5, W+0.5), ylim=(-0.5, W+0.5), zlim=(-0.2, C+0.8))

    _draw_block(ax2, step.kernel,
                face_color=C_FILTER_FACE, edge_color=C_FILTER_EDGE)
    _style_ax(ax2, f"Filter {k}×{k}×{C}",
              xlim=(-0.5, k+0.5), ylim=(-0.5, k+0.5), zlim=(-0.2, C+0.8))

    # Izlaz — active je trenutna ćelija koja se popunjava
    out_active = np.zeros((out_size, out_size), dtype=bool)
    out_active[step.out_row, step.out_col] = True

    # Izlazna mapa: NaN za nepopunjene
    out_display = step.output_map.astype(float).copy()
    out_display[empty] = np.nan

    _draw_block(ax3, out_display,
                face_color=C_OUTPUT_FACE, edge_color=C_OUTPUT_EDGE,
                active_mask=out_active, empty_mask=empty)
    _style_ax(ax3, f"Izlaz {out_size}×{out_size}",
              xlim=(-0.5, out_size+0.5), ylim=(-0.5, out_size+0.5),
              zlim=(-0.2, 1.8))

    # ── 2D panel ──────────────────────────────────────────────────────
    # patch: isječak ulaza koji se trenutno množi s filterom
    patch_2d = step.input_padded[
        step.filter_row : step.filter_row + k,
        step.filter_col : step.filter_col + k,
        0              # kanal 0 za prikaz (višekanalni se sumiraju)
    ]
    kernel_2d = step.kernel[:, :, 0]

    # Produkti samo za kanal 0 (suma po svim kanalima je conv_sum)
    products_2d = (patch_2d * kernel_2d).astype(int)

    _draw_calc_panel(ax_calc,
                     patch=patch_2d,
                     kernel=kernel_2d,
                     products=products_2d,
                     conv_sum=step.output_value - step.bias,
                     output_val=step.output_value,
                     out_row=step.out_row,
                     out_col=step.out_col,
                     bias=step.bias)

    # ── Naslov ────────────────────────────────────────────────────────
    bias_str = f" + bias({step.bias})" if step.bias else ""
    fig.suptitle(
        f"Konvolucija  │  Korak {step.step_idx + 1}/{step.total_steps}"
        f"  │  Σ(patch × filter){bias_str} = {step.output_value:.0f}"
        f"  →  izlaz[{step.out_row},{step.out_col}]",
        color=C_ACCENT, fontsize=10, fontweight='bold', y=0.985
    )
    return fig


# ─── Pooling ─────────────────────────────────────────────────────────────────

def render_pool_step(step, config, fig=None):
    """
    Crta jedan korak pooling-a.

    step   — PoolStep dataclass
    config — PoolConfig dataclass
    """
    if fig is None:
        fig = plt.figure(figsize=(14, 7), facecolor=C_BG)
    else:
        fig.clear()
        fig.patch.set_facecolor(C_BG)

    gs = GridSpec(2, 3, figure=fig,
                  height_ratios=[3, 2],
                  hspace=0.35, wspace=0.15,
                  left=0.02, right=0.98,
                  top=0.93, bottom=0.04)

    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    ax2 = fig.add_subplot(gs[0, 1], projection='3d')
    ax3 = fig.add_subplot(gs[0, 2], projection='3d')
    ax_calc = fig.add_subplot(gs[1, :])

    H = config.input_size
    k = config.kernel_size
    out_size = config.output_size()

    # Active mask na ulazu
    active = np.zeros((H, H), dtype=bool)
    for dr in range(k):
        for dc in range(k):
            active[step.filter_row + dr, step.filter_col + dc] = True

    # Empty mask na izlazu
    empty = np.ones((out_size, out_size), dtype=bool)
    for idx in range(step.step_idx + 1):
        ri, ci = divmod(idx, out_size)
        if ri < out_size:
            empty[ri, ci] = False

    # ── 3D blokovi ────────────────────────────────────────────────────
    _draw_block(ax1, step.input_map,
                face_color=C_INPUT_FACE, edge_color=C_INPUT_EDGE,
                active_mask=active)
    _style_ax(ax1, f"Ulaz {H}×{H}",
              xlim=(-0.5, H+0.5), ylim=(-0.5, H+0.5), zlim=(-0.2, 1.8))

    # Srednji subplot: težine (weighted) ili prozor (ostali)
    if step.pool_type.value == "Weighted Avg":
        w_disp = step.weights if step.weights.ndim == 3 \
                 else step.weights[:, :, np.newaxis]
        _draw_block(ax2, w_disp, face_color="#2a1a3a", edge_color="#AA00FF")
        _style_ax(ax2, f"Težine {k}×{k}",
                  xlim=(-0.5, k+0.5), ylim=(-0.5, k+0.5), zlim=(-0.2, 1.8))
    else:
        _draw_block(ax2, step.patch,
                    face_color=C_FILTER_FACE, edge_color=C_FILTER_EDGE)
        _style_ax(ax2, f"Prozor {k}×{k}  [{step.pool_type.value}]",
                  xlim=(-0.5, k+0.5), ylim=(-0.5, k+0.5), zlim=(-0.2, 1.8))

    # Izlaz
    out_active = np.zeros((out_size, out_size), dtype=bool)
    out_active[step.out_row, step.out_col] = True

    out_display = step.output_map.astype(float).copy()
    out_display[empty] = np.nan

    _draw_block(ax3, out_display,
                face_color=C_OUTPUT_FACE, edge_color=C_OUTPUT_EDGE,
                active_mask=out_active, empty_mask=empty)
    _style_ax(ax3, f"Izlaz {out_size}×{out_size}",
              xlim=(-0.5, out_size+0.5), ylim=(-0.5, out_size+0.5),
              zlim=(-0.2, 1.8))

    # ── 2D panel ──────────────────────────────────────────────────────
    patch_2d  = step.patch[:, :, 0] if step.patch.ndim == 3 else step.patch
    pt        = step.pool_type.value.lower().replace(" ", "_")
    pool_key  = ("max" if "max" in pt
                 else "avg" if "avg" in pt and "weight" not in pt
                 else "l2" if "l2" in pt
                 else "weighted")

    weights_2d = None
    if pool_key == "weighted":
        weights_2d = step.weights if step.weights.ndim == 2 \
                     else step.weights[:, :, 0]

    _draw_calc_panel(ax_calc,
                     patch=patch_2d,
                     kernel=None,
                     products=None,
                     conv_sum=step.output_value,
                     output_val=step.output_value,
                     out_row=step.out_row,
                     out_col=step.out_col,
                     pool_type=pool_key,
                     weights=weights_2d)

    fig.suptitle(
        f"{step.pool_type.value}  │  Korak {step.step_idx + 1}/{step.total_steps}"
        f"  │  {step.formula_str}",
        color=C_ACCENT, fontsize=10, fontweight='bold', y=0.985
    )
    return fig


# ─── Pattern detekcija ───────────────────────────────────────────────────────

def render_pattern_step(step, fig=None):
    """
    Crta korak detekcije obrazaca.

    Layout:
      [3D ulaz 12×12] [3D filter] [3D izlaz 10×10] [2D heatmap izlaza]
      [            2D panel za računanje                               ]
    """
    if fig is None:
        fig = plt.figure(figsize=(16, 7), facecolor=C_BG)
    else:
        fig.clear()
        fig.patch.set_facecolor(C_BG)

    gs = GridSpec(2, 4, figure=fig,
                  height_ratios=[3, 2],
                  hspace=0.35, wspace=0.2,
                  left=0.02, right=0.98,
                  top=0.93, bottom=0.04)

    ax_inp  = fig.add_subplot(gs[0, 0], projection='3d')
    ax_flt  = fig.add_subplot(gs[0, 1], projection='3d')
    ax_out  = fig.add_subplot(gs[0, 2], projection='3d')
    ax_heat = fig.add_subplot(gs[0, 3])
    ax_calc = fig.add_subplot(gs[1, :])

    fi = step.filter_idx
    k  = 3
    out_size = 10

    # Match mask za ulaz — sve pozitivne i negativne pozicije
    from core.pattern import POSITIVE_POSITIONS, NEGATIVE_POSITIONS
    match_mask = {}
    for ffi in range(3):
        pr, pc = POSITIVE_POSITIONS[ffi]
        nr, nc = NEGATIVE_POSITIONS[ffi]
        match_mask[(pr, pc)] = 'positive'
        match_mask[(nr, nc)] = 'negative'

    # Active mask — trenutni region
    active_inp = np.zeros((12, 12), dtype=bool)
    for dr in range(k):
        for dc in range(k):
            active_inp[step.filter_row + dr, step.filter_col + dc] = True

    # ── 3D ulaz ───────────────────────────────────────────────────────
    _draw_block(ax_inp, step.input_map,
                face_color=C_INPUT_FACE, edge_color=C_INPUT_EDGE,
                active_mask=active_inp, match_mask=match_mask)
    _style_ax(ax_inp, "Ulaz 12×12",
              xlim=(-0.5, 12.5), ylim=(-0.5, 12.5), zlim=(-0.2, 1.8),
              elev=28, azim=-48)

    # ── 3D filter ─────────────────────────────────────────────────────
    filter_edge_colors = [C_FILTER_EDGE, "#FF9800", "#E040FB"]
    _draw_block(ax_flt, step.kernel,
                face_color=C_FILTER_FACE,
                edge_color=filter_edge_colors[fi])
    _style_ax(ax_flt, f"Filter {fi+1} (3×3)",
              xlim=(-0.5, k+0.5), ylim=(-0.5, k+0.5), zlim=(-0.2, 1.8))

    # ── 3D izlaz ──────────────────────────────────────────────────────
    empty_out = np.ones((out_size, out_size), dtype=bool)
    for idx in range(step.step_idx + 1):
        ri, ci = divmod(idx, out_size)
        if ri < out_size:
            empty_out[ri, ci] = False

    out_active = np.zeros((out_size, out_size), dtype=bool)
    out_active[step.out_row, step.out_col] = True

    out_display = step.all_output_maps[fi].copy()
    out_display[empty_out] = np.nan

    match_out = {}
    if step.match_type == 'positive':
        match_out[(step.out_row, step.out_col)] = 'positive'
    elif step.match_type == 'negative':
        match_out[(step.out_row, step.out_col)] = 'negative'

    _draw_block(ax_out, out_display,
                face_color=C_OUTPUT_FACE,
                edge_color=filter_edge_colors[fi],
                active_mask=out_active,
                empty_mask=empty_out,
                match_mask=match_out)
    _style_ax(ax_out, f"Izlaz filtera {fi+1} (10×10)",
              xlim=(-0.5, out_size+0.5), ylim=(-0.5, out_size+0.5),
              zlim=(-0.2, 1.8))

    # ── Heatmap ───────────────────────────────────────────────────────
    ax_heat.set_facecolor(C_BG)
    valid_out = step.all_output_maps[fi].copy()
    valid_out[np.isnan(valid_out)] = 0
    im = ax_heat.imshow(valid_out, cmap='RdYlGn',
                        aspect='auto', vmin=-30, vmax=30)
    ax_heat.set_title(f"Heatmap izlaza {fi+1}\nmax={np.nanmax(step.all_output_maps[fi]):.0f}",
                      color=C_TEXT_PRIMARY, fontsize=8)
    ax_heat.tick_params(colors=C_TEXT_DIM, labelsize=6)
    for spine in ax_heat.spines.values():
        spine.set_edgecolor(C_GRID_LINE)

    # Označi pozitivnu i negativnu poziciju na heatmapu
    pr, pc = POSITIVE_POSITIONS[fi]
    nr, nc = NEGATIVE_POSITIONS[fi]
    for (mr, mc), ec in [(pr, pc, C_POSITIVE_EDGE), (nr, nc, C_NEGATIVE_EDGE)]:
        ax_heat.add_patch(plt.Rectangle(
            (mc - 0.5, mr - 0.5), 1, 1,
            linewidth=1.5, edgecolor=ec, facecolor='none'
        ))

    # ── 2D panel ──────────────────────────────────────────────────────
    patch_2d  = step.input_map[
        step.filter_row : step.filter_row + k,
        step.filter_col : step.filter_col + k
    ]
    kernel_2d = step.kernel[:, :, 0] if step.kernel.ndim == 3 else step.kernel
    products  = (patch_2d * kernel_2d).astype(float)

    _draw_calc_panel(ax_calc,
                     patch=patch_2d,
                     kernel=kernel_2d,
                     products=products,
                     conv_sum=step.output_value,
                     output_val=step.output_value,
                     out_row=step.out_row,
                     out_col=step.out_col,
                     bias=0)

    # Naslov s informacijom o poklapanju
    match_labels = {
        'positive': f"  ✓  MAKSIMUM — savršeno poklapanje s filterom {fi+1}!",
        'negative': f"  ✗  MINIMUM — negativno poklapanje s filterom {fi+1}!",
        'neutral':  "",
    }
    match_colors = {
        'positive': C_POSITIVE_EDGE,
        'negative': C_NEGATIVE_EDGE,
        'neutral':  C_ACCENT,
    }
    mtype = getattr(step, 'match_type', 'neutral') or 'neutral'

    fig.suptitle(
        f"Detekcija obrazaca  │  Filter {fi+1}/3"
        f"  │  Korak {step.step_idx+1}/{step.total_steps}"
        f"  │  Σ = {step.output_value:.0f}"
        f"{match_labels.get(mtype, '')}",
        color=match_colors.get(mtype, C_ACCENT),
        fontsize=9.5, fontweight='bold', y=0.985
    )
    return fig