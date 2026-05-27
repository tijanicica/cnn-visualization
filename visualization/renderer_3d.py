# visualization/renderer_3d.py
import numpy as np
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.figure import Figure

# ------------------------------------------------------------------
# Boje (RGBA)
# ------------------------------------------------------------------
COLOR_INPUT = (0.20, 0.60, 0.86, 0.40)  # plava, providna
COLOR_FILTER = (0.95, 0.50, 0.10, 0.60)  # narandžasta
COLOR_ACTIVE = (0.90, 0.10, 0.50, 0.70)  # roze/magenta (podrazumevana)
COLOR_OUTPUT = (0.60, 0.20, 0.80, 0.60)  # ljubičasta
COLOR_POSITIVE = (0.00, 0.80, 0.00, 0.85)  # zelena - DETEKCIJA
COLOR_NEGATIVE = (0.80, 0.00, 0.00, 0.85)  # crvena - NEGATIVNA DETEKCIJA
COLOR_EDGE = (0.0, 0.0, 0.0, 0.8)  # crna ivica

FORMULA_BBOX = dict(boxstyle="round,pad=1.2", fc="#2c313c", ec="white", lw=2, alpha=0.95)


def _cube_faces(x0: float, y0: float, z0: float, s: float) -> list:
    x1, y1, z1 = x0 + s, y0 + s, z0 + s
    return [
        [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)],
        [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],
        [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],
        [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)],
        [(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)],
        [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)],
    ]


def _draw_block(
        ax, data: np.ndarray, origin: tuple, cell_size: float = 1.0, gap: float = 0.08, channel_gap: float = 1.5,
        face_color=COLOR_INPUT, highlight_mask: np.ndarray = None, highlight_color=COLOR_ACTIVE,
        color_matrix: np.ndarray = None, show_values: bool = True, values_mask: np.ndarray = None,
        value_fmt: str = "{}", font_size=13  # <--- DODATO font_size sa default 13
):
    if data.ndim == 2:
        data = data[:, :, np.newaxis]

    rows, cols, channels = data.shape
    x0, y0, z0 = origin
    s = cell_size - gap
    all_faces, all_colors = [], []

    for c in range(channels):
        for r in range(rows):
            for col in range(cols):
                val = data[r, col, c]
                if np.isnan(val): continue

                cx = x0 + col * cell_size
                cy = y0 + c * (cell_size + channel_gap)
                cz = z0 + (rows - 1 - r) * cell_size
                all_faces.extend(_cube_faces(cx, cy, cz, s))

                if color_matrix is not None:
                    color = color_matrix[r, col]
                elif highlight_mask is not None and highlight_mask[r, col]:
                    color = highlight_color
                else:
                    color = face_color
                all_colors.extend([color] * 6)

    if not all_faces: return

    poly = Poly3DCollection(all_faces, facecolors=all_colors, edgecolors=COLOR_EDGE, linewidths=0.3)
    ax.add_collection3d(poly)

    if show_values:
        for c in range(channels):
            for r in range(rows):
                for col in range(cols):
                    if values_mask is not None and not values_mask[r, col]: continue
                    val = data[r, col, c]
                    if np.isnan(val): continue
                    cx = x0 + col * cell_size + s / 2
                    cy = y0 + c * (cell_size + channel_gap) + s / 2
                    cz = z0 + (rows - 1 - r) * cell_size + s / 2
                    ax.text(cx, cy, cz, value_fmt.format(val), ha='center', va='center',
                            fontsize=font_size, fontweight='bold', color='white', zorder=5)


def _set_ax_limits(ax, max_ext: float):
    ax.set_xlim(0, max_ext)
    ax.set_ylim(0, max_ext)
    ax.set_zlim(0, max_ext)
    ax.set_axis_off()
    ax.set_facecolor('white')


def _add_legend(fig, is_pattern=False):
    patches = [
        mpatches.Patch(color=COLOR_INPUT[:3], label='Ulaz'),
        mpatches.Patch(color=COLOR_ACTIVE[:3], label='Aktivan region'),
        mpatches.Patch(color=COLOR_FILTER[:3], label='Filter'),
        mpatches.Patch(color=COLOR_OUTPUT[:3], label='Izlaz')
    ]
    if is_pattern:
        patches.append(mpatches.Patch(color=COLOR_POSITIVE[:3], label='Pozitivna Detekcija'))
        patches.append(mpatches.Patch(color=COLOR_NEGATIVE[:3], label='Negativna Detekcija'))

    legend = fig.legend(handles=patches, loc='lower center', ncol=len(patches), fontsize=10)
    if legend:
        frame = legend.get_frame()
        frame.set_facecolor('#2c313c')
        frame.set_edgecolor('#3e4451')
        for text in legend.get_texts(): text.set_color("white")


# ------------------------------------------------------------------
# POMOĆNA FUNKCIJA: RASPISUJE FORMULU HORIZONTALNO!
# ------------------------------------------------------------------
def _build_detailed_formula(region, filt, ch_sums, total_sum, bias, output_val):
    if region.ndim == 2:
        region = region[:, :, np.newaxis]
        filt = filt[:, :, np.newaxis]
    C = region.shape[2]
    lines = []
    for c in range(C):
        r_flat = region[:, :, c].flatten()
        f_flat = filt[:, :, c].flatten()
        pairs = [f"{r_flat[i]:.0f}×{f_flat[i]:.0f}" for i in range(len(r_flat))]
        ch_str = " + ".join(pairs)
        prefix = f"K{c + 1}: [ " if C > 1 else "Σ = [ "
        lines.append(f"{prefix}{ch_str} ] = {ch_sums[c]}")
    summary = ""
    if C > 1:
        summary += f"Ukupna suma: {' + '.join(str(x) for x in ch_sums)} = {total_sum}    |    "
    if bias != 0:
        summary += f"IZLAZ: {total_sum} + bias({bias}) = {output_val}"
    else:
        summary += f"IZLAZ: {output_val}"
    lines.append(summary)
    return "\n".join(lines)


# ==================================================================
# 1. KONVOLUCIJA
# ==================================================================
def render_convolution(fig: Figure, engine, step: dict) -> None:
    fig.clear()
    axes = [fig.add_subplot(131, projection='3d'), fig.add_subplot(132, projection='3d'),
            fig.add_subplot(133, projection='3d')]
    padded, filt, out = engine.padded_input, engine.filter_weights, engine.output_map
    rows_p, cols_p, C = padded.shape
    F = engine.filter_size
    mask = np.zeros((rows_p, cols_p), dtype=bool)
    mask[step["in_row"]: step["in_row"] + F, step["in_col"]: step["in_col"] + F] = True
    _draw_block(axes[0], padded, origin=(0, 0, 0), highlight_mask=mask, values_mask=mask, channel_gap=1.5)
    axes[0].set_title(f"ULAZNA MAPA {'(Padded)' if engine.padding else ''}", fontsize=11, fontweight='bold', pad=10,
                      color='white')
    _draw_block(axes[1], filt, origin=(0, 0, 0), face_color=COLOR_FILTER, value_fmt="{:.0f}", channel_gap=1.5)
    axes[1].set_title("FILTER", fontsize=11, fontweight='bold', pad=10, color='white')
    formula = _build_detailed_formula(step["region"], filt, step['ch_sums'], step['conv_sum'], engine.bias,
                                      step['output_val'])
    f_size = 16 if F <= 3 else 12
    fig.text(0.5, 0.98, formula, ha='center', va='top', color='white', fontsize=f_size, fontweight='bold',
             bbox=FORMULA_BBOX)
    _draw_block(axes[2], out, origin=(0, 0, 0), face_color=COLOR_OUTPUT, value_fmt="{:.0f}", channel_gap=1.5)
    axes[2].set_title(f"IZLAZNA MAPA\nNova vrednost: {step['output_val']}", fontsize=11, fontweight='bold', pad=10,
                      color='white')
    y_extent = C * 1.0 + (C - 1) * 1.5
    max_ext = max(rows_p, cols_p, F, engine.output_size, y_extent)
    for ax in axes:
        _set_ax_limits(ax, max_ext)
        ax.view_init(elev=20, azim=-55)
    _add_legend(fig)
    fig.tight_layout(rect=[0, 0.05, 1, 0.78])


# ==================================================================
# 2. POOLING
# ==================================================================
# ==================================================================
# 2. POOLING - ISPRAVLJENO ZA MULTI-CHANNEL
# ==================================================================
def render_pooling(fig: Figure, engine, step: dict) -> None:
    fig.clear()
    has_weights = engine.pool_type == "weighted"
    # Ako ima težina (Weighted Avg), trebaju nam 3 subplota, inače 2
    axes = [fig.add_subplot(131 if has_weights else 121, projection='3d'),
            fig.add_subplot(132 if has_weights else 122, projection='3d')]
    if has_weights: axes.append(fig.add_subplot(133, projection='3d'))

    inp, H, F = engine.input_map, engine.input_size, engine.filter_size
    C = engine.channels  # Uzimamo broj kanala

    # Highlight maska za aktivni region
    mask = np.zeros((H, H), dtype=bool)
    mask[step["in_row"]: step["in_row"] + F, step["in_col"]: step["in_col"] + F] = True

    # 1. Ulaz (Crtamo sve kanale sa razmakom 1.5)
    _draw_block(axes[0], inp, origin=(0, 0, 0), highlight_mask=mask, values_mask=mask, channel_gap=1.5)
    axes[0].set_title(f"ULAZ\n{engine.pool_type.upper()} pooling", fontsize=11, fontweight='bold', pad=10,
                      color='white')

    # --- FORMULA ZA POOLING (Fokus na Kanal 1) ---
    vals = step["regions"][0].flatten()
    result = step["output_vals"][0]

    if engine.pool_type == "max":
        formula = f"K1: MAX( {', '.join(map(str, vals.astype(int)))} ) = {result:.1f}"
    elif engine.pool_type == "avg":
        formula = f"K1: AVG( {', '.join(map(str, vals.astype(int)))} ) = {result:.2f}"
    elif engine.pool_type == "l2":
        formula = f"K1: L2 NORM( {', '.join(map(str, vals.astype(int)))} ) = {result:.2f}"
    else:
        # Prikaz postupka za weighted average
        r_flat = step["regions"][0].flatten()
        w_flat = engine.weights.flatten()
        pairs = [f"{r_flat[i]:.0f}×{w_flat[i]:.0f}" for i in range(len(r_flat))]
        formula = f"K1 W_AVG: [ {' + '.join(pairs)} ] / Suma_težina({np.sum(engine.weights)}) = {result:.2f}"

    if C > 1:
        formula += f"\n(Isto se primenjuje na ostalih {C - 1} kanala)"

    fig.text(0.5, 0.98, formula, ha='center', va='top', color='white', fontsize=16, fontweight='bold',
             bbox=FORMULA_BBOX)

    ax_idx = 1
    # 2. Težine (samo ako je Weighted Avg)
    if has_weights:
        _draw_block(axes[1], engine.weights, origin=(0, 0, 0), face_color=COLOR_FILTER, value_fmt="{:.0f}")
        axes[1].set_title("TEŽINE", fontsize=11, fontweight='bold', pad=10, color='white')
        ax_idx = 2

    # 3. IZLAZ - PROMENA: Sada šaljemo CELU mapu (engine.output_map) umesto samo engine.output_map[:, :, 0]
    out_map = engine.output_map
    _draw_block(axes[ax_idx], out_map, origin=(0, 0, 0), face_color=COLOR_OUTPUT, value_fmt="{:.1f}", channel_gap=1.5)
    axes[ax_idx].set_title(f"IZLAZ\nNova vrednost (K1): {result:.1f}", fontsize=11, fontweight='bold', pad=10,
                           color='white')

    # Postavljanje granica osa (uzimajući u obzir razmak kanala)
    y_extent = C * 1.0 + (C - 1) * 1.5
    max_ext = max(H, F, engine.output_size, y_extent)

    for ax in axes:
        _set_ax_limits(ax, max_ext)
        ax.view_init(elev=20, azim=-55)

    _add_legend(fig)
    fig.tight_layout(rect=[0, 0.05, 1, 0.78])


# ==================================================================
# 3. PATTERN
# ==================================================================
def render_pattern(fig: Figure, engine, step: dict, filter_idx: int) -> None:
    fig.clear()
    axes = [fig.add_subplot(131, projection='3d'), fig.add_subplot(132, projection='3d'),
            fig.add_subplot(133, projection='3d')]
    inp, F = engine.input_map, engine.FILTER_SIZE
    r0, c0 = step["filter_row"], step["filter_col"]

    # --- 1. BOJE ZA ULAZNU MAPU ---
    active_color = COLOR_ACTIVE
    if step["match_type"] == "positive":
        active_color = COLOR_POSITIVE
    elif step["match_type"] == "negative":
        active_color = COLOR_NEGATIVE

    color_matrix_in = np.empty((engine.MAP_SIZE, engine.MAP_SIZE, 4))
    color_matrix_in[:] = COLOR_INPUT
    for sr in engine.special_regions:
        if sr["filter_idx"] == filter_idx:
            sr_r, sr_c = sr["row"], sr["col"]
            # Diskretna pozadina šablona na ulazu
            color_matrix_in[sr_r:sr_r + F, sr_c:sr_c + F] = (0, 0.5, 0, 0.25) if sr["type"] == "positive" else (
            0.5, 0, 0, 0.25)

    color_matrix_in[r0:r0 + F, c0:c0 + F] = active_color
    _draw_block(axes[0], inp, origin=(0, 0, 0), color_matrix=color_matrix_in,
                show_values=True, value_fmt="{:.0f}", font_size=8)
    axes[0].set_title(f"ULAZNA MAPA 12×12\nFilter {filter_idx + 1}", fontsize=11, fontweight='bold', pad=10,
                      color='white')

    # --- 2. FILTER ---
    _draw_block(axes[1], engine.filters[filter_idx], origin=(0, 0, 0), face_color=COLOR_FILTER, font_size=13)
    axes[1].set_title(f"FILTER {filter_idx + 1}", fontsize=11, fontweight='bold', pad=10, color='white')

    # --- 3. FORMULA ---
    formula = _build_detailed_formula(step["region"], engine.filters[filter_idx], [step['output_value']],
                                      step['output_value'], 0, step['output_value'])
    label = " ← DETEKTOVAN OBRAZAC!" if step["match_type"] == "positive" else (
        " ← NEGATIVAN OBRAZAC!" if step["match_type"] == "negative" else "")
    if label: formula += f"\n{label}"

    text_color = "white"
    if step["match_type"] == "positive":
        text_color = "#98c379"
    elif step["match_type"] == "negative":
        text_color = "#e06c75"
    fig.text(0.5, 0.98, formula, ha='center', va='top', color=text_color, fontsize=16, fontweight='bold',
             bbox=FORMULA_BBOX)

    # --- 4. IZLAZNA MAPA SA TRAJNIM BOJAMA ---
    out_map = engine.output_maps[filter_idx]
    out_rows, out_cols = out_map.shape

    # Kreiramo matricu boja za izlaz (podrazumevano ljubičasta)
    color_matrix_out = np.empty((out_rows, out_cols, 4))
    color_matrix_out[:] = COLOR_OUTPUT

    # Prolazimo kroz sve specijalne regione (detekcije)
    for sr in engine.special_regions:
        if sr["filter_idx"] == filter_idx:
            # U detekciji šablona, pošto je korak 1, pozicija šablona u ulazu (r,c)
            # je ista kao pozicija rezultata u izlazu (r,c)
            r, c = sr["row"], sr["col"]

            # Ako je ta kockica već izračunata (nije NaN), bojimo je trajno
            if not np.isnan(out_map[r, c]):
                color_matrix_out[r, c] = COLOR_POSITIVE if sr["type"] == "positive" else COLOR_NEGATIVE

    # Isticanje TRENUTNE kockice (da blinka dok se računa)
    mask = np.zeros(out_map.shape, dtype=bool)
    mask[step["out_row"], step["out_col"]] = True
    # Trenutna kockica je uvek intenzivna boja detekcije ili podrazumevana aktivna
    current_out_color = active_color

    _draw_block(axes[2], out_map, origin=(0, 0, 0), color_matrix=color_matrix_out,
                highlight_mask=mask, highlight_color=current_out_color,
                value_fmt="{:.0f}", font_size=13)
    axes[2].set_title(f"IZLAZ FILTERA {filter_idx + 1}", fontsize=11, fontweight='bold', pad=10, color='white')

    # Podešavanja osa
    for ax, max_d in zip(axes, [engine.MAP_SIZE, F, engine.output_size]):
        _set_ax_limits(ax, max_ext=max_d)
        ax.view_init(elev=20, azim=-55)

    _add_legend(fig, is_pattern=True)
    fig.tight_layout(rect=[0, 0.05, 1, 0.78])