import numpy as np
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.figure import Figure

# ------------------------------------------------------------------
# Boje (RGBA)
# ------------------------------------------------------------------
COLOR_INPUT = (0.20, 0.60, 0.86, 0.35)  # plava, providna
COLOR_FILTER = (0.78, 0.36, 0.05, 0.75) # narandžasta
COLOR_ACTIVE = (0.90, 0.20, 0.65, 0.70)# zuta — aktivan region
COLOR_OUTPUT = (0.60, 0.20, 0.80, 0.60)  # ljubičasta
COLOR_POSITIVE = (0.10, 0.90, 0.10, 0.80)  # jarko zelena
COLOR_NEGATIVE = (0.90, 0.10, 0.10, 0.80)  # crvena
COLOR_EDGE = (0.0, 0.0, 0.0, 0.8)  # crna ivica


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
        ax,
        data: np.ndarray,
        origin: tuple,
        cell_size: float = 1.0,
        gap: float = 0.08,
        channel_gap: float = 1.5,  # NOVO: Ogroman razmak između kanala!
        face_color=COLOR_INPUT,
        highlight_mask: np.ndarray = None,
        highlight_color=COLOR_ACTIVE,
        color_matrix: np.ndarray = None,
        show_values: bool = True,
        values_mask: np.ndarray = None,
        value_fmt: str = "{}",
):
    if data.ndim == 2:
        data = data[:, :, np.newaxis]

    rows, cols, channels = data.shape
    x0, y0, z0 = origin
    s = cell_size - gap

    all_faces = []
    all_colors = []

    for c in range(channels):
        for r in range(rows):
            for col in range(cols):
                val = data[r, col, c]
                if np.isnan(val):
                    continue

                cx = x0 + col * cell_size
                # NOVO: Dodajemo channel_gap da fizički razdvojimo kanale po Y osi
                cy = y0 + c * (cell_size + channel_gap)
                cz = z0 + (rows - 1 - r) * cell_size

                faces = _cube_faces(cx, cy, cz, s)
                all_faces.extend(faces)

                if color_matrix is not None:
                    color = color_matrix[r, col]
                elif highlight_mask is not None and highlight_mask[r, col]:
                    color = highlight_color
                else:
                    color = face_color

                all_colors.extend([color] * 6)

    if not all_faces:
        return

    poly = Poly3DCollection(
        all_faces, facecolors=all_colors, edgecolors=COLOR_EDGE, linewidths=0.3
    )
    ax.add_collection3d(poly)

    if show_values:
        for c in range(channels):
            for r in range(rows):
                for col in range(cols):
                    if values_mask is not None and not values_mask[r, col]:
                        continue

                    val = data[r, col, c]
                    if np.isnan(val):
                        continue

                    cx = x0 + col * cell_size + s / 2
                    cy = y0 + c * (cell_size + channel_gap) + s / 2
                    cz = z0 + (rows - 1 - r) * cell_size + s / 2
                    ax.text(
                        cx, cy, cz, value_fmt.format(val),
                        ha='center', va='center', fontsize=7,
                        fontweight='bold', color='white', zorder=5,
                    )

def _set_ax_limits(ax, max_ext: float):
    # Pojednostavljeno: pravimo savršenu kocku prostora da se slika ne bi deformisala
    ax.set_xlim(0, max_ext)
    ax.set_ylim(0, max_ext)
    ax.set_zlim(0, max_ext)
    ax.set_axis_off()


def render_convolution(fig: Figure, engine, step: dict) -> None:
    fig.clear()
    axes = [fig.add_subplot(131, projection='3d'), fig.add_subplot(132, projection='3d'),
            fig.add_subplot(133, projection='3d')]

    padded, filt, out = engine.padded_input, engine.filter_weights, engine.output_map
    rows_p, cols_p, C = padded.shape
    F = engine.filter_size

    mask = np.zeros((rows_p, cols_p), dtype=bool)
    mask[step["in_row"]: step["in_row"] + F, step["in_col"]: step["in_col"] + F] = True

    # Ulaz
    _draw_block(axes[0], padded, origin=(0, 0, 0), highlight_mask=mask, values_mask=mask, channel_gap=1.5)
    axes[0].set_title(f"ULAZNA MAPA {'(Padded)' if engine.padding else ''}", fontsize=11, fontweight='bold', pad=15)

    # Filter
    _draw_block(axes[1], filt, origin=(0, 0, 0), face_color=COLOR_FILTER, value_fmt="{:.0f}", channel_gap=1.5)

    # NOVO: Generisanje pametne formule koja prikazuje sume po kanalima
    if C > 1:
        ch_parts = " + ".join([f"{s}(K{i + 1})" for i, s in enumerate(step['ch_sums'])])
        formula = f"Σ = {ch_parts} = {step['conv_sum']}"
    else:
        formula = f"Σ(ulaz × filter) = {step['conv_sum']}"

    if engine.bias:
        formula += f"\nUkupno: {step['conv_sum']} + bias({engine.bias}) = {step['output_val']}"

    axes[1].set_title(f"FILTER\n{formula}", fontsize=11, fontweight='bold', pad=15)

    # Izlaz
    _draw_block(axes[2], out, origin=(0, 0, 0), face_color=COLOR_OUTPUT, value_fmt="{:.0f}", channel_gap=1.5)
    axes[2].set_title(f"IZLAZNA MAPA\nNova vrednost: {step['output_val']}", fontsize=11, fontweight='bold', pad=15)

    # Računanje dimenzija za prostor (uzimajući u obzir razmak kanala)
    y_extent = C * 1.0 + (C - 1) * 1.5
    max_ext = max(rows_p, cols_p, F, engine.output_size, y_extent)

    for ax in axes:
        _set_ax_limits(ax, max_ext)
        ax.view_init(elev=20, azim=-55)  # Blago promenjen ugao za bolji pogled na razmak

    patches = [
        mpatches.Patch(color=COLOR_INPUT[:3], label='Ulaz'),
        mpatches.Patch(color=COLOR_ACTIVE[:3], label='Aktivan region'),
        mpatches.Patch(color=COLOR_FILTER[:3], label='Filter'),
        mpatches.Patch(color=COLOR_OUTPUT[:3], label='Izlaz')
    ]
    fig.legend(handles=patches, loc='lower center', ncol=4, fontsize=9)
    fig.tight_layout(rect=[0, 0.05, 1, 1])

def render_pooling(fig: Figure, engine, step: dict) -> None:
    fig.clear()
    has_weights = engine.pool_type == "weighted"
    axes = [fig.add_subplot(131 if has_weights else 121, projection='3d'),
            fig.add_subplot(132 if has_weights else 122, projection='3d')]
    if has_weights: axes.append(fig.add_subplot(133, projection='3d'))

    inp, H, F = engine.input_map, engine.input_size, engine.filter_size
    mask = np.zeros((H, H), dtype=bool)
    mask[step["in_row"]: step["in_row"] + F, step["in_col"]: step["in_col"] + F] = True

    _draw_block(axes[0], inp, origin=(0, 0, 0), highlight_mask=mask)
    axes[0].set_title(f"Ulaz\n{engine.pool_type.upper()} pooling", fontsize=8, pad=4)

    ax_idx = 1
    if has_weights:
        _draw_block(axes[1], engine.weights, origin=(0, 0, 0), face_color=COLOR_FILTER, value_fmt="{:.0f}")
        axes[1].set_title("Težine", fontsize=9, pad=4)
        ax_idx = 2

    out_ch0 = engine.output_map[:, :, 0]
    _draw_block(axes[ax_idx], out_ch0, origin=(0, 0, 0), face_color=COLOR_OUTPUT, value_fmt="{:.1f}")
    axes[ax_idx].set_title(f"Izlaz [{step['out_row']},{step['out_col']}]", fontsize=9, pad=4)

    for ax in axes:
        _set_ax_limits(ax, max(H, F, engine.output_size))
        ax.view_init(elev=25, azim=-50)
    fig.tight_layout()


def render_pattern(fig: Figure, engine, step: dict, filter_idx: int) -> None:
    fig.clear()
    axes = [fig.add_subplot(131, projection='3d'), fig.add_subplot(132, projection='3d'),
            fig.add_subplot(133, projection='3d')]

    inp, F = engine.input_map, engine.FILTER_SIZE
    r0, c0 = step["filter_row"], step["filter_col"]

    # NOVO: Efikasno formiranje boja bez preklapanja (_draw_block se poziva samo jednom za ulaz)
    color_matrix = np.empty((engine.MAP_SIZE, engine.MAP_SIZE, 4))
    color_matrix[:] = COLOR_INPUT

    for sr in engine.special_regions:
        if sr["filter_idx"] == filter_idx:
            sr_r, sr_c = sr["row"], sr["col"]
            color_matrix[sr_r:sr_r + F, sr_c:sr_c + F] = COLOR_POSITIVE if sr["type"] == "positive" else COLOR_NEGATIVE

    color_matrix[r0:r0 + F, c0:c0 + F] = COLOR_ACTIVE

    _draw_block(axes[0], inp, origin=(0, 0, 0), color_matrix=color_matrix, show_values=False)

    label = " ← MAX!" if step["match_type"] == "positive" else (" ← MIN!" if step["match_type"] == "negative" else "")
    axes[0].set_title(f"Ulazna mapa 12×12\nFilter {filter_idx + 1}", fontsize=8)

    _draw_block(axes[1], engine.filters[filter_idx], origin=(0, 0, 0), face_color=COLOR_FILTER)
    axes[1].set_title(f"Filter {filter_idx + 1}\nΣ = {step['output_value']:.1f}{label}", fontsize=8, pad=4)

    _draw_block(axes[2], engine.output_maps[filter_idx], origin=(0, 0, 0), face_color=COLOR_OUTPUT, value_fmt="{:.0f}")
    axes[2].set_title(f"Izlaz filtera {filter_idx + 1}", fontsize=8)

    for ax, max_d in zip(axes, [engine.MAP_SIZE, F, engine.output_size]):
        _set_ax_limits(ax, max_d)
        ax.view_init(elev=25, azim=-50)

    patches = [
        mpatches.Patch(color=COLOR_INPUT[:3], label='Ulaz'),
        mpatches.Patch(color=COLOR_ACTIVE[:3], label='Aktivan region'),
        mpatches.Patch(color=COLOR_POSITIVE[:3], label='Pozitivno poklapanje'),
        mpatches.Patch(color=COLOR_NEGATIVE[:3], label='Negativno poklapanje'),
        mpatches.Patch(color=COLOR_OUTPUT[:3], label='Izlaz')
    ]
    fig.legend(handles=patches, loc='lower center', ncol=5, fontsize=7)
    fig.tight_layout(rect=[0, 0.05, 1, 1])