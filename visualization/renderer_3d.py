# visualization/renderer_3d.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.figure import Figure


# ------------------------------------------------------------------
# Boje (RGBA) — koristimo ih konzistentno kroz cio renderer
# ------------------------------------------------------------------
COLOR_INPUT      = (0.20, 0.60, 0.86, 0.35)   # plava, providna
COLOR_FILTER     = (0.95, 0.50, 0.10, 0.50)   # narandžasta
COLOR_ACTIVE     = (0.10, 0.90, 0.50, 0.70)   # zelena — aktivan region
COLOR_OUTPUT     = (0.60, 0.20, 0.80, 0.60)   # ljubičasta
COLOR_POSITIVE   = (0.10, 0.90, 0.10, 0.80)   # jarko zelena — pozitivno poklapanje
COLOR_NEGATIVE   = (0.90, 0.10, 0.10, 0.80)   # crvena — negativno poklapanje
COLOR_EDGE       = (0.0,  0.0,  0.0,  0.8)    # crna ivica


def _cube_faces(x0: float, y0: float, z0: float, s: float) -> list:
    """
    Vraća 6 površina kocke kao listu, svaka površina je lista 4 tačke.
    
    x0, y0, z0 — koordinate gornjeg lijevog prednjeg ugla
    s          — veličina stranice kocke
    
    Svaka površina: [(x,y,z), (x,y,z), (x,y,z), (x,y,z)]
    Redoslijed tačaka: obavezno konzistentan (konveksni polygon)
    """
    x1, y1, z1 = x0 + s, y0 + s, z0 + s
    return [
        # Donja površina (z = z0)
        [(x0,y0,z0), (x1,y0,z0), (x1,y1,z0), (x0,y1,z0)],
        # Gornja površina (z = z1)
        [(x0,y0,z1), (x1,y0,z1), (x1,y1,z1), (x0,y1,z1)],
        # Prednja površina (y = y0)
        [(x0,y0,z0), (x1,y0,z0), (x1,y0,z1), (x0,y0,z1)],
        # Zadnja površina (y = y1)
        [(x0,y1,z0), (x1,y1,z0), (x1,y1,z1), (x0,y1,z1)],
        # Lijeva površina (x = x0)
        [(x0,y0,z0), (x0,y1,z0), (x0,y1,z1), (x0,y0,z1)],
        # Desna površina (x = x1)
        [(x1,y0,z0), (x1,y1,z0), (x1,y1,z1), (x1,y0,z1)],
    ]


def _draw_block(
    ax,
    data: np.ndarray,
    origin: tuple,
    cell_size: float = 1.0,
    gap: float = 0.08,
    face_color=COLOR_INPUT,
    highlight_mask: np.ndarray = None,
    highlight_color=COLOR_ACTIVE,
    show_values: bool = True,
    value_fmt: str = "{}",
):
    """
    Crta 3D blok kocki koji predstavlja jednu mapu obeležja ili filter.

    data          — numpy niz shape (rows, cols) ili (rows, cols, channels)
                    Channels se crtaju duž Y ose (dubina).
    origin        — (x0, y0, z0) gdje počinje blok u 3D prostoru
    cell_size     — veličina jedne kocke
    gap           — razmak između kocki (estetski)
    face_color    — RGBA boja površina kocki
    highlight_mask — bool niz istog shape kao data[:,:,0]; 
                     True = ta kocka se crta u highlight_color
    show_values   — da li upisujemo vrijednost u centar kocke
    value_fmt     — format string za vrijednost, npr. "{:.1f}"
    """
    # Normalizuj shape na (rows, cols, channels)
    if data.ndim == 2:
        data = data[:, :, np.newaxis]   # dodaj dimenziju kanala

    rows, cols, channels = data.shape
    x0, y0, z0 = origin
    s = cell_size - gap   # efektivna veličina kocke (malo manja od ćelije)

    all_faces  = []
    all_colors = []

    for c in range(channels):
        for r in range(rows):
            for col in range(cols):

                # Koordinate ove kocke
                # X: kolona, Z: red (invertovan da (0,0) bude gore-lijevo),
                # Y: kanal (dubina)
                cx = x0 + col * cell_size
                cy = y0 + c   * cell_size
                cz = z0 + (rows - 1 - r) * cell_size  # invertujemo Z

                faces = _cube_faces(cx, cy, cz, s)
                all_faces.extend(faces)

                # Boja: highlight ako je maska True na toj poziciji
                if (
                    highlight_mask is not None
                    and highlight_mask[r, col]
                ):
                    color = highlight_color
                else:
                    color = face_color

                # Svaka kocka ima 6 površina — ista boja za sve
                all_colors.extend([color] * 6)

    # Dodaj sve površine odjednom — mnogo efikasnije od jednog po jednog
    poly = Poly3DCollection(
        all_faces,
        facecolors=all_colors,
        edgecolors=COLOR_EDGE,
        linewidths=0.3,
    )
    ax.add_collection3d(poly)

    # Upiši vrijednosti u centre kocki
    if show_values:
        for c in range(channels):
            for r in range(rows):
                for col in range(cols):
                    cx = x0 + col * cell_size + s / 2
                    cy = y0 + c   * cell_size + s / 2
                    cz = z0 + (rows - 1 - r) * cell_size + s / 2

                    val = data[r, col, c]
                    ax.text(
                        cx, cy, cz,
                        value_fmt.format(val),
                        ha='center', va='center',
                        fontsize=7,
                        fontweight='bold',
                        color='white',
                        zorder=5,
                    )


def _set_ax_limits(ax, max_dim: int, cell_size: float = 1.0):
    """
    Postavi jednake granice na sve 3 ose da kocke ne izgledaju
    izobličeno. Matplotlib 3D nema auto-aspect-ratio, mora ručno.
    """
    lim = max_dim * cell_size + 1.0
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_zlim(0, lim)
    ax.set_xlabel('Kolona', fontsize=8, labelpad=2)
    ax.set_ylabel('Kanal',  fontsize=8, labelpad=2)
    ax.set_zlabel('Red',    fontsize=8, labelpad=2)
    ax.tick_params(labelsize=6)


# ==================================================================
# Glavne render funkcije — po jednu za svaki task
# ==================================================================

def render_convolution(
    fig: Figure,
    engine,          # ConvolutionEngine instanca
    step: dict,      # trenutni korak iz engine.steps
) -> None:
    """
    Crta stanje konvolucije u datom koraku:
      - Lijevo:   ulazna mapa (s istaknutim aktivnim regionom)
      - Sredina:  filter
      - Desno:    izlazna mapa (popunjena do trenutnog koraka)

    fig — Matplotlib Figure u koji crtamo (dolazi iz GUI-a)
    """
    fig.clear()

    # Kreiraj subplot sa 3D projekcijom
    # 1 red, 3 kolone
    axes = [
        fig.add_subplot(131, projection='3d'),
        fig.add_subplot(132, projection='3d'),
        fig.add_subplot(133, projection='3d'),
    ]

    padded = engine.padded_input   # shape (H, W, C)
    filt   = engine.filter_weights # shape (F, F, C)
    out    = engine.output_map     # shape (out, out)

    rows_p, cols_p, _ = padded.shape
    F = engine.filter_size

    # -- Highlight maska: True za ćelije koje filter trenutno pokriva --
    mask = np.zeros((rows_p, cols_p), dtype=bool)
    r0, c0 = step["in_row"], step["in_col"]
    mask[r0 : r0 + F, c0 : c0 + F] = True

    # -- 1. Ulazna mapa --
    _draw_block(
        axes[0], padded,
        origin=(0, 0, 0),
        face_color=COLOR_INPUT,
        highlight_mask=mask,
        highlight_color=COLOR_ACTIVE,
    )
    axes[0].set_title(
        f"Ulaz {'(padded)' if engine.padding else ''}",
        fontsize=9, pad=4
    )

    # -- 2. Filter --
    _draw_block(
        axes[1], filt,
        origin=(0, 0, 0),
        face_color=COLOR_FILTER,
    )
    # Ispod filtera: formula trenutnog koraka
    products = step["products"]   # shape (F, F, C)
    conv_sum = step["conv_sum"]
    out_val  = step["output_val"]

    formula = f"Σ(ulaz × filter) = {conv_sum}"
    if engine.bias:
        formula += f"  + bias({engine.bias}) = {out_val}"

    axes[1].set_title(f"Filter\n{formula}", fontsize=8, pad=4)

    # -- 3. Izlazna mapa --
    # out je 2D — render kao jednokanalni blok
    _draw_block(
        axes[2], out,
        origin=(0, 0, 0),
        face_color=COLOR_OUTPUT,
        value_fmt="{:d}",
    )
    axes[2].set_title(
        f"Izlaz [{step['out_row']},{step['out_col']}] = {out_val}",
        fontsize=9, pad=4
    )

    # Podesi granice i kut gledanja za sve ose
    max_dim = max(rows_p, cols_p, F, engine.output_size)
    for ax in axes:
        _set_ax_limits(ax, max_dim)
        ax.view_init(elev=25, azim=-50)  # kut koji daje dobar 3D prikaz

    # Legenda
    patches = [
        mpatches.Patch(color=COLOR_INPUT[:3],   label='Ulaz'),
        mpatches.Patch(color=COLOR_ACTIVE[:3],  label='Aktivan region'),
        mpatches.Patch(color=COLOR_FILTER[:3],  label='Filter'),
        mpatches.Patch(color=COLOR_OUTPUT[:3],  label='Izlaz'),
    ]
    fig.legend(handles=patches, loc='lower center', ncol=4, fontsize=8)
    fig.tight_layout(rect=[0, 0.05, 1, 1])


def render_pooling(
    fig: Figure,
    engine,       # PoolingEngine
    step: dict,
) -> None:
    """
    Crta stanje pooling sloja:
      - Lijevo:  ulazna mapa (s istaknutim aktivnim prozorom)
      - Desno:   izlazna mapa
    Sredina je izostavljena jer pooling nema filter s težinama
    (osim weighted avg gdje prikazujemo težine).
    """
    fig.clear()

    has_weights = engine.pool_type == "weighted"
    n_cols = 3 if has_weights else 2

    axes = [
        fig.add_subplot(100 + n_cols * 10 + i + 1, projection='3d')
        for i in range(n_cols)
    ]

    inp = engine.input_map   # (H, W, C)
    H = engine.input_size
    F = engine.filter_size

    # Highlight maska
    mask = np.zeros((H, H), dtype=bool)
    r0, c0 = step["in_row"], step["in_col"]
    mask[r0 : r0 + F, c0 : c0 + F] = True

    # -- 1. Ulazna mapa --
    _draw_block(
        axes[0], inp,
        origin=(0, 0, 0),
        face_color=COLOR_INPUT,
        highlight_mask=mask,
    )

    # Napravi naslov s formulom za trenutni korak
    vals   = step["regions"][0].flatten()   # prvi kanal za prikaz
    result = step["output_vals"][0]

    if engine.pool_type == "max":
        formula = f"max({', '.join(map(str, vals.astype(int)))}) = {result:.2f}"
    elif engine.pool_type == "avg":
        formula = f"avg({', '.join(map(str, vals.astype(int)))}) = {result:.2f}"
    elif engine.pool_type == "l2":
        formula = f"L2({', '.join(map(str, vals.astype(int)))}) = {result:.2f}"
    else:
        formula = f"w·avg = {result:.2f}"

    axes[0].set_title(f"Ulaz\n{formula}", fontsize=8, pad=4)

    # -- 2. Težine (samo za weighted avg) --
    ax_idx = 1
    if has_weights:
        _draw_block(
            axes[1], engine.weights,
            origin=(0, 0, 0),
            face_color=COLOR_FILTER,
            value_fmt="{:.0f}",
        )
        axes[1].set_title("Težine", fontsize=9, pad=4)
        ax_idx = 2

    # -- 3. Izlazna mapa (samo prvi kanal za prikaz) --
    out_ch0 = np.nan_to_num(engine.output_map[:, :, 0], nan=0.0)
    _draw_block(
        axes[ax_idx], out_ch0,
        origin=(0, 0, 0),
        face_color=COLOR_OUTPUT,
        value_fmt="{:.1f}",
    )
    axes[ax_idx].set_title(
        f"Izlaz [{step['out_row']},{step['out_col']}]",
        fontsize=9, pad=4
    )

    max_dim = max(H, F, engine.output_size)
    for ax in axes:
        _set_ax_limits(ax, max_dim)
        ax.view_init(elev=25, azim=-50)

    fig.tight_layout()


def render_pattern(
    fig: Figure,
    engine,        # PatternEngine
    step: dict,
    filter_idx: int,
) -> None:
    """
    Crta detekciju obrazaca:
      - Lijevo:  12×12 ulazna mapa
                 zelena = pozitivno poklapanje ovog filtera
                 crvena = negativno poklapanje
                 plava  = aktivan region
      - Sredina: trenutni filter (3×3)
      - Desno:   izlazna mapa za ovaj filter
    """
    fig.clear()

    axes = [
        fig.add_subplot(131, projection='3d'),
        fig.add_subplot(132, projection='3d'),
        fig.add_subplot(133, projection='3d'),
    ]

    inp = engine.input_map   # (12, 12)
    F   = engine.FILTER_SIZE
    filt = engine.filters[filter_idx]  # (3, 3)
    out  = engine.output_maps[filter_idx]  # (10, 10)

    r0, c0 = step["filter_row"], step["filter_col"]


    # Pronađi specijalne regione za ovaj filter
    pos_mask = np.zeros((engine.MAP_SIZE, engine.MAP_SIZE), dtype=bool)
    neg_mask = np.zeros((engine.MAP_SIZE, engine.MAP_SIZE), dtype=bool)
    act_mask = np.zeros((engine.MAP_SIZE, engine.MAP_SIZE), dtype=bool)

    for sr in engine.special_regions:
        if sr["filter_idx"] == filter_idx:
            sr_r, sr_c = sr["row"], sr["col"]
            if sr["type"] == "positive":
                pos_mask[sr_r:sr_r+F, sr_c:sr_c+F] = True
            else:
                neg_mask[sr_r:sr_r+F, sr_c:sr_c+F] = True

    act_mask[r0:r0+F, c0:c0+F] = True

    # Crta ulaznu mapu — boje po prioritetu: active > positive > negative > default
    # Koristimo custom boju po kocki, pa moramo proći ručno
    # Napravimo combined mapu boja
    def get_color(r, c):
        if act_mask[r, c]:
            return COLOR_ACTIVE
        if pos_mask[r, c]:
            return COLOR_POSITIVE
        if neg_mask[r, c]:
            return COLOR_NEGATIVE
        return COLOR_INPUT

    # _draw_block podržava samo jednu highlight boju, pa ćemo
    # pozvati _draw_block bez highlighta i ručno prebojiti aktivni region
    _draw_block(axes[0], inp, origin=(0, 0, 0),
                face_color=COLOR_INPUT, show_values=False)

    # Precrtaj specijalne regione s pravim bojama
    act_only_mask = act_mask & ~pos_mask & ~neg_mask

    for mask, color in [
        (pos_mask, COLOR_POSITIVE),
        (neg_mask, COLOR_NEGATIVE),
        (act_only_mask, COLOR_ACTIVE),  # ← samo "čiste" aktivne ćelije
    ]:
        if np.any(mask):
            _draw_block(
                axes[0], inp,
                origin=(0, 0, 0),
                face_color=color,
                highlight_mask=mask,
                highlight_color=color,
                show_values=False,   # ne prikazuj vrijednosti na 12×12 (premalo)
            )

    conv_sum = step["output_value"]
    special = step["match_type"]
    label    = ""
    if special == "positive":
        label = " ← MAX (pozitivno poklapanje!)"
    elif special == "negative":
        label = " ← MIN (negativno poklapanje!)"

    axes[0].set_title(f"Ulazna mapa 12×12\nFilter {filter_idx+1}", fontsize=8)

    # Filter
    _draw_block(axes[1], filt, origin=(0, 0, 0), face_color=COLOR_FILTER)
    axes[1].set_title(
        f"Filter {filter_idx+1}\nΣ = {conv_sum:.1f}{label}",
        fontsize=8, pad=4
    )

    # Izlazna mapa
    out_display = np.nan_to_num(out, nan=0.0)
    _draw_block(axes[2], out_display, origin=(0, 0, 0),
                face_color=COLOR_OUTPUT, value_fmt="{:.0f}")
    axes[2].set_title(f"Izlaz filtera {filter_idx+1}", fontsize=8)

    limits = [
        (engine.MAP_SIZE, engine.MAP_SIZE),  # ulaz 12×12
        (F, F),  # filter 3×3
        (engine.output_size, engine.output_size),  # izlaz 10×10
    ]
    for ax, (rows, cols) in zip(axes, limits):
        _set_ax_limits(ax, max(rows, cols))

    # Legenda
    patches = [
        mpatches.Patch(color=COLOR_INPUT[:3],    label='Ulaz'),
        mpatches.Patch(color=COLOR_ACTIVE[:3],   label='Aktivan region'),
        mpatches.Patch(color=COLOR_POSITIVE[:3], label='Pozitivno poklapanje'),
        mpatches.Patch(color=COLOR_NEGATIVE[:3], label='Negativno poklapanje'),
        mpatches.Patch(color=COLOR_OUTPUT[:3],   label='Izlaz'),
    ]
    fig.legend(handles=patches, loc='lower center', ncol=5, fontsize=7)
    fig.tight_layout(rect=[0, 0.05, 1, 1])

