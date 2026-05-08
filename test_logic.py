# test_logic.py
"""
Testira core module i renderer bez GUI-a.
Sačuva slike u test_output/ folder.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # bez ekrana — samo sačuvaj fajl
import matplotlib.pyplot as plt

sys.path.insert(0, '.')
os.makedirs('test_output', exist_ok=True)

# ── Test 1: ConvolutionEngine ─────────────────────────────────────
print("=== Test 1: ConvolutionEngine ===")
from core.convolution import ConvolutionEngine

eng = ConvolutionEngine(
    input_size=5, channels=1,
    filter_size=3, stride=1,
    padding=False, bias=1
)

print(f"Ulaz shape:  {eng.padded_input.shape}")
print(f"Filter shape: {eng.filter_weights.shape}")
print(f"Izlaz size:  {eng.output_size}×{eng.output_size}")
print(f"Ukupno koraka: {len(eng.steps)}")

step = eng.get_current_step()
print(f"Korak 0: izlaz[{step['out_row']},{step['out_col']}] = {step['output_val']}")

eng.next_step()
step = eng.get_current_step()
print(f"Korak 1: izlaz[{step['out_row']},{step['out_col']}] = {step['output_val']}")
print("ConvolutionEngine — OK\n")

# ── Test 2: Renderer konvolucija ──────────────────────────────────
print("=== Test 2: Renderer konvolucija ===")
from visualization.renderer_3d import render_convolution

fig = plt.figure(figsize=(14, 5))
eng.reset()
# Prođi 4 koraka
for _ in range(4):
    eng.next_step()

step = eng.get_current_step()
render_convolution(fig, eng, step)
fig.savefig('test_output/conv_step4.png', dpi=100, bbox_inches='tight')
plt.close(fig)
print("Sačuvano: test_output/conv_step4.png\n")

# ── Test 3: PoolingEngine ─────────────────────────────────────────
print("=== Test 3: PoolingEngine ===")
from core.pooling import PoolingEngine

for pool_type in ["max", "avg", "l2", "weighted"]:
    eng_p = PoolingEngine(
        input_size=5, channels=1,
        filter_size=2, stride=1,
        pool_type=pool_type
    )
    step = eng_p.get_current_step()
    print(f"  {pool_type:10s} → izlaz[0,0] = {step['output_vals'][0]:.2f}")

print("PoolingEngine — OK\n")

# ── Test 4: Renderer pooling ──────────────────────────────────────
print("=== Test 4: Renderer pooling ===")
from visualization.renderer_3d import render_pooling

fig = plt.figure(figsize=(14, 5))
eng_p = PoolingEngine(input_size=5, channels=1,
                      filter_size=2, stride=1, pool_type="max")
eng_p.next_step()
eng_p.next_step()
step = eng_p.get_current_step()
render_pooling(fig, eng_p, step)
fig.savefig('test_output/pool_step2.png', dpi=100, bbox_inches='tight')
plt.close(fig)
print("Sačuvano: test_output/pool_step2.png\n")

# ── Test 5: PatternEngine ─────────────────────────────────────────
print("=== Test 5: PatternEngine ===")
from core.pattern import PatternEngine

eng_pat = PatternEngine()
print(f"Filteri shape: {eng_pat.filters.shape}")
print(f"Ulaz shape: {eng_pat.input_map.shape}")
print(f"Specijalni regioni:")
for sr in eng_pat.special_regions:
    print(f"  Filter {sr['filter_idx']+1} {sr['type']:10s} → ({sr['row']},{sr['col']})")

# Prođi do prvog specijalnog koraka
steps = eng_pat.steps_per_filter[0]
special_steps = [s for s in steps if s["match_type"] != "neutral"]
print(f"Specijalni koraci za filter 1: {len(special_steps)}")
for s in special_steps:
    print(f"  [{s['out_row']},{s['out_col']}] match={s['match_type']} Σ={s['output_value']:.1f}")
print("PatternEngine — OK\n")

# ── Test 6: Renderer pattern ──────────────────────────────────────
print("=== Test 6: Renderer pattern ===")
from visualization.renderer_3d import render_pattern

fig = plt.figure(figsize=(14, 5))
eng_pat2 = PatternEngine()

# Nađi specijalni korak (pozitivno poklapanje) za filter 0
steps = eng_pat2.steps_per_filter[0]
pos_step = next((s for s in steps if s["match_type"] == "positive"), steps[5])

render_pattern(fig, eng_pat2, pos_step, filter_idx=0)
fig.savefig('test_output/pattern_positive.png', dpi=100, bbox_inches='tight')
plt.close(fig)
print("Sačuvano: test_output/pattern_positive.png\n")

print("=" * 40)
print("Svi testovi prošli. Pogledaj test_output/ folder.")