"""
Benchmark: spectral_greedy_large (Technique 1+2) vs degree_greedy_large.
Runs BA graphs up to V=1M and reports cover quality + runtime.
"""
import sys, time
sys.path.insert(0, "../src")

import numpy as np
import igraph as ig
from quantum_walk_mvc.heuristics_igraph import (
    spectral_greedy_large, degree_greedy_large,
    warmup_jit, check_is_vertex_cover_igraph,
)

print("Compiling JIT...", end=" ", flush=True)
warmup_jit()
print("done\n")

# Previous timings (old lazy-heap Python version) for comparison
OLD_TIMES = {
    1_000:       0.003,
    5_000:       0.093,
    10_000:      0.464,
    50_000:     13.050,
    100_000:    59.229,
    500_000:  3052.022,
    1_000_000: 15895.668,
}

hdr = f"{'n':>8}  {'E':>8}  {'sp_cov':>7}  {'dg_cov':>7}  {'valid':>6}  {'sp_t(s)':>8}  {'dg_t(s)':>8}  {'vs_dg':>7}  {'vs_old':>8}"
print(hdr)
print("-" * len(hdr))

sizes = [
    (1_000, 2),
    (5_000, 2),
    (10_000, 2),
    (50_000, 2),
    (100_000, 2),
    (500_000, 2),
    (1_000_000, 2),
]

for n, m in sizes:
    g = ig.Graph.Barabasi(n, m, directed=False)

    t0 = time.perf_counter()
    cov_sp = spectral_greedy_large(g)
    t_sp = time.perf_counter() - t0

    t0 = time.perf_counter()
    cov_dg = degree_greedy_large(g)
    t_dg = time.perf_counter() - t0

    valid = check_is_vertex_cover_igraph(g, set(cov_sp))
    ratio_dg  = t_sp / t_dg  if t_dg  > 0 else float("inf")
    ratio_old = OLD_TIMES.get(n, float("nan")) / t_sp

    print(
        f"{n:>8}  {g.ecount():>8}  {len(cov_sp):>7}  {len(cov_dg):>7}  "
        f"{str(valid):>6}  {t_sp:>8.3f}  {t_dg:>8.3f}  "
        f"{ratio_dg:>6.1f}x  {ratio_old:>7.1f}x"
    )

print("\nDone.")
