#!/usr/bin/env python3
"""
Demonstration: Bloqade-based CTQW-MVC on a Random Geometric Graph.

A Random Geometric Graph (RGG) is a unit-disk graph by construction: n nodes
are placed uniformly in the unit square and edges connect pairs within a fixed
radius.  Scaling those positions to micrometers maps the graph exactly onto a
Rydberg atom array with no spurious blockade interactions.

Physical mapping
----------------
    Graph vertex              <-->  Rydberg atom (87Rb)
    Graph edge                <-->  Atom pair within blockade radius R_b
    Hamiltonian H = I - Gamma <-->  Uniform Rabi drive at Delta = 0
    Vertex freeze             <-->  Large local detuning  Delta_i >> Omega
    P(m -> out)               <-->  Marginal excitation probability <n_m>

Usage
-----
    cd quantum-walk-mvc
    python experiments/run_bloqade_demo.py

If Bloqade is not installed, the script falls back to the classical scipy
CTQW emulator and prints an informative message.
"""

from __future__ import annotations

import sys
import os
import time
import math
import warnings

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

import networkx as nx
import numpy as np

# --------------------------------------------------------------------------- #
# Bloqade availability                                                         #
# --------------------------------------------------------------------------- #

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from bloqade.analog import start as _test_start  # noqa: F401
    BLOQADE_AVAILABLE = True
except ImportError:
    BLOQADE_AVAILABLE = False

# --------------------------------------------------------------------------- #
# Package imports                                                              #
# --------------------------------------------------------------------------- #

from quantum_walk_mvc.bloqade_mvc import (
    quantum_walk_mvc_bloqade,
    embed_unit_disk_graph,
    get_rydberg_interaction_matrix,
    C6_RB87,
    BLOQADE_AVAILABLE as _PKG_BLOQADE,
)
from quantum_walk_mvc.core import quantum_walk_mvc_sparse
from quantum_walk_mvc.heuristics import (
    greedy_2_approx_vertex_cover,
    check_is_vertex_cover,
)

# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #


def _divider(char: str = "=", width: int = 72) -> str:
    return char * width


def _print_section(title: str) -> None:
    print()
    print(_divider())
    print(f"  {title}")
    print(_divider())


def _verify_cover(graph: nx.Graph, vc: list, label: str) -> bool:
    vc_set = set(vc)
    valid = check_is_vertex_cover(graph, vc_set)
    status = "VALID  " if valid else "INVALID"
    print(f"    [{status}]  {label}: {len(vc)} vertices  ->  {sorted(vc)}")
    return valid


def _layout_report(
    graph: nx.Graph,
    positions: dict,
    blockade_radius: float,
) -> None:
    """Print interaction distances for all atom pairs."""
    nodes = list(positions.keys())
    n = len(nodes)
    adj_set = {(min(u, v), max(u, v)) for u, v in graph.edges()}
    V = get_rydberg_interaction_matrix(positions, C6=C6_RB87)

    edge_dists, non_edge_dists = [], []
    violations = 0

    for i in range(n):
        for j in range(i + 1, n):
            ni, nj = nodes[i], nodes[j]
            pair = (min(ni, nj), max(ni, nj))
            Vij = V[i, j]
            dist = (C6_RB87 / Vij) ** (1.0 / 6) if Vij > 1e-9 else float("inf")
            if pair in adj_set:
                edge_dists.append(dist)
            else:
                non_edge_dists.append(dist)
                if dist < blockade_radius:
                    violations += 1

    print(f"\n  Atom layout  ({n} atoms,  {graph.number_of_edges()} edges)")
    print(f"  {'Parameter':<35}  {'Value':>10}")
    print(f"  {'-'*46}")
    print(f"  {'Blockade radius R_b':<35}  {blockade_radius:>9.2f} μm")
    print(f"  {'Edge distances  (min / mean / max)':<35}  "
          f"{min(edge_dists):>5.2f} / {np.mean(edge_dists):>5.2f} / {max(edge_dists):>5.2f} μm")
    print(f"  {'Non-edge distances (min / mean)':<35}  "
          f"{min(non_edge_dists):>5.2f} / {np.mean(non_edge_dists):>5.2f} μm")
    print(f"  {'Spurious blockade violations':<35}  {violations:>10d}")
    if violations == 0:
        print(f"\n  ✓  Perfect unit-disk embedding — zero spurious interactions.")
    else:
        print(f"\n  ✗  {violations} non-edge pair(s) within blockade radius.")


def _print_atom_positions(positions: dict) -> None:
    print("\n  Atom positions (μm):")
    print(f"  {'Atom':>5}   {'x (μm)':>10}   {'y (μm)':>10}")
    print(f"  {'-'*32}")
    for node, (x, y) in sorted(positions.items()):
        print(f"  {node:>5}   {x:>+10.3f}   {y:>+10.3f}")


# --------------------------------------------------------------------------- #
# Main demo                                                                    #
# --------------------------------------------------------------------------- #


def run_demo() -> None:

    # ------------------------------------------------------------------ #
    # Physical parameters                                                  #
    # ------------------------------------------------------------------ #
    Omega          = 4.0 * math.pi   # rad/μs  (peak Rabi frequency ≈ 12.57)
    blockade_radius = 7.5            # μm  →  R_b = (C6/Omega)^(1/6) ≈ 7.5 μm
    t_evolution    = 0.05            # μs  (walk time; well within coherent window)
    n_shots        = 1000            # measurement shots per iteration
    freeze_detuning = 100.0          # rad/μs  (>> Omega: suppresses frozen atoms)

    _print_section("Physical parameters — 87Rb Rydberg platform")
    print(f"""
  Rabi frequency Ω        : {Omega:.4f}  rad/μs  (= 4π)
  Blockade radius R_b     : {blockade_radius:.2f}  μm   [R_b = (C6/Ω)^(1/6)]
  Walk time t_evolution   : {t_evolution:.4f}  μs
  Measurement shots       : {n_shots}
  Freeze detuning Δ_i     : {freeze_detuning:.1f}  rad/μs  (>> Ω → suppresses dynamics)
  C6 coefficient (87Rb)   : {C6_RB87:.0f}  rad·μs⁻¹·μm⁶""")

    # ------------------------------------------------------------------ #
    # Build unit-disk graph                                                #
    # ------------------------------------------------------------------ #
    _print_section("Step 1: Generate Random Geometric Graph (unit-disk graph)")

    print("""
  A Random Geometric Graph (RGG) places n nodes uniformly in [0,1]²
  and connects pairs within a fixed radius.  It is a unit-disk graph
  by construction: scaling the positions to μm maps EVERY edge within
  R_b and EVERY non-edge outside R_b — with zero spurious interactions.
""")

    G, positions = embed_unit_disk_graph(
        n=10,
        connection_radius=0.38,
        blockade_radius=blockade_radius,
        seed=7,
    )

    print(f"  Nodes  : {G.number_of_nodes()}")
    print(f"  Edges  : {G.number_of_edges()}")
    print(f"  Degree sequence : {sorted([d for _, d in G.degree()], reverse=True)}")

    _print_atom_positions(positions)
    _layout_report(G, positions, blockade_radius)

    # ------------------------------------------------------------------ #
    # Compute exact MVC for reference (brute force, small graph)          #
    # ------------------------------------------------------------------ #
    from itertools import combinations

    optimal_vc = None
    for size in range(1, G.number_of_nodes() + 1):
        for candidate in combinations(G.nodes(), size):
            if check_is_vertex_cover(G, set(candidate)):
                optimal_vc = list(candidate)
                break
        if optimal_vc is not None:
            break
    optimal_size = len(optimal_vc)

    print(f"\n  Optimal MVC (brute force): size = {optimal_size},  {sorted(optimal_vc)}")

    # ------------------------------------------------------------------ #
    # Bloqade MVC                                                          #
    # ------------------------------------------------------------------ #
    _print_section("Step 2: Bloqade Rydberg Emulator — CTQW-MVC")

    if BLOQADE_AVAILABLE:
        print(f"\n  Running Bloqade emulator"
              f"  (Ω = {Omega:.3f} rad/μs,  t = {t_evolution} μs,"
              f"  shots = {n_shots}/iter) ...")
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bloqade_vc = quantum_walk_mvc_bloqade(
                graph=G,
                t_evolution=t_evolution,
                Omega=Omega,
                n_shots=n_shots,
                blockade_radius=blockade_radius,
                freeze_detuning=freeze_detuning,
                embed_seed=7,
            )
        bloqade_time = time.perf_counter() - t0
        mode = "Rydberg emulator"
        print(f"  Completed in {bloqade_time:.3f} s")
    else:
        print(
            "\n  Bloqade not installed — using classical CTQW fallback.\n"
            "  Install with:  pip install bloqade\n"
        )
        t0 = time.perf_counter()
        bloqade_vc = quantum_walk_mvc_bloqade(graph=G, t_evolution=t_evolution)
        bloqade_time = time.perf_counter() - t0
        mode = "classical fallback"

    bloqade_valid = _verify_cover(G, bloqade_vc,
                                   f"Bloqade CTQW-MVC ({mode})")

    # ------------------------------------------------------------------ #
    # Classical sparse CTQW                                               #
    # ------------------------------------------------------------------ #
    _print_section("Step 3: Classical Sparse CTQW-MVC (scipy)")

    t0 = time.perf_counter()
    classical_vc = quantum_walk_mvc_sparse(G, t_opt=t_evolution)
    classical_time = time.perf_counter() - t0
    classical_valid = _verify_cover(G, classical_vc, "Classical CTQW-MVC (scipy)")

    # ------------------------------------------------------------------ #
    # Greedy 2-approximation baseline                                     #
    # ------------------------------------------------------------------ #
    _print_section("Step 4: Greedy 2-Approximation Baseline")

    t0 = time.perf_counter()
    greedy_vc = greedy_2_approx_vertex_cover(G)
    greedy_time = time.perf_counter() - t0
    greedy_valid = _verify_cover(G, greedy_vc, "Greedy 2-approx")

    # ------------------------------------------------------------------ #
    # Comparison table                                                     #
    # ------------------------------------------------------------------ #
    _print_section("Comparison Table")

    rows = [
        ("Algorithm",                        "Size", "Valid",  "Ratio",  "Time"),
        (_divider("-", 28),                  "----", "-----",  "-----",  "------------"),
        (
            f"Bloqade CTQW  ({mode[:3]})",
            str(len(bloqade_vc)),
            "Yes" if bloqade_valid  else "No",
            f"{len(bloqade_vc)/optimal_size:.3f}",
            f"{bloqade_time:.3f} s",
        ),
        (
            "Classical CTQW  (scipy)",
            str(len(classical_vc)),
            "Yes" if classical_valid else "No",
            f"{len(classical_vc)/optimal_size:.3f}",
            f"{classical_time*1000:.3f} ms",
        ),
        (
            "Greedy 2-approx",
            str(len(greedy_vc)),
            "Yes" if greedy_valid   else "No",
            f"{len(greedy_vc)/optimal_size:.3f}",
            f"{greedy_time*1000:.4f} ms",
        ),
        (
            "Optimal (brute force)",
            str(optimal_size),
            "Yes",
            "1.000",
            "—",
        ),
    ]

    col_w = [max(len(r[c]) for r in rows) for c in range(len(rows[0]))]
    fmt = "  ".join(f"{{:<{w}}}" for w in col_w)
    print()
    for row in rows:
        print("  " + fmt.format(*row))
    print()

    # ------------------------------------------------------------------ #
    # Notes                                                               #
    # ------------------------------------------------------------------ #
    _print_section("Implementation notes")
    print(f"""
  Unit-disk embedding
  -------------------
  The Random Geometric Graph positions are scaled so the median edge
  length equals 0.9 × R_b = {0.9*blockade_radius:.2f} μm.  Every edge pair is
  within R_b (blockade active) and every non-edge pair is outside R_b
  (no spurious interactions).

  Freezing mechanism
  ------------------
  After selecting vertex m*, a large local detuning Δ_m* = {freeze_detuning:.0f} rad/μs
  is applied to atom m*.  Since Δ >> Ω ≈ {Omega:.1f} rad/μs, the effective
  Rabi coupling ~Ω²/(2Δ) ≈ {Omega**2/(2*freeze_detuning):.3f} rad/μs → 0.
  The atom stays in |g⟩ and is decoupled from further walk dynamics.
  This is the hardware realisation of H_F = (Ω-1)·P_m from the paper.

  Shot noise
  ----------
  Finite n_shots = {n_shots} introduces statistical fluctuations in the
  probability estimates.  Increase n_shots for more stable selection,
  at the cost of longer wall-clock time per iteration.
""")


if __name__ == "__main__":
    run_demo()
