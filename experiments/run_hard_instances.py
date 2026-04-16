#!/usr/bin/env python3
"""
Hard-instance analysis: CTQW on graph families known to challenge MVC heuristics.

Tests graph families where greedy and local-search heuristics struggle:
  - Crown graphs H_n (bipartite, vertex-transitive, OPT = n)
  - Complete bipartite K_{n,n} (OPT = n)
  - Random 3-regular graphs (expander-like, hard for local search)
  - Cycle graphs C_n (OPT = ceil(n/2))
  - Wheel graphs W_n (hub-heavy, OPT = n-1 for spokes)

For each family and size, runs CTQW + degree-greedy + 2-approx + SA with
exact MVC by exhaustive search (N <= 20) as reference.

Outputs:
    results_hard_instances.csv
    figure_hard_instances.pdf

Usage
-----
    python run_hard_instances.py
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import combinations

from quantum_walk_mvc.core import quantum_walk_mvc_sparse
from quantum_walk_mvc.heuristics import (
    degree_greedy_vertex_cover,
    greedy_2_approx_vertex_cover,
    simulated_annealing_vertex_cover,
    check_is_vertex_cover,
    get_exact_vertex_cover_sage,
    SAGE_AVAILABLE,
)

RESULTS_FILE = "results_hard_instances.csv"
FIGURE_FILE  = "figure_hard_instances.pdf"
T_EVOLUTION  = 0.01
BRUTE_FORCE_LIMIT = 20

ALGORITHMS = {
    "CTQW":          lambda G: quantum_walk_mvc_sparse(G, T_EVOLUTION),
    "Degree-Greedy": degree_greedy_vertex_cover,
    "2-Approx":      greedy_2_approx_vertex_cover,
    "SA":            lambda G: simulated_annealing_vertex_cover(
                                   G, initial_temperature=100.0,
                                   cooling_rate=0.99, max_iterations=5000),
}

COLORS  = {"CTQW": "#1f77b4", "Degree-Greedy": "#d62728",
           "2-Approx": "#e07b39", "SA": "#9467bd"}
MARKERS = {"CTQW": "o", "Degree-Greedy": "s", "2-Approx": "^", "SA": "v"}

# ---------------------------------------------------------------------------
# Graph families
# ---------------------------------------------------------------------------

def crown_graph(n: int) -> nx.Graph:
    """H_n = K_{n,n} minus a perfect matching.  OPT = n."""
    G = nx.complete_bipartite_graph(n, n)
    for i in range(n):
        G.remove_edge(i, n + i)
    return G


def GRAPH_FAMILIES(max_n: int = 18) -> list[tuple[str, int, nx.Graph]]:
    """
    Yield (family_name, n_param, graph) tuples.
    All graphs have N <= max_n so exact brute-force is feasible.
    """
    families = []

    # Crown graphs H_n  (N = 2n, OPT = n)
    for n in range(3, max_n // 2 + 1):
        G = crown_graph(n)
        if G.number_of_nodes() <= max_n:
            families.append((f"Crown", n, G))

    # Complete bipartite K_{n,n}  (N = 2n, OPT = n)
    for n in range(2, max_n // 2 + 1):
        G = nx.complete_bipartite_graph(n, n)
        if G.number_of_nodes() <= max_n:
            families.append(("K_{n,n}", n, G))

    # Cycle graphs C_n  (OPT = ceil(n/2))
    for n in range(4, max_n + 1):
        G = nx.cycle_graph(n)
        families.append(("Cycle", n, G))

    # Random 3-regular graphs  (10 seeds per N)
    for n in [10, 12, 14, 16, 18]:
        for seed in range(10):
            try:
                G = nx.random_regular_graph(3, n, seed=seed * 100 + n)
                if nx.is_connected(G):
                    families.append(("3-Regular", n, G))
            except Exception:
                pass

    # Petersen-like cage graphs
    families.append(("Petersen", 10, nx.petersen_graph()))

    return families


# ---------------------------------------------------------------------------
# Exact reference
# ---------------------------------------------------------------------------
def exact_mvc_brute(G: nx.Graph) -> int:
    nodes = list(G.nodes())
    n = len(nodes)
    for size in range(n + 1):
        for cand in combinations(nodes, size):
            if check_is_vertex_cover(G, set(cand)):
                return size
    return n


def get_reference(G: nx.Graph) -> int:
    n = G.number_of_nodes()
    if n <= BRUTE_FORCE_LIMIT and SAGE_AVAILABLE:
        try:
            _, exact = get_exact_vertex_cover_sage(G)
            return exact
        except Exception:
            pass
    if n <= BRUTE_FORCE_LIMIT:
        return exact_mvc_brute(G)
    return None   # signal: too large for exact


# ---------------------------------------------------------------------------
# Run algorithms
# ---------------------------------------------------------------------------
def run_all(G: nx.Graph, ref: int) -> dict:
    row = {}
    for name, fn in ALGORITHMS.items():
        try:
            t0 = time.perf_counter()
            vc = fn(G)
            elapsed = time.perf_counter() - t0
            s = len(vc)
            valid = check_is_vertex_cover(G, set(vc))
        except Exception as e:
            s, elapsed, valid = -1, float("nan"), False
        row[f"{name}_size"]  = s
        row[f"{name}_time"]  = elapsed
        row[f"{name}_ratio"] = s / ref if (s > 0 and ref > 0) else float("nan")
        row[f"{name}_valid"] = valid
    return row


# ---------------------------------------------------------------------------
# Build dataset
# ---------------------------------------------------------------------------
def build_dataset() -> pd.DataFrame:
    records = []
    families = GRAPH_FAMILIES(max_n=20)
    total = len(families)
    print(f"Total graph instances: {total}")

    for i, (fname, n_param, G) in enumerate(families):
        ref = get_reference(G)
        if ref is None:
            print(f"  Skipping {fname}(n={n_param}): too large for exact solver")
            continue

        row = {
            "family":   fname,
            "n_param":  n_param,
            "num_nodes": G.number_of_nodes(),
            "num_edges": G.number_of_edges(),
            "exact_vc_size": ref,
            "is_bipartite": nx.is_bipartite(G),
            "is_regular":   nx.is_regular(G),
        }
        row.update(run_all(G, ref))
        records.append(row)

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{total} done...")

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Figure: box plot of approximation ratios per family per algorithm
# ---------------------------------------------------------------------------
def make_figure(df: pd.DataFrame, outfile: str) -> None:
    families = df["family"].unique()
    algo_names = list(ALGORITHMS.keys())

    # One subplot per algorithm, x-axis = family
    fig, axes = plt.subplots(1, len(algo_names), figsize=(14, 4.5), sharey=True)
    fig.subplots_adjust(wspace=0.07, left=0.07, right=0.97, top=0.88, bottom=0.20)

    x_positions = np.arange(len(families))
    width = 0.6

    for ax, name in zip(axes, algo_names):
        col = f"{name}_ratio"
        data_per_family = [df[df["family"] == f][col].dropna().values
                           for f in families]
        bp = ax.boxplot(data_per_family, positions=x_positions,
                        widths=width, patch_artist=True,
                        medianprops=dict(color="white", linewidth=2),
                        boxprops=dict(facecolor=COLORS[name], alpha=0.7),
                        whiskerprops=dict(linewidth=1.2),
                        capprops=dict(linewidth=1.2),
                        flierprops=dict(marker="x", markersize=4,
                                        markeredgecolor=COLORS[name]))

        ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.axhline(1.3606, color="crimson", linestyle=":", linewidth=1.0,
                   alpha=0.8, label=r"$\approx 1.3606$ bound")
        ax.set_title(name, fontsize=10.5, color=COLORS[name], fontweight="bold")
        ax.set_xticks(x_positions)
        ax.set_xticklabels(families, rotation=35, ha="right", fontsize=8.5)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        ax.set_ylim(bottom=0.90)

    axes[0].set_ylabel("Approximation ratio  $|C|/|C^*|$", fontsize=10)

    # Add inapproximability line to legend only once
    axes[-1].legend(fontsize=8.5, loc="upper right")

    fig.suptitle("CTQW and classical heuristics on hard MVC instances\n"
                 "(exact reference, $N \\leq 20$)",
                 fontsize=11, y=1.01)

    fig.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved: {outfile}")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def print_summary(df: pd.DataFrame) -> None:
    algo_names = list(ALGORITHMS.keys())
    print("\n" + "=" * 70)
    print("  Approximation ratio summary on hard instances")
    print("=" * 70)
    print(f"  {'Family':>12}  " +
          "  ".join(f"{n:>14}" for n in algo_names))
    print("  " + "-" * 68)
    for fname in df["family"].unique():
        sub = df[df["family"] == fname]
        vals = []
        for name in algo_names:
            col = f"{name}_ratio"
            m = sub[col].mean() if col in sub.columns else float("nan")
            mx = sub[col].max() if col in sub.columns else float("nan")
            vals.append(f"{m:.3f}({mx:.3f})")
        print(f"  {fname:>12}  " + "  ".join(f"{v:>14}" for v in vals))
    print("  (format: mean(max))")

    print(f"\n  Inapproximability bound: ≈ 1.3606 (Dinur & Safra 2005)")
    print(f"  Instances where CTQW ratio > 1.2: "
          f"{(df['CTQW_ratio'] > 1.2).sum()} / {len(df)}")
    print(f"  Instances where CTQW ratio > 1.3606: "
          f"{(df['CTQW_ratio'] > 1.3606).sum()} / {len(df)}")
    print(f"  Max CTQW ratio seen: {df['CTQW_ratio'].max():.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Building hard-instance dataset...")
    df = build_dataset()
    df.to_csv(RESULTS_FILE, index=False)
    print(f"\nResults saved: {RESULTS_FILE}  ({len(df)} rows)")

    print_summary(df)
    make_figure(df, FIGURE_FILE)


if __name__ == "__main__":
    main()
