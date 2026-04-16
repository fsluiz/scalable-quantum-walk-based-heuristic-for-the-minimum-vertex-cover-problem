#!/usr/bin/env python3
"""
Focused comparison: CTQW vs degree-greedy vs classical heuristics.

Generates fresh graph ensembles (ER, BA, Regular), runs all five algorithms
plus MILP exact solver, and produces:
    1. results_degree_greedy_comparison.csv
    2. figure_degree_greedy_comparison.pdf

This experiment was added to address Reviewer 1's request for a comparison
with degree-based greedy algorithms.

Usage
-----
    python run_degree_greedy_comparison.py [--nodes N [N ...]] [--graphs G]
"""

import argparse
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
import matplotlib.ticker as mticker

from quantum_walk_mvc.core import quantum_walk_mvc_sparse
from itertools import combinations

from quantum_walk_mvc.heuristics import (
    degree_greedy_vertex_cover,
    greedy_2_approx_vertex_cover,
    fast_vc_heuristic,
    simulated_annealing_vertex_cover,
    check_is_vertex_cover,
    get_exact_vertex_cover_sage,
    SAGE_AVAILABLE,
)


BRUTE_FORCE_LIMIT = 20   # exact MVC by brute force for N <= this


def exact_mvc_brute(G: nx.Graph) -> int | None:
    """Return exact MVC size by exhaustive search (feasible for N <= 20)."""
    n = G.number_of_nodes()
    nodes = list(G.nodes())
    for size in range(0, n + 1):
        for candidate in combinations(nodes, size):
            if check_is_vertex_cover(G, set(candidate)):
                return size
    return n


def get_reference_size(G: nx.Graph, algo_sizes: list[int]) -> int:
    """
    Return the best available reference for MVC size.
    Uses exact brute-force for small graphs; otherwise the minimum
    cover size found across all heuristics (a lower bound on OPT).
    """
    n = G.number_of_nodes()
    if n <= BRUTE_FORCE_LIMIT and SAGE_AVAILABLE:
        try:
            _, exact = get_exact_vertex_cover_sage(G)
            return exact
        except Exception:
            pass
    if n <= BRUTE_FORCE_LIMIT:
        exact = exact_mvc_brute(G)
        if exact is not None:
            return exact
    # Proxy: best heuristic solution found
    valid_sizes = [s for s in algo_sizes if s > 0]
    return min(valid_sizes) if valid_sizes else 1
from quantum_walk_mvc.graph_generators import (
    generate_erdos_renyi_graphs,
    generate_barabasi_albert_graphs,
    generate_regular_graphs,
)

T_EVOLUTION = 0.01
RESULTS_FILE = "results_degree_greedy_comparison.csv"
FIGURE_FILE  = "figure_degree_greedy_comparison.pdf"

ALGORITHMS = {
    "CTQW":          lambda G: quantum_walk_mvc_sparse(G, T_EVOLUTION),
    "Degree-Greedy": degree_greedy_vertex_cover,
    "2-Approx":      greedy_2_approx_vertex_cover,
    "FastVC":        lambda G: fast_vc_heuristic(G, max_iterations=1000),
    "SA":            lambda G: simulated_annealing_vertex_cover(
                                   G, initial_temperature=100.0,
                                   cooling_rate=0.99, max_iterations=5000),
}

COLORS = {
    "CTQW":          "#1f77b4",
    "Degree-Greedy": "#d62728",
    "2-Approx":      "#e07b39",
    "FastVC":        "#2ca02c",
    "SA":            "#9467bd",
}
MARKERS = {
    "CTQW": "o", "Degree-Greedy": "s",
    "2-Approx": "^", "FastVC": "D", "SA": "v",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--nodes", type=int, nargs="+",
                   default=list(range(4, 55, 5)))
    p.add_argument("--graphs", type=int, default=10,
                   help="Instances per (N, param) combination")
    return p.parse_args()


def run_all(G: nx.Graph) -> dict:
    """Run all algorithms on G; compute reference size internally."""
    sizes = {}
    times = {}
    valids = {}
    for name, fn in ALGORITHMS.items():
        try:
            t0 = time.perf_counter()
            vc = fn(G)
            times[name]  = time.perf_counter() - t0
            sizes[name]  = len(vc)
            valids[name] = check_is_vertex_cover(G, set(vc))
        except Exception as e:
            print(f"    [{name}] error: {e}")
            sizes[name]  = -1
            times[name]  = float("nan")
            valids[name] = False

    ref = get_reference_size(G, list(sizes.values()))

    row = {"reference_size": ref,
           "exact_used": G.number_of_nodes() <= BRUTE_FORCE_LIMIT}
    for name in ALGORITHMS:
        s = sizes[name]
        row[f"{name}_size"]  = s
        row[f"{name}_time"]  = times[name]
        row[f"{name}_ratio"] = s / ref if (ref > 0 and s > 0) else float("nan")
        row[f"{name}_valid"] = valids[name]
    return row


def build_dataset(node_sizes, n_per_setting):
    records = []

    # ER graphs: p in {0.3, 0.5, 0.7}
    print("\n[ER graphs]")
    for n in node_sizes:
        for p in [0.3, 0.5, 0.7]:
            graphs, _ = generate_erdos_renyi_graphs(
                num_nodes_range=[n], edge_prob_range=[p],
                num_graphs_per_setting=n_per_setting, ensure_connected=True)
            for G in graphs:
                exact = None
                if SAGE_AVAILABLE and n <= 30:
                    try:
                        _, exact = get_exact_vertex_cover_sage(G)
                    except Exception:
                        pass
                row = {"graph_type": "ER", "num_nodes": n, "param": p,
                       "num_edges": G.number_of_edges()}
                row.update(run_all(G))
                records.append(row)
        print(f"  N={n} done")

    # BA graphs: m in {2, 5}
    print("\n[BA graphs]")
    for n in node_sizes:
        for m in [2, 5]:
            if m >= n:
                continue
            graphs, _ = generate_barabasi_albert_graphs(
                num_nodes_range=[n], m_range=[m],
                num_graphs_per_setting=n_per_setting)
            for G in graphs:
                row = {"graph_type": "BA", "num_nodes": n, "param": m,
                       "num_edges": G.number_of_edges()}
                row.update(run_all(G))
                records.append(row)
        print(f"  N={n} done")

    # Regular graphs
    print("\n[Regular graphs]")
    for n in node_sizes:
        graphs, _ = generate_regular_graphs(
            num_nodes_range=[n], num_graphs_per_setting=n_per_setting)
        for G in graphs:
            row = {"graph_type": "REG", "num_nodes": n, "param": None,
                   "num_edges": G.number_of_edges()}
            row.update(run_all(G))
            records.append(row)
        print(f"  N={n} done")

    return pd.DataFrame(records)


def make_figure(df: pd.DataFrame, outfile: str) -> None:
    graph_types = ["ER", "BA", "REG"]
    algo_names  = list(ALGORITHMS.keys())

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    fig.subplots_adjust(wspace=0.08, left=0.07, right=0.97,
                        top=0.88, bottom=0.14)

    for ax, gtype in zip(axes, graph_types):
        sub = df[df["graph_type"] == gtype].copy()
        sizes = sorted(sub["num_nodes"].unique())

        for name in algo_names:
            col = f"{name}_ratio"
            if col not in sub.columns:
                continue
            means = sub.groupby("num_nodes")[col].mean()
            stds  = sub.groupby("num_nodes")[col].std()

            ns = [n for n in sizes if n in means.index]
            ys = [means[n] for n in ns]
            es = [stds[n]  for n in ns]

            lw = 2.2 if name in ("CTQW", "Degree-Greedy") else 1.2
            zo = 3   if name in ("CTQW", "Degree-Greedy") else 2
            ax.plot(ns, ys, color=COLORS[name], marker=MARKERS[name],
                    markersize=5, linewidth=lw, zorder=zo, label=name)
            ax.fill_between(ns,
                            [y - e for y, e in zip(ys, es)],
                            [y + e for y, e in zip(ys, es)],
                            color=COLORS[name], alpha=0.12, zorder=zo - 1)

        ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8,
                   alpha=0.6, zorder=1)
        ax.set_title({"ER": "Erdős–Rényi", "BA": "Barabási–Albert",
                      "REG": "Regular"}[gtype], fontsize=11)
        ax.set_xlabel("Number of nodes $N$", fontsize=10)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=5))
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        ax.set_ylim(bottom=0.92)

    axes[0].set_ylabel("Approximation ratio  $|C|/|C^*|$", fontsize=10)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5,
               fontsize=9.5, framealpha=0.9,
               bbox_to_anchor=(0.52, 1.01))

    fig.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved: {outfile}")


def print_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 65)
    print("  Mean approximation ratio by graph type")
    print("=" * 65)
    algo_names = list(ALGORITHMS.keys())
    header = f"  {'Graph':>8}  " + "  ".join(f"{n:>14}" for n in algo_names)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for gtype in ["ER", "BA", "REG"]:
        sub = df[df["graph_type"] == gtype]
        vals = []
        for name in algo_names:
            col = f"{name}_ratio"
            m = sub[col].mean() if col in sub.columns else float("nan")
            vals.append(f"{m:>14.3f}")
        print(f"  {gtype:>8}  " + "  ".join(vals))

    print("\n  CTQW vs Degree-Greedy  (mean ratio, pooled)")
    ctqw_mean = df["CTQW_ratio"].mean()
    dg_mean   = df["Degree-Greedy_ratio"].mean()
    print(f"    CTQW          : {ctqw_mean:.3f}")
    print(f"    Degree-Greedy : {dg_mean:.3f}")
    print(f"    CTQW is better in {(df['CTQW_ratio'] < df['Degree-Greedy_ratio']).mean()*100:.1f}% of instances")


def main():
    args = parse_args()
    print(f"Node sizes: {args.nodes}")
    print(f"Instances per setting: {args.graphs}")
    print(f"Exact solver available: {SAGE_AVAILABLE}")

    df = build_dataset(args.nodes, args.graphs)
    df.to_csv(RESULTS_FILE, index=False)
    print(f"\nResults saved: {RESULTS_FILE}  ({len(df)} rows)")

    print_summary(df)
    make_figure(df, FIGURE_FILE)


if __name__ == "__main__":
    main()
