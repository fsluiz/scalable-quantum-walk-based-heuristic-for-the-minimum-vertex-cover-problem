#!/usr/bin/env python3
"""
Sensitivity analysis: approximation ratio vs evolution time t.

For each t in a logarithmic grid, runs quantum_walk_mvc_sparse on fresh
graph ensembles (ER, BA, Regular) and records the approximation ratio.
Also overlays the t_opt = 4/(pi * sqrt(N)) + 0.1 prediction.

Outputs:
    results_t_sensitivity.csv
    figure_t_sensitivity.pdf

Usage
-----
    python run_t_sensitivity.py [--nodes N [N ...]] [--graphs G]
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
from itertools import combinations

from quantum_walk_mvc.core import quantum_walk_mvc_sparse
from quantum_walk_mvc.heuristics import (
    check_is_vertex_cover,
    get_exact_vertex_cover_sage,
    SAGE_AVAILABLE,
)
from quantum_walk_mvc.graph_generators import (
    generate_erdos_renyi_graphs,
    generate_barabasi_albert_graphs,
    generate_regular_graphs,
)

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
RESULTS_FILE = "results_t_sensitivity.csv"
FIGURE_FILE  = "figure_t_sensitivity.pdf"
BRUTE_FORCE_LIMIT = 20

# Logarithmically-spaced grid of evolution times
T_VALUES = np.logspace(-3, 1, 30)   # 0.001 … 10

# Theoretical coherent transport window: t_opt = 4/(pi * sqrt(N)) + 0.1
def t_opt(N: int) -> float:
    return 4.0 / (np.pi * np.sqrt(N)) + 0.1


# ---------------------------------------------------------------------------
# Reference size helpers
# ---------------------------------------------------------------------------
def exact_mvc_brute(G: nx.Graph) -> int:
    n = G.number_of_nodes()
    nodes = list(G.nodes())
    for size in range(0, n + 1):
        for cand in combinations(nodes, size):
            if check_is_vertex_cover(G, set(cand)):
                return size
    return n


def get_exact_reference(G: nx.Graph) -> int | None:
    """Return exact MVC size for small graphs; None for large ones."""
    n = G.number_of_nodes()
    if n <= BRUTE_FORCE_LIMIT and SAGE_AVAILABLE:
        try:
            _, exact = get_exact_vertex_cover_sage(G)
            return exact
        except Exception:
            pass
    if n <= BRUTE_FORCE_LIMIT:
        return exact_mvc_brute(G)
    return None


# ---------------------------------------------------------------------------
# Core experiment
# ---------------------------------------------------------------------------
def run_graph(G: nx.Graph) -> tuple[dict, int]:
    """
    Run CTQW for all t values.  Returns ({t: vc_size}, reference_size).

    For N <= BRUTE_FORCE_LIMIT  : reference = exact MVC
    For N >  BRUTE_FORCE_LIMIT  : reference = min vc_size across all t
                                   (best proxy available without exact solver)
    """
    sizes = {}
    for t in T_VALUES:
        try:
            vc = quantum_walk_mvc_sparse(G, t)
            sizes[t] = len(vc)
        except Exception:
            sizes[t] = None

    exact = get_exact_reference(G)
    if exact is not None:
        ref = exact
    else:
        valid = [s for s in sizes.values() if s is not None]
        ref = min(valid) if valid else G.number_of_nodes()

    ratios = {}
    for t, s in sizes.items():
        ratios[t] = s / ref if (s is not None and ref > 0) else float("nan")

    return ratios, ref


def build_dataset(node_sizes, n_per_setting):
    records = []

    def process(gtype, n, param, graphs):
        for G in graphs:
            t_ratios, ref = run_graph(G)
            row = {
                "graph_type": gtype,
                "num_nodes": n,
                "param": param,
                "num_edges": G.number_of_edges(),
                "reference_size": ref,
                "exact_ref": get_exact_reference(G) is not None,
                "t_opt_theory": t_opt(n),
            }
            for t, ratio in t_ratios.items():
                row[f"ratio_t{t:.6f}"] = ratio
            records.append(row)

    print("\n[ER graphs]")
    for n in node_sizes:
        graphs, _ = generate_erdos_renyi_graphs(
            num_nodes_range=[n], edge_prob_range=[0.5],
            num_graphs_per_setting=n_per_setting, ensure_connected=True)
        process("ER", n, 0.5, graphs)
        print(f"  N={n} done")

    print("\n[BA graphs]")
    for n in node_sizes:
        graphs, _ = generate_barabasi_albert_graphs(
            num_nodes_range=[n], m_range=[2],
            num_graphs_per_setting=n_per_setting)
        process("BA", n, 2, graphs)
        print(f"  N={n} done")

    print("\n[Regular graphs]")
    for n in node_sizes:
        graphs, _ = generate_regular_graphs(
            num_nodes_range=[n], num_graphs_per_setting=n_per_setting)
        process("REG", n, None, graphs)
        print(f"  N={n} done")

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
GRAPH_LABELS = {"ER": "Erdős–Rényi ($p=0.5$)",
                "BA": "Barabási–Albert ($m=2$)",
                "REG": "Regular"}

NODE_COLORS = {
    4:  "#1f77b4",
    10: "#ff7f0e",
    20: "#2ca02c",
    30: "#d62728",
    40: "#9467bd",
    50: "#8c564b",
}

def make_figure(df: pd.DataFrame, outfile: str) -> None:
    graph_types = ["ER", "BA", "REG"]
    t_cols  = [c for c in df.columns if c.startswith("ratio_t")]
    t_vals  = np.array([float(c[len("ratio_t"):]) for c in t_cols])

    node_sizes = sorted(df["num_nodes"].unique())

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    fig.subplots_adjust(wspace=0.06, left=0.07, right=0.97,
                        top=0.88, bottom=0.16)

    for ax, gtype in zip(axes, graph_types):
        sub = df[df["graph_type"] == gtype]

        for n in node_sizes:
            color = NODE_COLORS.get(n, "#333333")
            sub_n = sub[sub["num_nodes"] == n]
            if sub_n.empty:
                continue

            means = sub_n[t_cols].mean().values
            stds  = sub_n[t_cols].std().values

            ax.plot(t_vals, means, color=color, linewidth=1.8,
                    label=f"$N={n}$")
            ax.fill_between(t_vals, means - stds, means + stds,
                            color=color, alpha=0.12)

            # Mark t_opt for this N
            t_th = t_opt(n)
            # Find closest t_val to theory
            idx_th = np.argmin(np.abs(t_vals - t_th))
            ax.axvline(t_th, color=color, linestyle=":", linewidth=0.8, alpha=0.5)

        # Mark the fixed t=0.01 used in the main experiments
        ax.axvline(0.01, color="black", linestyle="--", linewidth=1.0,
                   alpha=0.8, label="$t=0.01$ (paper)")

        ax.axhline(1.0, color="black", linestyle="-", linewidth=0.6,
                   alpha=0.4)
        ax.set_title(GRAPH_LABELS[gtype], fontsize=10.5)
        ax.set_xlabel("Evolution time $t$", fontsize=10)
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(
            mticker.LogFormatterSciNotation(labelOnlyBase=False))
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        ax.grid(axis="x", linestyle=":", alpha=0.3)
        ax.set_ylim(bottom=0.90)

    axes[0].set_ylabel("Approximation ratio  $|C|/|C^*|$", fontsize=10)

    # Unified legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(node_sizes) + 1,
               fontsize=9, framealpha=0.9, bbox_to_anchor=(0.52, 1.02))

    fig.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved: {outfile}")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def print_summary(df: pd.DataFrame) -> None:
    t_cols = [c for c in df.columns if c.startswith("ratio_t")]
    t_vals = np.array([float(c[len("ratio_t"):]) for c in t_cols])

    print("\n" + "=" * 60)
    print("  Optimal t per graph type (minimum mean ratio)")
    print("=" * 60)
    for gtype in ["ER", "BA", "REG"]:
        sub = df[df["graph_type"] == gtype]
        means = sub[t_cols].mean().values
        best_idx = np.argmin(means)
        print(f"  {gtype:>4}: best t = {t_vals[best_idx]:.4f}  "
              f"(ratio = {means[best_idx]:.4f})")

    print(f"\n  Ratio at t=0.01:")
    t01_col = t_cols[np.argmin(np.abs(t_vals - 0.01))]
    for gtype in ["ER", "BA", "REG"]:
        sub = df[df["graph_type"] == gtype]
        m = sub[t01_col].mean()
        print(f"  {gtype:>4}: {m:.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--nodes", type=int, nargs="+",
                   default=[4, 10, 20, 30, 40, 50])
    p.add_argument("--graphs", type=int, default=10,
                   help="Graph instances per (N, type) setting")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"Node sizes : {args.nodes}")
    print(f"Instances  : {args.graphs}")
    print(f"t values   : {len(T_VALUES)} points in [{T_VALUES[0]:.3f}, {T_VALUES[-1]:.1f}]")

    df = build_dataset(args.nodes, args.graphs)
    df.to_csv(RESULTS_FILE, index=False)
    print(f"\nResults saved: {RESULTS_FILE}  ({len(df)} rows)")

    print_summary(df)
    make_figure(df, FIGURE_FILE)


if __name__ == "__main__":
    main()
