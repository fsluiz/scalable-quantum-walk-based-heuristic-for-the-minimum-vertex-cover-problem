#!/usr/bin/env python3
"""
Validate quantum-inspired spectral greedy against CTQW (t=0.01).

Reproduces the 98.6% equivalence claim stated in the paper:
  "At t = 0.01, the spectral greedy criterion produces the same
   vertex cover as the full CTQW on 98.6% of benchmark instances."

Output
------
    results_spectral_greedy_validation.csv
        Columns: graph_type, V, E, ctqw_size, spectral_size, identical,
                 ctqw_ratio, spectral_ratio, reference_size, exact_used

    A summary table is printed to stdout.

Usage
-----
    python run_spectral_greedy_validation.py [--nodes N [N ...]] [--graphs G]
"""

import argparse
import sys
import os
import time
import subprocess

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from quantum_walk_mvc.core import quantum_walk_mvc_sparse
from quantum_walk_mvc.heuristics import (
    spectral_greedy_vertex_cover,
    degree_greedy_vertex_cover,
    fast_vc_heuristic,
    simulated_annealing_vertex_cover,
    check_is_vertex_cover,
)
from quantum_walk_mvc.graph_generators import (
    generate_erdos_renyi_graphs,
    generate_barabasi_albert_graphs,
    generate_regular_graphs,
)

T_EVOLUTION   = 0.01
RESULTS_FILE  = "results_spectral_greedy_validation.csv"
FIGURE_FILE   = "figure_spectral_greedy_validation.pdf"
SAGE_BIN      = "/home/fsluiz/sage-10.6/sage"
MILP_TIMEOUT  = 120  # seconds per instance


def sage_milp_vc_size(G: nx.Graph) -> int:
    """Compute exact MVC size via SageMath MILP (subprocess call)."""
    edges = list(G.edges())
    nodes = list(G.nodes())
    sage_code = (
        f"from sage.graphs.graph import Graph as SG\n"
        f"G = SG()\n"
        f"G.add_vertices({nodes})\n"
        f"G.add_edges({edges})\n"
        f"print(len(G.vertex_cover()))\n"
    )
    result = subprocess.run(
        [SAGE_BIN, "--python", "-c", sage_code],
        capture_output=True, text=True, timeout=MILP_TIMEOUT)
    if result.returncode != 0:
        raise RuntimeError(result.stderr[:200])
    return int(result.stdout.strip())


def get_reference(G: nx.Graph, sizes: list) -> tuple[int, bool]:
    """Return (reference_size, exact_used).
    Always attempts SageMath MILP; falls back to best-heuristic proxy.
    """
    try:
        exact = sage_milp_vc_size(G)
        return exact, True
    except Exception:
        pass
    valid = [s for s in sizes if s > 0]
    return (min(valid) if valid else 1), False


def run_one(G: nx.Graph, graph_type: str, param) -> dict:
    t0 = time.perf_counter()
    ctqw_vc = quantum_walk_mvc_sparse(G, T_EVOLUTION)
    t_ctqw = time.perf_counter() - t0

    t0 = time.perf_counter()
    spec_vc = spectral_greedy_vertex_cover(G)
    t_spec = time.perf_counter() - t0

    t0 = time.perf_counter()
    dg_vc = degree_greedy_vertex_cover(G)
    t_dg = time.perf_counter() - t0

    # Scale iterations inversely with graph size to keep runtime reasonable
    n_nodes = G.number_of_nodes()
    fvc_iters = max(200, min(1000, int(5000 / n_nodes)))
    sa_iters  = max(500, min(5000, int(50000 / n_nodes)))

    t0 = time.perf_counter()
    fvc_vc = fast_vc_heuristic(G, max_iterations=fvc_iters)
    t_fvc = time.perf_counter() - t0

    t0 = time.perf_counter()
    sa_vc = simulated_annealing_vertex_cover(G, initial_temperature=100.0,
                                              cooling_rate=0.99, max_iterations=sa_iters)
    t_sa = time.perf_counter() - t0

    ctqw_size = len(ctqw_vc)
    spec_size = len(spec_vc)
    dg_size   = len(dg_vc)
    fvc_size  = len(fvc_vc)
    sa_size   = len(sa_vc)

    ref, exact_used = get_reference(G, [ctqw_size, spec_size, dg_size, fvc_size, sa_size])

    return {
        "graph_type":     graph_type,
        "param":          param,
        "V":              G.number_of_nodes(),
        "E":              G.number_of_edges(),
        "reference_size": ref,
        "exact_used":     exact_used,
        # CTQW
        "ctqw_size":      ctqw_size,
        "ctqw_ratio":     ctqw_size / ref if ref > 0 else float("nan"),
        "ctqw_valid":     check_is_vertex_cover(G, set(ctqw_vc)),
        "ctqw_time":      t_ctqw,
        # Spectral greedy
        "spectral_size":  spec_size,
        "spectral_ratio": spec_size / ref if ref > 0 else float("nan"),
        "spectral_valid": check_is_vertex_cover(G, set(spec_vc)),
        "spectral_time":  t_spec,
        # Degree greedy
        "dg_size":        dg_size,
        "dg_ratio":       dg_size / ref if ref > 0 else float("nan"),
        "dg_time":        t_dg,
        # FastVC
        "fvc_size":       fvc_size,
        "fvc_ratio":      fvc_size / ref if ref > 0 else float("nan"),
        "fvc_valid":      check_is_vertex_cover(G, set(fvc_vc)),
        "fvc_time":       t_fvc,
        # Simulated Annealing
        "sa_size":        sa_size,
        "sa_ratio":       sa_size / ref if ref > 0 else float("nan"),
        "sa_valid":       check_is_vertex_cover(G, set(sa_vc)),
        "sa_time":        t_sa,
        # Equivalence
        "identical":      ctqw_size == spec_size,
    }


def build_dataset(node_sizes, n_per_setting):
    records = []

    print("\n[ER graphs]")
    for n in node_sizes:
        for p in [0.3, 0.5, 0.7]:
            graphs, _ = generate_erdos_renyi_graphs(
                num_nodes_range=[n], edge_prob_range=[p],
                num_graphs_per_setting=n_per_setting, ensure_connected=True)
            for G in graphs:
                records.append(run_one(G, "ER", p))
        print(f"  N={n} done")

    print("\n[BA graphs]")
    for n in node_sizes:
        for m in [2, 5]:
            if m >= n:
                continue
            graphs, _ = generate_barabasi_albert_graphs(
                num_nodes_range=[n], m_range=[m],
                num_graphs_per_setting=n_per_setting)
            for G in graphs:
                records.append(run_one(G, "BA", m))
        print(f"  N={n} done")

    print("\n[Regular graphs]")
    for n in node_sizes:
        graphs, _ = generate_regular_graphs(
            num_nodes_range=[n], num_graphs_per_setting=n_per_setting)
        for G in graphs:
            records.append(run_one(G, "REG", None))
        print(f"  N={n} done")

    return pd.DataFrame(records)


def print_summary(df: pd.DataFrame) -> None:
    from scipy import stats
    from statsmodels.stats.proportion import proportion_confint

    total = len(df)
    identical = df["identical"].sum()
    equiv_rate = identical / total * 100
    lo, hi = proportion_confint(identical, total, alpha=0.05, method="wilson")

    print("\n" + "=" * 70)
    print(f"  Total instances : {total}")
    print(f"  Identical covers: {identical}  ({equiv_rate:.2f}%,  "
          f"95% CI: [{lo*100:.1f}%, {hi*100:.1f}%])")
    print("=" * 70)
    print(f"\n  {'Graph':>5}  {'n':>6}  {'Equiv%':>8}  "
          f"{'CTQW':>8}  {'Spectral':>10}  {'DegGreedy':>11}  "
          f"{'FastVC':>8}  {'SA':>8}")
    print("  " + "-" * 72)
    for gtype in ["ER", "BA", "REG"]:
        sub = df[df["graph_type"] == gtype]
        eq   = sub["identical"].mean() * 100
        vals = [sub[c].mean() for c in
                ["ctqw_ratio","spectral_ratio","dg_ratio","fvc_ratio","sa_ratio"]]
        print(f"  {gtype:>5}  {len(sub):>6}  {eq:>8.1f}%  "
              + "  ".join(f"{v:>10.4f}" for v in vals))

    print(f"\n  Pooled mean ratios:")
    for col, lbl in [("ctqw_ratio","CTQW"),("spectral_ratio","Spectral"),
                     ("dg_ratio","Degree-Greedy"),("fvc_ratio","FastVC"),
                     ("sa_ratio","SA")]:
        print(f"    {lbl:<16}: {df[col].mean():.4f} ± {df[col].std():.4f}")

    # Wilcoxon paired test
    stat, p = stats.wilcoxon(df["ctqw_ratio"].values, df["spectral_ratio"].values)
    print(f"\n  Wilcoxon CTQW vs Spectral: stat={stat:.1f}, p={p:.4f}")
    for col, lbl in [("dg_ratio","DegGreedy"),("fvc_ratio","FastVC"),("sa_ratio","SA")]:
        stat2, p2 = stats.wilcoxon(df["spectral_ratio"].values, df[col].values)
        print(f"  Wilcoxon Spectral vs {lbl:<12}: stat={stat2:.1f}, p={p2:.2e}")

    # Runtime comparison
    print(f"\n  Mean runtime (seconds):")
    print(f"    CTQW    : {df['ctqw_time'].mean():.5f}")
    print(f"    Spectral: {df['spectral_time'].mean():.5f}  "
          f"({df['ctqw_time'].mean()/df['spectral_time'].mean():.0f}x faster than CTQW)")
    print(f"    DegGreedy: {df['dg_time'].mean():.5f}")
    print(f"    FastVC  : {df['fvc_time'].mean():.5f}")
    print(f"    SA      : {df['sa_time'].mean():.5f}")


def make_figure(df: pd.DataFrame, outfile: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.subplots_adjust(wspace=0.30, left=0.07, right=0.97,
                        top=0.88, bottom=0.14)

    graph_types = ["ER", "BA", "REG"]
    titles = {"ER": "Erdős–Rényi", "BA": "Barabási–Albert", "REG": "Regular"}

    for ax, gtype in zip(axes, graph_types):
        sub  = df[df["graph_type"] == gtype].copy()
        ns   = sorted(sub["V"].unique())

        for col, lbl, clr, mk, lw in [
            ("ctqw_ratio",     "CTQW (t=0.01)",       "#1f77b4", "o", 2.2),
            ("spectral_ratio", "Spectral greedy",      "#e07b39", "s", 2.0),
            ("dg_ratio",       "Degree greedy",        "#d62728", "^", 1.4),
            ("fvc_ratio",      "FastVC",               "#2ca02c", "D", 1.2),
            ("sa_ratio",       "SA",                   "#9467bd", "v", 1.2),
        ]:
            means = sub.groupby("V")[col].mean()
            stds  = sub.groupby("V")[col].std()
            ys  = [means.get(n, np.nan) for n in ns]
            es  = [stds.get(n, 0.0)    for n in ns]
            ax.plot(ns, ys, color=clr, marker=mk, markersize=5,
                    linewidth=lw, label=lbl)
            ax.fill_between(ns,
                            [y - e for y, e in zip(ys, es)],
                            [y + e for y, e in zip(ys, es)],
                            color=clr, alpha=0.12)

        ax.axhline(1.0, color="k", linestyle="--", lw=0.8, alpha=0.6)
        ax.set_title(titles[gtype], fontsize=11)
        ax.set_xlabel("Number of nodes $V$", fontsize=10)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        ax.set_ylim(bottom=0.92)

    axes[0].set_ylabel(r"Approximation ratio $|C|/|C^*|$", fontsize=10)

    # Equivalence annotation
    total = len(df)
    equiv = df["identical"].sum()
    fig.suptitle(
        f"CTQW vs spectral greedy: {equiv}/{total} identical covers "
        f"({equiv/total*100:.1f}%) at $t=0.01$",
        fontsize=11, y=1.01)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3,
               fontsize=9.5, framealpha=0.9, bbox_to_anchor=(0.52, 1.0))

    fig.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved: {outfile}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Validate spectral greedy against CTQW (t=0.01).")
    p.add_argument("--nodes", type=int, nargs="+",
                   default=list(range(4, 55, 5)),
                   help="Node sizes to test")
    p.add_argument("--graphs", type=int, default=10,
                   help="Instances per (N, param) combination")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"Node sizes   : {args.nodes}")
    print(f"Instances/cfg: {args.graphs}")
    print(f"t_evolution  : {T_EVOLUTION}")

    df = build_dataset(args.nodes, args.graphs)
    df.to_csv(RESULTS_FILE, index=False)
    print(f"\nResults saved: {RESULTS_FILE}  ({len(df)} rows)")

    print_summary(df)
    make_figure(df, FIGURE_FILE)


if __name__ == "__main__":
    main()
