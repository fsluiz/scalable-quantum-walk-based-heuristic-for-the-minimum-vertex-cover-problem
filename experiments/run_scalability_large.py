#!/usr/bin/env python3
"""
Large-scale scalability benchmark for the quantum-inspired spectral greedy MVC heuristic.

Tests spectral greedy (Algorithm 2) against degree-greedy and SA on synthetic
graph ensembles with V ∈ {1k, 2k, 5k, 10k, 20k, 50k, 100k} using igraph for
fast graph operations.  CTQW is excluded at this scale (O(V³) per step).

Since MILP exact solutions are unavailable for V > 200, quality is measured via:
  - Virtual Best Solver (VBS): min cover size across all heuristics
  - Pairwise win/tie/loss statistics

Outputs
-------
    results_scalability_large.csv
    figure_scalability_runtime.pdf
    figure_scalability_quality.pdf

Usage
-----
    python run_scalability_large.py [--nodes N [N ...]] [--graphs G]
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import igraph as ig
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from quantum_walk_mvc.heuristics_igraph import (
    spectral_greedy_large,
    degree_greedy_large,
    check_is_vertex_cover_igraph,
    warmup_jit,
)

RESULTS_FILE = "results_scalability_large.csv"


# ---------------------------------------------------------------------------
# SA for igraph (sparse, incremental)
# ---------------------------------------------------------------------------

def simulated_annealing_igraph(g: ig.Graph,
                                initial_temperature: float = 50.0,
                                cooling_rate: float = 0.995,
                                max_iterations: int = 2000) -> list:
    """SA heuristic adapted for igraph. Budget scales with graph size."""
    import random
    n = g.vcount()
    adj = [set(g.neighbors(v)) for v in range(n)]

    # Initialise with degree greedy
    cover = set(degree_greedy_large(g))
    best_cover = set(cover)

    temperature = initial_temperature
    for _ in range(max_iterations):
        if temperature < 0.01:
            break
        action = random.choice(["add", "remove"])
        if action == "remove" and cover:
            v = random.choice(list(cover))
            new_cover = cover - {v}
            if check_is_vertex_cover_igraph(g, new_cover):
                cover = new_cover
                if len(cover) < len(best_cover):
                    best_cover = set(cover)
            else:
                delta = 1
                if random.random() < np.exp(-delta / temperature):
                    cover = new_cover
        elif action == "add":
            candidates = [v for v in range(n) if v not in cover]
            if candidates:
                v = random.choice(candidates)
                cover = cover | {v}
                if len(cover) < len(best_cover):
                    best_cover = set(cover)
        temperature *= cooling_rate

    # Repair if needed
    if not check_is_vertex_cover_igraph(g, best_cover):
        for u, v in g.get_edgelist():
            if u not in best_cover and v not in best_cover:
                best_cover.add(u if g.degree(u) >= g.degree(v) else v)

    return list(best_cover)


# ---------------------------------------------------------------------------
# Graph generators using igraph
# ---------------------------------------------------------------------------

def gen_ba(n: int, m: int) -> ig.Graph:
    return ig.Graph.Barabasi(n, m, directed=False)

def gen_er(n: int, p: float) -> ig.Graph:
    g = ig.Graph.Erdos_Renyi(n, p, directed=False)
    while not g.is_connected():
        g = ig.Graph.Erdos_Renyi(n, p, directed=False)
    return g

def gen_regular(n: int, k: int) -> ig.Graph:
    """k-regular graph; uses igraph's K_Regular generator."""
    return ig.Graph.K_Regular(n, k)


# ---------------------------------------------------------------------------
# Single-instance runner
# ---------------------------------------------------------------------------

def run_one(g: ig.Graph, graph_type: str, param) -> dict:
    n = g.vcount()
    sa_iters = max(500, min(3000, int(30000 / max(n, 1))))

    results = {}
    for name, fn, kwargs in [
        ("spectral", spectral_greedy_large, {}),
        ("dg",       degree_greedy_large,   {}),
        ("sa",       simulated_annealing_igraph,
                     {"max_iterations": sa_iters}),
    ]:
        t0 = time.perf_counter()
        cover = fn(g, **kwargs)
        elapsed = time.perf_counter() - t0
        size = len(cover)
        valid = check_is_vertex_cover_igraph(g, set(cover))
        results[name] = {"size": size, "time": elapsed, "valid": valid}

    vbs = min(results[k]["size"] for k in results)

    return {
        "graph_type":      graph_type,
        "param":           param,
        "V":               n,
        "E":               g.ecount(),
        "vbs_size":        vbs,
        # Spectral
        "sp_size":         results["spectral"]["size"],
        "sp_ratio":        results["spectral"]["size"] / vbs if vbs > 0 else float("nan"),
        "sp_valid":        results["spectral"]["valid"],
        "sp_time":         results["spectral"]["time"],
        # Degree greedy
        "dg_size":         results["dg"]["size"],
        "dg_ratio":        results["dg"]["size"] / vbs if vbs > 0 else float("nan"),
        "dg_valid":        results["dg"]["valid"],
        "dg_time":         results["dg"]["time"],
        # SA
        "sa_size":         results["sa"]["size"],
        "sa_ratio":        results["sa"]["size"] / vbs if vbs > 0 else float("nan"),
        "sa_valid":        results["sa"]["valid"],
        "sa_time":         results["sa"]["time"],
        # Pairwise: spectral vs dg
        "sp_lt_dg":        int(results["spectral"]["size"] < results["dg"]["size"]),
        "sp_eq_dg":        int(results["spectral"]["size"] == results["dg"]["size"]),
        "sp_gt_dg":        int(results["spectral"]["size"] > results["dg"]["size"]),
    }


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_dataset(node_sizes, n_per_setting):
    records = []

    print("\n[BA graphs]")
    for n in node_sizes:
        for m in [2, 5]:
            if m >= n:
                continue
            for _ in range(n_per_setting):
                g = gen_ba(n, m)
                records.append(run_one(g, "BA", m))
        print(f"  N={n} done", flush=True)

    print("\n[ER graphs]")
    for n in node_sizes:
        for p in [0.01, 0.05]:
            # Skip if too dense (memory / time)
            if n * p > 20:
                p_actual = 15.0 / n
            else:
                p_actual = p
            for _ in range(n_per_setting):
                try:
                    g = gen_er(n, p_actual)
                    records.append(run_one(g, "ER", round(p_actual, 4)))
                except Exception as e:
                    print(f"    ER n={n} p={p_actual:.4f} failed: {e}")
        print(f"  N={n} done", flush=True)

    print("\n[Regular graphs]")
    for n in node_sizes:
        for k in [3, 4]:
            if n < k + 2 or (n * k) % 2 != 0:
                continue
            for _ in range(n_per_setting):
                try:
                    g = gen_regular(n, k)
                    records.append(run_one(g, "REG", k))
                except Exception as e:
                    print(f"    REG n={n} k={k} failed: {e}")
        print(f"  N={n} done", flush=True)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print(f"  Total instances : {len(df)}")
    for gtype in ["BA", "ER", "REG"]:
        sub = df[df["graph_type"] == gtype]
        if len(sub) == 0:
            continue
        sp_lt = sub["sp_lt_dg"].sum()
        sp_eq = sub["sp_eq_dg"].sum()
        sp_gt = sub["sp_gt_dg"].sum()
        print(f"\n  {gtype} ({len(sub)} instances):")
        print(f"    sp vs dg: {sp_lt} wins / {sp_eq} ties / {sp_gt} losses")
        for col, lbl in [("sp_ratio","Spectral"),("dg_ratio","DG"),("sa_ratio","SA")]:
            print(f"    {lbl:10s} VBS-ratio: {sub[col].mean():.4f} ± {sub[col].std():.4f}")

    print("\n  Pooled VBS-ratios:")
    for col, lbl in [("sp_ratio","Spectral"),("dg_ratio","DG"),("sa_ratio","SA")]:
        print(f"    {lbl:10s}: {df[col].mean():.4f} ± {df[col].std():.4f}")

    print("\n  Mean runtime (s):")
    for ns in sorted(df["V"].unique()):
        sub = df[df["V"] == ns]
        print(f"    V={ns:7d}  sp={sub['sp_time'].mean():7.3f}  "
              f"dg={sub['dg_time'].mean():7.3f}  sa={sub['sa_time'].mean():7.3f}")

    # Runtime scaling exponent (log-log regression on BA m=2)
    ba2 = df[(df["graph_type"] == "BA") & (df["param"] == 2)].copy()
    if len(ba2) > 4:
        vgs = ba2.groupby("V")["sp_time"].mean()
        log_v = np.log(vgs.index.values)
        log_t = np.log(vgs.values)
        slope, intercept, r, _, _ = stats.linregress(log_v, log_t)
        print(f"\n  Spectral runtime scaling (BA m=2, log-log): "
              f"exponent={slope:.2f}, R²={r**2:.3f}")

    # Wilcoxon spectral vs DG (if variable)
    diff = df["sp_size"] - df["dg_size"]
    if diff.std() > 0:
        stat, p = stats.wilcoxon(df["sp_size"], df["dg_size"], alternative="less")
        print(f"\n  Wilcoxon spectral < DG: W={stat:.0f}, p={p:.2e}")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def make_figures(df: pd.DataFrame) -> None:
    # --- Figure 1: Runtime scaling ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.subplots_adjust(wspace=0.30, left=0.08, right=0.97, top=0.88, bottom=0.14)

    ba2 = df[(df["graph_type"] == "BA") & (df["param"] == 2)]
    ns = sorted(ba2["V"].unique())

    for col, lbl, clr, mk in [
        ("sp_time", "Spectral greedy", "#e07b39", "s"),
        ("dg_time", "Degree greedy",   "#d62728", "^"),
        ("sa_time", "SA",              "#9467bd", "v"),
    ]:
        means = ba2.groupby("V")[col].mean()
        ys = [means.get(n, np.nan) for n in ns]
        axes[0].plot(ns, ys, marker=mk, label=lbl, linewidth=1.8, markersize=5)

    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_xlabel("Number of vertices $V$", fontsize=10)
    axes[0].set_ylabel("Wall-clock time (s)", fontsize=10)
    axes[0].set_title("Runtime scaling (BA, $m=2$)", fontsize=11)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, which="both", linestyle=":", alpha=0.5)

    # Speedup ratio sp vs dg
    sp_means = ba2.groupby("V")["sp_time"].mean()
    dg_means = ba2.groupby("V")["dg_time"].mean()
    speedups = [dg_means.get(n, np.nan) / sp_means.get(n, 1.0) for n in ns]
    axes[1].plot(ns, speedups, color="#1f77b4", marker="o", linewidth=2)
    axes[1].axhline(1.0, color="k", linestyle="--", lw=0.8, alpha=0.6)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Number of vertices $V$", fontsize=10)
    axes[1].set_ylabel("Speedup vs degree-greedy", fontsize=10)
    axes[1].set_title("Spectral vs DG speedup (BA, $m=2$)", fontsize=11)
    axes[1].grid(axis="y", linestyle=":", alpha=0.5)

    fig.suptitle("Large-scale runtime: spectral greedy (igraph + lazy heap)",
                 fontsize=11, y=1.02)
    fig.savefig("figure_scalability_runtime.pdf", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # --- Figure 2: Cover quality (VBS ratio) ---
    gtypes = [g for g in ["BA", "ER", "REG"] if g in df["graph_type"].values]
    fig, axes = plt.subplots(1, len(gtypes), figsize=(5 * len(gtypes), 4.5))
    if len(gtypes) == 1:
        axes = [axes]
    fig.subplots_adjust(wspace=0.28, left=0.07, right=0.97, top=0.88, bottom=0.14)
    titles = {"BA": "Barabási–Albert", "ER": "Erdős–Rényi", "REG": "Regular"}

    for ax, gtype in zip(axes, gtypes):
        sub = df[df["graph_type"] == gtype]
        ns_g = sorted(sub["V"].unique())
        for col, lbl, clr, mk, lw in [
            ("sp_ratio", "Spectral greedy", "#e07b39", "s", 2.2),
            ("dg_ratio", "Degree greedy",   "#d62728", "^", 1.4),
            ("sa_ratio", "SA",              "#9467bd", "v", 1.2),
        ]:
            means = sub.groupby("V")[col].mean()
            stds  = sub.groupby("V")[col].std().fillna(0)
            ys = [means.get(n, np.nan) for n in ns_g]
            es = [stds.get(n, 0.0) for n in ns_g]
            ax.plot(ns_g, ys, color=clr, marker=mk, markersize=5,
                    linewidth=lw, label=lbl)
            ax.fill_between(ns_g,
                            [y - e for y, e in zip(ys, es)],
                            [y + e for y, e in zip(ys, es)],
                            color=clr, alpha=0.12)
        ax.axhline(1.0, color="k", linestyle="--", lw=0.8, alpha=0.6)
        ax.set_title(titles.get(gtype, gtype), fontsize=11)
        ax.set_xlabel("Number of vertices $V$", fontsize=10)
        ax.set_xscale("log")
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        ax.set_ylim(bottom=0.98)
    axes[0].set_ylabel(r"Ratio $|C|/|C_{\rm VBS}|$", fontsize=10)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3,
               fontsize=9.5, framealpha=0.9, bbox_to_anchor=(0.52, 1.0))
    fig.suptitle("Cover quality vs. virtual best solver (VBS), "
                 "$V \\in [10^3, 10^5]$", fontsize=11, y=1.03)
    fig.savefig("figure_scalability_quality.pdf", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("\nFigures saved: figure_scalability_runtime.pdf, figure_scalability_quality.pdf")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Large-scale MVC scalability benchmark.")
    p.add_argument("--nodes", type=int, nargs="+",
                   default=[1000, 2000, 5000, 10000, 20000, 50000, 100000],
                   help="Vertex counts to test")
    p.add_argument("--graphs", type=int, default=5,
                   help="Instances per (V, param) combination")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"Node sizes   : {args.nodes}")
    print(f"Instances/cfg: {args.graphs}")

    print("\nWarming up Numba JIT...", end=" ", flush=True)
    warmup_jit()
    print("done")

    df = build_dataset(args.nodes, args.graphs)
    df.to_csv(RESULTS_FILE, index=False)
    print(f"\nResults saved: {RESULTS_FILE}  ({len(df)} rows)")

    print_summary(df)
    make_figures(df)


if __name__ == "__main__":
    main()
