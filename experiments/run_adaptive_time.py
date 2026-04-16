#!/usr/bin/env python3
"""
Adaptive-time CTQW-MVC experiment.

At each greedy step, if the confidence margin delta_k < tau,
try increasing evolution times t_1 < t_2 < ... < t_max until
delta >= tau or t_max is reached.  Use the t that maximises delta.

Compares:
    baseline  — fixed t = 0.01 (published algorithm)
    adaptive  — t chosen per-step from T_GRID

Outputs
-------
    results_adaptive_time.csv      — per-instance summary
    figure_adaptive_time.pdf       — comparison figure
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import networkx as nx
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from quantum_walk_mvc.graph_generators import (
    generate_erdos_renyi_graphs,
    generate_barabasi_albert_graphs,
    generate_regular_graphs,
)
from quantum_walk_mvc.heuristics import check_is_vertex_cover

# ── configuration ────────────────────────────────────────────────────────────
T_BASE        = 0.01
T_GRID        = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
TAU           = 0.05          # confidence threshold; try larger t if delta < tau
BENCHMARK_CSV = "../results/results_degree_greedy_comparison.csv"
RESULTS_FILE  = "results_adaptive_time.csv"
FIGURE_FILE   = "figure_adaptive_time.pdf"
# ─────────────────────────────────────────────────────────────────────────────


def _probs_from_gamma(adj_csr, t: float):
    """Return (probs, active_mask, degrees) for current adjacency at time t."""
    degrees = np.array(adj_csr.sum(axis=1)).flatten()
    active_mask = degrees > 0
    deg_safe = degrees.copy()
    deg_safe[deg_safe == 0] = 1e-10
    D_inv_sqrt = sp.diags(1.0 / np.sqrt(deg_safe))
    Gamma = D_inv_sqrt @ adj_csr @ D_inv_sqrt
    U = spla.expm(1j * t * Gamma)
    probs = 1.0 - np.abs(U.diagonal()) ** 2
    return probs, active_mask, degrees, Gamma


def ctqw_fixed(G: nx.Graph, t: float = T_BASE) -> list:
    """Original algorithm — fixed t."""
    adj = nx.to_scipy_sparse_array(G, dtype=float, format="csr")
    cover = []
    while adj.sum() > 0:
        probs, active_mask, _, _ = _probs_from_gamma(adj, t)
        probs[~active_mask] = -1.0
        v = int(np.argmax(probs))
        cover.append(v)
        adj = adj.tolil()
        adj[v, :] = 0
        adj[:, v] = 0
        adj = adj.tocsr()
    return cover


def ctqw_adaptive(G: nx.Graph, tau: float = TAU,
                  t_grid=None) -> dict:
    """
    Adaptive-t algorithm.

    Returns
    -------
    dict with keys:
        cover        : list of selected vertices
        t_used       : list of t chosen at each step
        delta_used   : list of delta achieved at each step
        n_adapted    : number of steps where t > T_BASE was chosen
    """
    if t_grid is None:
        t_grid = T_GRID

    adj = nx.to_scipy_sparse_array(G, dtype=float, format="csr")
    cover, t_used, delta_used = [], [], []

    while adj.sum() > 0:
        degrees = np.array(adj.sum(axis=1)).flatten()
        active_mask = degrees > 0
        deg_safe = degrees.copy()
        deg_safe[deg_safe == 0] = 1e-10
        D_inv_sqrt = sp.diags(1.0 / np.sqrt(deg_safe))
        adj_csr = adj if sp.issparse(adj) else adj.tocsr()
        Gamma = D_inv_sqrt @ adj_csr @ D_inv_sqrt

        best_t, best_v, best_delta = t_grid[0], None, -1.0

        for t in t_grid:
            U = spla.expm(1j * t * Gamma)
            probs = 1.0 - np.abs(U.diagonal()) ** 2
            probs_a = probs.copy()
            probs_a[~active_mask] = -1.0

            sorted_p = np.sort(probs_a[active_mask])[::-1]
            delta = float(sorted_p[0] - (sorted_p[1] if len(sorted_p) > 1 else 0.0))

            if delta > best_delta:
                best_delta = delta
                best_t = t
                best_v = int(np.argmax(probs_a))

            if delta >= tau:
                break   # good enough — stop trying larger t

        cover.append(best_v)
        t_used.append(best_t)
        delta_used.append(best_delta)

        adj = adj.tolil()
        adj[best_v, :] = 0
        adj[:, best_v] = 0
        adj = adj.tocsr()

    n_adapted = sum(1 for t in t_used if t > T_BASE)
    return {
        "cover":      cover,
        "t_used":     t_used,
        "delta_used": delta_used,
        "n_adapted":  n_adapted,
    }


# ── graph reconstruction ──────────────────────────────────────────────────────
def build_graph_list(n_per_setting=10):
    graphs = []
    node_sizes = list(range(4, 55, 5))

    for n in node_sizes:
        for p in [0.3, 0.5, 0.7]:
            gs, _ = generate_erdos_renyi_graphs(
                num_nodes_range=[n], edge_prob_range=[p],
                num_graphs_per_setting=n_per_setting, ensure_connected=True)
            for G in gs:
                graphs.append(("ER", n, float(p), G))

    for n in node_sizes:
        for m in [2, 5]:
            if m >= n:
                continue
            gs, _ = generate_barabasi_albert_graphs(
                num_nodes_range=[n], m_range=[m],
                num_graphs_per_setting=n_per_setting)
            for G in gs:
                graphs.append(("BA", n, float(m), G))

    for n in node_sizes:
        gs, _ = generate_regular_graphs(
            num_nodes_range=[n], num_graphs_per_setting=n_per_setting)
        for G in gs:
            graphs.append(("REG", n, None, G))

    return graphs


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ref_df = pd.read_csv(BENCHMARK_CSV)
    ref_lookup, graph_counters = {}, {}
    for _, row in ref_df.iterrows():
        gt = row["graph_type"]
        n  = int(row["num_nodes"])
        p  = float(row["param"]) if str(row["param"]) not in ("nan", "", "None") else None
        e  = int(row["num_edges"])
        key = (gt, n, p, e)
        ref_lookup.setdefault(key, []).append(int(row["reference_size"]))

    print("Building graph list...")
    all_graphs = build_graph_list()
    print(f"Analysing {len(all_graphs)} instances...\n")

    records = []
    inst_counters = {}
    t0 = time.time()

    for i, (gt, n, param, G) in enumerate(all_graphs):
        e   = G.number_of_edges()
        p   = float(param) if param is not None else None
        key = (gt, n, p, e)

        ci = inst_counters.get(key, 0)
        inst_counters[key] = ci + 1

        milp_ref = None
        if key in ref_lookup and ci < len(ref_lookup[key]):
            milp_ref = ref_lookup[key][ci]

        # baseline
        cov_fix = ctqw_fixed(G, T_BASE)
        sz_fix  = len(cov_fix)
        r_fix   = sz_fix / milp_ref if milp_ref else float("nan")

        # adaptive
        res_adp  = ctqw_adaptive(G, tau=TAU)
        cov_adp  = res_adp["cover"]
        sz_adp   = len(cov_adp)
        r_adp    = sz_adp / milp_ref if milp_ref else float("nan")

        records.append({
            "graph_type":   gt,
            "num_nodes":    n,
            "param":        param,
            "num_edges":    e,
            "milp_ref":     milp_ref,
            # baseline
            "size_fixed":   sz_fix,
            "ratio_fixed":  r_fix,
            "opt_fixed":    (r_fix == 1.0) if milp_ref else None,
            # adaptive
            "size_adaptive": sz_adp,
            "ratio_adaptive": r_adp,
            "opt_adaptive":  (r_adp == 1.0) if milp_ref else None,
            "n_adapted":     res_adp["n_adapted"],
            "n_steps":       len(cov_adp),
            "mean_t_used":   float(np.mean(res_adp["t_used"])),
            "mean_delta":    float(np.mean(res_adp["delta_used"])),
        })

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{len(all_graphs)} done  ({elapsed:.1f}s)")

    df = pd.DataFrame(records)
    df.to_csv(RESULTS_FILE, index=False)
    print(f"\nSaved: {RESULTS_FILE}  ({len(df)} rows)")

    # ── summary ───────────────────────────────────────────────────────────────
    has_ref = df["milp_ref"].notna()
    d = df[has_ref]

    n_fix = (d["opt_fixed"] == True).sum()
    n_adp = (d["opt_adaptive"] == True).sum()
    total = len(d)

    print(f"\n{'':=<55}")
    print(f"  Fixed t={T_BASE}:   {n_fix}/{total} optimal  ({100*n_fix/total:.1f}%)")
    print(f"  Adaptive t:    {n_adp}/{total} optimal  ({100*n_adp/total:.1f}%)")
    print(f"  Improvement:   +{n_adp-n_fix} instances")

    improved = d[(d["opt_fixed"] == False) & (d["opt_adaptive"] == True)]
    regressed = d[(d["opt_fixed"] == True) & (d["opt_adaptive"] == False)]
    print(f"\n  Fixed→Optimal (fixed fixed by adaptive): {len(improved)}")
    print(f"  Optimal→Worse  (adaptive broke optimal):  {len(regressed)}")

    # breakdown by graph type
    print(f"\n  {'Type':>5}  {'Fixed%':>8}  {'Adap%':>8}  {'Δ':>6}")
    print(f"  {'':->5}  {'':->8}  {'':->8}  {'':->6}")
    for gt in ["ER", "BA", "REG"]:
        sub = d[d["graph_type"] == gt]
        if len(sub) == 0:
            continue
        pf = 100 * (sub["opt_fixed"] == True).sum() / len(sub)
        pa = 100 * (sub["opt_adaptive"] == True).sum() / len(sub)
        print(f"  {gt:>5}  {pf:>7.1f}%  {pa:>7.1f}%  {pa-pf:>+6.1f}%")

    print(f"\n  Mean ratio  — fixed:    {d['ratio_fixed'].mean():.5f}")
    print(f"  Mean ratio  — adaptive: {d['ratio_adaptive'].mean():.5f}")
    print(f"  Mean steps adapted per instance: "
          f"{d['n_adapted'].mean():.2f} / {d['n_steps'].mean():.2f}")
    print(f"  Mean t used (adaptive): {d['mean_t_used'].mean():.4f}")
    print(f"{'':=<55}\n")

    make_figure(df)


# ── figure ────────────────────────────────────────────────────────────────────
def make_figure(df: pd.DataFrame) -> None:
    has_ref = df["milp_ref"].notna()
    d = df[has_ref]

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.30,
                           left=0.09, right=0.97, top=0.93, bottom=0.10)
    colors = {"ER": "#1f77b4", "BA": "#ff7f0e", "REG": "#2ca02c"}

    # Panel A — ratio scatter: fixed vs adaptive
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(d["ratio_fixed"], d["ratio_adaptive"],
                c=[colors.get(g, "gray") for g in d["graph_type"]],
                s=12, alpha=0.5)
    r_max = max(d["ratio_fixed"].max(), d["ratio_adaptive"].max()) * 1.02
    ax1.plot([1, r_max], [1, r_max], "k--", linewidth=0.9, label="y = x")
    ax1.set_xlabel("Ratio (fixed t)", fontsize=10)
    ax1.set_ylabel("Ratio (adaptive t)", fontsize=10)
    ax1.set_title("(a) Fixed vs adaptive ratio", fontsize=11)
    from matplotlib.patches import Patch
    ax1.legend(handles=[Patch(color=c, label=g) for g, c in colors.items()],
               fontsize=8)
    ax1.grid(linestyle=":", alpha=0.5)

    # Panel B — histogram of ratios, both algorithms
    ax2 = fig.add_subplot(gs[0, 1])
    bins = np.linspace(1.0, max(d["ratio_fixed"].quantile(0.99),
                                d["ratio_adaptive"].quantile(0.99)) + 0.01, 40)
    ax2.hist(d["ratio_fixed"],    bins=bins, alpha=0.6, color="#d62728",
             label=f"Fixed t={T_BASE}", density=True)
    ax2.hist(d["ratio_adaptive"], bins=bins, alpha=0.6, color="#2ca02c",
             label="Adaptive t", density=True)
    ax2.set_xlabel("Approximation ratio", fontsize=10)
    ax2.set_ylabel("Density", fontsize=10)
    ax2.set_title("(b) Ratio distribution", fontsize=11)
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", linestyle=":", alpha=0.5)

    # Panel C — n_adapted per instance (how often adaptive kicks in)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.hist(d["n_adapted"] / d["n_steps"].clip(lower=1), bins=30,
             color="#9467bd", alpha=0.75, edgecolor="white")
    ax3.set_xlabel("Fraction of steps using t > 0.01", fontsize=10)
    ax3.set_ylabel("Count", fontsize=10)
    ax3.set_title("(c) How often adaptive t is triggered", fontsize=11)
    ax3.grid(axis="y", linestyle=":", alpha=0.5)

    # Panel D — mean_t_used vs ratio (adaptive)
    ax4 = fig.add_subplot(gs[1, 1])
    opt_mask  = d["opt_adaptive"] == True
    nopt_mask = d["opt_adaptive"] == False
    ax4.scatter(d.loc[opt_mask,  "mean_t_used"], d.loc[opt_mask,  "ratio_adaptive"],
                s=12, alpha=0.4, color="#2ca02c", label="Optimal")
    ax4.scatter(d.loc[nopt_mask, "mean_t_used"], d.loc[nopt_mask, "ratio_adaptive"],
                s=14, alpha=0.65, color="#d62728", label="Sub-optimal")
    ax4.set_xlabel("Mean t used (adaptive)", fontsize=10)
    ax4.set_ylabel("Ratio (adaptive)", fontsize=10)
    ax4.set_title("(d) Mean t vs solution quality", fontsize=11)
    ax4.legend(fontsize=8)
    ax4.grid(linestyle=":", alpha=0.5)

    fig.suptitle(
        f"Adaptive evolution time (τ={TAU})  vs  fixed t={T_BASE}",
        fontsize=12)
    fig.savefig(FIGURE_FILE, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {FIGURE_FILE}")


if __name__ == "__main__":
    main()
