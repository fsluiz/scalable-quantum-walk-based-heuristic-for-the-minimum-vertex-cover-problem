#!/usr/bin/env python3
"""
Confidence-gap analysis for CTQW-MVC.

At each iteration k, records:
    delta_k  = P(m*_k → out) − max_{m≠m*_k} P(m → out)   [confidence margin]
    gap_k    = second-smallest eigenvalue of H_k (spectral gap)
    prob_k   = P(m*_k → out)

Then correlates these with whether the final cover is optimal (ratio=1) or
suboptimal (ratio>1), using the MILP reference from the benchmark CSV.

Also computes the matching lower bound at the end of each run and checks
whether  |C| == |max_matching(G)|  (optimality certificate).

Outputs
-------
    results_confidence_analysis.csv  — per-instance summary
    results_confidence_steps.csv     — per-step data (all iterations)
    figure_confidence_analysis.pdf   — 4-panel diagnostic figure

Usage
-----
    python run_confidence_analysis.py [--samples N]
    --samples : max instances to analyse (default: all 566)
"""

import argparse
import sys
import os
import time

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

T_EVOLUTION   = 0.01
RESULTS_FILE  = "results_confidence_analysis.csv"
STEPS_FILE    = "results_confidence_steps.csv"
FIGURE_FILE   = "figure_confidence_analysis.pdf"
BENCHMARK_CSV = "../results/results_degree_greedy_comparison.csv"


# ---------------------------------------------------------------------------
# Instrumented CTQW with per-step confidence metrics
# ---------------------------------------------------------------------------
def ctqw_mvc_instrumented(graph: nx.Graph, t: float) -> dict:
    """
    Run CTQW-MVC and record delta_k, spectral_gap_k, prob_k at each step.

    Returns
    -------
    dict with keys:
        cover        : list of selected vertices
        steps        : list of dicts (step, vertex, prob, delta, spectral_gap)
        min_delta    : minimum confidence margin across all steps
        min_gap      : minimum spectral gap across all steps
        n_uncertain  : number of steps with delta < 0.01
    """
    adj = nx.to_scipy_sparse_array(graph, dtype=float, format='lil')
    cover = []
    steps = []

    while adj.sum() > 0:
        degrees = np.array(adj.sum(axis=1)).flatten()
        degrees_safe = degrees.copy()
        degrees_safe[degrees_safe == 0] = 1e-10
        inv_sqrt_d = 1.0 / np.sqrt(degrees_safe)

        D_inv_sqrt = sp.diags(inv_sqrt_d)
        Gamma = D_inv_sqrt @ adj.tocsr() @ D_inv_sqrt

        # Evolution operator
        U = spla.expm(1j * t * Gamma)
        diag_sq = np.abs(U.diagonal())**2
        probs = 1.0 - diag_sq

        # Zero out isolated vertices (degree=0) — they can't be selected
        active_mask = degrees > 0
        probs_active = probs.copy()
        probs_active[~active_mask] = -1.0

        # Confidence margin: gap between best and second-best
        sorted_p = np.sort(probs_active[active_mask])[::-1]
        best_prob = sorted_p[0]
        second_prob = sorted_p[1] if len(sorted_p) > 1 else 0.0
        delta = best_prob - second_prob

        # Spectral gap of Gamma (smallest non-zero eigenvalue of Laplacian = I - Gamma)
        # For small matrices use dense; for large use ARPACK
        n_active = int(active_mask.sum())
        if n_active <= 200:
            G_dense = Gamma.toarray()
            L_dense = np.eye(G_dense.shape[0]) - G_dense
            # Only active submatrix
            idx = np.where(active_mask)[0]
            L_sub = L_dense[np.ix_(idx, idx)]
            eigvals = np.linalg.eigvalsh(L_sub)
            eigvals_sorted = np.sort(eigvals)
            # gap = difference between smallest two eigenvalues
            spectral_gap = float(eigvals_sorted[1] - eigvals_sorted[0]) if len(eigvals_sorted) > 1 else 0.0
        else:
            spectral_gap = float('nan')  # skip for very large graphs

        best_vertex = int(np.argmax(probs_active))
        cover.append(best_vertex)

        steps.append({
            'step':         len(steps),
            'vertex':       best_vertex,
            'prob':         float(best_prob),
            'delta':        float(delta),
            'spectral_gap': spectral_gap,
            'n_active':     n_active,
        })

        adj[best_vertex, :] = 0
        adj[:, best_vertex] = 0

    min_delta = min(s['delta'] for s in steps) if steps else 1.0
    min_gap   = min(s['spectral_gap'] for s in steps
                    if not np.isnan(s['spectral_gap'])) if steps else 1.0
    n_uncertain = sum(1 for s in steps if s['delta'] < 0.01)

    return {
        'cover':       cover,
        'steps':       steps,
        'min_delta':   min_delta,
        'min_gap':     float(min_gap),
        'n_uncertain': n_uncertain,
    }


# ---------------------------------------------------------------------------
# Matching lower bound (optimality certificate)
# ---------------------------------------------------------------------------
def matching_lower_bound(G: nx.Graph) -> int:
    """Return |max matching| — a lower bound on min vertex cover."""
    matching = nx.max_weight_matching(G, maxcardinality=True)
    return len(matching)


# ---------------------------------------------------------------------------
# Graph reconstruction (same seeds as benchmark)
# ---------------------------------------------------------------------------
def build_graph_list(n_per_setting=10):
    """Reconstruct graphs in the same order as run_degree_greedy_comparison.py."""
    graphs = []

    node_sizes = list(range(4, 55, 5))

    for n in node_sizes:
        for p in [0.3, 0.5, 0.7]:
            gs, _ = generate_erdos_renyi_graphs(
                num_nodes_range=[n], edge_prob_range=[p],
                num_graphs_per_setting=n_per_setting, ensure_connected=True)
            for G in gs:
                graphs.append(('ER', n, float(p), G))

    for n in node_sizes:
        for m in [2, 5]:
            if m >= n:
                continue
            gs, _ = generate_barabasi_albert_graphs(
                num_nodes_range=[n], m_range=[m],
                num_graphs_per_setting=n_per_setting)
            for G in gs:
                graphs.append(('BA', n, float(m), G))

    for n in node_sizes:
        gs, _ = generate_regular_graphs(
            num_nodes_range=[n], num_graphs_per_setting=n_per_setting)
        for G in gs:
            graphs.append(('REG', n, None, G))

    return graphs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, default=None,
                        help='Max instances to analyse (default: all)')
    args = parser.parse_args()

    # Load MILP references
    ref_df = pd.read_csv(BENCHMARK_CSV)
    # Build lookup by (graph_type, num_nodes, param, num_edges)
    ref_lookup = {}
    counters   = {}
    for _, row in ref_df.iterrows():
        gt  = row['graph_type']
        n   = int(row['num_nodes'])
        p   = float(row['param']) if str(row['param']) not in ('nan', '', 'None') else None
        e   = int(row['num_edges'])
        key = (gt, n, p, e)
        if key not in ref_lookup:
            ref_lookup[key] = []
        ref_lookup[key].append(int(row['reference_size']))

    print("Building graph list...")
    all_graphs = build_graph_list()
    if args.samples:
        all_graphs = all_graphs[:args.samples]
    print(f"Analysing {len(all_graphs)} instances...")

    instance_records = []
    step_records     = []
    graph_counters   = {}

    t0 = time.time()
    for i, (gt, n, param, G) in enumerate(all_graphs):
        e   = G.number_of_edges()
        p   = float(param) if param is not None else None
        key = (gt, n, p, e)

        # Get MILP reference
        if key not in graph_counters:
            graph_counters[key] = 0
        ci  = graph_counters[key]
        graph_counters[key] += 1

        milp_ref = None
        if key in ref_lookup and ci < len(ref_lookup[key]):
            milp_ref = ref_lookup[key][ci]

        # Run instrumented CTQW
        result = ctqw_mvc_instrumented(G, T_EVOLUTION)
        cover_size = len(result['cover'])
        ratio = cover_size / milp_ref if milp_ref else float('nan')
        is_optimal = (ratio == 1.0) if milp_ref else None

        # Matching lower bound
        lb = matching_lower_bound(G)
        lb_tight = (cover_size == lb)

        inst_row = {
            'graph_type':   gt,
            'num_nodes':    n,
            'param':        param,
            'num_edges':    e,
            'milp_ref':     milp_ref,
            'cover_size':   cover_size,
            'ratio':        ratio,
            'is_optimal':   is_optimal,
            'matching_lb':  lb,
            'lb_tight':     lb_tight,
            'n_steps':      len(result['steps']),
            'min_delta':    result['min_delta'],
            'min_gap':      result['min_gap'],
            'n_uncertain':  result['n_uncertain'],
        }
        instance_records.append(inst_row)

        # Per-step records
        for s in result['steps']:
            step_records.append({
                'instance_id': i,
                'graph_type':  gt,
                'num_nodes':   n,
                'is_optimal':  is_optimal,
                **s,
            })

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{len(all_graphs)} done  ({elapsed:.1f}s)")

    print(f"\nTotal time: {time.time()-t0:.1f}s")

    inst_df = pd.DataFrame(instance_records)
    step_df = pd.DataFrame(step_records)

    inst_df.to_csv(RESULTS_FILE, index=False)
    step_df.to_csv(STEPS_FILE, index=False)
    print(f"Saved: {RESULTS_FILE}  ({len(inst_df)} instances)")
    print(f"Saved: {STEPS_FILE}  ({len(step_df)} steps)")

    # ---- Summary statistics ----
    opt  = inst_df[inst_df['is_optimal'] == True]
    sub  = inst_df[inst_df['is_optimal'] == False]
    print(f"\nOptimal instances:     {len(opt)}/{len(inst_df)} ({100*len(opt)/len(inst_df):.1f}%)")
    print(f"Sub-optimal instances: {len(sub)}")
    if len(sub) > 0:
        print(f"\nMin delta (optimal):     mean={opt['min_delta'].mean():.4f}  "
              f"median={opt['min_delta'].median():.4f}")
        print(f"Min delta (sub-optimal): mean={sub['min_delta'].mean():.4f}  "
              f"median={sub['min_delta'].median():.4f}")
        print(f"\nMin gap (optimal):       mean={opt['min_gap'].mean():.4f}")
        print(f"Min gap (sub-optimal):   mean={sub['min_gap'].mean():.4f}")
    print(f"\nMatching LB tight (= optimal cert): "
          f"{inst_df['lb_tight'].sum()}/{len(inst_df)}")

    make_figure(inst_df, step_df)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def make_figure(inst_df: pd.DataFrame, step_df: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.30,
                            left=0.09, right=0.97, top=0.93, bottom=0.10)

    opt = inst_df[inst_df['is_optimal'] == True]
    sub = inst_df[inst_df['is_optimal'] == False]

    # ---- Panel A: histogram of min_delta by optimality ----
    ax1 = fig.add_subplot(gs[0, 0])
    bins = np.linspace(0, inst_df['min_delta'].quantile(0.99), 40)
    ax1.hist(opt['min_delta'], bins=bins, alpha=0.65, color='#2ca02c',
             label=f'Optimal ({len(opt)})', density=True)
    ax1.hist(sub['min_delta'], bins=bins, alpha=0.65, color='#d62728',
             label=f'Sub-optimal ({len(sub)})', density=True)
    ax1.axvline(0.01, color='k', linestyle='--', linewidth=1.2,
                label='δ = 0.01 threshold')
    ax1.set_xlabel('Min confidence margin $\\delta_{\\min}$', fontsize=10)
    ax1.set_ylabel('Density', fontsize=10)
    ax1.set_title('(a) Confidence margin distribution', fontsize=11)
    ax1.legend(fontsize=8)
    ax1.grid(axis='y', linestyle=':', alpha=0.5)

    # ---- Panel B: min_delta vs ratio (scatter) ----
    ax2 = fig.add_subplot(gs[0, 1])
    c_opt = '#2ca02c'
    c_sub = '#d62728'
    if len(opt) > 0:
        ax2.scatter(opt['min_delta'], opt['ratio'], s=15, alpha=0.4,
                    color=c_opt, label='Optimal')
    if len(sub) > 0:
        ax2.scatter(sub['min_delta'], sub['ratio'], s=15, alpha=0.6,
                    color=c_sub, label='Sub-optimal')
    ax2.axvline(0.01, color='k', linestyle='--', linewidth=1.2)
    ax2.set_xlabel('Min confidence margin $\\delta_{\\min}$', fontsize=10)
    ax2.set_ylabel('Approximation ratio $|C|/|C^*|$', fontsize=10)
    ax2.set_title('(b) Margin vs. solution quality', fontsize=11)
    ax2.legend(fontsize=8)
    ax2.grid(linestyle=':', alpha=0.5)

    # ---- Panel C: step-level delta for optimal vs sub-optimal instances ----
    ax3 = fig.add_subplot(gs[1, 0])
    step_opt = step_df[step_df['is_optimal'] == True]
    step_sub = step_df[step_df['is_optimal'] == False]
    bins3 = np.linspace(0, step_df['delta'].quantile(0.98), 50)
    ax3.hist(step_opt['delta'], bins=bins3, alpha=0.6, color=c_opt,
             label='Optimal steps', density=True)
    if len(step_sub) > 0:
        ax3.hist(step_sub['delta'], bins=bins3, alpha=0.6, color=c_sub,
                 label='Sub-optimal steps', density=True)
    ax3.axvline(0.01, color='k', linestyle='--', linewidth=1.2)
    ax3.set_xlabel('Per-step confidence margin $\\delta_k$', fontsize=10)
    ax3.set_ylabel('Density', fontsize=10)
    ax3.set_title('(c) Per-step margin (all iterations)', fontsize=11)
    ax3.legend(fontsize=8)
    ax3.grid(axis='y', linestyle=':', alpha=0.5)

    # ---- Panel D: spectral gap vs min_delta (scatter, colored by optimality) ----
    ax4 = fig.add_subplot(gs[1, 1])
    valid = inst_df[inst_df['min_gap'].notna() & (inst_df['min_gap'] != float('nan'))]
    v_opt = valid[valid['is_optimal'] == True]
    v_sub = valid[valid['is_optimal'] == False]
    if len(v_opt) > 0:
        ax4.scatter(v_opt['min_gap'], v_opt['min_delta'], s=15, alpha=0.4,
                    color=c_opt, label='Optimal')
    if len(v_sub) > 0:
        ax4.scatter(v_sub['min_gap'], v_sub['min_delta'], s=20, alpha=0.7,
                    color=c_sub, label='Sub-optimal')
    ax4.set_xlabel('Min spectral gap $\\lambda_{\\min}$', fontsize=10)
    ax4.set_ylabel('Min confidence margin $\\delta_{\\min}$', fontsize=10)
    ax4.set_title('(d) Spectral gap vs. confidence margin', fontsize=11)
    ax4.legend(fontsize=8)
    ax4.grid(linestyle=':', alpha=0.5)

    fig.suptitle(
        'CTQW-MVC optimality diagnosis: confidence margin and spectral gap',
        fontsize=12)
    fig.savefig(FIGURE_FILE, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure saved: {FIGURE_FILE}")


if __name__ == '__main__':
    main()
