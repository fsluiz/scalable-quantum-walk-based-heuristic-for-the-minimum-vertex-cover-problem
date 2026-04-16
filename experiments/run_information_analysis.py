#!/usr/bin/env python3
"""
Information-theoretic analysis of CTQW-MVC.

Per-step quantities computed at each iteration k:
  H_k      : Shannon entropy of the normalised transition-probability distribution
  D_kl_k   : KL-divergence from uniform  D_KL(P_k || U_k)
  excess_k : H_k - H_max_k  (entropy excess vs. theoretical maximum = log(n_active))

Instance-level summary:
  H_mean, H_std, H_slope  (linear trend of entropy profile)
  H_integral              (area under entropy curve — total "uncertainty budget")
  MI_consec               (mutual information between consecutive vertex selections)

Outputs
-------
  results_information_steps.csv   — per-step data
  results_information_inst.csv    — per-instance summary
  figure_information_analysis.pdf — 5-panel figure
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

T = 0.01
BENCHMARK_CSV = "../results/results_degree_greedy_comparison.csv"
STEPS_FILE    = "results_information_steps.csv"
INST_FILE     = "results_information_inst.csv"
FIGURE_FILE   = "figure_information_analysis.pdf"

# -----------------------------------------------------------------------
def ctqw_probs(adj):
    degrees = np.array(adj.sum(axis=1)).flatten()
    ds = degrees.copy(); ds[ds == 0] = 1e-10
    D  = sp.diags(1.0 / np.sqrt(ds))
    Gamma = D @ adj.tocsr() @ D
    U = spla.expm(1j * T * Gamma)
    probs = 1.0 - np.abs(U.diagonal())**2
    probs[degrees == 0] = -1.0
    return probs, degrees

def shannon_entropy(p_raw):
    """Shannon entropy of a raw (unnormalised, non-negative) probability vector."""
    p = p_raw[p_raw > 0]
    if len(p) == 0:
        return 0.0
    p = p / p.sum()
    return float(-np.sum(p * np.log(p + 1e-300)))

def kl_uniform(p_raw):
    """KL( P_k || U_k ) where U_k is uniform over active vertices."""
    p = p_raw[p_raw > 0]
    if len(p) == 0:
        return 0.0
    n = len(p)
    p = p / p.sum()
    return float(np.sum(p * np.log(p * n + 1e-300)))   # = log(n) - H

def run_instrumented(G):
    """Run CTQW-MVC and return per-step information measures."""
    adj   = nx.to_scipy_sparse_array(G, dtype=float, format='lil')
    cover = []
    steps = []

    while adj.sum() > 0:
        probs, degrees = ctqw_probs(adj)
        active_mask    = degrees > 0
        p_active       = probs.copy(); p_active[~active_mask] = 0.0
        n_active       = int(active_mask.sum())

        H_k      = shannon_entropy(p_active)
        H_max_k  = np.log(n_active) if n_active > 0 else 0.0
        D_kl_k   = kl_uniform(p_active)
        H_norm_k = H_k / H_max_k if H_max_k > 0 else 1.0   # in [0,1]

        # Confidence margin
        sorted_p = np.sort(p_active[active_mask])[::-1]
        best_p   = sorted_p[0]
        sec_p    = sorted_p[1] if len(sorted_p) > 1 else 0.0
        rel_d    = (best_p - sec_p) / best_p if best_p > 1e-15 else 1.0

        best_v = int(np.argmax(p_active))
        cover.append(best_v)
        steps.append({
            'step':      len(steps),
            'n_active':  n_active,
            'H':         H_k,
            'H_max':     H_max_k,
            'H_norm':    H_norm_k,
            'D_kl':      D_kl_k,
            'rel_delta': rel_d,
            'vertex':    best_v,
            'best_prob': float(best_p),
        })

        adj[best_v, :] = 0; adj[:, best_v] = 0

    return cover, steps

def mutual_information_consecutive(vertices, n_total):
    """
    Empirical MI between consecutive selections over the sequence
    v_0, v_1, ..., v_{K-1}  treated as a single trajectory.
    Uses plugin estimator on the joint distribution of (v_k, v_{k+1}).
    """
    K = len(vertices)
    if K < 2:
        return 0.0
    pairs = [(vertices[i], vertices[i+1]) for i in range(K - 1)]
    counts = {}
    for a, b in pairs:
        counts[(a, b)] = counts.get((a, b), 0) + 1
    N = len(pairs)
    # marginals
    count_a = {}; count_b = {}
    for (a, b), c in counts.items():
        count_a[a] = count_a.get(a, 0) + c
        count_b[b] = count_b.get(b, 0) + c
    mi = 0.0
    for (a, b), c in counts.items():
        p_ab = c / N
        p_a  = count_a[a] / N
        p_b  = count_b[b] / N
        mi  += p_ab * np.log(p_ab / (p_a * p_b + 1e-300))
    return float(mi)

# -----------------------------------------------------------------------
# Load benchmark references
ref_df     = pd.read_csv(BENCHMARK_CSV)
ref_lookup = {}
for _, row in ref_df.iterrows():
    gt = row['graph_type']
    n  = int(row['num_nodes'])
    p  = float(row['param']) if str(row['param']) not in ('nan','','None') else None
    e  = int(row['num_edges'])
    key = (gt, n, p, e)
    if key not in ref_lookup: ref_lookup[key] = []
    ref_lookup[key].append(int(row['reference_size']))

# -----------------------------------------------------------------------
from quantum_walk_mvc.graph_generators import (
    generate_erdos_renyi_graphs, generate_barabasi_albert_graphs, generate_regular_graphs
)

print("Building graphs...")
graphs = []
for n in range(4, 55, 5):
    for p in [0.3, 0.5, 0.7]:
        gs, _ = generate_erdos_renyi_graphs([n],[p],10,ensure_connected=True)
        for G in gs: graphs.append(('ER', n, float(p), G))
for n in range(4, 55, 5):
    for m in [2, 5]:
        if m >= n: continue
        gs, _ = generate_barabasi_albert_graphs([n],[m],10)
        for G in gs: graphs.append(('BA', n, float(m), G))
for n in range(4, 55, 5):
    gs, _ = generate_regular_graphs([n],10)
    for G in gs: graphs.append(('REG', n, None, G))

print(f"Analysing {len(graphs)} instances...")

import time
t0       = time.time()
counters = {}
step_rows = []
inst_rows = []

for inst_id, (gt, n, param, G) in enumerate(graphs):
    e   = G.number_of_edges()
    p   = float(param) if param is not None else None
    key = (gt, n, p, e)
    if key not in counters: counters[key] = 0
    ci   = counters[key]; counters[key] += 1
    milp = ref_lookup.get(key,[None])[ci] if key in ref_lookup and ci < len(ref_lookup[key]) else None

    cover, steps = run_instrumented(G)
    cover_size   = len(cover)
    ratio        = cover_size / milp if milp else float('nan')
    is_optimal   = (ratio == 1.0) if milp else None

    # MI between consecutive selections
    mi = mutual_information_consecutive(cover, n)

    # Entropy profile: linear slope (positive = entropy grows, negative = shrinks)
    H_vals = [s['H_norm'] for s in steps]
    if len(H_vals) > 2:
        slope, _, r_slope, _, _ = stats.linregress(range(len(H_vals)), H_vals)
    else:
        slope, r_slope = 0.0, 0.0

    inst_rows.append({
        'inst_id':    inst_id,
        'graph_type': gt,
        'num_nodes':  n,
        'param':      param,
        'num_edges':  e,
        'milp_ref':   milp,
        'cover_size': cover_size,
        'ratio':      ratio,
        'is_optimal': is_optimal,
        'H_mean':     float(np.mean(H_vals)),
        'H_std':      float(np.std(H_vals)),
        'H_first':    float(H_vals[0])  if H_vals else 0.0,
        'H_last':     float(H_vals[-1]) if H_vals else 0.0,
        'H_slope':    float(slope),
        'H_integral': float(np.trapz(H_vals) / len(H_vals)),
        'D_kl_mean':  float(np.mean([s['D_kl']    for s in steps])),
        'MI_consec':  mi,
        'n_steps':    len(steps),
    })

    for s in steps:
        step_rows.append({
            'inst_id':   inst_id,
            'graph_type': gt,
            'num_nodes':  n,
            'is_optimal': is_optimal,
            'ratio':      ratio,
            **s,
        })

    if (inst_id + 1) % 100 == 0:
        print(f"  {inst_id+1}/{len(graphs)}  ({time.time()-t0:.1f}s)")

print(f"Done in {time.time()-t0:.1f}s")

step_df = pd.DataFrame(step_rows)
inst_df = pd.DataFrame(inst_rows)
step_df.to_csv(STEPS_FILE, index=False)
inst_df.to_csv(INST_FILE,  index=False)
print(f"Saved {STEPS_FILE}  ({len(step_df)} rows)")
print(f"Saved {INST_FILE}   ({len(inst_df)} rows)")

# -----------------------------------------------------------------------
# Quick stats
opt = inst_df[inst_df['is_optimal'] == True]
sub = inst_df[inst_df['is_optimal'] == False]

print(f"\n{'Metric':<20} {'Optimal':>12} {'Sub-opt':>12} {'p-value':>12}")
print("-"*58)
for col, label in [
    ('H_mean',    'H_norm mean'),
    ('H_slope',   'H slope'),
    ('H_integral','H integral'),
    ('D_kl_mean', 'KL-div mean'),
    ('MI_consec', 'MI consec.'),
]:
    vo = opt[col].dropna()
    vs = sub[col].dropna()
    if len(vs) > 5:
        _, p = stats.mannwhitneyu(vo, vs, alternative='two-sided')
        print(f"{label:<20} {vo.mean():>12.5f} {vs.mean():>12.5f} {p:>12.3e}")

# -----------------------------------------------------------------------
# Figure
fig = plt.figure(figsize=(15, 10))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.32,
                        left=0.07, right=0.97, top=0.93, bottom=0.09)

c_opt, c_sub = '#2ca02c', '#d62728'

# Panel A: mean entropy profile (normalised step position) for opt vs sub
ax1 = fig.add_subplot(gs[0, 0])
s_opt = step_df[step_df['is_optimal'] == True].copy()
s_sub = step_df[step_df['is_optimal'] == False].copy()
# Normalise step position to [0,1] within each instance
step_df['step_frac'] = step_df['step'] / step_df['n_active'].clip(lower=1)
s_opt = step_df[step_df['is_optimal'] == True].copy()
s_sub = step_df[step_df['is_optimal'] == False].copy()

bins = np.linspace(0, 1, 20)
for sub_s, color, label in [(s_opt, c_opt, 'Optimal'), (s_sub, c_sub, 'Sub-optimal')]:
    sub_s = sub_s.copy()
    sub_s['bin'] = pd.cut(sub_s['step_frac'], bins=bins)
    g = sub_s.groupby('bin')['H_norm'].mean()
    xs = [(b.left + b.right)/2 for b in g.index]
    ax1.plot(xs, g.values, color=color, linewidth=2, label=label)
    se = sub_s.groupby('bin')['H_norm'].sem()
    ax1.fill_between(xs, g.values - se.values, g.values + se.values,
                     color=color, alpha=0.15)
ax1.axhline(1.0, color='gray', linestyle=':', linewidth=1, label='Uniform')
ax1.set_xlabel('Normalised step (fraction of cover)', fontsize=10)
ax1.set_ylabel('Normalised entropy $H_k / \\log n_k$', fontsize=10)
ax1.set_title('(a) Entropy profile across steps', fontsize=11)
ax1.legend(fontsize=8); ax1.grid(axis='y', linestyle=':', alpha=0.5)

# Panel B: H_mean histogram
ax2 = fig.add_subplot(gs[0, 1])
bins2 = np.linspace(0.5, 1.0, 30)
ax2.hist(opt['H_mean'], bins=bins2, alpha=0.65, color=c_opt,
         label=f'Optimal ({len(opt)})', density=True)
ax2.hist(sub['H_mean'], bins=bins2, alpha=0.65, color=c_sub,
         label=f'Sub-optimal ({len(sub)})', density=True)
ax2.set_xlabel('Mean normalised entropy $\\bar{H}$', fontsize=10)
ax2.set_ylabel('Density', fontsize=10)
ax2.set_title('(b) Mean entropy distribution', fontsize=11)
ax2.legend(fontsize=8); ax2.grid(axis='y', linestyle=':', alpha=0.5)

# Panel C: KL-divergence mean
ax3 = fig.add_subplot(gs[0, 2])
ax3.hist(opt['D_kl_mean'], bins=30, alpha=0.65, color=c_opt,
         label='Optimal', density=True)
ax3.hist(sub['D_kl_mean'], bins=30, alpha=0.65, color=c_sub,
         label='Sub-optimal', density=True)
ax3.set_xlabel('Mean KL-divergence from uniform $\\bar{D}_{\\rm KL}$', fontsize=10)
ax3.set_ylabel('Density', fontsize=10)
ax3.set_title('(c) KL-divergence distribution', fontsize=11)
ax3.legend(fontsize=8); ax3.grid(axis='y', linestyle=':', alpha=0.5)

# Panel D: entropy slope scatter
ax4 = fig.add_subplot(gs[1, 0])
ax4.scatter(opt['H_slope'], opt['H_mean'], s=12, alpha=0.4, color=c_opt, label='Optimal')
ax4.scatter(sub['H_slope'], sub['H_mean'], s=20, alpha=0.7, color=c_sub, label='Sub-optimal')
ax4.axvline(0, color='k', linestyle='--', linewidth=0.8)
ax4.set_xlabel('Entropy slope (linear trend)', fontsize=10)
ax4.set_ylabel('Mean normalised entropy', fontsize=10)
ax4.set_title('(d) Entropy slope vs. mean', fontsize=11)
ax4.legend(fontsize=8); ax4.grid(linestyle=':', alpha=0.5)

# Panel E: MI between consecutive selections
ax5 = fig.add_subplot(gs[1, 1])
ax5.hist(opt['MI_consec'], bins=30, alpha=0.65, color=c_opt,
         label='Optimal', density=True)
ax5.hist(sub['MI_consec'], bins=30, alpha=0.65, color=c_sub,
         label='Sub-optimal', density=True)
ax5.set_xlabel('MI between consecutive selections $I(m_k; m_{k+1})$', fontsize=10)
ax5.set_ylabel('Density', fontsize=10)
ax5.set_title('(e) Sequential mutual information', fontsize=11)
ax5.legend(fontsize=8); ax5.grid(axis='y', linestyle=':', alpha=0.5)

# Panel F: entropy integral vs ratio
ax6 = fig.add_subplot(gs[1, 2])
valid = inst_df[inst_df['ratio'].notna()]
ax6.scatter(valid[valid['is_optimal']==True]['H_integral'],
            valid[valid['is_optimal']==True]['ratio'],
            s=12, alpha=0.4, color=c_opt, label='Optimal')
ax6.scatter(valid[valid['is_optimal']==False]['H_integral'],
            valid[valid['is_optimal']==False]['ratio'],
            s=25, alpha=0.7, color=c_sub, label='Sub-optimal')
ax6.set_xlabel('Entropy integral $\\int H_k\\,dk / K$', fontsize=10)
ax6.set_ylabel('Approximation ratio $|C|/|C^*|$', fontsize=10)
ax6.set_title('(f) Entropy integral vs. solution quality', fontsize=11)
ax6.legend(fontsize=8); ax6.grid(linestyle=':', alpha=0.5)

fig.suptitle('Information-theoretic diagnosis of CTQW-MVC: entropy, KL-divergence, and mutual information',
             fontsize=12)
fig.savefig(FIGURE_FILE, dpi=180, bbox_inches='tight')
plt.close(fig)
print(f"Figure saved: {FIGURE_FILE}")
