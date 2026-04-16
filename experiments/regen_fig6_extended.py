#!/usr/bin/env python3
"""
Regenerate Figure 6 (figure_degree_greedy_comparison.pdf) extended to N≈150
using legacy MILP data from data/legacy/ directory.

For each row in the legacy CSVs, reconstructs the graph from the stored
edge list, computes degree-greedy cover, then regenerates the comparison plot.

Large input files (ER ~375 MB, BA ~134 MB) are NOT tracked in the repository.
Download them from Zenodo (see data/legacy/README.md) and place them under
data/legacy/ before running this script.  The REG dataset is already in
data/legacy/ as a .zip file.

Output:
  - results/results_dg_extended.csv          (processed data, tracked)
  - paper/figures/figure_degree_greedy_comparison.pdf  (updated figure)
"""

import sys, os, ast, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from quantum_walk_mvc.heuristics import degree_greedy_vertex_cover

# ── paths ──────────────────────────────────────────────────────────────────
ROOT        = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.join(ROOT, "..")
DATA_DIR    = os.path.join(REPO_ROOT, "data", "legacy")
FIG_OUT     = os.path.join(REPO_ROOT, "paper", "figures",
                            "figure_degree_greedy_comparison.pdf")
CSV_OUT     = os.path.join(REPO_ROOT, "results", "results_dg_extended.csv")

ER_CSV  = os.path.join(DATA_DIR,
    "quantum_full_and_with_tsa_vs_classic_mvc_analysis_resultsN4-153_"
    "P02040506070809_G10.csv")
BA_CSV  = os.path.join(DATA_DIR,
    "quantum__classic_exact_mvc_analysis_results_BAN4-203_"
    "P1235101520_G10.csv")
REG_CSV = os.path.join(DATA_DIR,
    "quantum_full_and_with_tsa_vs_classic_mvc_regular_graphs_"
    "analysis_resultsN4-152_G10.csv")

N_MAX = 150   # extend up to this N (MILP still 0% NaN throughout)


# ── helpers ────────────────────────────────────────────────────────────────

def parse_graph_data(edge_str: str) -> nx.Graph:
    """Reconstruct NetworkX graph from stored edge-list string."""
    edges = ast.literal_eval(edge_str)
    G = nx.Graph()
    for u, v, *_ in edges:
        G.add_edge(int(u), int(v))
    return G


def compute_dg_for_df(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Add degree-greedy cover size column to dataframe subset (N ≤ N_MAX)."""
    sub = df[df["num_nodes"] <= N_MAX].copy()
    dg_sizes = []
    n_total = len(sub)
    t0 = time.time()
    for i, (_, row) in enumerate(sub.iterrows()):
        G = parse_graph_data(row["graph_data"])
        vc = degree_greedy_vertex_cover(G)
        dg_sizes.append(len(vc))
        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n_total - i - 1) / rate
            print(f"  [{label}] {i+1}/{n_total}  "
                  f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")
    sub["dg_size"] = dg_sizes
    sub["dg_ratio"] = sub["dg_size"] / sub["exact_vc_size"]
    sub["ctqw_ratio"] = sub["quantum_vc_size"] / sub["exact_vc_size"]
    sub["approx2_ratio"] = sub["2approx_vc_size"] / sub["exact_vc_size"]
    sub["fastvc_ratio"] = sub["fastvc_vc_size"] / sub["exact_vc_size"]
    sub["sa_ratio"] = sub["sa_vc_size"] / sub["exact_vc_size"]
    return sub


# ── load & process ─────────────────────────────────────────────────────────

print("Loading legacy CSVs and computing degree-greedy covers...")

print(f"\n[ER]  loading {ER_CSV}")
df_er_raw = pd.read_csv(ER_CSV)
# Filter: p ∈ {0.4, 0.5, 0.7} — close to original {0.3, 0.5, 0.7}
# Use 0.5 and 0.7 exactly plus 0.4 as closest to 0.3
df_er_raw["p"] = df_er_raw["graph_params"].apply(
    lambda x: ast.literal_eval(x)["p"])
df_er_raw = df_er_raw[df_er_raw["p"].isin([0.4, 0.5, 0.7])]
print(f"  After p filter: {len(df_er_raw)} rows, N={df_er_raw.num_nodes.min()}–{df_er_raw.num_nodes.max()}")
df_er = compute_dg_for_df(df_er_raw, "ER")
df_er["graph_type"] = "ER"

print(f"\n[BA]  loading {BA_CSV}")
df_ba_raw = pd.read_csv(BA_CSV)
# No m column; use all BA data (diverse m values representative of BA topology)
# Filter to N_MAX; this gives ~9910 rows, too many — subsample for speed
# Actually degree-greedy is fast, so run all
print(f"  Total BA rows ≤ {N_MAX}: {len(df_ba_raw[df_ba_raw.num_nodes <= N_MAX])}")
df_ba = compute_dg_for_df(df_ba_raw, "BA")
df_ba["graph_type"] = "BA"

print(f"\n[REG] loading {REG_CSV}")
df_reg_raw = pd.read_csv(REG_CSV)
print(f"  Total REG rows ≤ {N_MAX}: {len(df_reg_raw[df_reg_raw.num_nodes <= N_MAX])}")
df_reg = compute_dg_for_df(df_reg_raw, "REG")
df_reg["graph_type"] = "REG"

# Combine
keep_cols = ["graph_type", "num_nodes", "num_edges", "exact_vc_size",
             "dg_size", "dg_ratio",
             "ctqw_ratio", "approx2_ratio", "fastvc_ratio", "sa_ratio"]
df_all = pd.concat([df_er[keep_cols], df_ba[keep_cols], df_reg[keep_cols]],
                   ignore_index=True)
df_all.to_csv(CSV_OUT, index=False)
print(f"\nExtended dataset saved: {CSV_OUT}  ({len(df_all)} rows)")


# ── statistics for Table IV update ────────────────────────────────────────

print("\n" + "=" * 70)
print("  Mean approximation ratio by graph type  (N ∈ [4, %d], MILP ref.)" % N_MAX)
print("=" * 70)
algo_labels = [
    ("CTQW",          "ctqw_ratio"),
    ("Degree-Greedy", "dg_ratio"),
    ("2-Approx",      "approx2_ratio"),
    ("FastVC",        "fastvc_ratio"),
    ("SA",            "sa_ratio"),
]
header = f"  {'Graph':>6}  {'N_inst':>7}  " + "  ".join(f"{n:>14}" for n, _ in algo_labels)
print(header)
print("  " + "-" * (len(header) - 2))
for gtype in ["ER", "BA", "REG", "All"]:
    sub = df_all if gtype == "All" else df_all[df_all["graph_type"] == gtype]
    n_inst = len(sub)
    vals = []
    for name, col in algo_labels:
        m = sub[col].mean()
        vals.append(f"{m:>14.4f}")
    print(f"  {gtype:>6}  {n_inst:>7}  " + "  ".join(vals))

print()
ctqw_wins = (df_all["ctqw_ratio"] < df_all["dg_ratio"]).sum()
ctqw_ties = (df_all["ctqw_ratio"] == df_all["dg_ratio"]).sum()
ctqw_loses = (df_all["ctqw_ratio"] > df_all["dg_ratio"]).sum()
n_total = len(df_all)
print(f"  CTQW vs DG: wins={ctqw_wins}({100*ctqw_wins/n_total:.1f}%)  "
      f"ties={ctqw_ties}({100*ctqw_ties/n_total:.1f}%)  "
      f"loses={ctqw_loses}({100*ctqw_loses/n_total:.1f}%)")

# Sub-optimality rate
subopt = (df_all["ctqw_ratio"] > 1.0).sum()
print(f"  CTQW sub-optimal: {subopt}/{n_total} = {100*subopt/n_total:.1f}%")
print(f"  Max CTQW ratio: {df_all['ctqw_ratio'].max():.4f}")
print(f"  Max DG ratio:   {df_all['dg_ratio'].max():.4f}")

# Wilcoxon test for updated Table III
from scipy.stats import wilcoxon
print()
print("  Wilcoxon signed-rank tests (CTQW vs baseline, n=%d)" % n_total)
for name, col in algo_labels[1:]:
    diff = df_all["ctqw_ratio"].values - df_all[col].values
    nz = diff[diff != 0]
    if len(nz) == 0:
        print(f"    vs {name:<14}: no differences")
        continue
    stat, p = wilcoxon(df_all["ctqw_ratio"], df_all[col], alternative="less")
    n_neq = (diff != 0).sum()
    r = abs(stat) / np.sqrt(n_neq) if n_neq > 0 else 0
    print(f"    vs {name:<14}: p={p:.2e}  n_neq={n_neq}  r={r:.2f}")


# ── figure ─────────────────────────────────────────────────────────────────

COLORS = {
    "CTQW":          "#1f77b4",
    "Degree-Greedy": "#d62728",
    "2-Approx":      "#e07b39",
    "FastVC":        "#2ca02c",
    "SA":            "#9467bd",
}
MARKERS = {"CTQW": "o", "Degree-Greedy": "s", "2-Approx": "^",
           "FastVC": "D", "SA": "v"}

ALGO_PLOT = [
    ("CTQW",          "ctqw_ratio"),
    ("Degree-Greedy", "dg_ratio"),
    ("2-Approx",      "approx2_ratio"),
    ("FastVC",        "fastvc_ratio"),
    ("SA",            "sa_ratio"),
]

graph_types   = ["ER", "BA", "REG"]
type_titles   = {"ER": "Erdős–Rényi", "BA": "Barabási–Albert", "REG": "Regular"}

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
fig.subplots_adjust(wspace=0.08, left=0.07, right=0.97, top=0.88, bottom=0.14)

for ax, gtype in zip(axes, graph_types):
    sub = df_all[df_all["graph_type"] == gtype].copy()
    sizes = sorted(sub["num_nodes"].unique())

    for name, col in ALGO_PLOT:
        means = sub.groupby("num_nodes")[col].mean()
        stds  = sub.groupby("num_nodes")[col].std()
        ns = [n for n in sizes if n in means.index]
        ys = [means[n] for n in ns]
        es = [stds[n]  for n in ns]

        lw = 2.2 if name in ("CTQW", "Degree-Greedy") else 1.2
        zo = 3   if name in ("CTQW", "Degree-Greedy") else 2
        ax.plot(ns, ys, color=COLORS[name], marker=MARKERS[name],
                markersize=4, linewidth=lw, zorder=zo, label=name)
        ax.fill_between(ns,
                        [y - e for y, e in zip(ys, es)],
                        [y + e for y, e in zip(ys, es)],
                        color=COLORS[name], alpha=0.12, zorder=zo - 1)

    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.6, zorder=1)
    ax.set_title(type_titles[gtype], fontsize=11)
    ax.set_xlabel("Number of nodes $N$", fontsize=10)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=5))
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.set_ylim(bottom=0.92)

axes[0].set_ylabel("Approximation ratio  $|C|/|C^*|$", fontsize=10)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=5,
           fontsize=9.5, framealpha=0.9, bbox_to_anchor=(0.52, 1.01))

fig.savefig(FIG_OUT, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"\nFigure saved: {FIG_OUT}")
