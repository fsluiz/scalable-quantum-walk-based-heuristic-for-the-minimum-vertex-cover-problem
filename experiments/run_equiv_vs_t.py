#!/usr/bin/env python3
"""Rec 2: Agreement rate (CTQW vs spectral greedy) as a function of t."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from quantum_walk_mvc.core import quantum_walk_mvc_sparse
from quantum_walk_mvc.heuristics import spectral_greedy_vertex_cover
from quantum_walk_mvc.graph_generators import (
    generate_erdos_renyi_graphs, generate_barabasi_albert_graphs, generate_regular_graphs)

T_VALUES = np.logspace(-3, 1, 25)
NODE_SIZES = [4, 10, 20, 30, 40, 50]
N_PER = 10

records = []
for t in T_VALUES:
    for n in NODE_SIZES:
        graphs_er, _ = generate_erdos_renyi_graphs([n],[0.5],N_PER,ensure_connected=True)
        graphs_ba, _ = generate_barabasi_albert_graphs([n],[2],N_PER)
        graphs_re, _ = generate_regular_graphs([n],num_graphs_per_setting=N_PER)
        for gtype, graphs in [("ER",graphs_er),("BA",graphs_ba),("REG",graphs_re)]:
            for G in graphs:
                ctqw_vc = quantum_walk_mvc_sparse(G, t)
                spec_vc = spectral_greedy_vertex_cover(G)
                records.append({"t": t, "V": n, "graph_type": gtype,
                                 "identical": len(ctqw_vc)==len(spec_vc)})
    print(f"  t={t:.4f} done")

df = pd.DataFrame(records)
df.to_csv("results_equiv_vs_t.csv", index=False)

# Plot
fig, ax = plt.subplots(figsize=(7,4))
for gt, clr in [("ER","#1f77b4"),("BA","#e07b39"),("REG","#2ca02c"),("ALL","#333333")]:
    if gt == "ALL":
        sub = df
        lw, ls = 2.2, "-"
        lbl = "All (pooled)"
    else:
        sub = df[df["graph_type"]==gt]
        lw, ls = 1.4, "--"
        lbl = gt
    eq = sub.groupby("t")["identical"].mean()*100
    ax.semilogx(eq.index, eq.values, color=clr, lw=lw, ls=ls, label=lbl)

ax.axvline(0.01, color="k", ls=":", lw=1, label="$t=0.01$")
ax.axhline(98.9, color="grey", ls=":", lw=0.8, alpha=0.6)
ax.set_xlabel(r"Evolution time $t$ (log scale)", fontsize=11)
ax.set_ylabel("Agreement rate (%)", fontsize=11)
ax.set_title("CTQW vs spectral greedy: agreement vs $t$", fontsize=11)
ax.legend(fontsize=9); ax.grid(ls=":",alpha=0.5)
ax.set_ylim(0,105)
fig.tight_layout()
fig.savefig("figure_equiv_vs_t.pdf", dpi=180, bbox_inches="tight")
plt.close(fig)
print("Done. figure_equiv_vs_t.pdf saved.")
