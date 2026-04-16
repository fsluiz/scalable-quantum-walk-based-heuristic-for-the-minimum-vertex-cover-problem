#!/usr/bin/env python3
"""Regenerate figure_noise_scaling.pdf from saved CSVs with fixed Panel B labels."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import shutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

df_hw = pd.read_csv("../results/results_noise_scaling_hw.csv")
df_nl = pd.read_csv("../results/results_noise_scaling_noiseless.csv")

print("Noiseless data:")
print(df_nl[["V", "n_qubits", "vc_size", "ref", "ratio", "ref_type"]].to_string())
print(f"\nRatio < 1 in noiseless: {(df_nl['ratio'] < 1-1e-9).sum()}")
print(f"Ratio < 1 in hw noiseless: {(df_hw['noiseless_ratio'] < 1-1e-9).sum()}")

BACKEND_NAME = "ibm_marrakesh"
FIGURE_FILE  = "figure_noise_scaling.pdf"
clr_nl   = "#2ca02c"
clr_fake = "#ff7f0e"
clr_hw   = "#d62728"

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.subplots_adjust(wspace=0.32, left=0.07, right=0.97, top=0.88, bottom=0.18)

# ── Panel A: ratio vs circuit depth (hardware) ──────────────────────────────
ax = axes[0]
for V, sub in df_hw.groupby("V"):
    d = sub["circuit_depth"].iloc[0]
    n = sub["n_qubits"].iloc[0]
    ax.scatter(d, sub["noiseless_ratio"].mean(), marker="o", s=60, color=clr_nl, zorder=4)
    if sub["fake_ratio_mean"].notna().any():
        ax.errorbar(d, sub["fake_ratio_mean"].mean(), yerr=sub["fake_ratio_std"].mean(),
                    fmt="s", color=clr_fake, markersize=8, capsize=4, zorder=4)
    ax.errorbar(d, sub["hw_ratio"].mean(),
                yerr=sub["hw_ratio"].std() if len(sub) > 1 else 0,
                fmt="D", color=clr_hw, markersize=8, capsize=4, zorder=5)
    ax.annotate(f"$n={n}$q\n$V={V}$", (d, sub["hw_ratio"].mean()),
                textcoords="offset points", xytext=(6, 4), fontsize=8)

ax.axhline(1.0, color="k", linestyle="--", lw=0.8, alpha=0.5)
ax.set_xscale("log")
ax.set_xlabel("Circuit depth (log scale)", fontsize=10)
ax.set_ylabel(r"Approximation ratio $|C|/|C^*|$", fontsize=10)
ax.set_title("(a) Ratio vs circuit depth", fontsize=10)
legend_elements = [
    Line2D([0],[0], marker="o", color="w", markerfacecolor=clr_nl, ms=8, label="Noiseless"),
    Line2D([0],[0], marker="s", color="w", markerfacecolor=clr_fake, ms=8, label="FakeMarrakesh"),
    Line2D([0],[0], marker="D", color="w", markerfacecolor=clr_hw, ms=8, label="ibm_marrakesh"),
]
ax.legend(handles=legend_elements, fontsize=8.5)
ax.grid(linestyle=":", alpha=0.5)
ax.set_ylim(0.85, df_hw["hw_ratio"].max() + 0.35)

# ── Panel B: noiseless ratio vs MILP — FIXED ────────────────────────────────
ax = axes[1]
ax.semilogx(df_nl["V"], df_nl["ratio"], "o-", color=clr_nl, linewidth=2, markersize=7)
ax.axhline(1.0, color="k", linestyle="--", lw=0.8, alpha=0.5, label="Optimal (1.000)")
for _, row in df_nl.iterrows():
    ax.annotate(f"$n={int(row.n_qubits)}$q", (row.V, row.ratio),
                textcoords="offset points", xytext=(4, 3), fontsize=7.5)
ax.set_xlabel(r"Graph size $V$ (log scale)", fontsize=10)
ax.set_ylabel(r"$|C_{\rm CTQW}|/|C^*_{\rm MILP}|$ (noiseless)", fontsize=10)
ax.set_title(r"(b) Noiseless ratio vs. MILP, $V=4$--$128$", fontsize=10)
ax.legend(fontsize=8.5)
ax.grid(linestyle=":", alpha=0.5)
r_max = max(df_nl["ratio"].max(), 1.0)
ax.set_ylim(0.97, r_max + 0.05)

# ── Panel C: qubit count and depth vs V ────────────────────────────────────
ax  = axes[2]
ax2 = ax.twinx()
vs  = np.array([4, 8, 16, 32, 64, 128, 256, 512, 1024])
ns  = np.ceil(np.log2(vs)).astype(int)
depths_est = [20 * (4 ** (n - 2)) for n in ns]

ax.plot(vs, ns, "o-", color="#1f77b4", lw=2, ms=6, label="$n$ qubits")
ax2.semilogy(vs, depths_est, "s--", color="#9467bd", lw=1.5, ms=6, label="Depth (est.)")
ax.axvspan(0, 16, alpha=0.06, color=clr_hw, label=r"Hardware feasible ($V\leq16$)")
ax.axvspan(16, 1024, alpha=0.06, color=clr_nl, label="Classical simulation")
ax.set_xscale("log")
ax.set_xlabel(r"Graph size $V$ (log scale)", fontsize=10)
ax.set_ylabel(r"Qubits $n = \lceil\log_2 V\rceil$", fontsize=10, color="#1f77b4")
ax2.set_ylabel("Circuit depth (estimated)", fontsize=10, color="#9467bd")
ax.tick_params(axis="y", labelcolor="#1f77b4")
ax2.tick_params(axis="y", labelcolor="#9467bd")
ax.set_title(r"(c) Qubit count and depth vs $V$", fontsize=10)
lines1, labs1 = ax.get_legend_handles_labels()
lines2, labs2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labs1 + labs2, fontsize=7.5, loc="upper left")
ax.grid(linestyle=":", alpha=0.5)

fig.suptitle(
    rf"CTQW-MVC noise scaling: binary encoding ($\lceil\log_2 V\rceil$ qubits) [{BACKEND_NAME}]",
    fontsize=11, y=1.01)
fig.savefig(FIGURE_FILE, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"\nFigure saved: {FIGURE_FILE}")

shutil.copy(FIGURE_FILE, "../paper/figures/figure_noise_scaling.pdf")
print("Copied to paper/figures/figure_noise_scaling.pdf")
