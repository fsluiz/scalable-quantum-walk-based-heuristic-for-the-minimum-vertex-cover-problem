#!/usr/bin/env python3
"""
CTQW-MVC on real IBM Quantum hardware (ibm_torino).

Optimised for ~10 minutes of quantum access.

Strategy
--------
  - V=4 (n=2 qubits), 6 ER(p=0.6) instances
  - V=8 (n=3 qubits), 3 BA(m=2) instances
  - 512 shots per circuit
  - Per-iteration batching: all per-vertex circuits for ALL active seeds
    are submitted as a single Sampler job per iteration.
    Overhead: ~4-5 job submissions total (one per iteration depth).
  - Circuits are pre-transpiled offline before opening the session
    to minimise billed time.

Outputs
-------
  results_real_hardware.csv
  figure_real_hardware.pdf

Prerequisites
-------------
  pip install qiskit-ibm-runtime

  Either set env var IBMQ_TOKEN=<your token>  OR  run once:
      from qiskit_ibm_runtime import QiskitRuntimeService
      QiskitRuntimeService.save_account(channel="ibm_quantum_platform", token="<token>")

Usage
-----
    python run_real_hardware.py [--dry-run]

  --dry-run   Use AerSimulator (no real hardware) to test the script locally.
"""

import sys, os, argparse, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import networkx as nx
import scipy.linalg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from itertools import combinations

# Qiskit
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector

from quantum_walk_mvc.heuristics import check_is_vertex_cover
from quantum_walk_mvc.core import quantum_walk_mvc_sparse

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BACKEND_NAME = "ibm_marrakesh"
N_SHOTS      = 512
T_EVOLUTION  = 0.01
RESULTS_FILE = "results_real_hardware.csv"
FIGURE_FILE  = "figure_real_hardware.pdf"


# ---------------------------------------------------------------------------
# Hamiltonian helpers (shared with run_qiskit_binary_encoding.py)
# ---------------------------------------------------------------------------

def build_normalized_laplacian(G: nx.Graph) -> np.ndarray:
    """H = I - D^{-1/2} A D^{-1/2}."""
    A = nx.to_numpy_array(G, dtype=float)
    deg = A.sum(axis=1)
    deg_safe = np.where(deg == 0, 1e-10, deg)
    inv_sqrt = 1.0 / np.sqrt(deg_safe)
    D_inv_sqrt = np.diag(inv_sqrt)
    Gamma = D_inv_sqrt @ A @ D_inv_sqrt
    return np.eye(len(G)) - Gamma


def evolution_unitary(H: np.ndarray, t: float) -> np.ndarray:
    """U(t) = exp(-i H t)."""
    return scipy.linalg.expm(-1j * t * H)


def get_probs_noiseless(G: nx.Graph, t: float) -> dict:
    """Exact P(m→out) via statevector for each vertex m."""
    V = G.number_of_nodes()
    n = int(np.ceil(np.log2(max(V, 2))))
    dim = 2 ** n
    H = build_normalized_laplacian(G)
    H_pad = np.eye(dim, dtype=complex)
    H_pad[:V, :V] = H
    U = evolution_unitary(H_pad, t)
    nodes = list(G.nodes())
    probs = {}
    for idx, node in enumerate(nodes):
        state = np.zeros(dim, dtype=complex)
        state[idx] = 1.0
        p_stay = abs((U @ state)[idx]) ** 2
        probs[node] = 1.0 - p_stay
    return probs


# ---------------------------------------------------------------------------
# Circuit builders
# ---------------------------------------------------------------------------

def build_vertex_circuit(G: nx.Graph, vertex_idx: int, t: float) -> QuantumCircuit:
    """
    Build circuit that prepares |vertex_idx⟩, applies U(t), measures.

    Returns a QuantumCircuit with classical register named 'meas'.
    P(m→out) is estimated as P(measuring anything != |vertex_idx⟩).
    """
    V = G.number_of_nodes()
    n = int(np.ceil(np.log2(max(V, 2))))
    dim = 2 ** n

    H = build_normalized_laplacian(G)
    H_pad = np.eye(dim, dtype=complex)
    H_pad[:V, :V] = H
    U = evolution_unitary(H_pad, t)

    qc = QuantumCircuit(n, name=f"v{vertex_idx}")
    qc.add_register(__import__('qiskit').ClassicalRegister(n, name="meas"))

    # Encode vertex_idx in binary (big-endian → Qiskit little-endian)
    bits = format(vertex_idx, f"0{n}b")
    for bit_pos, bit_val in enumerate(reversed(bits)):
        if bit_val == "1":
            qc.x(bit_pos)

    qc.unitary(U, list(range(n)), label="U(t)")
    qc.measure(list(range(n)), qc.cregs[0])

    return qc


# ---------------------------------------------------------------------------
# Exact reference (brute force, V <= 10)
# ---------------------------------------------------------------------------

def exact_mvc_brute(G: nx.Graph) -> int:
    nodes = list(G.nodes())
    for size in range(len(nodes) + 1):
        for cand in combinations(nodes, size):
            if check_is_vertex_cover(G, set(cand)):
                return size
    return len(nodes)


# ---------------------------------------------------------------------------
# Iterative MVC from a probability oracle
# ---------------------------------------------------------------------------

def ctqw_mvc_from_oracle(G: nx.Graph, prob_fn) -> list:
    """
    Iterative CTQW-MVC.
    prob_fn(G_residual) -> {node: P(m→out)}
    """
    G_res = G.copy()
    cover = []
    while G_res.number_of_edges() > 0:
        probs = prob_fn(G_res)
        active = [v for v in G_res.nodes() if G_res.degree(v) > 0]
        if not active:
            break
        best = max(active, key=lambda v: probs.get(v, 0.0))
        cover.append(best)
        G_res.remove_node(best)
    return cover


# ---------------------------------------------------------------------------
# Pre-transpilation (done offline, before session opens)
# ---------------------------------------------------------------------------

def pretranspile_all_circuits(
    graphs: list,
    t: float,
    backend,
    opt_level: int = 3,
) -> dict:
    """
    Pre-transpile all V circuits for each graph.

    Returns dict: {(seed_idx, vertex_idx): transpiled_qc}
    Also caches the raw graph state so we can rebuild sub-graphs later.
    """
    print("Pre-transpiling circuits (offline)...")
    transpiled = {}
    t0 = time.time()

    for si, (label, G) in enumerate(graphs):
        V = G.number_of_nodes()
        nodes = list(G.nodes())
        for vi, node in enumerate(nodes):
            qc = build_vertex_circuit(G, vi, t)
            qc_t = transpile(qc, backend=backend, optimization_level=opt_level)
            transpiled[(si, vi)] = qc_t

    print(f"  Pre-transpilation done in {time.time()-t0:.1f}s  "
          f"({len(transpiled)} circuits)")
    return transpiled


# ---------------------------------------------------------------------------
# Batched iterative MVC on real (or simulated) backend
# ---------------------------------------------------------------------------

def run_batched_mvc(
    graphs: list,
    transpiled_base: dict,
    sampler,
    t: float,
    n_shots: int,
) -> list:
    """
    Run the full iterative CTQW-MVC for all graphs.

    Each iteration submits ONE Sampler job containing all per-vertex
    circuits for all graphs that still have uncovered edges.

    Returns list of dicts with per-graph results.
    """
    V_list  = [G.number_of_nodes() for _, G in graphs]
    n_list  = [int(np.ceil(np.log2(max(V, 2)))) for V in V_list]
    nodes_list = [list(G.nodes()) for _, G in graphs]

    # State per graph
    residuals = [G.copy() for _, G in graphs]
    covers    = [[] for _ in graphs]
    done      = [False] * len(graphs)

    iteration = 0
    total_jobs = 0
    t_quantum  = 0.0

    while not all(done):
        iteration += 1

        # --- collect circuits to run this iteration ---
        job_meta = []   # (graph_idx, vertex_node, vertex_idx_in_residual, transpiled_qc)

        for si, G_res in enumerate(residuals):
            if done[si]:
                continue
            if G_res.number_of_edges() == 0:
                done[si] = True
                continue

            active_nodes = [v for v in G_res.nodes() if G_res.degree(v) > 0]
            original_nodes = nodes_list[si]

            for node in active_nodes:
                # Original index in the full graph (for the pre-transpiled circuit)
                orig_idx = original_nodes.index(node)
                qc_t = transpiled_base[(si, orig_idx)]
                job_meta.append((si, node, orig_idx, qc_t))

        if not job_meta:
            break

        circuits = [m[3] for m in job_meta]
        print(f"  Iteration {iteration}: submitting {len(circuits)} circuits "
              f"({sum(1 for m in job_meta if not done[m[0]])} active graphs)...")

        # --- submit ONE batched job ---
        # Both IBM Runtime SamplerV2 and Aer SamplerV2 accept:
        #   sampler.run(pubs, shots=N)  where pubs = list of circuits
        t_submit = time.time()
        job = sampler.run(circuits, shots=n_shots)
        result = job.result()
        t_quantum += time.time() - t_submit
        total_jobs += 1

        # --- parse results and update algorithm state ---
        # Collect all probs per graph for this iteration
        probs_this_iter = {si: {} for si in range(len(graphs))}

        for res_idx, (si, node, orig_idx, _) in enumerate(job_meta):
            n = n_list[si]
            m_bitstring = format(orig_idx, f"0{n}b")

            # SamplerV2: result[res_idx].data.<register_name>.get_counts()
            try:
                counts = result[res_idx].data.meas.get_counts()
            except AttributeError:
                # Fallback for SamplerV1
                counts = result.quasi_dists[res_idx]
                counts = {format(k, f"0{n}b"): v for k, v in counts.items()}

            total = sum(counts.values())
            count_stay = counts.get(m_bitstring, 0)
            p_out = 1.0 - count_stay / total if total > 0 else 0.0
            probs_this_iter[si][node] = p_out

        # Select best vertex per graph and update state
        for si in range(len(graphs)):
            if done[si]:
                continue
            G_res = residuals[si]
            if G_res.number_of_edges() == 0:
                done[si] = True
                continue

            active_nodes = [v for v in G_res.nodes() if G_res.degree(v) > 0]
            probs = probs_this_iter[si]

            if not probs or not active_nodes:
                done[si] = True
                continue

            best = max(active_nodes, key=lambda v: probs.get(v, 0.0))
            covers[si].append(best)
            residuals[si].remove_node(best)

            if residuals[si].number_of_edges() == 0:
                done[si] = True

    print(f"\n  Total Sampler jobs submitted: {total_jobs}")
    print(f"  Quantum execution time: {t_quantum:.1f}s")

    return covers, total_jobs, t_quantum


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(dry_run: bool = False):
    # --- Set up backend ---
    if dry_run:
        from qiskit_ibm_runtime.fake_provider import FakeMarrakesh
        from qiskit_aer import AerSimulator
        print("[DRY-RUN] Using AerSimulator (FakeMarrakesh noise model).")
        backend = AerSimulator.from_backend(FakeMarrakesh())
        use_session = False
    else:
        from qiskit_ibm_runtime import QiskitRuntimeService
        token = os.environ.get("IBMQ_TOKEN")
        if token:
            service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token)
        else:
            service = QiskitRuntimeService(channel="ibm_quantum_platform")  # use saved account
        backend = service.backend(BACKEND_NAME)
        use_session = True
        print(f"Connected to backend: {backend.name}")
        print(f"  Status: {backend.status().status_msg}")
        print(f"  Pending jobs: {backend.status().pending_jobs}")

    # --- Define test graphs ---
    # Conservative selection for ~7m30s budget (open plan, no session):
    # ~5 batched jobs × 60s/job ≈ 5 min, leaving 2.5 min buffer.
    graphs = []

    # V=4 (n=2 qubits, depth~20) — 3 ER instances
    for seed in [0, 1, 3]:
        G = nx.erdos_renyi_graph(4, 0.6, seed=seed)
        if nx.is_connected(G) and G.number_of_edges() > 0:
            graphs.append((f"ER(V=4,p=0.6,s{seed})", G))

    # V=8 (n=3 qubits, depth~128) — 2 BA instances
    for seed in [0, 1]:
        G = nx.barabasi_albert_graph(8, 2, seed=seed)
        graphs.append((f"BA(V=8,m=2,s{seed})", G))

    print(f"\nTest graphs: {len(graphs)} instances")
    for label, G in graphs:
        V = G.number_of_nodes()
        n = int(np.ceil(np.log2(max(V, 2))))
        print(f"  {label}  V={V}  n={n}q  edges={G.number_of_edges()}")

    # --- Noiseless reference ---
    print("\nComputing noiseless references...")
    noiseless_covers = []
    classical_covers = []
    exact_sizes      = []

    for label, G in graphs:
        V = G.number_of_nodes()
        ref = exact_mvc_brute(G) if V <= 10 else len(quantum_walk_mvc_sparse(G, T_EVOLUTION))
        vc_nl  = ctqw_mvc_from_oracle(G, lambda g: get_probs_noiseless(g, T_EVOLUTION))
        vc_cl  = quantum_walk_mvc_sparse(G, T_EVOLUTION)
        noiseless_covers.append(vc_nl)
        classical_covers.append(vc_cl)
        exact_sizes.append(ref)
        print(f"  {label}: ref={ref}  noiseless={len(vc_nl)}  classical={len(vc_cl)}")

    # --- Pre-transpile all circuits ---
    transpiled = pretranspile_all_circuits(graphs, T_EVOLUTION, backend)

    # --- Print circuit stats ---
    print("\nCircuit statistics after transpilation:")
    seen_V = set()
    for si, (label, G) in enumerate(graphs):
        V = G.number_of_nodes()
        if V in seen_V:
            continue
        seen_V.add(V)
        qc_example = transpiled[(si, 0)]
        ops = qc_example.count_ops()
        two_q = ops.get("cz", 0) + ops.get("cx", 0) + ops.get("ecr", 0)
        print(f"  V={V}: depth={qc_example.depth()}  2q_gates={two_q}  "
              f"total_gates={sum(ops.values())}")

    # Estimate runtime
    n_v4 = sum(1 for _, G in graphs if G.number_of_nodes() == 4)
    n_v8 = sum(1 for _, G in graphs if G.number_of_nodes() == 8)
    est_jobs   = 5   # conservative upper bound on iteration depth
    est_circs  = est_jobs * (n_v4 * 4 + n_v8 * 8)
    est_min    = est_circs * 0.3 / 60  # ~0.3s per circuit in batched job
    print(f"\nEstimated: ~{est_jobs} jobs, ~{est_circs} circuits, ~{est_min:.1f} min")

    # --- FakeMarrakesh simulation (local, same noise model as real backend) ---
    print("\nRunning FakeMarrakesh noise simulation (local)...")
    from qiskit_ibm_runtime.fake_provider import FakeMarrakesh
    from qiskit_aer import AerSimulator
    from qiskit_aer.primitives import SamplerV2 as AerSamplerV2

    fake_backend  = AerSimulator.from_backend(FakeMarrakesh())
    transpiled_fm = pretranspile_all_circuits(graphs, T_EVOLUTION, fake_backend)
    fake_sampler  = AerSamplerV2()

    # Run N_FAKE_REPS repetitions to estimate variance
    N_FAKE_REPS = 5
    fake_covers_all = []   # list of reps, each rep is list of covers per graph
    for rep in range(N_FAKE_REPS):
        covers_rep, _, _ = run_batched_mvc(
            graphs, transpiled_fm, fake_sampler, T_EVOLUTION, N_SHOTS
        )
        fake_covers_all.append(covers_rep)
        print(f"  FakeMarrakesh rep {rep+1}/{N_FAKE_REPS} done")

    # --- Real hardware execution ---
    print(f"\n{'[DRY-RUN] ' if dry_run else ''}Running on {BACKEND_NAME}...")
    t_wall_start = time.time()

    if use_session:
        # Real hardware: job mode (open plan compatible — no Session needed)
        from qiskit_ibm_runtime import SamplerV2
        sampler = SamplerV2(mode=backend)
        hw_covers, n_jobs, t_q = run_batched_mvc(
            graphs, transpiled, sampler, T_EVOLUTION, N_SHOTS
        )
    else:
        # Dry-run: re-use FakeMarrakesh sampler
        hw_covers, n_jobs, t_q = run_batched_mvc(
            graphs, transpiled_fm, fake_sampler, T_EVOLUTION, N_SHOTS
        )

    t_wall_total = time.time() - t_wall_start
    print(f"\nTotal wall time: {t_wall_total/60:.2f} min  "
          f"(quantum exec: {t_q:.1f}s)")

    # --- Build results table ---
    records = []
    for i, (label, G) in enumerate(graphs):
        V = G.number_of_nodes()
        n = int(np.ceil(np.log2(max(V, 2))))
        ref  = exact_sizes[i]
        hw   = len(hw_covers[i])
        nl   = len(noiseless_covers[i])
        cl   = len(classical_covers[i])
        valid_hw = check_is_vertex_cover(G, set(hw_covers[i]))

        # FakeMarrakesh stats across reps
        fm_sizes = [len(fake_covers_all[r][i]) for r in range(N_FAKE_REPS)]
        fm_mean  = float(np.mean(fm_sizes))
        fm_std   = float(np.std(fm_sizes))

        graph_type = "ER(V=4)" if V == 4 else "BA(V=8)"
        records.append({
            "label":              label,
            "graph_type":         graph_type,
            "V":                  V,
            "n_qubits":           n,
            "ref_size":           ref,
            "classical_size":     cl,
            "noiseless_size":     nl,
            "fake_size_mean":     round(fm_mean, 3),
            "fake_size_std":      round(fm_std, 3),
            "hardware_size":      hw,
            "classical_ratio":    cl / ref,
            "noiseless_ratio":    nl / ref,
            "fake_ratio_mean":    fm_mean / ref,
            "fake_ratio_std":     fm_std  / ref,
            "hardware_ratio":     hw / ref,
            "hardware_valid":     valid_hw,
            "backend":            BACKEND_NAME if not dry_run else "dry_run",
            "n_shots":            N_SHOTS,
            "fake_reps":          N_FAKE_REPS,
        })
        print(f"  {label}: ref={ref}  nl={nl}  "
              f"fake={fm_mean:.1f}±{fm_std:.1f}  hw={hw}  valid={valid_hw}  "
              f"hw_ratio={hw/ref:.3f}")

    df = pd.DataFrame(records)
    df.to_csv(RESULTS_FILE, index=False)
    print(f"\nResults saved: {RESULTS_FILE}")

    make_figure(df, dry_run)
    print_summary(df)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(df: pd.DataFrame, dry_run: bool = False) -> None:
    graph_types = ["ER(V=4)", "BA(V=8)"]
    clr_nl   = "#2ca02c"
    clr_fake = "#ff7f0e"
    clr_hw   = "#d62728"

    fig = plt.figure(figsize=(14, 5))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.32,
                           left=0.07, right=0.97, top=0.88, bottom=0.18)

    # -----------------------------------------------------------------
    # Panel A: mean ± std approximation ratio — noiseless / fake / real
    # -----------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0])
    x = np.arange(len(graph_types))
    w = 0.22

    cols = [
        ("noiseless_ratio",  None,             "Noiseless",           clr_nl),
        ("fake_ratio_mean",  "fake_ratio_std", f"FakeMarrakesh (n={df['fake_reps'].iloc[0]}×)", clr_fake),
        ("hardware_ratio",   None,             f"ibm_marrakesh (real)", clr_hw),
    ]
    for i, (col_m, col_s, lbl, clr) in enumerate(cols):
        means = [df[df["graph_type"] == gt][col_m].mean() for gt in graph_types]
        if col_s:
            # For FakeMarrakesh: propagate within-instance std + between-instance std
            stds = [np.sqrt(
                        df[df["graph_type"]==gt][col_s].mean()**2 +
                        df[df["graph_type"]==gt][col_m].std()**2
                    ) for gt in graph_types]
        else:
            stds = [df[df["graph_type"] == gt][col_m].std() for gt in graph_types]
        ax1.bar(x + (i - 1) * w, means, w,
                yerr=stds, capsize=4, label=lbl, color=clr, alpha=0.82)

    ax1.axhline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.6,
                label="Optimal (ratio=1)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(graph_types, fontsize=10)
    ax1.set_ylabel("Approximation ratio $|C|/|C^*|$", fontsize=10)
    ax1.set_title("(a) Solution quality: three settings", fontsize=10)
    ax1.legend(fontsize=7.5)
    ax1.grid(axis="y", linestyle=":", alpha=0.5)
    ymax = df[["noiseless_ratio","fake_ratio_mean","hardware_ratio"]].max().max() + 0.25
    ax1.set_ylim(0.85, max(ymax, 1.8))

    # -----------------------------------------------------------------
    # Panel B: per-instance scatter — FakeMarrakesh mean vs real hardware
    # -----------------------------------------------------------------
    ax2 = fig.add_subplot(gs[1])
    for gt in graph_types:
        sub = df[df["graph_type"] == gt]
        clr = clr_fake if gt == "ER(V=4)" else "#9467bd"
        ax2.errorbar(
            sub["fake_ratio_mean"], sub["hardware_ratio"],
            xerr=sub["fake_ratio_std"],
            fmt="o", color=clr, markersize=7, capsize=4,
            label=gt, zorder=3, alpha=0.85
        )
    lo = 0.9
    hi = max(df[["fake_ratio_mean","hardware_ratio"]].max().max(), 1.8) + 0.1
    ax2.plot([lo, hi], [lo, hi], "k--", linewidth=0.9, alpha=0.5,
             label="y = x (perfect model)")
    ax2.set_xlabel("FakeMarrakesh ratio (mean ± std)", fontsize=10)
    ax2.set_ylabel("Real ibm_marrakesh ratio", fontsize=10)
    ax2.set_title("(b) Noise model vs real hardware", fontsize=10)
    ax2.legend(fontsize=8.5)
    ax2.grid(linestyle=":", alpha=0.5)
    ax2.set_xlim(lo, hi); ax2.set_ylim(lo, hi)

    # -----------------------------------------------------------------
    # Panel C: degradation vs circuit depth
    # -----------------------------------------------------------------
    ax3 = fig.add_subplot(gs[2])
    depth_map = {4: 20, 8: 127}
    for gt in graph_types:
        sub = df[df["graph_type"] == gt]
        V   = sub["V"].iloc[0]
        d   = depth_map.get(V, V)
        # FakeMarrakesh
        ax3.errorbar(d, sub["fake_ratio_mean"].mean(),
                     yerr=sub["fake_ratio_std"].mean(),
                     fmt="s", color=clr_fake, markersize=9, capsize=4,
                     label="FakeMarrakesh" if gt == "ER(V=4)" else "")
        # Real hardware
        ax3.scatter(d, sub["hardware_ratio"].mean(),
                    marker="D", s=80, color=clr_hw, zorder=4,
                    label="ibm_marrakesh" if gt == "ER(V=4)" else "")
        ax3.annotate(gt, (d, sub["hardware_ratio"].mean()),
                     textcoords="offset points", xytext=(6, 4), fontsize=8)

    ax3.axhline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    ax3.set_xlabel("Circuit depth (FakeTorino transpilation)", fontsize=10)
    ax3.set_ylabel("Mean approximation ratio", fontsize=10)
    ax3.set_title("(c) Degradation vs circuit depth", fontsize=10)
    ax3.legend(fontsize=8.5)
    ax3.grid(linestyle=":", alpha=0.5)
    ax3.set_ylim(0.9, max(df["hardware_ratio"].max() + 0.3, 2.0))

    label_suffix = " [DRY-RUN]" if dry_run else ""
    fig.suptitle(
        f"CTQW-MVC: noiseless / FakeMarrakesh / ibm_marrakesh (real){label_suffix}",
        fontsize=11, y=1.01)
    fig.savefig(FIGURE_FILE, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {FIGURE_FILE}")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame) -> None:
    backend = df["backend"].iloc[0]
    shots   = df["n_shots"].iloc[0]
    reps    = df["fake_reps"].iloc[0]

    print("\n" + "=" * 80)
    print(f"  CTQW-MVC  |  backend={backend}  shots={shots}  "
          f"FakeMarrakesh reps={reps}")
    print("=" * 80)
    print(f"  {'Graph':>22}  {'Ref':>4}  {'Noiseless':>9}  "
          f"{'Fake(mean±std)':>16}  {'Hardware':>9}  {'Valid':>5}")
    print("  " + "-" * 72)
    for _, row in df.iterrows():
        fake_str = f"{row['fake_ratio_mean']:.3f}±{row['fake_ratio_std']:.3f}"
        print(f"  {row['label']:>22}  {row['ref_size']:>4}  "
              f"{row['noiseless_ratio']:>9.3f}  "
              f"{fake_str:>16}  "
              f"{row['hardware_ratio']:>9.3f}  "
              f"{'yes' if row['hardware_valid'] else 'NO':>5}")

    print()
    print(f"  {'Graph type':>12}  {'Noiseless':>9}  {'FakeMarrakesh':>14}  "
          f"{'ibm_marrakesh':>14}  {'Valid%':>7}")
    print("  " + "-" * 62)
    for gt in ["ER(V=4)", "BA(V=8)"]:
        sub = df[df["graph_type"] == gt]
        if sub.empty:
            continue
        nl  = sub["noiseless_ratio"].mean()
        fm  = sub["fake_ratio_mean"].mean()
        fms = sub["fake_ratio_std"].mean()
        hw  = sub["hardware_ratio"].mean()
        hws = sub["hardware_ratio"].std()
        vf  = sub["hardware_valid"].mean() * 100
        print(f"  {gt:>12}  {nl:>9.3f}  "
              f"{fm:.3f}±{fms:.3f}{'':>5}  "
              f"{hw:.3f}±{hws:.3f}{'':>5}  {vf:>6.0f}%")

    print()
    print("  FakeMarrakesh vs real: positive difference → model is pessimistic")
    print("  ratio = |C| / |C*|  (1.000 = optimal;  higher = worse)")
    print("  valid = output is a legitimate vertex cover despite noise")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CTQW-MVC on real IBM Quantum hardware."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Use local AerSimulator instead of real hardware (for testing)."
    )
    args = parser.parse_args()
    main(dry_run=args.dry_run)
