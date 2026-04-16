#!/usr/bin/env python3
"""
Benchmark spectral greedy vs degree-greedy vs SA on real-world graphs.

Datasets used:
  Built-in networkx (small, exact MILP reference available):
    karate      — Zachary karate club      (34 V, 78 E)
    les_mis     — Les Misérables            (77 V, 254 E)
    football    — US college football      (115 V, 613 E)
    polbooks    — Political books          (105 V, 441 E)

  SNAP (downloaded, VBS reference):
    email-Eu    — EU email network         (1005 V, 25571 E)
    ca-GrQc     — Arxiv GR-QC collabs      (5242 V, 14496 E)
    ca-HepTh    — Arxiv Hep-Th collabs     (9877 V, 25998 E)
    ca-AstroPh  — Arxiv AstroPh collabs    (18772 V, 198110 E)

Usage:
    python bench_real_graphs.py
    python bench_real_graphs.py --no-exact   # skip MILP (faster)
"""

import os, sys, time, urllib.request, gzip, shutil, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import networkx as nx
import igraph as ig

from quantum_walk_mvc.heuristics_igraph import (
    spectral_greedy_large, degree_greedy_large,
    check_is_vertex_cover_igraph, warmup_jit,
)

# ---------------------------------------------------------------------------
# Optional: exact MILP via SageMath
# ---------------------------------------------------------------------------
try:
    from sage.all import Graph as SageGraph, MixedIntegerLinearProgram
    SAGE_AVAILABLE = True
except ImportError:
    SAGE_AVAILABLE = False

# ---------------------------------------------------------------------------
# Simulated Annealing (reuses run_scalability_large.py version)
# ---------------------------------------------------------------------------
import random

def simulated_annealing_igraph(g, initial_temperature=50.0,
                                cooling_rate=0.995, max_iterations=3000):
    n = g.vcount()
    adj = [set(g.neighbors(v)) for v in range(n)]
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
                if random.random() < np.exp(-1 / temperature):
                    cover = new_cover
        elif action == "add":
            candidates = [v for v in range(n) if v not in cover]
            if candidates:
                v = random.choice(candidates)
                cover = cover | {v}
                if len(cover) < len(best_cover):
                    best_cover = set(cover)
        temperature *= cooling_rate
    if not check_is_vertex_cover_igraph(g, best_cover):
        for u, v in g.get_edgelist():
            if u not in best_cover and v not in best_cover:
                best_cover.add(u)
    return list(best_cover)


def milp_exact(g_ig):
    """Exact MVC via SageMath MILP (only for small graphs)."""
    if not SAGE_AVAILABLE:
        return None
    edges = g_ig.get_edgelist()
    n = g_ig.vcount()
    sg = SageGraph([(u, v) for u, v in edges])
    p = MixedIntegerLinearProgram(maximization=False)
    x = p.new_variable(binary=True)
    p.set_objective(sum(x[v] for v in range(n)))
    for u, v in edges:
        p.add_constraint(x[u] + x[v] >= 1)
    p.solve()
    return sum(1 for v in range(n) if p.get_values(x[v]) > 0.5)


# ---------------------------------------------------------------------------
# Graph loaders
# ---------------------------------------------------------------------------

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "snap")
os.makedirs(DATA_DIR, exist_ok=True)

SNAP_URLS = {
    # Collaboration networks
    "ca-GrQc":    ("https://snap.stanford.edu/data/ca-GrQc.txt.gz",    "ca-GrQc.txt.gz",    "ca-GrQc.txt"),
    "ca-HepTh":   ("https://snap.stanford.edu/data/ca-HepTh.txt.gz",   "ca-HepTh.txt.gz",   "ca-HepTh.txt"),
    "ca-AstroPh": ("https://snap.stanford.edu/data/ca-AstroPh.txt.gz", "ca-AstroPh.txt.gz", "ca-AstroPh.txt"),
    # Email / communication
    "email-Eu":   ("https://snap.stanford.edu/data/email-Eu-core.txt.gz", "email-Eu-core.txt.gz", "email-Eu-core.txt"),
    # P2P networks
    "p2p-Gnut04": ("https://snap.stanford.edu/data/p2p-Gnutella04.txt.gz", "p2p-Gnutella04.txt.gz", "p2p-Gnutella04.txt"),
    # Road network (sparse)
    "roadNet-PA": ("https://snap.stanford.edu/data/roadNet-PA.txt.gz", "roadNet-PA.txt.gz", "roadNet-PA.txt"),
}


def download_snap(name):
    url, gz_name, txt_name = SNAP_URLS[name]
    gz_path  = os.path.join(DATA_DIR, gz_name)
    txt_path = os.path.join(DATA_DIR, txt_name)
    if not os.path.exists(txt_path):
        if not os.path.exists(gz_path):
            print(f"  Downloading {name}...", end=" ", flush=True)
            try:
                urllib.request.urlretrieve(url, gz_path)
                print("OK")
            except Exception as e:
                print(f"FAILED ({e})")
                return None
        print(f"  Decompressing {name}...", end=" ", flush=True)
        with gzip.open(gz_path, "rb") as f_in, open(txt_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        print("OK")
    return txt_path


def nx_to_igraph(G):
    """Convert undirected networkx graph to igraph (largest connected component)."""
    G = G.to_undirected()
    G.remove_edges_from(nx.selfloop_edges(G))
    lcc = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    mapping = {v: i for i, v in enumerate(lcc.nodes())}
    edges = [(mapping[u], mapping[v]) for u, v in lcc.edges()]
    n = lcc.number_of_nodes()
    return ig.Graph(n=n, edges=edges)


def load_snap_igraph(name):
    txt_path = download_snap(name)
    if txt_path is None:
        return None
    G = nx.Graph()
    with open(txt_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                G.add_edge(int(parts[0]), int(parts[1]))
    return nx_to_igraph(G)


def load_builtin_graphs():
    graphs = {}
    G = nx.karate_club_graph()
    graphs["karate"] = (nx_to_igraph(G), True)        # True = use exact MILP

    G = nx.les_miserables_graph()
    graphs["les_mis"] = (nx_to_igraph(G), True)

    # Football — download from Pajek
    football_path = os.path.join(DATA_DIR, "football.gml")
    if not os.path.exists(football_path):
        try:
            url = "http://www-personal.umich.edu/~mejn/netdata/football.zip"
            import zipfile, io
            with urllib.request.urlopen(url) as r:
                zf = zipfile.ZipFile(io.BytesIO(r.read()))
                for name in zf.namelist():
                    if name.endswith(".gml"):
                        with zf.open(name) as f, open(football_path, "wb") as out:
                            shutil.copyfileobj(f, out)
                        break
        except Exception:
            football_path = None

    if football_path and os.path.exists(football_path):
        try:
            G = nx.read_gml(football_path, label="id")
            graphs["football"] = (nx_to_igraph(G), True)
        except Exception:
            pass

    # Political books — download
    polbooks_path = os.path.join(DATA_DIR, "polbooks.gml")
    if not os.path.exists(polbooks_path):
        try:
            url = "http://www-personal.umich.edu/~mejn/netdata/polbooks.zip"
            import zipfile, io
            with urllib.request.urlopen(url) as r:
                zf = zipfile.ZipFile(io.BytesIO(r.read()))
                for name in zf.namelist():
                    if name.endswith(".gml"):
                        with zf.open(name) as f, open(polbooks_path, "wb") as out:
                            shutil.copyfileobj(f, out)
                        break
        except Exception:
            polbooks_path = None

    if polbooks_path and os.path.exists(polbooks_path):
        try:
            G = nx.read_gml(polbooks_path, label="id")
            graphs["polbooks"] = (nx_to_igraph(G), True)
        except Exception:
            pass

    return graphs


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(name, g, use_exact=False, no_exact=False):
    V = g.vcount()
    E = g.ecount()

    t0 = time.perf_counter()
    sp_cover = spectral_greedy_large(g)
    sp_t = time.perf_counter() - t0

    t0 = time.perf_counter()
    dg_cover = degree_greedy_large(g)
    dg_t = time.perf_counter() - t0

    # Skip SA for very large graphs (O(V²) SA becomes intractable)
    if V > 50_000:
        sa_cover, sa_t, sa_sz = [], 0.0, None
    else:
        t0 = time.perf_counter()
        sa_cover = simulated_annealing_igraph(g)
        sa_t = time.perf_counter() - t0

    sp_sz = len(sp_cover)
    dg_sz = len(dg_cover)
    sa_sz = len(sa_cover) if sa_cover else None
    vbs   = min(sp_sz, dg_sz, *([sa_sz] if sa_sz is not None else []))

    # Validity checks
    sp_valid = check_is_vertex_cover_igraph(g, set(sp_cover))
    dg_valid = check_is_vertex_cover_igraph(g, set(dg_cover))
    sa_valid = check_is_vertex_cover_igraph(g, set(sa_cover)) if sa_cover else None

    # Exact reference
    exact_sz = None
    if use_exact and not no_exact and V <= 200:
        exact_sz = milp_exact(g)

    ref     = exact_sz if exact_sz is not None else vbs
    ref_lbl = "MILP" if exact_sz is not None else "VBS"

    sp_ratio = sp_sz / ref
    dg_ratio = dg_sz / ref
    sa_ratio = (sa_sz / ref) if sa_sz is not None else float("nan")

    return dict(
        name=name, V=V, E=E,
        sp=sp_sz, dg=dg_sz, sa=sa_sz, vbs=vbs, ref=ref, ref_lbl=ref_lbl,
        sp_ratio=sp_ratio, dg_ratio=dg_ratio, sa_ratio=sa_ratio,
        sp_valid=sp_valid, dg_valid=dg_valid, sa_valid=sa_valid,
        sp_t=sp_t, dg_t=dg_t, sa_t=sa_t,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-exact", action="store_true",
                        help="Skip MILP exact solver (faster)")
    parser.add_argument("--no-snap", action="store_true",
                        help="Skip SNAP dataset download")
    args = parser.parse_args()

    print("Warming up Numba JIT...", end=" ", flush=True)
    warmup_jit()
    print("done\n")

    results = []

    # ── Built-in networkx graphs ────────────────────────────────────────────
    print("Loading built-in graphs...")
    builtin = load_builtin_graphs()
    for name, (g, use_exact) in builtin.items():
        print(f"  Benchmarking {name} (V={g.vcount()}, E={g.ecount()})...",
              end=" ", flush=True)
        r = run_benchmark(name, g, use_exact=use_exact, no_exact=args.no_exact)
        results.append(r)
        ref_lbl = r["ref_lbl"]
        sa_str = f"{r['sa']}" if r['sa'] is not None else "N/A"
        print(f"sp={r['sp']} dg={r['dg']} sa={sa_str} "
              f"({ref_lbl}={r['ref']}) "
              f"sp/ref={r['sp_ratio']:.3f}")

    # ── SNAP datasets ───────────────────────────────────────────────────────
    if not args.no_snap:
        print("\nLoading SNAP datasets...")
        for snap_name in ["email-Eu", "ca-GrQc", "ca-HepTh", "ca-AstroPh",
                          "p2p-Gnut04", "roadNet-PA"]:
            g = load_snap_igraph(snap_name)
            if g is None:
                print(f"  {snap_name}: skipped (download failed)")
                continue
            print(f"  Benchmarking {snap_name} (V={g.vcount()}, E={g.ecount()})...",
                  end=" ", flush=True)
            r = run_benchmark(snap_name, g, use_exact=False)
            results.append(r)
            sa_str = f"{r['sa']}" if r['sa'] is not None else "N/A"
            print(f"sp={r['sp']} dg={r['dg']} sa={sa_str} "
                  f"VBS={r['vbs']} "
                  f"sp/VBS={r['sp_ratio']:.3f}")

    # ── Summary table ───────────────────────────────────────────────────────
    print()
    print("=" * 90)
    print(f"{'Graph':<14} {'V':>9} {'E':>9}  {'Ref':>7}  "
          f"{'sp_cov':>7} {'dg_cov':>7} {'sa_cov':>7}  "
          f"{'sp/ref':>7} {'dg/ref':>7} {'sa/ref':>7}  "
          f"{'REF':>5}")
    print("-" * 100)
    for r in results:
        sa_cov_str   = f"{r['sa']:>7}" if r['sa'] is not None else "    N/A"
        sa_ratio_str = f"{r['sa_ratio']:>7.3f}" if r['sa'] is not None else "    N/A"
        print(f"{r['name']:<14} {r['V']:>9} {r['E']:>9}  {r['ref']:>7}  "
              f"{r['sp']:>7} {r['dg']:>7} {sa_cov_str}  "
              f"{r['sp_ratio']:>7.3f} {r['dg_ratio']:>7.3f} {sa_ratio_str}  "
              f"{r['ref_lbl']:>5}")
    print("=" * 90)

    # ── Aggregate statistics ─────────────────────────────────────────────
    sp_ratios = [r["sp_ratio"] for r in results]
    dg_ratios = [r["dg_ratio"] for r in results]
    sa_ratios = [r["sa_ratio"] for r in results if r["sa"] is not None]

    print(f"\nPooled over {len(results)} real-world instances:")
    print(f"  Spectral greedy : mean={np.mean(sp_ratios):.4f}, "
          f"max={np.max(sp_ratios):.4f}, "
          f"wins vs DG={sum(s<d for s,d in zip(sp_ratios,dg_ratios))}/{len(results)}")
    print(f"  Degree greedy   : mean={np.mean(dg_ratios):.4f}, "
          f"max={np.max(dg_ratios):.4f}")
    print(f"  Simul. Annealing: mean={np.mean(sa_ratios):.4f}, "
          f"max={np.max(sa_ratios):.4f}")

    print("\nValidity check (all should be True):")
    for r in results:
        ok = r["sp_valid"] and r["dg_valid"] and r["sa_valid"]
        print(f"  {r['name']:<14}: sp={r['sp_valid']} dg={r['dg_valid']} "
              f"sa={r['sa_valid']}  {'OK' if ok else '*** INVALID ***'}")

    # ── Save CSV ────────────────────────────────────────────────────────────
    import csv
    csv_path = os.path.join(os.path.dirname(__file__),
                            "results_real_graphs.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {csv_path}")
