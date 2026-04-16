"""
Large-scale heuristics for Minimum Vertex Cover using igraph + Numba.

Provides the same algorithms as heuristics.py but tuned for graphs with
V ~ 10^3 – 10^6 vertices by combining:
  - igraph (C-based) for graph generation and I/O
  - Technique 1: incremental nbr_sum maintenance (O(1) score update per dirty vertex)
  - Technique 2: indexed max-heap fully inside Numba (zero Python overhead per update)
  - cache=True: LLVM bitcode saved to __pycache__; first run compiles (~5-10 min),
    subsequent runs load from cache (~3 s startup)
  - Degree greedy: Numba JIT linear scan

Algorithm complexity: O(E · k_avg · log V)
Observed: n=1M BA graph in ~5 s after cache warm (vs ~16 000 s naive Python version)

Public API
----------
spectral_greedy_large(g)       — quantum-inspired spectral greedy (Algorithm 2)
degree_greedy_large(g)         — maximum-degree greedy baseline
warmup_jit()                   — pre-compile / load Numba cache (call once)
check_is_vertex_cover_igraph() — validity check
"""

import heapq
import numpy as np
import scipy.sparse as sp

try:
    import igraph as ig
    IGRAPH_AVAILABLE = True
except ImportError:
    IGRAPH_AVAILABLE = False

try:
    import numba as nb
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _igraph_to_csr(g):
    """Convert igraph.Graph to scipy CSR (undirected, symmetric)."""
    n = g.vcount()
    if g.ecount() == 0:
        return sp.csr_matrix((n, n), dtype=np.float64)
    src, dst = map(np.array, zip(*g.get_edgelist()))
    row = np.concatenate([src, dst]).astype(np.int32)
    col = np.concatenate([dst, src]).astype(np.int32)
    data = np.ones(len(row), dtype=np.float64)
    return sp.csr_matrix((data, (row, col)), shape=(n, n))


def check_is_vertex_cover_igraph(g, cover_set: set) -> bool:
    """Verify cover validity using igraph edge list."""
    return all(u in cover_set or v in cover_set for u, v in g.get_edgelist())


# ---------------------------------------------------------------------------
# Numba heap primitives (indexed max-heap)
#
# Invariants:
#   heap[i]  = vertex at position i
#   pos[v]   = position of v in heap (-1 if not present)
#   score[v] = ordering key (max at heap[0])
#
# cache=True saves compiled LLVM bitcode to __pycache__ so that subsequent
# runs skip the ~5-minute compilation and load in ~3 seconds instead.
# ---------------------------------------------------------------------------

if NUMBA_AVAILABLE:

    @nb.njit(cache=True)
    def _sift_up(heap, pos, score, i):
        """Bubble heap[i] toward the root until max-heap property holds."""
        while i > 0:
            p = (i - 1) >> 1
            if score[heap[i]] > score[heap[p]]:
                tmp = heap[i]; heap[i] = heap[p]; heap[p] = tmp
                pos[heap[i]] = i;  pos[heap[p]] = p
                i = p
            else:
                break

    @nb.njit(cache=True)
    def _sift_down(heap, pos, score, sz, i):
        """Push heap[i] downward until max-heap property holds."""
        while True:
            mx = i
            l = (i << 1) + 1
            r = l + 1
            if l < sz and score[heap[l]] > score[heap[mx]]:
                mx = l
            if r < sz and score[heap[r]] > score[heap[mx]]:
                mx = r
            if mx == i:
                break
            tmp = heap[i]; heap[i] = heap[mx]; heap[mx] = tmp
            pos[heap[i]] = i;  pos[heap[mx]] = mx
            i = mx

    @nb.njit(cache=True)
    def _heap_fix(heap, pos, score, sz, v):
        """Restore heap property for vertex v after its score changed."""
        i = pos[v]
        if i < 0:
            return
        _sift_up(heap, pos, score, i)
        _sift_down(heap, pos, score, sz, pos[v])

    # -----------------------------------------------------------------------
    # Core algorithm — full Numba, all ops inlined via cached sub-functions
    # -----------------------------------------------------------------------

    @nb.njit(cache=True)
    def _spectral_greedy_numba(indptr, indices, n):
        """
        Spectral greedy — fully compiled, zero Python boundary crossings.

        Technique 1 — incremental nbr_sum:
          Maintains nbr_sum[v] = Σ_{j∈N(v), active} inv_deg[j].
          Removing m* propagates Δ = Δinv_deg[j] to 2-hop neighbours
          in O(1) per vertex (always Δ > 0 since degrees only decrease).

        Technique 2 — indexed max-heap inside Numba:
          heap/pos arrays give O(log V) extract-max and key-update with
          no Python call overhead per vertex (vs ~7 M heappush calls for
          n=1 M with Python heapq, each costing ~500 ns).

        Compilation: first run ~5–10 min (LLVM IR + optimisation);
                     cached runs load in ~3 s.

        Complexity: O(E · k_avg · log V).
        """
        # ── Initialise ───────────────────────────────────────────────────
        deg = np.empty(n, dtype=np.float64)
        for v in range(n):
            deg[v] = float(indptr[v + 1] - indptr[v])

        inv_deg = np.zeros(n, dtype=np.float64)
        for v in range(n):
            if deg[v] > 0.0:
                inv_deg[v] = 1.0 / deg[v]

        # nbr_sum[v] = Σ_{j∈N(v)} inv_deg[j]
        nbr_sum = np.zeros(n, dtype=np.float64)
        for v in range(n):
            for ki in range(indptr[v], indptr[v + 1]):
                nbr_sum[v] += inv_deg[indices[ki]]

        score = np.empty(n, dtype=np.float64)
        for v in range(n):
            score[v] = inv_deg[v] * nbr_sum[v]

        active = np.ones(n, dtype=np.bool_)

        # ── Build indexed max-heap (Floyd O(N)) ──────────────────────────
        heap = np.empty(n, dtype=np.int64)
        pos  = np.full(n, -1, dtype=np.int64)
        sz   = 0

        for v in range(n):
            if deg[v] > 0.0:
                heap[sz] = v
                pos[v]   = sz
                sz      += 1

        for i in range(sz // 2 - 1, -1, -1):
            _sift_down(heap, pos, score, sz, i)

        cover = np.empty(n, dtype=np.int64)
        csz   = 0

        # ── Main greedy loop ─────────────────────────────────────────────
        while sz > 0:
            best = heap[0]

            # All remaining vertices have deg=0 → all edges are covered.
            if deg[best] <= 0.0:
                break

            cover[csz] = best
            csz += 1
            active[best] = False

            # Pop root: move last element to root, sift down.
            sz -= 1
            if sz > 0:
                heap[0]      = heap[sz]
                pos[heap[0]] = 0
                _sift_down(heap, pos, score, sz, 0)
            pos[best] = -1

            # ── Incremental update (Technique 1) ─────────────────────────
            for ki in range(indptr[best], indptr[best + 1]):
                j = indices[ki]
                if not active[j]:
                    continue

                old_inv_j  = inv_deg[j]
                nbr_sum[j] -= inv_deg[best]
                deg[j]     -= 1.0
                inv_deg[j]  = 1.0 / deg[j] if deg[j] > 0.0 else 0.0
                delta       = inv_deg[j] - old_inv_j   # ≥ 0

                # Propagate to 2-hop neighbours: Δ > 0 → sift up only.
                for kk in range(indptr[j], indptr[j + 1]):
                    k = indices[kk]
                    if active[k] and k != best:
                        nbr_sum[k] += delta
                        score[k]    = inv_deg[k] * nbr_sum[k]
                        pk = pos[k]
                        if pk > 0:
                            _sift_up(heap, pos, score, pk)

                # j's score may go either direction → full fix.
                score[j] = inv_deg[j] * nbr_sum[j]
                _heap_fix(heap, pos, score, sz, j)

        return cover[:csz]

    # -----------------------------------------------------------------------
    # Degree greedy core
    # -----------------------------------------------------------------------

    @nb.njit(cache=True)
    def _degree_greedy_core(indptr, indices, n):
        """Maximum-degree greedy inner loop (O(V²), Numba JIT)."""
        degrees    = np.zeros(n, dtype=np.float64)
        for v in range(n):
            degrees[v] = float(indptr[v + 1] - indptr[v])

        active     = np.ones(n, dtype=np.bool_)
        cover      = np.empty(n, dtype=np.int32)
        cover_size = 0

        while True:
            best_deg = -1.0
            best_v   = -1
            for v in range(n):
                if active[v] and degrees[v] > best_deg:
                    best_deg = degrees[v]
                    best_v   = v

            if best_v == -1 or best_deg == 0.0:
                break

            cover[cover_size] = best_v
            cover_size       += 1
            active[best_v]    = False
            degrees[best_v]   = 0.0

            for k in range(indptr[best_v], indptr[best_v + 1]):
                j = indices[k]
                if active[j]:
                    degrees[j] -= 1.0

        return cover[:cover_size]


# ---------------------------------------------------------------------------
# Public wrappers
# ---------------------------------------------------------------------------

def spectral_greedy_large(g) -> list:
    """
    Quantum-inspired spectral greedy vertex cover for large graphs.

    Score: s(v) = (1/d_v) Σ_{j∈N(v)} 1/d_j  (two-hop spectral quantity,
    Eq. (eq:short_time) of the paper).

    Uses Technique 1 (incremental nbr_sum) + Technique 2 (Numba indexed
    max-heap) when Numba is available; falls back to pure Python otherwise.

    NOTE — first-run compilation:
      The Numba kernels are compiled once and cached to __pycache__.
      First invocation takes ~5–10 min; subsequent runs load in ~3 s.
      Call warmup_jit() at script start to trigger compilation early.

    Parameters
    ----------
    g : igraph.Graph (undirected)

    Returns
    -------
    list of int : vertex indices forming a valid vertex cover
    """
    if not IGRAPH_AVAILABLE:
        raise ImportError("igraph required: pip install igraph")

    n = g.vcount()
    if g.ecount() == 0:
        return []

    if NUMBA_AVAILABLE:
        A       = _igraph_to_csr(g)
        indptr  = A.indptr.astype(np.int64)
        indices = A.indices.astype(np.int64)
        return list(_spectral_greedy_numba(indptr, indices, n))

    # ── Pure-Python fallback (Technique 1 only) ───────────────────────────
    A       = _igraph_to_csr(g)
    indptr  = A.indptr.astype(np.int64)
    indices = A.indices.astype(np.int64)

    deg     = np.asarray(A.sum(axis=1)).ravel().astype(np.float64)
    inv_deg = np.where(deg > 0, 1.0 / np.where(deg > 0, deg, 1.0), 0.0)
    nbr_sum = (A @ inv_deg).astype(np.float64)
    score   = inv_deg * nbr_sum
    active  = np.ones(n, dtype=bool)

    version = np.zeros(n, dtype=np.int64)
    heap = [(-score[v], int(v), 0) for v in range(n) if deg[v] > 0]
    heapq.heapify(heap)
    cover = []

    while heap:
        neg_s, best, ver = heapq.heappop(heap)
        if not active[best] or ver != version[best] or deg[best] == 0:
            continue
        cover.append(best)
        active[best] = False

        for ki in range(int(indptr[best]), int(indptr[best + 1])):
            j = int(indices[ki])
            if not active[j]:
                continue
            old_inv_j   = inv_deg[j]
            nbr_sum[j] -= inv_deg[best]
            deg[j]     -= 1.0
            inv_deg[j]  = 1.0 / deg[j] if deg[j] > 0.0 else 0.0
            delta       = inv_deg[j] - old_inv_j

            for kk in range(int(indptr[j]), int(indptr[j + 1])):
                k = int(indices[kk])
                if active[k] and k != best:
                    nbr_sum[k] += delta
                    score[k]    = inv_deg[k] * nbr_sum[k]
                    version[k] += 1
                    heapq.heappush(heap, (-score[k], k, int(version[k])))

            score[j] = inv_deg[j] * nbr_sum[j]
            version[j] += 1
            heapq.heappush(heap, (-score[j], j, int(version[j])))

    return cover


def degree_greedy_large(g) -> list:
    """
    Maximum-degree greedy vertex cover for large graphs (igraph input).

    Parameters
    ----------
    g : igraph.Graph (undirected)

    Returns
    -------
    list of int : vertex indices forming a valid vertex cover
    """
    if not IGRAPH_AVAILABLE:
        raise ImportError("igraph required")

    n = g.vcount()
    if g.ecount() == 0:
        return []

    if NUMBA_AVAILABLE:
        A       = _igraph_to_csr(g)
        indptr  = A.indptr.astype(np.int64)
        indices = A.indices.astype(np.int32)
        return list(_degree_greedy_core(indptr, indices, n))

    # Pure-numpy fallback
    A       = _igraph_to_csr(g)
    indptr  = A.indptr
    indices = A.indices
    degrees = np.array(A.sum(axis=1)).ravel()
    active  = np.ones(n, dtype=bool)
    cover   = []
    while True:
        mask = active & (degrees > 0)
        if not mask.any():
            break
        best = int(np.argmax(np.where(mask, degrees, -1.0)))
        cover.append(best)
        active[best]  = False
        degrees[best] = 0.0
        for j in indices[indptr[best]:indptr[best + 1]]:
            if active[j]:
                degrees[j] -= 1.0
    return cover


# ---------------------------------------------------------------------------
# JIT warm-up helper
# ---------------------------------------------------------------------------

def warmup_jit():
    """
    Pre-compile (or load from cache) all Numba kernels.

    Call once at script start.  First call: ~5–10 min compilation.
    Subsequent calls on the same machine: ~3 s cache load.
    """
    if not IGRAPH_AVAILABLE or not NUMBA_AVAILABLE:
        return
    g = ig.Graph.Barabasi(20, 2, directed=False)
    spectral_greedy_large(g)
    degree_greedy_large(g)
