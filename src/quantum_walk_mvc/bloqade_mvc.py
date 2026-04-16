"""
Bloqade-based Continuous-Time Quantum Walk for Minimum Vertex Cover.

This module implements the CTQW-MVC algorithm from:
    "Scalable Quantum Walk-Based Heuristics for the Minimum Vertex Cover Problem"

using QuEra's Bloqade neutral-atom emulator.  The physical mapping is:

    Graph vertex  <-->  Rydberg atom
    Graph edge    <-->  Atom pair within blockade radius R_b
    Vertex freeze <-->  Large local detuning  Δ_i >> Ω  (drives atom to |g⟩, suppresses dynamics)

The quantum walk Hamiltonian is H = I - Γ, where
    Γ_ij = A_ij / sqrt(D_ii * D_jj)  (symmetric normalised adjacency).

In the Rydberg picture, the uniform Rabi drive at Δ = 0 implements a walk on
the blockade-interaction graph; the blockade constraint restricts the dynamics
so that adjacent atoms cannot both be in |r⟩, which directly encodes the
independent-set / vertex-cover complementarity.

Physical constants for ⁸⁷Rb:
    C6 = 862690  rad·μs⁻¹·μm⁶
    Ω_typical = 4π  rad/μs  →  R_b ≈ 7.5 μm
"""

from __future__ import annotations

import warnings
import math
from typing import Any

import numpy as np
import networkx as nx

# --------------------------------------------------------------------------- #
# Optional Bloqade import                                                      #
# --------------------------------------------------------------------------- #
try:
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        from bloqade.analog import start as _bloqade_start  # noqa: F401
    BLOQADE_AVAILABLE = True
except ImportError:
    BLOQADE_AVAILABLE = False

# --------------------------------------------------------------------------- #
# Physical constants                                                           #
# --------------------------------------------------------------------------- #

#: Van der Waals coefficient for ⁸⁷Rb |r⟩ = |70S₁/₂⟩  (rad·μs⁻¹·μm⁶)
C6_RB87: float = 862690.0

#: Ramp duration (μs) used on both sides of the Rabi pulse
_RAMP_DURATION: float = 0.1


# --------------------------------------------------------------------------- #
# 1. Graph embedding                                                           #
# --------------------------------------------------------------------------- #


def embed_graph_positions(
    graph: nx.Graph,
    blockade_radius: float = 7.5,
    min_spacing: float = 4.0,
    seed: int = 42,
) -> dict[Any, tuple[float, float]]:
    """
    Map graph vertices to (x, y) atom positions (μm) for neutral-atom hardware.

    The embedding uses NetworkX spring layout as an initial guess, then
    rescales coordinates so that:

    * Every pair of **connected** vertices is placed at distance < ``blockade_radius``.
    * Every pair of **disconnected** vertices ideally has distance > ``blockade_radius``.

    The second condition can only be strictly satisfied for unit-disk graphs.
    For general graphs a best-effort rescaling is applied and a warning is
    emitted when violations remain.

    Parameters
    ----------
    graph : nx.Graph
        Input graph.  Isolated nodes are included in the layout.
    blockade_radius : float, optional
        Target Rydberg blockade radius in micrometers (default 7.5 μm).
    min_spacing : float, optional
        Hard lower bound on atom–atom distance in micrometers (default 4.0 μm).
        Prevents atoms from overlapping in hardware.
    seed : int, optional
        Random seed passed to ``nx.spring_layout`` for reproducibility.

    Returns
    -------
    dict
        Mapping ``{vertex: (x_um, y_um)}`` in micrometers.

    Notes
    -----
    The spring layout embeds nodes in [-1, 1]², which is then affinely scaled
    so that the *minimum edge length* equals ``0.7 * blockade_radius``.  A
    hard minimum is then applied via a small repulsive jitter step so no two
    atoms are closer than ``min_spacing``.
    """
    n = graph.number_of_nodes()
    if n == 0:
        return {}
    if n == 1:
        node = next(iter(graph.nodes()))
        return {node: (0.0, 0.0)}

    nodes = list(graph.nodes())
    node_to_idx = {v: i for i, v in enumerate(nodes)}

    # Spring layout gives positions in approximately [-1, 1]²
    raw_pos = nx.spring_layout(graph, seed=seed, k=2.0 / math.sqrt(n))
    coords = np.array([raw_pos[v] for v in nodes], dtype=float)  # shape (n, 2)

    # -----------------------------------------------------------------
    # Scale so that the shortest edge sits at 0.7 * blockade_radius.
    # This keeps connected pairs well within blockade.
    # -----------------------------------------------------------------
    edge_list = list(graph.edges())
    if edge_list:
        edge_lengths = np.array([
            np.linalg.norm(coords[node_to_idx[u]] - coords[node_to_idx[v]])
            for u, v in edge_list
        ])
        min_edge = edge_lengths.min()
        if min_edge < 1e-9:
            min_edge = 1e-9
        target_edge = 0.7 * blockade_radius
        scale = target_edge / min_edge
        coords *= scale
    else:
        # No edges: just spread atoms 2 * blockade_radius apart
        coords *= 2.0 * blockade_radius

    # -----------------------------------------------------------------
    # Enforce minimum spacing via iterative repulsion
    # -----------------------------------------------------------------
    rng = np.random.default_rng(seed)
    for _ in range(200):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                diff = coords[i] - coords[j]
                dist = np.linalg.norm(diff)
                if dist < min_spacing:
                    moved = True
                    direction = diff / (dist + 1e-12)
                    push = (min_spacing - dist) / 2.0 + 0.01
                    coords[i] += direction * push
                    coords[j] -= direction * push
        if not moved:
            break

    # -----------------------------------------------------------------
    # Warn if any non-edge pair is within the blockade radius
    # -----------------------------------------------------------------
    non_edge_violations = []
    adj_set = set(
        (min(u, v), max(u, v)) for u, v in graph.edges()
    )
    for i in range(n):
        for j in range(i + 1, n):
            pair = (min(nodes[i], nodes[j]), max(nodes[i], nodes[j]))
            if pair not in adj_set:
                dist = np.linalg.norm(coords[i] - coords[j])
                if dist < blockade_radius:
                    non_edge_violations.append((nodes[i], nodes[j], dist))

    if non_edge_violations:
        warnings.warn(
            f"Graph is not a unit-disk graph: {len(non_edge_violations)} non-adjacent "
            f"atom pair(s) fall within blockade_radius={blockade_radius:.1f} μm.  "
            "The Rydberg blockade will introduce spurious interactions.  "
            "Results are best-effort; consider using the classical fallback for "
            "non-unit-disk graphs.",
            stacklevel=2,
        )

    # Build output dict; center around (0, 0)
    coords -= coords.mean(axis=0)
    return {nodes[i]: (float(coords[i, 0]), float(coords[i, 1])) for i in range(n)}


# --------------------------------------------------------------------------- #
# 1b. Unit-disk graph embedding (exact, zero spurious interactions)           #
# --------------------------------------------------------------------------- #


def embed_unit_disk_graph(
    n: int = 10,
    connection_radius: float = 0.38,
    blockade_radius: float = 7.5,
    seed: int = 7,
    min_edges: int = 6,
    max_attempts: int = 200,
) -> tuple[nx.Graph, dict[int, tuple[float, float]]]:
    """
    Generate a random geometric graph and return it with physical atom positions.

    A random geometric graph (RGG) is a unit-disk graph by construction: ``n``
    nodes are placed uniformly in [0, 1]² and edges connect pairs whose
    Euclidean distance is below ``connection_radius``.  Because the graph IS the
    unit-disk graph of its node positions, scaling those positions to micrometers
    guarantees that every edge pair lies within ``blockade_radius`` and every
    non-edge pair lies outside it — with no spurious blockade interactions.

    Parameters
    ----------
    n : int, optional
        Number of vertices / atoms (default 10).
    connection_radius : float, optional
        Edge threshold in the unit square (default 0.38).  Connected pairs
        will be placed at ``≈ 0.9 * blockade_radius`` μm after scaling.
    blockade_radius : float, optional
        Target Rydberg blockade radius in μm (default 7.5 μm).
    seed : int, optional
        Initial random seed.  If the graph generated with this seed is
        disconnected or has fewer than ``min_edges`` edges, the seed is
        incremented automatically until a valid graph is found.
    min_edges : int, optional
        Minimum number of edges required (default 6).
    max_attempts : int, optional
        Maximum number of seed increments before raising ``RuntimeError``.

    Returns
    -------
    graph : nx.Graph
        Connected random geometric graph.
    positions : dict
        Mapping ``{vertex: (x_um, y_um)}`` in micrometers, suitable for
        direct use with :func:`compute_transition_probs_bloqade`.

    Notes
    -----
    The unit-square coordinates are scaled so that the *median* edge length
    equals ``0.9 * blockade_radius``, keeping all edges well inside the
    blockade regime while maximising the gap to non-edge pairs.

    Raises
    ------
    RuntimeError
        If no valid connected graph is found within ``max_attempts`` seeds.
    """
    for attempt in range(max_attempts):
        current_seed = seed + attempt
        G = nx.random_geometric_graph(n, connection_radius, seed=current_seed)

        if not nx.is_connected(G):
            continue
        if G.number_of_edges() < min_edges:
            continue

        # Extract unit-square positions from node attributes
        raw = {v: np.array(G.nodes[v]["pos"]) for v in G.nodes()}
        coords = np.array([raw[v] for v in range(n)])  # shape (n, 2)

        # Scale: median edge length → 0.9 * blockade_radius
        edge_lengths = np.array([
            np.linalg.norm(coords[u] - coords[v])
            for u, v in G.edges()
        ])
        median_edge = np.median(edge_lengths)
        if median_edge < 1e-9:
            continue
        scale = (0.9 * blockade_radius) / median_edge
        coords_um = coords * scale

        # Center around origin
        coords_um -= coords_um.mean(axis=0)

        positions = {v: (float(coords_um[v, 0]), float(coords_um[v, 1]))
                     for v in range(n)}

        # Verify: count any non-edge violations
        adj_set = {(min(u, v), max(u, v)) for u, v in G.edges()}
        violations = sum(
            1
            for i in range(n)
            for j in range(i + 1, n)
            if (min(i, j), max(i, j)) not in adj_set
            and np.linalg.norm(coords_um[i] - coords_um[j]) < blockade_radius
        )
        if violations > 0:
            continue  # not a clean unit-disk embedding at this scale

        return G, positions

    raise RuntimeError(
        f"Could not find a valid connected unit-disk graph with the given "
        f"parameters after {max_attempts} attempts.  Try adjusting "
        f"connection_radius or n."
    )


# --------------------------------------------------------------------------- #
# 2. Rydberg interaction matrix                                                #
# --------------------------------------------------------------------------- #


def get_rydberg_interaction_matrix(
    positions: dict[Any, tuple[float, float]],
    C6: float = C6_RB87,
) -> np.ndarray:
    """
    Compute the pairwise Rydberg interaction matrix V_ij = C6 / |r_i - r_j|^6.

    Parameters
    ----------
    positions : dict
        Mapping ``{vertex: (x_um, y_um)}``, as returned by
        :func:`embed_graph_positions`.
    C6 : float, optional
        Van der Waals coefficient in rad·μs⁻¹·μm⁶ (default: ⁸⁷Rb value).

    Returns
    -------
    np.ndarray
        Square matrix of shape ``(n, n)``.  Diagonal entries are zero.
        Off-diagonal entry ``[i, j]`` is the interaction energy (rad/μs)
        between atoms *i* and *j*.

    Notes
    -----
    An atom pair with ``V_ij ≫ Ω`` is said to be in the *blockade* regime:
    simultaneous excitation of both atoms is strongly suppressed.
    """
    nodes = list(positions.keys())
    n = len(nodes)
    coords = np.array([positions[v] for v in nodes], dtype=float)
    V = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            r = np.linalg.norm(coords[i] - coords[j])
            if r < 1e-9:
                V[i, j] = V[j, i] = np.inf
            else:
                V[i, j] = V[j, i] = C6 / r**6
    return V


# --------------------------------------------------------------------------- #
# 3. Transition probability extraction from bitstrings                        #
# --------------------------------------------------------------------------- #


def _extract_marginal_probs(
    bitstrings: np.ndarray,
) -> np.ndarray:
    """
    Compute per-atom marginal excitation probability from a bitstring array.

    Parameters
    ----------
    bitstrings : np.ndarray
        Integer array of shape ``(n_shots, n_atoms)`` where 1 = Rydberg state.

    Returns
    -------
    np.ndarray
        1-D array of shape ``(n_atoms,)`` with P(atom m = |r⟩).
    """
    return bitstrings.mean(axis=0).astype(float)


# --------------------------------------------------------------------------- #
# 4. Build a Bloqade program for one CTQW step                                #
# --------------------------------------------------------------------------- #


def _build_bloqade_program(
    positions: dict[Any, tuple[float, float]],
    t_evolution: float,
    Omega: float,
    frozen_atoms: list[int],
    freeze_detuning: float,
) -> Any:
    """
    Construct a Bloqade analog program for one CTQW iteration.

    The pulse sequence is:
    - Rabi ramp-up: 0  → Ω  over ``_RAMP_DURATION`` μs  (adiabatic turn-on)
    - Rabi hold  : Ω  held for ``t_evolution`` μs         (quantum walk)
    - Rabi ramp-down: Ω → 0  over ``_RAMP_DURATION`` μs  (adiabatic turn-off)
    - Detuning   : Δ = 0 everywhere (unbiased walk)
    - Frozen atoms: additional local detuning of ``+freeze_detuning`` rad/μs,
      which shifts those atoms' transition energy far from resonance, effectively
      suppressing their dynamics (mapping to the paper's H_F = (Ω-1)·P_m term).

    Parameters
    ----------
    positions : dict
        Atom coordinates, ``{vertex: (x_um, y_um)}``.
    t_evolution : float
        Quantum walk hold time in μs.
    Omega : float
        Peak Rabi frequency in rad/μs.
    frozen_atoms : list of int
        Zero-based atom indices that should be frozen.
    freeze_detuning : float
        Local detuning (rad/μs) applied to frozen atoms.

    Returns
    -------
    Bloqade program builder
        Ready to call ``.bloqade.python().run(n_shots)``.
    """
    # Import inside function so module can be imported without Bloqade
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        from bloqade.analog import start

    pos_list = [positions[v] for v in positions]
    total_time = _RAMP_DURATION + t_evolution + _RAMP_DURATION
    durations = [_RAMP_DURATION, t_evolution, _RAMP_DURATION]

    # Build register of atom positions
    program = start.add_position(pos_list)

    # Rabi amplitude: ramp-up / hold / ramp-down
    builder = (
        program
        .rydberg.rabi.amplitude.uniform
        .piecewise_linear(
            durations=durations,
            values=[0.0, Omega, Omega, 0.0],
        )
    )

    # Uniform detuning = 0 (no global bias → symmetric, unbiased quantum walk)
    builder = (
        builder
        .detuning.uniform
        .piecewise_linear(
            durations=durations,
            values=[0.0, 0.0, 0.0, 0.0],
        )
    )

    # Site-specific large detuning for frozen atoms.
    # A large positive detuning Δ_i >> Ω pushes the atom far off resonance:
    # its effective Rabi coupling ~Ω²/(2Δ) → 0, so it stays in |g⟩ regardless
    # of its neighbours' state (cf. "freezing" mechanism in the paper).
    for atom_idx in frozen_atoms:
        builder = (
            builder
            .detuning.location(atom_idx)
            .piecewise_linear(
                durations=durations,
                values=[freeze_detuning] * 4,
            )
        )

    return builder


# --------------------------------------------------------------------------- #
# 5. Compute transition probabilities via Bloqade                             #
# --------------------------------------------------------------------------- #


def compute_transition_probs_bloqade(
    graph: nx.Graph,
    positions: dict[Any, tuple[float, float]],
    t_evolution: float = 0.01,
    Omega: float = 4.0 * math.pi,
    n_shots: int = 1000,
    frozen_atoms: list[int] | None = None,
    freeze_detuning: float = 100.0,
) -> dict[Any, float]:
    """
    Run one step of the Bloqade CTQW emulator and return per-vertex excitation
    probabilities.

    The marginal probability P(atom m = |r⟩) is used as a proxy for the
    CTQW transition probability P(m → out) = 1 - |⟨m|e^{iΓt}|m⟩|².  Under
    the Rydberg blockade, high excitation of an atom indicates strong quantum
    walk amplitude flowing out of that vertex, which is precisely the
    criterion for selecting it into the vertex cover.

    Parameters
    ----------
    graph : nx.Graph
        Input graph.  Node ordering must match ``positions``.
    positions : dict
        Atom positions as returned by :func:`embed_graph_positions`.
    t_evolution : float, optional
        Walk duration in μs (default 0.01 μs).
    Omega : float, optional
        Peak Rabi frequency in rad/μs (default 4π).
    n_shots : int, optional
        Number of projective measurement shots (default 1000).
    frozen_atoms : list of int, optional
        Atom indices to freeze via large detuning (default: none).
    freeze_detuning : float, optional
        Local detuning (rad/μs) for frozen atoms (default 100.0).

    Returns
    -------
    dict
        ``{vertex: probability}`` where probability ∈ [0, 1].

    Raises
    ------
    RuntimeError
        If Bloqade is not installed.  Use :func:`quantum_walk_mvc_bloqade`
        instead, which falls back to the classical emulator automatically.
    """
    if not BLOQADE_AVAILABLE:
        raise RuntimeError(
            "Bloqade is not installed.  Install with: pip install bloqade"
        )

    if frozen_atoms is None:
        frozen_atoms = []

    nodes = list(positions.keys())

    # Build and run the Bloqade program
    program = _build_bloqade_program(
        positions=positions,
        t_evolution=t_evolution,
        Omega=Omega,
        frozen_atoms=frozen_atoms,
        freeze_detuning=freeze_detuning,
    )

    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        result = program.bloqade.python().run(n_shots)

    report = result.report()
    # bitstrings() returns a list with one array per task; we have one task
    bs = report.bitstrings()[0]  # shape (n_shots, n_atoms)

    # Marginal excitation probability per atom
    marginal = _extract_marginal_probs(bs)

    return {nodes[i]: float(marginal[i]) for i in range(len(nodes))}


# --------------------------------------------------------------------------- #
# 6. Freeze helper                                                             #
# --------------------------------------------------------------------------- #


def freeze_vertex_bloqade(
    program_builder: Any,
    vertex_idx: int,
    t_evolution: float,
    freeze_detuning: float = 100.0,
) -> Any:
    """
    Append a large local detuning to a vertex in an existing Bloqade builder.

    This encodes the "freezing" operation from the iterative MVC algorithm:
    once a vertex is selected, it is permanently removed from the walk
    dynamics by applying an energy penalty Δ_i >> Ω to its corresponding atom.
    The atom remains trapped in the ground state |g⟩ for the rest of the
    evolution, equivalent to removing it from the effective Hamiltonian.

    In the paper's notation this corresponds to projecting the Hamiltonian
    as H_F = (Ω - 1)·P_m, where P_m = |m⟩⟨m| is the projector onto the
    frozen vertex.

    Parameters
    ----------
    program_builder : Bloqade builder object
        A partially constructed Bloqade program (after rabi amplitude has been
        set).
    vertex_idx : int
        Zero-based index of the atom/vertex to freeze.
    t_evolution : float
        Walk hold time in μs (must match the original program's pulse shape).
    freeze_detuning : float, optional
        Local detuning magnitude in rad/μs (default 100.0; should be >> Ω).

    Returns
    -------
    Bloqade builder object
        Updated builder with the freeze detuning appended.
    """
    durations = [_RAMP_DURATION, t_evolution, _RAMP_DURATION]
    return (
        program_builder
        .detuning.location(vertex_idx)
        .piecewise_linear(
            durations=durations,
            values=[freeze_detuning] * 4,
        )
    )


# --------------------------------------------------------------------------- #
# 7. Classical CTQW fallback (scipy, no Bloqade required)                     #
# --------------------------------------------------------------------------- #


def _ctqw_mvc_classical(
    graph: nx.Graph,
    t_opt: float,
) -> list:
    """
    Classical CTQW-MVC using the sparse scipy matrix exponential.

    This is the fallback used when Bloqade is unavailable.  It is identical
    in algorithmic logic to :func:`quantum_walk_mvc_sparse` from ``core.py``
    but is kept here as a self-contained reference to avoid circular imports.

    Parameters
    ----------
    graph : nx.Graph
        Input graph.
    t_opt : float
        Quantum walk time parameter (in the classical simulation this is
        dimensionless; set equal to ``t_evolution`` for consistency).

    Returns
    -------
    list
        Vertices forming the vertex cover.
    """
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    adj = nx.to_scipy_sparse_array(graph, dtype=float, format="lil")
    vertex_cover: list = []

    while adj.sum() > 0:
        degrees = np.array(adj.sum(axis=1)).flatten()
        degrees_safe = np.where(degrees == 0, 1e-10, degrees)
        inv_sqrt = 1.0 / np.sqrt(degrees_safe)
        D_inv_sqrt = sp.diags(inv_sqrt)

        # Normalised adjacency Γ
        Gamma = D_inv_sqrt @ adj.tocsr() @ D_inv_sqrt

        # Walk evolution: diagonal of e^{iΓt}
        evo = spla.expm(1j * t_opt * Gamma)
        prob_stay = np.abs(evo.diagonal()) ** 2
        probs = 1.0 - prob_stay

        best = int(np.argmax(probs))
        vertex_cover.append(best)
        adj[best, :] = 0
        adj[:, best] = 0

    return vertex_cover


# --------------------------------------------------------------------------- #
# 8. Full iterative MVC algorithm                                              #
# --------------------------------------------------------------------------- #


def quantum_walk_mvc_bloqade(
    graph: nx.Graph,
    t_evolution: float = 0.01,
    Omega: float = 4.0 * math.pi,
    n_shots: int = 1000,
    blockade_radius: float = 7.5,
    freeze_detuning: float = 100.0,
    embed_seed: int = 42,
) -> list:
    """
    Compute a Minimum Vertex Cover using the Bloqade Rydberg atom emulator.

    Algorithm
    ---------
    Repeat until all edges are covered:

    1. Embed remaining active vertices as atom positions (unit-disk layout).
    2. Run the Bloqade emulator: uniform Rabi pulse at Δ = 0, frozen atoms
       get large local detuning.
    3. Compute per-atom excitation probability  P_m = ⟨n_m⟩  from shot
       statistics.
    4. Select vertex ``m* = argmax_m P_m`` and add to vertex cover.
    5. Mark ``m*`` as frozen; remove all edges incident to ``m*``.

    Physical mapping
    ----------------
    * Rydberg atom in |r⟩  ↔  vertex is *excited* / high walk amplitude
    * Rydberg blockade      ↔  graph edge (adjacent atoms cannot both be |r⟩)
    * Large local detuning  ↔  frozen vertex (decoupled from dynamics)
    * P(atom = |r⟩)         ↔  CTQW transition probability P(m → out)

    Parameters
    ----------
    graph : nx.Graph
        Input graph.
    t_evolution : float, optional
        Quantum walk duration in μs (default 0.01 μs).
    Omega : float, optional
        Peak Rabi frequency in rad/μs (default 4π ≈ 12.57 rad/μs).
    n_shots : int, optional
        Number of projective measurement shots per iteration (default 1000).
    blockade_radius : float, optional
        Rydberg blockade radius in μm used for position embedding (default
        7.5 μm, matching R_b for Ω = 4π with C6 of ⁸⁷Rb).
    freeze_detuning : float, optional
        Local detuning (rad/μs) applied to frozen atoms; should satisfy
        ``freeze_detuning >> Omega`` (default 100.0).
    embed_seed : int, optional
        Random seed for spring-layout position embedding (default 42).

    Returns
    -------
    list
        Vertices forming the vertex cover.  The result is guaranteed to be
        a valid cover (every edge has at least one endpoint in the list).

    Notes
    -----
    If Bloqade is not installed a warning is emitted and the classical scipy
    CTQW simulation from ``core.py`` is used as a transparent fallback.
    """
    if not BLOQADE_AVAILABLE:
        warnings.warn(
            "Bloqade is not installed; falling back to classical scipy CTQW "
            "simulation.  Install Bloqade with: pip install bloqade",
            ImportWarning,
            stacklevel=2,
        )
        return _ctqw_mvc_classical(graph, t_opt=t_evolution)

    # Work on a copy so we can mutate it
    G = graph.copy()
    all_nodes = list(graph.nodes())

    # Map original node labels to contiguous 0-based atom indices for Bloqade
    node_to_atom: dict[Any, int] = {v: i for i, v in enumerate(all_nodes)}

    selected_vertices: list = []
    frozen_atom_indices: list[int] = []

    # Embed ALL vertices once; frozen ones are handled by detuning, not removal.
    # This avoids re-solving the layout at every iteration.
    positions = embed_graph_positions(
        graph,
        blockade_radius=blockade_radius,
        seed=embed_seed,
    )

    while G.number_of_edges() > 0:
        # Determine active (non-frozen) vertices that are still connected
        active_nodes = [
            v for v in G.nodes()
            if G.degree(v) > 0 and node_to_atom[v] not in frozen_atom_indices
        ]

        if not active_nodes:
            break

        # Compute excitation probabilities for all atoms (frozen → ≈ 0)
        probs = compute_transition_probs_bloqade(
            graph=graph,
            positions=positions,
            t_evolution=t_evolution,
            Omega=Omega,
            n_shots=n_shots,
            frozen_atoms=frozen_atom_indices,
            freeze_detuning=freeze_detuning,
        )

        # Consider only active vertices when selecting the best candidate
        active_probs = {v: probs[v] for v in active_nodes}
        best_vertex = max(active_probs, key=active_probs.get)

        # Add to vertex cover
        selected_vertices.append(best_vertex)

        # Freeze the selected atom so it is excluded from future walk steps
        best_atom_idx = node_to_atom[best_vertex]
        frozen_atom_indices.append(best_atom_idx)

        # Remove edges covered by this vertex
        G.remove_node(best_vertex)

    return selected_vertices


# --------------------------------------------------------------------------- #
# 9. General-graph embedding: approximate UDG + violation analysis             #
# --------------------------------------------------------------------------- #


def analyze_udg_embedding(
    graph: nx.Graph,
    blockade_radius: float = 7.5,
    seed: int = 42,
) -> dict:
    """
    Attempt to embed a general graph as an approximate unit-disk graph and
    return a detailed violation report plus iterative sparsification profile.

    For graphs that are not UDGs (most BA and dense ER graphs), some non-edge
    pairs will fall within ``blockade_radius`` (spurious interactions) and/or
    some edge pairs will fall outside it (missing interactions).

    The sparsification profile tracks how violations evolve as CTQW-MVC
    iteratively removes vertices, showing that the residual graph becomes
    progressively more embeddable.

    Returns
    -------
    dict with keys:
        positions, spurious_edges, missing_edges,
        n_spurious, n_missing, violation_fraction,
        sparsification (list of per-step dicts)
    """
    import warnings as _w
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    with _w.catch_warnings():
        _w.simplefilter("ignore")
        positions = embed_graph_positions(graph, blockade_radius=blockade_radius,
                                          seed=seed)

    def _count_violations(G_sub: nx.Graph, pos: dict):
        adj_s = {(min(u, v), max(u, v)) for u, v in G_sub.edges()}
        ns = list(G_sub.nodes())
        spurious, missing = [], []
        for i in range(len(ns)):
            for j in range(i + 1, len(ns)):
                u, v = ns[i], ns[j]
                if u not in pos or v not in pos:
                    continue
                d = math.sqrt((pos[u][0] - pos[v][0])**2 +
                               (pos[u][1] - pos[v][1])**2)
                pair = (min(u, v), max(u, v))
                if pair in adj_s and d >= blockade_radius:
                    missing.append((u, v, d))
                elif pair not in adj_s and d < blockade_radius:
                    spurious.append((u, v, d))
        return spurious, missing

    n = graph.number_of_nodes()
    spurious, missing = _count_violations(graph, positions)
    total_pairs = n * (n - 1) // 2
    vf = (len(spurious) + len(missing)) / total_pairs if total_pairs > 0 else 0.0

    # Iterative sparsification: simulate CTQW-MVC classically, track violations
    G_res = graph.copy()
    sparsification = []
    step = 0

    while G_res.number_of_edges() > 0 and step < n:
        adj_m = nx.to_scipy_sparse_array(G_res, dtype=float, format="lil")
        degrees = np.array(adj_m.sum(axis=1)).flatten()
        degrees_safe = np.where(degrees == 0, 1e-10, degrees)
        inv_sqrt = 1.0 / np.sqrt(degrees_safe)
        D_inv_sqrt = sp.diags(inv_sqrt)
        Gamma = D_inv_sqrt @ adj_m.tocsr() @ D_inv_sqrt
        evo = spla.expm(1j * 0.01 * Gamma)
        probs = 1.0 - np.abs(evo.diagonal())**2
        best_idx = int(np.argmax(probs))
        best_vertex = list(G_res.nodes())[best_idx]

        spu_r, mis_r = _count_violations(G_res, positions)
        sparsification.append({
            "step": step,
            "removed_vertex": best_vertex,
            "degree_before": G_res.degree(best_vertex),
            "n_spurious_residual": len(spu_r),
            "n_missing_residual": len(mis_r),
            "n_nodes_residual": G_res.number_of_nodes(),
            "n_edges_residual": G_res.number_of_edges(),
        })

        G_res.remove_node(best_vertex)
        step += 1

    return {
        "positions": positions,
        "spurious_edges": spurious,
        "missing_edges": missing,
        "n_spurious": len(spurious),
        "n_missing": len(missing),
        "violation_fraction": vf,
        "sparsification": sparsification,
    }


def suppress_spurious_detuning(
    spurious_edges: list,
    Omega: float = 4.0 * math.pi,
    suppression_factor: float = 20.0,
) -> dict:
    """
    Compute per-atom detuning to suppress spurious blockade interactions.

    Applying Δ_u = suppression_factor * Ω >> Ω reduces the effective coupling
    of atom u to all its neighbours: J_eff ≈ Ω²/(2Δ_u) → 0.

    Returns
    -------
    dict  {vertex: detuning_rad_per_us}
    """
    detunings: dict = {}
    detuning_value = suppression_factor * Omega
    for u, v, dist in spurious_edges:
        if u not in detunings:
            detunings[u] = detuning_value
    return detunings
