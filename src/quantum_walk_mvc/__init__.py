"""
Quantum Walk Minimum Vertex Cover (quantum-walk-mvc)

A Python package for solving the Minimum Vertex Cover problem using
continuous-time quantum walk algorithms.

This package implements:
- Quantum walk based heuristics for MVC
- Classical comparison algorithms (exact, 2-approx, FastVC, SA)
- Graph generators for benchmarking (BA, ER, regular graphs)
- Experiment runner with metrics collection

Example:
    >>> from quantum_walk_mvc import quantum_walk_iterative_vertex_cover
    >>> from quantum_walk_mvc.graph_generators import generate_erdos_renyi_graphs
    >>> 
    >>> graphs, params = generate_erdos_renyi_graphs([10], [0.5], 1)
    >>> vertex_cover = quantum_walk_iterative_vertex_cover(graphs[0], t_max=0.01)
    >>> print(f"Vertex cover size: {len(vertex_cover)}")

For more information, see the README.md file.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .core import (
    quantum_walk_iterative_vertex_cover,
    quantum_walk_mvc_sparse,
    construct_hamiltonian,
    qubits_number,
    QUTIP_AVAILABLE
)

from .heuristics import (
    get_exact_vertex_cover_sage,
    greedy_2_approx_vertex_cover,
    degree_greedy_vertex_cover,
    spectral_greedy_vertex_cover,
    fast_vc_heuristic,
    simulated_annealing_vertex_cover,
    check_is_vertex_cover,
    SAGE_AVAILABLE
)

from .graph_generators import (
    generate_barabasi_albert_graphs,
    generate_erdos_renyi_graphs,
    generate_regular_graphs,
    get_graph_properties
)

from .utils import (
    run_experiments,
    summarize_results
)

from .bloqade_mvc import (
    quantum_walk_mvc_bloqade,
    embed_graph_positions,
    embed_unit_disk_graph,
    get_rydberg_interaction_matrix,
    compute_transition_probs_bloqade,
    BLOQADE_AVAILABLE,
)

__all__ = [
    # Core quantum algorithms
    'quantum_walk_iterative_vertex_cover',
    'quantum_walk_mvc_sparse',
    'construct_hamiltonian',
    'qubits_number',
    'QUTIP_AVAILABLE',
    
    # Classical heuristics
    'get_exact_vertex_cover_sage',
    'greedy_2_approx_vertex_cover',
    'degree_greedy_vertex_cover',
    'spectral_greedy_vertex_cover',
    'fast_vc_heuristic',
    'simulated_annealing_vertex_cover',
    'check_is_vertex_cover',
    'SAGE_AVAILABLE',
    
    # Graph generators
    'generate_barabasi_albert_graphs',
    'generate_erdos_renyi_graphs',
    'generate_regular_graphs',
    'get_graph_properties',
    
    # Utilities
    'run_experiments',
    'summarize_results',
]
