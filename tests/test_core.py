"""
Unit tests for quantum-walk-mvc package.

Run with: pytest tests/
"""

import pytest
import numpy as np
import networkx as nx
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from quantum_walk_mvc import (
    quantum_walk_iterative_vertex_cover,
    quantum_walk_mvc_sparse,
    construct_hamiltonian,
    qubits_number,
    greedy_2_approx_vertex_cover,
    fast_vc_heuristic,
    simulated_annealing_vertex_cover,
    check_is_vertex_cover,
    generate_erdos_renyi_graphs,
    generate_barabasi_albert_graphs,
    generate_regular_graphs,
)


class TestQubitsNumber:
    """Tests for qubits_number function."""
    
    def test_zero_nodes(self):
        assert qubits_number(0) == 0
    
    def test_one_node(self):
        assert qubits_number(1) == 1
    
    def test_two_nodes(self):
        assert qubits_number(2) == 1
    
    def test_power_of_two(self):
        assert qubits_number(4) == 2
        assert qubits_number(8) == 3
        assert qubits_number(16) == 4
    
    def test_non_power_of_two(self):
        assert qubits_number(3) == 2
        assert qubits_number(5) == 3
        assert qubits_number(10) == 4


class TestHamiltonian:
    """Tests for construct_hamiltonian function."""
    
    def test_simple_graph(self):
        G = nx.Graph()
        G.add_edge(0, 1, weight=1.0)
        H = construct_hamiltonian(G)
        assert H is not None
        assert H.full().shape[0] == 2  # 2^1 = 2 dimensions
    
    def test_empty_graph(self):
        G = nx.Graph()
        H = construct_hamiltonian(G)
        assert H.full().shape == (1, 1)


class TestVertexCoverValidity:
    """Tests for check_is_vertex_cover function."""
    
    def test_valid_cover(self):
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2), (2, 3)])
        # {1, 2} covers all edges
        assert check_is_vertex_cover(G, {1, 2})
    
    def test_invalid_cover(self):
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2), (2, 3)])
        # {0} doesn't cover edge (1, 2) or (2, 3)
        assert not check_is_vertex_cover(G, {0})
    
    def test_full_cover(self):
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2)])
        # All nodes is always a valid cover
        assert check_is_vertex_cover(G, {0, 1, 2})


class TestQuantumWalkAlgorithm:
    """Tests for quantum walk vertex cover algorithms."""
    
    def test_simple_path(self):
        """Test on a simple path graph P_4."""
        G = nx.path_graph(4)
        for u, v in G.edges():
            G[u][v]['weight'] = 1.0
        
        vc = quantum_walk_iterative_vertex_cover(G, t_max=0.01)
        assert check_is_vertex_cover(G, set(vc))
    
    def test_cycle_graph(self):
        """Test on a cycle graph C_6."""
        G = nx.cycle_graph(6)
        for u, v in G.edges():
            G[u][v]['weight'] = 1.0
        
        vc = quantum_walk_iterative_vertex_cover(G, t_max=0.01)
        assert check_is_vertex_cover(G, set(vc))
    
    def test_complete_graph(self):
        """Test on complete graph K_5."""
        G = nx.complete_graph(5)
        for u, v in G.edges():
            G[u][v]['weight'] = 1.0
        
        vc = quantum_walk_iterative_vertex_cover(G, t_max=0.01)
        assert check_is_vertex_cover(G, set(vc))


class TestSparseQuantumWalk:
    """Tests for sparse matrix quantum walk implementation."""
    
    def test_simple_graph(self):
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2), (2, 0)])
        for u, v in G.edges():
            G[u][v]['weight'] = 1.0
        
        vc = quantum_walk_mvc_sparse(G, t_opt=0.01)
        assert check_is_vertex_cover(G, set(vc))


class TestClassicalHeuristics:
    """Tests for classical comparison algorithms."""
    
    def test_greedy_2_approx(self):
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])
        
        vc = greedy_2_approx_vertex_cover(G)
        assert check_is_vertex_cover(G, set(vc))
    
    def test_fast_vc(self):
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2), (2, 3)])
        
        vc = fast_vc_heuristic(G, max_iterations=100)
        assert check_is_vertex_cover(G, set(vc))
    
    def test_simulated_annealing(self):
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2), (2, 3)])
        
        vc = simulated_annealing_vertex_cover(G, max_iterations=100)
        assert check_is_vertex_cover(G, set(vc))


class TestGraphGenerators:
    """Tests for graph generation functions."""
    
    def test_erdos_renyi_generation(self):
        graphs, params = generate_erdos_renyi_graphs(
            num_nodes_range=[10],
            edge_prob_range=[0.5],
            num_graphs_per_setting=2
        )
        assert len(graphs) == 2
        assert all(G.number_of_nodes() == 10 for G in graphs)
    
    def test_barabasi_albert_generation(self):
        graphs, params = generate_barabasi_albert_graphs(
            num_nodes_range=[10],
            m_range=[2],
            num_graphs_per_setting=2
        )
        assert len(graphs) == 2
        assert all(G.number_of_nodes() == 10 for G in graphs)
    
    def test_regular_graph_generation(self):
        graphs, params = generate_regular_graphs(
            num_nodes_range=[10],
            num_graphs_per_setting=2
        )
        assert len(graphs) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
