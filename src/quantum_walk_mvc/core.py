"""
Core quantum walk algorithms for Minimum Vertex Cover (MVC).

This module implements continuous-time quantum walk (CTQW) based heuristics
for solving the Minimum Vertex Cover problem on graphs.

References:
    - [Your paper reference here]
"""

import numpy as np
import networkx as nx
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.linalg import expm

# Try to import QuTiP, fallback to mock implementation if not available
try:
    from qutip import Qobj, basis, tensor
    QUTIP_AVAILABLE = True
except ImportError:
    QUTIP_AVAILABLE = False
    
    class MockQobj:
        """
        Mock implementation of QuTiP's Qobj for environments where QuTiP is not available.
        
        Warning:
            This is a simplified implementation for compatibility only.
            For production use, install QuTiP for full functionality.
        """
        def __init__(self, data, dims=None):
            if np.isscalar(data):
                self.data = np.array([[data]])
            elif data.ndim == 1:
                self.data = data.reshape(-1, 1)
            else:
                self.data = data
            self.dims = dims
            
        def expm(self):
            """Compute matrix exponential."""
            return MockQobj(expm(self.data), self.dims)
        
        def dag(self):
            """Compute conjugate transpose."""
            return MockQobj(self.data.conj().T, self.dims)
        
        def full(self):
            """Return the underlying numpy array."""
            return self.data
        
        def __mul__(self, other):
            if isinstance(other, (int, float, complex)):
                return MockQobj(self.data * other, self.dims)
            if isinstance(other, MockQobj):
                result_data = self.data @ other.data
                if np.isscalar(result_data):
                    result_data = np.array([[result_data]])
                return MockQobj(result_data, self.dims)
            raise TypeError("Invalid operand for multiplication.")
        
        def __rmul__(self, other):
            return self.__mul__(other)
        
        def __abs__(self):
            return np.abs(self.data)

    def mock_basis(n, k):
        """Create a basis vector."""
        vec = np.zeros((n, 1))
        vec[k] = 1
        return MockQobj(vec)

    def mock_tensor(*args):
        """Compute tensor product of multiple objects."""
        res = args[0].data
        for i in range(1, len(args)):
            res = np.kron(res, args[i].data)
        return MockQobj(res)

    Qobj = MockQobj
    basis = mock_basis
    tensor = mock_tensor


def qubits_number(num_nodes: int) -> int:
    """
    Calculate the number of qubits required to represent graph nodes.
    
    Args:
        num_nodes: Number of nodes in the graph.
        
    Returns:
        Number of qubits needed (ceiling of log2(num_nodes)).
    """
    if num_nodes <= 0:
        return 0
    if num_nodes == 1:
        return 1
    return int(np.ceil(np.log2(num_nodes)))


def construct_hamiltonian(graph: nx.Graph) -> Qobj:
    """
    Construct the Hamiltonian based on the normalized graph Laplacian.
    
    The Hamiltonian is defined as H = I - A_norm, where A_norm is the
    symmetric normalized adjacency matrix: A_norm = D^(-1/2) @ A @ D^(-1/2).
    
    Args:
        graph: NetworkX graph object.
        
    Returns:
        QuTiP Qobj representing the Hamiltonian matrix, padded to 2^n dimensions.
    """
    num_nodes = len(graph.nodes)
    if num_nodes == 0:
        return Qobj(np.array([[0]]), dims=[[1], [1]])

    # Get adjacency matrix
    adj_matrix_sparse = nx.adjacency_matrix(graph, weight='weight')
    adj_matrix_dense = adj_matrix_sparse.todense().astype(complex)

    # Compute degree values
    degree_values = np.array(adj_matrix_sparse.sum(axis=1)).flatten()

    # Create D^(-1/2) diagonal matrix
    D_inv_sqrt_diag = np.where(degree_values != 0, 1.0 / np.sqrt(degree_values), 0.0)
    D_inv_sqrt = np.diag(D_inv_sqrt_diag)

    # Compute normalized adjacency: A_norm = D^(-1/2) @ A @ D^(-1/2)
    A_norm = D_inv_sqrt @ adj_matrix_dense @ D_inv_sqrt

    # Remove self-loops and construct Laplacian: H = I - A_norm
    np.fill_diagonal(A_norm, 0)
    H_data = np.eye(num_nodes, dtype=complex) - A_norm

    # Pad to 2^n dimensions for qubit representation
    qubits_n = qubits_number(num_nodes)
    dim = 2 ** qubits_n
    
    padded_H_data = np.zeros((dim, dim), dtype=complex)
    padded_H_data[:num_nodes, :num_nodes] = H_data
    
    dims = [[2] * qubits_n, [2] * qubits_n]
    return Qobj(padded_H_data, dims=dims)


def qubit_basis_states(num_qubits: int) -> list:
    """
    Create computational basis states for a given number of qubits.
    
    Args:
        num_qubits: Number of qubits.
        
    Returns:
        List of basis state vectors as Qobj objects.
    """
    dim = 2 ** num_qubits
    return [basis(dim, i) for i in range(dim)]


def quantum_walk_iterative_vertex_cover(graph_original: nx.Graph, t_max: float) -> list:
    """
    Compute vertex cover using iterative continuous-time quantum walk.
    
    This algorithm iteratively selects vertices based on transition probabilities
    computed from the quantum walk evolution operator. At each step, the vertex
    with the highest transition probability sum is added to the cover.
    
    Args:
        graph_original: Input NetworkX graph.
        t_max: Evolution time parameter for the quantum walk.
        
    Returns:
        List of vertices forming the vertex cover.
    """
    G = graph_original.copy()
    original_nodes_list = sorted(list(graph_original.nodes()))
    node_to_idx = {node: i for i, node in enumerate(original_nodes_list)}
    
    num_original_nodes = len(original_nodes_list)
    
    # Initialize Hamiltonian
    H_full_qobj_initial = construct_hamiltonian(graph_original)
    current_H_data = H_full_qobj_initial.full()
    
    selected_vertices = set()
    active_nodes = set(graph_original.nodes())

    # Pre-calculate basis matrix
    num_qubits_full = qubits_number(num_original_nodes)
    basis_vectors_full = qubit_basis_states(num_qubits_full)
    basis_matrix_full = np.column_stack([bv.full() for bv in basis_vectors_full])

    while G.number_of_edges() > 0 and len(active_nodes) > 0:
        # Create Qobj from current Hamiltonian
        current_H_qobj = Qobj(current_H_data, dims=H_full_qobj_initial.dims)
        
        # Compute evolution operator: U(t) = exp(-i*H*t)
        evolution_operator = (-1j * current_H_qobj * t_max).expm()
        
        # Calculate transition probabilities
        transition_probs = np.abs(
            basis_matrix_full.T.conj() @ (evolution_operator.full() @ basis_matrix_full)
        )**2

        # Compute vertex selection probabilities
        active_indices = np.array([node_to_idx[node] for node in active_nodes])
        
        temp_probs = transition_probs.copy()
        np.fill_diagonal(temp_probs, 0)  # Exclude self-loops
        
        # Extract submatrix for active nodes
        sub_matrix = temp_probs[np.ix_(active_indices, active_indices)]
        
        # Sum probabilities for each active node
        current_vertices_prob = {}
        for i, u_original in enumerate(active_nodes):
            current_vertices_prob[u_original] = np.sum(sub_matrix[i, :])
            
        if not current_vertices_prob:
            break

        # Select vertex with maximum probability
        max_vertex_original_id = max(current_vertices_prob, key=current_vertices_prob.get)
        selected_vertices.add(max_vertex_original_id)
        
        # Update graph and Hamiltonian
        if max_vertex_original_id in G:
            G.remove_node(max_vertex_original_id)
        active_nodes.remove(max_vertex_original_id)
        
        # Zero out corresponding row/column in Hamiltonian
        max_vertex_idx = node_to_idx[max_vertex_original_id]
        current_H_data[:, max_vertex_idx] = 0
        current_H_data[max_vertex_idx, :] = 0

    return list(selected_vertices)


def quantum_walk_mvc_sparse(graph: nx.Graph, t_opt: float) -> list:
    """
    Compute vertex cover using sparse matrix quantum walk (TSA variant).
    
    This implementation uses sparse matrix operations for improved performance
    on large graphs. It computes the evolution operator using scipy's sparse
    matrix exponential.
    
    Args:
        graph: Input NetworkX graph.
        t_opt: Optimal evolution time parameter.
        
    Returns:
        List of vertices forming the vertex cover.
    """
    # Build index-to-node mapping to return original NetworkX node IDs
    node_list = list(graph.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(node_list)}

    # Use sparse matrix representation
    adj_matrix_current = nx.to_scipy_sparse_array(graph, nodelist=node_list, dtype=float, format='lil')

    vertex_cover = []

    while adj_matrix_current.sum() > 0:
        # Compute degrees
        degrees = np.array(adj_matrix_current.sum(axis=1)).flatten()

        # Build normalization matrix
        degrees_safe = degrees.copy()
        degrees_safe[degrees_safe == 0] = 1e-10
        inv_sqrt_degrees = 1.0 / np.sqrt(degrees_safe)

        D_inv_sqrt = sp.diags(inv_sqrt_degrees)

        # Construct normalized adjacency (Gamma matrix)
        adj_matrix_current_csr = adj_matrix_current.tocsr()
        Gamma = D_inv_sqrt @ adj_matrix_current_csr @ D_inv_sqrt

        # Compute evolution operator using sparse matrix exponential
        evolution_operator = spla.expm(1j * t_opt * Gamma)

        # Extract probabilities from diagonal
        prob_m_to_m = np.abs(evolution_operator.diagonal())**2
        probabilities = 1 - prob_m_to_m

        # Select best vertex (matrix index → original node ID)
        best_idx = np.argmax(probabilities)
        best_vertex = node_list[best_idx]

        # Add to cover and zero out connections
        vertex_cover.append(best_vertex)
        adj_matrix_current[best_idx, :] = 0
        adj_matrix_current[:, best_idx] = 0

    return vertex_cover
