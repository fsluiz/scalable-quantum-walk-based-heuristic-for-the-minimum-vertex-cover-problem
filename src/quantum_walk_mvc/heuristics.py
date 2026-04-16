"""
Classical heuristic algorithms for Minimum Vertex Cover (MVC).

This module provides classical algorithms for comparison with quantum walk approaches:
- Exact solver using SageMath (ILP-based)
- Greedy 2-approximation algorithm
- FastVC local search heuristic
- Simulated Annealing metaheuristic
"""

import random
import numpy as np
import networkx as nx

# Try to import SageMath Graph for exact solver
try:
    from sage.graphs.graph import Graph as SageGraph
    SAGE_AVAILABLE = True
except ImportError:
    SAGE_AVAILABLE = False


def check_is_vertex_cover(graph: nx.Graph, vc_set: set) -> bool:
    """
    Verify if a given set of vertices forms a valid vertex cover.
    
    A vertex cover is valid if every edge in the graph has at least
    one endpoint in the cover set.
    
    Args:
        graph: NetworkX graph object.
        vc_set: Set of vertices to check.
        
    Returns:
        True if vc_set is a valid vertex cover, False otherwise.
    """
    for u, v in graph.edges():
        if u not in vc_set and v not in vc_set:
            return False
    return True


def get_exact_vertex_cover_sage(nx_graph: nx.Graph) -> tuple:
    """
    Compute the exact Minimum Vertex Cover using SageMath's ILP solver.
    
    This function converts a NetworkX graph to a SageMath graph and uses
    SageMath's built-in vertex_cover() method which employs integer linear
    programming for exact solution.
    
    Args:
        nx_graph: NetworkX graph object.
        
    Returns:
        Tuple of (vertex_cover_list, cover_size).
        
    Raises:
        ImportError: If SageMath is not available.
        
    Note:
        This solver has exponential worst-case complexity. For large graphs
        (n > 30), consider using heuristic methods instead.
    """
    if not SAGE_AVAILABLE:
        raise ImportError(
            "SageMath is required for exact vertex cover computation. "
            "Please install SageMath or use heuristic methods."
        )
    
    # Convert NetworkX graph to SageMath graph
    sage_graph = SageGraph()
    for u, v in nx_graph.edges():
        if u not in sage_graph.vertices():
            sage_graph.add_vertex(u)
        if v not in sage_graph.vertices():
            sage_graph.add_vertex(v)
        sage_graph.add_edge(u, v)
    
    # Compute exact vertex cover using SageMath's ILP solver
    exact_vc_set = sage_graph.vertex_cover()
    exact_vc_size = len(exact_vc_set)
    
    return list(exact_vc_set), exact_vc_size


def greedy_2_approx_vertex_cover(graph: nx.Graph) -> list:
    """
    Compute a vertex cover using the greedy 2-approximation algorithm.
    
    This algorithm repeatedly selects an arbitrary uncovered edge and adds
    both endpoints to the cover. It guarantees a cover of size at most
    2 * OPT, where OPT is the optimal cover size.
    
    Time Complexity: O(E) where E is the number of edges.
    
    Args:
        graph: NetworkX graph object.
        
    Returns:
        List of vertices forming the vertex cover.
        
    Reference:
        Vazirani, V. V. (2001). Approximation Algorithms. Springer.
    """
    G_copy = graph.copy()
    cover = set()
    
    while G_copy.number_of_edges() > 0:
        # Pick an arbitrary edge (u, v)
        u, v = next(iter(G_copy.edges()))
        
        # Add both endpoints to the cover
        cover.add(u)
        cover.add(v)
        
        # Remove all edges incident to u and v
        G_copy.remove_nodes_from([u, v])
            
    return list(cover)


def degree_greedy_vertex_cover(graph: nx.Graph) -> list:
    """
    Compute a vertex cover using the maximum-degree greedy heuristic.

    At each step, the vertex with the highest current degree is selected and
    added to the cover; it and its incident edges are then removed from the
    graph.  This is repeated until no edges remain.

    The degree-greedy rule is a natural local baseline: it captures the most
    'connected' vertex at each step purely from topological degree, without
    exploiting global spectral information.

    Time Complexity: O(V * E) in the worst case.

    Args:
        graph: NetworkX graph object.

    Returns:
        List of vertices forming the vertex cover.
    """
    G = graph.copy()
    cover = []
    while G.number_of_edges() > 0:
        max_vertex = max(G.nodes(), key=lambda v: G.degree(v))
        cover.append(max_vertex)
        G.remove_node(max_vertex)
    return cover


def fast_vc_heuristic(graph: nx.Graph, max_iterations: int = 1000) -> list:
    """
    Compute a vertex cover using a FastVC-inspired local search heuristic.
    
    This algorithm starts with a greedy solution and iteratively tries to
    improve it by removing vertices that are not essential for coverage,
    and adding vertices to cover any remaining uncovered edges.
    
    Args:
        graph: NetworkX graph object.
        max_iterations: Maximum number of improvement iterations.
        
    Returns:
        List of vertices forming the vertex cover.
        
    Reference:
        Cai, S. (2015). Balance between Complexity and Quality: Local Search
        for Minimum Vertex Cover in Massive Graphs. IJCAI.
    """
    V = list(graph.nodes())
    E = list(graph.edges())
    
    # Initialize with greedy 2-approximation solution
    current_cover = set(greedy_2_approx_vertex_cover(graph))
    best_cover = set(current_cover)
    
    for _ in range(max_iterations):
        improved = False
        
        # Try to remove a vertex while maintaining valid cover
        for v in list(current_cover):
            temp_cover = current_cover.copy()
            temp_cover.remove(v)
            
            # Check if temp_cover is still valid
            is_valid = True
            for u1, v1 in E:
                if u1 not in temp_cover and v1 not in temp_cover:
                    is_valid = False
                    break
            
            if is_valid:
                current_cover = temp_cover
                improved = True
                if len(current_cover) < len(best_cover):
                    best_cover = current_cover.copy()
                break
        
        if not improved:
            # Find uncovered edges and add a vertex to cover them
            uncovered_edges = []
            for u, v in E:
                if u not in current_cover and v not in current_cover:
                    uncovered_edges.append((u, v))
            
            if len(uncovered_edges) > 0:
                # Add the endpoint with higher degree
                u, v = random.choice(uncovered_edges)
                if graph.degree(u) >= graph.degree(v):
                    current_cover.add(u)
                else:
                    current_cover.add(v)
            else:
                break
    
    return list(best_cover)


def spectral_greedy_vertex_cover(graph: nx.Graph) -> list:
    """
    Quantum-inspired spectral greedy heuristic for Minimum Vertex Cover.

    Derived from the short-time expansion of the CTQW transition probability:

        P(m -> out) ≈ t^2 [Γ^2]_mm + O(t^4)

    where [Γ^2]_mm = (1/d_m) Σ_{j ∈ N(m)} 1/d_j.

    Since t^2 is a common factor, the argmax over vertices is equivalent to
    selecting the vertex with the highest spectral score s(m) at each step.
    This O(|E|)-per-iteration algorithm reproduces the CTQW (t=0.01) cover
    on ~98.6% of benchmark instances.

    Time Complexity: O(V * E) in the worst case; O(V * k_max) for sparse graphs.

    Args:
        graph: NetworkX graph object.

    Returns:
        List of vertices forming the vertex cover.

    Reference:
        Derived from CTQW short-time expansion; see Algorithm 2 in the paper.
    """
    G = graph.copy()
    cover = []
    while G.number_of_edges() > 0:
        scores = {}
        for m in list(G.nodes()):
            d_m = G.degree(m)
            if d_m == 0:
                continue
            scores[m] = sum(1.0 / G.degree(j) for j in G.neighbors(m)) / d_m
        if not scores:
            break
        m_star = max(scores, key=scores.get)
        cover.append(m_star)
        G.remove_node(m_star)
    return cover


def simulated_annealing_vertex_cover(
    graph: nx.Graph,
    initial_temperature: float = 100.0,
    cooling_rate: float = 0.99,
    max_iterations: int = 5000
) -> list:
    """
    Compute a vertex cover using Simulated Annealing metaheuristic.
    
    This algorithm uses a probabilistic approach to escape local minima
    by occasionally accepting worse solutions. The acceptance probability
    decreases over time according to a cooling schedule.
    
    Args:
        graph: NetworkX graph object.
        initial_temperature: Starting temperature for annealing.
        cooling_rate: Multiplicative cooling factor (0 < rate < 1).
        max_iterations: Maximum number of iterations.
        
    Returns:
        List of vertices forming the vertex cover.
        
    Reference:
        Kirkpatrick, S., Gelatt, C. D., & Vecchi, M. P. (1983).
        Optimization by Simulated Annealing. Science, 220(4598), 671-680.
    """
    V = list(graph.nodes())
    num_nodes = len(V)

    def cost_function(vc):
        """Cost is the size of the vertex cover."""
        return len(vc)

    # Initialize with greedy solution
    current_cover = set(greedy_2_approx_vertex_cover(graph))
    
    if not check_is_vertex_cover(graph, current_cover):
        # Fallback: start with all nodes and prune
        current_cover = set(V)
        for v in list(current_cover):
            temp_cover = current_cover.copy()
            temp_cover.remove(v)
            if check_is_vertex_cover(graph, temp_cover):
                current_cover = temp_cover

    best_cover = current_cover.copy()
    current_cost = cost_function(current_cover)
    best_cost = current_cost
    
    temperature = initial_temperature

    for i in range(max_iterations):
        if temperature <= 0.01:
            break

        # Generate neighbor solution
        neighbor_cover = current_cover.copy()
        action = random.choice(['add', 'remove'])
        
        if action == 'add' and len(neighbor_cover) < num_nodes:
            candidates = [v for v in V if v not in neighbor_cover]
            if candidates:
                node_to_add = random.choice(candidates)
                neighbor_cover.add(node_to_add)
        elif action == 'remove' and len(neighbor_cover) > 0:
            node_to_remove = random.choice(list(neighbor_cover))
            neighbor_cover.remove(node_to_remove)
        
        # Repair if not valid
        if not check_is_vertex_cover(graph, neighbor_cover):
            for u, v in graph.edges():
                if u not in neighbor_cover and v not in neighbor_cover:
                    if graph.degree(u) >= graph.degree(v):
                        neighbor_cover.add(u)
                    else:
                        neighbor_cover.add(v)
        
        neighbor_cost = cost_function(neighbor_cover)

        # Acceptance criterion
        if neighbor_cost < current_cost:
            current_cover = neighbor_cover
            current_cost = neighbor_cost
            if current_cost < best_cost:
                best_cover = current_cover.copy()
                best_cost = current_cost
        else:
            delta_cost = neighbor_cost - current_cost
            acceptance_probability = np.exp(-delta_cost / temperature)
            if random.random() < acceptance_probability:
                current_cover = neighbor_cover
                current_cost = neighbor_cost
        
        temperature *= cooling_rate
        
    # Final pruning pass
    final_optimized_cover = best_cover.copy()
    for v in list(final_optimized_cover):
        temp_cover = final_optimized_cover.copy()
        temp_cover.remove(v)
        if check_is_vertex_cover(graph, temp_cover):
            final_optimized_cover = temp_cover

    return list(final_optimized_cover)
