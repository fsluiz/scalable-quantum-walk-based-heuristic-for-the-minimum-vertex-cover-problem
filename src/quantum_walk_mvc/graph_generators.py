"""
Graph generators for benchmarking quantum walk vertex cover algorithms.

This module provides functions to generate various types of graphs used
in the experimental evaluation:
- Barabási-Albert (scale-free) graphs
- Erdős-Rényi (random) graphs
- Random regular graphs
"""

import random
import networkx as nx
from networkx.algorithms.graph_hashing import weisfeiler_lehman_graph_hash


def generate_barabasi_albert_graphs(
    num_nodes_range: list,
    m_range: list,
    num_graphs_per_setting: int = 5
) -> tuple:
    """
    Generate Barabási-Albert (scale-free) graphs for testing.
    
    Barabási-Albert graphs are generated using preferential attachment,
    resulting in scale-free degree distributions commonly found in
    real-world networks.
    
    Args:
        num_nodes_range: List of node counts to generate graphs for.
        m_range: List of m values (edges added per new node in BA model).
        num_graphs_per_setting: Number of graphs to generate per (n, m) pair.
        
    Returns:
        Tuple of (graphs_list, params_list) where params_list contains
        dictionaries with 'N', 'm', and 'seed' for each graph.
        
    Example:
        >>> graphs, params = generate_barabasi_albert_graphs([10, 20], [2, 3], 5)
        >>> len(graphs)
        20  # 2 sizes * 2 m-values * 5 graphs each
    """
    graphs = []
    graph_params = []
    
    print("Generating Barabási-Albert test graphs...")
    
    for n in num_nodes_range:
        for m in m_range:
            for i in range(num_graphs_per_setting):
                if n == 0:
                    G = nx.Graph()
                elif n == 1:
                    G = nx.Graph()
                    G.add_node(0)
                elif m >= n:
                    # m must be < n for BA model
                    continue
                else:
                    G = nx.barabasi_albert_graph(n, m, seed=i)

                # Assign unit weights to edges
                for u, v in G.edges():
                    G[u][v]['weight'] = 1.0
                    
                graphs.append(G)
                graph_params.append({'N': n, 'm': m, 'seed': i})
                
    print(f"Total graphs generated: {len(graphs)}")
    return graphs, graph_params


def generate_erdos_renyi_graphs(
    num_nodes_range: list,
    edge_prob_range: list,
    num_graphs_per_setting: int = 5,
    ensure_connected: bool = True
) -> tuple:
    """
    Generate Erdős-Rényi (random) graphs for testing.
    
    Each edge is included independently with probability p. If connectivity
    is required and the generated graph is disconnected, the function
    attempts regeneration or falls back to Watts-Strogatz model.
    
    Args:
        num_nodes_range: List of node counts to generate graphs for.
        edge_prob_range: List of edge probabilities p ∈ (0, 1).
        num_graphs_per_setting: Number of graphs to generate per (n, p) pair.
        ensure_connected: If True, only return connected graphs.
        
    Returns:
        Tuple of (graphs_list, params_list) where params_list contains
        dictionaries with 'N', 'p', and 'seed' for each graph.
        
    Example:
        >>> graphs, params = generate_erdos_renyi_graphs([10, 20], [0.3, 0.5], 5)
    """
    graphs = []
    graph_params = []
    
    print("Generating Erdős-Rényi test graphs...")
    
    for n in num_nodes_range:
        for p in edge_prob_range:
            for i in range(num_graphs_per_setting):
                if n == 0:
                    G = nx.Graph()
                elif n == 1:
                    G = nx.Graph()
                    G.add_node(0)
                else:
                    G = nx.erdos_renyi_graph(n, p, seed=i)
                    
                    if ensure_connected:
                        # Retry if disconnected
                        attempts = 0
                        while not nx.is_connected(G) and attempts < 5:
                            G = nx.erdos_renyi_graph(n, p, seed=i + attempts)
                            attempts += 1
                        
                        # Fallback to Watts-Strogatz if still disconnected
                        if not nx.is_connected(G) and n > 1:
                            k_val = max(2, min(n - 1, int(n * p * n / 2)))
                            if k_val < 2 and n > 1:
                                k_val = 2
                            
                            try:
                                G_ws = nx.connected_watts_strogatz_graph(n, k_val, 0.1, seed=i)
                                if nx.is_connected(G_ws):
                                    G = G_ws
                                else:
                                    continue
                            except nx.NetworkXError:
                                continue

                # Assign unit weights to edges
                for u, v in G.edges():
                    G[u][v]['weight'] = 1.0
                    
                graphs.append(G)
                graph_params.append({'N': n, 'p': p, 'seed': i})
                
    print(f"Total graphs generated: {len(graphs)}")
    return graphs, graph_params


def generate_regular_graphs(
    num_nodes_range: list,
    num_graphs_per_setting: int = 5,
    k_values: list = None
) -> tuple:
    """
    Generate random regular graphs (non-complete, connected) for testing.
    
    Regular graphs have the same degree k for all vertices. This function
    generates random k-regular graphs and ensures uniqueness using
    Weisfeiler-Lehman graph hashing.
    
    Args:
        num_nodes_range: List of node counts to generate graphs for.
        num_graphs_per_setting: Number of unique graphs to generate per (n, k).
        k_values: Optional list of degree values. If None, uses [2, n/4, n/2].
        
    Returns:
        Tuple of (graphs_list, params_list) where params_list contains
        dictionaries with 'n' and 'k' for each graph.
        
    Note:
        A k-regular graph on n vertices exists only if n*k is even.
        
    Example:
        >>> graphs, params = generate_regular_graphs([10, 20], num_graphs_per_setting=5)
    """
    graphs = []
    graph_params = []
    seen_hashes = set()

    print("Generating non-complete regular graphs (ensuring uniqueness)...")

    for n in num_nodes_range:
        if n < 3:
            print(f"Warning: Skipping N={n}, non-complete regular graphs are trivial or impossible.")
            continue

        # Determine k values to test
        if k_values is None:
            test_ks = [2, int(0.25 * n), int(0.5 * n)]
        else:
            test_ks = k_values
            
        # Filter valid k values: 2 <= k < n-1 and n*k must be even
        possible_ks = sorted(list(set([
            k for k in test_ks 
            if 2 <= k < n - 1 and (n * k) % 2 == 0
        ])))

        if not possible_ks:
            print(f"Warning: No valid k for N={n}. Skipping.")
            continue

        for k in possible_ks:
            print(f"  Attempting to generate {num_graphs_per_setting} unique graphs for N={n}, k={k}...")
            generated_count = 0
            attempts = 0
            max_attempts = 300

            base_seed = abs(hash((n, k))) % 100000
            while generated_count < num_graphs_per_setting and attempts < max_attempts:
                try:
                    G = nx.random_regular_graph(k, n, seed=base_seed + attempts)
                    
                    if nx.is_connected(G):
                        # Check uniqueness using graph hash
                        h = weisfeiler_lehman_graph_hash(G)
                        if h not in seen_hashes:
                            seen_hashes.add(h)
                            G.graph['n'] = n
                            G.graph['k'] = k
                            
                            # Assign unit weights
                            for u, v in G.edges():
                                G[u][v]['weight'] = 1.0

                            graphs.append(G)
                            graph_params.append({'n': n, 'k': k})
                            generated_count += 1
                            
                except nx.NetworkXError:
                    pass

                attempts += 1

            if generated_count < num_graphs_per_setting:
                print(f"  Warning: Only {generated_count}/{num_graphs_per_setting} unique graphs found for N={n}, k={k}.")

    print(f"\nTotal graphs generated: {len(graphs)} (all distinct)")
    return graphs, graph_params


def get_graph_properties(graph: nx.Graph) -> dict:
    """
    Compute various graph properties for analysis.
    
    Args:
        graph: NetworkX graph object.
        
    Returns:
        Dictionary containing graph properties:
        - num_nodes: Number of vertices
        - num_edges: Number of edges
        - is_connected: Whether the graph is connected
        - is_planar: Whether the graph is planar
        - is_eulerian: Whether the graph has an Eulerian circuit
        - is_bipartite: Whether the graph is bipartite
        - is_tree: Whether the graph is a tree
        - is_complete: Whether the graph is complete
        - density: Graph density
    """
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    
    properties = {
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'is_connected': nx.is_connected(graph) if num_nodes > 0 else True,
        'is_eulerian': nx.is_eulerian(graph),
        'is_bipartite': nx.is_bipartite(graph),
        'is_tree': nx.is_tree(graph),
        'density': nx.density(graph)
    }
    
    # Planarity check can be expensive
    try:
        properties['is_planar'] = nx.is_planar(graph)
    except nx.NetworkXException:
        properties['is_planar'] = False
    
    # Check if complete
    if num_nodes <= 1:
        properties['is_complete'] = True
    else:
        expected_edges = (num_nodes * (num_nodes - 1)) // 2
        properties['is_complete'] = (num_edges == expected_edges)
    
    return properties
