"""
Utility functions for running experiments and collecting metrics.

This module provides the experiment runner that executes all algorithms
on test graphs and collects performance metrics.
"""

import os
import time
import pandas as pd
import networkx as nx

from .core import quantum_walk_iterative_vertex_cover, quantum_walk_mvc_sparse
from .heuristics import (
    get_exact_vertex_cover_sage,
    greedy_2_approx_vertex_cover,
    fast_vc_heuristic,
    simulated_annealing_vertex_cover,
    check_is_vertex_cover,
    SAGE_AVAILABLE
)
from .graph_generators import get_graph_properties


def run_experiments(
    test_graphs: list,
    t_max_quantum: float = 0.01,
    results_filename: str = 'mvc_results.csv',
    run_exact_solver: bool = True,
    save_interval: int = 5
) -> pd.DataFrame:
    """
    Run all MVC algorithms on test graphs and collect metrics.
    
    This function executes the quantum walk algorithms and classical heuristics
    on each test graph, recording execution times and solution sizes. Results
    are saved incrementally to allow resuming interrupted experiments.
    
    Args:
        test_graphs: List of NetworkX graphs to test.
        t_max_quantum: Evolution time parameter for quantum walk algorithms.
        results_filename: Path to save results CSV file.
        run_exact_solver: Whether to run the exact SageMath solver.
        save_interval: Save results every N graphs processed.
        
    Returns:
        DataFrame containing all experimental results.
        
    Note:
        The exact solver has exponential complexity. For graphs with n > 30,
        set run_exact_solver=False to skip it.
    """
    # Load existing results or initialize new DataFrame
    if os.path.exists(results_filename):
        print(f"Loading existing results from '{results_filename}'...")
        all_results_df = pd.read_csv(results_filename)
        
        # Convert boolean columns back to bool type
        bool_cols = ['is_connected', 'is_planar', 'is_eulerian', 
                     'is_bipartite', 'is_tree', 'is_complete']
        for col in bool_cols:
            if col in all_results_df.columns:
                all_results_df[col] = all_results_df[col].astype(bool)
        
        # Determine which graphs have been processed
        processed_signatures = set(
            all_results_df['num_nodes'].astype(str) + "_" + 
            all_results_df['num_edges'].astype(str) + "_" + 
            all_results_df['graph_iteration_index'].astype(str)
        )
        print(f"Found {len(all_results_df)} previously saved results.")
    else:
        print(f"No existing results file found. Starting fresh.")
        all_results_df = pd.DataFrame()
        processed_signatures = set()

    results_to_append = []
    total_graphs = len(test_graphs)
    print(f"\nStarting/Continuing tests on {total_graphs} graphs...")
    
    for i, G_nx in enumerate(test_graphs):
        graph_signature = f"{G_nx.number_of_nodes()}_{G_nx.number_of_edges()}_{i}"
        
        if graph_signature in processed_signatures:
            print(f"Skipping graph {i + 1}/{total_graphs} - already processed.")
            continue

        num_nodes = G_nx.number_of_nodes()
        num_edges = G_nx.number_of_edges()
        print(f"Processing graph {i + 1}/{total_graphs} (N={num_nodes}, M={num_edges})...")

        # Get graph properties
        properties = get_graph_properties(G_nx)
        
        # Store graph data for reproducibility
        graph_data = [(u, v, d.get('weight', 1.0)) for u, v, d in G_nx.edges(data=True)]
        
        # Initialize result dictionary
        current_result = {
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'graph_data': str(graph_data),
            'graph_iteration_index': i,
            **properties
        }

        # --- Run Quantum Walk Algorithm (Full Hamiltonian) ---
        try:
            start_time = time.perf_counter()
            quantum_vc = quantum_walk_iterative_vertex_cover(G_nx, t_max_quantum)
            end_time = time.perf_counter()
            
            current_result['quantum_vc'] = str(quantum_vc)
            current_result['quantum_vc_size'] = len(quantum_vc)
            current_result['quantum_time'] = end_time - start_time
            current_result['quantum_is_valid'] = check_is_vertex_cover(G_nx, set(quantum_vc))
        except Exception as e:
            print(f"  Error in quantum algorithm: {e}")
            current_result['quantum_vc_size'] = -1
            current_result['quantum_time'] = float('inf')
            current_result['quantum_is_valid'] = False

        # --- Run Quantum Walk Algorithm (Sparse/TSA variant) ---
        try:
            start_time = time.perf_counter()
            quantum_tsa_vc = quantum_walk_mvc_sparse(G_nx, t_max_quantum)
            end_time = time.perf_counter()
            
            current_result['quantum_tsa_vc'] = str(quantum_tsa_vc)
            current_result['quantum_tsa_vc_size'] = len(quantum_tsa_vc)
            current_result['quantum_tsa_time'] = end_time - start_time
            current_result['quantum_tsa_is_valid'] = check_is_vertex_cover(G_nx, set(quantum_tsa_vc))
        except Exception as e:
            print(f"  Error in quantum TSA algorithm: {e}")
            current_result['quantum_tsa_vc_size'] = -1
            current_result['quantum_tsa_time'] = float('inf')
            current_result['quantum_tsa_is_valid'] = False

        # --- Run Exact Solver (SageMath) ---
        if run_exact_solver and SAGE_AVAILABLE:
            try:
                start_time = time.perf_counter()
                exact_vc, exact_vc_size = get_exact_vertex_cover_sage(G_nx)
                end_time = time.perf_counter()
                
                current_result['exact_vc'] = str(exact_vc)
                current_result['exact_vc_size'] = exact_vc_size
                current_result['exact_time'] = end_time - start_time
            except Exception as e:
                print(f"  Error in exact solver: {e}")
                current_result['exact_vc_size'] = -1
                current_result['exact_time'] = float('inf')
        else:
            current_result['exact_vc_size'] = -1
            current_result['exact_time'] = float('inf')

        # --- Run Greedy 2-Approximation ---
        try:
            start_time = time.perf_counter()
            approx_vc = greedy_2_approx_vertex_cover(G_nx)
            end_time = time.perf_counter()
            
            current_result['2approx_vc'] = str(approx_vc)
            current_result['2approx_vc_size'] = len(approx_vc)
            current_result['2approx_time'] = end_time - start_time
        except Exception as e:
            print(f"  Error in 2-approx algorithm: {e}")
            current_result['2approx_vc_size'] = -1
            current_result['2approx_time'] = float('inf')

        # --- Run FastVC Heuristic ---
        try:
            start_time = time.perf_counter()
            fastvc_vc = fast_vc_heuristic(G_nx)
            end_time = time.perf_counter()
            
            current_result['fastvc_vc'] = str(fastvc_vc)
            current_result['fastvc_vc_size'] = len(fastvc_vc)
            current_result['fastvc_time'] = end_time - start_time
        except Exception as e:
            print(f"  Error in FastVC heuristic: {e}")
            current_result['fastvc_vc_size'] = -1
            current_result['fastvc_time'] = float('inf')

        # --- Run Simulated Annealing ---
        try:
            start_time = time.perf_counter()
            sa_vc = simulated_annealing_vertex_cover(G_nx)
            end_time = time.perf_counter()
            
            current_result['sa_vc'] = str(sa_vc)
            current_result['sa_vc_size'] = len(sa_vc)
            current_result['sa_time'] = end_time - start_time
        except Exception as e:
            print(f"  Error in SA heuristic: {e}")
            current_result['sa_vc_size'] = -1
            current_result['sa_time'] = float('inf')

        # --- Calculate Approximation Ratios ---
        if current_result.get('exact_vc_size', -1) > 0:
            opt = current_result['exact_vc_size']
            current_result['quantum_approx_ratio'] = current_result['quantum_vc_size'] / opt
            current_result['quantum_tsa_approx_ratio'] = current_result['quantum_tsa_vc_size'] / opt
            current_result['2approx_approx_ratio'] = current_result['2approx_vc_size'] / opt
            current_result['fastvc_approx_ratio'] = current_result['fastvc_vc_size'] / opt
            current_result['sa_approx_ratio'] = current_result['sa_vc_size'] / opt
        else:
            current_result['quantum_approx_ratio'] = float('nan')
            current_result['quantum_tsa_approx_ratio'] = float('nan')
            current_result['2approx_approx_ratio'] = float('nan')
            current_result['fastvc_approx_ratio'] = float('nan')
            current_result['sa_approx_ratio'] = float('nan')
            
        results_to_append.append(current_result)
        
        # Save periodically
        if (i + 1) % save_interval == 0 or (i + 1) == total_graphs:
            new_results_df = pd.DataFrame(results_to_append)
            all_results_df = pd.concat([all_results_df, new_results_df], ignore_index=True)
            all_results_df.to_csv(results_filename, index=False)
            results_to_append = []
            print(f"  Partial results saved. Total: {len(all_results_df)} rows.")
            
            # Update processed signatures
            processed_signatures = set(
                all_results_df['num_nodes'].astype(str) + "_" + 
                all_results_df['num_edges'].astype(str) + "_" + 
                all_results_df['graph_iteration_index'].astype(str)
            )

    print("\nExperiments completed.")
    return all_results_df


def summarize_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate summary statistics from experimental results.
    
    Args:
        results_df: DataFrame containing experimental results.
        
    Returns:
        DataFrame with summary statistics grouped by graph size.
    """
    summary_cols = [
        'quantum_vc_size', 'quantum_tsa_vc_size', 'exact_vc_size',
        '2approx_vc_size', 'fastvc_vc_size', 'sa_vc_size',
        'quantum_time', 'quantum_tsa_time', 'exact_time',
        '2approx_time', 'fastvc_time', 'sa_time',
        'quantum_approx_ratio', 'quantum_tsa_approx_ratio',
        '2approx_approx_ratio', 'fastvc_approx_ratio', 'sa_approx_ratio'
    ]
    
    available_cols = [c for c in summary_cols if c in results_df.columns]
    
    summary = results_df.groupby('num_nodes')[available_cols].agg(['mean', 'std', 'min', 'max'])
    
    return summary
