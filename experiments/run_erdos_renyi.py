#!/usr/bin/env python3
"""
Experiment script for Erdős-Rényi (random) graphs.

This script generates ER graphs with various parameters and runs all MVC
algorithms to collect comparative performance data.

Usage:
    python run_erdos_renyi.py [--nodes N1 N2 ...] [--prob P1 P2 ...] [--graphs G]
    
Example:
    python run_erdos_renyi.py --nodes 100 200 --prob 0.1 0.3 0.5 --graphs 10
"""

import argparse
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from quantum_walk_mvc import run_experiments
from quantum_walk_mvc.graph_generators import generate_erdos_renyi_graphs


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run MVC experiments on Erdős-Rényi graphs'
    )
    parser.add_argument(
        '--nodes', 
        type=int, 
        nargs='+',
        default=[100, 200, 300, 400, 500],
        help='List of node counts to test'
    )
    parser.add_argument(
        '--prob', 
        type=float, 
        nargs='+',
        default=[0.1, 0.2, 0.3, 0.4, 0.5],
        help='List of edge probabilities'
    )
    parser.add_argument(
        '--graphs', 
        type=int, 
        default=10,
        help='Number of graphs per (n, p) configuration'
    )
    parser.add_argument(
        '--t-max', 
        type=float, 
        default=0.01,
        help='Quantum walk evolution time parameter'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default=None,
        help='Output filename (default: auto-generated)'
    )
    parser.add_argument(
        '--no-exact',
        action='store_true',
        help='Skip exact solver (recommended for large graphs)'
    )
    
    return parser.parse_args()


def main():
    """Main entry point for ER graph experiments."""
    args = parse_args()
    
    print("=" * 60)
    print("Quantum Walk MVC - Erdős-Rényi Graphs Experiment")
    print("=" * 60)
    print(f"Node counts: {args.nodes}")
    print(f"Edge probabilities: {args.prob}")
    print(f"Graphs per setting: {args.graphs}")
    print(f"t_max: {args.t_max}")
    print(f"Run exact solver: {not args.no_exact}")
    print("=" * 60)
    
    # Generate output filename
    if args.output is None:
        prob_str = ''.join([str(p).replace('.', '') for p in args.prob])
        filename_suffix = f"N{args.nodes[0]}-{args.nodes[-1]}_p{prob_str}_G{args.graphs}"
        results_filename = f'results_erdos_renyi_{filename_suffix}.csv'
    else:
        results_filename = args.output
    
    print(f"\nOutput file: {results_filename}")
    
    # Generate test graphs
    print("\n[1/2] Generating Erdős-Rényi graphs...")
    graphs, params = generate_erdos_renyi_graphs(
        num_nodes_range=args.nodes,
        edge_prob_range=args.prob,
        num_graphs_per_setting=args.graphs,
        ensure_connected=True
    )
    
    if len(graphs) == 0:
        print("Error: No graphs generated. Check parameters.")
        sys.exit(1)
    
    # Run experiments
    print("\n[2/2] Running experiments...")
    results_df = run_experiments(
        test_graphs=graphs,
        t_max_quantum=args.t_max,
        results_filename=results_filename,
        run_exact_solver=not args.no_exact
    )
    
    # Add graph parameters to results
    if len(results_df) == len(params):
        results_df['graph_params'] = [str(p) for p in params]
        results_df.to_csv(results_filename, index=False)
    
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETED")
    print(f"Results saved to: {results_filename}")
    print(f"Total graphs processed: {len(results_df)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
