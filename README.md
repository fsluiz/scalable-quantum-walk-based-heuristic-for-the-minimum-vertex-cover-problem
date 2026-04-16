# Quantum Walk Minimum Vertex Cover

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2512.02940-b31b1b.svg)](https://doi.org/10.48550/arXiv.2512.02940)

Code and data for the paper:

> **Scalable Quantum Walk-Based Heuristics for the Minimum Vertex Cover Problem**  
> F. S. Luiz, A. K. F. Iwakami, D. H. Moraes, M. C. de Oliveira  
> arXiv:2512.02940

## Overview

A continuous-time quantum walk (CTQW) heuristic for the **Minimum Vertex Cover (MVC)** problem using a binary encoding that requires only ⌈log₂V⌉ qubits for V-node graphs — an exponential reduction compared to qubit-per-node approaches.

A short-time expansion reveals that the quantum selection criterion reduces to a classical spectral quantity computable in O(|E|) per iteration, yielding a **quantum-inspired spectral greedy heuristic** (Algorithm 2) that matches the CTQW solution on 98.3 % of instances.

**Key results (14,680 instances, N ∈ [4, 150], exact MILP reference):**
- CTQW/spectral greedy: mean ratio 1.015 vs 1.023 for degree-greedy
- 100 % best cover at large scale (N ∈ [10³, 10⁵], 210 instances)
- Worst-case ratio 1.188, well below inapproximability threshold ≈ 1.3606

## Repository Structure

```
quantum-walk-mvc/
├── src/quantum_walk_mvc/       # Python package
│   ├── core.py                 # CTQW algorithms (dense + sparse)
│   ├── heuristics.py           # Classical baselines + Sage exact solver
│   ├── heuristics_igraph.py    # Fast Numba/igraph spectral greedy (large graphs)
│   ├── graph_generators.py     # BA, ER, regular graph factories
│   ├── bloqade_mvc.py          # QuEra neutral-atom hardware support
│   └── utils.py                # Experiment runner, incremental CSV saving
├── experiments/                # Reproducibility scripts
│   ├── run_*.py                # Experiment drivers (one per paper figure/table)
│   ├── bench_real_graphs.py    # SNAP real-world network benchmark
│   └── regen_fig6_extended.py  # Regenerate Figure 6 from legacy data
├── results/                    # All key CSV results (tracked)
├── data/
│   ├── snap/                   # SNAP real-world graphs (.gz, tracked)
│   └── legacy/                 # Large legacy benchmark CSVs (Zenodo only)
│       └── README.md           # Download instructions
├── paper/                      # Paper source (LaTeX + figures)
│   ├── main_v2.tex
│   ├── lib.bib
│   └── figures/
├── tests/                      # Unit tests
└── notebooks/                  # Jupyter notebooks
```

## Installation

```bash
git clone https://github.com/fsluiz/quantum-walk-mvc.git
cd quantum-walk-mvc
pip install -e ".[dev,quantum,visualization]"
```

**Optional dependencies:**
- `qiskit` + `qiskit-ibm-runtime` + `qiskit-aer` — quantum hardware/simulation experiments
- `igraph` + `numba` — fast large-scale benchmarks (bench_real_graphs.py, run_scalability_large.py)
- SageMath ≥ 10 — exact MILP solver (`heuristics.get_exact_vertex_cover_sage`)

## Quick Start

```python
import networkx as nx
from quantum_walk_mvc.core import quantum_walk_mvc_sparse

G = nx.barabasi_albert_graph(100, 2, seed=0)
cover = quantum_walk_mvc_sparse(G, t_opt=0.01)
print(f"Cover size: {len(cover)}")   # → typically ~43–46
```

## Running Experiments

All scripts run from the repo root:

```bash
# Classical simulation — ER graphs
python experiments/run_erdos_renyi.py --nodes 100 500 1000 --prob 0.3 0.5 --graphs 10

# Degree-greedy comparison (Figure 6 — using legacy data, see data/legacy/README.md)
python experiments/regen_fig6_extended.py

# Real-world SNAP networks benchmark (Table VI)
python experiments/bench_real_graphs.py

# Large-scale scalability (Table V, Figure 7)
python experiments/run_scalability_large.py

# Qiskit noiseless + FakeTorino noise (V=4,8,12,16)
python experiments/run_qiskit_binary_encoding.py
```

## Results

All published data is in `results/`:

| File | Description |
|------|-------------|
| `results_dg_extended.csv` | 14,680-instance benchmark N∈[4,150], MILP ref. + DG |
| `results_real_graphs.csv` | SNAP real-world networks (8 instances) |
| `results_scalability_large.csv` | Large-scale N∈[10³,10⁵], 210 instances |
| `results_spectral_greedy_validation.csv` | CTQW vs spectral greedy equivalence |
| `results_hard_instances.csv` | Hard graph families (crown, bipartite, Petersen) |
| `results_t_sensitivity.csv` | Evolution time sensitivity sweep |
| `results_noise_scaling_hw.csv` | ibm_marrakesh hardware runs |
| `results_noise_scaling_noiseless.csv` | Noiseless CTQW vs MILP |
| `results_real_hardware.csv` | ibm_marrakesh (ER V=4, BA V=8) |
| `results_qiskit_binary_encoding.csv` | Qiskit noiseless + FakeTorino |

## Large Data (Zenodo)

The legacy benchmark CSVs (ER 375 MB, BA 134 MB) used to generate Figure 6 are
archived on Zenodo. See `data/legacy/README.md` for download instructions.
The processed output (`results/results_dg_extended.csv`) is sufficient to
reproduce all paper figures without re-running the full pipeline.

## Running Tests

```bash
pytest tests/ -v
```

## Citation

```bibtex
@article{luiz2025qwmvc,
  title   = {Scalable Quantum Walk-Based Heuristics for the Minimum Vertex Cover Problem},
  author  = {Luiz, F. S. and Iwakami, A. K. F. and Moraes, D. H. and de Oliveira, M. C.},
  journal = {arXiv preprint},
  year    = {2025},
  doi     = {10.48550/arXiv.2512.02940}
}
```

## Acknowledgments

FSL thanks the UNICAMP Postdoctoral Researcher Program for financial support.
AKFI acknowledges FAPESP (No. 2023/13524-0). MCO acknowledges CNPq INCT-AQC
(No. 408884/2024-0) and FAPESP/CRISQuaM (No. 2013/07276-1).

## Contact

Issues and questions: open a GitHub issue or contact [fsluiz@unicamp.br](mailto:fsluiz@unicamp.br).
