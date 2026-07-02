# Boundary Conditions for Geometric Methods in Clinical Epidemiology

Code and data for the paper:

> **When Does Geometry Help Causal Inference? Boundary Conditions for Sheaf, Curvature, and Subspace Methods in Clinical Epidemiology**
>
> Elliot Tower

## Summary

Geometric and topological methods (sheaf cohomology, Grassmannian holonomy, discrete curvature) are increasingly proposed for causal inference in clinical epidemiology. This paper characterizes when they help and when they reduce to standard statistics, identifying three structural conditions that separate the two regimes:

1. **Subspace-valued data** — Grassmannian holonomy detects global inconsistency invisible to all pairwise and scalar alternatives. On scalar data, the sheaf test reduces algebraically to Cochran's Q.
2. **Cyclic consistency constraints** — Composed parallel transport accumulates signal as sqrt(m) over m loop steps, a separation mechanism unavailable to pairwise tests.
3. **Edge-specific heterogeneity** — Per-edge sheaf Q tests recover planted DAG structure that global tests miss.

Real-data validation on 61 published Mendelian randomization pairs across five clinical domains, ADNI longitudinal DAG analysis, and ABIDE multi-site cortical thickness holonomy.

## Repository structure

```
paper/              LaTeX source (v1-v4c), references, TMLR style files
figures/            All paper figures
experiments/
  ms_heterogeneity/ MS simulation experiments
  ad_heterogeneity/ AD simulation experiments
  clinical_epi/     Confound detection, HTE, prerequisites
  batch3_expansion/
    01_systematic_mr/       61-pair MR H1 classification pipeline
    02_enigma_holonomy/     Berry phase boundary analysis + ABIDE real-data test
    03_bnlearn_curvature/   Benchmark DAG curvature validation
```

## Key results

| Experiment | Result |
|---|---|
| Sheaf Q on scalar data | Algebraically identical to Cochran's Q |
| Berry phase holonomy (simulation) | Detected at p < 0.001; pairwise tests fail |
| Detection boundary | 80% power at 500 subjects/site on Gr(3,34) |
| ABIDE real data (20 sites, n=25-175) | Non-significant (p=0.16), confirming boundary |
| H1 classifier (61 MR pairs) | 85.2% accuracy, zero false positives |
| Per-edge DAG (ADNI) | Three-way edge classification |
| Forman-Ricci curvature | Acts as degree-deficit feature (AUROC 0.677) |
| Ollivier-Ricci curvature | Below chance (AUROC 0.466) |
| PCA + CATE clustering | Destroys treatment effect signal |

## Requirements

Experiments use standard scientific Python:

```
numpy, scipy, matplotlib, pandas, tqdm
```

Specific pipelines may additionally need `networkx`, `scikit-learn`, or `statsmodels`. No GPU required.

## Running experiments

Each experiment directory contains a self-contained `pipeline.py`:

```bash
# Berry phase boundary analysis
cd experiments/batch3_expansion/02_enigma_holonomy
uv run --no-project --with numpy --with scipy --with matplotlib --with tqdm --with pandas \
  python pipeline.py --berry-boundary --output results/boundary_r05

# ABIDE real-data holonomy
uv run --no-project --with numpy --with scipy --with matplotlib --with tqdm --with pandas \
  python pipeline.py --data-dir data/abide/ --k 3 --n-perms 2000 --output results/abide_k3 --plot

# Systematic MR classification
cd experiments/batch3_expansion/01_systematic_mr
uv run --no-project --with numpy --with scipy --with matplotlib --with tqdm --with pandas \
  python pipeline.py

# bnlearn curvature benchmark
cd experiments/batch3_expansion/03_bnlearn_curvature
uv run --no-project --with numpy --with scipy --with matplotlib --with networkx --with pandas \
  python pipeline.py
```

## Data

- **ABIDE cortical thickness**: 1,031 subjects across 20 sites, 34 Desikan-Killiany regions (left hemisphere). Downloaded from the ABIDE S3 bucket via `download_abide.py`. Included in `experiments/batch3_expansion/02_enigma_holonomy/data/abide/`.
- **MR pairs**: 61 curated Mendelian randomization pairs across MS, AD, cardiometabolic, cancer, and psychiatric domains. In `experiments/batch3_expansion/01_systematic_mr/data/pairs_curated.json`.

## License

MIT. See [LICENSE](LICENSE).
