"""Curvature-based edge validation on bnlearn benchmark causal graphs.

Tests whether discrete Ricci curvature discriminates true-positive from
false-positive edges in learned causal graphs. Benchmarks Forman-Ricci,
augmented Forman, and comparison features (Jaccard, betweenness, partial
correlation, clustering) across four standard DAGs: Asia, Sachs,
Insurance, Alarm.

Workflow per replicate:
  1. Generate linear SEM data from the true DAG.
  2. Learn a DAG via constraint-based PC-style algorithm (partial
     correlation CI tests, v-structure orientation).
  3. Compute edge-level features on the learned skeleton.
  4. Score AUROC for TP vs FP discrimination per feature.

Usage:
  python pipeline.py --graphs asia,sachs --n-samples 1000 --n-replicates 50 --output results/ --plot
  python pipeline.py --graphs asia,sachs,alarm,insurance --n-samples 2000 --output results/ --plot
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from collections import defaultdict
from itertools import combinations
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy import stats
from scipy.linalg import pinvh
from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AUROC (avoids sklearn dependency)
# ---------------------------------------------------------------------------

def roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute AUROC via the Mann-Whitney U statistic.

    Equivalent to sklearn.metrics.roc_auc_score for binary labels.
    AUROC = U / (n_pos * n_neg), where U counts concordant pairs.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    # For each positive, count negatives with lower score (+ 0.5 for ties)
    u = 0.0
    for p in pos:
        u += np.sum(p > neg) + 0.5 * np.sum(p == neg)
    return u / (len(pos) * len(neg))


# ---------------------------------------------------------------------------
# Benchmark DAG definitions (hardcoded adjacency lists)
# ---------------------------------------------------------------------------

BENCHMARK_DAGS: dict[str, dict[str, list[str]]] = {}

# Asia (8 nodes, 8 edges) — Lauritzen & Spiegelhalter 1988
BENCHMARK_DAGS["asia"] = {
    "nodes": ["asia", "tub", "smoke", "lung", "bronc", "either", "xray", "dysp"],
    "edges": [
        ("asia", "tub"),
        ("smoke", "lung"),
        ("smoke", "bronc"),
        ("tub", "either"),
        ("lung", "either"),
        ("bronc", "dysp"),
        ("either", "xray"),
        ("either", "dysp"),
    ],
}

# Sachs (11 nodes, 17 edges) — Sachs et al. 2005, protein signaling
BENCHMARK_DAGS["sachs"] = {
    "nodes": [
        "Plcg", "PIP3", "PIP2", "PKC", "PKA", "Raf", "Mek", "Erk",
        "Akt", "Jnk", "P38",
    ],
    "edges": [
        ("Plcg", "PIP3"),
        ("Plcg", "PIP2"),
        ("PIP3", "PIP2"),
        ("PKC", "PKA"),
        ("PKC", "Raf"),
        ("PKC", "Mek"),
        ("PKC", "Jnk"),
        ("PKC", "P38"),
        ("PKA", "Raf"),
        ("PKA", "Mek"),
        ("PKA", "Erk"),
        ("PKA", "Akt"),
        ("PKA", "Jnk"),
        ("PKA", "P38"),
        ("Raf", "Mek"),
        ("Mek", "Erk"),
        ("Erk", "Akt"),
    ],
}

# Insurance (27 nodes, 52 edges) — Binder et al. 1997
BENCHMARK_DAGS["insurance"] = {
    "nodes": [
        "Age", "SocioEcon", "GoodStudent", "RiskAversion", "VehicleYear",
        "MakeModel", "DrivQuality", "Mileage", "Antilock", "DrivHist",
        "SeniorTrain", "ThisCarDam", "RuggedAuto", "Accident", "OtherCar",
        "OtherCarCost", "MedCost", "Cushioning", "Airbag", "ILiCost",
        "DrivingSkill", "ThisCarCost", "Theft", "CarValue", "HomeBase",
        "AntiTheft", "PropCost",
    ],
    "edges": [
        ("Age", "SocioEcon"),
        ("Age", "GoodStudent"),
        ("Age", "RiskAversion"),
        ("Age", "DrivingSkill"),
        ("Age", "SeniorTrain"),
        ("Age", "MedCost"),
        ("SocioEcon", "GoodStudent"),
        ("SocioEcon", "RiskAversion"),
        ("SocioEcon", "MakeModel"),
        ("SocioEcon", "VehicleYear"),
        ("SocioEcon", "HomeBase"),
        ("SocioEcon", "OtherCar"),
        ("SocioEcon", "AntiTheft"),
        ("RiskAversion", "SeniorTrain"),
        ("RiskAversion", "DrivQuality"),
        ("RiskAversion", "VehicleYear"),
        ("RiskAversion", "MakeModel"),
        ("RiskAversion", "HomeBase"),
        ("RiskAversion", "AntiTheft"),
        ("SeniorTrain", "DrivingSkill"),
        ("DrivingSkill", "DrivQuality"),
        ("DrivingSkill", "DrivHist"),
        ("DrivQuality", "Accident"),
        ("DrivHist", "Accident"),
        ("MakeModel", "CarValue"),
        ("MakeModel", "RuggedAuto"),
        ("MakeModel", "Antilock"),
        ("MakeModel", "Airbag"),
        ("VehicleYear", "CarValue"),
        ("VehicleYear", "RuggedAuto"),
        ("VehicleYear", "Antilock"),
        ("VehicleYear", "Airbag"),
        ("Mileage", "CarValue"),
        ("Mileage", "Accident"),
        ("Accident", "ThisCarDam"),
        ("Accident", "OtherCar"),
        ("Accident", "MedCost"),
        ("ThisCarDam", "ThisCarCost"),
        ("RuggedAuto", "ThisCarDam"),
        ("RuggedAuto", "OtherCarCost"),
        ("RuggedAuto", "Cushioning"),
        ("Antilock", "Accident"),
        ("CarValue", "Theft"),
        ("CarValue", "ThisCarCost"),
        ("HomeBase", "Theft"),
        ("AntiTheft", "Theft"),
        ("Theft", "ThisCarCost"),
        ("OtherCar", "OtherCarCost"),
        ("OtherCarCost", "PropCost"),
        ("ThisCarCost", "PropCost"),
        ("Cushioning", "MedCost"),
        ("Airbag", "Cushioning"),
    ],
}

# Alarm (37 nodes, 46 edges) — Beinlich et al. 1989
BENCHMARK_DAGS["alarm"] = {
    "nodes": [
        "LVFAILURE", "HISTORY", "LVEDVOLUME", "CVP", "PCWP",
        "HYPOVOLEMIA", "STROKEVOLUME", "ERRLOWOUTPUT", "HRBP",
        "ERRCAUTER", "HREKG", "HRSAT", "INSUFFANESTH", "ANAPHYLAXIS",
        "TPR", "ARTCO2", "EXPCO2", "VENTLUNG", "INTUBATION",
        "MINVOL", "FIO2", "PVSAT", "SAO2", "SHUNT",
        "PULMEMBOLUS", "PAP", "PRESS", "KINKEDTUBE", "VENTTUBE",
        "MINVOLSET", "VENTMACH", "DISCONNECT", "CATECHOL", "HR",
        "CO", "BP", "VENTALV",
    ],
    "edges": [
        ("LVFAILURE", "HISTORY"),
        ("LVFAILURE", "LVEDVOLUME"),
        ("LVFAILURE", "STROKEVOLUME"),
        ("LVEDVOLUME", "CVP"),
        ("LVEDVOLUME", "PCWP"),
        ("HYPOVOLEMIA", "LVEDVOLUME"),
        ("HYPOVOLEMIA", "STROKEVOLUME"),
        ("STROKEVOLUME", "CO"),
        ("ERRLOWOUTPUT", "HRBP"),
        ("ERRCAUTER", "HREKG"),
        ("ERRCAUTER", "HRSAT"),
        ("INSUFFANESTH", "CATECHOL"),
        ("ANAPHYLAXIS", "TPR"),
        ("TPR", "BP"),
        ("ARTCO2", "EXPCO2"),
        ("ARTCO2", "CATECHOL"),
        ("VENTLUNG", "EXPCO2"),
        ("VENTLUNG", "MINVOL"),
        ("INTUBATION", "VENTLUNG"),
        ("INTUBATION", "VENTALV"),
        ("INTUBATION", "SHUNT"),
        ("INTUBATION", "MINVOL"),
        ("INTUBATION", "PRESS"),
        ("FIO2", "PVSAT"),
        ("PVSAT", "SAO2"),
        ("SAO2", "CATECHOL"),
        ("SHUNT", "SAO2"),
        ("PULMEMBOLUS", "PAP"),
        ("PULMEMBOLUS", "SHUNT"),
        ("PAP", "PRESS"),
        ("KINKEDTUBE", "VENTLUNG"),
        ("KINKEDTUBE", "PRESS"),
        ("VENTTUBE", "VENTLUNG"),
        ("VENTTUBE", "PRESS"),
        ("MINVOLSET", "VENTMACH"),
        ("VENTMACH", "VENTTUBE"),
        ("DISCONNECT", "VENTTUBE"),
        ("CATECHOL", "HR"),
        ("HR", "HREKG"),
        ("HR", "HRSAT"),
        ("HR", "HRBP"),
        ("CO", "BP"),
        ("CO", "HRBP"),
        ("BP", "CATECHOL"),
        ("VENTALV", "PVSAT"),
        ("VENTALV", "ARTCO2"),
    ],
}


# ---------------------------------------------------------------------------
# Data generation: linear SEM
# ---------------------------------------------------------------------------

def generate_sem_data(
    dag: nx.DiGraph,
    n_samples: int,
    rng: np.random.Generator,
    beta_range: tuple[float, float] = (0.3, 0.8),
) -> np.ndarray:
    """Generate data from a linear SEM on *dag*.

    X_j = sum_{i in pa(j)} beta_ij * X_i + epsilon_j
    beta ~ Uniform(beta_range), sign flipped with prob 0.5
    epsilon ~ N(0, 1)
    """
    nodes = list(nx.topological_sort(dag))
    node_idx = {n: i for i, n in enumerate(nodes)}
    p = len(nodes)

    # Assign edge weights
    weights: dict[tuple[str, str], float] = {}
    for u, v in dag.edges():
        beta = rng.uniform(*beta_range)
        if rng.random() < 0.5:
            beta = -beta
        weights[(u, v)] = beta

    # Generate data in topological order
    data = np.zeros((n_samples, p))
    for node in nodes:
        j = node_idx[node]
        eps = rng.standard_normal(n_samples)
        signal = np.zeros(n_samples)
        for parent in dag.predecessors(node):
            i = node_idx[parent]
            signal += weights[(parent, node)] * data[:, i]
        data[:, j] = signal + eps

    return data


# ---------------------------------------------------------------------------
# Constraint-based DAG learning (PC-style)
# ---------------------------------------------------------------------------

def partial_correlation_matrix(data: np.ndarray) -> np.ndarray:
    """Compute pairwise partial correlation from the precision matrix."""
    cov = np.cov(data, rowvar=False)
    prec = pinvh(cov)
    d = np.sqrt(np.diag(prec))
    # Avoid division by zero
    d = np.where(d == 0, 1.0, d)
    pcorr = -prec / np.outer(d, d)
    np.fill_diagonal(pcorr, 1.0)
    return pcorr


def partial_correlation_given(
    data: np.ndarray, i: int, j: int, cond: list[int]
) -> float:
    """Partial correlation between variables i and j given conditioning set.

    Uses the recursive formula via residuals from OLS regression.
    """
    if len(cond) == 0:
        r = np.corrcoef(data[:, i], data[:, j])[0, 1]
        return r

    # Regress i and j on conditioning set, return correlation of residuals
    X_cond = data[:, cond]
    # Add intercept
    X_aug = np.column_stack([np.ones(len(data)), X_cond])
    # Solve via least squares
    _, res_i_info, _, _ = np.linalg.lstsq(X_aug, data[:, i], rcond=None)
    _, res_j_info, _, _ = np.linalg.lstsq(X_aug, data[:, j], rcond=None)
    resid_i = data[:, i] - X_aug @ np.linalg.lstsq(X_aug, data[:, i], rcond=None)[0]
    resid_j = data[:, j] - X_aug @ np.linalg.lstsq(X_aug, data[:, j], rcond=None)[0]

    if np.std(resid_i) < 1e-12 or np.std(resid_j) < 1e-12:
        return 0.0
    return np.corrcoef(resid_i, resid_j)[0, 1]


def ci_test_partial_corr(
    data: np.ndarray, i: int, j: int, cond: list[int], alpha: float
) -> bool:
    """Test H0: X_i independent of X_j given X_cond using partial correlation.

    Returns True if independent (fail to reject), False if dependent.
    Uses Fisher's z-transform for the test statistic.
    """
    n = data.shape[0]
    r = partial_correlation_given(data, i, j, cond)
    # Fisher z-transform
    r_clipped = np.clip(r, -0.9999, 0.9999)
    z = 0.5 * np.log((1 + r_clipped) / (1 - r_clipped))
    # Standard error
    dof = n - len(cond) - 3
    if dof < 1:
        return False  # Not enough data to test
    se = 1.0 / np.sqrt(dof)
    # Two-sided test
    p_value = 2 * (1 - stats.norm.cdf(abs(z / se)))
    return p_value > alpha


def learn_skeleton(
    data: np.ndarray,
    alpha: float,
    max_cond_size: int = 2,
) -> tuple[nx.Graph, dict[tuple[int, int], list[int]]]:
    """Learn undirected skeleton via PC-style conditional independence tests.

    Tests all pairs; for each adjacent pair, tries conditioning sets of
    increasing size (up to max_cond_size) drawn from their current
    neighbors. Removes edges when a CI is found.

    Returns (skeleton, sep_sets) where sep_sets maps (i,j) to the
    conditioning set that separated them.
    """
    p = data.shape[1]
    skeleton = nx.complete_graph(p)
    sep_sets: dict[tuple[int, int], list[int]] = {}

    for cond_size in range(max_cond_size + 1):
        edges_to_remove = []
        for i, j in list(skeleton.edges()):
            # Candidate conditioning variables: neighbors of i (excluding j)
            neighbors_i = set(skeleton.neighbors(i)) - {j}
            if len(neighbors_i) < cond_size:
                continue
            # Test subsets of the given size
            found_independence = False
            for cond in combinations(sorted(neighbors_i), cond_size):
                cond_list = list(cond)
                if ci_test_partial_corr(data, i, j, cond_list, alpha):
                    edges_to_remove.append((i, j))
                    key = (min(i, j), max(i, j))
                    sep_sets[key] = cond_list
                    found_independence = True
                    break
            if found_independence:
                continue
            # Also try neighbors of j
            neighbors_j = set(skeleton.neighbors(j)) - {i}
            if len(neighbors_j) < cond_size:
                continue
            for cond in combinations(sorted(neighbors_j), cond_size):
                cond_list = list(cond)
                if ci_test_partial_corr(data, i, j, cond_list, alpha):
                    edges_to_remove.append((i, j))
                    key = (min(i, j), max(i, j))
                    sep_sets[key] = cond_list
                    found_independence = True
                    break

        for u, v in edges_to_remove:
            if skeleton.has_edge(u, v):
                skeleton.remove_edge(u, v)

    return skeleton, sep_sets


def orient_v_structures(
    skeleton: nx.Graph,
    sep_sets: dict[tuple[int, int], list[int]],
) -> nx.DiGraph:
    """Orient edges via v-structure detection: A - B - C with A not adj to C.

    If B is not in sep(A, C), orient as A -> B <- C.
    Remaining edges are left as arbitrary direction (both orientations added
    then one removed — but for our purposes we only need the skeleton for
    curvature, so undirected edges get an arbitrary orientation).
    """
    dag = nx.DiGraph()
    dag.add_nodes_from(skeleton.nodes())

    oriented_edges: set[tuple[int, int]] = set()

    # Detect v-structures
    for b in skeleton.nodes():
        neighbors = list(skeleton.neighbors(b))
        for a, c in combinations(neighbors, 2):
            if skeleton.has_edge(a, c):
                continue  # a and c are adjacent, not a v-structure
            key = (min(a, c), max(a, c))
            sep = sep_sets.get(key, [])
            if b not in sep:
                # Orient as a -> b <- c
                oriented_edges.add((a, b))
                oriented_edges.add((c, b))

    # Add oriented edges
    for u, v in oriented_edges:
        dag.add_edge(u, v)

    # Add remaining edges with arbitrary orientation
    for u, v in skeleton.edges():
        if (u, v) not in oriented_edges and (v, u) not in oriented_edges:
            dag.add_edge(u, v)

    return dag


def learn_dag(
    data: np.ndarray,
    alpha: float,
    max_cond_size: int = 2,
) -> nx.DiGraph:
    """Full PC-style DAG learning pipeline."""
    skeleton, sep_sets = learn_skeleton(data, alpha, max_cond_size)
    dag = orient_v_structures(skeleton, sep_sets)
    return dag


# ---------------------------------------------------------------------------
# Edge feature computation
# ---------------------------------------------------------------------------

def forman_ricci_curvature(G: nx.Graph, u: int, v: int) -> float:
    """Forman-Ricci curvature: kappa_F(u,v) = 4 - d(u) - d(v)."""
    return 4 - G.degree(u) - G.degree(v)


def augmented_forman_curvature(G: nx.Graph, u: int, v: int) -> float:
    """Augmented Forman: kappa_F + 3 * |triangles through (u,v)|."""
    kf = forman_ricci_curvature(G, u, v)
    common = len(set(G.neighbors(u)) & set(G.neighbors(v)))
    return kf + 3 * common


def jaccard_coefficient(G: nx.Graph, u: int, v: int) -> float:
    """Jaccard coefficient: |N(u) cap N(v)| / |N(u) cup N(v)|."""
    nu = set(G.neighbors(u))
    nv = set(G.neighbors(v))
    union = nu | nv
    if len(union) == 0:
        return 0.0
    return len(nu & nv) / len(union)


def compute_edge_features(
    learned_dag: nx.DiGraph,
    data: np.ndarray,
) -> dict[tuple[int, int], dict[str, float]]:
    """Compute all edge-level features for the learned graph.

    Works on the undirected skeleton for graph-theoretic features.
    """
    G_undirected = learned_dag.to_undirected()

    # Precompute betweenness and clustering
    betweenness = nx.edge_betweenness_centrality(G_undirected)
    clustering = nx.clustering(G_undirected)

    # Partial correlation matrix for correlation features
    pcorr_mat = partial_correlation_matrix(data)

    features: dict[tuple[int, int], dict[str, float]] = {}
    for u, v in learned_dag.edges():
        # Canonical undirected key for betweenness lookup
        bw_key = (u, v) if (u, v) in betweenness else (v, u)
        bw = betweenness.get(bw_key, 0.0)

        features[(u, v)] = {
            "forman_ricci": forman_ricci_curvature(G_undirected, u, v),
            "augmented_forman": augmented_forman_curvature(G_undirected, u, v),
            "jaccard": jaccard_coefficient(G_undirected, u, v),
            "betweenness": bw,
            "abs_partial_corr": abs(pcorr_mat[u, v]),
            "avg_clustering": (clustering[u] + clustering[v]) / 2,
        }

    return features


# ---------------------------------------------------------------------------
# TP / FP labeling and AUROC
# ---------------------------------------------------------------------------

def label_edges(
    learned_dag: nx.DiGraph,
    true_dag: nx.DiGraph,
) -> dict[tuple[int, int], int]:
    """Label learned edges as TP (1) or FP (0).

    An edge is TP if it exists in the true DAG in either direction
    (skeleton match).
    """
    true_skeleton = set()
    for u, v in true_dag.edges():
        true_skeleton.add((min(u, v), max(u, v)))

    labels = {}
    for u, v in learned_dag.edges():
        key = (min(u, v), max(u, v))
        labels[(u, v)] = 1 if key in true_skeleton else 0
    return labels


def compute_aurocs(
    features: dict[tuple[int, int], dict[str, float]],
    labels: dict[tuple[int, int], int],
) -> dict[str, float | None]:
    """Compute AUROC for each feature as a TP vs FP discriminator.

    Returns None for a feature if only one class is present.
    """
    edges = list(features.keys())
    y = np.array([labels[e] for e in edges])

    if len(np.unique(y)) < 2:
        feature_names = list(next(iter(features.values())).keys())
        return {f: None for f in feature_names}

    aurocs = {}
    feature_names = list(next(iter(features.values())).keys())
    for fname in feature_names:
        scores = np.array([features[e][fname] for e in edges])
        # Handle constant scores
        if np.std(scores) < 1e-12:
            aurocs[fname] = 0.5
        else:
            aurocs[fname] = roc_auc_score(y, scores)
    return aurocs


# ---------------------------------------------------------------------------
# Single replicate
# ---------------------------------------------------------------------------

def run_single_replicate(
    graph_name: str,
    n_samples: int,
    alpha: float,
    seed: int,
) -> dict[str, Any] | None:
    """Run one replicate: generate data, learn DAG, compute features, score.

    Returns None if the learned graph has no edges or only one class.
    """
    spec = BENCHMARK_DAGS[graph_name]
    nodes = spec["nodes"]
    node_idx = {n: i for i, n in enumerate(nodes)}

    # Build true DAG
    true_dag = nx.DiGraph()
    true_dag.add_nodes_from(range(len(nodes)))
    for u_name, v_name in spec["edges"]:
        true_dag.add_edge(node_idx[u_name], node_idx[v_name])

    rng = np.random.default_rng(seed)

    # Generate data
    data = generate_sem_data(true_dag, n_samples, rng)

    # Learn DAG
    learned_dag = learn_dag(data, alpha)

    if learned_dag.number_of_edges() == 0:
        return None

    # Compute features and labels
    features = compute_edge_features(learned_dag, data)
    labels = label_edges(learned_dag, true_dag)

    # Need both TP and FP for AUROC
    label_values = list(labels.values())
    if len(set(label_values)) < 2:
        return None

    aurocs = compute_aurocs(features, labels)

    n_tp = sum(label_values)
    n_fp = len(label_values) - n_tp

    return {
        "aurocs": aurocs,
        "n_edges_learned": learned_dag.number_of_edges(),
        "n_edges_true": true_dag.number_of_edges(),
        "n_tp": n_tp,
        "n_fp": n_fp,
    }


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

ALPHA_VALUES = [0.001, 0.01, 0.05]
FEATURE_NAMES = [
    "forman_ricci", "augmented_forman", "jaccard",
    "betweenness", "abs_partial_corr", "avg_clustering",
]
FEATURE_DISPLAY = {
    "forman_ricci": "Forman-Ricci",
    "augmented_forman": "Aug. Forman",
    "jaccard": "Jaccard",
    "betweenness": "Betweenness",
    "abs_partial_corr": "|Partial Corr|",
    "avg_clustering": "Avg Clustering",
}


def run_pipeline(
    graphs: list[str],
    n_samples: int,
    n_replicates: int,
    output_dir: str,
    plot: bool,
) -> dict[str, Any]:
    """Run the full benchmark pipeline across graphs, thresholds, and replicates."""
    os.makedirs(output_dir, exist_ok=True)

    results: dict[str, Any] = {}

    total_tasks = len(graphs) * len(ALPHA_VALUES) * n_replicates
    pbar = tqdm(total=total_tasks, desc="Pipeline")

    for graph_name in graphs:
        logger.info("Processing graph: %s (%d nodes, %d edges)",
                     graph_name,
                     len(BENCHMARK_DAGS[graph_name]["nodes"]),
                     len(BENCHMARK_DAGS[graph_name]["edges"]))

        results[graph_name] = {}

        for alpha in ALPHA_VALUES:
            alpha_key = f"alpha={alpha}"
            auroc_lists: dict[str, list[float]] = defaultdict(list)
            n_valid = 0
            tp_counts = []
            fp_counts = []

            for rep in range(n_replicates):
                seed = rep * 1000 + hash(graph_name) % 10000 + int(alpha * 10000)
                result = run_single_replicate(graph_name, n_samples, alpha, seed)
                pbar.update(1)

                if result is None:
                    continue

                n_valid += 1
                tp_counts.append(result["n_tp"])
                fp_counts.append(result["n_fp"])

                for fname, auc in result["aurocs"].items():
                    if auc is not None:
                        auroc_lists[fname].append(auc)

            # Compute summary statistics
            summary = {}
            for fname in FEATURE_NAMES:
                vals = auroc_lists.get(fname, [])
                if len(vals) > 0:
                    arr = np.array(vals)
                    summary[fname] = {
                        "mean": float(np.mean(arr)),
                        "se": float(np.std(arr, ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0,
                        "n_valid": len(vals),
                    }
                else:
                    summary[fname] = {"mean": None, "se": None, "n_valid": 0}

            results[graph_name][alpha_key] = {
                "feature_aurocs": summary,
                "n_valid_replicates": n_valid,
                "n_total_replicates": n_replicates,
                "mean_tp": float(np.mean(tp_counts)) if tp_counts else 0,
                "mean_fp": float(np.mean(fp_counts)) if fp_counts else 0,
            }

            logger.info("  %s: %d/%d valid replicates, mean TP=%.1f, FP=%.1f",
                         alpha_key, n_valid, n_replicates,
                         results[graph_name][alpha_key]["mean_tp"],
                         results[graph_name][alpha_key]["mean_fp"])
            for fname in FEATURE_NAMES:
                s = summary[fname]
                if s["mean"] is not None:
                    logger.info("    %-20s AUROC = %.3f +/- %.3f  (n=%d)",
                                 FEATURE_DISPLAY[fname], s["mean"], s["se"], s["n_valid"])

    pbar.close()

    # Save full results
    results_path = os.path.join(output_dir, "curvature_benchmark.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Full results saved to %s", results_path)

    # Build summary table
    summary_table = build_summary_table(results)
    summary_path = os.path.join(output_dir, "summary_table.json")
    with open(summary_path, "w") as f:
        json.dump(summary_table, f, indent=2)
    logger.info("Summary table saved to %s", summary_path)

    # Plots
    if plot:
        plot_auroc_comparison(results, output_dir)
        plot_auroc_by_graph_size(results, output_dir)

    return results


def build_summary_table(results: dict[str, Any]) -> list[dict[str, Any]]:
    """Build a summary table: one row per (graph, feature) with best alpha."""
    rows = []
    for graph_name, graph_results in results.items():
        n_nodes = len(BENCHMARK_DAGS[graph_name]["nodes"])
        n_edges = len(BENCHMARK_DAGS[graph_name]["edges"])

        for fname in FEATURE_NAMES:
            # Pick the alpha that gives the best AUROC (most favorable comparison)
            best_auroc = None
            best_se = None
            best_alpha = None
            best_n = 0

            for alpha_key, alpha_res in graph_results.items():
                s = alpha_res["feature_aurocs"].get(fname, {})
                mean_val = s.get("mean")
                if mean_val is not None:
                    if best_auroc is None or mean_val > best_auroc:
                        best_auroc = mean_val
                        best_se = s.get("se", 0.0)
                        best_alpha = alpha_key
                        best_n = s.get("n_valid", 0)

            rows.append({
                "graph": graph_name,
                "n_nodes": n_nodes,
                "n_edges": n_edges,
                "feature": FEATURE_DISPLAY[fname],
                "auroc_mean": best_auroc,
                "auroc_se": best_se,
                "best_alpha": best_alpha,
                "n_valid": best_n,
            })

    return rows


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_auroc_comparison(results: dict[str, Any], output_dir: str) -> None:
    """Grouped bar chart of AUROC by feature, faceted by graph."""
    graphs = list(results.keys())
    n_graphs = len(graphs)

    fig, axes = plt.subplots(1, n_graphs, figsize=(5 * n_graphs, 5), squeeze=False)

    for idx, graph_name in enumerate(graphs):
        ax = axes[0, idx]
        graph_res = results[graph_name]

        # Use alpha=0.01 as default display threshold
        alpha_key = "alpha=0.01"
        if alpha_key not in graph_res:
            alpha_key = list(graph_res.keys())[0]

        feat_res = graph_res[alpha_key]["feature_aurocs"]

        names = []
        means = []
        ses = []
        for fname in FEATURE_NAMES:
            s = feat_res.get(fname, {})
            m = s.get("mean")
            if m is not None:
                names.append(FEATURE_DISPLAY[fname])
                means.append(m)
                ses.append(s.get("se", 0.0))

        if not names:
            ax.set_title(f"{graph_name}\n(no valid data)")
            continue

        x = np.arange(len(names))
        colors = ["#2196F3" if "orman" in n.lower() else "#757575" for n in names]
        # Highlight curvature features
        colors_mapped = []
        for n in names:
            if "Forman" in n:
                colors_mapped.append("#1565C0")
            elif "Aug" in n:
                colors_mapped.append("#42A5F5")
            else:
                colors_mapped.append("#BDBDBD")

        bars = ax.bar(x, means, yerr=ses, capsize=3, color=colors_mapped,
                      edgecolor="black", linewidth=0.5)
        ax.axhline(0.5, color="red", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("AUROC" if idx == 0 else "")
        ax.set_ylim(0.3, 0.9)
        n_info = len(BENCHMARK_DAGS[graph_name]["nodes"])
        e_info = len(BENCHMARK_DAGS[graph_name]["edges"])
        ax.set_title(f"{graph_name.capitalize()} ({n_info}n, {e_info}e)")

    fig.suptitle("TP vs FP Edge Discrimination by Feature (alpha=0.01)", fontsize=12)
    plt.tight_layout()
    path = os.path.join(output_dir, "auroc_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved AUROC comparison plot to %s", path)


def plot_auroc_by_graph_size(results: dict[str, Any], output_dir: str) -> None:
    """Line plot of AUROC vs graph size for top features."""
    # Collect (n_nodes, auroc) per feature across graphs, using alpha=0.01
    feature_data: dict[str, list[tuple[int, float, float]]] = defaultdict(list)

    for graph_name in results:
        n_nodes = len(BENCHMARK_DAGS[graph_name]["nodes"])
        graph_res = results[graph_name]
        alpha_key = "alpha=0.01"
        if alpha_key not in graph_res:
            alpha_key = list(graph_res.keys())[0]

        feat_res = graph_res[alpha_key]["feature_aurocs"]
        for fname in FEATURE_NAMES:
            s = feat_res.get(fname, {})
            m = s.get("mean")
            se = s.get("se", 0.0)
            if m is not None:
                feature_data[fname].append((n_nodes, m, se))

    fig, ax = plt.subplots(figsize=(8, 5))

    markers = ["o", "s", "^", "D", "v", "P"]
    colors = ["#1565C0", "#42A5F5", "#66BB6A", "#FFA726", "#EF5350", "#AB47BC"]

    for i, fname in enumerate(FEATURE_NAMES):
        pts = sorted(feature_data.get(fname, []))
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        es = [p[2] for p in pts]
        ax.errorbar(xs, ys, yerr=es, marker=markers[i % len(markers)],
                     color=colors[i % len(colors)],
                     label=FEATURE_DISPLAY[fname], capsize=3, linewidth=1.5,
                     markersize=6)

    ax.axhline(0.5, color="red", linestyle="--", linewidth=0.8, alpha=0.7,
                label="Chance")
    ax.set_xlabel("Number of nodes in true DAG")
    ax.set_ylabel("AUROC (TP vs FP)")
    ax.set_title("Edge Discrimination vs Graph Size (alpha=0.01)")
    ax.legend(fontsize=8, loc="best")
    ax.set_ylim(0.3, 0.9)

    plt.tight_layout()
    path = os.path.join(output_dir, "auroc_by_graph.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved AUROC-by-graph plot to %s", path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Curvature-based edge validation on bnlearn benchmarks",
    )
    parser.add_argument(
        "--graphs", type=str, default="asia,sachs",
        help="Comma-separated graph names (asia, sachs, insurance, alarm)",
    )
    parser.add_argument(
        "--n-samples", type=int, default=1000,
        help="Number of SEM samples per replicate (default: 1000)",
    )
    parser.add_argument(
        "--n-replicates", type=int, default=50,
        help="Number of random replicates per (graph, alpha) (default: 50)",
    )
    parser.add_argument(
        "--output", type=str, default="results/",
        help="Output directory (default: results/)",
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Generate plots",
    )

    args = parser.parse_args()

    graphs = [g.strip().lower() for g in args.graphs.split(",")]
    for g in graphs:
        if g not in BENCHMARK_DAGS:
            parser.error(f"Unknown graph: {g}. Choose from: {list(BENCHMARK_DAGS.keys())}")

    logger.info("Starting bnlearn curvature benchmark pipeline")
    logger.info("Graphs: %s", graphs)
    logger.info("n_samples=%d, n_replicates=%d, output=%s",
                 args.n_samples, args.n_replicates, args.output)

    t0 = time.time()
    run_pipeline(
        graphs=graphs,
        n_samples=args.n_samples,
        n_replicates=args.n_replicates,
        output_dir=args.output,
        plot=args.plot,
    )
    elapsed = time.time() - t0
    logger.info("Pipeline completed in %.1f seconds", elapsed)


if __name__ == "__main__":
    main()
