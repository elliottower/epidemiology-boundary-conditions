"""Rescue experiments for batch1 negatives.

Experiment 5: curvature_rescue
  Batch1 ORC failed (AUROC=0.466). Test multiple topological edge features
  (Forman-Ricci, Jaccard, betweenness, partial corr magnitude) on
  linear and nonlinear SEMs across thresholds.

Experiment 6: hte_rescue
  Batch1 PCA-on-KNN-CATE failed (ARI=-0.011). Test better CATE estimators
  (T-learner RF/GBM, S-learner RF vs KNN) and clustering methods
  (direct CATE, spectral, CATE+covariates) for treatment effect subtype recovery.
"""
from __future__ import annotations

import numpy as np
from datetime import datetime
from scipy.stats import mannwhitneyu
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import networkx as nx


# ======================================================================
# SHARED HELPERS (from batch1, kept DRY)
# ======================================================================

def _random_dag(n_nodes, edge_prob, rng):
    adj = (rng.random((n_nodes, n_nodes)) < edge_prob).astype(float)
    return np.triu(adj, k=1)


def _simulate_linear_sem(adj, n_samples, rng, noise_std=1.0):
    n = adj.shape[0]
    W = adj * rng.uniform(0.5, 2.0, adj.shape) * rng.choice([-1, 1], adj.shape)
    X = np.zeros((n_samples, n))
    for i in range(n):
        parents = np.where(W[:, i] != 0)[0]
        if len(parents) > 0:
            X[:, i] = X[:, parents] @ W[parents, i] + rng.normal(0, noise_std, n_samples)
        else:
            X[:, i] = rng.normal(0, noise_std, n_samples)
    return X, W


def _simulate_nonlinear_sem(adj, n_samples, rng, noise_std=1.0):
    n = adj.shape[0]
    W = adj * rng.uniform(0.5, 2.0, adj.shape) * rng.choice([-1, 1], adj.shape)
    X = np.zeros((n_samples, n))
    for i in range(n):
        parents = np.where(W[:, i] != 0)[0]
        if len(parents) > 0:
            linear = X[:, parents] @ W[parents, i]
            X[:, i] = linear + 0.4 * np.sin(2 * linear) + rng.normal(0, noise_std, n_samples)
        else:
            X[:, i] = rng.normal(0, noise_std, n_samples)
    return X, W


def _learn_graph_pcorr(X, threshold=0.12):
    n = X.shape[1]
    cov = np.cov(X.T)
    precision = np.linalg.inv(cov + 1e-6 * np.eye(n))
    d = np.sqrt(np.diag(precision))
    partial_corr = -precision / np.outer(d, d)
    np.fill_diagonal(partial_corr, 0)
    adj = (np.abs(partial_corr) > threshold).astype(float)
    return adj, partial_corr


def _compute_edge_features(G, partial_corr):
    betweenness = nx.edge_betweenness_centrality(G)
    clustering = nx.clustering(G)

    features = {}
    for u, v in G.edges():
        d_u = G.degree(u)
        d_v = G.degree(v)
        common = len(set(G.neighbors(u)) & set(G.neighbors(v)))
        nb_union = len(set(G.neighbors(u)) | set(G.neighbors(v)))

        features[(u, v)] = {
            "forman": 4 - d_u - d_v,
            "forman_aug": 4 - d_u - d_v + 3 * common,
            "jaccard": common / nb_union if nb_union > 0 else 0,
            "betweenness": betweenness.get((u, v), betweenness.get((v, u), 0)),
            "pcorr_mag": abs(partial_corr[u, v]),
            "avg_clustering": (clustering[u] + clustering[v]) / 2,
        }
    return features


# ======================================================================
# 5. CURVATURE RESCUE
# ======================================================================

def run_curvature_rescue(seed=42):
    """Test multiple topological features for TP vs FP edge discrimination.

    Batch1 ORC got AUROC=0.466. We test cheaper combinatorial features:
    - Forman-Ricci: 4 - d(u) - d(v)
    - Augmented Forman: + 3 * common_neighbors
    - Jaccard coefficient (neighbor overlap)
    - Edge betweenness centrality
    - Partial correlation magnitude (non-topological baseline)
    - Average endpoint clustering coefficient

    Across conditions:
    - Linear vs nonlinear SEM
    - Loose (0.10) vs moderate (0.15) vs strict (0.25) threshold
    """
    print(f"[{datetime.now():%H:%M:%S}] curvature_rescue: starting")
    rng = np.random.default_rng(seed)

    n_graphs = 60
    node_counts = [10, 15, 20]
    n_samples = 2000

    feature_names = [
        "forman", "forman_aug", "jaccard",
        "betweenness", "pcorr_mag", "avg_clustering",
    ]

    conditions = [
        ("linear", "loose", _simulate_linear_sem, 0.10),
        ("linear", "moderate", _simulate_linear_sem, 0.15),
        ("linear", "strict", _simulate_linear_sem, 0.25),
        ("nonlinear", "loose", _simulate_nonlinear_sem, 0.10),
        ("nonlinear", "moderate", _simulate_nonlinear_sem, 0.15),
        ("nonlinear", "strict", _simulate_nonlinear_sem, 0.25),
    ]

    results_by_condition = []

    for sem_type, thresh_name, sim_fn, threshold in conditions:
        print(f"\n  Condition: {sem_type} SEM, threshold={threshold} ({thresh_name})")

        tp_by_feature = {f: [] for f in feature_names}
        fp_by_feature = {f: [] for f in feature_names}
        n_used = 0
        total_tp = 0
        total_fp = 0

        for g_idx in tqdm(range(n_graphs), desc=f"    {sem_type}/{thresh_name}"):
            n_nodes = int(rng.choice(node_counts))
            edge_prob = float(rng.uniform(0.15, 0.35))

            true_adj = _random_dag(n_nodes, edge_prob, rng)
            true_skeleton = ((true_adj + true_adj.T) > 0).astype(float)
            n_true_edges = int(np.sum(true_skeleton) / 2)
            if n_true_edges < 3:
                continue

            X, W = sim_fn(true_adj, n_samples, rng)
            learned_adj, pcorr = _learn_graph_pcorr(X, threshold=threshold)

            G = nx.Graph()
            G.add_nodes_from(range(n_nodes))
            for i in range(n_nodes):
                for j in range(i + 1, n_nodes):
                    if learned_adj[i, j] > 0:
                        G.add_edge(i, j)

            if G.number_of_edges() < 5:
                continue

            edge_feats = _compute_edge_features(G, pcorr)

            for (u, v), feats in edge_feats.items():
                is_tp = true_skeleton[u, v] > 0 or true_skeleton[v, u] > 0
                bucket = tp_by_feature if is_tp else fp_by_feature
                for fname in feature_names:
                    bucket[fname].append(feats[fname])
                if is_tp:
                    total_tp += 1
                else:
                    total_fp += 1

            n_used += 1

        condition_results = {
            "sem_type": sem_type,
            "threshold": threshold,
            "threshold_name": thresh_name,
            "n_graphs": n_used,
            "total_tp": total_tp,
            "total_fp": total_fp,
            "features": {},
        }

        for fname in feature_names:
            tp = np.array(tp_by_feature[fname])
            fp = np.array(fp_by_feature[fname])

            if len(tp) < 10 or len(fp) < 10:
                condition_results["features"][fname] = {
                    "auroc": None, "n_tp": len(tp), "n_fp": len(fp),
                }
                continue

            U, p = mannwhitneyu(tp, fp, alternative="two-sided")
            auroc = float(U / (len(tp) * len(fp)))

            condition_results["features"][fname] = {
                "auroc": auroc,
                "tp_mean": float(np.mean(tp)),
                "fp_mean": float(np.mean(fp)),
                "tp_std": float(np.std(tp)),
                "fp_std": float(np.std(fp)),
                "n_tp": len(tp),
                "n_fp": len(fp),
                "mann_whitney_p": float(p),
                "direction": "TP > FP" if np.mean(tp) > np.mean(fp) else "FP > TP",
            }

            marker = "+" if auroc > 0.55 else ("~" if auroc > 0.52 else "-")
            print(
                f"      {fname:18s}: AUROC={auroc:.3f}  "
                f"TP={np.mean(tp):.3f}+/-{np.std(tp):.3f}  "
                f"FP={np.mean(fp):.3f}+/-{np.std(fp):.3f}  [{marker}]"
            )

        results_by_condition.append(condition_results)

    summary = []
    for cond in results_by_condition:
        best_feat = None
        best_auroc = 0.5
        for fname, fdata in cond["features"].items():
            if fdata.get("auroc") is not None:
                effective = max(fdata["auroc"], 1 - fdata["auroc"])
                if effective > best_auroc:
                    best_auroc = effective
                    best_feat = fname
        summary.append({
            "condition": f"{cond['sem_type']}/{cond['threshold_name']}",
            "best_feature": best_feat,
            "best_auroc": best_auroc,
        })
        print(f"\n    Best for {cond['sem_type']}/{cond['threshold_name']}: "
              f"{best_feat} (AUROC={best_auroc:.3f})")

    result = {
        "conditions": results_by_condition,
        "summary": summary,
        "batch1_orc_auroc": 0.466,
    }
    print(f"\n[{datetime.now():%H:%M:%S}] curvature_rescue: done")
    return result


# ======================================================================
# 6. HTE RESCUE
# ======================================================================

def _simulate_ms_trial(n, n_covariates, rng):
    subtype_probs = [0.40, 0.40, 0.20]
    subtypes = rng.choice(3, size=n, p=subtype_probs)

    covariates = rng.normal(0, 1, (n, n_covariates))
    covariates[subtypes == 0, 0] -= 0.8
    covariates[subtypes == 0, 1] -= 0.5
    covariates[subtypes == 0, 3] += 0.6
    covariates[subtypes == 1, 0] += 0.6
    covariates[subtypes == 1, 1] += 0.4
    covariates[subtypes == 1, 4] += 0.5
    covariates[subtypes == 2, 2] += 0.7

    propensity = 0.5 + 0.1 * covariates[:, 0] - 0.05 * covariates[:, 1]
    propensity = np.clip(propensity, 0.15, 0.85)
    treatment = rng.binomial(1, propensity)

    effect = np.zeros(n)
    effect[subtypes == 0] = 2.0 * treatment[subtypes == 0]
    effect[subtypes == 1] = (
        -1.0 * treatment[subtypes == 1]
        + 1.5 * (1 - treatment[subtypes == 1])
    )

    baseline = (
        0.5 * covariates[:, 1]
        + 0.3 * covariates[:, 0]
        + rng.normal(0, 0.5, n)
    )
    outcome = baseline + effect + rng.normal(0, 1.5, n)

    true_cate = np.zeros(n)
    true_cate[subtypes == 0] = 2.0
    true_cate[subtypes == 1] = -2.5
    true_cate[subtypes == 2] = 0.0

    return covariates, treatment, outcome, subtypes, true_cate


def _cate_knn(covs, trt, out):
    knn1 = KNeighborsRegressor(n_neighbors=30).fit(covs[trt == 1], out[trt == 1])
    knn0 = KNeighborsRegressor(n_neighbors=30).fit(covs[trt == 0], out[trt == 0])
    return knn1.predict(covs) - knn0.predict(covs)


def _cate_rf_tlearner(covs, trt, out):
    rf1 = RandomForestRegressor(
        n_estimators=200, max_depth=10, random_state=0,
    ).fit(covs[trt == 1], out[trt == 1])
    rf0 = RandomForestRegressor(
        n_estimators=200, max_depth=10, random_state=0,
    ).fit(covs[trt == 0], out[trt == 0])
    return rf1.predict(covs) - rf0.predict(covs)


def _cate_gbm_tlearner(covs, trt, out):
    gb1 = GradientBoostingRegressor(
        n_estimators=200, max_depth=4, random_state=0,
    ).fit(covs[trt == 1], out[trt == 1])
    gb0 = GradientBoostingRegressor(
        n_estimators=200, max_depth=4, random_state=0,
    ).fit(covs[trt == 0], out[trt == 0])
    return gb1.predict(covs) - gb0.predict(covs)


def _cate_rf_slearner(covs, trt, out):
    X = np.column_stack([covs, trt])
    rf = RandomForestRegressor(
        n_estimators=200, max_depth=10, random_state=0,
    ).fit(X, out)
    X1 = np.column_stack([covs, np.ones(len(covs))])
    X0 = np.column_stack([covs, np.zeros(len(covs))])
    return rf.predict(X1) - rf.predict(X0)


def _cluster_pca_kmeans(covs, cate_hat):
    weighted = covs * cate_hat[:, np.newaxis]
    pca = PCA(n_components=3).fit(weighted)
    proj = pca.transform(weighted)
    return KMeans(n_clusters=3, n_init=10, random_state=0).fit_predict(proj[:, :2])


def _cluster_cate_kmeans(covs, cate_hat):
    return KMeans(
        n_clusters=3, n_init=10, random_state=0,
    ).fit_predict(cate_hat.reshape(-1, 1))


def _cluster_cate_cov(covs, cate_hat):
    combined = np.column_stack([covs, cate_hat])
    combined = StandardScaler().fit_transform(combined)
    return KMeans(n_clusters=3, n_init=10, random_state=0).fit_predict(combined)


def _cluster_spectral(covs, cate_hat):
    cate_scaled = (cate_hat - cate_hat.mean()) / (cate_hat.std() + 1e-8)
    n = len(cate_hat)
    # Subsample for spectral (eigendecomp is O(n^3))
    if n > 1000:
        idx = np.linspace(0, n - 1, 1000, dtype=int)
        sub_cate = cate_scaled[idx]
    else:
        idx = np.arange(n)
        sub_cate = cate_scaled

    gamma = 1.0 / (2 * sub_cate.std() ** 2 + 1e-8)
    affinity = rbf_kernel(sub_cate.reshape(-1, 1), gamma=gamma)

    labels_sub = SpectralClustering(
        n_clusters=3, affinity="precomputed", random_state=0,
    ).fit_predict(affinity)

    if n > 1000:
        knn = KNeighborsClassifier(n_neighbors=5).fit(
            sub_cate.reshape(-1, 1), labels_sub,
        )
        return knn.predict(cate_scaled.reshape(-1, 1))
    return labels_sub


CATE_METHODS = [
    ("knn", _cate_knn),
    ("rf_tlearner", _cate_rf_tlearner),
    ("gbm_tlearner", _cate_gbm_tlearner),
    ("rf_slearner", _cate_rf_slearner),
]

CLUSTER_METHODS = [
    ("pca_kmeans", _cluster_pca_kmeans),
    ("cate_kmeans", _cluster_cate_kmeans),
    ("cate_cov_kmeans", _cluster_cate_cov),
    ("spectral_cate", _cluster_spectral),
]


def run_hte_rescue(seed=42):
    """Rescue treatment heterogeneity detection with better CATE estimation.

    Batch1 PCA-on-KNN-CATE got ARI=-0.011. We test:

    CATE estimators:
    - KNN (batch1 baseline)
    - T-learner with RandomForest
    - T-learner with GradientBoosting
    - S-learner with RandomForest (treatment as feature — handles confounding)

    Clustering methods:
    - PCA on cov*CATE + K-means (batch1 baseline)
    - K-means on CATE directly
    - K-means on [covariates, CATE] concatenated
    - Spectral clustering on CATE RBF kernel

    Plus oracle (true CATE) and naive (raw covariate K-means) baselines.
    """
    print(f"[{datetime.now():%H:%M:%S}] hte_rescue: starting")
    rng = np.random.default_rng(seed)

    n_patients = 3000
    n_covariates = 10
    n_reps = 100

    all_results = {}

    for cate_name, cate_fn in CATE_METHODS:
        for clust_name, clust_fn in CLUSTER_METHODS:
            combo = f"{cate_name}/{clust_name}"
            print(f"\n  Testing: {combo}")
            aris = []

            for _ in tqdm(range(n_reps), desc=f"    {combo}"):
                covs, trt, out, sub, true_cate = _simulate_ms_trial(
                    n_patients, n_covariates, rng,
                )
                cate_hat = cate_fn(covs, trt, out)
                clusters = clust_fn(covs, cate_hat)
                aris.append(adjusted_rand_score(sub, clusters))

            all_results[combo] = {
                "ari_mean": float(np.mean(aris)),
                "ari_std": float(np.std(aris)),
                "ari_median": float(np.median(aris)),
                "ari_q25": float(np.percentile(aris, 25)),
                "ari_q75": float(np.percentile(aris, 75)),
            }
            r = all_results[combo]
            print(f"    ARI = {r['ari_mean']:.3f} +/- {r['ari_std']:.3f} "
                  f"(median={r['ari_median']:.3f})")

    # Oracle: clustering on true CATE (upper bound)
    print("\n  Oracle: clustering on true CATE")
    oracle_results = {}
    for clust_name, clust_fn in CLUSTER_METHODS:
        aris = []
        for _ in tqdm(range(n_reps), desc=f"    oracle/{clust_name}"):
            covs, trt, out, sub, true_cate = _simulate_ms_trial(
                n_patients, n_covariates, rng,
            )
            clusters = clust_fn(covs, true_cate)
            aris.append(adjusted_rand_score(sub, clusters))

        combo = f"oracle/{clust_name}"
        oracle_results[combo] = {
            "ari_mean": float(np.mean(aris)),
            "ari_std": float(np.std(aris)),
        }
        r = oracle_results[combo]
        print(f"    {combo}: ARI = {r['ari_mean']:.3f} +/- {r['ari_std']:.3f}")

    # Naive baseline
    print("\n  Naive baseline: K-means on raw covariates")
    naive_aris = []
    for _ in tqdm(range(n_reps), desc="    naive"):
        covs, trt, out, sub, true_cate = _simulate_ms_trial(
            n_patients, n_covariates, rng,
        )
        clusters = KMeans(
            n_clusters=3, n_init=10, random_state=0,
        ).fit_predict(covs)
        naive_aris.append(adjusted_rand_score(sub, clusters))

    naive_result = {
        "ari_mean": float(np.mean(naive_aris)),
        "ari_std": float(np.std(naive_aris)),
    }
    print(f"    Naive ARI = {naive_result['ari_mean']:.3f} "
          f"+/- {naive_result['ari_std']:.3f}")

    # Summary table
    print("\n  === SUMMARY ===")
    print(f"  {'Method':<30s} {'ARI':>8s}")
    print(f"  {'-' * 40}")
    print(f"  {'naive (raw cov K-means)':<30s} "
          f"{naive_result['ari_mean']:>8.3f}")
    baseline_ari = all_results.get("knn/pca_kmeans", {}).get("ari_mean")
    if baseline_ari is not None:
        print(f"  {'batch1 (knn/pca_kmeans)':<30s} {baseline_ari:>8.3f}")
    for combo in sorted(all_results.keys()):
        if combo != "knn/pca_kmeans":
            print(f"  {combo:<30s} {all_results[combo]['ari_mean']:>8.3f}")
    for combo in sorted(oracle_results.keys()):
        print(f"  {combo:<30s} {oracle_results[combo]['ari_mean']:>8.3f}")

    result = {
        "estimated_cate_results": all_results,
        "oracle_results": oracle_results,
        "naive_baseline": naive_result,
        "batch1_reference": {
            "knn_pca_kmeans_ari": -0.011,
            "naive_ari": 0.219,
        },
        "n_patients": n_patients,
        "n_reps": n_reps,
    }
    print(f"\n[{datetime.now():%H:%M:%S}] hte_rescue: done")
    return result
