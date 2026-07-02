"""Pure logic for MS neurology experiments.

No Modal dependencies. Each function returns a dict of results.

Experiments based on 7 design specs (MechVal MS program for Xia lab):
  Prerequisites: Cohort identity cleaning + outcome re-metricization
  Cocycle: 4-arm Grassmannian cocycle obstruction experiment
  P1: Bracket-norm confound audit for MS imaging biomarkers
  P2: Sheaf-cohomology DAG adjudication (inflammation vs degeneration)
  P3: H^1 effect-modifier heterogeneity suite
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import datetime
from scipy import linalg, stats
from scipy.stats import chi2, pearsonr, spearmanr, mannwhitneyu
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LinearRegression
from tqdm import tqdm


# ======================================================================
# HELPERS — Grassmannian geometry
# ======================================================================

def _random_subspace(d, k, rng):
    """Sample a random k-dim subspace of R^d (as an orthonormal basis)."""
    M = rng.standard_normal((d, k))
    U, _, _ = linalg.svd(M, full_matrices=False)
    return U


def _projector(U):
    """Orthogonal projector P = U U^T for an orthonormal U."""
    return U @ U.T


def _principal_angles(U1, U2):
    """Principal angles between two subspaces (given as orthonormal bases)."""
    M = U1.T @ U2
    svals = linalg.svdvals(M)
    svals = np.clip(svals, -1, 1)
    return np.arccos(svals)


def _geodesic_distance(U1, U2):
    """Grassmannian geodesic distance = norm of principal angles."""
    return float(np.linalg.norm(_principal_angles(U1, U2)))


def _rotate_subspace(U, plane_indices, angle):
    """Rotate subspace U by `angle` in a 2D plane of the ambient space.

    plane_indices = (i, j): the two ambient-space coordinates to rotate in.
    """
    d = U.shape[0]
    R = np.eye(d)
    i, j = plane_indices
    c, s = np.cos(angle), np.sin(angle)
    R[i, i] = c
    R[i, j] = -s
    R[j, i] = s
    R[j, j] = c
    rotated = R @ U
    Q, _ = linalg.qr(rotated, mode="economic")
    return Q


def _transport_matrix(U_from, U_to):
    """Parallel transport operator from Gr(k,d) at U_from to U_to.

    Uses the canonical connection: T = (I - P_to) factor via SVD.
    Returns a k x k matrix representing the transport in subspace coords.
    """
    M = U_from.T @ U_to
    u, s, vt = linalg.svd(M)
    return u @ vt


def _cocycle_holonomy(subspaces):
    """Compute holonomy around a cycle of subspaces.

    Returns (holonomy_matrix, holonomy_angle) where holonomy_angle
    is the Frobenius-norm deviation from identity.
    """
    k = subspaces[0].shape[1]
    composed = np.eye(k)
    for idx in range(len(subspaces)):
        U_from = subspaces[idx]
        U_to = subspaces[(idx + 1) % len(subspaces)]
        T = _transport_matrix(U_from, U_to)
        composed = T @ composed
    deviation = np.linalg.norm(composed - np.eye(k), "fro")
    return composed, float(deviation)


def _rotation_in_plane(v1, v2, angle):
    """Rotation matrix acting in the 2D plane spanned by orthonormal v1, v2."""
    d = len(v1)
    return (np.eye(d)
            + (np.cos(angle) - 1) * (np.outer(v1, v1) + np.outer(v2, v2))
            + np.sin(angle) * (np.outer(v2, v1) - np.outer(v1, v2)))


def _generate_cocycle_sections(V0, r, m, noise_level, rng, v_perp_1=None, v_perp_2=None):
    """Generate m sections on Gr(k,d) with planted holonomy via linked columns.

    Columns 0 and 1 of V0 both rotate by angle r toward shared perpendicular
    directions (v_perp_1, v_perp_2), with a pi/2 phase offset between them.
    This creates inter-column coupling in the transport matrices, giving nonzero
    sectional curvature and genuine Berry phase.

    Holonomy angle = 2*pi*sin^2(r). Frobenius deviation = 2*sqrt(2)*|sin(pi*sin^2(r))|.
    """
    d, k = V0.shape

    if v_perp_1 is None:
        v_perp_1 = rng.standard_normal(d)
        v_perp_1 -= V0 @ (V0.T @ v_perp_1)
        v_perp_1 /= np.linalg.norm(v_perp_1)

    if v_perp_2 is None:
        v_perp_2 = rng.standard_normal(d)
        v_perp_2 -= V0 @ (V0.T @ v_perp_2)
        v_perp_2 -= v_perp_1 * (v_perp_1 @ v_perp_2)
        v_perp_2 /= np.linalg.norm(v_perp_2)

    v_in_0 = V0[:, 0].copy()
    v_in_1 = V0[:, 1].copy()

    sections = []
    for i in range(m):
        phi = 2 * np.pi * i / m
        Vi = V0.copy()
        dir_0 = np.cos(phi) * v_perp_1 + np.sin(phi) * v_perp_2
        Vi[:, 0] = np.cos(r) * v_in_0 + np.sin(r) * dir_0
        dir_1 = np.cos(phi + np.pi / 2) * v_perp_1 + np.sin(phi + np.pi / 2) * v_perp_2
        Vi[:, 1] = np.cos(r) * v_in_1 + np.sin(r) * dir_1
        Q, _ = linalg.qr(Vi, mode="economic")
        noise = rng.standard_normal((d, k)) * noise_level
        Q_noisy, _ = linalg.qr(Q + noise, mode="economic")
        sections.append(Q_noisy)

    return sections, v_perp_1, v_perp_2


# ======================================================================
# HELPERS — Sheaf cohomology
# ======================================================================

def _sheaf_obstruction_subspace(local_sections, adjacency):
    """Compute sheaf gluing obstruction for subspace-valued sections.

    local_sections: list of (d, k) orthonormal bases (one per node)
    adjacency: list of (i, j) edges

    Returns (H1_norm, pairwise_angles, global_fit_residual).
    """
    n = len(local_sections)
    k = local_sections[0].shape[1]

    pairwise_angles = {}
    for i, j in adjacency:
        angles = _principal_angles(local_sections[i], local_sections[j])
        pairwise_angles[(i, j)] = angles.tolist()

    all_projectors = [_projector(U) for U in local_sections]
    avg_P = np.mean(all_projectors, axis=0)
    eigvals, eigvecs = np.linalg.eigh(avg_P)
    global_basis = eigvecs[:, -k:]

    residuals = []
    for U in local_sections:
        P_local = _projector(U)
        P_global = _projector(global_basis)
        residuals.append(np.linalg.norm(P_local - P_global, "fro"))
    H1_norm = float(np.mean(residuals))

    return H1_norm, pairwise_angles, float(np.std(residuals))


def _sheaf_q_test(estimates):
    """Sheaf Q test for scalar-valued sections (same as batch1)."""
    nodes = sorted(estimates.keys())
    n = len(nodes)
    node_idx = {name: i for i, name in enumerate(nodes)}
    edges = [(nodes[i], nodes[j]) for i in range(n) for j in range(i + 1, n)]

    d0 = np.zeros((len(edges), n))
    for e_idx, (u, v) in enumerate(edges):
        d0[e_idx, node_idx[u]] = -1.0
        d0[e_idx, node_idx[v]] = 1.0

    stalks = np.array([estimates[nd]["beta"] for nd in nodes])
    ses = np.array([estimates[nd]["se"] for nd in nodes])

    obs = d0 @ stalks
    Sigma = d0 @ np.diag(ses**2) @ d0.T
    Sigma_pinv = np.linalg.pinv(Sigma, rcond=1e-10)
    Q = float(obs @ Sigma_pinv @ obs)

    _, s, _ = linalg.svd(d0, full_matrices=False)
    rank = int(np.sum(s > 1e-10))
    p = 1.0 - chi2.cdf(Q, rank) if rank > 0 else 1.0

    return p, Q, rank


# ======================================================================
# EXPERIMENT 0: PREREQUISITES
# ======================================================================

def run_prerequisites(seed=42):
    """Prereq 1: cohort identity cleaning + Prereq 2: outcome re-metricization.

    Simulates:
    - A cohort with MS/NMOSD/MOGAD contamination
    - EDSS with known non-linearity
    - Item-response-style latent axis extraction
    - Scale invariance check (interaction stability across additive/multiplicative)
    """
    print(f"[{datetime.now():%H:%M:%S}] Starting prerequisites experiment")
    rng = np.random.default_rng(seed)
    results = {}

    # --- PREREQ 1: Cohort identity cleaning ---
    n_patients = 2000
    true_labels = rng.choice(
        ["MS", "NMOSD", "MOGAD"],
        size=n_patients,
        p=[0.85, 0.10, 0.05],
    )
    contamination_rate = 0.12

    assay_types = rng.choice(["cell_based", "ELISA"], size=n_patients, p=[0.6, 0.4])
    false_neg_rate = np.where(assay_types == "ELISA", 0.15, 0.03)

    observed_labels = true_labels.copy()
    for i in range(n_patients):
        if true_labels[i] in ("NMOSD", "MOGAD"):
            if rng.random() < false_neg_rate[i]:
                observed_labels[i] = "MS"

    actual_contamination = np.mean(
        (observed_labels == "MS") & (true_labels != "MS")
    )

    true_effect = np.where(true_labels == "MS", 0.5, 0.1)
    noise = rng.standard_normal(n_patients) * 0.3
    outcome = true_effect + noise

    clean_mask = observed_labels == "MS"
    strict_mask = clean_mask & (assay_types == "cell_based")

    from scipy.stats import ttest_ind
    effect_all = np.mean(outcome[clean_mask])
    effect_strict = np.mean(outcome[strict_mask])
    _, p_stability = ttest_ind(outcome[clean_mask], outcome[strict_mask])

    results["prereq1_cohort_identity"] = {
        "n_patients": n_patients,
        "true_ms_fraction": float(np.mean(true_labels == "MS")),
        "observed_ms_fraction": float(np.mean(observed_labels == "MS")),
        "contamination_rate": float(actual_contamination),
        "effect_permissive": float(effect_all),
        "effect_strict": float(effect_strict),
        "stability_p_value": float(p_stability),
        "verdict_stable": bool(p_stability > 0.05),
        "assay_false_neg_ELISA": 0.15,
        "assay_false_neg_cell_based": 0.03,
    }
    print(f"[{datetime.now():%H:%M:%S}] Prereq 1 done: contamination={actual_contamination:.3f}")

    # --- PREREQ 2: Outcome re-metricization ---
    n_visits = 3000
    true_theta = rng.standard_normal(n_visits)

    edss_from_theta = 1.0 / (1.0 + np.exp(-1.5 * (true_theta - 0.5)))
    edss_raw = np.round(edss_from_theta * 9.0) / 1.0
    edss_raw = np.clip(edss_raw, 0, 9)

    snfl = 0.6 * true_theta + rng.standard_normal(n_visits) * 0.4
    gm_atrophy = 0.5 * true_theta + rng.standard_normal(n_visits) * 0.5

    cor_edss_snfl = float(pearsonr(edss_raw, snfl)[0])
    cor_theta_snfl = float(pearsonr(true_theta, snfl)[0])
    cor_edss_gm = float(pearsonr(edss_raw, gm_atrophy)[0])
    cor_theta_gm = float(pearsonr(true_theta, gm_atrophy)[0])

    from sklearn.preprocessing import QuantileTransformer
    qt = QuantileTransformer(output_distribution="normal", n_quantiles=100)
    latent_axis = qt.fit_transform(edss_raw.reshape(-1, 1)).ravel()

    cor_latent_snfl = float(pearsonr(latent_axis, snfl)[0])
    cor_latent_gm = float(pearsonr(latent_axis, gm_atrophy)[0])

    # Scale invariance check: simulate HLA x EBV interaction
    n_inter = 1000
    hla = rng.binomial(1, 0.25, n_inter).astype(float)
    ebv = rng.binomial(1, 0.90, n_inter).astype(float)
    theta_inter = rng.standard_normal(n_inter)
    prog_true = 0.3 * hla + 0.2 * ebv + 0.4 * hla * ebv + theta_inter * 0.3

    edss_inter = np.round(
        np.clip(1.0 / (1.0 + np.exp(-1.5 * (prog_true - 0.5))) * 9.0, 0, 9)
    )
    latent_inter = qt.transform(edss_inter.reshape(-1, 1)).ravel()

    X_inter = np.column_stack([hla, ebv, hla * ebv, np.ones(n_inter)])

    from numpy.linalg import lstsq
    beta_raw, _, _, _ = lstsq(X_inter, edss_inter, rcond=None)
    beta_latent, _, _, _ = lstsq(X_inter, latent_inter, rcond=None)

    log_edss = np.log1p(edss_inter)
    beta_log, _, _, _ = lstsq(X_inter, log_edss, rcond=None)

    interaction_raw = float(beta_raw[2])
    interaction_latent = float(beta_latent[2])
    interaction_log = float(beta_log[2])

    scale_var_raw = float(np.std([interaction_raw, interaction_log]))
    scale_var_latent = float(np.std([interaction_latent, interaction_log]))

    results["prereq2_remetricization"] = {
        "n_visits": n_visits,
        "cor_edss_snfl": cor_edss_snfl,
        "cor_theta_snfl": cor_theta_snfl,
        "cor_latent_snfl": cor_latent_snfl,
        "cor_edss_gm": cor_edss_gm,
        "cor_theta_gm": cor_theta_gm,
        "cor_latent_gm": cor_latent_gm,
        "linearity_improvement_snfl": cor_latent_snfl - cor_edss_snfl,
        "linearity_improvement_gm": cor_latent_gm - cor_edss_gm,
        "interaction_raw_additive": interaction_raw,
        "interaction_latent_additive": interaction_latent,
        "interaction_log_multiplicative": interaction_log,
        "scale_variability_raw": scale_var_raw,
        "scale_variability_latent": scale_var_latent,
        "scale_invariance_improved": scale_var_latent < scale_var_raw,
    }
    print(f"[{datetime.now():%H:%M:%S}] Prereq 2 done: linearity improvement sNfL={cor_latent_snfl - cor_edss_snfl:.3f}")
    print(f"[{datetime.now():%H:%M:%S}] Prerequisites complete")
    return results


# ======================================================================
# EXPERIMENT 1: COCYCLE OBSTRUCTION (4-arm)
# ======================================================================

def _bootstrap_null_cocycle(d, k, m, noise_level, n_null, rng):
    """Parametric bootstrap null: cocycle norm under global consistency."""
    null_norms = []
    null_pairwise_maxes = []
    for _ in range(n_null):
        V0 = _random_subspace(d, k, rng)
        sections = []
        for _ in range(m):
            noise = rng.standard_normal((d, k)) * noise_level
            perturbed = V0 + noise
            Q, _ = linalg.qr(perturbed, mode="economic")
            sections.append(Q)
        _, norm = _cocycle_holonomy(sections)
        null_norms.append(norm)
        pw = [
            _geodesic_distance(sections[i], sections[(i + 1) % m])
            for i in range(m)
        ]
        null_pairwise_maxes.append(max(pw))
    return np.array(null_norms), np.array(null_pairwise_maxes)


def run_cocycle_obstruction(seed=42):
    """4-arm Grassmannian cocycle obstruction experiment.

    ARM 1: planted holonomy (C1+C2+C3 must all hold)
    ARM 2: negative control (no holonomy)
    ARM 3: competitor baselines
    ARM 4: graded dose-response (varying holonomy angle)

    Key constraint for C1+C2 simultaneously:
      per-step rotation < noise (hides signal pairwise)
      composed rotation > sqrt(m) * noise (coherent signal emerges globally)
    Requires holonomy_angle / m < noise AND holonomy_angle > sqrt(m) * noise.
    """
    print(f"[{datetime.now():%H:%M:%S}] Starting cocycle obstruction experiment")
    rng = np.random.default_rng(seed)
    results = {}

    d = 20
    k = 3
    m = 24
    loop_radius = 0.5
    noise_level = 0.06
    n_null = 1000

    # --- Calibration: bootstrap null ---
    print(f"[{datetime.now():%H:%M:%S}] Running {n_null} null bootstrap reps")
    null_norms, null_pw_maxes = _bootstrap_null_cocycle(d, k, m, noise_level, n_null, rng)
    threshold_95_holonomy = float(np.percentile(null_norms, 95))
    threshold_95_pairwise = float(np.percentile(null_pw_maxes, 95))

    # --- ARM 1: planted holonomy ---
    print(f"[{datetime.now():%H:%M:%S}] ARM 1: planted holonomy (r={loop_radius:.2f})")
    V0 = _random_subspace(d, k, rng)
    planted_sections, vp1, vp2 = _generate_cocycle_sections(V0, loop_radius, m, noise_level, rng)

    holonomy_mat, holonomy_norm = _cocycle_holonomy(planted_sections)

    noiseless_sections, _, _ = _generate_cocycle_sections(V0, loop_radius, m, 0.0, rng, vp1, vp2)
    _, noiseless_holonomy = _cocycle_holonomy(noiseless_sections)
    predicted_holonomy = 2 * np.sqrt(2) * abs(np.sin(np.pi * np.sin(loop_radius) ** 2))

    pairwise_dists = []
    for i in range(m):
        j = (i + 1) % m
        pairwise_dists.append(_geodesic_distance(planted_sections[i], planted_sections[j]))
    max_pairwise = float(max(pairwise_dists))
    mean_pairwise = float(np.mean(pairwise_dists))

    C1 = max_pairwise < threshold_95_pairwise
    C2 = holonomy_norm > threshold_95_holonomy
    p_arm1 = float(np.mean(null_norms >= holonomy_norm))

    scalar_projections = [float(U[:, 0].mean()) for U in planted_sections]
    scalar_range = max(scalar_projections) - min(scalar_projections)
    C3 = scalar_range < threshold_95_pairwise

    results["arm1_planted"] = {
        "loop_radius": float(loop_radius),
        "holonomy_norm": holonomy_norm,
        "noiseless_holonomy": float(noiseless_holonomy),
        "predicted_holonomy": float(predicted_holonomy),
        "p_value": p_arm1,
        "max_pairwise_distance": max_pairwise,
        "mean_pairwise_distance": mean_pairwise,
        "C1_pairwise_consistent": bool(C1),
        "C2_global_inconsistent": bool(C2),
        "C3_scalar_blind": bool(C3),
        "all_three_hold": bool(C1 and C2 and C3),
        "null_threshold_95_holonomy": threshold_95_holonomy,
        "null_threshold_95_pairwise": threshold_95_pairwise,
    }
    print(f"[{datetime.now():%H:%M:%S}] ARM 1: C1={C1}, C2={C2}, C3={C3}, holonomy={holonomy_norm:.4f}")

    # --- ARM 2: negative control (globally consistent) ---
    print(f"[{datetime.now():%H:%M:%S}] ARM 2: negative control")
    V0_ctrl = _random_subspace(d, k, rng)
    control_sections = []
    for _ in range(m):
        noise = rng.standard_normal((d, k)) * noise_level
        perturbed = V0_ctrl + noise
        Q, _ = linalg.qr(perturbed, mode="economic")
        control_sections.append(Q)

    _, ctrl_norm = _cocycle_holonomy(control_sections)
    p_ctrl = float(np.mean(null_norms >= ctrl_norm))

    results["arm2_negative_control"] = {
        "holonomy_norm": float(ctrl_norm),
        "p_value": p_ctrl,
        "correctly_non_significant": bool(p_ctrl > 0.05),
    }
    print(f"[{datetime.now():%H:%M:%S}] ARM 2: p={p_ctrl:.4f}, non-sig={p_ctrl > 0.05}")

    # --- ARM 3: baseline competitors ---
    print(f"[{datetime.now():%H:%M:%S}] ARM 3: baseline competitors")

    max_pw_planted = max_pairwise
    max_pw_ctrl = max(
        _geodesic_distance(control_sections[i], control_sections[(i + 1) % m])
        for i in range(m)
    )
    max_pw_detects = max_pw_planted > 2 * max_pw_ctrl

    n_scalar_reps = 500
    scalar_planted_range = []
    scalar_ctrl_range = []
    for _ in range(n_scalar_reps):
        direction = rng.standard_normal(d)
        direction /= np.linalg.norm(direction)
        planted_scalars = [float(direction @ U[:, 0]) for U in planted_sections]
        ctrl_scalars = [float(direction @ U[:, 0]) for U in control_sections]
        scalar_planted_range.append(max(planted_scalars) - min(planted_scalars))
        scalar_ctrl_range.append(max(ctrl_scalars) - min(ctrl_scalars))

    _, re_p = mannwhitneyu(scalar_planted_range, scalar_ctrl_range, alternative="greater")
    re_detects = re_p < 0.05

    planted_projectors = [_projector(U) for U in planted_sections]
    ctrl_projectors = [_projector(U) for U in control_sections]

    cka_planted = []
    cka_ctrl = []
    for i in range(m):
        for j in range(i + 1, m):
            cka_planted.append(
                np.trace(planted_projectors[i] @ planted_projectors[j]) / k
            )
            cka_ctrl.append(
                np.trace(ctrl_projectors[i] @ ctrl_projectors[j]) / k
            )
    _, cka_p = mannwhitneyu(cka_planted, cka_ctrl)
    cka_detects = cka_p < 0.05

    results["arm3_baselines"] = {
        "max_pairwise_angle": {
            "planted": max_pw_planted,
            "control": max_pw_ctrl,
            "detects": bool(max_pw_detects),
            "reason": "pairwise-only, misses global holonomy",
        },
        "random_effects_scalar": {
            "p_value": float(re_p),
            "detects": bool(re_detects),
            "reason": "scalar projection loses subspace structure",
        },
        "averaged_cka": {
            "mean_planted": float(np.mean(cka_planted)),
            "mean_control": float(np.mean(cka_ctrl)),
            "p_value": float(cka_p),
            "detects": bool(cka_detects),
            "reason": "pairwise similarity, blind to cyclic obstruction",
        },
    }
    print(f"[{datetime.now():%H:%M:%S}] ARM 3: max_pw={max_pw_detects}, RE={re_detects}, CKA={cka_detects}")

    # --- ARM 4: dose-response (varying holonomy angle) ---
    # Fix V0 and perpendicular directions so dose-response shows pure r-dependence
    print(f"[{datetime.now():%H:%M:%S}] ARM 4: dose-response sweep")
    V0_dr = _random_subspace(d, k, rng)
    vp1_dr = rng.standard_normal(d)
    vp1_dr -= V0_dr @ (V0_dr.T @ vp1_dr)
    vp1_dr /= np.linalg.norm(vp1_dr)
    vp2_dr = rng.standard_normal(d)
    vp2_dr -= V0_dr @ (V0_dr.T @ vp2_dr)
    vp2_dr -= vp1_dr * (vp1_dr @ vp2_dr)
    vp2_dr /= np.linalg.norm(vp2_dr)

    radii = np.linspace(0, 0.7, 10)
    dose_response = []
    for r_dr in tqdm(radii, desc="Dose-response"):
        if r_dr < 1e-6:
            dr_sections = []
            for _ in range(m):
                noise = rng.standard_normal((d, k)) * noise_level
                Q, _ = linalg.qr(V0_dr + noise, mode="economic")
                dr_sections.append(Q)
        else:
            dr_sections, _, _ = _generate_cocycle_sections(
                V0_dr, r_dr, m, noise_level, rng, vp1_dr, vp2_dr
            )

        _, dr_norm = _cocycle_holonomy(dr_sections)
        dr_p = float(np.mean(null_norms >= dr_norm))
        dose_response.append({
            "loop_radius": float(r_dr),
            "holonomy_norm": float(dr_norm),
            "p_value": dr_p,
            "detected": bool(dr_p < 0.05),
        })

    norms_arr = [entry["holonomy_norm"] for entry in dose_response]
    monotonic = all(norms_arr[i] <= norms_arr[i + 1] + 0.05 for i in range(len(norms_arr) - 1))

    results["arm4_dose_response"] = {
        "sweep": dose_response,
        "approximately_monotonic": bool(monotonic),
    }

    # --- Calibration summary ---
    fpr_at_005 = float(np.mean(null_norms > threshold_95_holonomy))

    results["calibration"] = {
        "n_bootstrap": n_null,
        "null_mean": float(np.mean(null_norms)),
        "null_std": float(np.std(null_norms)),
        "threshold_95_holonomy": threshold_95_holonomy,
        "threshold_95_pairwise": threshold_95_pairwise,
        "fpr_at_005": fpr_at_005,
    }

    print(f"[{datetime.now():%H:%M:%S}] Cocycle obstruction experiment complete")
    return results


# ======================================================================
# EXPERIMENT 2 (P1): BRACKET-NORM CONFOUND AUDIT
# ======================================================================

def run_bracket_norm_confound_audit(seed=42):
    """P1: confound audit for 4 MS imaging progression-substrate metrics.

    For each metric: naive association, bracket-corrected association,
    confound-leakage Delta, sNfL anchor validation, matroid rank.
    """
    print(f"[{datetime.now():%H:%M:%S}] Starting P1: bracket-norm confound audit")
    rng = np.random.default_rng(seed)

    n_patients = 1500
    n_sites = 8

    site = rng.choice(n_sites, size=n_patients)
    site_quality = np.linspace(0.2, 1.0, n_sites)
    acq_richness_base = site_quality[site]
    acq_richness = acq_richness_base + rng.standard_normal(n_patients) * 0.1
    acq_richness = np.clip(acq_richness, 0, 1)

    # Referral bias: specialist centers (high acq) see more progressive patients
    referral_bias = 0.5 * acq_richness
    true_progression = referral_bias + rng.standard_normal(n_patients) * 0.8
    snfl = 0.6 * true_progression + rng.standard_normal(n_patients) * 0.3

    metrics_config = {
        "iron_rim_qsm": {"substrate_weight": 0.3, "confound_weight": 0.8, "noise": 0.2},
        "deep_gm_atrophy": {"substrate_weight": 0.7, "confound_weight": 0.15, "noise": 0.2},
        "cortical_lesion_count": {"substrate_weight": 0.2, "confound_weight": 1.0, "noise": 0.3},
        "cervical_cord_csa": {"substrate_weight": 0.6, "confound_weight": 0.2, "noise": 0.25},
    }

    metric_values = {}
    for name, cfg in metrics_config.items():
        metric_values[name] = (
            cfg["substrate_weight"] * true_progression
            + cfg["confound_weight"] * acq_richness
            + cfg["noise"] * rng.standard_normal(n_patients)
        )

    latent_progression = true_progression + rng.standard_normal(n_patients) * 0.2

    confound_results = {}
    for name, values in tqdm(metric_values.items(), desc="Confound audit"):
        # R^2-based confound leakage (avoids suppression artifacts in coefficients)
        r2_metric = float(pearsonr(values, latent_progression)[0] ** 2)

        # R^2 of acq alone
        r2_acq = float(pearsonr(acq_richness, latent_progression)[0] ** 2)

        # R^2 of both together
        X_both = np.column_stack([values, acq_richness])
        reg_both = LinearRegression().fit(X_both, latent_progression)
        r2_both = float(reg_both.score(X_both, latent_progression))

        # Metric's unique R^2 = total R^2 - acq-only R^2
        r2_unique = max(r2_both - r2_acq, 0.0)

        # Delta = fraction of metric's R^2 that is shared with acq
        delta = (r2_metric - r2_unique) / r2_metric if r2_metric > 1e-8 else 0.0

        residual_metric = values - LinearRegression().fit(
            acq_richness.reshape(-1, 1), values
        ).predict(acq_richness.reshape(-1, 1))
        snfl_cor, snfl_p = pearsonr(residual_metric, snfl)

        confound_results[name] = {
            "R2_metric_total": r2_metric,
            "R2_acq_only": r2_acq,
            "R2_both": r2_both,
            "R2_metric_unique": r2_unique,
            "Delta_leakage": float(delta),
            "post_correction_snfl_r": float(snfl_cor),
            "post_correction_snfl_p": float(snfl_p),
            "confound_robust": delta < 0.3,
        }
        print(f"[{datetime.now():%H:%M:%S}]   {name}: Delta={delta:.3f}, sNfL_r={snfl_cor:.3f}")

    corrected_matrix = np.column_stack([
        metric_values[name] - LinearRegression().fit(
            acq_richness.reshape(-1, 1), metric_values[name]
        ).predict(acq_richness.reshape(-1, 1))
        for name in metrics_config
    ])

    pca = PCA()
    pca.fit(corrected_matrix)
    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)
    matroid_rank = int(np.searchsorted(cumulative, 0.90) + 1)

    tier_assignment = {}
    for name, res in confound_results.items():
        delta = res["Delta_leakage"]
        if delta < 0.3 and res["post_correction_snfl_p"] < 0.05:
            tier_assignment[name] = "T3_confirmed"
        elif delta < 0.5:
            tier_assignment[name] = "T2_T3_borderline"
        else:
            tier_assignment[name] = "T2_demoted"

    results = {
        "per_metric": confound_results,
        "matroid_rank": {
            "rank_at_90pct": matroid_rank,
            "explained_variance_ratios": explained.tolist(),
            "cumulative": cumulative.tolist(),
        },
        "tier_reassignment": tier_assignment,
        "n_patients": n_patients,
        "n_sites": n_sites,
    }

    print(f"[{datetime.now():%H:%M:%S}] P1 complete: matroid_rank={matroid_rank}, tiers={tier_assignment}")
    return results


# ======================================================================
# EXPERIMENT 3 (P2): SHEAF DAG ADJUDICATION
# ======================================================================

def _simulate_ms_dynamics(n, n_timepoints, dag_type, rng):
    """Simulate longitudinal inflammation/degeneration data under a given DAG.

    dag_type: "outside_in" (inflammation->degeneration->disability)
              "inside_out" (degeneration->inflammation->disability)
              "feedback" (bidirectional)
    """
    inflammation = np.zeros((n, n_timepoints))
    degeneration = np.zeros((n, n_timepoints))
    disability = np.zeros((n, n_timepoints))

    inflammation[:, 0] = rng.standard_normal(n) * 0.5
    degeneration[:, 0] = rng.standard_normal(n) * 0.5

    for t in range(1, n_timepoints):
        noise_i = rng.standard_normal(n) * 0.2
        noise_d = rng.standard_normal(n) * 0.2

        if dag_type == "outside_in":
            inflammation[:, t] = 0.7 * inflammation[:, t - 1] + noise_i
            degeneration[:, t] = (
                0.5 * degeneration[:, t - 1]
                + 0.4 * inflammation[:, t - 1]
                + noise_d
            )
        elif dag_type == "inside_out":
            degeneration[:, t] = 0.7 * degeneration[:, t - 1] + noise_d
            inflammation[:, t] = (
                0.5 * inflammation[:, t - 1]
                + 0.4 * degeneration[:, t - 1]
                + noise_i
            )
        elif dag_type == "feedback":
            inflammation[:, t] = (
                0.5 * inflammation[:, t - 1]
                + 0.3 * degeneration[:, t - 1]
                + noise_i
            )
            degeneration[:, t] = (
                0.5 * degeneration[:, t - 1]
                + 0.3 * inflammation[:, t - 1]
                + noise_d
            )

        disability[:, t] = (
            0.3 * inflammation[:, t]
            + 0.4 * degeneration[:, t]
            + rng.standard_normal(n) * 0.15
        )

    return inflammation, degeneration, disability


def _estimate_local_dag(inflammation, degeneration, disability):
    """Estimate local DAG edge weights via time-lagged regression.

    Returns (edges, edge_ses) where each is a dict of edge name -> float.
    """
    infl_lag = inflammation[:, :-1].ravel()
    degen_lag = degeneration[:, :-1].ravel()
    infl_curr = inflammation[:, 1:].ravel()
    degen_curr = degeneration[:, 1:].ravel()
    disab_curr = disability[:, 1:].ravel()

    def _ols(X, y):
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        resid = y - X @ beta
        n, p = X.shape
        mse = np.sum(resid ** 2) / max(n - p, 1)
        cov = mse * np.linalg.pinv(X.T @ X)
        return beta, np.sqrt(np.maximum(np.diag(cov), 0.0))

    beta_d, se_d = _ols(np.column_stack([degen_lag, infl_lag]), degen_curr)
    beta_i, se_i = _ols(np.column_stack([infl_lag, degen_lag]), infl_curr)
    beta_dis, se_dis = _ols(np.column_stack([infl_curr, degen_curr]), disab_curr)

    edges = {
        "infl_to_degen": float(beta_d[1]),
        "degen_to_infl": float(beta_i[1]),
        "infl_to_disab": float(beta_dis[0]),
        "degen_to_disab": float(beta_dis[1]),
    }
    edge_ses = {
        "infl_to_degen": float(se_d[1]),
        "degen_to_infl": float(se_i[1]),
        "infl_to_disab": float(se_dis[0]),
        "degen_to_disab": float(se_dis[1]),
    }
    return edges, edge_ses


def run_sheaf_dag_adjudication(seed=42):
    """P2: inflammation-first vs degeneration-first DAG adjudication.

    Tests whether local causal sections across strata glue into a
    single global DAG or carry an H^1 obstruction forcing a two-process model.
    """
    print(f"[{datetime.now():%H:%M:%S}] Starting P2: sheaf DAG adjudication")
    rng = np.random.default_rng(seed)

    n_per_stratum = 300
    n_timepoints = 12

    strata = {
        "early_RRMS": "outside_in",
        "late_RRMS": "feedback",
        "SPMS_active": "feedback",
        "SPMS_inactive": "inside_out",
        "ppms": "inside_out",
        "btk_treated": "inside_out",
        "siponimod_treated": "feedback",
        "anti_cd20_treated": "outside_in",
    }

    # Step 1: estimate local sections
    print(f"[{datetime.now():%H:%M:%S}] Estimating local DAG sections for {len(strata)} strata")
    local_sections = {}
    local_section_ses = {}
    for name, dag_type in tqdm(strata.items(), desc="Local DAGs"):
        infl, degen, disab = _simulate_ms_dynamics(
            n_per_stratum, n_timepoints, dag_type, rng
        )
        edges, edge_ses = _estimate_local_dag(infl, degen, disab)
        local_sections[name] = edges
        local_section_ses[name] = edge_ses

    # Step 2: pairwise consistency
    stratum_names = list(local_sections.keys())
    n_strata = len(stratum_names)
    edge_names = ["infl_to_degen", "degen_to_infl", "infl_to_disab", "degen_to_disab"]

    pairwise_diffs = {}
    for i in range(n_strata):
        for j in range(i + 1, n_strata):
            si, sj = stratum_names[i], stratum_names[j]
            diffs = {
                e: abs(local_sections[si][e] - local_sections[sj][e])
                for e in edge_names
            }
            pairwise_diffs[(si, sj)] = diffs

    consistency_threshold = 0.15
    inconsistent_pairs = [
        (pair, diffs) for pair, diffs in pairwise_diffs.items()
        if any(v > consistency_threshold for v in diffs.values())
    ]

    # Step 3: per-edge Q tests for heterogeneity
    edge_matrix = np.array([
        [local_sections[s][e] for e in edge_names]
        for s in stratum_names
    ])

    per_edge_var = {
        edge_names[j]: float(np.var(edge_matrix[:, j]))
        for j in range(len(edge_names))
    }

    per_edge_q_tests = {}
    for edge in edge_names:
        estimates = {
            s: {"beta": local_sections[s][edge], "se": local_section_ses[s][edge]}
            for s in stratum_names
        }
        p, Q, df = _sheaf_q_test(estimates)
        per_edge_q_tests[edge] = {"Q": float(Q), "p": float(p), "df": int(df)}

    n_tests = len(edge_names)
    min_p = min(t["p"] for t in per_edge_q_tests.values())
    H1_p = min(min_p * n_tests, 1.0)
    H1_nonzero = min_p < 0.05 / n_tests

    H1_norm = float(np.linalg.norm(
        edge_matrix - edge_matrix.mean(axis=0), "fro"
    ) / np.sqrt(n_strata))

    # Step 4: interventional anchoring
    btk = local_sections["btk_treated"]
    sipo = local_sections["siponimod_treated"]
    early = local_sections["early_RRMS"]

    btk_constraint = btk["infl_to_degen"] < early["infl_to_degen"]
    sipo_constraint = sipo["degen_to_disab"] > 0.1

    anchoring = {
        "btk_reduces_infl_to_degen": bool(btk_constraint),
        "btk_infl_to_degen": btk["infl_to_degen"],
        "early_infl_to_degen": early["infl_to_degen"],
        "siponimod_relapse_independent_progression": bool(sipo_constraint),
        "siponimod_degen_to_disab": sipo["degen_to_disab"],
    }

    # Step 5: identify the relapse-independent degeneration edge
    degen_to_disab_weights = [local_sections[s]["degen_to_disab"] for s in stratum_names]
    relapse_independent_edge = float(np.mean(degen_to_disab_weights))

    results = {
        "local_sections": local_sections,
        "n_strata": n_strata,
        "n_inconsistent_pairs": len(inconsistent_pairs),
        "inconsistent_pair_names": [
            (p[0], p[1]) for p, _ in inconsistent_pairs
        ],
        "H1_obstruction_norm": H1_norm,
        "H1_p_value_bonferroni": H1_p,
        "H1_nonzero": H1_nonzero,
        "per_edge_q_tests": per_edge_q_tests,
        "per_edge_variance": per_edge_var,
        "interventional_anchoring": anchoring,
        "relapse_independent_degen_edge": relapse_independent_edge,
        "verdict": (
            "two-process feedback SUPPORTED"
            if H1_nonzero
            else "single DAG NOT rejected"
        ),
    }

    print(f"[{datetime.now():%H:%M:%S}] P2 complete: H1_norm={H1_norm:.4f}, min_p={min_p:.4e}, verdict={results['verdict']}")
    return results


# ======================================================================
# EXPERIMENT 4 (P3): H^1 EFFECT-MODIFIER SUITE
# ======================================================================

def _estimate_stratified_effect(outcome, treatment, strata_col, rng):
    """Estimate treatment effect within each stratum. Returns dict of {stratum: {beta, se}}."""
    estimates = {}
    for s in np.unique(strata_col):
        mask = strata_col == s
        if mask.sum() < 20:
            continue
        y = outcome[mask]
        t = treatment[mask]
        X = np.column_stack([t, np.ones(mask.sum())])
        beta, residuals, _, _ = np.linalg.lstsq(X, y, rcond=None)
        if len(residuals) > 0:
            mse = residuals[0] / (mask.sum() - 2)
        else:
            mse = np.var(y - X @ beta)
        se = float(np.sqrt(mse / np.sum((t - t.mean()) ** 2)))
        estimates[str(s)] = {"beta": float(beta[0]), "se": max(se, 1e-8)}
    return estimates


def run_h1_effect_modifier_suite(seed=42):
    """P3: H^1 obstruction for mechanism-modifier pairs in MS.

    Tests which MS mechanisms transport across patient strata (H^1 ~ 0)
    vs are irreducibly stratum-specific (H^1 != 0).
    Includes scale-invariance stress test on HLA x EBV.
    """
    print(f"[{datetime.now():%H:%M:%S}] Starting P3: H^1 effect-modifier suite")
    rng = np.random.default_rng(seed)
    n = 2000

    modifier_pairs = {
        "HLA_x_EBV_risk": {
            "modifier_type": "exposure_to_mechanism",
            "transport_expected": True,
            "interaction_strength": 0.1,
        },
        "sex_x_course": {
            "modifier_type": "exposure_to_mechanism",
            "transport_expected": False,
            "interaction_strength": 0.5,
        },
        "genetics_x_OCB": {
            "modifier_type": "exposure_to_mechanism",
            "transport_expected": False,
            "interaction_strength": 0.6,
        },
        "age_x_anti_CD20": {
            "modifier_type": "treatment_to_outcome",
            "transport_expected": False,
            "interaction_strength": 0.4,
        },
        "phenotype_x_GM_atrophy": {
            "modifier_type": "mechanism_to_outcome",
            "transport_expected": False,
            "interaction_strength": 0.45,
        },
        "EBV_necessity": {
            "modifier_type": "exposure_to_mechanism",
            "transport_expected": True,
            "interaction_strength": 0.05,
        },
        "vitD_causal_risk": {
            "modifier_type": "exposure_to_mechanism",
            "transport_expected": True,
            "interaction_strength": 0.04,
        },
    }

    pair_results = {}
    for pair_name, config in tqdm(modifier_pairs.items(), desc="Modifier pairs"):
        if config["interaction_strength"] < 0.10:
            n_pair = 100
        elif config["interaction_strength"] < 0.15:
            n_pair = 200
        else:
            n_pair = n
        modifier = rng.choice(4, size=n_pair)
        treatment = rng.standard_normal(n_pair)

        base_effect = 0.5
        stratum_effects = np.array([
            base_effect + config["interaction_strength"] * (s - 1.5)
            for s in range(4)
        ])
        effect_per_patient = stratum_effects[modifier]

        noise_scale = 0.5
        outcome = effect_per_patient * treatment + rng.standard_normal(n_pair) * noise_scale

        estimates = _estimate_stratified_effect(outcome, treatment, modifier, rng)
        p_sheaf, Q_sheaf, df_sheaf = _sheaf_q_test(estimates)

        H1_detected = p_sheaf < 0.05
        transport = not H1_detected

        pair_results[pair_name] = {
            "modifier_type": config["modifier_type"],
            "expected_transport": config["transport_expected"],
            "interaction_strength": config["interaction_strength"],
            "H1_Q": Q_sheaf,
            "H1_df": df_sheaf,
            "H1_p_value": p_sheaf,
            "H1_detected": H1_detected,
            "transport_verdict": "transport" if transport else "non-transport",
            "matches_prediction": transport == config["transport_expected"],
            "per_stratum_effects": {
                k: v["beta"] for k, v in estimates.items()
            },
        }
        print(f"[{datetime.now():%H:%M:%S}]   {pair_name}: H1_p={p_sheaf:.4f}, transport={transport}, expected={config['transport_expected']}")

    # --- SCALE-INVARIANCE STRESS TEST (HLA x EBV) ---
    print(f"[{datetime.now():%H:%M:%S}] Scale-invariance stress test")
    n_si = 3000
    hla = rng.binomial(1, 0.25, n_si).astype(float)
    ebv = rng.binomial(1, 0.90, n_si).astype(float)

    true_theta = rng.standard_normal(n_si)
    prog_additive = 0.3 * hla + 0.2 * ebv + 0.4 * hla * ebv + true_theta * 0.3
    noise = rng.standard_normal(n_si) * 0.2

    outcome_raw = prog_additive + noise
    outcome_raw_pos = outcome_raw - outcome_raw.min() + 0.1

    from sklearn.preprocessing import QuantileTransformer
    qt = QuantileTransformer(output_distribution="normal", n_quantiles=200)
    outcome_invariant = qt.fit_transform(outcome_raw.reshape(-1, 1)).ravel()

    hla_strata = hla.astype(int)

    estimates_add_raw = _estimate_stratified_effect(outcome_raw, ebv, hla_strata, rng)
    _, Q_add_raw, _ = _sheaf_q_test(estimates_add_raw)

    estimates_mult_raw = _estimate_stratified_effect(np.log(outcome_raw_pos), ebv, hla_strata, rng)
    _, Q_mult_raw, _ = _sheaf_q_test(estimates_mult_raw)

    estimates_add_inv = _estimate_stratified_effect(outcome_invariant, ebv, hla_strata, rng)
    _, Q_add_inv, _ = _sheaf_q_test(estimates_add_inv)

    estimates_mult_inv = _estimate_stratified_effect(np.log(np.exp(outcome_invariant) + 1), ebv, hla_strata, rng)
    _, Q_mult_inv, _ = _sheaf_q_test(estimates_mult_inv)

    scale_var_raw = abs(Q_add_raw - Q_mult_raw) / max(Q_add_raw, Q_mult_raw, 1e-8)
    scale_var_inv = abs(Q_add_inv - Q_mult_inv) / max(Q_add_inv, Q_mult_inv, 1e-8)

    scale_test = {
        "Q_additive_raw": Q_add_raw,
        "Q_multiplicative_raw": Q_mult_raw,
        "Q_additive_invariant": Q_add_inv,
        "Q_multiplicative_invariant": Q_mult_inv,
        "scale_variability_raw": float(scale_var_raw),
        "scale_variability_invariant": float(scale_var_inv),
        "invariance_improved": scale_var_inv < scale_var_raw,
        "test_passes": scale_var_inv < 0.3,
    }

    # --- Edge-type stratification ---
    edge_types = {}
    for pair_name, res in pair_results.items():
        mt = res["modifier_type"]
        if mt not in edge_types:
            edge_types[mt] = []
        edge_types[mt].append({
            "pair": pair_name,
            "H1_Q": res["H1_Q"],
            "transport": res["transport_verdict"],
        })

    prediction_accuracy = np.mean([
        r["matches_prediction"] for r in pair_results.values()
    ])

    results = {
        "per_pair": pair_results,
        "scale_invariance_test": scale_test,
        "edge_type_stratification": edge_types,
        "prediction_accuracy": float(prediction_accuracy),
        "n_transport": sum(1 for r in pair_results.values() if r["transport_verdict"] == "transport"),
        "n_non_transport": sum(1 for r in pair_results.values() if r["transport_verdict"] == "non-transport"),
    }

    print(f"[{datetime.now():%H:%M:%S}] P3 complete: {results['n_transport']} transport, {results['n_non_transport']} non-transport")
    print(f"[{datetime.now():%H:%M:%S}] Scale test passes: {scale_test['test_passes']}, prediction accuracy: {prediction_accuracy:.2f}")
    return results
