"""ENIGMA holonomy pipeline — detect global inconsistency in multi-site neuroimaging subspaces.

Grassmannian holonomy (Berry phase) detects global inconsistency in subspace-valued
data that pairwise tests miss. This script provides:

1. Real data mode: load site-level covariance matrices from ENIGMA/ADNI CSV/NPY files,
   compute PCA subspaces per site, run holonomy test with permutation null.
2. Calibrated simulation mode: generate realistic multi-site cortical thickness data
   calibrated to published ENIGMA parameters (Grasby et al. 2020 Science, Fortin 2018),
   then run the same holonomy test.

Usage:
    python pipeline.py --simulate --n-sites 50 --k 3 --output results/ --plot
    python pipeline.py --data-dir /path/to/enigma/ --k 5 --output results/ --plot
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import linalg
from tqdm import tqdm

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Plot style ────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.linewidth": 1.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})


# ======================================================================
# GRASSMANNIAN GEOMETRY
# ======================================================================

def principal_angles(U1: np.ndarray, U2: np.ndarray) -> np.ndarray:
    """Principal angles between two subspaces given as orthonormal bases."""
    M = U1.T @ U2
    svals = linalg.svdvals(M)
    svals = np.clip(svals, -1.0, 1.0)
    return np.arccos(svals)


def geodesic_distance(U1: np.ndarray, U2: np.ndarray) -> float:
    """Grassmannian geodesic distance = L2 norm of principal angles."""
    return float(np.linalg.norm(principal_angles(U1, U2)))


def transport_matrix(U_from: np.ndarray, U_to: np.ndarray) -> np.ndarray:
    """Parallel transport on Gr(k,d) via the canonical (Levi-Civita) connection.

    Given orthonormal bases U_from, U_to for two k-subspaces, returns the
    k x k orthogonal matrix T such that a tangent vector v at U_from maps
    to T @ v at U_to along the geodesic.
    """
    M = U_from.T @ U_to
    u, _, vt = linalg.svd(M)
    return u @ vt


def compose_holonomy(subspaces: list[np.ndarray]) -> tuple[np.ndarray, float]:
    """Compose parallel transport around a closed cycle of subspaces.

    Returns (Phi, deviation) where Phi is the k x k holonomy matrix and
    deviation = ||Phi - I_k||_F.  Non-zero deviation indicates curvature-
    induced global inconsistency that pairwise comparisons cannot detect.
    """
    k = subspaces[0].shape[1]
    composed = np.eye(k)
    for idx in range(len(subspaces)):
        U_from = subspaces[idx]
        U_to = subspaces[(idx + 1) % len(subspaces)]
        T = transport_matrix(U_from, U_to)
        composed = T @ composed
    deviation = float(np.linalg.norm(composed - np.eye(k), "fro"))
    return composed, deviation


def pairwise_distance_matrix(subspaces: list[np.ndarray]) -> np.ndarray:
    """Compute full pairwise geodesic distance matrix on Gr(k,d)."""
    m = len(subspaces)
    D = np.zeros((m, m))
    for i in range(m):
        for j in range(i + 1, m):
            d = geodesic_distance(subspaces[i], subspaces[j])
            D[i, j] = d
            D[j, i] = d
    return D


def pairwise_principal_angles(subspaces: list[np.ndarray]) -> dict:
    """Compute principal angles for every pair.  Returns dict (i,j) -> angles."""
    m = len(subspaces)
    result = {}
    for i in range(m):
        for j in range(i + 1, m):
            result[(i, j)] = principal_angles(subspaces[i], subspaces[j]).tolist()
    return result


# ======================================================================
# PERMUTATION TEST
# ======================================================================

def holonomy_permutation_test(
    data_matrices: list[np.ndarray],
    cycle_indices: list[int],
    k: int,
    n_perms: int,
    rng: np.random.Generator,
) -> dict:
    """Permutation test for holonomy significance.

    Shuffles subject-to-site assignment, recomputes PCA subspaces, and
    measures holonomy around the given cycle.  Returns observed holonomy,
    null distribution, and p-value.
    """
    subspaces = [pca_subspace(data_matrices[i], k) for i in cycle_indices]
    _, observed = compose_holonomy(subspaces)

    pooled = np.vstack(data_matrices)
    site_sizes = [len(m) for m in data_matrices]

    null_dist = np.zeros(n_perms)
    for p in tqdm(range(n_perms), desc="Permutation test", unit="perm"):
        perm = rng.permutation(len(pooled))
        shuffled = pooled[perm]

        perm_matrices = []
        offset = 0
        for size in site_sizes:
            perm_matrices.append(shuffled[offset:offset + size])
            offset += size

        perm_subspaces = [pca_subspace(perm_matrices[i], k) for i in cycle_indices]
        _, dev = compose_holonomy(perm_subspaces)
        null_dist[p] = dev

    p_value = float(np.mean(null_dist >= observed))

    return {
        "observed_holonomy": observed,
        "p_value": p_value,
        "null_mean": float(np.mean(null_dist)),
        "null_std": float(np.std(null_dist)),
        "null_percentiles": {
            str(q): float(np.percentile(null_dist, q))
            for q in [50, 90, 95, 99]
        },
        "null_distribution": null_dist.tolist(),
    }


def berry_phase_boundary_analysis(
    n_sites: int = 24,
    d: int = 34,
    k: int = 3,
    radius: float = 0.5,
    noise_std: float = 0.05,
    sample_sizes: list[int] | None = None,
    n_trials: int = 50,
    seed: int = 42,
    output_dir: Path | None = None,
    do_plot: bool = False,
) -> dict:
    """Map the detection boundary for Berry phase in ENIGMA-realistic data.

    For each sample size, generates n_trials realizations of:
    - Berry phase simulation (planted cyclic rotation at given radius)
    - Null simulation (same biological + site effects, no cyclic rotation)

    Reports separation (sigma), recovery fraction (observed/planted),
    and detection power at alpha=0.05.
    """
    if sample_sizes is None:
        sample_sizes = [100, 200, 500, 1000, 2000, 5000]

    rng_base = np.random.default_rng(seed)

    # Shared biological structure
    Q, _ = linalg.qr(rng_base.standard_normal((d, d)))
    ev = np.array([1.0 / (j + 1) ** 2 for j in range(d)]) * 0.3
    bio_cov = Q @ np.diag(ev) @ Q.T
    bio_mean = rng_base.standard_normal(d) * 0.1 + 2.5
    e1, e2 = Q[:, k], Q[:, k + 1]

    # Planted holonomy from ideal subspaces
    planted_subs = []
    for s in range(n_sites):
        theta = 2 * np.pi * s / n_sites
        V = Q[:, :k].copy()
        t1 = np.cos(theta) * e1 + np.sin(theta) * e2
        t2 = np.cos(theta + np.pi / 2) * e1 + np.sin(theta + np.pi / 2) * e2
        V[:, 0] = np.cos(radius) * Q[:, 0] + np.sin(radius) * t1
        V[:, 1] = np.cos(radius) * Q[:, 1] + np.sin(radius) * t2
        V, _ = linalg.qr(V, mode='economic')
        planted_subs.append(V)
    _, planted_holonomy = compose_holonomy(planted_subs)
    max_adj = max(
        geodesic_distance(planted_subs[i], planted_subs[(i + 1) % n_sites])
        for i in range(n_sites)
    )

    log.info("Planted holonomy=%.4f, max_adjacent=%.4f", planted_holonomy, max_adj)
    log.info("Testing %d sample sizes: %s", len(sample_sizes), sample_sizes)

    results_by_n = {}

    for n_subj in tqdm(sample_sizes, desc="Sample sizes"):
        berry_holonomies = []
        null_holonomies = []

        for trial in tqdm(range(n_trials), desc=f"  n={n_subj}", leave=False):
            rng_trial = np.random.default_rng(seed * 10000 + n_subj * 100 + trial)

            # --- Berry phase condition ---
            bp_subs = []
            for s in range(n_sites):
                theta = 2 * np.pi * s / n_sites
                V = Q[:, :k].copy()
                t1 = np.cos(theta) * e1 + np.sin(theta) * e2
                t2 = np.cos(theta + np.pi / 2) * e1 + np.sin(theta + np.pi / 2) * e2
                V[:, 0] = np.cos(radius) * Q[:, 0] + np.sin(radius) * t1
                V[:, 1] = np.cos(radius) * Q[:, 1] + np.sin(radius) * t2
                V, _ = linalg.qr(V, mode='economic')

                rot_cov = V @ np.diag(ev[:k]) @ V.T
                rot_cov += Q[:, k:] @ np.diag(ev[k:]) @ Q[:, k:].T

                X = rng_trial.multivariate_normal(bio_mean, rot_cov, size=n_subj)
                X += rng_trial.standard_normal(X.shape) * noise_std
                site_bias = rng_trial.standard_normal(d) * 0.05
                site_scale = 1.0 + rng_trial.standard_normal(d) * 0.03
                X = X * site_scale + site_bias
                bp_subs.append(pca_subspace(X, k))

            _, h_bp = compose_holonomy(bp_subs)
            berry_holonomies.append(h_bp)

            # --- Null condition (no Berry phase) ---
            null_subs = []
            rng_null = np.random.default_rng(seed * 20000 + n_subj * 100 + trial)
            for s in range(n_sites):
                X = rng_null.multivariate_normal(bio_mean, bio_cov, size=n_subj)
                X += rng_null.standard_normal(X.shape) * noise_std
                site_bias = rng_null.standard_normal(d) * 0.05
                site_scale = 1.0 + rng_null.standard_normal(d) * 0.03
                X = X * site_scale + site_bias
                null_subs.append(pca_subspace(X, k))

            _, h_null = compose_holonomy(null_subs)
            null_holonomies.append(h_null)

        bp_arr = np.array(berry_holonomies)
        null_arr = np.array(null_holonomies)
        separation = (bp_arr.mean() - null_arr.mean()) / null_arr.std() if null_arr.std() > 0 else float('inf')

        # Detection power at alpha=0.05: fraction of Berry trials above
        # the 95th percentile of the null distribution
        threshold = float(np.percentile(null_arr, 95))
        power = float(np.mean(bp_arr >= threshold))

        results_by_n[n_subj] = {
            "berry_mean": float(bp_arr.mean()),
            "berry_std": float(bp_arr.std()),
            "null_mean": float(null_arr.mean()),
            "null_std": float(null_arr.std()),
            "separation_sigma": float(separation),
            "recovery_fraction": float(bp_arr.mean() / planted_holonomy),
            "detection_power": power,
            "threshold_95": threshold,
        }
        log.info(
            "  n=%d: berry=%.3f±%.3f, null=%.3f±%.3f, sep=%.1fσ, power=%.2f",
            n_subj, bp_arr.mean(), bp_arr.std(),
            null_arr.mean(), null_arr.std(), separation, power,
        )

    result = {
        "timestamp": datetime.now().isoformat(),
        "params": {
            "n_sites": n_sites, "d": d, "k": k, "radius": radius,
            "noise_std": noise_std, "n_trials": n_trials, "seed": seed,
        },
        "planted_holonomy": float(planted_holonomy),
        "max_adjacent_distance": float(max_adj),
        "by_sample_size": results_by_n,
    }

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "boundary_results.json", "w") as f:
            json.dump(result, f, indent=2)
        log.info("Saved boundary results: %s", output_dir / "boundary_results.json")

        if do_plot:
            _plot_boundary(result, output_dir)

    return result


def _plot_boundary(result: dict, output_dir: Path) -> None:
    by_n = result["by_sample_size"]
    # Keys may be int or str depending on JSON serialization path
    def _get(n):
        return by_n.get(n, by_n.get(str(n), by_n.get(int(n) if isinstance(n, str) else n)))

    ns = sorted(int(n) for n in by_n.keys())
    berry_means = [_get(n)["berry_mean"] for n in ns]
    berry_stds = [_get(n)["berry_std"] for n in ns]
    null_means = [_get(n)["null_mean"] for n in ns]
    null_stds = [_get(n)["null_std"] for n in ns]
    powers = [_get(n)["detection_power"] for n in ns]
    seps = [_get(n)["separation_sigma"] for n in ns]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Panel A: Holonomy vs sample size
    ax = axes[0]
    ax.fill_between(ns, [m - s for m, s in zip(berry_means, berry_stds)],
                    [m + s for m, s in zip(berry_means, berry_stds)],
                    alpha=0.2, color="#2980B9")
    ax.plot(ns, berry_means, "o-", color="#2980B9", label="Berry phase")
    ax.fill_between(ns, [m - s for m, s in zip(null_means, null_stds)],
                    [m + s for m, s in zip(null_means, null_stds)],
                    alpha=0.2, color="#95A5A6")
    ax.plot(ns, null_means, "s-", color="#95A5A6", label="Null (no rotation)")
    ax.axhline(result["planted_holonomy"], color="#C0392B", linestyle="--", linewidth=1, label="Planted")
    ax.set_xscale("log")
    ax.set_xlabel("Subjects per site")
    ax.set_ylabel(r"Holonomy $\|\Phi - I_k\|_F$")
    ax.set_title("Signal recovery")
    ax.legend(frameon=False, fontsize=9)

    # Panel B: Separation
    ax = axes[1]
    ax.plot(ns, seps, "o-", color="#2C3E50")
    ax.axhline(1.96, color="#E74C3C", linestyle=":", label=r"$z = 1.96$")
    ax.set_xscale("log")
    ax.set_xlabel("Subjects per site")
    ax.set_ylabel(r"Separation ($\sigma$)")
    ax.set_title("Signal-to-noise")
    ax.legend(frameon=False, fontsize=9)

    # Panel C: Detection power
    ax = axes[2]
    ax.plot(ns, powers, "o-", color="#27AE60")
    ax.axhline(0.80, color="#E74C3C", linestyle=":", label="80% power")
    ax.set_xscale("log")
    ax.set_xlabel("Subjects per site")
    ax.set_ylabel("Detection power")
    ax.set_title(r"Power at $\alpha = 0.05$")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(frameon=False, fontsize=9)

    fig.suptitle(
        f"Berry phase detection boundary (Gr({result['params']['k']},{result['params']['d']}), "
        f"m={result['params']['n_sites']}, r={result['params']['radius']})",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "detection_boundary.png")
    plt.close(fig)
    log.info("Saved detection boundary plot: %s", output_dir / "detection_boundary.png")


# ======================================================================
# PCA SUBSPACE EXTRACTION
# ======================================================================

def pca_subspace(X: np.ndarray, k: int) -> np.ndarray:
    """Extract the top-k PCA subspace from a (n_subjects, d) data matrix.

    Returns a (d, k) orthonormal basis for the top-k principal components.
    """
    X_centered = X - X.mean(axis=0, keepdims=True)
    cov = (X_centered.T @ X_centered) / (X_centered.shape[0] - 1)
    eigenvalues, eigenvectors = linalg.eigh(cov)
    # eigh returns ascending order; take the last k columns
    idx = np.argsort(eigenvalues)[::-1][:k]
    return eigenvectors[:, idx]


# ======================================================================
# REAL DATA LOADING
# ======================================================================

def load_site_data(data_dir: Path) -> tuple[list[np.ndarray], list[str]]:
    """Load per-site cortical thickness/surface area matrices.

    Expects one of:
    - A directory of per-site CSV/NPY files named site_*.csv or site_*.npy,
      each with shape (n_subjects, d_regions).
    - A single CSV with a 'site' column and d region columns.
    - A single NPY file + a site_labels.npy file.
    """
    data_dir = Path(data_dir)

    # Case 1: directory of per-site files
    site_csvs = sorted(data_dir.glob("site_*.csv"))
    site_npys = sorted(data_dir.glob("site_*.npy"))

    if site_csvs:
        log.info("Loading %d site CSV files from %s", len(site_csvs), data_dir)
        matrices = []
        names = []
        for f in site_csvs:
            df = pd.read_csv(f)
            matrices.append(df.values.astype(np.float64))
            names.append(f.stem)
        return matrices, names

    if site_npys:
        log.info("Loading %d site NPY files from %s", len(site_npys), data_dir)
        matrices = []
        names = []
        for f in site_npys:
            matrices.append(np.load(f).astype(np.float64))
            names.append(f.stem)
        return matrices, names

    # Case 2: single combined CSV with 'site' column
    combined_csvs = list(data_dir.glob("*.csv"))
    if combined_csvs:
        f = combined_csvs[0]
        log.info("Loading combined CSV %s", f)
        df = pd.read_csv(f)
        if "site" not in df.columns:
            raise ValueError(f"Combined CSV {f} must have a 'site' column")
        matrices = []
        names = []
        for site_name, group in df.groupby("site"):
            region_cols = [c for c in group.columns if c != "site"]
            matrices.append(group[region_cols].values.astype(np.float64))
            names.append(str(site_name))
        return matrices, names

    # Case 3: single NPY + labels
    data_file = data_dir / "data.npy"
    labels_file = data_dir / "site_labels.npy"
    if data_file.exists() and labels_file.exists():
        log.info("Loading data.npy + site_labels.npy from %s", data_dir)
        X = np.load(data_file).astype(np.float64)
        labels = np.load(labels_file)
        matrices = []
        names = []
        for site in np.unique(labels):
            mask = labels == site
            matrices.append(X[mask])
            names.append(str(site))
        return matrices, names

    raise FileNotFoundError(
        f"No recognized data format in {data_dir}. "
        "Expected: site_*.csv, site_*.npy, a combined CSV with 'site' column, "
        "or data.npy + site_labels.npy."
    )


# ======================================================================
# CALIBRATED SIMULATION
# ======================================================================

# Desikan-Killiany atlas: 34 cortical regions per hemisphere (68 total).
# We use 34 (one hemisphere) following standard ENIGMA cortical analyses.
DESIKAN_REGIONS = [
    "bankssts", "caudalanteriorcingulate", "caudalmiddlefrontal",
    "cuneus", "entorhinal", "fusiform", "inferiorparietal",
    "inferiortemporal", "isthmuscingulate", "lateraloccipital",
    "lateralorbitofrontal", "lingual", "medialorbitofrontal",
    "middletemporal", "parahippocampal", "paracentral",
    "parsopercularis", "parsorbitalis", "parstriangularis",
    "pericalcarine", "postcentral", "posteriorcingulate",
    "precentral", "precuneus", "rostralanteriorcingulate",
    "rostralmiddlefrontal", "superiorfrontal", "superiorparietal",
    "superiortemporal", "supramarginal", "frontalpole",
    "temporalpole", "transversetemporal", "insula",
]


def _make_biological_covariance(d: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a shared biological covariance matrix.

    Eigenvalue decay follows ~1/j^2, consistent with the observation that
    cortical thickness covariance is dominated by a few principal components
    (Grasby et al. 2020 Science).
    """
    # Random orthogonal basis
    Q, _ = linalg.qr(rng.standard_normal((d, d)))
    # Eigenvalue decay ~1/j^2
    eigenvalues = np.array([1.0 / (j + 1) ** 2 for j in range(d)])
    # Scale so total variance is realistic (CT variance ~ 0.1-0.5 mm^2)
    eigenvalues *= 0.3
    return Q @ np.diag(eigenvalues) @ Q.T, Q, eigenvalues


def _make_scanner_rotation(d: int, k: int, strength: float, rng: np.random.Generator) -> np.ndarray:
    """Create a rotation matrix that preferentially rotates the top-k subspace.

    This models systematic scanner effects that distort the principal structure
    of cortical measurements (e.g., site-specific Freesurfer pipeline differences,
    scanner field inhomogeneity, reconstruction algorithm versions).
    """
    # Random skew-symmetric matrix concentrated in the top-k block
    A = np.zeros((d, d))
    # Rotate within the top-k subspace and between top-k and complement
    for _ in range(k):
        i = rng.integers(0, k)
        j = rng.integers(k, d)
        val = rng.standard_normal() * strength
        A[i, j] = val
        A[j, i] = -val
    # Matrix exponential of skew-symmetric = orthogonal
    R = linalg.expm(A)
    return R


def generate_enigma_simulation(
    n_sites: int = 50,
    n_subjects_range: tuple[int, int] = (50, 500),
    d: int = 34,
    k: int = 3,
    n_confounded: int = 5,
    confound_strength: float = 0.4,
    site_bias_std: float = 0.05,
    scanner_scale_std: float = 0.03,
    seed: int = 42,
) -> dict:
    """Generate calibrated multi-site cortical thickness simulation.

    Parameters calibrated to:
    - Grasby et al. 2020 (Science): cortical thickness heritability, covariance structure
    - Fortin et al. 2018 (NeuroImage): ComBat site effect magnitudes
    - Thompson et al. 2020 (Brain): ENIGMA consortium site count/sizes

    Returns dict with data_matrices, site_names, site_labels (clean/confounded),
    and ground truth parameters.
    """
    rng = np.random.default_rng(seed)
    log.info("Generating calibrated ENIGMA simulation: %d sites, d=%d, k=%d", n_sites, d, k)

    # Shared biological covariance
    bio_cov, bio_eigvecs, bio_eigvals = _make_biological_covariance(d, rng)
    bio_mean = rng.standard_normal(d) * 0.1 + 2.5  # ~2.5mm mean cortical thickness

    # Assign site types
    site_labels = ["clean"] * n_sites
    confounded_indices = rng.choice(n_sites, size=n_confounded, replace=False)
    for idx in confounded_indices:
        site_labels[idx] = "confounded"

    # Generate per-site scanner rotation for confounded sites
    scanner_rotations = {}
    for idx in confounded_indices:
        scanner_rotations[idx] = _make_scanner_rotation(d, k, confound_strength, rng)

    data_matrices = []
    site_names = []

    for s in range(n_sites):
        n_subj = rng.integers(n_subjects_range[0], n_subjects_range[1] + 1)

        # Site-specific additive bias (Fortin 2018: ~5% of mean)
        site_bias = rng.standard_normal(d) * site_bias_std

        # Site-specific multiplicative scaling (scanner gain)
        site_scale = 1.0 + rng.standard_normal(d) * scanner_scale_std

        # Generate subjects from biological distribution
        X = rng.multivariate_normal(bio_mean, bio_cov, size=n_subj)

        # Apply site effects (additive + multiplicative)
        X = X * site_scale[np.newaxis, :] + site_bias[np.newaxis, :]

        # Apply systematic scanner distortion for confounded sites
        if site_labels[s] == "confounded":
            R = scanner_rotations[s]
            X = X @ R.T

        data_matrices.append(X)
        site_names.append(f"site_{s:03d}_{site_labels[s]}")

    log.info(
        "Generated %d sites (%d clean, %d confounded), %d total subjects",
        n_sites,
        site_labels.count("clean"),
        site_labels.count("confounded"),
        sum(len(m) for m in data_matrices),
    )

    return {
        "data_matrices": data_matrices,
        "site_names": site_names,
        "site_labels": site_labels,
        "confounded_indices": confounded_indices.tolist(),
        "params": {
            "n_sites": n_sites,
            "d": d,
            "k": k,
            "n_confounded": n_confounded,
            "confound_strength": confound_strength,
            "site_bias_std": site_bias_std,
            "scanner_scale_std": scanner_scale_std,
            "seed": seed,
        },
    }


def generate_berry_phase_simulation(
    n_sites: int = 24,
    n_subjects_per_site: int = 200,
    d: int = 34,
    k: int = 3,
    radius: float = 0.5,
    noise_std: float = 0.05,
    site_bias_std: float = 0.05,
    scanner_scale_std: float = 0.03,
    seed: int = 42,
) -> dict:
    """Generate Berry-phase simulation calibrated to ENIGMA parameters.

    Plants a cyclic inconsistency: each adjacent pair of sites has a small
    subspace rotation (below noise threshold), but the composed transport
    around the full cycle yields a large holonomy.  This is the signature
    that pairwise tests miss and holonomy catches.

    The rotation follows the linked-column construction from the paper's
    Gr(k,d) theory (columns 1-2 of the PCA basis rotate toward shared
    perpendicular directions with pi/2 phase offset), applied to
    ENIGMA-realistic 34-region cortical thickness data with site effects
    calibrated to Fortin 2018.
    """
    rng = np.random.default_rng(seed)
    log.info("Generating Berry phase simulation: %d sites, d=%d, k=%d, r=%.2f", n_sites, d, k, radius)

    bio_cov, bio_eigvecs, bio_eigvals = _make_biological_covariance(d, rng)
    bio_mean = rng.standard_normal(d) * 0.1 + 2.5

    # Two perpendicular directions in the complement of the top-k subspace,
    # expressed in the biological eigenvector basis
    e1 = bio_eigvecs[:, k]
    e2 = bio_eigvecs[:, k + 1]

    data_matrices = []
    site_names = []
    planted_subspaces = []

    for s in range(n_sites):
        theta = 2 * np.pi * s / n_sites

        # Linked-column Berry phase: both columns rotate in the SAME
        # (e1, e2) plane with a pi/2 phase offset.  Each column tilts
        # by angle `radius` from its original direction toward a target
        # that revolves in (e1, e2) as theta sweeps the cycle.
        V = bio_eigvecs[:, :k].copy()
        target_1 = np.cos(theta) * e1 + np.sin(theta) * e2
        target_2 = np.cos(theta + np.pi / 2) * e1 + np.sin(theta + np.pi / 2) * e2
        V[:, 0] = np.cos(radius) * bio_eigvecs[:, 0] + np.sin(radius) * target_1
        V[:, 1] = np.cos(radius) * bio_eigvecs[:, 1] + np.sin(radius) * target_2
        V, _ = linalg.qr(V, mode='economic')
        planted_subspaces.append(V)

        # Generate subjects: use rotated covariance
        rotated_eigvals = bio_eigvals.copy()
        rotated_cov = V @ np.diag(rotated_eigvals[:k]) @ V.T
        # Add noise from remaining components
        complement = bio_eigvecs[:, k:]
        rotated_cov += complement @ np.diag(rotated_eigvals[k:]) @ complement.T

        X = rng.multivariate_normal(bio_mean, rotated_cov, size=n_subjects_per_site)

        # Add realistic ENIGMA-calibrated site effects
        site_bias = rng.standard_normal(d) * site_bias_std
        site_scale = 1.0 + rng.standard_normal(d) * scanner_scale_std
        X = X * site_scale[np.newaxis, :] + site_bias[np.newaxis, :]

        # Add per-subject noise
        X += rng.standard_normal(X.shape) * noise_std

        data_matrices.append(X)
        site_names.append(f"site_{s:03d}")

    # Compute planted holonomy from the clean subspaces (before noise)
    _, planted_holonomy = compose_holonomy(planted_subspaces)

    # Compute max pairwise distance on planted subspaces
    planted_D = pairwise_distance_matrix(planted_subspaces)
    max_planted_dist = float(np.max(planted_D))
    max_adjacent_dist = max(
        geodesic_distance(planted_subspaces[i], planted_subspaces[(i + 1) % n_sites])
        for i in range(n_sites)
    )

    log.info(
        "Planted Berry phase: holonomy=%.4f, max_pairwise=%.4f, max_adjacent=%.4f",
        planted_holonomy, max_planted_dist, max_adjacent_dist,
    )
    log.info("Generated %d sites, %d subjects each", n_sites, n_subjects_per_site)

    return {
        "data_matrices": data_matrices,
        "site_names": site_names,
        "site_labels": None,
        "params": {
            "mode": "berry_phase",
            "n_sites": n_sites,
            "n_subjects_per_site": n_subjects_per_site,
            "d": d,
            "k": k,
            "radius": radius,
            "noise_std": noise_std,
            "site_bias_std": site_bias_std,
            "scanner_scale_std": scanner_scale_std,
            "seed": seed,
            "planted_holonomy": planted_holonomy,
            "max_planted_pairwise": max_planted_dist,
            "max_adjacent_distance": max_adjacent_dist,
        },
    }


# ======================================================================
# CYCLE CONSTRUCTION
# ======================================================================

def build_cycles(
    n_sites: int,
    site_labels: list[str] | None = None,
    rng: np.random.Generator | None = None,
) -> dict[str, list[int]]:
    """Build named cycles of site indices for holonomy testing.

    Returns a dict mapping cycle name to list of site indices.
    When site_labels are available (simulation), builds targeted cycles:
    - all_sites: every site in order
    - clean_only: only clean sites
    - confounded_only: only confounded sites
    - mixed: alternating clean and confounded

    When site_labels are unavailable (real data), builds:
    - all_sites: all sites in order
    - random_half: random subset of half the sites
    """
    cycles = {}

    cycles["all_sites"] = list(range(n_sites))

    if site_labels is not None:
        clean_idx = [i for i, lab in enumerate(site_labels) if lab == "clean"]
        conf_idx = [i for i, lab in enumerate(site_labels) if lab == "confounded"]

        if len(clean_idx) >= 3:
            cycles["clean_only"] = clean_idx
        if len(conf_idx) >= 3:
            cycles["confounded_only"] = conf_idx

        # Mixed cycle: alternate clean and confounded
        if clean_idx and conf_idx:
            mixed = []
            n_pairs = min(len(clean_idx), len(conf_idx))
            for i in range(n_pairs):
                mixed.append(clean_idx[i])
                mixed.append(conf_idx[i])
            if len(mixed) >= 3:
                cycles["mixed"] = mixed
    else:
        if rng is None:
            rng = np.random.default_rng(0)
        half = rng.choice(n_sites, size=n_sites // 2, replace=False).tolist()
        if len(half) >= 3:
            cycles["random_half"] = sorted(half)

    return cycles


# ======================================================================
# PLOTTING
# ======================================================================

def plot_null_distribution(
    observed: float,
    null_dist: list[float],
    p_value: float,
    cycle_name: str,
    outpath: Path,
) -> None:
    """Histogram of null distribution vs observed holonomy."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    null_arr = np.array(null_dist)
    ax.hist(null_arr, bins=50, color="#B0C4DE", edgecolor="#708090", alpha=0.85, label="Null")
    ax.axvline(observed, color="#C0392B", linewidth=2.0, linestyle="--", label=f"Observed ({observed:.3f})")

    pct95 = np.percentile(null_arr, 95)
    ax.axvline(pct95, color="#7F8C8D", linewidth=1.2, linestyle=":", label=f"95th pctl ({pct95:.3f})")

    ax.set_xlabel(r"Holonomy $\|\Phi - I_k\|_F$")
    ax.set_ylabel("Count")
    ax.set_title(f"Holonomy permutation test — {cycle_name}\n(p = {p_value:.4f})")
    ax.legend(frameon=False)

    fig.savefig(outpath)
    plt.close(fig)
    log.info("Saved null distribution plot: %s", outpath)


def plot_distance_heatmap(
    D: np.ndarray,
    site_names: list[str],
    site_labels: list[str] | None,
    outpath: Path,
) -> None:
    """Heatmap of pairwise subspace distances."""
    fig, ax = plt.subplots(figsize=(10, 8.5))

    im = ax.imshow(D, cmap="YlOrRd", aspect="equal")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, label="Geodesic distance")

    # Tick labels: show every nth label to avoid crowding
    n = len(site_names)
    step = max(1, n // 20)
    tick_positions = list(range(0, n, step))
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)

    short_names = [site_names[i].replace("site_", "S") for i in tick_positions]
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(short_names, fontsize=8)

    # Highlight confounded sites with markers on the axes
    if site_labels is not None:
        conf_ticks = [i for i in tick_positions if site_labels[i] == "confounded"]
        for ct in conf_ticks:
            pos = tick_positions.index(ct)
            ax.get_xticklabels()[pos].set_color("#C0392B")
            ax.get_yticklabels()[pos].set_color("#C0392B")

    ax.set_title("Pairwise Grassmannian subspace distances")
    ax.set_xlabel("Site")
    ax.set_ylabel("Site")

    fig.savefig(outpath)
    plt.close(fig)
    log.info("Saved distance heatmap: %s", outpath)


# ======================================================================
# MAIN PIPELINE
# ======================================================================

def run_pipeline(
    data_matrices: list[np.ndarray],
    site_names: list[str],
    site_labels: list[str] | None,
    k: int,
    n_perms: int,
    output_dir: Path,
    do_plot: bool,
    seed: int = 42,
) -> dict:
    """Run the full holonomy analysis pipeline."""
    rng = np.random.default_rng(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    n_sites = len(data_matrices)

    log.info("Running holonomy pipeline: %d sites, k=%d, %d permutations", n_sites, k, n_perms)

    # Step 1: compute PCA subspaces
    log.info("Computing PCA subspaces (k=%d) for %d sites", k, n_sites)
    subspaces = []
    for i, X in enumerate(data_matrices):
        d = X.shape[1]
        if k > d:
            raise ValueError(f"k={k} exceeds dimensionality d={d} of site {site_names[i]}")
        if X.shape[0] < k:
            log.warning("Site %s has only %d subjects (< k=%d), subspace may be degenerate",
                        site_names[i], X.shape[0], k)
        subspaces.append(pca_subspace(X, k))

    # Step 2: pairwise distances
    log.info("Computing pairwise subspace distances")
    D = pairwise_distance_matrix(subspaces)
    pair_angles = pairwise_principal_angles(subspaces)

    max_dist = float(np.max(D))
    mean_dist = float(np.mean(D[np.triu_indices(n_sites, k=1)]))
    log.info("Pairwise distances: max=%.4f, mean=%.4f", max_dist, mean_dist)

    # Step 3: build cycles and run holonomy tests
    cycles = build_cycles(n_sites, site_labels, rng)
    log.info("Testing %d cycles: %s", len(cycles), list(cycles.keys()))

    cycle_results = {}
    for name, indices in cycles.items():
        log.info("Cycle '%s': %d sites", name, len(indices))

        cycle_subspaces = [subspaces[i] for i in indices]
        _, observed_holonomy = compose_holonomy(cycle_subspaces)
        log.info("  Observed holonomy: %.4f", observed_holonomy)

        # Permutation test
        perm_result = holonomy_permutation_test(
            data_matrices, indices, k, n_perms, rng,
        )
        log.info("  p-value: %.4f (null mean=%.4f, std=%.4f)",
                 perm_result["p_value"], perm_result["null_mean"], perm_result["null_std"])

        cycle_results[name] = {
            "n_sites_in_cycle": len(indices),
            "site_indices": indices,
            **perm_result,
        }

        # Plot null distribution
        if do_plot:
            plot_null_distribution(
                perm_result["observed_holonomy"],
                perm_result["null_distribution"],
                perm_result["p_value"],
                name,
                output_dir / f"holonomy_null_{name}.png",
            )

    # Step 4: assemble results
    results = {
        "timestamp": datetime.now().isoformat(),
        "n_sites": n_sites,
        "k": k,
        "n_perms": n_perms,
        "pairwise_distances": {
            "max": max_dist,
            "mean": mean_dist,
        },
        "cycles": cycle_results,
    }

    # Strip null distributions from JSON (large arrays) but keep summary stats
    results_for_json = json.loads(json.dumps(results))
    for cname in results_for_json["cycles"]:
        results_for_json["cycles"][cname].pop("null_distribution", None)

    results_path = output_dir / "holonomy_results.json"
    with open(results_path, "w") as f:
        json.dump(results_for_json, f, indent=2)
    log.info("Saved results: %s", results_path)

    # Save full null distributions as NPY
    for cname, cdata in cycle_results.items():
        np.save(output_dir / f"null_dist_{cname}.npy", np.array(cdata["null_distribution"]))

    # Plot distance heatmap
    if do_plot:
        plot_distance_heatmap(D, site_names, site_labels, output_dir / "subspace_distances.png")

    return results


# ======================================================================
# CLI
# ======================================================================

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ENIGMA holonomy pipeline: detect global inconsistency in multi-site neuroimaging subspaces",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python pipeline.py --simulate --n-sites 50 --k 3 --output results/ --plot
  python pipeline.py --data-dir /path/to/enigma/ --k 5 --output results/ --plot
""",
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--data-dir", type=Path, help="Directory containing site-level data files")
    mode.add_argument("--simulate", action="store_true", help="Run calibrated ENIGMA simulation (independent confounds)")
    mode.add_argument("--berry-phase", action="store_true",
                      help="Run Berry phase simulation (cyclic inconsistency planted in ENIGMA-realistic data)")
    mode.add_argument("--berry-boundary", action="store_true",
                      help="Map detection boundary: Berry phase vs null across sample sizes")

    parser.add_argument("--n-sites", type=int, default=50, help="Number of sites (default: 50)")
    parser.add_argument("--n-subjects", type=int, default=200,
                        help="Subjects per site (berry-phase mode, default: 200)")
    parser.add_argument("--n-trials", type=int, default=50,
                        help="Trials per sample size (berry-boundary mode, default: 50)")
    parser.add_argument("--n-confounded", type=int, default=5,
                        help="Number of confounded sites (simulate mode, default: 5)")
    parser.add_argument("--confound-strength", type=float, default=0.4,
                        help="Scanner distortion strength (simulate mode, default: 0.4)")
    parser.add_argument("--radius", type=float, default=0.5,
                        help="Berry phase radius (berry-phase mode, default: 0.5)")
    parser.add_argument("--noise-std", type=float, default=0.05,
                        help="Per-subject noise std (berry-phase mode, default: 0.05)")
    parser.add_argument("--k", type=int, default=3, help="PCA subspace dimension (default: 3)")
    parser.add_argument("--n-perms", type=int, default=1000, help="Number of permutations (default: 1000)")
    parser.add_argument("--output", type=Path, default=Path("results"), help="Output directory (default: results/)")
    parser.add_argument("--plot", action="store_true", help="Generate plots")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)

    log.info("=== ENIGMA Holonomy Pipeline ===")
    if args.berry_boundary:
        mode_str = "Berry phase boundary analysis"
    elif args.berry_phase:
        mode_str = "Berry phase simulation"
    elif args.simulate:
        mode_str = "ENIGMA simulation"
    else:
        mode_str = f"real data ({args.data_dir})"
    log.info("Mode: %s", mode_str)
    log.info("k=%d, n_perms=%d, seed=%d", args.k, args.n_perms, args.seed)

    if args.berry_boundary:
        results = berry_phase_boundary_analysis(
            n_sites=args.n_sites,
            d=34,
            k=args.k,
            radius=args.radius,
            noise_std=args.noise_std,
            n_trials=args.n_trials,
            seed=args.seed,
            output_dir=args.output,
            do_plot=args.plot,
        )
        log.info("=== Boundary Summary ===")
        for n_subj, data in sorted(results["by_sample_size"].items(), key=lambda x: int(x[0])):
            log.info(
                "  n=%s: sep=%.1fσ, power=%.2f, recovery=%.0f%%",
                n_subj, data["separation_sigma"], data["detection_power"],
                data["recovery_fraction"] * 100,
            )
        return results

    if args.berry_phase:
        sim = generate_berry_phase_simulation(
            n_sites=args.n_sites,
            n_subjects_per_site=args.n_subjects,
            d=34,
            k=args.k,
            radius=args.radius,
            noise_std=args.noise_std,
            seed=args.seed,
        )
        data_matrices = sim["data_matrices"]
        site_names = sim["site_names"]
        site_labels = sim["site_labels"]

        args.output.mkdir(parents=True, exist_ok=True)
        with open(args.output / "simulation_params.json", "w") as f:
            json.dump(sim["params"], f, indent=2)
    elif args.simulate:
        sim = generate_enigma_simulation(
            n_sites=args.n_sites,
            d=34,
            k=args.k,
            n_confounded=args.n_confounded,
            confound_strength=args.confound_strength,
            seed=args.seed,
        )
        data_matrices = sim["data_matrices"]
        site_names = sim["site_names"]
        site_labels = sim["site_labels"]

        args.output.mkdir(parents=True, exist_ok=True)
        with open(args.output / "simulation_params.json", "w") as f:
            json.dump(sim["params"], f, indent=2)
    else:
        data_matrices, site_names = load_site_data(args.data_dir)
        site_labels = None

    results = run_pipeline(
        data_matrices=data_matrices,
        site_names=site_names,
        site_labels=site_labels,
        k=args.k,
        n_perms=args.n_perms,
        output_dir=args.output,
        do_plot=args.plot,
        seed=args.seed,
    )

    # Summary
    log.info("=== Summary ===")
    for cname, cdata in results["cycles"].items():
        log.info(
            "  %s: holonomy=%.4f, p=%.4f",
            cname, cdata["observed_holonomy"], cdata["p_value"],
        )

    return results


if __name__ == "__main__":
    main()
