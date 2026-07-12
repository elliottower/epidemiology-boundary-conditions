"""Batch 4 sensitivity experiments for Papers A and B.

Preregistered in PREREGISTRATION.md — run AFTER committing that file.
All experiments are CPU-only simulations with planted ground truth.
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import chi2, pearsonr
from tqdm import tqdm

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def compute_cochran_q(betas: np.ndarray, ses: np.ndarray):
    w = 1.0 / (ses ** 2)
    beta_pooled = np.sum(w * betas) / np.sum(w)
    Q = float(np.sum(w * (betas - beta_pooled) ** 2))
    k = len(betas)
    df = k - 1
    p = float(1.0 - chi2.cdf(Q, df))
    I2 = float(max(0.0, (Q - df) / Q)) if Q > 0 else 0.0
    C = np.sum(w) - np.sum(w ** 2) / np.sum(w)
    tau2 = float(max(0.0, (Q - df) / C)) if C > 0 else 0.0
    return {"Q": Q, "p": p, "I2": I2, "tau2": tau2, "beta_pooled": float(beta_pooled), "k": k, "df": df}


# ======================================================================
# Experiment 1: Heterogeneity-Ratio Sweep (Paper B)
# ======================================================================

def exp1_heterogeneity_ratio_sweep():
    """Sweep mechanism/downstream variance ratio to show smooth boundary."""
    print(f"[{datetime.now():%H:%M:%S}] Exp 1: Heterogeneity-ratio sweep")
    rng = np.random.default_rng(2026_07_12_01)

    K = 8
    n_reps = 200
    sigma_down = 0.01
    ratios = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]

    base_mean = 0.30
    se_per_stratum = 0.03

    results = []
    for R in tqdm(ratios, desc="  ratio sweep"):
        sigma_mech = R * sigma_down
        global_rejects = 0
        peredge_correct = 0

        for _ in range(n_reps):
            mech_coeffs_fwd = base_mean + rng.normal(0, sigma_mech, K)
            mech_coeffs_rev = base_mean + rng.normal(0, sigma_mech, K)
            down_coeffs_1 = base_mean + rng.normal(0, sigma_down, K)
            down_coeffs_2 = base_mean + rng.normal(0, sigma_down, K)

            betas_mech_fwd = mech_coeffs_fwd + rng.normal(0, se_per_stratum, K)
            betas_mech_rev = mech_coeffs_rev + rng.normal(0, se_per_stratum, K)
            betas_down_1 = down_coeffs_1 + rng.normal(0, se_per_stratum, K)
            betas_down_2 = down_coeffs_2 + rng.normal(0, se_per_stratum, K)
            ses = np.full(K, se_per_stratum)

            q_mech_fwd = compute_cochran_q(betas_mech_fwd, ses)
            q_mech_rev = compute_cochran_q(betas_mech_rev, ses)
            q_down_1 = compute_cochran_q(betas_down_1, ses)
            q_down_2 = compute_cochran_q(betas_down_2, ses)

            all_Qs = [q_mech_fwd["Q"], q_mech_rev["Q"], q_down_1["Q"], q_down_2["Q"]]
            global_Q = sum(all_Qs)
            global_p = float(1.0 - chi2.cdf(global_Q, 4 * (K - 1)))
            if global_p < 0.05:
                global_rejects += 1

            mech_detected = q_mech_fwd["p"] < 0.0125 and q_mech_rev["p"] < 0.0125
            down_stable = q_down_1["p"] > 0.0125 and q_down_2["p"] > 0.0125
            if mech_detected and down_stable:
                peredge_correct += 1

        results.append({
            "ratio": R,
            "sigma_mech": sigma_mech,
            "sigma_down": sigma_down,
            "global_power": global_rejects / n_reps,
            "peredge_correct": peredge_correct / n_reps,
            "peredge_advantage": (peredge_correct - global_rejects) / n_reps,
        })
        print(f"    R={R:>5}: global={results[-1]['global_power']:.3f}, "
              f"per-edge={results[-1]['peredge_correct']:.3f}")

    with open(RESULTS_DIR / "exp1_ratio_sweep.json", "w") as f:
        json.dump({"experiment": "heterogeneity_ratio_sweep", "results": results}, f, indent=2)
    print(f"[{datetime.now():%H:%M:%S}] Exp 1 complete\n")
    return results


# ======================================================================
# Experiment 2: Partial Confounding Degradation (Paper B)
# ======================================================================

def exp2_partial_confounding():
    """Degrade confounding from 100% to partial, measure AUROC of partial correlation."""
    print(f"[{datetime.now():%H:%M:%S}] Exp 2: Partial confounding degradation")
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(2026_07_12_02)

    n_real = 10
    n_confounded = 10
    n_patients = 3000
    n_reps = 200
    fractions = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

    results = []
    for f in tqdm(fractions, desc="  confound fraction"):
        aurocs = []
        for _ in range(n_reps):
            disease_severity = rng.normal(0, 1, n_patients)
            roi_size = 100 - 10 * disease_severity + rng.normal(0, 8, n_patients)
            roi_size = np.clip(roi_size, 10, 200)
            disability = 2 * disease_severity + rng.normal(0, 0.8, n_patients)

            is_real = []
            partial_corrs = []

            for i in range(n_real):
                coeff = rng.uniform(1.0, 3.0) * rng.choice([-1, 1])
                bio = coeff * disease_severity + rng.normal(0, 0.8, n_patients)
                is_real.append(1)
                X = np.column_stack([roi_size, np.ones(n_patients)])
                bio_resid = bio - X @ np.linalg.lstsq(X, bio, rcond=None)[0]
                dis_resid = disability - X @ np.linalg.lstsq(X, disability, rcond=None)[0]
                partial_corrs.append(abs(pearsonr(bio_resid, dis_resid)[0]))

            for i in range(n_confounded):
                coeff_confound = rng.uniform(0.3, 1.5)
                coeff_real = rng.uniform(1.0, 3.0) * rng.choice([-1, 1])
                bio = f * coeff_confound * roi_size + (1 - f) * coeff_real * disease_severity
                bio += rng.normal(0, 0.8, n_patients)
                is_real.append(0)
                X = np.column_stack([roi_size, np.ones(n_patients)])
                bio_resid = bio - X @ np.linalg.lstsq(X, bio, rcond=None)[0]
                dis_resid = disability - X @ np.linalg.lstsq(X, disability, rcond=None)[0]
                partial_corrs.append(abs(pearsonr(bio_resid, dis_resid)[0]))

            is_real = np.array(is_real)
            partial_corrs = np.array(partial_corrs)
            aurocs.append(roc_auc_score(is_real, partial_corrs))

        results.append({
            "confound_fraction": f,
            "auroc_mean": float(np.mean(aurocs)),
            "auroc_std": float(np.std(aurocs)),
            "auroc_ci_lo": float(np.quantile(aurocs, 0.025)),
            "auroc_ci_hi": float(np.quantile(aurocs, 0.975)),
        })
        print(f"    f={f:.1f}: AUROC={results[-1]['auroc_mean']:.3f} "
              f"[{results[-1]['auroc_ci_lo']:.3f}, {results[-1]['auroc_ci_hi']:.3f}]")

    with open(RESULTS_DIR / "exp2_partial_confounding.json", "w") as f:
        json.dump({"experiment": "partial_confounding", "results": results}, f, indent=2)
    print(f"[{datetime.now():%H:%M:%S}] Exp 2 complete\n")
    return results


# ======================================================================
# Experiment 3: Minimum-Detectable Heterogeneity (Paper A)
# ======================================================================

def exp3_minimum_detectable_heterogeneity():
    """Compute minimum between-stratum SD detectable at 80% power with 3-4 strata."""
    print(f"[{datetime.now():%H:%M:%S}] Exp 3: Minimum-detectable heterogeneity")
    rng = np.random.default_rng(2026_07_12_03)

    pairs_path = Path(__file__).resolve().parent.parent / "batch3_expansion" / "01_systematic_mr" / "data" / "pairs_curated.json"
    with open(pairs_path) as fp:
        pairs = json.load(fp)

    all_ses = []
    for p in pairs:
        for s in p["strata"]:
            all_ses.append(s["se"])
    all_ses = np.array(all_ses)
    median_se = float(np.median(all_ses))
    q25_se = float(np.quantile(all_ses, 0.25))
    q75_se = float(np.quantile(all_ses, 0.75))
    print(f"  Empirical SE: median={median_se:.3f}, IQR=[{q25_se:.3f}, {q75_se:.3f}]")

    n_reps = 2000
    beta_0 = 0.20
    tau_values = np.linspace(0.01, 0.50, 25)

    results = []
    for K in [3, 4]:
        se_set = np.array([median_se] * K)
        power_curve = []
        for tau in tqdm(tau_values, desc=f"  K={K}"):
            rejects = 0
            for _ in range(n_reps):
                betas = beta_0 + rng.normal(0, tau, K)
                betas_obs = betas + rng.normal(0, se_set)
                stats = compute_cochran_q(betas_obs, se_set)
                if stats["p"] < 0.05:
                    rejects += 1
            power_curve.append({
                "tau": float(tau),
                "power": rejects / n_reps,
            })

        tau_80 = None
        for pt in power_curve:
            if pt["power"] >= 0.80:
                tau_80 = pt["tau"]
                break

        results.append({
            "K": K,
            "se_used": float(median_se),
            "tau_80": tau_80,
            "tau_80_over_se": tau_80 / median_se if tau_80 else None,
            "power_curve": power_curve,
        })
        print(f"    K={K}: τ_80 = {tau_80:.3f} (τ/SE = {tau_80/median_se:.2f})" if tau_80 else f"    K={K}: τ_80 not reached")

    observed_taus = []
    for p in pairs:
        betas = np.array([s["beta"] for s in p["strata"]])
        ses = np.array([s["se"] for s in p["strata"]])
        stats = compute_cochran_q(betas, ses)
        observed_taus.append({
            "id": p["id"],
            "expected": p["expected"],
            "k": len(p["strata"]),
            "tau": np.sqrt(stats["tau2"]),
            "Q": stats["Q"],
            "p": stats["p"],
        })

    tau_80_k3 = results[0]["tau_80"]
    tau_80_k4 = results[1]["tau_80"]
    for ot in observed_taus:
        threshold = tau_80_k3 if ot["k"] == 3 else tau_80_k4
        ot["above_tau80"] = bool(ot["tau"] > threshold) if threshold else None

    detected_above = sum(1 for ot in observed_taus
                         if ot["expected"] == "non-transport" and ot["p"] < 0.05
                         and ot.get("above_tau80", False))
    detected_total = sum(1 for ot in observed_taus
                         if ot["expected"] == "non-transport" and ot["p"] < 0.05)
    missed_below = sum(1 for ot in observed_taus
                       if ot["expected"] == "non-transport" and ot["p"] >= 0.05
                       and ot.get("above_tau80") is not None and not ot["above_tau80"])
    missed_total = sum(1 for ot in observed_taus
                       if ot["expected"] == "non-transport" and ot["p"] >= 0.05)

    validation = {
        "detected_above_tau80": f"{detected_above}/{detected_total}",
        "missed_below_tau80": f"{missed_below}/{missed_total}",
    }
    print(f"  Validation: detected above τ_80: {validation['detected_above_tau80']}, "
          f"missed below τ_80: {validation['missed_below_tau80']}")

    output = {
        "experiment": "minimum_detectable_heterogeneity",
        "empirical_se": {"median": median_se, "q25": q25_se, "q75": q75_se},
        "results": results,
        "observed_taus": observed_taus,
        "validation": validation,
    }
    with open(RESULTS_DIR / "exp3_min_detectable_het.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"[{datetime.now():%H:%M:%S}] Exp 3 complete\n")
    return output


# ======================================================================
# Experiment 4: LD-Attributable Heterogeneity (Paper A)
# ======================================================================

def exp4_ld_heterogeneity():
    """Simulate LD-only heterogeneity and compare Q to observed values."""
    print(f"[{datetime.now():%H:%M:%S}] Exp 4: LD-attributable heterogeneity simulation")
    rng = np.random.default_rng(2026_07_12_04)

    pairs_path = Path(__file__).resolve().parent.parent / "batch3_expansion" / "01_systematic_mr" / "data" / "pairs_curated.json"
    with open(pairs_path) as fp:
        pairs = json.load(fp)

    transport_Qs = [compute_cochran_q(
        np.array([s["beta"] for s in p["strata"]]),
        np.array([s["se"] for s in p["strata"]])
    )["Q"] for p in pairs if p["expected"] == "transport"]

    nontransport_Qs = [compute_cochran_q(
        np.array([s["beta"] for s in p["strata"]]),
        np.array([s["se"] for s in p["strata"]])
    )["Q"] for p in pairs if p["expected"] == "non-transport"]

    print(f"  Observed: transport Q median={np.median(transport_Qs):.3f}, "
          f"non-transport Q median={np.median(nontransport_Qs):.3f}")

    K = 4
    n_reps = 500
    n_gwas = 50000
    beta_causal = 0.20
    gamma_0 = 0.30
    sigma_ld_fracs = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]

    results = []
    for sigma_frac in tqdm(sigma_ld_fracs, desc="  LD sweep"):
        sigma_ld = sigma_frac * gamma_0
        Q_values = []
        for _ in range(n_reps):
            gammas = gamma_0 + rng.normal(0, sigma_ld, K) if sigma_ld > 0 else np.full(K, gamma_0)
            gammas = np.clip(gammas, 0.05, 1.0)

            betas_mr = np.zeros(K)
            ses_mr = np.zeros(K)
            for k_idx in range(K):
                x = rng.normal(0, 1, n_gwas)
                z = gammas[k_idx] * x + rng.normal(0, np.sqrt(1 - gammas[k_idx]**2), n_gwas)
                y = beta_causal * x + rng.normal(0, 1, n_gwas)
                cov_zy = np.cov(z, y)[0, 1]
                var_z = np.var(z)
                beta_wald = cov_zy / var_z
                se_wald = 1.0 / (np.abs(gammas[k_idx]) * np.sqrt(n_gwas))
                betas_mr[k_idx] = beta_wald
                ses_mr[k_idx] = se_wald

            stats = compute_cochran_q(betas_mr, ses_mr)
            Q_values.append(stats["Q"])

        results.append({
            "sigma_ld_frac": sigma_frac,
            "sigma_ld": float(sigma_ld),
            "Q_median": float(np.median(Q_values)),
            "Q_mean": float(np.mean(Q_values)),
            "Q_95th": float(np.quantile(Q_values, 0.95)),
            "Q_99th": float(np.quantile(Q_values, 0.99)),
            "Q_max": float(np.max(Q_values)),
            "frac_above_7": float(np.mean(np.array(Q_values) > 7.0)),
        })
        print(f"    σ_LD/γ₀={sigma_frac:.2f}: Q median={results[-1]['Q_median']:.3f}, "
              f"95th={results[-1]['Q_95th']:.3f}, frac>7={results[-1]['frac_above_7']:.3f}")

    output = {
        "experiment": "ld_heterogeneity",
        "observed_transport_Q": {"median": float(np.median(transport_Qs)),
                                  "mean": float(np.mean(transport_Qs)),
                                  "max": float(np.max(transport_Qs))},
        "observed_nontransport_Q": {"median": float(np.median(nontransport_Qs)),
                                     "mean": float(np.mean(nontransport_Qs)),
                                     "min": float(np.min(nontransport_Qs))},
        "results": results,
    }
    with open(RESULTS_DIR / "exp4_ld_heterogeneity.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"[{datetime.now():%H:%M:%S}] Exp 4 complete\n")
    return output


# ======================================================================

def main():
    print("=" * 70)
    print("Batch 4 Sensitivity Experiments")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 70)

    exp = sys.argv[1] if len(sys.argv) > 1 else "all"

    if exp in ("all", "1"):
        exp1_heterogeneity_ratio_sweep()
    if exp in ("all", "2"):
        exp2_partial_confounding()
    if exp in ("all", "3"):
        exp3_minimum_detectable_heterogeneity()
    if exp in ("all", "4"):
        exp4_ld_heterogeneity()

    print("=" * 70)
    print(f"Finished: {datetime.now().isoformat()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
