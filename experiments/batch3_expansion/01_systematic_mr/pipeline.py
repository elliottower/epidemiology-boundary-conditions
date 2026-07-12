"""Systematic Cochran's Q-test pipeline for MR transportability classification."""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2, ncx2
from tqdm import tqdm

matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def compute_cochran_q(betas: np.ndarray, ses: np.ndarray):
    """Fixed-effects meta-analysis with Cochran's Q heterogeneity test.

    Returns dict with Q, p, I2, tau2, beta_pooled, and power.
    """
    w = 1.0 / (ses ** 2)
    beta_pooled = np.sum(w * betas) / np.sum(w)
    Q = float(np.sum(w * (betas - beta_pooled) ** 2))
    k = len(betas)
    df = k - 1
    p = float(1.0 - chi2.cdf(Q, df))
    I2 = float(max(0.0, (Q - df) / Q)) if Q > 0 else 0.0

    C = np.sum(w) - np.sum(w ** 2) / np.sum(w)
    tau2 = float(max(0.0, (Q - df) / C)) if C > 0 else 0.0

    return {
        "Q": Q,
        "df": df,
        "p": p,
        "I2": I2,
        "tau2": tau2,
        "beta_pooled": float(beta_pooled),
        "k": k,
    }


def compute_power(Q: float, df: int, alpha: float) -> float:
    """Power of the Q-test via noncentral chi-squared."""
    lambda_ncp = max(0.0, Q - df)
    if lambda_ncp == 0.0:
        return alpha
    crit = chi2.ppf(1 - alpha, df)
    return float(1.0 - ncx2.cdf(crit, df, lambda_ncp))


def classify_pair(p_value: float, alpha: float) -> str:
    return "non-transport" if p_value < alpha else "transport"


def run_pipeline(pairs: list, alpha: float) -> list:
    results = []
    for pair in tqdm(pairs, desc="Computing Q-tests"):
        betas = np.array([s["beta"] for s in pair["strata"]])
        ses = np.array([s["se"] for s in pair["strata"]])

        stats = compute_cochran_q(betas, ses)
        power = compute_power(stats["Q"], stats["df"], alpha)
        verdict = classify_pair(stats["p"], alpha)
        correct = verdict == pair["expected"]

        results.append({
            "id": pair["id"],
            "exposure": pair["exposure"],
            "outcome": pair["outcome"],
            "domain": pair["domain"],
            "expected": pair["expected"],
            "verdict": verdict,
            "correct": correct,
            **stats,
            "power": power,
            "approximate": any(s.get("approximate", False) for s in pair["strata"]),
        })
    return results


def evaluate_accuracy(results: list):
    n = len(results)
    n_correct = sum(r["correct"] for r in results)

    tp = sum(1 for r in results if r["expected"] == "non-transport" and r["verdict"] == "non-transport")
    fn = sum(1 for r in results if r["expected"] == "non-transport" and r["verdict"] == "transport")
    fp = sum(1 for r in results if r["expected"] == "transport" and r["verdict"] == "non-transport")
    tn = sum(1 for r in results if r["expected"] == "transport" and r["verdict"] == "transport")

    accuracy = n_correct / n if n > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "n_correct": n_correct,
        "n_total": n,
    }


def bootstrap_ci(results: list, n_boot: int = 2000, ci: float = 0.95):
    """Bootstrap 95% CIs for accuracy, sensitivity, specificity."""
    rng = np.random.default_rng()
    arr = np.array([(r["expected"], r["verdict"]) for r in results], dtype=object)

    def _metric(indices):
        sub = arr[indices]
        n = len(sub)
        correct = sum(1 for e, v in sub if e == v)
        tp = sum(1 for e, v in sub if e == "non-transport" and v == "non-transport")
        fn = sum(1 for e, v in sub if e == "non-transport" and v == "transport")
        fp = sum(1 for e, v in sub if e == "transport" and v == "non-transport")
        tn = sum(1 for e, v in sub if e == "transport" and v == "transport")
        acc = correct / n if n > 0 else 0.0
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        return acc, sens, spec

    accs, senss, specs = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, len(arr), size=len(arr))
        a, s, sp = _metric(idx)
        accs.append(a)
        senss.append(s)
        specs.append(sp)

    lo = (1 - ci) / 2
    hi = 1 - lo
    return {
        "accuracy_ci": (float(np.quantile(accs, lo)), float(np.quantile(accs, hi))),
        "sensitivity_ci": (float(np.quantile(senss, lo)), float(np.quantile(senss, hi))),
        "specificity_ci": (float(np.quantile(specs, lo)), float(np.quantile(specs, hi))),
    }


def alpha_sweep(results_by_alpha: dict):
    """Sweep alpha from 0.001 to 0.5 and return per-alpha metrics."""
    sweep = []
    for alpha_val, res in sorted(results_by_alpha.items()):
        metrics = evaluate_accuracy(res)
        sweep.append({"alpha": alpha_val, **metrics})
    return sweep


def plot_roc_like(sweep: list, output_dir: Path):
    """Sensitivity vs 1-specificity across alpha thresholds."""
    alphas = [s["alpha"] for s in sweep]
    sens = [s["sensitivity"] for s in sweep]
    one_minus_spec = [1 - s["specificity"] for s in sweep]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(one_minus_spec, sens, "o-", color="#2c3e50", markersize=5, linewidth=1.5)

    for i, a in enumerate(alphas):
        if a in (0.01, 0.05, 0.10, 0.20):
            ax.annotate(
                f"α={a}",
                (one_minus_spec[i], sens[i]),
                textcoords="offset points",
                xytext=(8, -4),
                fontsize=8,
                color="#7f8c8d",
            )

    ax.plot([0, 1], [0, 1], "--", color="#bdc3c7", linewidth=0.8)
    ax.set_xlabel("1 − Specificity (false positive rate)")
    ax.set_ylabel("Sensitivity (true positive rate)")
    ax.set_title("Q-test classification across α thresholds")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    for ext in ("png", "pdf"):
        fig.savefig(output_dir / f"roc_like_alpha_sweep.{ext}", dpi=200)
    plt.close(fig)
    log.info("Saved ROC-like alpha sweep plot")


def plot_dot_plot(results: list, alpha: float, output_dir: Path):
    """Dot plot of Q p-values for all pairs, colored by expected label."""
    sorted_res = sorted(results, key=lambda r: r["p"])

    fig, ax = plt.subplots(figsize=(8, max(6, len(sorted_res) * 0.22)))

    y_pos = np.arange(len(sorted_res))
    colors = []
    markers = []
    for r in sorted_res:
        if r["expected"] == "non-transport":
            colors.append("#e74c3c")
        else:
            colors.append("#2980b9")
        markers.append("x" if not r["correct"] else "o")

    neg_log_p = [-np.log10(max(r["p"], 1e-20)) for r in sorted_res]

    for i, (y, x, c, m) in enumerate(zip(y_pos, neg_log_p, colors, markers)):
        kwargs = {"c": c, "marker": m, "s": 40, "zorder": 3, "linewidths": 0.5}
        if m != "x":
            kwargs["edgecolors"] = c
        ax.scatter(x, y, **kwargs)

    ax.axvline(-np.log10(alpha), color="#7f8c8d", linestyle="--", linewidth=1, label=f"α = {alpha}")

    labels = [f"{r['id']} ({r['domain']})" for r in sorted_res]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("−log₁₀(p)")
    ax.set_title(f"Cochran's Q p-values (n={len(results)} pairs, α={alpha})")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#e74c3c", markersize=8, label="Non-transport (expected)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2980b9", markersize=8, label="Transport (expected)"),
        Line2D([0], [0], marker="x", color="k", markersize=8, label="Misclassified", linestyle="None"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)
    fig.tight_layout()

    for ext in ("png", "pdf"):
        fig.savefig(output_dir / f"dot_plot_all_pairs.{ext}", dpi=200)
    plt.close(fig)
    log.info("Saved dot plot")


def plot_power_scatter(results: list, output_dir: Path):
    """Power vs Q scatter, sized by number of strata."""
    fig, ax = plt.subplots(figsize=(6, 5))

    for r in results:
        c = "#e74c3c" if r["expected"] == "non-transport" else "#2980b9"
        m = "x" if not r["correct"] else "o"
        kwargs = {"c": c, "marker": m, "s": r["k"] * 25, "linewidths": 0.5, "zorder": 3}
        if m != "x":
            kwargs["edgecolors"] = c
        ax.scatter(r["Q"], r["power"], **kwargs)

    ax.set_xlabel("Cochran's Q")
    ax.set_ylabel("Power (1 − β)")
    ax.set_title("Power vs Q statistic")
    ax.axhline(0.80, color="#7f8c8d", linestyle="--", linewidth=0.8, label="80% power")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8)
    fig.tight_layout()

    for ext in ("png", "pdf"):
        fig.savefig(output_dir / f"power_vs_Q.{ext}", dpi=200)
    plt.close(fig)
    log.info("Saved power vs Q plot")


def dersimonian_laird(betas: np.ndarray, ses: np.ndarray):
    """DerSimonian-Laird random-effects meta-analysis.

    Returns dict with beta_re, se_re, tau2, Q, I2.
    """
    w = 1.0 / (ses ** 2)
    beta_fe = np.sum(w * betas) / np.sum(w)
    k = len(betas)
    df = k - 1
    Q = float(np.sum(w * (betas - beta_fe) ** 2))
    C = np.sum(w) - np.sum(w ** 2) / np.sum(w)
    tau2 = float(max(0.0, (Q - df) / C)) if C > 0 else 0.0
    I2 = float(max(0.0, (Q - df) / Q)) if Q > 0 else 0.0

    w_re = 1.0 / (ses ** 2 + tau2)
    beta_re = float(np.sum(w_re * betas) / np.sum(w_re))
    se_re = float(1.0 / np.sqrt(np.sum(w_re)))

    return {
        "beta_fe": float(beta_fe),
        "beta_re": beta_re,
        "se_re": se_re,
        "tau2": tau2,
        "Q": Q,
        "I2": I2,
        "k": k,
    }


def sensitivity_exclude_approximate(pairs: list, alpha: float):
    """Report accuracy separately for exact-only vs approximate-containing pairs."""
    exact = [p for p in pairs if not any(s.get("approximate", False) for s in p["strata"])]
    approx = [p for p in pairs if any(s.get("approximate", False) for s in p["strata"])]

    exact_results = run_pipeline(exact, alpha) if exact else []
    approx_results = run_pipeline(approx, alpha) if approx else []

    exact_metrics = evaluate_accuracy(exact_results) if exact_results else None
    approx_metrics = evaluate_accuracy(approx_results) if approx_results else None

    return {
        "n_exact": len(exact),
        "n_approximate": len(approx),
        "exact_ids": [p["id"] for p in exact],
        "exact_metrics": exact_metrics,
        "approximate_metrics": approx_metrics,
    }


def sensitivity_leave_one_out(pairs: list, alpha: float):
    """For each pair, drop each stratum and check if classification changes."""
    loo_results = []
    for pair in tqdm(pairs, desc="Leave-one-out"):
        strata = pair["strata"]
        if len(strata) <= 2:
            loo_results.append({
                "id": pair["id"],
                "k": len(strata),
                "stable": True,
                "note": "only 2 strata, LOO gives k=1 (untestable)",
            })
            continue

        betas_full = np.array([s["beta"] for s in strata])
        ses_full = np.array([s["se"] for s in strata])
        full_stats = compute_cochran_q(betas_full, ses_full)
        full_verdict = classify_pair(full_stats["p"], alpha)

        dropped = []
        for i in range(len(strata)):
            betas_loo = np.delete(betas_full, i)
            ses_loo = np.delete(ses_full, i)
            loo_stats = compute_cochran_q(betas_loo, ses_loo)
            loo_verdict = classify_pair(loo_stats["p"], alpha)
            dropped.append({
                "dropped_stratum": strata[i]["name"],
                "Q_loo": loo_stats["Q"],
                "p_loo": loo_stats["p"],
                "verdict_loo": loo_verdict,
                "changed": loo_verdict != full_verdict,
            })

        n_changed = sum(1 for d in dropped if d["changed"])
        loo_results.append({
            "id": pair["id"],
            "expected": pair["expected"],
            "k": len(strata),
            "full_Q": full_stats["Q"],
            "full_p": full_stats["p"],
            "full_verdict": full_verdict,
            "n_changed": n_changed,
            "stable": n_changed == 0,
            "dropped": dropped,
        })

    n_stable = sum(1 for r in loo_results if r["stable"])
    return {
        "n_pairs": len(pairs),
        "n_stable": n_stable,
        "pct_stable": n_stable / len(pairs) if pairs else 0.0,
        "per_pair": loo_results,
    }


def sensitivity_random_effects(pairs: list, alpha: float):
    """Compare fixed-effects and random-effects pooled estimates."""
    re_results = []
    for pair in tqdm(pairs, desc="Random-effects comparison"):
        betas = np.array([s["beta"] for s in pair["strata"]])
        ses = np.array([s["se"] for s in pair["strata"]])
        re = dersimonian_laird(betas, ses)
        fe_stats = compute_cochran_q(betas, ses)
        fe_verdict = classify_pair(fe_stats["p"], alpha)

        se_ratio = re["se_re"] / (1.0 / np.sqrt(np.sum(1.0 / ses ** 2))) if re["tau2"] > 0 else 1.0

        re_results.append({
            "id": pair["id"],
            "expected": pair["expected"],
            "verdict": fe_verdict,
            "beta_fe": fe_stats["beta_pooled"],
            "beta_re": re["beta_re"],
            "beta_shift": abs(re["beta_re"] - fe_stats["beta_pooled"]),
            "se_ratio": float(se_ratio),
            "tau2": re["tau2"],
            "I2": re["I2"],
        })

    return {
        "n_pairs": len(pairs),
        "per_pair": re_results,
    }


def run_sensitivity_analyses(pairs: list, alpha: float, output_dir: Path):
    """Run all stratum-level sensitivity analyses and save results."""
    log.info("Running sensitivity analyses...")

    exclude_approx = sensitivity_exclude_approximate(pairs, alpha)
    log.info(
        f"Exclude-approximate: {exclude_approx['n_exact']} exact pairs, "
        f"{exclude_approx['n_approximate']} with approximate strata"
    )
    if exclude_approx["exact_metrics"]:
        log.info(f"  Exact-only accuracy: {exclude_approx['exact_metrics']['accuracy']:.3f}")
    if exclude_approx["approximate_metrics"]:
        log.info(f"  Approximate-containing accuracy: {exclude_approx['approximate_metrics']['accuracy']:.3f}")

    loo = sensitivity_leave_one_out(pairs, alpha)
    log.info(
        f"Leave-one-out: {loo['n_stable']}/{loo['n_pairs']} pairs stable "
        f"({loo['pct_stable']:.1%})"
    )
    unstable = [r for r in loo["per_pair"] if not r["stable"]]
    for u in unstable:
        changed = [d for d in u.get("dropped", []) if d["changed"]]
        log.info(f"  Unstable: {u['id']} — flips when dropping: {[d['dropped_stratum'] for d in changed]}")

    re = sensitivity_random_effects(pairs, alpha)
    shifts = [r["beta_shift"] for r in re["per_pair"]]
    se_ratios = [r["se_ratio"] for r in re["per_pair"]]
    log.info(
        f"Random-effects: mean |beta_FE - beta_RE| = {np.mean(shifts):.4f}, "
        f"max = {np.max(shifts):.4f}"
    )
    log.info(
        f"  SE inflation (RE/FE): mean = {np.mean(se_ratios):.3f}, "
        f"max = {np.max(se_ratios):.3f}"
    )

    sensitivity_results = {
        "alpha": alpha,
        "exclude_approximate": exclude_approx,
        "leave_one_out": {
            "n_pairs": loo["n_pairs"],
            "n_stable": loo["n_stable"],
            "pct_stable": loo["pct_stable"],
            "unstable_pairs": [
                {
                    "id": r["id"],
                    "expected": r.get("expected"),
                    "full_Q": r.get("full_Q"),
                    "full_p": r.get("full_p"),
                    "n_changed": r.get("n_changed"),
                    "flipped_strata": [
                        d["dropped_stratum"]
                        for d in r.get("dropped", [])
                        if d["changed"]
                    ],
                }
                for r in loo["per_pair"]
                if not r["stable"]
            ],
        },
        "random_effects": {
            "mean_beta_shift": float(np.mean(shifts)),
            "max_beta_shift": float(np.max(shifts)),
            "mean_se_ratio": float(np.mean(se_ratios)),
            "max_se_ratio": float(np.max(se_ratios)),
            "per_pair": re["per_pair"],
        },
    }

    with open(output_dir / "sensitivity_results.json", "w") as f:
        json.dump(sensitivity_results, f, indent=2)
    log.info(f"Saved sensitivity results to {output_dir / 'sensitivity_results.json'}")

    return sensitivity_results


def main():
    parser = argparse.ArgumentParser(description="Systematic MR Q-test pipeline")
    parser.add_argument("--input", type=str, default="data/pairs_curated.json", help="Path to curated pairs JSON")
    parser.add_argument("--output", type=str, default="results/", help="Output directory")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance threshold for Q-test")
    parser.add_argument("--plot", action="store_true", help="Generate plots")
    parser.add_argument("--sensitivity", action="store_true", help="Run stratum-level sensitivity analyses")
    args = parser.parse_args()

    log.info("Starting systematic MR Q-test pipeline")
    log.info(f"Input: {args.input}, Alpha: {args.alpha}, Output: {args.output}")

    script_dir = Path(__file__).resolve().parent
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = script_dir / input_path
    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = script_dir / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(input_path) as f:
        pairs = json.load(f)
    log.info(f"Loaded {len(pairs)} pairs")

    results = run_pipeline(pairs, args.alpha)

    metrics = evaluate_accuracy(results)
    cis = bootstrap_ci(results)
    log.info(
        f"Accuracy: {metrics['accuracy']:.3f} "
        f"[{cis['accuracy_ci'][0]:.3f}, {cis['accuracy_ci'][1]:.3f}]"
    )
    log.info(
        f"Sensitivity: {metrics['sensitivity']:.3f} "
        f"[{cis['sensitivity_ci'][0]:.3f}, {cis['sensitivity_ci'][1]:.3f}]"
    )
    log.info(
        f"Specificity: {metrics['specificity']:.3f} "
        f"[{cis['specificity_ci'][0]:.3f}, {cis['specificity_ci'][1]:.3f}]"
    )

    full_results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "alpha": args.alpha,
            "n_pairs": len(pairs),
            "input_file": str(input_path),
        },
        "per_pair": {r["id"]: r for r in results},
    }
    with open(output_dir / "full_results.json", "w") as f:
        json.dump(full_results, f, indent=2)
    log.info(f"Saved full results to {output_dir / 'full_results.json'}")

    alphas_to_sweep = [0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.50]
    sweep_results = {}
    for a in tqdm(alphas_to_sweep, desc="Alpha sweep"):
        sweep_results[a] = run_pipeline(pairs, a)
    sweep = alpha_sweep(sweep_results)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "alpha": args.alpha,
        "n_pairs": len(pairs),
        "domains": {},
        "metrics": metrics,
        "bootstrap_95ci": cis,
        "alpha_sweep": sweep,
        "misclassified": [r["id"] for r in results if not r["correct"]],
    }
    for r in results:
        d = r["domain"]
        if d not in summary["domains"]:
            summary["domains"][d] = {"n": 0, "n_correct": 0}
        summary["domains"][d]["n"] += 1
        if r["correct"]:
            summary["domains"][d]["n_correct"] += 1
    for d in summary["domains"]:
        info = summary["domains"][d]
        info["accuracy"] = info["n_correct"] / info["n"] if info["n"] > 0 else 0.0

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Saved summary to {output_dir / 'summary.json'}")

    if args.plot:
        log.info("Generating plots...")
        plot_roc_like(sweep, output_dir)
        plot_dot_plot(results, args.alpha, output_dir)
        plot_power_scatter(results, output_dir)

    if args.sensitivity:
        run_sensitivity_analyses(pairs, args.alpha, output_dir)

    log.info("Pipeline complete")


if __name__ == "__main__":
    main()
