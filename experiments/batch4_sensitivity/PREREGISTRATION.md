# Preregistration: Paper A & B Sensitivity Experiments

**Date:** 2026-07-12
**Author:** Elliot Tower
**Commit SHA:** (to be filled after commit)

## Overview

Four experiments addressing reviewer-anticipated gaps in Papers A and B.
All use simulated data with planted ground truth; no real-data analysis
beyond the existing 61-pair catalog (which is frozen).

---

## Experiment 1: Heterogeneity-Ratio Sweep (Paper B, SPEC B2)

### Hypothesis
The boundary between "global test sufficient" and "per-edge test required"
is a smooth function of the mechanism/downstream variance ratio, not a
designed cliff at the current 100x setting.

### Method
- 3-node DAG (A↔B→C) with 8 strata, matching the MS/AD SEM structure.
- Mechanism edges (A→B, B→A) have between-stratum coefficient SD = σ_mech.
- Downstream edges (A→C, B→C) have between-stratum coefficient SD = σ_down.
- Sweep the ratio R = σ_mech / σ_down across {1, 2, 5, 10, 20, 50, 100, 200, 500, 1000}.
- At each ratio, run 200 replicates of: generate SEM data (n=3000 per stratum),
  compute global Frobenius test p-value and per-edge Q p-values.
- Hold σ_down = 0.005 fixed, vary σ_mech = R × σ_down.

### Analysis plan
- For each R, report: (a) fraction of replicates where global test rejects at α=0.05,
  (b) fraction where per-edge test correctly identifies mechanism edges as heterogeneous
  AND downstream edges as homogeneous.
- Plot both as a function of log10(R).
- Report the R value at which per-edge advantage first exceeds 20 percentage points
  over global test.

### Success criterion
- The transition is smooth (no step function at any particular R).
- Per-edge test dominates global test for R ≥ ~10.
- At R = 1 (no edge-specific structure), global and per-edge perform comparably.

### Failure modes
- If per-edge test dominates even at R = 1, the advantage is from multiple testing
  correction, not edge-specific structure.
- If the transition is sharp (step function), the boundary is an artifact of the
  thresholding, not the geometry.

---

## Experiment 2: Partial Confounding Degradation (Paper B, SPEC B2)

### Hypothesis
Partial correlation confound detection degrades gracefully from AUROC=1.0
as confounding strength decreases from 100% to partial.

### Method
- 10 real biomarkers (disease_severity → biomarker → disability) and
  10 confounded biomarkers (ROI_size → biomarker, ROI_size → disability),
  matching the existing confound_collapse_audit setup.
- Introduce a confounding fraction parameter f ∈ {0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0}.
  At f=1.0, confounded biomarkers correlate ONLY with ROI_size (current setup).
  At f=0.5, confounded biomarkers = 0.5 × ROI_size_component + 0.5 × disease_component.
  At f=0.0, confounded biomarkers are actually real (null control).
- n = 3000 patients, 200 replicates per f value.
- Compute partial correlation AUROC for distinguishing real from confounded.

### Analysis plan
- Plot AUROC vs f. Report AUROC at each f value with 95% CI.
- Report the f value at which AUROC first drops below 0.80.
- Report whether AUROC at f=0.0 is ~0.50 (null baseline).

### Success criterion
- Monotonic degradation: AUROC(f=1.0) > AUROC(f=0.7) > ... > AUROC(f=0.0) ≈ 0.50.
- AUROC remains above 0.80 for f ≥ 0.50 (partial confounding is still detectable
  when the confound explains ≥50% of biomarker variance).

### Failure modes
- If AUROC drops sharply (e.g., 1.0 at f=1.0 to 0.55 at f=0.9), the method is
  fragile to partial confounding — a genuine negative result worth reporting.
- If AUROC at f=0.0 is not ~0.50, the simulation has a design flaw.

---

## Experiment 3: Minimum-Detectable Heterogeneity (Paper A, SPEC A2)

### Hypothesis
With 3–4 strata (the structure of our 61-pair catalog), the Q test detects
between-stratum SD ≥ X at 80% power, where X is computable from the typical
within-stratum SE in our data.

### Method
- For each of K ∈ {3, 4} strata, sweep between-stratum SD (τ) from 0.01 to 0.50
  in 20 steps.
- Within-stratum SEs drawn from the empirical distribution of our 61-pair catalog
  (median SE, IQR).
- At each (K, τ), generate 2000 replicates: draw β_k ~ N(β_0, τ²), compute Q,
  test at α = 0.05.
- Report power = fraction rejecting.

### Analysis plan
- For K=3 and K=4, find τ_80 = minimum τ at which power ≥ 0.80.
- Report τ_80 relative to typical within-stratum SE (i.e., τ_80 / median_SE).
- Compare to the observed τ distribution in the 61-pair catalog:
  how many detected non-transport pairs have τ > τ_80?
  how many misclassified pairs have τ < τ_80?

### Success criterion
- τ_80 is well-defined and falls in a plausible range (e.g., τ_80 / median_SE ∈ [1, 5]).
- All 9 misclassified pairs have τ < τ_80 (consistent with the power explanation).
- All correctly detected non-transport pairs have τ > τ_80.

### Failure modes
- If τ_80 is very large (>10× median SE), the Q test is fundamentally underpowered
  for this data structure — an honest limitation.

---

## Experiment 4: LD-Attributable Heterogeneity Simulation (Paper A, SPEC A1)

### Hypothesis
Between-ancestry heterogeneity in MR estimates has contributions from both
(a) LD/allele-frequency differences and (b) causal effect modification.
The Q values observed in our transport pairs (Q < 0.70) are consistent with
LD-only variation, while non-transport pairs (Q > 7) exceed LD-only levels.

### Method
- Simulate an MR setting with 4 ancestry strata matching our data structure.
- True causal effect β_causal is CONSTANT across all strata (transport null).
- IV-exposure association strength varies by ancestry to model LD differences:
  γ_k ~ N(γ_0, σ_LD²), where σ_LD controls LD heterogeneity.
- The Wald ratio β_MR,k = β_causal × γ_k / γ_k + noise reduces to
  β_causal + noise/γ_k, so LD variation enters through the denominator.
- Sweep σ_LD / γ_0 (relative LD variation) across {0.0, 0.05, 0.10, 0.15, 0.20, 0.30}.
- At each level, generate 500 replicates with 4 strata, n=50000 per stratum
  (matching large GWAS), compute Q.
- Compare the Q distribution to observed Q values from our 39 transport pairs
  and 22 non-transport pairs.

### Analysis plan
- For each σ_LD level, report the Q distribution (median, 95th percentile).
- Identify the σ_LD level whose Q distribution best matches the observed
  transport-pair Q values (Q < 0.70, mean ~0.50).
- Show that even at the maximum plausible σ_LD, the Q distribution does not
  reach the non-transport range (Q > 7).
- If LD-only Q can reach non-transport levels, report the σ_LD required
  and assess whether it is realistic.

### Success criterion
- Transport-pair Q values are consistent with moderate LD variation (σ_LD/γ_0 ~ 0.05–0.15).
- Non-transport Q values (Q > 7) are NOT reachable under any plausible LD-only model.
- The separation between LD-attributable Q and causal-heterogeneity Q provides
  a natural threshold.

### Failure modes
- If LD-only variation can produce Q > 7 at realistic σ_LD, we cannot distinguish
  LD from causal heterogeneity — an important negative result that should be
  discussed honestly.
- If transport-pair Q values are too LOW relative to any LD model (even σ_LD=0
  produces Q > observed), the simulation calibration is wrong.

---

## Shared parameters

- All experiments use `numpy.random.default_rng(seed)` with seed specified per experiment.
- Seeds: Exp1=2026_07_12_01, Exp2=2026_07_12_02, Exp3=2026_07_12_03, Exp4=2026_07_12_04.
- Results saved to `experiments/batch4_sensitivity/results/` as JSON.
- No GPU required; all run on CPU.
