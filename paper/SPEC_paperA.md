# SPEC — Paper A → Genetic Epidemiology

**Target framing:** "Sheaf Q tests classify Mendelian Randomization transportability and recover causal DAG structure." A methods-contribution paper validated on published MR estimates — *not* an epidemiological discovery claim.

## A1. Required new analyses (acceptance-critical)
- [ ] **MR pleiotropy sensitivity on all 61 pairs**, not just cardiometabolic: MR-Egger intercept, weighted-median, weighted-mode. Report whether between-stratum heterogeneity survives pleiotropy adjustment. *This is the single biggest gate for this venue.*
- [ ] **Steiger filtering / directionality** check per pair — confirm instruments act on exposure→outcome, not reverse. Genetic Epi reviewers expect this by default.
- [ ] **I² and τ² reported per pair** alongside Q (meta-analytic standard; you already compute τ² internally — surface it).
- [ ] **Decompose the ancestry-heterogeneity caveat empirically**: for ≥5 pairs, show how much between-stratum Q is attributable to allele-frequency/LD differences vs. residual (e.g., LD-score or allele-frequency covariate adjustment). Currently §7.5 discusses this only in prose — reviewers will want at least one quantitative probe.

## A2. Required robustness (pre-empt reviewer 2)
- [ ] **Sensitivity table excluding the ††calibrated/approximate MR pairs** — report accuracy on published-exact pairs only, as a row in the main accuracy table.
- [ ] **Reframe post-hoc power** as a **pre-specified minimum-detectable-heterogeneity** analysis: "with 3–4 strata we detect between-stratum SD ≥ X at 80%." Keep the bimodal 0.38/0.62 gap as the headline diagnostic.
- [ ] **ADNI per-edge DAG** — add multiple-testing correction statement (Bonferroni/BH across edges) and a stability check (bootstrap or split-half) since structure is not planted here.

## A3. Reporting / compliance
- [ ] **STROBE-MR checklist** as supplement.
- [ ] **Data-source table** for all 61 pairs with GWAS consortium, ancestry, sample size, PMID (verify every pair has a citation + ancestry label).
- [ ] Note **no preregistration** explicitly and justify it (secondary analysis of published summary statistics).
- [ ] Code/data availability: Zenodo DOI + GitHub.

## A4. Structure/writing
- [ ] **Cut all curvature, holonomy, PCA-CATE, and boundary-condition framing** → those go to Paper B. Verify none leaked in.
- [ ] **Abstract ≤ 250 words**, lead with: sheaf Q method → 85.2% accuracy, zero FP, 5 domains → ADNI three-way finding.
- [ ] Keep AD as a **consistency check only** — do not imply cross-domain generalization from simulation.

**Acceptance odds after A1–A4:** ~45%. Without A1 (MR sensitivity): ~30%.
