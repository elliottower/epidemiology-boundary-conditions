# SPEC — Paper B → Entropy (special issue: causal inference / information geometry)

**Target framing:** "When do geometric methods help causal inference? Boundary conditions for holonomy, curvature, and subspace methods." A decision-procedure + negative-results paper. Cites Paper A for the positive sheaf result.

## B1. Required split surgery (do first)
- [ ] **Remove the full real-data MR validation** and cite Paper A instead — keep only a one-paragraph pointer.
- [ ] **Keep in B:** Condition 1 (holonomy/Berry), Condition 3 (curvature + PCA-CATE), boundary framework (Figure 4), AD consistency check.
- [ ] **Keep the per-edge Q (Condition 2)** only as an *illustration of the boundary*, citing Paper A for the empirical DAG results — avoid duplicating the ADNI analysis.

## B2. Required new analyses (acceptance-critical for a geometry audience)
- [ ] **Prove the Berry-phase closed form** ‖Φ − I_k‖_F = 2√2 |sin(π sin²r)| in an appendix.
- [ ] **Heterogeneity-ratio sweep**: vary the mechanism/downstream variance ratio across a continuum to show the boundary is smooth, not a designed cliff.
- [ ] **Partial + unobserved confounding** in the confound-detection sim — show graceful degradation from AUROC=1.0.
- [ ] **Information-geometry connection (venue fit!):** explicit linkage between sheaf Laplacian / holonomy and information geometry (Fisher metric, KL divergence, or entropy of the obstruction). *This is the venue-specific gate.*

## B3. Required robustness
- [ ] **Berry-phase power boundary**: add a second real-data null beyond ABIDE if feasible, or clearly frame ABIDE as the single boundary-consistent null.
- [ ] **Curvature negative result:** add a one-line general statement that Forman–Ricci discrimination direction = sign of (mean FP-degree − mean TP-degree).

## B4. Structure/writing
- [ ] **Make Figure 4 (three conditions) the structural spine** — organize Results strictly as Condition 1 / 2 / 3.
- [ ] **Abstract ≤ 200 words**, lead with the three-condition decision procedure.
- [ ] Reduce repetition of p<10^{-300} and the "existence and consistency" hedge.
- [ ] Add **information-theoretic keyword set** for the special-issue match.

**Acceptance odds after B1–B4:** ~45–50%. Without B2's information-geometry linkage: ~30%.

## Cross-paper guardrails
- [ ] **Anonymity/citation:** Paper B cites Paper A as "under review" (anonymized if double-blind).
- [ ] **One-experiment-one-paper rule:** MR sensitivity + ADNI + Cochran reduction → A only; heterogeneity sweep + partial confounding + curvature + holonomy + info-geometry → B only.
- [ ] **Shared appendix** (Cochran's Q reduction) lives in A; B references it.
