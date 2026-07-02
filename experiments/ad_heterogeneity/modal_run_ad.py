"""Modal wrapper for AD neurology experiments.

5 experiments run sequentially in one container. Logic lives in experiments.py
and is inlined via add_local_dir.

Usage:
    modal run --detach geometry/paper_clinical/ad_heterogeneity/modal_run_ad.py
"""
import json
from datetime import datetime
from pathlib import Path

import modal

app = modal.App("paper-clinical-ad")
volume = modal.Volume.from_name("neuro-epi-results", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .run_commands("echo 'rebuild-v1'")
    .pip_install(
        "numpy>=1.24,<2",
        "scipy>=1.11",
        "pandas>=2.1",
        "scikit-learn>=1.3",
        "statsmodels>=0.14",
        "matplotlib>=3.8",
        "tqdm>=4.66",
    )
    .add_local_dir(
        "geometry/paper_clinical/ad_heterogeneity",
        remote_path="/root/ad_experiments",
    )
)


def _save(result, name, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"{name}_{ts}.json"

    def make_serializable(obj):
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        if hasattr(obj, "tolist"):
            return obj.tolist()
        if isinstance(obj, dict):
            return {str(k): make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [make_serializable(v) for v in obj]
        if isinstance(obj, float) and (obj != obj):
            return None
        return obj

    with open(path, "w") as f:
        json.dump(make_serializable(result), f, indent=2, default=str)
    print(f"[{datetime.now():%H:%M:%S}] Saved: {path}")


@app.function(image=image, volumes={"/vol": volume}, timeout=86400)
def run_all():
    """Run all 5 AD experiments sequentially in one container."""
    import sys
    sys.path.insert(0, "/root/ad_experiments")
    from experiments import (
        run_prerequisites,
        run_cocycle_obstruction,
        run_bracket_norm_confound_audit,
        run_sheaf_dag_adjudication,
        run_h1_effect_modifier_suite,
    )

    experiments = [
        ("prerequisites", run_prerequisites),
        ("cocycle_obstruction", run_cocycle_obstruction),
        ("p1_bracket_norm", run_bracket_norm_confound_audit),
        ("p2_sheaf_dag", run_sheaf_dag_adjudication),
        ("p3_h1_modifier", run_h1_effect_modifier_suite),
    ]

    base_dir = Path("/vol/geometry_results/paper_clinical/ad_heterogeneity")
    results = {}
    for name, fn in experiments:
        print(f"\n{'='*60}\n  STARTING: {name}\n{'='*60}")
        try:
            result = fn()
            _save(result, name, base_dir / name)
            volume.commit()
            results[name] = "DONE"
            print(f"  {name}: DONE")
        except Exception as e:
            results[name] = f"FAILED: {e}"
            import traceback
            traceback.print_exc()
            print(f"  {name}: FAILED ({e})")

    print(f"\n\nFinal status: {results}")
    return results


@app.local_entrypoint()
def main():
    handle = run_all.spawn()
    print(f"Spawned AD experiments orchestrator: {handle.object_id}")
    print("5 experiments will run sequentially:")
    print("  0. prerequisites (AD/FTD/DLB cohort cleaning + MMSE re-metricization)")
    print("  1. cocycle_obstruction (4-arm Grassmannian holonomy on ATN subspaces)")
    print("  2. p1_bracket_norm (confound audit: tau PET, amyloid PET, FDG, p-tau217)")
    print("  3. p2_sheaf_dag (amyloid cascade vs multi-process adjudication)")
    print("  4. p3_h1_modifier (APOE4, sex, TREM2/IL-6 effect-modifier suite)")
    print()
    print("Check results: modal volume ls neuro-epi-results /geometry_results/paper_clinical/ad_heterogeneity/")
