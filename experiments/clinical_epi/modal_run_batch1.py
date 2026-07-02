"""Modal wrapper for batch1 professor-pitch experiments.

Experiments run sequentially in one container. Logic lives in experiments.py
but is inlined here via copy_local_dir since modal.Mount was removed in 1.x.

Usage:
    modal run --detach geometry/samprofessors/batch1/modal_run_batch1.py
"""
import json
from datetime import datetime
from pathlib import Path

import modal

app = modal.App("samprofessors-batch1")
volume = modal.Volume.from_name("neuro-epi-results", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy>=1.24,<2",
        "scipy>=1.11",
        "pandas>=2.1",
        "scikit-learn>=1.3",
        "statsmodels>=0.14",
        "matplotlib>=3.8",
        "tqdm>=4.66",
        "networkx>=3.2",
        "pot>=0.9",
    )
    .add_local_dir(
        "geometry/samprofessors/batch1",
        remote_path="/root/batch1",
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
    """Run all 4 experiments sequentially in one container."""
    import sys
    sys.path.insert(0, "/root/batch1")
    from experiments import (
        run_sheaf_federated_ehr,
        run_confound_collapse_audit,
        run_curvature_causal_graph,
        run_treatment_heterogeneity,
    )

    experiments = [
        ("sheaf_ehr", run_sheaf_federated_ehr),
        ("confound_audit", run_confound_collapse_audit),
        ("curvature_graph", run_curvature_causal_graph),
        ("treatment_het", run_treatment_heterogeneity),
    ]

    results = {}
    for name, fn in experiments:
        print(f"\n{'='*60}\n  STARTING: {name}\n{'='*60}")
        try:
            result = fn()
            _save(result, name, Path(f"/vol/geometry_results/samprofessors/batch1/{name}"))
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
    print(f"Spawned batch1 orchestrator: {handle.object_id}")
    print("4 experiments will run sequentially:")
    print("  1. sheaf_ehr (Visweswaran)")
    print("  2. confound_audit (Xia)")
    print("  3. curvature_graph (Visweswaran)")
    print("  4. treatment_het (Xia)")
    print()
    print("Check results: modal volume ls neuro-epi-results /geometry_results/samprofessors/batch1/")
