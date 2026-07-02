"""Modal wrapper for batch1 rescue experiments.

Rescues the two batch1 negatives:
5. curvature_rescue — Forman-Ricci + topological features (replaces ORC)
6. hte_rescue — RF/GBM CATE estimators + better clustering (replaces KNN/PCA)

Usage:
    modal run --detach geometry/samprofessors/batch1/modal_run_rescue.py
"""
import json
from datetime import datetime
from pathlib import Path

import modal

app = modal.App("samprofessors-batch1-rescue")
volume = modal.Volume.from_name("neuro-epi-results", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy>=1.24,<2",
        "scipy>=1.11",
        "scikit-learn>=1.3",
        "matplotlib>=3.8",
        "tqdm>=4.66",
        "networkx>=3.2",
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
def run_rescue():
    """Run both rescue experiments sequentially."""
    import sys
    sys.path.insert(0, "/root/batch1")
    from experiments_rescue import run_curvature_rescue, run_hte_rescue

    base = Path("/vol/geometry_results/samprofessors/batch1")

    experiments = [
        ("curvature_rescue", run_curvature_rescue),
        ("hte_rescue", run_hte_rescue),
    ]

    results = {}
    for name, fn in experiments:
        print(f"\n{'=' * 60}\n  STARTING: {name}\n{'=' * 60}")
        try:
            result = fn()
            _save(result, name, base / name)
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
    handle = run_rescue.spawn()
    print(f"Spawned rescue orchestrator: {handle.object_id}")
    print("2 rescue experiments will run sequentially:")
    print("  5. curvature_rescue (Forman-Ricci + topological features)")
    print("  6. hte_rescue (RF/GBM CATE + better clustering)")
    print()
    print("Check results: modal volume ls neuro-epi-results "
          "/geometry_results/samprofessors/batch1/")
