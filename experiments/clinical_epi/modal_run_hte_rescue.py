"""Modal wrapper for HTE rescue only (experiment 6).

The combined rescue app completed curvature but died before HTE finished.
This runs just the HTE rescue.

Usage:
    modal run --detach geometry/samprofessors/batch1/modal_run_hte_rescue.py
"""
import json
from datetime import datetime
from pathlib import Path

import modal

app = modal.App("samprofessors-batch1-hte-rescue")
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
def run_hte():
    """Run HTE rescue experiment."""
    import sys
    sys.path.insert(0, "/root/batch1")
    from experiments_rescue import run_hte_rescue

    base = Path("/vol/geometry_results/samprofessors/batch1")

    print(f"\n{'=' * 60}\n  STARTING: hte_rescue\n{'=' * 60}")
    result = run_hte_rescue()
    _save(result, "hte_rescue", base / "hte_rescue")
    volume.commit()
    print("  hte_rescue: DONE")
    return result


@app.local_entrypoint()
def main():
    handle = run_hte.spawn()
    print(f"Spawned HTE rescue: {handle.object_id}")
    print("  6. hte_rescue (RF/GBM CATE + better clustering)")
    print()
    print("Check results: modal volume ls neuro-epi-results "
          "/geometry_results/samprofessors/batch1/hte_rescue/")
