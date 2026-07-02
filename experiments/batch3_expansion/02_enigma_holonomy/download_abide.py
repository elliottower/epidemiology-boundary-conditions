"""Download ABIDE FreeSurfer cortical thickness data and prepare for holonomy analysis.

Downloads left-hemisphere aparc.stats files from S3 for all ABIDE I subjects,
extracts Desikan-Killiany cortical thickness (34 ROIs), and saves per-site
matrices for the holonomy pipeline.

Usage:
    python download_abide.py --output data/abide/
    python download_abide.py --output data/abide/ --both-hemispheres
"""
from __future__ import annotations

import argparse
import io
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PHENOTYPIC_URL = (
    "https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/"
    "Phenotypic_V1_0b_preprocessed1.csv"
)

STATS_URL_TEMPLATE = (
    "https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/"
    "Outputs/freesurfer/5.1/{file_id}/stats/{hemi}.aparc.stats"
)

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


def parse_aparc_stats(text: str) -> dict[str, float] | None:
    """Parse FreeSurfer aparc.stats text, return region -> ThickAvg dict."""
    result = {}
    for line in text.split("\n"):
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        if len(parts) >= 5:
            region = parts[0]
            try:
                thick = float(parts[4])
                result[region] = thick
            except (ValueError, IndexError):
                continue
    if not result:
        return None
    return result


def download_subject(file_id: str, hemispheres: list[str], session: requests.Session) -> dict[str, float] | None:
    """Download and parse aparc.stats for one subject."""
    combined = {}
    for hemi in hemispheres:
        url = STATS_URL_TEMPLATE.format(file_id=file_id, hemi=hemi)
        try:
            resp = session.get(url, timeout=15)
            if resp.status_code != 200:
                return None
            parsed = parse_aparc_stats(resp.text)
            if parsed is None:
                return None
            prefix = f"{hemi}_" if len(hemispheres) > 1 else ""
            for region, thick in parsed.items():
                combined[f"{prefix}{region}"] = thick
        except (requests.RequestException, Exception):
            return None
    return combined


def main():
    parser = argparse.ArgumentParser(description="Download ABIDE cortical thickness data")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument("--both-hemispheres", action="store_true",
                        help="Download both hemispheres (68 regions instead of 34)")
    parser.add_argument("--min-site-size", type=int, default=15,
                        help="Minimum subjects per site to include (default: 15)")
    args = parser.parse_args()

    hemispheres = ["lh", "rh"] if args.both_hemispheres else ["lh"]
    n_regions = 34 * len(hemispheres)
    log.info("Downloading ABIDE cortical thickness: %d hemisphere(s), %d regions", len(hemispheres), n_regions)

    # Download phenotypic data
    log.info("Fetching phenotypic data...")
    resp = requests.get(PHENOTYPIC_URL, timeout=30)
    pheno = pd.read_csv(io.StringIO(resp.text))
    log.info("Phenotypic data: %d subjects, %d sites", len(pheno), pheno["SITE_ID"].nunique())

    # Download aparc.stats for each subject
    session = requests.Session()
    records = []
    failed = 0

    for _, row in tqdm(pheno.iterrows(), total=len(pheno), desc="Downloading subjects"):
        file_id = row["FILE_ID"]
        if pd.isna(file_id) or file_id == "no_filename":
            failed += 1
            continue

        data = download_subject(str(file_id), hemispheres, session)
        if data is None:
            failed += 1
            continue

        data["subject_id"] = row["SUB_ID"]
        data["site_id"] = row["SITE_ID"]
        data["dx_group"] = row["DX_GROUP"]
        data["age"] = row["AGE_AT_SCAN"]
        data["sex"] = row["SEX"]
        records.append(data)

    log.info("Downloaded: %d succeeded, %d failed", len(records), failed)

    if not records:
        log.error("No data downloaded. Exiting.")
        return

    # Build DataFrame
    df = pd.DataFrame(records)
    log.info("Data shape: %s", df.shape)

    # Get region columns (either with or without hemisphere prefix)
    if args.both_hemispheres:
        region_cols = [f"{h}_{r}" for h in hemispheres for r in DESIKAN_REGIONS]
    else:
        region_cols = DESIKAN_REGIONS

    available_regions = [c for c in region_cols if c in df.columns]
    log.info("Available regions: %d / %d", len(available_regions), len(region_cols))

    # Save per-site matrices
    args.output.mkdir(parents=True, exist_ok=True)

    site_info = {}
    for site_id, group in df.groupby("site_id"):
        if len(group) < args.min_site_size:
            log.info("Skipping site %s: only %d subjects (< %d)", site_id, len(group), args.min_site_size)
            continue

        matrix = group[available_regions].values.astype(np.float64)

        # Check for NaN rows and drop them
        valid_rows = ~np.isnan(matrix).any(axis=1)
        if valid_rows.sum() < args.min_site_size:
            log.info("Skipping site %s: only %d valid rows after NaN removal", site_id, valid_rows.sum())
            continue
        matrix = matrix[valid_rows]

        site_file = args.output / f"site_{site_id}.npy"
        np.save(site_file, matrix)

        site_info[str(site_id)] = {
            "n_subjects": int(matrix.shape[0]),
            "n_regions": int(matrix.shape[1]),
            "file": site_file.name,
        }
        log.info("Site %s: %d subjects x %d regions", site_id, matrix.shape[0], matrix.shape[1])

    # Save metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "source": "ABIDE I (Autism Brain Imaging Data Exchange)",
        "atlas": "Desikan-Killiany (FreeSurfer aparc)",
        "measure": "cortical_thickness_mm",
        "hemispheres": hemispheres,
        "n_regions": len(available_regions),
        "regions": available_regions,
        "n_sites": len(site_info),
        "total_subjects": sum(s["n_subjects"] for s in site_info.values()),
        "sites": site_info,
    }

    with open(args.output / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Also save the full DataFrame
    df.to_csv(args.output / "all_subjects.csv", index=False)

    log.info("=== Summary ===")
    log.info("Sites: %d", len(site_info))
    log.info("Total subjects: %d", sum(s["n_subjects"] for s in site_info.values()))
    log.info("Regions: %d", len(available_regions))
    log.info("Output: %s", args.output)


if __name__ == "__main__":
    main()
