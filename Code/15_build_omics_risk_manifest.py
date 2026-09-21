"""Export one frozen imaging-risk definition for downstream omics analyses."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def build_manifest(predictions: pd.DataFrame, subset: pd.DataFrame) -> pd.DataFrame:
    for frame in (predictions, subset):
        if frame.patient_id.isna().any() or frame.patient_id.duplicated().any():
            raise ValueError("Patient identifiers must be complete and unique.")
    if not set(subset.patient_id).issubset(predictions.patient_id):
        raise ValueError("Every omics patient must have a frozen imaging prediction.")
    fields = ["patient_id", "subtype", "vit_km_score", "vit_cutoff", "vit_risk_group"]
    result = subset[["patient_id", "omics_type"]].merge(
        predictions[fields], on="patient_id", how="left", validate="one_to_one")
    if not result.subtype.eq("GBM").all():
        raise ValueError("The specified omics analysis is GBM-only.")
    if not result.omics_type.isin(["Single-cell RNA-seq", "Bulk transcriptomics"]).all():
        raise ValueError("Each patient must belong to exactly one nonoverlapping omics subset.")
    numeric = result[["vit_km_score", "vit_cutoff"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or result.vit_cutoff.nunique() != 1:
        raise ValueError("Risk scores need a single finite frozen GBM cutoff.")
    expected = np.where(result.vit_km_score >= result.vit_cutoff, "High risk", "Low risk")
    if not np.array_equal(expected, result.vit_risk_group):
        raise ValueError("Risk groups disagree with the final ViT scores and cutoff.")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    pred = pd.read_csv(args.root / "Results/clinicoradiologic_crvit_patient_predictions.csv",
                       dtype={"patient_id": str})
    subset = pd.read_csv(args.root / "Data/center1_gbm_multiomics_subset_patient_level.csv",
                         dtype={"patient_id": str})
    manifest = build_manifest(pred, subset)
    manifest.to_csv(args.root / "Results/omics_frozen_vit_risk_manifest.csv", index=False)
    counts = manifest.groupby(["omics_type", "vit_risk_group"]).size().reset_index(name="patients")
    counts.to_csv(args.root / "QA/omics_risk_group_counts.csv", index=False)
    print(counts.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
