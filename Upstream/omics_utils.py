"""Patient-level linkage for post-model transcriptomic interpretation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from io_utils import require_columns


def validate_risk_manifest(frame: pd.DataFrame) -> None:
    fields = ["patient_id", "subtype", "omics_type", "vit_km_score", "vit_cutoff", "vit_risk_group"]
    require_columns(frame, fields, "The frozen ViT risk manifest")
    if frame.empty or frame[fields].isna().any().any() or frame.patient_id.duplicated().any():
        raise ValueError("The risk manifest must contain unique, complete patient records.")
    values = frame[["vit_km_score", "vit_cutoff"]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or frame.vit_cutoff.nunique() != 1 or not frame.subtype.eq("GBM").all():
        raise ValueError("Use the final GBM ViT scores and its single frozen training cutoff.")
    if not frame.omics_type.isin(["Single-cell RNA-seq", "Bulk transcriptomics"]).all():
        raise ValueError("Omics cohorts must be disjoint and explicitly identified.")
    expected = np.where(frame.vit_km_score >= frame.vit_cutoff, "High risk", "Low risk")
    if not np.array_equal(expected, frame.vit_risk_group):
        raise ValueError("Stale imaging-risk groups were detected.")


def attach_risk(metadata: pd.DataFrame, manifest: pd.DataFrame, omics_type: str) -> pd.DataFrame:
    validate_risk_manifest(manifest)
    require_columns(metadata, ["patient_id"], "The omics metadata")
    if {"risk_group_3dvit", "training_cutoff_3dvit"}.intersection(metadata.columns):
        raise ValueError("Remove obsolete risk fields; use only the frozen ViT manifest.")
    expected = set(manifest.loc[manifest.omics_type.eq(omics_type), "patient_id"])
    if set(metadata.patient_id) != expected:
        raise ValueError("Omics patient membership differs from the frozen manifest.")
    result = metadata.merge(manifest[["patient_id", "vit_risk_group"]], on="patient_id",
                            how="left", validate="many_to_one", suffixes=("_supplied", ""))
    if "vit_risk_group_supplied" in result and not result.vit_risk_group.eq(result.vit_risk_group_supplied).all():
        raise ValueError("Supplied groups disagree with final imaging predictions.")
    if "risk_group" in result and not result.risk_group.eq(result.vit_risk_group).all():
        raise ValueError("Legacy risk groups disagree with final imaging predictions.")
    return result.drop(columns=["vit_risk_group_supplied"], errors="ignore")
