"""Evaluate held-out automatic masks against independent manual references."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import binary_erosion, distance_transform_edt


def mask_metrics(predicted: np.ndarray, reference: np.ndarray, spacing_zyx: tuple) -> tuple[float, float]:
    predicted, reference = predicted.astype(bool), reference.astype(bool)
    if predicted.shape != reference.shape or predicted.ndim != 3:
        raise ValueError("Masks must share a 3D grid.")
    if not reference.any():
        raise ValueError("The manual tumor-core reference is empty.")
    if not predicted.any():
        return 0.0, float("inf")
    dice = 2 * np.count_nonzero(predicted & reference) / (predicted.sum() + reference.sum())
    a = predicted & ~binary_erosion(predicted)
    b = reference & ~binary_erosion(reference)
    a_to_b = distance_transform_edt(~b, sampling=spacing_zyx)[a]
    b_to_a = distance_transform_edt(~a, sampling=spacing_zyx)[b]
    hd95 = max(np.percentile(a_to_b, 95), np.percentile(b_to_a, 95))
    return float(dice), float(hd95)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import SimpleITK as sitk
    spec = importlib.util.spec_from_file_location("prepare", Path(__file__).with_name("02_prepare_nnunet_dataset.py"))
    prepare = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prepare)
    records = json.loads(args.evaluation_manifest.read_text(encoding="utf-8"))
    if not records or len({r["patient_id"] for r in records}) != len(records):
        raise ValueError("Evaluation needs one held-out mask per patient.")
    rows = []
    for record in records:
        expected = "out_of_fold" if record["training"] else "frozen_ensemble"
        if record["prediction_type"] != expected:
            raise ValueError("In-sample or manually corrected predictions cannot enter this evaluation.")
        reference = sitk.ReadImage(record["reference_mask"])
        prediction = sitk.ReadImage(record["prediction_mask"])
        if not prepare.same_geometry(prediction, reference):
            raise ValueError("Predictions and references must share physical geometry.")
        dice, hd95 = mask_metrics(sitk.GetArrayFromImage(prediction) > 0,
                                 sitk.GetArrayFromImage(reference) > 0,
                                 tuple(reversed(reference.GetSpacing())))
        rows.append({"patient_id": record["patient_id"], "cohort": record["cohort"],
                     "dice_similarity": dice, "hd95_mm": hd95,
                     "prediction_type": expected, "empty_prediction": not np.isfinite(hd95)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    if any(row["empty_prediction"] for row in rows):
        raise SystemExit("Empty automatic masks were retained as failures; review before finite-HD95 summaries.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
