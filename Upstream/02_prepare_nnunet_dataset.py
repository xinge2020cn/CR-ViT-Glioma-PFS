"""Prepare registered full-volume MRI for patient-level nnU-Net evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from io_utils import load_config, require_columns


def validate_manifest(frame: pd.DataFrame, config: dict) -> None:
    data = config["data"]
    columns = [data["patient_id_column"], data["cohort_column"],
               data["reference_mask_column"], "preprocessing_stage",
               *data["image_columns"].values()]
    require_columns(frame, columns, "The segmentation manifest")
    identifiers = frame[data["patient_id_column"]]
    if (frame.empty or frame[columns].isna().any().any() or identifiers.duplicated().any()
            or identifiers.astype(str).str.strip().eq("").any()):
        raise ValueError("Segmentation needs complete, unique patient records.")
    if not frame.preprocessing_stage.eq("registered_normalized_full_volume").all():
        raise ValueError("Run full-volume registration and normalization before dataset preparation.")
    training = frame[data["cohort_column"]].str.lower().eq(str(data["training_cohort"]).lower())
    if training.sum() < 5:
        raise ValueError("Five-fold segmentation requires at least five training patients.")


def same_geometry(first, second) -> bool:
    return first.GetSize() == second.GetSize() and all(
        np.allclose(a, b, rtol=0, atol=1e-5) for a, b in (
            (first.GetSpacing(), second.GetSpacing()), (first.GetOrigin(), second.GetOrigin()),
            (first.GetDirection(), second.GetDirection())))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    data = config["data"]
    frame = pd.read_csv(args.manifest, dtype={data["patient_id_column"]: str})
    validate_manifest(frame, config)
    expected_name = f"Dataset{int(config['nnunet']['dataset_id']):03d}_{config['nnunet']['dataset_name']}"
    if args.output_dir.name != expected_name:
        raise ValueError(f"The nnU-Net dataset directory must be named {expected_name}.")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Use a new, empty dataset directory to avoid mixing releases.")
    import SimpleITK as sitk

    records = []
    for index, (_, row) in enumerate(frame.iterrows()):
        training = str(row[data["cohort_column"]]).lower() == str(data["training_cohort"]).lower()
        name = f"case_{index:06d}"
        folder = args.output_dir / ("imagesTr" if training else "imagesTs")
        folder.mkdir(parents=True, exist_ok=True)
        first = None
        for channel, column in enumerate(data["image_columns"].values()):
            path = Path(row[column])
            path = path if path.is_absolute() else args.manifest.parent / path
            image = sitk.ReadImage(str(path), sitk.sitkFloat32)
            first = image if first is None else first
            if not same_geometry(first, image):
                raise ValueError(f"Misaligned MRI channels in {name}.")
            if not np.allclose(image.GetSpacing(), config["mri_preprocessing"]["target_spacing_mm_xyz"]):
                raise ValueError(f"Unexpected full-volume spacing in {name}.")
            sitk.WriteImage(image, str(folder / f"{name}_{channel:04d}.nii.gz"), useCompression=True)
        reference = Path(row[data["reference_mask_column"]])
        reference = reference if reference.is_absolute() else args.manifest.parent / reference
        mask = sitk.ReadImage(str(reference))
        if not same_geometry(first, mask):
            raise ValueError(f"Reference-mask geometry mismatch in {name}.")
        if not np.any(sitk.GetArrayViewFromImage(mask) > 0):
            raise ValueError(f"Empty reference tumor core in {name}.")
        labels = args.output_dir / ("labelsTr" if training else "labelsTs")
        labels.mkdir(exist_ok=True)
        destination = labels / f"{name}.nii.gz"
        sitk.WriteImage(sitk.Cast(mask > 0, sitk.sitkUInt8), str(destination), useCompression=True)
        records.append({"case": name, "patient_id": str(row[data["patient_id_column"]]),
                        "cohort": str(row[data["cohort_column"]]), "training": training,
                        "reference_mask": str(destination.resolve())})
    metadata = {
        "channel_names": {str(i): name for i, name in enumerate(data["image_columns"])},
        "labels": config["nnunet"]["label_map"], "numTraining": sum(r["training"] for r in records),
        "file_ending": ".nii.gz", "name": config["nnunet"]["dataset_name"],
    }
    for filename, value in (("dataset.json", metadata), ("case_manifest.json", records)):
        (args.output_dir / filename).write_text(json.dumps(value, indent=2), encoding="utf-8")
    print(f"Prepared {metadata['numTraining']} training patients and {len(records) - metadata['numTraining']} validation patients.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
