"""Create a nnU-Net v2 dataset for four-channel tumor-core segmentation."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from io_utils import load_config, require_columns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--copy-validation", action="store_true")
    return parser.parse_args()


def case_name(value: object, index: int) -> str:
    raw = "".join(character if str(character).isalnum() or str(character) in "-_" else "_" for character in str(value))
    return f"case_{index:04d}_{raw}"


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    data_cfg = config["data"]
    frame = pd.read_csv(args.manifest)
    required = [data_cfg["patient_id_column"], data_cfg["cohort_column"], data_cfg["mask_column"]]
    required += list(data_cfg["image_columns"].values())
    require_columns(frame, required, "The nnU-Net manifest")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    images_tr = args.output_dir / "imagesTr"
    labels_tr = args.output_dir / "labelsTr"
    images_ts = args.output_dir / "imagesTs"
    images_tr.mkdir(exist_ok=True)
    labels_tr.mkdir(exist_ok=True)
    selected = frame[
        frame[data_cfg["cohort_column"]].astype(str).str.lower()
        == str(config["data"]["training_cohort"]).lower()
    ].copy()
    try:
        import SimpleITK as sitk
    except ImportError as exc:
        raise SystemExit("SimpleITK is required to prepare a nnU-Net dataset.") from exc
    channel_items = list(data_cfg["image_columns"].items())
    records = []
    manifest_dir = args.manifest.parent.resolve()
    for index, (_, row) in enumerate(selected.reset_index(drop=True).iterrows(), start=1):
        name = case_name(row[data_cfg["patient_id_column"]], index)
        image_paths = []
        for channel_index, (_, column) in enumerate(channel_items):
            source = Path(str(row[column]))
            if not source.is_absolute():
                source = manifest_dir / source
            if not source.exists():
                raise FileNotFoundError(source)
            destination = images_tr / f"{name}_{channel_index:04d}.nii.gz"
            image = sitk.ReadImage(str(source))
            sitk.WriteImage(image, str(destination), useCompression=True)
            image_paths.append(str(destination.name))
        mask_source = Path(str(row[data_cfg["mask_column"]]))
        if not mask_source.is_absolute():
            mask_source = manifest_dir / mask_source
        if not mask_source.exists():
            raise FileNotFoundError(mask_source)
        mask_destination = labels_tr / f"{name}.nii.gz"
        mask = sitk.Cast(sitk.ReadImage(str(mask_source)) > 0, sitk.sitkUInt8)
        sitk.WriteImage(mask, str(mask_destination), useCompression=True)
        records.append({"case": name, "patient_id": str(row[data_cfg["patient_id_column"]]), "images": image_paths})
    dataset_id = int(config["nnunet"]["dataset_id"])
    dataset_name = str(config["nnunet"]["dataset_name"])
    dataset_json = {
        "channel_names": {str(index): name for index, (name, _) in enumerate(channel_items)},
        "labels": config["nnunet"]["label_map"],
        "numTraining": len(records),
        "file_ending": ".nii.gz",
        "name": dataset_name,
        "description": "Four-channel adult-type diffuse glioma tumor-core segmentation",
    }
    with (args.output_dir / "dataset.json").open("w", encoding="utf-8") as handle:
        json.dump(dataset_json, handle, indent=2)
    with (args.output_dir / "case_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2)
    if args.copy_validation:
        images_ts.mkdir(exist_ok=True)
        validation = frame.loc[~frame.index.isin(selected.index)].copy()
        for index, (_, row) in enumerate(validation.iterrows(), start=1):
            name = case_name(row[data_cfg["patient_id_column"]], index)
            for channel_index, (_, column) in enumerate(channel_items):
                source = Path(str(row[column]))
                if not source.is_absolute():
                    source = manifest_dir / source
                destination = images_ts / f"{name}_{channel_index:04d}.nii.gz"
                sitk.WriteImage(sitk.ReadImage(str(source)), str(destination), useCompression=True)
    command = (
        f"nnUNetv2_plan_and_preprocess -d {dataset_id} --verify_dataset_integrity\n"
        f"nnUNetv2_train Dataset{dataset_id:03d} {config['nnunet']['configuration']} all "
        f"-tr {config['nnunet']['trainer']}"
    )
    (args.output_dir / "nnunet_commands.txt").write_text(command + "\n", encoding="utf-8")
    print(f"Prepared {len(records)} training cases in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
