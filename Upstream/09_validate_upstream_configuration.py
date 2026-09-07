"""Validate the unified upstream configuration without requiring raw inputs."""

from __future__ import annotations

import argparse
from pathlib import Path

from io_utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    required_sections = [
        "project",
        "data",
        "mri_preprocessing",
        "nnunet",
        "survival_model",
        "attribution",
        "bulk_ssgsea",
        "single_cell",
    ]
    missing_sections = [section for section in required_sections if section not in config]
    if missing_sections:
        raise SystemExit(f"Missing configuration sections: {', '.join(missing_sections)}")
    if int(config["project"]["random_seed"]) != 2026:
        raise SystemExit("The unified project random seed must be 2026.")
    mri = config["mri_preprocessing"]
    target = [int(value) for value in mri["target_size_voxels_xyz"]]
    patch = [int(value) for value in config["survival_model"]["patch_size_xyz"]]
    if len(target) != 3 or len(patch) != 3 or any(target[i] % patch[i] for i in range(3)):
        raise SystemExit("The target size must be divisible by the 3D patch size in every axis.")
    if list(config["data"]["image_columns"].keys()) != ["T1WI", "T2WI", "T2_FLAIR", "CE_T1WI"]:
        raise SystemExit("The workflow requires the four MRI sequences in the configured order.")
    if mri["reference_sequence"] != "CE_T1WI":
        raise SystemExit("CE_T1WI must be the registration reference.")
    if [float(value) for value in mri["target_spacing_mm_xyz"]] != [1.0, 1.0, 5.0]:
        raise SystemExit("The configured MRI spacing must be 1 x 1 x 5 mm.")
    if float(mri["tumor_core_expansion_mm"]) != 5.0:
        raise SystemExit("The configured tumor-core expansion must be 5 mm.")
    if config["nnunet"]["configuration"] != "3d_fullres":
        raise SystemExit("The configured nnU-Net mode must be 3d_fullres.")
    if set(config["survival_model"]["models"]) != {"3d_vit", "3d_resnet18"}:
        raise SystemExit("The survival model list must contain 3d_vit and 3d_resnet18.")
    print("Upstream configuration is valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
