"""Validate the unified upstream configuration without requiring raw inputs."""

from __future__ import annotations

import argparse
from pathlib import Path

from io_utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def validate_config(config: dict) -> None:
    """Check the current manuscript profile, not whether any model has been trained."""
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
    if target != [192, 192, 24] or patch != [16, 16, 4]:
        raise SystemExit("The manuscript profile requires input 192 x 192 x 24 and patch 16 x 16 x 4.")
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
    model = config["survival_model"]
    expected = {
        "input_array_order": "CZYX", "embed_dim": 256, "transformer_depth": 6,
        "attention_heads": 8, "mlp_ratio": 4.0, "dropout": 0.1,
        "survival_head_hidden_dim": 64, "survival_head_dropout": 0.3,
        "cnn_head_dropout": 0.3, "batch_size": 8, "cox_risk_set": "full_cohort",
        "epochs": 200, "learning_rate": 0.0001, "weight_decay": 0.0001,
        "warmup_epochs": 10, "learning_rate_schedule": "linear_warmup_cosine",
        "validation_fraction": 0.2, "validation_stratification": "event_and_time_quartile",
        "early_stopping_patience": 30, "cox_tie_method": "breslow",
    }
    for key, value in expected.items():
        if model.get(key) != value:
            raise SystemExit(f"The manuscript survival profile requires {key}={value!r}.")
    aug_expected = {
        "enabled": True, "rotation_degrees_xyz": [10.0, 10.0, 10.0],
        "translation_voxels_xyz": [5.0, 5.0, 1.0], "spatial_scale_range": [0.9, 1.1],
        "intensity_scale_range": [0.9, 1.1], "intensity_shift_max": 0.1,
        "noise_std_max": 0.05, "left_right_flip": False,
    }
    for key, value in aug_expected.items():
        if model["augmentation"].get(key) != value:
            raise SystemExit(f"The manuscript augmentation profile requires {key}={value!r}.")
    attr = config["attribution"]
    attr_expected = {
        "methods": ["occlusion"], "models": ["3d_vit"],
        "occlusion_block_size_xyz": [16, 16, 4], "occlusion_stride_xyz": [8, 8, 2],
        "occlusion_fill_strategy": "training_channel_mean",
        "visualization": "absolute_magnitude", "save_case_maps": True,
    }
    for key, value in attr_expected.items():
        if attr.get(key) != value:
            raise SystemExit(f"The manuscript attribution profile requires {key}={value!r}.")
    retired = {key for key in attr if "shap" in key or key in {"occlusion_fill_value", "save_channel_maps"}}
    if retired:
        raise SystemExit(f"Remove obsolete attribution parameters: {sorted(retired)}")


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    validate_config(config)
    print("Upstream manuscript configuration is valid; this does not verify a trained checkpoint or results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
