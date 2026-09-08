"""Joint-channel 3D occlusion sensitivity for a locked ViT risk model.

No training is performed. Signed maps are original-minus-occluded score changes
averaged across overlapping windows; displayed magnitudes are abs(mean(delta)).
They describe perturbation sensitivity, not causal biological importance.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from io_utils import load_config, read_manifest, write_json


TRAINING_ONLY_KEYS = {
    "models", "batch_size", "epochs", "learning_rate", "weight_decay",
    "validation_fraction", "early_stopping_patience", "gradient_clip_norm",
    "cox_tie_method", "augmentation", "warmup_epochs", "scheduler",
    "learning_rate_schedule", "validation_stratification", "refit_full_training",
    "score_output_name", "split_stratification", "max_epochs",
}


def load_model_module():
    path = Path(__file__).with_name("04_train_survival_models.py")
    spec = importlib.util.spec_from_file_location("survival_models", path)
    if spec is None or spec.loader is None:
        raise ImportError("Unable to load the survival model module.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_attribution_config(config: dict) -> dict:
    cfg = config["attribution"]
    if cfg.get("methods") != ["occlusion"]:
        raise ValueError("attribution.methods must be [occlusion]; unsupported methods are not run.")
    if cfg.get("models", ["3d_vit"]) != ["3d_vit"]:
        raise ValueError("This workflow supports the locked 3d_vit model only.")
    if cfg.get("occlusion_fill_strategy") != "training_channel_mean":
        raise ValueError("occlusion_fill_strategy must be training_channel_mean.")
    if cfg.get("visualization", "absolute_magnitude") != "absolute_magnitude":
        raise ValueError("visualization must be absolute_magnitude.")
    if not cfg.get("save_case_maps", True):
        raise ValueError("Raw case maps must be retained (save_case_maps: true).")
    if config["survival_model"].get("input_array_order") != "CZYX":
        raise ValueError("The locked model input_array_order must be CZYX.")
    if len(config["data"]["image_columns"]) != 4:
        raise ValueError("Joint-channel occlusion requires four MRI channels.")
    for key in ("occlusion_block_size_xyz", "occlusion_stride_xyz"):
        values = cfg.get(key)
        if not isinstance(values, (list, tuple)) or len(values) != 3:
            raise ValueError(f"{key} must contain three positive integer XYZ values.")
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in values):
            raise ValueError(f"{key} must contain three positive integer XYZ values.")
    if list(cfg["occlusion_block_size_xyz"]) != list(config["survival_model"]["patch_size_xyz"]):
        raise ValueError("The occlusion window must match the documented ViT patch size.")
    for size, block, stride in zip(config["mri_preprocessing"]["target_size_voxels_xyz"],
                                  cfg["occlusion_block_size_xyz"], cfg["occlusion_stride_xyz"]):
        window_starts(int(size), block, stride)
    return cfg


def input_shape(config: dict) -> tuple:
    return (len(config["data"]["image_columns"]), *reversed(config["mri_preprocessing"]["target_size_voxels_xyz"]))


def load_image(path: str | Path, config: dict) -> torch.Tensor:
    with np.load(path, allow_pickle=False) as bundle:
        image = np.asarray(bundle["image"], dtype=np.float32)
        if "sequence_names" not in bundle:
            raise ValueError("Processed image must retain sequence_names to verify channel order.")
        if bundle["sequence_names"].tolist() != list(config["data"]["image_columns"]):
            raise ValueError("MRI channel order mismatch.")
        if "preprocessing_config_json" in bundle:
            recorded = json.loads(str(bundle["preprocessing_config_json"].item()))
            if recorded != config["mri_preprocessing"]:
                raise ValueError("Processed image preprocessing metadata does not match checkpoint.")
    if tuple(image.shape) != input_shape(config):
        raise ValueError(f"Processed image shape {image.shape} does not match CZYX {input_shape(config)}.")
    if not np.isfinite(image).all():
        raise ValueError("Processed image contains non-finite values.")
    return torch.from_numpy(image).unsqueeze(0)


def compute_channel_means(training: pd.DataFrame, config: dict) -> tuple[np.ndarray, dict]:
    """All unaugmented, complete preprocessed volumes of this subtype's training set."""
    if training.empty:
        raise ValueError("No subtype-specific training volumes for channel means.")
    data = config["data"]
    cohort = training[data["cohort_column"]].astype(str).str.strip().str.casefold()
    if not cohort.eq(str(data["training_cohort"]).strip().casefold()).all():
        raise ValueError("Channel means must not include validation-cohort volumes.")
    if training[data["subtype_column"]].astype(str).nunique() != 1:
        raise ValueError("Channel means must be subtype-specific.")
    if training[data["patient_id_column"]].duplicated().any():
        raise ValueError("Training volumes must contain one row per patient.")
    sums = np.zeros(4, dtype=np.float64)
    voxel_count, metadata_verified, identifiers = 0, True, []
    for _, row in training.iterrows():
        image = load_image(row["processed_npz"], config).squeeze(0).numpy()
        sums += image.sum(axis=(1, 2, 3), dtype=np.float64)
        voxel_count += int(np.prod(image.shape[1:]))
        with np.load(row["processed_npz"], allow_pickle=False) as bundle:
            metadata_verified &= "preprocessing_config_json" in bundle
        identifiers.append(str(row[data["patient_id_column"]]))
    means = sums / voxel_count
    return means.astype(np.float32), {
        "subtype": str(training.iloc[0][data["subtype_column"]]),
        "training_patient_count": len(training), "training_patient_ids": identifiers,
        "voxels_per_channel": voxel_count, "channel_names": list(data["image_columns"]),
        "channel_means": means.tolist(),
        "mean_definition": "voxel-weighted mean over complete preprocessed training volumes; no augmentation",
        "preprocessing_metadata_verified": bool(metadata_verified),
    }


def architecture_signature(config: dict) -> dict:
    return {
        "survival_model": {key: value for key, value in config["survival_model"].items() if key not in TRAINING_ONLY_KEYS},
        "mri_preprocessing": config["mri_preprocessing"],
        "image_columns": list(config["data"]["image_columns"].items()),
    }


def load_locked_model(checkpoint_path: Path, external_config: dict, module, device):
    # Never fall back to unrestricted pickle loading for downloaded checkpoints.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("config"), dict):
        raise ValueError("Checkpoint must contain its original training config and state_dict.")
    for key in ("model_name", "subtype", "state_dict"):
        if key not in checkpoint:
            raise ValueError(f"Checkpoint is missing {key}.")
    if str(checkpoint["model_name"]) != "3d_vit":
        raise ValueError("Attribution requires a 3d_vit checkpoint.")
    if architecture_signature(checkpoint["config"]) != architecture_signature(external_config):
        raise ValueError("Checkpoint architecture, channel order, or preprocessing differs from supplied config; do not adapt weights silently.")
    model = module.make_model(str(checkpoint["model_name"]), checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval().requires_grad_(False)
    return model, checkpoint


def window_starts(size: int, block: int, stride: int) -> list[int]:
    """Full-sized windows only; the final one ends at the image boundary."""
    if not 0 < block <= size or not 0 < stride <= block:
        raise ValueError("Window must fit image and 0 < stride <= window is required for coverage.")
    starts = list(range(0, size - block + 1, stride))
    if starts[-1] != size - block:
        starts.append(size - block)
    return starts


def scalar_score(model, image: torch.Tensor) -> float:
    result = model(image)
    if not isinstance(result, torch.Tensor) or result.numel() != 1:
        raise ValueError("Locked model must emit one continuous risk score per volume.")
    value = float(result.detach().cpu().item())
    if not np.isfinite(value):
        raise ValueError("Locked model emitted a non-finite risk score.")
    return value


def occlusion_map(model, image: torch.Tensor, block_zyx: tuple[int, int, int],
                  stride_zyx: tuple[int, int, int], channel_means, device) -> dict:
    if image.ndim != 5 or image.shape[0] != 1 or image.shape[1] != 4:
        raise ValueError("Occlusion expects one four-channel image: 1,C,Z,Y,X.")
    if not torch.isfinite(image).all():
        raise ValueError("Image contains non-finite values.")
    if len(block_zyx) != 3 or len(stride_zyx) != 3:
        raise ValueError("Window and stride must contain three ZYX dimensions.")
    fill = torch.as_tensor(channel_means, dtype=image.dtype, device=device)
    if fill.shape != (4,) or not torch.isfinite(fill).all():
        raise ValueError("One finite mean per MRI channel is required.")
    fill = fill.reshape(1, 4, 1, 1, 1)
    shape = tuple(image.shape[2:])
    axes = [window_starts(size, block, stride) for size, block, stride in zip(shape, block_zyx, stride_zyx)]
    accumulated = np.zeros(shape, dtype=np.float64)
    counts = np.zeros(shape, dtype=np.uint32)
    bounds, deltas = [], []
    model.to(device).eval().requires_grad_(False)
    image = image.to(device)
    with torch.inference_mode():
        baseline = scalar_score(model, image)
        for z, y, x in itertools.product(*axes):
            ends = (z + block_zyx[0], y + block_zyx[1], x + block_zyx[2])
            region = (slice(z, ends[0]), slice(y, ends[1]), slice(x, ends[2]))
            occluded = image.clone()
            occluded[(slice(None), slice(None), *region)] = fill
            delta = baseline - scalar_score(model, occluded)
            accumulated[region] += delta
            counts[region] += 1
            bounds.append((z, y, x, *ends))
            deltas.append(delta)
    if not counts.min() > 0:
        raise RuntimeError("Occlusion did not cover complete input volume.")
    signed = (accumulated / counts).astype(np.float32)
    return {
        "baseline_risk": baseline, "occlusion_signed_zyx": signed,
        "occlusion_absolute_zyx": np.abs(signed), "occlusion_overlap_count_zyx": counts,
        "window_bounds_zyx": np.asarray(bounds, dtype=np.int32),
        "window_score_differences": np.asarray(deltas, dtype=np.float64),
    }


def display_geometry(processed_path: str | Path, expected_shape) -> tuple[dict, dict]:
    """Keep exact input crop; an expanded crop mask is not a tumor-core mask."""
    arrays, metadata = {}, {"axial_slice_index": None, "slice_selection": "unavailable: explicit unexpanded tumor_core_mask not supplied"}
    with np.load(processed_path, allow_pickle=False) as bundle:
        arrays["image_czyx"] = np.asarray(bundle["image"], dtype=np.float32)
        for key in ("spacing_xyz", "sequence_names", "crop_index_xyz", "crop_size_xyz", "origin_xyz", "direction_xyz"):
            if key in bundle:
                arrays[key] = bundle[key]
        metadata["preprocessing_metadata_verified"] = "preprocessing_config_json" in bundle
        if "tumor_core_mask" in bundle:
            core = np.asarray(bundle["tumor_core_mask"])
            if core.shape != tuple(expected_shape) or not np.isfinite(core).all():
                raise ValueError("tumor_core_mask must be finite and match the model ZYX grid.")
            core = core > 0
            if not core.any():
                raise ValueError("tumor_core_mask is empty; largest-core slice cannot be selected.")
            index = int(core.sum(axis=(1, 2)).argmax())
            arrays["tumor_core_mask_zyx"] = core.astype(np.uint8)
            arrays["axial_slice_index"] = np.asarray(index, dtype=np.int32)
            metadata.update(axial_slice_index=index, slice_selection="largest unexpanded tumor-core area; first slice on tie")
    return arrays, metadata


def safe_component(value: str) -> str:
    safe = "".join(character if character.isalnum() or character in "-_" else "_" for character in value)
    if not safe:
        raise ValueError("Output identifier cannot be empty.")
    return safe


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    attribution = validate_attribution_config(config)
    frame = read_manifest(args.manifest, config)
    if "processed_npz" not in frame.columns:
        raise ValueError("Attribution manifest must contain processed_npz.")
    frame["processed_npz"] = frame["processed_npz"].map(
        lambda value: str((args.manifest.parent / str(value)).resolve()) if not Path(str(value)).is_absolute() else str(value)
    )
    paths = sorted(args.checkpoint_dir.glob("3d_vit_*.pt"))
    if not paths:
        raise FileNotFoundError("No 3d_vit_*.pt locked checkpoints found; no maps generated.")
    module = load_model_module()
    device_name = str(config["project"].get("device", "auto")).lower()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device_name == "auto" else torch.device(device_name)
    output_dir, map_dir = args.output_dir, args.output_dir / "maps"
    map_dir.mkdir(parents=True, exist_ok=True)
    rows, failures, checkpoints = [], [], []
    patient_col, subtype_col, cohort_col = (config["data"][key] for key in ("patient_id_column", "subtype_column", "cohort_column"))
    training_value = str(config["data"]["training_cohort"]).strip().casefold()
    seen_subtypes, targets = set(), set()
    block = tuple(reversed(attribution["occlusion_block_size_xyz"]))
    stride = tuple(reversed(attribution["occlusion_stride_xyz"]))
    for checkpoint_path in paths:
        try:
            model, checkpoint = load_locked_model(checkpoint_path, config, module, device)
            subtype, model_name = str(checkpoint["subtype"]), str(checkpoint["model_name"])
            if subtype in seen_subtypes:
                raise ValueError(f"Multiple locked checkpoints for subtype {subtype}.")
            seen_subtypes.add(subtype)
            subset = frame[frame[subtype_col].astype(str) == subtype].copy()
            if subset.empty:
                raise ValueError(f"No cases match checkpoint subtype {subtype}.")
            training = subset[subset[cohort_col].astype(str).str.strip().str.casefold() == training_value]
            means, mean_metadata = compute_channel_means(training, checkpoint["config"])
            checkpoint_hash = sha256(checkpoint_path)
            checkpoints.append({"checkpoint": str(checkpoint_path.resolve()), "checkpoint_sha256": checkpoint_hash, **mean_metadata})
        except Exception as exc:
            failures.append({"checkpoint": str(checkpoint_path), "patient_id": "", "error": str(exc)})
            continue
        for _, row in subset.iterrows():
            patient_id = str(row[patient_col])
            try:
                image = load_image(row["processed_npz"], checkpoint["config"])
                geometry, geometry_meta = display_geometry(row["processed_npz"], image.shape[2:])
                maps = occlusion_map(model, image, block, stride, means, device)
                path = map_dir / f"{model_name}_{safe_component(subtype)}_{safe_component(patient_id)}.npz"
                if path in targets:
                    raise ValueError("Sanitized output identifiers collide.")
                targets.add(path)
                metadata = {
                    "method": "3D occlusion sensitivity", "patient_id": patient_id, "subtype": subtype,
                    "model": model_name, "checkpoint_sha256": checkpoint_hash,
                    "processed_npz_sha256": sha256(Path(row["processed_npz"])), "input_array_order": "CZYX",
                    "window_size_xyz": attribution["occlusion_block_size_xyz"], "stride_xyz": attribution["occlusion_stride_xyz"],
                    "boundary_policy": "full-size final window anchored at image boundary",
                    "channels_occluded": "all four jointly", "channel_means": means.tolist(),
                    "signed_definition": "mean(original risk minus occluded risk) over covering windows",
                    "display_definition": "absolute value of overlap-averaged signed map; not direction of risk", **geometry_meta,
                }
                np.savez_compressed(path, **maps, **geometry, channel_means=means,
                                    metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))
                rows.append({"patient_id": patient_id, "subtype": subtype, "model": model_name,
                             "baseline_risk": maps["baseline_risk"],
                             "occlusion_signed_mean": float(maps["occlusion_signed_zyx"].mean()),
                             "occlusion_absolute_mean": float(maps["occlusion_absolute_zyx"].mean()),
                             "window_count": len(maps["window_score_differences"]),
                             "axial_slice_index": geometry_meta["axial_slice_index"], "map_path": str(path.resolve())})
            except Exception as exc:
                failures.append({"checkpoint": str(checkpoint_path), "patient_id": patient_id, "error": str(exc)})
    for subtype in sorted(set(frame[subtype_col].dropna().astype(str)) - seen_subtypes):
        failures.append({"checkpoint": "", "patient_id": "", "error": f"No compatible locked checkpoint for subtype {subtype}."})
    pd.DataFrame(rows, columns=["patient_id", "subtype", "model", "baseline_risk", "occlusion_signed_mean", "occlusion_absolute_mean", "window_count", "axial_slice_index", "map_path"]).to_csv(output_dir / "attribution_summary.csv", index=False)
    pd.DataFrame(failures, columns=["checkpoint", "patient_id", "error"]).to_csv(output_dir / "attribution_failures.csv", index=False)
    write_json(output_dir / "attribution_metadata.json", {
        "config": config, "checkpoints": checkpoints, "generated_cases": len(rows), "failure_count": len(failures),
        "interpretation": "perturbation sensitivity, not causal attribution",
        "status": "complete" if rows and not failures else "incomplete",
    })
    return 0 if rows and not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
