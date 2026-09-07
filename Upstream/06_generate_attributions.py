"""Generate patch-occlusion and DeepSHAP maps for frozen survival models."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from io_utils import load_config, read_manifest


def load_model_module():
    path = Path(__file__).with_name("04_train_survival_models.py")
    spec = importlib.util.spec_from_file_location("survival_models", path)
    if spec is None or spec.loader is None:
        raise ImportError("Unable to load the survival model module.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_image(path: str) -> torch.Tensor:
    with np.load(path, allow_pickle=False) as bundle:
        image = np.asarray(bundle["image"], dtype=np.float32)
    if image.ndim != 4:
        raise ValueError("Each processed image must have shape CZYX.")
    return torch.from_numpy(image).unsqueeze(0)


def occlusion_map(model, image: torch.Tensor, block_zyx: tuple[int, int, int], stride_zyx: tuple[int, int, int], fill: float, device):
    model.eval()
    with torch.no_grad():
        baseline = float(model(image.to(device)).cpu().item())
    _, channels, depth, height, width = image.shape
    values = np.zeros((depth, height, width), dtype=np.float32)
    counts = np.zeros_like(values)
    channel_values = np.zeros((channels, depth, height, width), dtype=np.float32)
    channel_counts = np.zeros_like(channel_values)
    for z in range(0, depth, stride_zyx[0]):
        for y in range(0, height, stride_zyx[1]):
            for x in range(0, width, stride_zyx[2]):
                z_end = min(depth, z + block_zyx[0])
                y_end = min(height, y + block_zyx[1])
                x_end = min(width, x + block_zyx[2])
                occluded = image.clone()
                occluded[:, :, z:z_end, y:y_end, x:x_end] = fill
                with torch.no_grad():
                    score = float(model(occluded.to(device)).cpu().item())
                delta = baseline - score
                values[z:z_end, y:y_end, x:x_end] += delta
                counts[z:z_end, y:y_end, x:x_end] += 1.0
                for channel in range(channels):
                    channel_occluded = image.clone()
                    channel_occluded[:, channel, z:z_end, y:y_end, x:x_end] = fill
                    with torch.no_grad():
                        channel_score = float(model(channel_occluded.to(device)).cpu().item())
                    channel_delta = baseline - channel_score
                    channel_values[channel, z:z_end, y:y_end, x:x_end] += channel_delta
                    channel_counts[channel, z:z_end, y:y_end, x:x_end] += 1.0
    values /= np.maximum(counts, 1.0)
    channel_values /= np.maximum(channel_counts, 1.0)
    return baseline, values, channel_values


def deep_shap_map(model, image: torch.Tensor, background: torch.Tensor, device):
    import shap

    model.eval()
    explainer = shap.DeepExplainer(model, background.to(device))
    values = explainer.shap_values(image.to(device), check_additivity=False)
    if isinstance(values, list):
        values = values[0]
    return np.asarray(values, dtype=np.float32).squeeze(0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    frame = read_manifest(args.manifest, config)
    if "processed_npz" not in frame.columns:
        raise ValueError("The attribution manifest must contain the processed_npz column.")
    module = load_model_module()
    device = torch.device("cuda" if torch.cuda.is_available() and str(config["project"]["device"]).lower() == "auto" else "cpu")
    output_dir = args.output_dir
    map_dir = output_dir / "maps"
    map_dir.mkdir(parents=True, exist_ok=True)
    attribution_rows = []
    failures = []
    patient_col = config["data"]["patient_id_column"]
    subtype_col = config["data"]["subtype_column"]
    for checkpoint_path in sorted(args.checkpoint_dir.glob("*.pt")):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model_name = str(checkpoint["model_name"])
        subtype = str(checkpoint["subtype"])
        model = module.make_model(model_name, config)
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device).eval()
        subset = frame[frame[subtype_col].astype(str) == subtype].copy()
        training = subset[subset[config["data"]["cohort_column"]].astype(str).str.lower() == str(config["data"]["training_cohort"]).lower()]
        background_cases = training.head(int(config["attribution"]["shap_background_cases"]))
        if background_cases.empty:
            failures.append({"checkpoint": str(checkpoint_path), "error": "No training background cases."})
            continue
        background = torch.cat([load_image(path) for path in background_cases["processed_npz"]], dim=0)
        block_xyz = [int(value) for value in config["attribution"]["occlusion_block_size_xyz"]]
        stride_xyz = [int(value) for value in config["attribution"]["occlusion_stride_xyz"]]
        block_zyx = tuple(reversed(block_xyz))
        stride_zyx = tuple(reversed(stride_xyz))
        for _, row in subset.iterrows():
            patient_id = str(row[patient_col])
            safe_id = "".join(character if character.isalnum() or character in "-_" else "_" for character in patient_id)
            try:
                image = load_image(row["processed_npz"])
                baseline, occlusion, channel_occlusion = occlusion_map(
                    model,
                    image,
                    block_zyx,
                    stride_zyx,
                    float(config["attribution"]["occlusion_fill_value"]),
                    device,
                )
                shap_values = deep_shap_map(model, image, background, device)
                output_path = map_dir / f"{model_name}_{safe_id}.npz"
                np.savez_compressed(
                    output_path,
                    occlusion_zyx=occlusion,
                    occlusion_channel_zyx=channel_occlusion,
                    deep_shap_czyx=shap_values,
                )
                row_values = {
                    "patient_id": patient_id,
                    "subtype": subtype,
                    "model": model_name,
                    "baseline_risk": baseline,
                    "occlusion_mean": float(np.mean(occlusion)),
                    "occlusion_abs_mean": float(np.mean(np.abs(occlusion))),
                    "deep_shap_abs_mean": float(np.mean(np.abs(shap_values))),
                    "map_path": str(output_path.resolve()),
                }
                for channel_index in range(channel_occlusion.shape[0]):
                    row_values[f"occlusion_channel_{channel_index + 1}_abs_mean"] = float(
                        np.mean(np.abs(channel_occlusion[channel_index]))
                    )
                attribution_rows.append(row_values)
            except Exception as exc:
                failures.append({"checkpoint": str(checkpoint_path), "patient_id": patient_id, "error": str(exc)})
    pd.DataFrame(attribution_rows).to_csv(output_dir / "attribution_summary.csv", index=False)
    pd.DataFrame(failures).to_csv(output_dir / "attribution_failures.csv", index=False)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
