"""Register, normalize, crop, and resize the four MRI sequences."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from io_utils import load_config, read_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_image(path: Path, sitk):
    if not path.exists():
        raise FileNotFoundError(path)
    image = sitk.ReadImage(str(path))
    return image


def orient(image, orientation: str, sitk):
    return sitk.DICOMOrient(image, orientation)


def register_rigid(fixed, moving, registration_cfg: dict, sitk, random_seed: int):
    fixed_float = sitk.Cast(fixed, sitk.sitkFloat32)
    moving_float = sitk.Cast(moving, sitk.sitkFloat32)
    initial = sitk.CenteredTransformInitializer(
        fixed_float,
        moving_float,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )
    method = sitk.ImageRegistrationMethod()
    method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=int(registration_cfg["metric_bins"]))
    method.SetMetricSamplingStrategy(method.RANDOM)
    method.SetMetricSamplingPercentage(0.2, seed=random_seed)
    method.SetInterpolator(sitk.sitkLinear)
    method.SetOptimizerAsGradientDescent(
        learningRate=float(registration_cfg["learning_rate"]),
        numberOfIterations=int(registration_cfg["iterations"]),
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
    )
    method.SetOptimizerScalesFromPhysicalShift()
    method.SetShrinkFactorsPerLevel([int(x) for x in registration_cfg["shrink_factors"]])
    method.SetSmoothingSigmasPerLevel([float(x) for x in registration_cfg["smoothing_sigmas_mm"]])
    method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    method.SetInitialTransform(initial, inPlace=False)
    transform = method.Execute(fixed_float, moving_float)
    return sitk.Resample(moving, fixed, transform, sitk.sitkLinear, 0.0, moving.GetPixelID())


def resample_to_spacing(image, spacing_xyz: list[float], interpolator, sitk):
    old_size = image.GetSize()
    old_spacing = image.GetSpacing()
    new_size = [
        max(1, int(round(size * old_sp / new_sp)))
        for size, old_sp, new_sp in zip(old_size, old_spacing, spacing_xyz)
    ]
    return sitk.Resample(
        image,
        new_size,
        sitk.Transform(),
        interpolator,
        image.GetOrigin(),
        spacing_xyz,
        image.GetDirection(),
        0.0,
        image.GetPixelID(),
    )


def resample_to_reference(image, reference, interpolator, sitk):
    return sitk.Resample(image, reference, sitk.Transform(), interpolator, 0.0, image.GetPixelID())


def n4_and_zscore(image, brain_mask, n4_cfg: dict, intensity_cfg: dict, sitk):
    image = sitk.Cast(image, sitk.sitkFloat32)
    brain_mask = sitk.Cast(brain_mask > 0, sitk.sitkUInt8)
    if n4_cfg["enabled"]:
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetMaximumNumberOfIterations([int(value) for value in n4_cfg["iterations"]])
        corrector.SetConvergenceThreshold(float(n4_cfg["convergence_threshold"]))
        image = corrector.Execute(image, brain_mask)
    values = sitk.GetArrayViewFromImage(image).astype(np.float32)
    mask = sitk.GetArrayViewFromImage(brain_mask) > 0
    if not np.any(mask):
        raise ValueError("The brain mask is empty after resampling.")
    mean = float(values[mask].mean())
    std = float(values[mask].std())
    if not np.isfinite(std) or std <= 0:
        raise ValueError("Brain-mask intensity standard deviation is not positive.")
    normalized = (values - mean) / std
    limits = intensity_cfg["clip_after_normalization"]
    normalized = np.clip(normalized, float(limits[0]), float(limits[1])).astype(np.float32)
    normalized[~mask] = 0.0
    output = sitk.GetImageFromArray(normalized)
    output.CopyInformation(image)
    return output


def expand_mask(mask, expansion_mm: float, sitk):
    spacing = mask.GetSpacing()
    radius = [max(1, int(math.ceil(expansion_mm / float(value)))) for value in spacing]
    binary = sitk.Cast(mask > 0, sitk.sitkUInt8)
    return sitk.BinaryDilate(binary, radius, sitk.sitkBall)


def crop_and_resize(images, mask, target_size_xyz: list[int], sitk):
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(sitk.Cast(mask > 0, sitk.sitkUInt8))
    if not stats.HasLabel(1):
        raise ValueError("The expanded tumor mask contains no foreground voxels.")
    x, y, z, sx, sy, sz = stats.GetBoundingBox(1)
    index = [int(x), int(y), int(z)]
    size = [int(sx), int(sy), int(sz)]
    crop_mask = sitk.RegionOfInterest(mask, size, index)
    crop_images = [sitk.RegionOfInterest(image, size, index) for image in images]
    crop_size = crop_mask.GetSize()
    crop_spacing = crop_mask.GetSpacing()
    output_spacing = [
        float(crop_size[i] * crop_spacing[i] / target_size_xyz[i])
        for i in range(3)
    ]
    resized_images = [
        sitk.Resample(
            image,
            target_size_xyz,
            sitk.Transform(),
            sitk.sitkLinear,
            crop_mask.GetOrigin(),
            output_spacing,
            crop_mask.GetDirection(),
            0.0,
            sitk.sitkFloat32,
        )
        for image in crop_images
    ]
    resized_mask = sitk.Resample(
        crop_mask,
        target_size_xyz,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        crop_mask.GetOrigin(),
        output_spacing,
        crop_mask.GetDirection(),
        0,
        sitk.sitkUInt8,
    )
    return resized_images, resized_mask, output_spacing


def process_case(row: pd.Series, config: dict, output_dir: Path, sitk, manifest_dir: Path) -> str:
    def input_path(value: object) -> Path:
        path = Path(str(value))
        return path if path.is_absolute() else manifest_dir / path

    data_cfg = config["data"]
    pre_cfg = config["mri_preprocessing"]
    sequence_columns = list(data_cfg["image_columns"].items())
    images = {
        sequence: orient(read_image(input_path(row[column]), sitk), pre_cfg["orientation"], sitk)
        for sequence, column in sequence_columns
    }
    reference_sequence = pre_cfg["reference_sequence"]
    reference = images[reference_sequence]
    registration_cfg = pre_cfg["registration"]
    aligned = {}
    for sequence, image in images.items():
        aligned[sequence] = image if sequence == reference_sequence else register_rigid(
            reference, image, registration_cfg, sitk, int(config["project"]["random_seed"])
        )
    tumor_mask = orient(read_image(input_path(row[data_cfg["mask_column"]]), sitk), pre_cfg["orientation"], sitk)
    brain_mask = orient(read_image(input_path(row[data_cfg["brain_mask_column"]]), sitk), pre_cfg["orientation"], sitk)
    tumor_mask = resample_to_reference(tumor_mask, reference, sitk.sitkNearestNeighbor, sitk)
    brain_mask = resample_to_reference(brain_mask, reference, sitk.sitkNearestNeighbor, sitk)
    target_spacing = [float(x) for x in pre_cfg["target_spacing_mm_xyz"]]
    resampled_images = [
        resample_to_spacing(aligned[sequence], target_spacing, sitk.sitkLinear, sitk)
        for sequence, _ in sequence_columns
    ]
    resampled_tumor = resample_to_spacing(tumor_mask, target_spacing, sitk.sitkNearestNeighbor, sitk)
    resampled_brain = resample_to_spacing(brain_mask, target_spacing, sitk.sitkNearestNeighbor, sitk)
    normalized = [
        n4_and_zscore(image, resampled_brain, pre_cfg["n4_bias_correction"], pre_cfg["intensity_normalization"], sitk)
        for image in resampled_images
    ]
    expanded = expand_mask(resampled_tumor, float(pre_cfg["tumor_core_expansion_mm"]), sitk)
    resized_images, resized_mask, spacing = crop_and_resize(
        normalized,
        expanded,
        [int(x) for x in pre_cfg["target_size_voxels_xyz"]],
        sitk,
    )
    arrays = [sitk.GetArrayFromImage(image).astype(np.float32) for image in resized_images]
    image_array = np.stack(arrays, axis=0)
    mask_array = (sitk.GetArrayFromImage(resized_mask) > 0).astype(np.uint8)
    patient_id = str(row[data_cfg["patient_id_column"]])
    safe_id = "".join(character if character.isalnum() or character in "-_" else "_" for character in patient_id)
    output_path = output_dir / f"{safe_id}.npz"
    np.savez_compressed(
        output_path,
        image=image_array,
        tumor_mask=mask_array,
        spacing_xyz=np.asarray(spacing, dtype=np.float32),
        sequence_names=np.asarray([sequence for sequence, _ in sequence_columns]),
    )
    return str(output_path.resolve())


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    frame = read_manifest(args.manifest, config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        import SimpleITK as sitk
    except ImportError as exc:
        raise SystemExit("SimpleITK is required for MRI preprocessing.") from exc
    processed_paths = []
    for index, row in frame.iterrows():
        print(f"Processing patient {index + 1}/{len(frame)}", flush=True)
        processed_paths.append(process_case(row, config, args.output_dir, sitk, args.manifest.parent.resolve()))
    output_manifest = frame.copy()
    output_manifest["processed_npz"] = processed_paths
    output_manifest.to_csv(args.output_dir / "processed_manifest.csv", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
