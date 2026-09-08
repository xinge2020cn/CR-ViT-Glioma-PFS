# Upstream analysis modules

This directory contains upstream implementations for four-sequence MRI preprocessing, segmentation dataset preparation, subtype-specific 3D survival models, occlusion sensitivity, bulk ssGSEA, and single-cell analysis. Coverage is not equivalent to a verified reproduction of every manuscript method; see `../CODE_COVERAGE.md` for unresolved differences and the retired fusion entry point.

To run the upstream workflow, prepare a private or governed input directory and create a manifest with one row per patient. No patient-level inputs are included in this repository.

## Unified parameter file

All settings are in `config/article_defaults.yml`. The effective configuration is the single source of truth and must be versioned with each analysis run.

## Input manifest

The manifest is a CSV file with one row per patient. A header template is provided as `manifest_template.csv`. The required columns are:

```text
patient_id,subtype,cohort,T1WI,T2WI,T2_FLAIR,CE_T1WI,tumor_mask,brain_mask,pfs_time_months,event
```

The four MRI and mask columns contain paths to NIfTI files or other formats supported by SimpleITK. The `cohort` value `training` is used for model fitting. Other cohort values are scored as frozen validation data. Patient identifiers must be unique, and all image and mask files must share a patient-specific coordinate system.

## Normal workflow

Run commands from this directory's parent package:

```powershell
python Upstream/09_validate_upstream_configuration.py --config Upstream/config/article_defaults.yml
python Upstream/01_preprocess_mri.py --config Upstream/config/article_defaults.yml --manifest inputs/manifest.csv --output-dir Working/preprocessed
python Upstream/02_prepare_nnunet_dataset.py --config Upstream/config/article_defaults.yml --manifest inputs/manifest.csv --output-dir Working/Dataset501
python Upstream/03_train_nnunet.py --config Upstream/config/article_defaults.yml --dataset-root Working/Dataset501
python Upstream/04_train_survival_models.py --config Upstream/config/article_defaults.yml --manifest Working/preprocessed/processed_manifest.csv --output-dir Working/survival
python Upstream/06_generate_attributions.py --config Upstream/config/article_defaults.yml --manifest Working/preprocessed/processed_manifest.csv --checkpoint-dir Working/survival/checkpoints --output-dir Working/attributions
python Upstream/07_run_bulk_ssgsea.py --config Upstream/config/article_defaults.yml --expression inputs/bulk_expression.csv --metadata inputs/bulk_metadata.csv --signatures inputs/signatures.gmt --output-dir Working/bulk_ssgsea
python Upstream/08_run_single_cell_analysis.py --config Upstream/config/article_defaults.yml --input inputs/single_cell.h5ad --marker-sets inputs/neural_lineage_marker_sets.yml --output-dir Working/single_cell
```

The nnU-Net command is deliberately separate because it requires the nnU-Net environment variables and a compute environment selected by the investigator. The training wrapper prints the exact command and can execute it after `--run` is supplied.

## Occlusion sensitivity: fixed-model inference only

`06_generate_attributions.py` no longer runs DeepSHAP or per-channel occlusion. It jointly occludes the four registered input channels and measures `original risk - occluded risk`. Overlapping-window changes are averaged at each voxel **before** taking the absolute value for display. A large absolute value indicates stronger output sensitivity, not necessarily an increase in risk and not a demonstrated biological mechanism.

The configured dimensions are in **XYZ**; saved tensors are **CZYX**. The input is 192 x 192 x 24 voxels, the window 16 x 16 x 4, and the stride 8 x 8 x 2. Border windows are aligned with the end of each dimension to retain their full size. At these input/window/stride settings the grid is exactly divisible, so every window is full-sized.

Replacement intensities are four scalar channel means calculated exclusively from the corresponding subtype's processed training volumes. They are not recalculated from validation patients. Keep the same channel ordering, preprocessing configuration and input geometry as model training. The checkpoint's saved configuration and state dictionary must match. Missing checkpoints, incompatible dimensions and nonfinite values must cause a failure rather than a substitute map.

Preprocessing now saves the unexpanded `tumor_core_mask` separately from `expanded_crop_mask`. Only the unexpanded core is used to select the axial slice with its largest area. The older `tumor_mask` output incorrectly stored the expanded region; regenerate those old processed inputs before claiming core-based slice selection. The saved input-grid origin, direction, spacing, crop bounding box and preprocessing configuration support geometry checks. Resizing a patient-specific crop changes its voxel spacing: input-window dimensions must not be described as one fixed physical size for all patients.

Raw signed and absolute arrays and provenance are saved, without adding artificial smoothing or inventing numeric color-bar limits. Figure-level display limits should be fixed and reported when actual maps are rendered. Models, MRI and output artifacts remain local, outside this public code release.

`10_render_occlusion_panels.py` renders independent MRI and occlusion PNG images plus a numeric color bar from the saved arrays. It requires the explicit unexpanded core mask, verifies largest-area slice selection and rejects mismatched geometry or an incorrectly formed absolute map. Supply only maps from one checkpoint per run. The default shared color limit is the maximum across supplied display slices; a fixed `--vmax` can be used consistently across figure batches, with any display clipping recorded in JSON. No smoothing is applied. The MRI underlay uses recorded 2nd/98th-percentile display limits. The output images are suitable for separate insertion into an editable PPT; no PPT or clinical heatmap is manufactured by the public tests.

```powershell
python Upstream/10_render_occlusion_panels.py --maps Working/attributions/maps/3d_vit_GBM_CASE_A.npz Working/attributions/maps/3d_vit_GBM_CASE_B.npz --output-dir Working/occlusion_panels/GBM
```

The example filenames must be replaced with actual stage-06 outputs. An all-zero map is flagged rather than assigned a misleading arbitrary color range.

The output-difference and overlapping-window averaging definition follows the established [occlusion sensitivity formulation](https://captum.ai/api/occlusion.html). Window dimensions, stride and subtype-training mean replacement are study-specific settings, not a universal standard.

## Training implementation and validation status

The revised model defaults follow the current supplement's architecture and tuning settings. Cox loss uses Breslow ties and the full fitting-cohort risk set. `batch_size: 8` controls forward microbatches; graphs are retained until the full-cohort loss is computed. This is **not** eight-patient independent Cox risk-set training, and it can require substantial GPU memory. Memory capacity and the reported training platform have not been validated in this revision. Training, checkpoint selection and full-cohort refitting must be performed and logged before these implementations are used to support study results.

`05_fit_crvit_fusion.R` is explicitly retired: its previous joint selection of CNN, ViT and clinical variables at lambda.1se did not implement the current manuscript method. It now stops immediately instead of silently producing an incompatible score.

## Implementation checks

Run from the repository root with the upstream dependencies installed:

```powershell
python -m unittest discover -s Upstream/tests -v
python Upstream/09_validate_upstream_configuration.py --config Upstream/config/article_defaults.yml
```

Small generated-array and toy-network tests verify numerical behavior only. They do not substitute for a governed-data end-to-end run, a trained study checkpoint or clinical validation.

## Reproducibility rules

- Split at the patient level. Never place scans, masks, or cells from the same patient in different model partitions.
- Fit preprocessing-derived statistics, model weights, thresholds, and feature-selection objects on the training cohort only.
- Keep temporal and spatial cohorts frozen during model selection.
- Keep the manifest, effective configuration, package version, dependency versions, and command log with every run.
- Use the raw MRI, mask, and omics inputs that correspond to the manifest.
