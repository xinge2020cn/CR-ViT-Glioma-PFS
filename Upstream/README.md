# Upstream analysis modules

This directory contains the executable upstream modules that are needed before the patient-level statistical workflow in `../Code/`. The modules follow the article workflow: four-sequence MRI preprocessing, tumor-core segmentation dataset preparation, subtype-specific 3D survival models, image attribution, bulk ssGSEA, and single-cell analysis.

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
Rscript Upstream/05_fit_crvit_fusion.R --manifest inputs/manifest.csv --prediction-dir Working/survival/predictions --output-dir Working/crvit
python Upstream/06_generate_attributions.py --config Upstream/config/article_defaults.yml --manifest Working/preprocessed/processed_manifest.csv --checkpoint-dir Working/survival/checkpoints --output-dir Working/attributions
python Upstream/07_run_bulk_ssgsea.py --config Upstream/config/article_defaults.yml --expression inputs/bulk_expression.csv --metadata inputs/bulk_metadata.csv --signatures inputs/signatures.gmt --output-dir Working/bulk_ssgsea
python Upstream/08_run_single_cell_analysis.py --config Upstream/config/article_defaults.yml --input inputs/single_cell.h5ad --marker-sets inputs/neural_lineage_marker_sets.yml --output-dir Working/single_cell
```

The nnU-Net command is deliberately separate because it requires the nnU-Net environment variables and a compute environment selected by the investigator. The training wrapper prints the exact command and can execute it after `--run` is supplied.

## Reproducibility rules

- Split at the patient level. Never place scans, masks, or cells from the same patient in different model partitions.
- Fit preprocessing-derived statistics, model weights, thresholds, and feature-selection objects on the training cohort only.
- Keep temporal and spatial cohorts frozen during model selection.
- Keep the manifest, effective configuration, package version, dependency versions, and command log with every run.
- Use the raw MRI, mask, and omics inputs that correspond to the manifest.
