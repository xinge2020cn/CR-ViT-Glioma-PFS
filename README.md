# Public Analysis Code Package

This repository contains the R and Python workflow used for the article's imaging, survival, segmentation, reader-agreement, and multi-omics analyses.

This is a code-only release. Patient-level tables, MRI volumes, masks, expression matrices, model checkpoints, generated results, quality-assurance reports, and figure exports are not included. Provide governed inputs locally before running the workflow.

## Scope

The workflow covers the following analysis stages:

1. Input validation and cohort integrity checks.
2. Training-only LASSO-Cox selection of conventional clinical and MRI predictors.
3. Cox refitting and combination of the clinicoradiologic score with supplied 3D-CNN and 3D-ViT scores.
4. Apparent training and frozen validation performance evaluation.
5. Harrell C-index, time-dependent AUC, Brier score, integrated metrics, calibration, decision-curve analysis, proportional-hazards checks, paired bootstrap comparisons, and Kaplan-Meier risk-group analyses.
6. Segmentation-performance summaries and two-reader agreement analysis from supplied patient-level records.

## Upstream modules

The `Upstream/` directory contains the raw-input workflow for MRI preprocessing, nnU-Net dataset preparation, survival-model training, image attribution, bulk ssGSEA, and single-cell analysis. These modules run on an external patient-level manifest plus the corresponding MRI, mask, and omics files. The downstream scripts in `Code/` evaluate externally supplied patient-level score tables.

See `CODE_COVERAGE.md` for the stage-by-stage verification record.

## Directory structure

```text
Code/
  00_validate_inputs.py
  01_run_primary_analysis.R
  02_build_pfs_summary.R
  03_audit_model_scores.R
  04_audit_lasso_selection.R
  05_build_multivariable_cox_results.R
  06_build_km_statistics.R
  07_build_segmentation_summary.py
  08_build_reader_agreement.py
  09_build_figure_s1.py
  10_build_figure_components.py
  11_audit_public_release.py
  12_build_data_dictionary.py
Upstream/
  config/article_defaults.yml
  01_preprocess_mri.py
  02_prepare_nnunet_dataset.py
  03_train_nnunet.py
  04_train_survival_models.py
  05_fit_crvit_fusion.R
  06_generate_attributions.py
  07_run_bulk_ssgsea.py
  08_run_single_cell_analysis.py
  09_validate_upstream_configuration.py
  requirements-upstream.txt
  manifest_template.csv

.gitignore
CODE_COVERAGE.md
LICENSE_PENDING.md
README.md
r-packages.txt
requirements.txt
run_all.R
```

## Local input layout

The downstream workflow requires external patient-level tables. The repository does not provide patient rows. Required column names and validation rules are implemented in the analysis workflow.

The upstream workflow requires an external manifest together with MRI, mask, expression, marker, and ligand-receptor files.

## Run the complete workflow

From a local analysis workspace containing this repository and the governed input files:

```powershell
Rscript .\run_all.R
```

For a quick smoke test:

```powershell
Rscript .\run_all.R --bootstrap=100 --workers=2
```

The default analysis uses 1,000 patient-level bootstrap resamples. Figure jitter uses fixed seeds only for visual reproducibility. The workflow consumes externally supplied patient-level outcomes and model scores; it does not create them.

## Required software

- R 4.5 or later.
- Python 3.10 or later.
- R packages listed in `r-packages.txt`.
- Python packages listed in `requirements.txt`.
- Upstream dependencies listed in `Upstream/requirements-upstream.txt` when the raw-input workflow is used.

## Public-release checklist

Before publishing a derived release, confirm that patient identifiers are de-identified, all data-use and ethics requirements are satisfied, the intended model-training and omics source code is present, dependency versions are recorded, and a project-selected open-source license replaces `LICENSE_PENDING.md`.
