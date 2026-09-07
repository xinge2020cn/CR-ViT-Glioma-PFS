# Article Workflow Code Coverage

This file records which article workflow stages are supported by executable code in this package and which stages cannot be verified from the supplied source directory.

| Article workflow stage | Package implementation | Coverage status |
| --- | --- | --- |
| Cohort and input integrity | `Code/00_validate_inputs.py` | Executable validation of supplied patient-level tables |
| Clinical and MRI descriptive summaries | `Code/02_build_pfs_summary.R`, `Code/03_audit_model_scores.R` | Executable from supplied tables |
| Training-only LASSO-Cox selection | `Code/01_run_primary_analysis.R`, `Code/04_audit_lasso_selection.R` | Executable from supplied tables |
| Clinicoradiologic, 3D-CNN, 3D-ViT, and CR-ViT score evaluation | `Code/01_run_primary_analysis.R` | Executable when model scores are supplied as columns |
| Validation Cox models and proportional-hazards checks | `Code/05_build_multivariable_cox_results.R` | Executable from supplied tables |
| Kaplan-Meier risk-group analysis | `Code/06_build_km_statistics.R` | Executable from supplied tables |
| Segmentation-performance summary | `Code/07_build_segmentation_summary.py`, `Code/09_build_figure_s1.py` | Executable from supplied patient-level metrics |
| Two-reader agreement | `Code/08_build_reader_agreement.py` | Executable from supplied reader ratings |
| Figure 3 and Supplementary Figures S1-S6 components | `Code/09_build_figure_s1.py`, `Code/10_build_figure_components.py` | Executable after statistical outputs are created |
| MRI registration, resampling, bias correction, skull stripping, and intensity normalization | No verified source implementation or raw image inputs were supplied | Not reproducible from the current source directory |
| nnU-Net training and manual-mask adjudication | No verified source implementation, masks, or training data were supplied | Not reproducible from the current source directory |
| 3D-CNN and 3D-ViT architecture training, frozen-model inference, and attribution maps | No verified source implementation or image inputs were supplied | Not reproducible from the current source directory |
| Single-cell, bulk transcriptomic, ssGSEA, CNV, pathway, and ligand-receptor analyses | No verified expression matrices or analysis scripts were supplied | Not reproducible from the current source directory |

The missing upstream stages are intentionally identified rather than replaced with newly invented code. Adding them requires the original source implementation, exact configuration, raw or governed-access inputs, and a documented data-use and ethics basis.
