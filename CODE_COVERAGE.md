# Article Workflow Code Coverage

This file summarizes the article workflow stages covered by executable code in this package.

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
| MRI registration, resampling, bias correction, skull stripping, and intensity normalization | `Upstream/01_preprocess_mri.py` | Executable from a patient-level MRI manifest |
| nnU-Net dataset preparation and full-resolution training | `Upstream/02_prepare_nnunet_dataset.py`, `Upstream/03_train_nnunet.py` | Executable with nnU-Net and segmentation inputs |
| 3D-CNN and 3D-ViT architecture training, frozen-model inference, and attribution maps | `Upstream/04_train_survival_models.py`, `Upstream/06_generate_attributions.py` | Executable from processed MRI arrays |
| Clinicoradiologic LASSO-Cox fusion | `Upstream/05_fit_crvit_fusion.R` | Executable from model predictions and clinical data |
| Bulk transcriptomic ssGSEA | `Upstream/07_run_bulk_ssgsea.py` | Executable from expression, metadata, and GMT files |
| Single-cell QC, clustering, neural-lineage state scoring, CNV, and ligand-receptor scoring | `Upstream/08_run_single_cell_analysis.py` | Executable from an h5ad object and supplied marker/pair files |

All effective settings for the upstream workflow are kept in `Upstream/config/article_defaults.yml`.
