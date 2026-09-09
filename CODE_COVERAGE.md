# Article Workflow Code Coverage

This file summarizes code coverage from the 2026-09-08 manuscript audit and the 2026-09-09 downstream statistical synchronization. "Executable" alone does not establish that the implementation reproduces the current manuscript. The local simulated-data analysis was rerun after an outcome-blind tumor-composition repair; no study neural-network training or clinical-observation validation was performed.

| Article workflow stage | Package implementation | Coverage status |
| --- | --- | --- |
| Cohort and input integrity | `Code/00_validate_inputs.py` | Executable validation of external patient-level tables |
| Clinical and MRI descriptive summaries | `Code/02_build_pfs_summary.R`, `Code/03_audit_model_scores.R` | Executable from external tables |
| Training-only LASSO-Cox selection | `Code/01_run_primary_analysis.R`, `Code/04_audit_lasso_selection.R` | Executable from external tables |
| Clinicoradiologic, 3D-CNN, 3D-ViT, and CR-ViT score evaluation | `Code/01_run_primary_analysis.R` | Executable when model scores are supplied as columns |
| Validation Cox models and proportional-hazards checks | `Code/05_build_multivariable_cox_results.R` | Executable from external tables |
| Kaplan-Meier risk-group analysis | `Code/06_build_km_statistics.R` | Executable from external tables |
| Segmentation-performance summary | `Code/07_build_segmentation_summary.py`, `Code/09_build_figure_s1.py` | Executable from external patient-level metrics |
| Two-reader agreement | `Code/08_build_reader_agreement.py` | Executable from external reader ratings |
| MRI registration, resampling, bias correction, skull stripping, and intensity normalization | `Upstream/01_preprocess_mri.py` | Executable from a patient-level MRI manifest |
| nnU-Net dataset preparation and full-resolution training | `Upstream/02_prepare_nnunet_dataset.py`, `Upstream/03_train_nnunet.py` | Executable with nnU-Net and segmentation inputs |
| 3D-CNN and 3D-ViT training | `Upstream/04_train_survival_models.py` | Architecture/defaults and risk-set handling revised against the supplement; generated-input tests only, no study training/checkpoint validation |
| Fixed-ViT occlusion sensitivity | `Upstream/06_generate_attributions.py` | Occlusion-only implementation; matching checkpoints and processed inputs required; no study maps regenerated |
| Independent PNG attribution panels and numeric color bar | `Upstream/10_render_occlusion_panels.py` | Reads saved occlusion arrays, verifies core-based slice and geometry, shares a numeric color scale within one checkpoint; no study panels regenerated |
| Legacy upstream fusion | `Upstream/05_fit_crvit_fusion.R` | Retired and fails explicitly; former formula and lambda choice do not match the current supplement |
| Bulk transcriptomic ssGSEA | `Upstream/07_run_bulk_ssgsea.py` | Executable from expression, metadata, and GMT files |
| Single-cell QC, clustering, neural-lineage state scoring, CNV, and ligand-receptor scoring | `Upstream/08_run_single_cell_analysis.py` | Executable from an h5ad object and supplied marker/pair files |

All effective settings for the upstream workflow are kept in `Upstream/config/article_defaults.yml`.

## Changes checked against the current methods

- Removed the former DeepSHAP runtime dependency and attribution branch. Occlusion uses joint four-channel replacement, subtype-training channel means, overlapping-window averaging, and a separate absolute-value display array.
- Separated the unexpanded tumor core from the expanded crop mask and retained resized-grid geometry and preprocessing metadata.
- Aligned ViT embedding/depth/heads and survival bottleneck, CNN head dropout, augmentation, optimization defaults and shared patient partitioning with the current supplement. Train/tune Cox computations now use full risk sets instead of averaging unrelated microbatch losses. This changes a future training implementation; it does not retroactively validate existing scores.
- Manifest checks reject nonbinary events before integer conversion and reject nonfinite survival times.

## Open items: do not claim complete manuscript reproduction

1. **Combined-model formula reconciled:** current Supplementary Appendix S4 specifies the selected individual clinical/MRI terms jointly with standardized ViT in an unpenalized Cox model, matching `Code/01_run_primary_analysis.R`. The former two-linear-predictor fusion is not used. The 2026-09-09 local simulated-data repair reran the downstream analysis and synchronized model outputs; this does not validate the upstream neural-network training.
2. **Cross-fitting scope:** outer folds in the downstream code operate on supplied CNN/ViT scores; this cannot by itself establish out-of-fold neural-network training. Check the provenance of each supplied score and the training-derived scaling before describing end-to-end cross-fitted predictions.
3. **Biological methods:** the generic Python single-cell module does not reproduce the supplement's Seurat/fastCNV/CytoTRACE/edgeR workflow. A generic marker score or optional CNV wrapper is not evidence that those named analyses were executed. Biological results are untouched; their exact production pipeline still needs to be supplied and checked.
4. **Segmentation and MRI geometry:** supplied masks must genuinely represent the manuscript's tumor core, be in the declared reference space and have their original generation/reader provenance. An external brain mask is consumed; this package does not independently prove the stated skull-stripping or segmentation evaluation occurred. The nnU-Net wrapper currently uses fold `all`; it does not implement the supplement's five-fold out-of-fold segmentation and locked ensemble evaluation. Dataset preparation also does not enforce consumption of the registered/normalized four-sequence outputs. Those steps need an explicit, verified integration before reporting that the full S1 procedure was reproduced.
5. **Study artifacts and runtime:** the repository contains no study checkpoint, MRI, training logs or complete governed-data run. Reported hardware, dependency versions, endpoint metrics, segmentation metrics, agreement estimates and attribution maps have not been independently reproduced here. Tests use generated arrays and toy networks only.
6. **Document synchronization:** separate tracked-change copies update the local statistical results and remove obsolete SHAP caption abbreviations. Original MRI/heatmap displays remain illustrative; no study maps were recomputed.
7. **Predictor scales reconciled:** current Appendix S4 specifies log-transformed tumor volume and MRI ordinal coding 0–3. Browser inputs retain original clinical units and apply the same coding internally. A log-volume coefficient is not the effect of one cubic centimetre.
