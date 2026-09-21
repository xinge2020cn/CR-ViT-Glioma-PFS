# Raw-input workflow

These modules implement methods from MRI preprocessing through post-model transcriptomic interpretation. Review the configuration and input contracts before running the workflow.

All operations consume external inputs and write to an investigator-controlled output directory. No patient records or trained models are supplied.

## Configuration and software

Effective settings are in `config/article_defaults.yml`. Install the Python and R dependencies listed alongside this guide. Segmentation requires nnU-Net 2.4.2; single-cell analysis uses Seurat 5, fastCNV, original CytoTRACE, edgeR and CellChat. The original CytoTRACE and CytoTRACE2 are not interchangeable.

Commands below run from the repository root. Replace the example paths with governed input and output locations.

## MRI and segmentation

The manifest contains one row per patient. Preserve identifiers as strings. The four MRI sequences must be mapped correctly; brain masks, manual reference tumor-core masks and finalized tumor-core masks have separate roles. Both mask types must be in CE-T1 reference space. The manifest template contains field names only.

First produce registered, normalized, uncropped volumes and independent manual evaluation references:

```powershell
python Upstream/01_preprocess_mri.py --config Upstream/config/article_defaults.yml --manifest private/manifest.csv --stage segmentation --output-dir work/registered
python Upstream/02_prepare_nnunet_dataset.py --config Upstream/config/article_defaults.yml --manifest work/registered/processed_manifest.csv --output-dir work/nnUNet_raw/Dataset501_GliomaTumorCore
python Upstream/03_train_nnunet.py --config Upstream/config/article_defaults.yml --dataset-root work/nnUNet_raw/Dataset501_GliomaTumorCore
```

The final command prints the plan. Add `--run` to execute planning, the five patient-level folds and frozen validation-ensemble inference. Only training patients enter planning or model fitting. The training cohort is evaluated using its held-out fold predictions; validation uses the five final checkpoints. An evaluation manifest is produced only after all predictions exist.

```powershell
python Upstream/11_evaluate_segmentation.py --evaluation-manifest work/nnUNet_results/Dataset501_GliomaTumorCore/evaluation_manifest.json --output work/segmentation_metrics.csv
```

Dice and HD95 are calculated before manual correction. HD95 is the maximum of the two directed 95th-percentile surface distances, measured in millimetres. Empty predictions are retained as failures and stop finite-HD95 summary reporting.

Inspect and, when needed, correct automatic masks outside this workflow. Supply the finalized masks in the original CE-T1 reference space for survival preprocessing; do not replace independent evaluation references with corrected predictions.

## Survival models and spatial interpretation

```powershell
python Upstream/09_validate_upstream_configuration.py --config Upstream/config/article_defaults.yml
python Upstream/01_preprocess_mri.py --config Upstream/config/article_defaults.yml --manifest private/finalized_manifest.csv --stage survival --output-dir work/preprocessed
python Upstream/04_train_survival_models.py --config Upstream/config/article_defaults.yml --manifest work/preprocessed/processed_manifest.csv --output-dir work/survival
python Upstream/06_generate_attributions.py --config Upstream/config/article_defaults.yml --manifest work/preprocessed/processed_manifest.csv --checkpoint-dir work/survival/checkpoints --output-dir work/attributions
```

The processed tensor is CZYX; configured sizes are XYZ. Neural models share patient partitions, use complete fitting-cohort Cox risk sets and do not tune on external validation patients. The active downstream CR-ViT model combines selected individual conventional predictors and the training-standardized ViT score; the retired fusion script must not be used.

Occlusion requires matching locked checkpoints. It replaces all four channels jointly with subtype-training means, averages signed score changes over overlapping windows and then takes the absolute value for display. The unexpanded tumor core determines the representative slice. Saved arrays can be rendered independently with the attribution-panel renderer.

## Frozen biological risk definition

Run the downstream workflow first to obtain its omics risk manifest. It supplies `patient_id`, `omics_type`, `vit_km_score`, `vit_cutoff` and `vit_risk_group`. The cutoff comes from the imaging training cohort and is never selected using transcriptomic data.

Single-cell and bulk membership must be nonoverlapping and agree exactly with this manifest. Metadata carrying conflicting old groups is rejected. Do not adjust groups to reproduce manuscript counts.

## Single-cell analysis

Input is a Seurat object containing raw RNA counts and per-cell `patient_id` metadata. The workflow is staged because cell-type and joint CNV-expression annotation require investigator review.

```powershell
Rscript Upstream/08_run_single_cell_analysis.R --stage=cluster --config=Upstream/config/article_defaults.yml --input=private/raw_cells.rds --risk-manifest=private/frozen_risk.csv --output-dir=work/clustering
```

Review the cluster markers and assign one `cell_type` per `cell_id`. Select documented nonmalignant CNV reference labels; they are not inferred from the most abundant cluster.

```powershell
Rscript Upstream/08_run_single_cell_analysis.R --stage=cnv --config=Upstream/config/article_defaults.yml --input=work/clustering/clustered.rds --risk-manifest=private/frozen_risk.csv --annotations=private/cell_types.csv --reference-types=T_cells,Myeloid --output-dir=work/cnv
```

The reference labels in this example must be replaced by reviewed labels present in the data. Review the neural-lineage expression and CNV profiles and supply `cell_id,neural_state` assignments for every neural-lineage cell. Allowed labels are the configured states or `Reference`. State identities must not be assigned from imaging-risk labels.

```powershell
Rscript Upstream/08_run_single_cell_analysis.R --stage=interpret --config=Upstream/config/article_defaults.yml --input=work/cnv/cnv_annotated.rds --risk-manifest=private/frozen_risk.csv --state-annotations=private/neural_states.csv --hallmark-gmt=private/hallmark.gmt --kegg-gmt=private/kegg.gmt --output-dir=work/states
Rscript Upstream/12_run_cell_communication.R --config=Upstream/config/article_defaults.yml --input=work/states/characterized.rds --risk-manifest=private/frozen_risk.csv --output-dir=work/communication
```

State abundance uses tumors as edgeR replicates, with the total cells assigned to identified neural states as the denominator; the reference population is excluded. State-preferential genes are tested with paired-tumor pseudobulk comparisons against other identified states. Enrichment uses the tested-gene universe, separate Hallmark/KEGG families and BH correction. The resulting gene lists are frozen before bulk evaluation.

Communication is estimated separately for each tumor. Missing inferred interactions contribute zero to patient-level comparisons. Outgoing and incoming strengths are summarized per tumor; risk comparisons use two-sided rank-sum tests with BH correction. These are predictions of communication, not experimentally verified signaling.

## Bulk evaluation

Expression rows are unique gene symbols and columns are unique sample IDs. Metadata supplies `sample_id` and `patient_id`, with one sample per tumor.

```powershell
python Upstream/07_run_bulk_ssgsea.py --config Upstream/config/article_defaults.yml --expression private/bulk_expression.csv --metadata private/bulk_metadata.csv --signatures work/states/state_signatures.gmt --risk-manifest private/frozen_risk.csv --output-dir work/bulk
```

Scores use rank-based ssGSEA. Group comparisons use the final ViT labels, not legacy groups or CR-ViT labels. All requested signatures must pass the configured gene-overlap filter.
