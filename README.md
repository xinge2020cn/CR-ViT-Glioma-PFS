# CR-ViT-Glioma-PFS

R and Python implementations of the imaging, survival and post-model transcriptomic analysis workflow.

This is a code-only release. Patient-level records, imaging, expression matrices, trained weights and generated outputs are not distributed. Inputs and outputs belong in a separate governed analysis workspace.

## Analysis workflow

- Validate patient linkage, tumor-core composition and required variables.
- Summarize baseline characteristics and reverse-Kaplan-Meier follow-up.
- Select conventional predictors with training-only, ten-fold LASSO-Cox cross-validation using partial-likelihood deviance and `lambda.min`.
- Refit the selected individual terms by unpenalized Cox regression; enter the training-standardized ViT score jointly for CR-ViT.
- Evaluate apparent training and frozen validation predictions using concordance, time-dependent AUC, Brier scores, integrated metrics, calibration and decision curves.
- Calculate paired bootstrap comparisons, proportional-hazards checks and Kaplan-Meier analyses using frozen training-median thresholds.
- Summarize segmentation and reader agreement, export individual figure panels and validate computed outputs.
- Export a single final ViT risk definition for nonoverlapping single-cell and bulk cohorts.

## Running an analysis

Install the dependencies in `requirements.txt` and `r-packages.txt`. Use R 4.5 or later and Python 3.10 or later.

Supply the required input tables in the external workspace. The input validator defines their required fields. Run from the repository:

```powershell
Rscript run_all.R --root="path/to/analysis-workspace" --python="path/to/python"
```

The default run uses 1,000 patient-level bootstrap resamples. A reduced-bootstrap smoke test is available:

```powershell
Rscript run_all.R --root="path/to/analysis-workspace" --bootstrap=100 --workers=2
```

Reduced-bootstrap runs are implementation checks, not final inferential analyses. The workflow reads outcomes and neural scores supplied by the investigator; it does not generate them.

The source-release audit is separate from analysis-output validation:

```powershell
python Code/11_audit_public_release.py
```

## Raw-input methods

The upstream workflow covers four-sequence MRI registration and normalization, five-fold nnU-Net segmentation, subtype-specific CNN/ViT survival modeling, fixed-model occlusion sensitivity, Seurat-based single-cell analysis and bulk ssGSEA.

Use the upstream guide for command-line interfaces. Annotation requires investigator-reviewed cell identities, CNV references and joint CNV-expression state assignments. No automated step forces the number of clusters, state proportions or statistical significance to match a manuscript.

## License

This repository makes its source code publicly viewable. No open-source license is granted, and all rights are reserved.
