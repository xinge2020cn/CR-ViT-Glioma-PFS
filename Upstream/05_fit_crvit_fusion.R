#!/usr/bin/env Rscript
# Compatibility guard for the retired upstream fusion command.
stop(paste(
  "This entry point is retired. Run Code/01_run_primary_analysis.R with the",
  "validated patient-level input tables. The current CR-ViT model jointly",
  "refits the LASSO-selected individual clinical/MRI predictors and the",
  "training-standardized ViT score in an unpenalized Cox model."
))
