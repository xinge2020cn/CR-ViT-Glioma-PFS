args <- commandArgs(trailingOnly = TRUE)

get_arg <- function(prefix, default) {
  hit <- grep(paste0("^", prefix, "="), args, value = TRUE)
  if (length(hit)) sub(paste0("^", prefix, "="), "", hit[[1]]) else default
}

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_dir <- if (length(script_arg)) {
  dirname(normalizePath(sub("^--file=", "", script_arg[[1]]), winslash = "/", mustWork = FALSE))
} else {
  getwd()
}
root <- normalizePath(get_arg("--root", script_dir), winslash = "/", mustWork = FALSE)
code_dir <- file.path(script_dir, "Code")
rscript <- Sys.which("Rscript")
if (!nzchar(rscript)) rscript <- file.path(R.home("bin"), "Rscript.exe")
python <- get_arg("--python", Sys.which("python"))
if (!nzchar(python)) stop("Python was not found. Pass --python=<path>.")
if (!dir.exists(file.path(root, "Data"))) {
  stop("External inputs were not found. Pass --root=<analysis-workspace>.")
}

dir.create(file.path(root, "Results"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(root, "QA"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(root, "Figures"), recursive = TRUE, showWarnings = FALSE)

run_python <- function(script, extra = character()) {
  message("Running ", script)
  status <- system2(
    python,
    c(shQuote(file.path(code_dir, script)), shQuote(paste0("--root=", root)), extra)
  )
  if (!identical(status, 0L)) stop(script, " failed with status ", status)
}

run_r <- function(script, extra = character()) {
  message("Running ", script)
  status <- system2(
    rscript,
    c(shQuote(file.path(code_dir, script)), shQuote(paste0("--root=", root)), extra)
  )
  if (!identical(status, 0L)) stop(script, " failed with status ", status)
}

run_python("00_validate_inputs.py")
run_r("13_build_cohort_summaries.R")

run_r(
  "01_run_primary_analysis.R",
  c(
    paste0("--bootstrap=", get_arg("--bootstrap", "1000")),
    paste0("--workers=", get_arg("--workers", "3")),
    paste0("--seed=", get_arg("--seed", "20260901"))
  )
)
for (script in c(
  "02_build_pfs_summary.R",
  "03_audit_model_scores.R",
  "04_audit_lasso_selection.R",
  "05_build_multivariable_cox_results.R",
  "06_build_km_statistics.R"
)) {
  run_r(script, if (identical(script, "04_audit_lasso_selection.R")) paste0("--seed=", get_arg("--seed", "20260901")) else character())
}

run_python("07_build_segmentation_summary.py", c(
  "--bootstrap=2000", paste0("--seed=", get_arg("--segmentation-seed", "20260919"))
))
run_python("08_build_reader_agreement.py", c(
  "--bootstrap=2000", paste0("--seed=", get_arg("--reader-seed", "20260931"))
))
run_python("09_build_figure_s1.py")
run_python("10_build_figure_components.py")
run_python("12_build_data_dictionary.py")
run_python("15_build_omics_risk_manifest.py")
run_python("14_validate_analysis_outputs.py")

message("Analysis completed. Outputs and integrity checks are available in the analysis workspace.")
