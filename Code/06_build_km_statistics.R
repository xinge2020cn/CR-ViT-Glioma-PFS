suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(survival)
  library(tibble)
})

get_arg <- function(prefix, default) {
  hit <- grep(paste0("^", prefix, "="), commandArgs(trailingOnly = TRUE), value = TRUE)
  if (length(hit)) sub(paste0("^", prefix, "="), "", hit[[1]]) else default
}

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_dir <- if (length(script_arg)) {
  dirname(normalizePath(sub("^--file=", "", script_arg[[1]]), winslash = "/", mustWork = FALSE))
} else {
  getwd()
}
default_root <- normalizePath(file.path(script_dir, ".."), winslash = "/", mustWork = FALSE)
root <- normalizePath(get_arg("--root", default_root), winslash = "/", mustWork = FALSE)
dir.create(file.path(root, "Results"), recursive = TRUE, showWarnings = FALSE)
pred <- read_csv(
  file.path(root, "Results", "clinicoradiologic_crvit_patient_predictions.csv"),
  show_col_types = FALSE
)
multiomics_ids <- read_csv(
  file.path(root, "Data", "center1_gbm_multiomics_subset_patient_level.csv"),
  show_col_types = FALSE
)$patient_id

specs <- list()
for (cohort_name in c("Training cohort", "Temporal validation cohort", "Spatial validation cohort")) {
  for (subtype_name in c("IDHmut-intact", "IDHmut-codel", "GBM")) {
    specs[[length(specs) + 1L]] <- list(
      cohort = cohort_name,
      subtype = subtype_name,
      ids = NULL
    )
  }
}
specs[[length(specs) + 1L]] <- list(
  cohort = "GBM multi-omics subset",
  subtype = "GBM",
  ids = multiomics_ids
)

rows <- list()
for (spec in specs) {
  d <- if (is.null(spec$ids)) {
    pred |> filter(cohort == spec$cohort, subtype == spec$subtype)
  } else {
    pred |> filter(patient_id %in% spec$ids, subtype == spec$subtype)
  }
  d <- d |>
    mutate(
      vit_risk_group = factor(vit_risk_group, levels = c("Low risk", "High risk"))
    )
  fit <- coxph(
    Surv(pfs_time_months, event) ~ vit_risk_group,
    data = d,
    ties = "efron",
    x = TRUE
  )
  fit_summary <- summary(fit)
  zph <- cox.zph(fit, transform = "km")
  logrank <- survdiff(Surv(pfs_time_months, event) ~ vit_risk_group, data = d)
  rows[[length(rows) + 1L]] <- tibble(
    cohort = spec$cohort,
    subtype = spec$subtype,
    n = nrow(d),
    events = sum(d$event),
    analysis_horizon_months = NA_real_,
    low_risk_n = sum(d$vit_risk_group == "Low risk"),
    high_risk_n = sum(d$vit_risk_group == "High risk"),
    high_vs_low_hr = fit_summary$conf.int[1, "exp(coef)"],
    ci_low = fit_summary$conf.int[1, "lower .95"],
    ci_high = fit_summary$conf.int[1, "upper .95"],
    cox_p_value = fit_summary$coefficients[1, "Pr(>|z|)"],
    logrank_p_value = pchisq(logrank$chisq, df = 1, lower.tail = FALSE),
    ph_p_risk_group = zph$table["vit_risk_group", "p"]
  )
}

out <- bind_rows(rows)
write_excel_csv(out, file.path(root, "Results", "vit_km_statistics.csv"), na = "")
if (any(out$ph_p_risk_group <= 0.05)) {
  warning("At least one full-follow-up ViT risk-group Cox model failed the PH check; see vit_km_statistics.csv.")
}
print(out, n = Inf)
