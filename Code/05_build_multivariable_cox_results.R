suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(survival)
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

root <- normalizePath(
  get_arg("--root", default_root),
  winslash = "/", mustWork = FALSE
)
dat <- read_csv(file.path(root, "Data", "imaging_3center_patient_level.csv"), show_col_types = FALSE)
pred <- read_csv(file.path(root, "Results", "clinicoradiologic_crvit_patient_predictions.csv"), show_col_types = FALSE) %>%
  select(patient_id, vit_score, clinic_score, vit_cutoff, crvit_cutoff)
selected <- read_csv(file.path(root, "Results", "clinicoradiologic_selected_predictors.csv"), show_col_types = FALSE)
multiomics_ids <- read_csv(file.path(root, "Data", "center1_gbm_multiomics_subset_patient_level.csv"), show_col_types = FALSE)$patient_id

dat <- dat %>%
  left_join(pred, by = "patient_id") %>%
  mutate(
    cohort = factor(cohort),
    sex = factor(sex, levels = c("Female", "Male")),
    who_grade = factor(who_grade, levels = c("Grade 2", "Grade 3", "Grade 4"), ordered = TRUE),
    mgmt_promoter_methylation = factor(mgmt_promoter_methylation, levels = c("Methylated", "Unmethylated")),
    extent_of_resection = factor(extent_of_resection, levels = c("Gross total resection", "Subtotal resection")),
    tumor_location = factor(tumor_location),
    ependymal_involvement = factor(ependymal_involvement, levels = c("Absent", "Present")),
    deep_white_matter_invasion = factor(deep_white_matter_invasion, levels = c("Absent", "Present")),
    enhancing_tumor_crossing_midline = factor(enhancing_tumor_crossing_midline, levels = c("Absent", "Present")),
    enhancing_ord = match(enhancing_proportion, c("None/minimal (<=5%)", "Mild (>5-33%)", "Moderate (>33-67%)", "Extensive (>67%)")) - 1,
    necrotic_ord = match(necrotic_proportion, c("None/minimal (<=5%)", "Mild (>5-33%)", "Moderate (>33-67%)", "Extensive (>67%)")) - 1,
    flair_ord = match(peritumoral_flair_extent, c("None/minimal (<=5%)", "Mild (>5-33%)", "Moderate (>33-67%)", "Extensive (>67%)")) - 1,
    log_tumor_volume = log(tumor_volume_cm3)
  )

if (anyDuplicated(dat$patient_id) || any(!complete.cases(dat))) {
  stop("The adjusted Cox analysis requires complete and unique imaging patients.")
}

analysis_specs <- list(
  list(population = "Pooled validation cohorts", subtype = "IDHmut-intact", ids = NULL, strata_cohort = TRUE),
  list(population = "Pooled validation cohorts", subtype = "IDHmut-codel", ids = NULL, strata_cohort = TRUE),
  list(population = "Pooled validation cohorts", subtype = "GBM", ids = NULL, strata_cohort = TRUE),
  list(population = "GBM multi-omics subset", subtype = "GBM", ids = multiomics_ids, strata_cohort = FALSE)
)

table_rows <- list()
coefficient_rows <- list()
ph_rows <- list()
for (spec in analysis_specs) {
  d <- if (is.null(spec$ids)) {
    dat %>% filter(subtype == spec$subtype, cohort %in% c("Temporal validation cohort", "Spatial validation cohort"))
  } else {
    dat %>% filter(patient_id %in% spec$ids, subtype == spec$subtype)
  }
  d <- droplevels(d)
  adjustment_terms <- selected %>%
    filter(subtype == spec$subtype) %>%
    pull(model_variable) %>%
    unique()
  rhs <- c("vit_score", adjustment_terms)
  if (spec$strata_cohort) rhs <- c(rhs, "strata(cohort)")
  model_formula <- as.formula(paste("Surv(pfs_time_months, event) ~", paste(rhs, collapse = " + ")))
  fit <- coxph(model_formula, data = d, ties = "efron", x = TRUE, model = TRUE)
  sm <- summary(fit)
  zph <- cox.zph(fit, transform = "km")
  vit_i <- which(rownames(sm$coefficients) == "vit_score")

  table_rows[[length(table_rows) + 1L]] <- tibble(
    population = spec$population,
    subtype = spec$subtype,
    n = nrow(d),
    events = sum(d$event),
    adjusted_hr_per_training_sd_vit = sm$conf.int[vit_i, "exp(coef)"],
    ci_low = sm$conf.int[vit_i, "lower .95"],
    ci_high = sm$conf.int[vit_i, "upper .95"],
    p_value = sm$coefficients[vit_i, "Pr(>|z|)"],
    ph_p_vit = zph$table["vit_score", "p"],
    ph_p_global = zph$table["GLOBAL", "p"],
    adjustment_variables = paste(adjustment_terms, collapse = "; "),
    cohort_stratified = spec$strata_cohort
  )

  coefficient_rows[[length(coefficient_rows) + 1L]] <- tibble(
    population = spec$population,
    subtype = spec$subtype,
    term = rownames(sm$coefficients),
    hazard_ratio = unname(sm$conf.int[, "exp(coef)"]),
    ci_low = unname(sm$conf.int[, "lower .95"]),
    ci_high = unname(sm$conf.int[, "upper .95"]),
    p_value = unname(sm$coefficients[, "Pr(>|z|)"])
  )

  ph_rows[[length(ph_rows) + 1L]] <- as.data.frame(zph$table) %>%
    tibble::rownames_to_column("term") %>%
    transmute(
      population = spec$population, subtype = spec$subtype,
      term, chisq, df, p_value = p
    )
}

table3 <- bind_rows(table_rows)
coefficients <- bind_rows(coefficient_rows)
ph <- bind_rows(ph_rows)
cutoffs <- dat %>%
  filter(cohort == "Training cohort") %>%
  group_by(subtype) %>%
  summarise(
    training_n = n(),
    vit_cutoff = first(vit_cutoff),
    crvit_cutoff = first(crvit_cutoff),
    .groups = "drop"
  )

write_excel_csv(table3, file.path(root, "Results", "Table3_validation_multivariable_cox.csv"))
write_excel_csv(coefficients, file.path(root, "Results", "Table3_validation_multivariable_cox_full_coefficients.csv"))
write_excel_csv(ph, file.path(root, "Results", "Table3_validation_proportional_hazards.csv"))
write_excel_csv(cutoffs, file.path(root, "Results", "training_derived_score_cutoffs.csv"))

if (any(table3$ph_p_vit <= 0.05) || any(table3$ph_p_global <= 0.05)) {
  warning("At least one adjusted Cox model has evidence against proportional hazards; inspect the written diagnostics and consider an appropriate alternative. Do not alter observations to obtain a nonsignificant test.")
}
message("Rebuilt Table 3 using validation-only Cox models adjusted for LASSO-selected conventional variables; the nested multi-omics result is integrated as the final row.")
