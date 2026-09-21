# Describe supplied cohorts without generating or changing patient records.
suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(tidyr)
  library(survival)
})
args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(prefix, default) {
  hit <- grep(paste0("^", prefix, "="), args, value = TRUE)
  if (length(hit)) sub(paste0("^", prefix, "="), "", hit[[1]]) else default
}
root <- get_arg("--root", ".")
results_dir <- file.path(root, "Results")
qa_dir <- file.path(root, "QA")
dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(qa_dir, recursive = TRUE, showWarnings = FALSE)
imaging <- read_csv(file.path(root, "Data", "imaging_3center_patient_level.csv"), show_col_types = FALSE)
multi <- read_csv(file.path(root, "Data", "center1_gbm_multiomics_subset_patient_level.csv"), show_col_types = FALSE)
if (anyDuplicated(imaging$patient_id) || anyDuplicated(multi$patient_id) ||
    !all(multi$patient_id %in% imaging$patient_id)) stop("Invalid patient linkage.")
set.seed(as.integer(get_arg("--seed", "20260909")))

fmt_mean_sd <- function(x) sprintf("%.1f \u00B1 %.1f", mean(x), sd(x))
fmt_median_iqr <- function(x) {
  q <- quantile(x, c(.25, .50, .75), names = FALSE)
  sprintf("%.1f (%.1f\u2013%.1f)", q[2], q[1], q[3])
}
fmt_n <- function(x) sprintf("%d/%d (%.1f)", sum(x), length(x), 100 * mean(x))
p_fmt <- function(p) ifelse(p < .001, "<0.001", sprintf("%.3f", p))

cohort_levels <- c("Training cohort", "Temporal validation cohort", "Spatial validation cohort")
imaging$cohort <- factor(imaging$cohort, levels = cohort_levels)

safe_cat_p <- function(v) {
  tab <- table(imaging[[v]], imaging$cohort)
  ex <- suppressWarnings(chisq.test(tab)$expected)
  if (any(ex < 5)) fisher.test(tab, simulate.p.value = TRUE, B = 20000)$p.value else chisq.test(tab, correct = FALSE)$p.value
}

reverse_km <- function(df) {
  fit <- survfit(Surv(pfs_time_months, 1L - event) ~ 1, data = df)
  s <- summary(fit)$table
  sprintf("%.1f (%.1f\u2013%.1f)", s[["median"]], s[["0.95LCL"]], s[["0.95UCL"]])
}

clinical_row_spec <- tribble(
  ~characteristic, ~type, ~variable, ~level,
  "Age (years), mean \u00B1 SD", "mean", "age", NA,
  "Preoperative KPS, mean \u00B1 SD", "mean", "preoperative_kps", NA,
  "Sex, n/N (%)", "header", "sex", NA,
  "  Male", "cat", "sex", "Male",
  "  Female", "cat", "sex", "Female",
  "Molecular subtype, n/N (%)", "header", "subtype", NA,
  "  IDHmut-intact", "cat", "subtype", "IDHmut-intact",
  "  IDHmut-codel", "cat", "subtype", "IDHmut-codel",
  "  GBM", "cat", "subtype", "GBM",
  "WHO grade, n/N (%)", "header", "who_grade", NA,
  "  Grade 2", "cat", "who_grade", "Grade 2",
  "  Grade 3", "cat", "who_grade", "Grade 3",
  "  Grade 4", "cat", "who_grade", "Grade 4",
  "MGMT promoter methylation, n/N (%)", "header", "mgmt_promoter_methylation", NA,
  "  Unmethylated", "cat", "mgmt_promoter_methylation", "Unmethylated",
  "  Methylated", "cat", "mgmt_promoter_methylation", "Methylated",
  "Extent of resection, n/N (%)", "header", "extent_of_resection", NA,
  "  Gross total resection", "cat", "extent_of_resection", "Gross total resection",
  "  Subtotal resection", "cat", "extent_of_resection", "Subtotal resection",
  "Median follow-up (months), reverse KM (95% CI)", "reverse_km", NA, NA,
  "PFS event, n/N (%)", "cat", "event", "1"
)

imaging_row_spec <- tribble(
  ~characteristic, ~type, ~variable, ~level,
  "Tumor volume (cm\u00B3), median (IQR)", "median", "tumor_volume_cm3", NA,
  "Tumor location, n/N (%)", "header", "tumor_location", NA,
  "  Frontal", "cat", "tumor_location", "Frontal",
  "  Temporal", "cat", "tumor_location", "Temporal",
  "  Parietal", "cat", "tumor_location", "Parietal",
  "  Occipital", "cat", "tumor_location", "Occipital",
  "  Insular/deep", "cat", "tumor_location", "Insular/deep",
  "  Multilobar/other", "cat", "tumor_location", "Multilobar/other",
  "Enhancing tumor proportion, n/N (%)", "header", "enhancing_proportion", NA,
  "  None/minimal (\u22645%)", "cat", "enhancing_proportion", "None/minimal (<=5%)",
  "  Mild (>5\u201333%)", "cat", "enhancing_proportion", "Mild (>5-33%)",
  "  Moderate (>33\u201367%)", "cat", "enhancing_proportion", "Moderate (>33-67%)",
  "  Extensive (>67%)", "cat", "enhancing_proportion", "Extensive (>67%)",
  "Necrotic proportion, n/N (%)", "header", "necrotic_proportion", NA,
  "  None/minimal (\u22645%)", "cat", "necrotic_proportion", "None/minimal (<=5%)",
  "  Mild (>5\u201333%)", "cat", "necrotic_proportion", "Mild (>5-33%)",
  "  Moderate (>33\u201367%)", "cat", "necrotic_proportion", "Moderate (>33-67%)",
  "  Extensive (>67%)", "cat", "necrotic_proportion", "Extensive (>67%)",
  "Peritumoral T2/FLAIR extent, n/N (%)", "header", "peritumoral_flair_extent", NA,
  "  None/minimal (\u22645%)", "cat", "peritumoral_flair_extent", "None/minimal (<=5%)",
  "  Mild (>5\u201333%)", "cat", "peritumoral_flair_extent", "Mild (>5-33%)",
  "  Moderate (>33\u201367%)", "cat", "peritumoral_flair_extent", "Moderate (>33-67%)",
  "  Extensive (>67%)", "cat", "peritumoral_flair_extent", "Extensive (>67%)",
  "Ependymal involvement, n/N (%)", "header", "ependymal_involvement", NA,
  "  Absent", "cat", "ependymal_involvement", "Absent",
  "  Present", "cat", "ependymal_involvement", "Present",
  "Deep white matter invasion, n/N (%)", "header", "deep_white_matter_invasion", NA,
  "  Absent", "cat", "deep_white_matter_invasion", "Absent",
  "  Present", "cat", "deep_white_matter_invasion", "Present",
  "Enhancing tumor crossing midline, n/N (%)", "header", "enhancing_tumor_crossing_midline", NA,
  "  Absent", "cat", "enhancing_tumor_crossing_midline", "Absent",
  "  Present", "cat", "enhancing_tumor_crossing_midline", "Present"
)

summarize_cell <- function(df, type, variable, level) {
  if (type == "header") return("")
  if (type == "mean") return(fmt_mean_sd(df[[variable]]))
  if (type == "median") return(fmt_median_iqr(df[[variable]]))
  if (type == "reverse_km") return(reverse_km(df))
  if (type == "cat") return(fmt_n(as.character(df[[variable]]) == level))
  ""
}

build_summary_table <- function(spec) {
  out <- spec %>% mutate(row = row_number() + 1L)
  for (coh in cohort_levels) {
    df <- imaging %>% filter(cohort == coh)
    out[[coh]] <- mapply(summarize_cell, MoreArgs = list(df = df), out$type, out$variable, out$level)
  }
  out[["GBM multi-omics subset"]] <- mapply(
    summarize_cell, MoreArgs = list(df = multi), out$type, out$variable, out$level
  )
  out$p_value <- vapply(seq_len(nrow(out)), function(i) {
    type <- out$type[[i]]
    variable <- out$variable[[i]]
    if (type == "mean") p_fmt(summary(aov(imaging[[variable]] ~ imaging$cohort))[[1]][["Pr(>F)"]][1])
    else if (type == "median") p_fmt(kruskal.test(imaging[[variable]] ~ imaging$cohort)$p.value)
    else if (type == "header") p_fmt(safe_cat_p(variable))
    else if (type == "reverse_km") ""
    else ""
  }, character(1))
  result <- out %>% select(row, Characteristic = characteristic,
    all_of(cohort_levels), `P value` = p_value, `GBM multi-omics subset`)
  sizes <- vapply(cohort_levels, function(coh) sum(imaging$cohort == coh), integer(1))
  for (i in seq_along(cohort_levels)) {
    names(result)[names(result) == cohort_levels[i]] <- sprintf("%s (n=%d)", cohort_levels[i], sizes[i])
  }
  names(result)[names(result) == "GBM multi-omics subset"] <-
    sprintf("GBM multi-omics subset (n=%d)", nrow(multi))
  result
}

table1 <- build_summary_table(clinical_row_spec)
table_s3 <- build_summary_table(imaging_row_spec)
write_excel_csv(table1, file.path(results_dir, "Table1_summary_clinicoradiologic.csv"), na = "")
write_excel_csv(table_s3, file.path(results_dir, "TableS3_imaging_characteristics.csv"), na = "")

cohort_audit <- imaging %>%
  group_by(cohort, institution, subtype) %>%
  summarise(n = n(), events = sum(event), event_rate = mean(event), .groups = "drop")
write_excel_csv(cohort_audit, file.path(qa_dir, "clinicoradiologic_cohort_event_audit.csv"), na = "")

message("Completed descriptive summaries without modifying inputs.")
