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

revision_root <- normalizePath(
  get_arg("--root", default_root),
  winslash = "/", mustWork = FALSE
)

data <- read_csv(file.path(revision_root, "Data", "imaging_3center_patient_level.csv"), show_col_types = FALSE)

score_summary <- data %>%
  group_by(cohort, subtype) %>%
  summarise(
    n = n(),
    events = sum(event),
    vit_mean = mean(risk_score_3dvit),
    vit_sd = sd(risk_score_3dvit),
    vit_q01 = quantile(risk_score_3dvit, 0.01),
    vit_q50 = median(risk_score_3dvit),
    vit_q99 = quantile(risk_score_3dvit, 0.99),
    cnn_mean = mean(risk_score_3dcnn),
    cnn_sd = sd(risk_score_3dcnn),
    cnn_q01 = quantile(risk_score_3dcnn, 0.01),
    cnn_q50 = median(risk_score_3dcnn),
    cnn_q99 = quantile(risk_score_3dcnn, 0.99),
    vit_cnn_spearman = cor(risk_score_3dvit, risk_score_3dcnn, method = "spearman"),
    vit_cindex = concordance(
      Surv(pfs_time_months, event) ~ risk_score_3dvit,
      reverse = TRUE
    )$concordance,
    cnn_cindex = concordance(
      Surv(pfs_time_months, event) ~ risk_score_3dcnn,
      reverse = TRUE
    )$concordance,
    .groups = "drop"
  )

training_cutoffs <- data %>%
  filter(cohort == "Training cohort") %>%
  group_by(subtype) %>%
  summarise(
    vit_training_median_cutoff = median(risk_score_3dvit),
    cnn_training_median_cutoff = median(risk_score_3dcnn),
    .groups = "drop"
  )

score_summary <- score_summary %>% left_join(training_cutoffs, by = "subtype")
write_excel_csv(score_summary, file.path(revision_root, "QA", "model_score_distribution_audit.csv"), na = "")
print(score_summary, n = Inf, width = Inf)
