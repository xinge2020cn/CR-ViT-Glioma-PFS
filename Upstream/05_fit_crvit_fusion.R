#!/usr/bin/env Rscript

# Retired implementation retained only for source-history inspection.
# It jointly selected CNN, ViT and clinical predictors at lambda.1se and does
# not implement the current manuscript's two-stage CR-ViT specification.
stop(paste(
  "This legacy fusion entry point has been retired.",
  "Do not use it to generate current manuscript scores.",
  "See CODE_COVERAGE.md for the unresolved fusion audit; no existing scores",
  "are changed by the occlusion-only revision."
))

args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(name, default = NULL) {
  prefix <- paste0("--", name, "=")
  hit <- args[startsWith(args, prefix)]
  if (length(hit) == 0) return(default)
  sub(prefix, "", hit[[1]], fixed = TRUE)
}

manifest_path <- get_arg("manifest")
prediction_dir <- get_arg("prediction-dir")
output_dir <- get_arg("output-dir", "Working/crvit")
if (is.null(manifest_path) || is.null(prediction_dir)) {
  stop("Usage: Rscript 05_fit_crvit_fusion.R --manifest=FILE --prediction-dir=DIR --output-dir=DIR")
}

suppressPackageStartupMessages({
  library(survival)
  library(glmnet)
  library(readr)
  library(dplyr)
})

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
data <- read_csv(manifest_path, show_col_types = FALSE)
required <- c("patient_id", "subtype", "cohort", "pfs_time_months", "event")
missing <- setdiff(required, names(data))
if (length(missing) > 0) stop("Missing manifest columns: ", paste(missing, collapse = ", "))

read_prediction <- function(path, model_name) {
  if (!file.exists(path)) stop("Missing prediction file: ", path)
  prediction <- read_csv(path, show_col_types = FALSE)
  if (!all(c("patient_id", "risk_score") %in% names(prediction))) {
    stop("Prediction file must contain patient_id and risk_score: ", path)
  }
  prediction <- prediction[, c("patient_id", "risk_score")]
  names(prediction)[names(prediction) == "risk_score"] <- model_name
  prediction
}

vit <- read_prediction(file.path(prediction_dir, "3d_vit_predictions.csv"), "risk_score_3dvit")
cnn <- read_prediction(file.path(prediction_dir, "3d_resnet18_predictions.csv"), "risk_score_3dcnn")
data <- data %>%
  select(-any_of(c("risk_score_3dvit", "risk_score_3dcnn"))) %>%
  left_join(vit, by = "patient_id") %>%
  left_join(cnn, by = "patient_id")
if (anyNA(data$risk_score_3dvit) || anyNA(data$risk_score_3dcnn)) {
  stop("Prediction files do not cover every patient in the manifest.")
}

clinical_candidates <- c(
  "age", "sex", "preoperative_kps", "who_grade", "mgmt_promoter_methylation",
  "extent_of_resection", "postoperative_radiotherapy", "postoperative_chemotherapy",
  "tumor_location", "tumor_volume_cm3", "enhancing_proportion", "necrotic_proportion",
  "peritumoral_flair_extent", "ependymal_involvement", "deep_white_matter_invasion",
  "enhancing_tumor_crossing_midline"
)
clinical_candidates <- intersect(clinical_candidates, names(data))
model_variables <- c("risk_score_3dvit", "risk_score_3dcnn", clinical_candidates)
for (column in model_variables) {
  if (is.character(data[[column]])) data[[column]] <- factor(data[[column]])
}

all_coefficients <- list()
all_predictions <- list()
for (subtype_value in sort(unique(as.character(data$subtype)))) {
  subtype_data <- data %>% filter(as.character(.data$subtype) == subtype_value)
  training <- subtype_data %>% filter(grepl("train", tolower(as.character(cohort))))
  complete <- complete.cases(training[, c("pfs_time_months", "event", model_variables)])
  training <- training[complete, , drop = FALSE]
  if (nrow(training) < 10 || sum(training$event == 1) < 2) next
  x_formula <- as.formula(paste("~", paste(model_variables, collapse = " + "), "-1"))
  x_train <- model.matrix(x_formula, data = training)
  x_all <- model.matrix(x_formula, data = subtype_data)
  y_train <- Surv(training$pfs_time_months, training$event)
  folds <- max(3, min(10, nrow(training)))
  fit <- cv.glmnet(x_train, y_train, family = "cox", alpha = 1, nfolds = folds, type.measure = "C")
  coefficients <- as.matrix(coef(fit, s = "lambda.1se"))
  coefficient_table <- tibble(
    subtype = subtype_value,
    feature = rownames(coefficients),
    coefficient = as.numeric(coefficients[, 1]),
    selected = as.numeric(coefficients[, 1]) != 0
  )
  all_coefficients[[subtype_value]] <- coefficient_table
  score <- as.numeric(predict(fit, newx = x_all, s = "lambda.1se", type = "link"))
  all_predictions[[subtype_value]] <- tibble(
    patient_id = subtype_data$patient_id,
    subtype = subtype_value,
    cohort = subtype_data$cohort,
    risk_score_crvit = score
  )
  saveRDS(fit, file.path(output_dir, paste0("lasso_cox_", gsub("[^A-Za-z0-9_-]", "_", subtype_value), ".rds")))
}

if (length(all_predictions) == 0) stop("No subtype had enough training events for LASSO-Cox fitting.")
write_csv(bind_rows(all_coefficients), file.path(output_dir, "lasso_selected_coefficients.csv"))
write_csv(bind_rows(all_predictions), file.path(output_dir, "crvit_predictions.csv"))
