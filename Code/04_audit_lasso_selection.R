suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(survival)
  library(glmnet)
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
seed <- as.integer(get_arg("--seed", "2026"))
dir.create(file.path(root, "QA"), recursive = TRUE, showWarnings = FALSE)

dat <- read_csv(file.path(root, "Data", "imaging_3center_patient_level.csv"), show_col_types = FALSE) %>%
  mutate(
    subtype = factor(subtype, levels = c("IDHmut-intact", "IDHmut-codel", "GBM")),
    sex = factor(sex, levels = c("Female", "Male")),
    who_grade = factor(who_grade, levels = c("Grade 2", "Grade 3", "Grade 4"), ordered = TRUE),
    mgmt_promoter_methylation = factor(mgmt_promoter_methylation, levels = c("Methylated", "Unmethylated")),
    extent_of_resection = factor(extent_of_resection, levels = c("Gross total resection", "Subtotal resection")),
    tumor_location = factor(tumor_location, levels = c("Frontal", "Temporal", "Parietal", "Occipital", "Insular/deep", "Multilobar/other")),
    ependymal_involvement = factor(ependymal_involvement, levels = c("Absent", "Present")),
    deep_white_matter_invasion = factor(deep_white_matter_invasion, levels = c("Absent", "Present")),
    enhancing_tumor_crossing_midline = factor(enhancing_tumor_crossing_midline, levels = c("Absent", "Present")),
    enhancing_ord = match(enhancing_proportion, c("None/minimal (<=5%)", "Mild (>5-33%)", "Moderate (>33-67%)", "Extensive (>67%)")) - 1,
    necrotic_ord = match(necrotic_proportion, c("None/minimal (<=5%)", "Mild (>5-33%)", "Moderate (>33-67%)", "Extensive (>67%)")) - 1,
    flair_ord = match(peritumoral_flair_extent, c("None/minimal (<=5%)", "Mild (>5-33%)", "Moderate (>33-67%)", "Extensive (>67%)")) - 1,
    log_tumor_volume = log(tumor_volume_cm3)
  )

candidate_formula <- ~ age + sex + preoperative_kps + who_grade +
  mgmt_promoter_methylation + extent_of_resection + tumor_location +
  log_tumor_volume + enhancing_ord + necrotic_ord + flair_ord +
  ependymal_involvement + deep_white_matter_invasion + enhancing_tumor_crossing_midline
candidate_terms <- attr(terms(candidate_formula), "term.labels")

make_x <- function(df) {
  x <- model.matrix(candidate_formula, data = df)[, -1, drop = FALSE]
  sds <- apply(x, 2, sd)
  x[, is.finite(sds) & sds > 0, drop = FALSE]
}

group_column <- function(column_name) {
  hits <- candidate_terms[startsWith(column_name, candidate_terms)]
  if (!length(hits)) stop("Unable to map model-matrix column: ", column_name)
  hits[[which.max(nchar(hits))]]
}

stratified_folds <- function(df, k, seed_offset) {
  set.seed(seed + seed_offset)
  strata <- interaction(df$event, dplyr::ntile(df$pfs_time_months, 4), drop = TRUE)
  fold <- integer(nrow(df))
  for (st in levels(strata)) {
    idx <- which(strata == st)
    fold[idx] <- sample(rep(seq_len(k), length.out = length(idx)))
  }
  fold
}

rows <- list()
coef_rows <- list()
for (subtype_name in levels(dat$subtype)) {
  train <- dat %>% filter(subtype == subtype_name, cohort == "Training cohort") %>% arrange(patient_id)
  x <- make_x(train)
  y <- Surv(train$pfs_time_months, train$event)
  foldid <- stratified_folds(train, 10L, match(subtype_name, levels(dat$subtype)) * 10000L)
  for (measure in c("deviance", "C")) {
    cv <- cv.glmnet(
      x, y, family = "cox", alpha = 1, foldid = foldid,
      nfolds = 10, standardize = TRUE, type.measure = measure
    )
    for (rule in c("lambda.1se", "lambda.min")) {
      b <- as.matrix(coef(cv, s = rule))
      cols <- rownames(b)[abs(as.numeric(b)) > 1e-10]
      terms_selected <- unique(vapply(cols, group_column, character(1)))
      rows[[length(rows) + 1L]] <- tibble(
        subtype = subtype_name,
        type_measure = measure,
        lambda_rule = rule,
        lambda = if (rule == "lambda.1se") cv$lambda.1se else cv$lambda.min,
        n_predictor_groups = length(terms_selected),
        selected_predictors = paste(terms_selected, collapse = "; ")
      )
      if (length(cols)) {
        coef_rows[[length(coef_rows) + 1L]] <- tibble(
          subtype = subtype_name,
          type_measure = measure,
          lambda_rule = rule,
          model_matrix_column = cols,
          predictor_group = vapply(cols, group_column, character(1)),
          coefficient = as.numeric(b[cols, , drop = FALSE])
        )
      }
    }
  }
}

write_excel_csv(bind_rows(rows), file.path(root, "QA", "lasso_lambda_rule_audit.csv"))
write_excel_csv(bind_rows(coef_rows), file.path(root, "QA", "lasso_lambda_rule_coefficients.csv"))
print(bind_rows(rows), n = Inf)
