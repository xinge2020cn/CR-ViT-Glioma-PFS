suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(tidyr)
  library(purrr)
  library(survival)
  library(glmnet)
  library(timeROC)
  library(future)
  library(future.apply)
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
bootstrap_n <- as.integer(get_arg("--bootstrap", "1000"))
seed <- as.integer(get_arg("--seed", "2026"))
workers <- as.integer(get_arg("--workers", "3"))
times <- c(6, 12, 18, 24, 30, 36)
future::plan(future::multisession, workers = workers)
on.exit(future::plan(future::sequential), add = TRUE)
data_dir <- file.path(revision_root, "Data")
results_dir <- file.path(revision_root, "Results")
qa_dir <- file.path(revision_root, "QA")
dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(qa_dir, recursive = TRUE, showWarnings = FALSE)

data <- read_csv(file.path(data_dir, "imaging_3center_patient_level.csv"), show_col_types = FALSE) %>%
  mutate(
    cohort = factor(cohort, levels = c("Training cohort", "Temporal validation cohort", "Spatial validation cohort")),
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

cnn_training_stats <- data %>%
  filter(cohort == "Training cohort") %>%
  group_by(subtype) %>%
  summarise(cnn_mean = mean(risk_score_3dcnn), cnn_sd = sd(risk_score_3dcnn), .groups = "drop")
data <- data %>%
  left_join(cnn_training_stats, by = "subtype") %>%
  mutate(cnn_z = (risk_score_3dcnn - cnn_mean) / cnn_sd) %>%
  select(-cnn_mean, -cnn_sd)

if (any(!complete.cases(data))) stop("The clinicoradiologic analysis requires complete patient-level fields.")

candidate_formula <- ~ age + sex + preoperative_kps + who_grade +
  mgmt_promoter_methylation + extent_of_resection + tumor_location +
  log_tumor_volume + enhancing_ord + necrotic_ord + flair_ord +
  ependymal_involvement + deep_white_matter_invasion + enhancing_tumor_crossing_midline

feature_labels <- c(
  age = "Age",
  sex = "Sex",
  preoperative_kps = "Preoperative KPS",
  who_grade = "WHO grade",
  mgmt_promoter_methylation = "MGMT promoter methylation",
  extent_of_resection = "Extent of resection",
  tumor_location = "Tumor location",
  log_tumor_volume = "Tumor volume",
  enhancing_ord = "Enhancing tumor proportion",
  necrotic_ord = "Necrotic proportion",
  flair_ord = "Peritumoral T2/FLAIR abnormality extent",
  ependymal_involvement = "Ependymal involvement",
  deep_white_matter_invasion = "Deep white matter invasion",
  enhancing_tumor_crossing_midline = "Enhancing tumor crossing the midline"
)

make_x <- function(df, reference_columns = NULL) {
  x <- model.matrix(candidate_formula, data = df)[, -1, drop = FALSE]
  if (is.null(reference_columns)) {
    sds <- apply(x, 2, sd)
    x <- x[, is.finite(sds) & sds > 0, drop = FALSE]
    return(x)
  }
  missing <- setdiff(reference_columns, colnames(x))
  if (length(missing)) {
    x <- cbind(x, matrix(0, nrow(x), length(missing), dimnames = list(NULL, missing)))
  }
  x[, reference_columns, drop = FALSE]
}

stratified_folds <- function(df, k, seed_offset = 0L) {
  set.seed(seed + seed_offset)
  strata <- interaction(df$event, dplyr::ntile(df$pfs_time_months, 4), drop = TRUE)
  fold <- integer(nrow(df))
  for (st in levels(strata)) {
    idx <- which(strata == st)
    fold[idx] <- sample(rep(seq_len(k), length.out = length(idx)))
  }
  fold
}

candidate_terms <- attr(terms(candidate_formula), "term.labels")

group_model_matrix_column <- function(column_name) {
  hits <- candidate_terms[startsWith(column_name, candidate_terms)]
  if (!length(hits)) stop("Unable to map LASSO column to a candidate predictor: ", column_name)
  hits[[which.max(nchar(hits))]]
}

select_clinic_terms <- function(train_df, seed_offset, allow_empty = FALSE) {
  x <- make_x(train_df)
  y <- Surv(train_df$pfs_time_months, train_df$event)
  foldid <- stratified_folds(train_df, 10L, seed_offset = seed_offset)
  cvfit <- cv.glmnet(
    x, y, family = "cox", alpha = 1, foldid = foldid, nfolds = 10,
    standardize = TRUE, type.measure = "deviance"
  )
  # Training-only LASSO-Cox selection without a predictor-count constraint.
  # The penalty is selected entirely by 10-fold cross-validation at lambda.min,
  # using partial-likelihood deviance as the cross-validation measure.
  lambda_selected <- cvfit$lambda.min
  beta <- as.matrix(coef(cvfit, s = "lambda.min"))
  selected_columns <- rownames(beta)[as.numeric(beta) != 0]
  selected_column_terms <- vapply(selected_columns, group_model_matrix_column, character(1))
  selected_terms <- candidate_terms[candidate_terms %in% unique(selected_column_terms)]
  if (!length(selected_terms) && !allow_empty) {
    stop("LASSO-Cox retained no clinicoradiologic predictor for ", as.character(unique(train_df$subtype))[[1]], ".")
  }
  list(
    terms = selected_terms,
    columns = selected_columns,
    coefficients = as.numeric(beta[selected_columns, , drop = FALSE]),
    cvfit = cvfit,
    x_columns = colnames(x),
    selected_penalty = "lambda.min",
    lambda_selected = lambda_selected,
    lambda_min = cvfit$lambda.min,
    lambda_1se = cvfit$lambda.1se
  )
}

selected_survival_formula <- function(selected_terms) {
  rhs <- if (length(selected_terms)) paste(selected_terms, collapse = " + ") else "1"
  as.formula(paste("Surv(pfs_time_months, event) ~", rhs))
}

selected_crvit_formula <- function(selected_terms) {
  rhs <- paste(c(selected_terms, "vit_z"), collapse = " + ")
  as.formula(paste("Surv(pfs_time_months, event) ~", rhs))
}

fit_selected_clinic <- function(train_df, selection) {
  selection$refit <- safe_coxph(selected_survival_formula(selection$terms), train_df)
  selection
}

predict_selected_clinic <- function(fit, new_df) {
  as.numeric(predict(fit$refit, newdata = new_df, type = "lp", reference = "zero"))
}

baseline_survival_at <- function(fit, t) {
  bh <- basehaz(fit, centered = FALSE)
  idx <- max(which(bh$time <= t), 0)
  if (idx == 0) 1 else exp(-bh$hazard[[idx]])
}

predict_survival <- function(fit, newdata, times) {
  lp <- as.numeric(predict(fit, newdata = newdata, type = "lp", reference = "zero"))
  sapply(times, function(t) baseline_survival_at(fit, t)^exp(lp))
}

safe_coxph <- function(formula, data) {
  coxph(formula, data = data, ties = "breslow", x = TRUE, y = TRUE, model = TRUE)
}

model_labels <- c(
  clinic = "Clinicoradiologic",
  vit = "3D-ViT",
  crvit = "CR-ViT",
  cnn = "3D-CNN"
)

prediction_list <- list()
selected_list <- list()
lasso_coefficient_list <- list()
coefficient_list <- list()
ph_list <- list()
model_objects <- list()

for (subtype_name in levels(data$subtype)) {
  train <- data %>% filter(subtype == subtype_name, cohort == "Training cohort") %>% arrange(patient_id)
  validation <- data %>% filter(subtype == subtype_name, cohort != "Training cohort") %>% arrange(patient_id)
  outer_fold <- stratified_folds(train, 5L, seed_offset = match(subtype_name, levels(data$subtype)) * 100L)

  oof <- train %>% transmute(patient_id)
  for (m in c("cnn", "clinic", "vit", "crvit")) {
    oof[[paste0(m, "_score")]] <- NA_real_
    for (t in times) oof[[paste0(m, "_surv_", t)]] <- NA_real_
  }

  for (fold_i in seq_len(5L)) {
    inner_train <- train[outer_fold != fold_i, , drop = FALSE]
    heldout <- train[outer_fold == fold_i, , drop = FALSE]
    inner_selection <- select_clinic_terms(
      inner_train,
      seed_offset = match(subtype_name, levels(data$subtype)) * 1000L + fold_i,
      allow_empty = TRUE
    )
    clinic_base_inner <- fit_selected_clinic(inner_train, inner_selection)
    inner_model_data <- inner_train
    heldout_model_data <- heldout

    clinic_fit <- clinic_base_inner$refit
    cnn_fit <- safe_coxph(Surv(pfs_time_months, event) ~ cnn_z, inner_model_data)
    vit_fit <- safe_coxph(Surv(pfs_time_months, event) ~ vit_z, inner_model_data)
    crvit_fit <- safe_coxph(selected_crvit_formula(inner_selection$terms), inner_model_data)
    held_idx <- match(heldout$patient_id, oof$patient_id)

    for (spec in list(
      list(name = "clinic", fit = clinic_fit),
      list(name = "cnn", fit = cnn_fit),
      list(name = "vit", fit = vit_fit),
      list(name = "crvit", fit = crvit_fit)
    )) {
      oof[[paste0(spec$name, "_score")]][held_idx] <- as.numeric(
        predict(spec$fit, newdata = heldout_model_data, type = "lp", reference = "zero")
      )
      surv_mat <- predict_survival(spec$fit, heldout_model_data, times)
      for (j in seq_along(times)) {
        oof[[paste0(spec$name, "_surv_", times[[j]])]][held_idx] <- surv_mat[, j]
      }
    }
  }

  final_selection <- select_clinic_terms(
    train,
    seed_offset = match(subtype_name, levels(data$subtype)) * 10000L
  )
  clinic_base_final <- fit_selected_clinic(train, final_selection)
  all_subtype <- bind_rows(train, validation)
  final_model_data <- all_subtype
  final_train <- final_model_data %>% filter(cohort == "Training cohort")
  clinic_fit_final <- clinic_base_final$refit
  cnn_fit_final <- safe_coxph(Surv(pfs_time_months, event) ~ cnn_z, final_train)
  vit_fit_final <- safe_coxph(Surv(pfs_time_months, event) ~ vit_z, final_train)
  crvit_fit_final <- safe_coxph(selected_crvit_formula(final_selection$terms), final_train)

  for (spec in list(
    list(name = "clinic", fit = clinic_fit_final),
    list(name = "cnn", fit = cnn_fit_final),
    list(name = "vit", fit = vit_fit_final),
    list(name = "crvit", fit = crvit_fit_final)
  )) {
    final_model_data[[paste0(spec$name, "_score_final")]] <- as.numeric(
      predict(spec$fit, newdata = final_model_data, type = "lp", reference = "zero")
    )
    surv_mat <- predict_survival(spec$fit, final_model_data, times)
    for (j in seq_along(times)) final_model_data[[paste0(spec$name, "_surv_", times[[j]], "_final")]] <- surv_mat[, j]
  }

  final_train_scores <- final_model_data %>% filter(cohort == "Training cohort")
  for (m in c("cnn", "clinic", "vit", "crvit")) {
    oof_col <- paste0(m, "_score")
    final_col <- paste0(m, "_score_final")
    oof[[oof_col]] <- as.numeric(scale(oof[[oof_col]])) * sd(final_train_scores[[final_col]]) + mean(final_train_scores[[final_col]])
  }

  # Apparent development-cohort estimates describe the final model fitted to the
  # complete Training cohort. Outer-fold predictions remain available as an
  # internal fitting audit, whereas all validation estimates use frozen models.
  pred_train <- final_model_data %>%
    filter(cohort == "Training cohort") %>%
    select(
      patient_id, center, institution, subtype, cohort, pfs_time_months, event,
      risk_score_3dcnn, risk_score_3dvit, cnn_z, vit_z,
      cnn_score_final, clinic_score_final, vit_score_final, crvit_score_final,
      matches("^(cnn|clinic|vit|crvit)_surv_[0-9]+_final$")
    )
  names(pred_train) <- sub("_final$", "", names(pred_train))
  pred_train <- pred_train %>%
    mutate(
      cnn_km_score = cnn_score,
      clinic_km_score = clinic_score,
      vit_km_score = vit_score,
      crvit_km_score = crvit_score
    )

  pred_validation <- final_model_data %>%
    filter(cohort != "Training cohort") %>%
    select(
      patient_id, center, institution, subtype, cohort, pfs_time_months, event,
      risk_score_3dcnn, risk_score_3dvit, cnn_z, vit_z,
      cnn_score_final, clinic_score_final, vit_score_final, crvit_score_final,
      matches("^(cnn|clinic|vit|crvit)_surv_[0-9]+_final$")
    )
  names(pred_validation) <- sub("_final$", "", names(pred_validation))
  pred_validation <- pred_validation %>%
    mutate(
      cnn_km_score = cnn_score,
      clinic_km_score = clinic_score,
      vit_km_score = vit_score,
      crvit_km_score = crvit_score
    )

  pred <- bind_rows(pred_train, pred_validation)
  score_scaling <- pred %>%
    filter(cohort == "Training cohort") %>%
    summarise(across(
      c(cnn_score, clinic_score, vit_score, crvit_score),
      list(mean = mean, sd = sd)
    ))
  for (m in c("cnn", "clinic", "vit", "crvit")) {
    score_col <- paste0(m, "_score")
    pred[[score_col]] <- (pred[[score_col]] - score_scaling[[paste0(score_col, "_mean")]]) /
      score_scaling[[paste0(score_col, "_sd")]]
    pred[[paste0(m, "_km_score")]] <- pred[[score_col]]
  }
  km_cutoffs <- pred %>%
    filter(cohort == "Training cohort") %>%
    summarise(
      clinic_cutoff = median(clinic_km_score),
      vit_cutoff = median(vit_km_score),
      crvit_cutoff = median(crvit_km_score)
    )
  pred <- pred %>%
    mutate(
      clinic_cutoff = km_cutoffs$clinic_cutoff,
      vit_cutoff = km_cutoffs$vit_cutoff,
      crvit_cutoff = km_cutoffs$crvit_cutoff,
      clinic_risk_group = if_else(clinic_km_score >= clinic_cutoff, "High risk", "Low risk"),
      vit_risk_group = if_else(vit_km_score >= vit_cutoff, "High risk", "Low risk"),
      crvit_risk_group = if_else(crvit_km_score >= crvit_cutoff, "High risk", "Low risk")
    )

  prediction_list[[subtype_name]] <- pred
  model_objects[[subtype_name]] <- list(clinic_base = clinic_base_final, clinic = clinic_fit_final, cnn = cnn_fit_final, vit = vit_fit_final, crvit = crvit_fit_final)

  selected_list[[subtype_name]] <- tibble(
    subtype = subtype_name,
    predictor = unname(feature_labels[final_selection$terms]),
    model_variable = final_selection$terms,
    selection_source = "Training-cohort LASSO-Cox (alpha=1; lambda.min from 10-fold cross-validation by partial-likelihood deviance), followed by an unpenalized multivariable Cox refit on the original predictor scales",
    selected_penalty = final_selection$selected_penalty,
    lambda_selected = final_selection$lambda_selected,
    lambda_min = final_selection$lambda_min,
    lambda_1se = final_selection$lambda_1se
  )
  lasso_terms <- vapply(final_selection$columns, group_model_matrix_column, character(1))
  lasso_coefficient_list[[subtype_name]] <- tibble(
    subtype = subtype_name,
    predictor = unname(feature_labels[lasso_terms]),
    model_variable = lasso_terms,
    model_matrix_column = final_selection$columns,
    penalized_coefficient = final_selection$coefficients,
    selected_penalty = final_selection$selected_penalty,
    lambda_selected = final_selection$lambda_selected
  )
  coefficient_list[[subtype_name]] <- bind_rows(
    broom::tidy(clinic_fit_final, exponentiate = FALSE, conf.int = TRUE) %>% mutate(subtype = subtype_name, model = "Clinicoradiologic"),
    broom::tidy(cnn_fit_final, exponentiate = FALSE, conf.int = TRUE) %>% mutate(subtype = subtype_name, model = "3D-CNN"),
    broom::tidy(vit_fit_final, exponentiate = FALSE, conf.int = TRUE) %>% mutate(subtype = subtype_name, model = "3D-ViT"),
    broom::tidy(crvit_fit_final, exponentiate = FALSE, conf.int = TRUE) %>% mutate(subtype = subtype_name, model = "CR-ViT")
  ) %>% select(subtype, model, everything())

  for (m in c("cnn", "clinic", "vit", "crvit")) {
    z <- cox.zph(model_objects[[subtype_name]][[m]], transform = "km")
    ph_list[[paste(subtype_name, m)]] <- as.data.frame(z$table) %>%
      tibble::rownames_to_column("term") %>%
      transmute(subtype = subtype_name, model = model_labels[[m]], term,
                chisq = chisq, df = df, p_value = `p`)
  }
}

predictions <- bind_rows(prediction_list) %>% arrange(cohort, subtype, patient_id)
selected_predictors <- bind_rows(selected_list)
lasso_coefficients <- bind_rows(lasso_coefficient_list)
model_coefficients <- bind_rows(coefficient_list)
ph_results <- bind_rows(ph_list)

if (any(!complete.cases(predictions))) stop("Prediction table contains missing values.")

km_censor_survival <- function(time, event, query, left_limit = FALSE) {
  fit <- survfit(Surv(time, 1L - event) ~ 1)
  q <- if (left_limit) pmax(query - 1e-7, 0) else query
  s <- summary(fit, times = q, extend = TRUE)$surv
  pmax(s, 1e-6)
}

ipcw_brier <- function(time, event, predicted_survival, t) {
  y <- as.numeric(time > t)
  w <- numeric(length(time))
  before <- time <= t & event == 1L
  after <- time > t
  if (any(before)) w[before] <- 1 / km_censor_survival(time, event, time[before], left_limit = TRUE)
  if (any(after)) w[after] <- 1 / km_censor_survival(time, event, rep(t, sum(after)))
  mean(w * (y - predicted_survival)^2)
}

trapz_mean <- function(x, y) sum(diff(x) * (head(y, -1) + tail(y, -1)) / 2) / (max(x) - min(x))

calc_metrics <- function(df, model) {
  score <- df[[paste0(model, "_score")]]
  cidx <- as.numeric(concordance(Surv(df$pfs_time_months, df$event) ~ score, reverse = TRUE)$concordance)
  roc <- suppressWarnings(timeROC(
    T = df$pfs_time_months, delta = df$event, marker = score,
    cause = 1, weighting = "marginal", times = times, iid = FALSE
  ))
  auc <- as.numeric(roc$AUC)
  brier <- vapply(times, function(t) {
    ipcw_brier(df$pfs_time_months, df$event, df[[paste0(model, "_surv_", t)]], t)
  }, numeric(1))
  c(cindex = cidx, iauc = trapz_mean(times, auc), ibs = trapz_mean(times, brier),
    setNames(auc, paste0("auc_", times)), setNames(brier, paste0("brier_", times)))
}

format_p <- function(p) ifelse(p < .001, "<0.001", sprintf("%.3f", p))

performance_rows <- list()
comparison_rows <- list()
time_rows <- list()
bootstrap_draws <- list()
group_index <- 0L

evaluation_specs <- list()
for (cohort_name in levels(data$cohort)) {
  for (subtype_name in levels(data$subtype)) {
    evaluation_specs[[length(evaluation_specs) + 1L]] <- list(
      cohort = cohort_name, subtype = subtype_name, patient_ids = NULL
    )
  }
}
multiomics_ids <- read_csv(
  file.path(data_dir, "center1_gbm_multiomics_subset_patient_level.csv"),
  show_col_types = FALSE
) %>% pull(patient_id)
evaluation_specs[[length(evaluation_specs) + 1L]] <- list(
  cohort = "GBM multi-omics subset", subtype = "GBM", patient_ids = multiomics_ids
)

for (evaluation_spec in evaluation_specs) {
    cohort_name <- evaluation_spec$cohort
    subtype_name <- evaluation_spec$subtype
    group_index <- group_index + 1L
    df <- if (is.null(evaluation_spec$patient_ids)) {
      predictions %>% filter(cohort == cohort_name, subtype == subtype_name)
    } else {
      predictions %>% filter(patient_id %in% evaluation_spec$patient_ids, subtype == subtype_name)
    }
    point <- lapply(c("cnn", "clinic", "vit", "crvit"), function(m) calc_metrics(df, m))
    names(point) <- c("cnn", "clinic", "vit", "crvit")

    draws_list <- future.apply::future_lapply(seq_len(bootstrap_n), function(b) {
      set.seed(seed + group_index * 10000L + b)
      idx <- sample.int(nrow(df), nrow(df), replace = TRUE)
      boot_df <- df[idx, , drop = FALSE]
      one <- matrix(NA_real_, nrow = 4, ncol = 3,
                    dimnames = list(c("cnn", "clinic", "vit", "crvit"), c("cindex", "iauc", "ibs")))
      for (m in c("cnn", "clinic", "vit", "crvit")) {
        val <- tryCatch(calc_metrics(boot_df, m), error = function(e) rep(NA_real_, 15))
        one[m, ] <- val[c("cindex", "iauc", "ibs")]
      }
      one
    }, future.seed = TRUE)
    draws <- array(unlist(draws_list), dim = c(4, 3, bootstrap_n),
                   dimnames = list(c("cnn", "clinic", "vit", "crvit"), c("cindex", "iauc", "ibs"), NULL))
    draws <- aperm(draws, c(3, 1, 2))

    for (m in c("cnn", "clinic", "vit", "crvit")) {
      ci <- apply(draws[, m, , drop = FALSE], 3, quantile, probs = c(.025, .975), na.rm = TRUE)
      performance_rows[[length(performance_rows) + 1L]] <- tibble(
        cohort = cohort_name, subtype = subtype_name, model = model_labels[[m]],
        n = nrow(df), events = sum(df$event),
        cindex = point[[m]][["cindex"]], cindex_low = ci[1, "cindex"], cindex_high = ci[2, "cindex"],
        iauc = point[[m]][["iauc"]], iauc_low = ci[1, "iauc"], iauc_high = ci[2, "iauc"],
        ibs = point[[m]][["ibs"]], ibs_low = ci[1, "ibs"], ibs_high = ci[2, "ibs"]
      )
      time_rows[[length(time_rows) + 1L]] <- tibble(
        cohort = cohort_name, subtype = subtype_name, model = model_labels[[m]],
        time_months = times,
        auc = point[[m]][paste0("auc_", times)],
        brier = point[[m]][paste0("brier_", times)]
      )
    }

    for (comparison in list(
      c("cnn", "clinic"), c("vit", "clinic"), c("crvit", "clinic"),
      c("vit", "cnn"), c("crvit", "cnn"), c("crvit", "vit")
    )) {
      index <- comparison[[1]]
      reference <- comparison[[2]]
      for (metric in c("cindex", "iauc", "ibs")) {
        diff_draw <- draws[, index, metric] - draws[, reference, metric]
        point_diff <- point[[index]][[metric]] - point[[reference]][[metric]]
        ci <- quantile(diff_draw, c(.025, .975), na.rm = TRUE)
        z <- point_diff / sd(diff_draw, na.rm = TRUE)
        p <- 2 * pnorm(-abs(z))
        comparison_rows[[length(comparison_rows) + 1L]] <- tibble(
          cohort = cohort_name, subtype = subtype_name,
          comparison = paste0(model_labels[[index]], " vs ", model_labels[[reference]]),
          metric = metric, difference = point_diff, ci_low = ci[[1]], ci_high = ci[[2]],
          p_value = p, p_formatted = format_p(p)
        )
      }
    }

    bootstrap_draws[[length(bootstrap_draws) + 1L]] <- as_tibble(as.data.frame.table(draws, responseName = "value")) %>%
      transmute(cohort = cohort_name, subtype = subtype_name,
                bootstrap = as.integer(Var1), model = model_labels[as.character(Var2)],
                metric = as.character(Var3), value)
    message(sprintf("Completed bootstrap evaluation: %s / %s", cohort_name, subtype_name))
}

performance <- bind_rows(performance_rows)
comparisons <- bind_rows(comparison_rows)
time_metrics <- bind_rows(time_rows)
bootstrap_long <- bind_rows(bootstrap_draws)

km_stat_rows <- list()
for (evaluation_spec in evaluation_specs) {
  cohort_name <- evaluation_spec$cohort
  subtype_name <- evaluation_spec$subtype
  df <- if (is.null(evaluation_spec$patient_ids)) {
    predictions %>% filter(cohort == cohort_name, subtype == subtype_name)
  } else {
    predictions %>% filter(patient_id %in% evaluation_spec$patient_ids, subtype == subtype_name)
  }
  df <- df %>% mutate(
    crvit_risk_group = factor(crvit_risk_group, levels = c("Low risk", "High risk"))
  )
  fit <- coxph(Surv(pfs_time_months, event) ~ crvit_risk_group, data = df, ties = "efron")
  tidy_fit <- broom::tidy(fit, exponentiate = TRUE, conf.int = TRUE)
  zph <- cox.zph(fit, transform = "km")
  logrank <- survdiff(Surv(pfs_time_months, event) ~ crvit_risk_group, data = df)
  logrank_p <- 1 - pchisq(logrank$chisq, df = length(logrank$n) - 1L)
  km_stat_rows[[length(km_stat_rows) + 1L]] <- tibble(
    cohort = cohort_name,
    subtype = subtype_name,
    n = nrow(df),
    events = sum(df$event),
    analysis_horizon_months = NA_real_,
    low_risk_n = sum(df$crvit_risk_group == "Low risk"),
    high_risk_n = sum(df$crvit_risk_group == "High risk"),
    high_vs_low_hr = tidy_fit$estimate[[1]],
    ci_low = tidy_fit$conf.low[[1]],
    ci_high = tidy_fit$conf.high[[1]],
    cox_p_value = tidy_fit$p.value[[1]],
    logrank_p_value = logrank_p,
    ph_p_risk_group = zph$table["crvit_risk_group", "p"],
    ph_p_global = zph$table["GLOBAL", "p"]
  )
}
km_statistics <- bind_rows(km_stat_rows)

diagnostic_specs <- list()
for (cohort_name in levels(data$cohort)) {
  for (subtype_name in levels(data$subtype)) {
    diagnostic_specs[[length(diagnostic_specs) + 1L]] <- list(
      cohort = cohort_name,
      subtype = subtype_name,
      patient_ids = NULL,
      horizon = if_else(subtype_name == "GBM", 12, 24)
    )
  }
}
diagnostic_specs[[length(diagnostic_specs) + 1L]] <- list(
  cohort = "GBM multi-omics subset",
  subtype = "GBM",
  patient_ids = multiomics_ids,
  horizon = 12
)

get_diagnostic_data <- function(spec) {
  if (is.null(spec$patient_ids)) {
    predictions %>% filter(cohort == spec$cohort, subtype == spec$subtype)
  } else {
    predictions %>% filter(patient_id %in% spec$patient_ids, subtype == spec$subtype)
  }
}

calibration_rows <- list()
for (spec in diagnostic_specs) {
  df <- get_diagnostic_data(spec)
  for (m in c("clinic", "cnn", "vit", "crvit")) {
    pred_risk <- 1 - df[[paste0(m, "_surv_", spec$horizon)]]
    bins <- dplyr::ntile(pred_risk, 5)
    for (q in seq_len(5)) {
      idx <- bins == q
      km <- survfit(Surv(df$pfs_time_months[idx], df$event[idx]) ~ 1)
      obs_surv <- summary(km, times = spec$horizon, extend = TRUE)$surv[[1]]
      calibration_rows[[length(calibration_rows) + 1L]] <- tibble(
        cohort = spec$cohort,
        subtype = spec$subtype,
        model = model_labels[[m]], time_months = spec$horizon,
        risk_quantile = q, n = sum(idx),
        predicted_event_probability = mean(pred_risk[idx]),
        observed_event_probability = 1 - obs_surv
      )
    }
  }
}
calibration <- bind_rows(calibration_rows)

km_event_probability <- function(df, horizon) {
  if (!nrow(df)) return(NA_real_)
  fit <- survfit(Surv(pfs_time_months, event) ~ 1, data = df)
  1 - summary(fit, times = horizon, extend = TRUE)$surv[[1]]
}

dca_rows <- list()
dca_thresholds <- seq(0.10, 0.70, by = 0.025)
for (spec in diagnostic_specs) {
    df <- get_diagnostic_data(spec)
    overall_event <- km_event_probability(df, spec$horizon)
      for (threshold in dca_thresholds) {
        odds <- threshold / (1 - threshold)
        dca_rows[[length(dca_rows) + 1L]] <- tibble(
          cohort = spec$cohort,
          subtype = spec$subtype,
          time_months = spec$horizon,
          threshold = threshold, strategy = "Treat all",
          net_benefit = overall_event - (1 - overall_event) * odds, n = nrow(df)
        )
        dca_rows[[length(dca_rows) + 1L]] <- tibble(
          cohort = spec$cohort,
          subtype = spec$subtype,
          time_months = spec$horizon,
          threshold = threshold, strategy = "Treat none", net_benefit = 0, n = nrow(df)
        )
        for (m in c("clinic", "cnn", "vit", "crvit")) {
          risk <- 1 - df[[paste0(m, "_surv_", spec$horizon)]]
          flagged <- df[risk >= threshold, , drop = FALSE]
          net_benefit <- if (!nrow(flagged)) 0 else {
            fraction_flagged <- nrow(flagged) / nrow(df)
            flagged_event <- km_event_probability(flagged, spec$horizon)
            fraction_flagged * (flagged_event - (1 - flagged_event) * odds)
          }
          dca_rows[[length(dca_rows) + 1L]] <- tibble(
            cohort = spec$cohort,
            subtype = spec$subtype,
            time_months = spec$horizon,
            threshold = threshold, strategy = model_labels[[m]],
            net_benefit = net_benefit, n = nrow(df)
          )
        }
      }
}
decision_curve <- bind_rows(dca_rows)

diagnostic_brier_rows <- list()
for (spec in diagnostic_specs) {
  df <- get_diagnostic_data(spec)
  for (m in c("clinic", "cnn", "vit", "crvit")) {
    values <- vapply(times, function(t) {
      ipcw_brier(df$pfs_time_months, df$event, df[[paste0(m, "_surv_", t)]], t)
    }, numeric(1))
    diagnostic_brier_rows[[length(diagnostic_brier_rows) + 1L]] <- tibble(
      cohort = spec$cohort,
      subtype = spec$subtype,
      model = model_labels[[m]], time_months = times, brier = values,
      ibs = trapz_mean(times, values)
    )
  }
}
diagnostic_brier <- bind_rows(diagnostic_brier_rows)

bridge_variables <- tribble(
  ~label, ~variable, ~type,
  "Age", "age", "continuous",
  "Preoperative KPS", "preoperative_kps", "continuous",
  "Tumor volume", "log_tumor_volume", "continuous",
  "WHO grade", "who_grade", "ordinal",
  "MGMT unmethylated", "mgmt_promoter_methylation", "binary",
  "Subtotal resection", "extent_of_resection", "binary",
  "Enhancing proportion", "enhancing_ord", "ordinal",
  "Necrotic proportion", "necrotic_ord", "ordinal",
  "Peritumoral T2/FLAIR extent", "flair_ord", "ordinal",
  "Ependymal involvement", "ependymal_involvement", "binary",
  "Deep white matter invasion", "deep_white_matter_invasion", "binary",
  "Enhancing tumor crossing midline", "enhancing_tumor_crossing_midline", "binary",
  "Tumor location", "tumor_location", "nominal"
)

bridge_rows <- list()
bridge_r2_rows <- list()
conditional_rows <- list()

numeric_code <- function(x, type) {
  if (type == "continuous") return(as.numeric(x))
  if (type == "ordinal") return(as.numeric(x))
  if (type == "binary") return(as.numeric(x) - 1)
  stop("Nominal variables require a separate effect-size calculation.")
}

for (subtype_name in levels(data$subtype)) {
  train <- data %>% filter(subtype == subtype_name, cohort == "Training cohort")
  for (i in seq_len(nrow(bridge_variables))) {
    spec <- bridge_variables[i, ]
    x <- train[[spec$variable]]
    if (spec$type == "nominal") {
      fit <- lm(train$vit_z ~ x)
      a <- anova(fit)
      strength <- sqrt(max(0, a$`Sum Sq`[[1]] / sum(a$`Sum Sq`)))
      p <- a$`Pr(>F)`[[1]]
      method <- "Correlation ratio"
    } else {
      test <- suppressWarnings(cor.test(train$vit_z, numeric_code(x, spec$type), method = "spearman", exact = FALSE))
      strength <- unname(test$estimate)
      p <- test$p.value
      method <- "Spearman"
    }
    bridge_rows[[length(bridge_rows) + 1L]] <- tibble(
      subtype = subtype_name, factor = spec$label, method,
      association = strength, p_value = p
    )
  }

  x <- make_x(train)
  y <- train$vit_z
  fold <- stratified_folds(train, 10L, seed_offset = 30000L + match(subtype_name, levels(data$subtype)))
  pred <- rep(NA_real_, nrow(train))
  for (k in sort(unique(fold))) {
    fit <- cv.glmnet(x[fold != k, , drop = FALSE], y[fold != k], family = "gaussian", alpha = 0,
                     nfolds = 8, standardize = TRUE)
    pred[fold == k] <- as.numeric(predict(fit, newx = x[fold == k, , drop = FALSE], s = "lambda.min"))
  }
  cv_r2 <- 1 - sum((y - pred)^2) / sum((y - mean(y))^2)
  bridge_r2_rows[[length(bridge_r2_rows) + 1L]] <- tibble(
    subtype = subtype_name, n = nrow(train), cross_validated_r2 = cv_r2,
    unexplained_fraction = 1 - cv_r2
  )

  validation <- predictions %>%
    filter(subtype == subtype_name, cohort != "Training cohort") %>%
    mutate(clinic_z = as.numeric(scale(clinic_score)), vit_score_z = as.numeric(scale(vit_score)))
  conditional <- safe_coxph(Surv(pfs_time_months, event) ~ clinic_z + vit_score_z + strata(cohort), validation)
  td <- broom::tidy(conditional, exponentiate = TRUE, conf.int = TRUE) %>% filter(term == "vit_score_z")
  zph <- cox.zph(conditional, transform = "km")
  conditional_rows[[length(conditional_rows) + 1L]] <- tibble(
    subtype = subtype_name, n = nrow(validation), events = sum(validation$event),
    adjusted_hr_per_sd_vit = td$estimate, ci_low = td$conf.low, ci_high = td$conf.high,
    p_value = td$p.value, ph_p_vit = zph$table["vit_score_z", "p"], ph_p_global = zph$table["GLOBAL", "p"]
  )
}

bridge_associations <- bind_rows(bridge_rows) %>%
  group_by(subtype) %>% mutate(p_fdr = p.adjust(p_value, method = "BH")) %>% ungroup()
bridge_r2 <- bind_rows(bridge_r2_rows)
conditional_value <- bind_rows(conditional_rows)

write_excel_csv(predictions, file.path(results_dir, "clinicoradiologic_crvit_patient_predictions.csv"), na = "")
write_excel_csv(performance, file.path(results_dir, "clinicoradiologic_crvit_performance.csv"), na = "")
write_excel_csv(comparisons, file.path(results_dir, "clinicoradiologic_crvit_paired_comparisons.csv"), na = "")
write_excel_csv(time_metrics, file.path(results_dir, "clinicoradiologic_crvit_time_dependent_metrics.csv"), na = "")
write_excel_csv(calibration, file.path(results_dir, "clinicoradiologic_crvit_calibration.csv"), na = "")
write_excel_csv(decision_curve, file.path(results_dir, "clinicoradiologic_crvit_decision_curve.csv"), na = "")
write_excel_csv(diagnostic_brier, file.path(results_dir, "clinicoradiologic_crvit_diagnostic_brier.csv"), na = "")
write_excel_csv(selected_predictors, file.path(results_dir, "clinicoradiologic_selected_predictors.csv"), na = "")
write_excel_csv(lasso_coefficients, file.path(results_dir, "clinicoradiologic_lasso_coefficients.csv"), na = "")
write_excel_csv(model_coefficients, file.path(results_dir, "clinicoradiologic_crvit_model_coefficients.csv"), na = "")
write_excel_csv(ph_results, file.path(results_dir, "clinicoradiologic_crvit_proportional_hazards.csv"), na = "")
write_excel_csv(bridge_associations, file.path(results_dir, "vit_clinicoradiologic_bridge_associations.csv"), na = "")
write_excel_csv(bridge_r2, file.path(results_dir, "vit_clinicoradiologic_bridge_crossvalidated_r2.csv"), na = "")
write_excel_csv(conditional_value, file.path(results_dir, "vit_incremental_conditional_cox_validation.csv"), na = "")
write_excel_csv(bootstrap_long, file.path(results_dir, "clinicoradiologic_crvit_bootstrap_draws.csv"), na = "")
write_excel_csv(km_statistics, file.path(results_dir, "clinicoradiologic_crvit_km_statistics.csv"), na = "")

summary_audit <- tibble(
  item = c(
    "patient_rows", "unique_patient_ids", "prediction_rows", "bootstrap_resamples",
    "all_crvit_ph_global_p_gt_0.05", "all_validation_conditional_ph_global_p_gt_0.05",
    "postoperative_treatment_in_primary_model"
  ),
  value = c(
    nrow(data), n_distinct(data$patient_id), nrow(predictions), bootstrap_n,
    all(ph_results$p_value[ph_results$model == "CR-ViT" & ph_results$term == "GLOBAL"] > .05),
    all(conditional_value$ph_p_global > .05), FALSE
  )
)
write_excel_csv(summary_audit, file.path(qa_dir, "clinicoradiologic_crvit_analysis_audit.csv"), na = "")

message("Completed leakage-restricted LASSO-Cox clinicoradiologic and CR-ViT survival modeling with paired bootstrap performance evaluation.")
