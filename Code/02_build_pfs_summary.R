suppressPackageStartupMessages(library(survival))

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
data_file <- file.path(root, "Data", "imaging_3center_patient_level.csv")
subset_file <- file.path(root, "Data", "center1_gbm_multiomics_subset_patient_level.csv")
out_file <- file.path(root, "Results", "pfs_descriptive_summary.csv")

d <- read.csv(data_file, check.names = FALSE, stringsAsFactors = FALSE)
subset_data <- read.csv(subset_file, check.names = FALSE, stringsAsFactors = FALSE)
names(d)[1] <- sub("^\\ufeff|^\\xef\\xbb\\xbf", "", names(d)[1])
names(subset_data)[1] <- sub("^\\ufeff|^\\xef\\xbb\\xbf", "", names(subset_data)[1])
if (!"patient_id" %in% names(d)) names(d)[1] <- "patient_id"
if (!"patient_id" %in% names(subset_data)) names(subset_data)[1] <- "patient_id"
subset_ids <- unique(subset_data$patient_id)

km_value <- function(fit, name) {
  value <- unname(summary(fit)$table[[name]])
  if (!length(value) || !is.finite(value)) NA_real_ else value
}

summarize_group <- function(x, group_type, group_label) {
  pfs_fit <- survfit(Surv(pfs_time_months, event) ~ 1, data = x)
  follow_fit <- survfit(Surv(pfs_time_months, 1 - event) ~ 1, data = x)
  landmarks <- summary(pfs_fit, times = c(12, 24), extend = TRUE)$surv
  data.frame(
    group_type = group_type,
    group_label = group_label,
    n = nrow(x),
    events = sum(x$event),
    event_rate = mean(x$event),
    reverse_km_followup_median_months = km_value(follow_fit, "median"),
    median_pfs_months = km_value(pfs_fit, "median"),
    median_pfs_lower_95ci = km_value(pfs_fit, "0.95LCL"),
    median_pfs_upper_95ci = km_value(pfs_fit, "0.95UCL"),
    pfs_12_months = landmarks[[1]],
    pfs_24_months = landmarks[[2]],
    stringsAsFactors = FALSE
  )
}

rows <- list(summarize_group(d, "overall", "All imaging patients"))
for (label in c("Training cohort", "Temporal validation cohort", "Spatial validation cohort")) {
  rows[[length(rows) + 1]] <- summarize_group(d[d$cohort == label, ], "cohort", label)
}
for (label in c("IDHmut-intact", "IDHmut-codel", "GBM")) {
  rows[[length(rows) + 1]] <- summarize_group(d[d$subtype == label, ], "molecular subtype", label)
}
rows[[length(rows) + 1]] <- summarize_group(
  d[d$patient_id %in% subset_ids, ], "nested subset", "GBM multi-omics subset"
)

out <- do.call(rbind, rows)
write.csv(out, out_file, row.names = FALSE, na = "")
message("PFS descriptive summary written for overall, cohort, subtype, and nested-subset reporting.")
