# Patient-level inference helpers; no target cohort sizes or results are imposed.
read_risk_manifest <- function(path) {
  risk <- read.csv(path, stringsAsFactors = FALSE, colClasses = c(patient_id = "character"))
  needed <- c("patient_id", "subtype", "omics_type", "vit_km_score", "vit_cutoff", "vit_risk_group")
  if (!all(needed %in% names(risk)) || anyNA(risk[needed]) || anyDuplicated(risk$patient_id))
    stop("Invalid frozen risk manifest.")
  if (!all(risk$subtype == "GBM") || length(unique(risk$vit_cutoff)) != 1 ||
      any(!is.finite(risk$vit_km_score)) || any(!is.finite(risk$vit_cutoff)) ||
      !all(risk$omics_type %in% c("Single-cell RNA-seq", "Bulk transcriptomics")))
    stop("Risk manifest must use disjoint GBM subsets and a single frozen cutoff.")
  expected <- ifelse(risk$vit_km_score >= risk$vit_cutoff, "High risk", "Low risk")
  if (!all(risk$vit_risk_group == expected)) stop("Stale imaging-risk groups.")
  risk
}

attach_frozen_risk <- function(metadata, risk) {
  if (any(c("risk_group_3dvit", "training_cutoff_3dvit") %in% names(metadata)))
    stop("Remove obsolete risk fields; use only the frozen ViT manifest.")
  ids <- risk$patient_id[risk$omics_type == "Single-cell RNA-seq"]
  if (!setequal(unique(metadata$patient_id), ids))
    stop("Single-cell patient membership differs from the frozen risk manifest.")
  values <- risk$vit_risk_group[match(metadata$patient_id, risk$patient_id)]
  for (column in intersect(c("risk_group", "vit_risk_group"), names(metadata))) {
    if (anyNA(metadata[[column]]) || !all(metadata[[column]] == values))
      stop("Supplied single-cell risk groups disagree with frozen predictions.")
  }
  metadata$vit_risk_group <- values
  metadata
}

state_abundance <- function(metadata, states) {
  patients <- sort(unique(metadata$patient_id))
  selected <- metadata[metadata$neural_state %in% states, , drop = FALSE]
  counts <- table(factor(selected$neural_state, levels = states),
                  factor(selected$patient_id, levels = patients))
  dimnames(counts) <- list(state = states, patient_id = patients)
  totals <- colSums(counts)
  if (any(totals == 0)) stop("Some tumors have no cells assigned to an identified neural state.")
  abundance <- as.data.frame(counts, responseName = "cell_count")
  abundance$denominator <- totals[match(abundance$patient_id, patients)]
  abundance$cell_fraction <- abundance$cell_count / abundance$denominator
  groups <- unique(metadata[c("patient_id", "vit_risk_group")])
  if (anyDuplicated(groups$patient_id)) stop("Each tumor must have a single risk group.")
  group <- factor(groups$vit_risk_group[match(patients, groups$patient_id)],
                  levels = c("Low risk", "High risk"))
  if (anyNA(group) || any(table(group) < 2)) stop("At least two tumors per group are required.")
  design <- model.matrix(~ group)
  y <- edgeR::DGEList(counts = counts, lib.size = totals, norm.factors = rep(1, length(totals)))
  # The denominator is all identified-state neural cells, excluding the reference population.
  y <- edgeR::estimateDisp(y, design, trend.method = "none", robust = TRUE)
  fit <- edgeR::glmQLFit(y, design, robust = TRUE, abundance.trend = FALSE)
  result <- edgeR::topTags(edgeR::glmQLFTest(fit, coef = 2), n = Inf, sort.by = "none")$table
  result$state <- rownames(result)
  abundance$vit_risk_group <- groups$vit_risk_group[match(abundance$patient_id, groups$patient_id)]
  list(abundance = abundance, comparison = result)
}

paired_state_markers <- function(counts, metadata, target_state, states) {
  keep <- metadata$neural_state %in% states
  metadata <- metadata[keep, , drop = FALSE]
  counts <- counts[, rownames(metadata), drop = FALSE]
  metadata$contrast <- ifelse(metadata$neural_state == target_state, "target", "other")
  paired <- table(metadata$patient_id, metadata$contrast)
  patients <- rownames(paired)[rowSums(paired > 0) == 2]
  if (length(patients) < 3) stop("State-expression comparison needs at least three paired tumors.")
  metadata <- metadata[metadata$patient_id %in% patients, , drop = FALSE]
  counts <- counts[, rownames(metadata), drop = FALSE]
  samples <- unique(metadata[c("patient_id", "contrast")])
  keys <- paste(samples$patient_id, samples$contrast, sep = "::")
  membership <- match(paste(metadata$patient_id, metadata$contrast, sep = "::"), keys)
  aggregate <- counts %*% Matrix::sparseMatrix(i = seq_len(nrow(metadata)), j = membership,
                                             x = 1, dims = c(nrow(metadata), nrow(samples)))
  colnames(aggregate) <- keys
  samples$contrast <- factor(samples$contrast, levels = c("other", "target"))
  design <- model.matrix(~ factor(patient_id) + contrast, data = samples)
  y <- edgeR::DGEList(counts = aggregate)
  keep <- edgeR::filterByExpr(y, design)
  if (!any(keep)) stop("No genes passed the pseudobulk expression filter.")
  y <- edgeR::calcNormFactors(y[keep, , keep.lib.sizes = FALSE])
  y <- edgeR::estimateDisp(y, design, robust = TRUE)
  fit <- edgeR::glmQLFit(y, design, robust = TRUE)
  result <- edgeR::topTags(edgeR::glmQLFTest(fit, coef = ncol(design)), n = Inf)$table
  result$gene <- rownames(result)
  result
}

read_gene_sets <- function(path) {
  fields <- strsplit(readLines(path, warn = FALSE), "\t", fixed = TRUE)
  if (!length(fields) || any(lengths(fields) < 3)) stop("Invalid GMT gene sets.")
  labels <- vapply(fields, "[[", character(1), 1)
  if (anyDuplicated(labels)) stop("Gene-set names must be unique.")
  setNames(lapply(fields, function(x) unique(x[-c(1, 2)])), labels)
}

enrich_sets <- function(selected, universe, sets) {
  selected <- intersect(selected, universe)
  result <- do.call(rbind, lapply(names(sets), function(name) {
    genes <- intersect(sets[[name]], universe)
    overlap <- intersect(genes, selected)
    data.frame(pathway = name, gene_count = length(overlap), set_size = length(genes),
               signature_size = length(selected), universe_size = length(universe),
               gene_ratio = if (length(selected)) length(overlap) / length(selected) else NA_real_,
               p_value = if (length(selected) && length(genes))
                 phyper(length(overlap) - 1, length(genes), length(universe) - length(genes),
                        length(selected), lower.tail = FALSE) else 1)
  }))
  result$fdr_bh <- p.adjust(result$p_value, method = "BH")
  result
}
