# Constructed counts test the implementation, not study biology.
script <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[[1]])
source(file.path(dirname(normalizePath(script)), "..", "single_cell_helpers.R"))
set.seed(14)
states <- c("State 1", "State 2", "State 3")
meta <- do.call(rbind, lapply(seq_len(12), function(i) {
  labels <- sample(c(states, "Reference"), 120, replace = TRUE, prob = c(.3, .2, .3, .2))
  data.frame(patient_id = paste0("p", i), vit_risk_group = if (i <= 6) "Low risk" else "High risk",
             neural_state = labels, row.names = paste0("p", i, "_", seq_along(labels)))
}))
result <- state_abundance(meta, states)
stopifnot(nrow(result$abundance) == 36, nrow(result$comparison) == 3,
          all(abs(tapply(result$abundance$cell_fraction, result$abundance$patient_id, sum) - 1) < 1e-12))
expected <- table(meta$patient_id[meta$neural_state != "Reference"])
stopifnot(all(result$abundance$denominator == expected[as.character(result$abundance$patient_id)]))
counts <- Matrix::Matrix(matrix(rpois(60 * nrow(meta), 3), nrow = 60,
                 dimnames = list(paste0("G", 1:60), rownames(meta))), sparse = TRUE)
markers <- paired_state_markers(counts, meta, "State 2", states)
stopifnot(nrow(markers) > 0, all(is.finite(markers$FDR)), all(markers$FDR >= 0 & markers$FDR <= 1))
enrichment <- enrich_sets(paste0("G", 1:10), paste0("G", 1:60),
                          list(set_a = paste0("G", 1:15), set_b = paste0("G", 31:50)))
stopifnot(nrow(enrichment) == 2, enrichment$gene_count[1] == 10,
          enrichment$gene_count[2] == 0, all(enrichment$fdr_bh >= enrichment$p_value))
cat("Patient-replicated state abundance, paired pseudobulk and enrichment tests passed.\n")
