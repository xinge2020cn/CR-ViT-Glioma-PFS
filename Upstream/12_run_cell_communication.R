# Patient-replicated ligand-receptor analysis on characterized cell populations.
suppressPackageStartupMessages({
  library(Seurat)
  library(CellChat)
})
args <- commandArgs(trailingOnly = TRUE)
arg <- function(name) {
  hit <- grep(paste0("^--", name, "="), args, value = TRUE)
  if (!length(hit)) stop("Missing --", name, "=...")
  sub(paste0("^--", name, "="), "", hit[[1]])
}
script <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[[1]])
source(file.path(dirname(normalizePath(script)), "single_cell_helpers.R"))
config <- yaml::read_yaml(arg("config"))
out <- arg("output-dir")
dir.create(out, recursive = TRUE, showWarnings = FALSE)
set.seed(config$project$random_seed)
object <- readRDS(arg("input"))
object@meta.data <- attach_frozen_risk(object[[]], read_risk_manifest(arg("risk-manifest")))
if (!"analysis_population" %in% names(object[[]])) stop("Run state characterization first.")
DefaultAssay(object) <- "RNA"
if (inherits(object[["RNA"]], "Assay5")) object <- JoinLayers(object, assay = "RNA")
object <- NormalizeData(object, verbose = FALSE)
expression <- GetAssayData(object, assay = "RNA", layer = "data")
meta <- object[[]]
patients <- sort(unique(meta$patient_id))
rows <- list()
for (patient in patients) {
  cells <- rownames(meta)[meta$patient_id == patient]
  input <- expression[, cells, drop = FALSE]
  cell_meta <- meta[cells, , drop = FALSE]
  chat <- createCellChat(input, meta = cell_meta, group.by = "analysis_population")
  chat@DB <- CellChatDB.human
  chat <- subsetData(chat)
  chat <- identifyOverExpressedGenes(chat)
  chat <- identifyOverExpressedInteractions(chat)
  chat <- computeCommunProb(chat, type = "triMean", raw.use = TRUE,
                           nboot = config$single_cell$communication_bootstraps,
                           seed.use = config$project$random_seed)
  chat <- filterCommunication(chat, min.cells = config$single_cell$communication_min_cells)
  chat <- computeCommunProbPathway(chat)
  chat <- aggregateNet(chat)
  probability <- chat@net$prob
  axes <- dimnames(probability)
  if (length(axes) != 3 || any(lengths(axes) == 0)) stop("No estimable interactions for ", patient)
  frame <- expand.grid(sender = axes[[1]], receiver = axes[[2]], interaction = axes[[3]],
                       KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE)
  frame$strength <- as.vector(probability)
  frame$p_value_within_tumor <- as.vector(chat@net$pval)
  frame$fdr_within_tumor <- p.adjust(frame$p_value_within_tumor, "BH")
  frame$patient_id <- patient
  rows[[patient]] <- frame
  saveRDS(chat, file.path(out, paste0("cellchat_", match(patient, patients), ".rds")))
}
raw <- do.call(rbind, rows)
keys <- unique(raw[c("sender", "receiver", "interaction")])
keys$key <- seq_len(nrow(keys))
raw <- merge(raw, keys, by = c("sender", "receiver", "interaction"))
grid <- expand.grid(patient_id = patients, key = keys$key, stringsAsFactors = FALSE)
grid <- merge(grid, raw[c("patient_id", "key", "strength")], all.x = TRUE)
grid$strength[is.na(grid$strength)] <- 0
patient_meta <- unique(meta[c("patient_id", "vit_risk_group")])
grid <- merge(grid, patient_meta, by = "patient_id")
comparison <- do.call(rbind, lapply(split(grid, grid$key), function(frame) {
  low <- frame$strength[frame$vit_risk_group == "Low risk"]
  high <- frame$strength[frame$vit_risk_group == "High risk"]
  if (length(low) < 2 || length(high) < 2) stop("At least two tumors per risk group are required.")
  p <- if (length(unique(c(low, high))) == 1) 1 else
    wilcox.test(low, high, alternative = "two.sided", exact = FALSE)$p.value
  data.frame(key = frame$key[[1]], n_low = length(low), n_high = length(high),
             mean_low = mean(low), mean_high = mean(high),
             difference_high_minus_low = mean(high) - mean(low), p_value = p)
}))
comparison$fdr_bh <- p.adjust(comparison$p_value, "BH")
comparison <- merge(keys, comparison, by = "key")
full <- merge(grid, keys, by = "key")
outgoing <- aggregate(strength ~ patient_id + vit_risk_group + sender, full, sum)
incoming <- aggregate(strength ~ patient_id + vit_risk_group + receiver, full, sum)
counts <- aggregate(I(fdr_within_tumor < .05 & strength > 0) ~ patient_id, raw, sum)
names(counts)[2] <- "supported_interactions"
write.csv(raw, file.path(out, "patient_ligand_receptor_scores.csv"), row.names = FALSE)
write.csv(comparison, file.path(out, "risk_group_interaction_comparison.csv"), row.names = FALSE)
write.csv(outgoing, file.path(out, "patient_outgoing_signaling.csv"), row.names = FALSE)
write.csv(incoming, file.path(out, "patient_incoming_signaling.csv"), row.names = FALSE)
write.csv(counts, file.path(out, "patient_interaction_counts.csv"), row.names = FALSE)
writeLines(capture.output(sessionInfo()), file.path(out, "communication_session_info.txt"))
