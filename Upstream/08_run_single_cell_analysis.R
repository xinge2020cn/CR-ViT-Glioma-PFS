# Seurat, fastCNV, CytoTRACE and tumor-replicated biological interpretation.
suppressPackageStartupMessages(library(Seurat))
args <- commandArgs(trailingOnly = TRUE)
arg <- function(name, default = NULL) {
  hit <- grep(paste0("^--", name, "="), args, value = TRUE)
  value <- if (length(hit)) sub(paste0("^--", name, "="), "", hit[[1]]) else default
  if (is.null(value)) stop("Missing --", name, "=...")
  value
}
script <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[[1]])
source(file.path(dirname(normalizePath(script)), "single_cell_helpers.R"))
config <- yaml::read_yaml(arg("config"))
cfg <- config$single_cell
stage <- arg("stage")
if (!stage %in% c("cluster", "cnv", "interpret")) stop("Stage must be cluster, cnv or interpret.")
out <- arg("output-dir")
dir.create(out, recursive = TRUE, showWarnings = FALSE)
set.seed(config$project$random_seed)
risk <- read_risk_manifest(arg("risk-manifest"))
object <- readRDS(arg("input"))
if (!inherits(object, "Seurat")) stop("Input must be a Seurat object with raw RNA counts.")
object@meta.data <- attach_frozen_risk(object[[]], risk)
DefaultAssay(object) <- "RNA"
if (inherits(object[["RNA"]], "Assay5")) object <- JoinLayers(object, assay = "RNA")
counts <- GetAssayData(object, assay = "RNA", layer = "counts")
values <- if (inherits(counts, "sparseMatrix")) counts@x else as.vector(counts)
if (any(!is.finite(values)) || any(values < 0) || any(values != round(values)))
  stop("RNA counts must be finite nonnegative integers.")
if (stage == "cluster") {
  object[["percent.mt"]] <- PercentageFeatureSet(object, pattern = "^MT-")
  cells <- rownames(object[[]])[object$nFeature_RNA >= cfg$min_genes_per_cell &
                                  object$percent.mt <= cfg$max_mitochondrial_percent]
  object <- subset(object, cells = cells)
  object@meta.data <- attach_frozen_risk(object[[]], risk)
  counts <- GetAssayData(object, layer = "counts")
  genes <- rownames(counts)[Matrix::rowSums(counts > 0) >= cfg$min_cells_per_gene]
  object <- subset(object, features = genes)
  pieces <- SplitObject(object, split.by = "patient_id")
  dims <- seq_len(min(cfg$pca_components, min(vapply(pieces, ncol, numeric(1))) - 1L,
                      length(genes) - 1L))
  if (length(dims) < 3) stop("Too few cells or genes for integration.")
  pieces <- lapply(pieces, function(x) FindVariableFeatures(NormalizeData(x, verbose = FALSE),
                     nfeatures = cfg$highly_variable_genes, verbose = FALSE))
  features <- SelectIntegrationFeatures(pieces, nfeatures = cfg$highly_variable_genes)
  anchors <- FindIntegrationAnchors(pieces, anchor.features = features, dims = dims)
  object <- IntegrateData(anchors, dims = dims)
  DefaultAssay(object) <- "integrated"
  object <- ScaleData(object, verbose = FALSE)
  object <- RunPCA(object, npcs = length(dims), verbose = FALSE)
  object <- FindNeighbors(object, dims = dims, k.param = cfg$neighbors, verbose = FALSE)
  object <- FindClusters(object, resolution = cfg$cluster_resolution,
                         random.seed = config$project$random_seed, verbose = FALSE)
  object <- RunUMAP(object, dims = dims, seed.use = config$project$random_seed, verbose = FALSE)
  DefaultAssay(object) <- "RNA"
  if (inherits(object[["RNA"]], "Assay5")) object <- JoinLayers(object, assay = "RNA")
  markers <- FindAllMarkers(object, assay = "RNA", only.pos = TRUE, test.use = "wilcox")
  write.csv(markers, file.path(out, "cluster_markers.csv"), row.names = FALSE)
  write.csv(cbind(cell_id = colnames(object), object[[]], Embeddings(object, "umap")),
            file.path(out, "cell_clusters.csv"), row.names = FALSE)
  saveRDS(object, file.path(out, "clustered.rds"))
} else if (stage == "cnv") {
  if (!requireNamespace("fastCNV", quietly = TRUE) ||
      as.character(packageVersion("fastCNV")) != "1.1.9")
    stop("This interface requires fastCNV 1.1.9.")
  annotation <- read.csv(arg("annotations"), stringsAsFactors = FALSE)
  if (!all(c("cell_id", "cell_type") %in% names(annotation)) ||
      anyDuplicated(annotation$cell_id) || anyNA(annotation$cell_type) ||
      !setequal(annotation$cell_id, colnames(object))) stop("Complete reviewed cell-type annotations are required.")
  object$cell_type <- annotation$cell_type[match(colnames(object), annotation$cell_id)]
  references <- strsplit(arg("reference-types"), ",", fixed = TRUE)[[1]]
  if (!all(references %in% object$cell_type)) stop("CNV reference labels are absent.")
  pieces <- SplitObject(object, split.by = "patient_id")
  cnv <- fastCNV::fastCNV(seuratObj = pieces, sampleName = names(pieces), assay = "RNA",
                          referenceVar = "cell_type", referenceLabel = references,
                          prepareCounts = FALSE, pooledReference = TRUE,
                          doPlot = TRUE, printPlot = FALSE, savePath = out)
  saveRDS(cnv, file.path(out, "fastcnv_objects.rds"))
  meta <- do.call(rbind, unname(lapply(cnv, function(x) x[[]])))
  if (!setequal(rownames(meta), colnames(object)) || anyDuplicated(rownames(meta)) ||
      !"cnv_fraction" %in% names(meta)) stop("fastCNV output cells or metadata could not be linked.")
  object@meta.data <- meta[colnames(object), , drop = FALSE]
  neural <- subset(object, cells = colnames(object)[object$cell_type %in% unlist(cfg$neural_cell_types)])
  if (!"integrated" %in% Assays(neural)) stop("Integrated expression is required for neural reclustering.")
  DefaultAssay(neural) <- "integrated"
  dims <- seq_len(min(cfg$pca_components, ncol(neural) - 1L, length(VariableFeatures(neural)) - 1L))
  neural <- RunPCA(ScaleData(neural, verbose = FALSE), npcs = length(dims), verbose = FALSE)
  neural <- FindNeighbors(neural, dims = dims, k.param = cfg$neighbors, verbose = FALSE)
  neural <- FindClusters(neural, resolution = cfg$neural_cluster_resolution,
                         random.seed = config$project$random_seed, verbose = FALSE)
  neural <- RunUMAP(neural, dims = dims, seed.use = config$project$random_seed, verbose = FALSE)
  object$neural_cluster <- NA_character_
  object$neural_cluster[match(colnames(neural), colnames(object))] <- as.character(Idents(neural))
  write.csv(cbind(cell_id = colnames(neural), neural[[]], Embeddings(neural, "pca")),
            file.path(out, "neural_cnv_expression_profiles.csv"), row.names = FALSE)
  DefaultAssay(neural) <- "RNA"
  write.csv(FindAllMarkers(neural, only.pos = TRUE), file.path(out, "neural_cluster_markers.csv"), row.names = FALSE)
  saveRDS(neural, file.path(out, "neural_reclustered.rds"))
  saveRDS(object, file.path(out, "cnv_annotated.rds"))
} else {
  if (!requireNamespace("CytoTRACE", quietly = TRUE))
    stop("Install the original CytoTRACE package before state characterization.")
  states <- unlist(cfg$identified_states)
  annotation <- read.csv(arg("state-annotations"), stringsAsFactors = FALSE)
  neural_ids <- colnames(object)[object$cell_type %in% unlist(cfg$neural_cell_types)]
  if (!all(c("cell_id", "neural_state") %in% names(annotation)) ||
      anyDuplicated(annotation$cell_id) || !setequal(annotation$cell_id, neural_ids) ||
      anyNA(annotation$neural_state) ||
      !all(annotation$neural_state %in% c(states, "Reference")))
    stop("Provide reviewed CNV-expression state assignments for every neural-lineage cell.")
  if (!"cnv_fraction" %in% names(object[[]])) stop("Run fastCNV before interpreting states.")
  object$neural_state <- NA_character_
  object$neural_state[match(annotation$cell_id, colnames(object))] <- annotation$neural_state
  meta <- object[[]]
  result <- state_abundance(meta, states)
  write.csv(result$abundance, file.path(out, "patient_state_abundance.csv"), row.names = FALSE)
  write.csv(result$comparison, file.path(out, "state_abundance_edger.csv"), row.names = FALSE)
  types <- as.data.frame(table(patient_id = meta$patient_id, cell_type = meta$cell_type))
  types$fraction <- types$Freq / ave(types$Freq, types$patient_id, FUN = sum)
  write.csv(types, file.path(out, "patient_cell_type_composition.csv"), row.names = FALSE)
  cyto <- CytoTRACE::CytoTRACE(as.matrix(counts), enableFast = TRUE, ncores = 1)
  object$CytoTRACE <- cyto$CytoTRACE[colnames(object)]
  object$analysis_population <- ifelse(is.na(object$neural_state), object$cell_type, object$neural_state)
  write.csv(cbind(cell_id = colnames(object), object[[]]), file.path(out, "cell_state_characterization.csv"), row.names = FALSE)
  cyto_summary <- aggregate(CytoTRACE ~ patient_id + analysis_population + vit_risk_group,
                            object[[]], median, na.rm = TRUE)
  write.csv(cyto_summary, file.path(out, "patient_cytotrace_summary.csv"), row.names = FALSE)
  hallmark <- read_gene_sets(arg("hallmark-gmt"))
  kegg <- read_gene_sets(arg("kegg-gmt"))
  signatures <- character()
  for (state in states) {
    markers <- paired_state_markers(counts, meta, state, states)
    genes <- markers$gene[markers$FDR < cfg$marker_fdr & markers$logFC > cfg$marker_log2fc]
    slug <- gsub(" ", "_", state, fixed = TRUE)
    write.csv(markers, file.path(out, paste0(slug, "_pseudobulk_markers.csv")), row.names = FALSE)
    write.csv(enrich_sets(genes, markers$gene, hallmark), file.path(out, paste0(slug, "_hallmark.csv")), row.names = FALSE)
    write.csv(enrich_sets(genes, markers$gene, kegg), file.path(out, paste0(slug, "_kegg.csv")), row.names = FALSE)
    if (length(genes)) signatures <- c(signatures, paste(c(slug, "paired_tumor_state_signature", genes), collapse = "\t"))
  }
  writeLines(signatures, file.path(out, "state_signatures.gmt"))
  saveRDS(object, file.path(out, "characterized.rds"))
}
writeLines(capture.output(sessionInfo()), file.path(out, paste0(stage, "_session_info.txt")))
