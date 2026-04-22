# ============================================================
# SC3 clustering for PBMC 10k with SVM mode (local version)
#
# Source notebook: SC3_10k_R.ipynb
# All Google Drive / Colab dependencies removed.
#
# Prerequisites (install once):
#   BiocManager::install(c("SC3", "SingleCellExperiment", "zellkonverter"))
#   install.packages("Matrix")
#
# Input:  <repo_root>/pbmc10k_preprocessed.h5ad
# Output: <repo_root>/sc3_labels_10k.csv
#
# Notes:
#   - SVM mode (svm_num_cells=5000) is used for scalability on 11k cells.
#   - sc3_run_svm() predicts cluster labels for non-training cells.
#   - biology=FALSE skips DEG/marker gene computation (reduces run time).
# ============================================================

library(SC3)
library(SingleCellExperiment)
library(Matrix)
library(zellkonverter)

# ── Paths ────────────────────────────────────────────────────
.args <- commandArgs(trailingOnly = FALSE)
.file_flag <- grep("^--file=", .args, value = TRUE)
if (length(.file_flag) > 0) {
    SCRIPT_DIR <- dirname(normalizePath(sub("^--file=", "", .file_flag[1])))
} else if (!is.null(sys.frames()[[1]]$ofile)) {
    SCRIPT_DIR <- dirname(normalizePath(sys.frames()[[1]]$ofile))
} else {
    SCRIPT_DIR <- getwd()
    message("Warning: could not detect script directory; using CWD: ", SCRIPT_DIR)
}

BASE_DIR  <- normalizePath(file.path(SCRIPT_DIR, ".."))
H5AD_PATH <- file.path(BASE_DIR, "pbmc10k_preprocessed.h5ad")
OUT_CSV   <- file.path(BASE_DIR, "sc3_labels_10k.csv")

# ============================================================
# CELL 2: Load PBMC 10k preprocessed data
# ============================================================
cat("Reading:", H5AD_PATH, "\n")
sce <- readH5AD(H5AD_PATH)
cat("Loaded:", ncol(sce), "cells x", nrow(sce), "genes\n")
print(sce)

# ============================================================
# CELL 3: Prepare SCE for SC3
# ============================================================

# Use X as logcounts
assay(sce, "logcounts") <- assay(sce, "X")

# SC3 needs counts as a regular dense matrix
# ~2063 x 11037 ≈ 22M values, manageable in RAM
assay(sce, "counts") <- as.matrix(assay(sce, "counts"))

rowData(sce)$feature_symbol <- rownames(sce)
sce <- sce[!duplicated(rownames(sce)), ]

cat("After dedup:", ncol(sce), "cells x", nrow(sce), "genes\n")
cat("Assays available:", paste(assayNames(sce), collapse = ", "), "\n")
cat("counts class:", class(assay(sce, "counts")), "\n")
cat("counts dim:", dim(assay(sce, "counts")), "\n")

# Verify rowSums works (SC3 internal check)
test <- rowSums(assay(sce, "counts") == 0)
cat("rowSums test passed, length:", length(test), "\n")

# ============================================================
# CELL 4: Run SC3 with SVM mode
# ============================================================
set.seed(42)

sce <- sc3(
    sce,
    ks            = 6:12,
    biology       = FALSE,
    n_cores       = 8,
    svm_num_cells = 5000
)

cat("SC3 SVM mode complete\n")
print(head(colData(sce)))

cat("Training indices stored:",
    length(metadata(sce)$sc3$svm_train_inds), "\n")
cat("First few training indices:",
    head(metadata(sce)$sc3$svm_train_inds), "\n")
cat("Total cells:", ncol(sce), "\n")
cat("Cells to predict:",
    ncol(sce) - length(metadata(sce)$sc3$svm_train_inds), "\n")

# ============================================================
# CELL 4b: SVM prediction for remaining cells
# ============================================================
sce <- sc3_run_svm(sce, ks = 6:12)

cat("SVM prediction complete\n")

for (k in 6:12) {
    col   <- paste0("sc3_", k, "_clusters")
    n_na  <- sum(is.na(colData(sce)[[col]]))
    cat(sprintf("k=%d | NAs remaining: %d\n", k, n_na))
}

print(head(colData(sce)[, grep("sc3_", colnames(colData(sce)))]))

# ============================================================
# CELL 5: Export SC3 labels
# ============================================================
library(tibble)

sc3_cols  <- paste0("sc3_", 6:12, "_clusters")
labels_df <- as.data.frame(colData(sce)[, sc3_cols])
labels_df <- tibble::rownames_to_column(labels_df, var = "cell")

for (col in sc3_cols) {
    labels_df[[col]] <- as.integer(as.character(labels_df[[col]]))
}

write.csv(labels_df, OUT_CSV, row.names = FALSE)
cat("Exported", nrow(labels_df), "cells to", OUT_CSV, "\n")
cat("Columns:", paste(names(labels_df), collapse = ", "), "\n")
print(head(labels_df))
