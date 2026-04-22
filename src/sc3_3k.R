# ============================================================
# SC3 clustering for PBMC 3k (local version)
#
# Source notebook: SC3_3k_R.ipynb
# All Google Drive / Colab dependencies removed.
#
# Prerequisites (install once):
#   BiocManager::install(c("SC3", "SingleCellExperiment", "scater", "zellkonverter"))
#   install.packages(c("Matrix", "mclust", "cluster"))
#
# Input:  <repo_root>/pbmc3k_preprocessed.h5ad
# Output: <repo_root>/sc3_labels.csv
#         src/figures/sc3_3k/sc3_silhouette.png
# ============================================================

library(SC3)
library(SingleCellExperiment)
library(Matrix)
library(zellkonverter)
library(mclust)
library(cluster)

# ── Paths ────────────────────────────────────────────────────
# Reliably locate the script directory whether called via Rscript or source()
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

BASE_DIR   <- normalizePath(file.path(SCRIPT_DIR, ".."))
H5AD_PATH  <- file.path(BASE_DIR, "pbmc3k_preprocessed.h5ad")
OUT_CSV    <- file.path(BASE_DIR, "sc3_labels.csv")
FIG_DIR    <- file.path(SCRIPT_DIR, "figures", "sc3_3k")
dir.create(FIG_DIR, recursive = TRUE, showWarnings = FALSE)

cat("Reading:", H5AD_PATH, "\n")
sce <- readH5AD(H5AD_PATH)
cat("Loaded:", ncol(sce), "cells x", nrow(sce), "genes\n")
print(sce)

# ============================================================
# CELL 3: Prepare SCE for SC3
# ============================================================
cat("Available assays:", paste(assayNames(sce), collapse = ", "), "\n")

log_counts <- as.matrix(assay(sce, "X"))
raw_counts <- as.matrix(assay(sce, "counts"))

cat(paste("Matrix dimensions:",
          nrow(log_counts), "genes x", ncol(log_counts), "cells\n"))
cat(paste("Any NA:", any(is.na(log_counts)), "\n"))

sce_sc3 <- SingleCellExperiment(
    assays = list(
        counts    = raw_counts,
        logcounts = log_counts
    )
)
rownames(sce_sc3) <- rownames(sce)
colnames(sce_sc3) <- colnames(sce)

rowData(sce_sc3)$feature_symbol <- rownames(sce_sc3)
sce_sc3 <- sce_sc3[!duplicated(rowData(sce_sc3)$feature_symbol), ]

cat(paste("SC3 input:", nrow(sce_sc3), "genes x", ncol(sce_sc3), "cells\n"))
cat("Running SC3 for k = 3 to 8 (8 cores)...\n")
cat("This will take 20-40 minutes. Please wait.\n")

set.seed(42)
sce_sc3 <- sc3(sce_sc3, ks = 3:8, biology = FALSE, n_cores = 8)

cat("SC3 done.\n")
print(sce_sc3)

# ============================================================
# CELL 4: Evaluate SC3 results (silhouette + resampling ARI)
# ============================================================
k_range     <- 3:8
sc3_results <- list()

pca_embed <- reducedDim(sce, "X_pca")[, 1:20]

cat("SC3 clustering results:\n\n")
for (k in k_range) {
    col_name <- paste0("sc3_", k, "_clusters")
    labels   <- colData(sce_sc3)[[col_name]]
    sil      <- silhouette(as.integer(labels), dist(pca_embed))
    sil_score <- mean(sil[, 3])
    sc3_results[[as.character(k)]] <- list(
        labels     = labels,
        n_clusters = k,
        silhouette = sil_score
    )
    cat(paste0("  k=", k, " | Silhouette: ", round(sil_score, 4), "\n"))
}

sil_scores <- sapply(sc3_results, function(x) x$silhouette)
best_k     <- names(which.max(sil_scores))
cat(paste("\nBest k by silhouette:", best_k,
          "(", round(max(sil_scores), 4), ")\n"))

# ── Resampling stability for best k ──────────────────────────
cat(paste("\nRunning resampling stability for SC3 (k=", best_k, ")...\n"))

base_labels   <- as.integer(
    colData(sce_sc3)[[paste0("sc3_", best_k, "_clusters")]]
)
N_RUNS        <- 5
resample_aris <- c()

for (i in 1:N_RUNS) {
    set.seed(i)
    idx      <- sample(ncol(sce_sc3), floor(ncol(sce_sc3) * 0.8))
    sce_sub  <- sce_sc3[, idx]
    sce_sub  <- sc3(sce_sub, ks = as.integer(best_k),
                    biology = FALSE, n_cores = 8)
    sub_labels <- as.integer(
        colData(sce_sub)[[paste0("sc3_", best_k, "_clusters")]]
    )
    ari <- adjustedRandIndex(base_labels[idx], sub_labels)
    resample_aris <- c(resample_aris, ari)
    cat(paste0("  Run ", i, " | ARI: ", round(ari, 4), "\n"))
}

mean_ari <- round(mean(resample_aris), 4)
std_ari  <- round(sd(resample_aris), 4)
cat(paste("\nSC3 Resampling Stability:",
          mean_ari, "+/-", std_ari, "\n"))

# ── Summary table ─────────────────────────────────────────────
sc3_summary <- data.frame(
    k          = k_range,
    silhouette = sapply(k_range,
                        function(k) round(sc3_results[[as.character(k)]]$silhouette, 4))
)
cat("\nSC3 Summary Table:\n")
print(sc3_summary)

cat(paste("\nBest k:", best_k))
cat(paste("\nBest silhouette:", round(max(sil_scores), 4)))
cat(paste("\nResampling ARI:", mean_ari, "+/-", std_ari, "\n"))

# ── Silhouette plot ────────────────────────────────────────────
k_values <- k_range
sil_vals <- sapply(k_values,
                   function(k) sc3_results[[as.character(k)]]$silhouette)

png(file.path(FIG_DIR, "sc3_silhouette.png"),
    width = 700, height = 450)
par(mar = c(5, 5, 4, 2))
plot(k_values, sil_vals, type = "b", pch = 19,
     col = "steelblue", lwd = 2, cex = 1.5,
     xlab = "Number of clusters (k)",
     ylab = "Silhouette Score (higher is better)",
     main = "SC3: Silhouette Score across k values (PBMC 3k)",
     ylim = c(0, 0.4))
abline(v = as.integer(best_k), col = "red", lty = 2, lwd = 1.5)
text(as.integer(best_k), max(sil_vals) + 0.02,
     paste("Best k =", best_k), col = "red", cex = 0.9)
grid()
dev.off()
cat("Saved silhouette plot to", file.path(FIG_DIR, "sc3_silhouette.png"), "\n")

# ============================================================
# CELL 5: Export cluster labels (k=3 to 8)
# ============================================================
library(tibble)

sc3_cols  <- paste0("sc3_", 3:8, "_clusters")
labels_df <- as.data.frame(colData(sce_sc3)[, sc3_cols])
labels_df <- tibble::rownames_to_column(labels_df, var = "cell")

# Convert factors to integers and rename columns to sc3_k3 ... sc3_k8
for (col in sc3_cols) {
    labels_df[[col]] <- as.integer(as.character(labels_df[[col]]))
}

# Rename columns: sc3_3_clusters -> sc3_k3
colnames(labels_df) <- c(
    "cell",
    paste0("sc3_k", 3:8)
)

# Use cell barcode as row index (drop the "cell" column from index)
rownames(labels_df) <- labels_df$cell

write.csv(labels_df, OUT_CSV, row.names = TRUE)
cat("Exported", nrow(labels_df), "cells to", OUT_CSV, "\n")
cat("Columns:", paste(names(labels_df), collapse = ", "), "\n")
print(head(labels_df))
