# PBMC Clustering Pipeline

Systematic comparison of six clustering algorithms on single-cell RNA-seq data (PBMC 3k and 10k datasets), with MAGIC imputation as an ablation and SC3 as a Bioconductor baseline.

## Methods compared

| Method | Category |
|--------|----------|
| K-means | Partitional |
| Hierarchical Ward | Agglomerative |
| DBSCAN | Density-based |
| HDBSCAN | Density-based |
| Leiden (PCA graph) | Graph-based |
| scVI + Leiden | Deep generative |
| SC3 (R) | Consensus (Bioconductor) |

## Evaluation

- **Internal**: Silhouette score, Davies-Bouldin index, Calinski-Harabász score
- **Stability**: Seed ARI (10 seeds), resampling ARI (80% subsampling × 10 runs)
- **Biological**: Wilcoxon marker gene recovery against known PBMC cell types
- **Cross-method**: NMI heatmap (Leiden vs SC3 across all k / resolution combinations)
- **Trajectory**: PAGA connectivity graph + PAGA-initialized UMAP

## Datasets

| Dataset | Cells | Source |
|---------|-------|--------|
| PBMC 3k | ~2,700 | 10x Genomics (hg19/) |
| PBMC 10k | ~11,000 | 10x Genomics (filtered_feature_bc_matrix/) |

Raw data is not tracked in git (too large). Download from [10x Genomics](https://www.10xgenomics.com/resources/datasets).

## Project structure

```
502/
├── run_pipeline.sh          # single command to run everything
├── src/
│   ├── pbmc_3k.py           # 3k baseline pipeline
│   ├── pbmc_3k_magic.py     # 3k + MAGIC imputation pipeline
│   ├── pbmc_10k.py          # 10k pipeline + combined results
│   ├── sc3_3k.R             # SC3 clustering for 3k
│   ├── sc3_10k.R            # SC3 + SVM mode for 10k
│   ├── requirements.txt     # Python dependencies
│   ├── figures/             # all plots (generated)
│   └── results/             # CSV summaries (generated)
├── sc3_labels.csv           # SC3 labels for 3k (committed so R is optional)
└── sc3_labels_10k.csv       # SC3 labels for 10k
```

## Setup

**Python** (requires 3.11; 3.14 is not yet supported by all packages):

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r src/requirements.txt
pip install "jax[cpu]"
```

**R** (optional — SC3 labels are pre-committed so you can skip R):

```bash
brew install r   # macOS
```

```r
install.packages(c("BiocManager", "Matrix", "mclust", "cluster", "tibble"))
BiocManager::install(c("SC3", "SingleCellExperiment", "zellkonverter", "scater"))
```

## Running

```bash
source .venv/bin/activate
bash run_pipeline.sh
```

The pipeline has 7 steps. Each Python script checks for a preprocessed `.h5ad` checkpoint and skips expensive steps if one exists. Expected runtimes:

| Condition | Estimated time |
|-----------|----------------|
| Fresh run (no checkpoints, no GPU) | 3–5 hours |
| Checkpoints exist, no GPU | 30–60 minutes |
| Checkpoints exist, GPU | ~15 minutes |

## Outputs

- `src/figures/3k/` — ~25 figures for the 3k baseline pipeline
- `src/figures/magic/` — ~25 figures for the MAGIC pipeline
- `src/figures/10k/` — ~25 figures for the 10k pipeline
- `src/figures/sc3_3k/` — SC3 silhouette plot
- `src/results/results_combined.csv` — unified metrics table across all datasets and methods
- `src/results/nmi_leiden_vs_sc3_*.csv` — NMI matrices
