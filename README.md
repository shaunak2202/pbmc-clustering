# PBMC Clustering Pipeline

A systematic comparison of clustering algorithms on human blood cell data from single-cell RNA sequencing (scRNA-seq). Seven methods are tested across two dataset sizes, evaluated on cluster quality, stability, and biological accuracy.

---

## What This Project Does

Single-cell RNA sequencing measures gene activity in thousands of individual cells at once. A key step in the analysis is **clustering** — grouping cells that behave similarly so they can be identified as known cell types (like T cells, B cells, or NK cells).

This project asks: **which clustering algorithm does the best job?**

Seven methods are compared side by side on two publicly available datasets of human peripheral blood mononuclear cells (PBMCs) — one with ~2,700 cells and one with ~11,000 cells. Every method is tested with the same preprocessing and scored on the same metrics, so the comparison is fair.

---

## Methods Compared

| Method | Type | What it does |
|--------|------|--------------|
| **K-means** | Partitional | Assigns cells to k groups by minimizing distance to cluster centers |
| **Hierarchical (Ward)** | Agglomerative | Builds a tree of cells by merging closest groups bottom-up |
| **DBSCAN** | Density-based | Finds clusters as dense regions, marks sparse cells as noise |
| **HDBSCAN** | Density-based | Like DBSCAN but automatically finds cluster density thresholds |
| **Leiden** | Graph-based | Detects communities in a cell-neighbor graph (standard in scRNA-seq) |
| **scVI + Leiden** | Deep generative | Learns a compressed representation with a neural network, then clusters |
| **SC3** | Consensus (R) | Combines multiple clustering runs via Bioconductor's SC3 package |

---

## Datasets

Both datasets come from [10x Genomics](https://www.10xgenomics.com/resources/datasets) and contain PBMCs (white blood cells) from a healthy human donor.

| Dataset | Cells | Genes measured |
|---------|-------|---------------|
| PBMC 3k | ~2,700 | ~33,000 |
| PBMC 10k | ~11,000 | ~33,000 |

Raw data is not included in this repo (files are several hundred MB). Download instructions are in the [Setup](#setup) section.

---

## Key Results

### Biological Accuracy — How many known cell types does each method find?

**PBMC 3k** (6 known cell types: T cells, NK cells, B cells, Monocytes, Platelets, Dendritic):

| Method | Cell types recovered |
|--------|---------------------|
| K-means | 4 / 6 |
| Hierarchical Ward | **6 / 6** |
| DBSCAN | 5 / 6 |
| HDBSCAN | 4 / 6 |
| Leiden | **6 / 6** |
| scVI + Leiden | **6 / 6** |

**PBMC 10k** (10 known cell types):

| Method | Cell types recovered |
|--------|---------------------|
| K-means | 8 / 10 |
| Hierarchical Ward | 8 / 10 |
| DBSCAN | 7 / 10 |
| HDBSCAN | 8 / 10 |
| Leiden | 8 / 10 |
| scVI + Leiden | **10 / 10** |
| SC3 | 7 / 10 |

scVI + Leiden is the only method to recover all 10 cell types on the larger dataset, including rare subtypes like CD8+ T cells and both monocyte subtypes.

### Cluster Quality (PBMC 10k, best configuration per method)

Higher silhouette = better-separated clusters. Lower Davies-Bouldin = less overlap.

| Method | Silhouette | Davies-Bouldin |
|--------|-----------|----------------|
| K-means (k=6) | 0.52 | 0.71 |
| Hierarchical Ward (n=6) | 0.51 | 0.68 |
| HDBSCAN | **0.55** | 0.59 |
| DBSCAN | 0.37 | **0.51** |
| Leiden (res=0.1) | 0.37 | 1.18 |
| scVI + Leiden | 0.20 | 1.50 |

Note: scVI scores low on internal metrics because its latent space is not Euclidean — but it recovers the most cell types biologically.

### Stability — Do results change with different random seeds or subsampled data?

All methods are highly stable on the 10k dataset. On the smaller 3k dataset, K-means shows meaningful variation across seeds (ARI = 0.74 ± 0.24), while all other methods score above 0.86.

### Agreement Between Methods

Leiden (res=0.4) and SC3 (k=4) on the 3k dataset agree with **NMI = 0.91** — two very different algorithmic approaches arriving at nearly the same answer, which is strong evidence the clusters reflect real biology.

---

## Project Structure

```
pbmc-clustering/
├── run_pipeline.sh              # Run everything with one command
├── src/
│   ├── pbmc_3k.py               # 3k dataset: preprocessing + all clustering
│   ├── pbmc_3k_magic.py         # 3k dataset with MAGIC imputation (ablation)
│   ├── pbmc_10k.py              # 10k dataset + combined cross-dataset results
│   ├── sc3_3k.R                 # SC3 clustering for 3k (optional, needs R)
│   ├── sc3_10k.R                # SC3 clustering for 10k with SVM scaling
│   ├── requirements.txt         # Python dependencies
│   ├── figures/                 # All generated plots (created when pipeline runs)
│   └── results/                 # CSV metric summaries (created when pipeline runs)
├── sc3_labels.csv               # Pre-computed SC3 labels for 3k (R not required)
├── sc3_labels_10k.csv           # Pre-computed SC3 labels for 10k
└── README.md
```

The `sc3_labels*.csv` files are committed to the repo so you can run the full comparison — including SC3 — without installing R.

---

## Setup

### 1. Get the data

Download the two PBMC datasets from 10x Genomics and place them in the project root:

- **PBMC 3k**: download the "Gene / Cell Matrix (filtered)" — extract into a folder named `hg19/`
- **PBMC 10k**: download the "Feature / Barcode Matrix (filtered)" — extract into `filtered_feature_bc_matrix/`

Your folder structure should look like:
```
pbmc-clustering/
├── hg19/
│   ├── barcodes.tsv
│   ├── genes.tsv
│   └── matrix.mtx
└── filtered_feature_bc_matrix/
    ├── barcodes.tsv.gz
    ├── features.tsv.gz
    └── matrix.mtx.gz
```

> If you have pre-computed `.h5ad` checkpoint files from a previous run (e.g., from Google Colab), place them in the project root. The pipeline will detect and use them automatically, skipping expensive recomputation.

### 2. Install Python

This project requires **Python 3.11**. Python 3.12+ is not yet supported by all dependencies.

```bash
# macOS (using Homebrew)
brew install python@3.11

# Create a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r src/requirements.txt
pip install "jax[cpu]"
```

### 3. Install R (optional)

R is only needed to re-run the SC3 clustering. Pre-computed SC3 labels are already included in the repo, so **you can skip this step** and still get full SC3 comparison results.

```bash
# macOS
brew install r
```

Then in an R console:
```r
install.packages(c("BiocManager", "Matrix", "mclust", "cluster", "tibble"))
BiocManager::install(c("SC3", "SingleCellExperiment", "zellkonverter", "scater"))
```

---

## Running the Pipeline

```bash
source .venv/bin/activate
bash run_pipeline.sh
```

The pipeline runs in 7 steps. If R is not installed, the SC3 steps are automatically skipped (pre-computed labels are used instead). If checkpoint `.h5ad` files exist, heavy preprocessing is skipped.

### Expected runtimes

| Situation | Estimated time |
|-----------|----------------|
| Fresh run, no GPU | 3–5 hours |
| Checkpoints exist, no GPU | 30–60 minutes |
| Checkpoints exist, GPU available | ~15 minutes |

---

## Outputs

After the pipeline finishes, all results are in `src/`:

### Figures (`src/figures/`)
- `3k/` — ~34 plots for the 3k baseline: UMAPs, metric sweeps, marker gene heatmaps, stability charts, PAGA trajectory, NMI heatmap
- `magic/` — same set of plots after MAGIC imputation (ablation study)
- `10k/` — ~31 plots for the 10k dataset, plus cross-dataset comparison charts

### Results (`src/results/`)
| File | What it contains |
|------|-----------------|
| `results_combined.csv` | All methods × both datasets: silhouette, Davies-Bouldin, Calinski-Harabász |
| `summary_3k.csv` | Internal metrics for PBMC 3k |
| `summary_10k.csv` | Internal metrics for PBMC 10k |
| `biological_validation_summary.csv` | Which cell types each method recovers (3k) |
| `biological_validation_10k.csv` | Same for 10k |
| `nmi_leiden_vs_sc3_3k.csv` | NMI scores between Leiden and SC3 across all parameter combos (3k) |
| `nmi_leiden_vs_sc3_10k.csv` | Same for 10k |

---

## What is MAGIC?

MAGIC (Markov Affinity-based Graph Imputation of Cells) is a method that smooths out the noise in scRNA-seq data by sharing information between similar cells. This project uses it as an ablation study — running the same pipeline on the 3k dataset before and after MAGIC imputation to see whether smoothing improves or changes the clustering. Conservative parameters are used (t=2, knn=5) to avoid over-smoothing.

---

## Dependencies

Core Python packages: `scanpy`, `scvi-tools`, `leidenalg`, `igraph`, `hdbscan`, `umap-learn`, `magic-impute`, `scikit-learn`, `numpy`, `pandas`, `matplotlib`, `seaborn`

Full list with pinned versions: `src/requirements.txt`
