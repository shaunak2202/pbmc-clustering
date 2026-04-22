"""
PBMC 3k clustering pipeline WITH MAGIC imputation.

Mirrors the MAGIC_502_New_3k notebook exactly, replacing all Google
Drive / Colab dependencies with local paths.

Run order:
  1. sc3_3k.R  (generates sc3_labels.csv at BASE_DIR)
  2. python pbmc_3k_magic.py

Outputs
-------
figures/magic/   – all plots
results/         – CSV summaries
BASE_DIR/pbmc3k_magic_preprocessed.h5ad
BASE_DIR/pbmc3k_magic_scvi.h5ad
"""

import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import scanpy as sc
import anndata
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path

# ── Reproducibility ──────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

# ── Scanpy settings ──────────────────────────────────────────
sc.settings.verbosity = 2
sc.settings.set_figure_params(dpi=100, facecolor="white", figsize=(6, 4))
warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
DATA_3K     = BASE_DIR / "hg19"
CKPT_MAGIC  = BASE_DIR / "pbmc3k_magic_preprocessed.h5ad"
SCVI_MAGIC  = BASE_DIR / "pbmc3k_magic_scvi.h5ad"
SC3_LABELS  = BASE_DIR / "sc3_labels.csv"
FIG_DIR     = Path(__file__).resolve().parent / "figures" / "magic"
RES_DIR     = Path(__file__).resolve().parent / "results"

FIG_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR.mkdir(parents=True, exist_ok=True)


def savefig(name: str) -> None:
    plt.savefig(FIG_DIR / name, dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================
# CELL 3: Load data (use preprocessed checkpoint if available)
# ============================================================
if CKPT_MAGIC.exists():
    print(f"Loading MAGIC preprocessed checkpoint: {CKPT_MAGIC}")
    adata = sc.read_h5ad(CKPT_MAGIC)
    print(f"Loaded: {adata.shape[0]} cells x {adata.shape[1]} genes")
    skip_preprocessing = True
else:
    print("Loading PBMC 3k raw data...")
    adata = sc.read_10x_mtx(str(DATA_3K), var_names="gene_symbols", cache=True)
    adata.var_names_make_unique()
    print(f"\nDataset shape: {adata.shape[0]} cells x {adata.shape[1]} genes")
    skip_preprocessing = False

# ============================================================
# CELL 4: Quality Control
# ============================================================
if not skip_preprocessing:
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
    )

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].hist(adata.obs["n_genes_by_counts"], bins=60, color="steelblue", edgecolor="white")
    axes[0].set_xlabel("Genes per cell")
    axes[0].set_title("Genes per Cell")
    axes[1].hist(adata.obs["total_counts"], bins=60, color="seagreen", edgecolor="white")
    axes[1].set_xlabel("Total counts")
    axes[1].set_title("Total Counts per Cell")
    axes[2].hist(adata.obs["pct_counts_mt"], bins=60, color="salmon", edgecolor="white")
    axes[2].set_xlabel("% MT counts")
    axes[2].set_title("Mitochondrial %")
    plt.suptitle("PBMC 3k - Pre-filter QC", fontsize=13, fontweight="bold")
    plt.tight_layout()
    savefig("qc_prefilter.png")

MIN_GENES  = 200
MAX_GENES  = 2500
MAX_MT_PCT = 5.0

if not skip_preprocessing:
    print(f"Before filtering: {adata.n_obs} cells")
    sc.pp.filter_cells(adata, min_genes=MIN_GENES)
    sc.pp.filter_cells(adata, max_genes=MAX_GENES)
    adata = adata[adata.obs["pct_counts_mt"] < MAX_MT_PCT].copy()
    sc.pp.filter_genes(adata, min_cells=3)
    print(f"After filtering:  {adata.n_obs} cells, {adata.n_vars} genes")

# ============================================================
# CELL 5: Preprocessing with MAGIC imputation
# ============================================================
if not skip_preprocessing:
    import magic

    # Save raw counts before normalization (needed for scVI later)
    adata.layers["counts"] = adata.X.copy()

    # 1. Normalize to 10,000 counts per cell
    sc.pp.normalize_total(adata, target_sum=1e4)

    # 2. Log1p transform
    sc.pp.log1p(adata)

    # 3. MAGIC imputation — conservative parameters to avoid over-smoothing
    print("Running MAGIC imputation...")
    magic_operator = magic.MAGIC(
        t=2,        # lighter than default t=3, reduces cross-boundary bleeding
        knn=5,      # small neighborhood, borrows from closest cells only
        decay=15,   # steep distance decay, far neighbors contribute very little
        random_state=SEED
    )
    adata.X = magic_operator.fit_transform(adata.X, genes="all_genes")
    print(f"MAGIC imputation complete. Data shape: {adata.shape}")

    # 4. Store normalized log counts (pre-HVG, post-MAGIC)
    adata.raw = adata

    # 5. Highly variable genes
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    print(f"Highly variable genes selected: {adata.var['highly_variable'].sum()}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, norm, title in zip(
        axes, [True, False], ["Normalized dispersion", "Raw dispersion"]
    ):
        disp_col = "dispersions_norm" if norm else "dispersions"
        ax.scatter(
            adata.var["means"][~adata.var["highly_variable"]],
            adata.var[disp_col][~adata.var["highly_variable"]],
            s=2, color="lightgray", label="other genes", alpha=0.7,
        )
        ax.scatter(
            adata.var["means"][adata.var["highly_variable"]],
            adata.var[disp_col][adata.var["highly_variable"]],
            s=4, color="#e8473f", label="highly variable", alpha=0.9,
        )
        ax.set_xlabel("Mean expression")
        ax.set_ylabel(title)
        ax.legend(markerscale=3, fontsize=9)
    plt.suptitle("Highly Variable Genes (MAGIC)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    savefig("hvg_magic.png")

    # 6. Filter to HVGs
    adata = adata[:, adata.var["highly_variable"]].copy()

    # 7. Scale
    sc.pp.scale(adata, max_value=10)

    # 8. PCA
    sc.tl.pca(adata, svd_solver="arpack", n_comps=50, random_state=SEED)

    pca_var = adata.uns["pca"]["variance_ratio"]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(1, 21), pca_var[:20], color="steelblue", edgecolor="white")
    ax.plot(range(1, 21), pca_var[:20], "o-", color="#e8473f", markersize=5)
    ax.set_xlabel("PC ranking")
    ax.set_ylabel("Variance ratio")
    ax.set_title("PCA Variance Ratio MAGIC (Top 20 PCs)")
    plt.tight_layout()
    savefig("pca_variance_magic.png")

    # 9. Neighbors + UMAP
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=20, random_state=SEED)
    sc.tl.umap(adata, random_state=SEED)

    print(f"\nAfter HVG selection: {adata.shape[0]} cells x {adata.shape[1]} genes")

    # Save checkpoint
    adata.write(str(CKPT_MAGIC))
    print(f"Checkpoint saved to {CKPT_MAGIC}")

# ============================================================
# CELL 6: Baseline 1 - K-means Clustering
# ============================================================
from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score, normalized_mutual_info_score)

X_pca = adata.obsm["X_pca"][:, :20]
k_range = range(3, 11)
kmeans_results = {}

print("Running K-means for k = 3 to 10...\n")
for k in k_range:
    km     = KMeans(n_clusters=k, random_state=SEED, n_init=10)
    labels = km.fit_predict(X_pca)
    sil    = silhouette_score(X_pca, labels)
    db     = davies_bouldin_score(X_pca, labels)
    ch     = calinski_harabasz_score(X_pca, labels)
    kmeans_results[k] = {"labels": labels, "silhouette": sil, "davies_bouldin": db, "calinski_harabasz": ch}
    print(f"  k={k} | Silhouette: {sil:.4f} | DB: {db:.4f} | CH: {ch:.1f}")

best_k = max(kmeans_results, key=lambda k: kmeans_results[k]["silhouette"])
print(f"\nBest k by silhouette: {best_k}")

adata.obs["kmeans_best"] = pd.Categorical(
    kmeans_results[best_k]["labels"].astype(str)
)

fig, axes = plt.subplots(1, 4, figsize=(20, 4))
plot_ks = [4, 6, 8, 10]
for ax, k in zip(axes, plot_ks):
    adata.obs[f"kmeans_{k}"] = pd.Categorical(
        kmeans_results[k]["labels"].astype(str)
    )
    sc.pl.umap(adata, color=f"kmeans_{k}", ax=ax, show=False,
               title=f"K-means k={k}", legend_loc="on data")
plt.suptitle("K-means Clustering on PBMC 3k (MAGIC)", fontsize=13, fontweight="bold")
plt.tight_layout()
savefig("kmeans_umap.png")

summary = pd.DataFrame({
    "k": list(k_range),
    "silhouette":     [kmeans_results[k]["silhouette"]     for k in k_range],
    "davies_bouldin": [kmeans_results[k]["davies_bouldin"] for k in k_range],
})
print("\nK-means Summary:")
print(summary.to_string(index=False))

# ============================================================
# CELL 7: Baseline 2 - Hierarchical Clustering
# ============================================================
from sklearn.cluster import AgglomerativeClustering

linkages          = ["ward", "average"]
n_clusters_range  = range(3, 11)
hier_results      = {}

print("Running Hierarchical Clustering...\n")
for linkage in linkages:
    hier_results[linkage] = {}
    for n in n_clusters_range:
        model  = AgglomerativeClustering(n_clusters=n, linkage=linkage)
        labels = model.fit_predict(X_pca)
        sil    = silhouette_score(X_pca, labels)
        db     = davies_bouldin_score(X_pca, labels)
        ch     = calinski_harabasz_score(X_pca, labels)
        hier_results[linkage][n] = {
            "labels": labels, "silhouette": sil, "davies_bouldin": db, "calinski_harabasz": ch
        }
        print(
            f"  linkage={linkage} | n={n} | "
            f"Silhouette: {sil:.4f} | DB: {db:.4f} | CH: {ch:.1f}"
        )
    print()

for linkage in linkages:
    best_n = max(
        hier_results[linkage],
        key=lambda n: hier_results[linkage][n]["silhouette"],
    )
    print(f"Best n for {linkage} linkage: {best_n}")
    adata.obs[f"hier_{linkage}_best"] = pd.Categorical(
        hier_results[linkage][best_n]["labels"].astype(str)
    )

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, linkage in zip(axes, linkages):
    best_n = max(
        hier_results[linkage],
        key=lambda n: hier_results[linkage][n]["silhouette"],
    )
    sc.pl.umap(adata, color=f"hier_{linkage}_best", ax=ax, show=False,
               title=f"Hierarchical ({linkage}, n={best_n})", legend_loc="on data")
plt.suptitle(
    "Hierarchical Clustering on PBMC 3k (MAGIC)", fontsize=13, fontweight="bold"
)
plt.tight_layout()
savefig("hierarchical_umap.png")

rows = []
for linkage in linkages:
    for n in n_clusters_range:
        rows.append({
            "linkage":       linkage,
            "n_clusters":    n,
            "silhouette":    hier_results[linkage][n]["silhouette"],
            "davies_bouldin":hier_results[linkage][n]["davies_bouldin"],
        })
print("\nHierarchical Clustering Summary:")
print(pd.DataFrame(rows).to_string(index=False))

# ============================================================
# CELL 8: Baseline 3 - DBSCAN and HDBSCAN
# ============================================================
from sklearn.cluster import DBSCAN
import hdbscan

X_umap = adata.obsm["X_umap"]

print("Running DBSCAN on UMAP embedding...\n")
eps_range      = [0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
min_samples    = 5
dbscan_results = {}

for eps in eps_range:
    model      = DBSCAN(eps=eps, min_samples=min_samples)
    labels     = model.fit_predict(X_umap)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_pct  = (labels == -1).sum() / len(labels) * 100
    if n_clusters > 1:
        mask = labels != -1
        sil  = silhouette_score(X_umap[mask], labels[mask])
        db   = davies_bouldin_score(X_umap[mask], labels[mask])
    else:
        sil, db = -1, -1
    dbscan_results[eps] = {
        "labels": labels, "n_clusters": n_clusters,
        "noise_pct": noise_pct, "silhouette": sil, "davies_bouldin": db,
    }
    print(
        f"  eps={eps} | clusters={n_clusters} | noise={noise_pct:.1f}% | "
        f"Silhouette: {sil:.4f} | DB: {db:.4f}"
    )

print("\nRunning HDBSCAN on PCA embedding...\n")
min_cluster_sizes = [10, 20, 30, 50, 75]
hdbscan_results   = {}

for mcs in min_cluster_sizes:
    clusterer  = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=5)
    labels     = clusterer.fit_predict(X_pca)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_pct  = (labels == -1).sum() / len(labels) * 100
    if n_clusters > 1:
        mask = labels != -1
        sil  = silhouette_score(X_pca[mask], labels[mask])
        db   = davies_bouldin_score(X_pca[mask], labels[mask])
    else:
        sil, db = -1, -1
    hdbscan_results[mcs] = {
        "labels": labels, "n_clusters": n_clusters,
        "noise_pct": noise_pct, "silhouette": sil, "davies_bouldin": db,
    }
    print(
        f"  min_cluster_size={mcs} | clusters={n_clusters} | "
        f"noise={noise_pct:.1f}% | Silhouette: {sil:.4f} | DB: {db:.4f}"
    )

valid_dbscan  = {e: v for e, v in dbscan_results.items()  if v["silhouette"] > 0}
valid_hdbscan = {m: v for m, v in hdbscan_results.items() if v["silhouette"] > 0}

best_eps = max(valid_dbscan,  key=lambda e: valid_dbscan[e]["silhouette"])
best_mcs = max(valid_hdbscan, key=lambda m: valid_hdbscan[m]["silhouette"])

print(
    f"\nBest DBSCAN eps: {best_eps} | "
    f"{valid_dbscan[best_eps]['n_clusters']} clusters | "
    f"noise={valid_dbscan[best_eps]['noise_pct']:.1f}%"
)
print(
    f"Best HDBSCAN min_cluster_size: {best_mcs} | "
    f"{valid_hdbscan[best_mcs]['n_clusters']} clusters | "
    f"noise={valid_hdbscan[best_mcs]['noise_pct']:.1f}%"
)

adata.obs["dbscan_best"]  = pd.Categorical(dbscan_results[best_eps]["labels"].astype(str))
adata.obs["hdbscan_best"] = pd.Categorical(hdbscan_results[best_mcs]["labels"].astype(str))

fig, axes = plt.subplots(1, 4, figsize=(22, 5))
for ax, eps in zip(axes[:2], [0.5, best_eps]):
    col = f"dbscan_eps_{eps}"
    adata.obs[col] = pd.Categorical(dbscan_results[eps]["labels"].astype(str))
    n   = dbscan_results[eps]["n_clusters"]
    noi = dbscan_results[eps]["noise_pct"]
    sil = dbscan_results[eps]["silhouette"]
    sc.pl.umap(adata, color=col, ax=ax, show=False, legend_loc="on data",
               title=f"DBSCAN eps={eps}\n{n} clusters | noise={noi:.1f}% | sil={sil:.2f}")

for ax, mcs in zip(axes[2:], [10, 50]):
    col = f"hdbscan_mcs_{mcs}"
    adata.obs[col] = pd.Categorical(hdbscan_results[mcs]["labels"].astype(str))
    n   = hdbscan_results[mcs]["n_clusters"]
    noi = hdbscan_results[mcs]["noise_pct"]
    sil = hdbscan_results[mcs]["silhouette"]
    sc.pl.umap(adata, color=col, ax=ax, show=False, legend_loc="on data",
               title=f"HDBSCAN min_size={mcs}\n{n} clusters | noise={noi:.1f}% | sil={sil:.2f}")

plt.suptitle(
    "Density-based Clustering on PBMC 3k (MAGIC)", fontsize=13, fontweight="bold"
)
plt.tight_layout()
savefig("dbscan_hdbscan_umap.png")

# ============================================================
# CELL 9: scRNA-seq Method 1 - Leiden Clustering
# ============================================================
resolutions    = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.5]
leiden_results = {}

print("Running Leiden across resolution sweep...\n")
for res in resolutions:
    sc.tl.leiden(adata, resolution=res, random_state=SEED,
                 key_added=f"leiden_{res}")
    labels     = adata.obs[f"leiden_{res}"].astype(int).values
    n_clusters = len(set(labels))
    sil        = silhouette_score(X_pca, labels)
    db         = davies_bouldin_score(X_pca, labels)
    ch         = calinski_harabasz_score(X_pca, labels)
    leiden_results[res] = {
        "n_clusters": n_clusters, "silhouette": sil, "davies_bouldin": db, "calinski_harabasz": ch
    }
    print(
        f"  res={res} | clusters={n_clusters} | "
        f"Silhouette: {sil:.4f} | DB: {db:.4f} | CH: {ch:.1f}"
    )

best_res = max(leiden_results, key=lambda r: leiden_results[r]["silhouette"])
print(
    f"\nBest resolution by silhouette: {best_res} "
    f"({leiden_results[best_res]['n_clusters']} clusters)"
)

plot_res = sorted(set([0.3, 0.5, 0.8, best_res]))
fig, axes = plt.subplots(1, len(plot_res), figsize=(6 * len(plot_res), 5))
if len(plot_res) == 1:
    axes = [axes]
for ax, res in zip(axes, plot_res):
    n = leiden_results[res]["n_clusters"]
    s = leiden_results[res]["silhouette"]
    sc.pl.umap(adata, color=f"leiden_{res}", ax=ax, show=False,
               legend_loc="on data",
               title=f"Leiden res={res}\n{n} clusters | sil={s:.2f}")
plt.suptitle("Leiden Clustering on PBMC 3k (MAGIC)", fontsize=13, fontweight="bold")
plt.tight_layout()
savefig("leiden_umap.png")

fig, ax1 = plt.subplots(figsize=(8, 4))
ax2 = ax1.twinx()
ax1.plot(resolutions,
         [leiden_results[r]["n_clusters"] for r in resolutions],
         "o-", color="steelblue", label="n_clusters")
ax2.plot(resolutions,
         [leiden_results[r]["silhouette"] for r in resolutions],
         "s--", color="#e8473f", label="silhouette")
ax1.set_xlabel("Resolution")
ax1.set_ylabel("Number of clusters", color="steelblue")
ax2.set_ylabel("Silhouette score", color="#e8473f")
ax1.set_title("Leiden: Resolution vs Cluster Count and Silhouette")
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")
plt.tight_layout()
savefig("leiden_resolution_sweep.png")

summary = pd.DataFrame({
    "resolution":    resolutions,
    "n_clusters":    [leiden_results[r]["n_clusters"]    for r in resolutions],
    "silhouette":    [leiden_results[r]["silhouette"]    for r in resolutions],
    "davies_bouldin":[leiden_results[r]["davies_bouldin"]for r in resolutions],
})
print("\nLeiden Summary:")
print(summary.to_string(index=False))

# ============================================================
# CELL 10: scRNA-seq Method 2 - scVI + Leiden
# ============================================================
import scvi

scvi_resolutions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]

if SCVI_MAGIC.exists():
    print(f"Loading scVI checkpoint: {SCVI_MAGIC}")
    adata_scvi = sc.read_h5ad(SCVI_MAGIC)
else:
    # scVI needs raw counts
    adata_scvi = adata.copy()
    adata_scvi.X = adata_scvi.layers["counts"]

    scvi.model.SCVI.setup_anndata(adata_scvi)
    model_scvi = scvi.model.SCVI(adata_scvi, n_layers=2, n_latent=30)
    print("Training scVI model...\n")
    model_scvi.train(max_epochs=200, early_stopping=True)

    adata_scvi.obsm["X_scVI"] = model_scvi.get_latent_representation()
    print(f"\nscVI latent space shape: {adata_scvi.obsm['X_scVI'].shape}")

    sc.pp.neighbors(adata_scvi, use_rep="X_scVI", n_neighbors=15, random_state=SEED)
    sc.tl.umap(adata_scvi, random_state=SEED)

    for res in scvi_resolutions:
        sc.tl.leiden(adata_scvi, resolution=res, random_state=SEED,
                     key_added=f"scvi_leiden_{res}")

    adata_scvi.write(str(SCVI_MAGIC))
    print(f"\nSaved scVI results to {SCVI_MAGIC}")

X_scvi               = adata_scvi.obsm["X_scVI"]
scvi_leiden_results  = {}

print("\nRunning Leiden on scVI latent space...\n")
for res in scvi_resolutions:
    key = f"scvi_leiden_{res}"
    if key not in adata_scvi.obs.columns:
        sc.tl.leiden(adata_scvi, resolution=res, random_state=SEED, key_added=key)
    labels     = adata_scvi.obs[key].astype(int).values
    n_clusters = len(set(labels))
    sil        = silhouette_score(X_scvi, labels)
    db         = davies_bouldin_score(X_scvi, labels)
    ch         = calinski_harabasz_score(X_scvi, labels)
    scvi_leiden_results[res] = {
        "n_clusters": n_clusters, "silhouette": sil, "davies_bouldin": db, "calinski_harabasz": ch
    }
    print(
        f"  res={res} | clusters={n_clusters} | "
        f"Silhouette: {sil:.4f} | DB: {db:.4f}"
    )

best_scvi_res = max(
    scvi_leiden_results, key=lambda r: scvi_leiden_results[r]["silhouette"]
)
print(
    f"\nBest resolution by silhouette: {best_scvi_res} "
    f"({scvi_leiden_results[best_scvi_res]['n_clusters']} clusters)"
)

plot_res = sorted(set([0.3, 0.5, 0.8, best_scvi_res]))
fig, axes = plt.subplots(1, len(plot_res), figsize=(6 * len(plot_res), 5))
if len(plot_res) == 1:
    axes = [axes]
for ax, res in zip(axes, plot_res):
    n = scvi_leiden_results[res]["n_clusters"]
    s = scvi_leiden_results[res]["silhouette"]
    sc.pl.umap(adata_scvi, color=f"scvi_leiden_{res}", ax=ax, show=False,
               legend_loc="on data",
               title=f"scVI+Leiden res={res}\n{n} clusters | sil={s:.2f}")
plt.suptitle(
    "scVI + Leiden Clustering on PBMC 3k (MAGIC)", fontsize=13, fontweight="bold"
)
plt.tight_layout()
savefig("scvi_leiden_umap.png")

fig, ax1 = plt.subplots(figsize=(8, 4))
ax2 = ax1.twinx()
ax1.plot(scvi_resolutions,
         [scvi_leiden_results[r]["n_clusters"] for r in scvi_resolutions],
         "o-", color="steelblue", label="n_clusters")
ax2.plot(scvi_resolutions,
         [scvi_leiden_results[r]["silhouette"] for r in scvi_resolutions],
         "s--", color="#e8473f", label="silhouette")
ax1.set_xlabel("Resolution")
ax1.set_ylabel("Number of clusters", color="steelblue")
ax2.set_ylabel("Silhouette score", color="#e8473f")
ax1.set_title("scVI+Leiden: Resolution vs Cluster Count and Silhouette")
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")
plt.tight_layout()
savefig("scvi_leiden_resolution_sweep.png")

# ============================================================
# CELL 11: Stability Analysis (MAGIC best parameters)
# ============================================================
from sklearn.metrics import adjusted_rand_score
from sklearn.utils import resample as sklearn_resample
from matplotlib.patches import Patch

X_pca  = adata.obsm["X_pca"][:, :20]
X_umap = adata.obsm["X_umap"]

N_RUNS         = 10
SUBSAMPLE_FRAC = 0.8
seeds          = list(range(N_RUNS))

# MAGIC best parameters
BEST_K_KMEANS   = 7    # was 4 in baseline
BEST_N_HIER     = 7    # was 5 in baseline
BEST_EPS_DBSCAN = 0.8  # was 0.5 in baseline
BEST_RES_LEIDEN = 0.2  # was 0.3 in baseline
BEST_RES_SCVI   = 0.1  # was 0.6 in baseline

stability_results = {}


def pairwise_ari(labels_list):
    aris = []
    for i in range(len(labels_list)):
        for j in range(i + 1, len(labels_list)):
            aris.append(adjusted_rand_score(labels_list[i], labels_list[j]))
    return np.mean(aris), np.std(aris)


# 1. K-means seed stability (k=7)
print(f"K-means seed stability (k={BEST_K_KMEANS})...")
km_labels = []
for s in seeds:
    km = KMeans(n_clusters=BEST_K_KMEANS, random_state=s, n_init=10)
    km_labels.append(km.fit_predict(X_pca))
mean_ari, std_ari = pairwise_ari(km_labels)
stability_results["KMeans_seed"] = {"mean_ARI": mean_ari, "std_ARI": std_ari}
print(f"  Mean ARI: {mean_ari:.4f} +/- {std_ari:.4f}")

# 2. K-means resampling stability (k=7)
print(f"K-means resampling stability (k={BEST_K_KMEANS})...")
base_km_labels = KMeans(
    n_clusters=BEST_K_KMEANS, random_state=SEED, n_init=10
).fit_predict(X_pca)
resample_aris = []
for s in seeds:
    np.random.seed(s)
    idx        = np.random.choice(len(X_pca), int(len(X_pca) * SUBSAMPLE_FRAC), replace=False)
    labels_sub = KMeans(
        n_clusters=BEST_K_KMEANS, random_state=s, n_init=10
    ).fit_predict(X_pca[idx])
    resample_aris.append(adjusted_rand_score(base_km_labels[idx], labels_sub))
stability_results["KMeans_resample"] = {
    "mean_ARI": np.mean(resample_aris), "std_ARI": np.std(resample_aris)
}
print(f"  Mean ARI: {np.mean(resample_aris):.4f} +/- {np.std(resample_aris):.4f}")

# 3. Hierarchical Ward stability (n=7)
print(f"Hierarchical Ward stability (n={BEST_N_HIER})...")
base_hier = AgglomerativeClustering(
    n_clusters=BEST_N_HIER, linkage="ward"
).fit_predict(X_pca)
hier_aris = []
for s in seeds:
    np.random.seed(s)
    idx        = np.random.choice(len(X_pca), int(len(X_pca) * SUBSAMPLE_FRAC), replace=False)
    labels_sub = AgglomerativeClustering(
        n_clusters=BEST_N_HIER, linkage="ward"
    ).fit_predict(X_pca[idx])
    hier_aris.append(adjusted_rand_score(base_hier[idx], labels_sub))
stability_results["Hierarchical_seed"]     = {"mean_ARI": 1.0, "std_ARI": 0.0}
stability_results["Hierarchical_resample"] = {
    "mean_ARI": np.mean(hier_aris), "std_ARI": np.std(hier_aris)
}
print(f"  Seed ARI: 1.0 (deterministic)")
print(f"  Resample Mean ARI: {np.mean(hier_aris):.4f} +/- {np.std(hier_aris):.4f}")

# 4. DBSCAN resampling stability (eps=0.8)
print(f"DBSCAN resampling stability (eps={BEST_EPS_DBSCAN})...")
base_dbscan = DBSCAN(eps=BEST_EPS_DBSCAN, min_samples=5).fit_predict(X_umap)
dbscan_aris = []
for s in seeds:
    np.random.seed(s)
    idx        = np.random.choice(len(X_umap), int(len(X_umap) * SUBSAMPLE_FRAC), replace=False)
    labels_sub = DBSCAN(eps=BEST_EPS_DBSCAN, min_samples=5).fit_predict(X_umap[idx])
    mask       = labels_sub != -1
    if mask.sum() > 10:
        dbscan_aris.append(
            adjusted_rand_score(base_dbscan[idx][mask], labels_sub[mask])
        )
stability_results["DBSCAN_seed"]     = {"mean_ARI": 1.0, "std_ARI": 0.0}
stability_results["DBSCAN_resample"] = {
    "mean_ARI": np.mean(dbscan_aris), "std_ARI": np.std(dbscan_aris)
}
print(f"  Seed ARI: 1.0 (deterministic)")
print(f"  Resample Mean ARI: {np.mean(dbscan_aris):.4f} +/- {np.std(dbscan_aris):.4f}")

# 5. Leiden seed stability (res=0.2)
print(f"Leiden seed stability (res={BEST_RES_LEIDEN})...")
leiden_seed_labels = []
for s in seeds:
    sc.tl.leiden(adata, resolution=BEST_RES_LEIDEN, random_state=s,
                 key_added=f"leiden_seed_{s}")
    leiden_seed_labels.append(adata.obs[f"leiden_seed_{s}"].astype(int).values)
mean_ari, std_ari = pairwise_ari(leiden_seed_labels)
stability_results["Leiden_seed"] = {"mean_ARI": mean_ari, "std_ARI": std_ari}
print(f"  Mean ARI: {mean_ari:.4f} +/- {std_ari:.4f}")

# Leiden resampling stability
print(f"Leiden resampling stability (res={BEST_RES_LEIDEN})...")
base_leiden = adata.obs[f"leiden_{BEST_RES_LEIDEN}"].astype(int).values
leiden_resample_aris = []
for s in seeds:
    np.random.seed(s)
    idx       = np.random.choice(adata.n_obs, int(adata.n_obs * SUBSAMPLE_FRAC), replace=False)
    adata_sub = adata[idx].copy()
    sc.pp.neighbors(adata_sub, n_neighbors=15, n_pcs=20, random_state=s)
    sc.tl.leiden(adata_sub, resolution=BEST_RES_LEIDEN, random_state=s,
                 key_added="leiden_sub")
    leiden_resample_aris.append(
        adjusted_rand_score(base_leiden[idx],
                            adata_sub.obs["leiden_sub"].astype(int).values)
    )
stability_results["Leiden_resample"] = {
    "mean_ARI": np.mean(leiden_resample_aris),
    "std_ARI":  np.std(leiden_resample_aris),
}
print(
    f"  Mean ARI: {np.mean(leiden_resample_aris):.4f} "
    f"+/- {np.std(leiden_resample_aris):.4f}"
)

# 6. scVI+Leiden stability (res=0.1)
print(f"scVI+Leiden seed stability (res={BEST_RES_SCVI})...")

if f"scvi_leiden_{BEST_RES_SCVI}" not in adata_scvi.obs.columns:
    sc.tl.leiden(adata_scvi, resolution=BEST_RES_SCVI, random_state=SEED,
                 key_added=f"scvi_leiden_{BEST_RES_SCVI}")

sc.pp.neighbors(adata_scvi, use_rep="X_scVI", n_neighbors=15,
                key_added="scvi_neighbors", random_state=SEED)

ref_scvi  = adata_scvi.obs[f"scvi_leiden_{BEST_RES_SCVI}"].astype(int).values
X_scvi_3k = adata_scvi.obsm["X_scVI"]
N_scvi    = len(X_scvi_3k)

scvi_seed_aris = []
for s in seeds:
    sc.tl.leiden(adata_scvi, resolution=BEST_RES_SCVI, random_state=s,
                 neighbors_key="scvi_neighbors",
                 key_added=f"scvi_seed_{s}")
    ari = adjusted_rand_score(
        ref_scvi, adata_scvi.obs[f"scvi_seed_{s}"].astype(int).values
    )
    scvi_seed_aris.append(ari)
    print(f"  Seed {s}: ARI={ari:.4f}")
stability_results["scVI_Leiden_seed"] = {
    "mean_ARI": np.mean(scvi_seed_aris), "std_ARI": np.std(scvi_seed_aris)
}
print(
    f"  Seed Mean ARI: {np.mean(scvi_seed_aris):.4f} "
    f"+/- {np.std(scvi_seed_aris):.4f}"
)

print(f"scVI+Leiden resampling stability (res={BEST_RES_SCVI})...")
scvi_resample_aris = []
for i in range(10):
    idx   = sklearn_resample(
        np.arange(N_scvi), n_samples=int(0.8 * N_scvi), random_state=i, replace=False
    )
    X_sub     = X_scvi_3k[idx]
    adata_sub = anndata.AnnData(X=X_sub)
    sc.pp.neighbors(adata_sub, n_neighbors=15, use_rep="X", random_state=SEED)
    sc.tl.leiden(adata_sub, resolution=BEST_RES_SCVI, random_state=SEED,
                 key_added="leiden_sub")
    ari = adjusted_rand_score(
        ref_scvi[idx], adata_sub.obs["leiden_sub"].astype(int).values
    )
    scvi_resample_aris.append(ari)
    print(f"  Resample run {i+1}/10: ARI={ari:.4f}")
stability_results["scVI_Leiden_resample"] = {
    "mean_ARI": np.mean(scvi_resample_aris),
    "std_ARI":  np.std(scvi_resample_aris),
}
print(
    f"  Resample Mean ARI: {np.mean(scvi_resample_aris):.4f} "
    f"+/- {np.std(scvi_resample_aris):.4f}"
)
print(f"  Stable: {'YES' if np.mean(scvi_resample_aris) >= 0.8 else 'NO'}")

summary_df = pd.DataFrame(stability_results).T.reset_index()
summary_df.columns = ["method", "mean_ARI", "std_ARI"]
print("\nStability Summary (MAGIC):")
print(summary_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 7))
colors = [
    "steelblue" if "seed" in m.lower() else "#e8473f"
    for m in summary_df["method"]
]
ax.barh(summary_df["method"], summary_df["mean_ARI"],
        xerr=summary_df["std_ARI"], color=colors,
        edgecolor="white", capsize=4, alpha=0.85)
ax.axvline(x=0.8, color="gray", linestyle="--", linewidth=1.5)
ax.set_xlabel("Mean ARI")
ax.set_title(
    "Stability Analysis - Seed and Resampling (PBMC 3k MAGIC)",
    fontsize=13, fontweight="bold",
)
ax.set_xlim(0, 1.1)
legend_elements = [
    Patch(facecolor="steelblue", label="Seed ARI"),
    Patch(facecolor="#e8473f",   label="Resample ARI"),
]
ax.legend(handles=legend_elements, loc="lower right")
plt.tight_layout()
savefig("stability_summary_magic.png")

# ============================================================
# CELL 12: Biological Validation (MAGIC)
# ============================================================
# Re-compute best labels with MAGIC best parameters
km_best = KMeans(n_clusters=7, random_state=SEED, n_init=10)
adata.obs["kmeans_best"] = pd.Categorical(km_best.fit_predict(X_pca).astype(str))

hw_best = AgglomerativeClustering(n_clusters=7, linkage="ward")
adata.obs["hier_ward_best"] = pd.Categorical(hw_best.fit_predict(X_pca).astype(str))

db_best = DBSCAN(eps=0.8, min_samples=5)
adata.obs["dbscan_best"] = pd.Categorical(db_best.fit_predict(X_umap).astype(str))

adata.obs["leiden_best"] = adata.obs["leiden_0.2"].copy()

adata.obs["scvi_leiden_best"] = pd.Categorical(
    adata_scvi.obs["scvi_leiden_0.1"].values.astype(str)
)

print("All best labels stored:")
for key in ["kmeans_best", "hier_ward_best", "dbscan_best",
            "leiden_best", "scvi_leiden_best"]:
    print(f"  {key}: {adata.obs[key].nunique()} clusters")

MARKERS = {
    "T cells":         ["CD3D", "CD3E", "CD3G"],
    "CD4+ T cells":    ["CD4", "IL7R"],
    "CD8+ T cells":    ["CD8A", "CD8B"],
    "NK cells":        ["NKG7", "GZMA", "PRF1", "GNLY"],
    "B cells":         ["CD79A", "CD79B", "MS4A1"],
    "CD14+ Monocytes": ["CD14", "LYZ", "CST3"],
    "CD16+ Monocytes": ["FCGR3A", "MS4A7"],
    "Dendritic cells": ["FCER1A", "CST3", "HLA-DQA1"],
    "Platelets":       ["PF4", "PPBP"],
    "pDC":             ["IL3RA", "GZMB", "SERPINF1"],
}

methods = {
    "K-means":     "kmeans_best",
    "Hier. Ward":  "hier_ward_best",
    "DBSCAN":      "dbscan_best",
    "Leiden PCA":  "leiden_best",
    "scVI+Leiden": "scvi_leiden_best",
}


def validate_method(adata_in, label_key, method_name):
    adata_in.obs[label_key] = adata_in.obs[label_key].astype(str)
    sc.tl.rank_genes_groups(
        adata_in, groupby=label_key,
        method="wilcoxon", key_added=f"rank_{label_key}",
        pts=True,
    )
    n_clusters = adata_in.obs[label_key].nunique()
    detected   = set()
    print(f"\n{'='*55}")
    print(f"{method_name} ({n_clusters} clusters)")
    print(f"{'='*55}")
    for cluster in sorted(adata_in.obs[label_key].unique()):
        top_genes = sc.get.rank_genes_groups_df(
            adata_in, group=cluster, key=f"rank_{label_key}"
        ).head(20)["names"].tolist()
        matched = []
        for cell_type, markers in MARKERS.items():
            hits = [m for m in markers if m in top_genes]
            if hits:
                matched.append(f"{cell_type}({','.join(hits)})")
                detected.add(cell_type)
        annotation = " | ".join(matched) if matched else "Unknown"
        print(f"  Cluster {cluster}: {annotation}")
    score = len(detected)
    print(f"\n  Detected cell types: {score}/{len(MARKERS)}")
    print(f"  {', '.join(sorted(detected))}")
    return score, detected


validation_scores   = {}
validation_detected = {}

for method_name, label_key in methods.items():
    score, detected = validate_method(adata, label_key, method_name)
    validation_scores[method_name]   = score
    validation_detected[method_name] = detected

print(f"\n{'='*55}")
print("Biological Validation Summary - PBMC 3k (MAGIC)")
print(f"{'='*55}")
print(f"{'Method':<15} {'Score':>8} {'Cell Types Detected'}")
print("-" * 55)
for m, score in validation_scores.items():
    print(
        f"{m:<15} {score:>5}/{len(MARKERS)}  "
        f"{', '.join(sorted(validation_detected[m]))}"
    )

fig, ax = plt.subplots(figsize=(12, 5))
methods_list = list(validation_scores.keys())
scores_list  = [validation_scores[m] for m in methods_list]
colors_list  = ["steelblue", "seagreen", "salmon", "#e8473f", "mediumpurple"]
bars = ax.bar(methods_list, scores_list, color=colors_list,
              edgecolor="white", width=0.5)
ax.axhline(len(MARKERS), color="gray", linestyle="--", linewidth=1,
           label=f"Max ({len(MARKERS)} types)")
ax.set_ylabel("Cell types detected")
ax.set_title(
    "Biological Validation - PBMC 3k (MAGIC)", fontsize=13, fontweight="bold"
)
ax.set_ylim(0, len(MARKERS) + 1)
ax.legend()
for bar, score in zip(bars, scores_list):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
            str(score), ha="center", va="bottom", fontweight="bold")
plt.tight_layout()
savefig("biological_validation_3k_magic.png")

val_df = pd.DataFrame({
    "method":   list(validation_scores.keys()),
    "score":    list(validation_scores.values()),
    "detected": [", ".join(sorted(v)) for v in validation_detected.values()],
})
val_df.to_csv(RES_DIR / "biological_validation_3k_magic.csv", index=False)
print(f"Saved validation summary to {RES_DIR}/biological_validation_3k_magic.csv")

# ============================================================
# CELL 15: Load SC3 labels and full integration
# ============================================================

if SC3_LABELS.exists():
    print(f"\nLoading SC3 labels from {SC3_LABELS}...")
    sc3_df = pd.read_csv(SC3_LABELS, index_col=0)

    print(f"SC3 df columns: {sc3_df.columns.tolist()}")
    overlap = set(adata.obs_names) & set(sc3_df.index)
    print(f"Matching barcodes: {len(overlap)} out of {adata.n_obs}")

    adata.obs["sc3_k4"] = sc3_df.loc[adata.obs_names, "sc3_k4"].astype(str)
    adata.obs["sc3_k4"] = pd.Categorical(adata.obs["sc3_k4"])
    print(f"\nSC3 k=4 cluster distribution:")
    print(adata.obs["sc3_k4"].value_counts().sort_index())

    # ============================================================
    # CELL 16: SC3 UMAP, marker genes and comparison
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sc.pl.umap(adata, color="leiden_0.3", ax=axes[0], show=False,
               legend_loc="on data", title="Leiden res=0.3 (6 clusters)")
    sc.pl.umap(adata, color="sc3_k4", ax=axes[1], show=False,
               legend_loc="on data", title="SC3 k=4 (4 clusters)")
    plt.suptitle("Leiden vs SC3 - PBMC 3k", fontsize=13, fontweight="bold")
    plt.tight_layout()
    savefig("sc3_vs_leiden_umap.png")

    known_markers = {
        "T cells":   ["CD3D", "CD3E", "IL7R", "CD4", "CD8A"],
        "NK cells":  ["GNLY", "NKG7", "GZMB", "GZMA", "PRF1"],
        "B cells":   ["MS4A1", "CD79A", "CD79B", "CD19", "BANK1"],
        "Monocytes": ["CD14", "LYZ", "CST3", "FCGR3A", "MS4A7"],
        "Dendritic": ["FCER1A", "CST3", "IL3RA", "CLEC4C"],
        "Platelets": ["PPBP", "PF4", "GP1BB"],
    }

    adata_markers_sc3 = adata.raw.to_adata()
    adata_markers_sc3.obs["sc3_k4"] = adata.obs["sc3_k4"]

    sc.tl.rank_genes_groups(
        adata_markers_sc3, groupby="sc3_k4",
        method="wilcoxon", key_added="rank_genes_sc3", n_genes=20,
    )

    print("SC3 k=4 marker genes per cluster:\n")
    sc3_annotations = {}
    sc3_recovered   = []

    for cluster in sorted(
        adata_markers_sc3.obs["sc3_k4"].unique(), key=lambda x: int(x)
    ):
        genes = sc.get.rank_genes_groups_df(
            adata_markers_sc3, group=cluster, key="rank_genes_sc3"
        )["names"].tolist()[:10]
        print(f"Cluster {cluster}: {', '.join(genes[:10])}")
        print(f"  Matches:")
        matched = False
        for cell_type, markers in known_markers.items():
            hits = [g for g in genes if g in markers]
            if hits:
                print(f"    {cell_type}: {hits}")
                if cell_type not in sc3_recovered:
                    sc3_recovered.append(cell_type)
                if not matched:
                    sc3_annotations[cluster] = cell_type
                    matched = True
        if not matched:
            sc3_annotations[cluster] = "Unknown"
        print()

    print(f"SC3 recovered cell types: {sc3_recovered}")
    print(f"SC3 biological validation: {len(sc3_recovered)}/6")

    adata.obs["sc3_cell_type"] = adata.obs["sc3_k4"].map(
        {k: f"{v} ({k})" for k, v in sc3_annotations.items()}
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sc.pl.umap(adata, color="sc3_k4", ax=axes[0], show=False,
               legend_loc="on data", title="SC3 k=4 (cluster numbers)")
    sc.pl.umap(adata, color="sc3_cell_type", ax=axes[1], show=False,
               legend_loc="on data", title="SC3 k=4 (cell type annotations)")
    plt.suptitle("SC3 Clustering - PBMC 3k", fontsize=13, fontweight="bold")
    plt.tight_layout()
    savefig("sc3_annotations_umap.png")

    marker_genes_flat = [
        "CD3D", "IL7R", "CD14", "LYZ", "MS4A1",
        "CD79A", "GNLY", "NKG7", "FCGR3A", "PPBP",
    ]
    sc.pl.dotplot(adata_markers_sc3, marker_genes_flat,
                  groupby="sc3_k4",
                  title="Known PBMC Markers across SC3 Clusters")
    savefig("sc3_dotplot.png")

    # Biological validation heatmap (all methods + SC3)
    all_cell_types = ["T cells", "NK cells", "B cells",
                      "Monocytes", "Platelets", "Dendritic"]

    method_validation_updated = {
        "K-means (k=4)":         ["NK cells", "Platelets", "Monocytes", "Dendritic"],
        "Hier. Ward (n=5)":      ["Monocytes", "Platelets", "NK cells", "B cells", "Dendritic", "T cells"],
        "DBSCAN (eps=0.5)":      ["Monocytes", "Platelets", "B cells", "Dendritic", "T cells"],
        "HDBSCAN (min=10)":      ["B cells", "Monocytes", "Dendritic", "T cells"],
        "Leiden PCA (res=0.3)":  ["Monocytes", "Platelets", "NK cells", "B cells", "Dendritic", "T cells"],
        "scVI+Leiden (res=0.6)": ["Monocytes", "Platelets", "NK cells", "B cells", "Dendritic", "T cells"],
        "SC3 (k=4)":             sc3_recovered,
    }

    methods_list_hm = list(method_validation_updated.keys())
    matrix = np.zeros((len(methods_list_hm), len(all_cell_types)))
    for i, (method, recovered) in enumerate(method_validation_updated.items()):
        for j, ct in enumerate(all_cell_types):
            matrix[i, j] = 1 if ct in recovered else 0

    fig, ax = plt.subplots(figsize=(13, 7))
    ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(all_cell_types)))
    ax.set_xticklabels(all_cell_types, fontsize=11, fontweight="bold")
    ax.set_yticks(range(len(methods_list_hm)))
    ax.set_yticklabels(methods_list_hm, fontsize=10)
    for i in range(len(methods_list_hm)):
        for j in range(len(all_cell_types)):
            text  = "Yes" if matrix[i, j] == 1 else "No"
            color = "white" if matrix[i, j] == 0 else "black"
            ax.text(j, i, text, ha="center", va="center",
                    fontsize=11, fontweight="bold", color=color)
    ax.set_title(
        "Biological Validation - All Methods including SC3\n(PBMC 3k)",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlabel("Cell Type", fontsize=11)
    ax.set_ylabel("Clustering Method", fontsize=11)
    plt.tight_layout()
    savefig("biological_validation_heatmap_final.png")

    print("\nSC3 cluster to cell type mapping:")
    for k, v in sc3_annotations.items():
        print(f"  Cluster {k}: {v}")

    # ============================================================
    # CELL 17: SC3 k=6 - check if Platelets are recovered
    # ============================================================
    adata.obs["sc3_k6"] = sc3_df.loc[adata.obs_names, "sc3_k6"].astype(str)
    adata.obs["sc3_k6"] = pd.Categorical(adata.obs["sc3_k6"])
    print(f"SC3 k=6 cluster distribution:")
    print(adata.obs["sc3_k6"].value_counts().sort_index())

    adata_markers_sc3_k6 = adata.raw.to_adata()
    adata_markers_sc3_k6.obs["sc3_k6"] = adata.obs["sc3_k6"]

    sc.tl.rank_genes_groups(
        adata_markers_sc3_k6, groupby="sc3_k6",
        method="wilcoxon", key_added="rank_genes_sc3_k6", n_genes=20,
    )

    print("\nSC3 k=6 marker genes per cluster:\n")
    sc3_k6_annotations = {}
    sc3_k6_recovered   = []

    for cluster in sorted(
        adata_markers_sc3_k6.obs["sc3_k6"].unique(), key=lambda x: int(x)
    ):
        genes = sc.get.rank_genes_groups_df(
            adata_markers_sc3_k6, group=cluster, key="rank_genes_sc3_k6"
        )["names"].tolist()[:10]
        print(f"Cluster {cluster}: {', '.join(genes[:10])}")
        print(f"  Matches:")
        matched = False
        for cell_type, markers in known_markers.items():
            hits = [g for g in genes if g in markers]
            if hits:
                print(f"    {cell_type}: {hits}")
                if cell_type not in sc3_k6_recovered:
                    sc3_k6_recovered.append(cell_type)
                if not matched:
                    sc3_k6_annotations[cluster] = cell_type
                    matched = True
        if not matched:
            sc3_k6_annotations[cluster] = "Unknown"
        print()

    print(f"SC3 k=6 recovered cell types: {sc3_k6_recovered}")
    print(f"SC3 k=6 biological validation: {len(sc3_k6_recovered)}/6")

    adata.obs["sc3_k6_cell_type"] = adata.obs["sc3_k6"].map(
        {k: f"{v} ({k})" for k, v in sc3_k6_annotations.items()}
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sc.pl.umap(adata, color="sc3_k6", ax=axes[0], show=False,
               legend_loc="on data", title="SC3 k=6 (cluster numbers)")
    sc.pl.umap(adata, color="sc3_k6_cell_type", ax=axes[1], show=False,
               legend_loc="on data", title="SC3 k=6 (cell type annotations)")
    plt.suptitle("SC3 k=6 Clustering - PBMC 3k", fontsize=13, fontweight="bold")
    plt.tight_layout()
    savefig("sc3_k6_umap.png")

    print("\nSC3 k=6 cluster to cell type mapping:")
    for k, v in sc3_k6_annotations.items():
        print(f"  Cluster {k}: {v}")
else:
    print(
        f"\nSC3 labels not found at {SC3_LABELS}. "
        "Run sc3_3k.R first to generate them."
    )

# ============================================================
# EXTRA PRESENTATION FIGURES
# ============================================================
print("\n── Generating extra presentation figures ──")
import matplotlib.patches as mpatches

# 1. UMAP colored by QC metrics
if "n_genes_by_counts" in adata.obs.columns:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, col, label in zip(
        axes,
        ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
        ["Genes per cell", "Total UMI counts", "% Mitochondrial"],
    ):
        sc.pl.umap(adata, color=col, ax=ax, show=False,
                   color_map="viridis", title=label, vmin=0)
    plt.suptitle("UMAP Colored by QC Metrics — PBMC 3k (MAGIC)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    savefig("umap_qc_overlay.png")

# 2. Post-filter QC violin
if "n_genes_by_counts" in adata.obs.columns:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, col, title, color in zip(
        axes,
        ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
        ["Genes per cell", "Total counts", "% Mitochondrial"],
        ["steelblue", "seagreen", "salmon"],
    ):
        parts = ax.violinplot(adata.obs[col].values, showmedians=True, showextrema=True)
        for pc in parts["bodies"]:
            pc.set_facecolor(color); pc.set_alpha(0.7)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel(title, fontsize=10)
        ax.set_xticks([])
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.suptitle("PBMC 3k (MAGIC) — Post-filter Cell Quality", fontsize=13, fontweight="bold")
    plt.tight_layout()
    savefig("qc_postfilter_violin.png")

# 3. PCA scatter colored by best Leiden and scVI
_cc_m = f"leiden_{best_res}" if f"leiden_{best_res}" in adata.obs.columns else "leiden_0.2"
_sc_m = "scvi_leiden_best"   if "scvi_leiden_best"   in adata.obs.columns else _cc_m
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sc.pl.pca(adata, color=_cc_m, ax=axes[0], show=False,
          title=f"PCA — Leiden (res={best_res})")
sc.pl.pca(adata, color=_sc_m, ax=axes[1], show=False,
          title="PCA — scVI+Leiden")
plt.suptitle("PCA Space — PBMC 3k (MAGIC)", fontsize=13, fontweight="bold")
plt.tight_layout()
savefig("pca_scatter.png")

# 4. K-means silhouette + DB sweep
_ks_m   = list(k_range)
_sils_m = [kmeans_results[k]["silhouette"]     for k in _ks_m]
_dbs_m  = [kmeans_results[k]["davies_bouldin"] for k in _ks_m]
_ck_m   = ["#e8473f" if k == best_k else "steelblue" for k in _ks_m]
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
bars1 = axes[0].bar(_ks_m, _sils_m, color=_ck_m, edgecolor="white", width=0.6)
axes[0].set_xlabel("k (clusters)"); axes[0].set_ylabel("Silhouette Score (higher = better)")
axes[0].set_title("K-means: Silhouette Score vs k", fontsize=12, fontweight="bold")
axes[0].set_xticks(_ks_m)
for bar, v in zip(bars1, _sils_m):
    axes[0].text(bar.get_x() + bar.get_width()/2, v + 0.003, f"{v:.3f}",
                 ha="center", va="bottom", fontsize=8, fontweight="bold")
axes[0].spines["top"].set_visible(False); axes[0].spines["right"].set_visible(False)
bars2 = axes[1].bar(_ks_m, _dbs_m, color=_ck_m, edgecolor="white", width=0.6)
axes[1].set_xlabel("k (clusters)"); axes[1].set_ylabel("Davies-Bouldin Index (lower = better)")
axes[1].set_title("K-means: Davies-Bouldin Index vs k", fontsize=12, fontweight="bold")
axes[1].set_xticks(_ks_m)
for bar, v in zip(bars2, _dbs_m):
    axes[1].text(bar.get_x() + bar.get_width()/2, v + 0.01, f"{v:.3f}",
                 ha="center", va="bottom", fontsize=8, fontweight="bold")
axes[1].spines["top"].set_visible(False); axes[1].spines["right"].set_visible(False)
plt.suptitle("K-means Metric Sweep — PBMC 3k (MAGIC)", fontsize=13, fontweight="bold")
plt.tight_layout()
savefig("kmeans_metrics_sweep.png")

# 5. K-means all-k UMAP grid
_ncols_km = 4
_nrows_km = int(np.ceil(len(_ks_m) / _ncols_km))
fig, axes = plt.subplots(_nrows_km, _ncols_km, figsize=(5 * _ncols_km, 4.5 * _nrows_km))
axes = np.array(axes).flatten()
for _im, _km in enumerate(k_range):
    _col_m = f"kmeans_{_km}"
    if _col_m not in adata.obs.columns:
        adata.obs[_col_m] = pd.Categorical(kmeans_results[_km]["labels"].astype(str))
    sc.pl.umap(adata, color=_col_m, ax=axes[_im], show=False, legend_loc="on data",
               title=f"k={_km}  sil={kmeans_results[_km]['silhouette']:.3f}")
for _jm in range(_im + 1, len(axes)):
    axes[_jm].axis("off")
plt.suptitle("K-means: All k Values — PBMC 3k (MAGIC)", fontsize=14, fontweight="bold")
plt.tight_layout()
savefig("kmeans_all_k_grid.png")

# 6. Hierarchical silhouette sweep (Ward vs Average)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
_ns_m = list(n_clusters_range)
for ax, linkage, color in zip(axes, ["ward", "average"], ["steelblue", "#e8473f"]):
    _sils_hm = [hier_results[linkage][n]["silhouette"] for n in _ns_m]
    _best_nm = max(hier_results[linkage], key=lambda n: hier_results[linkage][n]["silhouette"])
    ax.plot(_ns_m, _sils_hm, "o-", color=color, linewidth=2, markersize=7)
    ax.axvline(_best_nm, color="gray", linestyle="--", linewidth=1.2,
               label=f"Best n={_best_nm}")
    for n, s in zip(_ns_m, _sils_hm):
        ax.text(n, s + 0.003, f"{s:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_xlabel("n clusters"); ax.set_ylabel("Silhouette Score")
    ax.set_title(f"Hierarchical ({linkage}) — Best n={_best_nm}",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(_ns_m); ax.legend()
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.suptitle("Hierarchical Silhouette Sweep — PBMC 3k (MAGIC)", fontsize=13, fontweight="bold")
plt.tight_layout()
savefig("hierarchical_metrics_sweep.png")

# 7. All-methods UMAP panel (2 × 3 grid)
_best_k_m   = best_k
_best_res_m = best_res
_best_scvi_m = best_scvi_res
_best_eps_m  = best_eps
_best_mcs_m  = best_mcs
_best_n_w_m  = max(hier_results["ward"], key=lambda n: hier_results["ward"][n]["silhouette"])

_panel_m = [
    ("kmeans_best",          f"K-means (k={_best_k_m})"),
    ("hier_ward_best",       f"Hierarchical Ward (n={_best_n_w_m})"),
    ("dbscan_best",          f"DBSCAN (eps={_best_eps_m})"),
    ("hdbscan_best",         f"HDBSCAN (mcs={_best_mcs_m})"),
    (f"leiden_{_best_res_m}",f"Leiden PCA (res={_best_res_m})"),
    ("scvi_leiden_best",     f"scVI+Leiden (res={_best_scvi_m})"),
]
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
axes = axes.flatten()
for ax, (key, title) in zip(axes, _panel_m):
    if key in adata.obs.columns:
        sc.pl.umap(adata, color=key, ax=ax, show=False, legend_loc="on data", title=title)
    else:
        ax.axis("off")
plt.suptitle("All Clustering Methods — Best Result (PBMC 3k MAGIC)",
             fontsize=14, fontweight="bold")
plt.tight_layout()
savefig("all_methods_umap_panel.png")

# 8. Leiden all-resolutions grid
_res_list_m = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.5]
_ncols_lm   = 5
_nrows_lm   = int(np.ceil(len(_res_list_m) / _ncols_lm))
fig, axes = plt.subplots(_nrows_lm, _ncols_lm, figsize=(5 * _ncols_lm, 4.5 * _nrows_lm))
axes = np.array(axes).flatten()
for _ilm, _res in enumerate(_res_list_m):
    _key_r = f"leiden_{_res}"
    if _key_r in adata.obs.columns:
        sc.pl.umap(adata, color=_key_r, ax=axes[_ilm], show=False, legend_loc="on data",
                   title=f"res={_res}  k={leiden_results[_res]['n_clusters']}  "
                         f"sil={leiden_results[_res]['silhouette']:.3f}")
    else:
        axes[_ilm].axis("off")
for _jlm in range(_ilm + 1, len(axes)):
    axes[_jlm].axis("off")
plt.suptitle("Leiden: Full Resolution Grid — PBMC 3k (MAGIC)", fontsize=14, fontweight="bold")
plt.tight_layout()
savefig("leiden_all_resolutions_grid.png")

# 9. MAGIC smooth gene expression on UMAP (shows imputation effect)
_magic_genes = [g for g in ["CD3D", "LYZ", "MS4A1", "GNLY", "PPBP", "CD14"]
                if g in adata.var_names or (adata.raw is not None and g in adata.raw.var_names)]
if _magic_genes:
    _ncols_ge = 3
    _nrows_ge = int(np.ceil(len(_magic_genes) / _ncols_ge))
    fig, axes = plt.subplots(_nrows_ge, _ncols_ge,
                             figsize=(6 * _ncols_ge, 5 * _nrows_ge))
    axes = np.array(axes).flatten()
    for _ig, gene in enumerate(_magic_genes):
        sc.pl.umap(adata, color=gene, use_raw=True, ax=axes[_ig], show=False,
                   color_map="RdPu", title=f"{gene} (MAGIC-imputed)")
    for _jg in range(_ig + 1, len(axes)):
        axes[_jg].axis("off")
    plt.suptitle("Marker Gene Expression after MAGIC Imputation — PBMC 3k",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    savefig("magic_gene_expression_umap.png")

# 10. Heatmap and violin of key marker genes
try:
    _adata_hm_m = adata.raw.to_adata()
    _adata_hm_m.obs["cluster"] = adata.obs[f"leiden_{_best_res_m}"].values
    sc.tl.rank_genes_groups(_adata_hm_m, groupby="cluster", method="wilcoxon",
                             key_added="rank_hm_m", n_genes=10)
    _mkeys_m = ["CD3D", "IL7R", "CD14", "LYZ", "MS4A1",
                "CD79A", "GNLY", "NKG7", "FCGR3A", "PPBP"]
    _mkeys_m = [g for g in _mkeys_m if g in _adata_hm_m.var_names]
    if _mkeys_m:
        sc.pl.heatmap(_adata_hm_m, var_names=_mkeys_m, groupby="cluster",
                      show_gene_labels=True, cmap="RdBu_r", figsize=(10, 6), show=False)
        plt.suptitle("Marker Gene Heatmap — PBMC 3k (MAGIC)", fontsize=13, fontweight="bold")
        savefig("heatmap_markers.png")
        sc.pl.violin(_adata_hm_m, keys=_mkeys_m[:5], groupby="cluster",
                     rotation=45, stripplot=False, figsize=(14, 4), show=False)
        plt.suptitle("Marker Gene Expression per Cluster — PBMC 3k (MAGIC)",
                     fontsize=13, fontweight="bold")
        savefig("violin_key_markers.png")
except Exception as e:
    print(f"  Heatmap/Violin skipped: {e}")

# 11. Biological validation heatmap (all methods)
_all_ct_m   = list(MARKERS.keys())
_methods_vm = list(validation_detected.keys())
_matrix_vm  = np.zeros((len(_methods_vm), len(_all_ct_m)))
for _iv, m in enumerate(_methods_vm):
    for _jv, ct in enumerate(_all_ct_m):
        _matrix_vm[_iv, _jv] = 1 if ct in validation_detected[m] else 0
fig, ax = plt.subplots(figsize=(16, 6))
ax.imshow(_matrix_vm, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
ax.set_xticks(range(len(_all_ct_m)))
ax.set_xticklabels(_all_ct_m, fontsize=9, fontweight="bold", rotation=30, ha="right")
ax.set_yticks(range(len(_methods_vm)))
ax.set_yticklabels(_methods_vm, fontsize=11)
for _iv in range(len(_methods_vm)):
    for _jv in range(len(_all_ct_m)):
        ax.text(_jv, _iv, "Y" if _matrix_vm[_iv, _jv] == 1 else "N",
                ha="center", va="center", fontsize=10, fontweight="bold",
                color="white" if _matrix_vm[_iv, _jv] == 0 else "black")
ax.set_title("Biological Validation — All Methods (PBMC 3k MAGIC)",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Cell Type"); ax.set_ylabel("Method")
plt.tight_layout()
savefig("biological_validation_heatmap.png")

# 12. MAGIC vs baseline best parameters comparison bar chart
_methods_comp = ["K-means", "Hier. Ward", "DBSCAN", "Leiden", "scVI+Leiden"]
_sil_base = [0.3308, 0.3278, 0.5660, 0.3052, 0.1054]
_sil_magic = [
    kmeans_results[_best_k_m]["silhouette"],
    hier_results["ward"][_best_n_w_m]["silhouette"],
    dbscan_results.get(_best_eps_m, {}).get("silhouette", 0.0),
    leiden_results.get(_best_res_m, {}).get("silhouette", 0.0),
    scvi_leiden_results.get(_best_scvi_m, {}).get("silhouette", 0.0),
]
_x_comp = np.arange(len(_methods_comp))
_w_comp  = 0.35
fig, ax = plt.subplots(figsize=(12, 6))
b1 = ax.bar(_x_comp - _w_comp/2, _sil_base,  _w_comp,
            label="Baseline (no MAGIC)", color="steelblue",  alpha=0.85)
b2 = ax.bar(_x_comp + _w_comp/2, _sil_magic, _w_comp,
            label="MAGIC imputed",       color="#e8473f", alpha=0.85)
ax.set_xticks(_x_comp)
ax.set_xticklabels(_methods_comp, rotation=15, ha="right", fontsize=11)
ax.set_ylabel("Silhouette Score (higher = better)", fontsize=11)
ax.set_title("MAGIC vs Baseline: Silhouette Score Comparison",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=11)
ax.set_ylim(0, 0.75)
for bar, v in zip(b1, _sil_base):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.01, f"{v:.3f}",
            ha="center", va="bottom", fontsize=8)
for bar, v in zip(b2, _sil_magic):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.01, f"{v:.3f}",
            ha="center", va="bottom", fontsize=8)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
savefig("magic_vs_baseline_silhouette.png")

# NMI between SC3 and Leiden (MAGIC)
if SC3_LABELS.exists() and "sc3_k4" in adata.obs.columns:
    _leiden_cols_m = [c for c in adata.obs.columns
                      if c.startswith("leiden_") and not c.startswith("leiden_seed")]
    _sc3_cols_m    = [c for c in adata.obs.columns if c.startswith("sc3_k")]
    nmi_rows_m = []
    for lc in _leiden_cols_m:
        for sc3c in _sc3_cols_m:
            nmi = normalized_mutual_info_score(
                adata.obs[lc].astype(str), adata.obs[sc3c].astype(str))
            nmi_rows_m.append({"leiden": lc, "sc3": sc3c, "NMI": round(nmi, 4)})
    if nmi_rows_m:
        nmi_df_m = pd.DataFrame(nmi_rows_m)
        nmi_pivot_m = nmi_df_m.pivot(index="leiden", columns="sc3", values="NMI")
        fig, ax = plt.subplots(figsize=(10, 7))
        im = ax.imshow(nmi_pivot_m.values, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(len(nmi_pivot_m.columns)))
        ax.set_xticklabels(nmi_pivot_m.columns, fontsize=10)
        ax.set_yticks(range(len(nmi_pivot_m.index)))
        ax.set_yticklabels(nmi_pivot_m.index, fontsize=9)
        plt.colorbar(im, ax=ax, label="NMI")
        for i in range(len(nmi_pivot_m.index)):
            for j in range(len(nmi_pivot_m.columns)):
                ax.text(j, i, f"{nmi_pivot_m.values[i,j]:.2f}",
                        ha="center", va="center", fontsize=8)
        ax.set_title("NMI: Leiden vs SC3 — PBMC 3k (MAGIC)", fontsize=13, fontweight="bold")
        ax.set_xlabel("SC3 k"); ax.set_ylabel("Leiden resolution")
        plt.tight_layout()
        savefig("nmi_leiden_vs_sc3_magic.png")
        nmi_df_m.to_csv(RES_DIR / "nmi_leiden_vs_sc3_magic.csv", index=False)

# Calinski-Harabász comparison
_ch_methods_m = ["K-means", "Hier. Ward", "Leiden PCA", "scVI+Leiden"]
_best_n_ward_m = max(hier_results["ward"], key=lambda n: hier_results["ward"][n]["silhouette"])
_ch_vals_m = [
    kmeans_results[_best_k_m]["calinski_harabasz"],
    hier_results["ward"][_best_n_ward_m]["calinski_harabasz"],
    leiden_results[_best_res_m]["calinski_harabasz"],
    scvi_leiden_results[_best_scvi_m]["calinski_harabasz"],
]
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(_ch_methods_m, _ch_vals_m,
              color=["#4e79a7", "#4e79a7", "#59a14f", "#e15759"],
              edgecolor="white", width=0.5)
for bar, v in zip(bars, _ch_vals_m):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            f"{v:.0f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_ylabel("Calinski-Harabász Score (higher = better)", fontsize=11)
ax.set_title("Calinski-Harabász Score by Method — PBMC 3k (MAGIC)", fontsize=13, fontweight="bold")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
savefig("calinski_harabasz_magic.png")

# PAGA trajectory
try:
    _paga_key_m = f"leiden_{_best_res_m}" if f"leiden_{_best_res_m}" in adata.obs.columns else "leiden_0.2"
    sc.tl.paga(adata, groups=_paga_key_m)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sc.pl.paga(adata, ax=axes[0], show=False, fontsize=10,
               title=f"PAGA — connectivity graph (Leiden res={_best_res_m})")
    adata.obsm["X_umap_orig_m"] = adata.obsm["X_umap"].copy()
    sc.tl.umap(adata, init_pos="paga", random_state=SEED)
    sc.pl.umap(adata, color=_paga_key_m, ax=axes[1], show=False, legend_loc="on data",
               title=f"PAGA-initialized UMAP (Leiden res={_best_res_m})")
    adata.obsm["X_umap"] = adata.obsm["X_umap_orig_m"]
    plt.suptitle("PAGA Trajectory — PBMC 3k (MAGIC)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    savefig("paga_magic.png")
except Exception as e:
    print(f"  PAGA skipped: {e}")

# Standardized summary CSV
_std_magic = pd.DataFrame({
    "dataset":   "PBMC 3k MAGIC",
    "method":    ["K-means", "Hier. Ward", "DBSCAN", "HDBSCAN", "Leiden PCA", "scVI+Leiden"],
    "n_clusters":[kmeans_results[_best_k_m]["labels"].max() + 1,
                  hier_results["ward"][_best_n_ward_m]["labels"].max() + 1,
                  dbscan_results[_best_eps_m]["n_clusters"],
                  hdbscan_results[_best_mcs_m]["n_clusters"],
                  leiden_results[_best_res_m]["n_clusters"],
                  scvi_leiden_results[_best_scvi_m]["n_clusters"]],
    "silhouette":  _sil_magic,
    "calinski_harabasz": [_ch_vals_m[0], _ch_vals_m[1], np.nan, np.nan,
                          _ch_vals_m[2], _ch_vals_m[3]],
})
_std_magic.to_csv(RES_DIR / "summary_magic.csv", index=False)

print("\nDone. Figures saved to", FIG_DIR)
print("Results saved to", RES_DIR)
