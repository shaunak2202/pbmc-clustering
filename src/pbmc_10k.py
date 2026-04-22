"""
PBMC 10k Clustering Pipeline
Full local version — no Google Colab or Google Drive required.

Run order:
  1. python pbmc_10k.py
  2. Rscript sc3_10k.R          (produces sc3_labels_10k.csv in project root)
  3. python pbmc_10k.py          (re-run; picks up SC3 labels automatically)

Dependencies (install once):
  pip install scanpy==1.10.1 scvi-tools leidenalg==0.10.2 igraph==0.11.4
              umap-learn==0.5.6 hdbscan scikit-learn matplotlib seaborn
              pandas numpy anndata
"""

import os
import shutil
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # headless — remove this line for interactive plots
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import anndata
import scanpy as sc
import scvi
import hdbscan
from pathlib import Path
from collections import Counter
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             adjusted_rand_score, calinski_harabasz_score,
                             normalized_mutual_info_score)
from sklearn.utils import resample

os.environ["JAX_PLATFORM_NAME"] = "cpu"

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).resolve().parent.parent
DATA_10K_RAW    = BASE_DIR / "filtered_feature_bc_matrix"
DATA_10K_STAGED = Path(__file__).resolve().parent / "data" / "pbmc10k"
CKPT_10K        = BASE_DIR / "pbmc10k_preprocessed.h5ad"
SCVI_10K        = BASE_DIR / "pbmc10k_scvi.h5ad"
SCVI_MODEL_DIR  = BASE_DIR / "pbmc10k_scvi_model"
SC3_LABELS_10K  = BASE_DIR / "sc3_labels_10k.csv"
FIG_DIR         = Path(__file__).resolve().parent / "figures" / "10k"
RES_DIR         = Path(__file__).resolve().parent / "results"

FIG_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
sc.settings.verbosity = 2
sc.settings.set_figure_params(dpi=100, facecolor="white", figsize=(6, 4))
warnings.filterwarnings("ignore")


def savefig(name):
    plt.savefig(FIG_DIR / name, dpi=150, bbox_inches="tight")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: Load data (use preprocessed checkpoint if available)
# ═══════════════════════════════════════════════════════════════════════════
if CKPT_10K.exists():
    print(f"Loading preprocessed checkpoint: {CKPT_10K}")
    adata = sc.read_h5ad(CKPT_10K)
    print(f"Loaded: {adata.shape[0]} cells x {adata.shape[1]} genes")
    skip_preprocessing = True
else:
    # Stage raw data into a temp directory with standardized filenames
    # (Scanpy expects barcodes.tsv, not barcodes_fixed.tsv)
    DATA_10K_STAGED.mkdir(parents=True, exist_ok=True)
    shutil.copy2(DATA_10K_RAW / "barcodes_fixed.tsv", DATA_10K_STAGED / "barcodes.tsv")
    shutil.copy2(DATA_10K_RAW / "matrix.mtx",          DATA_10K_STAGED / "matrix.mtx")
    shutil.copy2(DATA_10K_RAW / "features.tsv",        DATA_10K_STAGED / "genes.tsv")
    print(f"Staged files: {list(DATA_10K_STAGED.iterdir())}")

    adata = sc.read_10x_mtx(str(DATA_10K_STAGED), var_names="gene_symbols", cache=True)
    adata.var_names_make_unique()
    print(f"Loaded: {adata.shape[0]} cells x {adata.shape[1]} genes")
    skip_preprocessing = False

# ═══════════════════════════════════════════════════════════════════════════
# STEP 2: Quality Control
# ═══════════════════════════════════════════════════════════════════════════
if not skip_preprocessing:
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None,
                               log1p=False, inplace=True)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].hist(adata.obs["n_genes_by_counts"], bins=60, color="steelblue", edgecolor="white")
    axes[0].set_xlabel("Genes per cell"); axes[0].set_title("Genes per Cell")
    axes[1].hist(adata.obs["total_counts"], bins=60, color="seagreen", edgecolor="white")
    axes[1].set_xlabel("Total counts"); axes[1].set_title("Total Counts per Cell")
    axes[2].hist(adata.obs["pct_counts_mt"], bins=60, color="salmon", edgecolor="white")
    axes[2].set_xlabel("% MT counts"); axes[2].set_title("Mitochondrial %")
    plt.suptitle("PBMC 10k - Pre-filter QC", fontsize=13, fontweight="bold")
    plt.tight_layout(); savefig("qc_prefilter_10k.png")

    # Looser thresholds for the larger 10k dataset
    MIN_GENES, MAX_GENES, MAX_MT_PCT = 200, 5000, 15.0
    print(f"Before filtering: {adata.n_obs} cells")
    sc.pp.filter_cells(adata, min_genes=MIN_GENES)
    sc.pp.filter_cells(adata, max_genes=MAX_GENES)
    adata = adata[adata.obs["pct_counts_mt"] < MAX_MT_PCT].copy()
    sc.pp.filter_genes(adata, min_cells=3)
    print(f"After filtering:  {adata.n_obs} cells, {adata.n_vars} genes")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 3: Preprocessing
# ═══════════════════════════════════════════════════════════════════════════
if not skip_preprocessing:
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.raw = adata

    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    print(f"Highly variable genes selected: {adata.var['highly_variable'].sum()}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, norm, title in zip(axes, [True, False],
                               ["Normalized dispersion", "Raw dispersion"]):
        disp_col = "dispersions_norm" if norm else "dispersions"
        ax.scatter(adata.var["means"][~adata.var["highly_variable"]],
                   adata.var[disp_col][~adata.var["highly_variable"]],
                   s=2, color="lightgray", label="other genes", alpha=0.7)
        ax.scatter(adata.var["means"][adata.var["highly_variable"]],
                   adata.var[disp_col][adata.var["highly_variable"]],
                   s=4, color="#e8473f", label="highly variable", alpha=0.9)
        ax.set_xlabel("Mean expression"); ax.set_ylabel(title)
        ax.legend(markerscale=3, fontsize=9)
    plt.suptitle("Highly Variable Genes - PBMC 10k", fontsize=13, fontweight="bold")
    plt.tight_layout(); savefig("hvg_10k.png")

    adata = adata[:, adata.var["highly_variable"]].copy()
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, svd_solver="arpack", n_comps=50, random_state=SEED)

    pca_var = adata.uns["pca"]["variance_ratio"]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(1, 21), pca_var[:20], color="steelblue", edgecolor="white")
    ax.plot(range(1, 21), pca_var[:20], "o-", color="#e8473f", markersize=5)
    ax.set_xlabel("PC ranking"); ax.set_ylabel("Variance ratio")
    ax.set_title("PCA Variance Ratio - PBMC 10k (Top 20 PCs)")
    plt.tight_layout(); savefig("pca_variance_10k.png")

    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=20, random_state=SEED)
    sc.tl.umap(adata, random_state=SEED)
    print(f"After HVG selection: {adata.shape[0]} cells x {adata.shape[1]} genes")

    adata.write(CKPT_10K)
    print(f"Saved preprocessed checkpoint: {CKPT_10K}")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 4: K-means Clustering
# ═══════════════════════════════════════════════════════════════════════════
print("\n── K-means Clustering ──")
X_pca = adata.obsm["X_pca"][:, :20]
k_range = range(3, 12)
kmeans_results = {}

for k in k_range:
    km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
    labels = km.fit_predict(X_pca)
    sil = silhouette_score(X_pca, labels)
    db  = davies_bouldin_score(X_pca, labels)
    ch  = calinski_harabasz_score(X_pca, labels)
    kmeans_results[k] = {"labels": labels, "silhouette": sil, "davies_bouldin": db, "calinski_harabasz": ch}
    print(f"  k={k} | Silhouette: {sil:.4f} | DB: {db:.4f} | CH: {ch:.1f}")

best_k = max(kmeans_results, key=lambda k: kmeans_results[k]["silhouette"])
print(f"\nBest k by silhouette: {best_k}")
adata.obs["kmeans_best"] = pd.Categorical(kmeans_results[best_k]["labels"].astype(str))

fig, axes = plt.subplots(1, 4, figsize=(22, 5))
for ax, k in zip(axes, [4, 6, 8, 10]):
    adata.obs[f"kmeans_{k}"] = pd.Categorical(kmeans_results[k]["labels"].astype(str))
    sc.pl.umap(adata, color=f"kmeans_{k}", ax=ax, show=False,
               legend_loc="on data",
               title=f"K-means k={k}\nsil={kmeans_results[k]['silhouette']:.2f}")
plt.suptitle("K-means Clustering - PBMC 10k", fontsize=13, fontweight="bold")
plt.tight_layout(); savefig("kmeans_umap_10k.png")

summary_km = pd.DataFrame({
    "k": list(k_range),
    "silhouette":     [kmeans_results[k]["silhouette"] for k in k_range],
    "davies_bouldin": [kmeans_results[k]["davies_bouldin"] for k in k_range],
})
print("\nK-means Summary - PBMC 10k:"); print(summary_km.to_string(index=False))

# ═══════════════════════════════════════════════════════════════════════════
# STEP 5: Hierarchical Clustering (Ward only for 10k)
# ═══════════════════════════════════════════════════════════════════════════
print("\n── Hierarchical Ward Clustering ──")
n_clusters_range = range(3, 12)
hier_results = {}

for n in n_clusters_range:
    model  = AgglomerativeClustering(n_clusters=n, linkage="ward")
    labels = model.fit_predict(X_pca)
    sil    = silhouette_score(X_pca, labels)
    db     = davies_bouldin_score(X_pca, labels)
    ch     = calinski_harabasz_score(X_pca, labels)
    counts = Counter(labels)
    max_pct = max(counts.values()) / len(labels) * 100
    min_pct = min(counts.values()) / len(labels) * 100
    hier_results[n] = {"labels": labels, "silhouette": sil, "davies_bouldin": db,
                       "calinski_harabasz": ch, "max_pct": max_pct, "min_pct": min_pct}
    print(f"  n={n} | Silhouette: {sil:.4f} | DB: {db:.4f} | "
          f"Max: {max_pct:.1f}% | Min: {min_pct:.1f}%")

best_n = max(hier_results, key=lambda n: hier_results[n]["silhouette"])
print(f"\nBest n by silhouette: {best_n}")
adata.obs["hier_ward_best"] = pd.Categorical(hier_results[best_n]["labels"].astype(str))

plot_ns = list(dict.fromkeys([4, 6, 8, best_n]))
fig, axes = plt.subplots(1, len(plot_ns), figsize=(5.5 * len(plot_ns), 5))
if len(plot_ns) == 1: axes = [axes]
for ax, n in zip(axes, plot_ns):
    col = f"hier_ward_{n}"
    adata.obs[col] = pd.Categorical(hier_results[n]["labels"].astype(str))
    sc.pl.umap(adata, color=col, ax=ax, show=False, legend_loc="on data",
               title=f"Hier. Ward n={n}\nsil={hier_results[n]['silhouette']:.2f}")
plt.suptitle("Hierarchical Ward Clustering - PBMC 10k", fontsize=13, fontweight="bold")
plt.tight_layout(); savefig("hierarchical_umap_10k.png")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 6: DBSCAN / HDBSCAN
# ═══════════════════════════════════════════════════════════════════════════
print("\n── DBSCAN / HDBSCAN ──")
X_umap = adata.obsm["X_umap"]

# DBSCAN on UMAP (preferred for scRNA-seq — high-dim PCA is problematic for DBSCAN)
eps_range = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
dbscan_results = {}
for eps in eps_range:
    db_model = DBSCAN(eps=eps, min_samples=10)
    labels   = db_model.fit_predict(X_umap)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_pct  = (labels == -1).sum() / len(labels) * 100
    if n_clusters >= 2 and noise_pct < 50:
        valid_mask = labels != -1
        sil = silhouette_score(X_umap[valid_mask], labels[valid_mask])
        dbs = davies_bouldin_score(X_umap[valid_mask], labels[valid_mask])
    else:
        sil, dbs = np.nan, np.nan
    dbscan_results[eps] = {"labels": labels, "n_clusters": n_clusters,
                           "noise_pct": noise_pct, "silhouette": sil, "davies_bouldin": dbs}
    print(f"  DBSCAN eps={eps:.1f} | clusters={n_clusters:2d} | "
          f"noise={noise_pct:5.1f}% | sil={sil:.4f} | DB={dbs:.4f}")

valid_eps = {e: r for e, r in dbscan_results.items()
             if r["n_clusters"] >= 3 and r["noise_pct"] < 10 and not np.isnan(r["silhouette"])}
if valid_eps:
    best_eps = max(valid_eps, key=lambda e: valid_eps[e]["silhouette"])
else:
    fallback = {e: r for e, r in dbscan_results.items() if r["n_clusters"] >= 2}
    best_eps = min(fallback, key=lambda e: fallback[e]["noise_pct"])
print(f"\nBest DBSCAN eps: {best_eps}")

db_labels = pd.Series(dbscan_results[best_eps]["labels"]).astype(str).replace("-1", "Noise")
adata.obs["dbscan_best"] = pd.Categorical(db_labels)

# HDBSCAN on PCA
min_cluster_sizes = [10, 20, 30, 50, 100]
fixed_min_samples = 10
hdbscan_results = {}
for mcs in min_cluster_sizes:
    clusterer = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=fixed_min_samples,
                                  core_dist_n_jobs=-1)
    labels     = clusterer.fit_predict(X_pca)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_pct  = (labels == -1).sum() / len(labels) * 100
    if n_clusters >= 2 and noise_pct < 50:
        valid_mask = labels != -1
        sil = silhouette_score(X_pca[valid_mask], labels[valid_mask])
        dbs = davies_bouldin_score(X_pca[valid_mask], labels[valid_mask])
    else:
        sil, dbs = np.nan, np.nan
    hdbscan_results[mcs] = {"labels": labels, "n_clusters": n_clusters,
                             "noise_pct": noise_pct, "silhouette": sil, "davies_bouldin": dbs}
    print(f"  HDBSCAN mcs={mcs:3d} | clusters={n_clusters:2d} | "
          f"noise={noise_pct:5.1f}% | sil={sil:.4f} | DB={dbs:.4f}")

valid_hdb = {m: r for m, r in hdbscan_results.items()
             if r["n_clusters"] >= 3 and r["noise_pct"] < 30 and not np.isnan(r["silhouette"])}
if valid_hdb:
    best_mcs = max(valid_hdb, key=lambda m: valid_hdb[m]["silhouette"])
else:
    fallback_hdb = {m: r for m, r in hdbscan_results.items() if r["n_clusters"] >= 2}
    best_mcs = min(fallback_hdb, key=lambda m: fallback_hdb[m]["noise_pct"])
print(f"Best HDBSCAN mcs: {best_mcs}")

hdb_labels = pd.Series(hdbscan_results[best_mcs]["labels"]).astype(str).replace("-1", "Noise")
adata.obs["hdbscan_best"] = pd.Categorical(hdb_labels)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sc.pl.umap(adata, color="dbscan_best", ax=axes[0], show=False, legend_loc="on data",
           title=f"DBSCAN eps={best_eps}\n"
                 f"clusters={dbscan_results[best_eps]['n_clusters']} | "
                 f"noise={dbscan_results[best_eps]['noise_pct']:.1f}%")
sc.pl.umap(adata, color="hdbscan_best", ax=axes[1], show=False, legend_loc="on data",
           title=f"HDBSCAN mcs={best_mcs}\n"
                 f"clusters={hdbscan_results[best_mcs]['n_clusters']} | "
                 f"noise={hdbscan_results[best_mcs]['noise_pct']:.1f}%")
plt.suptitle("DBSCAN / HDBSCAN - PBMC 10k", fontsize=13, fontweight="bold")
plt.tight_layout(); savefig("dbscan_hdbscan_umap_10k.png")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 7: Leiden Clustering
# ═══════════════════════════════════════════════════════════════════════════
print("\n── Leiden Clustering ──")
resolutions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
leiden_results = {}

for res in resolutions:
    key = f"leiden_{res}"
    sc.tl.leiden(adata, resolution=res, random_state=SEED, key_added=key)
    labels     = adata.obs[key].cat.codes.values
    n_clusters = len(set(labels))
    counts     = Counter(labels)
    max_pct    = max(counts.values()) / len(labels) * 100
    min_pct    = min(counts.values()) / len(labels) * 100
    if n_clusters >= 2:
        sil = silhouette_score(X_pca, labels)
        db  = davies_bouldin_score(X_pca, labels)
        ch  = calinski_harabasz_score(X_pca, labels)
    else:
        sil, db, ch = np.nan, np.nan, np.nan
    leiden_results[res] = {"labels": labels, "n_clusters": n_clusters,
                           "silhouette": sil, "davies_bouldin": db, "calinski_harabasz": ch,
                           "max_pct": max_pct, "min_pct": min_pct}
    print(f"  res={res:.1f} | clusters={n_clusters:2d} | sil={sil:.4f} | DB={db:.4f} | "
          f"CH={ch:.0f} | max={max_pct:.1f}% | min={min_pct:.1f}%")

valid_res = {r: v for r, v in leiden_results.items() if not np.isnan(v["silhouette"])}
best_leiden_res = max(valid_res, key=lambda r: valid_res[r]["silhouette"])
print(f"\nBest resolution by silhouette: {best_leiden_res}")
adata.obs["leiden_best"] = adata.obs[f"leiden_{best_leiden_res}"]

plot_res = list(dict.fromkeys([0.2, 0.3, 0.5, best_leiden_res]))
fig, axes = plt.subplots(1, len(plot_res), figsize=(5.5 * len(plot_res), 5))
if len(plot_res) == 1: axes = [axes]
for ax, res in zip(axes, plot_res):
    sc.pl.umap(adata, color=f"leiden_{res}", ax=ax, show=False, legend_loc="on data",
               title=f"Leiden res={res}\n"
                     f"k={leiden_results[res]['n_clusters']} | "
                     f"sil={leiden_results[res]['silhouette']:.3f}")
plt.suptitle("Leiden Clustering - PBMC 10k", fontsize=13, fontweight="bold")
plt.tight_layout(); savefig("leiden_umap_10k.png")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 8: Cell Type Composition (Leiden res=0.1)
# ═══════════════════════════════════════════════════════════════════════════
label_key = "leiden_0.1"
cluster_labels_10k = {
    "0": "T cells", "1": "CD14+ Monocytes", "2": "B cells",
    "3": "T cells / NK mix", "4": "CD16+ Monocytes", "5": "NK cells",
    "6": "Dendritic cells", "7": "Platelets", "8": "Unassigned",
    "9": "Putative rare population",
}
cluster_counts = adata.obs[label_key].astype(str).value_counts().sort_index()
cluster_names  = [cluster_labels_10k.get(str(c), f"Cluster {c}") for c in cluster_counts.index]
cluster_pcts   = cluster_counts / cluster_counts.sum() * 100
color_map = {
    "T cells": "steelblue", "CD14+ Monocytes": "#e8473f", "B cells": "seagreen",
    "T cells / NK mix": "salmon", "CD16+ Monocytes": "mediumpurple",
    "NK cells": "darkorange", "Dendritic cells": "#138d90",
    "Platelets": "gold", "Unassigned": "gray", "Putative rare population": "#7f8c8d",
}
colors_bar = [color_map.get(n, "lightgray") for n in cluster_names]

fig, ax = plt.subplots(figsize=(11, 6))
bars = ax.barh(cluster_names, cluster_pcts, color=colors_bar, edgecolor="white", alpha=0.85)
ax.set_xlabel("Percentage of cells (%)", fontsize=11)
ax.set_title("Cell Type Composition - PBMC 10k (Leiden res=0.1)",
             fontsize=13, fontweight="bold")
ax.set_xlim(0, max(cluster_pcts) + 10)
for bar, pct, count in zip(bars, cluster_pcts, cluster_counts):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
            f"{pct:.1f}% ({count} cells)", va="center", fontsize=9)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout(); savefig("cell_type_composition_10k_bar_only.png")

# Top 3 marker genes per annotated cell type
adata_markers = adata.raw.to_adata()
adata_markers.obs["cell_type"] = adata.obs[label_key].astype(str).map(cluster_labels_10k)
adata_markers = adata_markers[~adata_markers.obs["cell_type"].isna()].copy()
sc.tl.rank_genes_groups(adata_markers, groupby="cell_type",
                         method="wilcoxon", key_added="rank_genes_ct_10k", pts=True)
cell_types_list = list(pd.unique(adata_markers.obs["cell_type"]))
n_cols = 3
n_rows = int(np.ceil(len(cell_types_list) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4.5 * n_rows))
axes = np.array(axes).reshape(-1)
for i, ct in enumerate(cell_types_list):
    df     = sc.get.rank_genes_groups_df(adata_markers, group=ct, key="rank_genes_ct_10k").head(3)
    genes  = df["names"].tolist(); scores = df["scores"].tolist()
    ax     = axes[i]
    c_list = plt.cm.Blues(np.linspace(0.4, 0.9, 3))
    ax.bar(range(3), scores, color=c_list, edgecolor="white", width=0.6)
    for j, (gene, score) in enumerate(zip(genes, scores)):
        ax.text(j, score + max(scores) * 0.01, gene,
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticks(range(3)); ax.set_xticklabels([f"Rank {k+1}" for k in range(3)], fontsize=9)
    ax.set_ylabel("Wilcoxon Score", fontsize=9)
    ax.set_title(ct, fontsize=11, fontweight="bold")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
for j in range(i + 1, len(axes)): axes[j].axis("off")
plt.suptitle("Top 3 Marker Genes per Cell Type - PBMC 10k", fontsize=14, fontweight="bold")
plt.tight_layout(); savefig("top_genes_per_celltype_10k.png")

print("Cell Type Composition Summary - PBMC 10k:")
for name, count, pct in zip(cluster_names, cluster_counts, cluster_pcts):
    print(f"  {name:<28} {count:>8}  {pct:.1f}%")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 9: scVI + Leiden
# ═══════════════════════════════════════════════════════════════════════════
print("\n── scVI + Leiden ──")
if SCVI_10K.exists():
    print(f"Loading scVI checkpoint: {SCVI_10K}")
    adata = sc.read_h5ad(SCVI_10K)
    X_pca  = adata.obsm["X_pca"][:, :20]
    X_umap = adata.obsm["X_umap"]
else:
    scvi.settings.seed = SEED
    adata_scvi_tmp = adata.copy()
    adata_scvi_tmp.X = adata_scvi_tmp.layers["counts"].copy()
    scvi.model.SCVI.setup_anndata(adata_scvi_tmp)
    model = scvi.model.SCVI(adata_scvi_tmp, n_layers=2, n_latent=10, gene_likelihood="nb")
    print("Training scVI model...")
    model.train(max_epochs=200, early_stopping=True, early_stopping_patience=10,
                plan_kwargs={"lr": 1e-3})
    print("Training complete.")
    adata.obsm["X_scvi"] = model.get_latent_representation()
    print(f"scVI latent shape: {adata.obsm['X_scvi'].shape}")
    model.save(str(SCVI_MODEL_DIR), overwrite=True)
    sc.pp.neighbors(adata, use_rep="X_scvi", n_neighbors=15, key_added="scvi_neighbors")

    scvi_resolutions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
    for res in scvi_resolutions:
        sc.tl.leiden(adata, resolution=res, random_state=SEED,
                     neighbors_key="scvi_neighbors", key_added=f"scvi_leiden_{res}")

    # Compute scVI UMAP for visualization, then restore PCA UMAP
    sc.tl.umap(adata, neighbors_key="scvi_neighbors", random_state=SEED)
    adata.obsm["X_umap_scvi"] = adata.obsm["X_umap"].copy()
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=20, random_state=SEED)
    sc.tl.umap(adata, random_state=SEED)

    adata.write(SCVI_10K)
    print(f"Saved scVI checkpoint: {SCVI_10K}")

X_scvi = adata.obsm.get("X_scvi", adata.obsm.get("X_scVI"))
scvi_resolutions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
scvi_results = {}

# Recompute scvi_neighbors if needed
if "scvi_neighbors" not in adata.uns:
    sc.pp.neighbors(adata, use_rep="X_scvi" if "X_scvi" in adata.obsm else "X_scVI",
                    n_neighbors=15, key_added="scvi_neighbors")

for res in scvi_resolutions:
    key = f"scvi_leiden_{res}"
    if key not in adata.obs.columns:
        sc.tl.leiden(adata, resolution=res, random_state=SEED,
                     neighbors_key="scvi_neighbors", key_added=key)
    labels     = adata.obs[key].cat.codes.values
    n_clusters = len(set(labels))
    counts     = Counter(labels)
    max_pct    = max(counts.values()) / len(labels) * 100
    min_pct    = min(counts.values()) / len(labels) * 100
    if n_clusters >= 2:
        sil = silhouette_score(X_scvi, labels)
        db  = davies_bouldin_score(X_scvi, labels)
        ch  = calinski_harabasz_score(X_scvi, labels)
    else:
        sil, db, ch = np.nan, np.nan, np.nan
    scvi_results[res] = {"labels": labels, "n_clusters": n_clusters,
                         "silhouette": sil, "davies_bouldin": db, "calinski_harabasz": ch,
                         "max_pct": max_pct, "min_pct": min_pct}
    print(f"  res={res:.1f} | clusters={n_clusters:2d} | sil={sil:.4f} | DB={db:.4f}")

valid_scvi_res = {r: v for r, v in scvi_results.items() if not np.isnan(v["silhouette"])}
best_scvi_res  = max(valid_scvi_res, key=lambda r: valid_scvi_res[r]["silhouette"])
print(f"\nBest scVI resolution by silhouette: {best_scvi_res}")
adata.obs["scvi_leiden_best"] = adata.obs[f"scvi_leiden_{best_scvi_res}"]

X_umap_scvi = adata.obsm.get("X_umap_scvi", adata.obsm["X_umap"])
plot_res = list(dict.fromkeys([0.1, 0.2, 0.4, best_scvi_res]))
fig, axes = plt.subplots(1, len(plot_res), figsize=(5.5 * len(plot_res), 5))
if len(plot_res) == 1: axes = [axes]
for ax, res in zip(axes, plot_res):
    key = f"scvi_leiden_{res}"
    adata.obsm["X_umap_tmp"] = X_umap_scvi
    sc.pl.embedding(adata, basis="X_umap_tmp", color=key, ax=ax, show=False,
                    legend_loc="on data",
                    title=f"scVI+Leiden res={res}\n"
                          f"k={scvi_results[res]['n_clusters']} | "
                          f"sil={scvi_results[res]['silhouette']:.3f}")
plt.suptitle("scVI + Leiden - PBMC 10k", fontsize=13, fontweight="bold")
plt.tight_layout(); savefig("scvi_leiden_umap_10k.png")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 10: Load SC3 Labels (if available)
# ═══════════════════════════════════════════════════════════════════════════
sc3_metrics = {}
if SC3_LABELS_10K.exists():
    print(f"\n── Loading SC3 Labels ──")
    sc3_df = pd.read_csv(SC3_LABELS_10K)
    sc3_df = sc3_df.set_index("cell")
    missing = adata.obs_names.difference(sc3_df.index)
    if len(missing) > 0:
        print(f"Warning: {len(missing)} cells missing from SC3 label file.")
    sc3_df = sc3_df.loc[adata.obs_names]
    print(f"SC3 labels loaded: {sc3_df.shape[0]} cells x {sc3_df.shape[1]} k values")

    for col in sc3_df.columns:
        k = int(col.split("_")[1])
        labels     = sc3_df[col].values
        n_clusters = len(set(labels))
        counts_c   = Counter(labels)
        max_pct    = max(counts_c.values()) / len(labels) * 100
        min_pct    = min(counts_c.values()) / len(labels) * 100
        if n_clusters >= 2:
            sil = silhouette_score(X_pca, labels)
            db  = davies_bouldin_score(X_pca, labels)
        else:
            sil, db = np.nan, np.nan
        sc3_metrics[k] = {"labels": labels, "n_clusters": n_clusters,
                          "silhouette": sil, "davies_bouldin": db,
                          "max_pct": max_pct, "min_pct": min_pct}
        adata.obs[f"sc3_k{k}"] = pd.Categorical(labels.astype(str))
        print(f"  k={k:2d} | clusters={n_clusters:2d} | sil={sil:.4f} | DB={db:.4f}")

    valid_k    = {k: v for k, v in sc3_metrics.items() if not np.isnan(v["silhouette"])}
    best_k_sc3 = max(valid_k, key=lambda k: valid_k[k]["silhouette"])
    print(f"\nBest k by silhouette: {best_k_sc3}")
    adata.obs["sc3_best"] = pd.Categorical(sc3_metrics[best_k_sc3]["labels"].astype(str))

    plot_ks = list(dict.fromkeys([6, 8, 10, best_k_sc3]))
    fig, axes = plt.subplots(1, len(plot_ks), figsize=(5.5 * len(plot_ks), 5))
    if len(plot_ks) == 1: axes = [axes]
    for ax, k in zip(axes, plot_ks):
        sc.pl.umap(adata, color=f"sc3_k{k}", ax=ax, show=False, legend_loc="on data",
                   title=f"SC3 k={k}\nsil={sc3_metrics[k]['silhouette']:.3f} | "
                         f"DB={sc3_metrics[k]['davies_bouldin']:.3f}")
    plt.suptitle("SC3 Clustering - PBMC 10k", fontsize=13, fontweight="bold")
    plt.tight_layout(); savefig("sc3_umap_10k.png")

    ks   = sorted(sc3_metrics.keys())
    sils = [sc3_metrics[k]["silhouette"] for k in ks]
    dbs  = [sc3_metrics[k]["davies_bouldin"] for k in ks]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ks, sils, "o-", color="steelblue", label="Silhouette", linewidth=2)
    ax2 = ax.twinx()
    ax2.plot(ks, dbs, "s--", color="#e8473f", label="Davies-Bouldin", linewidth=2)
    ax.set_xlabel("Number of clusters (k)"); ax.set_ylabel("Silhouette score", color="steelblue")
    ax2.set_ylabel("Davies-Bouldin score", color="#e8473f")
    ax.set_title("SC3 Metric Sweep - PBMC 10k", fontsize=13, fontweight="bold")
    ax.axvline(best_k_sc3, color="gray", linestyle="--", linewidth=1)
    lines1, lbs1 = ax.get_legend_handles_labels()
    lines2, lbs2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lbs1 + lbs2, loc="upper right")
    plt.tight_layout(); savefig("sc3_metrics_10k.png")

    # NMI between SC3 and Leiden
    _leiden_cols_10 = [c for c in adata.obs.columns
                       if c.startswith("leiden_") and not c.startswith("leiden_seed")
                       and not c.startswith("leiden_best")]
    _sc3_cols_10    = [c for c in adata.obs.columns if c.startswith("sc3_k")
                       and not c.startswith("sc3_best")]
    nmi_rows_10 = []
    for lc in _leiden_cols_10:
        for sc3c in _sc3_cols_10:
            nmi = normalized_mutual_info_score(
                adata.obs[lc].astype(str), adata.obs[sc3c].astype(str))
            nmi_rows_10.append({"leiden": lc, "sc3": sc3c, "NMI": round(nmi, 4)})
    if nmi_rows_10:
        nmi_df_10 = pd.DataFrame(nmi_rows_10)
        nmi_pivot_10 = nmi_df_10.pivot(index="leiden", columns="sc3", values="NMI")
        fig, ax = plt.subplots(figsize=(12, 7))
        im = ax.imshow(nmi_pivot_10.values, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(len(nmi_pivot_10.columns)))
        ax.set_xticklabels(nmi_pivot_10.columns, fontsize=10)
        ax.set_yticks(range(len(nmi_pivot_10.index)))
        ax.set_yticklabels(nmi_pivot_10.index, fontsize=9)
        plt.colorbar(im, ax=ax, label="NMI")
        for i in range(len(nmi_pivot_10.index)):
            for j in range(len(nmi_pivot_10.columns)):
                ax.text(j, i, f"{nmi_pivot_10.values[i,j]:.2f}",
                        ha="center", va="center", fontsize=8)
        ax.set_title("NMI: Leiden vs SC3 — PBMC 10k", fontsize=13, fontweight="bold")
        ax.set_xlabel("SC3 k"); ax.set_ylabel("Leiden resolution")
        plt.tight_layout(); savefig("nmi_leiden_vs_sc3_10k.png")
        nmi_df_10.to_csv(RES_DIR / "nmi_leiden_vs_sc3_10k.csv", index=False)
else:
    print(f"\nSC3 labels not found at {SC3_LABELS_10K}.")
    print("Run: Rscript sc3_10k.R   (from the src/ directory)")
    print("Then re-run this script to generate SC3 comparison plots.")
    best_k_sc3 = 6

# ═══════════════════════════════════════════════════════════════════════════
# STEP 11: Stability Analysis
# ═══════════════════════════════════════════════════════════════════════════
print("\n── Stability Analysis ──")
X_pca  = adata.obsm["X_pca"][:, :20]
X_umap = adata.obsm["X_umap"]
X_scvi_arr = adata.obsm.get("X_scvi", adata.obsm.get("X_scVI"))
N      = len(X_pca)
SEEDS  = [0, 7, 13, 21, 42, 55, 77, 88, 99, 123]

ref_labels = {
    "kmeans":      adata.obs["kmeans_best"].cat.codes.values,
    "hier_ward":   adata.obs["hier_ward_best"].cat.codes.values,
    "dbscan":      adata.obs["dbscan_best"].cat.codes.values,
    "hdbscan":     adata.obs["hdbscan_best"].cat.codes.values,
    "leiden_pca":  adata.obs["leiden_best"].cat.codes.values,
    "scvi_leiden": adata.obs["scvi_leiden_best"].cat.codes.values,
}

BEST_K       = best_k
BEST_N       = best_n
BEST_EPS     = best_eps
BEST_MCS     = best_mcs
BEST_LEIDEN  = best_leiden_res
BEST_SCVI    = best_scvi_res

# Seed ARI
print("=" * 60); print("Seed ARI (10 seeds)"); print("=" * 60)
seed_aris = {m: [] for m in ref_labels}
for s in SEEDS:
    km = KMeans(n_clusters=BEST_K, random_state=s, n_init=10)
    seed_aris["kmeans"].append(adjusted_rand_score(ref_labels["kmeans"], km.fit_predict(X_pca)))
    hw = AgglomerativeClustering(n_clusters=BEST_N, linkage="ward")
    seed_aris["hier_ward"].append(adjusted_rand_score(ref_labels["hier_ward"], hw.fit_predict(X_pca)))
    db = DBSCAN(eps=BEST_EPS, min_samples=10)
    seed_aris["dbscan"].append(adjusted_rand_score(ref_labels["dbscan"], db.fit_predict(X_umap)))
    hdb = hdbscan.HDBSCAN(min_cluster_size=BEST_MCS, min_samples=10, core_dist_n_jobs=-1)
    seed_aris["hdbscan"].append(adjusted_rand_score(ref_labels["hdbscan"], hdb.fit_predict(X_pca)))
    sc.tl.leiden(adata, resolution=BEST_LEIDEN, random_state=s, key_added=f"leiden_seed_{s}")
    seed_aris["leiden_pca"].append(adjusted_rand_score(
        ref_labels["leiden_pca"], adata.obs[f"leiden_seed_{s}"].cat.codes.values))
    sc.tl.leiden(adata, resolution=BEST_SCVI, random_state=s,
                 neighbors_key="scvi_neighbors", key_added=f"scvi_seed_{s}")
    seed_aris["scvi_leiden"].append(adjusted_rand_score(
        ref_labels["scvi_leiden"], adata.obs[f"scvi_seed_{s}"].cat.codes.values))

print(f"{'Method':<15} {'Mean ARI':>10} {'Std':>8} {'Stable?':>10}")
print("-" * 50)
for m, aris in seed_aris.items():
    mean, std = np.mean(aris), np.std(aris)
    print(f"{m:<15} {mean:>10.4f} {std:>8.4f} {'YES' if mean >= 0.8 else 'NO':>10}")

# Resampling ARI
print(f"\n{'='*60}"); print("Resampling ARI (80% subsampling x 10 runs)"); print("=" * 60)
resample_aris = {m: [] for m in ref_labels}
for i in range(10):
    idx = resample(np.arange(N), n_samples=int(0.8 * N), random_state=i, replace=False)
    Xp = X_pca[idx]; Xu = X_umap[idx]; Xs = X_scvi_arr[idx]

    km = KMeans(n_clusters=BEST_K, random_state=42, n_init=10)
    resample_aris["kmeans"].append(adjusted_rand_score(ref_labels["kmeans"][idx], km.fit_predict(Xp)))
    hw = AgglomerativeClustering(n_clusters=BEST_N, linkage="ward")
    resample_aris["hier_ward"].append(adjusted_rand_score(ref_labels["hier_ward"][idx], hw.fit_predict(Xp)))
    db = DBSCAN(eps=BEST_EPS, min_samples=10)
    resample_aris["dbscan"].append(adjusted_rand_score(ref_labels["dbscan"][idx], db.fit_predict(Xu)))
    hdb = hdbscan.HDBSCAN(min_cluster_size=BEST_MCS, min_samples=10, core_dist_n_jobs=-1)
    resample_aris["hdbscan"].append(adjusted_rand_score(ref_labels["hdbscan"][idx], hdb.fit_predict(Xp)))

    adata_sub = anndata.AnnData(X=Xp)
    sc.pp.neighbors(adata_sub, n_neighbors=15, use_rep="X", random_state=42)
    sc.tl.leiden(adata_sub, resolution=BEST_LEIDEN, random_state=42, key_added="leiden_sub")
    resample_aris["leiden_pca"].append(adjusted_rand_score(
        ref_labels["leiden_pca"][idx], adata_sub.obs["leiden_sub"].cat.codes.values))

    adata_sub2 = anndata.AnnData(X=Xs)
    sc.pp.neighbors(adata_sub2, n_neighbors=15, use_rep="X", random_state=42)
    sc.tl.leiden(adata_sub2, resolution=BEST_SCVI, random_state=42, key_added="leiden_sub")
    resample_aris["scvi_leiden"].append(adjusted_rand_score(
        ref_labels["scvi_leiden"][idx], adata_sub2.obs["leiden_sub"].cat.codes.values))

    print(f"  Resample run {i+1}/10 done")

print(f"\n{'Method':<15} {'Mean ARI':>10} {'Std':>8} {'Stable?':>10}")
print("-" * 50)
for m, aris in resample_aris.items():
    mean, std = np.mean(aris), np.std(aris)
    print(f"{m:<15} {mean:>10.4f} {std:>8.4f} {'YES' if mean >= 0.8 else 'NO':>10}")

method_labels_map = {"kmeans": "KMeans", "hier_ward": "Hierarchical",
                     "dbscan": "DBSCAN", "hdbscan": "HDBSCAN",
                     "leiden_pca": "Leiden", "scvi_leiden": "scVI+Leiden"}
bar_data = []
for m in ref_labels:
    bar_data.append({"label": f"{method_labels_map[m]}_seed",   "mean": np.mean(seed_aris[m]),
                     "std": np.std(seed_aris[m]),   "type": "seed"})
    bar_data.append({"label": f"{method_labels_map[m]}_resample", "mean": np.mean(resample_aris[m]),
                     "std": np.std(resample_aris[m]), "type": "resample"})
df_bar = pd.DataFrame(bar_data)

fig, ax = plt.subplots(figsize=(10, 9))
colors_stab = {"seed": "steelblue", "resample": "#e8473f"}
for i, row in df_bar.iterrows():
    ax.barh(i, row["mean"], xerr=row["std"], color=colors_stab[row["type"]],
            alpha=0.85, error_kw=dict(ecolor="black", capsize=4))
ax.set_yticks(list(range(len(df_bar)))); ax.set_yticklabels(df_bar["label"])
ax.axvline(0.8, color="gray", linestyle="--", linewidth=1.5)
ax.set_xlabel("Adjusted Rand Index (ARI)")
ax.set_title("Stability Analysis - Seed and Resampling (PBMC 10k)",
             fontsize=13, fontweight="bold")
ax.set_xlim(0, 1.05)
ax.legend(handles=[mpatches.Patch(facecolor="steelblue", label="Seed ARI"),
                   mpatches.Patch(facecolor="#e8473f", label="Resample ARI")],
          loc="lower right")
plt.tight_layout(); savefig("stability_10k.png")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 12: Biological Validation — All Methods
# ═══════════════════════════════════════════════════════════════════════════
print("\n── Biological Validation (All Methods) ──")
MARKERS = {
    "T cells":          ["CD3D", "CD3E", "CD3G"],
    "CD4+ T cells":     ["CD4", "IL7R"],
    "CD8+ T cells":     ["CD8A", "CD8B"],
    "NK cells":         ["NKG7", "GZMA", "PRF1", "GNLY"],
    "B cells":          ["CD79A", "CD79B", "MS4A1"],
    "CD14+ Monocytes":  ["CD14", "LYZ", "CST3"],
    "CD16+ Monocytes":  ["FCGR3A", "MS4A7"],
    "Dendritic cells":  ["FCER1A", "CST3", "HLA-DQA1"],
    "Platelets":        ["PF4", "PPBP"],
    "pDC":              ["IL3RA", "GZMB", "SERPINF1"],
}
methods_to_validate = {
    "K-means":     "kmeans_best", "Hier. Ward": "hier_ward_best",
    "DBSCAN":      "dbscan_best", "HDBSCAN":    "hdbscan_best",
    "Leiden PCA":  "leiden_best", "scVI+Leiden": "scvi_leiden_best",
}
if sc3_metrics:
    methods_to_validate["SC3"] = "sc3_best"

def validate_method(adata_v, label_key, method_name, exclude_noise=True):
    ad = adata_v.copy()
    labels = ad.obs[label_key].astype(str)
    if exclude_noise:
        keep_mask = ~labels.isin(["Noise", "-1"])
        ad = ad[keep_mask].copy()
        labels = ad.obs[label_key].astype(str)
    n_clusters = labels.nunique()
    print(f"\n{'='*55}\n{method_name} ({n_clusters} clusters)\n{'='*55}")
    if n_clusters < 2:
        print("  Not enough non-noise clusters.")
        return 0, set()
    sc.tl.rank_genes_groups(ad, groupby=label_key, method="wilcoxon",
                             key_added=f"rank_{label_key}", pts=True, use_raw=True)
    detected = set()
    for cluster in sorted(labels.unique(), key=lambda x: str(x)):
        top_genes = sc.get.rank_genes_groups_df(ad, group=cluster,
                                                  key=f"rank_{label_key}").head(20)["names"].tolist()
        matched = []
        for cell_type, markers in MARKERS.items():
            hits = [m for m in markers if m in top_genes]
            if hits:
                matched.append(f"{cell_type}({','.join(hits)})")
                detected.add(cell_type)
        print(f"  Cluster {cluster}: {' | '.join(matched) if matched else 'Unknown'}")
    score = len(detected)
    print(f"\n  Detected: {score}/{len(MARKERS)}  {', '.join(sorted(detected))}")
    return score, detected

validation_scores   = {}
validation_detected = {}
for method_name, label_key in methods_to_validate.items():
    score, detected = validate_method(
        adata, label_key, method_name,
        exclude_noise=(method_name in ["DBSCAN", "HDBSCAN"]))
    validation_scores[method_name]   = score
    validation_detected[method_name] = detected

print(f"\n{'='*60}\nBiological Validation Summary - PBMC 10k\n{'='*60}")
for m, score in validation_scores.items():
    print(f"  {m:<15} {score}/{len(MARKERS)}  {', '.join(sorted(validation_detected[m]))}")

fig, ax = plt.subplots(figsize=(12, 5))
methods_list_v = list(validation_scores.keys())
scores_v = [validation_scores[m] for m in methods_list_v]
colors_v = ["steelblue", "seagreen", "salmon", "goldenrod", "#e8473f", "mediumpurple", "darkorange"]
bars = ax.bar(methods_list_v, scores_v, color=colors_v[:len(methods_list_v)],
              edgecolor="white", width=0.55)
ax.axhline(len(MARKERS), color="gray", linestyle="--", linewidth=1)
ax.set_ylabel("Cell types detected"); ax.set_title("Biological Validation - PBMC 10k",
             fontsize=13, fontweight="bold")
ax.set_ylim(0, len(MARKERS) + 1)
for bar, score in zip(bars, scores_v):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
            str(score), ha="center", va="bottom", fontweight="bold")
plt.tight_layout(); savefig("biological_validation_10k.png")

val_df = pd.DataFrame({
    "method": list(validation_scores.keys()),
    "score":  list(validation_scores.values()),
    "detected": [", ".join(sorted(v)) if v else "None" for v in validation_detected.values()],
})
val_df.to_csv(RES_DIR / "biological_validation_10k.csv", index=False)

# ═══════════════════════════════════════════════════════════════════════════
# STEP 13: Cross-Method Comparison Summary
# ═══════════════════════════════════════════════════════════════════════════
print("\n── Cross-Method Comparison ──")
BEST_SC3 = best_k_sc3 if sc3_metrics else 6

methods_info = {
    "K-means":    {"n_clusters": BEST_K,   "silhouette": kmeans_results[BEST_K]["silhouette"],
                   "davies_bouldin": kmeans_results[BEST_K]["davies_bouldin"],
                   "seed_ari": np.mean(seed_aris["kmeans"]),
                   "resample_ari": np.mean(resample_aris["kmeans"]),
                   "bio_score": validation_scores.get("K-means", np.nan), "noise_pct": 0.0, "space": "PCA"},
    "Hier. Ward": {"n_clusters": BEST_N,   "silhouette": hier_results[BEST_N]["silhouette"],
                   "davies_bouldin": hier_results[BEST_N]["davies_bouldin"],
                   "seed_ari": np.mean(seed_aris["hier_ward"]),
                   "resample_ari": np.mean(resample_aris["hier_ward"]),
                   "bio_score": validation_scores.get("Hier. Ward", np.nan), "noise_pct": 0.0, "space": "PCA"},
    "DBSCAN":     {"n_clusters": dbscan_results[BEST_EPS]["n_clusters"],
                   "silhouette": dbscan_results[BEST_EPS]["silhouette"],
                   "davies_bouldin": dbscan_results[BEST_EPS]["davies_bouldin"],
                   "seed_ari": np.mean(seed_aris["dbscan"]),
                   "resample_ari": np.mean(resample_aris["dbscan"]),
                   "bio_score": validation_scores.get("DBSCAN", np.nan),
                   "noise_pct": dbscan_results[BEST_EPS]["noise_pct"], "space": "UMAP*"},
    "HDBSCAN":    {"n_clusters": hdbscan_results[BEST_MCS]["n_clusters"],
                   "silhouette": hdbscan_results[BEST_MCS]["silhouette"],
                   "davies_bouldin": hdbscan_results[BEST_MCS]["davies_bouldin"],
                   "seed_ari": np.mean(seed_aris["hdbscan"]),
                   "resample_ari": np.mean(resample_aris["hdbscan"]),
                   "bio_score": validation_scores.get("HDBSCAN", np.nan),
                   "noise_pct": hdbscan_results[BEST_MCS]["noise_pct"], "space": "PCA"},
    "Leiden PCA": {"n_clusters": leiden_results[BEST_LEIDEN]["n_clusters"],
                   "silhouette": leiden_results[BEST_LEIDEN]["silhouette"],
                   "davies_bouldin": leiden_results[BEST_LEIDEN]["davies_bouldin"],
                   "seed_ari": np.mean(seed_aris["leiden_pca"]),
                   "resample_ari": np.mean(resample_aris["leiden_pca"]),
                   "bio_score": validation_scores.get("Leiden PCA", np.nan), "noise_pct": 0.0, "space": "PCA"},
    "scVI+Leiden":{"n_clusters": scvi_results[BEST_SCVI]["n_clusters"],
                   "silhouette": scvi_results[BEST_SCVI]["silhouette"],
                   "davies_bouldin": scvi_results[BEST_SCVI]["davies_bouldin"],
                   "seed_ari": np.mean(seed_aris["scvi_leiden"]),
                   "resample_ari": np.mean(resample_aris["scvi_leiden"]),
                   "bio_score": validation_scores.get("scVI+Leiden", np.nan), "noise_pct": 0.0, "space": "scVI"},
}
if sc3_metrics:
    methods_info["SC3"] = {
        "n_clusters": sc3_metrics[BEST_SC3]["n_clusters"],
        "silhouette": sc3_metrics[BEST_SC3]["silhouette"],
        "davies_bouldin": sc3_metrics[BEST_SC3]["davies_bouldin"],
        "seed_ari": np.nan, "resample_ari": np.nan,
        "bio_score": validation_scores.get("SC3", np.nan), "noise_pct": 0.0, "space": "PCA",
    }

df_compare = pd.DataFrame(methods_info).T
df_compare.index.name = "Method"
print("=" * 90); print("Cross-Method Comparison - PBMC 10k"); print("=" * 90)
print(df_compare.to_string())
df_compare.to_csv(RES_DIR / "results_summary_10k.csv")

# Clean display version
display_df = pd.DataFrame({
    "Clusters":     df_compare["n_clusters"].astype(float).round(0).astype("Int64"),
    "Silhouette":   df_compare["silhouette"].astype(float).round(3),
    "DB Index":     df_compare["davies_bouldin"].astype(float).round(3),
    "Seed ARI":     df_compare["seed_ari"].astype(float).round(3),
    "Resample ARI": df_compare["resample_ari"].astype(float).round(3),
    "Bio Score":    [f"{int(v)}/{len(MARKERS)}" if not pd.isna(v) else "N/A"
                     for v in df_compare["bio_score"]],
    "Noise %":      df_compare["noise_pct"].astype(float).round(1),
    "Space":        df_compare["space"],
}, index=df_compare.index)

fig, ax = plt.subplots(figsize=(15, 4.8))
ax.axis("off")
table = ax.table(cellText=display_df.values, colLabels=display_df.columns,
                 rowLabels=display_df.index, cellLoc="center", loc="center")
table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1.15, 2.0)
for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor("black"); cell.set_linewidth(0.5)
    if row == 0 or col == -1:
        cell.set_facecolor("#f0f0f0"); cell.set_text_props(fontweight="bold")
    else:
        cell.set_facecolor("white")
plt.title("Cross-Method Comparison - PBMC 10k", fontsize=13, fontweight="bold", pad=20)
plt.tight_layout(); savefig("comparison_table_10k.png")
display_df.to_csv(RES_DIR / "results_summary_10k_display.csv")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 14: Cross-Dataset Comparison (3k vs 10k) — Biological Validation
# ═══════════════════════════════════════════════════════════════════════════
print("\n── Cross-Dataset Comparison ──")
results_3k = {
    "K-means":    {"silhouette": 0.331, "db": 1.182, "seed_ari": 0.742,
                   "resample_ari": 0.685, "bio": 4, "clusters": 4, "noise": 0.0},
    "Hier. Ward": {"silhouette": 0.328, "db": 1.061, "seed_ari": 1.000,
                   "resample_ari": 0.926, "bio": 6, "clusters": 5, "noise": 0.0},
    "DBSCAN":     {"silhouette": 0.566, "db": 0.439, "seed_ari": 1.000,
                   "resample_ari": 0.997, "bio": 5, "clusters": 4, "noise": 0.1},
    "HDBSCAN":    {"silhouette": 0.380, "db": 1.019, "seed_ari": 1.000,
                   "resample_ari": 0.864, "bio": 4, "clusters": 3, "noise": 10.7},
    "Leiden PCA": {"silhouette": 0.305, "db": 1.223, "seed_ari": 0.971,
                   "resample_ari": 0.970, "bio": 6, "clusters": 6, "noise": 0.0},
    "scVI+Leiden":{"silhouette": 0.105, "db": 2.344, "seed_ari": 0.996,
                   "resample_ari": 0.976, "bio": 6, "clusters": 5, "noise": 0.0},
    "SC3":        {"silhouette": 0.291, "db": 1.495, "seed_ari": 1.000,
                   "resample_ari": 0.985, "bio": 5, "clusters": 4, "noise": 0.0},
}
results_10k = {m: {"bio": validation_scores.get(name, np.nan),
                   "seed_ari": np.mean(seed_aris.get(key, [np.nan])),
                   "resample_ari": np.mean(resample_aris.get(key, [np.nan])),
                   "silhouette": methods_info[m]["silhouette"],
                   "db": methods_info[m]["davies_bouldin"]}
               for m, name, key in [
                   ("K-means",    "K-means",    "kmeans"),
                   ("Hier. Ward", "Hier. Ward",  "hier_ward"),
                   ("DBSCAN",     "DBSCAN",      "dbscan"),
                   ("HDBSCAN",    "HDBSCAN",     "hdbscan"),
                   ("Leiden PCA", "Leiden PCA",  "leiden_pca"),
                   ("scVI+Leiden","scVI+Leiden", "scvi_leiden"),
               ]}
if sc3_metrics:
    results_10k["SC3"] = {"bio": validation_scores.get("SC3", np.nan), "seed_ari": np.nan,
                          "resample_ari": np.nan,
                          "silhouette": sc3_metrics[BEST_SC3]["silhouette"],
                          "db": sc3_metrics[BEST_SC3]["davies_bouldin"]}

methods_cross = [m for m in results_10k.keys() if m in results_3k]
x = np.arange(len(methods_cross)); width = 0.35

bio_3k_pct  = [results_3k[m]["bio"] / 6 * 100 if pd.notna(results_3k[m]["bio"]) else 0
               for m in methods_cross]
bio_10k_pct = [results_10k[m]["bio"] / len(MARKERS) * 100 if pd.notna(results_10k[m]["bio"]) else 0
               for m in methods_cross]

fig, ax = plt.subplots(figsize=(13, 6), constrained_layout=True)
bars1 = ax.bar(x - width/2, bio_3k_pct, width, label="PBMC 3k", color="steelblue", alpha=0.85)
bars2 = ax.bar(x + width/2, bio_10k_pct, width, label="PBMC 10k", color="#e8473f", alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(methods_cross, rotation=25, ha="right", fontsize=10)
ax.set_ylabel("Cell types detected (%)", fontsize=11); ax.set_ylim(0, 115)
ax.set_title("Biological Validation Comparison - PBMC 3k vs 10k",
             fontsize=14, fontweight="bold")
ax.legend(fontsize=10)
for i, m in enumerate(methods_cross):
    ax.text(x[i] - width/2, bio_3k_pct[i] + 1,
            f"{int(results_3k[m]['bio'])}/6", ha="center", va="bottom", fontsize=8)
    if pd.notna(results_10k[m]["bio"]):
        ax.text(x[i] + width/2, bio_10k_pct[i] + 1,
                f"{int(results_10k[m]['bio'])}/{len(MARKERS)}", ha="center", va="bottom", fontsize=8)
plt.savefig(FIG_DIR / "cross_dataset_biological_validation_only.png", dpi=150, bbox_inches="tight")
plt.close()

# Silhouette & DB cross-dataset
results_3k["SC3"]["silhouette"] = 0.2909; results_3k["SC3"]["db"] = 1.4952
methods_sil = [m for m in methods_cross
               if pd.notna(results_3k[m]["silhouette"]) and pd.notna(results_10k[m]["silhouette"])]
sil_3k  = [results_3k[m]["silhouette"] for m in methods_sil]
sil_10k = [results_10k[m]["silhouette"] for m in methods_sil]
db_3k   = [results_3k[m]["db"] for m in methods_sil]
db_10k  = [results_10k[m]["db"] for m in methods_sil]
x2 = np.arange(len(methods_sil))

fig, axes = plt.subplots(1, 2, figsize=(18, 6), constrained_layout=True)
for ax, vals3, vals10, ylabel, title in [
    (axes[0], sil_3k,  sil_10k, "Silhouette Score",    "PBMC 3k vs 10k: Silhouette"),
    (axes[1], db_3k,   db_10k,  "Davies-Bouldin Index", "PBMC 3k vs 10k: Davies-Bouldin"),
]:
    b1 = ax.bar(x2 - width/2, vals3,  width, label="PBMC 3k",  color="steelblue", alpha=0.85)
    b2 = ax.bar(x2 + width/2, vals10, width, label="PBMC 10k", color="#e8473f",   alpha=0.85)
    ax.set_xticks(x2); ax.set_xticklabels(methods_sil, rotation=25, ha="right", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=11); ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    for bar, val in zip(b1, vals3): ax.text(bar.get_x() + bar.get_width()/2,
        val + 0.02, f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    for bar, val in zip(b2, vals10): ax.text(bar.get_x() + bar.get_width()/2,
        val + 0.02, f"{val:.3f}", ha="center", va="bottom", fontsize=8)
plt.suptitle("Geometric Clustering Metrics: PBMC 3k vs PBMC 10k",
             fontsize=15, fontweight="bold")
plt.savefig(FIG_DIR / "silhouette_db_3k_vs_10k_with_sc3.png", dpi=150, bbox_inches="tight")
plt.close()

# Seed vs Resampling stability cross-dataset
methods_stab = [m for m in methods_cross if m != "SC3"]
seed_3k_vals     = [results_3k[m]["seed_ari"] for m in methods_stab]
resample_3k_vals = [results_3k[m]["resample_ari"] for m in methods_stab]
seed_10k_vals    = [np.mean(seed_aris.get(k, [np.nan]))
                    for m, k in zip(methods_stab,
                                    ["kmeans", "hier_ward", "dbscan", "hdbscan", "leiden_pca", "scvi_leiden"])
                    if m in methods_stab]
resample_10k_vals= [np.mean(resample_aris.get(k, [np.nan]))
                    for m, k in zip(methods_stab,
                                    ["kmeans", "hier_ward", "dbscan", "hdbscan", "leiden_pca", "scvi_leiden"])
                    if m in methods_stab]

x3 = np.arange(len(methods_stab))
fig, axes = plt.subplots(1, 2, figsize=(18, 6), constrained_layout=True)
for ax, seed_v, resamp_v, title in [
    (axes[0], seed_3k_vals,  resample_3k_vals,  "PBMC 3k: Seed vs Resampling Stability"),
    (axes[1], seed_10k_vals, resample_10k_vals, "PBMC 10k: Seed vs Resampling Stability"),
]:
    b1 = ax.bar(x3 - width/2, [0 if pd.isna(v) else v for v in seed_v],
                width, label="Seed ARI", color="steelblue", alpha=0.85)
    b2 = ax.bar(x3 + width/2, [0 if pd.isna(v) else v for v in resamp_v],
                width, label="Resample ARI", color="#e8473f", alpha=0.85)
    ax.axhline(0.8, color="gray", linestyle="--", linewidth=1.5)
    ax.set_xticks(x3); ax.set_xticklabels(methods_stab, rotation=25, ha="right", fontsize=10)
    ax.set_ylabel("ARI", fontsize=11); ax.set_ylim(0, 1.1)
    ax.set_title(title, fontsize=13, fontweight="bold"); ax.legend(fontsize=10)
    for i, bar in enumerate(b1):
        label = "N/A" if pd.isna(seed_v[i]) else f"{seed_v[i]:.3f}"
        ax.text(bar.get_x() + bar.get_width()/2, max(bar.get_height(), 0) + 0.01,
                label, ha="center", va="bottom", fontsize=8)
    for i, bar in enumerate(b2):
        label = "N/A" if pd.isna(resamp_v[i]) else f"{resamp_v[i]:.3f}"
        ax.text(bar.get_x() + bar.get_width()/2, max(bar.get_height(), 0) + 0.01,
                label, ha="center", va="bottom", fontsize=8)
plt.suptitle("Seed vs Resampling Stability Across Methods", fontsize=15, fontweight="bold")
plt.savefig(FIG_DIR / "seed_vs_resampling_all_methods_no_sc3.png", dpi=150, bbox_inches="tight")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════
# EXTRA PRESENTATION FIGURES
# ═══════════════════════════════════════════════════════════════════════════
print("\n── Generating extra presentation figures ──")

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
    plt.suptitle("UMAP Colored by QC Metrics — PBMC 10k", fontsize=13, fontweight="bold")
    plt.tight_layout()
    savefig("umap_qc_overlay_10k.png")

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
    plt.suptitle("PBMC 10k — Post-filter Cell Quality Distribution", fontsize=13, fontweight="bold")
    plt.tight_layout()
    savefig("qc_postfilter_violin_10k.png")

# 3. PCA scatter
_cc10 = "leiden_best"    if "leiden_best"    in adata.obs.columns else f"leiden_{best_leiden_res}"
_ct10 = "scvi_leiden_best" if "scvi_leiden_best" in adata.obs.columns else _cc10
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sc.pl.pca(adata, color=_cc10, ax=axes[0], show=False, title="PCA — Leiden cluster IDs")
sc.pl.pca(adata, color=_ct10, ax=axes[1], show=False, title="PCA — scVI+Leiden cluster IDs")
plt.suptitle("PCA Space — PBMC 10k", fontsize=13, fontweight="bold")
plt.tight_layout()
savefig("pca_scatter_10k.png")

# 4. K-means silhouette + DB sweep
_ks10  = list(k_range)
_sils10 = [kmeans_results[k]["silhouette"]     for k in _ks10]
_dbs10  = [kmeans_results[k]["davies_bouldin"] for k in _ks10]
_ck10   = ["#e8473f" if k == best_k else "steelblue" for k in _ks10]
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
bars1 = axes[0].bar(_ks10, _sils10, color=_ck10, edgecolor="white", width=0.6)
axes[0].set_xlabel("k (clusters)"); axes[0].set_ylabel("Silhouette Score (higher = better)")
axes[0].set_title("K-means: Silhouette Score vs k", fontsize=12, fontweight="bold")
axes[0].set_xticks(_ks10)
for bar, v in zip(bars1, _sils10):
    axes[0].text(bar.get_x() + bar.get_width()/2, v + 0.003, f"{v:.3f}",
                 ha="center", va="bottom", fontsize=8, fontweight="bold")
axes[0].spines["top"].set_visible(False); axes[0].spines["right"].set_visible(False)
bars2 = axes[1].bar(_ks10, _dbs10, color=_ck10, edgecolor="white", width=0.6)
axes[1].set_xlabel("k (clusters)"); axes[1].set_ylabel("Davies-Bouldin Index (lower = better)")
axes[1].set_title("K-means: Davies-Bouldin Index vs k", fontsize=12, fontweight="bold")
axes[1].set_xticks(_ks10)
for bar, v in zip(bars2, _dbs10):
    axes[1].text(bar.get_x() + bar.get_width()/2, v + 0.01, f"{v:.3f}",
                 ha="center", va="bottom", fontsize=8, fontweight="bold")
axes[1].spines["top"].set_visible(False); axes[1].spines["right"].set_visible(False)
plt.suptitle("K-means Metric Sweep — PBMC 10k", fontsize=13, fontweight="bold")
plt.tight_layout()
savefig("kmeans_metrics_sweep_10k.png")

# 5. K-means all-k UMAP grid
_ncols_k10 = 4
_nrows_k10  = int(np.ceil(len(_ks10) / _ncols_k10))
fig, axes = plt.subplots(_nrows_k10, _ncols_k10, figsize=(5 * _ncols_k10, 4.5 * _nrows_k10))
axes = np.array(axes).flatten()
for _i10, _k10 in enumerate(k_range):
    _col10 = f"kmeans_{_k10}"
    if _col10 not in adata.obs.columns:
        adata.obs[_col10] = pd.Categorical(kmeans_results[_k10]["labels"].astype(str))
    sc.pl.umap(adata, color=_col10, ax=axes[_i10], show=False, legend_loc="on data",
               title=f"k={_k10}  sil={kmeans_results[_k10]['silhouette']:.3f}")
for _j10 in range(_i10 + 1, len(axes)):
    axes[_j10].axis("off")
plt.suptitle("K-means: All k Values — PBMC 10k", fontsize=14, fontweight="bold")
plt.tight_layout()
savefig("kmeans_all_k_grid_10k.png")

# 6. Hierarchical silhouette sweep (Ward only for 10k)
_ns10  = list(n_clusters_range)
_sils_w = [hier_results[n]["silhouette"]     for n in _ns10]
_dbs_w  = [hier_results[n]["davies_bouldin"] for n in _ns10]
_best_n10 = max(hier_results, key=lambda n: hier_results[n]["silhouette"])
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(_ns10, _sils_w, "o-", color="steelblue", linewidth=2, markersize=7)
axes[0].axvline(_best_n10, color="gray", linestyle="--", linewidth=1.2,
                label=f"Best n={_best_n10}")
for n, s in zip(_ns10, _sils_w):
    axes[0].text(n, s + 0.003, f"{s:.3f}", ha="center", va="bottom", fontsize=7)
axes[0].set_xlabel("n clusters"); axes[0].set_ylabel("Silhouette Score")
axes[0].set_title(f"Hierarchical Ward Silhouette Sweep (Best n={_best_n10})",
                  fontsize=12, fontweight="bold")
axes[0].set_xticks(_ns10); axes[0].legend()
axes[0].spines["top"].set_visible(False); axes[0].spines["right"].set_visible(False)
axes[1].plot(_ns10, _dbs_w, "s-", color="#e8473f", linewidth=2, markersize=7)
axes[1].axvline(_best_n10, color="gray", linestyle="--", linewidth=1.2)
for n, d in zip(_ns10, _dbs_w):
    axes[1].text(n, d + 0.01, f"{d:.3f}", ha="center", va="bottom", fontsize=7)
axes[1].set_xlabel("n clusters"); axes[1].set_ylabel("Davies-Bouldin Index")
axes[1].set_title("Hierarchical Ward DB Index Sweep", fontsize=12, fontweight="bold")
axes[1].set_xticks(_ns10)
axes[1].spines["top"].set_visible(False); axes[1].spines["right"].set_visible(False)
plt.suptitle("Hierarchical Ward Sweep — PBMC 10k", fontsize=13, fontweight="bold")
plt.tight_layout()
savefig("hierarchical_metrics_sweep_10k.png")

# 7. Leiden resolution sweep (line plot)
_res10  = resolutions
_nc10   = [leiden_results[r]["n_clusters"] for r in _res10]
_sil10  = [leiden_results[r]["silhouette"]  for r in _res10]
fig, ax1 = plt.subplots(figsize=(9, 4))
ax2 = ax1.twinx()
ax1.plot(_res10, _nc10,  "o-", color="steelblue", linewidth=2, label="n_clusters")
ax2.plot(_res10, _sil10, "s--", color="#e8473f",   linewidth=2, label="silhouette")
ax1.axvline(best_leiden_res, color="gray", linestyle=":", linewidth=1.5)
ax1.set_xlabel("Resolution"); ax1.set_ylabel("Number of clusters", color="steelblue")
ax2.set_ylabel("Silhouette score", color="#e8473f")
ax1.set_title(f"Leiden Resolution Sweep — PBMC 10k  (best={best_leiden_res})",
              fontsize=12, fontweight="bold")
ax1.legend(loc="upper left"); ax2.legend(loc="upper right")
plt.tight_layout()
savefig("leiden_resolution_sweep_10k.png")

# 8. All-methods UMAP panel (2 × 3 or 3 × 3 with SC3)
_panel10 = [
    ("kmeans_best",                    f"K-means (k={best_k})"),
    ("hier_ward_best",                 f"Hierarchical Ward (n={best_n})"),
    ("dbscan_best",                    f"DBSCAN (eps={best_eps})"),
    ("hdbscan_best",                   f"HDBSCAN (mcs={best_mcs})"),
    (f"leiden_{best_leiden_res}",      f"Leiden PCA (res={best_leiden_res})"),
    (f"scvi_leiden_{best_scvi_res}",   f"scVI+Leiden (res={best_scvi_res})"),
]
if sc3_metrics:
    _panel10.append((f"sc3_k{best_k_sc3}", f"SC3 (k={best_k_sc3})"))
_ncols_p10 = 3
_nrows_p10 = int(np.ceil(len(_panel10) / _ncols_p10))
fig, axes = plt.subplots(_nrows_p10, _ncols_p10, figsize=(6 * _ncols_p10, 5.5 * _nrows_p10))
axes = np.array(axes).flatten()
for _ip, (key, title) in enumerate(_panel10):
    if key in adata.obs.columns:
        sc.pl.umap(adata, color=key, ax=axes[_ip], show=False,
                   legend_loc="on data", title=title)
    else:
        axes[_ip].axis("off")
for _jp in range(_ip + 1, len(axes)):
    axes[_jp].axis("off")
plt.suptitle("All Clustering Methods — Best Result (PBMC 10k)",
             fontsize=14, fontweight="bold")
plt.tight_layout()
savefig("all_methods_umap_panel_10k.png")

# 9. Heatmap of top marker genes
try:
    _mkeys = ["CD3D", "IL7R", "CD14", "LYZ", "MS4A1",
              "CD79A", "GNLY", "NKG7", "FCGR3A", "PPBP"]
    _adata_hm = adata.raw.to_adata() if adata.raw else adata.copy()
    _adata_hm.obs["cluster"] = adata.obs[f"leiden_{best_leiden_res}"].values
    sc.tl.rank_genes_groups(_adata_hm, groupby="cluster", method="wilcoxon",
                             key_added="rank_hm", n_genes=10)
    sc.pl.heatmap(_adata_hm, var_names=_mkeys, groupby="cluster",
                  show_gene_labels=True, cmap="RdBu_r", figsize=(11, 7), show=False)
    plt.suptitle("Marker Gene Expression Heatmap — PBMC 10k", fontsize=13, fontweight="bold")
    savefig("heatmap_markers_10k.png")
except Exception as e:
    print(f"  Heatmap skipped: {e}")

# 10. Violin plots of key marker genes
try:
    _adata_vn = adata.raw.to_adata() if adata.raw else adata.copy()
    _adata_vn.obs["cluster"] = adata.obs[f"leiden_{best_leiden_res}"].values
    sc.pl.violin(_adata_vn, keys=["CD3D", "LYZ", "MS4A1", "GNLY", "PPBP"],
                 groupby="cluster", rotation=45, stripplot=False,
                 figsize=(14, 4), show=False)
    plt.suptitle("Marker Gene Expression per Cluster — PBMC 10k",
                 fontsize=13, fontweight="bold")
    savefig("violin_key_markers_10k.png")
except Exception as e:
    print(f"  Violin skipped: {e}")

# 11. Biological validation Yes/No heatmap (mirror 3k style)
_all_ct10 = list(MARKERS.keys())
_methods_v10 = list(validation_detected.keys())
_matrix_v10 = np.zeros((len(_methods_v10), len(_all_ct10)))
for _i, m in enumerate(_methods_v10):
    for _j, ct in enumerate(_all_ct10):
        _matrix_v10[_i, _j] = 1 if ct in validation_detected[m] else 0
fig, ax = plt.subplots(figsize=(16, 7))
ax.imshow(_matrix_v10, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
ax.set_xticks(range(len(_all_ct10)))
ax.set_xticklabels(_all_ct10, fontsize=9, fontweight="bold", rotation=30, ha="right")
ax.set_yticks(range(len(_methods_v10)))
ax.set_yticklabels(_methods_v10, fontsize=10)
for _i in range(len(_methods_v10)):
    for _j in range(len(_all_ct10)):
        ax.text(_j, _i, "Y" if _matrix_v10[_i, _j] == 1 else "N",
                ha="center", va="center", fontsize=10, fontweight="bold",
                color="white" if _matrix_v10[_i, _j] == 0 else "black")
ax.set_title("Biological Validation — All Methods (PBMC 10k)",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Cell Type"); ax.set_ylabel("Method")
plt.tight_layout()
savefig("biological_validation_heatmap_10k.png")

# 12. Leiden all-resolutions grid
_res_list10 = resolutions
_ncols_l10  = 5
_nrows_l10  = int(np.ceil(len(_res_list10) / _ncols_l10))
fig, axes = plt.subplots(_nrows_l10, _ncols_l10,
                         figsize=(5 * _ncols_l10, 4.5 * _nrows_l10))
axes = np.array(axes).flatten()
for _il, _res in enumerate(_res_list10):
    _k_label = leiden_results[_res]["n_clusters"]
    _s_label = leiden_results[_res]["silhouette"]
    sc.pl.umap(adata, color=f"leiden_{_res}", ax=axes[_il], show=False,
               legend_loc="on data",
               title=f"res={_res}  k={_k_label}  sil={_s_label:.3f}")
for _jl in range(_il + 1, len(axes)):
    axes[_jl].axis("off")
plt.suptitle("Leiden: Full Resolution Grid — PBMC 10k", fontsize=14, fontweight="bold")
plt.tight_layout()
savefig("leiden_all_resolutions_grid_10k.png")

# 13. Calinski-Harabász comparison
_ch_methods_10 = ["K-means", "Hier. Ward", "Leiden PCA", "scVI+Leiden"]
_ch_vals_10 = [
    kmeans_results[best_k]["calinski_harabasz"],
    hier_results[best_n]["calinski_harabasz"],
    leiden_results[best_leiden_res]["calinski_harabasz"],
    scvi_results[best_scvi_res]["calinski_harabasz"],
]
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(_ch_methods_10, _ch_vals_10,
              color=["#4e79a7", "#4e79a7", "#59a14f", "#e15759"],
              edgecolor="white", width=0.5)
for bar, v in zip(bars, _ch_vals_10):
    if not np.isnan(v):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f"{v:.0f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_ylabel("Calinski-Harabász Score (higher = better)", fontsize=11)
ax.set_title("Calinski-Harabász Score by Method — PBMC 10k", fontsize=13, fontweight="bold")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
savefig("calinski_harabasz_10k.png")

# 14. PAGA trajectory
try:
    _paga_key_10 = f"leiden_{best_leiden_res}"
    sc.tl.paga(adata, groups=_paga_key_10)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sc.pl.paga(adata, ax=axes[0], show=False, fontsize=10,
               title=f"PAGA — connectivity graph (Leiden res={best_leiden_res})")
    adata.obsm["X_umap_orig_10"] = adata.obsm["X_umap"].copy()
    sc.tl.umap(adata, init_pos="paga", random_state=SEED)
    sc.pl.umap(adata, color=_paga_key_10, ax=axes[1], show=False, legend_loc="on data",
               title=f"PAGA-initialized UMAP (Leiden res={best_leiden_res})")
    adata.obsm["X_umap"] = adata.obsm["X_umap_orig_10"]
    plt.suptitle("PAGA Trajectory — PBMC 10k", fontsize=13, fontweight="bold")
    plt.tight_layout()
    savefig("paga_10k.png")
except Exception as e:
    print(f"  PAGA skipped: {e}")

# 15. Standardized summary CSV
_std_10k = pd.DataFrame({
    "dataset":   "PBMC 10k",
    "method":    ["K-means", "Hier. Ward", "DBSCAN", "HDBSCAN", "Leiden PCA", "scVI+Leiden"],
    "n_clusters":[kmeans_results[best_k]["labels"].max() + 1,
                  hier_results[best_n]["labels"].max() + 1,
                  dbscan_results[best_eps]["n_clusters"],
                  hdbscan_results[best_mcs]["n_clusters"],
                  leiden_results[best_leiden_res]["n_clusters"],
                  scvi_results[best_scvi_res]["n_clusters"]],
    "silhouette":[kmeans_results[best_k]["silhouette"],
                  hier_results[best_n]["silhouette"],
                  dbscan_results[best_eps]["silhouette"],
                  hdbscan_results[best_mcs]["silhouette"],
                  leiden_results[best_leiden_res]["silhouette"],
                  scvi_results[best_scvi_res]["silhouette"]],
    "davies_bouldin":[kmeans_results[best_k]["davies_bouldin"],
                      hier_results[best_n]["davies_bouldin"],
                      dbscan_results[best_eps]["davies_bouldin"],
                      hdbscan_results[best_mcs]["davies_bouldin"],
                      leiden_results[best_leiden_res]["davies_bouldin"],
                      scvi_results[best_scvi_res]["davies_bouldin"]],
    "calinski_harabasz":[kmeans_results[best_k]["calinski_harabasz"],
                         hier_results[best_n]["calinski_harabasz"],
                         np.nan, np.nan,
                         leiden_results[best_leiden_res]["calinski_harabasz"],
                         scvi_results[best_scvi_res]["calinski_harabasz"]],
})
_std_10k.to_csv(RES_DIR / "summary_10k.csv", index=False)

# ═══════════════════════════════════════════════════════════════════════════
# COMBINED RESULTS: merge 3k, 3k-MAGIC, and 10k summaries
# ═══════════════════════════════════════════════════════════════════════════
print("\n── Generating combined results CSV ──")
_summary_files = [
    RES_DIR / "summary_3k.csv",
    RES_DIR / "summary_magic.csv",
    RES_DIR / "summary_10k.csv",
]
_available = [f for f in _summary_files if f.exists()]
if _available:
    _combined = pd.concat([pd.read_csv(f) for f in _available], ignore_index=True)
    _combined.to_csv(RES_DIR / "results_combined.csv", index=False)
    print(f"Combined results saved to {RES_DIR}/results_combined.csv")
    print(_combined.to_string(index=False))

    # Combined silhouette comparison bar chart (all datasets × all methods)
    _datasets    = _combined["dataset"].unique()
    _methods_cb  = _combined["method"].unique()
    _x_cb = np.arange(len(_methods_cb))
    _w_cb = 0.25
    _ds_colors = ["steelblue", "#e8473f", "seagreen"]
    fig, ax = plt.subplots(figsize=(16, 6))
    for _di, (ds, color) in enumerate(zip(_datasets, _ds_colors)):
        _ds_df = _combined[_combined["dataset"] == ds].set_index("method")
        _sil_vals = [_ds_df.loc[m, "silhouette"] if m in _ds_df.index and not pd.isna(_ds_df.loc[m, "silhouette"]) else 0
                     for m in _methods_cb]
        bars = ax.bar(_x_cb + _di * _w_cb, _sil_vals, _w_cb,
                      label=ds, color=color, alpha=0.85, edgecolor="white")
        for bar, v in zip(bars, _sil_vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=7, rotation=90)
    ax.set_xticks(_x_cb + _w_cb)
    ax.set_xticklabels(_methods_cb, rotation=15, ha="right", fontsize=10)
    ax.set_ylabel("Silhouette Score (higher = better)", fontsize=11)
    ax.set_title("Silhouette Score: All Datasets × All Methods", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    savefig("combined_silhouette_comparison.png")

    # Combined Calinski-Harabász chart
    fig, ax = plt.subplots(figsize=(16, 6))
    for _di, (ds, color) in enumerate(zip(_datasets, _ds_colors)):
        _ds_df = _combined[_combined["dataset"] == ds].set_index("method")
        _ch_vals_cb = [_ds_df.loc[m, "calinski_harabasz"]
                       if m in _ds_df.index and "calinski_harabasz" in _ds_df.columns
                       and not pd.isna(_ds_df.loc[m, "calinski_harabasz"]) else 0
                       for m in _methods_cb]
        bars = ax.bar(_x_cb + _di * _w_cb, _ch_vals_cb, _w_cb,
                      label=ds, color=color, alpha=0.85, edgecolor="white")
        for bar, v in zip(bars, _ch_vals_cb):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f"{v:.0f}", ha="center", va="bottom", fontsize=7, rotation=90)
    ax.set_xticks(_x_cb + _w_cb)
    ax.set_xticklabels(_methods_cb, rotation=15, ha="right", fontsize=10)
    ax.set_ylabel("Calinski-Harabász Score (higher = better)", fontsize=11)
    ax.set_title("Calinski-Harabász Score: All Datasets × All Methods", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    savefig("combined_calinski_harabasz_comparison.png")
else:
    print("  No summary CSVs found yet — run 3k and MAGIC pipelines first.")

print(f"\nDone. Figures saved to: {FIG_DIR}")
print(f"Results saved to:       {RES_DIR}")
print("\nNote: DBSCAN scores are in UMAP space and are not directly comparable to PCA-based methods.")
