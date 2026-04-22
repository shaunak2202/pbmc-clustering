"""
PBMC 3k Clustering Pipeline
Full local version — no Google Colab or Google Drive required.

Run order:
  1. python pbmc_3k.py
  2. Rscript sc3_3k.R          (produces sc3_labels.csv in project root)
  3. python pbmc_3k.py          (re-run; picks up SC3 labels automatically)

Dependencies (install once):
  pip install scanpy==1.10.1 scvi-tools leidenalg==0.10.2 igraph==0.11.4
              scrublet==0.2.3 umap-learn==0.5.6 hdbscan scikit-learn
              matplotlib seaborn pandas numpy anndata
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # headless — remove this line for interactive plots
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
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
from sklearn.utils import resample as sklearn_resample

os.environ["JAX_PLATFORM_NAME"] = "cpu"

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
DATA_3K     = BASE_DIR / "hg19"
CKPT_3K     = BASE_DIR / "pbmc3k_preprocessed.h5ad"
SCVI_3K     = BASE_DIR / "pbmc3k_scvi.h5ad"
SC3_LABELS  = BASE_DIR / "sc3_labels.csv"
FIG_DIR     = Path(__file__).resolve().parent / "figures" / "3k"
RES_DIR     = Path(__file__).resolve().parent / "results"

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
if CKPT_3K.exists():
    print(f"Loading preprocessed checkpoint: {CKPT_3K}")
    adata = sc.read_h5ad(CKPT_3K)
    print(f"Loaded: {adata.shape[0]} cells x {adata.shape[1]} genes")
    skip_preprocessing = True
else:
    print(f"Loading raw PBMC 3k data from: {DATA_3K}")
    adata = sc.read_10x_mtx(str(DATA_3K), var_names="gene_symbols", cache=True)
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
    plt.suptitle("PBMC 3k - Pre-filter QC", fontsize=13, fontweight="bold")
    plt.tight_layout(); savefig("qc_prefilter.png")

    MIN_GENES, MAX_GENES, MAX_MT_PCT = 200, 2500, 5.0
    print(f"Before filtering: {adata.n_obs} cells")
    sc.pp.filter_cells(adata, min_genes=MIN_GENES)
    sc.pp.filter_cells(adata, max_genes=MAX_GENES)
    adata = adata[adata.obs["pct_counts_mt"] < MAX_MT_PCT].copy()
    sc.pp.filter_genes(adata, min_cells=3)
    print(f"After filtering:  {adata.n_obs} cells, {adata.n_vars} genes")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 3: Preprocessing (normalize → log → HVG → scale → PCA → UMAP)
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
    plt.suptitle("Highly Variable Genes", fontsize=13, fontweight="bold")
    plt.tight_layout(); savefig("hvg.png")

    adata = adata[:, adata.var["highly_variable"]].copy()
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, svd_solver="arpack", n_comps=50, random_state=SEED)

    pca_var = adata.uns["pca"]["variance_ratio"]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(1, 21), pca_var[:20], color="steelblue", edgecolor="white")
    ax.plot(range(1, 21), pca_var[:20], "o-", color="#e8473f", markersize=5)
    ax.set_xlabel("PC ranking"); ax.set_ylabel("Variance ratio")
    ax.set_title("PCA Variance Ratio (Top 20 PCs)")
    plt.tight_layout(); savefig("pca_variance.png")

    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=20, random_state=SEED)
    sc.tl.umap(adata, random_state=SEED)
    print(f"After HVG selection: {adata.shape[0]} cells x {adata.shape[1]} genes")

    adata.write(CKPT_3K)
    print(f"Saved preprocessed checkpoint: {CKPT_3K}")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 4: K-means Clustering
# ═══════════════════════════════════════════════════════════════════════════
print("\n── K-means Clustering ──")
X_pca = adata.obsm["X_pca"][:, :20]
k_range = range(3, 11)
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

fig, axes = plt.subplots(1, 4, figsize=(20, 4))
for ax, k in zip(axes, [4, 6, 8, 10]):
    adata.obs[f"kmeans_{k}"] = pd.Categorical(kmeans_results[k]["labels"].astype(str))
    sc.pl.umap(adata, color=f"kmeans_{k}", ax=ax, show=False,
               title=f"K-means k={k}", legend_loc="on data")
plt.suptitle("K-means Clustering on PBMC 3k", fontsize=13, fontweight="bold")
plt.tight_layout(); savefig("kmeans_umap.png")

summary_km = pd.DataFrame({
    "k": list(k_range),
    "silhouette": [kmeans_results[k]["silhouette"] for k in k_range],
    "davies_bouldin": [kmeans_results[k]["davies_bouldin"] for k in k_range],
})
print("\nK-means Summary:"); print(summary_km.to_string(index=False))

# ═══════════════════════════════════════════════════════════════════════════
# STEP 5: Hierarchical Clustering
# ═══════════════════════════════════════════════════════════════════════════
print("\n── Hierarchical Clustering ──")
linkages = ["ward", "average"]
n_clusters_range = range(3, 11)
hier_results = {}

for linkage in linkages:
    hier_results[linkage] = {}
    for n in n_clusters_range:
        model = AgglomerativeClustering(n_clusters=n, linkage=linkage)
        labels = model.fit_predict(X_pca)
        sil = silhouette_score(X_pca, labels)
        db  = davies_bouldin_score(X_pca, labels)
        ch  = calinski_harabasz_score(X_pca, labels)
        hier_results[linkage][n] = {"labels": labels, "silhouette": sil, "davies_bouldin": db, "calinski_harabasz": ch}
        print(f"  linkage={linkage} | n={n} | Silhouette: {sil:.4f} | DB: {db:.4f} | CH: {ch:.1f}")
    print()

for linkage in linkages:
    best_n = max(hier_results[linkage], key=lambda n: hier_results[linkage][n]["silhouette"])
    print(f"Best n for {linkage} linkage: {best_n}")
    adata.obs[f"hier_{linkage}_best"] = pd.Categorical(
        hier_results[linkage][best_n]["labels"].astype(str))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, linkage in zip(axes, linkages):
    best_n = max(hier_results[linkage], key=lambda n: hier_results[linkage][n]["silhouette"])
    sc.pl.umap(adata, color=f"hier_{linkage}_best", ax=ax, show=False,
               title=f"Hierarchical ({linkage}, n={best_n})", legend_loc="on data")
plt.suptitle("Hierarchical Clustering on PBMC 3k", fontsize=13, fontweight="bold")
plt.tight_layout(); savefig("hierarchical_umap.png")

rows = []
for linkage in linkages:
    for n in n_clusters_range:
        rows.append({"linkage": linkage, "n_clusters": n,
                     "silhouette": hier_results[linkage][n]["silhouette"],
                     "davies_bouldin": hier_results[linkage][n]["davies_bouldin"]})
print("\nHierarchical Summary:"); print(pd.DataFrame(rows).to_string(index=False))

# ═══════════════════════════════════════════════════════════════════════════
# STEP 6: DBSCAN and HDBSCAN
# ═══════════════════════════════════════════════════════════════════════════
print("\n── DBSCAN / HDBSCAN ──")
X_umap = adata.obsm["X_umap"]

# DBSCAN on UMAP
eps_range = [0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
dbscan_results = {}
for eps in eps_range:
    model  = DBSCAN(eps=eps, min_samples=5)
    labels = model.fit_predict(X_umap)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_pct  = (labels == -1).sum() / len(labels) * 100
    if n_clusters > 1:
        mask = labels != -1
        sil = silhouette_score(X_umap[mask], labels[mask])
        db  = davies_bouldin_score(X_umap[mask], labels[mask])
    else:
        sil, db = -1, -1
    dbscan_results[eps] = {"labels": labels, "n_clusters": n_clusters,
                           "noise_pct": noise_pct, "silhouette": sil, "davies_bouldin": db}
    print(f"  DBSCAN eps={eps} | clusters={n_clusters} | noise={noise_pct:.1f}% | "
          f"Silhouette: {sil:.4f} | DB: {db:.4f}")

# HDBSCAN on PCA
min_cluster_sizes = [10, 20, 30, 50, 75]
hdbscan_results = {}
for mcs in min_cluster_sizes:
    clusterer  = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=5)
    labels     = clusterer.fit_predict(X_pca)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_pct  = (labels == -1).sum() / len(labels) * 100
    if n_clusters > 1:
        mask = labels != -1
        sil = silhouette_score(X_pca[mask], labels[mask])
        db  = davies_bouldin_score(X_pca[mask], labels[mask])
    else:
        sil, db = -1, -1
    hdbscan_results[mcs] = {"labels": labels, "n_clusters": n_clusters,
                             "noise_pct": noise_pct, "silhouette": sil, "davies_bouldin": db}
    print(f"  HDBSCAN mcs={mcs} | clusters={n_clusters} | noise={noise_pct:.1f}% | "
          f"Silhouette: {sil:.4f} | DB: {db:.4f}")

valid_dbscan  = {e: v for e, v in dbscan_results.items() if v["silhouette"] > 0}
valid_hdbscan = {m: v for m, v in hdbscan_results.items() if v["silhouette"] > 0}
best_eps = max(valid_dbscan, key=lambda e: valid_dbscan[e]["silhouette"])
best_mcs = max(valid_hdbscan, key=lambda m: valid_hdbscan[m]["silhouette"])
print(f"\nBest DBSCAN eps={best_eps} | Best HDBSCAN mcs={best_mcs}")

adata.obs["dbscan_best"]  = pd.Categorical(dbscan_results[best_eps]["labels"].astype(str))
adata.obs["hdbscan_best"] = pd.Categorical(hdbscan_results[best_mcs]["labels"].astype(str))

fig, axes = plt.subplots(1, 4, figsize=(22, 5))
for ax, eps in zip(axes[:2], [0.5, best_eps]):
    col = f"dbscan_eps_{eps}"
    adata.obs[col] = pd.Categorical(dbscan_results[eps]["labels"].astype(str))
    sc.pl.umap(adata, color=col, ax=ax, show=False, legend_loc="on data",
               title=f"DBSCAN eps={eps}\n{dbscan_results[eps]['n_clusters']} clusters | "
                     f"noise={dbscan_results[eps]['noise_pct']:.1f}% | "
                     f"sil={dbscan_results[eps]['silhouette']:.2f}")
for ax, mcs in zip(axes[2:], [10, 50]):
    col = f"hdbscan_mcs_{mcs}"
    adata.obs[col] = pd.Categorical(hdbscan_results[mcs]["labels"].astype(str))
    sc.pl.umap(adata, color=col, ax=ax, show=False, legend_loc="on data",
               title=f"HDBSCAN mcs={mcs}\n{hdbscan_results[mcs]['n_clusters']} clusters | "
                     f"noise={hdbscan_results[mcs]['noise_pct']:.1f}% | "
                     f"sil={hdbscan_results[mcs]['silhouette']:.2f}")
plt.suptitle("Density-based Clustering on PBMC 3k", fontsize=13, fontweight="bold")
plt.tight_layout(); savefig("dbscan_hdbscan_umap.png")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 7: Leiden Clustering
# ═══════════════════════════════════════════════════════════════════════════
print("\n── Leiden Clustering ──")
resolutions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.5]
leiden_results = {}

for res in resolutions:
    sc.tl.leiden(adata, resolution=res, random_state=SEED, key_added=f"leiden_{res}")
    labels     = adata.obs[f"leiden_{res}"].astype(int).values
    n_clusters = len(set(labels))
    sil = silhouette_score(X_pca, labels)
    db  = davies_bouldin_score(X_pca, labels)
    ch  = calinski_harabasz_score(X_pca, labels)
    leiden_results[res] = {"n_clusters": n_clusters, "silhouette": sil, "davies_bouldin": db, "calinski_harabasz": ch}
    print(f"  res={res} | clusters={n_clusters} | Silhouette: {sil:.4f} | DB: {db:.4f} | CH: {ch:.1f}")

best_res = max(leiden_results, key=lambda r: leiden_results[r]["silhouette"])
print(f"\nBest resolution: {best_res} ({leiden_results[best_res]['n_clusters']} clusters)")

plot_res = sorted(set([0.3, 0.5, 0.8, best_res]))
fig, axes = plt.subplots(1, len(plot_res), figsize=(6 * len(plot_res), 5))
if len(plot_res) == 1: axes = [axes]
for ax, res in zip(axes, plot_res):
    sc.pl.umap(adata, color=f"leiden_{res}", ax=ax, show=False, legend_loc="on data",
               title=f"Leiden res={res}\n{leiden_results[res]['n_clusters']} clusters | "
                     f"sil={leiden_results[res]['silhouette']:.2f}")
plt.suptitle("Leiden Clustering on PBMC 3k", fontsize=13, fontweight="bold")
plt.tight_layout(); savefig("leiden_umap.png")

fig, ax1 = plt.subplots(figsize=(8, 4))
ax2 = ax1.twinx()
ax1.plot(resolutions, [leiden_results[r]["n_clusters"] for r in resolutions],
         "o-", color="steelblue", label="n_clusters")
ax2.plot(resolutions, [leiden_results[r]["silhouette"] for r in resolutions],
         "s--", color="#e8473f", label="silhouette")
ax1.set_xlabel("Resolution"); ax1.set_ylabel("Number of clusters", color="steelblue")
ax2.set_ylabel("Silhouette score", color="#e8473f")
ax1.set_title("Leiden: Resolution vs Cluster Count and Silhouette")
ax1.legend(loc="upper left"); ax2.legend(loc="upper right")
plt.tight_layout(); savefig("leiden_resolution_sweep.png")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 8: scVI + Leiden
# ═══════════════════════════════════════════════════════════════════════════
print("\n── scVI + Leiden ──")
if SCVI_3K.exists():
    print(f"Loading scVI checkpoint: {SCVI_3K}")
    adata_scvi = sc.read_h5ad(SCVI_3K)
else:
    adata_scvi = adata.copy()
    adata_scvi.X = adata_scvi.layers["counts"]
    scvi.model.SCVI.setup_anndata(adata_scvi)
    model = scvi.model.SCVI(adata_scvi, n_layers=2, n_latent=30)
    print("Training scVI model...")
    model.train(max_epochs=200, early_stopping=True)
    adata_scvi.obsm["X_scVI"] = model.get_latent_representation()
    print(f"scVI latent space shape: {adata_scvi.obsm['X_scVI'].shape}")
    sc.pp.neighbors(adata_scvi, use_rep="X_scVI", n_neighbors=15, random_state=SEED)
    sc.tl.umap(adata_scvi, random_state=SEED)

    scvi_resolutions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
    for res in scvi_resolutions:
        sc.tl.leiden(adata_scvi, resolution=res, random_state=SEED,
                     key_added=f"scvi_leiden_{res}")

    adata_scvi.write(SCVI_3K)
    print(f"Saved scVI checkpoint: {SCVI_3K}")

X_scvi = adata_scvi.obsm["X_scVI"]
scvi_resolutions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
scvi_leiden_results = {}

for res in scvi_resolutions:
    key = f"scvi_leiden_{res}"
    if key not in adata_scvi.obs.columns:
        sc.tl.leiden(adata_scvi, resolution=res, random_state=SEED, key_added=key)
    labels     = adata_scvi.obs[key].astype(int).values
    n_clusters = len(set(labels))
    sil = silhouette_score(X_scvi, labels)
    db  = davies_bouldin_score(X_scvi, labels)
    ch  = calinski_harabasz_score(X_scvi, labels)
    scvi_leiden_results[res] = {"n_clusters": n_clusters, "silhouette": sil, "davies_bouldin": db, "calinski_harabasz": ch}
    print(f"  res={res} | clusters={n_clusters} | Silhouette: {sil:.4f} | DB: {db:.4f} | CH: {ch:.1f}")

best_scvi_res = max(scvi_leiden_results, key=lambda r: scvi_leiden_results[r]["silhouette"])
print(f"\nBest scVI resolution: {best_scvi_res}")

# Add scVI neighbors key for stability analysis
if "scvi_neighbors" not in adata_scvi.uns:
    sc.pp.neighbors(adata_scvi, use_rep="X_scVI", n_neighbors=15,
                    key_added="scvi_neighbors", random_state=SEED)

plot_res = sorted(set([0.3, 0.5, 0.8, best_scvi_res]))
fig, axes = plt.subplots(1, len(plot_res), figsize=(6 * len(plot_res), 5))
if len(plot_res) == 1: axes = [axes]
for ax, res in zip(axes, plot_res):
    sc.pl.umap(adata_scvi, color=f"scvi_leiden_{res}", ax=ax, show=False,
               legend_loc="on data",
               title=f"scVI+Leiden res={res}\n"
                     f"{scvi_leiden_results[res]['n_clusters']} clusters | "
                     f"sil={scvi_leiden_results[res]['silhouette']:.2f}")
plt.suptitle("scVI + Leiden Clustering on PBMC 3k", fontsize=13, fontweight="bold")
plt.tight_layout(); savefig("scvi_leiden_umap.png")

fig, ax1 = plt.subplots(figsize=(8, 4))
ax2 = ax1.twinx()
ax1.plot(scvi_resolutions, [scvi_leiden_results[r]["n_clusters"] for r in scvi_resolutions],
         "o-", color="steelblue", label="n_clusters")
ax2.plot(scvi_resolutions, [scvi_leiden_results[r]["silhouette"] for r in scvi_resolutions],
         "s--", color="#e8473f", label="silhouette")
ax1.set_xlabel("Resolution"); ax1.set_ylabel("Number of clusters", color="steelblue")
ax2.set_ylabel("Silhouette score", color="#e8473f")
ax1.set_title("scVI+Leiden: Resolution vs Cluster Count and Silhouette")
ax1.legend(loc="upper left"); ax2.legend(loc="upper right")
plt.tight_layout(); savefig("scvi_leiden_resolution_sweep.png")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 9: Stability Analysis (Seed + Resampling)
# ═══════════════════════════════════════════════════════════════════════════
print("\n── Stability Analysis ──")
N_RUNS = 10
SUBSAMPLE_FRAC = 0.8
seeds = list(range(N_RUNS))
stability_results = {}

def pairwise_ari(labels_list):
    aris = []
    for i in range(len(labels_list)):
        for j in range(i + 1, len(labels_list)):
            aris.append(adjusted_rand_score(labels_list[i], labels_list[j]))
    return np.mean(aris), np.std(aris)

# K-means seed stability (k=4)
print("K-means seed stability (k=4)...")
km_seed_labels = [KMeans(n_clusters=4, random_state=s, n_init=10).fit_predict(X_pca)
                  for s in seeds]
mean_ari, std_ari = pairwise_ari(km_seed_labels)
stability_results["KMeans_seed"] = {"mean_ARI": mean_ari, "std_ARI": std_ari}
print(f"  Mean ARI: {mean_ari:.4f} +/- {std_ari:.4f}")

# K-means resampling stability (k=4)
print("K-means resampling stability (k=4)...")
base_km = KMeans(n_clusters=4, random_state=SEED, n_init=10).fit_predict(X_pca)
km_resample_aris = []
for s in seeds:
    np.random.seed(s)
    idx = np.random.choice(len(X_pca), int(len(X_pca) * SUBSAMPLE_FRAC), replace=False)
    sub_labels = KMeans(n_clusters=4, random_state=s, n_init=10).fit_predict(X_pca[idx])
    km_resample_aris.append(adjusted_rand_score(base_km[idx], sub_labels))
stability_results["KMeans_resample"] = {"mean_ARI": np.mean(km_resample_aris),
                                        "std_ARI": np.std(km_resample_aris)}
print(f"  Mean ARI: {np.mean(km_resample_aris):.4f} +/- {np.std(km_resample_aris):.4f}")

# Hierarchical Ward stability (n=5)
print("Hierarchical Ward stability (n=5)...")
base_hier = AgglomerativeClustering(n_clusters=5, linkage="ward").fit_predict(X_pca)
hier_aris = []
for s in seeds:
    np.random.seed(s)
    idx = np.random.choice(len(X_pca), int(len(X_pca) * SUBSAMPLE_FRAC), replace=False)
    sub_labels = AgglomerativeClustering(n_clusters=5, linkage="ward").fit_predict(X_pca[idx])
    hier_aris.append(adjusted_rand_score(base_hier[idx], sub_labels))
stability_results["Hierarchical_seed"]     = {"mean_ARI": 1.0, "std_ARI": 0.0}
stability_results["Hierarchical_resample"] = {"mean_ARI": np.mean(hier_aris),
                                              "std_ARI": np.std(hier_aris)}
print(f"  Seed ARI: 1.0 (deterministic)")
print(f"  Resample Mean ARI: {np.mean(hier_aris):.4f} +/- {np.std(hier_aris):.4f}")

# DBSCAN resampling stability (eps=0.5)
print("DBSCAN resampling stability (eps=0.5)...")
base_db = DBSCAN(eps=0.5, min_samples=5).fit_predict(X_umap)
db_aris = []
for s in seeds:
    np.random.seed(s)
    idx = np.random.choice(len(X_umap), int(len(X_umap) * SUBSAMPLE_FRAC), replace=False)
    sub_labels = DBSCAN(eps=0.5, min_samples=5).fit_predict(X_umap[idx])
    mask = sub_labels != -1
    if mask.sum() > 10:
        db_aris.append(adjusted_rand_score(base_db[idx][mask], sub_labels[mask]))
stability_results["DBSCAN_seed"]     = {"mean_ARI": 1.0, "std_ARI": 0.0}
stability_results["DBSCAN_resample"] = {"mean_ARI": np.mean(db_aris),
                                        "std_ARI": np.std(db_aris)}
print(f"  Seed ARI: 1.0 (deterministic)")
print(f"  Resample Mean ARI: {np.mean(db_aris):.4f} +/- {np.std(db_aris):.4f}")

# HDBSCAN stability (mcs=10)
print("HDBSCAN stability (min_cluster_size=10)...")
best_ms = 10
base_hdb = hdbscan.HDBSCAN(min_cluster_size=best_ms, min_samples=best_ms,
                             core_dist_n_jobs=-1).fit_predict(X_pca)
noise_pct = (base_hdb == -1).sum() / len(base_hdb) * 100
print(f"  Base noise: {noise_pct:.1f}% cells assigned as noise")
hdb_aris = []
for s in seeds:
    np.random.seed(s)
    idx = np.random.choice(len(X_pca), int(len(X_pca) * SUBSAMPLE_FRAC), replace=False)
    sub_labels = hdbscan.HDBSCAN(min_cluster_size=best_ms, min_samples=best_ms,
                                  core_dist_n_jobs=-1).fit_predict(X_pca[idx])
    mask_base = base_hdb[idx] != -1
    mask_sub  = sub_labels != -1
    mask = mask_base & mask_sub
    if mask.sum() > 10:
        hdb_aris.append(adjusted_rand_score(base_hdb[idx][mask], sub_labels[mask]))
stability_results["HDBSCAN_seed"]     = {"mean_ARI": 1.0, "std_ARI": 0.0}
stability_results["HDBSCAN_resample"] = {
    "mean_ARI": np.mean(hdb_aris) if hdb_aris else 0.0,
    "std_ARI":  np.std(hdb_aris) if hdb_aris else 0.0,
}
print(f"  Seed ARI: 1.0 (deterministic)")
if hdb_aris:
    print(f"  Resample Mean ARI: {np.mean(hdb_aris):.4f} +/- {np.std(hdb_aris):.4f}")

# Leiden seed stability (res=0.3)
print("Leiden seed stability (res=0.3)...")
leiden_seed_labels = []
for s in seeds:
    sc.tl.leiden(adata, resolution=0.3, random_state=s, key_added=f"leiden_seed_{s}")
    leiden_seed_labels.append(adata.obs[f"leiden_seed_{s}"].astype(int).values)
mean_ari, std_ari = pairwise_ari(leiden_seed_labels)
stability_results["Leiden_seed"] = {"mean_ARI": mean_ari, "std_ARI": std_ari}
print(f"  Mean ARI: {mean_ari:.4f} +/- {std_ari:.4f}")

# Leiden resampling stability (res=0.3)
print("Leiden resampling stability (res=0.3)...")
base_leiden = adata.obs["leiden_0.3"].astype(int).values
leiden_resample_aris = []
for s in seeds:
    np.random.seed(s)
    idx = np.random.choice(adata.n_obs, int(adata.n_obs * SUBSAMPLE_FRAC), replace=False)
    adata_sub = adata[idx].copy()
    sc.pp.neighbors(adata_sub, n_neighbors=15, n_pcs=20, random_state=s)
    sc.tl.leiden(adata_sub, resolution=0.3, random_state=s, key_added="leiden_sub")
    leiden_resample_aris.append(
        adjusted_rand_score(base_leiden[idx],
                            adata_sub.obs["leiden_sub"].astype(int).values))
stability_results["Leiden_resample"] = {"mean_ARI": np.mean(leiden_resample_aris),
                                        "std_ARI": np.std(leiden_resample_aris)}
print(f"  Mean ARI: {np.mean(leiden_resample_aris):.4f} +/- {np.std(leiden_resample_aris):.4f}")

# scVI+Leiden seed stability (res=0.6)
print("scVI+Leiden seed stability (res=0.6)...")
ref_scvi  = adata_scvi.obs["scvi_leiden_0.6"].astype(int).values
N_scvi    = len(X_scvi)
scvi_seed_aris = []
for s in seeds:
    sc.tl.leiden(adata_scvi, resolution=0.6, random_state=s,
                 neighbors_key="scvi_neighbors", key_added=f"scvi_seed_{s}")
    ari = adjusted_rand_score(ref_scvi, adata_scvi.obs[f"scvi_seed_{s}"].astype(int).values)
    scvi_seed_aris.append(ari)
    print(f"  Seed {s}: ARI={ari:.4f}")
stability_results["scVI_Leiden_seed"] = {"mean_ARI": np.mean(scvi_seed_aris),
                                         "std_ARI":  np.std(scvi_seed_aris)}

print("scVI+Leiden resampling stability (res=0.6)...")
scvi_resample_aris = []
for i in range(10):
    idx = sklearn_resample(np.arange(N_scvi), n_samples=int(0.8 * N_scvi),
                           random_state=i, replace=False)
    adata_sub = anndata.AnnData(X=X_scvi[idx])
    sc.pp.neighbors(adata_sub, n_neighbors=15, use_rep="X", random_state=SEED)
    sc.tl.leiden(adata_sub, resolution=0.6, random_state=SEED, key_added="leiden_sub")
    ari = adjusted_rand_score(ref_scvi[idx], adata_sub.obs["leiden_sub"].astype(int).values)
    scvi_resample_aris.append(ari)
    print(f"  Resample run {i+1}/10: ARI={ari:.4f}")
stability_results["scVI_Leiden_resample"] = {"mean_ARI": np.mean(scvi_resample_aris),
                                             "std_ARI":  np.std(scvi_resample_aris)}
print(f"  Resample Mean ARI: {np.mean(scvi_resample_aris):.4f} +/- {np.std(scvi_resample_aris):.4f}")
print(f"  Stable: {'YES' if np.mean(scvi_resample_aris) >= 0.8 else 'NO'}")

summary_stab = pd.DataFrame(stability_results).T.reset_index()
summary_stab.columns = ["method", "mean_ARI", "std_ARI"]
print("\nStability Summary:"); print(summary_stab.to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 8))
colors = ["steelblue" if "seed" in m.lower() else "#e8473f"
          for m in summary_stab["method"]]
ax.barh(summary_stab["method"], summary_stab["mean_ARI"],
        xerr=summary_stab["std_ARI"], color=colors,
        edgecolor="white", capsize=4, alpha=0.85)
ax.axvline(x=0.8, color="gray", linestyle="--", linewidth=1.5)
ax.set_xlabel("Mean ARI")
ax.set_title("Stability Analysis - Seed and Resampling (PBMC 3k)",
             fontsize=13, fontweight="bold")
ax.set_xlim(0, 1.1)
ax.legend(handles=[mpatches.Patch(facecolor="steelblue", label="Seed ARI"),
                   mpatches.Patch(facecolor="#e8473f", label="Resample ARI")],
          loc="lower right")
plt.tight_layout(); savefig("stability_summary.png")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 10: Cell Type Composition
# ═══════════════════════════════════════════════════════════════════════════
print("\n── Cell Type Composition ──")
cluster_labels_3k = {
    "0": "T cells", "1": "Monocytes (CD14+)", "2": "NK cells",
    "3": "B cells", "4": "Monocytes (CD16+)", "5": "Platelets",
}
cluster_counts = adata.obs["leiden_0.3"].value_counts().sort_index()
cluster_names  = [cluster_labels_3k.get(str(c), f"Cluster {c}") for c in cluster_counts.index]
cluster_pcts   = cluster_counts / cluster_counts.sum() * 100
colors_bar = ["steelblue", "#e8473f", "seagreen", "salmon", "mediumpurple", "darkorange"]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(cluster_names, cluster_pcts, color=colors_bar, edgecolor="white", alpha=0.85)
ax.set_xlabel("Percentage of cells (%)", fontsize=11)
ax.set_title("Cell Type Composition - PBMC 3k (Leiden res=0.3)",
             fontsize=13, fontweight="bold")
ax.set_xlim(0, max(cluster_pcts) + 12)
for bar, pct, count in zip(bars, cluster_pcts, cluster_counts):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{pct:.1f}% ({count} cells)", va="center", fontsize=10)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout(); savefig("cell_composition_3k.png")

print("Cell Type Composition Summary - PBMC 3k:")
for name, count, pct in zip(cluster_names, cluster_counts, cluster_pcts):
    print(f"  {name:<25} {count:>6} cells  {pct:.1f}%")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 11: Marker Gene Validation (Leiden res=0.3)
# ═══════════════════════════════════════════════════════════════════════════
print("\n── Marker Gene Validation ──")
cluster_labels = {
    "0": "T cells", "1": "Monocytes", "2": "NK cells",
    "3": "B cells", "4": "Monocytes (CD16+)", "5": "Platelets",
}
adata.obs["best_clustering"] = adata.obs["leiden_0.3"]
adata_markers = adata.raw.to_adata()
adata_markers.obs["best_clustering"] = adata.obs["leiden_0.3"]
sc.tl.rank_genes_groups(adata_markers, groupby="best_clustering",
                         method="wilcoxon", key_added="rank_genes", n_genes=20)

n_genes_show = 5
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()
for i, cluster in enumerate(sorted(adata_markers.obs["best_clustering"].unique())):
    df     = sc.get.rank_genes_groups_df(adata_markers, group=cluster,
                                          key="rank_genes").head(n_genes_show)
    genes  = df["names"].tolist()
    scores = df["scores"].tolist()
    ax = axes[i]
    colors_c = plt.cm.RdYlBu_r(np.linspace(0.3, 0.9, n_genes_show))
    ax.bar(range(n_genes_show), scores, color=colors_c, edgecolor="white", width=0.6)
    for j, (gene, score) in enumerate(zip(genes, scores)):
        ax.text(j, score + max(scores) * 0.01, gene,
                ha="center", va="bottom", fontsize=9, fontweight="bold", rotation=15)
    ax.set_xticks(range(n_genes_show))
    ax.set_xticklabels([f"Rank {k+1}" for k in range(n_genes_show)], fontsize=9)
    ax.set_xlabel("Gene Rank (1 = most specific)", fontsize=10)
    ax.set_ylabel("Wilcoxon Score", fontsize=9)
    ax.set_title(f"Cluster {cluster} — {cluster_labels.get(cluster, '?')} vs rest",
                 fontsize=11, fontweight="bold")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.suptitle("Top 5 Marker Genes per Cluster (Leiden res=0.3)", fontsize=14, fontweight="bold")
plt.tight_layout(); savefig("marker_genes_per_cluster.png")

known_markers = {
    "T cells":   ["CD3D", "CD3E", "IL7R", "CD4", "CD8A"],
    "NK cells":  ["GNLY", "NKG7", "GZMB", "GZMA", "PRF1"],
    "B cells":   ["MS4A1", "CD79A", "CD79B", "CD19", "BANK1"],
    "Monocytes": ["CD14", "LYZ", "CST3", "FCGR3A", "MS4A7"],
    "Dendritic": ["FCER1A", "CST3", "IL3RA", "CLEC4C"],
    "Platelets": ["PPBP", "PF4", "GP1BB"],
}

print("Top marker genes per cluster:")
cluster_annotations = {}
for cluster in sorted(adata_markers.obs["best_clustering"].unique()):
    genes = sc.get.rank_genes_groups_df(adata_markers, group=cluster,
                                         key="rank_genes")["names"].tolist()[:10]
    print(f"Cluster {cluster}: {', '.join(genes)}")
    matched = False
    for cell_type, markers in known_markers.items():
        overlap = [g for g in genes if g in markers]
        if overlap:
            print(f"    {cell_type}: {overlap}")
            if not matched:
                cluster_annotations[cluster] = cell_type
                matched = True
    if not matched:
        cluster_annotations[cluster] = "Unknown"
    print()

adata.obs["cell_type"] = adata.obs["best_clustering"].map(
    {k: f"{v} ({k})" for k, v in cluster_annotations.items()})

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sc.pl.umap(adata, color="best_clustering", ax=axes[0], show=False,
           legend_loc="on data", title="Leiden res=0.3 (cluster numbers)")
sc.pl.umap(adata, color="cell_type", ax=axes[1], show=False,
           legend_loc="on data", title="Leiden res=0.3 (cell type annotations)")
plt.suptitle("PBMC 3k - Cluster Annotations", fontsize=13, fontweight="bold")
plt.tight_layout(); savefig("cell_type_annotations.png")

marker_genes_flat = ["CD3D", "IL7R", "CD14", "LYZ", "MS4A1",
                     "CD79A", "GNLY", "NKG7", "FCGR3A", "PPBP"]
sc.pl.dotplot(adata_markers, marker_genes_flat, groupby="best_clustering",
              title="Known PBMC Marker Genes across Clusters")
savefig("dotplot_markers.png")

print("Final cluster to cell type mapping:")
for k, v in cluster_annotations.items():
    print(f"  Cluster {k}: {v}")

# Top 3 marker genes bar chart per cell type
adata_markers2 = adata.raw.to_adata()
adata_markers2.obs["cell_type"] = adata.obs["leiden_0.3"].map(cluster_labels_3k)
sc.tl.rank_genes_groups(adata_markers2, groupby="cell_type",
                         method="wilcoxon", key_added="rank_genes_ct", pts=True)
cell_types_list = list(cluster_labels_3k.values())
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()
for i, ct in enumerate(cell_types_list):
    df     = sc.get.rank_genes_groups_df(adata_markers2, group=ct, key="rank_genes_ct").head(3)
    genes  = df["names"].tolist()
    scores = df["scores"].tolist()
    ax     = axes[i]
    c_list = plt.cm.Blues(np.linspace(0.4, 0.9, 3))
    ax.bar(range(3), scores, color=c_list, edgecolor="white", width=0.6)
    for j, (gene, score) in enumerate(zip(genes, scores)):
        ax.text(j, score + max(scores) * 0.01, gene,
                ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_xticks(range(3))
    ax.set_xticklabels([f"Rank {k+1}" for k in range(3)], fontsize=9)
    ax.set_ylabel("Wilcoxon Score", fontsize=9)
    ax.set_title(ct, fontsize=11, fontweight="bold")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.suptitle("Top 3 Marker Genes per Cell Type - PBMC 3k", fontsize=14, fontweight="bold")
plt.tight_layout(); savefig("top_genes_per_celltype_3k.png")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 12: Biological Validation — All Methods
# ═══════════════════════════════════════════════════════════════════════════
print("\n── Biological Validation (All Methods) ──")
adata_raw = adata.raw.to_adata()
all_cell_types = ["T cells", "NK cells", "B cells", "Monocytes", "Platelets", "Dendritic"]

def run_marker_analysis(adata_r, labels, method_name):
    adata_tmp = adata_r.copy()
    adata_tmp.obs["cluster"] = pd.Categorical(labels.astype(str))
    adata_tmp = adata_tmp[adata_tmp.obs["cluster"] != "-1"].copy()
    if len(adata_tmp.obs["cluster"].unique()) < 2:
        print(f"  {method_name}: not enough clusters, skipping")
        return {}, []
    sc.tl.rank_genes_groups(adata_tmp, groupby="cluster", method="wilcoxon",
                             key_added="rank_genes", n_genes=20)
    recovered = set()
    ann = {}
    for cluster in adata_tmp.obs["cluster"].unique():
        genes = sc.get.rank_genes_groups_df(adata_tmp, group=cluster,
                                             key="rank_genes")["names"].tolist()[:15]
        for cell_type, markers in known_markers.items():
            if any(g in markers for g in genes):
                recovered.add(cell_type)
                if cluster not in ann:
                    ann[cluster] = cell_type
    return ann, list(recovered)

method_validation = {}
km_labels_val = KMeans(n_clusters=4, random_state=SEED, n_init=10).fit_predict(X_pca)
_, km_rec = run_marker_analysis(adata_raw, km_labels_val, "K-means")
method_validation["K-means\n(k=4)"] = km_rec
print(f"K-means recovered: {km_rec}")

hw_labels_val = AgglomerativeClustering(n_clusters=5, linkage="ward").fit_predict(X_pca)
_, hw_rec = run_marker_analysis(adata_raw, hw_labels_val, "Hier. Ward")
method_validation["Hier. Ward\n(n=5)"] = hw_rec
print(f"Hierarchical Ward recovered: {hw_rec}")

db_labels_val = dbscan_results[0.5]["labels"]
_, db_rec = run_marker_analysis(adata_raw, db_labels_val, "DBSCAN")
method_validation["DBSCAN\n(eps=0.5)"] = db_rec
print(f"DBSCAN recovered: {db_rec}")

hdb_labels_val = hdbscan_results[10]["labels"]
_, hdb_rec = run_marker_analysis(adata_raw, hdb_labels_val, "HDBSCAN")
method_validation["HDBSCAN\n(min=10)"] = hdb_rec
print(f"HDBSCAN recovered: {hdb_rec}")

ld_labels_val = adata.obs["leiden_0.3"].astype(int).values
_, ld_rec = run_marker_analysis(adata_raw, ld_labels_val, "Leiden PCA")
method_validation["Leiden PCA\n(res=0.3)"] = ld_rec
print(f"Leiden PCA recovered: {ld_rec}")

scvi_labels_val = adata_scvi.obs["scvi_leiden_0.6"].astype(int).values
adata_raw_scvi = adata_scvi.raw.to_adata() if adata_scvi.raw else adata_raw.copy()
_, scvi_rec = run_marker_analysis(adata_raw_scvi, scvi_labels_val, "scVI+Leiden")
method_validation["scVI+Leiden\n(res=0.6)"] = scvi_rec
print(f"scVI+Leiden recovered: {scvi_rec}")

val_rows = []
for method, recovered in method_validation.items():
    row = {"Method": method.replace("\n", " ")}
    for ct in all_cell_types:
        row[ct] = "Yes" if ct in recovered else "No"
    row["Total recovered"] = f"{sum(1 for ct in all_cell_types if ct in recovered)}/{len(all_cell_types)}"
    val_rows.append(row)
val_df = pd.DataFrame(val_rows)
print("\nBiological Validation Summary:"); print(val_df.to_string(index=False))
val_df.to_csv(RES_DIR / "biological_validation_summary.csv", index=False)

matrix = np.zeros((len(method_validation), len(all_cell_types)))
for i, (method, recovered) in enumerate(method_validation.items()):
    for j, ct in enumerate(all_cell_types):
        matrix[i, j] = 1 if ct in recovered else 0
fig, ax = plt.subplots(figsize=(12, 6))
ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
ax.set_xticks(range(len(all_cell_types)))
ax.set_xticklabels(all_cell_types, fontsize=11, fontweight="bold")
methods_labels = [m.replace("\n", " ") for m in method_validation.keys()]
ax.set_yticks(range(len(methods_labels))); ax.set_yticklabels(methods_labels, fontsize=11)
for i in range(len(methods_labels)):
    for j in range(len(all_cell_types)):
        ax.text(j, i, "Yes" if matrix[i, j] == 1 else "No",
                ha="center", va="center", fontsize=11, fontweight="bold",
                color="white" if matrix[i, j] == 0 else "black")
ax.set_title("Biological Validation - Cell Type Recovery per Method\n(PBMC 3k)",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Cell Type", fontsize=11); ax.set_ylabel("Clustering Method", fontsize=11)
plt.tight_layout(); savefig("biological_validation_heatmap.png")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 13: Cross-Method Comparison Summary
# ═══════════════════════════════════════════════════════════════════════════
print("\n── Cross-Method Comparison ──")

def cluster_balance(labels):
    labels = [l for l in labels if l != -1]
    counts = Counter(labels)
    total = len(labels)
    return (round(max(counts.values()) / total * 100, 1),
            round(min(counts.values()) / total * 100, 1))

km_max,  km_min  = cluster_balance(km_labels_val)
hw_max,  hw_min  = cluster_balance(hw_labels_val)
db_max,  db_min  = cluster_balance(db_labels_val)
hdb_max, hdb_min = cluster_balance(hdb_labels_val)
ld_max,  ld_min  = cluster_balance(ld_labels_val)

results = {
    "Method":          ["K-means (k=4)", "Hierarchical Ward (n=5)", "DBSCAN (eps=0.5)",
                        "HDBSCAN (min_size=10)", "Leiden PCA (res=0.3)", "scVI + Leiden (res=0.6)"],
    "N_clusters":      [4,      5,      4,      3,      6,      5     ],
    "Silhouette":      [0.3308, 0.3278, 0.5660, 0.3801, 0.3052, 0.1054],
    "Davies_Bouldin":  [1.1815, 1.0606, 0.4387, 1.0191, 1.2232, 2.3436],
    "Seed_ARI":        [0.742,  1.0,    1.0,    1.0,    0.971,  np.nan],
    "Resample_ARI":    [0.685,  0.927,  0.997,  np.nan, 0.970,  np.nan],
    "Noise_pct":       [0,      0,      0.1,    30.1,   0,      0     ],
    "Max_cluster_pct": [km_max, hw_max, db_max, hdb_max, ld_max, "N/A"],
    "Min_cluster_pct": [km_min, hw_min, db_min, hdb_min, ld_min, "N/A"],
    "Biological_valid":["Yes",  "Partial", "Partial", "No", "Yes", "Yes"],
}
df_compare = pd.DataFrame(results)
print("Cross-Method Comparison Table:"); print(df_compare.to_string(index=False))
df_compare.to_csv(RES_DIR / "results_summary.csv", index=False)

methods_short = ["K-means\n(k=4)", "Hier. Ward\n(n=5)", "DBSCAN\n(eps=0.5)",
                 "HDBSCAN\n(min=10)", "Leiden PCA\n(res=0.3)", "scVI+Leiden\n(res=0.6)"]
colors_m = ["#4e79a7", "#4e79a7", "#f28e2b", "#f28e2b", "#59a14f", "#e15759"]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sil_vals = [0.3308, 0.3278, 0.5660, 0.3801, 0.3052, 0.1054]
db_vals  = [1.1815, 1.0606, 0.4387, 1.0191, 1.2232, 2.3436]
bars1 = axes[0].bar(methods_short, sil_vals, color=colors_m, edgecolor="white", width=0.6)
axes[0].set_ylabel("Silhouette Score (higher is better)", fontsize=11)
axes[0].set_title("Silhouette Score by Method", fontsize=12, fontweight="bold")
axes[0].set_ylim(0, 0.75)
for bar, val in zip(bars1, sil_vals):
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
axes[0].tick_params(axis="x", labelsize=9)
axes[0].spines["top"].set_visible(False); axes[0].spines["right"].set_visible(False)

bars2 = axes[1].bar(methods_short, db_vals, color=colors_m, edgecolor="white", width=0.6)
axes[1].set_ylabel("Davies-Bouldin Score (lower is better)", fontsize=11)
axes[1].set_title("Davies-Bouldin Score by Method", fontsize=12, fontweight="bold")
for bar, val in zip(bars2, db_vals):
    axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
axes[1].tick_params(axis="x", labelsize=9)
axes[1].spines["top"].set_visible(False); axes[1].spines["right"].set_visible(False)

fig.legend(handles=[mpatches.Patch(color="#4e79a7", label="Baseline methods"),
                    mpatches.Patch(color="#f28e2b", label="Density-based methods"),
                    mpatches.Patch(color="#59a14f", label="scRNA-seq specific"),
                    mpatches.Patch(color="#e15759", label="Deep learning")],
           loc="lower center", ncol=4, fontsize=10, bbox_to_anchor=(0.5, -0.02))
plt.suptitle("Clustering Quality Metrics - All Methods (PBMC 3k)",
             fontsize=13, fontweight="bold")
plt.tight_layout(); savefig("comparison_quality_metrics.png")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 14: SC3 Integration (requires sc3_labels.csv from sc3_3k.R)
# ═══════════════════════════════════════════════════════════════════════════
if SC3_LABELS.exists():
    print(f"\n── SC3 Integration ──")
    sc3_df = pd.read_csv(SC3_LABELS, index_col=0)
    overlap = set(adata.obs_names) & set(sc3_df.index)
    print(f"Matching barcodes: {len(overlap)} out of {adata.n_obs}")

    for col in [c for c in sc3_df.columns if c.startswith("sc3_k")]:
        adata.obs[col] = sc3_df.loc[adata.obs_names, col].astype(str)
        adata.obs[col] = pd.Categorical(adata.obs[col])

    # NMI between SC3 and all Leiden resolutions
    leiden_cols = [c for c in adata.obs.columns if c.startswith("leiden_") and not c.startswith("leiden_seed")]
    sc3_k_cols  = [c for c in sc3_df.columns if c.startswith("sc3_k")]
    nmi_rows = []
    for lc in leiden_cols:
        for sc3c in sc3_k_cols:
            nmi = normalized_mutual_info_score(
                adata.obs[lc].astype(str), adata.obs[sc3c].astype(str))
            nmi_rows.append({"leiden": lc, "sc3": sc3c, "NMI": round(nmi, 4)})
    nmi_df = pd.DataFrame(nmi_rows)
    print("\nNMI between Leiden and SC3 clusterings:")
    print(nmi_df.pivot(index="leiden", columns="sc3", values="NMI").to_string())
    nmi_df.to_csv(RES_DIR / "nmi_leiden_vs_sc3_3k.csv", index=False)

    # NMI heatmap
    nmi_pivot = nmi_df.pivot(index="leiden", columns="sc3", values="NMI")
    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(nmi_pivot.values, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(nmi_pivot.columns))); ax.set_xticklabels(nmi_pivot.columns, fontsize=10)
    ax.set_yticks(range(len(nmi_pivot.index)));   ax.set_yticklabels(nmi_pivot.index,   fontsize=9)
    plt.colorbar(im, ax=ax, label="NMI")
    for i in range(len(nmi_pivot.index)):
        for j in range(len(nmi_pivot.columns)):
            ax.text(j, i, f"{nmi_pivot.values[i,j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("NMI: Leiden vs SC3 — PBMC 3k", fontsize=13, fontweight="bold")
    ax.set_xlabel("SC3 k"); ax.set_ylabel("Leiden resolution")
    plt.tight_layout(); savefig("nmi_leiden_vs_sc3_3k.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sc.pl.umap(adata, color="leiden_0.3", ax=axes[0], show=False,
               legend_loc="on data", title="Leiden res=0.3 (6 clusters)")
    sc.pl.umap(adata, color="sc3_k4", ax=axes[1], show=False,
               legend_loc="on data", title="SC3 k=4 (4 clusters)")
    plt.suptitle("Leiden vs SC3 - PBMC 3k", fontsize=13, fontweight="bold")
    plt.tight_layout(); savefig("sc3_vs_leiden_umap.png")

    adata_markers_sc3 = adata.raw.to_adata()
    adata_markers_sc3.obs["sc3_k4"] = adata.obs["sc3_k4"]
    sc.tl.rank_genes_groups(adata_markers_sc3, groupby="sc3_k4",
                             method="wilcoxon", key_added="rank_genes_sc3", n_genes=20)

    print("SC3 k=4 marker genes per cluster:")
    sc3_annotations = {}
    sc3_recovered   = []
    for cluster in sorted(adata_markers_sc3.obs["sc3_k4"].unique(), key=lambda x: int(x)):
        genes = sc.get.rank_genes_groups_df(adata_markers_sc3, group=cluster,
                                             key="rank_genes_sc3")["names"].tolist()[:10]
        print(f"Cluster {cluster}: {', '.join(genes)}")
        matched = False
        for cell_type, markers in known_markers.items():
            overlap = [g for g in genes if g in markers]
            if overlap:
                print(f"    {cell_type}: {overlap}")
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
        {k: f"{v} ({k})" for k, v in sc3_annotations.items()})
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sc.pl.umap(adata, color="sc3_k4", ax=axes[0], show=False,
               legend_loc="on data", title="SC3 k=4 (cluster numbers)")
    sc.pl.umap(adata, color="sc3_cell_type", ax=axes[1], show=False,
               legend_loc="on data", title="SC3 k=4 (cell type annotations)")
    plt.suptitle("SC3 Clustering - PBMC 3k", fontsize=13, fontweight="bold")
    plt.tight_layout(); savefig("sc3_annotations_umap.png")

    marker_genes_flat = ["CD3D", "IL7R", "CD14", "LYZ", "MS4A1",
                         "CD79A", "GNLY", "NKG7", "FCGR3A", "PPBP"]
    sc.pl.dotplot(adata_markers_sc3, marker_genes_flat, groupby="sc3_k4",
                  title="Known PBMC Markers across SC3 Clusters")
    savefig("sc3_dotplot.png")

    # Final biological validation heatmap including SC3
    method_validation_final = {
        "K-means (k=4)":         ["NK cells", "Platelets", "Monocytes", "Dendritic"],
        "Hier. Ward (n=5)":      ["Monocytes", "Platelets", "NK cells", "B cells",
                                   "Dendritic", "T cells"],
        "DBSCAN (eps=0.5)":      ["Monocytes", "Platelets", "B cells", "Dendritic", "T cells"],
        "HDBSCAN (min=10)":      ["B cells", "Monocytes", "Dendritic", "T cells"],
        "Leiden PCA (res=0.3)":  ["Monocytes", "Platelets", "NK cells", "B cells",
                                   "Dendritic", "T cells"],
        "scVI+Leiden (res=0.6)": ["Monocytes", "Platelets", "NK cells", "B cells",
                                   "Dendritic", "T cells"],
        "SC3 (k=4)":             sc3_recovered,
    }
    methods_final = list(method_validation_final.keys())
    matrix_final = np.zeros((len(methods_final), len(all_cell_types)))
    for i, (method, recovered) in enumerate(method_validation_final.items()):
        for j, ct in enumerate(all_cell_types):
            matrix_final[i, j] = 1 if ct in recovered else 0
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.imshow(matrix_final, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(all_cell_types)))
    ax.set_xticklabels(all_cell_types, fontsize=11, fontweight="bold")
    ax.set_yticks(range(len(methods_final))); ax.set_yticklabels(methods_final, fontsize=10)
    for i in range(len(methods_final)):
        for j in range(len(all_cell_types)):
            ax.text(j, i, "Yes" if matrix_final[i, j] == 1 else "No",
                    ha="center", va="center", fontsize=11, fontweight="bold",
                    color="white" if matrix_final[i, j] == 0 else "black")
    ax.set_title("Biological Validation - All Methods including SC3\n(PBMC 3k)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Cell Type", fontsize=11); ax.set_ylabel("Clustering Method", fontsize=11)
    plt.tight_layout(); savefig("biological_validation_heatmap_final.png")

    # SC3 k=6
    if "sc3_k6" in adata.obs.columns:
        adata_markers_sc3_k6 = adata.raw.to_adata()
        adata_markers_sc3_k6.obs["sc3_k6"] = adata.obs["sc3_k6"]
        sc.tl.rank_genes_groups(adata_markers_sc3_k6, groupby="sc3_k6",
                                 method="wilcoxon", key_added="rank_genes_sc3_k6", n_genes=20)
        sc3_k6_annotations = {}
        sc3_k6_recovered   = []
        for cluster in sorted(adata_markers_sc3_k6.obs["sc3_k6"].unique(),
                               key=lambda x: int(x)):
            genes = sc.get.rank_genes_groups_df(adata_markers_sc3_k6, group=cluster,
                                                 key="rank_genes_sc3_k6")["names"].tolist()[:10]
            matched = False
            for cell_type, markers in known_markers.items():
                overlap = [g for g in genes if g in markers]
                if overlap:
                    if cell_type not in sc3_k6_recovered:
                        sc3_k6_recovered.append(cell_type)
                    if not matched:
                        sc3_k6_annotations[cluster] = cell_type
                        matched = True
            if not matched:
                sc3_k6_annotations[cluster] = "Unknown"
        print(f"SC3 k=6 recovered: {sc3_k6_recovered}")
        adata.obs["sc3_k6_cell_type"] = adata.obs["sc3_k6"].map(
            {k: f"{v} ({k})" for k, v in sc3_k6_annotations.items()})
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        sc.pl.umap(adata, color="sc3_k6", ax=axes[0], show=False,
                   legend_loc="on data", title="SC3 k=6 (cluster numbers)")
        sc.pl.umap(adata, color="sc3_k6_cell_type", ax=axes[1], show=False,
                   legend_loc="on data", title="SC3 k=6 (cell type annotations)")
        plt.suptitle("SC3 k=6 Clustering - PBMC 3k", fontsize=13, fontweight="bold")
        plt.tight_layout(); savefig("sc3_k6_umap.png")
else:
    print(f"\nSC3 labels not found at {SC3_LABELS}.")
    print("Run: Rscript sc3_3k.R   (from the src/ directory)")
    print("Then re-run this script to generate SC3 comparison plots.")

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
    plt.suptitle("UMAP Colored by QC Metrics — PBMC 3k", fontsize=13, fontweight="bold")
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
    plt.suptitle("PBMC 3k — Post-filter Cell Quality Distribution", fontsize=13, fontweight="bold")
    plt.tight_layout()
    savefig("qc_postfilter_violin.png")

# 3. PCA scatter colored by leiden / cell type
_cc = "best_clustering" if "best_clustering" in adata.obs.columns else "leiden_0.3"
_ct = "cell_type"       if "cell_type"        in adata.obs.columns else _cc
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sc.pl.pca(adata, color=_cc, ax=axes[0], show=False, title="PCA — cluster IDs")
sc.pl.pca(adata, color=_ct, ax=axes[1], show=False, title="PCA — cell type annotations")
plt.suptitle("PCA Space — PBMC 3k", fontsize=13, fontweight="bold")
plt.tight_layout()
savefig("pca_scatter.png")

# 4. K-means silhouette + DB sweep
_ks   = list(k_range)
_sils = [kmeans_results[k]["silhouette"]     for k in _ks]
_dbs  = [kmeans_results[k]["davies_bouldin"] for k in _ks]
_ck   = ["#e8473f" if k == best_k else "steelblue" for k in _ks]
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
bars1 = axes[0].bar(_ks, _sils, color=_ck, edgecolor="white", width=0.6)
axes[0].set_xlabel("k (clusters)"); axes[0].set_ylabel("Silhouette Score (higher = better)")
axes[0].set_title("K-means: Silhouette Score vs k", fontsize=12, fontweight="bold")
axes[0].set_xticks(_ks)
for bar, v in zip(bars1, _sils):
    axes[0].text(bar.get_x() + bar.get_width()/2, v + 0.003, f"{v:.3f}",
                 ha="center", va="bottom", fontsize=8, fontweight="bold")
axes[0].spines["top"].set_visible(False); axes[0].spines["right"].set_visible(False)
bars2 = axes[1].bar(_ks, _dbs, color=_ck, edgecolor="white", width=0.6)
axes[1].set_xlabel("k (clusters)"); axes[1].set_ylabel("Davies-Bouldin Index (lower = better)")
axes[1].set_title("K-means: Davies-Bouldin Index vs k", fontsize=12, fontweight="bold")
axes[1].set_xticks(_ks)
for bar, v in zip(bars2, _dbs):
    axes[1].text(bar.get_x() + bar.get_width()/2, v + 0.01, f"{v:.3f}",
                 ha="center", va="bottom", fontsize=8, fontweight="bold")
axes[1].spines["top"].set_visible(False); axes[1].spines["right"].set_visible(False)
plt.suptitle("K-means Metric Sweep — PBMC 3k", fontsize=13, fontweight="bold")
plt.tight_layout()
savefig("kmeans_metrics_sweep.png")

# 5. K-means all-k UMAP grid
_ncols_k = 4
_nrows_k  = int(np.ceil(len(_ks) / _ncols_k))
fig, axes = plt.subplots(_nrows_k, _ncols_k, figsize=(5 * _ncols_k, 4.5 * _nrows_k))
axes = np.array(axes).flatten()
for _i, _k in enumerate(k_range):
    _col = f"kmeans_{_k}"
    if _col not in adata.obs.columns:
        adata.obs[_col] = pd.Categorical(kmeans_results[_k]["labels"].astype(str))
    sc.pl.umap(adata, color=_col, ax=axes[_i], show=False, legend_loc="on data",
               title=f"k={_k}  sil={kmeans_results[_k]['silhouette']:.3f}")
for _j in range(_i + 1, len(axes)):
    axes[_j].axis("off")
plt.suptitle("K-means: All k Values — PBMC 3k", fontsize=14, fontweight="bold")
plt.tight_layout()
savefig("kmeans_all_k_grid.png")

# 6. Hierarchical silhouette sweep (Ward vs Average)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
_ns = list(n_clusters_range)
for ax, linkage, color in zip(axes, ["ward", "average"], ["steelblue", "#e8473f"]):
    _sils_h = [hier_results[linkage][n]["silhouette"] for n in _ns]
    _best_n = max(hier_results[linkage], key=lambda n: hier_results[linkage][n]["silhouette"])
    ax.plot(_ns, _sils_h, "o-", color=color, linewidth=2, markersize=7)
    ax.axvline(_best_n, color="gray", linestyle="--", linewidth=1.2, label=f"Best n={_best_n}")
    for n, s in zip(_ns, _sils_h):
        ax.text(n, s + 0.003, f"{s:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_xlabel("n clusters"); ax.set_ylabel("Silhouette Score")
    ax.set_title(f"Hierarchical ({linkage}) — Best n={_best_n}", fontsize=12, fontweight="bold")
    ax.set_xticks(_ns); ax.legend()
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.suptitle("Hierarchical Silhouette Sweep — PBMC 3k", fontsize=13, fontweight="bold")
plt.tight_layout()
savefig("hierarchical_metrics_sweep.png")

# 7. All-methods UMAP panel (2 × 3 grid, one per method)
adata.obs["_scvi_panel"] = pd.Categorical(
    adata_scvi.obs[f"scvi_leiden_{best_scvi_res}"].values.astype(str)
)
_panel = [
    ("kmeans_best",           f"K-means (k={best_k})"),
    ("hier_ward_best",        "Hierarchical Ward (best n)"),
    ("dbscan_best",           f"DBSCAN (eps={best_eps})"),
    ("hdbscan_best",          f"HDBSCAN (mcs={best_mcs})"),
    (f"leiden_{best_res}",    f"Leiden PCA (res={best_res})"),
    ("_scvi_panel",           f"scVI+Leiden (res={best_scvi_res})"),
]
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
axes = axes.flatten()
for ax, (key, title) in zip(axes, _panel):
    sc.pl.umap(adata, color=key, ax=ax, show=False, legend_loc="on data", title=title)
plt.suptitle("All Clustering Methods — Best Result (PBMC 3k)", fontsize=14, fontweight="bold")
plt.tight_layout()
savefig("all_methods_umap_panel.png")

# 8. Heatmap of top marker genes across clusters
try:
    _marker_flat = ["CD3D", "IL7R", "CD14", "LYZ", "MS4A1",
                    "CD79A", "GNLY", "NKG7", "FCGR3A", "PPBP"]
    sc.pl.heatmap(adata_markers, var_names=_marker_flat, groupby="best_clustering",
                  show_gene_labels=True, cmap="RdBu_r", figsize=(10, 6), show=False)
    plt.suptitle("Marker Gene Expression Heatmap — PBMC 3k", fontsize=13, fontweight="bold")
    savefig("heatmap_markers.png")
except Exception as e:
    print(f"  Heatmap skipped: {e}")

# 9. Violin plots of key marker genes
try:
    sc.pl.violin(adata_markers, keys=["CD3D", "LYZ", "MS4A1", "GNLY", "PPBP"],
                 groupby="best_clustering", rotation=45, stripplot=False,
                 figsize=(14, 4), show=False)
    plt.suptitle("Marker Gene Expression per Cluster — PBMC 3k", fontsize=13, fontweight="bold")
    savefig("violin_key_markers.png")
except Exception as e:
    print(f"  Violin skipped: {e}")

# 10. scVI latent UMAP annotated with cell types
if "cell_type" in adata.obs.columns:
    adata_scvi.obs["_ct_ref"] = adata.obs["cell_type"].values
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sc.pl.umap(adata_scvi, color=f"scvi_leiden_{best_scvi_res}",
               ax=axes[0], show=False, legend_loc="on data",
               title=f"scVI+Leiden (res={best_scvi_res})")
    sc.pl.umap(adata_scvi, color="_ct_ref",
               ax=axes[1], show=False, legend_loc="on data",
               title="Cell type annotations (from Leiden)")
    plt.suptitle("scVI Latent Space — PBMC 3k", fontsize=13, fontweight="bold")
    plt.tight_layout()
    savefig("scvi_umap_annotated.png")

# 11. Leiden all-resolutions grid
_res_list = resolutions
_ncols_l  = 5
_nrows_l  = int(np.ceil(len(_res_list) / _ncols_l))
fig, axes = plt.subplots(_nrows_l, _ncols_l, figsize=(5 * _ncols_l, 4.5 * _nrows_l))
axes = np.array(axes).flatten()
for _i, _res in enumerate(_res_list):
    sc.pl.umap(adata, color=f"leiden_{_res}", ax=axes[_i], show=False,
               legend_loc="on data",
               title=f"res={_res}  k={leiden_results[_res]['n_clusters']}  "
                     f"sil={leiden_results[_res]['silhouette']:.3f}")
for _j in range(_i + 1, len(axes)):
    axes[_j].axis("off")
plt.suptitle("Leiden: Full Resolution Grid — PBMC 3k", fontsize=14, fontweight="bold")
plt.tight_layout()
savefig("leiden_all_resolutions_grid.png")

# 12. Comprehensive metrics summary table
_tbl = pd.DataFrame({
    "Clusters":      [4,      5,      4,      3,      6,      5     ],
    "Silhouette":    [0.331,  0.328,  0.566,  0.380,  0.305,  0.105 ],
    "DB Index":      [1.182,  1.061,  0.439,  1.019,  1.223,  2.344 ],
    "Seed ARI":      [0.742,  1.000,  1.000,  1.000,  0.971,  0.996 ],
    "Resample ARI":  [0.685,  0.927,  0.997,  "N/A",  0.970,  0.976 ],
    "Bio Validation":["4/6",  "6/6",  "5/6",  "4/6",  "6/6",  "6/6" ],
    "Noise %":       [0.0,    0.0,    0.1,    30.1,   0.0,    0.0   ],
}, index=["K-means (k=4)", "Hier. Ward (n=5)", "DBSCAN (eps=0.5)",
          "HDBSCAN (min=10)", "Leiden (res=0.3)", "scVI+Leiden (res=0.6)"])
fig, ax = plt.subplots(figsize=(15, 4.2))
ax.axis("off")
_tbl_obj = ax.table(cellText=_tbl.values, colLabels=_tbl.columns,
                    rowLabels=_tbl.index, cellLoc="center", loc="center")
_tbl_obj.auto_set_font_size(False); _tbl_obj.set_fontsize(10); _tbl_obj.scale(1.1, 2.1)
for (row, col), cell in _tbl_obj.get_celld().items():
    cell.set_edgecolor("black"); cell.set_linewidth(0.5)
    if row == 0 or col == -1:
        cell.set_facecolor("#f0f0f0"); cell.set_text_props(fontweight="bold")
    else:
        cell.set_facecolor("white")
plt.title("Clustering Methods Summary — PBMC 3k", fontsize=13, fontweight="bold", pad=20)
plt.tight_layout()
savefig("metrics_table_3k.png")

# 13. Calinski-Harabasz score comparison bar chart
_ch_methods = ["K-means\n(k=4)", "Hier. Ward\n(n=5)", "Leiden PCA\n(res=0.3)", "scVI+Leiden\n(res=0.6)"]
_best_ward_n3k = max(hier_results["ward"], key=lambda n: hier_results["ward"][n]["silhouette"])
_ch_vals = [
    kmeans_results[best_k]["calinski_harabasz"],
    hier_results["ward"][_best_ward_n3k]["calinski_harabasz"],
    leiden_results[best_res]["calinski_harabasz"],
    scvi_leiden_results[best_scvi_res]["calinski_harabasz"],
]
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(_ch_methods, _ch_vals,
              color=["#4e79a7", "#4e79a7", "#59a14f", "#e15759"],
              edgecolor="white", width=0.5)
for bar, v in zip(bars, _ch_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            f"{v:.0f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_ylabel("Calinski-Harabász Score (higher = better)", fontsize=11)
ax.set_title("Calinski-Harabász Score by Method — PBMC 3k", fontsize=13, fontweight="bold")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
savefig("calinski_harabasz_3k.png")

# 14. PAGA trajectory graph
try:
    _paga_key = "leiden_0.3" if "leiden_0.3" in adata.obs.columns else f"leiden_{best_res}"
    sc.tl.paga(adata, groups=_paga_key)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sc.pl.paga(adata, ax=axes[0], show=False, fontsize=10,
               title="PAGA — connectivity graph (Leiden res=0.3)")
    # PAGA-initialized UMAP (stored separately to not disturb existing X_umap)
    adata.obsm["X_umap_orig"] = adata.obsm["X_umap"].copy()
    sc.tl.umap(adata, init_pos="paga", random_state=SEED)
    sc.pl.umap(adata, color=_paga_key, ax=axes[1], show=False, legend_loc="on data",
               title="PAGA-initialized UMAP (Leiden res=0.3)")
    adata.obsm["X_umap"] = adata.obsm["X_umap_orig"]   # restore original
    plt.suptitle("PAGA Trajectory — PBMC 3k", fontsize=13, fontweight="bold")
    plt.tight_layout()
    savefig("paga_3k.png")
except Exception as e:
    print(f"  PAGA skipped: {e}")

# 15. Save standardized summary CSV (used later by combined results script)
_std_3k = pd.DataFrame({
    "dataset":   "PBMC 3k",
    "method":    ["K-means", "Hier. Ward", "DBSCAN", "HDBSCAN", "Leiden PCA", "scVI+Leiden"],
    "n_clusters":[4, 5, 4, 3, 6, 5],
    "silhouette":[kmeans_results[best_k]["silhouette"],
                  hier_results["ward"][_best_ward_n3k]["silhouette"],
                  dbscan_results[best_eps]["silhouette"],
                  hdbscan_results[best_mcs]["silhouette"],
                  leiden_results[best_res]["silhouette"],
                  scvi_leiden_results[best_scvi_res]["silhouette"]],
    "davies_bouldin":[kmeans_results[best_k]["davies_bouldin"],
                      hier_results["ward"][_best_ward_n3k]["davies_bouldin"],
                      dbscan_results[best_eps]["davies_bouldin"],
                      hdbscan_results[best_mcs]["davies_bouldin"],
                      leiden_results[best_res]["davies_bouldin"],
                      scvi_leiden_results[best_scvi_res]["davies_bouldin"]],
    "calinski_harabasz":[kmeans_results[best_k]["calinski_harabasz"],
                         hier_results["ward"][_best_ward_n3k]["calinski_harabasz"],
                         np.nan, np.nan,
                         leiden_results[best_res]["calinski_harabasz"],
                         scvi_leiden_results[best_scvi_res]["calinski_harabasz"]],
})
_std_3k.to_csv(RES_DIR / "summary_3k.csv", index=False)

print(f"\nDone. Figures saved to: {FIG_DIR}")
print(f"Results saved to:       {RES_DIR}")
