# spare_seq/__init__.py

__version__ = "1.0.0"

# --- 1. Spatial Clustering & QC ---
from .cluster import (
    init_spclue_env,
    run_spclue_clustering,
    plot_umap_qc,
    plot_spatial_clusters,
    export_clean_cluster_bins
)

# --- 2. End-to-End Differential Editing Analysis & Annotation ---
from .DESeq2 import (
    run_test_vs_control,
    run_cluster_vs_others,
    run_annotation
)

# --- 3. Gene-Ratio Correlation Analysis & Visualization ---
from .correlation import (
    extract_and_normalize_genes,
    calculate_ratio_matrix,
    run_correlations,
    filter_and_plot_correlations
)