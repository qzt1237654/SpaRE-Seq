# spare_seq/__init__.py

__version__ = "0.1.0"

# --- 1. Preprocessing (Line 1 & Line 3) ---
from .preprocessing import (
    init_spclue_env,
    run_spclue_clustering,
    plot_spatial_clusters,
    export_clean_cluster_bins,
    prepare_cluster_vs_others_matrices
)

# --- 2. Differential Editing Analysis (Line 1 & Line 3) ---
from .differential import (
    run_deseq2_cluster_vs_others,
    clean_and_filter_deseq2_results
)

# --- 3. Gene-Ratio Correlation Analysis (Line 2) ---
from .correlation import (
    extract_and_normalize_genes,
    calculate_ratio_matrix,
    run_correlations,
    filter_correlation_results
)

# --- 4. Visualization ---
from .visualization import (
    plot_correlation_chr_distribution,
    plot_volcano_onlyC,
    plot_correlation_boxplots
)