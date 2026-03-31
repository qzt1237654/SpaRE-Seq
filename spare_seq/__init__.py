# spare_seq/__init__.py

__version__ = "0.1.0"

# --- 1. Preprocessing (Line 1 & Line 3) ---
from .preprocessing import (
    init_spclue_env,
    run_spclue_clustering,
    plot_spatial_clusters,
    export_clean_cluster_bins,
    prepare_stress_vs_control_matrices,
    prepare_onlyC_cluster_vs_others_matrices
)

# --- 2. Differential Editing Analysis (Line 1 & Line 3) ---
from .differential_editing import (
    run_deseq2_stress_vs_control,        # Line 1: Stress vs Control
    filter_stress_deseq2_results,        # Line 1: Cleaner & Filter
    run_deseq2_onlyC_cluster_vs_others,  # Line 3: Internal Comparison
    filter_onlyC_deseq2_results          # Line 3: Cleaner & Filter
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

# --- 5. Genomic Annotation ---
from .annotation import (
    build_or_load_gtf_db,
    get_region_type,
    annotate_clusters
)