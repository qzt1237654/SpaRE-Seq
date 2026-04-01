# spare_seq/__init__.py

__version__ = "0.1.0"

# --- 1. Spatial Clustering & QC ---
from .cluster import (
    init_spclue_env,
    run_spclue_clustering,
    plot_umap_qc,
    plot_spatial_clusters,
    export_clean_cluster_bins
)

# --- 2. End-to-End Differential Editing Analysis ---
from .DESeq2 import (
    run_pipeline_stress_vs_control,
    run_pipeline_cluster_vs_others
)

# --- 3. Gene-Ratio Correlation Analysis ---
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