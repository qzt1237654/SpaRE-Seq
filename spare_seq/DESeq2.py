import os
import gc
import re
import numpy as np
import pandas as pd
import anndata
from scipy.sparse import issparse

# ==========================================
# 1. Stress vs Control Pipeline
# ==========================================

def run_pipeline_stress_vs_control(data_raw_dir, list_dir, output_dir, ctrl_samples, stress_samples, n_clusters=6, layer_key='AGcount_A', p_thresh=0.05, lfc_thresh=1.0):
    """
    End-to-end pipeline: Generates matrices -> Runs DESeq2 -> Filters significant sites.
    Compares Stress vs Control within each cluster.
    """
    from rpy2.robjects import r
    matrix_dir = os.path.join(output_dir, "raw_matrices")
    deseq2_dir = os.path.join(output_dir, "deseq2_raw_results")
    final_dir = os.path.join(output_dir, "significant_results")
    os.makedirs(matrix_dir, exist_ok=True)
    os.makedirs(deseq2_dir, exist_ok=True)
    os.makedirs(final_dir, exist_ok=True)

    print("📊 [1/3] Generating input matrices from h5ad data...")
    all_samples = ctrl_samples + stress_samples
    all_sites = set()
    for s in all_samples:
        path = os.path.join(data_raw_dir, f"GSM81924{s}_AtoI_tissue.h5ad")
        if os.path.exists(path):
            temp = anndata.read_h5ad(path, backed='r')
            all_sites.update(temp.var_names)
            temp.file.close()
            
    site_list = sorted(list(all_sites))
    site_to_idx = {site: i for i, site in enumerate(site_list)}
    n_sites = len(site_list)
    
    for c in range(n_clusters):
        c_rows, c_labels = [], []
        for s in all_samples:
            vec_G, vec_A = np.zeros(n_sites), np.zeros(n_sites)
            h5ad_path = os.path.join(data_raw_dir, f"GSM81924{s}_AtoI_tissue.h5ad")
            txt_path = os.path.join(list_dir, f"{s}_cluster{c}.txt")
            
            if os.path.exists(h5ad_path) and os.path.exists(txt_path):
                adata_sub = anndata.read_h5ad(h5ad_path)
                with open(txt_path, 'r') as f:
                    target_ids = [line.strip() for line in f if line.strip()]
                sub = adata_sub[adata_sub.obs_names.isin(target_ids)]
                
                if sub.n_obs > 0:
                    current_map = [site_to_idx[g] for g in adata_sub.var_names]
                    sum_G = sub.X.sum(axis=0)
                    vec_G[current_map] = np.array(sum_G).flatten() if issparse(sum_G) else np.ravel(sum_G)
                    if layer_key in sub.layers:
                        sum_A = sub.layers[layer_key].sum(axis=0)
                        vec_A[current_map] = np.array(sum_A).flatten() if issparse(sum_A) else np.ravel(sum_A)
                del adata_sub; gc.collect()
            
            c_rows.extend([vec_G, vec_A])
            c_labels.extend([f"{s}_G", f"{s}_A"])
            
        df_cluster = pd.DataFrame(np.vstack(c_rows), index=c_labels, columns=site_list)
        g_data = df_cluster.iloc[0::2].values
        a_data = df_cluster.iloc[1::2].values
        ag_median = np.median(g_data + a_data, axis=0)
        g_count = (g_data > 0).sum(axis=0)
        
        valid_sites = df_cluster.columns[(ag_median >= 10) & (g_count >= 3)]
        df_filtered = df_cluster[valid_sites].T
        
        c_dir = os.path.join(matrix_dir, f"cluster{c}")
        os.makedirs(c_dir, exist_ok=True)
        df_filtered[[f"{s}_G" for s in ctrl_samples]].to_csv(os.path.join(c_dir, f"G_control_cluster{c}.csv"))
        df_filtered[[f"{s}_A" for s in ctrl_samples]].to_csv(os.path.join(c_dir, f"A_control_cluster{c}.csv"))
        df_filtered[[f"{s}_G" for s in stress_samples]].to_csv(os.path.join(c_dir, f"G_stress_cluster{c}.csv"))
        df_filtered[[f"{s}_A" for s in stress_samples]].to_csv(os.path.join(c_dir, f"A_stress_cluster{c}.csv"))

    print("🧬 [2/3] Executing DESeq2 via rpy2...")
    r_input_path = matrix_dir.replace('\\', '/')
    r_output_path = deseq2_dir.replace('\\', '/')
    
    r_code = f"""
    library(DESeq2)
    input_path <- "{r_input_path}"
    output_dir <- "{r_output_path}"
    
    for (cluster_id in 0:{n_clusters-1}) {{
        c_dir <- file.path(input_path, paste0("cluster", cluster_id))
        g_ctrl_file <- file.path(c_dir, paste0("G_control_cluster", cluster_id, ".csv"))
        if (!file.exists(g_ctrl_file)) next
        
        G_control <- read.csv(g_ctrl_file, row.names=1)
        A_control <- read.csv(file.path(c_dir, paste0("A_control_cluster", cluster_id, ".csv")), row.names=1)
        G_stress  <- read.csv(file.path(c_dir, paste0("G_stress_cluster", cluster_id, ".csv")), row.names=1)
        A_stress  <- read.csv(file.path(c_dir, paste0("A_stress_cluster", cluster_id, ".csv")), row.names=1)
        
        counts <- cbind(G_control, G_stress, A_control, A_stress)
        num_ctrl <- ncol(G_control)
        num_stress <- ncol(G_stress)
        
        design <- data.frame(
            Trt = factor(rep(c(rep("control", num_ctrl), rep("stress", num_stress)), 2), levels = c("control", "stress")),
            Type = factor(c(rep("Edit_G", (num_ctrl + num_stress)), rep("Base_A", (num_ctrl + num_stress))), levels = c("Base_A", "Edit_G"))
        )
        
        dds <- DESeqDataSetFromMatrix(countData = round(as.matrix(counts)), colData = design, design = ~ Type + Trt + Type:Trt)
        
        total_coverage <- cbind((G_control + A_control), (G_stress + A_stress))
        log_data <- log(as.matrix(total_coverage))
        log_data[is.infinite(log_data)] <- NA
        log_s <- log_data - rowMeans(log_data, na.rm = TRUE)
        sf <- exp(apply(log_s, 2, function(x) median(x, na.rm = TRUE)))
        sizeFactors(dds) <- rep(sf, 2)
        
        dds <- DESeq(dds, test = "Wald", quiet = TRUE)
        res <- results(dds, name = "TypeEdit_G.Trtstress")
        
        write.csv(as.data.frame(res), file = file.path(output_dir, paste0("StressVsCtrl_Cluster", cluster_id, "_DESeq2_Results.csv")))
    }}
    """
    try:
        r(r_code)
    except Exception as e:
        print(f"❌ DESeq2 execution failed: {e}")
        return

    print(f"🧹 [3/3] Filtering results (P < {p_thresh}, |Log2FC| > {lfc_thresh})...")
    for c in range(n_clusters):
        file_path = os.path.join(deseq2_dir, f"StressVsCtrl_Cluster{c}_DESeq2_Results.csv")
        if not os.path.exists(file_path): continue
        
        df = pd.read_csv(file_path, index_col=0)
        df.index = [re.sub(r'^X(\d)', r'\1', str(idx)) for idx in df.index]
        
        df_sig = df.dropna(subset=['pvalue', 'log2FoldChange'])
        df_sig = df_sig[(df_sig['pvalue'] < p_thresh) & (df_sig['log2FoldChange'].abs() > lfc_thresh)].copy()
        df_final = df_sig[['pvalue', 'log2FoldChange']].sort_values(by='log2FoldChange', ascending=False)
        
        df_final.to_csv(os.path.join(final_dir, f"Cluster{c}_Significant_Sites.csv"))
        with open(os.path.join(final_dir, f"Cluster{c}_sites_list.txt"), 'w') as f:
            for site in df_final.index: f.write(f"{site}\n")
        print(f"✅ Cluster {c}: Retained {len(df_final)} significant sites.")


# ==========================================
# 2. Target Cluster vs Others Pipeline
# ==========================================

def run_pipeline_cluster_vs_others(data_raw_dir, list_dir, output_dir, samples, n_clusters=6, layer_key='AGcount_A', p_thresh=0.05, lfc_thresh=1.0):
    """
    End-to-end pipeline: Generates matrices -> Runs DESeq2 -> Filters significant sites.
    Compares one specific cluster against all other combined clusters.
    """
    from rpy2.robjects import r
    matrix_dir = os.path.join(output_dir, "raw_matrices")
    deseq2_dir = os.path.join(output_dir, "deseq2_raw_results")
    final_dir = os.path.join(output_dir, "significant_results")
    os.makedirs(matrix_dir, exist_ok=True)
    os.makedirs(deseq2_dir, exist_ok=True)
    os.makedirs(final_dir, exist_ok=True)

    print("📊 [1/3] Generating input matrices from h5ad data...")
    all_sites = set()
    for s in samples:
        path = os.path.join(data_raw_dir, f"GSM81924{s}_AtoI_tissue.h5ad")
        if os.path.exists(path):
            adata_ref = anndata.read_h5ad(path, backed='r')
            all_sites.update(adata_ref.var_names)
            adata_ref.file.close()
            
    site_list = sorted(list(all_sites))
    site_to_idx = {s: i for i, s in enumerate(site_list)}
    n_sites = len(site_list)

    cluster_dict = {}
    for c in range(n_clusters):
        rows, labels = [], []
        for s in samples:
            v_G, v_A = np.zeros(n_sites), np.zeros(n_sites)
            h5ad_path = os.path.join(data_raw_dir, f"GSM81924{s}_AtoI_tissue.h5ad")
            txt_path = os.path.join(list_dir, f"{s}_cluster{c}.txt")
            if os.path.exists(h5ad_path) and os.path.exists(txt_path):
                adata_sub = anndata.read_h5ad(h5ad_path)
                with open(txt_path, 'r') as f:
                    t_ids = [line.strip() for line in f if line.strip()]
                sub = adata_sub[adata_sub.obs_names.isin(t_ids)]
                if sub.n_obs > 0:
                    cur_map = [site_to_idx[g] for g in adata_sub.var_names]
                    sum_G = sub.X.sum(axis=0)
                    v_G[cur_map] = np.array(sum_G).flatten() if issparse(sum_G) else np.ravel(sum_G)
                    if layer_key in sub.layers:
                        sum_A = sub.layers[layer_key].sum(axis=0)
                        v_A[cur_map] = np.array(sum_A).flatten() if issparse(sum_A) else np.ravel(sum_A)
                del adata_sub; gc.collect()
            rows.extend([v_G, v_A]); labels.extend([f"{s}_G", f"{s}_A"])
        cluster_dict[c] = pd.DataFrame(np.vstack(rows), index=labels, columns=site_list)

    for target_c in range(n_clusters):
        df_target = cluster_dict[target_c].copy()
        g_data_1 = df_target.iloc[0::2].values
        a_data_1 = df_target.iloc[1::2].values
        mask_1 = (np.median(g_data_1 + a_data_1, axis=0) >= 10) & ((g_data_1 > 0).sum(axis=0) >= 3)
        valid_sites_1 = df_target.columns[mask_1]
        
        df_t_sub = df_target[valid_sites_1]
        other_dfs = [cluster_dict[i][valid_sites_1] for i in range(n_clusters) if i != target_c]
        df_ex_sub = sum(other_dfs)
        
        df_t_sub.index = [f"{x}_{target_c}" for x in df_t_sub.index]
        df_ex_sub.index = [f"{x}_no{target_c}" for x in df_ex_sub.index]
        df_concat = pd.concat([df_t_sub, df_ex_sub], axis=0)
        
        g_rows = [r for r in df_concat.index if "_G_" in r]
        a_rows = [r for r in df_concat.index if "_A_" in r]
        mask_2 = (np.median(df_concat.loc[g_rows].values + df_concat.loc[a_rows].values, axis=0) > 10) & \
                 ((df_concat.loc[g_rows].values > 0).sum(axis=0) >= 3)
        final_sites = df_concat.columns[mask_2]
        
        c_dir = os.path.join(matrix_dir, f"cluster{target_c}")
        no_c_dir = os.path.join(matrix_dir, f"no_cluster{target_c}")
        os.makedirs(c_dir, exist_ok=True); os.makedirs(no_c_dir, exist_ok=True)
        df_concat.loc[df_t_sub.index, final_sites].to_csv(os.path.join(c_dir, f"cluster{target_c}_filtered.csv"))
        df_concat.loc[df_ex_sub.index, final_sites].to_csv(os.path.join(no_c_dir, f"no_cluster{target_c}_filtered.csv"))

    print("🧬 [2/3] Executing DESeq2 via rpy2...")
    r_input_path = matrix_dir.replace('\\', '/')
    r_output_path = deseq2_dir.replace('\\', '/')
    
    r_code = f"""
    library(DESeq2)
    input_path <- "{r_input_path}"
    output_dir <- "{r_output_path}"
    
    for (cluster_id in 0:{n_clusters-1}) {{
        file_cluster <- file.path(input_path, paste0("cluster", cluster_id), paste0("cluster", cluster_id, "_filtered.csv"))
        file_except  <- file.path(input_path, paste0("no_cluster", cluster_id), paste0("no_cluster", cluster_id, "_filtered.csv"))
        
        if (!file.exists(file_cluster)) next
        
        data_cluster <- t(read.csv(file_cluster, row.names=1))
        data_except  <- t(read.csv(file_except, row.names=1))
        
        G_cluster <- data_cluster[, grep("_G_", colnames(data_cluster)), drop=FALSE]
        A_cluster <- data_cluster[, grep("_A_", colnames(data_cluster)), drop=FALSE]
        G_except  <- data_except[, grep("_G_", colnames(data_except)), drop=FALSE]
        A_except  <- data_except[, grep("_A_", colnames(data_except)), drop=FALSE]
        
        counts <- cbind(G_except, G_cluster, A_except, A_cluster)
        num_cluster <- ncol(G_cluster)
        num_except  <- ncol(G_except)
        
        design <- data.frame(
            Group = factor(rep(c(rep("Except", num_except), rep("Cluster", num_cluster)), 2), levels = c("Except", "Cluster")),
            Type  = factor(c(rep("Edit_G", (num_except + num_cluster)), rep("Base_A", (num_except + num_cluster))), levels = c("Base_A", "Edit_G"))
        )
        
        dds <- DESeqDataSetFromMatrix(countData = round(as.matrix(counts)), colData = design, design = ~ Type + Group + Type:Group)
        
        total_coverage <- cbind((G_except + A_except), (G_cluster + A_cluster))
        log_data <- log(as.matrix(total_coverage))
        log_data[is.infinite(log_data)] <- NA
        log_s <- log_data - rowMeans(log_data, na.rm = TRUE)
        sf <- exp(apply(log_s, 2, function(x) median(x, na.rm = TRUE)))
        sizeFactors(dds) <- rep(sf, 2)
        
        dds <- DESeq(dds, test = "Wald", quiet = TRUE)
        res <- results(dds, name = "TypeEdit_G.GroupCluster")
        
        write.csv(as.data.frame(res), file = file.path(output_dir, paste0("OnlyC_Cluster", cluster_id, "_DESeq2_Results.csv")))
    }}
    """
    try:
        r(r_code)
    except Exception as e:
        print(f"❌ DESeq2 execution failed: {e}")
        return

    print(f"🧹 [3/3] Filtering results (P < {p_thresh}, |Log2FC| > {lfc_thresh})...")
    for c in range(n_clusters):
        file_path = os.path.join(deseq2_dir, f"OnlyC_Cluster{c}_DESeq2_Results.csv")
        if not os.path.exists(file_path): continue
        
        df = pd.read_csv(file_path, index_col=0)
        df.index = [re.sub(r'^X(\d)', r'\1', str(idx)) for idx in df.index]
        
        df_sig = df.dropna(subset=['pvalue', 'log2FoldChange'])
        df_sig = df_sig[(df_sig['pvalue'] < p_thresh) & (df_sig['log2FoldChange'].abs() > lfc_thresh)].copy()
        df_final = df_sig[['pvalue', 'log2FoldChange']].sort_values(by='log2FoldChange', ascending=False)
        
        df_final.to_csv(os.path.join(final_dir, f"Cluster{c}_Significant_Sites.csv"))
        with open(os.path.join(final_dir, f"Cluster{c}_sites_list.txt"), 'w') as f:
            for site in df_final.index: f.write(f"{site}\n")
        print(f"✅ Cluster {c}: Retained {len(df_final)} significant sites.")