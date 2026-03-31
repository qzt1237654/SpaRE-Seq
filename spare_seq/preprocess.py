import os
import gc
import anndata
import numpy as np
import pandas as pd
from scipy.sparse import issparse

def prepare_stress_vs_control_matrices(data_raw_dir, list_dir, output_dir, ctrl_samples, stress_samples, n_clusters=6, layer_key='AGcount_A'):
    """
    Line 1: Extracts data, applies depth/presence filters, and splits into 4 matrices 
    (G_ctrl, A_ctrl, G_stress, A_stress) per cluster for DESeq2.
    """
    all_samples = ctrl_samples + stress_samples
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Scan for global site union
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
    
    # 2. Extract and reconstruct per cluster
    for c in range(n_clusters):
        c_rows, c_labels = [], []
        for s in all_samples:
            vec_G, vec_A = np.zeros(n_sites), np.zeros(n_sites)
            h5ad_path = os.path.join(data_raw_dir, f"GSM81924{s}_AtoI_tissue.h5ad")
            txt_path = os.path.join(list_dir, f"{s}_cluster{c}.txt")
            
            if os.path.exists(h5ad_path) and os.path.exists(txt_path):
                adata = anndata.read_h5ad(h5ad_path)
                with open(txt_path, 'r') as f:
                    target_ids = [line.strip() for line in f if line.strip()]
                sub = adata[adata.obs_names.isin(target_ids)]
                
                if sub.n_obs > 0:
                    current_map = [site_to_idx[g] for g in adata.var_names]
                    sum_G = sub.X.sum(axis=0)
                    vec_G[current_map] = np.array(sum_G).flatten() if issparse(sum_G) else np.ravel(sum_G)
                    if layer_key in sub.layers:
                        sum_A = sub.layers[layer_key].sum(axis=0)
                        vec_A[current_map] = np.array(sum_A).flatten() if issparse(sum_A) else np.ravel(sum_A)
                del adata; gc.collect()
            
            c_rows.extend([vec_G, vec_A])
            c_labels.extend([f"{s}_G", f"{s}_A"])
            
        df_cluster = pd.DataFrame(np.vstack(c_rows), index=c_labels, columns=site_list)
        
        # 3. Apply Hard Filter (Median >= 10, G count >= 3)
        g_data = df_cluster.iloc[0::2].values
        a_data = df_cluster.iloc[1::2].values
        ag_median = np.median(g_data + a_data, axis=0)
        g_count = (g_data > 0).sum(axis=0)
        
        valid_sites = df_cluster.columns[(ag_median >= 10) & (g_count >= 3)]
        df_filtered = df_cluster[valid_sites].T  # Transpose to (Sites x Samples)
        
        # 4. Split and Save into 4 groups
        c_dir = os.path.join(output_dir, f"cluster{c}")
        os.makedirs(c_dir, exist_ok=True)
        
        df_filtered[[f"{s}_G" for s in ctrl_samples]].to_csv(os.path.join(c_dir, f"G_control_cluster{c}.csv"))
        df_filtered[[f"{s}_A" for s in ctrl_samples]].to_csv(os.path.join(c_dir, f"A_control_cluster{c}.csv"))
        df_filtered[[f"{s}_G" for s in stress_samples]].to_csv(os.path.join(c_dir, f"G_stress_cluster{c}.csv"))
        df_filtered[[f"{s}_A" for s in stress_samples]].to_csv(os.path.join(c_dir, f"A_stress_cluster{c}.csv"))


def prepare_onlyC_cluster_vs_others_matrices(data_raw_dir, list_dir, output_dir, samples, n_clusters=6, layer_key='AGcount_A'):
    """
    Line 3: In-memory two-step filtering and Target vs Except matrix generation.
    Step 1: Filter on target cluster (Median>=10, G>=3).
    Step 2: Sum remaining clusters, concat, and filter again on the combined matrix.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Sync global sites
    print("🔍 Synchronizing global sites...")
    all_sites = set()
    for s in samples:
        path = os.path.join(data_raw_dir, f"GSM81924{s}_AtoI_tissue.h5ad")
        if os.path.exists(path):
            adata = anndata.read_h5ad(path, backed='r')
            all_sites.update(adata.var_names)
            adata.file.close()
            
    site_list = sorted(list(all_sites))
    site_to_idx = {s: i for i, s in enumerate(site_list)}
    n_sites = len(site_list)

    # 2. Build base matrices for all clusters (shape: 6xN for 3 samples)
    cluster_dict = {}
    for c in range(n_clusters):
        rows, labels = [], []
        for s in samples:
            v_G, v_A = np.zeros(n_sites), np.zeros(n_sites)
            h5ad_path = os.path.join(data_raw_dir, f"GSM81924{s}_AtoI_tissue.h5ad")
            txt_path = os.path.join(list_dir, f"{s}_cluster{c}.txt")
            
            if os.path.exists(h5ad_path) and os.path.exists(txt_path):
                adata = anndata.read_h5ad(h5ad_path)
                with open(txt_path, 'r') as f:
                    t_ids = [line.strip() for line in f if line.strip()]
                sub = adata[adata.obs_names.isin(t_ids)]
                
                if sub.n_obs > 0:
                    cur_map = [site_to_idx[g] for g in adata.var_names]
                    sum_G = sub.X.sum(axis=0)
                    v_G[cur_map] = np.array(sum_G).flatten() if issparse(sum_G) else np.ravel(sum_G)
                    if layer_key in sub.layers:
                        sum_A = sub.layers[layer_key].sum(axis=0)
                        v_A[cur_map] = np.array(sum_A).flatten() if issparse(sum_A) else np.ravel(sum_A)
                del adata; gc.collect()
                
            rows.extend([v_G, v_A])
            labels.extend([f"{s}_G", f"{s}_A"])
            
        cluster_dict[c] = pd.DataFrame(np.vstack(rows), index=labels, columns=site_list)

    # 3. Apply Two-Step Filter and split into DESeq2 inputs
    for target_c in range(n_clusters):
        df_target = cluster_dict[target_c].copy()
        
        # Filter 1: Apply threshold to target cluster only
        g_data_1 = df_target.iloc[0::2].values
        a_data_1 = df_target.iloc[1::2].values
        mask_1 = (np.median(g_data_1 + a_data_1, axis=0) >= 10) & ((g_data_1 > 0).sum(axis=0) >= 3)
        valid_sites_1 = df_target.columns[mask_1]
        
        # Calculate Except matrix (sum of other clusters) using only valid_sites_1
        df_t_sub = df_target[valid_sites_1]
        other_dfs = [cluster_dict[i][valid_sites_1] for i in range(n_clusters) if i != target_c]
        df_ex_sub = sum(other_dfs)
        
        # Rename indices for concat
        df_t_sub.index = [f"{x}_{target_c}" for x in df_t_sub.index]
        df_ex_sub.index = [f"{x}_no{target_c}" for x in df_ex_sub.index]
        df_concat = pd.concat([df_t_sub, df_ex_sub], axis=0)
        
        # Filter 2: Apply threshold to concatenated matrix
        g_rows = [r for r in df_concat.index if "_G_" in r]
        a_rows = [r for r in df_concat.index if "_A_" in r]
        df_G = df_concat.loc[g_rows]
        df_A = df_concat.loc[a_rows]
        
        # Note: Your code used > 10 for the combined matrix
        mask_2 = (np.median(df_G.values + df_A.values, axis=0) > 10) & ((df_G.values > 0).sum(axis=0) >= 3)
        final_sites = df_concat.columns[mask_2]
        
        # Split and export
        c_dir = os.path.join(output_dir, f"cluster{target_c}")
        no_c_dir = os.path.join(output_dir, f"no_cluster{target_c}")
        os.makedirs(c_dir, exist_ok=True); os.makedirs(no_c_dir, exist_ok=True)
        
        df_concat.loc[df_t_sub.index, final_sites].to_csv(os.path.join(c_dir, f"cluster{target_c}_filtered.csv"))
        df_concat.loc[df_ex_sub.index, final_sites].to_csv(os.path.join(no_c_dir, f"no_cluster{target_c}_filtered.csv"))
        print(f"✅ Cluster {target_c}: Final valid sites = {len(final_sites)}")