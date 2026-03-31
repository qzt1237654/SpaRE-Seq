import os
import sys
import random
import numpy as np
import torch
import scanpy as sc
import pandas as pd
import anndata
from sklearn.decomposition import PCA
from scipy.sparse import block_diag, issparse
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
import gc

# ==========================================
# 1. Basic & Environment Functions
# ==========================================

def init_spclue_env(r_home, r_user, spclue_path, seed=0):
    """Initialize R environment, spCLUE path, and fix random seeds."""
    os.environ['R_HOME'] = r_home
    os.environ['R_USER'] = r_user
    sys.path.append(spclue_path)
    
    # Fix seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    from rpy2.robjects import numpy2ri
    from rpy2.robjects.packages import importr
    numpy2ri.activate()
    
    try:
        importr('mclust')
        print("✅ R environment and mclust loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load mclust: {e}")

    import spCLUE
    spCLUE.fix_seed(seed)
    return spCLUE

# ==========================================
# 2. Clustering & Export Functions
# ==========================================

def run_spclue_clustering(sample_names, h5ad_template_path, spclue_mod, n_clusters=6, device_name='cuda'):
    """Full pipeline for data loading, graph building, and spCLUE training."""
    adata_s = []
    batch_list = []
    
    print("📂 Loading and preprocessing data...")
    for i, name in enumerate(sample_names):
        path = h5ad_template_path.format(name=name)
        adata_temp = sc.read_h5ad(path)
        adata_temp.var_names_make_unique()
        if hasattr(adata_temp.X, "toarray"):
            adata_temp.X = adata_temp.X.toarray()
            
        adata_temp = spclue_mod.preprocess(adata_temp)
        adata_temp.obs['batch_name'] = str(i)
        adata_s.append(adata_temp)
        batch_list += [i] * adata_temp.shape[0]

    adata = sc.concat(adata_s, index_unique="_")
    batch_list = np.array(batch_list)
    adata.obsm["X_pca"] = PCA(n_components=200, random_state=0).fit_transform(adata.X)

    print("🕸️ Building graphs and training spCLUE...")
    g_spatial = block_diag([spclue_mod.prepare_graph(cur, "spatial") for cur in adata_s])
    g_expr = block_diag([spclue_mod.prepare_graph(cur, "expr") for cur in adata_s])
    
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    model = spclue_mod.spCLUE(adata.obsm["X_pca"], {"spatial": g_spatial, "expr": g_expr}, 
                             n_clusters, batch_list, epochs=500, batch_train=True, device=device)
    _, adata.obsm["spCLUE"] = model.trainBatch()
    
    sc.pp.neighbors(adata, n_neighbors=50, use_rep="spCLUE")
    spclue_mod.clustering(adata, n_clusters, key='spCLUE', cluster_methods="leiden")
    
    try:
        spclue_mod.batch_refine_label(adata, key="leiden", batch_key="batch_name")
        print("✅ Clustering and refinement complete.")
    except:
        adata.obs["leiden_refined"] = adata.obs["leiden"]
        
    return adata

def plot_spatial_clusters(adata, sample_names, save_path, flip_dict=None):
    """Plot spatial distribution with optional flipping."""
    if flip_dict is None: flip_dict = {}
    valid_labels = [x for x in adata.obs["leiden_refined"].unique() if pd.notna(x)]
    unique_labels = sorted([int(float(x)) for x in valid_labels])
    tmp_colors = sns.color_palette("tab10", len(unique_labels))
    fixed_palette = {str(lbl): mcolors.to_hex(tmp_colors[i]) for i, lbl in enumerate(unique_labels)}

    rows, cols = (len(sample_names) + 2) // 3, 3
    fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*4))
    axes = axes.flatten()

    for i, s_id in enumerate(sample_names):
        ax = axes[i]
        cur = adata[adata.obs["batch_name"].astype(str) == str(i)].copy()
        if cur.shape[0] > 0:
            coords = cur.obsm["spatial"].copy()
            action = flip_dict.get(str(s_id), "")
            if action == "lr": coords[:, 0] = -coords[:, 0]
            elif action == "ud": coords[:, 1] = -coords[:, 1]
            cur.obsm["spatial"] = coords
            sc.pl.embedding(cur, basis="spatial", color="leiden_refined", palette=fixed_palette, ax=ax, show=False, frameon=False)
            ax.set_title(f"Sample: {s_id} {action.upper()}")
        else: ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def export_clean_cluster_bins(adata, sample_names, h5ad_template_path, output_dir):
    """Export bin IDs cleaned of concat suffixes."""
    os.makedirs(output_dir, exist_ok=True)
    # Get total clusters from data
    unique_clusters = sorted(adata.obs["leiden_refined"].unique())
    
    for i, s_id in enumerate(sample_names):
        raw_adata = sc.read_h5ad(h5ad_template_path.format(name=s_id), backed='r')
        raw_bins = set(raw_adata.obs_names)
        raw_adata.file.close()
        
        sample_mask = adata.obs["batch_name"].astype(str) == str(i)
        for n in unique_clusters:
            cluster_mask = adata.obs["leiden_refined"] == n
            bins = adata.obs_names[sample_mask & cluster_mask].tolist()
            # Remove the "_0", "_1" etc. suffixes added by concat
            clean_bins = [b.rsplit('_', 1)[0] if (b not in raw_bins and "_" in b) else b for b in bins]
            
            with open(os.path.join(output_dir, f"{s_id}_cluster{n}.txt"), 'w') as f:
                for b in clean_bins: f.write(f"{b}\n")

# ==========================================
# 3. Matrix Generation Functions (Lines 1 & 3)
# ==========================================

def prepare_stress_vs_control_matrices(data_raw_dir, list_dir, output_dir, ctrl_samples, stress_samples, n_clusters=6, layer_key='AGcount_A'):
    """Line 1: Extracts data, applies filters, and splits into 4 matrices."""
    all_samples = ctrl_samples + stress_samples
    os.makedirs(output_dir, exist_ok=True)
    
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
        
        c_dir = os.path.join(output_dir, f"cluster{c}")
        os.makedirs(c_dir, exist_ok=True)
        df_filtered[[f"{s}_G" for s in ctrl_samples]].to_csv(os.path.join(c_dir, f"G_control_cluster{c}.csv"))
        df_filtered[[f"{s}_A" for s in ctrl_samples]].to_csv(os.path.join(c_dir, f"A_control_cluster{c}.csv"))
        df_filtered[[f"{s}_G" for s in stress_samples]].to_csv(os.path.join(c_dir, f"G_stress_cluster{c}.csv"))
        df_filtered[[f"{s}_A" for s in stress_samples]].to_csv(os.path.join(c_dir, f"A_stress_cluster{c}.csv"))

def prepare_onlyC_cluster_vs_others_matrices(data_raw_dir, list_dir, output_dir, samples, n_clusters=6, layer_key='AGcount_A'):
    """Line 3: Target vs Except matrix generation with two-step filtering."""
    os.makedirs(output_dir, exist_ok=True)
    print("🔍 Synchronizing global sites...")
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
        
        c_dir = os.path.join(output_dir, f"cluster{target_c}")
        no_c_dir = os.path.join(output_dir, f"no_cluster{target_c}")
        os.makedirs(c_dir, exist_ok=True); os.makedirs(no_c_dir, exist_ok=True)
        df_concat.loc[df_t_sub.index, final_sites].to_csv(os.path.join(c_dir, f"cluster{target_c}_filtered.csv"))
        df_concat.loc[df_ex_sub.index, final_sites].to_csv(os.path.join(no_c_dir, f"no_cluster{target_c}_filtered.csv"))