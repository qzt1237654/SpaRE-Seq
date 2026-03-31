import scanpy as sc
import pandas as pd
import numpy as np
import torch
import os
import sys
import random
import gc
from sklearn.decomposition import PCA
from scipy.sparse import block_diag
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors

def init_spclue_env(r_home, r_user, spclue_path, seed=0):
    """Initialize the R environment, the spCLUE path and fix all random seeds"""
    os.environ['R_HOME'] = r_home
    os.environ['R_USER'] = r_user
    sys.path.append(spclue_path)
    
    import rpy2.robjects as robjects
    from rpy2.robjects import numpy2ri
    from rpy2.robjects.packages import importr
    numpy2ri.activate()
    
    try:
        importr('mclust')
        print("R environment and mclust loaded successfully ")
    except Exception as e:
        print(f"mclust failed to load. Please check the R environment: {e}")

    # 固定种子
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    
    import spCLUE
    spCLUE.fix_seed(seed)
    return spCLUE


def run_spclue_clustering(sample_names, h5ad_template_path, spclue_mod, n_clusters=6, device_name='cuda'):
    """Carry out the complete preprocessing and spCLUE clustering process"""
    adata_s = []
    batch_list = []
    
    print("start batch reading and preprocessing data...")
    for i, name in enumerate(sample_names):
        path = h5ad_template_path.format(name=name)
        adata_temp = sc.read_h5ad(path)
        adata_temp.var_names_make_unique()
        
        # spCLUE requires a dense matrix
        if hasattr(adata_temp.X, "toarray"):
            adata_temp.X = adata_temp.X.toarray()
            
        adata_temp = spclue_mod.preprocess(adata_temp)
        adata_temp.obs['batch_name'] = str(i)
        
        adata_s.append(adata_temp)
        batch_list += [i] * adata_temp.shape[0]

    batch_list = np.array(batch_list)
    adata = sc.concat(adata_s, index_unique="_")
    print(f"The merge is complete! Total cell count: {adata.n_obs}")

    # PCA dimensional reduction
    adata.obsm["X_pca"] = PCA(n_components=200, random_state=0).fit_transform(adata.X)

    # Construct Spatial and Expr graphs
    print("building spatial and expression graph...")
    # spatial graph
    g_spatial_list = []
    for adata_cur in adata_s:
        g_spatial = spclue_mod.prepare_graph(adata_cur, "spatial")
        g_spatial_list.append(g_spatial)
    g_spatial = block_diag(g_spatial_list)

    # expression graph
    g_expr_list = []
    for adata_cur in adata_s:
        g_expr = spclue_mod.prepare_graph(adata_cur, "expr")
        g_expr_list.append(g_expr)
    g_expr = block_diag(g_expr_list)

    graph_dict = {"spatial": g_spatial, "expr": g_expr}

    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    
    # train model
    print("training spCLUE model...")
    spCLUE_model = spclue_mod.spCLUE(
        adata.obsm["X_pca"], graph_dict, n_clusters, batch_list,
        epochs=500, batch_train=True, device=device
    )

    _, adata.obsm["spCLUE"] = spCLUE_model.trainBatch()
    adata.obs["batch_name"] = batch_list
    adata.obs["batch_name"] = adata.obs["batch_name"].astype("category")

    adata.obsm["spCLUE"] = np.ascontiguousarray(adata.obsm["spCLUE"]).astype(np.float64)
    
    sc.pp.neighbors(adata, n_neighbors=50, use_rep="spCLUE")
    spclue_mod.clustering(adata, n_clusters, key='spCLUE', cluster_methods="leiden")
    
    try:
        spclue_mod.batch_refine_label(adata, key="leiden", batch_key="batch_name")
        print(f"Label Refine succeeded!produce {leiden_refined} column.")
    except Exception as e:
        print(f"Refine failed: {e}")
        adata.obs["leiden_refined"] = adata.obs["leiden"] 
        
    return adata


def plot_spatial_clusters(adata, sample_names, save_path, flip_dict=None):
    """
    Draw spatial distribution map
    flip_dict example: {"50": "lr", "49": "ud"} (lr=left_right flip, ud=up_down_flip)
    """
    if flip_dict is None: flip_dict = {}
    
    valid_labels = [x for x in adata.obs["leiden_refined"].unique() if pd.notna(x)]
    unique_labels = sorted([int(float(x)) for x in valid_labels])
    tmp_colors = sns.color_palette("tab10", len(unique_labels))
    fixed_palette = {str(lbl): mcolors.to_hex(tmp_colors[i]) for i, lbl in enumerate(unique_labels)}

    rows, cols = (len(sample_names) + 2) // 3, 3
    fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*5))
    axes = axes.flatten()

    for i, s_id in enumerate(sample_names):
        ax = axes[i]
        cur = adata[adata.obs["batch_name"].astype(str) == str(i)].copy()
        
        if cur.shape[0] > 0:
            coords = cur.obsm["spatial"].copy()
            flip_info = ""
            
            # Automatically apply the flip logic
            action = flip_dict.get(str(s_id), "")
            if action == "lr":
                coords[:, 0] = -coords[:, 0]
                flip_info = " (L-R Flipped)"
            elif action == "ud":
                coords[:, 1] = -coords[:, 1]
                flip_info = " (U-D Flipped)"
                
            cur.obsm["spatial"] = coords
            cur.obs["leiden_refined"] = cur.obs["leiden_refined"].astype(str).astype('category')
            
            sc.pl.embedding(cur, basis="spatial", color="leiden_refined", 
                            palette=fixed_palette, ax=ax, show=False, size=100, frameon=False)
            ax.set_title(f"Sample: {s_id}{flip_info}")
            ax.set_xlabel(""); ax.set_ylabel("")
        else:
            ax.axis('off')

    for j in range(len(sample_names), len(axes)): axes[j].axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"The spatial clustering graph has been saved in: {save_path}")


def plot_umap_qc(adata, save_path):
    """Draw the UMAP quality control chart (batch fusion and clustering effects)"""
    if "X_umap" not in adata.obsm:
        sc.pp.neighbors(adata, use_rep="spCLUE", n_neighbors=15)
        sc.tl.umap(adata)
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    sc.pl.umap(adata, color='batch_name', ax=ax1, show=False, title='Batch Integration')
    sc.pl.umap(adata, color='leiden_refined', ax=ax2, show=False, title='Clustering Quality')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"The UMAP QC map has been saved: {save_path}")


def export_clean_cluster_bins(adata, sample_names, h5ad_template_path, output_dir, n_clusters=6):
    """Export the Bin ID, automatically compare it with the original h5ad to detect and remove abnormal suffixes"""
    os.makedirs(output_dir, exist_ok=True)
    print("Start intelligent export of Cluster Bins...")
    
    for i, s_id in enumerate(sample_names):
        # 1. Take a sneak peek at the original file to obtain the true Bin ID format
        raw_path = h5ad_template_path.format(name=s_id)
        if not os.path.exists(raw_path):
            print(f"The original file {raw_path} cannot be found. Skip the suffix comparison.")
            continue
            
        raw_adata = sc.read_h5ad(raw_path, backed='r')
        raw_bins_set = set(raw_adata.obs_names)
        raw_adata.file.close()
        
        # 2. Extract the clustering data of the current sample
        sample_mask = adata.obs["batch_name"].astype(str) == str(i)
        sample_adata = adata[sample_mask]
        
        for n in range(n_clusters):
            cluster_mask = sample_adata.obs["leiden_refined"].astype(str) == str(n)
            raw_bins_in_concat = sample_adata.obs_names[cluster_mask].tolist()
            
            if not raw_bins_in_concat: continue
            
            clean_bins = []
            # 3. Intelligent detection logic
            # If the current name is not in the original dataset and it contains "_"
            for b in raw_bins_in_concat:
                if b not in raw_bins_set and "_" in b:
                    stripped_b = b.rsplit('_', 1)[0]
                    # If the tail is still in the original data after being cut off, it indicates that the correct removal method has been found
                    if stripped_b in raw_bins_set:
                        clean_bins.append(stripped_b)
                    else:
                        clean_bins.append(b) # It's not right to cut it. Keep it as it is
                else:
                    clean_bins.append(b) 
                    
            # 4. save
            file_path = os.path.join(output_dir, f"{s_id}_cluster{n}.txt")
            with open(file_path, 'w') as f:
                for bin_id in clean_bins:
                    f.write(f"{bin_id}\n")
                    
        print(f"The sample {s_id} has been exported successfully, and the suffixes have been intelligently aligned.")