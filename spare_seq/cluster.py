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
    """初始化 R 环境、spCLUE 路径并固定所有随机种子"""
    os.environ['R_HOME'] = r_home
    os.environ['R_USER'] = r_user
    sys.path.append(spclue_path)
    
    import rpy2.robjects as robjects
    from rpy2.robjects import numpy2ri
    from rpy2.robjects.packages import importr
    numpy2ri.activate()
    
    try:
        importr('mclust')
        print("✅ R 环境与 mclust 加载成功")
    except Exception as e:
        print(f"❌ mclust 加载失败，请检查 R 环境: {e}")

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
    """执行完整的预处理与 spCLUE 聚类流程"""
    adata_s = []
    batch_list = []
    
    print("📂 开始批量读取与预处理数据...")
    for i, name in enumerate(sample_names):
        path = h5ad_template_path.format(name=name)
        adata_temp = sc.read_h5ad(path)
        adata_temp.var_names_make_unique()
        
        # spCLUE 要求密集矩阵
        if hasattr(adata_temp.X, "toarray"):
            adata_temp.X = adata_temp.X.toarray()
            
        adata_temp = spclue_mod.preprocess(adata_temp)
        adata_temp.obs['batch_name'] = str(i)
        
        adata_s.append(adata_temp)
        batch_list += [i] * adata_temp.shape[0]

    batch_list = np.array(batch_list)
    adata = sc.concat(adata_s, index_unique="_")
    print(f"🔗 合并完成！总细胞数: {adata.n_obs}")

    # PCA 降维
    adata.obsm["X_pca"] = PCA(n_components=200, random_state=0).fit_transform(adata.X)

    # 构建 Spatial 和 Expr Graph
    print("🕸️ 正在构建空间与表达图网络...")
    g_spatial = block_diag([spclue_mod.prepare_graph(cur, "spatial") for cur in adata_s])
    g_expr = block_diag([spclue_mod.prepare_graph(cur, "expr") for cur in adata_s])
    graph_dict = {"spatial": g_spatial, "expr": g_expr}

    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    
    # 训练模型
    print("🧠 开始训练 spCLUE 模型...")
    spCLUE_model = spclue_mod.spCLUE(
        adata.obsm["X_pca"], graph_dict, n_clusters, batch_list,
        epochs=500, batch_train=True, device=device
    )
    _, adata.obsm["spCLUE"] = spCLUE_model.trainBatch()
    
    # 格式转换与聚类
    adata.obs["batch_name"] = adata.obs["batch_name"].astype("category")
    adata.obsm["spCLUE"] = np.ascontiguousarray(adata.obsm["spCLUE"]).astype(np.float64)
    
    sc.pp.neighbors(adata, n_neighbors=50, use_rep="spCLUE")
    spclue_mod.clustering(adata, n_clusters, key='spCLUE', cluster_methods="leiden")
    
    try:
        spclue_mod.batch_refine_label(adata, key="leiden", batch_key="batch_name")
        print("✅ Label Refine 成功！生成了 leiden_refined 列")
    except Exception as e:
        print(f"❌ Refine 失败: {e}")
        adata.obs["leiden_refined"] = adata.obs["leiden"] # 降级处理
        
    return adata


def plot_spatial_clusters(adata, sample_names, save_path, flip_dict=None):
    """
    绘制空间分布图。
    flip_dict 示例: {"50": "lr", "49": "ud"} (lr=左右翻转, ud=上下翻转)
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
            
            # 自动应用翻转逻辑
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
    print(f"✅ 空间聚类图已保存: {save_path}")


def plot_umap_qc(adata, save_path):
    """绘制 UMAP 质量控制图 (批次融合与聚类效果)"""
    if "X_umap" not in adata.obsm:
        sc.pp.neighbors(adata, use_rep="spCLUE", n_neighbors=15)
        sc.tl.umap(adata)
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    sc.pl.umap(adata, color='batch_name', ax=ax1, show=False, title='Batch Integration')
    sc.pl.umap(adata, color='leiden_refined', ax=ax2, show=False, title='Clustering Quality')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ UMAP QC 图已保存: {save_path}")


def export_clean_cluster_bins(adata, sample_names, h5ad_template_path, output_dir, n_clusters=6):
    """导出 Bin ID，自动对比原始 h5ad 检测并切除异常后缀"""
    os.makedirs(output_dir, exist_ok=True)
    print("🧹 开始智能导出 Cluster Bins...")
    
    for i, s_id in enumerate(sample_names):
        # 1. 偷偷看一眼原始文件，获取真正的 Bin ID 格式
        raw_path = h5ad_template_path.format(name=s_id)
        if not os.path.exists(raw_path):
            print(f"⚠️ 找不到原文件 {raw_path}，跳过后缀对比。")
            continue
            
        raw_adata = sc.read_h5ad(raw_path, backed='r')
        raw_bins_set = set(raw_adata.obs_names)
        raw_adata.file.close()
        
        # 2. 提取当前样本的聚类数据
        sample_mask = adata.obs["batch_name"].astype(str) == str(i)
        sample_adata = adata[sample_mask]
        
        for n in range(n_clusters):
            cluster_mask = sample_adata.obs["leiden_refined"].astype(str) == str(n)
            raw_bins_in_concat = sample_adata.obs_names[cluster_mask].tolist()
            
            if not raw_bins_in_concat: continue
            
            clean_bins = []
            # 3. 智能检测逻辑
            # 如果目前的名字不在原数据集里，并且它带有 "_"
            for b in raw_bins_in_concat:
                if b not in raw_bins_set and "_" in b:
                    stripped_b = b.rsplit('_', 1)[0]
                    # 如果切掉尾巴后在原数据里，说明找到了正确的去除方式
                    if stripped_b in raw_bins_set:
                        clean_bins.append(stripped_b)
                    else:
                        clean_bins.append(b) # 切了也不对，原样保留
                else:
                    clean_bins.append(b) # 没毛病，直接用
                    
            # 4. 保存
            file_path = os.path.join(output_dir, f"{s_id}_cluster{n}.txt")
            with open(file_path, 'w') as f:
                for bin_id in clean_bins:
                    f.write(f"{bin_id}\n")
                    
        print(f"✅ 样本 {s_id} 导出完成，后缀已智能对齐。")