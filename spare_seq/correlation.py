import os
import anndata
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr
import warnings

def extract_and_normalize_genes(h5ad_dir, cluster_dir, save_dir, sample_ids, target_genes, n_clusters=6):
    """1. 提取特定基因在各个 Cluster 中的表达量总和并 normalization (log2(x+1))"""
    os.makedirs(save_dir, exist_ok=True)
    for s_id in sample_ids:
        h5ad_path = os.path.join(h5ad_dir, f"GSM81924{s_id}_GeneID_tissue.h5ad")
        if not os.path.exists(h5ad_path): 
            continue
        
        adata = anndata.read_h5ad(h5ad_path)
        current_matrix = np.zeros((len(target_genes), n_clusters))
        
        for n in range(n_clusters):
            txt_path = os.path.join(cluster_dir, f"{s_id}_cluster{n}.txt")
            if os.path.exists(txt_path):
                with open(txt_path, 'r') as f:
                    bin_indices = [line.strip() for line in f if line.strip()]
                valid_bins = [b for b in bin_indices if b in adata.obs_names]
                if valid_bins:
                    sub_X = adata[valid_bins, target_genes].X
                    gene_sums = np.array(sub_X.toarray().sum(axis=0)).flatten() if hasattr(sub_X, "toarray") else np.array(sub_X.sum(axis=0)).flatten()
                    current_matrix[:, n] = gene_sums
        
        # DataFrame 构建、log2(x+1) 标准化并转置
        df = pd.DataFrame(current_matrix, index=target_genes, columns=[f"Cluster_{i}" for i in range(n_clusters)])
        df_log = np.log2(df + 1).T
        df_log.to_csv(os.path.join(save_dir, f"Sample_{s_id}_gene_processed.csv"))
        adata.file.close()

def calculate_ratio_matrix(input_matrices_dir, save_dir, sample_ids, n_clusters=6):
    """2. 获取所有 Cluster 位点并集，重构并计算 Ratio = G / (G + A + 1)"""
    os.makedirs(save_dir, exist_ok=True)
    all_site_union = set()
    
    # 扫描并集
    for n in range(n_clusters):
        file_path = os.path.join(input_matrices_dir, f"Cluster_{n}_new.csv")
        if os.path.exists(file_path):
            all_site_union.update(pd.read_csv(file_path, index_col=0, nrows=0).columns)
    union_list = sorted(list(all_site_union))

    # 计算并对齐
    for s_id in tqdm(sample_ids, desc="Reconstructing Ratio Matrices"):
        sample_data_dict = {}
        for n in range(n_clusters):
            file_path = os.path.join(input_matrices_dir, f"Cluster_{n}_new.csv")
            if not os.path.exists(file_path): continue
            df = pd.read_csv(file_path, index_col=0)
            g, a = df.loc[f"{s_id}_G"], df.loc[f"{s_id}_A"]
            ratio_series = g / (g + a + 1)
            sample_data_dict[f"Cluster_{n}"] = ratio_series.reindex(union_list, fill_value=0)
            
        pd.DataFrame(sample_data_dict).T.to_csv(os.path.join(save_dir, f"Sample_{s_id}_ratio_matrix.csv"))

def run_correlations(ratio_dir, gene_dir, output_dir, sample_ids):
    """3 & 4. 遍历计算 Pearson 和 Spearman 相关性"""
    os.makedirs(output_dir, exist_ok=True)
    warnings.filterwarnings('ignore')
    
    for s_id in tqdm(sample_ids, desc="Running Correlations"):
        ratio_file = os.path.join(ratio_dir, f"Sample_{s_id}_ratio_matrix.csv")
        gene_file = os.path.join(gene_dir, f"Sample_{s_id}_gene_processed.csv")
        if not (os.path.exists(ratio_file) and os.path.exists(gene_file)): continue
        
        df_ratio = pd.read_csv(ratio_file, index_col=0)
        df_gene = pd.read_csv(gene_file, index_col=0)
        results = []
        
        for site in tqdm(df_ratio.columns, desc=f"Sample {s_id}", position=1, leave=False):
            vec_ratio = df_ratio[site]
            if vec_ratio.std() == 0: continue
            
            for gene in df_gene.columns:
                vec_gene = df_gene[gene]
                if vec_gene.std() == 0: continue
                try:
                    s_rho, s_p = spearmanr(vec_gene, vec_ratio)
                    p_r, p_p = pearsonr(vec_gene, vec_ratio)
                    results.append({'Site': site, 'Gene': gene, 'Spearman_Rho': s_rho, 'Spearman_P': s_p, 'Pearson_R': p_r, 'Pearson_P': p_p})
                except: continue
                
        if results:
            pd.DataFrame(results).to_csv(os.path.join(output_dir, f"Corr_Results_Sample_{s_id}.csv"), index=False)

def filter_correlation_results(input_dir, output_dir, sample_ids, spearman_rho=0.3, pearson_p=0.05):
    """5. 分别筛选 Spearman (Rho > cut) 和 Pearson (P < cut) 并排序导出"""
    os.makedirs(output_dir, exist_ok=True)
    for s_id in sample_ids:
        file_path = os.path.join(input_dir, f"Corr_Results_Sample_{s_id}.csv")
        if not os.path.exists(file_path): continue
        
        df = pd.read_csv(file_path)
        
        # Spearman
        df_s = df[df['Spearman_Rho'] > spearman_rho].sort_values(by='Spearman_Rho', ascending=False)
        df_s[['Site', 'Gene', 'Spearman_Rho', 'Spearman_P']].to_csv(os.path.join(output_dir, f"Sample_{s_id}_Spearman_Sig.csv"), index=False)
        
        # Pearson
        df_p = df[df['Pearson_P'] < pearson_p].sort_values(by='Pearson_R', ascending=False)
        df_p[['Site', 'Gene', 'Pearson_R', 'Pearson_P']].to_csv(os.path.join(output_dir, f"Sample_{s_id}_Pearson_Sig.csv"), index=False)