import os
import anndata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr
import warnings

# ==========================================
# INTERNAL PLOTTING TOOLS 
# ==========================================
def _plot_boxplots(input_dir, output_base_dir, sample_ids):
    warnings.filterwarnings('ignore')
    for metric, suffix in zip(['Spearman_Rho', 'Pearson_R'], ['spearman', 'pearson']):
        out_dir = os.path.join(output_base_dir, metric)
        os.makedirs(out_dir, exist_ok=True)
        for s_id in sample_ids:
            file_path = os.path.join(input_dir, f"Corr_Results_Sample_{s_id}.csv")
            if not os.path.exists(file_path): continue
            df = pd.read_csv(file_path)
            if df.empty or metric not in df.columns: continue
            
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=df, x='Gene', y=metric, palette='Set3', showfliers=False)
            plt.axhline(0, color='red', linestyle='--', linewidth=1) 
            plt.title(f"Sample {s_id}: Distribution of {metric}")
            plt.ylabel(f"{metric} Value"); plt.xlabel("Target Genes")
            plt.savefig(os.path.join(out_dir, f"Sample_{s_id}_Boxplot_{suffix}.png"), dpi=300, bbox_inches='tight')
            plt.close()

def _plot_chr_dist(filtered_dir, sample_ids, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    def chr_sort_key(c): return (0, int(c)) if str(c).isdigit() else (1, str(c))

    all_counts = []
    for s_id in sample_ids:
        for method in ["Spearman", "Pearson"]:
            file_path = os.path.join(filtered_dir, f"Sample_{s_id}_{method}_Sig.csv")
            if not os.path.exists(file_path): continue
            df = pd.read_csv(file_path)
            if 'Site' in df.columns:
                df['Chromosome'] = df['Site'].astype(str).str.split('_').str[0]
                counts = df['Chromosome'].value_counts().reset_index()
                counts.columns, counts['Sample'], counts['Method'] = ['Chromosome', 'Count'], s_id, method
                all_counts.append(counts)

    if not all_counts: return
    summary_df = pd.concat(all_counts, ignore_index=True)
    summary_df.to_csv(os.path.join(filtered_dir, "Chromosome_Distribution_Summary.csv"), index=False)
    
    for method in ["Spearman", "Pearson"]:
        plot_data = summary_df[summary_df['Method'] == method]
        if plot_data.empty: continue
        plt.figure(figsize=(14, 6))
        order = sorted(plot_data['Chromosome'].unique(), key=chr_sort_key)
        sns.barplot(data=plot_data, x='Chromosome', y='Count', hue='Sample', order=order, palette='Set2')
        plt.title(f'Chromosome Distribution ({method})', fontsize=15, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"Chr_Distribution_{method}.png"), dpi=300)
        plt.close()

def _plot_heatmaps(ratio_dir, plot_dir, matrix_dir, sample_ids, top_n):
    warnings.filterwarnings('ignore')
    for s in sample_ids:
        ratio_file = os.path.join(ratio_dir, f"Sample_{s}_ratio_matrix.csv")
        if not os.path.exists(ratio_file): continue
        
        df = pd.read_csv(ratio_file, index_col=0)
        df = df.dropna(thresh=3).fillna(0)
        
        if len(df) > top_n:
            variances = df.var(axis=1)
            df_plot = df.loc[variances.nlargest(top_n).index]
        else:
            df_plot = df

        if len(df_plot) < 2: continue

        show_y_labels = True if len(df_plot) <= 50 else False

        g = sns.clustermap(
            df_plot, cmap="YlOrRd", figsize=(8, 10),
            row_cluster=True, col_cluster=True, 
            yticklabels=show_y_labels, xticklabels=True, 
            cbar_pos=(0.02, 0.8, 0.05, 0.18),
            cbar_kws={'label': f'Ratio (Sample {s})'}
        )
        
        if show_y_labels:
            plt.setp(g.ax_heatmap.get_yticklabels(), fontsize=8)

        g.fig.suptitle(f'Editing Signatures - Sample {s} (Top {len(df_plot)} Sites)', y=1.02, fontsize=14)
        
        g.savefig(os.path.join(plot_dir, f"Heatmap_Sample_{s}.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        reordered_df = df_plot.iloc[g.dendrogram_row.reordered_ind]
        reordered_df.to_csv(os.path.join(matrix_dir, f"Heatmap_Ordered_Sites_Sample_{s}.csv"))

# ==========================================
# MAIN PIPELINE FUNCTIONS
# ==========================================
def extract_and_normalize_genes(h5ad_dir, cluster_dir, save_dir, sample_ids, target_genes, n_clusters=6):
    """1. Extract target gene expression sum and normalize (log2(x+1))"""
    os.makedirs(save_dir, exist_ok=True)
    for s_id in sample_ids:
        h5ad_path = os.path.join(h5ad_dir, f"GSM81924{s_id}_GeneID_tissue.h5ad")
        if not os.path.exists(h5ad_path): continue
        
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
        
        df = pd.DataFrame(current_matrix, index=target_genes, columns=[f"Cluster_{i}" for i in range(n_clusters)])
        df_log = np.log2(df + 1).T
        df_log.to_csv(os.path.join(save_dir, f"Sample_{s_id}_gene_processed.csv"))
        adata.file.close()

def calculate_ratio_matrix(input_matrices_dir, save_dir, sample_ids, n_clusters=6):
    """2. Reconstruct Ratio matrix = G / (G + A + 1)"""
    os.makedirs(save_dir, exist_ok=True)
    all_site_union = set()
    
    for n in range(n_clusters):
        file_path = os.path.join(input_matrices_dir, f"cluster{n}", f"cluster{n}_filtered.csv")
        if os.path.exists(file_path):
            all_site_union.update(pd.read_csv(file_path, index_col=0, nrows=0).columns)
    union_list = sorted(list(all_site_union))

    for s_id in tqdm(sample_ids, desc="Reconstructing Ratio Matrices"):
        sample_data_dict = {}
        for n in range(n_clusters):
            file_path = os.path.join(input_matrices_dir, f"cluster{n}", f"cluster{n}_filtered.csv")
            if not os.path.exists(file_path): continue
            df = pd.read_csv(file_path, index_col=0)
            try:
                g, a = df.loc[f"{s_id}_G_{n}"], df.loc[f"{s_id}_A_{n}"]
                ratio_series = g / (g + a + 1)
                sample_data_dict[f"Cluster_{n}"] = ratio_series.reindex(union_list, fill_value=0)
            except KeyError:
                continue
            
        if sample_data_dict:
            pd.DataFrame(sample_data_dict).T.to_csv(os.path.join(save_dir, f"Sample_{s_id}_ratio_matrix.csv"))

def run_correlations(ratio_dir, gene_dir, output_dir, sample_ids):
    """3 & 4. Calculate Pearson and Spearman correlations"""
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

def filter_and_plot_correlations(input_dir, output_dir, plot_dir, sample_ids, spearman_rho=0.3, pearson_p=0.05, ratio_dir=None, plot_boxplot=True, plot_chr_dist=True, plot_heatmap=True, heatmap_top_n=40):
    """
    5. Filter significant correlations and automatically generate requested plots.
    Includes switches to turn specific plots on or off.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"🧹 Filtering correlations (Spearman Rho > {spearman_rho}, Pearson P < {pearson_p})...")
    
    for s_id in sample_ids:
        file_path = os.path.join(input_dir, f"Corr_Results_Sample_{s_id}.csv")
        if not os.path.exists(file_path): continue
        df = pd.read_csv(file_path)
        
        df_s = df[df['Spearman_Rho'] > spearman_rho].sort_values(by='Spearman_Rho', ascending=False)
        df_s[['Site', 'Gene', 'Spearman_Rho', 'Spearman_P']].to_csv(os.path.join(output_dir, f"Sample_{s_id}_Spearman_Sig.csv"), index=False)
        
        df_p = df[df['Pearson_P'] < pearson_p].sort_values(by='Pearson_R', ascending=False)
        df_p[['Site', 'Gene', 'Pearson_R', 'Pearson_P']].to_csv(os.path.join(output_dir, f"Sample_{s_id}_Pearson_Sig.csv"), index=False)
        
    print("🎨 Generating correlation plots...")
    if plot_boxplot:
        _plot_boxplots(input_dir, os.path.join(plot_dir, "boxplots"), sample_ids)
    if plot_chr_dist:
        _plot_chr_dist(output_dir, sample_ids, os.path.join(plot_dir, "chr_distribution"))
        
    if plot_heatmap and ratio_dir is not None:
        print("🔥 is producing single sample Heatmaps and its CSV list...")
        heatmap_out_dir = os.path.join(plot_dir, "heatmaps")
        heatmap_matrix_dir = os.path.join(output_dir, "heatmap_ordered_matrices")
        os.makedirs(heatmap_out_dir, exist_ok=True)
        os.makedirs(heatmap_matrix_dir, exist_ok=True)
        _plot_heatmaps(ratio_dir, heatmap_out_dir, heatmap_matrix_dir, sample_ids, heatmap_top_n)
        
    print("✅ Filtering and visualization completed.")