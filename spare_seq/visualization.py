import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import matplotlib.colors as mcolors
from adjustText import adjust_text
import warnings

# --- 内部工具函数：染色体排序 ---
def _chr_sort_key(c):
    """确保染色体按 1, 2, ..., X, Y 排序，而不是按字符 1, 10, 11 排序"""
    c_str = str(c).replace('chr', '')
    if c_str.isdigit():
        return (0, int(c_str))
    else:
        return (1, c_str)

# --- 1. 空间聚类分布图 (带翻转逻辑) ---
def plot_spatial_clusters(adata, sample_order, batch_mapping, save_path):
    """
    绘制空间聚类图，包含自动翻转逻辑和固定色板。
    """
    valid_labels = [x for x in adata.obs["leiden_refined"].unique() if pd.notna(x)]
    unique_labels = sorted([int(x) for x in valid_labels])
    tmp_colors = sns.color_palette("tab10", len(unique_labels))
    fixed_palette = {str(lbl): mcolors.to_hex(tmp_colors[i]) for i, lbl in enumerate(unique_labels)}

    rows, cols = (len(sample_order) + 2) // 3, 3
    fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*5))
    axes = axes.flatten()

    for i, s_id in enumerate(sample_order):
        ax = axes[i]
        internal_idx = batch_mapping.get(s_id)
        cur = adata[adata.obs["batch_name"].astype(str) == str(internal_idx)].copy()
        
        if cur.shape[0] > 0:
            coords = cur.obsm["spatial"].copy()
            # 翻转逻辑
            if s_id in ["50", "52", "54"]:
                coords[:, 0] = -coords[:, 0] # 左右翻转
            elif s_id == "49":
                coords[:, 1] = -coords[:, 1] # 上下翻转
            
            cur.obsm["spatial"] = coords
            cur.obs["leiden_refined"] = cur.obs["leiden_refined"].astype(str).astype('category')
            
            sc.pl.embedding(cur, basis="spatial", color="leiden_refined", 
                            palette=fixed_palette, ax=ax, show=False, size=100, frameon=False)
            ax.set_title(f"Sample: {s_id}")
        else:
            ax.axis('off')

    for j in range(len(sample_order), len(axes)): axes[j].axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 空间聚类图已保存: {save_path}")

# --- 2. UMAP 质量检查图 ---
def plot_umap_comparison(adata, color_palette, save_path):
    """绘制 UMAP 对比图：检查批次融合与聚类效果"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    sc.pl.umap(adata, color='batch_name', ax=ax1, show=False, title='Batch Integration')
    sc.pl.umap(adata, color='leiden_refined', palette=color_palette, ax=ax2, show=False, title='Clustering')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"✅ UMAP 对比图已保存: {save_path}")

# --- 3. 差异分析火山图 ---
def plot_volcano(res_path, save_path, title="Volcano Plot", p_thresh=0.05, lfc_thresh=1.0, top_n=5):
    """绘制带自动标注的火山图"""
    if not os.path.exists(res_path): return
    df = pd.read_csv(res_path, index_col=0).dropna(subset=['pvalue', 'log2FoldChange'])
    
    df['neg_log_p'] = -np.log10(df['pvalue'].replace(0, 1e-300))
    df['Group'] = 'Not Significant'
    df.loc[(df['pvalue'] < p_thresh) & (df['log2FoldChange'] > lfc_thresh), 'Group'] = 'Up'
    df.loc[(df['pvalue'] < p_thresh) & (df['log2FoldChange'] < -lfc_thresh), 'Group'] = 'Down'

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='log2FoldChange', y='neg_log_p', hue='Group', 
                    palette={'Not': '#E0E0E0', 'Up': '#FF4B4B', 'Down': '#4B7BFF'}, alpha=0.6, s=10)
    
    # 自动标注 Top 位点
    texts = []
    top_points = pd.concat([df[df['Group']=='Up'].nlargest(top_n, 'neg_log_p'), 
                            df[df['Group']=='Down'].nlargest(top_n, 'neg_log_p')])
    for idx, row in top_points.iterrows():
        texts.append(plt.text(row['log2FoldChange'], row['neg_log_p'], str(idx), fontsize=8))
    
    if texts: adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
    
    plt.title(f"Volcano Plot: Cluster {n} (Test vs Ctrl)", fontsize=15, pad=15)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# --- 4. 染色体分布统计图 ---
def plot_chromosome_distribution(filtered_dir, sample_ids, save_dir):
    """统计并绘制显著位点的染色体分布图"""
    all_counts = []
    methods = ["Spearman", "Pearson"]

    for s_id in sample_ids:
        for method in methods:
            file_path = os.path.join(filtered_dir, f"Sample_{s_id}_{method}_Sig.csv")
            if not os.path.exists(file_path): continue
            
            df = pd.read_csv(file_path)
            if 'Site' in df.columns:
                df['Chromosome'] = df['Site'].astype(str).str.split('_').str[0]
                counts = df['Chromosome'].value_counts().reset_index()
                counts.columns = ['Chromosome', 'Count']
                counts['Sample'] = s_id
                counts['Method'] = method
                all_counts.append(counts)

    if not all_counts:
        print("⚠️ 未发现可统计的位点数据")
        return

    summary_df = pd.concat(all_counts, ignore_index=True)
    
    for method in methods:
        plot_data = summary_df[summary_df['Method'] == method]
        if plot_data.empty: continue
        
        plt.figure(figsize=(12, 5))
        order = sorted(plot_data['Chromosome'].unique(), key=_chr_sort_key)
        sns.barplot(data=plot_data, x='Chromosome', y='Count', hue='Sample', order=order, palette='Set2')
        
        plt.title(f'Chromosome Distribution ({method})')
        plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
        plt.tight_layout()
        
        out_path = os.path.join(save_dir, f"Chr_Dist_{method}.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
    
    print(f"✅ 染色体分布图已保存至: {save_dir}")

    
def plot_correlation_chr_distribution(filtered_dir, sample_ids, save_dir):
    """6. 绘制 Spearman 和 Pearson 筛选后位点的染色体分布图"""
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    os.makedirs(save_dir, exist_ok=True)
    
    def chr_sort_key(c):
        return (0, int(c)) if str(c).isdigit() else (1, str(c))

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
        plt.legend(title='Sample', bbox_to_anchor=(1.01, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"Chr_Distribution_{method}.png"), dpi=300)
        plt.close()

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from adjustText import adjust_text

def plot_volcano_onlyC(deseq2_dir, save_dir, p_thresh=0.05, lfc_thresh=1.0, top_n=5, n_clusters=6):
    """Plots volcano plots for Cluster vs Others comparisons with automatic text labeling."""
    os.makedirs(save_dir, exist_ok=True)
    import warnings
    warnings.filterwarnings('ignore')
    
    for n in range(n_clusters):
        file_path = os.path.join(deseq2_dir, f"OnlyC_Cluster{n}_DESeq2_Results.csv")
        if not os.path.exists(file_path): continue
        
        df = pd.read_csv(file_path, index_col=0).dropna(subset=['pvalue', 'log2FoldChange'])
        df = df.sample(n=min(50000, len(df)), random_state=42)
        
        df['neg_log_p'] = -np.log10(df['pvalue'].replace(0, 1e-300))
        df['Group'] = 'Not Significant'
        df.loc[(df['pvalue'] < p_thresh) & (df['log2FoldChange'] > lfc_thresh), 'Group'] = 'Up-regulated'
        df.loc[(df['pvalue'] < p_thresh) & (df['log2FoldChange'] < -lfc_thresh), 'Group'] = 'Down-regulated'
        
        top_points = pd.concat([df[df['Group'] == 'Up-regulated'].nlargest(top_n, 'neg_log_p'),
                                df[df['Group'] == 'Down-regulated'].nlargest(top_n, 'neg_log_p')])
        
        plt.figure(figsize=(9, 7))
        sns.scatterplot(data=df, x='log2FoldChange', y='neg_log_p', hue='Group', 
                        palette={'Not Significant': '#E0E0E0', 'Up-regulated': '#FF4B4B', 'Down-regulated': '#4B7BFF'},
                        alpha=0.7, s=15, edgecolor=None)
        
        texts = [plt.text(row['log2FoldChange'], row['neg_log_p'], str(idx), fontsize=9, fontweight='bold') 
                 for idx, row in top_points.iterrows()]
        if texts: adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.8), expand_points=(1.5, 1.5))
        
        plt.axhline(-np.log10(p_thresh), color='black', linestyle='--', linewidth=0.8)
        plt.axvline(lfc_thresh, color='gray', linestyle='--', linewidth=0.8)
        plt.axvline(-lfc_thresh, color='gray', linestyle='--', linewidth=0.8)
        
        plt.title(f"Volcano Plot: Cluster {n} vs Others", fontsize=15, pad=15)
        plt.xlabel("log2 Fold Change"); plt.ylabel("-log10(p-value)")
        limit = max(abs(df['log2FoldChange'].min()), abs(df['log2FoldChange'].max())) * 1.1
        plt.xlim(-limit, limit); plt.grid(True, linestyle=':', alpha=0.4)
        
        plt.savefig(os.path.join(save_dir, f"Cluster{n}_volcano_labeled_onlyC.png"), dpi=300, bbox_inches='tight')
        plt.close()



def plot_correlation_boxplots(input_dir, output_base_dir, sample_ids, metrics=None):
    """
    Plots boxplots for correlation metrics (Spearman/Pearson) across target genes.
    """
    if metrics is None:
        metrics = ['Spearman_Rho', 'Pearson_R']
        
    warnings.filterwarnings('ignore')

    for metric in metrics:
        # Auto-configure folder and file suffix based on the chosen metric
        folder_name = "Spearman_Rho" if "Spearman" in metric else "Pearson_R"
        suffix = "spearman" if "Spearman" in metric else "pearson"
        
        output_dir = os.path.join(output_base_dir, folder_name)
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"📊 Plotting boxplots for {metric}...")
        
        for s_id in sample_ids:
            file_path = os.path.join(input_dir, f"Corr_Results_Sample_{s_id}.csv")
            if not os.path.exists(file_path): 
                continue
            
            df = pd.read_csv(file_path)
            if df.empty or metric not in df.columns:
                continue
            
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=df, x='Gene', y=metric, palette='Set3', showfliers=False)
            
            plt.axhline(0, color='red', linestyle='--', linewidth=1) 
            plt.title(f"Sample {s_id}: Distribution of {metric} across Genes")
            plt.ylabel(f"{metric} Value")
            plt.xlabel("Target Genes")
            
            save_path = os.path.join(output_dir, f"Sample_{s_id}_Boxplot_{suffix}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
    print(f"✅ All correlation boxplots generated successfully in: {output_base_dir}")