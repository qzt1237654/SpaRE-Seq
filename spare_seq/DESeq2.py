import os
import gc
import re
import numpy as np
import pandas as pd
import anndata
from scipy.sparse import issparse
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text
import warnings
import gffutils

# ==========================================
# INTERNAL GENOMIC ANNOTATION TOOLS
# ==========================================
def _build_or_load_gtf_db(gtf_path, db_path):
    if not os.path.exists(db_path):
        print(f"Parsing GTF: {gtf_path}. This will create {db_path}...")
        db = gffutils.create_db(gtf_path, db_path, force=True, keep_order=True,
                                merge_strategy='merge', sort_attribute_values=True)
    else:
        db = gffutils.FeatureDB(db_path)
    return db

def _get_region_type(db, chrom, pos):
    try:
        features = list(db.region(seqid=str(chrom), start=pos, end=pos))
    except ValueError:
        return "Intergenic"
    if not features:
        return "Intergenic"

    types = [f.featuretype for f in features]
    if 'CDS' in types: return 'CDS'
    if 'three_prime_utr' in types: return "3' UTR"
    if 'five_prime_utr' in types: return "5' UTR"
    if 'exon' in types: return 'Exon (Non-coding)'
    if 'gene' in types: return 'Intron'
    return "Other"

def run_annotation(results_dir, gtf_path, db_path, num_clusters=6):
    """
    Annotate significant sites and calculate percentages for each region.
    """
    db = _build_or_load_gtf_db(gtf_path, db_path)
    summary_list = []

    for n in range(num_clusters):
        file_path = os.path.join(results_dir, f"Cluster{n}_Significant_Sites.csv")
        if not os.path.exists(file_path): continue
        
        df = pd.read_csv(file_path)
        total_sites = len(df)
        if total_sites == 0: continue

        df[['chr', 'pos']] = df.iloc[:, 0].str.split('_', expand=True)
        df['pos'] = df['pos'].astype(int)
        df['Region'] = df.apply(lambda x: _get_region_type(db, x['chr'], x['pos']), axis=1)
        
        df.to_csv(os.path.join(results_dir, f"Annotated_Cluster_{n}.csv"), index=False)
        
        counts = df['Region'].value_counts()
        cluster_stats = {"Cluster": n, "Total_Significant_Sites": total_sites}
        for region, count in counts.items():
            cluster_stats[f"{region}_Count"] = count
            cluster_stats[f"{region}_Pct (%)"] = round((count / total_sites) * 100, 2)
            
        summary_list.append(cluster_stats)

    if summary_list:
        summary_df = pd.DataFrame(summary_list).fillna(0)
        summary_df.to_csv(os.path.join(results_dir, "Annotation_Summary_Report.csv"), index=False)
        print(f"✅ Annotation completed for {results_dir}")
        return summary_df
    return None

# ==========================================
# 1. Stress vs Control Pipeline
# ==========================================
def run_test_vs_control(data_raw_dir, list_dir, output_dir, ctrl_samples, stress_samples, n_clusters=6, layer_key='AGcount_A', p_thresh=0.05, lfc_thresh=1.0, top_n_labels=5, plot_barplot=True):
    """End-to-end pipeline for Stress vs Control."""
    from rpy2.robjects import r
    warnings.filterwarnings('ignore')

    matrix_dir = os.path.join(output_dir, "raw_matrices")
    deseq2_dir = os.path.join(output_dir, "deseq2_raw_results")
    final_dir = os.path.join(output_dir, "significant_results")
    volcano_dir = os.path.join(output_dir, "volcano_plots")
    for d in [matrix_dir, deseq2_dir, final_dir, volcano_dir]: os.makedirs(d, exist_ok=True)

    print("📊 [1/4] Generating input matrices from h5ad data...")
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
                with open(txt_path, 'r') as f: target_ids = [line.strip() for line in f if line.strip()]
                sub = adata_sub[adata_sub.obs_names.isin(target_ids)]
                if sub.n_obs > 0:
                    current_map = [site_to_idx[g] for g in adata_sub.var_names]
                    sum_G = sub.X.sum(axis=0)
                    vec_G[current_map] = np.array(sum_G).flatten() if issparse(sum_G) else np.ravel(sum_G)
                    if layer_key in sub.layers:
                        sum_A = sub.layers[layer_key].sum(axis=0)
                        vec_A[current_map] = np.array(sum_A).flatten() if issparse(sum_A) else np.ravel(sum_A)
                del adata_sub; gc.collect()
            c_rows.extend([vec_G, vec_A]); c_labels.extend([f"{s}_G", f"{s}_A"])
            
        df_cluster = pd.DataFrame(np.vstack(c_rows), index=c_labels, columns=site_list)
        g_data = df_cluster.iloc[0::2].values; a_data = df_cluster.iloc[1::2].values
        ag_median = np.median(g_data + a_data, axis=0); g_count = (g_data > 0).sum(axis=0)
        valid_sites = df_cluster.columns[(ag_median >= 10) & (g_count >= 3)]
        df_filtered = df_cluster[valid_sites].T
        
        c_dir = os.path.join(matrix_dir, f"cluster{c}")
        os.makedirs(c_dir, exist_ok=True)
        df_filtered[[f"{s}_G" for s in ctrl_samples]].to_csv(os.path.join(c_dir, f"G_control_cluster{c}.csv"))
        df_filtered[[f"{s}_A" for s in ctrl_samples]].to_csv(os.path.join(c_dir, f"A_control_cluster{c}.csv"))
        df_filtered[[f"{s}_G" for s in stress_samples]].to_csv(os.path.join(c_dir, f"G_stress_cluster{c}.csv"))
        df_filtered[[f"{s}_A" for s in stress_samples]].to_csv(os.path.join(c_dir, f"A_stress_cluster{c}.csv"))

    print("🧬 [2/4] Executing DESeq2 via rpy2...")
    r_input_path = matrix_dir.replace('\\', '/'); r_output_path = deseq2_dir.replace('\\', '/')
    r_code = f"""
    suppressPackageStartupMessages(library(DESeq2)); options(warn=-1)
    input_path <- "{r_input_path}"; output_dir <- "{r_output_path}"
    for (cluster_id in 0:{n_clusters-1}) {{
        c_dir <- file.path(input_path, paste0("cluster", cluster_id))
        g_ctrl_file <- file.path(c_dir, paste0("G_control_cluster", cluster_id, ".csv"))
        if (!file.exists(g_ctrl_file)) next
        G_control <- read.csv(g_ctrl_file, row.names=1)
        A_control <- read.csv(file.path(c_dir, paste0("A_control_cluster", cluster_id, ".csv")), row.names=1)
        G_stress  <- read.csv(file.path(c_dir, paste0("G_stress_cluster", cluster_id, ".csv")), row.names=1)
        A_stress  <- read.csv(file.path(c_dir, paste0("A_stress_cluster", cluster_id, ".csv")), row.names=1)
        counts <- cbind(G_control, G_stress, A_control, A_stress)
        num_ctrl <- ncol(G_control); num_stress <- ncol(G_stress)
        design <- data.frame(
            Trt = factor(rep(c(rep("control", num_ctrl), rep("stress", num_stress)), 2), levels = c("control", "stress")),
            Type = factor(c(rep("Edit_G", (num_ctrl + num_stress)), rep("Base_A", (num_ctrl + num_stress))), levels = c("Base_A", "Edit_G"))
        )
        dds <- DESeqDataSetFromMatrix(countData = round(as.matrix(counts)), colData = design, design = ~ Type + Trt + Type:Trt)
        total_coverage <- cbind((G_control + A_control), (G_stress + A_stress))
        log_data <- log(as.matrix(total_coverage)); log_data[is.infinite(log_data)] <- NA
        log_s <- log_data - rowMeans(log_data, na.rm = TRUE)
        sf <- exp(apply(log_s, 2, function(x) median(x, na.rm = TRUE)))
        sizeFactors(dds) <- rep(sf, 2)
        dds <- DESeq(dds, test = "Wald", quiet = TRUE)
        res <- results(dds, name = "TypeEdit_G.Trtstress")
        write.csv(as.data.frame(res), file = file.path(output_dir, paste0("StressVsCtrl_Cluster", cluster_id, "_DESeq2_Results.csv")))
    }}
    """
    try: r(r_code)
    except Exception as e: print(f"❌ DESeq2 failed: {e}"); return

    print(f"🧹 [3/4] Cleaning prefix and filtering (P < {p_thresh}, |Log2FC| > {lfc_thresh})...")
    for c in range(n_clusters):
        file_path = os.path.join(deseq2_dir, f"StressVsCtrl_Cluster{c}_DESeq2_Results.csv")
        if not os.path.exists(file_path): continue
        df = pd.read_csv(file_path, index_col=0)
        df.index = [re.sub(r'^X(\d)', r'\1', str(idx)) for idx in df.index]
        df.to_csv(file_path)
        
        df_sig = df.dropna(subset=['pvalue', 'log2FoldChange'])
        df_sig = df_sig[(df_sig['pvalue'] < p_thresh) & (df_sig['log2FoldChange'].abs() > lfc_thresh)].copy()
        df_final = df_sig[['pvalue', 'log2FoldChange']].sort_values(by='log2FoldChange', ascending=False)
        df_final.to_csv(os.path.join(final_dir, f"Cluster{c}_Significant_Sites.csv"))
        with open(os.path.join(final_dir, f"Cluster{c}_sites_list.txt"), 'w') as f:
            for site in df_final.index: f.write(f"{site}\n")
        print(f"✅ Cluster {c}: Retained {len(df_final)} sites.")

    print(f"🌋 [4/4] Generating Volcano Plots with top {top_n_labels} labels...")
    for c in range(n_clusters):
        file_path = os.path.join(deseq2_dir, f"StressVsCtrl_Cluster{c}_DESeq2_Results.csv")
        if not os.path.exists(file_path): continue
        df = pd.read_csv(file_path, index_col=0).dropna(subset=['pvalue', 'log2FoldChange'])
        df = df.sample(n=min(50000, len(df)), random_state=42)
        df['neg_log_p'] = -np.log10(df['pvalue'].replace(0, 1e-300))
        df['Group'] = 'Not Significant'
        df.loc[(df['pvalue'] < p_thresh) & (df['log2FoldChange'] > lfc_thresh), 'Group'] = 'Up-regulated'
        df.loc[(df['pvalue'] < p_thresh) & (df['log2FoldChange'] < -lfc_thresh), 'Group'] = 'Down-regulated'
        
        top_points = pd.concat([df[df['Group'] == 'Up-regulated'].nlargest(top_n_labels, 'neg_log_p'),
                                df[df['Group'] == 'Down-regulated'].nlargest(top_n_labels, 'neg_log_p')])
        plt.figure(figsize=(9, 7))
        sns.scatterplot(data=df, x='log2FoldChange', y='neg_log_p', hue='Group', 
                        palette={'Not Significant': '#E0E0E0', 'Up-regulated': '#FF4B4B', 'Down-regulated': '#4B7BFF'},
                        alpha=0.7, s=15, edgecolor=None)
        texts = [plt.text(row['log2FoldChange'], row['neg_log_p'], str(idx), fontsize=9, fontweight='bold') for idx, row in top_points.iterrows()]
        if texts: adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.8), expand_points=(1.5, 1.5))
        plt.axhline(-np.log10(p_thresh), color='black', linestyle='--', linewidth=0.8)
        plt.axvline(lfc_thresh, color='gray', linestyle='--', linewidth=0.8)
        plt.axvline(-lfc_thresh, color='gray', linestyle='--', linewidth=0.8)
        plt.title(f"Volcano Plot: Cluster {c} (Stress vs Control)", fontsize=15, pad=15)
        plt.xlabel("log2 Fold Change"); plt.ylabel("-log10(p-value)")
        limit = max(abs(df['log2FoldChange'].min()), abs(df['log2FoldChange'].max())) * 1.1
        plt.xlim(-limit, limit); plt.grid(True, linestyle=':', alpha=0.4)
        plt.savefig(os.path.join(volcano_dir, f"Cluster{c}_volcano_labeled_StressVsCtrl.png"), dpi=300, bbox_inches='tight')
        plt.close()

    # ----------------------------------------------------
    # (Stress vs Control)
    # ----------------------------------------------------
    if plot_barplot:
        print("📊 The number of significant points of each Cluster is being counted and a bar chart is being drawn...")
        data = []
        for c in range(n_clusters):
            file_path = os.path.join(final_dir, f"Cluster{c}_Significant_Sites.csv")
            if os.path.exists(file_path):
                df_bar = pd.read_csv(file_path, index_col=0)
                # 由于已经按照阈值过滤过了，直接用正负号判断即可
                up_count = len(df_bar[df_bar['log2FoldChange'] > 0])
                down_count = len(df_bar[df_bar['log2FoldChange'] < 0])
                data.append({'Cluster': f'Cluster {c}', 'Count': up_count, 'Direction': 'Up-regulated (Stress > Ctrl)'})
                data.append({'Cluster': f'Cluster {c}', 'Count': down_count, 'Direction': 'Down-regulated (Stress < Ctrl)'})
            else:
                data.append({'Cluster': f'Cluster {c}', 'Count': 0, 'Direction': 'Up-regulated (Stress > Ctrl)'})
                data.append({'Cluster': f'Cluster {c}', 'Count': 0, 'Direction': 'Down-regulated (Stress < Ctrl)'})

        df_plot = pd.DataFrame(data)
        if not df_plot.empty and df_plot['Count'].sum() > 0:
            plt.figure(figsize=(10, 6))
            colors = {'Up-regulated (Stress > Ctrl)': '#e74c3c', 'Down-regulated (Stress < Ctrl)': '#3498db'}
            ax = sns.barplot(data=df_plot, y='Cluster', x='Count', hue='Direction', palette=colors)
            plt.title('Number of Stress-Responsive A-to-I Editing Sites per Cluster', fontsize=15, pad=15)
            plt.xlabel(f'Number of Significant Sites (p < {p_thresh}, |log2FC| > {lfc_thresh})', fontsize=12)
            plt.ylabel('Brain Region / Cell Type', fontsize=12)

            for p in ax.patches:
                width = p.get_width()
                if width > 0:
                    ax.text(width + max(df_plot['Count'].max(), 1) * 0.01, p.get_y() + p.get_height() / 2, 
                            f'{int(width)}', va='center', fontsize=10)

            plt.legend(title='Editing Change under Stress', loc='lower right')
            plt.grid(axis='x', linestyle='--', alpha=0.6)
            save_path = os.path.join(volcano_dir, "Stress_Responsive_Sites_Barplot.png")
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✨ The bar chart drawing is completed! The picture has been saved to: {save_path}")

    print(f"🎉 Pipeline completed successfully! Check the outputs in: {output_dir}")

# ==========================================
# 2. Target Cluster vs Others Pipeline
# ==========================================
def run_cluster_vs_others(data_raw_dir, list_dir, output_dir, samples, n_clusters=6, layer_key='AGcount_A', p_thresh=0.05, lfc_thresh=1.0, top_n_labels=5, plot_barplot=True):
    """End-to-end pipeline for Cluster vs Others."""
    from rpy2.robjects import r
    warnings.filterwarnings('ignore')

    matrix_dir = os.path.join(output_dir, "raw_matrices")
    deseq2_dir = os.path.join(output_dir, "deseq2_raw_results")
    final_dir = os.path.join(output_dir, "significant_results")
    volcano_dir = os.path.join(output_dir, "volcano_plots")
    for d in [matrix_dir, deseq2_dir, final_dir, volcano_dir]: os.makedirs(d, exist_ok=True)

    print("📊 [1/4] Generating input matrices from h5ad data...")
    all_sites = set()
    for s in samples:
        path = os.path.join(data_raw_dir, f"GSM81924{s}_AtoI_tissue.h5ad")
        if os.path.exists(path):
            adata_ref = anndata.read_h5ad(path, backed='r')
            all_sites.update(adata_ref.var_names)
            adata_ref.file.close()
            
    site_list = sorted(list(all_sites)); site_to_idx = {s: i for i, s in enumerate(site_list)}; n_sites = len(site_list)

    cluster_dict = {}
    for c in range(n_clusters):
        rows, labels = [], []
        for s in samples:
            v_G, v_A = np.zeros(n_sites), np.zeros(n_sites)
            h5ad_path = os.path.join(data_raw_dir, f"GSM81924{s}_AtoI_tissue.h5ad")
            txt_path = os.path.join(list_dir, f"{s}_cluster{c}.txt")
            if os.path.exists(h5ad_path) and os.path.exists(txt_path):
                adata_sub = anndata.read_h5ad(h5ad_path)
                with open(txt_path, 'r') as f: t_ids = [line.strip() for line in f if line.strip()]
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
        g_data_1 = df_target.iloc[0::2].values; a_data_1 = df_target.iloc[1::2].values
        mask_1 = (np.median(g_data_1 + a_data_1, axis=0) >= 10) & ((g_data_1 > 0).sum(axis=0) >= 3)
        valid_sites_1 = df_target.columns[mask_1]
        
        df_t_sub = df_target[valid_sites_1]
        other_dfs = [cluster_dict[i][valid_sites_1] for i in range(n_clusters) if i != target_c]
        df_ex_sub = sum(other_dfs)
        
        df_t_sub.index = [f"{x}_{target_c}" for x in df_t_sub.index]; df_ex_sub.index = [f"{x}_no{target_c}" for x in df_ex_sub.index]
        df_concat = pd.concat([df_t_sub, df_ex_sub], axis=0)
        
        g_rows = [r for r in df_concat.index if "_G_" in r]; a_rows = [r for r in df_concat.index if "_A_" in r]
        mask_2 = (np.median(df_concat.loc[g_rows].values + df_concat.loc[a_rows].values, axis=0) > 10) & ((df_concat.loc[g_rows].values > 0).sum(axis=0) >= 3)
        final_sites = df_concat.columns[mask_2]
        
        c_dir = os.path.join(matrix_dir, f"cluster{target_c}"); no_c_dir = os.path.join(matrix_dir, f"no_cluster{target_c}")
        os.makedirs(c_dir, exist_ok=True); os.makedirs(no_c_dir, exist_ok=True)
        df_concat.loc[df_t_sub.index, final_sites].to_csv(os.path.join(c_dir, f"cluster{target_c}_filtered.csv"))
        df_concat.loc[df_ex_sub.index, final_sites].to_csv(os.path.join(no_c_dir, f"no_cluster{target_c}_filtered.csv"))

    print("🧬 [2/4] Executing DESeq2 via rpy2...")
    r_input_path = matrix_dir.replace('\\', '/'); r_output_path = deseq2_dir.replace('\\', '/')
    r_code = f"""
    suppressPackageStartupMessages(library(DESeq2)); options(warn=-1)
    input_path <- "{r_input_path}"; output_dir <- "{r_output_path}"
    for (cluster_id in 0:{n_clusters-1}) {{
        file_cluster <- file.path(input_path, paste0("cluster", cluster_id), paste0("cluster", cluster_id, "_filtered.csv"))
        file_except  <- file.path(input_path, paste0("no_cluster", cluster_id), paste0("no_cluster", cluster_id, "_filtered.csv"))
        if (!file.exists(file_cluster)) next
        data_cluster <- t(read.csv(file_cluster, row.names=1)); data_except  <- t(read.csv(file_except, row.names=1))
        G_cluster <- data_cluster[, grep("_G_", colnames(data_cluster)), drop=FALSE]; A_cluster <- data_cluster[, grep("_A_", colnames(data_cluster)), drop=FALSE]
        G_except  <- data_except[, grep("_G_", colnames(data_except)), drop=FALSE]; A_except  <- data_except[, grep("_A_", colnames(data_except)), drop=FALSE]
        counts <- cbind(G_except, G_cluster, A_except, A_cluster)
        num_cluster <- ncol(G_cluster); num_except  <- ncol(G_except)
        design <- data.frame(
            Group = factor(rep(c(rep("Except", num_except), rep("Cluster", num_cluster)), 2), levels = c("Except", "Cluster")),
            Type  = factor(c(rep("Edit_G", (num_except + num_cluster)), rep("Base_A", (num_except + num_cluster))), levels = c("Base_A", "Edit_G"))
        )
        dds <- DESeqDataSetFromMatrix(countData = round(as.matrix(counts)), colData = design, design = ~ Type + Group + Type:Group)
        total_coverage <- cbind((G_except + A_except), (G_cluster + A_cluster))
        log_data <- log(as.matrix(total_coverage)); log_data[is.infinite(log_data)] <- NA
        log_s <- log_data - rowMeans(log_data, na.rm = TRUE)
        sf <- exp(apply(log_s, 2, function(x) median(x, na.rm = TRUE)))
        sizeFactors(dds) <- rep(sf, 2)
        dds <- DESeq(dds, test = "Wald", quiet = TRUE)
        res <- results(dds, name = "TypeEdit_G.GroupCluster")
        write.csv(as.data.frame(res), file = file.path(output_dir, paste0("OnlyC_Cluster", cluster_id, "_DESeq2_Results.csv")))
    }}
    """
    try: r(r_code)
    except Exception as e: print(f"❌ DESeq2 failed: {e}"); return

    print(f"🧹 [3/4] Cleaning prefix and filtering (P < {p_thresh}, |Log2FC| > {lfc_thresh})...")
    for c in range(n_clusters):
        file_path = os.path.join(deseq2_dir, f"OnlyC_Cluster{c}_DESeq2_Results.csv")
        if not os.path.exists(file_path): continue
        df = pd.read_csv(file_path, index_col=0)
        df.index = [re.sub(r'^X(\d)', r'\1', str(idx)) for idx in df.index]
        df.to_csv(file_path)
        
        df_sig = df.dropna(subset=['pvalue', 'log2FoldChange'])
        df_sig = df_sig[(df_sig['pvalue'] < p_thresh) & (df_sig['log2FoldChange'].abs() > lfc_thresh)].copy()
        df_final = df_sig[['pvalue', 'log2FoldChange']].sort_values(by='log2FoldChange', ascending=False)
        df_final.to_csv(os.path.join(final_dir, f"Cluster{c}_Significant_Sites.csv"))
        with open(os.path.join(final_dir, f"Cluster{c}_sites_list.txt"), 'w') as f:
            for site in df_final.index: f.write(f"{site}\n")
        print(f"✅ Cluster {c}: Retained {len(df_final)} sites.")

    print(f"🌋 [4/4] Generating Volcano Plots with top {top_n_labels} labels...")
    for c in range(n_clusters):
        file_path = os.path.join(deseq2_dir, f"OnlyC_Cluster{c}_DESeq2_Results.csv")
        if not os.path.exists(file_path): continue
        df = pd.read_csv(file_path, index_col=0).dropna(subset=['pvalue', 'log2FoldChange'])
        df = df.sample(n=min(50000, len(df)), random_state=42)
        df['neg_log_p'] = -np.log10(df['pvalue'].replace(0, 1e-300))
        df['Group'] = 'Not Significant'
        df.loc[(df['pvalue'] < p_thresh) & (df['log2FoldChange'] > lfc_thresh), 'Group'] = 'Up-regulated'
        df.loc[(df['pvalue'] < p_thresh) & (df['log2FoldChange'] < -lfc_thresh), 'Group'] = 'Down-regulated'
        
        top_points = pd.concat([df[df['Group'] == 'Up-regulated'].nlargest(top_n_labels, 'neg_log_p'),
                                df[df['Group'] == 'Down-regulated'].nlargest(top_n_labels, 'neg_log_p')])
        plt.figure(figsize=(9, 7))
        sns.scatterplot(data=df, x='log2FoldChange', y='neg_log_p', hue='Group', 
                        palette={'Not Significant': '#E0E0E0', 'Up-regulated': '#FF4B4B', 'Down-regulated': '#4B7BFF'},
                        alpha=0.7, s=15, edgecolor=None)
        texts = [plt.text(row['log2FoldChange'], row['neg_log_p'], str(idx), fontsize=9, fontweight='bold') for idx, row in top_points.iterrows()]
        if texts: adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.8), expand_points=(1.5, 1.5))
        plt.axhline(-np.log10(p_thresh), color='black', linestyle='--', linewidth=0.8)
        plt.axvline(lfc_thresh, color='gray', linestyle='--', linewidth=0.8)
        plt.axvline(-lfc_thresh, color='gray', linestyle='--', linewidth=0.8)
        plt.title(f"Volcano Plot: Cluster {c} (Cluster vs Others)", fontsize=15, pad=15)
        plt.xlabel("log2 Fold Change"); plt.ylabel("-log10(p-value)")
        limit = max(abs(df['log2FoldChange'].min()), abs(df['log2FoldChange'].max())) * 1.1
        plt.xlim(-limit, limit); plt.grid(True, linestyle=':', alpha=0.4)
        plt.savefig(os.path.join(volcano_dir, f"Cluster{c}_volcano_labeled_ClusterVsOthers.png"), dpi=300, bbox_inches='tight')
        plt.close()

    # ----------------------------------------------------
    #  (Cluster vs Others)
    # ----------------------------------------------------
    if plot_barplot:
        print("📊 The number of specific sites for each Cluster is being counted and a bar chart is being drawn...")
        data = []
        for c in range(n_clusters):
            file_path = os.path.join(final_dir, f"Cluster{c}_Significant_Sites.csv")
            if os.path.exists(file_path):
                df_bar = pd.read_csv(file_path, index_col=0)
                # A value greater than 0 indicates that this Cluster is edited higher than other regions
                up_count = len(df_bar[df_bar['log2FoldChange'] > 0])
                down_count = len(df_bar[df_bar['log2FoldChange'] < 0])
                data.append({'Cluster': f'Cluster {c}', 'Count': up_count, 'Direction': 'Higher in this Cluster'})
                data.append({'Cluster': f'Cluster {c}', 'Count': down_count, 'Direction': 'Lower in this Cluster'})
            else:
                data.append({'Cluster': f'Cluster {c}', 'Count': 0, 'Direction': 'Higher in this Cluster'})
                data.append({'Cluster': f'Cluster {c}', 'Count': 0, 'Direction': 'Lower in this Cluster'})

        df_plot = pd.DataFrame(data)
        if not df_plot.empty and df_plot['Count'].sum() > 0:
            plt.figure(figsize=(10, 6))
            colors = {'Higher in this Cluster': '#e74c3c', 'Lower in this Cluster': '#3498db'}
            ax = sns.barplot(data=df_plot, y='Cluster', x='Count', hue='Direction', palette=colors)
            plt.title('Number of Cluster-Specific A-to-I Editing Sites', fontsize=15, pad=15)
            plt.xlabel(f'Number of Significant Sites (p < {p_thresh}, |log2FC| > {lfc_thresh})', fontsize=12)
            plt.ylabel('Brain Region / Cell Type', fontsize=12)

            for p in ax.patches:
                width = p.get_width()
                if width > 0:
                    ax.text(width + max(df_plot['Count'].max(), 1) * 0.01, p.get_y() + p.get_height() / 2, 
                            f'{int(width)}', va='center', fontsize=10)

            plt.legend(title='Editing Change vs Others', loc='lower right')
            plt.grid(axis='x', linestyle='--', alpha=0.6)
            save_path = os.path.join(volcano_dir, "Cluster_Specific_Sites_Barplot.png")
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✨ The bar chart drawing is completed! The picture has been saved to: {save_path}")

    print(f"🎉 Pipeline completed successfully! Check the outputs in: {output_dir}")