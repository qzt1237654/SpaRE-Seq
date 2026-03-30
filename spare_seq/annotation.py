import pandas as pd
import gffutils
import os
import matplotlib.pyplot as plt

def build_or_load_gtf_db(gtf_path, db_path):
    """建立或加载 GTF 数据库"""
    if not os.path.exists(db_path):
        print("正在解析 GTF 文件建立地图数据库，请稍候（仅需一次）...")
        db = gffutils.create_db(gtf_path, db_path, force=True, keep_order=True,
                                merge_strategy='merge', sort_attribute_values=True)
    else:
        db = gffutils.FeatureDB(db_path)
    return db

def get_region_type(db, chrom, pos):
    """根据坐标在 GTF 数据库中查找该位点属于什么区域"""
    try:
        features = list(db.region(seqid=str(chrom), start=pos, end=pos))
    except ValueError:
        return "Intergenic"
    
    if not features:
        return "Intergenic"

    types = [f.featuretype for f in features]
    
    if 'CDS' in types: return 'CDS'
    if 'three_prime_utr' in types: return '3\' UTR'
    if 'five_prime_utr' in types: return '5\' UTR'
    if 'exon' in types: return 'Exon (Non-coding)'
    if 'gene' in types: return 'Intron'
    
    return "Other"

def annotate_clusters(results_dir, gtf_path, db_path, num_clusters=6):
    """
    核心调用函数：对指定文件夹下的所有 Cluster DESeq2 结果进行注释并统计。
    """
    db = build_or_load_gtf_db(gtf_path, db_path)
    summary_list = []

    for n in range(num_clusters):
        file_name = f"DESeq2_Result_cluster{n}.csv"
        file_path = os.path.join(results_dir, file_name)
        
        if not os.path.exists(file_path):
            continue
        
        df = pd.read_csv(file_path)
        df[['chr', 'pos']] = df.iloc[:, 0].str.split('_', expand=True)
        df['pos'] = df['pos'].astype(int)
        
        p_col = [c for c in df.columns if 'pvalue' in c.lower() or 'p.value' in c.lower()][0]
        significant_df = df[(df[p_col] < 0.05) & (df['log2FoldChange'].abs() > 1)].copy()
        
        if significant_df.empty:
            continue

        significant_df['Region'] = significant_df.apply(lambda x: get_region_type(db, x['chr'], x['pos']), axis=1)
        
        counts = significant_df['Region'].value_counts()
        summary_list.append({"Cluster": n, **counts.to_dict()})
        significant_df.to_csv(os.path.join(results_dir, f"Annotated_Cluster_{n}.csv"), index=False)

    if summary_list:
        summary_df = pd.DataFrame(summary_list).fillna(0)
        print("✅ 注释完成！")
        return summary_df
    else:
        print("⚠️ 所有 Cluster 均无显著位点。")
        return None