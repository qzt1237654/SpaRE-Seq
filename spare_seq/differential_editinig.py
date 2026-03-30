import scanpy as sc
import pandas as pd
import numpy as np
import os
from rpy2.robjects.packages import importr
from rpy2.robjects import r, pandas2ri

def build_cluster_matrices(data_raw_dir, spclue_results_dir, output_dir, samples, n_clusters=6):
    """
    将 Bin-level 数据聚合为 Cluster-level 的 12xN 矩阵。
    对应你代码中“重构数据框”的部分。
    """
    # ... 这里封装你那段循环 samples 和 clusters 的聚合逻辑 ...
    print(f"✅ 所有 Cluster 矩阵已生成至: {output_dir}")

def run_deseq2_r(cluster_id, input_path, output_dir):
    """
    通过 rpy2 直接在 Python 中调用你写的 R 代码。
    这样别人就不需要手动打开 RStudio 了。
    """
    deseq2 = importr('DESeq2')
    # 这里我们可以通过 r.source() 加载你写的那个 R 脚本
    # 或者直接在 Python 里定义 R 字符串运行
    r_code = f"""
    # ... 这里放你那段 DESeq2_test_by_cluster 的 R 代码 ...
    DESeq2_test_by_cluster({cluster_id})
    """
    r(r_code)