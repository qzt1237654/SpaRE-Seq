library(DESeq2)

# 定义 A-to-I 编辑差异分析函数 (Cluster vs Except_Cluster)
DESeq2_test_cluster_vs_others <- function(cluster_id) {
  set.seed(10)
  
  # 1. 路径配置 (对接你 Python 上一步的输出路径)
  input_path <- "C:/Users/28616/Desktop/DESeq2_onlyC"
  output_dir <- "C:/Users/28616/Desktop/DESeq2_Results"
  if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)
  
  # 2. 读取数据 (当前 Cluster vs 剩余其他 Cluster)
  file_cluster <- file.path(input_path, paste0("cluster", cluster_id), paste0("cluster", cluster_id, "_filtered.csv"))
  file_except  <- file.path(input_path, paste0("no_cluster", cluster_id), paste0("no_cluster", cluster_id, "_filtered.csv"))
  
  # 读取并转置矩阵 (DESeq2 要求行是位点，列是样本)
  data_cluster <- t(read.csv(file_cluster, row.names=1))
  data_except  <- t(read.csv(file_except, row.names=1))
  
  # 3. 自动分离 G 和 A 的计数矩阵
  # 通过列名中是否包含 "_G_" 或 "_A_" 来拆分
  G_cluster <- data_cluster[, grep("_G_", colnames(data_cluster)), drop=FALSE]
  A_cluster <- data_cluster[, grep("_A_", colnames(data_cluster)), drop=FALSE]
  
  G_except  <- data_except[, grep("_G_", colnames(data_except)), drop=FALSE]
  A_except  <- data_except[, grep("_A_", colnames(data_except)), drop=FALSE]
  
  num_cluster <- ncol(G_cluster)
  num_except  <- ncol(G_except)
  
  # 4. 构造全样本 Count 矩阵
  # 布局：[Except_G, Cluster_G, Except_A, Cluster_A]
  counts <- cbind(G_except, G_cluster, A_except, A_cluster)
  
  # 5. 构建 Design 矩阵 (描述样本属性)
  # Group: 分组 (Except / Cluster) -> Except 作为对照组
  # Type: 碱基类型 (Edit_G / Base_A) -> Base_A 作为基准
  design <- data.frame(
    Group = rep(c(rep("Except", num_except), rep("Cluster", num_cluster)), 2),
    Type  = c(rep("Edit_G", (num_except + num_cluster)), rep("Base_A", (num_except + num_cluster)))
  )
  
  # 设置因子水平，确保 Except 和 Base_A 作为对比基准
  design$Group <- factor(design$Group, levels = c("Except", "Cluster"))
  design$Type <- factor(design$Type, levels = c("Base_A", "Edit_G"))
  
  # 6. 构建 DESeq2 对象
  # 交互项 Type:Group 代表: 当前 Cluster 相较于其他所有 Cluster，其编辑效率 (G/A比例) 是否有显著变化
  model <- ~ Type + Group + Type:Group
  dds <- DESeqDataSetFromMatrix(countData = round(as.matrix(counts)),
                                colData = design,
                                design = model)
  
  # 7. 自定义 SizeFactor 计算函数 (基于该位点总覆盖度 A + G)
  sizeFactor_internal <- function(data) {
    data <- as.matrix(data)
    log_data <- log(data)
    log_data[is.infinite(log_data)] <- NA
    log_mean <- rowMeans(log_data, na.rm = TRUE)
    log_s <- log_data - log_mean
    s_size <- exp(apply(log_s, 2, function(x) median(x, na.rm = TRUE)))
    return(s_size)
  }
  
  # 计算总 Coverage (A+G) 以计算归一化因子
  # 同一个物理样本的 G 和 A 必须共享同一个归一化因子
  total_coverage <- cbind((G_except + A_except), (G_cluster + A_cluster))
  sf <- sizeFactor_internal(total_coverage)
  sizeFactors(dds) <- rep(sf, 2)
  
  # 8. 运行 Wald 检验
  dds <- DESeq(dds, test = "Wald")
  
  # 9. 提取交互项结果 (Differential Editing Analysis)
  # 提取的名称必须和 dds 模型里的内部名称完全一致
  # 这里提取的是 "TypeEdit_G.GroupCluster"
  res <- DESeq2::results(dds, name = "TypeEdit_G.GroupCluster")
  
  # 10. 输出结果
  output_file <- file.path(output_dir, paste0("OnlyC_Cluster", cluster_id, "_DESeq2_Results.csv"))
  write.csv(as.data.frame(res), file = output_file)
  
  message(paste("✅ Successfully processed Cluster:", cluster_id, "vs Except_Cluster"))
}

# --- 自动循环运行部分 (Cluster 0 到 5) ---
cat("\n🚀 -------------------------------------- 🚀\n")
cat("   Starting Differential Editing Analysis\n")
cat("🚀 -------------------------------------- 🚀\n")

for (id in 0:5) {
  tryCatch({
    cat("\n>>> Processing Cluster", id, "vs All Others <<<\n")
    DESeq2_test_cluster_vs_others(id)
  }, error = function(e) {
    cat("!!! Error in Cluster", id, ":", conditionMessage(e), "\n")
  })
}

cat("\n🎉 All 6 Cluster analyses completed!\n")