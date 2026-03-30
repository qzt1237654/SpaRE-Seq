library(DESeq2)

# 定义 A-to-I 编辑差异分析函数
DESeq2_test_by_cluster <- function(cluster_id) {
  set.seed(10)
  
  # 1. 路径配置
  input_path <- "C:/Users/28616/Desktop/Split_Matrices"
  output_dir <- "C:/Users/28616/Desktop/DESeq2_Results"
  if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)
  
  # 2. 读取数据
  # G 矩阵代表编辑后的计数 (Inosine)，A 矩阵代表原始碱基 A 的计数
  G_control <- read.csv(file.path(input_path, paste0("G_control_cluster", cluster_id, ".csv")), row.names=1)
  A_control <- read.csv(file.path(input_path, paste0("A_control_cluster", cluster_id, ".csv")), row.names=1)
  G_stress  <- read.csv(file.path(input_path, paste0("G_stress_cluster", cluster_id, ".csv")), row.names=1)
  A_stress  <- read.csv(file.path(input_path, paste0("A_stress_cluster", cluster_id, ".csv")), row.names=1)
  
  num_ctrl <- ncol(G_control)
  num_stress <- ncol(G_stress)
  
  # 3. 构造全样本 Count 矩阵
  # 布局：[Control_G, Stress_G, Control_A, Stress_A]
  counts <- cbind(G_control, G_stress, A_control, A_stress)
  
  # 4. 构建 Design 矩阵 (描述样本属性)
  # Trt: 分组 (control / stress)
  # Type: 碱基类型 (Edit_G / Base_A)
  design <- data.frame(
    Trt = rep(c(rep("control", num_ctrl), rep("stress", num_stress)), 2),
    Type = c(rep("Edit_G", (num_ctrl + num_stress)), rep("Base_A", (num_ctrl + num_stress)))
  )
  
  # 设置因子水平，确保对照组和原始碱基作为基准 (Reference)
  design$Trt <- factor(design$Trt, levels = c("control", "stress"))
  design$Type <- factor(design$Type, levels = c("Base_A", "Edit_G"))
  
  # 5. 构建 DESeq2 对象
  # 交互项 Type:Trt 代表编辑效率 (G/A) 随处理 (Trt) 的变化
  model <- ~ Type + Trt + Type:Trt
  dds <- DESeqDataSetFromMatrix(countData = round(as.matrix(counts)),
                                colData = design,
                                design = model)
  
  # 6. 自定义 SizeFactor 计算函数 (基于该位点总覆盖度 A + G)
  sizeFactor_internal <- function(data) {
    data <- as.matrix(data)
    log_data <- log(data)
    log_data[is.infinite(log_data)] <- NA
    log_mean <- rowMeans(log_data, na.rm = TRUE)
    log_s <- log_data - log_mean
    s_size <- exp(apply(log_s, 2, function(x) median(x, na.rm = TRUE)))
    return(s_size)
  }
  
  # 计算每个物理样本的总 Coverage (A+G) 并应用
  # 同一个样本的 G 和 A 必须共享同一个归一化因子
  total_coverage <- cbind((G_control + A_control), (G_stress + A_stress))
  sf <- sizeFactor_internal(total_coverage)
  sizeFactors(dds) <- rep(sf, 2)
  
  # 7. 运行 Wald 检验
  dds <- DESeq(dds, test = "Wald")
  
  # 8. 提取交互项结果 (Differential Editing Analysis)
  # 结果代表：log2((Stress_G/Stress_A) / (Control_G/Control_A))
  res <- DESeq2::results(dds, name = "TypeEdit_G.Trtstress")
  
  # 9. 输出结果
  output_file <- file.path(output_dir, paste0("DESeq2_Result_cluster", cluster_id, ".csv"))
  write.csv(as.data.frame(res), file = output_file)
  
  message(paste("Successfully processed A-to-I cluster:", cluster_id))
}

# --- 自动循环运行部分 (Cluster 0 到 8) ---

for (id in 0:8) {
  tryCatch({
    cat("\n>>> Processing A-to-I Editing for Cluster", id, "<<<\n")
    DESeq2_test_by_cluster(id)
  }, error = function(e) {
    cat("!!! Error in Cluster", id, ":", conditionMessage(e), "\n")
  })
}

cat("\nAll A-to-I editing analyses completed!\n")