# ==============================================================================
# 合并版 t-SNE 降维与可视化脚本 (动态参数驱动)
# ==============================================================================

# 自动检测并载入依赖的 R 包
required_packages <- c("ggplot2", "Rtsne")
for (pkg in required_packages) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org")
    library(pkg, character.only = TRUE)
  }
}

# 1. 接收来自 Python 的命令行参数
args <- commandArgs(trailingOnly = TRUE)

input_csv      <- ifelse(length(args) >= 1, args[1], "data_matrix.csv")
group_csv      <- ifelse(length(args) >= 2, args[2], "group_labels.csv")
output_dir     <- ifelse(length(args) >= 3, args[3], "./")
perplexity_val <- ifelse(length(args) >= 4, as.numeric(args[4]), 30)
point_size     <- ifelse(length(args) >= 5, as.numeric(args[5]), 1.2)
point_alpha    <- ifelse(length(args) >= 6, as.numeric(args[6]), 0.6)

# 2. 读取纯特征矩阵并降维
A <- read.csv(input_csv, header = TRUE, row.names = 1)
Atsne <- Rtsne(t(A), perplexity = perplexity_val, check_duplicates = FALSE)
colnames(Atsne$Y) <- c("TSNE1", "TSNE2")

# ==============================================================================
# 3. 读取动态分组与 【动态颜色】
# ==============================================================================
if (file.exists(group_csv)) {
  groups_df <- read.csv(group_csv, header = TRUE)
  types <- groups_df$Type

  # 【核心修复2】：从传过来的表里，提炼出 Type 和 Color 的对应关系
  color_map <- unique(groups_df[, c("Type", "Color")])
  # 转换为 ggplot 认得的命名向量 (例如: c("C"="#0000FF", "NC"="#8B00FF"))
  define_color <- setNames(as.character(color_map$Color), as.character(color_map$Type))
} else {
  types <- rep("Unknown", ncol(A))
  define_color <- c("Unknown" = "#808080")
}

Atsne_data <- data.frame(
  sample = colnames(A),
  Type = types,
  Atsne$Y
)

# 4. ggplot2 绘图
p <- ggplot(Atsne_data, aes(x = TSNE1, y = TSNE2, colour = Type)) +
  geom_point(size = point_size, alpha = point_alpha, stroke = 0) +
  scale_color_manual(values = define_color) +  # 【核心修复3】：强制套用自定义颜色字典！
  xlab("TSNE1") + ylab("TSNE2") +
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        legend.title = element_blank(),
        legend.text = element_text(face = "bold", color = "black", size = 10),
        panel.border = element_blank(),
        axis.line = element_line(color = "black", linewidth = 1), # 修复了 R 的 linewidth 警告
        axis.ticks = element_line(linewidth = 1),                 # 修复了 R 的 linewidth 警告
        axis.text = element_text(color = "black", size = 12, face = "bold", family = "sans"),
        axis.title = element_text(size = 14, face = "bold", family = "sans"),
        panel.background = element_blank())

# 5. 保存结果
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}
output_file <- file.path(output_dir, "tsne_merged_result.tiff")
ggsave(output_file, p, width = 12, height = 10, units = "cm", dpi = 300)

cat(sprintf("\n🎉 R script executed successfully!\nSaved to: %s\nParameters used -> Perplexity: %s, Point Size: %s, Alpha: %s\n",
            output_file, perplexity_val, point_size, point_alpha))