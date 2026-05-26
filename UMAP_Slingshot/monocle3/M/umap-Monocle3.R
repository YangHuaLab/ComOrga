# 启动干净的 R 环境
library(monocle3)
library(ggplot2)
library(viridis)
library(dplyr)
library(igraph)

# ====================================================================
# 1. 加载数据与构建对象 (M2 数据集)
# ====================================================================
setwd("F:/英文文章数据/time")
data <- read.csv("M2_CELL.csv", row.names = 1)

# 定义分组
groups <- factor(c(rep("M0_00H", 1000), 
                   rep("M2_01H", 1000),
                   rep("M2_06H", 1000),
                   rep("M2_16H", 1000),
                   rep("M2_24H", 1000)))

# 转换数据格式 (Monocle3 要求: 行=基因, 列=细胞)
expr_matrix <- t(data)

# 准备 metadata
cell_metadata <- data.frame(Group = groups, row.names = rownames(data))
gene_metadata <- data.frame(gene_short_name = rownames(expr_matrix), 
                            row.names = rownames(expr_matrix))

# 构建 cds 对象
cds <- new_cell_data_set(as.matrix(expr_matrix), 
                         cell_metadata = cell_metadata, 
                         gene_metadata = gene_metadata)

# ====================================================================
# 2. 预处理、降维、聚类与学习轨迹图 (🌟 完全使用标准默认参数)
# ====================================================================
# 预处理 (仅保留 norm_method="none" 以防对你已处理的数据重复标准化)
cds <- preprocess_cds(cds, num_dim = 50, norm_method = "none")

#  UMAP 降维 (移除了强制的 neighbors 和 dist 参数)
cds <- reduce_dimension(cds, reduction_method = "UMAP")

# 聚类 (移除了强制的极低 resolution)
cds <- cluster_cells(cds)

#认学习轨迹
cds <- learn_graph(cds)

# ====================================================================
# 3. 融入官方黑科技：编程方式指定根节点
# ====================================================================
get_earliest_principal_node <- function(cds, time_bin="M0_00H"){
  # 找到属于 M0_00H 的所有细胞的 ID
  cell_ids <- which(colData(cds)[, "Group"] == time_bin)
  
  # 找到这些细胞在 UMAP 图上最近的轨迹主节点
  closest_vertex <- cds@principal_graph_aux[["UMAP"]]$pr_graph_cell_proj_closest_vertex
  closest_vertex <- as.matrix(closest_vertex[colnames(cds), ])
  
  # 统计包含 M0_00H 细胞最多的那个节点，将其作为起点
  root_pr_nodes <- igraph::V(principal_graph(cds)[["UMAP"]])$name[as.numeric(names(which.max(table(closest_vertex[cell_ids,]))))]
  
  return(root_pr_nodes)
}

# 自动计算起点节点，并进行排序计算拟时间
root_node <- get_earliest_principal_node(cds, time_bin = "M0_00H")
cds <- order_cells(cds, root_pr_nodes = root_node)

# ====================================================================
# 4. 可视化基础设置 (提取你的专属完美主题)
# ====================================================================
group_colors <- c("M0_00H" = "#1F77B4", "M2_01H" = "#FF7F0E", "M2_06H" = "#2CA02C", 
                  "M2_16H" = "#D62728", "M2_24H" = "#9467BD")

my_theme <- theme_minimal() +
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        legend.title=element_blank(), 
        panel.border = element_blank(),
        legend.text=element_text(face="bold", color="black",size=10),
        axis.line = element_line(color="black", linewidth = 1),
        axis.ticks = element_line(linewidth = 1),
        axis.text = element_text(color="black",size = 12, face="bold", family = "sans"),
        axis.title = element_text(size = 14, face="bold", family = "sans"),
        panel.background = element_blank())

# ====================================================================
# 5. 画图 1：按 Group 分组填色的轨迹图
# ====================================================================
p_group <- plot_cells(cds, 
                      color_cells_by = "Group",
                      label_groups_by_cluster = FALSE,
                      label_leaves = FALSE,
                      label_branch_points = FALSE, # 如果你想显示算法找到的分支点黑圈，可以把这个改为 TRUE
                      label_roots = FALSE, 
                      cell_size = 0.5,
                      trajectory_graph_color = "black",
                      trajectory_graph_segment_size = 1.2) +
  scale_color_manual(values = group_colors) +
  my_theme +
  labs(title = "UMAP Groupings with Trajectory (M2)", x = "UMAP 1", y = "UMAP 2")

print(p_group)
ggsave("M2-Group_Monocle3_Default.tiff", plot = p_group, width = 12, height = 10, units = "cm", dpi = 600, compression = "lzw")

# ====================================================================
# 6. 画图 2：按 Pseudotime 渐变填色的轨迹图
# ====================================================================
p_time <- plot_cells(cds, 
                     color_cells_by = "pseudotime", 
                     label_groups_by_cluster = FALSE,
                     label_leaves = FALSE,
                     label_branch_points = FALSE,
                     label_roots = FALSE,
                     cell_size = 0.5,
                     show_trajectory_graph = FALSE) +
  scale_color_viridis_c() + 
  my_theme +
  labs(title = "UMAP Pseudotime with Trajectory (M2)", x = "UMAP 1", y = "UMAP 2")

print(p_time)
ggsave("M2-Pseudotime_Monocle3_Default.tiff", plot = p_time, width = 12, height = 10, units = "cm", dpi = 600, compression = "lzw")
