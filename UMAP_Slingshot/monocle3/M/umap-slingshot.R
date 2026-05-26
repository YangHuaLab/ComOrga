library(uwot) # Used for UMAP instead of Rtsne
library(slingshot)
library(ggplot2)
library(viridis)

# Load data
setwd("F:/英文文章数据/time")
data <- read.csv("M1_CELL.csv", row.names = 1)

# Define groups
groups <- factor(c(rep("M0_00H", 1000), 
                   rep("M1_01H", 1000),
                   rep("M1_06H", 1000),
                   rep("M1_16H", 1000),
                   rep("M1_24H", 1000)))

# ---------------------------------------------------------
# 1. Run UMAP instead of t-SNE (Better for Trajectories)
# ---------------------------------------------------------
set.seed(123)
umap_result <- umap(data, n_neighbors = 30, min_dist = 0.3)
umap_coords <- umap_result
rownames(umap_coords) <- rownames(data)
colnames(umap_coords) <- c("UMAP1", "UMAP2")

# ---------------------------------------------------------
# 2. Slingshot trajectory inference (Crucial Fixes Applied)
# ---------------------------------------------------------
# We provide the cluster labels and explicitly state the starting cluster.
sds <- slingshot(umap_coords, clusterLabels = groups, start.clus = "M0_00H")
curves <- slingCurves(sds)
pseudotime <- slingPseudotime(sds)

# Create data frame for ggplot
df <- data.frame(
  UMAP1 = umap_coords[, 1],
  UMAP2 = umap_coords[, 2],
  Pseudotime = pseudotime[, 1],  
  Group = groups
)

# Define group colors
group_colors <- c("M0_00H" = "#1F77B4", "M1_01H" = "#FF7F0E", "M1_06H" = "#2CA02C", 
                  "M1_16H" = "#D62728", "M1_24H" = "#9467BD")

# ---------------------------------------------------------
# Plot 1: UMAP Points colored by Pseudotime with Trajectory
# ---------------------------------------------------------
p1 <- ggplot(df, aes(x = UMAP1, y = UMAP2)) +
  geom_point(aes(color = Pseudotime), size = 0.5, alpha = 1) +
  scale_color_viridis_c() +
  theme_minimal() +
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        legend.title=element_blank(), 
        panel.border = element_blank(),
        legend.text=element_text(face="bold", color="black",size=10),
        axis.line = element_line(color="black", linewidth = 1),
        axis.ticks = element_line(linewidth = 1),
        axis.text = element_text(color="black",size = 12, face="bold", family = "sans"),
        axis.title = element_text(size = 14, face="bold", family = "sans"),
        panel.background = element_blank())+
  labs(title = "UMAP with Slingshot Trajectory",
       x = "UMAP 1",
       y = "UMAP 2",
       color = "Pseudotime")

# Add Slingshot trajectories
for (curve in curves) {
  curve_data <- curve$s[curve$ord, ]
  colnames(curve_data) <- c("UMAP1", "UMAP2")
  p1 <- p1 + geom_path(data = curve_data, aes(x = UMAP1, y = UMAP2), color = "black", linewidth = 1)
}

print(p1)
ggsave("M1-Pseudotime_UMAP.tiff", plot = p1, width = 12, height = 10, units = "cm", dpi = 600, compression = "lzw")

# ---------------------------------------------------------
# Plot 2: UMAP Points colored by Group with Trajectory
# ---------------------------------------------------------
p2 <- ggplot(df, aes(x = UMAP1, y = UMAP2)) +
  geom_point(aes(color = Group), size = 0.5, alpha = 1) +
  scale_color_manual(values = group_colors) +
  theme_minimal() +
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        legend.title=element_blank(), 
        panel.border = element_blank(),
        legend.text=element_text(face="bold", color="black",size=10),
        axis.line = element_line(color="black", linewidth = 1),
        axis.ticks = element_line(linewidth = 1),
        axis.text = element_text(color="black",size = 12, face="bold", family = "sans"),
        axis.title = element_text(size = 14, face="bold", family = "sans"),
        panel.background = element_blank())+
  labs(title = "UMAP Groupings with Trajectory",
       x = "UMAP 1",
       y = "UMAP 2",
       color = "Group")

print(p2)
ggsave("M1-Group_UMAP.tiff", plot = p2, width = 12, height = 10, units = "cm", dpi = 600, compression = "lzw")
