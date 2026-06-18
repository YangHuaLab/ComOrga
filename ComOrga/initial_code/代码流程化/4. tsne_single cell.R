library(ggplot2)
library(ggpubr)
library(ggthemes)
library(Rtsne)

setwd("G:/英文文章数据/230405_3D/cell_cellprofiler/处理数据/包含互作参数")
A <- read.csv("nuc+mi+er.csv",header = T,row.names = 1)

Atsne <- Rtsne(t(A), perplexity = 30,check_duplicates = FALSE)
Atsne$Y
colnames(Atsne$Y) <- c("TSNE1","TSNE2")
Atsne_data <- data.frame(sample=colnames(A),
                         Type=c(rep("Control",2000), rep("M1",2000),rep("M2",2000)),Atsne$Y)

define_color <- c ("#0000FF","#FF0000","#999999")
p <-ggplot(Atsne_data, aes(x=TSNE1, y=TSNE2, colour=Type)) + 
  geom_point(size=1.2,alpha = 0.6,stroke = 0)+ xlab("TSNE1")+ ylab("TSNE2")+
  scale_color_manual(values=define_color )+
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        legend.position = "none",
        #legend.title=element_blank(), 
        panel.border = element_blank(),
        axis.line = element_line(color="black", size = 1),
        axis.ticks = element_line(size = 1),
        axis.text = element_text(color="black",size = 12,face="bold", family = "sans"),
        axis.title = element_text(size = 14,face="bold", family = "sans"),
        panel.background = element_blank())




p
ggsave("nuc+mi+er.tiff", p, width = 8, height = 8, units = "cm", dpi = 300)