library(ggplot2)
library(ggpubr)
library(ggthemes)
library(Rtsne)
library(pals)
library(factoextra)
library(ggforce)

setwd("G:/英文文章数据/230405_3D/CELLPROFILER/处理数据/单个互作参数")
getwd()



A <- read.csv("mi-er.csv",header = T,row.names = 1)
Atsne <- Rtsne(t(A), perplexity =5)

colnames(Atsne$Y) <- c("TSNE1","TSNE2")
Atsne_data <- data.frame(sample=colnames(A),
                         Type=c(rep("M0",6),rep("M1",6),rep("M2",6)
                         ),Atsne$Y)



define_color <- c ("#0000FF","#FF0000","#999999")

p <- ggplot(Atsne_data, aes(x=TSNE1, y=TSNE2)) + 
  geom_point(aes(color=Type),size=3) +
  xlab("TSNE1")+ ylab("TSNE2") +
  scale_color_manual(values=define_color )+
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        #legend.position = "none",
        legend.title=element_blank(), 
        panel.border = element_blank(),
        legend.text=element_text(face="bold", color="black",size=10),
        axis.line = element_line(color="black", size = 1),
        axis.ticks = element_line(size = 1),
        axis.text = element_text(color="black",size = 12, face="bold", family = "sans"),
        axis.title = element_text(size = 14, face="bold", family = "sans"),
        panel.background = element_blank())


p
ggsave("mi-er.tiff", p, width = 8, height = 8, units = "cm", dpi = 300)


p1 <- ggplot(Atsne_data, aes(x=TSNE1, y=TSNE2)) + 
  geom_point(aes(color=Type),size=3) +
  xlab("TSNE1")+ ylab("TSNE2") +
  scale_color_manual(values=define_color )+
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        legend.position = "none",
        legend.title=element_blank(), 
        panel.border = element_blank(),
        legend.text=element_text(face="bold", color="black",size=10),
        axis.line = element_line(color="black", size = 1),
        axis.ticks = element_line(size = 1),
        axis.text = element_text(color="black",size = 12, face="bold", family = "sans"),
        axis.title = element_text(size = 14, face="bold", family = "sans"),
        panel.background = element_blank())


p1
ggsave("mi-er1.tiff", p1, width = 8, height = 8, units = "cm", dpi = 300)