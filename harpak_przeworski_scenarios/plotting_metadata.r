# will need to check and change working
# directory if using the r console on vscode.


# setwd("/home/paula/fwdpy11/polygenic_effect_sims/harpak_przeworski_scenarios")#of course, you don't want to have to change this for every machine you're running on.

# setting working directory to be whatever directory the file is in:
setwd(dirname(rstudioapi::getActiveDocumentContext()$path)) # https://stackoverflow.com/questions/3452086/getting-path-of-an-r-script/35842176#35842176


files <- list.files(pattern = "case_4_shared_ancestry")
print(files)

metadata <- NULL
for (f in files)
{
    dat <- read.table(f, header = T, sep = "\t")
    metadata <- rbind(metadata, dat)
}

# metadata # https://stackoverflow.com/questions/2104483/how-to-read-table-multiple-files-into-a-single-table-in-r
summary(metadata)

# metadata <- read.table("sim.txt", header = TRUE, sep = "\t")


library("ggplot2")
phenotype_plot <-
    ggplot(metadata, aes(x = (ind_phenotype), y = ind_fitness, color = deme)) +
    geom_point()

environmental_value_plot <-
    ggplot(metadata, aes(x = ind_environmental_value, group = deme, color = deme)) +
    geom_density()

genetic_value_plot <-
    ggplot(metadata, aes(x = ind_genetic_value, group = deme, color = deme)) +
    geom_density()



library("gridExtra")
library(cowplot)

pdf("case_4_shared_ancestry.pdf")
grid.arrange(phenotype_plot, environmental_value_plot, genetic_value_plot, nrow = 3)
dev.off()





####### SUMMARY STATISTICS
# for (i in unique(metadata$deme)) {
#    print(sd(metadata$ind_genetic_value + metadata$ind_environmental_value))
# }


library(dplyr)
group <- group_by(metadata, deme)
summary(group)

# print(group)
summarise(group, mean = mean(ind_phenotype), sd = sd(ind_phenotype))

summarise(group, mean = mean(ind_genetic_value), sd = sd(ind_genetic_value))

summarise(group, mean = mean(ind_environmental_value), sd = sd(ind_environmental_value))
###### OVERALL
phenotype <- metadata$ind_genetic_value + metadata$ind_environmental_value
print(sd(phenotype))
print(sd(metadata$ind_genetic_value))
print(sd(metadata$ind_environmental_value))


# packageurl <- "https://cran.r-project.org/src/contrib/Archive/nloptr/nloptr_1.2.1.tar.gz"
# install.packages(packageurl, repos = NULL, type = "source") # https://stackoverflow.com/questions/64939242/unable-to-install-ggpubr for ggpubr not working
# install.packages("ggpubr")
# library("ggpubr")


# ggarange(phenotype_plot, environmental_value_plot, genetic_value_plot, ncol = 3, nrow = 1)



# plots <- list(phenotype_plot, environmental_value_plot, genetic_value_plot)
# pdf("summary_plots.pdf")
# for (i in plots) {
#     var1 <- rnorm(i)
#     var2 <- rnorm(i)
#     plot(i)
# }
# dev.off()




# jpeg("summary_plots.jpg")
#     par(mfrow = c(1,3))
#     plot(phenotype_plot)
#     plot(environmental_value_plot)
#     plot(genetic_value_plot)
# dev.off()


# plot(phenotype_plot)

# jpeg("summary_plots.jpg")
#     par(mfrow = c(1,3))
#     phenotype_plot
#     environmental_value_plot
#     genetic_value_plot
# dev.off()


# png("summary_plots.png")
# print(plots)
# dev.off()

# library(ggplot2)
# library("ggpubr")
# figure <- ggarange