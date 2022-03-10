# will need to check and change working
# directory if using the r console on vscode.


# setwd("/home/paula/fwdpy11/polygenic_effect_sims/harpak_przeworski_scenarios")#of course, you don't want to have to change this for every machine you're running on.

# setting working directory to be whatever directory the file is in:
setwd(dirname(rstudioapi::getActiveDocumentContext()$path)) # https://stackoverflow.com/questions/3452086/getting-path-of-an-r-script/35842176#35842176


files <- list.files(pattern = "case_4_no_ancestry")
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

pdf("case_4_no_ancestry.pdf")
grid.arrange(phenotype_plot, environmental_value_plot, genetic_value_plot, nrow = 3)
dev.off()





####### SUMMARY STATISTICS


library(dplyr)
group <- group_by(metadata, deme)
summary(group)

# print(group)
summarise(group, mean = mean(ind_phenotype), sd = sd(ind_phenotype))

summarise(group, mean = mean(ind_genetic_value), sd = sd(ind_genetic_value))

summarise(group, mean = mean(ind_environmental_value), sd = sd(ind_environmental_value))
###### OVERALL
print(sd(metadata$phenotype))
print(sd(metadata$ind_genetic_value))
print(sd(metadata$ind_environmental_value))
