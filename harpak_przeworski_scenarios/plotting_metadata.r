# will need to check and change working
# directory if using the r console on vscode.


# setwd("/home/paula/fwdpy11/polygenic_effect_sims/harpak_przeworski_scenarios")#of course, you don't want to have to change this for every machine you're running on.

# setting working directory to be whatever directory the file is in:
setwd(dirname(rstudioapi::getActiveDocumentContext()$path)) # https://stackoverflow.com/questions/3452086/getting-path-of-an-r-script/35842176#35842176


metadata <- read.table("sim.txt", header = TRUE, sep = "\t")

metadata

library("ggplot2")
ggplot(metadata, aes(x = (ind_phenotype), y = ind_fitness, color = deme)) +
    geom_point()

ggplot(metadata, aes(x = ind_environmental_value, group = deme, color = deme)) +
    geom_density()

ggplot(metadata, aes(x = ind_genetic_value, group = deme, color = deme)) +
    geom_density()


####### SUMMARY STATISTICS
# for (i in unique(metadata$deme)) {
#    print(sd(metadata$ind_genetic_value + metadata$ind_environmental_value))
# }


library(dplyr)
group <- group_by(metadata, deme)
# print(group)
summarise(group, mean = mean(ind_phenotype), sd = sd(ind_phenotype))

summarise(group, mean = mean(ind_genetic_value), sd = sd(ind_genetic_value))

summarise(group, mean = mean(ind_environmental_value), sd = sd(ind_environmental_value))
###### OVERALL
phenotype <- metadata$ind_genetic_value + metadata$ind_environmental_value
print(sd(phenotype))
print(sd(metadata$ind_genetic_value))
print(sd(metadata$ind_environmental_value))