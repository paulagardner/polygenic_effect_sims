# will need to check and change working
# directory if using the r console on vscode.


# setwd("/home/paula/fwdpy11/polygenic_effect_sims/harpak_przeworski_scenarios")#of course, you don't want to have to change this for every machine you're running on.

# setting working directory to be whatever directory the file is in:
setwd(dirname(rstudioapi::getActiveDocumentContext()$path)) # https://stackoverflow.com/questions/3452086/getting-path-of-an-r-script/35842176#35842176


metadata <- read.table("harpak_przeworski_case3.txt", header = TRUE, sep = "\t")


library("ggplot2")
ggplot(metadata, aes(x = (ind_genetic_value + ind_environmental_value), y = ind_fitness, color = count)) +
    geom_point()

ggplot(metadata, aes(x = ind_environmental_value, group = count, color = count)) +
    geom_density()

ggplot(metadata, aes(x = ind_genetic_value, group = count, color = count)) +
    geom_density()