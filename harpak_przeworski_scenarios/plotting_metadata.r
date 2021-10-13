# will need to check and change working
# directory if using the r console on vscode.

setwd("/home/paula/fwdpy11/polygenic_effect_sims/harpak_przeworski_scenarios")
metadada <- read.table("harpak_przeworski.txt", header = TRUE, sep = "")
# tried fill = TRUE here, and while it
# does remove the scan error, it doesn't change that
# metadata, when called, shows as 'not found'

metadata