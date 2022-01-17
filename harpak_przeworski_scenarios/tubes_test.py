import argparse
import fwdpy11
import sys
import numpy as np
import json
import tskit

import demes
import demesdraw


model = fwdpy11.discrete_demography.from_demes(
    "pop_split.yml",
    burnin=10,
)
print(model)


# print(model.metadata["initial_sizes"])
# print(
#    model.metadata["total_simulation_length"]
# )  # if burnin is 10, I believe it's specifying 10N- 10*100 = 1000, plus
# my .yml file specifying ancestral population was 1000 generations ago = 2000 gen total length?


initial_sizes = [
    model.metadata["initial_sizes"][i]
    for i in sorted(model.metadata["initial_sizes"].keys())
]
# print(initial_sizes)


pop = fwdpy11.DiploidPopulation(
    initial_sizes, 1000.0
)  # second part is specifying genome size, if I read the documentation correctly
# print(pop.deme_sizes())
print()
print(pop.deme_sizes(as_dict=True))

graph = demes.load("pop_split.yml")
ax = demesdraw.tubes(graph)
ax.figure.savefig("tubes_test.svg")

demog = fwdpy11.discrete_demography.from_demes(
    graph
)  # so this contains all the information it seems your sim would need to run
# print(demog) #same as print(model)
# print(model.metadata)
