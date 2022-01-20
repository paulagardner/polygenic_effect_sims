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
print(
    pop.deme_sizes(as_dict=True)
)  # just checking- this is from fwdpy11.DiploidPopulation.deme_sizes

graph = demes.load("pop_split.yml")
ax = demesdraw.tubes(graph)
ax.figure.savefig("tubes_test.svg")

# demog = fwdpy11.discrete_demography.from_demes(
#    graph
# )
# print(demog) #same as print(model)
# print(model.metadata)


def yaml_function():
    import demes

    yaml = """
description:
  simple model w/o migration of a population split with even pop sizes
time_units: generations
defaults:
  epoch:
    start_size: 100
demes:
  - name: ancestral
    epochs:
      - end_time: 1000
  - name: A
    ancestors: [ancestral]
  - name: B
    ancestors: [ancestral]
"""
    return demes.loads(yaml)


graph = yaml_function()

demography = fwdpy11.discrete_demography.from_demes(
    graph,
    burnin=10,
)

initial_sizes = [
    demography.metadata["initial_sizes"][i]
    for i in sorted(demography.metadata["initial_sizes"].keys())
]

total_length = demography.metadata["total_simulation_length"]

pop = fwdpy11.DiploidPopulation(initial_sizes, 1000.0)

print()

print(demography)
print()
print(pop)
print()
print(graph.demes)
