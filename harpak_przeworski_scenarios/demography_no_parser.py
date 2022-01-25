import fwdpy11
import fwdpy11.discrete_demography
import tskit
import demes
import numpy as np
import json

POPT = 0.0
VS = 1.0
E_SD = 1.0
E_MEAN = 1.0
MU = 1e-3


demog = fwdpy11.discrete_demography.from_demes("pop_split.yaml", burnin=1)
pop = fwdpy11.DiploidPopulation(
    [v for _, v in demog.metadata["initial_sizes"].items()], 1.0
)
print(pop.N)
pdict = {
    "sregions": [fwdpy11.GaussianS(0, 1, 1, 0.25)],
    "nregions": [],
    "recregions": [fwdpy11.PoissonInterval(0, 1.00, 1e-3)],
    "rates": (0, MU, None),
    "gvalue": fwdpy11.Additive(
        2, fwdpy11.GSS(POPT, VS), fwdpy11.GaussianNoise(E_SD, E_MEAN)
    ),
    "prune_selected": False,
    "simlen": demog.metadata["total_simulation_length"],
    "demography": demog,
}
mparams = fwdpy11.ModelParams(**pdict)

seed = int(np.random.randint(0, 100000, 1)[0])
# randint returns numpt.int64, which json doesn't
# know how to handle.  So, we turn it
# to a regular Python int to circumvent this problem.
rng = fwdpy11.GSLrng(seed)

fwdpy11.evolvets(rng, pop, mparams, simplification_interval=100)
print(pop.N, pop.deme_sizes())

g = demes.load("pop_split.yaml")
ts = pop.dump_tables_to_tskit(demes_graph=g)
ts.dump("sim.trees")


tsl = tskit.load("sim.trees")
graph_dict = tsl.metadata["demes_graph"]
rebuilt_graph = demes.Graph.fromdict(graph_dict)
assert g == rebuilt_graph

print(rebuilt_graph)
print(graph_dict)
print()

# popt = tsl.model_params.gvalue.gvalue_to_fitness.optimum
# vs = tsl.model_params.gvalue.gvalue_to_fitness.VS

ind_md = fwdpy11.tskit_tools.decode_individual_metadata(tsl)
# print(ind_md)


provenance = json.loads(ts.provenance(0).record)
print(provenance)
fitness = np.array([md.w for md in ind_md])
print(fitness)

genetic_value = np.array([md.g for md in ind_md])
environmental_value = np.array([md.e for md in ind_md])
print(environmental_value)
phenotype = np.array([md.g + md.e for md in ind_md])
print(phenotype)
deme = np.array([md.deme for md in ind_md])
print(deme)


def fitness_phenotype_summary(ind_md):
    with open("sim.txt", "w") as output_file:
        output_file.write(
            f"{'individual'}\t{'deme'}\t{'Population_optimum'}\t{'strength_stabilizing_selection'}\t{'mean_E'}\t{'e_SD'}\t{'ind_fitness'}\t{'ind_genetic_value'}\t{'ind_environmental_value'}\t{'ind_phenotype'}"
        )
        for ind, listitem in enumerate(ind_md):
            output_file.write(
                f"\n{ind}\t{deme[ind]}\t{POPT}\t{VS}\t{MU}\t{E_SD}\t{fitness[ind]}\t{genetic_value[ind]}\t{environmental_value[ind]}\t{phenotype[ind]}"
            )


def main():
    fitness_phenotype_summary(ind_md=ind_md)


if __name__ == "__main__":
    main()
