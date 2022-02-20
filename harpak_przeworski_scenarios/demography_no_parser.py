import fwdpy11
import fwdpy11.discrete_demography
import tskit
import demes
import numpy as np
import json

POPT = 0.0  # maybe if you do this as a list, then you can specify different slices for fwdpy11.Additive Popt
# and then iterate over the individuals when you write to the text file
VS = 1.0
E_SD = 1.0
E_MEAN = 1.0
MU = 1e-3

burnin = 10

a = [fwdpy11.Additive(2.0, fwdpy11.GSS(fwdpy11.Optimum(i, VS=1.0))) for i in POPT]
print(a)
# make fwdpy11.Additive objects from the POPT specified using list comprehension

demog = fwdpy11.discrete_demography.from_demes("pop_split.yaml", burnin)
pop = fwdpy11.DiploidPopulation(
    [v for _, v in demog.metadata["initial_sizes"].items()], 1.0
)
print(pop.N)
print()
print()

initial_sizes = [
    demog.metadata["initial_sizes"][i]
    for i in sorted(demog.metadata["initial_sizes"].keys())
]


dbg = fwdpy11.DemographyDebugger(initial_sizes, demog.model)
print(dbg.report)


moving_optimum_deme_1 = fwdpy11.GSSmo(
    [
        fwdpy11.Optimum(when=0, optimum=0.0, VS=VS),
        fwdpy11.Optimum(when=burnin * pop.N, optimum=1.0, VS=VS),
    ]
)

moving_optimum_deme_2 = fwdpy11.GSSmo(
    [
        fwdpy11.Optimum(when=0, optimum=0.0, VS=VS),
        fwdpy11.Optimum(when=burnin * pop.N, optimum=-1.0, VS=VS),
    ]
)

""" optimum_deme_1 = fwdpy11.GSS(
    fwdpy11.Optimum(optimum=1.0, VS=VS, when=None),
)
optimum_deme_2 = fwdpy11.GSS(
    fwdpy11.Optimum(optimum=-1.0, VS=VS, when=None),
) """

""" optimum_deme_1 = fwdpy11.Optimum(optimum=1.0, VS=VS, when=None)

optimum_deme_2 = fwdpy11.Optimum(optimum=-1.0, VS=VS, when=None) """


pdict = {
    "sregions": [fwdpy11.GaussianS(0, 1, 1, 0.25)],
    "nregions": [],
    "recregions": [fwdpy11.PoissonInterval(0, 1.00, 1e-3)],
    "rates": (0, MU, None),
    "gvalue": [
        fwdpy11.Additive(2, fwdpy11.GSS(POPT, VS), fwdpy11.GaussianNoise(E_SD, E_MEAN)),
        fwdpy11.Additive(
            2,
            gvalue_to_fitness=moving_optimum_deme_1,
            # fwdpy11.GSS(moving_optimum_deme_1, VS), of course you don't want this- I had the right idea, but VS is already accounted for in the variable!
            noise=fwdpy11.GaussianNoise(E_SD, E_MEAN),
        ),
        fwdpy11.Additive(
            2,
            gvalue_to_fitness=moving_optimum_deme_2,
            noise=fwdpy11.GaussianNoise(E_SD, E_MEAN),
        ),
    ],
    # fwdpy11.Additive(gvalue_to_fitness= fwdpy11.GSS(optimum = moving_optimum_deme_2, VS), fwdpy11.GaussianNoise(E_SD, E_MEAN), ndemes = 3, scaling= 2)],
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
print(
    pop.N, pop.deme_sizes()
)  # it's at this point that pop actually has the two demes.

g = demes.load("pop_split.yaml")
ts = pop.dump_tables_to_tskit(demes_graph=g)
ts.dump("sim.trees")


tsl = tskit.load("sim.trees")
graph_dict = tsl.metadata["demes_graph"]
rebuilt_graph = demes.Graph.fromdict(graph_dict)
assert g == rebuilt_graph

# print(rebuilt_graph)
# print(graph_dict)

# popt = tsl.model_params.gvalue.gvalue_to_fitness.optimum
# vs = tsl.model_params.gvalue.gvalue_to_fitness.VS

ind_md = fwdpy11.tskit_tools.decode_individual_metadata(tsl)
# print(ind_md)


provenance = json.loads(ts.provenance(0).record)
fitness = np.array([md.w for md in ind_md])

genetic_value = np.array([md.g for md in ind_md])
environmental_value = np.array([md.e for md in ind_md])
phenotype = np.array([md.g + md.e for md in ind_md])
deme = np.array([md.deme for md in ind_md])


def fitness_phenotype_summary(ind_md):
    with open("sim.txt", "w") as output_file:
        output_file.write(
            f"{'individual'}\t{'deme'}\t{'Population_optimum'}\t{'strength_stabilizing_selection'}\t{'mean_E'}\t{'e_SD'}\t{'ind_fitness'}\t{'ind_genetic_value'}\t{'ind_environmental_value'}\t{'ind_phenotype'}"
        )
        for ind, ind_md in enumerate(ind_md):
            output_file.write(
                f"\n{ind}\t{deme[ind]}\t{POPT}\t{VS}\t{MU}\t{E_SD}\t{fitness[ind]}\t{genetic_value[ind]}\t{environmental_value[ind]}\t{phenotype[ind]}"
            )


def main():
    fitness_phenotype_summary(ind_md=ind_md)


if __name__ == "__main__":
    main()
