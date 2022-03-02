import fwdpy11
import fwdpy11.discrete_demography
import sys
import tskit
import demes
import numpy as np
import json
import argparse


def make_parser() -> argparse.ArgumentParser:
    # make an argument parser that can output help.
    ADHF = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(__file__, formatter_class=ADHF)
    # __file__ specifies the full path to this file

    parser.add_argument(
        "--yaml",
        "-y",
        type=str,
        default=None,
        help="yaml file input name (demes specification)",
    )
    return parser


# will need to add the above back in when I split things up into functions


def validate_args(args: argparse.Namespace):
    if args.yaml is None:
        raise ValueError(f"must specify a demes yaml model for the simulation")


def run_sim(args: argparse.Namespace) -> fwdpy11.DiploidPopulation:
    input_file = args.yaml
    # MU = args.MU
    # POPT = args.POPT
    # VS = args.VS
    # E_SD = args.E_SD
    # E_MEAN = args.E_MEAN
    # POPT = 0.0  # maybe if you do this as a list, then you can specify different slices for fwdpy11.Additive Popt
    # and then iterate over the individuals when you write to the text file

    POPT = 1.0
    # optima_list = [0.0, 0.1, -0.1]
    optima_list = [0.1, -0.1]

    VS = 1.0

    E_SD = 1.0
    # env_sd_list = [1.0, 1.3, 0.7]
    env_sd_list = [1.3, 0.7]

    E_MEAN = 1.0
    # env_mean_list = [1.0, 1.1, 0.9]
    env_mean_list = [1.1, 0.9]

    MU = 1e-3

    burnin = 10

    additive_objects = []
    for i in optima_list:
        additive_objects.append(
            fwdpy11.Additive(
                2.0, fwdpy11.GSS(i, VS), fwdpy11.GaussianNoise(E_SD, E_MEAN)
            )
        )
    print(additive_objects)
    # make fwdpy11.Additive objects from the POPT specified using list comprehension

    env_sd_objects = []
    for i in env_sd_list:
        env_sd_objects.append(
            fwdpy11.Additive(
                2.0, fwdpy11.GSS(POPT, VS), fwdpy11.GaussianNoise(i, E_MEAN)
            )
        )

    env_mean_objects = []
    for i in env_mean_list:
        env_mean_objects.append(
            fwdpy11.Additive(2.0, fwdpy11.GSS(POPT, VS), fwdpy11.GaussianNoise(E_SD, i))
        )

    demog = fwdpy11.discrete_demography.from_demes(input_file, burnin)
    # demog = fwdpy11.discrete_demography.from_demes("no_ancestry.yaml", burnin)
    pop = fwdpy11.DiploidPopulation(
        [v for _, v in demog.metadata["initial_sizes"].items()], 1.0
    )
    # print(pop.N)
    print()
    print()

    initial_sizes = [
        demog.metadata["initial_sizes"][i]
        for i in sorted(demog.metadata["initial_sizes"].keys())
    ]

    dbg = fwdpy11.DemographyDebugger(initial_sizes, demog.model)
    print(dbg.report)

    pdict = {
        "sregions": [fwdpy11.GaussianS(0, 1, 1, 0.25)],
        "nregions": [],
        "recregions": [fwdpy11.PoissonInterval(0, 1.00, 1e-3)],
        "rates": (0, MU, None),
        "gvalue": env_sd_objects,
        # fwdpy11.Additive(gvalue_to_fitness= fwdpy11.GSS(optimum = moving_optimum_deme_2, VS), fwdpy11.GaussianNoise(E_SD, E_MEAN), ndemes = 3, scaling= 2)],
        "prune_selected": False,
        "simlen": demog.metadata["total_simulation_length"],
        "demography": demog,
    }

    mparams = fwdpy11.ModelParams(**pdict)

    # print(mparams.asblack)
    # print(mparams)
    print(mparams.gvalue)
    # for fwdpy11.Additive in mparams.gvalue:
    get_gvalue_to_fitness = np.array([md.gvalue_to_fitness for md in mparams.gvalue])
    print(get_gvalue_to_fitness)

    popt = np.array([md.optimum for md in get_gvalue_to_fitness])
    print(popt)

    # popt2 = np.array([md.optimum for md in mparams.gvalue.gvalue_to_fitness])
    # print(popt2)
    # Not sure why the above doesn't work!

    get_gaussian_noise = np.array([md.noise for md in mparams.gvalue])
    print(get_gaussian_noise)

    e_sd = np.array([md.sd for md in get_gaussian_noise])
    print(e_sd)

    e_mean = np.array([md.mean for md in get_gaussian_noise])

    print()
    print()

    seed = int(np.random.randint(0, 100000, 1)[0])
    # randint returns numpt.int64, which json doesn't
    # know how to handle.  So, we turn it
    # to a regular Python int to circumvent this problem.
    rng = fwdpy11.GSLrng(seed)

    fwdpy11.evolvets(rng, pop, mparams, simplification_interval=100)
    # print(pop.N, pop.deme_sizes())  # it's at this point that pop actually has the multiple demes.
    return (pop, input_file)


# print(fwdpy11.ModelParams)


def fitness_phenotype_summary(pop, input_file):
    g = demes.load(input_file)
    # g = demes.load("no_ancestry.yaml")
    ts = pop.dump_tables_to_tskit(demes_graph=g)
    ts.dump("sim.trees")

    tsl = tskit.load("sim.trees")
    graph_dict = tsl.metadata["demes_graph"]
    rebuilt_graph = demes.Graph.fromdict(graph_dict)
    assert g == rebuilt_graph
    print()
    # print(tsl.metadata)
    print()
    # popt = tsl.model_params.gvalue.gvalue_to_fitness.optimum
    #
    # print(popt)
    # popt = tsl.metadata["model_params"]  # trying to be able to write popt to txt file
    # print(popt)

    # vs = tsl.model_params.gvalue.gvalue_to_fitness.VS
    ind_md = fwdpy11.tskit_tools.decode_individual_metadata(tsl)
    print(ind_md)

    print()

    provenance = json.loads(ts.provenance(0).record)
    fitness = np.array([md.w for md in ind_md])

    genetic_value = np.array([md.g for md in ind_md])
    environmental_value = np.array([md.e for md in ind_md])
    phenotype = np.array([md.g + md.e for md in ind_md])
    deme = np.array([md.deme for md in ind_md])
    """return (
        # pop,
        # params,
        # seed,
        # E_MEAN,
        # E_SD,
        yaml
    )"""  # this is from my other script, where I'd pass these to write_treefile
    H2_list = []
    for i in env_sd_list:
        H2_list.append(
            4 * MU * VS / ((4 * MU * VS) + (i ** 2))
        )  # square env_sd bc sd is sqrt(variance)

    print(H2_list)

    with open("sim.txt", "w") as output_file:
        output_file.write(
            f"{'individual'}\t{'deme'}\t{'Population_optimum'}\t{'strength_stabilizing_selection'}\t{'mutation_rate'}\t{'e_mean'}\t{'e_SD'}\t{'ind_fitness'}\t{'ind_genetic_value'}\t{'ind_environmental_value'}\t{'ind_phenotype'}"
        )
        for ind, ind_md in enumerate(ind_md):
            output_file.write(
                f"\n{ind}\t{deme[ind]}\t{popt}\t{VS}\t{MU}\t{e_mean}\t{e_sd}\t{fitness[ind]}\t{genetic_value[ind]}\t{environmental_value[ind]}\t{phenotype[ind]}"
            )  # I want to have POPT and others be writing the results of the list, but not sure how to do that without the ts_metadata function


def main():
    # build our parser
    parser = make_parser()

    # process sys.argv
    args = parser.parse_args(sys.argv[1:])

    # check input
    validate_args(args)

    (pop, input_file) = run_sim(args)

    print("did it work???????????")
    print(pop)
    print(input_file)

    fitness_phenotype_summary(
        pop=pop, input_file=input_file
    )  # make sure these and the above are in the same order


if __name__ == "__main__":
    main()
