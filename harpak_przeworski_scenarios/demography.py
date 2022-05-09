import fwdpy11
import fwdpy11.discrete_demography
import sys
import tskit
import demes
import numpy as np
import json
import argparse


# example to run: python demography.py -y no_ancestry.yaml --MU 1e-3 --VS 1.0 --POPT 0.0 0.0 --E_SD 1.5 0.5 --E_MEAN 0.0 0.0 --output_file test_case.txt
# #must specify 2, 3, etc sets of values for however many demes I'm doing, due to collapsing everything down to one iterator


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

    parser.add_argument("--MU", "--u", type=float, help="Mutation rate")

    parser.add_argument(
        "--VS",
        type=float,
        help="Variance of S, the inverse strength of stabilizing selection",
    )

    parser.add_argument(
        "--sim_type",
        type=str,
        help="which case from harpak+przeworski is being modeled",
    )

    parser.add_argument(
        "--POPT",
        type=float,
        nargs="+",
        default=0,
        help="Population optimum trait value",
    )

    parser.add_argument(
        "--E_SD",
        type=float,
        nargs="+",
        help="Environmental effect distribution's standard deviation",
    )

    parser.add_argument(
        "--E_MEAN",
        type=float,
        nargs="+",
        help="Environmental effect distribution's mean",
    )

    parser.add_argument(
        "--output_file", "-o", type=str, default=None, help="Output file name"
    )

    return parser


def validate_args(args: argparse.Namespace):
    if args.yaml is None:
        raise ValueError(f"must specify a demes yaml model for the simulation")

    if args.MU is None:
        raise ValueError(f"Mutation rate cannot be None")

    if args.VS is None:
        raise ValueError(
            f"In this simulation using stabilizing selection, VS cannot be None"
        )

    # see if you can write a statement for incorrect number of inputs for POPT, E_SD, VS here


def run_sim(args: argparse.Namespace) -> fwdpy11.DiploidPopulation:
    input_file = args.yaml

    VS = args.VS
    MU = args.MU

    burnin = 10

    POPT = args.POPT
    E_SD = args.E_SD
    E_MEAN = args.E_MEAN

    print(POPT)
    #print(env_sd_list)
    # optima_list = args.POPT
    # additive_objects = []
    # for i in optima_list:
    #     additive_objects.append(
    #         fwdpy11.Additive(
    #             2.0,
    #             fwdpy11.GSS(i, VS),
    #             fwdpy11.GaussianNoise(1, 0),
    #         )
    #     )
    # print(additive_objects)

    gvalue_objects = []
    for i, y, z in zip(
        range(len(E_SD)), range(len(POPT)), range(len(E_MEAN))
    ):  # length of a variable gives you an integer of how long that list is, and range gives you a list of values from 0 to length (so if len = 2, range = [0, 1,])
        gvalue_objects.append(
            fwdpy11.Additive(
                2.0,
                fwdpy11.GSS(
                    POPT[y], VS
                ),  # so why range(len(POPT)) here? It's letting us have an index,
                # so i'm saying when iterating over the y values of POPT, give me the yth value of that list, which depends on the loop
                fwdpy11.GaussianNoise(
                    E_SD[i], E_MEAN[z]
                ),  # if using this syntax, you'll have to specify all values for the three variables in the command line,
                # because zip will just not fully iterate over the longest list if lists are of unequal size
            )
        )

    demog = fwdpy11.discrete_demography.from_demes(input_file, burnin)
    # demog = fwdpy11.discrete_demography.from_demes("no_ancestry.yaml", burnin)
    pop = fwdpy11.DiploidPopulation(
        [v for _, v in demog.metadata["initial_sizes"].items()], 1.0
    )  # second parameter is genome length! Skylar was running into issues w/ sregions and recregions where she was specifying regions longer than her genome length

    # print(pop.N)

    fwdpy11.ConstantS
    initial_sizes = [
        demog.metadata["initial_sizes"][i]
        for i in sorted(demog.metadata["initial_sizes"].keys())
    ]

    dbg = fwdpy11.DemographyDebugger(initial_sizes, demog.model)
    print(dbg.report)

    pdict = {
        "sregions": [
            fwdpy11.GaussianS(0, 1, 1, 0.25)
        ],  # you can run into trouble w/ using ints vs. floats here
        "nregions": [],
        "recregions": [fwdpy11.PoissonInterval(0, 1.00, 1e-3)],
        "rates": (0, MU, None),
        "gvalue": gvalue_objects,  # ************************************************ what I'm hardcoding in. Maybe I can make this an arg in the argparser so that I'm changing what I'm varying?
        # (additive_objects if args.sim_type == popt_differs, env_sd_objects if args.sim_type == env_sd_differs, env_mean_objects if args.sim_type == env_mean_differs) #trying to specify argparser idea above
        "prune_selected": False,
        "simlen": demog.metadata["total_simulation_length"],
        "demography": demog,
    }

    mparams = fwdpy11.ModelParams(**pdict)

    print()

    seed = int(np.random.randint(0, 100000, 1)[0])
    # randint returns numpt.int64, which json doesn't
    # know how to handle.  So, we turn it
    # to a regular Python int to circumvent this problem.
    rng = fwdpy11.GSLrng(seed)

    fwdpy11.evolvets(rng, pop, mparams, simplification_interval=100)
    # print(pop.N, pop.deme_sizes())  # it's at this point that pop actually has the multiple demes.
    return (pop, mparams, input_file)


def write_treefile(pop, input_file):
    g = demes.load(input_file)
    # g = demes.load("no_ancestry.yaml")
    ts = pop.dump_tables_to_tskit(demes_graph=g)
    ts.dump("sim.trees")

    tsl = tskit.load("sim.trees")
    graph_dict = tsl.metadata["demes_graph"]
    rebuilt_graph = demes.Graph.fromdict(graph_dict)
    ind_md = fwdpy11.tskit_tools.decode_individual_metadata(tsl)
    assert g == rebuilt_graph
    print()
    # print(tsl.metadata)
    print()
    print()

    provenance = json.loads(ts.provenance(0).record)
    print("PROVENANCE IS")
    print(provenance)
    return ind_md


# print(fwdpy11.ModelParams)


def fitness_phenotype_summary(args, mparams, ind_md):
    VS = args.VS
    MU = args.MU
    # print(tsl.metadata)
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
    # popt = tsl.model_params.gvalue.gvalue_to_fitness.optimum
    #
    # print(popt)
    # popt = tsl.metadata["model_params"]  # trying to be able to write popt to txt file
    # print(popt)

    # vs = tsl.model_params.gvalue.gvalue_to_fitness.VS
    # print(ind_md)

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
    """
    for i in env_sd_list:
        H2_list.append(
            4 * MU * VS / ((4 * MU * VS) + (i ** 2))
        )  # square env_sd bc sd is sqrt(variance)
    """
    print(H2_list)

    print(ind_md)

    with open(args.output_file, "w") as output_file:
        output_file.write(
            f"{'individual'}\t{'deme'}\t{'strength_stabilizing_selection'}\t{'mutation_rate'}\t{'Population_optimum'}\t{'e_mean'}\t{'e_SD'}\t{'ind_fitness'}\t{'ind_genetic_value'}\t{'ind_environmental_value'}\t{'ind_phenotype'}"
        )
        for ind, ind_md in enumerate(ind_md):
            output_file.write(
                f"\n{ind}\t{deme[ind]}\t{VS}\t{MU}\t{popt}\t{e_mean}\t{e_sd}\t{fitness[ind]}\t{genetic_value[ind]}\t{environmental_value[ind]}\t{phenotype[ind]}"
            )  # I want to have POPT and others be writing the results of the list, but not sure how to do that without the ts_metadata function


def main():
    # build our parser
    parser = make_parser()

    # process sys.argv
    args = parser.parse_args(sys.argv[1:])

    # check input
    validate_args(args)

    # (pop, input_file) = run_sim(args)

    # fitness_phenotype_summary(
    #    pop=pop, input_file=input_file
    # )  # make sure these and the above are in the same order

    (pop, mparams, input_file) = run_sim(args)

    ind_md = write_treefile(
        pop=pop, input_file=input_file
    )  # defining ind_md is necessary (ask skylar for help explaining to yourself why), but if you don't specify variables for objects in the tuple
    # that run_sim outputs,it doesn't know which one to assign to which/which one to access

    fitness_phenotype_summary(args=args, mparams=mparams, ind_md=ind_md)


if __name__ == "__main__":
    main()
