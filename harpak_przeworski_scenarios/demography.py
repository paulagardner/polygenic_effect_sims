import fwdpy11
import fwdpy11.discrete_demography
import sys
import tskit
import demes
import numpy as np
import json
import argparse


# example to run: python demography.py -y no_ancestry.yaml --MU 1e-3 --VS 1.0 --POPT 1.0 -1.0 --E_SD 1 --E_MEAN 0 --output_file case_1_no_ancestry_rep1.txt
# probably better to run this until I get E_SD, E_MEAN etc working properly, so as not to fool myself into thinking the input is specifying anything:
# python demography.py -y no_ancestry.yaml --MU 1e-3 --VS 1.0 --POPT 1.0 -1.0 --output_file case_1_no_ancestry_rep1.txt


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


# will need to add the above back in when I split things up into functions


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
    # MU = args.MU
    # POPT = args.POPT

    # E_SD = args.E_SD
    # E_MEAN = args.E_MEAN
    # and then iterate over the individuals when you write to the text file

    # optima_list = [0.0, 0.1, -0.1]
    # optima_list = [0.1, -0.1]
    # this actually works to make a list just like the above, but passing it is a different issue

    VS = args.VS
    MU = args.MU

    # E_SD = 1.0
    # env_sd_list = [1.0, 1.3, 0.7]
    # env_sd_list = [1.3, 0.7]

    optima_list = args.POPT
    env_sd_list = args.E_SD
    env_mean_list = args.E_MEAN

    ##E_MEAN = 1.0
    # env_mean_list = [1.0, 1.1, 0.9]
    # env_mean_list = [1.1, 0.9]

    burnin = 10

    """
    POPT = args.POPT
    E_SD = args.E_SD
    E_MEAN = args.E_MEAN
    """  # I would like this, and the loops below, to be for i in POPT, E_SD, etc. if that's the case, hopefully it's because we can just be doing the above... and get rid of these defaults:
    POPT_default = 0.0
    E_SD_default = 1.0
    E_MEAN_default = 0.0
    #######HARDCODED IN!!! FIX!

    additive_objects = []
    for i in optima_list:
        additive_objects.append(
            fwdpy11.Additive(
                2.0,
                fwdpy11.GSS(i, VS),
                fwdpy11.GaussianNoise(E_SD_default, E_MEAN_default),
            )
        )
    print(additive_objects)
    # make fwdpy11.Additive objects from the POPT specified using list comprehension

    env_sd_objects = []
    for i in env_sd_list:
        env_sd_objects.append(
            fwdpy11.Additive(
                2.0,
                fwdpy11.GSS(POPT_default, VS),
                fwdpy11.GaussianNoise(i, E_MEAN_default),
            )
        )

    env_mean_objects = []
    for i in env_mean_list:
        env_mean_objects.append(
            fwdpy11.Additive(
                2.0,
                fwdpy11.GSS(POPT_default, VS),
                fwdpy11.GaussianNoise(E_SD_default, i),
            )
        )

    demog = fwdpy11.discrete_demography.from_demes(input_file, burnin)
    # demog = fwdpy11.discrete_demography.from_demes("no_ancestry.yaml", burnin)
    pop = fwdpy11.DiploidPopulation(
        [v for _, v in demog.metadata["initial_sizes"].items()], 1.0
    )
    # print(pop.N)
    print()
    print()

    fwdpy11.ConstantS
    initial_sizes = [
        demog.metadata["initial_sizes"][i]
        for i in sorted(demog.metadata["initial_sizes"].keys())
    ]

    dbg = fwdpy11.DemographyDebugger(initial_sizes, demog.model)
    print(dbg.report)

    """
    if args.sim_type == popt_differs:
        gvalue_objects = additive_objects
    if args.sim_type == env_mean_differs:
        gvalue_objects = env_mean_objects
    if args.sim_type == env_sd_differs:
        gvalue_objeccts = env_sd_objects
    """  ###########ASK SKYLAR ABOUT THIS. IF THIS IS THE WAY TO DO IT THEN gvalue_objects IS WHAT I WAS PUTTING INTO gvalue

    pdict = {
        "sregions": [fwdpy11.GaussianS(0, 1, 1, 0.25)],
        "nregions": [],
        "recregions": [fwdpy11.PoissonInterval(0, 1.00, 1e-3)],
        "rates": (0, MU, None),
        "gvalue": env_sd_objects,  # ************************************************ what I'm hardcoding in. Maybe I can make this an arg in the argparser so that I'm changing what I'm varying?
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
    )  # defining ind_md is necessary (ask skylar for help explaining to yourself why), but if you don't specify variables for objects in the tuple that run_sim outputs,

    fitness_phenotype_summary(args=args, mparams=mparams, ind_md=ind_md)


if __name__ == "__main__":
    main()
