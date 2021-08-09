import argparse
import fwdpy11
import sys
import numpy as np


def make_parser() -> argparse.ArgumentParser:
    # make an argument parser that can output help.
    # The __file__ is the full path to this file
    ADHF = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(__file__, formatter_class=ADHF)

    # Add an argument
    parser.add_argument(
        "--treefile", "-t", type=str, default=None, help="Tree file name (tskit format)"
    )

    # parser.add_argument("--seed", "-s", type=int, default=0, help="Random number seed")

    parser.add_argument("--N", type=int, help="Number of individuals")

    parser.add_argument("--MU", "--u", type=float, help="Mutation rate")

    parser.add_argument(
        "--POPT", type=float, default=0, help="Population optimum trait value"
    )

    parser.add_argument(
        "--VS",
        type=float,
        help="Variance of S, the inverse strength of stabilizing selection",
    )

    parser.add_argument(
        "--E_SD",
        type=float,
        help="Environmental effect distribution's standard deviation",
    )

    parser.add_argument(
        "--E_MEAN", type=float, help="Environmental effect distribution's mean"
    )

    return parser


def validate_args(args: argparse.Namespace):
    if args.treefile is None:
        raise ValueError(f"treefile cannot be None")

    # if args.seed < 0:
    # raise ValueError(f"invalid seed value: {args.seed}")

    if args.POPT is None:
        raise ValueError(f"Population optimum trait value cannot be None")

    if args.N is None:
        raise ValueError(
            f"Number of individuals cannot be None"
        )  # this was giving me trouble. Error was AttributeError: 'Namespace' object has no attribute 'N'

    if args.MU is None:
        raise ValueError(f"Mutation rate cannot be None")

    if args.VS is None:
        raise ValueError(
            f"In this simulation using stabilizing selection, VS cannot be None"
        )


def run_sim(args: argparse.Namespace) -> fwdpy11.DiploidPopulation:
    for rep in range(5):

        MU = args.MU
        POPT = args.POPT
        VS = args.VS
        E_SD = args.E_SD
        E_MEAN = args.E_MEAN
        N = args.N

        pdict = {
            "sregions": [fwdpy11.GaussianS(0, 1, 1, 0.25)],
            "nregions": [],
            "recregions": [fwdpy11.PoissonInterval(0, 1.00, 1e-3)],
            "rates": (0, MU, None),
            "gvalue": [
                fwdpy11.Additive(
                    2, fwdpy11.GSS(POPT, VS), fwdpy11.GaussianNoise(E_SD, E_MEAN)
                )
            ],
            "prune_selected": False,
            "simlen": 10 * N,
        }

        pop = fwdpy11.DiploidPopulation(N, 1.00)

        seed = np.random.randint(0, 100000, 1)[0]

        rng = fwdpy11.GSLrng(seed)

        params = fwdpy11.ModelParams(**pdict)

        fwdpy11.evolvets(rng, pop, params, 100)

        # md = np.array(pop.diploid_metadata, copy=False)

        # h2 = md["g"].var() / ((md["g"] + md["e"]).var())

        return pop # if you don't return pop you'll get an error in write_treefile: 'AttributeError: 'NoneType' object has no attribute 'dump_tables_to_tskit'


def write_treefile(pop: fwdpy11.DiploidPopulation, args: argparse.Namespace):
    ts = pop.dump_tables_to_tskit()
    ts.dump(args.treefile)


if __name__ == "__main__":
    # build our parser
    parser = make_parser()

    # process sys.argv
    args = parser.parse_args(sys.argv[1:])

    # check input
    validate_args(args)

    # evolve our population
    pop = run_sim(args)

    # write the output to a tskit "trees" file
    write_treefile(pop, args)