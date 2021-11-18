import argparse
import fwdpy11
import sys
import numpy as np
import json
import tskit

def make_parser() -> argparse.ArgumentParser:
    # make an argument parser that can output help.
    # The __file__ is the full path to this file
    ADHF = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(__file__, formatter_class=ADHF)

    # Add an argument
    parser.add_argument(
        "--treefile", "-t", type=str, default=None, help="Tree file output name (tskit format)"
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
        raise ValueError(f"treefile to be written cannot be None")

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


# example for how you might run this: python harpak_przeworski.py --N 100 --MU 1e-3 --POPT 0.0 --VS 1.0 --E_SD 1.0 --E_MEAN 1.0 --treefile harpak_przeworski.trees


def run_sim(args: argparse.Namespace) -> fwdpy11.DiploidPopulation:
    MU = args.MU
    POPT = args.POPT
    VS = args.VS
    E_SD = args.E_SD
    E_MEAN = args.E_MEAN
    N = args.N

    paramsdict = {
        "sregions": [fwdpy11.GaussianS(0, 1, 1, 0.25)],
        "nregions": [],
        "recregions": [fwdpy11.PoissonInterval(0, 1.00, 1e-3)],
        "rates": (0, MU, None),
        "gvalue": 
            fwdpy11.Additive(
                2, fwdpy11.GSS(POPT, VS), fwdpy11.GaussianNoise(E_SD, E_MEAN)
            )
        ,
        "prune_selected": False,
        "simlen": 10 * N,
    }

    pop = fwdpy11.DiploidPopulation(N, 1.00)
    
    # randint returns numpt.int64, which json doesn't
    # know how to handle.  So, we turn it 
    # to a regular Python int to circumvent this problem.
    seed = int(np.random.randint(0, 100000, 1)[0])

    rng = fwdpy11.GSLrng(seed)
 
    params = fwdpy11.ModelParams(**paramsdict) #as in example for dict, these can be the same variable. I keep them separate here for legibility

    fwdpy11.evolvets(rng, pop, params, simplification_interval=100)

    # md = np.array(pop.diploid_metadata, copy=False)

    # h2 = md["g"].var() / ((md["g"] + md["e"]).var())

    return pop, params, seed, E_MEAN, E_SD # if you don't return pop you'll get an error in write_treefile: 'AttributeError: 'NoneType' object has no attribute 'dump_tables_to_tskit'



def write_treefile(*, pop: fwdpy11.DiploidPopulation, seed: int, params: fwdpy11.ModelParams, args: argparse.Namespace, E_MEAN: int, E_SD:int): #note that technically, in line 159 below, you don't need the star here to be able to use the args in any order IF you've named them. However, the converse isn't true- once you have the star here, you need to have non-positional arguments when the fuction is called. 
    ts = pop.dump_tables_to_tskit(# The actual model params
    model_params= params, #why is it not necessary to have params as an argument in write_treefile, like seed or args? from my understanding, we're defining it here, and model_params is of class ModelParams(so far as I can tell)

    # Any dict you want.  Some of what I'm putting here is redundant...
    # This dict will get written to the "provenance" table
    parameters={"seed": seed, "simplification_interval": 100, "meanE": E_MEAN, "E_SD": E_SD}, 
    wrapped=True,)
    # The ts is a fwdpy11.WrappedTreeSequence.
    # To dump it, access the underling tskit.TreeSequence
    ts.ts.dump(args.treefile)



    print(ts.model_params)
    print()
    provenance = json.loads(ts.ts.provenance(0).record)
    print(provenance)
    print()
    print(ts.model_params.gvalue.gvalue_to_fitness.VS)
    print(ts.model_params.gvalue.gvalue_to_fitness.optimum)
    print()


def main():
    # build our parser
    parser = make_parser()

    # process sys.argv
    args = parser.parse_args(sys.argv[1:])

    # check input
    validate_args(args)

    # evolve our population
    pop,params,seed,E_MEAN,E_SD= run_sim(args) #if you keep as before, where pop = run_sim, AttributeError: 'function' object has no attribute 'params'

    # write the output to a tskit "trees" file
    print("params is", type(params))
    write_treefile(pop=pop, params=params, seed=seed, args=args, E_MEAN=E_MEAN, E_SD=E_SD)



if __name__ == "__main__":
    main()