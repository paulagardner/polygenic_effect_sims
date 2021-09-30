import argparse
import sys
import fwdpy11
import numpy as np

#ts = fwdpy11.tskit_tools.load(
    #sys.argv[1]
#}  # idea would be: calling sys.argv[1] is because treefile is our 'first' command line argument as defined in make_parser, right?

#the above was present in the file kevin sent as a sample. You've put it in your calc_stats function below, much the same way that in the main simulation file, run_sim is in fwdpy11.DiploidPopulation, since we want tskit_tools.load to load in the file, yes? And validate_args will not be doing it...

def make_parser() -> argparse.ArgumentParser:
     # make an argument parser that can output help.
    # The __file__ is the full path to this file
    ADHF = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(__file__, formatter_class=ADHF)

    # Add an argument
    parser.add_argument("--treefile", "-t", type = str, default = None, help = "Tree file input name")

    #in our case, we need the previous filname to load things in, and need to specify what file is going to come out of it 
    parser.add_argument("--metadata_output", "-o". type = str, default = None, help = "Output file name")

    return parser


def validate_args(args: argparse.Namespace):
    if args.treefile is None: 
        raise ValueError(f"treefile to be read cannot be None")

    if args.metadata_output is None: 
        raise ValueError(f"output file to be written cannot be None")



#do the function for which you have made the arguments 
def calc_stats(args: argparse.namespace) -> fwdpy11.tskit_tools.load:
    ind_md = ts.decode_individual_metadata()
    fitness = np.zeros(len(ind_md))
    phenotype = np.zeros(len(ind_md))
    
    for i, md in enumerate(ind_md):
        fitness[i] = md.w
        phenotype[i] = md.g + md.e

    print(f"Mean fitness = {fitness.mean()}.  Mean phenotype = {phenotype.mean()}")

#you want to write to a file for downstream analysis 
def write_nextfile():


#main
if __name__ == "__main__":
    # build our parser
    parser = make_parser()

    # process sys.argv
    args = parser.parse_args(sys.argv[1:])

    # check input
    validate_args(args)

    #do the function we want it to do (requires that function has a name)
    printvariable = calc_stats(args)

    #write the output to a file that can be analysed downstream 
    write_nextfile(printvariable, args)
