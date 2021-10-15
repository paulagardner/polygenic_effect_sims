import argparse
import sys
import fwdpy11
import numpy as np

# example usage: python md_analysis_with_parser.py harpak_przeworski.trees --metadata_output harpak_przeworski.txt or
# OR (positional argument style) python md_analysis_with_parser.py  --metadata_output harpak_przeworski.txt harpak_przeworski.trees. It seems if you only have one positional argument it doesn't matter where it goes. If the parser no longer has treefile as a positional argument, this will change.


def make_parser() -> argparse.ArgumentParser:
    # make an argument parser that can output help.

    ADHF = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(
        __file__,
        formatter_class=ADHF,
        description="process treefiles and output metadata",
    )
    ## __file__ is the full path to this file

    # Add an argument

    # in our case, we need the previous filname to load things in, and need to specify what file is going to come out of it. However, the treefiles argument will be put in at the end, since they will be positional arguments and those have to come at the end.
    parser.add_argument(
        "treefile",
        metavar="treefile",  # metavar vs dest, etc https://stackoverflow.com/questions/50965583/python-argparse-multiple-metavar-names
        nargs="+",
        type=str,
        default=None,
        help="Tree file input name",
    )
    # I want to be able to specify multiple treefiles so that I can have everything write to a single, output.txt file for processing later
    # https://docs.python.org/3/library/argparse.html
    # it isn't working in the same way it wasn't working when treefile() was a positional argument, but from Kevin: "that type of design usually will force you to write the tree file name to your output file, allowing you to match rows to particular files.". SO I think the issue is with my fitness_phenotype_summary function

    parser.add_argument(
        "--metadata_output", "-o", type=str, default=None, help="Output file name"
    )

    return parser


def validate_args(args: argparse.Namespace):
    if args.treefile is None:
        raise ValueError(f"treefile to be read cannot be None")

    if args.metadata_output is None:
        raise ValueError(f"output file to be written cannot be None")


# do the function for which you have made the arguments, write the output to another file
def fitness_phenotype_summary(args: argparse.Namespace) -> fwdpy11.tskit_tools.load:

    with open(
        args.metadata_output, "w"
    ) as output_file:  #'w' here is just the standard python switch(?) for write. Metadata_output is your parser argument, in the make_parser function
        # trying to fomat following this:
        output_file.write(
            # f"Treefile name\t\t\tMean fitness\t\t\tMean genetic value\t\t\tMean environmental value\t\t\tMean phenotype")
            f"{'Treefile_name':<30} {'Mean_fitness':<30} {'Mean_genetic_value':<30} {'Mean_environmental_value':<30} {'Mean_phenotype':<30}\n"
        )  # why formatting with underscores? If not, the way my plotting_metadata.r function works doesn't seem to read the headers in correctly

        input_file = args.treefile
        for input_file in args.treefile:
            ts = fwdpy11.tskit_tools.load(input_file)
            ind_md = ts.decode_individual_metadata()
            # fitness = np.zeros(len(ind_md))
            # phenotype = np.zeros(len(ind_md))
            # genetic_value = np.zeros(len(ind_md))
            # environmental_value = np.zeros(len(ind_md))

            fitness = np.array([md.w for md in ind_md])
            genetic_value = np.array([md.g for md in ind_md])
            environmental_value = np.array([md.e for md in ind_md])
            phenotype = np.array([md.g + md.e for md in ind_md])

            # Originally was using a for() loop of this format, but requires more lines:
            # fitness = np.zeros(len(ind_md))
            # phenotype = np.zeros(len(ind_md))
            # genetic_value = np.zeros(len(ind_md))
            # environmental_value = np.zeros(len(ind_md))

            # for i, md in enumerate(ind_md):
            #   fitness[i] = md.w

            #   genetic_value[i] = md.g

            #   environmental_value[i] = md.e

            #   phenotype[i] = md.g + md.e

            output_file.write(
                f"\n{input_file:<30} {fitness.mean():<30} {genetic_value.mean():<30} {environmental_value.mean():<30} {phenotype.mean():<30}\n"  # for reference #http://cis.bentley.edu/sandbox/wp-content/uploads/Documentation-on-f-strings.pdf
            )
            # why the newline character at the end? If you want to process things in R, you get a warning message "incomplete final line found" if you don't include it.

        print(
            f"{'Treefile name':<30} {'Mean fitness':<30} {'Mean genetic value':<30} {'Mean environmental value':<30} {'Mean phenotype':<30}\n{input_file:<30} {fitness.mean():<30} {genetic_value.mean():<30} {environmental_value.mean():<30} {phenotype.mean():<30}"
        )

        # f"{args.treefile}" #https://zetcode.com/python/argparse/ include args.treefile as your column name

        # you want to write to a file to look at and for future downstream analysis


# def write_nextfile(args: argparse.Namespace): #not really needed here,


if __name__ == "__main__":
    # build our parser
    parser = make_parser()

    # process sys.argv
    args = parser.parse_args(sys.argv[1:])

    # check input
    validate_args(args)

    # do the function we want it to do (requires that function has a name)
    # printvariable = fitness_phenotype_summary(args) #variable not necessary if you're not passing it to the next function(?)
    fitness_phenotype_summary(args)

    # write the output to a file that can be analysed downstream
    # write_nextfile(printvariable, args) #may not be necessary, as you've combined the process and write function
