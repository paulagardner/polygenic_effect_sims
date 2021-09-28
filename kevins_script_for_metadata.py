import sys
import fwdpy11
import numpy as np

ts = fwdpy11.tskit_tools.load(
    sys.argv[1]
)  # calling sys.argv[1] is because treefile is our 'first' command line argument as defined in make_parser, right?

ind_md = ts.decode_individual_metadata()

fitness = np.zeros(len(ind_md))
phenotype = np.zeros(len(ind_md))

for i, md in enumerate(ind_md):
    fitness[i] = md.w
    phenotype[i] = md.g + md.e

print(f"Mean fitness = {fitness.mean()}.  Mean phenotype = {phenotype.mean()}")