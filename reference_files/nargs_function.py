import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--param', type=float, help="A parameter", default=10.1)

parser.add_argument('treefiles', metavar='TREEFILE', type=str, nargs='+',
                    help='List of .trees files')
args = parser.parse_args()

print(f"The parameter is {args.param}")
print("The tree files are:")
for i in args.treefiles:
    try:
        with open(i, 'r') as f:
            pass
    except Exception as e:
        print(e)
