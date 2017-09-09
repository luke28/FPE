import os, sys
import argparse
import json
import numpy as np


def main():
    parser = argparse.ArgumentParser(formatter_class = argparse.RawTextHelpFormatter)
    parser.add_argument('--n_nodes', type = str, required = True)
    parser.add_argument('--input', type = str, required = True)
    args = parser.parse_args()
    js = json.loads(open(args.input, "r").read())
    coordinates = np.array(js["coordinates"])
    c = coordinates[: int(args.n_nodes)]
    with open("fpe_fea_" + str(c.shape[1]), "w") as f:
        for items in c:
            for item in items:
                f.write(str(item) + "\t")
            f.write("\n")

if __name__ == "__main__":
    main()
