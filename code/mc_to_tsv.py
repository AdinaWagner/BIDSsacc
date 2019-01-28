#!/usr/bin/env python

"""small script to transform motion txt files into tsv with header"""

import pandas as pd
from glob import glob
import numpy as np

def main(filepath,
         header
         ):
    files = sorted(glob(filepath))
    print("I found {} files.".format(len(files)))
    for f in files:
        data = pd.read_csv(f, sep = "  ", header = None)
        np.savetxt(f, data, delimiter='\t', header=header, comments='',
                    fmt='%1.4f')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input', help="txt files to be transformed")
    parser.add_argument('--header', nargs = '+', help="which header to use",
                        default = ['X', 'Y', 'Z', 'RotX', 'RotY', 'RotZ'])

    args = parser.parse_args()
    filepath = args.input
    print(filepath)
    header = '\t'.join(args.header)

    main(filepath, header)
