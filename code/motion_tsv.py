#!/usr/bin/env python

"""
short script to include headers in the motion parameter files
Ran this script in the root of the directory with
code/motion_tsv.py
Important: Don't datalad add any file to the dataset before running the command,
as Datalad/git annex then tries to protect the files from being overwritten!
"""
import pandas as pd
import numpy as np
from glob import glob

def main(file,
         header,
         sep
         ):
    files = glob(file)
    for f in sorted(files):
        data=pd.read_csv(f, sep=sep, header=None)
        np.savetxt(f, data, delimiter='\t', header=header, comments='')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inputs', help="Path to files that a header should be added to."
                                               "Please specify a path with appropriate wildcards for"
                                               "globbing.")
    parser.add_argument('--header', nargs='+', help="The new header to be added. Make sure it corresponds"
                                                    "to the number of columns in the input files. Specify"
                                                    "as consecutive strings.")
    parser.add_argument('-s', '--sep', help="What is the separator of the input files? Default: '\t'",
                        default='\t')
    args = parser.parse_args()

    header = '\t'.join(args.header)
    print("I will try to use this header: {}.".format(header))

    file = args.inputs
    print("I will glob for this file: {}.".format(file))

    sep = args.sep

    main(file, header, sep)