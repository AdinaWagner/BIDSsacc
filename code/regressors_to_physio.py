#!/usr/bin/env python

import pandas as pd
import json
from glob import glob
import os

"""
This script should take the three movie property _regressors.tsv files and merge their amplitudes into
one _physio.tsv.gz file. It should also generate a json side car with Onset time, frequency, and column names.
"""


def main(files,
         runs,
         sep,
         header,
         subs
         ):
    """read in individual regressors per run, and merge their amplitudes into a single physio file.
    Then creates an appropriate json side car. Frequency and onset are taken from the data."""
    for sub in subs:
        for run in runs:
            amplitudes = []
            names = []
            sampling_rate = []
            onset = []
            for file in sorted(files):
                if header:
                    data = pd.read_csv(file.format(sub, run), sep=sep)
                    amplitudes.append(data['amplitude'])
                    # get the frequency from duration data, and onset from onset
                    # we assume duration is in seconds
                    sampling_rate.append(1. / data['duration'][0])
                    onset.append(data['onset'][0])
                else:
                    data = pd.read_csv(file.format(sub, run), sep=sep, header=None)
                    amplitudes.append(data.iloc[:, 2])
                    # get the frequency from duration data, and onset from onset
                    # we assume duration is in seconds
                    sampling_rate.append(1. / data.iloc[:, 1][0])
                    onset.append(data[:, 0][0])
                # get the variable name for json
                names.append(file.split('/')[-1].split('desc-')[-1].split('_regressors')[0])
                # assert we have the same onsets and frequencies in all files:
            assert len(set(onset)) == 1
            assert len(set(sampling_rate)) == 1

            # build a new filename
            basename = file.format(sub, run).split('/')[-1].split('_desc')[0]
            pathname = os.path.dirname(file)
            # make a tsv.gz file from the amplitudes
            merge = pd.concat(amplitudes, axis=1)
            merge.to_csv(pathname + '/' + basename + '_physio.tsv.gz', compression='gzip', sep='\t')
            # write a json
            json_data = dict([("SamplingFrequency", sampling_rate[0]),
                              ("StartTime", onset[0]),
                              ("Columns", names)])
            with open(pathname + '/' + basename + '_physio.json', 'w') as outfile:
                json.dump(json_data, outfile)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', nargs='+', help="Which tsv files to combine, specified as strings with curly "
                                                         "braces for run eg \"sub-01/ses-movie/func/sub-01_ses-movie_"
                                                         "task-avmovie_run-{}_desc-pd_regressors.tsv\""
                                                         "\"sub-{}/ses-movie/func/sub-{}_ses-movie_task-avmovie_run-{}"
                                                         "_desc-rms_regressors.tsv\"")
    parser.add_argument('-r', '--run', nargs='+',
                        help="Which runs? Specify as consecutive integers eg. 1 2 4. Defaults to 8",
                        default=[1, 2, 3, 4, 5, 6, 7, 8])
    parser.add_argument('-s', '--sep', help="whats the seperator in the regressor files? Default: \t", default='\t')
    parser.add_argument('--header', help="Is there the 'onset', 'duration', 'amplitude' header present?. Default True",
                        default=True)
    parser.add_argument('--subject', nargs='+',
                        help="Which subjects to use? Specify as consecutive srings, as in '01' '03'."
                             "Defaults to all subjects in current directory.")

    args = parser.parse_args()

    files = args.input
    runs = args.run
    sep = args.sep
    header = args.header
    if args.subject:
        subs = args.subject
    else:
        subs = sorted(glob('sub-*'))

    main(files,
         runs,
         sep,
         header,
         subs)
