#!/usr/bin/env python

import pandas as pd
from glob import glob
import matplotlib.pyplot as plt

"""
A small last minute script to analyze and plot a couple of 
descriptive statistics on the saccades used in the event files.
"""


def main(input,
         outputpath):
    """
    plot a vertical bar plot of the counts of saccades,
    print the number of saccades per direction to the terminal
    """
    # read in:
    data = glob(input)
    assert len(data) == 120
    files = []
    for d in data:
        file = pd.read_csv(d, sep = '\t')
        files.append(file)

    # stack stuff together
    events = pd.concat(files, axis = 0).reset_index(drop = True)

    subset = events.trial_type[(events.trial_type == 'LEFT') |
                               (events.trial_type == 'RIGHT') |
                               (events.trial_type == 'UP') |
                               (events.trial_type == 'DOWN')]

    # plot
    fig, ax = plt.subplots()
    subset.value_counts().plot(ax=ax, kind='barh', title = 'total saccade counts per direction', colormap='viridis', grid = True)
    plt.savefig(outputpath)

    # count stuff and print it to terminal
    counts = events['trial_type'].value_counts()
    print('Left saccades: {0} \nRight saccades: {1} \nUp saccades: {2} \nDown saccades: {3}'.format(counts.LEFT,
                                                                                                    counts.RIGHT,
                                                                                                    counts.UP,
                                                                                                    counts.DOWN))

# argparse
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='input event files as a pattern to glob for')
    parser.add_argument('-o', '--output', help='Where the plot should be saved')

    args = parser.parse_args()


    datapath = args.input
    outputpath = args.output
    main(datapath, outputpath)