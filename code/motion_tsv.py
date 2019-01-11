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

motion_path = 'sub*/ses-movie/func/sub*_ses-movie_task-avmovie_run*_bold_desc-mcparams_motion.txt'
motion_files = glob(motion_path)
head_mc = 'X\tY\tZ\tRotX\tRotY\tRotZ'
for mc in sorted(motion_files):
    data=pd.read_csv(mc, sep='  ', header=None)
    np.savetxt(mc, data, delimiter='\t', header=head_mc, comments='')
