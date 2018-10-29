#!/usr/bin/python

"""
short script to include headers in the motion parameter files
"""
import pandas as pd
import numpy as np
from glob import glob
import os.path

motion_path = 'sub*/ses-movie/func/sub*_ses-movie_task-avmovie_run*_bold_desc-mcparams_motion.tsv'
motion_files = glob(motion_path)
head_mc = 'X\tY\tZ\tRotX\tRotY\tRotZ'
for mc in sorted(motion_files):
    data=pd.read_csv(mc, sep='  ', header=None)
    np.savetxt(mc, data, delimiter='\t', header=head_mc, comments='')
