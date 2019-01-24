#!/usr/bin/python

import numpy as np
import pandas as pd
import glob
from os.path import join as opj

"""
This script builds an event file for a fitlins model from remodnav and
visual confounds data. Execute it in the root of the directory.
Event files will be saved into the directory were the visual confound
regressors are.
"""

def ev3_info(subject_id,
             run_id,
             rootdir,
             remodnav,
             confounds
             ):
    """
    This function creates ev3 files from classified/labeled eyemovements,
    extracts information from the ev3 files about 'conditions', onsets and
    durations and returns and saves everything in a dataframe-event file.
    "subject_id", "run_id", "session_id", "task_id" should be strings following
    the format 'sub-01', 'run-1', 'ses-movie', 'task-avmovie' or similar.
    "confinfo", "experiment_dir", "eyemovement_dir" should be paths or
    directory names
    """
    path = rootdir + subject_id + remodnav + '*{}_desc-remodnav_events*.tsv'.format(run_id)
    infile = glob.glob(path)[0]
    print(infile)
    subs = ["sub-01", "sub-02", "sub-03", "sub-04", "sub-05", "sub-06", "sub-09",
            "sub-10", "sub-14", "sub-15", "sub-16", "sub-17", "sub-18", "sub-19", "sub-20"]
    # pixel need to be converted to visual degrees.
    if subject_id in subs:
        print("subject stems from fmri sample")
        conversion_factor = 0.018558123256059607
    # this would be for lab subjects
    else:
        conversion_factor = 0.026671197202630847

        # length and angle calculation basics

    def cart2pol(x, y):
        rho = np.sqrt(x ** 2 + y ** 2)
        theta = np.arctan2(y, x)
        return rho, theta

    # length in visual degrees and angle basics
    def calc_polar(x_start, x_end, y_start, y_end, conversion_factor):
        """function takes start and end coordinates in x and y direction and the
        appropriate conversion factor (lab or mri)"""
        # calculate saccade length from raw eye tracking data
        x, y = x_end - x_start, y_end - y_start
        # transform via cart2pol. rho = length, theta = angle
        rho, theta = cart2pol(x, y)
        # convert rho in px into visual degrees
        vis_deg = rho * conversion_factor
        return vis_deg, theta

    # extract and calculate necessary measures
    def polar_coordinates(data, conversion_factor):
        """data is an array with columns time, x_start, x_end, y_start, y_end.
        Conversion factor is either mri_deg_per_px or lab_deg_per_px"""
        angles = []
        lengths = []
        time = data['onset']  # extract time from data (first col)
        duration = data['duration']  # extract duration from data
        for row in data:
            length, angle = calc_polar(row['start_x'], row['end_x'], row['start_y'],
                                       row['end_y'], conversion_factor)
            angles.append(angle)
            lengths.append(length)
        return time.tolist(), duration.tolist(), angles, lengths

    df = np.recfromcsv(infile, delimiter='\t',
                       dtype={'names': ('onset', 'duration', 'label',
                                        'start_x', 'start_y', 'end_x', 'end_y', 'amp',
                                        'peak_vel', 'med_vel', 'avg_vel'),
                              'formats': ('f8', 'f8', 'U10', 'f8', 'f8', 'f8', 'f8',
                                          'f8', 'f8', 'f8', 'f8')})
    # take all types of saccades
    data = df[(df['label'] == 'SACC') | (df['label'] == 'ISAC')]
    # calculate angles and lengths, extract times and durations
    time, duration, angles, lengths = polar_coordinates(data, conversion_factor)
    trial_dummy = np.repeat('dummy_string_value', len(time))
    trans_data = np.column_stack((time, duration, angles, lengths))
    # group the saccades into directions. Relevant for the glm are only
    # 30° around the horizontal and vertical axis, but the rest is included
    # in the GLM as well for more power. If necessary later, I include the
    # necessary bounderies for more directions as well.
    # I computed the radians with this degree to radian converter for simplicity:
    # https://www.rapidtables.com/convert/number/degrees-to-radians.html?x=75

    up_idx = np.where(np.logical_and(trans_data[:, 2] >= 1.308996939,
                                     trans_data[:, 2] <= 1.8325957146))[0]
    down_idx = np.where(np.logical_and(trans_data[:, 2] <= -1.308996939,
                                       trans_data[:, 2] >= -1.8325957146))[0]
    left_idx = np.where(np.logical_or(trans_data[:, 2] >= 2.8797932658,
                                      trans_data[:, 2] <= -2.8797932658))[0]
    right_idx = np.where(np.logical_and(trans_data[:, 2] <= 0.2617993878,
                                        trans_data[:, 2] >= -0.2617993878))[0]
    # additional direction for possible later use:
    up_right_idx = np.where(np.logical_and(trans_data[:, 2] < 1.308996939,
                                           trans_data[:, 2] >= 0.7853981634))[0]
    right_up_idx = np.where(np.logical_and(trans_data[:, 2] < 0.7853981634,
                                           trans_data[:, 2] > 0.2617993878))[0]
    right_down_idx = np.where(np.logical_and(trans_data[:, 2] < -0.2617993878,
                                             trans_data[:, 2] >= -0.7853981634))[0]
    down_right_idx = np.where(np.logical_and(trans_data[:, 2] < -0.7853981634,
                                             trans_data[:, 2] > -1.308996939))[0]
    down_left_idx = np.where(np.logical_and(trans_data[:, 2] < -1.8325957146,
                                            trans_data[:, 2] > -2.3561944902))[0]
    left_down_idx = np.where(np.logical_and(trans_data[:, 2] <= -2.3561944902,
                                            trans_data[:, 2] > -2.8797932658))[0]
    left_up_idx = np.where(np.logical_and(trans_data[:, 2] < 2.8797932658,
                                          trans_data[:, 2] > 2.3561944902))[0]
    up_left_idx = np.where(np.logical_and(trans_data[:, 2] <= 2.3561944902,
                                          trans_data[:, 2] > 1.8325957146))[0]

    for i in up_idx:
        trial_dummy[i] = "UP"
    for i in down_idx:
        trial_dummy[i] = "DOWN"
    for i in left_idx:
        trial_dummy[i] = "LEFT"
    for i in right_idx:
        trial_dummy[i] = "RIGHT"
    for i in up_right_idx:
        trial_dummy[i] = "UP_RIGHT"
    for i in right_up_idx:
        trial_dummy[i] = "RIGHT_UP"
    for i in right_down_idx:
        trial_dummy[i] = "RIGHT_DOWN"
    for i in down_right_idx:
        trial_dummy[i] = "DOWN_RIGHT"
    for i in down_left_idx:
        trial_dummy[i] = "DOWN_LEFT"
    for i in left_down_idx:
        trial_dummy[i] = "LEFT_DOWN"
    for i in left_up_idx:
        trial_dummy[i] = "LEFT_UP"
    for i in up_left_idx:
        trial_dummy[i] = "UP_LEFT"

    # combine into dataframe
    rec = np.core.records.fromarrays(trans_data.transpose(), names='onset, duration, angle, amplitude',
                                     formats='f8, f8, f8, f8')
    df_saccs = pd.DataFrame.from_records(rec)
    df_saccs['trial_type'] = trial_dummy.tolist()

    # add confounds
    con_lrdiff_path = rootdir + subject_id + confounds + '*{}*desc-lrdiff*'.format(run_id)
    con_lrdiff = glob.glob(con_lrdiff_path)[0]
    lrdiff = pd.read_csv(con_lrdiff, sep=" ", header=None, names=['onset', 'duration', 'amplitude'])
    trial_dummy_lrdiff = np.repeat('LRDIFF', len(lrdiff))
    lrdiff['trial_type'] = trial_dummy_lrdiff

    con_rms_path = rootdir + subject_id + confounds + '*{}*desc-rms*'.format(run_id)
    con_rms = glob.glob(con_rms_path)[0]
    rms = pd.read_csv(con_rms, sep=" ", header=None, names=['onset', 'duration', 'amplitude'])
    trial_dummy_rms = np.repeat('RMS', len(rms))
    rms['trial_type'] = trial_dummy_rms

    con_pd_path = rootdir + subject_id + confounds + '*{}*desc-pd*'.format(run_id)
    con_pd = glob.glob(con_pd_path)[0]
    cpd = pd.read_csv(con_pd, sep='\t', header=None, names=['onset', 'duration', 'amplitude'])
    trial_dummy_cpd = np.repeat('PD', len(cpd))
    cpd['trial_type'] = trial_dummy_cpd

    dfs = [df_saccs, lrdiff, rms, cpd]
    df_event = pd.concat(dfs)
    # reindexing, because pandas orders columns alphabetically
    newindex = ['onset', 'duration', 'trial_type', 'amplitude']
    df_events = df_event.reindex(newindex, axis='columns')
    # assert that we did not miss any saccade direction
    assert 'dummy_string_value' not in np.unique(df_events['trial_type'].values)
    # assert that we found all 15 different types of trial_types
    assert len(np.unique(df_events['trial_type'].values)) == 15

    outfilepath = rootdir + subject_id + confounds
    df_events.to_csv(opj(outfilepath, '{}_ses-movie_task-avmovie_{}_events.tsv'.format(subject_id,
                                                                                       run_id)),
                     header=True,
                     index=False,
                     sep='\t')
    return df_events


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--remodnav', help='Specify a path to the remodnav files'
                                                 'based on the subject directory, e.g. '
                                                 'ses-movie/eyetrack/',
                        required=True)
    parser.add_argument('-c', '--confounds', help='Specify a path the confound files'
                                                  'based on the subject directory, e.g.'
                                                  'ses-movie/func',
                        required=True)
    parser.add_argument('--rootdir', help='What is the root of the directory (i.e.where'
                                          'the subject subdirectories lie). Defaults to'
                                          '"."', default='.')
    parser.add_argument('--runs', nargs='+', help='Which runs? Specify as consecutive strings'
                                                  'such as "run-1" "run-4" "run-8", defaults'
                                                  'to all 8 runs')
    parser.add_argument('--subjects', nargs='+', help='Which subjects? Specify as consecutive strings'
                                                      'such as "sub-01" "sub-04" "sub-09", defaults'
                                                      'to all subjects in the repo')
    args = parser.parse_args()

    remodnav = '/' + args.remodnav + '/'
    confounds = '/' + args.confounds + '/'
    rootdir = args.rootdir + '/'
    if args.runs:
        runs = [i for i in args.runs]
    else:
        runs = ['run-1', 'run-2', 'run-3', 'run-4', 'run-5', 'run-6', 'run-7', 'run-8']
    if args.subjects:
        subjects = [i for i in args.subjects]
    else:
        subjects = sorted([path.split('/')[-1] for path in glob.glob(rootdir + 'sub-*')])
        print('I found the following subjects in the specified root directory: {}'.format(subjects))

    for subject_id in subjects:
        for run_id in runs:
            ev3_info(subject_id,
                     run_id,
                     rootdir,
                     remodnav,
                     confounds)
