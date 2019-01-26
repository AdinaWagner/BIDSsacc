#!/usr/bin/env python

"""
This script replaces the Repetition time in nifti files.
"""

import nibabel as nb
from glob import glob

def main(filepath,
         TR
         ):
    """
    Read in nififiles and replace the TR in header.
    :param filepath:
    :param TR:
    :return:
    """
    files = glob(filepath)
    assert len(files) > 0
    print("I found {} file(s).".format(len(files)))
    for file in files:
        data = nb.load(file)
        zooms = data.header.get_zooms()
        assert len(zooms) == 4
        # build a tuple with old vx dimensions and new TR
        new_zooms = zooms[0:3] + (TR,)
        assert new_zooms[0:3] == zooms[0:3]
        data.header.set_zooms(new_zooms)
        nb.save(data, file)
        print('Saved the file under {}. Previous TR was {}, new TR is {}.'.format(file, zooms[3], TR))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input', help="Input files to be processed. You can specify a"
                                              "pattern such as 'sub-*/ses-movie/func/sub-*bold.nii.gz'"
                                              "to be globbed for.")
    parser.add_argument('--TR', help="New TR to be set in the header. Defaults to 2.0.",
                        default=2.0, type=float)

    args = parser.parse_args()

    TR = args.TR
    filepath = args.input

    # execute this thing
    main(filepath, TR)