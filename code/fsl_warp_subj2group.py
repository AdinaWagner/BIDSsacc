#!/usr/bin/python

import os
from glob import glob
from subprocess import call

"""
Transform aligned highpassed files into groupspace/MNI152.
Keyword to use according to BIDS Derivatives RC1: MNI152NLin6Sym
example filename for mask:
 - sub-001_task-rest_run-1_space-MNI305_desc-PFC_mask.nii.gz
example filename for bold:
 - <source_keywords>[_space-<space>][_desc-<label>]_<suffix>.<ext>

"""


def main(participants,
         basedir,
         inputfile,
         mask,
         reference,
         template):
    """
    performs the warping and saving.
    :param participants:
    :param basedir:
    :param inputfile:
    :param mask:
    :param reference:
    :param template:
    :return:
    """
    interpolation = 'nn'
    for participant in participants:
        # glob bold files and brain masks
        input_fns = glob(basedir + participant + inputfile)
        input_fns.append(glob(basedir + participant + mask))

        # get the template and reference
        reference_fn = basedir + reference.format(participant)
        template_fn = basedir + template.format(participant)
        #print(input_fns, reference_fn, template_fn)

        for inp in input_fns:
            # build an output name
            # lets hope it follows bids convention
            # TODO: clean this up at later stage
            session = inp.split('/')[-1].split('_')[1]
            task = inp.split('/')[-1].split('_')[2]
            run = inp.split('/')[-1].split('_')[3]
            # if we have a highpassed bold file:
            if 'desc-highpass' in inp.split('/')[-1].split('_'):
                outname = '_'.join([participant,
                                    session,
                                    task,
                                    run,
                                    'space-MNI152NLin6Sym',
                                    '_'.join(inp.split('/')[-1].split('_')[4:])])
            elif 'mask.nii.gz' in inp.split('/')[-1].split('_'):
                outname = '_'.join([participant,
                                    session,
                                    task,
                                    run,
                                    'space-MNI152NLin6Sym',
                                    '_'.join(mask.split('/')[-1].split('_')[4:])])
            output_dir = basedir + participant + '/' + '/'.join(inp.split('/')[2:-1]) + '/' + outname
            # print(outname, output_dir)
            fsl_cmd = ("fsl5.0-applywarp -i {0}  -o {1} -r {2} -w {3} --interp={4}".format(
                inp, output_dir, reference_fn, template_fn, interpolation))

            call(fsl_cmd, shell=True)

            print("Warped input files{0}; output file is {1}".format(inp, output_dir))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--subject', nargs='+', help="Which participants to warp. Default: all of them")
    parser.add_argument('-b', '--basedir', help="specify root of the directory", default='.')
    parser.add_argument('-i', '--inputfile', help="Highpassed bold files"
                                                  " that should be warped,"
                                                  " path starts at subject dir level,"
                                                  "eg ses-movie/func/sub-*highpassed_bold.nii.gz")
    parser.add_argument('-m', '--mask', help="Mask file that should be warped,"
                                             "path starts at subject dir level,"
                                             "eg ses-movie/func/sub-*desc-brain_mask.nii.gz")
    parser.add_argument('-t', '--template', help="Which template to use. Path from"
                                                 "root of the directory with curly brackets"
                                                 "as placeholders for participants, "
                                                 "eg sourcedata/tnt/{}/bold3Tp2/in_grpbold3Tp2/subj2tmpl_warp.nii.gz.")
    parser.add_argument('-r', '--reference', help="Which reference head image to use. Specify"
                                                  "from root of the directory with curly brackets"
                                                  "as placeholders for participants, eg"
                                                  "sourcedata/tnt/{}/bold3Tp2/in_grpbold3Tp2/head.nii.gz")

    args = parser.parse_args()

    basedir = args.basedir + '/'
    if args.subject:
        participants = [i for i in args.subject]
    else:
        participants = sorted([path.split('/')[-1] for path in glob(basedir + 'sub-*')])

    inputfile = args.inputfile
    mask = args.mask
    reference = args.reference
    template = args.template
    main(participants,
         basedir,
         inputfile,
         mask,
         reference,
         template)
