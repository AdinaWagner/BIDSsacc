#!/bin/bash

# This file is intended to restructure a variety of different inputs in
# BIDS_sacc into a BIDS conform dataset for further analysis with nipype and
# fitlins. Previously you will need to datalad install a number of datasets.


set -e
set -u

#find subjects
subs=$(find inputs/aligned -type d -name "sub-*" -printf "%f\n" | sort)

#prereqs
initial_dir=../BIDS_sacc
session=ses-movie
task=task-avmovie
# create directories if the don't exists yet
for sub in $subs; do
    sub_dir=$sub
    [ ! -d "${sub_dir}/${session}/func" ] && mkdir -p "${sub_dir}/${session}/func"
    [ ! -d "${sub_dir}/${session}/anat" ] && mkdir -p "${sub_dir}/${session}/anat"
    [ ! -d "${sub_dir}/${session}/xfm" ] && mkdir -p "${sub_dir}/${session}/xfm"
    [ ! -d "${sub_dir}/${session}/eyetrack" ] && mkdir -p "${sub_dir}/${session}/eyetrack"
done

[ ! -d "models" ] && mkdir -p "models"

# move aligned bold files, motion parameter files and annotation confound files
# into  into respective sub_dir/func dir
for sub in $subs; do
    cp inputs/aligned/${sub}/in_bold3Tp2/${sub}_task-avmovie_run-*_bold_mcparams.txt ${sub}/${session}/func
    for i in $(find ${sub}/${session}/func/*mcparams.txt); do
        mv $i $(echo $i | sed -e 's/task-avmovie/ses-movie_task-avmovie/');
    done
    for i in $(find ${sub}/${session}/func/*mcparams.txt); do
        mv $i $(echo $i | sed -e 's/bold_mcparams.txt/desc-mcparams_motion.tsv/');
    done
    cp inputs/aligned/${sub}/in_bold3Tp2/${sub}_task-avmovie_run-*_bold.nii.gz ${sub}/${session}/func
    for i in $(find ${sub}/${session}/func/*.nii.gz); do
        mv $i $(echo $i | sed -e 's/task/ses-movie_task/');
    done
    cp inputs/eyemovementlabels/${sub}/*.tsv ${sub}/${session}/eyetrack
    for i in $(find ${sub}/${session}/eyetrack/*.tsv); do
       mv $i $(echo $i | sed -e 's/task-movie/ses-movie_task-avmovie/');
    done
    cp inputs/annotation_confounds/fg_ad_run-*_lrdiff.ev3 ${sub}/${session}/func
    for i in $(find ${sub}/${session}/func/*lrdiff.ev3); do
        mv $i $(echo $i | sed -e "s/fg_ad/${sub}_${session}_${task}/");
    done
    for i in $(find ${sub}/${session}/func/*lrdiff.ev3); do
        mv $i $(echo $i | sed -e 's/lrdiff.ev3/desc-lrdiff_regressors.tsv/');
    done
    cp inputs/annotation_confounds/fg_ad_run-*_rms.ev3 ${sub}/${session}/func
    for i in $(find ${sub}/${session}/func/*rms.ev3); do
        mv $i $(echo $i | sed -e "s/fg_ad/${sub}_${session}_${task}/");
    done
    for i in $(find ${sub}/${session}/func/*rms.ev3); do
        mv $i $(echo $i | sed -e 's/rms.ev3/desc-rms_regressors.tsv/');
    done
    cp inputs/annotation_confounds/fg_av_ger_pd_run-*.txt ${sub}/${session}/func
    for i in $(find ${sub}/${session}/func/fg_av*.txt); do
        mv $i $(echo $i | sed -e "s/fg_av_ger_pd/${sub}_${session}_${task}/");
    done
    for i in $(find ${sub}/${session}/func/*.txt); do
        mv $i $(echo $i | sed -e 's/.txt/_desc-pd_regressors.tsv/');
    done
    cp inputs/tnt/${sub}/bold3Tp2/in_grpbold3Tp2/subj2tmpl_warp.nii.gz  ${sub}/${session}/xfm/
    for i in $(find ${sub}/${session}/xfm/subj2tmpl_warp.nii.gz); do
        mv $i $(echo $i | sed -e "s/subj2tmpl_warp/${sub}_from-BOLD_to-group_mode-image/");
    done
    cp inputs/tnt/${sub}/bold3Tp2/in_grpbold3Tp2/head.nii.gz  ${sub}/${session}/xfm/
    for i in $(find ${sub}/${session}/xfm/head.nii.gz); do
        mv $i $(echo $i | sed -e "s/head/NonstandardReference_space-group/");
    done
    for i in $(find ${sub}
done

# generate .json file to accompany mcparams file
for sub in $subs; do
    for run in $(seq 1 8); do
        cat <<EOT >  ${sub}/${session}/func/${sub}_${session}_${task}_run-${run}_desc-mcparams_motion.json
{
    "X": "movement in X direction",
    "Y": "movement in Y direction",
    "Z": "movement in Z direction",
    "RotX": "rotation in x direction",
    "RotY": "rotation in y direction",
    "RotZ": "rotation in z direction"
}
EOT
    done
done

for sub in $subs; do
    for i in $(find ${sub}/${session}/eyetrack/*.tsv); do
        mv $i $(echo $i | sed -e "s/events/desc-remodnav_events/")
    done
done

# create a .bidsignore file to hopefully get this directory bids compliant
cat <<EOT > .bidsignore
inputs/
workdir/
eyetrack/
CHANGELOG.md
README.md
model-frst_smdl.json
EOT



# create some accompanying .json files. For most of them I am unsure what they
# should contain, so I'm creating them just for the sake of having the filename.

# dataset_description.json file at root level:
cat <<EOT > dataset_description.json
{
    "Name" : "BIDS_sacc"
    "BIDSVersion" : "1.1.1"
    "PipelineDescription.Name": "alignment and nipype preproc",
    "PipelineDescription.Version": "studyforrest phase 2 aligned + nipype 1.1.3",
    "SourceDatasets": "url=medusa.ovgu.de:/home/data/psyinf/scratch/studyforrest-eyemovementlabels, url=medusa.ovgu.de:/home/data/psyinf/forrest_gump/collection/aligned"
}
EOT

# .json files for nipype preprocessing derivatives
for sub in $subs; do
    for run in $(seq 1 8); do
        cat <<EOT >${sub}/${session}/func/${sub}_${session}_${task}_run-${run}_desc-highpass.json
{
"Hiphpass-filter" : "50.0"
}

EOT
    done
done

for sub in $subs; do
    for run in $(seq 1 8); do
        cat <<EOT >${sub}/${session}/func/${sub}_${session}_${task}_run-${run}_type-brain_mask.json
{
"Type" : "brain"
}

EOT
    done
done

for sub in $subs; do
    for run in $(seq 1 8); do
        cat <<EOT >${sub}/${session}/func/${sub}_${session}_${task}_run-${run}_desc-smooth.json
{
"fwhm" : "5.0mm"
}

EOT
    done
done

for sub in $subs; do
    for run in $(seq 1 8); do
        cat <<EOT >${sub}/${session}/func/${sub}_${session}_${task}_run-${run}_desc-mean.json
{
"mean functional image" : ""
}

EOT
    done
done

