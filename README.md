# BIDSsacc
Transitional repository created in the middle of a fitlin/BIDS endeavour - Nothing much to see here yet. 

This repository contains a python script to execute a nipype preprocessing workflow.
Before executing the preprocessing workflow, the dataset has to be created:


```
datalad create BIDS_sacc
cd BIDS_sacc
datalad run-procedure setup_yoda_dataset
mkdir inputs

#install some not-BIDS conform datasets
datalad install -d . -s adina@medusa.ovgu.de:/home/data/psyinf/forrest_gump/collection/aligned/ inputs/aligned
datalad install -d . -s adina@medusa.ovgu.de:/home/data/psyinf/scratch/studyforrest-data-eyemovementlabels inputs/eyemovementlabels
datalad install -d . -s git@github.com:psychoinformatics-de/studyforrest-data-templatetransforms.git inputs/tnt
datalad get inputs/aligned/sub*/in_bold3Tp2/sub-*task-avmovie_run-*_bold*
datalad get inputs/eyemovementlabels/sub*/sub*task-movie_run-*_events.tsv
datalad get inputs/tnt/sub*/bold3Tp2/brain*
datalad get inputs/tnt/sub*/bold3Tp2/head*
datalad get inputs/tnt/sub*/bold3Tp2/in_grpbold3Tp2/*

#copy some custom annotation confounds
scp -r adina@medusa.ovgu.de:~/pd/adina/direction_saccades_glm/inputs/annotation_confounds inputs/annotation_confounds
#copy some descripitve files from studyforrest phase 2
scp adina@medusa.ovgu.de:/home/data/psyinf/forrest_gump/collection/phase2/task-movie_bold.json task-avmovie_bold.json
scp adina@medusa.ovgu.de:/home/data/psyinf/forrest_gump/collection/phase2/participants.tsv .
```

The dataset should become BIDS conform at some point. This involves a fair amount of restructuring and renaming, for which a
shell script is provided:

In the root of the directory, execute
```
code/restructure2bids.sh
```

In order to be have BIDS conform mcparameter files, the script motion_tsv.py adds a header to the motion parameter files. 

```
code/motion_tsv.py
```

To finally execute the nipype workflow, run

```
 code/nipype_v2.py --experiment_dir='.' --work_dir='workdir' --conf_path='ses-movie/func' --eyemovement_dir='eyetrack' --subject_id='sub-01' --run_id='run-1'
```
for preprocessing one subject, one run. Without the subject_id and run_id label, all 15 subjects across all 8 runs are preprocessed.
