# BIDSsacc

Data and code for a GLM to localize the frontal eye fields based on saccadic eyemovements.

This repository contains a python script to execute a nipype preprocessing workflow.
Dataset was build/retrieved with:


```
datalad create BIDS_sacc
cd BIDS_sacc
datalad run-procedure setup_yoda_dataset
mkdir inputs

#install some not-BIDS conform datasets
datalad install -d . -s adina@medusa.ovgu.de:/home/data/psyinf/forrest_gump/collection/aligned/ inputs/aligned
datalad install -d . -s adina@medusa.ovgu.de:/home/data/psyinf/scratch/studyforrest-data-eyemovementlabels inputs/eyemovementlabels
datalad install -d . -s git@github.com:psychoinformatics-de/studyforrest-data-templatetransforms.git inputs/tnt
datalad get sourcedata/aligned/sub*/in_bold3Tp2/sub-*task-avmovie_run-*_bold*
datalad get sourcedata/eyemovementlabels/sub*/sub*task-movie_run-*_events.tsv
datalad get sourcedata/tnt/sub*/bold3Tp2/brain*
datalad get sourcedata/tnt/sub*/bold3Tp2/head*
datalad get sourcedata/tnt/sub*/bold3Tp2/in_grpbold3Tp2/*

#copy some custom annotation confounds
scp -r adina@medusa.ovgu.de:~/pd/adina/direction_saccades_glm/inputs/annotation_confounds inputs/annotation_confounds
#copy some descripitve files from studyforrest phase 2
scp adina@medusa.ovgu.de:/home/data/psyinf/forrest_gump/collection/phase2/task-movie_bold.json task-avmovie_bold.json
scp adina@medusa.ovgu.de:/home/data/psyinf/forrest_gump/collection/phase2/participants.tsv .
```

The dataset should become BIDS conform at some point. This involves a fair amount of restructuring and renaming, for which a
shell script is provided:
[UPDATE: THIS SHELL SCRIPT IS CURRENTLY OUTDATED!]
In the root of the directory, execute
```
code/restructure2bids.sh
```

Several scripts are used to restructure and reshape confound or regressor files into BIDS conformity

```
code/motion_tsv.py
```
Is used to give change input files into properly formatted tsv files with a custom header. 

```
code/regressors_to_physio.py
```

combines dense movie stimulus properties into a single \_physio.tsv.gz file.

The nipype workflow for preprocessing can be recreated by running the following script:

```
 code/nipype_v2.py --experiment_dir='.' --work_dir='workdir' --conf_path='ses-movie/func' --eyemovement_dir='eyetrack' --subject_id='sub-01' --run_id='run-1'
```
for preprocessing one subject, one run. Without the subject_id and run_id label, all 15 subjects across all 8 runs are preprocessed.


The models/ directory contains model.json files necessary to build a fitlins model for the GLM analysis. 

Currently, the repository is very messy. The branch `full` is the branch which will be in an orderly first. Branches `slim` and `slim2` contain a slimmed down version (3 subjects with either 2 or 3 runs) used during model developement.

Note: getting the model to run required some changes to the sourcecode of pybids and fitlins. Many of those changes are already merged into the respective software's master. However, it is currently not guaranteed that the fitlins model here can be put into function immediately.
