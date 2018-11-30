#!/bin/sh

set -e

#Generate singularity image

sudo docker run --rm kaczmarj/neurodocker:master generate singularity \
--base neurodebian:stretch-non-free \
--pkg-manager apt \
--install fsl-5.0-core fsl-mni152-templates python-mvpa2 python-nipype python-matplotlib python-sklearn ipython \
--add-to-entrypoint "source /etc/fsl/5.0/fsl.sh" \
--run-bash "mkdir -p /backup/data/movieloc/saccs /data/movieloc/backup_store/saccs" \
> envs/Singularity.1.1
