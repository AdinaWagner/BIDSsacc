#!/bin/bash
# Remove bold files in subject space that follow BIDS naming convention

set -eu
git rm sub-*/*/*/*run-{1..8}_bold.nii.gz
