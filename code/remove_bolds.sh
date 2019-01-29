#!/bin/bash
# Remove bold files in subject space that follow BIDS naming convention

set -eu
git rm sub-*/*/*/*run-{1..3}_bold.nii.gz
