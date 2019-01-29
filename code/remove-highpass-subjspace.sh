#!/bin/bash

## We said "Fuck it all" and will remove everything that the model does not
## directly need from the directory.
## for now only two runs, later this needs to be {1 .. 8}

set -eu

git rm sub-*/*/*/*run-{1..3}_desc-highpass_bold*
