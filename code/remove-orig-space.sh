#!/bin/bash
# Apparently there is a bug that masks with various spaces are picked up and
# some selected randomly instead of the correct space. So we just remove them
# for now

set -eu
git rm sub-*/*/*/*space-orig* 
