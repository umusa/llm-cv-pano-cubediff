#!/usr/bin/env bash
# Export PYTHONPATH to include the parent directory
export PYTHONPATH=$(dirname $(realpath $0))/../..
time python ../data/polyhaven/build_tiny_set.py --out ../data/dataspace/polyhaven_tiny --n 700