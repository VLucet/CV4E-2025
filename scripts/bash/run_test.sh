#!/bin/bash

export PYTHONPATH=.
export DATASTORE='/mnt/class_data/group4/val/trailcam/'

# python3 scripts/python/0_clean_data.py
# python3 scripts/python/1_explore_data.py
# python3 scripts/python/2_split_data.py
python3 scripts/python/3_load_and_run.py