#!/bin/bash

# export PYTHONPATH=.

# python mdclassifier/train.py --config configs/config.yaml

# wandb sweep --name bigsweep --project cv4e-sweep configs/sweep.yaml

python scripts/wandb/sweep.py --config configs/sweep.yaml