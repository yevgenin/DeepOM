#!/bin/bash

###
# CS236781: Deep Learning
# jupyter-lab.sh
#
# This script is intended to help you run jupyter lab on the course servers.
#
# Example usage:
#
# 1. First, activate the relevant conda env
# $ conda activate cs236781-hwN
#
# 2. Run this script on a compute node:
# $ srun -c 2 --gres=gpu:1 --pty jupyter-lab.sh
#


unset XDG_RUNTIME_DIR

xvfb-run -a -s "-screen 0 1440x900x24" jupyter lab --no-browser --ip=$(hostname -I) --port-retries=100
