#!/usr/bin/env bash
set -euo pipefail
export PATH=/nas/liziyan103/anaconda3/envs/seatrack/bin:$PATH
export PYTHONPATH=/nas/liziyan103/seatrack/SEATrack
export CUDA_VISIBLE_DEVICES=4
export CONDA_NO_PLUGINS=true
export PYTHONUNBUFFERED=1
cd /nas/liziyan103/seatrack/SEATrack/Depthtrack_workspace
exec > >(tee -a /nas/liziyan103/seatrack/eval_depthtrack.log) 2>&1
date -Iseconds
vot evaluate --workspace ./ rgbd
python /nas/liziyan103/seatrack/check_depthtrack_results.py
vot analysis --workspace ./ --nocache --workers 1 --format html --name depthtrack_reproduction rgbd
vot analysis --workspace ./ --workers 1 --format json --name depthtrack_reproduction rgbd
date -Iseconds
touch /nas/liziyan103/seatrack/depthtrack_eval_complete
