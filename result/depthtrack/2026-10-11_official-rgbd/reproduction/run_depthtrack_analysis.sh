#!/usr/bin/env bash
set -euo pipefail
export PATH=/nas/liziyan103/anaconda3/envs/seatrack/bin:$PATH
export PYTHONPATH=/nas/liziyan103/seatrack/SEATrack
export PYTHONUNBUFFERED=1
cd /nas/liziyan103/seatrack/SEATrack/Depthtrack_workspace
exec > >(tee -a /nas/liziyan103/seatrack/analysis_depthtrack.log) 2>&1
date -Iseconds
python /nas/liziyan103/seatrack/check_depthtrack_results.py
vot analysis --workspace ./ --nocache --workers 1 --format html --name depthtrack_reproduction rgbd
vot analysis --workspace ./ --nocache --workers 1 --format json --name depthtrack_reproduction rgbd
date -Iseconds
