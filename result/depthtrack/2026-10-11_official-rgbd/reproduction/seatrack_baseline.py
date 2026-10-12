import os
import sys
env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, env_path)
from lib.test.vot.seatrack_class import run_vot_exp
# Inherit CUDA_VISIBLE_DEVICES from the evaluation launcher.


run_vot_exp('seatrack', 'rgbd', vis=False, out_conf=True, channel_type='rgbd')
