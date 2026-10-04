# RGB-T
CUDA_VISIBLE_DEVICES=0,1 python tracking/train.py --script seatrack --config rgbt --save_dir ./models --mode multiple

# RGB-D
python tracking/train.py --script seatrack --config rgbd --save_dir ./models --mode multiple

# RGB-E
python tracking/train.py --script seatrack --config rgbe --save_dir ./models --mode multiple