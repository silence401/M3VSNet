#!/usr/bin/env bash
#MVS_TRAINING="/home/silence401/cascade-stereo-master1/CasMVSNet/pcltest"
MVS_TRAINING="/home/silence401/下载/dataset/mvsnet/eth3d"
python train.py --dataset=dtu_yao --batch_size=1 --trainpath=$MVS_TRAINING --trainlist lists/dtu/train.txt --testlist lists/dtu/test.txt --numdepth=192 --logdir ./checkpoints/pcld192 $@
