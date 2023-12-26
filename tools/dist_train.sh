#!/usr/bin/env bash
set -x

timestamp=`date +"%y%m%d.%H%M%S"`

WORK_DIR=work_dirs/lanesegnet
CONFIG=projects/configs/lanesegnet_r50_8x1_24e_olv2_subset_A.py

GPUS=$1
PORT=${PORT:-28510}

python -m torch.distributed.run --nproc_per_node=$GPUS --master_port=$PORT \
    tools/train.py $CONFIG --launcher pytorch --work-dir ${WORK_DIR} ${@:2} \
    2>&1 | tee ${WORK_DIR}/train.${timestamp}.log
