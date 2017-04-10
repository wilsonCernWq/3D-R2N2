#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

# debug
SAVE_FREQ=40
VALIDATION_FREQ=60

NET_NAME=GANNet
DATASET='./experiments/dataset/shapenet_unit_test.json'
EXP_DETAIL=default_model
OUT_PATH='./output/'$NET_NAME/$EXP_DETAIL
LOG="$OUT_PATH/log.`date +'%Y-%m-%d_%H-%M-%S'`"

# Make the dir if it not there
mkdir -p $OUT_PATH
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

# export THEANO_FLAGS="floatX=float32,device=gpu,assert_no_cpu_op='raise'"

# python main.py \
#       --batch-size 24 \
#       --iter 100 \
#       --out $OUT_PATH \
#       --model $NET_NAME \
#       --dataset $DATASET \
#       --save-freq $SAVE_FREQ \
#       --valid-freq $VALIDATION_FREQ \
#       ${*:1}

python main.py \
      --test \
      --batch-size 1 \
      --out $OUT_PATH \
      --weights $OUT_PATH/weights.npy \
      --model $NET_NAME \
      --dataset $DATASET \
      --save-freq $SAVE_FREQ \
      --valid-freq $VALIDATION_FREQ \
      ${*:1}
