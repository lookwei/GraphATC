#!/bin/bash
cd ./graphatc/
python run.py 0 5311 \
    --EXP_NAME=GraphATC_ATC-GRAPH \
    --MODEL=DGCN \
    --TRAIN_METHOD=Jackknife \
    --AUTHOR=tian \
    --LEVEL=2 \
    --NUM_LAYERS=15 \
    --USE_BIDIRECTIONAL_RNN=True \
    --SPLIT_COMPONENT=True \
    --POLYMER_METHOD=2 \
    --BATCH_SIZE=256 \
    --LR=1e-3 \
    --EPOCHS=600 \
    --WRITER \
    --OSS False \
    --SAVE_MODEL \
    --GPU=0 \
