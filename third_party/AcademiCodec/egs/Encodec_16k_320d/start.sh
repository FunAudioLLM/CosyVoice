#!/bin/bash
source path.sh
log_root=logs
# 16kHz *.wav in train_data_dir
train_data_dir=dump/train
valid_data_dir=dump/valid

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python3 -m torch.distributed.launch --nproc_per_node 8 ${BIN_DIR}/main_launch.py \
        --BATCH_SIZE 16 \
        --N_EPOCHS 300 \
        --save_dir ${log_root} \
        --PATH ${log_root} \
        --train_data_path ${train_data_dir} \
        --valid_data_path ${valid_data_dir} \
        --sr 16000 \
        --ratios 8 5 4 2 \
        --target_bandwidths 1 1.5 2 4 6 12
