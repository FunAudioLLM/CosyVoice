#!/bin/bash
source path.sh

python3 ${BIN_DIR}/test.py \
       --input=./test_wav \
       --output=./output \
       --resume_path=checkpoint/encodec_16k_320d.pth \
       --sr=16000 \
       --ratios 8 5 4 2 \
       --target_bandwidths 1 1.5 2 4 6 12 \
       --target_bw=12 \
       -r
