#!/bin/bash
source path.sh

python3 ${BIN_DIR}/test.py \
       --input=./test_wav \
       --output=./output \
       --resume_path=checkpoint/encodec_24khz_240d.pth \
       --sr=24000 \
       --ratios 6 5 4 2 \
       --target_bandwidths 1 2 4 8 12 \
       --target_bw=12 \
       -r
       