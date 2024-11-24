#!/bin/bash
source path.sh

python3 ${BIN_DIR}/test.py \
       --input=./test_wav \
       --output=./output \
       --resume_path=checkpoint/Encodec_24khz_32d.pth \
       --sr=24000 \
       --ratios 2 2 2 4 \
       --target_bandwidths 7.5 15 \
       --target_bw=7.5 \
       -r

       