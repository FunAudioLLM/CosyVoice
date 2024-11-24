#!/bin/bash
source path.sh

ckpt=checkpoint/HiFi-Codec-24k-320d
echo checkpoint path: ${ckpt}

# the path of test wave
wav_dir=test_wav

outputdir=output
mkdir -p ${outputdir}

python3 ${BIN_DIR}/vqvae_copy_syn.py \
    --model_path=${ckpt} \
    --config_path=config_24k_320d.json \
    --input_wavdir=${wav_dir} \
    --outputdir=${outputdir} \
    --num_gens=10000
