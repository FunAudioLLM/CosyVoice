#!/usr/bin/env bash

# pip install pesq
# pip install pystoi
# pip install pyworld
# pip install pysptk
# pip install -U numpy
stage=1
stop_stage=2

#ref_dir=$1
#gen_dir=$2

ref_dir='your test folder'
gen_dir='the genereated samples'
echo ${ref_dir}
echo ${gen_dir}



if [ $stage -le 1 ] && [ "${stop_stage}" -ge 2 ];then
  echo "Compute PESQ"
  python metrics/compute_pesq.py \
    -r ${ref_dir} \
    -d ${gen_dir}
fi

if [ $stage -le 2 ] && [ "${stop_stage}" -ge 3 ];then
  echo "Compute STOI"
  python metrics/compute_stoi.py \
    -r ${ref_dir} \
    -d ${gen_dir}
fi
