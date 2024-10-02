#!/bin/bash
# Copyright 2024 Alibaba Inc. All Rights Reserved.
. ./path.sh || exit 1;

stage=5
stop_stage=5

raw_data_dir=/data/tts
output_raw_data_dir=data
pretrained_model_dir=../../../pretrained_models/CosyVoice-300M-25Hz

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "Data preparation, prepare wav.scp/text/utt2spk/spk2utt"
  python local/prepare_data.py $raw_data_dir $output_raw_data_dir
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  for x in train valid; do
    echo "Extract campplus speaker embedding, you will get spk2embedding.pt and utt2embedding.pt in $output_raw_data_dir/$x dir"
    tools/extract_embedding.py --dir $output_raw_data_dir/$x \
      --onnx_path $pretrained_model_dir/campplus.onnx
  done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  for x in train valid; do
    echo "Extract discrete speech token, you will get utt2speech_token.pt in $output_raw_data_dir/$x dir"
    tools/extract_speech_token.py --dir $output_raw_data_dir/$x \
      --onnx_path $pretrained_model_dir/speech_tokenizer_v1.onnx
  done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Prepare required parquet format data, you should have prepared wav.scp/text/utt2spk/spk2utt/utt2embedding.pt/spk2embedding.pt/utt2speech_token.pt"
  for x in train valid; do
    mkdir -p $output_raw_data_dir/$x/parquet
    tools/make_parquet_list.py --num_utts_per_parquet 1000 \
      --num_processes 8 \
      --src_dir $output_raw_data_dir/$x \
      --des_dir $output_raw_data_dir/$x/parquet
  done
fi

# # inference
# if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
#   echo "Run inference. Please make sure utt in tts_text is in prompt_data"
#   for mode in sft zero_shot; do
#     python cosyvoice/bin/inference.py --mode $mode \
#       --gpu 0 \
#       --config conf/cosyvoice.yaml \
#       --prompt_data $output_raw_data_dir/valid/parquet/data.list \
#       --prompt_utt2data $output_raw_data_dir/valid/parquet/utt2data.list \
#       --tts_text `pwd`/tts_text.json \
#       --llm_model $pretrained_model_dir/llm.pt \
#       --flow_model $pretrained_model_dir/flow.pt \
#       --hifigan_model $pretrained_model_dir/hift.pt \
#       --result_dir `pwd`/exp/cosyvoice/valid/$mode
#   done
# fi

# train llm
export CUDA_VISIBLE_DEVICES="0"
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
job_id=1986
dist_backend="nccl"
num_workers=4
prefetch=100
train_engine=torch_ddp
exp_name=ft_25hz_lr1e-5_warmup1k_maxframe2k

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "Run train. We only support llm traning for now. If your want to train from scratch, please use conf/cosyvoice.fromscratch.yaml"
  if [ $train_engine == 'deepspeed' ]; then
    echo "Notice deepspeed has its own optimizer config. Modify conf/ds_stage2.json if necessary"
  fi
  for model in flow; do
  # for model in llm; do
    echo "======================"
    echo "START TRAINING: $model"
    echo "======================"
    torchrun --nnodes=1 --nproc_per_node=$num_gpus \
        --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint="localhost:0" \
        cosyvoice/bin/train.py \
          --train_engine $train_engine \
          --config conf/cosyvoice.yaml \
          --train_data $output_raw_data_dir/train/parquet/data.list \
          --cv_data $output_raw_data_dir/valid/parquet/data.list \
          --model $model \
          --checkpoint $pretrained_model_dir/$model.pt \
          --model_dir `pwd`/exp/$exp_name/$model \
          --tensorboard_dir `pwd`/tensorboard/$exp_name/$model \
          --ddp.dist_backend $dist_backend \
          --num_workers ${num_workers} \
          --prefetch ${prefetch} \
          --pin_memory
          # --deepspeed_config ./conf/ds_stage2.json \
          # --deepspeed.save_states model+optimizer
  done
fi

# if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
#   echo "Export your model for inference speedup. Remember copy your llm or flow model to model_dir"
#   python cosyvoice/bin/export_jit.py --model_dir $pretrained_model_dir
#   python cosyvoice/bin/export_onnx.py --model_dir $pretrained_model_dir
# fi
