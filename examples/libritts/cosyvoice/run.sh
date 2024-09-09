#!/bin/bash
# Copyright 2024 Alibaba Inc. All Rights Reserved.
. ./path.sh || exit 1;

stage=-1
stop_stage=3

# data_url=www.openslr.org/resources/60
# data_dir=/mnt/lyuxiang.lx/data/tts/openslr/libritts
pretrained_model_dir=../../../pretrained_models/CosyVoice-300M

data_dir=/root/code/CosyVoice/examples/libritts/cosyvoice/original_data

# # 阶段 -1: 数据下载
# if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
#   echo "数据下载"
#   for part in dev-clean test-clean dev-other test-other train-clean-100 train-clean-360 train-other-500; do
#     local/download_and_untar.sh ${data_dir} ${data_url} ${part}
#   done
# fi

# # 阶段 0: 数据准备
# if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
#   echo "数据准备，准备 wav.scp/text/utt2spk/spk2utt"
#   for x in train test; do
#     mkdir -p data/$x
#     python local/prepare_data.py --src_dir $data_dir/$x --des_dir data/$x
#   done
# fi

# # 阶段 1: 提取 campplus 说话人嵌入
# if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
#   echo "提取 campplus 说话人嵌入，你将在 data/$x 目录下获得 spk2embedding.pt 和 utt2embedding.pt"
#   for x in train test; do
#     tools/extract_embedding.py --dir data/$x \
#       --onnx_path $pretrained_model_dir/campplus.onnx
#   done
# fi

# # 阶段 2: 提取离散语音标记
# if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
#   echo "提取离散语音标记，你将在 data/$x 目录下获得 utt2speech_token.pt"
#   for x in train test; do
#     tools/extract_speech_token.py --dir data/$x \
#       --onnx_path $pretrained_model_dir/speech_tokenizer_v1.onnx
#   done
# fi

# # 阶段 3: 准备所需的 parquet 格式数据
# if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
#   echo "准备所需的 parquet 格式数据，你应该已经准备好 wav.scp/text/utt2spk/spk2utt/utt2embedding.pt/spk2embedding.pt/utt2speech_token.pt"
#   for x in train test; do
#     mkdir -p data/$x/parquet
#     tools/make_parquet_list.py --num_utts_per_parquet 1000 \
#       --num_processes 10 \
#       --src_dir data/$x \
#       --des_dir data/$x/parquet
#   done
# fi

# # 阶段 4: 运行推理
# if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 3 ]; then
#   echo "运行推理。请确保 tts_text 中的 utt 在 prompt_data 中"
#   for mode in sft zero_shot; do
#     python cosyvoice/bin/inference.py --mode $mode \
#       --gpu 0 \
#       --config conf/cosyvoice.yaml \
#       --prompt_data data/train/parquet/data.list \
#       --prompt_utt2data data/train/parquet/utt2data.list \
#       --tts_text `pwd`/tts_text.json \
#       --llm_model $pretrained_model_dir/llm.pt \
#       --flow_model $pretrained_model_dir/flow.pt \
#       --hifigan_model $pretrained_model_dir/hift.pt \
#       --result_dir `pwd`/exp/cosyvoice/train/$mode
#   done
# fi

# 阶段 5: 训练 LLM
export CUDA_VISIBLE_DEVICES="0"
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
job_id=1986
dist_backend="nccl"
num_workers=1
prefetch=1
train_engine=torch_ddp
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 3 ]; then
  echo "运行训练。我们目前仅支持 LLM 训练。如果你想从头开始训练，请使用 conf/cosyvoice.fromscratch.yaml"
  if [ $train_engine == 'deepspeed' ]; then
    echo "注意 deepspeed 有自己的优化器配置。如有必要，请修改 conf/ds_stage2.json"
  fi
  cat data/{train-clean-100,train-clean-360,train-other-500}/parquet/data.list > data/train.data.list
  cat data/{dev-clean,dev-other}/parquet/data.list > data/dev.data.list
  for model in llm flow; do
    torchrun --nnodes=1 --nproc_per_node=$num_gpus \
        --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint="localhost:0" \
      cosyvoice/bin/train.py \
      --train_engine $train_engine \
      --config conf/cosyvoice.yaml \
      --train_data data/train.data.list \
      --cv_data data/dev.data.list \
      --model $model \
      --checkpoint $pretrained_model_dir/$model.pt \
      --model_dir `pwd`/exp/cosyvoice/$model/$train_engine \
      --tensorboard_dir `pwd`/tensorboard/cosyvoice/$model/$train_engine \
      --ddp.dist_backend $dist_backend \
      --num_workers ${num_workers} \
      --prefetch ${prefetch} \
      --pin_memory \
      --deepspeed_config ./conf/ds_stage2.json \
      --deepspeed.save_states model+optimizer
  done
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  echo "Export your model for inference speedup. Remember copy your llm or flow model to model_dir"
  python cosyvoice/bin/export_jit.py --model_dir $pretrained_model_dir
  python cosyvoice/bin/export_onnx.py --model_dir $pretrained_model_dir
fi