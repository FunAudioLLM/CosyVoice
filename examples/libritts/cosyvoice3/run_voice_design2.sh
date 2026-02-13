#!/bin/bash
# Copyright (c) 2025 Voice Design Extension
# Simplified Voice Design Training Pipeline for CosyVoice3
# (Skips Stage 4/5: Description Encoder Training)

. ./path.sh || exit 1;

stage=0
stop_stage=6

# Parse arguments
if [ $# -ge 2 ]; then
    stage=$1
    stop_stage=$2
fi

# Paths
data_dir=/mnt/data/train
pretrained_model_dir=$(realpath ../../../pretrained_models/Fun-CosyVoice3-0.5B)
description_file="/mnt/data/train/voice_design_final.jsonl"

# Fine-tuning settings
FINETUNE_MODELS=true

# GPU settings
export CUDA_VISIBLE_DEVICES="0"
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
job_id=1987
dist_backend="nccl"
num_workers=4
prefetch=100
train_engine=torch_ddp

# Helper to find latest checkpoint
get_latest_checkpoint() {
    local dir=$1
    local default=$2
    if [ -d "$dir" ]; then
        local latest=$(ls "$(realpath "$dir")"/epoch_*_whole.pt 2>/dev/null | sort -V | tail -n 1)
        if [ -n "$latest" ]; then
            echo "$latest"
            return
        fi
    fi
    echo "$default"
}

# STAGE 0: Data Preparation
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "Stage 0: Data preparation"
    mkdir -p data/train
    python local/prepare_data_voice_design.py \
        --src_dir $data_dir/azure_youtube_data \
        --des_dir data/train \
        --with_description \
        --description_file $description_file \
        --instruct "You are a helpful assistant.<|endofprompt|>"
fi

# STAGE 1: Extract Style Embeddings
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "Stage 1: Extract style embeddings from audio"
    for x in train; do
        python ../../../tools/extract_style_embedding.py \
            --dir data/$x \
            --model_path $pretrained_model_dir/campplus.onnx \
            --output_file utt2style_embedding.pt
    done
fi

# STAGE 2: Extract Speech Tokens
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "Stage 2: Extract speech tokens from audio"
    for x in train; do
        python ../../../tools/extract_speech_token.py \
            --dir data/$x \
            --onnx_path $pretrained_model_dir/speech_tokenizer_v3.onnx \
            --num_thread $num_workers
    done
fi

# STAGE 3: Parquet Creation
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "Stage 3: Create parquet files"
    for x in train; do
        mkdir -p data/$x/parquet
        python ../../../tools/make_parquet_list_voice_design.py \
            --num_utts_per_parquet 1000 \
            --num_processes 10 \
            --src_dir data/$x \
            --des_dir data/$x/parquet \
            --include_description \
            --include_style_embedding
    done
fi

# STAGE 4: Data List Preparation
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "Stage 4: Preparing data lists"
    train_parquet_dir=$(realpath data/train/parquet)
    find $train_parquet_dir -name "*.parquet" > data/train.data.list
    
    # Dev set (use 5% for validation if not exists)
    total_files=$(cat data/train.data.list | wc -l)
    val_files=$((total_files / 20))
    if [ $val_files -lt 1 ]; then val_files=1; fi
    shuf data/train.data.list > data/all_parquet.list.tmp
    head -n $((total_files - val_files)) data/all_parquet.list.tmp > data/train.data.list
    tail -n $val_files data/all_parquet.list.tmp > data/dev.data.list
    rm data/all_parquet.list.tmp
    echo "Generated data lists: $(wc -l < data/train.data.list) train, $(wc -l < data/dev.data.list) dev"
fi

# STAGE 5: Fine-tune LLM
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "Stage 5: Fine-tune LLM"
    if [ "$FINETUNE_MODELS" = "true" ]; then
        model_dir="exp/llm_sft/$train_engine"
        checkpoint=$(get_latest_checkpoint "$model_dir" "$pretrained_model_dir/llm.pt")
        echo "Using checkpoint: $checkpoint"

        cd ../../../
        torchrun --nnodes=1 --nproc_per_node="$num_gpus" \
            --rdzv_id="$job_id" --rdzv_backend="c10d" --rdzv_endpoint="localhost:1234" \
            cosyvoice/bin/train.py \
            --train_engine "$train_engine" \
            --config examples/libritts/cosyvoice3/conf/cosyvoice3_sft.yaml \
            --train_data examples/libritts/cosyvoice3/data/train.data.list \
            --cv_data examples/libritts/cosyvoice3/data/dev.data.list \
            --qwen_pretrain_path "$pretrained_model_dir/CosyVoice-BlankEN" \
            --onnx_path "$pretrained_model_dir" \
            --model llm \
            --checkpoint "$checkpoint" \
            --model_dir "$(pwd)/examples/libritts/cosyvoice3/exp/llm_sft/$train_engine" \
            --tensorboard_dir "$(pwd)/examples/libritts/cosyvoice3/tensorboard/llm_sft/$train_engine" \
            --ddp.dist_backend "$dist_backend" \
            --num_workers "$num_workers" \
            --prefetch "$prefetch" \
            --pin_memory \
            --use_amp \
            --deepspeed_config examples/libritts/cosyvoice3/conf/ds_stage2.json \
            --deepspeed.save_states model+optimizer
        cd examples/libritts/cosyvoice3
    fi
fi

# STAGE 6: Fine-tune Flow
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    echo "Stage 6: Fine-tune Flow"
    if [ "$FINETUNE_MODELS" = "true" ]; then
        model_dir="exp/flow_sft/$train_engine"
        checkpoint=$(get_latest_checkpoint "$model_dir" "$pretrained_model_dir/flow.pt")
        echo "Using checkpoint: $checkpoint"

        cd ../../../
        torchrun --nnodes=1 --nproc_per_node="$num_gpus" \
            --rdzv_id="$job_id" --rdzv_backend="c10d" --rdzv_endpoint="localhost:1235" \
            cosyvoice/bin/train.py \
            --train_engine "$train_engine" \
            --config examples/libritts/cosyvoice3/conf/cosyvoice3_sft.yaml \
            --train_data examples/libritts/cosyvoice3/data/train.data.list \
            --cv_data examples/libritts/cosyvoice3/data/dev.data.list \
            --model flow \
            --checkpoint "$checkpoint" \
            --model_dir "$(pwd)/examples/libritts/cosyvoice3/exp/flow_sft/$train_engine" \
            --qwen_pretrain_path "$pretrained_model_dir/CosyVoice-BlankEN" \
            --onnx_path "$pretrained_model_dir" \
            --tensorboard_dir "$(pwd)/examples/libritts/cosyvoice3/tensorboard/flow_sft/$train_engine" \
            --ddp.dist_backend "$dist_backend" \
            --num_workers "$num_workers" \
            --prefetch "$prefetch" \
            --pin_memory \
            --use_amp \
            --deepspeed_config examples/libritts/cosyvoice3/conf/ds_stage2.json \
            --deepspeed.save_states model+optimizer
        cd examples/libritts/cosyvoice3
    fi
fi

# STAGE 7: Model Averaging
average_num=3
if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    echo "Stage 7: Model averaging"
    for model in llm_sft flow_sft; do
        src_path="$(pwd)/exp/$model/$train_engine"
        dst_model="${src_path}/$(echo $model | cut -d'_' -f1).pt"
        if [ -d "$src_path" ]; then
            echo "Averaging $model model to $dst_model"
            python ../../../cosyvoice/bin/average_model.py \
                --dst_model $dst_model \
                --src_path $src_path \
                --num ${average_num} \
                --val_best
        fi
    done
fi
