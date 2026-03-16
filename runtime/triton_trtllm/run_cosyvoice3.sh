#!/bin/bash
# Copyright (c) 2026 NVIDIA (authors: Yuekai Zhang)
export CUDA_VISIBLE_DEVICES=0
cosyvoice_path=/workspace/CosyVoice

export PYTHONPATH=${cosyvoice_path}:$PYTHONPATH
export PYTHONPATH=${cosyvoice_path}/third_party/Matcha-TTS:$PYTHONPATH

stage=$1
stop_stage=$2

huggingface_llm_local_dir=$cosyvoice_path/runtime/triton_trtllm/hf_cosyvoice3_llm
cosyvoice3_official_model_dir=$cosyvoice_path/runtime/triton_trtllm/Fun-CosyVoice3-0.5B-2512

trt_dtype=bfloat16
trt_weights_dir=$cosyvoice_path/runtime/triton_trtllm/trt_weights_${trt_dtype}
trt_engines_dir=$cosyvoice_path/runtime/triton_trtllm/trt_engines_${trt_dtype}

model_repo_src=$cosyvoice_path/runtime/triton_trtllm/model_repo_cosyvoice3
model_repo=$cosyvoice_path/runtime/triton_trtllm/model_repo_cosyvoice3_copy
bls_instance_num=10

if [ $stage -le -1 ] && [ $stop_stage -ge -1 ]; then

    echo "Cloning CosyVoice"
    git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git $cosyvoice_path
    cd $cosyvoice_path
    git submodule update --init --recursive
    cd runtime/triton_trtllm
fi

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    echo "Downloading CosyVoice3 Checkpoints"
    # if s3 tokenizer version is not 0.3.0
    if [ $(pip3 show s3tokenizer | grep -o "0\.2\.[0-9]") != "0.3.0" ]; then
        pip3 install --upgrade x_transformers s3tokenizer
    fi
    huggingface-cli download --local-dir $huggingface_llm_local_dir yuekai/Fun-CosyVoice3-0.5B-2512-LLM-HF
    huggingface-cli download --local-dir $cosyvoice3_official_model_dir yuekai/Fun-CosyVoice3-0.5B-2512-FP16-ONNX
    huggingface-cli download --local-dir $cosyvoice3_official_model_dir FunAudioLLM/Fun-CosyVoice3-0.5B-2512
fi


if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    echo "Converting checkpoint to TensorRT weights"
    python3 scripts/convert_checkpoint.py --model_dir $huggingface_llm_local_dir \
                                --output_dir $trt_weights_dir \
                                --dtype $trt_dtype || exit 1

    echo "Building TensorRT engines"
    trtllm-build --checkpoint_dir $trt_weights_dir \
                --output_dir $trt_engines_dir \
                --max_batch_size 64 \
                --max_num_tokens 32768 \
                --gemm_plugin $trt_dtype || exit 1

    echo "Testing TensorRT engines"
    python3 ./scripts/test_llm.py --input_text "你好，请问你叫什么？" \
                    --tokenizer_dir $huggingface_llm_local_dir \
                    --top_k 50 --top_p 0.95 --temperature 0.8 \
                    --engine_dir=$trt_engines_dir  || exit 1
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    echo "Creating CosyVoice3 model repository"
    rm -rf $model_repo
    mkdir -p $model_repo

    # Copy all modules from template source
    cp -r ${model_repo_src}/cosyvoice3 $model_repo/
    cp -r ${model_repo_src}/token2wav $model_repo/
    cp -r ${model_repo_src}/vocoder $model_repo/
    cp -r ${model_repo_src}/audio_tokenizer $model_repo/
    cp -r ${model_repo_src}/speaker_embedding $model_repo/

    MAX_QUEUE_DELAY_MICROSECONDS=0
    MODEL_DIR=$cosyvoice3_official_model_dir
    LLM_TOKENIZER_DIR=$huggingface_llm_local_dir
    BLS_INSTANCE_NUM=$bls_instance_num
    TRITON_MAX_BATCH_SIZE=1
    DECOUPLED_MODE=True # False for offline TTS

    python3 scripts/fill_template.py -i ${model_repo}/cosyvoice3/config.pbtxt model_dir:${MODEL_DIR},bls_instance_num:${BLS_INSTANCE_NUM},llm_tokenizer_dir:${LLM_TOKENIZER_DIR},triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},decoupled_mode:${DECOUPLED_MODE},max_queue_delay_microseconds:${MAX_QUEUE_DELAY_MICROSECONDS}
    python3 scripts/fill_template.py -i ${model_repo}/token2wav/config.pbtxt model_dir:${MODEL_DIR},triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},max_queue_delay_microseconds:${MAX_QUEUE_DELAY_MICROSECONDS}
    python3 scripts/fill_template.py -i ${model_repo}/vocoder/config.pbtxt model_dir:${MODEL_DIR},triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},max_queue_delay_microseconds:${MAX_QUEUE_DELAY_MICROSECONDS}
    python3 scripts/fill_template.py -i ${model_repo}/audio_tokenizer/config.pbtxt model_dir:${MODEL_DIR},triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},max_queue_delay_microseconds:${MAX_QUEUE_DELAY_MICROSECONDS}
    python3 scripts/fill_template.py -i ${model_repo}/speaker_embedding/config.pbtxt model_dir:${MODEL_DIR},triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},max_queue_delay_microseconds:${MAX_QUEUE_DELAY_MICROSECONDS}

fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
   echo "Starting CosyVoice3 Triton server and LLM using trtllm-serve"
   CUDA_VISIBLE_DEVICES=0 mpirun -np 1 --allow-run-as-root --oversubscribe trtllm-serve serve --tokenizer $huggingface_llm_local_dir $trt_engines_dir --max_batch_size 64  --kv_cache_free_gpu_memory_fraction 0.4 &
   CUDA_VISIBLE_DEVICES=0 tritonserver --model-repository $model_repo --http-port 18000 --grpc-port 18001 --metrics-port 18002 &
   wait
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
    echo "Running benchmark client for CosyVoice3"
    num_task=4
    mode=streaming
    BLS_INSTANCE_NUM=$bls_instance_num

    python3 client_grpc.py \
        --server-addr localhost \
        --server-port 18001 \
        --model-name cosyvoice3 \
        --num-tasks $num_task \
        --mode $mode \
        --huggingface-dataset yuekai/seed_tts_cosy2 \
        --log-dir ./log_cosyvoice3_concurrent_tasks_${num_task}_${mode}_bls_${BLS_INSTANCE_NUM}

fi

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
    echo "stage 5: Python script CosyVoice3 TTS (LLM + CosyVoice3 Token2Wav) inference"

    datasets=(wenetspeech4tts) # wenetspeech4tts
    backend=trtllm  # hf, trtllm, vllm, trtllm-serve

    batch_sizes=(16 8 4 2 1)
    token2wav_batch_size=1 # Only support 1 for now

    for batch_size in ${batch_sizes[@]}; do
      for dataset in ${datasets[@]}; do
        output_dir=./cosyvoice3_${dataset}_${backend}_llm_batch_size_${batch_size}_token2wav_batch_size_${token2wav_batch_size}_offline_tts_trt
        CUDA_VISIBLE_DEVICES=0 \
            python3 infer_cosyvoice3.py \
                --output-dir $output_dir \
                --llm-model-name-or-path $huggingface_llm_local_dir \
                --token2wav-path $cosyvoice3_official_model_dir \
                --backend $backend \
                --batch-size $batch_size --token2wav-batch-size $token2wav_batch_size \
                --engine-dir $trt_engines_dir \
                --enable-trt \
                --epoch 3 \
                --split-name ${dataset} || exit 1
      done
    done
fi