#!/bin/bash
# Copyright (c) 2025 NVIDIA (authors: Yuekai Zhang)
export CUDA_VISIBLE_DEVICES=0
cosyvoice_path=/workspace/CosyVoice
export PYTHONPATH=${cosyvoice_path}:$PYTHONPATH
export PYTHONPATH=${cosyvoice_path}/third_party/Matcha-TTS:$PYTHONPATH
stage=$1
stop_stage=$2

huggingface_model_local_dir=./cosyvoice2_llm
model_scope_model_local_dir=./CosyVoice2-0.5B
trt_dtype=bfloat16
trt_weights_dir=./trt_weights_${trt_dtype}
trt_engines_dir=./trt_engines_${trt_dtype}

model_repo=./model_repo_cosyvoice2

if [ $stage -le -1 ] && [ $stop_stage -ge -1 ]; then
    echo "Cloning CosyVoice"
    git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git $cosyvoice_path
    cd $cosyvoice_path
    git submodule update --init --recursive
    cd runtime/triton_trtllm
fi

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    echo "Downloading CosyVoice2-0.5B"
    huggingface-cli download --local-dir $huggingface_model_local_dir yuekai/cosyvoice2_llm
    modelscope download --model iic/CosyVoice2-0.5B --local_dir $model_scope_model_local_dir
fi


if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    echo "Converting checkpoint to TensorRT weights"
    python3 scripts/convert_checkpoint.py --model_dir $huggingface_model_local_dir \
                                --output_dir $trt_weights_dir \
                                --dtype $trt_dtype || exit 1

    echo "Building TensorRT engines"
    trtllm-build --checkpoint_dir $trt_weights_dir \
                --output_dir $trt_engines_dir \
                --max_batch_size 16 \
                --max_num_tokens 32768 \
                --gemm_plugin $trt_dtype || exit 1

    echo "Testing TensorRT engines"
    python3 ./scripts/test_llm.py --input_text "你好，请问你叫什么？" \
                    --tokenizer_dir $huggingface_model_local_dir \
                    --top_k 50 --top_p 0.95 --temperature 0.8 \
                    --engine_dir=$trt_engines_dir  || exit 1
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    echo "Creating model repository"
    rm -rf $model_repo
    mkdir -p $model_repo
    cosyvoice2_dir="cosyvoice2"

    cp -r ./model_repo/${cosyvoice2_dir} $model_repo
    cp -r ./model_repo/audio_tokenizer $model_repo
    cp -r ./model_repo/tensorrt_llm $model_repo
    cp -r ./model_repo/token2wav $model_repo

    ENGINE_PATH=$trt_engines_dir
    MAX_QUEUE_DELAY_MICROSECONDS=0
    MODEL_DIR=$model_scope_model_local_dir
    LLM_TOKENIZER_DIR=$huggingface_model_local_dir
    BLS_INSTANCE_NUM=4
    TRITON_MAX_BATCH_SIZE=16
    DECOUPLED_MODE=False

    python3 scripts/fill_template.py -i ${model_repo}/token2wav/config.pbtxt model_dir:${MODEL_DIR},triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},max_queue_delay_microseconds:${MAX_QUEUE_DELAY_MICROSECONDS}
    python3 scripts/fill_template.py -i ${model_repo}/audio_tokenizer/config.pbtxt model_dir:${MODEL_DIR},triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},max_queue_delay_microseconds:${MAX_QUEUE_DELAY_MICROSECONDS}
    python3 scripts/fill_template.py -i ${model_repo}/${cosyvoice2_dir}/config.pbtxt model_dir:${MODEL_DIR},bls_instance_num:${BLS_INSTANCE_NUM},llm_tokenizer_dir:${LLM_TOKENIZER_DIR},triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},decoupled_mode:${DECOUPLED_MODE},max_queue_delay_microseconds:${MAX_QUEUE_DELAY_MICROSECONDS}
    python3 scripts/fill_template.py -i ${model_repo}/tensorrt_llm/config.pbtxt triton_backend:tensorrtllm,triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},decoupled_mode:${DECOUPLED_MODE},max_beam_width:1,engine_dir:${ENGINE_PATH},max_tokens_in_paged_kv_cache:2560,max_attention_window_size:2560,kv_cache_free_gpu_mem_fraction:0.5,exclude_input_in_output:True,enable_kv_cache_reuse:False,batching_strategy:inflight_fused_batching,max_queue_delay_microseconds:${MAX_QUEUE_DELAY_MICROSECONDS},encoder_input_features_data_type:TYPE_FP16,logits_datatype:TYPE_FP32

fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
   echo "Starting Triton server"
   tritonserver --model-repository $model_repo
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
    echo "Single request test http"
    python3 client_http.py \
        --reference-audio ./assets/prompt_audio.wav \
        --reference-text "吃燕窝就选燕之屋，本节目由26年专注高品质燕窝的燕之屋冠名播出。豆奶牛奶换着喝，营养更均衡，本节目由豆本豆豆奶特约播出。" \
        --target-text "身临其境，换新体验。塑造开源语音合成新范式，让智能语音更自然。" \
        --model-name cosyvoice2
fi

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
    echo "Running benchmark client grpc"
    num_task=4
    # set mode=streaming, when decoupled=True
    # set mode=offline, when decoupled=False
    mode=offline
    python3 client_grpc.py \
        --server-addr localhost \
        --model-name cosyvoice2 \
        --num-tasks $num_task \
        --mode $mode \
        --huggingface-dataset yuekai/seed_tts_cosy2 \
        --log-dir ./log_concurrent_tasks_${num_task}_${mode}_bls_4_${trt_dtype}
fi