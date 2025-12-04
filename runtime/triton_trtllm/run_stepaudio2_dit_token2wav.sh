#!/bin/bash
# Copyright (c) 2025 NVIDIA (authors: Yuekai Zhang)
export CUDA_VISIBLE_DEVICES=0
cosyvoice_path=/workspace/CosyVoice
stepaudio2_path=/workspace/Step-Audio2

export PYTHONPATH=${stepaudio2_path}:$PYTHONPATH
export PYTHONPATH=${cosyvoice_path}:$PYTHONPATH
export PYTHONPATH=${cosyvoice_path}/third_party/Matcha-TTS:$PYTHONPATH

stage=$1
stop_stage=$2

huggingface_model_local_dir=./cosyvoice2_llm
model_scope_model_local_dir=./CosyVoice2-0.5B
step_audio_model_dir=./Step-Audio-2-mini

trt_dtype=bfloat16
trt_weights_dir=./trt_weights_${trt_dtype}
trt_engines_dir=./trt_engines_${trt_dtype}

model_repo=./model_repo_cosyvoice2_dit
bls_instance_num=10

if [ $stage -le -1 ] && [ $stop_stage -ge -1 ]; then

    echo "Cloning Step-Audio2-mini"
    git clone https://github.com/yuekaizhang/Step-Audio2.git -b trt $stepaudio2_path

    echo "Cloning CosyVoice"
    git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git $cosyvoice_path
    cd $cosyvoice_path
    git submodule update --init --recursive
    cd runtime/triton_trtllm
fi

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    echo "Downloading CosyVoice2-0.5B"
    # see https://github.com/nvidia-china-sae/mair-hub/blob/main/rl-tutorial/cosyvoice_llm/pretrained_to_huggingface.py
    huggingface-cli download --local-dir $huggingface_model_local_dir yuekai/cosyvoice2_llm
    modelscope download --model iic/CosyVoice2-0.5B --local_dir $model_scope_model_local_dir

    echo "Step-Audio2-mini"
    huggingface-cli download --local-dir $step_audio_model_dir stepfun-ai/Step-Audio-2-mini
    cd $step_audio_model_dir/token2wav
    wget https://huggingface.co/yuekai/cosyvoice2_dit_flow_matching_onnx/resolve/main/flow.decoder.estimator.fp32.dynamic_batch.onnx -O flow.decoder.estimator.fp32.dynamic_batch.onnx
    wget https://huggingface.co/yuekai/cosyvoice2_dit_flow_matching_onnx/resolve/main/flow.decoder.estimator.chunk.fp32.dynamic_batch.simplify.onnx -O flow.decoder.estimator.chunk.fp32.dynamic_batch.simplify.onnx
    cd -
fi


if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    echo "Converting checkpoint to TensorRT weights"
    python3 scripts/convert_checkpoint.py --model_dir $huggingface_model_local_dir \
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
                    --tokenizer_dir $huggingface_model_local_dir \
                    --top_k 50 --top_p 0.95 --temperature 0.8 \
                    --engine_dir=$trt_engines_dir  || exit 1
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    echo "Creating model repository async mode"
    rm -rf $model_repo
    mkdir -p $model_repo
    cosyvoice2_dir="cosyvoice2_dit"
    token2wav_dir="token2wav_dit"

    cp -r ./model_repo/${cosyvoice2_dir} $model_repo
    cp -r ./model_repo/${token2wav_dir} $model_repo
    cp -r ./model_repo/audio_tokenizer $model_repo
    cp -r ./model_repo/speaker_embedding $model_repo


    ENGINE_PATH=$trt_engines_dir
    MAX_QUEUE_DELAY_MICROSECONDS=0
    MODEL_DIR=$model_scope_model_local_dir
    LLM_TOKENIZER_DIR=$huggingface_model_local_dir
    BLS_INSTANCE_NUM=$bls_instance_num
    TRITON_MAX_BATCH_SIZE=1
    DECOUPLED_MODE=True # Only streaming TTS mode is supported using Nvidia Triton for now
    STEP_AUDIO_MODEL_DIR=$step_audio_model_dir/token2wav

    python3 scripts/fill_template.py -i ${model_repo}/${token2wav_dir}/config.pbtxt model_dir:${STEP_AUDIO_MODEL_DIR},triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},max_queue_delay_microseconds:${MAX_QUEUE_DELAY_MICROSECONDS}
    python3 scripts/fill_template.py -i ${model_repo}/${cosyvoice2_dir}/config.pbtxt model_dir:${MODEL_DIR},bls_instance_num:${BLS_INSTANCE_NUM},llm_tokenizer_dir:${LLM_TOKENIZER_DIR},triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},decoupled_mode:${DECOUPLED_MODE},max_queue_delay_microseconds:${MAX_QUEUE_DELAY_MICROSECONDS}
    python3 scripts/fill_template.py -i ${model_repo}/audio_tokenizer/config.pbtxt model_dir:${MODEL_DIR},triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},max_queue_delay_microseconds:${MAX_QUEUE_DELAY_MICROSECONDS}
    python3 scripts/fill_template.py -i ${model_repo}/speaker_embedding/config.pbtxt model_dir:${MODEL_DIR},triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},max_queue_delay_microseconds:${MAX_QUEUE_DELAY_MICROSECONDS}

fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
   echo "Starting Token2wav Triton server and Cosyvoice2 llm using trtllm-serve"
   mpirun -np 1 --allow-run-as-root --oversubscribe trtllm-serve serve --tokenizer $huggingface_model_local_dir $trt_engines_dir --max_batch_size 64  --kv_cache_free_gpu_memory_fraction 0.4 &
   tritonserver --model-repository $model_repo --http-port 18000 &
   wait
    # Test using curl
    # curl http://localhost:8000/v1/chat/completions \
    #     -H "Content-Type: application/json" \
    #     -d '{
    #         "model": "",
    #         "messages":[{"role": "user", "content": "Where is New York?"},
    #                     {"role": "assistant", "content": "<|s_1708|><|s_2050|><|s_2159|>"}],
    #         "max_tokens": 512,
    #         "temperature": 0.8,
    #         "top_p": 0.95,
    #         "top_k": 50,
    #         "stop": ["<|eos1|>"],
    #         "repetition_penalty": 1.2,
    #         "stream": false
    #     }'
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
    echo "Running benchmark client"
    num_task=4
    mode=streaming
    BLS_INSTANCE_NUM=$bls_instance_num

    python3 client_grpc.py \
        --server-addr localhost \
        --server-port 8001 \
        --model-name cosyvoice2_dit \
        --num-tasks $num_task \
        --mode $mode \
        --huggingface-dataset yuekai/seed_tts_cosy2 \
        --log-dir ./log_single_gpu_concurrent_tasks_${num_task}_${mode}_bls_${BLS_INSTANCE_NUM}

fi

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
  echo "stage 5: Offline TTS (Cosyvoice2 LLM + Step-Audio2-mini DiT Token2Wav) inference using a single python script"

  datasets=(wenetspeech4tts) # wenetspeech4tts, test_zh, zero_shot_zh
  backend=trtllm # hf, trtllm, vllm, trtllm-serve

  batch_sizes=(16)
  token2wav_batch_size=1

  for batch_size in ${batch_sizes[@]}; do
    for dataset in ${datasets[@]}; do
    output_dir=./${dataset}_${backend}_llm_batch_size_${batch_size}_token2wav_batch_size_${token2wav_batch_size}
    CUDA_VISIBLE_DEVICES=1 \
        python3 offline_inference.py \
            --output-dir $output_dir \
            --llm-model-name-or-path $huggingface_model_local_dir \
            --token2wav-path $step_audio_model_dir/token2wav \
            --backend $backend \
            --batch-size $batch_size --token2wav-batch-size $token2wav_batch_size \
            --engine-dir $trt_engines_dir \
            --split-name ${dataset} || exit 1
    done
  done
fi

if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then
   echo "Running Step-Audio2-mini DiT Token2Wav inference using a single python script"
   export CUDA_VISIBLE_DEVICES=1
   # Note: Using pre-computed cosyvoice2 tokens
   python3 streaming_inference.py --enable-trt --strategy equal # equal, exponential
   # Offline Token2wav inference
   python3 token2wav_dit.py --enable-trt
fi


if [ $stage -le 7 ] && [ $stop_stage -ge 7 ]; then
   echo "Disaggregated Server: LLM and Token2wav on different GPUs"
   echo "Starting LLM server on GPU 0"
   export CUDA_VISIBLE_DEVICES=0
   mpirun -np 1 --allow-run-as-root --oversubscribe trtllm-serve serve --tokenizer $huggingface_model_local_dir $trt_engines_dir --max_batch_size 64  --kv_cache_free_gpu_memory_fraction 0.4 &
   echo "Starting Token2wav server on GPUs 1-3"
   Token2wav_num_gpus=3
   http_port=17000
   grpc_port=18000
   metrics_port=16000
   for i in $(seq 0 $(($Token2wav_num_gpus - 1))); do
       echo "Starting server on GPU $i"
       http_port=$((http_port + 1))
       grpc_port=$((grpc_port + 1))
       metrics_port=$((metrics_port + 1))
       # Two instances of Token2wav server on the same GPU
       CUDA_VISIBLE_DEVICES=$(($i + 1)) tritonserver --model-repository $model_repo --http-port $http_port --grpc-port $grpc_port --metrics-port $metrics_port &
       http_port=$((http_port + 1))
       grpc_port=$((grpc_port + 1))
       metrics_port=$((metrics_port + 1))
       CUDA_VISIBLE_DEVICES=$(($i + 1)) tritonserver --model-repository $model_repo --http-port $http_port --grpc-port $grpc_port --metrics-port $metrics_port &
   done
   wait
fi

if [ $stage -le 8 ] && [ $stop_stage -ge 8 ]; then
    echo "Running benchmark client for Disaggregated Server"
    per_gpu_instances=2
    mode=streaming
    BLS_INSTANCE_NUM=$bls_instance_num
    Token2wav_num_gpus=(1 2 3)
    concurrent_tasks=(1 2 3 4 5 6)
    for n_gpu in ${Token2wav_num_gpus[@]}; do
        echo "Test 1 GPU for LLM server and $n_gpu GPUs for Token2wav servers"
        for concurrent_task in ${concurrent_tasks[@]}; do
            num_instances=$((per_gpu_instances * n_gpu))
            for i in $(seq 1 $num_instances); do
                port=$(($i + 18000))
                python3 client_grpc.py \
                    --server-addr localhost \
                    --server-port $port \
                    --model-name cosyvoice2_dit \
                    --num-tasks $concurrent_task \
                    --mode $mode \
                    --huggingface-dataset yuekai/seed_tts_cosy2 \
                    --log-dir ./log_disagg_concurrent_tasks_${concurrent_task}_per_instance_total_token2wav_instances_${num_instances}_port_${port} &
            done
            wait
        done
    done
fi