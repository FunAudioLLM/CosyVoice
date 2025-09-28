#!/usr/bin/env bash

set -eou pipefail

stage=-1
stop_stage=4

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

export PYTHONPATH=/workspace/CosyVoice
model_scope_model_path=./CosyVoice2-0.5B
sft_model_path=./transformers_cosyvoice2_llm

if [ $stage -le -2 ] && [ $stop_stage -ge -2 ]; then
  log "stage -2: install dependencies locally if pre-built docker image is not available"
  conda create -n cosyvoice2 python=3.10 -y
  conda activate cosyvoice2
    # install verl
  git clone https://github.com/yuekaizhang/verl.git -b thread
  cd verl
  USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
  pip install --no-deps -e .
  cd -
  # install requirements
  pip install -r requirements.txt
  pip install -U nvidia-pytriton
  git clone https://github.com/yuekaizhang/PytritonSenseVoice.git && cd PytritonSenseVoice && pip install -e .
fi

if [ $stage -le -1 ] && [ $stop_stage -ge -1 ]; then
  log "stage -1: download official CosyVoice2-0.5B LLM model and convert to huggingface compatible checkpoint"
  modelscope download --model iic/CosyVoice2-0.5B --local_dir $model_scope_model_path 
  python3 pretrained_to_huggingface.py \
    --pretrained-cosyvoice2-path $model_scope_model_path \
    --save-path $sft_model_path

  # Or, you could use the following command to download the huggingface compatible checkpoint
  # huggingface-cli download --local-dir $sft_model_path yuekai/cosyvoice2_llm

  # Note: we remove the lm_head's bias to make it compatible with the Qwen2.5-0.5B model in Transformers.
fi

data_dir=data/parquet_aishell3
if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  log "stage 0: prepare data into verl format"
  mkdir -p $data_dir
  wget -O data/aishell-3.jsonl https://huggingface.co/datasets/SparkAudio/voxbox/resolve/main/metadata/aishell-3.jsonl
  # total 88035 samples
  head -n 80000 data/aishell-3.jsonl > data/train.jsonl
  tail -n 100 data/aishell-3.jsonl > data/test.jsonl
  python prepare_data.py \
    --train_file data/train.jsonl \
    --test_file data/test.jsonl \
    --local_dir $data_dir
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  log "stage 1: start token2wav asr server for reward function"
  python3 token2wav_asr_server.py --number-of-devices 8
fi 

exp_name=official_llm_aishell3_grpo
if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  log "stage 2: grpo train"
  export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
  export MKL_SERVICE_FORCE_INTEL=TRUE
  n_gpus_per_node=8
  micro_batch_size=4
  train_batch_size=32
  python3 -m verl.trainer.main_ppo \
      algorithm.adv_estimator=grpo \
      data.train_files=$data_dir/train.parquet \
      data.val_files=$data_dir/test.parquet \
      data.train_batch_size=$train_batch_size \
      data.max_prompt_length=1024 \
      data.max_response_length=512 \
      data.truncation='error' \
      actor_rollout_ref.model.use_remove_padding=False \
      actor_rollout_ref.model.path=$sft_model_path \
      actor_rollout_ref.actor.optim.lr=1e-6 \
      actor_rollout_ref.actor.ppo_mini_batch_size=32 \
      actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$micro_batch_size \
      actor_rollout_ref.actor.use_kl_loss=False \
      actor_rollout_ref.model.enable_gradient_checkpointing=True \
      actor_rollout_ref.actor.fsdp_config.param_offload=False \
      actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
      actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$micro_batch_size \
      actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
      actor_rollout_ref.rollout.name=vllm \
      actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
      actor_rollout_ref.rollout.do_sample=true \
      actor_rollout_ref.rollout.temperature=0.8 \
      actor_rollout_ref.rollout.top_p=0.95 \
      actor_rollout_ref.rollout.top_k=25 \
      actor_rollout_ref.rollout.n=4 \
      actor_rollout_ref.rollout.val_kwargs.do_sample=true \
      actor_rollout_ref.rollout.val_kwargs.temperature=0.8 \
      actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
      actor_rollout_ref.rollout.val_kwargs.top_k=25 \
      reward_model.reward_manager=prime \
      custom_reward_function.path=reward_tts.py \
      custom_reward_function.name=compute_score \
      trainer.project_name='cosyvoice2_grpo' \
      trainer.experiment_name=$exp_name \
      trainer.logger=['console','wandb'] \
      trainer.n_gpus_per_node=$n_gpus_per_node \
      trainer.nnodes=1 \
      trainer.save_freq=100 \
      trainer.test_freq=100 \
      trainer.resume_mode='auto' \
      trainer.total_epochs=1 \
      trainer.val_before_train=False
fi

steps=(100 200 300 400 500)
for step in ${steps[@]}; do
llm_path=./checkpoints/cosyvoice2_grpo/$exp_name/global_step_${step}
if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  log "stage 3: merge the model"
  python -m verl.model_merger merge \
      --backend fsdp \
      --local_dir $llm_path/actor \
      --target_dir $llm_path/merged_hf_model || exit 1
fi 

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
  log "stage 4: Test the model"
  dataset=zero_shot_zh # from CosyVoice3 test set
  # dataset=test_zh # from seed_tts test set
  output_dir=./outputs_${exp_name}_${step}_${dataset}

  token2wav_path=/workspace/CosyVoice2-0.5B
  model_path=$llm_path/merged_hf_model

  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  torchrun --nproc_per_node=8 \
      infer_dataset.py \
        --output-dir $output_dir \
        --llm-model-name-or-path $model_path \
        --token2wav-path $token2wav_path \
        --split-name ${dataset} || exit 1

  bash scripts/compute_wer.sh $output_dir ${dataset}
fi
done

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
  log "stage 5: Convert the RL trained model to CosyVoice repo format"
  python3 huggingface_to_pretrained.py \
    --hf-cosyvoice2-llm-path $llm_path/merged_hf_model \
    --output-path /workspace/CosyVoice2-0.5B/llm-new.pt
  # You need to manually move the llm-new.pt to overwrite /workspace/CosyVoice2-0.5B/llm.pt
  # However, we found that the RL trained model accuracy would slightly drop after this conversion.
  # Please be careful or use the huggingface format inference code.
fi