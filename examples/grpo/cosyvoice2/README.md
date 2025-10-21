# CosyVoice2 LLM Reinforcement Learning Recipe

This recipe demonstrates how to fine-tune the **CosyVoice2** large language model with reinforcement learning algorithms—specifically **GRPO**—using the [veRL](https://github.com/volcengine/verl) framework. Our experiments show that applying GRPO reduces the character error rate (CER) on the CosyVoice3 `zero_shot_zh` set from 4.08% to 3.36%.

## Table of Contents

- [Environment Setup](#environment-setup)
- [Data Preparation](#data-preparation)
- [Reward Function & ASR Server](#reward-function--asr-server)
- [Training](#training)
- [Evaluation](#evaluation)
- [Export Model](#export-model)
- [Results](#results)
- [Acknowledgement](#acknowledgement)

## Environment Setup
We recommend using the pre-built Docker image below. Alternatively, you can manually install the dependencies following the Dockerfile.
```bash
docker pull soar97/verl:app-verl0.4-vllm0.8.5-mcore0.12.2-te2.2
```
If Docker is not available, you can refer to `run.sh` `stage -2` to install the dependencies locally.

## Data Preparation

`prepare_data.py` expects a JSON/JSONL file with at least the following schema:

```jsonc
{
  "text": "An example sentence to be synthesized."
}
```
You can download the JSONL files from the metadata directory of the [SparkAudio/voxbox](https://huggingface.co/datasets/SparkAudio/voxbox/tree/main/metadata) dataset on Hugging Face.

Stage `0` converts raw JSONL files into the parquet format expected by veRL:

```bash
bash run.sh 0 0
```
Create two JSONL files—`train.jsonl` and `test.jsonl`.
The script will then generate two Parquet files:

```
data/parquet_tiny/train.parquet
data/parquet_tiny/test.parquet
```

Each sample is automatically wrapped into a CosyVoice2-style prompt so that the LLM learns to output CosyVoice2 speech tokens.


## Reward Function & ASR Server

To compute rewards, we run a lightweight server that:

1. Converts generated speech tokens back to a 16 kHz waveform with the **CosyVoice2** pretrained U-Net model.
2. Transcribes the waveform with **SenseVoice** ASR.
3. Calculates the pinyin-level error rate relative to the ground-truth text and maps it to a score between 0 and 1.

Start the server (stage `1`) in a dedicated terminal or on a separate GPU:

```bash
bash run.sh 1 1
# Triton server listens on ports 8000/8001/8002
```

The custom reward implementation is located in [`reward_tts.py`](./reward_tts.py) and calls the server to obtain the reward score.

## Training

Run stage `2` to start GRPO training:

```bash
bash run.sh 2 2
```

Key CLI arguments passed to `verl.trainer.main_ppo`:

* `algorithm.adv_estimator=grpo` – use GRPO instead of PPO.
* `data.train_files=data/parquet_aishell3/train.parquet` and `data.val_files=data/parquet_aishell3/test.parquet`
* `custom_reward_function.path=reward_tts.py` – custom reward function described above.

Adjust `CUDA_VISIBLE_DEVICES`, batch sizes, and other hyperparameters to match your hardware.
> [!TIP]
> Note: the lm_head bias is disabled during training to make the model compatible with VLLM and Transformers' Qwen model.

## Evaluation

After training is complete, collect the sharded FSDP weights and export a Hugging Face-style checkpoint (stage `3`):

```bash
bash run.sh 3 3   # merges weights into $llm_path/merged_hf_model
```

You can then evaluate the model on the CosyVoice3 zero-shot Chinese test set (stage `4`):

```bash
bash run.sh 4 4
```

This command launches distributed inference via `infer_dataset.py` and computes WER with `scripts/compute_wer.sh`.

> [!TIP]
> The script also supports the Seed-TTS test set by setting `dataset=test_zh`.

## Export Model

To use the RL-trained model with the official CosyVoice repository:

```bash
bash run.sh 5 5
```

The script converts the Hugging Face checkpoint back into the format expected by the CosyVoice repository.
> [!TIP]
>  However, we observed a slight accuracy drop when using the RL-trained model after conversion, compared with the Hugging Face format.

## Results

| Model | Seed-TTS `test_zh` CER | CosyVoice3 `zero_shot_zh` CER | Comment |
|-------|------------------------|------------------------------|---------|
| CosyVoice2 LLM (official) | 1.45% | 4.08% | See the [paper](https://arxiv.org/abs/2412.10117) |
| CosyVoice2 LLM + GRPO | 1.37% | **3.36%** | See the [decoding results](yuekai/official-cosyvoice-llm-grpo-aishell3), Hugging Face-format model |

## Acknowledgement

This work was inspired by the implementation in [ch-tts-llasa-rl-grpo](https://github.com/channel-io/ch-tts-llasa-rl-grpo).
