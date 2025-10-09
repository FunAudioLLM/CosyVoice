## Accelerating CosyVoice with DiT-based Token2Wav, NVIDIA Triton Inference Server and TensorRT-LLM

Contributed by Yuekai Zhang (NVIDIA).

This document describes how to accelerate CosyVoice with a DiT-based Token2Wav module from Step-Audio2, using NVIDIA Triton Inference Server and TensorRT-LLM.

### Quick Start

Launch the service directly with Docker Compose:
```sh
docker compose -f docker-compose.dit.yml up
```

### Build the Docker Image

To build the image from scratch:
```sh
docker build . -f Dockerfile.server -t soar97/triton-cosyvoice:25.06
```

### Run a Docker Container
```sh
your_mount_dir=/mnt:/mnt
docker run -it --name "cosyvoice-server" --gpus all --net host -v $your_mount_dir --shm-size=2g soar97/triton-cosyvoice:25.06
```

### Understanding `run_stepaudio2_dit_token2wav.sh`

The `run_stepaudio2_dit_token2wav.sh` script orchestrates the entire workflow through numbered stages.

You can run a subset of stages with:
```sh
bash run_stepaudio2_dit_token2wav.sh <start_stage> <stop_stage>
```
- `<start_stage>`: The stage to start from.
- `<stop_stage>`: The stage to stop after.

**Stages:**

- **Stage -1**: Clones the `Step-Audio2` and `CosyVoice` repositories.
- **Stage 0**: Downloads the `cosyvoice2_llm`, `CosyVoice2-0.5B`, and `Step-Audio-2-mini` models.
- **Stage 1**: Converts the HuggingFace checkpoint for the LLM to the TensorRT-LLM format and builds the TensorRT engines.
- **Stage 2**: Creates the Triton model repository, including configurations for `cosyvoice2_dit` and `token2wav_dit`.
- **Stage 3**: Launches the Triton Inference Server for Token2Wav module and uses `trtllm-serve` to deploy Cosyvoice2 LLM.
- **Stage 4**: Runs the gRPC benchmark client for performance testing.
- **Stage 5**: Runs the offline TTS inference benchmark test.
- **Stage 6**: Runs a standalone inference script for the Step-Audio2-mini DiT Token2Wav model.

### Export Models and Launch Server

Inside the Docker container, prepare the models and start the Triton server by running stages 0-3:
```sh
# This command runs stages 0, 1, 2, and 3
bash run_stepaudio2_dit_token2wav.sh 0 3
```

### Benchmark with client-server mode

To benchmark the running Triton server, run stage 4:
```sh
bash run_stepaudio2_dit_token2wav.sh 4 4

# You can customize parameters such as the number of tasks inside the script.
```
The following results were obtained by decoding on a single L20 GPU with the `yuekai/seed_tts_cosy2` dataset.

#### Total Request Latency

| Concurrent Tasks | RTF    | Average (ms) | 50th Percentile (ms) | 90th Percentile (ms) | 95th Percentile (ms) | 99th Percentile (ms) |
| ---------------- | ------ | ------------ | -------------------- | -------------------- | -------------------- | -------------------- |
| 1                | 0.1228 | 833.66       | 779.98               | 1297.05              | 1555.97              | 1653.02              |
| 2                | 0.0901 | 1166.23      | 1124.69              | 1762.76              | 1900.64              | 2204.14              |
| 4                | 0.0741 | 1849.30      | 1759.42              | 2624.50              | 2822.20              | 3128.42              |
| 6                | 0.0774 | 2936.13      | 3054.64              | 3849.60              | 3900.49              | 4245.79              |
| 8                | 0.0691 | 3408.56      | 3434.98              | 4547.13              | 5047.76              | 5346.53              |
| 10               | 0.0707 | 4306.56      | 4343.44              | 5769.64              | 5876.09              | 5939.79              |

#### First Chunk Latency

| Concurrent Tasks | Average (ms) | 50th Percentile (ms) | 90th Percentile (ms) | 95th Percentile (ms) | 99th Percentile (ms) |
| ---------------- | ------------ | -------------------- | -------------------- | -------------------- | -------------------- |
| 1                | 197.50       | 196.13               | 214.65               | 215.96               | 229.21               |
| 2                |  281.15       | 278.20               | 345.18               | 361.79               | 395.97               |
| 4                |  510.65       | 530.50               | 630.13               | 642.44               | 666.65               |
| 6                |  921.54       | 918.86               | 1079.97              | 1265.22              | 1524.41              |
| 8                |  1019.95      | 1085.26              | 1371.05              | 1402.24              | 1410.66              |
| 10               |  1214.98      | 1293.54              | 1575.36              | 1654.51              | 2161.76              |

### Benchmark with offline inference mode
For offline inference mode benchmark, please run stage 5:
```sh
bash run_stepaudio2_dit_token2wav.sh 5 5
```

The following results were obtained by decoding on a single L20 GPU with the `yuekai/seed_tts_cosy2` dataset.

#### Offline TTS (Cosyvoice2 0.5B LLM + StepAudio2 DiT Token2Wav)
| Backend | Batch Size | llm_time_seconds  | total_time_seconds | RTF |
|---------|------------|------------------|-----------------------|--|
| TRTLLM | 16 | 2.01 |  5.03 | 0.0292 |



### Acknowledgements

This work originates from the NVIDIA CISI project. For more multimodal resources, please see [mair-hub](https://github.com/nvidia-china-sae/mair-hub).
