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
- **Stage 7**: Launches servers in a disaggregated setup, with the LLM on GPU 0 and Token2Wav servers on GPUs 1-3.
- **Stage 8**: Runs the benchmark client for the disaggregated server configuration.
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


### Disaggregated Server
When the LLM and token2wav components are deployed on the same GPU, they compete for resources. To optimize performance, we use a disaggregated setup where the LLM is deployed on one dedicated L20 GPU, taking advantage of in-flight batching for inference. The token2wav module is deployed on separate, dedicated GPUs.

The table below shows the first chunk latency results for this configuration. In our tests, we deploy two token2wav instances on each dedicated token2wav GPU.

| token2wav_num_gpu | concurrent_task_per_instance | concurrent_tasks_per_gpu | avg (ms) | p50 (ms) | p90 (ms) | p99 (ms) |
|---|---|---|---|---|---|---|
| 1 | 1 | 1.00 | 218.53 | 217.86 | 254.07 | 296.49 |
| 2 | 1 | 1.33 | 218.82 | 219.21 | 256.62 | 303.13 |
| 3 | 1 | 1.50 | 229.08 | 223.27 | 302.13 | 324.41 |
| 4 | 1 | 1.60 | 203.87 | 198.23 | 254.92 | 279.31 |
| 1 | 2 | 2.00 | 293.46 | 280.53 | 370.81 | 407.40 |
| 2 | 2 | 2.67 | 263.38 | 236.84 | 350.82 | 397.39 |
| 3 | 2 | 3.00 | 308.09 | 275.48 | 385.22 | 521.45 |
| 4 | 2 | 3.20 | 271.85 | 253.25 | 359.03 | 387.91 |
| 1 | 3 | 3.00 | 389.15 | 373.01 | 469.22 | 542.89 |
| 2 | 3 | 4.00 | 403.48 | 394.80 | 481.24 | 507.75 |
| 3 | 3 | 4.50 | 406.33 | 391.28 | 495.43 | 571.29 |
| 4 | 3 | 4.80 | 436.72 | 383.81 | 638.44 | 879.23 |
| 1 | 4 | 4.00 | 520.12 | 493.98 | 610.38 | 739.85 |
| 2 | 4 | 5.33 | 494.60 | 490.50 | 605.93 | 708.09 |
| 3 | 4 | 6.00 | 538.23 | 508.33 | 687.62 | 736.96 |
| 4 | 4 | 6.40 | 579.68 | 546.20 | 721.53 | 958.04 |
| 1 | 5 | 5.00 | 635.02 | 623.30 | 786.85 | 819.84 |
| 2 | 5 | 6.67 | 598.23 | 617.09 | 741.00 | 788.96 |
| 3 | 5 | 7.50 | 644.78 | 684.40 | 786.45 | 1009.45 |
| 4 | 5 | 8.00 | 733.92 | 642.26 | 1024.79 | 1281.55 |
| 1 | 6 | 6.00 | 715.38 | 745.68 | 887.04 | 906.68 |
| 2 | 6 | 8.00 | 748.31 | 753.94 | 873.59 | 1007.14 |
| 3 | 6 | 9.00 | 900.27 | 822.28 | 1431.14 | 1800.23 |
| 4 | 6 | 9.60 | 857.54 | 820.33 | 1150.30 | 1298.53 |

The `concurrent_task_per_gpu` is calculated as:
`concurrent_task_per_gpu = concurrent_task_per_instance * num_token2wav_instance_per_gpu (2) * token2wav_gpus / (token2wav_gpus + llm_gpus (1))`

### Acknowledgements

This work originates from the NVIDIA CISI project. For more multimodal resources, please see [mair-hub](https://github.com/nvidia-china-sae/mair-hub).
