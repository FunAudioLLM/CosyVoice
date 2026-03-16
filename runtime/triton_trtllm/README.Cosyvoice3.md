## Accelerating CosyVoice3 with NVIDIA Triton Inference Server and TensorRT-LLM

Contributed by Yuekai Zhang (NVIDIA).

### Quick Start

Launch the service directly with Docker Compose:
```sh
docker compose -f docker-compose.cosyvoice3.yml up
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

### Understanding `run_cosyvoice3.sh`

The `run_cosyvoice3.sh` script orchestrates the entire workflow through numbered stages.

You can run a subset of stages with:
```sh
bash run_cosyvoice3.sh <start_stage> <stop_stage>
```
- `<start_stage>`: The stage to start from.
- `<stop_stage>`: The stage to stop after.

**Stages:**

- **Stage -1**: Clones the `CosyVoice` repository.
- **Stage 0**: Downloads the `Fun-CosyVoice3-0.5B-2512` model and its HuggingFace LLM checkpoint.
- **Stage 1**: Converts the HuggingFace checkpoint for the LLM to the TensorRT-LLM format and builds the TensorRT engines.
- **Stage 2**: Creates the Triton model repository, including configurations for `cosyvoice3`, `token2wav`, `vocoder`, `audio_tokenizer`, and `speaker_embedding`.
- **Stage 3**: Launches the Triton Inference Server for Token2Wav module and uses `trtllm-serve` to deploy CosyVoice3 LLM.
- **Stage 4**: Runs the gRPC benchmark client for performance testing.
- **Stage 5**: Runs the offline TTS inference benchmark test.

### Export Models and Launch Server

Inside the Docker container, prepare the models and start the Triton server by running stages 0-3:
```sh
# This command runs stages 0, 1, 2, and 3
bash run_cosyvoice3.sh 0 3
```

### Benchmark with client-server mode

To benchmark the running Triton server, run stage 4:
```sh
bash run_cosyvoice3.sh 4 4

# You can customize parameters such as the number of tasks inside the script.
```
The following results were obtained by decoding on a single L20 GPU.

#### Streaming TTS (Concurrent Tasks = 4)

**First Chunk Latency**

| Concurrent Tasks | Average (ms) | 50th Percentile (ms) | 90th Percentile (ms) | 95th Percentile (ms) | 99th Percentile (ms) |
| ---------------- | ------------ | -------------------- | -------------------- | -------------------- | -------------------- |
| 4                | 750.42       | 740.31               | 941.05               | 977.55               | 1002.37              |

### Benchmark with offline inference mode

For offline inference mode benchmark, please run stage 5:
```sh
bash run_cosyvoice3.sh 5 5
```

#### Offline TTS (CosyVoice3 0.5B LLM + Token2Wav with TensorRT)

| Backend | LLM Batch Size | llm_time (s) | token2wav_time (s) | pipeline_time (s) | RTF    |
|---------|------------|--------------|--------------------|--------------------|--------|
| TRTLLM  | 1          | 13.21        | 5.72               | 19.48              | 0.1091 |
| TRTLLM  | 2          | 8.46         | 6.02               | 14.91              | 0.0822 |
| TRTLLM  | 4          | 5.07         | 5.95               | 11.43              | 0.0630 |
| TRTLLM  | 8          | 2.98         | 6.11               | 9.53               | 0.0562 |
| TRTLLM  | 16         | 2.12         | 6.27               | 8.83               | 0.0501 |
