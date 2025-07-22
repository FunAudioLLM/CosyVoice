## Nvidia Triton Inference Serving Best Practice for Spark TTS

### Quick Start
Directly launch the service using docker compose.
```sh
docker compose up
```

### Build Image
Build the docker image from scratch. 
```sh
docker build . -f Dockerfile.server -t soar97/triton-spark-tts:25.02
```

### Create Docker Container
```sh
your_mount_dir=/mnt:/mnt
docker run -it --name "spark-tts-server" --gpus all --net host -v $your_mount_dir --shm-size=2g soar97/triton-spark-tts:25.02
```

### Understanding `run.sh`

The `run.sh` script automates various steps using stages. You can run specific stages using:
```sh
bash run.sh <start_stage> <stop_stage> [service_type]
```
- `<start_stage>`: The stage to begin execution from (0-5).
- `<stop_stage>`: The stage to end execution at (0-5).
- `[service_type]`: Optional, specifies the service type ('streaming' or 'offline', defaults may apply based on script logic). Required for stages 4 and 5.

Stages:
- **Stage 0**: Download Spark-TTS-0.5B model from HuggingFace.
- **Stage 1**: Convert HuggingFace checkpoint to TensorRT-LLM format and build TensorRT engines.
- **Stage 2**: Create the Triton model repository structure and configure model files (adjusts for streaming/offline).
- **Stage 3**: Launch the Triton Inference Server.
- **Stage 4**: Run the gRPC benchmark client.
- **Stage 5**: Run the single utterance client (gRPC for streaming, HTTP for offline).

### Export Models to TensorRT-LLM and Launch Server
Inside the docker container, you can prepare the models and launch the Triton server by running stages 0 through 3. This involves downloading the original model, converting it to TensorRT-LLM format, building the optimized TensorRT engines, creating the necessary model repository structure for Triton, and finally starting the server.
```sh
# This runs stages 0, 1, 2, and 3
bash run.sh 0 3
```
*Note: Stage 2 prepares the model repository differently based on whether you intend to run streaming or offline inference later. You might need to re-run stage 2 if switching service types.*


### Single Utterance Client
Run a single inference request. Specify `streaming` or `offline` as the third argument.

**Streaming Mode (gRPC):**
```sh
bash run.sh 5 5 streaming
```
This executes the `client_grpc.py` script with predefined example text and prompt audio in streaming mode.

**Offline Mode (HTTP):**
```sh
bash run.sh 5 5 offline
```

### Benchmark using Dataset
Run the benchmark client against the running Triton server. Specify `streaming` or `offline` as the third argument.
```sh
# Run benchmark in streaming mode
bash run.sh 4 4 streaming

# Run benchmark in offline mode
bash run.sh 4 4 offline

# You can also customize parameters like num_task directly in client_grpc.py or via args if supported
# Example from run.sh (streaming):
# python3 client_grpc.py \
#     --server-addr localhost \
#     --model-name spark_tts \
#     --num-tasks 2 \
#     --mode streaming \
#     --log-dir ./log_concurrent_tasks_2_streaming_new

# Example customizing dataset (requires modifying client_grpc.py or adding args):
# python3 client_grpc.py --num-tasks 2 --huggingface-dataset yuekai/seed_tts --split-name wenetspeech4tts --mode [streaming|offline]
```

### Benchmark Results
Decoding on a single L20 GPU, using 26 different prompt_audio/target_text [pairs](https://huggingface.co/datasets/yuekai/seed_tts), total audio duration 169 secs.

| Mode | Note   | Concurrency | Avg Latency     | First Chunk Latency (P50) |  RTF | 
|-------|-----------|-----------------------|---------|----------------|-|
| Offline | [Code Commit](https://github.com/SparkAudio/Spark-TTS/tree/4d769ff782a868524f29e0be851ca64f8b22ebf1/runtime/triton_trtllm) | 1                   | 876.24 ms |-| 0.1362|
| Offline | [Code Commit](https://github.com/SparkAudio/Spark-TTS/tree/4d769ff782a868524f29e0be851ca64f8b22ebf1/runtime/triton_trtllm) | 2                   | 920.97 ms |-|0.0737|
| Offline | [Code Commit](https://github.com/SparkAudio/Spark-TTS/tree/4d769ff782a868524f29e0be851ca64f8b22ebf1/runtime/triton_trtllm) | 4                   | 1611.51 ms |-| 0.0704|
| Streaming | [Code Commit](https://github.com/yuekaizhang/Spark-TTS/commit/0e978a327f99aa49f0735f86eb09372f16410d86) | 1                   | 913.28 ms |210.42 ms| 0.1501 |
| Streaming | [Code Commit](https://github.com/yuekaizhang/Spark-TTS/commit/0e978a327f99aa49f0735f86eb09372f16410d86) | 2                   | 1009.23 ms |226.08 ms |0.0862 |
| Streaming | [Code Commit](https://github.com/yuekaizhang/Spark-TTS/commit/0e978a327f99aa49f0735f86eb09372f16410d86) | 4                   | 1793.86 ms |1017.70 ms| 0.0824 |