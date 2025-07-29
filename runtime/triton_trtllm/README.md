## Best Practices for Serving CosyVoice with NVIDIA Triton Inference Server

### Quick Start
Launch the service directly with Docker Compose:
```sh
docker compose up
```

### Build the Docker Image
Build the image from scratch:
```sh
docker build . -f Dockerfile.server -t soar97/triton-cosyvoice:25.06
```

### Run a Docker Container
```sh
your_mount_dir=/mnt:/mnt
docker run -it --name "cosyvoice-server" --gpus all --net host -v $your_mount_dir --shm-size=2g soar97/triton-cosyvoice:25.06
```

### Understanding `run.sh`
The `run.sh` script orchestrates the entire workflow through numbered stages.

Run a subset of stages with:
```sh
bash run.sh <start_stage> <stop_stage> [service_type]
```
- `<start_stage>` – stage to start from (0-5).
- `<stop_stage>`  – stage to stop after (0-5).

Stages:
- **Stage 0** – Download the cosyvoice-2 0.5B model from HuggingFace.
- **Stage 1** – Convert the HuggingFace checkpoint to TensorRT-LLM format and build TensorRT engines.
- **Stage 2** – Create the Triton model repository and configure the model files (adjusts depending on whether `Decoupled=True/False` will be used later).
- **Stage 3** – Launch the Triton Inference Server.
- **Stage 4** – Run the single-utterance HTTP client.
- **Stage 5** – Run the gRPC benchmark client.

### Export Models to TensorRT-LLM and Launch the Server
Inside the Docker container, prepare the models and start the Triton server by running stages 0-3:
```sh
# Runs stages 0, 1, 2, and 3
bash run.sh 0 3
```
*Note: Stage 2 prepares the model repository differently depending on whether you intend to run with `Decoupled=False` or `Decoupled=True`. Rerun stage 2 if you switch the service type.*

### Single-Utterance HTTP Client
Send a single HTTP inference request:
```sh
bash run.sh 4 4
```

### Benchmark with a Dataset
Benchmark the running Triton server. Pass either `streaming` or `offline` as the third argument.
```sh
bash run.sh 5 5

# You can also customise parameters such as num_task and dataset split directly:
# python3 client_grpc.py --num-tasks 2 --huggingface-dataset yuekai/seed_tts_cosy2 --split-name test_zh --mode [streaming|offline]
```
> [!TIP]
> Only offline CosyVoice TTS is currently supported. Setting the client to `streaming` simply enables NVIDIA Triton’s decoupled mode so that responses are returned as soon as they are ready.

### Benchmark Results
Decoding on a single L20 GPU with 26 prompt_audio/target_text [pairs](https://huggingface.co/datasets/yuekai/seed_tts) (≈221 s of audio):

| Mode | Note | Concurrency | Avg Latency (ms) | P50 Latency (ms) | RTF |
|------|------|-------------|------------------|------------------|-----|
| Decoupled=False | [Commit](https://github.com/SparkAudio/cosyvoice/tree/4d769ff782a868524f29e0be851ca64f8b22ebf1/runtime/triton_trtllm) | 1 | 758.04 | 615.79 | 0.0891 |
| Decoupled=False | [Commit](https://github.com/SparkAudio/cosyvoice/tree/4d769ff782a868524f29e0be851ca64f8b22ebf1/runtime/triton_trtllm) | 2 | 1025.93 | 901.68 | 0.0657 |
| Decoupled=False | [Commit](https://github.com/SparkAudio/cosyvoice/tree/4d769ff782a868524f29e0be851ca64f8b22ebf1/runtime/triton_trtllm) | 4 | 1914.13 | 1783.58 | 0.0610 |
| Decoupled=True  | [Commit](https://github.com/SparkAudio/cosyvoice/tree/4d769ff782a868524f29e0be851ca64f8b22ebf1/runtime/triton_trtllm) | 1 | 659.87 | 655.63 | 0.0891 |
| Decoupled=True  | [Commit](https://github.com/SparkAudio/cosyvoice/tree/4d769ff782a868524f29e0be851ca64f8b22ebf1/runtime/triton_trtllm) | 2 | 1103.16 | 992.96 | 0.0693 |
| Decoupled=True  | [Commit](https://github.com/SparkAudio/cosyvoice/tree/4d769ff782a868524f29e0be851ca64f8b22ebf1/runtime/triton_trtllm) | 4 | 1790.91 | 1668.63 | 0.0604 |

### OpenAI-Compatible Server
To launch an OpenAI-compatible service, run:
```sh
git clone https://github.com/yuekaizhang/Triton-OpenAI-Speech.git
pip install -r requirements.txt
# After the Triton service is up, start the FastAPI bridge:
python3 tts_server.py --url http://localhost:8000 --ref_audios_dir ./ref_audios/ --port 10086 --default_sample_rate 24000
# Test with curl
bash test/test_cosyvoice.sh
```

### Acknowledgements
This section originates from the NVIDIA CISI project. We also provide other multimodal resources—see [mair-hub](https://github.com/nvidia-china-sae/mair-hub) for details.

