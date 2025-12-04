## Accelerating CosyVoice with NVIDIA Triton Inference Server and TensorRT-LLM

Contributed by Yuekai Zhang (NVIDIA).

### Quick Start

Launch the service directly with Docker Compose:
```sh
docker compose up
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

### Understanding `run.sh`

The `run.sh` script orchestrates the entire workflow through numbered stages.

You can run a subset of stages with:
```sh
bash run.sh <start_stage> <stop_stage> [service_type]
```
- `<start_stage>`: The stage to start from (0-5).
- `<stop_stage>`: The stage to stop after (0-5).

**Stages:**

- **Stage 0**: Downloads the `cosyvoice-2 0.5B` model from HuggingFace.
- **Stage 1**: Converts the HuggingFace checkpoint to the TensorRT-LLM format and builds the TensorRT engines.
- **Stage 2**: Creates the Triton model repository and configures the model files. The configuration is adjusted based on whether `Decoupled=True` (streaming) or `Decoupled=False` (offline) will be used.
- **Stage 3**: Launches the Triton Inference Server.
- **Stage 4**: Runs the single-utterance HTTP client for testing.
- **Stage 5**: Runs the gRPC benchmark client.
- **Stage 6**: Runs the offline inference benchmark test.

### Export Models and Launch Server

Inside the Docker container, prepare the models and start the Triton server by running stages 0-3:
```sh
# This command runs stages 0, 1, 2, and 3
bash run.sh 0 3
```
> [!TIP]
> Both streaming and offline (non-streaming) TTS modes are supported. For streaming TTS, set `Decoupled=True`. For offline TTS, set `Decoupled=False`. You need to rerun stage 2 if you switch between modes.

### Single-Utterance HTTP Client

Sends a single HTTP inference request. This is intended for testing the offline TTS mode (`Decoupled=False`):
```sh
bash run.sh 4 4
```

### Benchmark with client-server mode

To benchmark the running Triton server, pass `streaming` or `offline` as the third argument:
```sh
bash run.sh 5 5 # [streaming|offline]

# You can also customize parameters such as the number of tasks and the dataset split:
# python3 client_grpc.py --num-tasks 2 --huggingface-dataset yuekai/seed_tts_cosy2 --split-name test_zh --mode [streaming|offline]
```
> [!TIP]
> It is recommended to run the benchmark multiple times to get stable results after the initial server warm-up.

### Benchmark with offline inference mode
For offline inference mode benchmark, please check the below command:
```sh
# install FlashCosyVoice for token2wav batching
# git clone https://github.com/yuekaizhang/FlashCosyVoice.git /workspace/FlashCosyVoice -b trt
# cd /workspace/FlashCosyVoice
# pip install -e .
# cd -
# wget https://huggingface.co/yuekai/cosyvoice2_flow_onnx/resolve/main/flow.decoder.estimator.fp32.dynamic_batch.onnx -O $model_scope_model_local_dir/flow.decoder.estimator.fp32.dynamic_batch.onnx

bash run.sh 6 6

# You can also switch to huggingface backend by setting backend=hf
```


### Benchmark Results
The following results were obtained by decoding on a single L20 GPU with 26 prompt audio/target text pairs from the [yuekai/seed_tts](https://huggingface.co/datasets/yuekai/seed_tts) dataset (approximately 170 seconds of audio):

**Client-Server Mode: Streaming TTS (First Chunk Latency)**
| Mode | Concurrency | Avg Latency (ms) | P50 Latency (ms) | RTF |
|---|---|---|---|---|
| Streaming, use_spk2info_cache=False | 1 | 220.43 | 218.07 | 0.1237 |
| Streaming, use_spk2info_cache=False | 2 | 476.97 | 369.25 | 0.1022 |
| Streaming, use_spk2info_cache=False | 4 | 1107.34 | 1243.75| 0.0922 |
| Streaming, use_spk2info_cache=True | 1 | 189.88 | 184.81 | 0.1155 |
| Streaming, use_spk2info_cache=True | 2 | 323.04 | 316.83 | 0.0905 |
| Streaming, use_spk2info_cache=True | 4 | 977.68 | 903.68| 0.0733 |

> If your service only needs a fixed speaker, you can set `use_spk2info_cache=True` in `run.sh`. To add more speakers, refer to the instructions [here](https://github.com/qi-hua/async_cosyvoice?tab=readme-ov-file#9-spk2info-%E8%AF%B4%E6%98%8E).

**Client-Server Mode: Offline TTS (Full Sentence Latency)**
| Mode | Note | Concurrency | Avg Latency (ms) | P50 Latency (ms) | RTF |
|---|---|---|---|---|---|
| Offline, Decoupled=False, use_spk2info_cache=False | [Commit](https://github.com/yuekaizhang/CosyVoice/commit/b44f12110224cb11c03aee4084b1597e7b9331cb) | 1 | 758.04 | 615.79 | 0.0891 |
| Offline, Decoupled=False, use_spk2info_cache=False | [Commit](https://github.com/yuekaizhang/CosyVoice/commit/b44f12110224cb11c03aee4084b1597e7b9331cb) | 2 | 1025.93 | 901.68 | 0.0657 |
| Offline, Decoupled=False, use_spk2info_cache=False | [Commit](https://github.com/yuekaizhang/CosyVoice/commit/b44f12110224cb11c03aee4084b1597e7b9331cb) | 4 | 1914.13 | 1783.58 | 0.0610 |

**Offline Inference Mode: Hugginface LLM V.S. TensorRT-LLM**
| Backend | Batch Size | llm_time_seconds  | total_time_seconds | RTF |
|---------|------------|------------------|-----------------------|--|
| HF | 1 | 39.26 |  44.31 | 0.2494 |
| HF | 2 | 30.54 | 35.62 | 0.2064 |
| HF | 4 | 18.63 |  23.90 | 0.1421 |
| HF | 8 | 11.22 | 16.45 | 0.0947 |
| HF | 16 | 8.42 | 13.78 | 0.0821 |
| TRTLLM | 1 | 12.46 | 17.31 | 0.0987 |
| TRTLLM | 2 | 7.64 |12.65 | 0.0739 |
| TRTLLM | 4 | 4.89 |  9.38 | 0.0539 |
| TRTLLM | 8 | 2.92 |  7.23 | 0.0418 |
| TRTLLM | 16 | 2.01 |  6.63 | 0.0386 |
### OpenAI-Compatible Server

To launch an OpenAI-compatible API service, run the following commands:
```sh
git clone https://github.com/yuekaizhang/Triton-OpenAI-Speech.git
cd Triton-OpenAI-Speech
pip install -r requirements.txt

# After the Triton service is running, start the FastAPI bridge:
python3 tts_server.py --url http://localhost:8000 --ref_audios_dir ./ref_audios/ --port 10086 --default_sample_rate 24000

# Test the service with curl:
bash test/test_cosyvoice.sh
```
> [!NOTE]
> Currently, only the offline TTS mode is compatible with the OpenAI-compatible server.

### Acknowledgements

This work originates from the NVIDIA CISI project. For more multimodal resources, please see [mair-hub](https://github.com/nvidia-china-sae/mair-hub).

