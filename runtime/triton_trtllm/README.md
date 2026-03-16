# Accelerating CosyVoice with NVIDIA Triton Inference Server and TensorRT-LLM

Contributed by Yuekai Zhang (NVIDIA).

This repository provides three acceleration solutions for CosyVoice, each targeting a different model version and Token2Wav architecture. All solutions use TensorRT-LLM for LLM acceleration and NVIDIA Triton Inference Server for serving.

## Solutions

### [CosyVoice3](README.Cosyvoice3.md)

Acceleration solution for [Fun-CosyVoice3-0.5B-2512](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512), the latest CosyVoice model. The pipeline includes `audio_tokenizer`, `speaker_embedding`, `token2wav`, and `vocoder` modules managed by Triton, with the LLM served via `trtllm-serve`.

### [CosyVoice2 + UNet Token2Wav](README.Cosyvoice2.Unet.md)

The baseline acceleration solution for CosyVoice2, using the original UNet-based flow-matching Token2Wav module.

### [CosyVoice2 + DiT Token2Wav](README.Cosyvoice2.DiT.md)

Replaces the UNet Token2Wav with a DiT-based Token2Wav module from [Step-Audio2](https://github.com/stepfun-ai/Step-Audio-2). Supports disaggregated deployment where the LLM and Token2Wav run on separate GPUs for better resource utilization under high concurrency.



## Quick Start

Each solution can be launched with a single Docker Compose command:

```sh
# CosyVoice3
docker compose -f docker-compose.cosyvoice3.yml up

# CosyVoice2 + UNet Token2Wav
docker compose -f docker-compose.cosyvoice2.unet.yml up

# CosyVoice2 + DiT Token2Wav
docker compose -f docker-compose.cosyvoice2.dit.yml up
```

