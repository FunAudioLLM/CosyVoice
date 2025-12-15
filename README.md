[![SVG Banners](https://svg-banners.vercel.app/api?type=origin&text1=CosyVoice🤠&text2=Text-to-Speech%20💖%20Large%20Language%20Model&width=800&height=210)](https://github.com/Akshay090/svg-banners)

## 👉🏻 CosyVoice 👈🏻

**Fun-CosyVoice 3.0**: [Demos](https://funaudiollm.github.io/cosyvoice3/); [Paper](https://arxiv.org/abs/2505.17589); [Modelscope](https://www.modelscope.cn/studios/FunAudioLLM/Fun-CosyVoice3-0.5B); [CV3-Eval](https://github.com/FunAudioLLM/CV3-Eval)

**CosyVoice 2.0**: [Demos](https://funaudiollm.github.io/cosyvoice2/); [Paper](https://arxiv.org/abs/2412.10117); [Modelscope](https://www.modelscope.cn/studios/iic/CosyVoice2-0.5B); [HuggingFace](https://huggingface.co/spaces/FunAudioLLM/CosyVoice2-0.5B)

**CosyVoice 1.0**: [Demos](https://fun-audio-llm.github.io); [Paper](https://funaudiollm.github.io/pdf/CosyVoice_v1.pdf); [Modelscope](https://www.modelscope.cn/studios/iic/CosyVoice-300M)

## Highlight🔥

**Fun-CosyVoice 3.0** is an advanced text-to-speech (TTS) system based on large language models (LLM), surpassing its predecessor (CosyVoice 2.0) in content consistency, speaker similarity, and prosody naturalness. It is designed for zero-shot multilingual speech synthesis in the wild.
### Key Features
- **Language Coverage**: Covers 9 common languages (Chinese, English, Japanese, Korean, German, Spanish, French, Italian, Russian), 18+ Chinese dialects/accents (Guangdong, Minnan, Sichuan, Dongbei, Shan3xi, Shan1xi, Shanghai, Tianjin, Shan1dong, Ningxia, Gansu, etc.) and meanwhile supports both multi-lingual/cross-lingual zero-shot voice cloning.
- **Content Consistency & Naturalness**: Achieves state-of-the-art performance in content consistency, speaker similarity, and prosody naturalness.
- **Pronunciation Inpainting**: Supports pronunciation inpainting of Chinese Pinyin and English CMU phonemes, providing more controllability and thus suitable for production use.
- **Text Normalization**: Supports reading of numbers, special symbols and various text formats without a traditional frontend module.
- **Bi-Streaming**: Support both text-in streaming and audio-out streaming, and achieves latency as low as 150ms while maintaining high-quality audio output.
- **Instruct Support**: Supports various instructions such as languages, dialects, emotions, speed, volume, etc.


## Roadmap

- [x] 2025/12

    - [x] release Fun-CosyVoice3-0.5B-2512 base model, rl model and its training/inference script
    - [x] release Fun-CosyVoice3-0.5B modelscope gradio space

- [x] 2025/08

    - [x] Thanks to the contribution from NVIDIA Yuekai Zhang, add triton trtllm runtime support and cosyvoice2 grpo training support

- [x] 2025/07

    - [x] release Fun-CosyVoice 3.0 eval set

- [x] 2025/05

    - [x] add CosyVoice2-0.5B vllm support

- [x] 2024/12

    - [x] 25hz CosyVoice2-0.5B released

- [x] 2024/09

    - [x] 25hz CosyVoice-300M base model
    - [x] 25hz CosyVoice-300M voice conversion function

- [x] 2024/08

    - [x] Repetition Aware Sampling(RAS) inference for llm stability
    - [x] Streaming inference mode support, including kv cache and sdpa for rtf optimization

- [x] 2024/07

    - [x] Flow matching training support
    - [x] WeTextProcessing support when ttsfrd is not available
    - [x] Fastapi server and client

## Evaluation
| Model | CER (%) ↓ (test-zh) | WER (%) ↓ (test-en) | CER (%) ↓ (test-hard) |
|-----|------------------|------------------|------------------|
| Human | 1.26 | 2.14 | - |
| F5-TTS | 1.53 | 2.00 | 8.67 |
| SparkTTS | 1.20 | 1.98 | - |
| Seed-TTS | 1.12 | 2.25 | 7.59 |
| CosyVoice2 | 1.45 | 2.57 | 6.83 |
| FireRedTTS-2 | 1.14 | 1.95 | - |
| IndexTTS2 | 1.01 | 1.52 | 7.12 |
| VibeVoice | 1.16 | 3.04 | - |
| HiggsAudio | 1.79 | 2.44 | - |
| MiniMax-Speech | 0.83 | 1.65 | - |
| VoxPCM | 0.93 | 1.85 | 8.87 |
| GLM-TTS | 1.03 | - | - |
| GLM-TTS_RL | 0.89 | - | - |
| Fun-CosyVoice3-0.5B-2512 | 1.21 |  2.24 | 6.71 |
| Fun-CosyVoice3-0.5B-2512_RL | 0.81 | 1.68 | 5.44 |


## Install

### Clone and install

- Clone the repo
    ``` sh
    git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
    # If you failed to clone the submodule due to network failures, please run the following command until success
    cd CosyVoice
    git submodule update --init --recursive
    ```

- Install Conda: please see https://docs.conda.io/en/latest/miniconda.html
- Create Conda env:

    ``` sh
    conda create -n cosyvoice -y python=3.10
    conda activate cosyvoice
    pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com

    # If you encounter sox compatibility issues
    # ubuntu
    sudo apt-get install sox libsox-dev
    # centos
    sudo yum install sox sox-devel
    ```

### Model download

We strongly recommend that you download our pretrained `Fun-CosyVoice3-0.5B` `CosyVoice2-0.5B` `CosyVoice-300M` `CosyVoice-300M-SFT` `CosyVoice-300M-Instruct` model and `CosyVoice-ttsfrd` resource.

``` python
# SDK模型下载
from modelscope import snapshot_download
snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', local_dir='pretrained_models/Fun-CosyVoice3-0.5B')
snapshot_download('iic/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')
snapshot_download('iic/CosyVoice-300M', local_dir='pretrained_models/CosyVoice-300M')
snapshot_download('iic/CosyVoice-300M-SFT', local_dir='pretrained_models/CosyVoice-300M-SFT')
snapshot_download('iic/CosyVoice-300M-Instruct', local_dir='pretrained_models/CosyVoice-300M-Instruct')
snapshot_download('iic/CosyVoice-ttsfrd', local_dir='pretrained_models/CosyVoice-ttsfrd')
```

Optionally, you can unzip `ttsfrd` resource and install `ttsfrd` package for better text normalization performance.

Notice that this step is not necessary. If you do not install `ttsfrd` package, we will use wetext by default.

``` sh
cd pretrained_models/CosyVoice-ttsfrd/
unzip resource.zip -d .
pip install ttsfrd_dependency-0.1-py3-none-any.whl
pip install ttsfrd-0.4.2-cp310-cp310-linux_x86_64.whl
```

### Basic Usage

We strongly recommend using `Fun-CosyVoice3-0.5B` for better performance.
Follow the code in `example.py` for detailed usage of each model.
```sh
python example.py
```

#### CosyVoice2 vllm Usage
If you want to use vllm for inference, please install `vllm==v0.9.0`. Older vllm version do not support CosyVoice2 inference.

Notice that `vllm==v0.9.0` has a lot of specific requirements, for example `torch==2.7.0`. You can create a new env to in case your hardward do not support vllm and old env is corrupted.

``` sh
conda create -n cosyvoice_vllm --clone cosyvoice
conda activate cosyvoice_vllm
pip install vllm==v0.9.0 transformers==4.51.3 -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
python vllm_example.py
```

#### Start web demo

You can use our web demo page to get familiar with CosyVoice quickly.

Please see the demo website for details.

``` python
# change iic/CosyVoice-300M-SFT for sft inference, or iic/CosyVoice-300M-Instruct for instruct inference
python3 webui.py --port 50000 --model_dir pretrained_models/CosyVoice-300M
```

#### Advanced Usage

For advanced users, we have provided training and inference scripts in `examples/libritts/cosyvoice/run.sh`.

#### Build for deployment

Optionally, if you want service deployment,
You can run the following steps.

``` sh
cd runtime/python
docker build -t cosyvoice:v1.0 .
# change iic/CosyVoice-300M to iic/CosyVoice-300M-Instruct if you want to use instruct inference
# for grpc usage
docker run -d --runtime=nvidia -p 50000:50000 cosyvoice:v1.0 /bin/bash -c "cd /opt/CosyVoice/CosyVoice/runtime/python/grpc && python3 server.py --port 50000 --max_conc 4 --model_dir iic/CosyVoice-300M && sleep infinity"
cd grpc && python3 client.py --port 50000 --mode <sft|zero_shot|cross_lingual|instruct>
# for fastapi usage
docker run -d --runtime=nvidia -p 50000:50000 cosyvoice:v1.0 /bin/bash -c "cd /opt/CosyVoice/CosyVoice/runtime/python/fastapi && python3 server.py --port 50000 --model_dir iic/CosyVoice-300M && sleep infinity"
cd fastapi && python3 client.py --port 50000 --mode <sft|zero_shot|cross_lingual|instruct>
```

#### Using Nvidia TensorRT-LLM for deployment

Using TensorRT-LLM to accelerate cosyvoice2 llm could give 4x acceleration comparing with huggingface transformers implementation.
To quick start:

``` sh
cd runtime/triton_trtllm
docker compose up -d
```
For more details, you could check [here](https://github.com/FunAudioLLM/CosyVoice/tree/main/runtime/triton_trtllm)

## Discussion & Communication

You can directly discuss on [Github Issues](https://github.com/FunAudioLLM/CosyVoice/issues).

You can also scan the QR code to join our official Dingding chat group.

<img src="./asset/dingding.png" width="250px">

## Acknowledge

1. We borrowed a lot of code from [FunASR](https://github.com/modelscope/FunASR).
2. We borrowed a lot of code from [FunCodec](https://github.com/modelscope/FunCodec).
3. We borrowed a lot of code from [Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS).
4. We borrowed a lot of code from [AcademiCodec](https://github.com/yangdongchao/AcademiCodec).
5. We borrowed a lot of code from [WeNet](https://github.com/wenet-e2e/wenet).

## Citations

``` bibtex
@article{du2024cosyvoice,
  title={Cosyvoice: A scalable multilingual zero-shot text-to-speech synthesizer based on supervised semantic tokens},
  author={Du, Zhihao and Chen, Qian and Zhang, Shiliang and Hu, Kai and Lu, Heng and Yang, Yexin and Hu, Hangrui and Zheng, Siqi and Gu, Yue and Ma, Ziyang and others},
  journal={arXiv preprint arXiv:2407.05407},
  year={2024}
}

@article{du2024cosyvoice,
  title={Cosyvoice 2: Scalable streaming speech synthesis with large language models},
  author={Du, Zhihao and Wang, Yuxuan and Chen, Qian and Shi, Xian and Lv, Xiang and Zhao, Tianyu and Gao, Zhifu and Yang, Yexin and Gao, Changfeng and Wang, Hui and others},
  journal={arXiv preprint arXiv:2412.10117},
  year={2024}
}

@article{du2025cosyvoice,
  title={CosyVoice 3: Towards In-the-wild Speech Generation via Scaling-up and Post-training},
  author={Du, Zhihao and Gao, Changfeng and Wang, Yuxuan and Yu, Fan and Zhao, Tianyu and Wang, Hao and Lv, Xiang and Wang, Hui and Shi, Xian and An, Keyu and others},
  journal={arXiv preprint arXiv:2505.17589},
  year={2025}
}

@inproceedings{lyu2025build,
  title={Build LLM-Based Zero-Shot Streaming TTS System with Cosyvoice},
  author={Lyu, Xiang and Wang, Yuxuan and Zhao, Tianyu and Wang, Hao and Liu, Huadai and Du, Zhihao},
  booktitle={ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--2},
  year={2025},
  organization={IEEE}
}
```

## Disclaimer
The content provided above is for academic purposes only and is intended to demonstrate technical capabilities. Some examples are sourced from the internet. If any content infringes on your rights, please contact us to request its removal.
