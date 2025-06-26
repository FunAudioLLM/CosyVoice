# CosyVoice2 Minimal Inference Setup

This directory contains a streamlined version of the CosyVoice2 official inference code, designed to be more self-contained and with a focus on clear dependency management.

## 1. Setup Environment

It's recommended to use a virtual environment (e.g., conda or venv).

```bash
python -m venv cosyvoice_env
source cosyvoice_env/bin/activate  # On Windows: cosyvoice_env\Scripts\activate
```

Install the necessary Python packages:

```bash
pip install -r requirements.txt
```
If you have a CUDA-enabled GPU and want to use `onnxruntime-gpu`:
1. Uninstall `onnxruntime` if it was installed: `pip uninstall onnxruntime`
2. Install `onnxruntime-gpu`. Ensure its version is compatible with your CUDA toolkit.
   Refer to [ONNX Runtime Execution Providers](https://onnxruntime.ai/docs/execution-providers/) for more details.
   For example: `pip install onnxruntime-gpu`

## 2. Download Models

This script expects the official CosyVoice2 model files to be downloaded and organized locally.

Create a base directory for your models, for example, `./models` inside `cosyvoice_inference_official_style/` or a path like `/path/to/my/models/`.
The `run_inference.py` script takes a `--model_base_dir` argument pointing to this location.

Inside your `--model_base_dir`, you should have the specific model snapshot directory, e.g., `CosyVoice2-0.5B`.
The expected structure is:
```
<model_base_dir>/
└── CosyVoice2-0.5B/      # This is the snapshot directory for the model
    ├── cosyvoice2.yaml   # Or cosyvoice.yaml, depending on the snapshot
    ├── llm.pt
    ├── flow.pt
    ├── hift.pt
    ├── campplus.onnx
    ├── speech_tokenizer_v2.onnx
    ├── spk2info.pt
    ├── asset/            # Contains prompt examples like zero_shot_prompt.wav
    │   └── ...
    └── CosyVoice-BlankEN/  # Qwen pretrain path for tokenizer (referenced in yaml)
        └── ...           # (tokenizer.json, config.json, etc.)
```
Download these files from the Hugging Face model repository (e.g., `FunAudioLLM/CosyVoice2-0.5B`).

**Additionally, for text normalization (especially for Chinese using `ttsfrd`):**
The text normalization component `ttsfrd` (if its Python package is installed and preferred by the frontend) expects its resources in a specific relative path. If you intend to use `ttsfrd`, create the following structure at the root of this `cosyvoice_inference_official_style` package:
```
cosyvoice_inference_official_style/
├── pretrained_models/          # This name is fixed due to frontend.py's relative path logic
│   └── CosyVoice-ttsfrd/
│       └── resource/
│           └── # ... ttsfrd resource files ...
├── cosyvoice_lib/
├── models/                     # Example model_base_dir
└── run_inference.py
...
```
You'll need to obtain the `CosyVoice-ttsfrd` resources. These are often part of full CosyVoice or FunASR releases/snapshots. If `ttsfrd` is not found or its resources are missing, the system will fall back to `WeTextProcessing` for text normalization, which might have different quality/coverage.

## 3. Run Inference

The `run_inference.py` script provides examples for running text-to-speech.

**Command Line Arguments:**

*   `--model_base_dir` (Required): Base directory where your model snapshots are stored (e.g., `./models` or `/path/to/all_my_tts_models`). The script will look for a model snapshot named `CosyVoice2-0.5B` (by default, can be changed in script) inside this directory.
*   `--text`: Text to synthesize. Default: "你好，欢迎使用慷燕语音合成服务。"
*   `--speaker_id`: Speaker ID for SFT inference (e.g., one from `spk2info.pt` like " кан_Ян").
*   `--prompt_wav`: Path to a prompt WAV file for zero-shot inference.
*   `--prompt_text`: Text accompanying the prompt WAV for zero-shot (optional, a default is used if empty).
*   `--output_wav`: Path to save the offline synthesized WAV file. Default: `output_offline.wav`.

**Important:** You must provide either `--speaker_id` or `--prompt_wav`. If neither is provided, the script defaults to using `speaker_id=" кан_Ян"`.

**Example Usages:**

1.  **SFT (pre-defined speaker):**
    Make sure your `spk2info.pt` inside `<model_base_dir>/CosyVoice2-0.5B/` contains the speaker ID.
    ```bash
    python run_inference.py \
        --model_base_dir ./models \
        --text "Hello, this is a test of Cosy Voice." \
        --speaker_id " кан_Ян" \
        --output_wav sft_output.wav
    ```

2.  **Zero-shot (using a prompt audio):**
    You can use one of the prompt audios from the model's `asset` directory or your own 16kHz mono WAV file.
    ```bash
    python run_inference.py \
        --model_base_dir ./models \
        --text "This voice is generated using a prompt audio." \
        --prompt_wav ./models/CosyVoice2-0.5B/asset/zero_shot_prompt.wav \
        --prompt_text "This is a zero shot prompt." \
        --output_wav zeroshot_output.wav
    ```
    If the provided `--prompt_wav` path is not found, the script will attempt to use a default prompt located at `<model_base_dir>/CosyVoice2-0.5B/asset/zero_shot_prompt.wav`.

The script will perform both offline synthesis (saving to the output file) and then demonstrate streaming synthesis (printing chunk information to the console).

## Notes

*   **Model Paths in YAML:** The `cosyvoice2.yaml` (or `cosyvoice.yaml`) inside your model snapshot directory (e.g., `models/CosyVoice2-0.5B/`) uses `!ref <root_dir>/filename` to refer to model files. `HyperPyYAML` resolves `<root_dir>` to the directory containing the YAML file itself. Ensure paths for `campplus.onnx`, `speech_tokenizer_v2.onnx`, `spk2info.pt`, and the `CosyVoice-BlankEN` directory (for `qwen_pretrain_path`) are correct relative to this YAML or use this `<root_dir>` reference.
*   **GPU/CPU:** The script will attempt to use CUDA if available for PyTorch and ONNX Runtime.
*   **Dependencies:** This setup aims for a more minimal set of dependencies compared to the full original repository but still requires several libraries for core functionality.
```
