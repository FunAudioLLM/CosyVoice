# CosyVoice2 Minimal Inference Setup (GPU-Only)

This directory contains a streamlined version of the CosyVoice2 official inference code, designed for **CUDA-enabled GPU execution** and with a focus on clear dependency management.

## 1. Setup Environment

**A CUDA-enabled GPU is required.**

It's recommended to use a virtual environment (e.g., conda or venv).

```bash
python -m venv cosyvoice_env
source cosyvoice_env/bin/activate  # On Windows: cosyvoice_env\Scripts\activate
```

Install the necessary Python packages:
```bash
pip install -r requirements.txt
```
The `requirements.txt` file lists `onnxruntime-gpu`. Ensure its version is compatible with your CUDA toolkit. Refer to [ONNX Runtime Execution Providers](https://onnxruntime.ai/docs/execution-providers/) for more details if you encounter issues.

## 2. Dependencies and External Code

### Matcha-TTS
This version of CosyVoice2 relies on components from the `Matcha-TTS` library, specifically for parts of its flow-based acoustic model (e.g., `CausalCFMDecoder`, `SinusoidalPosEmb`). Since `Matcha-TTS` is not available as a standard PyPI package, you need to make its Python modules available.

1.  Clone the `Matcha-TTS` repository. It's often included as a submodule in the full CosyVoice repository (`third_party/Matcha-TTS`), or you can find it on Hugging Face (`FunAudioLLM/Matcha-TTS`).
    ```bash
    # Example:
    git clone https://huggingface.co/FunAudioLLM/Matcha-TTS
    ```
2.  Add the path to the directory containing the `matcha` package (i.e., the directory *above* the `matcha` folder itself) to your `PYTHONPATH`.
    For example, if you cloned `Matcha-TTS` such that you have `/path/to/Matcha-TTS/matcha/...`, you would add `/path/to/Matcha-TTS` to your `PYTHONPATH`:
    ```bash
    export PYTHONPATH=/path/to/Matcha-TTS:$PYTHONPATH
    ```
    Verify that you can run `python -c "import matcha.models.components"` without errors from your environment.

Failure to correctly set up `Matcha-TTS` will result in `ModuleNotFoundError` when the script tries to import components like `matcha.models.components.flow_matching.BASECFM`.

## 3. Download Models

This script expects the official CosyVoice2 model files to be downloaded and organized locally. **The `modelscope` library for automatic downloads has been removed.**

You must manually download the model snapshot for `FunAudioLLM/CosyVoice2-0.5B` from Hugging Face (or your chosen CosyVoice2 model version).

Create a base directory for your models (e.g., `./models` inside this `cosyvoice_inference_official_style/` directory, or `/path/to/my/all_tts_models/`). The `run_inference.py` script takes a `--model_base_dir` argument pointing to this location.

Inside your `--model_base_dir`, create a directory for the specific model, for example, `CosyVoice2-0.5B`. The structure should be:

```
<model_base_dir>/
└── CosyVoice2-0.5B/      # This is the downloaded snapshot directory
    ├── cosyvoice2.yaml   # Or cosyvoice.yaml, as found in the snapshot. This is CRITICAL.
    ├── llm.pt
    ├── flow.pt           # Contains weights for the FlowModel (encoder, parts of decoder not in ONNX)
    ├── hift.pt
    ├── campplus.onnx
    ├── speech_tokenizer_v2.onnx
    ├── spk2info.pt
    ├── flow.decoder.estimator.fp32.onnx # The ONNX model for the flow decoder's estimator
    ├── asset/            # Contains prompt examples like zero_shot_prompt.wav
    │   └── ...
    └── CosyVoice-BlankEN/  # Qwen pretrain path for tokenizer (referenced in yaml)
        └── ...           # (tokenizer.json, config.json, etc.)
```
Ensure all these files, especially `flow.decoder.estimator.fp32.onnx`, are present in this structure.

**Text Normalization:**
Text normalization now relies on `WeTextProcessing`. The `ttsfrd` specific setup has been removed for simplification.

## 4. Configuring ONNX Flow Decoder (Optional)

By default, the system uses the PyTorch implementation for the flow decoder's estimator. To use the ONNX version (`flow.decoder.estimator.fp32.onnx`):
You need to modify your `<model_base_dir>/CosyVoice2-0.5B/cosyvoice2.yaml` (or `cosyvoice.yaml`).
Locate the `flow` model configuration, then its `decoder` (which is `CausalCFMDecoder`), and then the `denoiser_params` for `CausalConditionalDecoder`. Add the `onnx_model_path` key:

```yaml
# Example snippet from your cosyvoice2.yaml (or cosyvoice.yaml)
# ... other parts of the YAML ...

flow: !new:cosyvoice_lib.flow.flow.CausalMaskedDiffWithXvec
  # ... other CausalMaskedDiffWithXvec params ...
  decoder: !new:matcha.models.decoder.CausalCFMDecoder # This is from Matcha-TTS
    # ... other CausalCFMDecoder params ...
    denoiser_cls: cosyvoice_lib.flow.decoder.CausalConditionalDecoder # Path to our decoder
    denoiser_params:
      in_channels: 240  # Or the actual value from your original YAML (packed input channels)
      out_channels: 80 # Or the actual value (mel channels)
      channels: [256, 256, 256, 256] # Example, use actual values
      # ... other CausalConditionalDecoder PyTorch parameters ...

      # Add this line to enable ONNX estimator:
      onnx_model_path: !ref <root_dir>/flow.decoder.estimator.fp32.onnx
      # onnx_providers: ['CUDAExecutionProvider'] # Default for this GPU-only setup
```
The `!ref <root_dir>/` syntax ensures the path is relative to the YAML file itself. Make sure `flow.decoder.estimator.fp32.onnx` is in the same directory as your `cosyvoice2.yaml`.

## 5. Run Inference

The `run_inference.py` script provides examples for running text-to-speech. **A CUDA GPU is required.**

**Command Line Arguments:**
*   `--model_base_dir` (Required): Base directory where your model snapshots are stored.
*   `--text`: Text to synthesize.
*   `--speaker_id`: Speaker ID for SFT inference.
*   `--prompt_wav`: Path to a prompt WAV file for zero-shot inference.
*   `--prompt_text`: Text accompanying the prompt WAV for zero-shot.
*   `--output_wav`: Path to save the offline synthesized WAV file.

**Important:** You must provide either `--speaker_id` or `--prompt_wav`. If neither is provided, the script defaults to using `speaker_id=" кан_Ян"`.

**Example Usages:**
(Same as before - ensure paths are correct)

1.  **SFT (pre-defined speaker):**
    ```bash
    python run_inference.py \
        --model_base_dir ./models \
        --text "Hello, this is a test of Cosy Voice." \
        --speaker_id " кан_Ян" \
        --output_wav sft_output.wav
    ```

2.  **Zero-shot (using a prompt audio):**
    ```bash
    python run_inference.py \
        --model_base_dir ./models \
        --text "This voice is generated using a prompt audio." \
        --prompt_wav ./models/CosyVoice2-0.5B/asset/zero_shot_prompt.wav \
        --prompt_text "This is a zero shot prompt." \
        --output_wav zeroshot_output.wav
    ```

## Notes

*   **Model Paths in YAML:** `HyperPyYAML` resolves `<root_dir>` to the directory containing the YAML file.
*   **Mel Spectrograms:** The dependency on `openai-whisper` for feature extraction has been removed. Mel spectrograms are now computed using `torchaudio`. (`openai-whisper` is still used for its tokenizer class).
*   **JIT/TRT/vLLM:** Support for these specialized model formats has been removed.
```
