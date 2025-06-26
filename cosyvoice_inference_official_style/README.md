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
1. Uninstall `onnxruntime` if it was installed via `requirements.txt`: `pip uninstall onnxruntime`
2. Install `onnxruntime-gpu`. Ensure its version is compatible with your CUDA toolkit.
   Refer to [ONNX Runtime Execution Providers](https://onnxruntime.ai/docs/execution-providers/) for more details.
   For example: `pip install onnxruntime-gpu`

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

This script expects the official CosyVoice2 model files to be downloaded and organized locally. **The `modelscope` library for automatic downloads has been removed to minimize dependencies.**

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

**Text Normalization Resources (`ttsfrd` - Optional but Recommended for Chinese):**
If you intend to use `ttsfrd` for better text normalization:
1. Obtain the `CosyVoice-ttsfrd` resources.
2. Create the following directory structure at the root of this `cosyvoice_inference_official_style` package:
   ```
   cosyvoice_inference_official_style/
   ├── pretrained_models/          # This name is fixed.
   │   └── CosyVoice-ttsfrd/
   │       └── resource/
   │           └── # ... place ttsfrd resource files here ...
   ...
   ```
If `ttsfrd` (the Python package, often `pyFunTTSExt`) is not installed or its resources are missing, the system will fall back to `WeTextProcessing`.

## 4. Configuring ONNX Flow Decoder (Optional)

By default, the system uses the PyTorch implementation for the flow decoder's estimator. To use the ONNX version (`flow.decoder.estimator.fp32.onnx`):
You need to modify your `<model_base_dir>/CosyVoice2-0.5B/cosyvoice2.yaml` (or `cosyvoice.yaml`).
Locate the `flow` model configuration, then its `decoder` (which is `CausalCFMDecoder`), and then the `denoiser_params` for `CausalConditionalDecoder`. Add the `onnx_model_path` key:

```yaml
# Example snippet from your cosyvoice2.yaml (or cosyvoice.yaml)
# ... other parts of the YAML ...

flow: !new:cosyvoice_lib.flow.flow.CausalMaskedDiffWithXvec # Ensure path matches if changed
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
      # Optional: specify ONNX providers, e.g., ['CUDAExecutionProvider', 'CPUExecutionProvider']
      # onnx_providers: ['CUDAExecutionProvider']
```
The `!ref <root_dir>/` syntax ensures the path is relative to the YAML file itself. Make sure `flow.decoder.estimator.fp32.onnx` is in the same directory as your `cosyvoice2.yaml`.

## 5. Run Inference

The `run_inference.py` script provides examples for running text-to-speech.

**Command Line Arguments:**
(Same as before)
*   `--model_base_dir` (Required)
*   `--text`
*   `--speaker_id`
*   `--prompt_wav`
*   `--prompt_text`
*   `--output_wav`

**Example Usages:**
(Same as before)

## Notes

*   **Model Paths in YAML:** `HyperPyYAML` resolves `<root_dir>` to the directory containing the YAML file. Ensure all paths referenced (like for `CosyVoice-BlankEN`) are correct.
*   **GPU/CPU:** The script attempts to use CUDA if available.
*   **Mel Spectrograms:** The dependency on `openai-whisper` for feature extraction has been removed. Mel spectrograms are now computed using `torchaudio`.
*   **JIT/TRT/vLLM:** Support for these specialized model formats has been removed from this version to simplify dependencies.
```
