# CosyVoice Rust Client & Server

A full-featured Rust implementation for CosyVoice text-to-speech with real-time audio streaming and voice cloning.

## Features

- **gRPC Server**: Bridges requests to Python TTS backend with GPU acceleration
- **gRPC Client**: Real-time audio streaming with WAV file output
- **Voice Cloning**: Zero-shot voice cloning using reference audio
- **GPU Support**: Utilizes NVIDIA RTX 2070 (or compatible GPU) via Python backend

## Architecture

```
┌─────────────┐     gRPC      ┌──────────────┐    HTTP     ┌─────────────┐
│ Rust Client │◄────────────►│ Rust Server  │◄───────────►│ Python TTS  │
│ (streaming) │   streaming   │ (gRPC proxy) │   FastAPI   │ (CUDA/GPU)  │
└─────────────┘               └──────────────┘              └─────────────┘
```

## Quick Start

### 1. Start Python Backend

```bash
cd /home/grant/github/CosyVoice
pixi run serve  # or: python runtime/python/fastapi/server.py --port 50000
```

### 2. Start Rust Server

```bash
cd rust
cargo run --release --bin cosyvoice-server -- --python-backend http://localhost:50000
```

### 3. Run Client

```bash
# Text-to-speech with default TARS voice
cargo run --release --bin cosyvoice-client -- \
    --text "Hello, I am TARS from Interstellar." \
    --output /tmp/output.wav

# Health check
cargo run --release --bin cosyvoice-client -- --health

# List speakers
cargo run --release --bin cosyvoice-client -- --text "" --list-speakers
```

## Client Options

| Option | Description |
|--------|-------------|
| `-t, --text` | Text to synthesize |
| `-o, --output` | Output WAV file path |
| `-v, --voice` | Reference voice audio file |
| `-p, --prompt-text` | Prompt text for zero-shot mode |
| `-m, --mode` | Mode: zero-shot, cross-lingual, sft, instruct |
| `--speaker` | Speaker ID for SFT/instruct modes |
| `--instruct` | Instruction text for instruct mode |
| `--no-play` | Disable audio playback |
| `--health` | Health check and exit |
| `--list-speakers` | List available speakers |

## Building

```bash
cd rust

# Build without audio playback (no ALSA required)
cargo build --release

# Build with audio playback (requires libasound2-dev)
cargo build --release --features playback
```

## Default Voice

The client uses `asset/interstellar-tars-01-resemble-denoised.wav` as the default reference voice for zero-shot voice cloning (TARS from Interstellar).

## GPU Requirements

- NVIDIA GPU with CUDA support
- PyTorch with CUDA enabled via pixi environment
- TensorRT optional for maximum performance
