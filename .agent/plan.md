# Rust TTS Client-Server with Voice Cloning and GPU Acceleration

## Context & Assumptions

### Context
CosyVoice is a multilingual TTS system supporting zero-shot voice cloning. The project has:
- Existing Python-based gRPC and FastAPI servers in `runtime/python/`
- Rust proto definitions already generated in `rust/proto/src/cosyvoice.rs` with:
  - `InferenceRequest` with SFT, ZeroShot, CrossLingual, Instruct modes
  - `AudioChunk` for streaming response
  - `CosyVoiceServiceClient` and `CosyVoiceServiceServer` gRPC stubs
- Empty `rust/client/src/` and `rust/server/src/` directories to implement
- Reference voice: `asset/interstellar-tars-01-resemble-denoised.wav` (16-bit 44100Hz mono)

### Assumptions
1. GPU utilization will be handled by the Python TTS backend (PyTorch with CUDA)
2. The Rust server will act as a gRPC bridge, calling the Python inference
3. For maximum GPU utilization, we'll use TensorRT when available
4. The client handles real-time audio streaming and file saving

---

## Requirements

### Functional
1. **Rust gRPC Server**: Bridges requests to Python TTS model with GPU acceleration
2. **Rust gRPC Client**: Streams audio in real-time, saves to WAV files
3. **Voice Cloning**: Zero-shot mode using the reference voice by default
4. **Audio Streaming**: Smooth real-time playback with chunk-based streaming
5. **File Saving**: Write streamed audio to WAV format

### Non-Functional
1. **GPU Utilization**: RTX 2070 with CUDA/TensorRT for inference
2. **Latency**: < 500ms first-audio-byte latency
3. **Dependencies**: Latest compatible versions, use pixi for Python, Cargo for Rust
4. **Cross-platform**: Linux primary (WSL2 environment)

---

## Acceptance Criteria

1. `cargo build --release` succeeds for both client and server crates
2. Server starts and reports GPU availability in health check
3. Client can perform zero-shot TTS with reference voice
4. Audio streams in real-time and plays smoothly
5. Audio saves to valid WAV file
6. Dependencies upgraded to latest compatible versions

---

## Architecture

```
┌─────────────┐     gRPC      ┌──────────────┐    Python    ┌─────────────┐
│ Rust Client │◄────────────►│ Rust Server  │◄────────────►│ CosyVoice   │
│ (streaming) │   streaming   │ (gRPC proxy) │   (PyO3)    │ TTS Model   │
└─────────────┘               └──────────────┘              └─────────────┘
      │                              │                            │
      ▼                              ▼                            ▼
 [Audio Output]              [Health/Metrics]              [CUDA/TensorRT]
 [WAV File]                                                [RTX 2070 GPU]
```

### Key Design Decisions
1. **Python interop via subprocess/HTTP**: Simpler than PyO3 for PyTorch models
2. **Enhanced gRPC server**: Rust server calls Python FastAPI backend
3. **Streaming playback**: Use `rodio` or `cpal` for real-time audio
4. **WAV writing**: Use `hound` crate for file output

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| GPU not accessible from Rust | Use Python backend for inference, Rust for streaming |
| Audio latency issues | Pre-buffer chunks, async streaming |
| WAV sample rate mismatch | Resample from 22050Hz model output |
| Dependency conflicts | Use pixi for Python isolation |

---

## Test Strategy

1. **Unit Tests**: Rust client/server logic with mock gRPC
2. **Integration Tests**: End-to-end TTS generation
3. **Manual Tests**: Audio playback quality, GPU utilization

---

## Phases

### Phase 1: Environment Setup & Dependencies
- Set up pixi for Python environment
- Upgrade Rust dependencies to latest
- Verify GPU detection

### Phase 2: Rust Server Implementation
- Implement gRPC service trait
- Add Python backend bridge
- Health check with GPU info

### Phase 3: Rust Client Implementation
- Connect to gRPC server
- Implement audio streaming
- Add WAV file saving
- Default voice cloning with reference audio

### Phase 4: Integration & Testing
- End-to-end testing
- Performance optimization
- Documentation

---

## Session ID
`20251225-174930-2db78e7`
