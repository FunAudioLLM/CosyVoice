# Backlog - Rust TTS Client-Server

Session: `20251225-174930-2db78e7`

## Atomic Work Items

### 1. Environment Setup with Pixi
- **Title**: Initialize pixi environment and upgrade Python dependencies
- **Description**: Set up pixi.toml for Python environment management with latest compatible CUDA/PyTorch dependencies
- **Acceptance Criteria**:
  - `pixi.toml` created with Python 3.10+ and CUDA 12.1
  - `pixi install` succeeds
  - GPU detected via `torch.cuda.is_available()`
- **Effort**: S

---

### 2. Upgrade Rust Dependencies
- **Title**: Update Cargo.toml with latest compatible crate versions
- **Description**: Create/update Cargo workspace with latest tonic, tokio, prost, hound, rodio crates
- **Acceptance Criteria**:
  - Workspace Cargo.toml at `rust/Cargo.toml`
  - `cargo build` succeeds with latest deps
  - No breaking API changes unhandled
- **Dependencies**: None
- **Effort**: S

---

### 3. Implement Rust gRPC Server
- **Title**: Create CosyVoice gRPC server implementation
- **Description**: Implement `cosy_voice_service_server::CosyVoiceService` trait in `rust/server/src/main.rs`, bridging to Python backend
- **Acceptance Criteria**:
  - Server compiles and starts on port 50051
  - Health check returns GPU info
  - Inference endpoint streams audio chunks
- **Dependencies**: #2
- **Effort**: L

---

### 4. Implement Rust gRPC Client
- **Title**: Create streaming TTS client with audio playback
- **Description**: Implement client that connects to server, sends zero-shot requests with reference voice, plays audio in real-time
- **Acceptance Criteria**:
  - Client connects and sends TTS request
  - Audio streams and plays in real-time using cpal/rodio
  - Saves output to WAV file using hound
  - Uses `interstellar-tars-01-resemble-denoised.wav` as default voice
- **Dependencies**: #3
- **Effort**: L

---

### 5. Enhanced Python Backend Server
- **Title**: Upgrade Python server with GPU optimizations
- **Description**: Enhance FastAPI server to use TensorRT when available, add GPU metrics endpoint
- **Acceptance Criteria**:
  - Server reports GPU memory usage
  - TensorRT used if available
  - Streaming latency < 300ms
- **Dependencies**: #1
- **Effort**: M

---

### 6. Integration Testing & Documentation
- **Title**: End-to-end tests and README updates
- **Description**: Create integration test, update README with usage instructions
- **Acceptance Criteria**:
  - Full TTS pipeline test passes
  - README documents setup and usage
  - Example commands provided
- **Dependencies**: #3, #4, #5
- **Effort**: M

---

## Summary

| # | Title | Effort | Dependencies |
|---|-------|--------|--------------|
| 1 | Pixi Environment Setup | S | - |
| 2 | Upgrade Rust Dependencies | S | - |
| 3 | Rust gRPC Server | L | #2 |
| 4 | Rust gRPC Client | L | #3 |
| 5 | Enhanced Python Backend | M | #1 |
| 6 | Integration & Docs | M | #3, #4, #5 |
