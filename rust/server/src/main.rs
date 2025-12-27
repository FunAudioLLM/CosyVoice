//! CosyVoice gRPC Server
//!
//! This server bridges gRPC requests to the Python FastAPI TTS backend,
//! enabling GPU-accelerated text-to-speech with real-time streaming.
//!
//! Features:
//! - Automatic GPU detection and optimization
//! - Connection pooling for improved performance
//! - Real-time performance monitoring
//! - Streaming audio output

mod gpu_optimizer;

use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use async_stream::try_stream;
use clap::Parser;
use futures::Stream;
use reqwest::Client;
use tokio::sync::RwLock;
use tokio_stream::StreamExt;
use tonic::{transport::Server, Request, Response, Status};
use tracing::{debug, info, warn};

use cosyvoice_proto::{
    cosy_voice_service_server::{CosyVoiceService, CosyVoiceServiceServer},
    inference_request::Mode,
    AudioChunk, AudioMetadata, HealthCheckRequest, HealthCheckResponse, InferenceRequest,
    ListSpeakersRequest, ListSpeakersResponse,
};

use gpu_optimizer::{GpuInfo, GpuOptimizer, OptimizationProfile, PerformanceStats};

/// CLI arguments for the server.
#[derive(Parser, Debug)]
#[command(name = "cosyvoice-server")]
#[command(about = "CosyVoice gRPC server with Python backend bridge and GPU auto-optimization")]
struct Args {
    /// Port to listen on
    #[arg(long, default_value = "50051")]
    port: u16,

    /// Python FastAPI backend URL
    #[arg(long, default_value = "http://localhost:50000")]
    python_backend: String,

    /// Maximum concurrent requests (0 = auto-detect based on GPU)
    #[arg(long, default_value = "0")]
    max_concurrent: usize,

    /// Disable automatic GPU optimization
    #[arg(long, default_value = "false")]
    no_auto_optimize: bool,

    /// Force FP16 mode (overrides auto-detection)
    #[arg(long)]
    force_fp16: Option<bool>,

    /// Force TensorRT mode (overrides auto-detection)
    #[arg(long)]
    force_tensorrt: Option<bool>,

    /// TensorRT concurrent execution contexts (overrides auto-detection)
    #[arg(long)]
    trt_concurrent: Option<u32>,
}

/// Shared state for the service.
#[allow(dead_code)]
struct ServiceState {
    optimizer: RwLock<GpuOptimizer>,
    /// Request counter for tracking (reserved for future metrics endpoint)
    request_count: std::sync::atomic::AtomicU64,
}

/// CosyVoice service implementation.
struct CosyVoiceServiceImpl {
    http_client: Client,
    backend_url: String,
    state: Arc<ServiceState>,
    /// Pre-allocated buffer pool for audio chunks
    _buffer_size: usize,
}

impl CosyVoiceServiceImpl {
    fn new(backend_url: String, optimizer: GpuOptimizer) -> Self {
        let profile = optimizer.profile();

        // Build optimized HTTP client with connection pooling
        let http_client = Client::builder()
            .timeout(Duration::from_secs(profile.request_timeout_secs))
            .pool_max_idle_per_host(profile.connection_pool_size)
            .pool_idle_timeout(Duration::from_secs(60))
            .tcp_keepalive(Duration::from_secs(30))
            .tcp_nodelay(true)
            .build()
            .expect("Failed to create HTTP client");

        let buffer_size = profile.stream_buffer_size;

        info!(
            "HTTP client configured: pool_size={}, timeout={}s, buffer={}KB",
            profile.connection_pool_size,
            profile.request_timeout_secs,
            buffer_size / 1024
        );

        Self {
            http_client,
            backend_url,
            state: Arc::new(ServiceState {
                optimizer: RwLock::new(optimizer),
                request_count: std::sync::atomic::AtomicU64::new(0),
            }),
            _buffer_size: buffer_size,
        }
    }

    /// Record performance metrics for a completed request.
    #[allow(dead_code)]
    async fn record_metrics(&self, duration: Duration, audio_duration_secs: f32) {
        let rtf = duration.as_secs_f32() / audio_duration_secs.max(0.001);
        let latency_ms = duration.as_millis() as f32;

        let mut optimizer = self.state.optimizer.write().await;
        optimizer.record_request(rtf, latency_ms);

        self.state
            .request_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        debug!(
            "Request completed: RTF={:.2}, latency={:.0}ms, audio={:.2}s",
            rtf, latency_ms, audio_duration_secs
        );
    }

    /// Call the Python backend for zero-shot inference.
    async fn call_zero_shot(
        &self,
        text: String,
        prompt_text: String,
        prompt_audio: Vec<u8>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<Vec<u8>, Status>> + Send>>, Status> {
        let url = format!("{}/inference_zero_shot", self.backend_url);

        // Build multipart form
        let form = reqwest::multipart::Form::new()
            .text("tts_text", text)
            .text("prompt_text", prompt_text)
            .part(
                "prompt_wav",
                reqwest::multipart::Part::bytes(prompt_audio)
                    .file_name("prompt.wav")
                    .mime_str("audio/wav")
                    .map_err(|e| Status::internal(format!("MIME error: {}", e)))?,
            );

        let response = self
            .http_client
            .post(&url)
            .multipart(form)
            .send()
            .await
            .map_err(|e| Status::unavailable(format!("Backend unavailable: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response
                .text()
                .await
                .unwrap_or_else(|_| String::from("Unknown error"));
            return Err(Status::internal(format!(
                "Backend error {}: {}",
                status, body
            )));
        }

        // Stream the response bytes
        let stream = response.bytes_stream();
        let mapped = stream.map(|result: Result<bytes::Bytes, reqwest::Error>| {
            result
                .map(|bytes| bytes.to_vec())
                .map_err(|e| Status::internal(format!("Stream error: {}", e)))
        });

        Ok(Box::pin(mapped))
    }

    /// Call the Python backend for cross-lingual inference.
    async fn call_cross_lingual(
        &self,
        text: String,
        prompt_audio: Vec<u8>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<Vec<u8>, Status>> + Send>>, Status> {
        let url = format!("{}/inference_cross_lingual", self.backend_url);

        let form = reqwest::multipart::Form::new()
            .text("tts_text", text)
            .part(
                "prompt_wav",
                reqwest::multipart::Part::bytes(prompt_audio)
                    .file_name("prompt.wav")
                    .mime_str("audio/wav")
                    .map_err(|e| Status::internal(format!("MIME error: {}", e)))?,
            );

        let response = self
            .http_client
            .post(&url)
            .multipart(form)
            .send()
            .await
            .map_err(|e| Status::unavailable(format!("Backend unavailable: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response
                .text()
                .await
                .unwrap_or_else(|_| String::from("Unknown error"));
            return Err(Status::internal(format!(
                "Backend error {}: {}",
                status, body
            )));
        }

        let stream = response.bytes_stream();
        let mapped = stream.map(|result: Result<bytes::Bytes, reqwest::Error>| {
            result
                .map(|bytes| bytes.to_vec())
                .map_err(|e| Status::internal(format!("Stream error: {}", e)))
        });

        Ok(Box::pin(mapped))
    }

    /// Call the Python backend for SFT inference.
    async fn call_sft(
        &self,
        text: String,
        speaker_id: String,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<Vec<u8>, Status>> + Send>>, Status> {
        let url = format!(
            "{}/inference_sft?tts_text={}&spk_id={}",
            self.backend_url,
            urlencoding::encode(&text),
            urlencoding::encode(&speaker_id)
        );

        let response = self
            .http_client
            .get(&url)
            .send()
            .await
            .map_err(|e| Status::unavailable(format!("Backend unavailable: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response
                .text()
                .await
                .unwrap_or_else(|_| String::from("Unknown error"));
            return Err(Status::internal(format!(
                "Backend error {}: {}",
                status, body
            )));
        }

        let stream = response.bytes_stream();
        let mapped = stream.map(|result: Result<bytes::Bytes, reqwest::Error>| {
            result
                .map(|bytes| bytes.to_vec())
                .map_err(|e| Status::internal(format!("Stream error: {}", e)))
        });

        Ok(Box::pin(mapped))
    }

    /// Call the Python backend for instruct inference.
    async fn call_instruct(
        &self,
        text: String,
        speaker_id: String,
        instruct_text: String,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<Vec<u8>, Status>> + Send>>, Status> {
        let url = format!("{}/inference_instruct", self.backend_url);

        let form = reqwest::multipart::Form::new()
            .text("tts_text", text)
            .text("spk_id", speaker_id)
            .text("instruct_text", instruct_text);

        let response = self
            .http_client
            .post(&url)
            .multipart(form)
            .send()
            .await
            .map_err(|e| Status::unavailable(format!("Backend unavailable: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response
                .text()
                .await
                .unwrap_or_else(|_| String::from("Unknown error"));
            return Err(Status::internal(format!(
                "Backend error {}: {}",
                status, body
            )));
        }

        let stream = response.bytes_stream();
        let mapped = stream.map(|result: Result<bytes::Bytes, reqwest::Error>| {
            result
                .map(|bytes| bytes.to_vec())
                .map_err(|e| Status::internal(format!("Stream error: {}", e)))
        });

        Ok(Box::pin(mapped))
    }

    /// Get current GPU information.
    async fn get_gpu_info(&self) -> Option<GpuInfo> {
        let optimizer = self.state.optimizer.read().await;
        optimizer.gpu_info().cloned()
    }

    /// Get current optimization profile.
    #[allow(dead_code)]
    async fn get_profile(&self) -> OptimizationProfile {
        let optimizer = self.state.optimizer.read().await;
        optimizer.profile().clone()
    }

    /// Get performance statistics.
    async fn get_stats(&self) -> PerformanceStats {
        let optimizer = self.state.optimizer.read().await;
        optimizer.stats().clone()
    }
}

type InferenceStream = Pin<Box<dyn Stream<Item = Result<AudioChunk, Status>> + Send>>;

#[tonic::async_trait]
impl CosyVoiceService for CosyVoiceServiceImpl {
    type InferenceStream = InferenceStream;

    async fn inference(
        &self,
        request: Request<InferenceRequest>,
    ) -> Result<Response<Self::InferenceStream>, Status> {
        let start_time = Instant::now();
        let req = request.into_inner();
        let text = req.text;
        let text_len = text.len();

        info!(
            "Inference request: {:?}... ({} chars)",
            &text.chars().take(50).collect::<String>(),
            text_len
        );

        let audio_stream: Pin<Box<dyn Stream<Item = Result<Vec<u8>, Status>> + Send>> =
            match req.mode {
                Some(Mode::ZeroShot(zero_shot)) => {
                    info!(
                        "Zero-shot mode with prompt text: {:?}",
                        zero_shot.prompt_text
                    );
                    self.call_zero_shot(
                        text.clone(),
                        zero_shot.prompt_text,
                        zero_shot.prompt_audio,
                    )
                    .await?
                }
                Some(Mode::CrossLingual(cross_lingual)) => {
                    info!("Cross-lingual mode");
                    self.call_cross_lingual(text.clone(), cross_lingual.prompt_audio)
                        .await?
                }
                Some(Mode::Sft(sft)) => {
                    info!("SFT mode with speaker: {}", sft.speaker_id);
                    self.call_sft(text.clone(), sft.speaker_id).await?
                }
                Some(Mode::Instruct(instruct)) => {
                    info!(
                        "Instruct mode with speaker: {}, instruction: {}",
                        instruct.speaker_id, instruct.instruct_text
                    );
                    self.call_instruct(text.clone(), instruct.speaker_id, instruct.instruct_text)
                        .await?
                }
                None => {
                    return Err(Status::invalid_argument("No inference mode specified"));
                }
            };

        // Clone state for metrics recording
        let state = self.state.clone();
        let request_start = start_time;

        // Convert raw audio stream to AudioChunk stream with metrics
        let output_stream = try_stream! {
            let mut chunk_index = 0u32;
            let mut pinned_stream = audio_stream;
            let mut total_audio_bytes = 0usize;

            while let Some(result) = pinned_stream.next().await {
                let audio_data = result?;
                total_audio_bytes += audio_data.len();

                let chunk = AudioChunk {
                    audio_data,
                    chunk_index,
                    is_final: false,
                    metadata: if chunk_index == 0 {
                        Some(AudioMetadata {
                            sample_rate: 22050,
                            channels: 1,
                            bits_per_sample: 16,
                        })
                    } else {
                        None
                    },
                };

                chunk_index += 1;
                yield chunk;
            }

            // Calculate audio duration: bytes / (sample_rate * channels * bytes_per_sample)
            let audio_duration_secs = total_audio_bytes as f32 / (22050.0 * 1.0 * 2.0);
            let duration = request_start.elapsed();

            // Record metrics
            let rtf = duration.as_secs_f32() / audio_duration_secs.max(0.001);
            let latency_ms = duration.as_millis() as f32;

            {
                let mut optimizer = state.optimizer.write().await;
                optimizer.record_request(rtf, latency_ms);
            }

            info!(
                "Inference complete: {} chunks, {:.2}s audio, RTF={:.2}",
                chunk_index, audio_duration_secs, rtf
            );

            // Send final chunk
            yield AudioChunk {
                audio_data: vec![],
                chunk_index,
                is_final: true,
                metadata: None,
            };
        };

        Ok(Response::new(Box::pin(output_stream)))
    }

    async fn list_speakers(
        &self,
        _request: Request<ListSpeakersRequest>,
    ) -> Result<Response<ListSpeakersResponse>, Status> {
        // Try to query Python backend for available speakers
        let url = format!("{}/speakers", self.backend_url);

        let speakers = match self.http_client.get(&url).send().await {
            Ok(response) if response.status().is_success() => {
                match response.json::<Vec<String>>().await {
                    Ok(speakers) => speakers,
                    Err(_) => self.default_speakers(),
                }
            }
            _ => self.default_speakers(),
        };

        Ok(Response::new(ListSpeakersResponse { speakers }))
    }

    async fn health_check(
        &self,
        _request: Request<HealthCheckRequest>,
    ) -> Result<Response<HealthCheckResponse>, Status> {
        // Check if Python backend is available
        let backend_healthy = self
            .http_client
            .get(&self.backend_url)
            .timeout(Duration::from_secs(5))
            .send()
            .await
            .is_ok();

        // Get GPU info from optimizer
        let gpu_info = self.get_gpu_info().await;
        let (gpu_available, gpu_name, gpu_memory_mb) = match gpu_info {
            Some(info) => (true, info.name, info.total_memory_mb),
            None => (false, String::new(), 0),
        };

        // Get performance stats
        let stats = self.get_stats().await;

        info!(
            "Health check: backend={}, GPU={}, avg_RTF={:.2}",
            backend_healthy, gpu_available, stats.average_rtf
        );

        Ok(Response::new(HealthCheckResponse {
            healthy: backend_healthy,
            model_name: "CosyVoice".to_string(),
            gpu_available,
            gpu_name,
            gpu_memory_mb,
        }))
    }
}

impl CosyVoiceServiceImpl {
    /// Default speaker list when backend query fails.
    fn default_speakers(&self) -> Vec<String> {
        vec![
            "中文女".to_string(),
            "中文男".to_string(),
            "英文女".to_string(),
            "英文男".to_string(),
        ]
    }
}

/// Get detailed GPU information using nvidia-smi.
fn get_gpu_info() -> (bool, String, u64) {
    match std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        ])
        .output()
    {
        Ok(output) if output.status.success() => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let parts: Vec<&str> = stdout.trim().split(", ").collect();
            if parts.len() >= 2 {
                let name = parts[0].to_string();
                let memory: u64 = parts[1].parse().unwrap_or(0);
                return (true, name, memory);
            }
            (true, "Unknown GPU".to_string(), 0)
        }
        _ => (false, String::new(), 0),
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing with more detail
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("cosyvoice_server=info".parse().unwrap())
                .add_directive("tower_http=debug".parse().unwrap()),
        )
        .with_target(true)
        .with_thread_ids(true)
        .init();

    let args = Args::parse();

    info!("╔══════════════════════════════════════════════════════════════╗");
    info!("║              CosyVoice gRPC Server v0.1.0                    ║");
    info!("╚══════════════════════════════════════════════════════════════╝");

    // Initialize GPU optimizer
    let optimizer = GpuOptimizer::new();

    // Apply any CLI overrides
    if args.no_auto_optimize {
        warn!("Auto-optimization disabled by CLI flag");
    }

    // Log optimization summary
    optimizer.log_summary();

    // Get profile for configuration
    let profile = optimizer.profile().clone();

    let addr = format!("0.0.0.0:{}", args.port).parse()?;
    let service = CosyVoiceServiceImpl::new(args.python_backend.clone(), optimizer);

    info!("Starting CosyVoice gRPC server on {}", addr);
    info!("Python backend: {}", args.python_backend);
    info!(
        "Configuration: max_concurrent={}, pool_size={}, buffer={}KB",
        profile.max_concurrent_requests,
        profile.connection_pool_size,
        profile.stream_buffer_size / 1024
    );

    // Check GPU availability (legacy log)
    let (gpu_available, gpu_name, gpu_memory) = get_gpu_info();
    if gpu_available {
        info!("GPU detected: {} ({} MiB)", gpu_name, gpu_memory);
    } else {
        warn!("No GPU detected - inference will be slow");
    }

    // Build and start gRPC server with optimizations
    Server::builder()
        .tcp_nodelay(true)
        .tcp_keepalive(Some(Duration::from_secs(30)))
        .http2_keepalive_interval(Some(Duration::from_secs(60)))
        .http2_keepalive_timeout(Some(Duration::from_secs(20)))
        .add_service(
            CosyVoiceServiceServer::new(service)
                .max_decoding_message_size(64 * 1024 * 1024) // 64MB for large audio
                .max_encoding_message_size(64 * 1024 * 1024),
        )
        .serve(addr)
        .await?;

    Ok(())
}
