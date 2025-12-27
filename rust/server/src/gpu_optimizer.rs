//! GPU Optimizer Module
//!
//! Automatic GPU detection, profiling, and parameter optimization for CosyVoice TTS.
//! Calculates optimal inference parameters based on GPU capabilities to maximize performance.

use serde::{Deserialize, Serialize};
use std::process::Command;
use tracing::{debug, info, warn};

/// GPU information detected from the system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuInfo {
    /// GPU model name (e.g., "NVIDIA GeForce RTX 2070")
    pub name: String,
    /// Total VRAM in MiB
    pub total_memory_mb: u64,
    /// Free VRAM in MiB
    pub free_memory_mb: u64,
    /// Current GPU utilization percentage
    pub gpu_utilization: u32,
    /// Current memory utilization percentage
    pub memory_utilization: u32,
    /// CUDA compute capability (e.g., "7.5" for Turing)
    pub compute_capability: String,
    /// Whether tensor cores are available
    pub tensor_cores_available: bool,
    /// GPU architecture tier
    pub architecture: GpuArchitecture,
    /// Driver version
    pub driver_version: String,
    /// CUDA version
    pub cuda_version: String,
}

/// GPU architecture classification for optimization decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GpuArchitecture {
    /// Kepler (CC 3.x) - Very old, minimal support
    Kepler,
    /// Maxwell (CC 5.x) - Old, basic CUDA
    Maxwell,
    /// Pascal (CC 6.x) - GTX 10xx series
    Pascal,
    /// Volta (CC 7.0) - Tesla V100, Titan V
    Volta,
    /// Turing (CC 7.5) - RTX 20xx, GTX 16xx
    Turing,
    /// Ampere (CC 8.x) - RTX 30xx, A100
    Ampere,
    /// Ada Lovelace (CC 8.9) - RTX 40xx
    AdaLovelace,
    /// Hopper (CC 9.0) - H100
    Hopper,
    /// Unknown architecture
    Unknown,
}

impl GpuArchitecture {
    /// Parse compute capability string to architecture.
    pub fn from_compute_capability(cc: &str) -> Self {
        let parts: Vec<&str> = cc.split('.').collect();
        if parts.is_empty() {
            return Self::Unknown;
        }

        match parts[0].parse::<u32>().unwrap_or(0) {
            3 => Self::Kepler,
            5 => Self::Maxwell,
            6 => Self::Pascal,
            7 => {
                if parts.len() > 1 && parts[1] == "5" {
                    Self::Turing
                } else {
                    Self::Volta
                }
            }
            8 => {
                if parts.len() > 1 && parts[1] == "9" {
                    Self::AdaLovelace
                } else {
                    Self::Ampere
                }
            }
            9 => Self::Hopper,
            _ => Self::Unknown,
        }
    }

    /// Check if architecture supports FP16 tensor core acceleration.
    pub fn supports_fp16_tensor_cores(&self) -> bool {
        matches!(
            self,
            Self::Volta | Self::Turing | Self::Ampere | Self::AdaLovelace | Self::Hopper
        )
    }

    /// Check if architecture supports TensorRT efficiently.
    pub fn supports_tensorrt(&self) -> bool {
        matches!(
            self,
            Self::Pascal
                | Self::Volta
                | Self::Turing
                | Self::Ampere
                | Self::AdaLovelace
                | Self::Hopper
        )
    }
}

/// Optimization profile for CosyVoice inference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationProfile {
    /// Enable FP16 (half-precision) inference
    pub fp16_enabled: bool,
    /// Enable TensorRT acceleration
    pub tensorrt_enabled: bool,
    /// TensorRT concurrent execution contexts
    pub trt_concurrent: u32,
    /// Enable vLLM for LLM acceleration
    pub vllm_enabled: bool,
    /// Streaming chunk buffer size in bytes
    pub stream_buffer_size: usize,
    /// Maximum concurrent inference requests
    pub max_concurrent_requests: usize,
    /// HTTP connection pool size
    pub connection_pool_size: usize,
    /// Request timeout in seconds
    pub request_timeout_secs: u64,
    /// Recommended model variant
    pub recommended_model: ModelVariant,
    /// Explanation of optimization choices
    pub optimization_notes: Vec<String>,
}

impl Default for OptimizationProfile {
    fn default() -> Self {
        Self {
            fp16_enabled: false,
            tensorrt_enabled: false,
            trt_concurrent: 1,
            vllm_enabled: false,
            stream_buffer_size: 16384,
            max_concurrent_requests: 2,
            connection_pool_size: 4,
            request_timeout_secs: 300,
            recommended_model: ModelVariant::CosyVoice300M,
            optimization_notes: vec!["Default conservative settings".to_string()],
        }
    }
}

/// Model variant recommendations based on GPU capability.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelVariant {
    /// CosyVoice 300M - Smallest, fastest
    CosyVoice300M,
    /// CosyVoice 500M - Balanced
    CosyVoice500M,
    /// CosyVoice2 - Enhanced quality
    CosyVoice2,
    /// CosyVoice3 0.5B - Latest with best quality
    FunCosyVoice3_05B,
}

/// Performance statistics for monitoring.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PerformanceStats {
    /// Total inference requests processed
    pub total_requests: u64,
    /// Average Real-Time Factor (RTF) - lower is better, <1 means faster than realtime
    pub average_rtf: f32,
    /// Minimum RTF observed
    pub min_rtf: f32,
    /// Maximum RTF observed
    pub max_rtf: f32,
    /// Average latency in milliseconds
    pub average_latency_ms: f32,
    /// Current GPU utilization
    pub current_gpu_utilization: f32,
    /// Current GPU memory utilization
    pub current_memory_utilization: f32,
    /// Requests per second
    pub requests_per_second: f32,
}

/// GPU Optimizer - detects GPU and calculates optimal parameters.
#[derive(Debug)]
pub struct GpuOptimizer {
    gpu_info: Option<GpuInfo>,
    profile: OptimizationProfile,
    stats: PerformanceStats,
}

impl GpuOptimizer {
    /// Create a new GPU optimizer with automatic detection.
    pub fn new() -> Self {
        let mut optimizer = Self {
            gpu_info: None,
            profile: OptimizationProfile::default(),
            stats: PerformanceStats::default(),
        };

        // Auto-detect and optimize
        if let Some(gpu_info) = optimizer.detect_gpu() {
            optimizer.gpu_info = Some(gpu_info.clone());
            optimizer.profile = optimizer.calculate_optimal_profile(&gpu_info);
            info!("GPU detected: {}", gpu_info.name);
            info!("Optimization profile: {:?}", optimizer.profile);
        } else {
            warn!("No GPU detected, using CPU fallback settings");
        }

        optimizer
    }

    /// Get detected GPU info.
    pub fn gpu_info(&self) -> Option<&GpuInfo> {
        self.gpu_info.as_ref()
    }

    /// Get current optimization profile.
    pub fn profile(&self) -> &OptimizationProfile {
        &self.profile
    }

    /// Get performance statistics.
    pub fn stats(&self) -> &PerformanceStats {
        &self.stats
    }

    /// Update performance statistics with a new request.
    pub fn record_request(&mut self, rtf: f32, latency_ms: f32) {
        let n = self.stats.total_requests as f32;
        self.stats.total_requests += 1;

        // Running average for RTF
        self.stats.average_rtf = (self.stats.average_rtf * n + rtf) / (n + 1.0);
        self.stats.average_latency_ms = (self.stats.average_latency_ms * n + latency_ms) / (n + 1.0);

        // Update min/max
        if self.stats.total_requests == 1 {
            self.stats.min_rtf = rtf;
            self.stats.max_rtf = rtf;
        } else {
            self.stats.min_rtf = self.stats.min_rtf.min(rtf);
            self.stats.max_rtf = self.stats.max_rtf.max(rtf);
        }
    }

    /// Detect GPU using nvidia-smi.
    fn detect_gpu(&self) -> Option<GpuInfo> {
        // Query comprehensive GPU info
        let output = Command::new("nvidia-smi")
            .args([
                "--query-gpu=name,memory.total,memory.free,utilization.gpu,utilization.memory,driver_version",
                "--format=csv,noheader,nounits",
            ])
            .output()
            .ok()?;

        if !output.status.success() {
            return None;
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let parts: Vec<&str> = stdout.trim().split(", ").collect();

        if parts.len() < 6 {
            warn!("Unexpected nvidia-smi output format: {}", stdout);
            return None;
        }

        let name = parts[0].to_string();
        let total_memory_mb: u64 = parts[1].parse().unwrap_or(0);
        let free_memory_mb: u64 = parts[2].parse().unwrap_or(0);
        let gpu_utilization: u32 = parts[3].trim().parse().unwrap_or(0);
        let memory_utilization: u32 = parts[4].trim().parse().unwrap_or(0);
        let driver_version = parts[5].to_string();

        // Get CUDA version and compute capability
        let (cuda_version, compute_capability) = self.get_cuda_info().unwrap_or_default();

        let architecture = GpuArchitecture::from_compute_capability(&compute_capability);
        let tensor_cores_available = architecture.supports_fp16_tensor_cores();

        Some(GpuInfo {
            name,
            total_memory_mb,
            free_memory_mb,
            gpu_utilization,
            memory_utilization,
            compute_capability,
            tensor_cores_available,
            architecture,
            driver_version,
            cuda_version,
        })
    }

    /// Get CUDA version and compute capability.
    fn get_cuda_info(&self) -> Option<(String, String)> {
        // Get CUDA version from nvidia-smi
        let cuda_output = Command::new("nvidia-smi")
            .args(["--query-gpu=cuda_version", "--format=csv,noheader"])
            .output()
            .ok()?;

        let cuda_version = if cuda_output.status.success() {
            String::from_utf8_lossy(&cuda_output.stdout).trim().to_string()
        } else {
            "Unknown".to_string()
        };

        // Try to get compute capability - this is trickier as nvidia-smi doesn't directly report it
        // We'll infer it from the GPU name
        let compute_capability = self.infer_compute_capability();

        Some((cuda_version, compute_capability))
    }

    /// Infer compute capability from GPU name.
    fn infer_compute_capability(&self) -> String {
        let output = Command::new("nvidia-smi")
            .args(["--query-gpu=name", "--format=csv,noheader"])
            .output()
            .ok();

        let name = match output {
            Some(o) if o.status.success() => {
                String::from_utf8_lossy(&o.stdout).trim().to_string()
            }
            _ => return "Unknown".to_string(),
        };

        let name_lower = name.to_lowercase();

        // Hopper (H100, etc.)
        if name_lower.contains("h100") || name_lower.contains("h200") {
            return "9.0".to_string();
        }

        // Ada Lovelace (RTX 40xx)
        if name_lower.contains("rtx 40")
            || name_lower.contains("rtx 4090")
            || name_lower.contains("rtx 4080")
            || name_lower.contains("rtx 4070")
            || name_lower.contains("rtx 4060")
            || name_lower.contains("l4")
            || name_lower.contains("l40")
        {
            return "8.9".to_string();
        }

        // Ampere (RTX 30xx, A100, etc.)
        if name_lower.contains("rtx 30")
            || name_lower.contains("rtx 3090")
            || name_lower.contains("rtx 3080")
            || name_lower.contains("rtx 3070")
            || name_lower.contains("rtx 3060")
            || name_lower.contains("a100")
            || name_lower.contains("a40")
            || name_lower.contains("a30")
            || name_lower.contains("a10")
        {
            return "8.6".to_string();
        }

        // Turing (RTX 20xx, GTX 16xx)
        if name_lower.contains("rtx 20")
            || name_lower.contains("rtx 2080")
            || name_lower.contains("rtx 2070")
            || name_lower.contains("rtx 2060")
            || name_lower.contains("gtx 16")
            || name_lower.contains("t4")
        {
            return "7.5".to_string();
        }

        // Volta (V100, Titan V)
        if name_lower.contains("v100") || name_lower.contains("titan v") {
            return "7.0".to_string();
        }

        // Pascal (GTX 10xx, P100, etc.)
        if name_lower.contains("gtx 10")
            || name_lower.contains("gtx 1080")
            || name_lower.contains("gtx 1070")
            || name_lower.contains("gtx 1060")
            || name_lower.contains("p100")
            || name_lower.contains("p40")
        {
            return "6.1".to_string();
        }

        // Maxwell (GTX 9xx, Titan X Maxwell)
        if name_lower.contains("gtx 9")
            || name_lower.contains("gtx 980")
            || name_lower.contains("gtx 970")
            || name_lower.contains("titan x") && !name_lower.contains("pascal")
        {
            return "5.2".to_string();
        }

        "Unknown".to_string()
    }

    /// Calculate optimal profile based on GPU capabilities.
    fn calculate_optimal_profile(&self, gpu_info: &GpuInfo) -> OptimizationProfile {
        let mut profile = OptimizationProfile::default();
        let mut notes = Vec::new();

        let vram_mb = gpu_info.total_memory_mb;
        let arch = gpu_info.architecture;

        // FP16 decision based on tensor core support
        if gpu_info.tensor_cores_available && arch.supports_fp16_tensor_cores() {
            profile.fp16_enabled = true;
            notes.push(format!(
                "FP16 enabled: {} has tensor cores ({:?})",
                gpu_info.name, arch
            ));
        } else {
            notes.push("FP16 disabled: No tensor core support".to_string());
        }

        // TensorRT decision
        if arch.supports_tensorrt() {
            profile.tensorrt_enabled = true;
            notes.push("TensorRT enabled for accelerated inference".to_string());
        }

        // TRT concurrency based on VRAM
        profile.trt_concurrent = match vram_mb {
            0..=4096 => {
                notes.push("Low VRAM: TRT concurrent=1".to_string());
                1
            }
            4097..=8192 => {
                notes.push("Medium VRAM (6-8GB): TRT concurrent=2".to_string());
                2
            }
            8193..=12288 => {
                notes.push("Good VRAM (8-12GB): TRT concurrent=3".to_string());
                3
            }
            12289..=24576 => {
                notes.push("High VRAM (12-24GB): TRT concurrent=4".to_string());
                4
            }
            _ => {
                notes.push("Very high VRAM (>24GB): TRT concurrent=6".to_string());
                6
            }
        };

        // vLLM for newer architectures with sufficient VRAM
        if matches!(arch, GpuArchitecture::Ampere | GpuArchitecture::AdaLovelace | GpuArchitecture::Hopper)
            && vram_mb >= 10000
        {
            profile.vllm_enabled = true;
            notes.push("vLLM enabled for LLM acceleration".to_string());
        }

        // Streaming buffer size based on VRAM
        profile.stream_buffer_size = match vram_mb {
            0..=4096 => 8192,
            4097..=8192 => 16384,
            8193..=16384 => 32768,
            _ => 65536,
        };
        notes.push(format!(
            "Stream buffer: {} bytes",
            profile.stream_buffer_size
        ));

        // Concurrent requests based on VRAM and architecture
        profile.max_concurrent_requests = match (arch, vram_mb) {
            (GpuArchitecture::Hopper, _) => 8,
            (GpuArchitecture::AdaLovelace, v) if v >= 16000 => 6,
            (GpuArchitecture::AdaLovelace, _) => 4,
            (GpuArchitecture::Ampere, v) if v >= 16000 => 6,
            (GpuArchitecture::Ampere, v) if v >= 10000 => 4,
            (GpuArchitecture::Ampere, _) => 3,
            (GpuArchitecture::Turing, v) if v >= 8000 => 3,
            (GpuArchitecture::Turing, _) => 2,
            _ => 2,
        };
        notes.push(format!(
            "Max concurrent requests: {}",
            profile.max_concurrent_requests
        ));

        // Connection pool size (slightly larger than concurrent requests)
        profile.connection_pool_size = (profile.max_concurrent_requests * 2).max(4);

        // Model recommendation based on VRAM
        profile.recommended_model = match vram_mb {
            0..=4096 => ModelVariant::CosyVoice300M,
            4097..=8192 => ModelVariant::CosyVoice500M,
            8193..=12288 => ModelVariant::FunCosyVoice3_05B,
            _ => ModelVariant::FunCosyVoice3_05B,
        };
        notes.push(format!(
            "Recommended model: {:?}",
            profile.recommended_model
        ));

        // Request timeout - longer for larger models
        profile.request_timeout_secs = match profile.recommended_model {
            ModelVariant::CosyVoice300M => 120,
            ModelVariant::CosyVoice500M => 180,
            _ => 300,
        };

        profile.optimization_notes = notes;
        profile
    }

    /// Refresh GPU stats (call periodically for monitoring).
    #[allow(dead_code)]
    pub fn refresh_gpu_stats(&mut self) {
        if let Some(ref mut gpu_info) = self.gpu_info {
            if let Ok(output) = Command::new("nvidia-smi")
                .args([
                    "--query-gpu=utilization.gpu,utilization.memory,memory.free",
                    "--format=csv,noheader,nounits",
                ])
                .output()
            {
                if output.status.success() {
                    let stdout = String::from_utf8_lossy(&output.stdout);
                    let parts: Vec<&str> = stdout.trim().split(", ").collect();
                    if parts.len() >= 3 {
                        gpu_info.gpu_utilization = parts[0].trim().parse().unwrap_or(0);
                        gpu_info.memory_utilization = parts[1].trim().parse().unwrap_or(0);
                        gpu_info.free_memory_mb = parts[2].trim().parse().unwrap_or(0);

                        self.stats.current_gpu_utilization = gpu_info.gpu_utilization as f32;
                        self.stats.current_memory_utilization = gpu_info.memory_utilization as f32;
                    }
                }
            }
        }
    }

    /// Generate CLI arguments for the Python backend based on profile.
    #[allow(dead_code)]
    pub fn python_backend_args(&self) -> Vec<String> {
        let mut args = Vec::new();

        if self.profile.fp16_enabled {
            args.push("--fp16".to_string());
        }

        if self.profile.tensorrt_enabled {
            args.push("--load-trt".to_string());
            args.push(format!("--trt-concurrent={}", self.profile.trt_concurrent));
        }

        if self.profile.vllm_enabled {
            args.push("--load-vllm".to_string());
        }

        args
    }

    /// Log a summary of the optimization configuration.
    pub fn log_summary(&self) {
        if let Some(ref gpu) = self.gpu_info {
            info!("╔══════════════════════════════════════════════════════════════╗");
            info!("║                    GPU OPTIMIZATION SUMMARY                   ║");
            info!("╠══════════════════════════════════════════════════════════════╣");
            info!("║ GPU: {:<54} ║", gpu.name);
            info!("║ VRAM: {} MiB total, {} MiB free                             ║",
                gpu.total_memory_mb, gpu.free_memory_mb);
            info!("║ Architecture: {:?} (Compute Capability {})         ║",
                gpu.architecture, gpu.compute_capability);
            info!("║ Tensor Cores: {}                                         ║",
                if gpu.tensor_cores_available { "Yes" } else { "No" });
            info!("╠══════════════════════════════════════════════════════════════╣");
            info!("║ FP16: {} | TensorRT: {} | TRT Concurrent: {}                ║",
                if self.profile.fp16_enabled { "ON " } else { "OFF" },
                if self.profile.tensorrt_enabled { "ON " } else { "OFF" },
                self.profile.trt_concurrent);
            info!("║ vLLM: {} | Max Concurrent: {} | Buffer: {} KB            ║",
                if self.profile.vllm_enabled { "ON " } else { "OFF" },
                self.profile.max_concurrent_requests,
                self.profile.stream_buffer_size / 1024);
            info!("╚══════════════════════════════════════════════════════════════╝");

            for note in &self.profile.optimization_notes {
                debug!("  → {}", note);
            }
        } else {
            warn!("No GPU detected - running in CPU mode (slow)");
        }
    }
}

impl Default for GpuOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_architecture_from_compute_capability() {
        assert_eq!(
            GpuArchitecture::from_compute_capability("7.5"),
            GpuArchitecture::Turing
        );
        assert_eq!(
            GpuArchitecture::from_compute_capability("8.6"),
            GpuArchitecture::Ampere
        );
        assert_eq!(
            GpuArchitecture::from_compute_capability("8.9"),
            GpuArchitecture::AdaLovelace
        );
        assert_eq!(
            GpuArchitecture::from_compute_capability("6.1"),
            GpuArchitecture::Pascal
        );
    }

    #[test]
    fn test_tensor_core_support() {
        assert!(GpuArchitecture::Turing.supports_fp16_tensor_cores());
        assert!(GpuArchitecture::Ampere.supports_fp16_tensor_cores());
        assert!(!GpuArchitecture::Pascal.supports_fp16_tensor_cores());
    }

    #[test]
    fn test_tensorrt_support() {
        assert!(GpuArchitecture::Pascal.supports_tensorrt());
        assert!(GpuArchitecture::Turing.supports_tensorrt());
        assert!(!GpuArchitecture::Maxwell.supports_tensorrt());
    }

    #[test]
    fn test_default_profile() {
        let profile = OptimizationProfile::default();
        assert!(!profile.fp16_enabled);
        assert!(!profile.tensorrt_enabled);
        assert_eq!(profile.trt_concurrent, 1);
    }
}
