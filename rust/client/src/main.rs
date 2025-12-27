//! CosyVoice gRPC Client
//!
//! A full-featured client for the CosyVoice TTS service with:
//! - Real-time audio streaming and playback (optional, requires `playback` feature)
//! - WAV file output
//! - Zero-shot voice cloning with custom reference audio

use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use hound::{WavReader, WavSpec, WavWriter};
use rubato::{FftFixedIn, Resampler};
use tokio_stream::StreamExt;
use tonic::transport::Channel;
use tracing::{info, warn};

use cosyvoice_proto::{
    cosy_voice_service_client::CosyVoiceServiceClient,
    inference_request::Mode,
    CrossLingualMode, HealthCheckRequest, InferenceRequest, InstructMode, ListSpeakersRequest,
    SftMode, ZeroShotMode,
};

#[cfg(feature = "playback")]
mod audio_player {
    use std::sync::mpsc::{channel, Receiver, Sender};
    use std::sync::{Arc, Mutex};

    use anyhow::{Context, Result};
    use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
    use cpal::SampleFormat;
    use tracing::{error, info, warn};

    /// Audio player using cpal.
    pub struct AudioPlayer {
        _stream: cpal::Stream,
        sender: Sender<Vec<i16>>,
    }

    impl AudioPlayer {
        pub fn new() -> Result<Self> {
            let host = cpal::default_host();
            let device = host
                .default_output_device()
                .context("No output device available")?;

            info!("Using audio device: {}", device.name()?);

            let config = device.default_output_config()?;
            let sample_format = config.sample_format();
            let config: cpal::StreamConfig = config.into();

            let (sender, receiver): (Sender<Vec<i16>>, Receiver<Vec<i16>>) = channel();
            let receiver = Arc::new(Mutex::new(receiver));
            let buffer: Arc<Mutex<Vec<i16>>> = Arc::new(Mutex::new(Vec::new()));

            let receiver_clone = receiver.clone();
            let buffer_clone = buffer.clone();

            let err_fn = |err| error!("Audio stream error: {}", err);

            let stream = match sample_format {
                SampleFormat::I16 => device.build_output_stream(
                    &config,
                    move |data: &mut [i16], _: &cpal::OutputCallbackInfo| {
                        if let Ok(rx) = receiver_clone.lock() {
                            while let Ok(new_data) = rx.try_recv() {
                                if let Ok(mut buf) = buffer_clone.lock() {
                                    buf.extend(new_data);
                                }
                            }
                        }

                        if let Ok(mut buf) = buffer_clone.lock() {
                            for sample in data.iter_mut() {
                                *sample = if !buf.is_empty() {
                                    buf.remove(0)
                                } else {
                                    0
                                };
                            }
                        }
                    },
                    err_fn,
                    None,
                )?,
                SampleFormat::F32 => device.build_output_stream(
                    &config,
                    move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                        if let Ok(rx) = receiver_clone.lock() {
                            while let Ok(new_data) = rx.try_recv() {
                                if let Ok(mut buf) = buffer_clone.lock() {
                                    buf.extend(new_data);
                                }
                            }
                        }

                        if let Ok(mut buf) = buffer_clone.lock() {
                            for sample in data.iter_mut() {
                                *sample = if !buf.is_empty() {
                                    buf.remove(0) as f32 / 32768.0
                                } else {
                                    0.0
                                };
                            }
                        }
                    },
                    err_fn,
                    None,
                )?,
                _ => return Err(anyhow::anyhow!("Unsupported sample format")),
            };

            stream.play()?;
            Ok(Self {
                _stream: stream,
                sender,
            })
        }

        pub fn play(&self, samples: Vec<i16>) {
            if let Err(e) = self.sender.send(samples) {
                warn!("Failed to send audio samples: {}", e);
            }
        }
    }
}

#[cfg(feature = "playback")]
use audio_player::AudioPlayer;

/// Default reference voice path (TARS from Interstellar).
const DEFAULT_VOICE_PATH: &str = "asset/interstellar-tars-01-resemble-denoised.wav";

/// CLI arguments for the client.
#[derive(Parser, Debug)]
#[command(name = "cosyvoice-client")]
#[command(about = "CosyVoice TTS client with real-time audio streaming")]
struct Args {
    /// Text to synthesize
    #[arg(long, short = 't')]
    text: String,

    /// Server address
    #[arg(long, default_value = "http://localhost:50051")]
    server: String,

    /// Output WAV file path
    #[arg(long, short = 'o')]
    output: Option<PathBuf>,

    /// Reference voice audio file (for zero-shot cloning)
    #[arg(long, short = 'v')]
    voice: Option<PathBuf>,

    /// Prompt text (what the reference voice says, for zero-shot mode)
    #[arg(long, short = 'p', default_value = "")]
    prompt_text: String,

    /// Inference mode: zero-shot, cross-lingual, sft, instruct
    #[arg(long, short = 'm', default_value = "zero-shot")]
    mode: String,

    /// Speaker ID (for sft/instruct modes)
    #[arg(long)]
    speaker: Option<String>,

    /// Instruction text (for instruct mode)
    #[arg(long)]
    instruct: Option<String>,

    /// Disable audio playback
    #[arg(long)]
    no_play: bool,

    /// List available speakers and exit
    #[arg(long)]
    list_speakers: bool,

    /// Health check and exit
    #[arg(long)]
    health: bool,
}

/// Load a WAV file and resample to 16kHz mono.
fn load_and_resample_wav(path: &PathBuf) -> Result<Vec<u8>> {
    let file = File::open(path).context("Failed to open voice file")?;
    let reader = WavReader::new(BufReader::new(file)).context("Failed to read WAV header")?;
    let spec = reader.spec();

    info!(
        "Loading voice file: {} Hz, {} channels, {} bits",
        spec.sample_rate, spec.channels, spec.bits_per_sample
    );

    // Read all samples as f32
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => reader
            .into_samples::<i16>()
            .map(|s| s.map(|v| v as f32 / 32768.0))
            .collect::<Result<Vec<_>, _>>()?,
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .collect::<Result<Vec<_>, _>>()?,
    };

    // Convert to mono if stereo
    let mono_samples: Vec<f32> = if spec.channels == 2 {
        samples
            .chunks(2)
            .map(|chunk| (chunk[0] + chunk[1]) / 2.0)
            .collect()
    } else {
        samples
    };

    // Resample to 16kHz if needed
    let target_rate = 16000;
    let resampled = if spec.sample_rate != target_rate {
        info!(
            "Resampling from {} Hz to {} Hz",
            spec.sample_rate, target_rate
        );

        let mut resampler = FftFixedIn::<f32>::new(
            spec.sample_rate as usize,
            target_rate as usize,
            mono_samples.len().min(1024),
            1,
            1,
        )?;

        let waves_in = vec![mono_samples];
        let waves_out = resampler.process(&waves_in, None)?;
        waves_out.into_iter().next().unwrap_or_default()
    } else {
        mono_samples
    };

    // Convert to 16-bit PCM bytes
    let pcm_bytes: Vec<u8> = resampled
        .iter()
        .flat_map(|&s| {
            let sample = (s * 32767.0).clamp(-32768.0, 32767.0) as i16;
            sample.to_le_bytes()
        })
        .collect();

    info!("Voice file loaded: {} bytes of 16kHz PCM", pcm_bytes.len());
    Ok(pcm_bytes)
}

/// Perform health check.
async fn health_check(client: &mut CosyVoiceServiceClient<Channel>) -> Result<()> {
    let response = client.health_check(HealthCheckRequest {}).await?;
    let health = response.into_inner();

    println!("Health Check:");
    println!("  Healthy: {}", health.healthy);
    println!("  Model: {}", health.model_name);
    println!("  GPU Available: {}", health.gpu_available);
    if health.gpu_available {
        println!("  GPU Name: {}", health.gpu_name);
        println!("  GPU Memory: {} MiB", health.gpu_memory_mb);
    }

    Ok(())
}

/// List available speakers.
async fn list_speakers(client: &mut CosyVoiceServiceClient<Channel>) -> Result<()> {
    let response = client.list_speakers(ListSpeakersRequest {}).await?;
    let speakers = response.into_inner();

    println!("Available Speakers:");
    for speaker in speakers.speakers {
        println!("  - {}", speaker);
    }

    Ok(())
}

/// Build inference request based on mode.
fn build_request(args: &Args, voice_audio: Vec<u8>) -> Result<InferenceRequest> {
    let mode = match args.mode.as_str() {
        "zero-shot" | "zeroshot" => {
            let prompt_text = if args.prompt_text.is_empty() {
                "This is a sample voice.".to_string()
            } else {
                args.prompt_text.clone()
            };

            Mode::ZeroShot(ZeroShotMode {
                prompt_text,
                prompt_audio: voice_audio,
            })
        }
        "cross-lingual" | "crosslingual" | "cross" => {
            Mode::CrossLingual(CrossLingualMode {
                prompt_audio: voice_audio,
            })
        }
        "sft" => {
            let speaker_id = args
                .speaker
                .clone()
                .unwrap_or_else(|| "中文女".to_string());
            Mode::Sft(SftMode { speaker_id })
        }
        "instruct" => {
            let speaker_id = args
                .speaker
                .clone()
                .unwrap_or_else(|| "中文女".to_string());
            let instruct_text = args
                .instruct
                .clone()
                .unwrap_or_else(|| "Speak naturally.".to_string());
            Mode::Instruct(InstructMode {
                speaker_id,
                instruct_text,
            })
        }
        _ => return Err(anyhow::anyhow!("Unknown mode: {}", args.mode)),
    };

    Ok(InferenceRequest {
        text: args.text.clone(),
        speed: 1.0,
        stream: true,
        text_frontend: true,
        mode: Some(mode),
    })
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("cosyvoice_client=info".parse().unwrap()),
        )
        .init();

    let args = Args::parse();

    info!("Connecting to server: {}", args.server);
    let mut client = CosyVoiceServiceClient::connect(args.server.clone())
        .await
        .context("Failed to connect to server")?;

    // Handle special commands
    if args.health {
        return health_check(&mut client).await;
    }

    if args.list_speakers {
        return list_speakers(&mut client).await;
    }

    // Load reference voice
    let voice_path = args
        .voice
        .clone()
        .unwrap_or_else(|| PathBuf::from(DEFAULT_VOICE_PATH));

    info!("Loading reference voice: {:?}", voice_path);
    let voice_audio = load_and_resample_wav(&voice_path)?;

    // Build request
    let request = build_request(&args, voice_audio)?;
    info!("Sending inference request (mode: {})", args.mode);

    // Start streaming
    let response = client.inference(request).await?;
    let mut stream = response.into_inner();

    // Setup audio player (only if playback feature is enabled)
    #[cfg(feature = "playback")]
    let player = if !args.no_play {
        match AudioPlayer::new() {
            Ok(p) => Some(p),
            Err(e) => {
                warn!("Failed to initialize audio player: {}. Audio playback disabled.", e);
                None
            }
        }
    } else {
        None
    };

    #[cfg(not(feature = "playback"))]
    {
        if !args.no_play {
            warn!("Audio playback not available. Build with --features playback to enable.");
            warn!("Use --output to save audio to a WAV file instead.");
        }
    }

    // Setup WAV writer
    let wav_spec = WavSpec {
        channels: 1,
        sample_rate: 22050,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut wav_writer = args
        .output
        .as_ref()
        .map(|path| {
            WavWriter::create(path, wav_spec).context("Failed to create WAV file")
        })
        .transpose()?;

    let mut total_samples = 0usize;
    let mut chunk_count = 0u32;

    // Process audio stream
    while let Some(result) = stream.next().await {
        let chunk = result?;

        if chunk.is_final {
            info!("Received final chunk");
            break;
        }

        if chunk.audio_data.is_empty() {
            continue;
        }

        // Convert bytes to i16 samples
        let samples: Vec<i16> = chunk
            .audio_data
            .chunks_exact(2)
            .map(|bytes| i16::from_le_bytes([bytes[0], bytes[1]]))
            .collect();

        total_samples += samples.len();
        chunk_count += 1;

        // Play audio (only if playback feature is enabled)
        #[cfg(feature = "playback")]
        if let Some(ref player) = player {
            player.play(samples.clone());
        }

        // Write to WAV
        if let Some(ref mut writer) = wav_writer {
            for sample in &samples {
                writer.write_sample(*sample)?;
            }
        }

        if chunk_count.is_multiple_of(10) {
            info!(
                "Received {} chunks, {} samples ({:.2}s of audio)",
                chunk_count,
                total_samples,
                total_samples as f64 / 22050.0
            );
        }
    }

    // Finalize WAV file
    if let Some(writer) = wav_writer {
        writer.finalize()?;
        info!("Saved audio to {:?}", args.output.unwrap());
    }

    info!(
        "Done! Total: {} samples ({:.2}s of audio)",
        total_samples,
        total_samples as f64 / 22050.0
    );

    // Wait a bit for audio to finish playing
    #[cfg(feature = "playback")]
    if player.is_some() {
        tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
    }

    Ok(())
}
