//! CosyVoice gRPC protocol definitions.
//!
//! This crate provides the generated Rust types and gRPC service definitions
//! for the CosyVoice TTS service.

#![allow(clippy::derive_partial_eq_without_eq)]

mod cosyvoice;

pub use cosyvoice::*;
