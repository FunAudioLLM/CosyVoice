#!/usr/bin/env python3
# Copyright (c) 2025 Voice Design Extension
#
# Licensed under the Apache License, Version 2.0

"""
Extract style embeddings from audio files.

This script extracts style embeddings from audio using CampPlus model.
These embeddings are used as ground truth (teacher signal) for training
the Description Encoder.

Usage:
    python tools/extract_style_embedding.py \
        --dir data/train \
        --model_path pretrained_models/Fun-CosyVoice3-0.5B/campplus.onnx \
        --output_file utt2style_embedding.pt
"""

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import onnxruntime as ort
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


def get_args():
    parser = argparse.ArgumentParser(description='Extract style embeddings from audio')
    parser.add_argument('--dir', required=True, help='Data directory containing wav.scp')
    parser.add_argument('--model_path', required=True, help='Path to CampPlus ONNX model')
    parser.add_argument('--output_file', default='utt2style_embedding.pt', help='Output filename')
    parser.add_argument('--sample_rate', default=16000, type=int, help='Target sample rate')
    parser.add_argument('--num_threads', default=4, type=int, help='Number of threads for ONNX')
    args = parser.parse_args()
    return args


class StyleExtractor:
    """Extract style embeddings using CampPlus ONNX model."""
    
    def __init__(self, model_path: str, num_threads: int = 4):
        """
        Initialize the style extractor.
        
        Args:
            model_path: Path to CampPlus ONNX model
            num_threads: Number of threads for inference
        """
        self.model_path = model_path
        
        # Set up ONNX session options
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = num_threads
        sess_options.inter_op_num_threads = num_threads
        
        # Create ONNX session
        providers = ['CPUExecutionProvider']
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            providers = [('CUDAExecutionProvider', {'device_id': 0})]

        self.session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=providers,
        )
        
        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        logging.info(f"Loaded CampPlus model from {model_path}")
    
    def extract(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """
        Extract style embedding from audio.
        
        Args:
            audio: Audio waveform numpy array [T]
            sample_rate: Sample rate (should be 16000)
            
        Returns:
            style_embedding: [192] numpy array
        """
        # Compute Fbank features
        # CampPlus expects [Batch, Time, 80]
        # torchaudio.compliance.kaldi.fbank returns [Time, 80]
        import torchaudio.compliance.kaldi as kaldi
        
        # Ensure waveform is tensor [1, T]
        waveform = torch.from_numpy(audio)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
            
        # Compute fbank
        # Standard params for Campplus: num_mel_bins=80, frame_length=25, frame_shift=10, dither=0.0
        feat = kaldi.fbank(waveform, num_mel_bins=80, frame_length=25, frame_shift=10, dither=0.0, energy_floor=0.0, sample_frequency=sample_rate)
        
        # Add batch dimension [1, Time, 80]
        feat = feat.unsqueeze(0).numpy()
        
        # Run inference
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: feat}
        )
        
        # Get embedding [1, 192] -> [192]
        embedding = outputs[0].squeeze(0)
        
        return embedding


def load_wav_scp(wav_scp_path: str) -> dict:
    """Load wav.scp file into dictionary."""
    utt2wav = {}
    with open(wav_scp_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                utt_id, wav_path = parts
                utt2wav[utt_id] = wav_path
    return utt2wav


def main():
    args = get_args()
    
    data_dir = Path(args.dir)
    wav_scp_path = data_dir / 'wav.scp'
    output_path = data_dir / args.output_file
    
    # Check inputs
    if not wav_scp_path.exists():
        raise FileNotFoundError(f"wav.scp not found at {wav_scp_path}")
    
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found at {args.model_path}")
    
    # Load wav.scp
    utt2wav = load_wav_scp(wav_scp_path)
    logging.info(f"Loaded {len(utt2wav)} utterances from {wav_scp_path}")
    
    # Initialize extractor
    extractor = StyleExtractor(args.model_path, args.num_threads)
    
    # Extract embeddings
    utt2embedding = {}
    failed_utts = []
    
    for utt_id, wav_path in tqdm(utt2wav.items(), desc="Extracting style embeddings"):
        try:
            # Load audio
            waveform, sr = torchaudio.load(wav_path)
            
            # Resample if needed
            if sr != args.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, args.sample_rate)
                waveform = resampler(waveform)
            
            # Convert to mono if needed
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            # Convert to numpy
            audio = waveform.squeeze(0).numpy()
            
            # Extract embedding
            embedding = extractor.extract(audio, args.sample_rate)
            utt2embedding[utt_id] = torch.from_numpy(embedding)
            
        except Exception as e:
            logging.warning(f"Failed to process {utt_id}: {e}")
            failed_utts.append(utt_id)
    
    # Save embeddings
    torch.save(utt2embedding, output_path)
    logging.info(f"Saved {len(utt2embedding)} embeddings to {output_path}")
    
    if failed_utts:
        logging.warning(f"Failed to process {len(failed_utts)} utterances: {failed_utts[:10]}...")


if __name__ == '__main__':
    main()
