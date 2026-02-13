#!/usr/bin/env python3
# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
# Copyright (c) 2025 Voice Design Extension
#
# Make parquet files with voice descriptions and style embeddings.

"""
Create parquet files with voice descriptions and style embeddings.

Usage:
    python tools/make_parquet_list_voice_design.py \
        --src_dir data/train \
        --des_dir data/train/parquet \
        --include_description \
        --include_style_embedding
"""

import argparse
import logging
import os
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


def get_args():
    parser = argparse.ArgumentParser(description='Make parquet files with voice design features')
    parser.add_argument('--src_dir', required=True, help='Source directory')
    parser.add_argument('--des_dir', required=True, help='Destination directory')
    parser.add_argument('--num_utts_per_parquet', default=1000, type=int, help='Utterances per parquet')
    parser.add_argument('--num_processes', default=10, type=int, help='Number of processes')
    parser.add_argument('--include_description', action='store_true', help='Include voice descriptions')
    parser.add_argument('--include_style_embedding', action='store_true', help='Include style embeddings')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing parquet files')
    args = parser.parse_args()
    return args


def load_scp(scp_path: str) -> dict:
    """Load SCP file into dictionary."""
    result = {}
    if not os.path.exists(scp_path):
        return result
    with open(scp_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                result[parts[0]] = parts[1]
    return result


def load_tsv(tsv_path: str) -> dict:
    """Load TSV file (utt2description) into dictionary."""
    result = {}
    if not os.path.exists(tsv_path):
        return result
    with open(tsv_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t', maxsplit=1)  # Use TAB delimiter
            if len(parts) == 2:
                result[parts[0]] = parts[1]
    return result


def load_pt(pt_path: str) -> dict:
    """Load PyTorch embedding file."""
    if not os.path.exists(pt_path):
        return {}
    return torch.load(pt_path)


def process_batch(args_tuple):
    """Process a batch of utterances into parquet with strict schema."""
    utt_list, utt2wav, utt2text, utt2spk, utt2description, utt2style, utt2speech_token, utt2instruct, output_path, include_desc, include_style, schema = args_tuple
    
    records = []
    for utt in utt_list:
        if utt not in utt2wav:
            continue
        
        # Base record
        record = {
            'utt': utt,
            'wav': utt2wav[utt],
            'text': utt2text.get(utt, ''),
            'spk': utt2spk.get(utt, ''),
            'instruct': utt2instruct.get(utt, ''),
            'description': '',
            'audio_data': b'',
            'style_embedding': [],
            'spk_embedding': [],
            'utt_embedding': [],
            'speech_token': []
        }
        
        # Read audio data
        try:
            with open(utt2wav[utt], 'rb') as f:
                record['audio_data'] = f.read()
        except Exception as e:
            logging.warning(f"Failed to read audio for {utt}: {e}")
            continue
            
        if include_desc:
            record['description'] = utt2description.get(utt, 'A neutral voice')
        
        if include_style and utt in utt2style:
            emb = utt2style[utt]
            if isinstance(emb, torch.Tensor):
                emb = emb.numpy()
            elif isinstance(emb, list):
                emb = np.array(emb, dtype=np.float32)
            
            # Use tolist() to avoid binary blobs and np.frombuffer during load
            emb_list = emb.tolist()
            record['style_embedding'] = emb_list
            record['spk_embedding'] = emb_list
            record['utt_embedding'] = emb_list
            
        if utt2speech_token and utt in utt2speech_token:
            token = utt2speech_token[utt]
            if isinstance(token, torch.Tensor):
                token = token.numpy()
            if isinstance(token, (np.ndarray, list)):
                record['speech_token'] = token.tolist()
        
        records.append(record)
    
    if records:
        # Create pandas DataFrame and save to parquet with strict schema
        df = pd.DataFrame(records)
        # Use pyarrow engine and pass the strict schema
        df.to_parquet(output_path, engine='pyarrow', schema=schema, index=False)
        return len(records)
    return 0


def main():
    args = get_args()
    
    src_dir = Path(args.src_dir)
    des_dir = Path(args.des_dir)
    if args.overwrite and des_dir.exists():
        logging.info(f"Overwriting {des_dir}, deleting existing parquet files...")
        for f in des_dir.glob('*.parquet'):
            f.unlink()
        if (des_dir / 'data.list').exists():
            (des_dir / 'data.list').unlink()
    des_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data files
    utt2wav = load_scp(src_dir / 'wav.scp')
    utt2text = load_scp(src_dir / 'text')
    utt2spk = load_scp(src_dir / 'utt2spk')
    utt2instruct = load_scp(src_dir / 'instruct')
    
    utt2description = {}
    if args.include_description:
        utt2description = load_tsv(src_dir / 'utt2description')
        logging.info(f"Loaded {len(utt2description)} descriptions")
    
    utt2style = {}
    if args.include_style_embedding:
        utt2style = load_pt(src_dir / 'utt2style_embedding.pt')
        logging.info(f"Loaded {len(utt2style)} style embeddings")
        
    utt2speech_token = {}
    speech_token_path = src_dir / 'utt2speech_token.pt'
    if speech_token_path.exists():
        utt2speech_token = load_pt(speech_token_path)
        logging.info(f"Loaded {len(utt2speech_token)} speech tokens")
    
    # Get all utterances
    all_utts = list(utt2wav.keys())
    logging.info(f"Total utterances: {len(all_utts)}")
    
    # Define strict schema for Parquet files
    schema = pa.schema([
        ('utt', pa.string()),
        ('wav', pa.string()),
        ('text', pa.string()),
        ('spk', pa.string()),
        ('instruct', pa.string()),
        ('audio_data', pa.binary()),
        ('description', pa.string()),
        ('style_embedding', pa.list_(pa.float32())),
        ('spk_embedding', pa.list_(pa.float32())),
        ('utt_embedding', pa.list_(pa.float32())),
        ('speech_token', pa.list_(pa.int32())),
    ])
    
    # Split into batches
    batches = []
    for i in range(0, len(all_utts), args.num_utts_per_parquet):
        batch_utts = all_utts[i:i + args.num_utts_per_parquet]
        batch_idx = i // args.num_utts_per_parquet
        output_path = des_dir / f'part_{batch_idx:05d}.parquet'
        
        batches.append((
            batch_utts, utt2wav, utt2text, utt2spk,
            utt2description, utt2style, utt2speech_token, utt2instruct, str(output_path),
            args.include_description, args.include_style_embedding, schema
        ))
    
    # Process in parallel
    total_processed = 0
    with ProcessPoolExecutor(max_workers=args.num_processes) as executor:
        results = list(tqdm(
            executor.map(process_batch, batches),
            total=len(batches),
            desc="Creating parquet files"
        ))
        total_processed = sum(results)
    
    # Write data.list
    data_list_path = des_dir / 'data.list'
    with open(data_list_path, 'w') as f:
        for batch_idx in range(len(batches)):
            parquet_path = des_dir / f'part_{batch_idx:05d}.parquet'
            if parquet_path.exists():
                f.write(f'{parquet_path}\n')
    
    logging.info(f"Processed {total_processed} utterances into {len(batches)} parquet files")
    logging.info(f"Data list: {data_list_path}")


if __name__ == '__main__':
    main()
