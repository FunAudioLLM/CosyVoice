#!/usr/bin/env python3
# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
# Copyright (c) 2025 Voice Design Extension
#
# Data preparation script with voice description support.

import argparse
import glob
import json
import logging
import os
from tqdm import tqdm

logger = logging.getLogger()


def load_descriptions(description_file: str) -> dict:
    """
    Load voice descriptions from JSONL file.
    Supports multiple descriptions for the same file using strictly 'description' field.
    """
    from collections import defaultdict
    path2descs = defaultdict(list)
    
    if not os.path.exists(description_file):
        logger.warning(f"Description file not found: {description_file}")
        return path2descs
    
    with open(description_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                if 'filename' in data and 'description' in data:
                    full_path = data['filename']
                    path_parts = full_path.split(os.sep)
                    if len(path_parts) >= 3:
                        key = os.path.join(path_parts[-3], path_parts[-2], path_parts[-1])
                    elif len(path_parts) == 2:
                        key = os.path.join(path_parts[-2], path_parts[-1])
                    else:
                        key = path_parts[-1]
                    
                    if len(path2descs) < 5:
                        print(f"DEBUG JSONL Key: {key}")

                    desc = data['description']
                    if desc:
                        path2descs[key].append(desc)
            except json.JSONDecodeError:
                continue
    
    logger.info(f"Loaded {sum(len(v) for v in path2descs.values())} descriptions for {len(path2descs)} files from {description_file}")
    return path2descs


def generate_default_description(spk_id: str) -> str:
    return f"Speaker {spk_id}"


def main():
    args = parser.parse_args()
    
    src_dir_abs = os.path.abspath(args.src_dir)
    dataset_prefix = os.path.basename(src_dir_abs)
    if not dataset_prefix:
        dataset_prefix = os.path.basename(os.path.dirname(src_dir_abs))
    
    # Find all wav files
    wavs = list(glob.glob('{}/**/*wav'.format(args.src_dir), recursive=True))
    if not wavs:
        wavs = list(glob.glob('{}/*wav'.format(args.src_dir)))
    
    logger.info(f"Dataset Prefix: {dataset_prefix} | Found {len(wavs)} wav files")
    
    # Load descriptions (now returns a list for each key)
    base2descs = {}
    if args.with_description and args.description_file:
        base2descs = load_descriptions(args.description_file)
    
    utt2wav, utt2text, utt2spk, spk2utt, utt2description = {}, {}, {}, {}, {}
    
    for wav in tqdm(wavs, desc="Processing"):
        path_parts = wav.split(os.sep)
        basename = os.path.basename(wav).replace('.wav', '')
        parent_dir = path_parts[-2] if len(path_parts) > 1 else "default"
        
        # Base Unique ID parts
        base_utt = f"{dataset_prefix}_{parent_dir}_{basename}"
        orig_spk = basename.split('_')[0]
        spk = f"{dataset_prefix}_{parent_dir}_{orig_spk}"
        
        # Check for transcription
        txt = wav.replace('.wav', '.normalized.txt')
        if not os.path.exists(txt):
            txt = wav.replace('.wav', '.txt')
            if not os.path.exists(txt):
                continue
        
        with open(txt, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            
        # Match descriptions
        if len(path_parts) >= 3:
            match_key = os.path.join(path_parts[-3], path_parts[-2], path_parts[-1])
        elif len(path_parts) == 2:
            match_key = os.path.join(path_parts[-2], path_parts[-1])
        else:
            match_key = path_parts[-1]
            
        if len(wavs) < 5:
             print(f"DEBUG File Key: {match_key} (from {wav})")
        
        descs = []
        if args.with_description:
            if match_key in base2descs:
                descs = base2descs[match_key]
            elif os.path.basename(wav) in base2descs:
                descs = base2descs[os.path.basename(wav)]
            
            if not descs:
                descs = [generate_default_description(spk)]
        else:
            descs = [None] # Placeholder when not using descriptions
            
        # Create multiple entries if multiple descriptions exist
        for i, desc in enumerate(descs):
            utt = f"{base_utt}_v{i}" if len(descs) > 1 else base_utt
            
            utt2wav[utt] = wav
            # Revert prepending for content text, will use instruct field instead
            utt2text[utt] = content
            utt2spk[utt] = spk
            
            if spk not in spk2utt:
                spk2utt[spk] = []
            spk2utt[spk].append(utt)
            
            if desc:
                utt2description[utt] = desc
    
    # Create output directory
    os.makedirs(args.des_dir, exist_ok=True)
    
    # Write files
    with open(f'{args.des_dir}/wav.scp', 'w', encoding='utf-8') as f:
        for k, v in sorted(utt2wav.items()):
            f.write(f'{k} {v}\n')
    
    with open(f'{args.des_dir}/text', 'w', encoding='utf-8') as f:
        for k, v in sorted(utt2text.items()):
            f.write(f'{k} {v}\n')
    
    with open(f'{args.des_dir}/utt2spk', 'w', encoding='utf-8') as f:
        for k, v in sorted(utt2spk.items()):
            f.write(f'{k} {v}\n')
    
    with open(f'{args.des_dir}/spk2utt', 'w', encoding='utf-8') as f:
        for k, v in sorted(spk2utt.items()):
            f.write(f'{k} {" ".join(v)}\n')
    
    with open(f'{args.des_dir}/instruct', 'w', encoding='utf-8') as f:
        for k in sorted(utt2text.keys()):
            # Combine global instruct with per-utterance description
            desc = utt2description.get(k, '')
            combined_instruct = f"{args.instruct} {desc}".strip()
            if combined_instruct:
                f.write(f'{k} {combined_instruct}\n')
            else:
                f.write(f'{k} You are a helpful assistant\n')
    
    if args.with_description:
        with open(f'{args.des_dir}/utt2description', 'w', encoding='utf-8') as f:
            for k, v in sorted(utt2description.items()):
                f.write(f'{k}\t{v}\n')
        logger.info(f"Wrote {len(utt2description)} descriptions to utt2description")
    
    logger.info(f"Processed {len(utt2wav)} utterances")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare data with Multi-Description support')
    parser.add_argument('--src_dir', type=str, required=True, help='Source directory with wav files')
    parser.add_argument('--des_dir', type=str, required=True, help='Destination directory')
    parser.add_argument('--instruct', type=str, default='', help='Instruction text')
    parser.add_argument('--with_description', action='store_true', help='Enable voice description processing')
    parser.add_argument('--description_file', type=str, default='', help='JSONL file with descriptions')
    
    main()
