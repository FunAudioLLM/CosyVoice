import argparse
import os
from loguru import logger
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from file import list_files

def process_filelist(filelist, output_dir):
    """
    Processes the filelist to create wav.scp, text, utt2spk, and spk2utt files.
    """
    utt2wav, utt2text, utt2spk, spk2utt = {}, {}, {}, {}

    for line in tqdm(filelist):
        wav, spk, text = line.split('|')
        utt = os.path.basename(wav).replace('.wav', '')
        utt2wav[utt] = wav
        utt2text[utt] = text
        utt2spk[utt] = spk
        if spk not in spk2utt:
            spk2utt[spk] = []
        spk2utt[spk].append(utt)

    os.makedirs(output_dir, exist_ok=True)
    
    with open(f'{output_dir}/wav.scp', 'w', encoding='utf-8') as f:
        for utt, wav in utt2wav.items():
            f.write(f'{utt} {wav}\n')

    with open(f'{output_dir}/text', 'w', encoding='utf-8') as f:
        for utt, text in utt2text.items():
            f.write(f'{utt} {text}\n')

    with open(f'{output_dir}/utt2spk', 'w', encoding='utf-8') as f:
        for utt, spk in utt2spk.items():
            f.write(f'{utt} {spk}\n')

    with open(f'{output_dir}/spk2utt', 'w', encoding='utf-8') as f:
        for spk, utts in spk2utt.items():
            f.write(f'{spk} {" ".join(utts)}\n')

    logger.success(f'Created utt2wav, utt2text, utt2spk, and spk2utt files in {output_dir}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str)
    parser.add_argument("output_dir", type=str)
    parser.add_argument("--valid_size", type=float, default=0.05, help="Proportion of validation set size")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f'Loading metadata from {args.data_dir}')
    files = list_files(args.data_dir, extensions=[".txt"], recursive=True)
    
    train_filelist = []
    valid_filelist = []
    
    for file in files:
        # if 'vivoice' in str(file) or 'voice_clone' in str(file):
        #     continue
        # if 'data_bac_nam' not in str(file):
        #     continue
        if file.name == "metadata.txt":
            logger.info(f'Loading {file}')
            subset_filelist = []
            with open(file, "r", encoding="utf-8") as f:
                for line in f.readlines():
                    line = line.strip().split("|")
                    if len(line) != 3:
                        print(line)
                        continue
                    if not os.path.exists(line[1]):
                        print(f"File {line[1]} not exist")
                        continue
                    if 'vivos' in str(file):
                        line[2] = line[2].lower().strip()
                    text = line[2]
                    if not text.endswith("."):
                        text += "."
                    text = text.replace(" .", ".").replace(" ,", ",")
                    line = f"{line[1]}|{line[0]}|{text}"
                    subset_filelist.append(line)
            
            # Split the current subset into train and valid sets
            train_subset, valid_subset = train_test_split(subset_filelist, test_size=args.valid_size, random_state=42)
            train_filelist.extend(train_subset)
            valid_filelist.extend(valid_subset)

    # Save the training and validation file lists
    with open(os.path.join(args.output_dir, "train_filelist.txt"), "w", encoding="utf-8") as f:
        for file in train_filelist:
            f.write(file + "\n")

    with open(os.path.join(args.output_dir, "valid_filelist.txt"), "w", encoding="utf-8") as f:
        for file in valid_filelist:
            f.write(file + "\n")

    # Process training and validation filelists to create utt2wav, utt2text, utt2spk, and spk2utt files
    logger.info('Processing train filelist...')
    process_filelist(train_filelist, os.path.join(args.output_dir, "train"))

    logger.info('Processing valid filelist...')
    process_filelist(valid_filelist, os.path.join(args.output_dir, "valid"))

    logger.success(f'Finished processing training and validation datasets.')

if __name__ == "__main__":
    main()