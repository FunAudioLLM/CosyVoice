import os
from loguru import logger
from tqdm import tqdm
import argparse
import soundfile

from transcribe import transcribe_audio_faster_whisper
from file import list_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str)
    args = parser.parse_args()

    logger.info(f'Loading metadata from {args.data_dir}')
    files = list_files(args.data_dir, extensions=[".txt"], recursive=True)
    
    for i, file in enumerate(files):
        if 'viet_bud500' in str(file) \
            or 'YODAS_vi000' in str(file) \
            or 'MSR-86K' in str(file) \
            or 'fpt' in str(file) \
            or 'fleurs' in str(file) \
            or 'meeting_' in str(file):
            continue
        if 'ctv_12_2021' not in str(file):
            continue
        if file.name == "transcripts.txt":
            logger.info(f'[{i+1}/{len(files)}] Processing {file}...')
            with open(file, "r", encoding="utf-8") as f, open(str(file).replace('.txt', '_whisper.txt'), 'w', encoding='utf-8') as f2:
                for line in tqdm(f.readlines()):
                    line = line.strip().split("|")
                    filepath = line[0]
                    org_text = line[1]
                    filepath = filepath.replace('/data/raw/train', '/home/andrew/data/asr')
                    if not os.path.exists(filepath):
                        print(f"File {filepath} not exist")
                        continue
                    try:
                        y, sr = soundfile.read(filepath)
                    except:
                        print(f"File {filepath} cannot be read")
                        continue
                    if len(y.shape) != 1:
                        print(f"File {filepath} is not mono")
                        continue
                    dur = y.shape[0] / sr
                    if y.shape[0]/ sr > 20:
                        print(f"File {filepath} is too long")
                        continue
                    text = None
                    for _ in range(3):
                        try:
                            text = transcribe_audio_faster_whisper(filepath)
                            break
                        except:
                            print(f"File {filepath} cannot be transcribed")
                    if text is None:
                        continue
                    if not text.endswith("."):
                        text += "."
                    text = text.replace(" .", ".").replace(" ,", ",")
                    line = f"{filepath}|{text}|{org_text}|{dur:4f}\n"
                    f2.write(line)
