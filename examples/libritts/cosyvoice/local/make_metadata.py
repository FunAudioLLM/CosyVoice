import os
from loguru import logger
from tqdm import tqdm
import argparse
import soundfile
import shutil

from transcribe import transcribe_audio_faster_whisper
from file import list_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str)
    args = parser.parse_args()

    logger.info(f'Loading metadata from {args.data_dir}')
    files = list_files(args.data_dir, extensions=[".txt"], recursive=True)
    
    for i, file in enumerate(files):
        if 'vivoice' in str(file):
            continue
        if 'voice_clone' not in str(file):
            continue
        if file.name == "metadata.txt":
            logger.info(f'[{i+1}/{len(files)}] Processing {file}...')
            with open(file, "r", encoding="utf-8") as f, open(str(file).replace('.txt', '_whisper2.txt'), 'w', encoding='utf-8') as f2:
                for line in tqdm(f.readlines()):
                    line = line.strip().split("|")
                    if len(line) != 3:
                        print(line)
                        continue
                    filepath = line[1]
                    filepath = filepath.replace('/home/andrew/data/tts', '/data/tts')
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
                    if y.shape[0]/ sr > 15:
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
                    line = f"{filepath}|{line[0]}|{text}\n"
                    f2.write(line)
