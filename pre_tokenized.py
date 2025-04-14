import os
import sys
import torch
import torchaudio
from cosyvoice.cli.frontend import CosyVoiceFrontEnd
from cosyvoice.utils.file_utils import load_wav
from hyperpyyaml import load_hyperpyyaml

import argparse

def main(model_dir, name):
    with open('{}/cosyvoice.yaml'.format(model_dir), 'r') as f:
                configs = load_hyperpyyaml(f, overrides={'qwen_pretrain_path': os.path.join(model_dir, 'CosyVoice-BlankEN')})
    frontend = CosyVoiceFrontEnd(configs['get_tokenizer'],
        configs['feat_extractor'],
        '{}/campplus.onnx'.format(model_dir),
        '{}/speech_tokenizer_v2.onnx'.format(model_dir),
        '{}/spk2info.pt'.format(model_dir),
        configs['allowed_special'])

    prompt_speech_16k = load_wav('./voice/{}/prompt.wav'.format(name), 16000)
    with open('./voice/{}/prompt.txt'.format(name), 'r') as f:
        prompt_text = f.read().strip()
    frontend.save_cache(prompt_text, prompt_speech_16k, configs['sample_rate'], './prompt_wav_cache/{}.pt'.format(name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir',
                        type=str,
                        default='pretrained_models/CosyVoice2-0.5B-trt',
                        help='local path or modelscope repo id')
    parser.add_argument('--name',
                        type=str,
                        help='name of the voice')
    args = parser.parse_args()
    main(args.model_dir, args.name)