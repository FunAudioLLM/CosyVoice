import argparse
import glob
import json
import os
from pathlib import Path

import soundfile as sf
from tqdm import tqdm

from academicodec.models.hificodec.vqvae_tester import VqvaeTester

parser = argparse.ArgumentParser()

#Path
parser.add_argument('--outputdir', type=str, required=True)
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--input_wavdir', type=str, required=True)
parser.add_argument('--config_path', type=str, required=True)
parser.add_argument('--num_gens', type=int, default=1024)

#Data
parser.add_argument('--sample_rate', type=int, default=24000)

args = parser.parse_args()

with open(args.config_path, 'r') as f:
    argdict = json.load(f)
    assert argdict['sampling_rate'] == args.sample_rate, \
        f"Sampling rate not consistent, stated {args.sample_rate}, but the model is trained on {argdict['sample_rate']}"
    argdict.update(args.__dict__)
    args.__dict__ = argdict

if __name__ == '__main__':
    Path(args.outputdir).mkdir(parents=True, exist_ok=True)
    print("Init model and load weights")
    model = VqvaeTester(config_path=args.config_path, model_path=args.model_path,sample_rate=args.sample_rate)
    model.cuda()
    model.vqvae.generator.remove_weight_norm()
    model.vqvae.encoder.remove_weight_norm()
    model.eval()
    print("Model ready")

    wav_paths = glob.glob(f"{args.input_wavdir}/*.wav")[:args.num_gens]
    print(f"Globbed {len(wav_paths)} wav files.")

    for wav_path in wav_paths:
        fid, wav = model(wav_path)
        wav = wav.squeeze().cpu().numpy()
        sf.write(
            os.path.join(args.outputdir, f'{fid}.wav'), wav, args.sample_rate)
