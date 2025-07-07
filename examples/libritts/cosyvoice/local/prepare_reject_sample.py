import argparse
import logging
import os
from tqdm import tqdm
import torch
import torchaudio
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav


logger = logging.getLogger()


def main():
    cosyvoice = CosyVoice2(args.ref_model)

    utt2wav, utt2text = {}, {}
    with open('{}/wav.scp'.format(args.src_dir)) as f:
        for l in f:
            l = l.split('\n')[0].split()
            utt2wav[l[0]] = l[1]
    with open('{}/text'.format(args.src_dir)) as f:
        for l in f:
            l = l.split('\n')[0].split()
            utt2text[l[0]] = ' '.join(l[1:])

    os.makedirs('{}/wav'.format(args.des_dir), exist_ok=True)
    with open('{}/wav.scp'.format(args.des_dir), 'w') as f:
        for utt, wav in tqdm(utt2wav.items()):
            prompt_speech_16k = load_wav(wav, 16000)
            if prompt_speech_16k.shape[1] >= 30 * 16000:
                continue
            speech_list = []
            for _, j in enumerate(cosyvoice.inference_zero_shot(utt2text[utt], utt2text[utt], prompt_speech_16k, stream=False, text_frontend=False)):
                speech_list.append(j['tts_speech'])
            negative_wav = os.path.abspath('{}/wav/{}'.format(args.des_dir, os.path.basename(wav)))
            torchaudio.save(negative_wav, torch.concat(speech_list, dim=1), cosyvoice.sample_rate, backend='soundfile')
            f.write('{} {}\n'.format(utt, negative_wav))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir',
                        type=str)
    parser.add_argument('--des_dir',
                        type=str)
    parser.add_argument('--ref_model',
                        type=str)
    args = parser.parse_args()
    main()
