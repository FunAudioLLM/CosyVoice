import argparse
import logging
import os
from tqdm import tqdm


logger = logging.getLogger()


def main():
    utt2wav, utt2text, utt2spk, spk2utt = {}, {}, {}, {}
    with open(os.path.join(args.src_dir, "TRANS.txt"), "r") as f:
        lines = f.readlines()[1:]
        lines = [l.split('\t') for l in lines]
    for wav, spk, content in tqdm(lines):
        wav, spk, content = wav.strip(), spk.strip(), content.strip()
        content = content.replace('[FIL]', '')
        content = content.replace('[SPK]', '')
        wav = os.path.join(args.src_dir, spk, wav)
        if not os.path.exists(wav):
            continue
        utt = os.path.basename(wav).replace('.wav', '')
        utt2wav[utt] = wav
        utt2text[utt] = content
        utt2spk[utt] = spk
        if spk not in spk2utt:
            spk2utt[spk] = []
        spk2utt[spk].append(utt)

    with open('{}/wav.scp'.format(args.des_dir), 'w') as f:
        for k, v in utt2wav.items():
            f.write('{} {}\n'.format(k, v))
    with open('{}/text'.format(args.des_dir), 'w') as f:
        for k, v in utt2text.items():
            f.write('{} {}\n'.format(k, v))
    with open('{}/utt2spk'.format(args.des_dir), 'w') as f:
        for k, v in utt2spk.items():
            f.write('{} {}\n'.format(k, v))
    with open('{}/spk2utt'.format(args.des_dir), 'w') as f:
        for k, v in spk2utt.items():
            f.write('{} {}\n'.format(k, ' '.join(v)))
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir',
                        type=str)
    parser.add_argument('--des_dir',
                        type=str)
    args = parser.parse_args()
    main()
