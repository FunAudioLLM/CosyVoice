import argparse
import glob
import os

import numpy as np
from pystoi import stoi
from scipy.io import wavfile
from tqdm import tqdm


def calculate_stoi(ref_dir, deg_dir):
    input_files = glob.glob(f"{deg_dir}/*.wav")
    if len(input_files) < 1:
        raise RuntimeError(f"Found no wavs in {ref_dir}")

    stoi_scores = []
    for deg_wav in tqdm(input_files):
        ref_wav = os.path.join(ref_dir, os.path.basename(deg_wav))
        rate, ref = wavfile.read(ref_wav)
        rate, deg = wavfile.read(deg_wav)
        min_len = min(len(ref), len(deg))
        ref = ref[:min_len]
        deg = deg[:min_len]
        cur_stoi = stoi(ref, deg, rate, extended=False)
        stoi_scores.append(cur_stoi)

    return np.mean(stoi_scores)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Compute STOI measure")

    parser.add_argument(
        '-r', '--ref_dir', required=True, help="Reference wave folder.")
    parser.add_argument(
        '-d', '--deg_dir', required=True, help="Degraded wave folder.")

    args = parser.parse_args()

    stoi_score = calculate_stoi(args.ref_dir, args.deg_dir)
    print(f"STOI: {stoi_score}")
