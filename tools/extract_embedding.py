#!/usr/bin/env python3
# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# you may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import onnxruntime
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from tqdm import tqdm


def single_job(utt, utt2wav, ort_session):
    try:
        audio, sample_rate = torchaudio.load(utt2wav[utt])
        if sample_rate != 16000:
            audio = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(audio)
        
        feat = kaldi.fbank(audio, num_mel_bins=80, dither=0, sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)
        embedding = ort_session.run(None, {ort_session.get_inputs()[0].name: feat.unsqueeze(dim=0).cpu().numpy()})[0].flatten().tolist()
        return utt, embedding
    except Exception as e:
        print(f"Error processing {utt}: {e}")
        return utt, None


def main(args, utt2wav, utt2spk, ort_session):
    utt2embedding, spk2embedding = {}, {}
    all_task = [executor.submit(single_job, utt, utt2wav, ort_session) for utt in utt2wav.keys()]
    
    for future in tqdm(as_completed(all_task)):
        utt, embedding = future.result()
        if embedding is not None:
            utt2embedding[utt] = embedding
            spk = utt2spk[utt]
            if spk not in spk2embedding:
                spk2embedding[spk] = []
            spk2embedding[spk].append(embedding)

    for k, v in spk2embedding.items():
        spk2embedding[k] = torch.tensor(v).mean(dim=0).tolist()
    
    torch.save(utt2embedding, f"{args.dir}/utt2embedding.pt")
    torch.save(spk2embedding, f"{args.dir}/spk2embedding.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True)
    parser.add_argument("--onnx_path", type=str, required=True)
    parser.add_argument("--num_thread", type=int, default=8)
    args = parser.parse_args()

    utt2wav, utt2spk = {}, {}
    with open(f'{args.dir}/wav.scp') as f:
        for l in f:
            l = l.strip().split()
            if len(l) == 2:
                utt2wav[l[0]] = l[1]
    a#!/usr/bin/env python3
# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# you may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import onnxruntime
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from tqdm import tqdm


def single_job(utt, utt2wav, ort_session):
    try:
        audio, sample_rate = torchaudio.load(utt2wav[utt])
        if sample_rate != 16000:
            audio = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(audio)
        
        feat = kaldi.fbank(audio, num_mel_bins=80, dither=0, sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)
        embedding = ort_session.run(None, {ort_session.get_inputs()[0].name: feat.unsqueeze(dim=0).cpu().numpy()})[0].flatten().tolist()
        return utt, embedding
    except Exception as e:
        print(f"Error processing {utt}: {e}")
        return utt, None


def main(args, utt2wav, utt2spk, ort_session):
    utt2embedding, spk2embedding = {}, {}
    all_task = [executor.submit(single_job, utt, utt2wav, ort_session) for utt in utt2wav.keys()]
    
    for future in tqdm(as_completed(all_task)):
        utt, embedding = future.result()
        if embedding is not None:
            utt2embedding[utt] = embedding
            spk = utt2spk[utt]
            if spk not in spk2embedding:
                spk2embedding[spk] = []
            spk2embedding[spk].append(embedding)

    for k, v in spk2embedding.items():
        spk2embedding[k] = torch.tensor(v).mean(dim=0).tolist()
    
    torch.save(utt2embedding, f"{args.dir}/utt2embedding.pt")
    torch.save(spk2embedding, f"{args.dir}/spk2embedding.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True)
    parser.add_argument("--onnx_path", type=str, required=True)
    parser.add_argument("--num_thread", type=int, default=8)
    args = parser.parse_args()

    utt2wav, utt2spk = {}, {}
    with open(f'{args.dir}/wav.scp') as f:
        for l in f:
            l = l.strip().split()
            if len(l) == 2:
                utt2wav[l[0]] = l[1]
    
    with open(f'{args.dir}/utt2spk') as f:
        for l in f:
            l = l.strip().split()
            if len(l) == 2:
                utt2spk[l[0]] = l[1]

    option = onnxruntime.SessionOptions()
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    option.intra_op_num_threads = 1
    providers = ["CPUExecutionProvider"]
    ort_session = onnxruntime.InferenceSession(args.onnx_path, sess_options=option, providers=providers)
    
    with ThreadPoolExecutor(max_workers=args.num_thread) as executor:
        main(args, utt2wav, utt2spk, ort_session)

    with open(f'{args.dir}/utt2spk') as f:
        for l in f:
            l = l.strip().split()
            if len(l) == 2:
                utt2spk[l[0]] = l[1]

    option = onnxruntime.SessionOptions()
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    option.intra_op_num_threads = 1
    providers = ["CPUExecutionProvider"]
    ort_session = onnxruntime.InferenceSession(args.onnx_path, sess_options=option, providers=providers)
    
    with ThreadPoolExecutor(max_workers=args.num_thread) as executor:
        main(args, utt2wav, utt2spk, ort_session)
