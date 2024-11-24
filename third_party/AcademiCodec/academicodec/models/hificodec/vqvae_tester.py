import os

import librosa
import torch
import torch.nn as nn

from academicodec.models.hificodec.vqvae import VQVAE


class VqvaeTester(nn.Module):
    def __init__(self, config_path, model_path, sample_rate=24000):
        super().__init__()
        self.vqvae = VQVAE(config_path, model_path, with_encoder=True)
        self.sample_rate = sample_rate

    @torch.no_grad()
    def forward(self, wav_path):
        # 单声道
        # wav.shape (T, ), 按照模型的 sr 读取
        wav, sr = librosa.load(wav_path, sr=self.sample_rate)
        fid = os.path.basename(wav_path)[:-4]
        wav = torch.tensor(wav).unsqueeze(0)
        wav = wav.cuda()
        # vq_codes is acoustic token
        vq_codes = self.vqvae.encode(wav)
        syn = self.vqvae(vq_codes)
        return fid, syn

    @torch.no_grad()
    def vq(self, wav_path):
        wav, sr = librosa.load(wav_path, sr=self.sample_rate)
        fid = os.path.basename(wav_path)[:-4]
        wav = torch.tensor(wav).unsqueeze(0)
        wav = wav.cuda()
        # vq_codes is acoustic token
        vq_codes = self.vqvae.encode(wav)
        return fid, vq_codes
