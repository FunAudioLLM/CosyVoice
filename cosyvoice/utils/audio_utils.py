# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torchaudio


def log_mel_spectrogram(audio, n_mels=128, n_fft=400, hop_length=160, sample_rate=16000):
    """Compute a log-mel spectrogram from a waveform tensor.

    This is a drop-in replacement for ``whisper.log_mel_spectrogram`` that uses
    only ``torch`` and ``torchaudio``, avoiding the heavy ``openai-whisper``
    dependency.  The output is numerically equivalent for the default Whisper
    parameters (n_fft=400, hop_length=160, sample_rate=16000).

    Args:
        audio: 1-D or 2-D float tensor of raw audio at *sample_rate* Hz.
        n_mels: Number of mel-frequency bins.
        n_fft: FFT window size.
        hop_length: Hop length for STFT.
        sample_rate: Expected sample rate of *audio*.

    Returns:
        Tensor of shape ``(n_mels, n_frames)`` (if 1-D input) or
        ``(batch, n_mels, n_frames)`` (if 2-D input).
    """
    window = torch.hann_window(n_fft).to(audio.device)
    stft = torch.stft(audio, n_fft, hop_length, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2

    mel_filters = torchaudio.functional.melscale_fbanks(
        n_freqs=n_fft // 2 + 1,
        f_min=0.0,
        f_max=sample_rate / 2.0,
        n_mels=n_mels,
        sample_rate=sample_rate,
        norm="slaney",
        mel_scale="slaney",
    ).to(audio.device)

    mel_spec = mel_filters.T @ magnitudes
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.amax() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec
