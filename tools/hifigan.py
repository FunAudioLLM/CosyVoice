import torch
import matplotlib.pyplot as plt
from IPython.display import Audio
import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Using {device} for inference')

fastpitch, generator_train_setup = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_fastpitch')
hifigan, vocoder_train_setup, denoiser = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_hifigan')

CHECKPOINT_SPECIFIC_ARGS = [
    'sampling_rate', 'hop_length', 'win_length', 'p_arpabet', 'text_cleaners',
    'symbol_set', 'max_wav_value', 'prepend_space_to_text',
    'append_space_to_text']

for k in CHECKPOINT_SPECIFIC_ARGS:

    v1 = generator_train_setup.get(k, None)
    v2 = vocoder_train_setup.get(k, None)

    assert v1 is None or v2 is None or v1 == v2, \
        f'{k} mismatch in spectrogram generator and vocoder'
        
fastpitch.to(device)
hifigan.to(device)
denoiser.to(device)

tp = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_textprocessing_utils', cmudict_path="cmudict-0.7b", heteronyms_path="heteronyms")

text = "Say this smoothly, to prove you are not a robot."
batches = tp.prepare_input_sequence([text], batch_size=1)
gen_kw = {
    'pace': 1.0,
    'speaker': 0,
    'pitch_tgt': None,
    'pitch_transform': None
}
denoising_strength = 0.005

def mel2wav(mel: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        audios = hifigan(mel).float()
        audios = denoiser(audios.squeeze(1), denoising_strength)
        audios = audios.squeeze(1) * vocoder_train_setup['max_wav_value']
    return audios

# plt.figure(figsize=(10,12))
# res_mel = mel[0].detach().cpu().numpy()
# plt.imshow(res_mel, origin='lower')
# plt.xlabel('time')
# plt.ylabel('frequency')
# _ = plt.title('Spectrogram')
# plt.savefig('audio.png')

# audio_numpy = audios[0].cpu().numpy()
# Audio(audio_numpy, rate=22050)

# from scipy.io.wavfile import write
# write("audio.wav", vocoder_train_setup['sampling_rate'], audio_numpy)