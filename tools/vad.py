from typing import Union
import numpy as np
import torch
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps

model = load_silero_vad()

def get_speech(
    audio_input: Union[str, np.ndarray, torch.Tensor],
    return_numpy: bool=False,
    min_duration: float=3,
    max_duration: float=5
) -> Union[torch.Tensor, np.ndarray]:
    if isinstance(audio_input, str):
        audio_input = read_audio(audio_input)
    speech_timestamps = get_speech_timestamps(audio_input, model)
    speech = [audio_input[t['start']:t['end']] \
        for t in speech_timestamps \
            if (t['end'] - t['start']) >= 16000*min_duration \
                and (t['end'] - t['start']) <= 16000*max_duration]
    if not speech:
        speech = audio_input[:int(max_duration*16000)]
    else:
        speech = speech[0]
    if return_numpy:
        speech = speech.cpu().numpy()
    return speech

if __name__ == '__main__':
    print(get_speech('samples/diep-chi.wav'))