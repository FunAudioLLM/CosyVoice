# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import argparse
import gradio as gr
import numpy as np
import torch
import torchaudio
import random
import librosa

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/third_party/Matcha-TTS'.format(ROOT_DIR))

from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav, logging
from cosyvoice.utils.common import set_all_random_seed

inference_mode_list = ['Pre-trained voice', '3-second fast replication', 'Cross-lingual replication', 'Natural language control']
instruct_dict = {
    'Pre-trained voice': '1. Select pre-trained voice\n2. Click the generate audio button',
    '3-second fast replication': '1. Select the prompt audio file, or record the prompt audio, ensuring it does not exceed 30 seconds. If both are provided, the prompt audio file will be prioritized\n2. Enter the prompt text\n3. Click the generate audio button',
    'Cross-lingual replication': '1. Select the prompt audio file, or record the prompt audio, ensuring it does not exceed 30 seconds. If both are provided, the prompt audio file will be prioritized\n2. Click the generate audio button',
    'Natural language control': '1. Select pre-trained voice\n2. Enter the instruct text\n3. Click the generate audio button'
}
stream_mode_list = [('No', False), ('Yes', True)]
max_val = 0.8

def generate_seed():
    seed = random.randint(1, 100000000)
    return {
        "__type__": "update",
        "value": seed
    }

def postprocess(speech, top_db=60, hop_length=220, win_length=440):
    speech, _ = librosa.effects.trim(
        speech, top_db=top_db,
        frame_length=win_length,
        hop_length=hop_length
    )
    if speech.abs().max() > max_val:
        speech = speech / speech.abs().max() * max_val
    speech = torch.concat([speech, torch.zeros(1, int(target_sr * 0.2))], dim=1)
    return speech

def change_instruction(mode_checkbox_group):
    return instruct_dict[mode_checkbox_group]

def generate_audio(tts_text, mode_checkbox_group, sft_dropdown, prompt_text, prompt_wav_upload, prompt_wav_record, instruct_text,
                   seed, stream, speed):
    if prompt_wav_upload is not None:
        prompt_wav = prompt_wav_upload
    elif prompt_wav_record is not None:
        prompt_wav = prompt_wav_record
    else:
        prompt_wav = None

    # Natural Language Control Mode
    if mode_checkbox_group in ['Natural language control']:
        if cosyvoice.frontend.instruct is False:
            gr.Warning('You are using natural language control mode, the {} model does not support this mode, please use the iic/CosyVoice-300M-Instruct model.'.format(args.model_dir))
            yield (target_sr, default_data)
        if instruct_text == '':
            gr.Warning('You are using natural language control mode, please enter the instruct text.')
            yield (target_sr, default_data)
        if prompt_wav is not None or prompt_text != '':
            gr.Info('You are using natural language control mode, the prompt audio/prompt text will be ignored.')

    # Cross-lingual Mode
    if mode_checkbox_group in ['Cross-lingual replication']:
        if cosyvoice.frontend.instruct is True:
            gr.Warning('You are using cross-lingual replication mode, the {} model does not support this mode, please use the iic/CosyVoice-300M model.'.format(args.model_dir))
            yield (target_sr, default_data)
        if instruct_text != '':
            gr.Info('You are using cross-lingual replication mode, the instruct text will be ignored.')
        if prompt_wav is None:
            gr.Warning('You are using cross-lingual replication mode, please provide the prompt audio.')
            yield (target_sr, default_data)
        gr.Info('You are using cross-lingual replication mode, please ensure that the synthesized text and the prompt text are in different languages.')

    # 3-second Fast Replication Mode
    if mode_checkbox_group in ['3-second fast replication', 'Cross-lingual replication']:
        if prompt_wav is None:
            gr.Warning('Prompt audio is empty, did you forget to input the prompt audio?')
            yield (target_sr, default_data)
        if torchaudio.info(prompt_wav).sample_rate < prompt_sr:
            gr.Warning('Prompt audio sampling rate {} is lower than {}.'.format(torchaudio.info(prompt_wav).sample_rate, prompt_sr))
            yield (target_sr, default_data)

    # Pre-trained Voice Mode
    if mode_checkbox_group in ['Pre-trained voice']:
        if instruct_text != '' or prompt_wav is not None or prompt_text != '':
            gr.Info('You are using pre-trained voice mode, the prompt text/prompt audio/instruct text will be ignored!')

    # 3-second Fast Replication Mode
    if mode_checkbox_group in ['3-second fast replication']:
        if prompt_text == '':
            gr.Warning('Prompt text is empty, did you forget to input the prompt text?')
            yield (target_sr, default_data)
        if instruct_text != '':
            gr.Info('You are using 3-second fast replication mode, the pre-trained voice/instruct text will be ignored!')

    if mode_checkbox_group == 'Pre-trained voice':
        logging.info('Get SFT inference request')
        set_all_random_seed(seed)
        for i in cosyvoice.inference_sft(tts_text, sft_dropdown, stream=stream, speed=speed):
            yield (target_sr, i['tts_speech'].numpy().flatten())
    elif mode_checkbox_group == '3-second fast replication':
        logging.info('Get zero-shot inference request')
        prompt_speech_16k = postprocess(load_wav(prompt_wav, prompt_sr))
        set_all_random_seed(seed)
        for i in cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k, stream=stream, speed=speed):
            yield (target_sr, i['tts_speech'].numpy().flatten())
    elif mode_checkbox_group == 'Cross-lingual replication':
        logging.info('Get cross-lingual inference request')
        prompt_speech_16k = postprocess(load_wav(prompt_wav, prompt_sr))
        set_all_random_seed(seed)
        for i in cosyvoice.inference_cross_lingual(tts_text, prompt_speech_16k, stream=stream, speed=speed):
            yield (target_sr, i['tts_speech'].numpy().flatten())
    else:
        logging.info('Get instruct inference request')
        set_all_random_seed(seed)
        for i in cosyvoice.inference_instruct(tts_text, sft_dropdown, instruct_text, stream=stream, speed=speed):
            yield (target_sr, i['tts_speech'].numpy().flatten())

def main():
    with gr.Blocks() as demo:
        gr.Markdown("### Codebase [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) \
                    Pre-trained models [CosyVoice-300M](https://www.modelscope.cn/models/iic/CosyVoice-300M) \
                    [CosyVoice-300M-Instruct](https://www.modelscope.cn/models/iic/CosyVoice-300M-Instruct) \
                    [CosyVoice-300M-SFT](https://www.modelscope.cn/models/iic/CosyVoice-300M-SFT)")
        gr.Markdown("#### Please enter the text to be synthesized, select the inference mode, and follow the instruction steps.")

        tts_text = gr.Textbox(label="Input synthesized text", lines=1, value="I am a newly launched generative voice model from the Tongyi Lab speech team, offering comfortable and natural voice synthesis capabilities.")
        with gr.Row():
            mode_checkbox_group = gr.Radio(choices=inference_mode_list, label='Select inference mode', value=inference_mode_list[0])
            instruction_text = gr.Text(label="Instruction steps", value=instruct_dict[inference_mode_list[0]], scale=0.5)
            sft_dropdown = gr.Dropdown(choices=sft_spk, label='Select pre-trained voice')
            stream = gr.Radio(choices=stream_mode_list, label='Enable streaming inference', value=stream_mode_list[0][1])
            speed = gr.Number(value=1, label="Speed adjustment (only supported for non-streaming inference)", minimum=0.5, maximum=2.0, step=0.1)
            with gr.Column(scale=0.25):
                seed_button = gr.Button(value="\U0001F3B2")
                seed = gr.Number(value=0, label="Random inference seed")

        with gr.Row():
            prompt_wav_upload = gr.Audio(sources='upload', type='filepath', label='Select prompt audio file, note sampling rate should be at least 16khz')
            prompt_wav_record = gr.Audio(sources='microphone', type='filepath', label='Record prompt audio file')
        prompt_text = gr.Textbox(label="Input prompt text", lines=1)
        instruct_text = gr.Textbox(label="Input instruct text", lines=1)
        output_audio = gr.Audio(label='Generated audio')

        mode_checkbox_group.change(fn=change_instruction, inputs=mode_checkbox_group, outputs=instruction_text)

        seed_button.click(fn=generate_seed, outputs=seed)

        generate_btn = gr.Button("Generate")
        generate_btn.click(fn=generate_audio, inputs=[tts_text, mode_checkbox_group, sft_dropdown, prompt_text, prompt_wav_upload, prompt_wav_record, instruct_text, seed, stream, speed], outputs=output_audio)

    demo.queue(max_size=4, default_concurrency_limit=2)
    demo.launch(server_name='0.0.0.0', server_port=args.port)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',
                        type=int,
                        default=8000)
    parser.add_argument('--model_dir',
                        type=str,
                        default='pretrained_models/CosyVoice-300M',
                        help='local path or modelscope repo id')
    args = parser.parse_args()
    cosyvoice = CosyVoice(args.model_dir)
    sft_spk = cosyvoice.list_avaliable_spks()
    prompt_sr, target_sr = 16000, 22050
    default_data = np.zeros(target_sr)
    main()

