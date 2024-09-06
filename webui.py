# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Liu Yue)
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
import os
import sys
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/third_party/Matcha-TTS'.format(ROOT_DIR))

import argparse
import gradio as gr
import numpy as np
import torch
import torchaudio
import random
import librosa
import i18n 
from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav, speed_change, logging

# Load available languages 
i18n.load_path.append('./locales/')
i18n.set('file_format', 'json')
i18n.set('filename_format', '{locale}.{format}')




def generate_seed():
    seed = random.randint(1, 100000000)
    return {
        "__type__": "update",
        "value": seed
    }

def set_all_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

max_val = 0.8
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

inference_mode_list = [i18n.t('inference_mode_list.pretrained_voice'), i18n.t('inference_mode_list.3s_fast_replication'), i18n.t('inference_mode_list.crosslingual'), i18n.t('inference_mode_list.natural_language_control')]
instruct_dict = {i18n.t('inference_mode_list.pretrained_voice'): i18n.t('instruct_dict.pretrained_voice'),
                 i18n.t('inference_mode_list.3s_fast_replication'): i18n.t('instruct_dict.3s_fast_replication'),
                 i18n.t('inference_mode_list.crosslingual'): i18n.t('instruct_dict.crosslingual'),
                 i18n.t('inference_mode_list.natural_language_control'): i18n.t('instruct_dict.natural_language_control')}
stream_mode_list = [(i18n.t('boolean.false'), False), (i18n.t('boolean.true'), True)]
def change_instruction(mode_checkbox_group):
    return instruct_dict[mode_checkbox_group]

def generate_audio(tts_text, mode_checkbox_group, sft_dropdown, prompt_text, prompt_wav_upload, prompt_wav_record, instruct_text, seed, stream, speed_factor):
    if prompt_wav_upload is not None:
        prompt_wav = prompt_wav_upload
    elif prompt_wav_record is not None:
        prompt_wav = prompt_wav_record
    else:
        prompt_wav = None
    # if instruct mode, please make sure that model is iic/CosyVoice-300M-Instruct and not cross_lingual mode
    if mode_checkbox_group in ['自然语言控制']:
        if cosyvoice.frontend.instruct is False:
            gr.Warning(i18n.t('warnings.nlp_model_warn').format(args.model_dir))
            return (target_sr, default_data)
        if instruct_text == '':
            gr.Warning(i18n.t('warnings.instruct_text'))
            return (target_sr, default_data)
        if prompt_wav is not None or prompt_text != '':
            gr.Info(i18n.t('info.prompt_wav'))
    # if cross_lingual mode, please make sure that model is iic/CosyVoice-300M and tts_text prompt_text are different language
    if mode_checkbox_group in ['跨语种复刻']:
        if cosyvoice.frontend.instruct is True:
            gr.Warning(i18n.t('warnings.no_crosslingual_support').format(args.model_dir))
            return (target_sr, default_data)
        if instruct_text != '':
            gr.Info(i18n.t('warnings.crosslingual_instruct_ignored'))
        if prompt_wav is None:
            gr.Warning(i18n.t('warnings.crosslingual_prompt_audio_required'))
            return (target_sr, default_data)
        gr.Info(i18n.t('info.crosslingual_language_reminder'))
    # if in zero_shot cross_lingual, please make sure that prompt_text and prompt_wav meets requirements
    if mode_checkbox_group in ['3s极速复刻', '跨语种复刻']:
        if prompt_wav is None:
            gr.Warning(i18n.t('warnings.prompt_audio_empty'))
            return (target_sr, default_data)
        if torchaudio.info(prompt_wav).sample_rate < prompt_sr:
            gr.Warning(i18n.t('warnings.sample_rate_error').format(torchaudio.info(prompt_wav).sample_rate, prompt_sr))
            return (target_sr, default_data)
    # sft mode only use sft_dropdown
    if mode_checkbox_group in ['预训练音色']:
        if instruct_text != '' or prompt_wav is not None or prompt_text != '':
            gr.Info(i18n.t('info.pretrained_voice_warning'))
    # zero_shot mode only use prompt_wav prompt text
    if mode_checkbox_group in ['3s极速复刻']:
        if prompt_text == '':
            gr.Warning(i18n.t('warnings.prompt_text_empty'))
            return (target_sr, default_data)
        if instruct_text != '':
            gr.Info(i18n.t('info.instruct_text_empty'))

    if mode_checkbox_group == '预训练音色':
        logging.info('get sft inference request')
        set_all_random_seed(seed)
        for i in cosyvoice.inference_sft(tts_text, sft_dropdown, stream=stream):
            yield (target_sr,  i['tts_speech'].numpy().flatten())
    elif mode_checkbox_group == '3s极速复刻':
        logging.info('get zero_shot inference request')
        prompt_speech_16k = postprocess(load_wav(prompt_wav, prompt_sr))
        set_all_random_seed(seed)
        for i in cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k, stream=stream):
            yield (target_sr,  i['tts_speech'].numpy().flatten())
    elif mode_checkbox_group == '跨语种复刻':
        logging.info('get cross_lingual inference request')
        prompt_speech_16k = postprocess(load_wav(prompt_wav, prompt_sr))
        set_all_random_seed(seed)
        for i in cosyvoice.inference_cross_lingual(tts_text, prompt_speech_16k, stream=stream):
            yield (target_sr,  i['tts_speech'].numpy().flatten())
    else:
        logging.info('get instruct inference request')
        set_all_random_seed(seed)
        for i in cosyvoice.inference_instruct(tts_text, sft_dropdown, instruct_text, stream=stream):
            yield (target_sr,  i['tts_speech'].numpy().flatten())

def main():
    with gr.Blocks() as demo:
        gr.Markdown(i18n.t('markdown.code_reference'))
        gr.Markdown(i18n.t('markdown.output_text_prompt'))

        tts_text = gr.Textbox(label=i18n.t('input_label.enter_synthesis_text'), lines=1, value=i18n.t('placeholders.enter_synthesis_text'))
        speed_factor = gr.Slider(minimum=0.25, maximum=4, step=0.05, label=i18n.t('input_label.speed_adjustment'), value=1.0, interactive=True)
        with gr.Row():
            mode_checkbox_group = gr.Radio(choices=inference_mode_list, label=i18n.t('input_label.select_inference_mode_radio'), value=inference_mode_list[0])
            instruction_text = gr.Text(label=i18n.t('input_label.instruction_text'), value=instruct_dict[inference_mode_list[0]], scale=0.5)
            sft_dropdown = gr.Dropdown(choices=sft_spk, label=i18n.t('input_label.sft_dropdown'), value=sft_spk[0], scale=0.25)
            stream = gr.Radio(choices=stream_mode_list, label=i18n.t('input_label.stream'), value=stream_mode_list[0][1])
            with gr.Column(scale=0.25):
                seed_button = gr.Button(value="\U0001F3B2")
                seed = gr.Number(value=0, label=i18n.t('input_label.seed_number'))

        with gr.Row():
            prompt_wav_upload = gr.Audio(sources='upload', type='filepath', label=i18n.t('input_label.prompt_wav_upload'))
            prompt_wav_record = gr.Audio(sources='microphone', type='filepath', label=i18n.t('input_label.prompt_wav_record'))
        prompt_text = gr.Textbox(label=i18n.t('input_label.prompt_text'), lines=1, placeholder=i18n.t('placeholders.prompt_text'), value='')
        instruct_text = gr.Textbox(label=i18n.t('input_label.instruct_text'), lines=1, placeholder=i18n.t('placeholders.instruct_text'), value='')

        generate_button = gr.Button(i18n.t('input_label.generate_button'))

        audio_output = gr.Audio(label=i18n.t('input_label.audio_output'), autoplay=True, streaming=True)

        seed_button.click(generate_seed, inputs=[], outputs=seed)
        generate_button.click(generate_audio,
                              inputs=[tts_text, mode_checkbox_group, sft_dropdown, prompt_text, prompt_wav_upload, prompt_wav_record, instruct_text, seed, stream, speed_factor],
                              outputs=[audio_output])
        mode_checkbox_group.change(fn=change_instruction, inputs=[mode_checkbox_group], outputs=[instruction_text])
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
    parser.add_argument('--locale',
                        type=str,
                        default='zh',
                        help='language locale')
    args = parser.parse_args()
    i18n.set('locale', args.locale)
    cosyvoice = CosyVoice(args.model_dir)
    sft_spk = cosyvoice.list_avaliable_spks()
    prompt_sr, target_sr = 16000, 22050
    default_data = np.zeros(target_sr)
    main()
