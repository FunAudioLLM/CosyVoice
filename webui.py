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

inference_mode_list = ['Single Language', 'Multi-Language']
instruct_dict = {
    'Single Language': '1. Choose the language model\n2. Click on generate audio',
    'Multi-Language': '1. Choose the multi-language model\n2. Provide prompt audio\n3. Click on generate audio'
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
                   seed, stream, speed, language_type):
    if prompt_wav_upload is not None:
        prompt_wav = prompt_wav_upload
    elif prompt_wav_record is not None:
        prompt_wav = prompt_wav_record
    else:
        prompt_wav = None
    
    if mode_checkbox_group == 'Multi-Language':
        if prompt_wav is None:
            gr.Warning('Prompt audio is missing, please provide an audio file.')
            yield (target_sr, default_data)
        if torchaudio.info(prompt_wav).sample_rate < prompt_sr:
            gr.Warning(f'Prompt audio sampling rate {torchaudio.info(prompt_wav).sample_rate} is lower than {prompt_sr}.')
            yield (target_sr, default_data)

    logging.info('Starting inference request')
    set_all_random_seed(seed)

    if mode_checkbox_group == 'Single Language':
        # Use single language model
        logging.info('Using single language model')
        for i in cosyvoice.inference_single_language(tts_text, sft_dropdown, stream=stream, speed=speed, language=language_type):
            yield (target_sr, i['tts_speech'].numpy().flatten())
    elif mode_checkbox_group == 'Multi-Language':
        # Use multi-language model
        prompt_speech_16k = postprocess(load_wav(prompt_wav, prompt_sr))
        logging.info('Using multi-language model')
        for i in cosyvoice.inference_multi_language(tts_text, prompt_speech_16k, stream=stream, speed=speed, language=language_type):
            yield (target_sr, i['tts_speech'].numpy().flatten())

def main():
    with gr.Blocks() as demo:
        gr.Markdown("### Code for Cross-Lingual and Single Language Training Support")

        tts_text = gr.Textbox(label="Enter text for synthesis", lines=1, value="Hello, this is a voice synthesis example.")
        language_type = gr.Dropdown(choices=['English', 'Spanish', 'French'], label='Select Language', value='English')
        with gr.Row():
            mode_checkbox_group = gr.Radio(choices=inference_mode_list, label='Select Mode', value=inference_mode_list[0])
            instruction_text = gr.Text(label="Instructions", value=instruct_dict[inference_mode_list[0]], scale=0.5)
            sft_dropdown = gr.Dropdown(choices=sft_spk, label='Choose Pre-trained Voice', value=sft_spk[0], scale=0.25)
            stream = gr.Radio(choices=stream_mode_list, label='Stream Mode', value=stream_mode_list[0][1])
            speed = gr.Number(value=1, label="Speed Adjustment", minimum=0.5, maximum=2.0, step=0.1)
            seed_button = gr.Button(value="🎲")
            seed = gr.Number(value=0, label="Random Seed")

        with gr.Row():
            prompt_wav_upload = gr.Audio(sources='upload', type='filepath', label='Upload Prompt Audio')
            prompt_wav_record = gr.Audio(sources='microphone', type='filepath', label='Record Prompt Audio')
        prompt_text = gr.Textbox(label="Enter Prompt Text", lines=1, placeholder="Enter prompt text matching the audio...")
        instruct_text = gr.Textbox(label="Enter Instructions", lines=1, placeholder="Instructions for synthesis...", value='')

        generate_button = gr.Button("Generate Audio")
        audio_output = gr.Audio(label="Synthesized Audio", autoplay=True, streaming=True)

        seed_button.click(generate_seed, inputs=[], outputs=seed)
        generate_button.click(generate_audio,
                              inputs=[tts_text, mode_checkbox_group, sft_dropdown, prompt_text, prompt_wav_upload, prompt_wav_record, instruct_text,
                                      seed, stream, speed, language_type],
                              outputs=[audio_output])
        mode_checkbox_group.change(fn=change_instruction, inputs=[mode_checkbox_group], outputs=[instruction_text])
    demo.queue(max_size=4, default_concurrency_limit=2)
    demo.launch(server_name='0.0.0.0', server_port=args.port)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--model_dir', type=str, default='pretrained_models/CosyVoice-300M', help='Model directory')
    args = parser.parse_args()
    cosyvoice = CosyVoice(args.model_dir)
    sft_spk = cosyvoice.list_avaliable_spks()
    prompt_sr, target_sr = 16000, 22050
    default_data = np.zeros(target_sr)
    main()
