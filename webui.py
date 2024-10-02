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

inference_mode_list = ['Cross-lingual replication', '3-second fast replication']
# inference_mode_list = ['Pre-trained voice', '3-second fast replication', 'Cross-lingual replication', 'Natural language control']
instruct_dict = {
    # 'Pre-trained voice': '1. Select pre-trained voice\n2. Click the generate audio button',
    '3-second fast replication': '1. Select the prompt audio file, or record the prompt audio, ensuring it does not exceed 30 seconds. If both are provided, the prompt audio file will be prioritized\n2. Enter the prompt text\n3. Click the generate audio button',
    'Cross-lingual replication': '1. Select the prompt audio file, or record the prompt audio, ensuring it does not exceed 30 seconds. If both are provided, the prompt audio file will be prioritized\n2. Click the generate audio button',
    # 'Natural language control': '1. Select pre-trained voice\n2. Enter the instruct text\n3. Click the generate audio button'
}
examples = [
    [
        "Please enter the text to be synthesized, select the inference mode, and follow the instruction steps. Bây giờ, em chỉ cần biết bán kính r của hình tròn là bao nhiêu, rồi thay vào công thức trên là ra ngay diện tích thôi. Em thử tính xem nào! Nếu cần giúp thêm bước nào, cứ hỏi anh nha!",
        "Cross-lingual replication",
        "nếu các cậu muốn vậy, đây là bảo mối dành cho trẻ em. Chúng là một đồ chơi để bay ra ngoài vũ trụ trong tương lai á. Đầu tiên là bảo bối tên lửa cấm trại ngoài vũ trụ.",
        "samples/doremon.mp3"
    ],
    [
        "Please enter the text to be synthesized, select the inference mode, and follow the instruction steps. Bây giờ, em chỉ cần biết bán kính r của hình tròn là bao nhiêu, rồi thay vào công thức trên là ra ngay diện tích thôi. Em thử tính xem nào! Nếu cần giúp thêm bước nào, cứ hỏi anh nha!",
        "Cross-lingual replication",
        "Các bạn đã bao giờ gặp phải cảm giác vội dối khi phải đối mặt với một cuộc sống hoàn toàn mới mà không ai nhắc nhở hay hướng dẫn từng bước, hay cảm thấy áp lực nặng nề từ việc học tập khiến cho bạn dễ dàng mất đi định hướng ngay từ những ngày đầu học đại học? Ở đại học, sẽ không còn ai giám sát hay nhắc nhở bạn như thời trung học nữa.",
        "samples/quynh.wav"
    ],
    [
        "Please enter the text to be synthesized, select the inference mode, and follow the instruction steps. Bây giờ, em chỉ cần biết bán kính r của hình tròn là bao nhiêu, rồi thay vào công thức trên là ra ngay diện tích thôi. Em thử tính xem nào! Nếu cần giúp thêm bước nào, cứ hỏi anh nha!",
        "Cross-lingual replication",
        "Ngày hôm nay ngồi đây với chúng ta không chỉ là một Sơn Tùng ca sĩ, nhạc sĩ mà cách đây ba bốn tháng thì bạn ấy còn vừa đảm nhiệm một vai trò mới là giám đốc công ty giải trí mang tên mình. Vai trò mới, niềm vui mới nhưng chắc chắn trách nhiệm và áp lực cũng mới đúng không?",
        "samples/diep-chi.wav"
    ],
    [
        "Please enter the text to be synthesized, select the inference mode, and follow the instruction steps. Bây giờ, em chỉ cần biết bán kính r của hình tròn là bao nhiêu, rồi thay vào công thức trên là ra ngay diện tích thôi. Em thử tính xem nào! Nếu cần giúp thêm bước nào, cứ hỏi anh nha!",
        "Cross-lingual replication",
        "Xin chào, tôi là một trợ lý AI có khả năng trò chuyện với bạn bằng giọng nói tự nhiên, được phát triển bởi nhóm Nón Lá. Tôi có thể hỗ trợ người khiếm thị, đọc sách nói, làm trợ lý ảo, review phim, làm waifu để an ủi bạn và phục vụ nhiều mục đích khác.",
        "samples/nu-nhe-nhang.wav"
    ],
    [
        "Please enter the text to be synthesized, select the inference mode, and follow the instruction steps. Bây giờ, em chỉ cần biết bán kính r của hình tròn là bao nhiêu, rồi thay vào công thức trên là ra ngay diện tích thôi. Em thử tính xem nào! Nếu cần giúp thêm bước nào, cứ hỏi anh nha!",
        "Cross-lingual replication",
        "Đẩy mạnh công tác thông tin tuyên truyền, kịp thời thông tin, cung cấp các thông tin trinh thống đến tất cả các đơn vị trên cả nước. Đây là tôi đang nói những nội dung mà nó không có ý nghĩa mục đích là test hệ thống của Tech Channel",
        "samples/atuan.wav"
    ],
    [
        "Please enter the text to be synthesized, select the inference mode, and follow the instruction steps. Bây giờ, em chỉ cần biết bán kính r của hình tròn là bao nhiêu, rồi thay vào công thức trên là ra ngay diện tích thôi. Em thử tính xem nào! Nếu cần giúp thêm bước nào, cứ hỏi anh nha!",
        "Cross-lingual replication",
        "Điểm thứ hai mà chúng tôi cũng xin thưa quý vị là cũng trong cái buổi nói chuyện đó thì tôi có đề cập tới một phụ nữ Việt Nam đã sống tại Mỹ 17 năm mà bị trục xuất Thưa quý vị bản tin này chúng tôi cũng đọc ở trên báo hoặc là trên mạng nhưng mà không có kiểm chứng chi tiết cho chính xác trước khi lên đường qua Âu Chộc thì chúng tôi xin rút lại câu chuyện này để tránh sự hiểu lầm về chính sách di trú của Hoa Kịch. Thưa quý vị",
        "samples/nguyen-ngoc-ngan.wav"
    ],
    [
        "Please enter the text to be synthesized, select the inference mode, and follow the instruction steps. Bây giờ, em chỉ cần biết bán kính r của hình tròn là bao nhiêu, rồi thay vào công thức trên là ra ngay diện tích thôi. Em thử tính xem nào! Nếu cần giúp thêm bước nào, cứ hỏi anh nha!",
        "Cross-lingual replication",
        "希望你以后能够做得比我还好哟",
        "samples/zero_shot_prompt.wav"
    ]
]
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

def generate_audio(
    tts_text, mode_checkbox_group,
    prompt_text, prompt_wav_upload, prompt_wav_record,
    seed, stream, speed
):
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
        if prompt_wav is not None or prompt_text != '':
            gr.Info('You are using natural language control mode, the prompt audio/prompt text will be ignored.')

    # Cross-lingual Mode
    if mode_checkbox_group in ['Cross-lingual replication']:
        if cosyvoice.frontend.instruct is True:
            gr.Warning('You are using cross-lingual replication mode, the {} model does not support this mode, please use the iic/CosyVoice-300M model.'.format(args.model_dir))
            yield (target_sr, default_data)
        if prompt_wav is None:
            gr.Warning('You are using cross-lingual replication mode, please provide the prompt audio.')
            yield (target_sr, default_data)

    # 3-second Fast Replication Mode
    if mode_checkbox_group in ['3-second fast replication', 'Cross-lingual replication']:
        if prompt_wav is None:
            gr.Warning('Audio reference must not be empty.')
            yield (target_sr, default_data)
        if torchaudio.info(prompt_wav).sample_rate < prompt_sr:
            gr.Warning('Audio reference sampling rate {} is lower than {}.'.format(torchaudio.info(prompt_wav).sample_rate, prompt_sr))
            yield (target_sr, default_data)

    # Pre-trained Voice Mode
    # 3-second Fast Replication Mode
    if mode_checkbox_group in ['3-second fast replication']:
        if prompt_text == '':
            gr.Warning('Reference text must not be empty.')
            yield (target_sr, default_data)

    if mode_checkbox_group == 'Pre-trained voice':
        logging.info('Get SFT inference request')
        set_all_random_seed(seed)
        for i in cosyvoice.inference_sft(tts_text, stream=stream, speed=speed):
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
        for i in cosyvoice.inference_instruct(tts_text,None, stream=stream, speed=speed):
            yield (target_sr, i['tts_speech'].numpy().flatten())

def main():
    with gr.Blocks() as demo:
        gr.Markdown("# 🍵 CosyVoice: A fast TTS architecture with conditional flow matching")
        with gr.Row():
            with gr.Column():
                tts_text = gr.Textbox(
                    label="Input text",
                    lines=5,
                    value="Please enter the text to be synthesized, select the inference mode, and follow the instruction steps. Bây giờ, em chỉ cần biết bán kính r của hình tròn là bao nhiêu, rồi thay vào công thức trên là ra ngay diện tích thôi. Em thử tính xem nào! Nếu cần giúp thêm bước nào, cứ hỏi anh nha! "
                )
                with gr.Tabs(elem_id="prompt_wav_tabs"):
                    with gr.TabItem("Upload"):
                        prompt_wav_upload = gr.Audio(sources='upload', type='filepath', label='Reference audio file')
                    with gr.TabItem("Record"):
                        prompt_wav_record = gr.Audio(sources='microphone', type='filepath', label='Record prompt audio file')

                prompt_text = gr.Textbox(label="Reference text", lines=5)
                mode_checkbox_group = gr.Radio(
                    choices=inference_mode_list,
                    label='Mode',
                    value=inference_mode_list[0]
                )
                stream = gr.Radio(
                    label='Enable streaming inference',
                    choices=stream_mode_list,
                    value=stream_mode_list[0][1],
                    visible=False
                )
                speed = gr.Slider(label="Speed", value=1, minimum=0.5, maximum=2.0, step=0.1)
                seed_button = gr.Button(value="\U0001F3B2", visible=False)
                seed = gr.Number(value=0, label="Random inference seed", visible=False)
            with gr.Column():
                output_audio = gr.Audio(label='Generated audio')
                generate_btn = gr.Button("Generate", variant="primary")
                gr.Examples(examples, inputs=[tts_text, mode_checkbox_group, prompt_text, prompt_wav_upload])

            seed_button.click(fn=generate_seed, outputs=seed)
            generate_btn.click(
                fn=generate_audio,
                inputs=[
                    tts_text, mode_checkbox_group,
                    prompt_text, prompt_wav_upload,
                    prompt_wav_record, seed,
                    stream, speed
                ],
                outputs=output_audio
            )
    demo.title = '🍵 CosyVoice: A fast TTS architecture with conditional flow matching'
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