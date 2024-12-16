import os
import sys
import argparse
import gradio as gr
import numpy as np
from typing import Union
import torch
import random
from loguru import logger as logging

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/third_party/Matcha-TTS'.format(ROOT_DIR))

from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav
from cosyvoice.utils.common import set_all_random_seed, fade_in_out_audio
from tools.vad import get_speech

inference_mode_list = ['Cross-lingual replication', '3-second fast replication', 'Natural language control']
# inference_mode_list = ['Pre-trained voice', '3-second fast replication', 'Cross-lingual replication', 'Natural language control']
instruct_dict = {
    # 'Pre-trained voice': '1. Select pre-trained voice\n2. Click the generate audio button',
    '3-second fast replication': '1. Select the prompt audio file, or record the prompt audio, ensuring it does not exceed 30 seconds. If both are provided, the prompt audio file will be prioritized\n2. Enter the prompt text\n3. Click the generate audio button',
    'Cross-lingual replication': '1. Select the prompt audio file, or record the prompt audio, ensuring it does not exceed 30 seconds. If both are provided, the prompt audio file will be prioritized\n2. Click the generate audio button',
    'Natural language control': '1. Select pre-trained voice\n2. Enter the instruct text\n3. Click the generate audio button'
}
examples = [
    # [
    #     "Thủ đô Hà Nội - trung tâm chính trị, kinh tế, văn hóa của cả nước, tự hào sở hữu nhiều địa danh lịch sử, văn hóa và cách mạng. Đặc biệt là Quảng trường Ba Đình, nơi 79 năm trước, Chủ tịch Hồ Chí Minh đã đọc Bản Tuyên ngôn Độc lập, khai sinh nước Việt Nam Dân chủ Cộng hòa.",
    #     "Cross-lingual replication",
    #     "",
    #     "samples/NSND Le Chuc - isolated.mp3"
    # ],
    [
        "Báo Nhân Dân, Cơ quan Trung ương của Đảng Cộng sản Việt Nam, Tiếng nói của Đảng, Nhà nước và nhân dân Việt Nam, ra số đầu ngày 11-3-1951 tại Chiến khu Việt Bắc.",
        "Cross-lingual replication",
        "",
        "samples/cdteam.wav"
    ],
    [
        "Bây giờ, em chỉ cần biết bán kính r của hình tròn là bao nhiêu, rồi thay vào công thức trên là ra ngay diện tích thôi. Em thử tính xem nào! Nếu cần giúp thêm bước nào, cứ hỏi anh nha!",
        "Cross-lingual replication",
        "",
        "samples/doremon.mp3"
    ],
    [
        "Bây giờ, em chỉ cần biết bán kính r của hình tròn là bao nhiêu, rồi thay vào công thức trên là ra ngay diện tích thôi. Em thử tính xem nào! Nếu cần giúp thêm bước nào, cứ hỏi anh nha!",
        "Cross-lingual replication",
        "",
        "samples/songtung-mtp.wav"
    ],
    [
        "Đây mới thực sự là phiên bản hoàn hảo nhất của ca khúc này. Giọng Amee cực kỳ hợp với bài này, vừa nhẹ nhàng, vừa có chút gì đó buồn man mác. Màu giọng của Kaidinh cực hợp với Amee và GreyD. Xin chúc mừng bộ 3 hoàn hảo, 10 điểm cho ca khúc và phần trình diễn của cả 3. Ai thấy mê bài này giống mình thì cho 1 like nha.",
        "Cross-lingual replication",
        "",
        "samples/quynh.wav"
    ],
    [
        "Ở lượt thi cá nhân, mỗi thí sinh sẽ lần lượt trả lời 6 câu hỏi thuộc các lĩnh vực: Khoa học tự nhiên, Khoa học xã hội, Văn hóa - Nghệ thuật - Thể thao, Danh nhân - Sự kiện, Lĩnh vực khác, Tiếng Anh, ... Trả lời mỗi câu trong vòng 5 giây (kể từ khi MC đọc xong câu hỏi). Mỗi câu trả lời đúng được 10 điểm, trả lời sai không có điểm nào.",
        "Cross-lingual replication",
        "",
        "samples/diep-chi.wav"
    ],
    [
        "Mảnh đất cố đô yên bình, mộng mơ là điểm dừng chân của những tâm hồn lạc lối với trăm nỗi bộn bề cuộc sống, là cuộc gặp gỡ của những người đang đi tìm cảm hứng sáng tác nghệ thuật, thi ca. Hay đơn giản, Huế là nơi lữ khách dừng chân, thưởng ngoạn thành phố miền Trung yên ả, rồi trót yêu, trót gắn bó chẳng nỡ rời đi.",
        "Cross-lingual replication",
        "",
        "samples/nu-nhe-nhang.wav"
    ],
    [
        "Nhiều di tích lịch sử và công trình kiến trúc mang tính biểu tượng của Thủ đô như Quảng trường Ba Đình, Phủ Chủ tịch, Tháp Rùa, Văn Miếu - Quốc Tử Giám... mãi sống trong lòng người dân Thủ đô.",
        "Cross-lingual replication",
        "",
        "samples/atuan.wav"
    ],
    [
        "Ban đầu những bác nông dân xung quanh mỗi lần nghe thấy đều nhanh chân chạy đến giúp đỡ. Tuy nhiên, sau khi      bị lừa gạt nhiều lần họ dần không quan tâm tiếng hét của cậu nữa. Rồi một hôm gặp sói thật, cậu kêu cứu nhưng không ai đến giúp và thế là cả đàn cừu bị sói ăn mất sạch.",
        "Cross-lingual replication",
        "",
        "samples/nguyen-ngoc-ngan.wav"
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

def preprocess_prompt_audio(
    speech: Union[str, np.ndarray],
    vad: bool = False,
    min_duration: float=3,
    max_duration: float=5
) -> torch.Tensor:
    if isinstance(speech, str):
        speech = load_wav(speech, prompt_sr)
    elif isinstance(speech, np.ndarray):
        speech = torch.from_numpy(speech)
    if speech.abs().max() > max_val:
        speech = speech / speech.abs().max() * max_val
    if vad:
        speech = get_speech(
            audio_input=speech.squeeze(0),
            min_duration=min_duration,
            max_duration=max_duration
        ).unsqueeze(0)
    else:
        speech = speech[:, :int(max_duration*prompt_sr)]
    # speech = fade_in_out_audio(speech)
    return speech

def change_instruction(mode_checkbox_group):
    return instruct_dict[mode_checkbox_group]

def pcm2float(data: np.ndarray, dtype='float32') -> np.ndarray:
    """Convert PCM datanal to floating point with a range from -1 to 1.
    """
    data = np.asarray(data)
    if data.dtype.kind not in 'iu':
        raise TypeError("'data' must be an array of integers")
    dtype = np.dtype(dtype)
    if dtype.kind != 'f':
        raise TypeError("'dtype' must be a floating point type")

    i = np.iinfo(data.dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (data.astype(dtype) - offset) / abs_max

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
    
    if isinstance(prompt_wav, str):
        prompt_wav = load_wav(prompt_wav, prompt_sr)
    if isinstance(prompt_wav, tuple):
        prompt_wav = pcm2float(prompt_wav[1])
        prompt_wav = prompt_wav[np.newaxis, :]

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
        set_all_random_seed(seed)
        for i in cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_wav, stream=stream, speed=speed):
            yield (target_sr, i['tts_speech'].numpy().flatten())
    elif mode_checkbox_group == 'Cross-lingual replication':
        logging.info('Get cross-lingual inference request')
        set_all_random_seed(seed)
        for i in cosyvoice.inference_cross_lingual(tts_text, prompt_wav, stream=stream, speed=speed):
            yield (target_sr, i['tts_speech'].numpy().flatten())
    else:
        logging.info('Get instruct inference request')
        set_all_random_seed(seed)
        for i in cosyvoice.inference_instruct(tts_text, prompt_wav, prompt_text, stream=stream, speed=speed):
            yield (target_sr, i['tts_speech'].numpy().flatten())

def generate_audio_vc(
    source_audio: Union[str, np.ndarray],
    target_audio: Union[str, np.ndarray]
):
    set_all_random_seed(0)
    if isinstance(source_audio, str):
        source_audio = load_wav(source_audio, prompt_sr)
    if isinstance(target_audio, str):
        target_audio = load_wav(target_audio, prompt_sr)
    if isinstance(source_audio, tuple):
        source_audio = pcm2float(source_audio[1])
        source_audio = source_audio[np.newaxis, :]
    if isinstance(target_audio, tuple):
        target_audio = pcm2float(target_audio[1])
        target_audio = target_audio[np.newaxis, :]
    for i in cosyvoice.inference_vc(source_audio, target_audio):
        yield (target_sr, i['tts_speech'].numpy().flatten())

def main():
    with gr.Blocks() as demo:
        gr.Markdown("# Text2Speech")
        with gr.Tab('TTS & Voice clone'):
            with gr.Row():
                with gr.Column():
                    tts_text = gr.Textbox(
                        label="Input text",
                        lines=5,
                    )
                    with gr.Tabs(elem_id="prompt_wav_tabs"):
                        with gr.TabItem("Upload"):
                            prompt_wav_upload = gr.Audio(sources='upload', type='filepath', label='Reference audio file')
                        with gr.TabItem("Record"):
                            prompt_wav_record = gr.Audio(sources='microphone', type='filepath', label='Record prompt audio file')
                    with gr.Row():
                        with gr.Column():
                            min_speech_dur = gr.Number(value=3, minimum=-1, maximum=30, label="Minimum speech duration")
                        with gr.Column():
                            max_speech_dur = gr.Number(value=5, minimum=-1, maximum=30, label="Maximum speech duration")
                        enable_vad = gr.Checkbox(value=True, label="Enable VAD")
                    
                    output_prompt_audio = gr.Audio(label='Processed prompt audio', type='numpy')
                    # trigger vad change upload
                    enable_vad.change(
                        fn=lambda a,b,c,d: (prompt_sr, preprocess_prompt_audio(a, b, c, d).numpy().flatten()),
                        inputs=[prompt_wav_upload, enable_vad, min_speech_dur, max_speech_dur],
                        outputs=output_prompt_audio
                    )
                    # trigger min/max speech dur change upload
                    min_speech_dur.change(
                        fn=lambda a,b,c,d: (prompt_sr, preprocess_prompt_audio(a, b, c, d).numpy().flatten()),
                        inputs=[prompt_wav_upload, enable_vad, min_speech_dur, max_speech_dur],
                        outputs=output_prompt_audio
                    )
                    max_speech_dur.change(
                        fn=lambda a,b,c,d: (prompt_sr, preprocess_prompt_audio(a, b, c, d).numpy().flatten()),
                        inputs=[prompt_wav_upload, enable_vad, min_speech_dur, max_speech_dur],
                        outputs=output_prompt_audio
                    )
                    # trigger input change
                    prompt_wav_upload.change(
                        fn=lambda a,b,c,d: (prompt_sr, preprocess_prompt_audio(a, b, c, d).numpy().flatten()),
                        inputs=[prompt_wav_upload, enable_vad, min_speech_dur, max_speech_dur],
                        outputs=output_prompt_audio
                    )
                    prompt_wav_record.stop_recording(
                        fn=lambda a,b,c,d: (prompt_sr, preprocess_prompt_audio(a, b, c, d).numpy().flatten()),
                        inputs=[prompt_wav_record, enable_vad, min_speech_dur, max_speech_dur],
                        outputs=output_prompt_audio
                    )
                    prompt_text = gr.Textbox(label="Reference/Prompt text", lines=5)
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
                    output_audio = gr.Audio(label='Generated audio', streaming=False)
                    generate_btn = gr.Button("Generate", variant="primary")
                    gr.Examples(examples, inputs=[tts_text, mode_checkbox_group, prompt_text, prompt_wav_upload])

                seed_button.click(fn=generate_seed, outputs=seed)
                generate_btn.click(
                    fn=generate_audio,
                    inputs=[
                        tts_text, mode_checkbox_group,
                        prompt_text, output_prompt_audio,
                        output_prompt_audio, seed,
                        stream, speed
                    ],
                    outputs=output_audio
                )
        with gr.Tab('Voice conversion'):
            with gr.Row():
                with gr.Column():
                    ref_wav_upload = gr.Audio(sources='upload', type='filepath', label='Reference audio')
                    target_wav_upload = gr.Audio(sources='upload', type='filepath', label='Target audio')
                with gr.Column():
                    output_ref_audio = gr.Audio(label='Pre-processed reference audio', sources='upload', type='numpy', editable=False)
                    output_target_audio = gr.Audio(label='Pre-processed target audio', sources='upload', type='numpy', editable=False)
            with gr.Row():
                with gr.Column():
                    min_speech_dur = gr.Number(value=3, minimum=-1, maximum=30, label="Minimum speech duration")
                with gr.Column():
                    max_speech_dur = gr.Number(value=5, minimum=-1, maximum=30, label="Maximum speech duration")
            with gr.Row():
                output_audio = gr.Audio(label='Generated audio', streaming=True)
            with gr.Row():
                generate_btn = gr.Button("Generate", variant="primary")
                ref_wav_upload.upload(
                    fn=lambda a: (prompt_sr, preprocess_prompt_audio(a, False, -1, -1).numpy().flatten()),
                    inputs=[ref_wav_upload],
                    outputs=[output_ref_audio]
                )
                target_wav_upload.upload(
                    fn=lambda a,b,c,d: (prompt_sr, preprocess_prompt_audio(a, b, c, d).numpy().flatten()),
                    inputs=[target_wav_upload, enable_vad, min_speech_dur, max_speech_dur],
                    outputs=[output_target_audio]
                )
                generate_btn.click(
                    fn=generate_audio_vc,
                    inputs=[
                        output_ref_audio,
                        output_target_audio
                    ],
                    outputs=output_audio
                )
                
    demo.title = 'Text2Speech: A fast TTS architecture with conditional flow matching'
    demo.queue(max_size=4, default_concurrency_limit=2)
    demo.launch(server_name='0.0.0.0', server_port=args.port)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--port', type=int, default=50000)
    args = parser.parse_args()
    cosyvoice = CosyVoice(args.model_dir)
    sft_spk = cosyvoice.list_avaliable_spks()
    prompt_sr, target_sr = 16000, 22050
    default_data = np.zeros(target_sr)
    main()