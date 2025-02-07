from cosyvoice.cli.cosyvoice import CosyVoice
import torch
import torchaudio
from cosyvoice.utils.file_utils import logging
import sounddevice as sd
from pyperclip import paste
import threading
import queue

def generate_audio(tts_text, sft_dropdown, instruct_text):
    print(tts_text)
    target_samplerate = 22050  # 目标采样率
    audio_buffer = []
    cosyvoice = CosyVoice('G:\\projects\\cozyvoice\\CosyVoice\\pretrained_models\\CosyVoice-300M-Instruct', load_jit=True, load_onnx=False, fp16=True)
    logging.info("{}\n".format(cosyvoice.list_avaliable_spks()))
    logging.info('get sft inference request\n')

    audio_queue = queue.Queue()
    def producer():
        for i, j in enumerate(cosyvoice.inference_instruct(tts_text, sft_dropdown, instruct_text, stream=False)):
            j['tts_speech'].flatten(0)
            audio_chunk = j['tts_speech'].numpy().flatten()
            audio_tensor = torch.tensor(audio_chunk, dtype=torch.float32)
            audio_buffer.append(audio_tensor)
            audio_queue.put(audio_chunk)
        audio_queue.put(None) 

    # 消费者线程：从队列中取出音频块并播放
    def consumer():
        while True:
            audio_chunk = audio_queue.get()
            if audio_chunk is None:
                break  # 结束标志
            playchunk(audio_chunk, target_samplerate)
            audio_queue.task_done()
    
    # 创建并启动生产者线程
    producer_thread = threading.Thread(target=producer)
    producer_thread.start()
    
    # 创建并启动消费者线程
    consumer_thread = threading.Thread(target=consumer)
    consumer_thread.start()
    
    # 等待生产者线程完成
    producer_thread.join()
    # 将所有音频块合并成一个完整的音频张量
    full_audio = torch.cat(audio_buffer, dim=0)

    # 保存音频文件
    output_filename = 'C:\\users\\alphabet\\desktop\\generated_audio.wav'
    torchaudio.save(output_filename, full_audio.unsqueeze(0), target_samplerate)    
    # 等待消费者线程完成
    consumer_thread.join()
    
def playchunk(audio_chunk, samplerate):
    sd.play(audio_chunk, samplerate)
    sd.wait()

if __name__=="__main__":
    tts_text:str = paste()
    sft_dropdown = "中文男"
    Instruct = "Kang Hui, a news anchor renowned for his composure and professionalism, is delivering a political news segment with a calm and steady tone, ensuring clarity and engagement with the audience."
    Instruct = "Zhang Xiuqi, a male political teacher known for his wisdom, humor, and passionate delivery, is explaining a complex political theory with a confident and slightly playful tone, at a moderate pace, ensuring clarity and engagement with his students."
    generate_audio(tts_text,sft_dropdown,Instruct)
