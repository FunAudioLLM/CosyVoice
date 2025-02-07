import tkinter as tk
from tkinter import messagebox
from cosyvoice.cli.cosyvoice import CosyVoice
import torch
import torchaudio
from cosyvoice.utils.file_utils import logging
import sounddevice as sd
import soundfile as sf
import threading
import queue
from pyperclip import paste
import os

# 全局变量，用于存储加载的模型
cosyvoice = None
audio_queue = queue.Queue()

def load_model():
    global cosyvoice
    if cosyvoice is None:
        cosyvoice = CosyVoice('G:\\projects\\cozyvoice\\CosyVoice\\pretrained_models\\CosyVoice-300M-Instruct', load_jit=True, load_onnx=False, fp16=True)

def generate_audio():
    target_samplerate = 22050  # 目标采样率
    audio_buffer = []
    tts_text = paste()  
    sft_dropdown = "中文男"
    instruct_text = "Zhang Xiuqi, a male political teacher known for his wisdom, humor, and passionate delivery, is explaining a complex political theory with a confident and slightly playful tone, at a moderate pace, ensuring clarity and engagement with his students."

    def producer():
        for i, j in enumerate(cosyvoice.inference_instruct(tts_text, sft_dropdown, instruct_text, stream=False)):
            j['tts_speech'].flatten(0)
            audio_chunk = j['tts_speech'].numpy().flatten()
            audio_tensor = torch.tensor(audio_chunk, dtype=torch.float32)
            audio_buffer.append(audio_tensor)
            audio_queue.put(audio_chunk)
        audio_queue.put(None)

    def consumer():
        while True:
            audio_chunk = audio_queue.get()
            if audio_chunk is None:
                break  # 结束标志
            playchunk(audio_chunk, target_samplerate)
            audio_queue.task_done()

    producer_thread = threading.Thread(target=producer)
    producer_thread.start()

    consumer_thread = threading.Thread(target=consumer)
    consumer_thread.start()

    producer_thread.join()
    full_audio = torch.cat(audio_buffer, dim=0)

    output_filename = 'C:\\users\\alphabet\\desktop\\generated_audio.wav'
    torchaudio.save(output_filename, full_audio.unsqueeze(0), target_samplerate)
    consumer_thread.join()

def playchunk(audio_chunk, samplerate):
    sd.play(audio_chunk, samplerate)
    sd.wait()

def play_generated_audio():
    output_filename = 'C:\\users\\alphabet\\desktop\\generated_audio.wav'
    if os.path.exists(output_filename):
        waveform, sample_rate = sf.read(output_filename)
        sd.play(waveform, sample_rate)
        sd.wait()
    else:
        messagebox.showwarning("警告", "音频文件不存在，请先生成音频")

def on_generate_button_click():
    global display_label
    display_label['text'] = paste()
    root.update_idletasks()
    threading.Thread(target=generate_audio).start()

def on_play_generated_audio_button_click():
    threading.Thread(target=play_generated_audio).start()
    
     
# 创建主窗口
root = tk.Tk()
root.title("CozyVoice TTS")

# 显示复制的文本
display_label = tk.Label(root, text="已复制的文本:",justify=tk.LEFT)
display_label.pack(pady=5)


# 生成按钮
generate_button = tk.Button(root, text="生成并播放音频", command=on_generate_button_click)
generate_button.pack(pady=20)

play_button = tk.Button(root,text='播放音频',command=on_play_generated_audio_button_click)
play_button.pack(pady=25)



# 加载模型
load_model()

# 运行主循环
root.mainloop()