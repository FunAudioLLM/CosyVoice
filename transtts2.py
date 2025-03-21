import sys
from PySide6.QtWidgets import QApplication, QLabel, QPushButton, QCheckBox, QMessageBox, QVBoxLayout, QWidget
from PySide6.QtCore import QThread, Signal
from cosyvoice.cli.cosyvoice import CosyVoice2
import torchaudio
import numpy as np
import sounddevice as sd
import soundfile as sf
import threading
import queue
from pyperclip import paste
import os

# 全局变量，用于存储加载的模型
model_path = './pretrained_models/CosyVoice2-0.5B'
audio_queue = queue.Queue()
cosyvoice = None

def load_model():
    global cosyvoice 
    cosyvoice = CosyVoice2(model_path, load_jit=True, load_trt=False, fp16=False)

def load_wav(wav, target_sr):
    speech, sample_rate = torchaudio.load(wav, backend='soundfile')
    speech = speech.mean(dim=0, keepdim=True)
    if sample_rate != target_sr:
        assert sample_rate > target_sr, 'wav sample rate {} must be greater than {}'.format(sample_rate, target_sr)
        speech = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)(speech)
    return speech
def generate_audio():
    audio_buffer = []
    tts_text = paste()  
    import re
    tts_text = re.sub(r'\s+', ' ', tts_text).strip()
    prompt_speech_16k = load_wav('G:/projects/cozyvoice/CosyVoice/asset/prompt.wav', 16000)
    prompt_text = "在我们的考研中啊，这一块儿其实近十五年都没有考过，主要我觉得呢，是排不上队，或者说呢，过于的专业了。"

    def producer():
        for i, j in enumerate(cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k, stream=is_stream)):
            audio_chunk = j['tts_speech'].squeeze(0).numpy()
            audio_buffer.append(audio_chunk)
            audio_queue.put(audio_chunk)
        audio_queue.put(None)

    def consumer():
        while True:
            audio_tensor = audio_queue.get()
            if audio_tensor is None:
                break  # 结束标志
            playchunk(audio_tensor, cosyvoice.sample_rate)
            audio_queue.task_done()

    producer_thread = threading.Thread(target=producer)
    producer_thread.start()

    consumer_thread = threading.Thread(target=consumer)
    consumer_thread.start()

    producer_thread.join()
    full_audio = np.concatenate(audio_buffer, axis=0)

    output_filename = 'C:/users/alphabet/desktop/generated_audio.wav'
    sf.write(output_filename, full_audio, cosyvoice.sample_rate)
    consumer_thread.join()

def playchunk(audio_chunk, samplerate):
    sd.play(audio_chunk, samplerate)
    sd.wait()

def play_generated_audio():
    output_filename = 'C:/users/alphabet/desktop/generated_audio.wav'
    if os.path.exists(output_filename):
        waveform, sample_rate = sf.read(output_filename)
        sd.play(waveform, sample_rate)
        sd.wait()
    else:
        QMessageBox.warning(None, "警告", "音频文件不存在，请先生成音频")

def on_generate_button_click():
    display_label.setText(paste())
    threading.Thread(target=generate_audio).start()

def on_play_generated_audio_button_click():
    threading.Thread(target=play_generated_audio).start()

class ModelLoaderThread(QThread):
    # 定义信号，用于通知主线程模型加载完成
    model_loaded = Signal(object)

    def run(self):
        global cosyvoice
        cosyvoice = CosyVoice2(model_path, load_jit=True, load_trt=False, fp16=False)
        self.model_loaded.emit(cosyvoice)  # 发送加载完成的模型对象

# 修改主窗口逻辑以支持异步加载
app = QApplication(sys.argv)
window = QWidget()
window.setWindowTitle("CozyVoice TTS")

# 布局
layout = QVBoxLayout()

# 显示复制的文本
display_label = QLabel("已复制的文本:")
display_label.setWordWrap(True)  # 启用自动换行
layout.addWidget(display_label)

# 生成按钮
generate_button = QPushButton("生成并播放音频")
generate_button.setEnabled(False)  # 初始禁用按钮
generate_button.clicked.connect(on_generate_button_click)
layout.addWidget(generate_button)

# 播放按钮
play_button = QPushButton("播放音频")
play_button.setEnabled(False)  # 初始禁用按钮
play_button.clicked.connect(on_play_generated_audio_button_click)
layout.addWidget(play_button)

# 流式生成复选框
is_stream = QCheckBox("流式生成")
layout.addWidget(is_stream)

# 设置布局
window.setLayout(layout)

# 异步加载模型
def on_model_loaded(model):
    generate_button.setEnabled(True)  # 启用按钮
    play_button.setEnabled(True)  # 启用按钮

model_loader_thread = ModelLoaderThread()
model_loader_thread.model_loaded.connect(on_model_loaded)
model_loader_thread.start()

# 运行主循环
window.show()
sys.exit(app.exec_())
