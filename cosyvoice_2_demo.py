import sys
import os

# 设置根目录并添加第三方库路径
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append("{}/third_party/Matcha-TTS".format(ROOT_DIR))


import logging
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
# 设置日志级别为 DEBUG
logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger().setLevel(logging.DEBUG)

# 确保设置影响所有模块
for name in logging.root.manager.loggerDict:
    logging.getLogger(name).setLevel(logging.DEBUG)


from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio
import argparse
import time
import numpy as np
import queue
import threading
import sounddevice as sd
import torch

# 添加命令行参数解析
parser = argparse.ArgumentParser(description="CosyVoice2 Demo")
parser.add_argument(
    "--model_dir",
    type=str,
    default="pretrained_models/CosyVoice2-0.5B",
    help="模型目录路径",
)
parser.add_argument(
    "--fp16", action="store_true", default=False, help="是否使用半精度(fp16)推理"
)
args = parser.parse_args()

print(f"使用模型目录: {args.model_dir}")
cosyvoice = CosyVoice2(args.model_dir, load_jit=False, load_trt=True, fp16=args.fp16)

prompt_speech_16k = load_wav("./asset/sqr3.wav", 16000)

# 预热机制：先用一个很短的文本做一次非流式推理，让模型完成首次编译和加载
# 这样后续的正式推理就不会有明显延迟
next(cosyvoice.inference_sft(
    "预热",
    "中文女",
    stream=False,
    speed=1.0,
    text_frontend=True
))

print("模型预热完成，准备正式生成")

# 创建音频播放类，用于流式播放无间断音频
class AudioPlayer:
    def __init__(self, sample_rate=22050, buffer_size=20):
        self.sample_rate = sample_rate
        self.audio_queue = queue.Queue(maxsize=buffer_size)
        self.is_playing = False
        self.all_audio = [] if buffer_size > 0 else None  # 仅当需要保存时才累积
        self.save_enabled = False
        self.stream = None
        
        # 设置ALSA缓冲区参数，预防underrun
        self.blocksize = 8192  # 更大的块大小
        self.latency = 'high'  # 较高的延迟但更稳定
        
    def start(self):
        """启动播放线程"""
        self.is_playing = True
        # 初始化输出流
        self.stream = sd.OutputStream(
            samplerate=self.sample_rate, 
            channels=1,
            blocksize=self.blocksize, 
            latency=self.latency
        )
        self.stream.start()
        
        # 启动播放线程
        self.play_thread = threading.Thread(target=self._play_continuously)
        self.play_thread.daemon = True
        self.play_thread.start()
        
    def enable_save(self, enabled=True):
        """启用或禁用保存完整音频"""
        self.save_enabled = enabled
        if not enabled:
            self.all_audio = None
        else:
            self.all_audio = []
        
    def add_audio(self, audio_chunk):
        """添加音频块到播放队列"""
        # 如果audio_chunk是PyTorch张量，转换为numpy数组
        if isinstance(audio_chunk, torch.Tensor):
            audio_chunk = audio_chunk.numpy().T
        
        # 在向队列添加数据前预先分块，避免大块数据造成的停顿
        chunk_size = 4000  # 分块大小
        if len(audio_chunk) > chunk_size:
            # 将大块分成多个小块添加到队列
            for i in range(0, len(audio_chunk), chunk_size):
                end = min(i + chunk_size, len(audio_chunk))
                self.audio_queue.put(audio_chunk[i:end])
                
                # 如果启用了保存，保存每个块
                if self.save_enabled and self.all_audio is not None:
                    self.all_audio.append(audio_chunk[i:end])
        else:
            # 小块直接加入队列
            self.audio_queue.put(audio_chunk)
            
            # 如果启用了保存，保存整块
            if self.save_enabled and self.all_audio is not None:
                self.all_audio.append(audio_chunk)
    
    def _play_continuously(self):
        """连续播放队列中的音频"""
        try:
            # 播放循环
            while self.is_playing:
                try:
                    # 从队列获取音频块（最多等待0.1秒）
                    audio_chunk = self.audio_queue.get(timeout=0.1)
                    
                    # 检查数据是否有效
                    if audio_chunk.size > 0:
                        # 直接写入音频流
                        self.stream.write(audio_chunk)
                    
                    # 标记任务完成
                    self.audio_queue.task_done()
                    
                except queue.Empty:
                    # 队列为空时暂停一小段时间，但保持循环以保持流的连续性
                    time.sleep(0.005)  # 更短的睡眠时间以提高响应性
        except Exception as e:
            logging.error(f"播放线程发生错误: {e}")
        finally:
            # 确保线程退出时清理资源
            if self.stream is not None and self.stream.active:
                self.stream.stop()
                self.stream.close()
                
    def stop(self):
        """停止播放"""
        self.is_playing = False
        if hasattr(self, 'play_thread'):
            self.play_thread.join(timeout=1.0)
        
        # 关闭音频流
        if self.stream is not None:
            if self.stream.active:
                self.stream.stop()
            self.stream.close()
            self.stream = None
    
    def wait_until_done(self, timeout=5.0):
        """等待队列中所有音频播放完毕，带超时机制"""
        # 计算开始等待的时间
        start_wait = time.time()
        
        # 等待队列清空，最多等待timeout秒
        while not self.audio_queue.empty():
            if time.time() - start_wait > timeout:
                logging.warning("等待音频播放完成超时")
                break
            time.sleep(0.1)
        
        # 额外等待一小段时间确保最后一块音频播放完毕
        time.sleep(0.2)

    def save_audio(self, filename):
        """保存累积的所有音频到文件"""
        if not self.save_enabled or self.all_audio is None or len(self.all_audio) == 0:
            logging.warning("没有可保存的音频数据或保存功能未启用")
            return False
            
        try:
            # 合并所有音频块
            combined = np.concatenate(self.all_audio, axis=0)
            # 转换为torch张量格式并保存
            audio_tensor = torch.tensor(combined.T).float()  # 转置回来
            torchaudio.save(filename, audio_tensor, self.sample_rate)
            return True
        except Exception as e:
            logging.error(f"保存音频失败: {e}")
            return False


# 使用改进的播放机制进行流式语音生成和播放
print("ATTENTION: 文本已经给到模型，开始生成语音啦！！！")
start_time = time.time()

# 创建音频播放器实例，使用更大的缓冲区
player = AudioPlayer(sample_rate=cosyvoice.sample_rate, buffer_size=30)
# 如果需要保存音频，启用保存功能
# player.enable_save(True)
player.start()

# 流式生成并添加到播放队列
for i, j in enumerate(
    cosyvoice.inference_zero_shot(
        # "这句话里面到底在使用了谁的语音呢？",
        "CosyVoice迎来全面升级，提供更准、更稳、更快、 更好的语音生成能力。make america great again. ",
        "我会把三段话切成3段，用来做",
        prompt_speech_16k,
        stream=True,
    )
):
    current_time = time.time()
    logging.info(f"第 {i} 次生成耗时: {current_time - start_time:.2f} 秒")
    start_time = current_time
    
    # 添加到播放队列
    player.add_audio(j["tts_speech"])
    
    # 适当的小暂停，让播放线程有时间处理
    if i > 0 and player.audio_queue.qsize() > 10:
        time.sleep(0.01)

# 等待播放完成
player.wait_until_done()
player.stop()

# 可选：保存完整音频
# if player.save_enabled:
#     player.save_audio("完整音频输出_1.wav")

print("ATTENTION: 文本已经给到模型，开始生成语音啦！！！")
start_time = time.time()

# 创建新的播放器实例
player = AudioPlayer(sample_rate=cosyvoice.sample_rate, buffer_size=30)
player.start()

for i, j in enumerate(
    cosyvoice.inference_sft(
        # "这句话里面到底在使用了谁的语音呢？",
        "CosyVoice迎来全面升级，提供更准、更稳、更快、 更好的语音生成能力。make america great again. ",
        # "中文女",
        "xiaoluo_mandarin",
        stream=True,
    )
):
    current_time = time.time()
    logging.info(f"第 {i} 次生成耗时: {current_time - start_time:.2f} 秒")
    start_time = current_time
    
    # 添加到播放队列
    player.add_audio(j["tts_speech"])
    
    # 适当的小暂停，让播放线程有时间处理
    if i > 0 and player.audio_queue.qsize() > 10:
        time.sleep(0.01)

# 等待播放完成
player.wait_until_done()
player.stop()

# 最后一个示例，保存到文件而不是播放
start_time = time.time()
for i, j in enumerate(
    cosyvoice.inference_zero_shot(
        # "这句话里面到底在使用了谁的语音呢？",
        "CosyVoice迎来全面升级，提供更准、更稳、更快、 更好的语音生成能力。make america great again. ",
        "我会把三段话切成3段，用来做",
        prompt_speech_16k,
        stream=True,
    )
):
    current_time = time.time()
    logging.info(f"第 {i} 次生成耗时: {current_time - start_time:.2f} 秒")
    start_time = current_time
    torchaudio.save(
        "zero_shot_{}.wav".format(i), j["tts_speech"], cosyvoice.sample_rate
    )

# # instruct usage
# for i, j in enumerate(
#     cosyvoice.inference_instruct2(
#         "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。",
#         "用四川话说这句话",
#         prompt_speech_16k,
#         stream=True,
#     )
# ):
#     torchaudio.save("instruct_{}.wav".format(i), j["tts_speech"], cosyvoice.sample_rate)

# # NOTE if you want to reproduce the results on https://funaudiollm.github.io/cosyvoice2, please add text_frontend=False during inference
# # zero_shot usage
# prompt_speech_16k = load_wav('./asset/zero_shot_prompt.wav', 16000)
# for i, j in enumerate(cosyvoice.inference_zero_shot('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '希望你以后能够做的比我还好呦。', prompt_speech_16k, stream=False)):
#     torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

# # fine grained control, for supported control, check cosyvoice/tokenizer/tokenizer.py#L248
# for i, j in enumerate(cosyvoice.inference_cross_lingual('在他讲述那个荒诞故事的过程中，他突然[laughter]停下来，因为他自己也被逗笑了[laughter]。', prompt_speech_16k, stream=False)):
#     torchaudio.save('fine_grained_control_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

# # instruct usage
# for i, j in enumerate(cosyvoice.inference_instruct2('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '用四川话说这句话', prompt_speech_16k, stream=False)):
#     torchaudio.save('instruct_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

# # bistream usage, you can use generator as input, this is useful when using text llm model as input
# # NOTE you should still have some basic sentence split logic because llm can not handle arbitrary sentence length
# def text_generator():
#     yield '收到好友从远方寄来的生日礼物，'
#     yield '那份意外的惊喜与深深的祝福'
#     yield '让我心中充满了甜蜜的快乐，'
#     yield '笑容如花儿般绽放。'
# for i, j in enumerate(cosyvoice.inference_zero_shot(text_generator(), '希望你以后能够做的比我还好呦。', prompt_speech_16k, stream=False)):
#     torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

# start_time = time.time()

# for i, j in enumerate(
#     cosyvoice.inference_cross_lingual(
#         "在他讲述那个荒诞故事的过程中，他突然[laughter]停下来，因为他自己也被逗笑了[laughter]。",
#         "没有用到的文本",
#         prompt_speech_16k,
#         stream=True,
#     )
# ):
#     current_time = time.time()
#     print(f"第 {i} 次生成耗时: {current_time - start_time:.2f} 秒")
#     start_time = current_time
#     torchaudio.save(
#         "fine_grained_control_{}.wav".format(i), j["tts_speech"], cosyvoice.sample_rate
#     )
