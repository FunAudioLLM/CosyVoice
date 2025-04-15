import sounddevice as sd
import numpy as np
import queue
import threading
import time
import logging
import torch


# class StreamPlayer:
#     def __init__(self, sample_rate=22050, channels=1, block_size=8192):
#         self.sample_rate = sample_rate
#         self.channels = channels
#         self.block_size = block_size
#         self.audio_queue = queue.Queue()
#         self.playing = False
#         self.play_thread = None
        
#     def start(self):
#         """启动播放线程"""
#         self.playing = True
#         self.play_thread = threading.Thread(target=self._play_loop)
#         # 将线程设置为守护线程，这样当主程序退出时，该线程会自动终止
#         # 避免因为播放线程未结束而导致程序无法正常退出
#         # 如果不设置daemon=True，则主程序结束时会等待该线程完成
#         self.play_thread.daemon = True
#         self.play_thread.start()
    
#     def _play_loop(self):
#         """播放线程循环函数"""
#         while self.playing:
#             try:
#                 # 从队列获取音频数据
#                 audio_data = self.audio_queue.get(timeout=0.2)
                
#                 # 如果获取到有效数据则播放
#                 if audio_data is not None and len(audio_data) > 0:
#                     try:
#                         # 确保数据格式正确
#                         sd.play(audio_data, self.sample_rate, blocksize=self.block_size)
#                         # 等待播放完成
#                         sd.wait()
#                     except sd.PortAudioError as e:
#                         print(f"音频播放错误: {e}")
#                         # 短暂暂停后继续
#                         time.sleep(0.5)
                
#                 self.audio_queue.task_done()
#             except queue.Empty:
#                 # 队列为空时短暂休眠
#                 time.sleep(0.01)
    
#     def play(self, audio_data):
#         """将音频数据添加到播放队列"""
#         self.audio_queue.put(audio_data)
    
#     def stop(self):
#         """停止播放并清理资源"""
#         self.playing = False
#         if self.play_thread and self.play_thread.is_alive():
#             self.play_thread.join(timeout=1.0)
        
#         # 清空队列
#         try:
#             while True:
#                 self.audio_queue.get_nowait()
#                 self.audio_queue.task_done()
#         except queue.Empty:
#             pass

# 实现自定义的音频播放器，使用sounddevice的回调机制
class StreamPlayer:
    def __init__(self, sample_rate=22050, channels=1, block_size=8192, latency='low'):
        self.sample_rate = sample_rate
        self.channels = channels
        self.audio_queue = queue.Queue()
        self.stream = None
        self.is_playing = False
        self.block_size = block_size
        self.latency = latency
        self.current_audio = None
        self.position = 0
        
    def _audio_callback(self, outdata, frames, time, status):
        """
        这是sounddevice回调函数，负责从队列提取音频并输出
        """
        if status:
            logging.warning(f"流状态: {status}")
            
        # 如果当前没有音频数据或已播放完
        if self.current_audio is None or self.position >= len(self.current_audio):
            try:
                # 使用get_nowait()如果队列为空会立即引发异常
                # 使用较小的超时值保持低延迟，同时给足够时间等待新数据
                self.current_audio = self.audio_queue.get(timeout=0.05)
                self.position = 0
                self.audio_queue.task_done()
            except queue.Empty:
                # 队列为空，但我们不希望有明显静音
                # 只填充很少的静音样本或重复最后一帧以减少感知中断
                outdata.fill(0)
                return
        
        # 计算可用的音频数据量
        available = len(self.current_audio) - self.position
        
        # 如果有足够数据，直接复制
        if available >= len(outdata):
            outdata[:] = self.current_audio[self.position:self.position+len(outdata)]
            self.position += len(outdata)
        else:
            # 复制剩余的可用数据
            outdata[:available] = self.current_audio[self.position:self.position+available]
            
            # 尝试立即获取下一段音频继续填充，而不是填充静音
            try:
                next_audio = self.audio_queue.get_nowait()
                self.audio_queue.task_done()
                
                # 计算需要从新音频填充的样本数
                remaining = len(outdata) - available
                
                # 确保不超出新音频的长度
                if remaining <= len(next_audio):
                    outdata[available:] = next_audio[:remaining]
                    # 更新当前音频和位置
                    self.current_audio = next_audio
                    self.position = remaining
                else:
                    # 新音频不够填充剩余部分
                    outdata[available:available+len(next_audio)] = next_audio
                    outdata[available+len(next_audio):].fill(0)  # 剩余部分仍需填充静音
                    self.current_audio = None
                    self.position = 0
            except queue.Empty:
                # 没有新的音频数据，只能填充静音
                outdata[available:].fill(0)
                self.current_audio = None
                self.position = 0
        
    def start(self):
        """启动音频流"""
        if self.stream is None or not self.stream.active:
            self.stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=self.channels, 
                callback=self._audio_callback,
                blocksize=self.block_size,
                latency=self.latency
            )
            self.stream.start()
            self.is_playing = True
            logging.debug("音频播放流已启动")
    
    def play(self, audio_data):
        """添加音频数据到播放队列"""
            
        # # 确保数据是正确的形状 (二维数组)
        # if len(audio_data.shape) == 1:
        #     audio_data = audio_data.reshape(-1, 1)
        
        # 添加到队列
        self.audio_queue.put(audio_data)
    
    def stop(self):
        """停止音频流"""
        if self.stream is not None and self.stream.active:
            self.stream.stop()
            self.stream.close()
            self.stream = None
            self.is_playing = False
            # 清空队列
            try:
                while True:
                    self.audio_queue.get_nowait()
                    self.audio_queue.task_done()
            except queue.Empty:
                pass
            logging.debug("音频播放流已停止")
    
    def is_empty(self):
        """检查队列是否为空"""
        return self.audio_queue.empty() and (self.current_audio is None or self.position >= len(self.current_audio))
