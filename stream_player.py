import sounddevice as sd
import numpy as np
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
    def __init__(self, sample_rate=22050, channels=1, block_size=4096, latency='high', max_buffer_size=3000000):
        """
        使用连续缓冲区的音频播放器
        
        参数:
            sample_rate: 采样率
            channels: 通道数
            block_size: 音频处理块大小
            latency: 延迟设置 ('low', 'high', 'medium')
            max_buffer_size: 缓冲区最大样本数，超过会截断
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.block_size = block_size
        self.latency = latency
        self.max_buffer_size = max_buffer_size
        
        # 音频缓冲区，所有新音频都拼接到这里
        self.buffer = np.zeros((0,), dtype=np.float32)
        self.buffer_lock = threading.Lock()  # 用于同步访问缓冲区
        
        # 当前播放位置
        self.position = 0
        self.stream = None
        self.is_playing = False
        
    def _audio_callback(self, outdata, frames, time, status):
        """
        sounddevice回调函数，从连续缓冲区读取数据
        """
        if status:
            logging.warning(f"流状态: {status}")
        
        with self.buffer_lock:
            # 计算可用的音频数据量 (缓冲区长度 - 当前位置)
            available = len(self.buffer) - self.position
            
            if available <= 0:
                # 缓冲区中没有可用数据，播放静音
                outdata.fill(0)
                return
            
            # 确定要播放的样本数
            play_length = min(len(outdata), available)
            
            # 复制数据到输出缓冲区
            outdata[:play_length] = self.buffer[self.position:self.position+play_length].reshape(-1, 1)
            
            # 如果没有足够数据填满输出缓冲区，剩余部分填充静音
            if play_length < len(outdata):
                outdata[play_length:].fill(0)
                logging.info(f"缓冲区数据不足，部分输出静音 ({play_length}/{len(outdata)})")
            
            # 更新位置
            self.position += play_length
            
            # 如果位置超过了设定的阈值，裁剪缓冲区
            if self.position > self.max_buffer_size // 2:
                # 保留后半部分缓冲区
                self.buffer = self.buffer[self.position - self.block_size:]
                # 重置位置，保留一个块的余量防止播放断裂
                self.position = self.block_size
                logging.debug(f"缓冲区已裁剪，新长度: {len(self.buffer)}")
    
    def start(self):
        """启动音频流"""
        if self.stream is None or not self.stream.active:
            try:
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
            except sd.PortAudioError as e:
                logging.error(f"启动音频流失败: {e}")
                raise
    
    def play(self, audio_data):
        """
        添加音频数据到连续缓冲区
        """
        if not isinstance(audio_data, np.ndarray):
            audio_data = np.array(audio_data, dtype=np.float32)
        
        # 保证数据是一维的
        if len(audio_data.shape) > 1:
            audio_data = audio_data.flatten()
        
        with self.buffer_lock:
            # 将新数据附加到缓冲区
            self.buffer = np.concatenate((self.buffer, audio_data))
            
            # 如果缓冲区超过最大大小，则裁剪
            if len(self.buffer) > self.max_buffer_size:
                # 保留后半部分，确保当前播放位置之后的数据不会丢失
                keep_from = max(0, self.position - self.block_size)
                self.buffer = self.buffer[keep_from:]
                self.position -= keep_from
                logging.debug(f"缓冲区已裁剪，新长度: {len(self.buffer)}")
            
            # logging.debug(f"缓冲区状态: {self.get_buffer_status()}")
            # logging.debug(f"添加了 {len(audio_data)} 个样本，当前缓冲区大小: {len(self.buffer)}, 当前位置: {self.position}")
    
    def stop(self):
        """停止音频流并清理资源"""
        if self.stream is not None and self.stream.active:
            self.stream.stop()
            self.stream.close()
            self.stream = None
            self.is_playing = False
            
            with self.buffer_lock:
                # 清空缓冲区
                self.buffer = np.zeros((0,), dtype=np.float32)
                self.position = 0
            
            logging.debug("音频播放流已停止，缓冲区已清空")
    
    def is_empty(self):
        """检查缓冲区是否为空"""
        with self.buffer_lock:
            return len(self.buffer) <= self.position
    
    def get_buffer_status(self):
        """获取缓冲区状态"""
        with self.buffer_lock:
            total = len(self.buffer)
            available = max(0, total - self.position)
            return {
                "total_size": total,
                "position": self.position,
                "available": available,
                "buffer_seconds": available / self.sample_rate if self.sample_rate > 0 else 0
            }

    def start_with_prebuffer(self, min_buffer_samples=16384):
        """启动音频流，但先等待缓冲区达到最小大小"""
        # 检查缓冲区大小
        prebuffer_wait_start = time.time()
        max_wait_time = 3.0  # 最多等待3秒
        
        while len(self.buffer) < min_buffer_samples:
            # 检查是否超时
            if time.time() - prebuffer_wait_start > max_wait_time:
                logging.warning(f"预缓冲超时，当前缓冲区大小: {len(self.buffer)}")
                break
                
            time.sleep(0.05)  # 短暂休眠，等待缓冲区填充
            logging.debug(f"等待预缓冲，当前大小: {len(self.buffer)}/{min_buffer_samples}")
        
        # 正常启动流
        self.start()
