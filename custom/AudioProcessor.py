import os
import uuid
import numpy as np
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from noisereduce import reduce_noise
from fastapi import UploadFile
from custom.file_utils import logging

class AudioProcessor:
    def __init__(self, input_dir="results/input", output_dir="results/output"):
        """
        初始化音频处理器。
        :param input_dir: 输入文件目录
        :param output_dir: 输出文件目录
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
    
    @staticmethod
    def volume_safely(audio: AudioSegment, volume_multiplier: float = 1.0) -> AudioSegment:
        """
        安全地调整音频音量。
        :param audio: AudioSegment 对象，音频数据。
        :param volume_multiplier: float，音量倍数，1.0 为原音量，大于 1 提高音量，小于 1 降低音量。
        :return: 调整后的 AudioSegment 对象。
        """
        logging.info(f"volume_multiplier: {volume_multiplier}")
        if volume_multiplier <= 0:
            raise ValueError("volume_multiplier 必须大于 0")

        # 计算增益（分贝），根据倍数调整
        gain_in_db = 20 * np.log10(volume_multiplier)

        # 应用增益调整音量
        audio = audio.apply_gain(gain_in_db)

        # 确保音频不削波（归一化到峰值 -0.1 dB 以下）
        audio = audio.normalize(headroom=0.1)

        return audio

    def generate_wav(self, audio_data, sample_rate, delay=0.0, volume_multiplier = 1.0):
        """
        使用 pydub 将音频数据转换为 WAV 格式，并支持添加延迟。
        :param audio_data: numpy 数组，音频数据
        :param sample_rate: int，采样率
        :param delay: float，延迟时间（单位：秒），默认为 0
        :param volume_multiplier: float，音量倍数，默认为 1.0
        :return: 文件路径，生成的 WAV 文件路径
        """
        # 确保 audio_data 是 numpy 数组
        if not isinstance(audio_data, np.ndarray):
            raise ValueError("audio_data 必须是 numpy 数组。")
        # 生成静音数据（如果有延迟需求）
        if delay > 0:
            num_silence_samples = int(delay * sample_rate)
            silence = np.zeros(num_silence_samples, dtype=audio_data.dtype)
            audio_data = np.concatenate((silence, audio_data), axis=0)
        # 检测音频数据类型并转换
        sample_width = 2
        if audio_data.dtype == np.float32:
            # 如果是 float32 数据，量化到 int16
            audio_data = (audio_data * 32767).astype(np.int16)
            sample_width = 2  # 16-bit (2 bytes per sample)
        elif audio_data.dtype == np.int16:
            sample_width = 2  # 16-bit (2 bytes per sample)
        elif audio_data.dtype == np.int8:
            audio_data = audio_data.astype(np.int16) * 256  # 转换为 int16
            sample_width = 2  # 16-bit
        else:
            raise ValueError("audio_data.dtype 不正确。")
        # 检测声道数
        if len(audio_data.shape) == 1:  # 单声道
            channels = 1
        elif len(audio_data.shape) == 2:  # 多声道
            channels = audio_data.shape[1]
        else:
            raise ValueError("audio_data.shape 格式不正确，必须是 1D 或 2D numpy 数组。")
        # 使用 pydub 生成音频段
        audio_segment = AudioSegment(
            audio_data.tobytes(),
            frame_rate=sample_rate,
            sample_width=sample_width,
            channels=channels
        )
        if volume_multiplier != 1.0:
            # 安全地增加音量
            audio_segment = self.volume_safely(audio_segment, volume_multiplier)
        # 指定保存文件的路径
        filename = f"{str(uuid.uuid4())}.wav"
        wav_path = os.path.join(self.output_dir, filename)
        # 如果文件已存在，先删除
        if os.path.exists(wav_path):
            os.remove(wav_path)
        # 导出 WAV 文件
        audio_segment.export(wav_path, format="wav")

        return wav_path        
    
    @staticmethod
    def audio_to_np_array(audio: AudioSegment):
        """将 AudioSegment 转换为 NumPy 数组"""
        return np.array(audio.get_array_of_samples())
    
    @staticmethod
    def np_array_to_audio(np_array, audio: AudioSegment):
        """将 NumPy 数组转换回 AudioSegment"""
        return AudioSegment(
            np_array.tobytes(),
            frame_rate=audio.frame_rate,
            sample_width=audio.sample_width,
            channels=audio.channels
        )

    async def save_upload_to_wav(
            self, 
            upload_file: UploadFile, 
            prefix: str, 
            volume_multiplier: float = 1.0, 
            nonsilent: bool = False, 
            reduce_noise_enabled: bool = True
        ):
        """保存上传文件并转换为 WAV 格式（如果需要）"""
        # 构造文件路径
        upload_path = os.path.join(self.input_dir, f'{prefix}{upload_file.filename}')
        # 删除同名已存在的文件
        if os.path.exists(upload_path):
            os.remove(upload_path)
        # 检查文件格式并转换为 WAV（如果需要）
        if not upload_path.lower().endswith(".wav"):
            wav_path = f"{os.path.splitext(upload_path)[0]}_new.wav"
        else:
            wav_path = upload_path

        logging.info(f"接收上传{upload_file.filename}请求 {upload_path}")

        try:
            # 保存上传的音频文件
            with open(upload_path, "wb") as f:
                f.write(await upload_file.read())
            # 加载音频
            audio = AudioSegment.from_file(upload_path)        
            # 降噪处理
            if reduce_noise_enabled:
                logging.info("reduce noise start")
                # 转换为 NumPy 数组
                audio_np = self.audio_to_np_array(audio)
                # 使用前 0.3 秒作为噪音参考（假设音频开头为背景噪音）
                noise_duration = int(audio.frame_rate * 0.3)  # 0.3 秒对应的采样点数量
                noise_profile = audio_np[:noise_duration]  # 提取前 0.3 秒
                # 使用较温和的降噪参数
                reduced_audio_np = reduce_noise(
                    y=audio_np,
                    sr=audio.frame_rate,
                    y_noise=noise_profile,
                    n_std_thresh_stationary=2.0,  # 提高阈值，减少过度降噪
                    prop_decrease=0.8  # 降低噪声衰减比例
                )
                # 转换回 AudioSegment
                audio = self.np_array_to_audio(reduced_audio_np, audio)
            # 去除前后静音
            if nonsilent:
                logging.info("nonsilent start")
                nonsilent_ranges = detect_nonsilent(audio, min_silence_len=300, silence_thresh=audio.dBFS - 16)
                if nonsilent_ranges:
                    start_trim = nonsilent_ranges[0][0]
                    end_trim = nonsilent_ranges[-1][1]
                    audio = audio[start_trim:end_trim]

            if volume_multiplier != 1.0:
                audio = self.volume_safely(audio, volume_multiplier=volume_multiplier)
            # 保存调整后的音频
            audio.export(wav_path, format="wav")

            return wav_path
        except Exception as e:
            raise Exception(f"{upload_file.filename}音频文件保存或转换失败: {str(e)}")
            