import os
import datetime
import traceback
from langdetect import detect
from custom.file_utils import logging

class TextProcessor:
    """
    文本处理工具类，提供多种文本相关功能。
    """
    @staticmethod
    def detect_language(text):
        """
        检测输入文本的语言。
        :param text: 输入文本
        :return: 返回检测到的语言代码（如 'en', 'zh-cn'）
        """
        try:
            lang = detect(text)
            logging.info(f'Detected language: {lang}')
            return lang
        except Exception as e:
            logging.error(f"Language detection failed: {e}")
            return None
    
    @staticmethod
    def ensure_sentence_ends_with_period(text):
        """
        确保输入文本以适当的句号结尾。
        :param text: 输入文本
        :return: 修改后的文本
        """
        if not text.strip():
            return text  # 空文本直接返回
        # 判断是否已经以句号结尾
        if text[-1] in ['.', '。', '！', '!', '？', '?']:
            return text
        # 根据文本内容添加适当的句号
        lang = TextProcessor.detect_language(text)
        if lang == 'zh-cn': # 中文文本
            return text + '。'
        else:  # 英文或其他
            return text + '.'

    @staticmethod
    def log_error(exception: Exception, log_dir='error'):
        """
        记录错误信息到指定目录，并按日期时间命名文件。

        :param exception: 捕获的异常对象
        :param log_dir: 错误日志存储的目录，默认为 'error'
        """
        # 确保日志目录存在
        os.makedirs(log_dir, exist_ok=True)
        # 获取当前时间戳，格式化为 YYYY-MM-DD_HH-MM-SS
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        # 创建日志文件路径
        log_file_path = os.path.join(log_dir, f'error_{timestamp}.log')
        # 使用 traceback 模块获取详细的错误信息
        error_traceback = traceback.format_exc()
        # 写入错误信息到文件
        with open(log_file_path, 'w') as log_file:
            log_file.write(f"错误发生时间: {timestamp}\n")
            log_file.write(f"错误信息: {str(exception)}\n")
            log_file.write("堆栈信息:\n")
            log_file.write(error_traceback + '\n')
        
        logging.info(f"错误信息已保存至: {log_file_path}")
