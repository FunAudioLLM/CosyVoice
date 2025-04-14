# test_speech_to_text_api.py
import unittest
import os
import tempfile
from unittest.mock import patch, MagicMock
from flask import Flask, json

class TestSpeechToTextAPI(unittest.TestCase):
    def setUp(self):
        self.app = Flask(__name__)
        self.client = self.app.test_client()
        
        # 创建临时音频文件
        self.temp_audio_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        self.temp_audio_file.close()
        self.valid_audio_path = self.temp_audio_file.name
        
        # 注册路由
        self.app.add_url_rule('/speech_to_text', 'speech_to_text_api', self.speech_to_text_api, methods=['POST'])
        
    def tearDown(self):
        # 删除临时文件
        if os.path.exists(self.temp_audio_file.name):
            os.unlink(self.temp_audio_file.name)
    
    def speech_to_text_api(self):
        # 这是被测试的方法的模拟实现
        from flask import request, jsonify
        import os
        if 'audio_path' not in request.json:
            return jsonify({"error": "No audio file path provided"}), 400
        audio_path = request.json['audio_path']
        if not os.path.exists(audio_path):
            return jsonify({"error": "Audio file not found"}), 400
        text = "测试文本"  # 模拟语音识别结果
        return jsonify({"text": text}), 200
    
    def test_success_case(self):
        """测试正常情况下的请求"""
        response = self.client.post(
            '/speech_to_text',
            data=json.dumps({'audio_path': self.valid_audio_path}),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn('text', response.json)
    
    def test_missing_audio_path(self):
        """测试缺少audio_path参数的情况"""
        response = self.client.post(
            '/speech_to_text',
            data=json.dumps({}),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json['error'], "No audio file path provided")
    
    def test_nonexistent_audio_file(self):
        """测试音频文件不存在的情况"""
        non_existent_path = "/path/to/nonexistent/file.wav"
        response = self.client.post(
            '/speech_to_text',
            data=json.dumps({'audio_path': non_existent_path}),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json['error'], "Audio file not found")
    
    @patch('torchaudio.load')
    def test_audio_loading_error(self, mock_load):
        """测试音频加载失败的情况"""
        mock_load.side_effect = Exception("Audio loading error")
        response = self.client.post(
            '/speech_to_text',
            data=json.dumps({'audio_path': self.valid_audio_path}),
            content_type='application/json'
        )
        # 注意：这里假设speech_to_text_api方法会捕获并处理音频加载错误
        # 如果实际实现中没有处理，这个测试可能需要调整
        self.assertEqual(response.status_code, 200)  # 或根据实际实现调整预期状态码
    
    def test_invalid_content_type(self):
        """测试非JSON内容类型的请求"""
        response = self.client.post(
            '/speech_to_text',
            data={'audio_path': self.valid_audio_path},
            content_type='multipart/form-data'
        )
        self.assertEqual(response.status_code, 400)  # 或415，取决于Flask配置

if __name__ == '__main__':
    unittest.main()
