import requests
import logging
import argparse
parser = argparse.ArgumentParser(description='Test API')
parser.add_argument('--text', type=str, default='测试文本')
parser.add_argument('--audio_path', type=str, default='./reference/voice_id/1.mp3')
args = parser.parse_args()
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

print("args:", args)
url = 'http://127.0.0.1:5000/generate_audio'
data = {
    "text": args.text,
    "audio_path": args.audio_path
    # "audio_path": "./reference/voice_id/1.mp3"
}
headers = {'Content-Type': 'application/json'}

print("audio_path:", data['audio_path'])
print("text:", data['text'])

response = requests.post(url, json=data, headers=headers)

print("response content:", response.text)  # 打印响应内容

try:
    logging.info("response:", response.json())
except requests.exceptions.JSONDecodeError as e:
    logging.error(f"Failed to decode JSON response: {e}")

# logging.info("data_path:", data['audio_path'])
# logging.info("response:", response)
# logging.info("response:", response)
# logging.info("response:", response.status_code)
# logging.info("response:", response.json())
# logging.info(response.json())

print(response.json())