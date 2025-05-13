import os
import sys
import logging
from pathlib import Path
from modelscope import snapshot_download
import torch
import torchaudio

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 模型配置
MODELS = {
    'CosyVoice-300M': {
        'id': 'iic/CosyVoice-300M',
        'required_files': ['campplus.onnx', 'cosyvoice.yaml']
    },
    'CosyVoice-300M-SFT': {
        'id': 'iic/CosyVoice-300M-SFT',
        'required_files': ['campplus.onnx', 'cosyvoice.yaml']
    },
    'CosyVoice-300M-Instruct': {
        'id': 'iic/CosyVoice-300M-Instruct',
        'required_files': ['campplus.onnx', 'cosyvoice.yaml']
    },
    'CosyVoice2-0.5B': {
        'id': 'iic/CosyVoice2-0.5B',
        'required_files': ['campplus.onnx', 'cosyvoice2.yaml']
    },
    'CosyVoice-ttsfrd': {
        'id': 'iic/CosyVoice-ttsfrd',
        'required_files': ['resource.zip']
    }
}

def check_model_files(model_dir: str, required_files: list) -> bool:
    """检查模型文件是否完整"""
    model_path = Path(model_dir)
    if not model_path.exists():
        logging.error(f"Model directory {model_dir} does not exist!")
        return False
    
    missing_files = []
    for file in required_files:
        if not (model_path / file).exists():
            missing_files.append(file)
    
    if missing_files:
        logging.error(f"Missing required files in {model_dir}:")
        for file in missing_files:
            logging.error(f"  - {file}")
        return False
    
    logging.info(f"All required files found in {model_dir}")
    return True

def download_model(model_name: str, model_info: dict) -> bool:
    """下载并检查模型"""
    model_dir = f"pretrained_models/{model_name}"
    
    # 如果目录已存在，先检查文件
    if os.path.exists(model_dir):
        logging.info(f"Model directory {model_dir} already exists, checking files...")
        if check_model_files(model_dir, model_info['required_files']):
            logging.info(f"Model {model_name} is already downloaded and complete.")
            return True
        else:
            logging.warning(f"Model {model_name} is incomplete, will re-download.")
    
    # 下载模型
    try:
        logging.info(f"Downloading {model_name}...")
        snapshot_download(model_info['id'], local_dir=model_dir)
        logging.info(f"Download completed for {model_name}")
        
        # 检查文件
        if check_model_files(model_dir, model_info['required_files']):
            logging.info(f"Model {model_name} downloaded and verified successfully.")
            
            # 如果是 ttsfrd 模型，解压 resource.zip
            if model_name == 'CosyVoice-ttsfrd':
                try:
                    import zipfile
                    zip_path = os.path.join(model_dir, 'resource.zip')
                    logging.info(f"Extracting {zip_path}...")
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(model_dir)
                    logging.info("Extraction completed successfully")
                except Exception as e:
                    logging.error(f"Error extracting resource.zip: {str(e)}")
                    return False
            
            return True
        else:
            logging.error(f"Model {model_name} download completed but files are incomplete.")
            return False
            
    except Exception as e:
        logging.error(f"Error downloading {model_name}: {str(e)}")
        return False

def main():
    # 创建模型目录
    os.makedirs("pretrained_models", exist_ok=True)
    
    # 下载所有模型
    success = True
    for model_name, model_info in MODELS.items():
        if not download_model(model_name, model_info):
            success = False
            logging.error(f"Failed to download or verify {model_name}")
    
    if success:
        logging.info("All models downloaded and verified successfully!")
        logging.info("\n安装 ttsfrd 包（可选）：")
        logging.info("cd pretrained_models/CosyVoice-ttsfrd/")
        logging.info("pip install ttsfrd_dependency-0.1-py3-none-any.whl")
        logging.info("pip install ttsfrd-0.4.2-cp310-cp310-linux_x86_64.whl")
    else:
        logging.error("Some models failed to download or verify. Please check the logs above.")
        sys.exit(1)

if __name__ == "__main__":
    main() 