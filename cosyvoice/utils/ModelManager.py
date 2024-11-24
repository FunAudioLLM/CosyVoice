import threading
from cosyvoice.cli.cosyvoice import CosyVoice

class ModelManager:
    def __init__(self):
        self.cosyvoice = None
        self.cosyvoice_sft = None
        self.cosyvoice_instruct = None
        self.sft_spk = None
        self.lock = threading.Lock()

    def load_models(self):
        with self.lock:  # 确保线程安全
            if self.cosyvoice is None:
                self.cosyvoice = CosyVoice('pretrained_models/CosyVoice-300M')
            if self.cosyvoice_sft is None:
                self.cosyvoice_sft = CosyVoice(
                    'pretrained_models/CosyVoice-300M-SFT',
                    load_jit=True, load_onnx=False, fp16=True)
                self.sft_spk = self.cosyvoice_sft.list_avaliable_spks()
            if self.cosyvoice_instruct is None:
                self.cosyvoice_instruct = CosyVoice('pretrained_models/CosyVoice-300M-Instruct')

    def get_model(self, model_type: str):
        """
        获取指定类型的模型实例。
        model_type: str, 可选值为 "cosyvoice", "cosyvoice_sft", "cosyvoice_instruct"
        """
        if model_type == "cosyvoice":
            return self.cosyvoice
        elif model_type == "cosyvoice_sft":
            return self.cosyvoice_sft
        elif model_type == "cosyvoice_instruct":
            return self.cosyvoice_instruct
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
