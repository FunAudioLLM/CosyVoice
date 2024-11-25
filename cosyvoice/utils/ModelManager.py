import threading
from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import logging

class ModelManager:
    def __init__(self):
        self.models = {
            "cosyvoice": None,
            "cosyvoice_sft": None,
            "cosyvoice_instruct": None,
        }
        self.sft_spk = None
        self.locks = {
            "cosyvoice": threading.Lock(),
            "cosyvoice_sft": threading.Lock(),
            "cosyvoice_instruct": threading.Lock(),
        }

    def _load_model(self, model_type: str):
        """
        内部方法：加载指定类型的模型。
        """
        logging.info(f"Loading model: {model_type}")
        if model_type == "cosyvoice":
            return CosyVoice('pretrained_models/CosyVoice-300M')
        elif model_type == "cosyvoice_sft":
            model = CosyVoice(
                'pretrained_models/CosyVoice-300M-SFT',
                load_jit=True, load_onnx=False, fp16=True
            )
            self.sft_spk = model.list_avaliable_spks()
            return model
        elif model_type == "cosyvoice_instruct":
            return CosyVoice('pretrained_models/CosyVoice-300M-Instruct')
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def get_model(self, model_type: str):
        """
        获取指定类型的模型实例，按需加载，确保线程安全。
        """
        logging.info(f"get_model: {model_type}")
        if model_type not in self.models:
            raise ValueError(f"Unsupported model type: {model_type}")

        # 如果模型尚未加载，则加载
        if self.models[model_type] is None:
            with self.locks[model_type]:  # 确保线程安全
                if self.models[model_type] is None:  # 双重检查锁定
                    self.models[model_type] = self._load_model(model_type)
        
        return self.models[model_type]