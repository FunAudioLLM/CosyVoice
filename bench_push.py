"""Push higher concurrency + short text benchmark."""
import sys
sys.path.append('third_party/Matcha-TTS')

from vllm import ModelRegistry
from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM
ModelRegistry.register_model('CosyVoice2ForCausalLM', CosyVoice2ForCausalLM)

from cosyvoice.cli.cosyvoice import AutoModel
import bench_cosyvoice3 as B


def main():
    m = AutoModel(model_dir='pretrained_models/Fun-CosyVoice3-0.5B', load_trt=True, load_vllm=True, fp16=False)
    print('===SHORT TEXT, push concurrency===', flush=True)
    B.bench_concurrent(m, text_name='short', concurrencies=(4, 8, 16, 32), per_round=4)
    print('===MEDIUM TEXT, push concurrency===', flush=True)
    B.bench_concurrent(m, text_name='medium', concurrencies=(8, 16, 32), per_round=2)


if __name__ == '__main__':
    main()
