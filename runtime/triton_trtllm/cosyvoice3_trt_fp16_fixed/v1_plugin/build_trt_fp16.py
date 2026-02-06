import sys
sys.path.insert(0, '/opt/code/CosyVoice')
sys.path.insert(0, '/opt/code/CosyVoice/third_party/Matcha-TTS')

from argparse import ArgumentParser
import os
import logging
import torch
from hyperpyyaml import load_hyperpyyaml
from cosyvoice.utils.file_utils import convert_onnx_to_trt
from cosyvoice.utils.common import TrtContextWrapper

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_args():
    parser = ArgumentParser()

    parser.add_argument(
        "--token2wav-path",
        type=str,
        default='/weights/Fun-CosyVoice3-0.5B-2512',
        help="Token2Wav path, default to %(default)r",
    )
    parser.add_argument(
        "--trt_plugin_lib_path",
        type=str,
        default='/opt/code/CosyVoice/runtime/triton_trtllm/cosyvoice3_trt_plugin/plugin_ln_3d_eps6/build/libpluginSo_cusLn3d_eps6.so',
    )
    
    args = parser.parse_args()
    return args


class CosyVoice3:

    def __init__(self, model_dir, load_trt=False, fp16=False, trt_concurrent=1, plugin_path=None):
        self.model_dir = model_dir
        self.fp16 = fp16

        hyper_yaml_path = '{}/cosyvoice3.yaml'.format(model_dir)
        if not os.path.exists(hyper_yaml_path):
            raise ValueError('{} not found!'.format(hyper_yaml_path))
        with open(hyper_yaml_path, 'r') as f:
            configs = load_hyperpyyaml(f, overrides={'qwen_pretrain_path': os.path.join(model_dir, 'CosyVoice-BlankEN')})

        self.model = CosyVoice3Model(configs['flow'], fp16)
        self.model.load('{}/flow.pt'.format(model_dir))
        if load_trt:
            if self.fp16 is True:
                logging.warning('DiT tensorRT fp16 engine have some performance issue, use at caution!')
            if plugin_path is None:
                trt_path = '{}/flow.decoder.estimator.{}.mygpu.plan'.format(model_dir, 'fp16' if self.fp16 is True else 'fp32')
                onnx_path = '{}/flow.decoder.estimator.fp32.onnx'.format(model_dir)
            else:
                trt_path = '{}/flow.decoder.estimator.{}.cus_ln.mygpu.plan'.format(model_dir, 'fp16' if self.fp16 is True else 'fp32')
                onnx_path = '{}/flow.decoder.estimator.fp32.cus_ln.onnx'.format(model_dir)
            logging.info('onnx_path: {}'.format(onnx_path))
            self.model.load_trt(trt_path,
                                onnx_path,
                                trt_concurrent,
                                self.fp16,
                                plugin_path)


class CosyVoice3Model:

    def __init__(self,
                 flow: torch.nn.Module,
                 fp16: bool = False):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.flow = flow
        self.fp16 = fp16

    def load(self, flow_model):
        self.flow.load_state_dict(torch.load(flow_model, map_location=self.device), strict=True)
        self.flow.to(self.device).eval()

    def load_trt(self, flow_decoder_estimator_model, flow_decoder_onnx_model, trt_concurrent, fp16, plugin_path=None):
        assert torch.cuda.is_available(), 'tensorrt only supports gpu!'

        import tensorrt as trt
        if plugin_path is not None:
            logger = trt.Logger(trt.Logger.INFO)
            import ctypes
            ctypes.CDLL(plugin_path)
            trt.init_libnvinfer_plugins(logger, "")

        if not os.path.exists(flow_decoder_estimator_model) or os.path.getsize(flow_decoder_estimator_model) == 0:
            convert_onnx_to_trt(flow_decoder_estimator_model, self.get_trt_kwargs(), flow_decoder_onnx_model, fp16)
        del self.flow.decoder.estimator
        with open(flow_decoder_estimator_model, 'rb') as f:
            estimator_engine = trt.Runtime(trt.Logger(trt.Logger.INFO)).deserialize_cuda_engine(f.read())
        assert estimator_engine is not None, 'failed to load trt {}'.format(flow_decoder_estimator_model)
        self.flow.decoder.estimator = TrtContextWrapper(estimator_engine, trt_concurrent=trt_concurrent, device=self.device)

    def get_trt_kwargs(self):
        min_shape = [(2, 80, 4), (2, 1, 4), (2, 80, 4), (2, 80, 4)]
        opt_shape = [(2, 80, 500), (2, 1, 500), (2, 80, 500), (2, 80, 500)]
        max_shape = [(2, 80, 3000), (2, 1, 3000), (2, 80, 3000), (2, 80, 3000)]
        input_names = ["x", "mask", "mu", "cond"]
        return {'min_shape': min_shape, 'opt_shape': opt_shape, 'max_shape': max_shape, 'input_names': input_names}


def main():
    logger.info('请先编译plugin_ln_3d_eps6文件夹')
    args = get_args()
    model_dir = args.token2wav_path
    # 官方fp16的trt结果有问题，会得到一些nan
    fp16=True

    plugin_path = args.trt_plugin_lib_path
    if not os.path.exists(plugin_path):
        plugin_path = None

    token2wav_model = CosyVoice3(
        model_dir, load_trt=True, fp16=fp16, plugin_path=plugin_path
    )


if __name__ == '__main__':
    main()
    print('all finish...')
