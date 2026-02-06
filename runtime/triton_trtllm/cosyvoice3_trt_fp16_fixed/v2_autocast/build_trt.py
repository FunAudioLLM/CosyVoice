import sys
sys.path.insert(0, '/opt/code/CosyVoice')
sys.path.insert(0, '/opt/code/CosyVoice/third_party/Matcha-TTS')

import json
import os
import logging
import torch
from hyperpyyaml import load_hyperpyyaml
from cosyvoice.utils.common import TrtContextWrapper
from collections import defaultdict
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def convert_onnx_to_trt(trt_model, trt_kwargs, onnx_model, fp16, autocast_mode=False):
    import tensorrt as trt
    logging.info("Converting onnx to trt...")
    if autocast_mode:
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    else:
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)
    config = builder.create_builder_config()
    # for see network detailed
    # config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 32)  # 4GB
    if not autocast_mode:
        if fp16:
            config.set_flag(trt.BuilderFlag.FP16)
    profile = builder.create_optimization_profile()
    # load onnx model
    with open(onnx_model, "rb") as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            raise ValueError('failed to parse {}'.format(onnx_model))
    # set input shapes
    for i in range(len(trt_kwargs['input_names'])):
        profile.set_shape(trt_kwargs['input_names'][i], trt_kwargs['min_shape'][i], trt_kwargs['opt_shape'][i], trt_kwargs['max_shape'][i])
    tensor_dtype = trt.DataType.HALF if fp16 else trt.DataType.FLOAT
    # set input and output data type
    for i in range(network.num_inputs):
        input_tensor = network.get_input(i)
        input_tensor.dtype = tensor_dtype
    for i in range(network.num_outputs):
        output_tensor = network.get_output(i)
        output_tensor.dtype = tensor_dtype
    config.add_optimization_profile(profile)
    engine_bytes = builder.build_serialized_network(network, config)
    # save trt engine
    with open(trt_model, "wb") as f:
        f.write(engine_bytes)
    logging.info("Succesfully convert onnx to trt...")


class CosyVoice3:

    def __init__(self, model_dir, load_trt=False, fp16=False, trt_concurrent=1, autocast_mode=False):
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
            if autocast_mode:
                onnx_path = '{}/flow.decoder.estimator.autocast_fp16.onnx'.format(model_dir)
                trt_path = '{}/flow.decoder.estimator.autocast_fp16.mygpu.plan'.format(model_dir)
            else:
                onnx_path = '{}/flow.decoder.estimator.fp32.onnx'.format(model_dir)
                trt_path = '{}/flow.decoder.estimator.{}.mygpu.plan'.format(model_dir, 'fp16' if self.fp16 is True else 'fp32')
            logging.info('onnx_path: {}'.format(onnx_path))
            logging.info('trt_path: {}'.format(trt_path))
            self.model.load_trt(trt_path,
                                onnx_path,
                                trt_concurrent,
                                self.fp16,
                                autocast_mode)


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

    def load_trt(self, flow_decoder_estimator_model, flow_decoder_onnx_model, trt_concurrent, fp16, autocast_mode=False):
        assert torch.cuda.is_available(), 'tensorrt only supports gpu!'
        if not os.path.exists(flow_decoder_estimator_model) or os.path.getsize(flow_decoder_estimator_model) == 0:
            convert_onnx_to_trt(flow_decoder_estimator_model, self.get_trt_kwargs(), flow_decoder_onnx_model, fp16, autocast_mode)
        del self.flow.decoder.estimator
        import tensorrt as trt
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
    model_dir = '/weights/Fun-CosyVoice3-0.5B-2512'
    # 官方fp16的trt结果有问题，会得到一些nan
    fp16=True
    autocast_mode = True
    token2wav_model = CosyVoice3(
        model_dir, load_trt=True, fp16=fp16, autocast_mode=autocast_mode
    )


if __name__ == '__main__':
    main()
    print('all finish...')
