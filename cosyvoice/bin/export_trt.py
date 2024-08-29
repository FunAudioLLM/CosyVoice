import argparse
import logging
import os
import sys

logging.getLogger('matplotlib').setLevel(logging.WARNING)

try:
    import tensorrt
except ImportError:
    error_msg_zh = [
        "step.1 下载 tensorrt .tar.gz 压缩包并解压，下载地址: https://developer.nvidia.com/tensorrt/download/10x",
        "step.2 使用 tensorrt whl 包进行安装根据 python 版本对应进行安装，如 pip install ${TensorRT-Path}/python/tensorrt-10.2.0-cp38-none-linux_x86_64.whl",
        "step.3 将 tensorrt 的 lib 路径添加进环境变量中，export LD_LIBRARY_PATH=${TensorRT-Path}/lib/"
    ]
    print("\n".join(error_msg_zh))
    sys.exit(1)

import torch
from cosyvoice.cli.cosyvoice import CosyVoice

def get_args():
    parser = argparse.ArgumentParser(description='Export your model for deployment')
    parser.add_argument('--model_dir',
                        type=str,
                        default='pretrained_models/CosyVoice-300M',
                        help='Local path to the model directory')

    parser.add_argument('--export_half',
                        action='store_true',
                        help='Export with half precision (FP16)')
    
    args = parser.parse_args()
    print(args)
    return args

def main():
    args = get_args()

    cosyvoice = CosyVoice(args.model_dir, load_jit=False, load_trt=False)
    estimator = cosyvoice.model.flow.decoder.estimator

    dtype = torch.float32 if not args.export_half else torch.float16
    device = torch.device("cuda")
    batch_size = 1
    seq_len = 256
    hidden_size = cosyvoice.model.flow.output_size
    x = torch.rand((batch_size, hidden_size, seq_len), dtype=dtype, device=device)
    mask = torch.ones((batch_size, 1, seq_len), dtype=dtype, device=device)
    mu = torch.rand((batch_size, hidden_size, seq_len), dtype=dtype, device=device)
    t = torch.rand((batch_size, ), dtype=dtype, device=device)
    spks = torch.rand((batch_size, hidden_size), dtype=dtype, device=device)
    cond = torch.rand((batch_size, hidden_size, seq_len), dtype=dtype, device=device)

    onnx_file_name = 'estimator_fp32.onnx' if not args.export_half else 'estimator_fp16.onnx'
    onnx_file_path = os.path.join(args.model_dir, onnx_file_name)
    dummy_input = (x, mask, mu, t, spks, cond)

    estimator = estimator.to(dtype)

    torch.onnx.export(
        estimator,
        dummy_input,
        onnx_file_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['x', 'mask', 'mu', 't', 'spks', 'cond'],
        output_names=['output'],
        dynamic_axes={
            'x': {2: 'seq_len'},
            'mask': {2: 'seq_len'},
            'mu': {2: 'seq_len'},
            'cond': {2: 'seq_len'},
            'output': {2: 'seq_len'},
        }
    )

    tensorrt_path = os.environ.get('tensorrt_root_dir')
    if not tensorrt_path:
        raise EnvironmentError("Please set the 'tensorrt_root_dir' environment variable.")

    if not os.path.isdir(tensorrt_path):
        raise FileNotFoundError(f"The directory {tensorrt_path} does not exist.")

    trt_lib_path = os.path.join(tensorrt_path, "lib")
    if trt_lib_path not in os.environ.get('LD_LIBRARY_PATH', ''):
        print(f"Adding TensorRT lib path {trt_lib_path} to LD_LIBRARY_PATH.")
        os.environ['LD_LIBRARY_PATH'] = f"{os.environ.get('LD_LIBRARY_PATH', '')}:{trt_lib_path}"

    trt_file_name = 'estimator_fp32.plan' if not args.export_half else 'estimator_fp16.plan'
    trt_file_path = os.path.join(args.model_dir, trt_file_name)

    trtexec_cmd = f"{tensorrt_path}/bin/trtexec --onnx={onnx_file_path} --saveEngine={trt_file_path} " \
                  "--minShapes=x:1x80x1,mask:1x1x1,mu:1x80x1,t:1,spks:1x80,cond:1x80x1 " \
                  "--maxShapes=x:1x80x4096,mask:1x1x4096,mu:1x80x4096,t:1,spks:1x80,cond:1x80x4096 --verbose " + \
                  ("--fp16" if args.export_half else "")
# /ossfs/workspace/TensorRT-10.2.0.19/bin/trtexec --onnx=estimator_fp32.onnx --saveEngine=estimator_fp32.plan --minShapes=x:1x80x1,mask:1x1x1,mu:1x80x1,t:1,spks:1x80,cond:1x80x1 --maxShapes=x:1x80x4096,mask:1x1x4096,mu:1x80x4096,t:1,spks:1x80,cond:1x80x4096 --verbose
    print("execute ", trtexec_cmd)

    os.system(trtexec_cmd)

    print("x.shape", x.shape)
    print("mask.shape", mask.shape)
    print("mu.shape", mu.shape)
    print("t.shape", t.shape)
    print("spks.shape", spks.shape)
    print("cond.shape", cond.shape)

if __name__ == "__main__":
    main()
