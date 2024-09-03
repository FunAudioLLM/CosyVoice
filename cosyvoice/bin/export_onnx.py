# Copyright (c) 2024 Antgroup Inc (authors: Zhoubofan, hexisyztem@icloud.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import os
import sys

logging.getLogger('matplotlib').setLevel(logging.WARNING)
import onnxruntime as ort
import numpy as np

# try:
#     import tensorrt
#     import tensorrt as trt
# except ImportError:
#     error_msg_zh = [
#         "step.1 下载 tensorrt .tar.gz 压缩包并解压，下载地址: https://developer.nvidia.com/tensorrt/download/10x",
#         "step.2 使用 tensorrt whl 包进行安装根据 python 版本对应进行安装，如 pip install ${TensorRT-Path}/python/tensorrt-10.2.0-cp38-none-linux_x86_64.whl",
#         "step.3 将 tensorrt 的 lib 路径添加进环境变量中，export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${TensorRT-Path}/lib/"
#     ]
#     print("\n".join(error_msg_zh))
#     sys.exit(1)

import torch
from cosyvoice.cli.cosyvoice import CosyVoice


def calculate_onnx(onnx_file, x, mask, mu, t, spks, cond):
    providers = ['CUDAExecutionProvider']
    sess_options = ort.SessionOptions()

    providers = [
        'CUDAExecutionProvider'
    ]

    # Load the ONNX model
    session = ort.InferenceSession(onnx_file, sess_options=sess_options, providers=providers)
    
    x_np = x.cpu().numpy()
    mask_np = mask.cpu().numpy()
    mu_np = mu.cpu().numpy()
    t_np = np.array(t.cpu()) 
    spks_np = spks.cpu().numpy()
    cond_np = cond.cpu().numpy()

    ort_inputs = {
        'x': x_np,
        'mask': mask_np,
        'mu': mu_np,
        't': t_np,
        'spks': spks_np,
        'cond': cond_np
    }

    output = session.run(None, ort_inputs)

    return output[0]

# def calculate_tensorrt(trt_file, x, mask, mu, t, spks, cond):
#     trt.init_libnvinfer_plugins(None, "")
#     logger = trt.Logger(trt.Logger.WARNING)
#     runtime = trt.Runtime(logger)
#     with open(trt_file, 'rb') as f:
#         serialized_engine = f.read()
#     engine = runtime.deserialize_cuda_engine(serialized_engine)
#     context = engine.create_execution_context()

#     bs = x.shape[0]
#     hs = x.shape[1]
#     seq_len = x.shape[2]

#     ret = torch.zeros_like(x)

#     # Set input shapes for dynamic dimensions
#     context.set_input_shape("x", x.shape)
#     context.set_input_shape("mask", mask.shape)
#     context.set_input_shape("mu", mu.shape)
#     context.set_input_shape("t", t.shape)
#     context.set_input_shape("spks", spks.shape)
#     context.set_input_shape("cond", cond.shape)

#     # bindings = [x.data_ptr(), mask.data_ptr(), mu.data_ptr(), t.data_ptr(), spks.data_ptr(), cond.data_ptr(), ret.data_ptr()]
#     # names = ['x', 'mask', 'mu', 't', 'spks', 'cond', 'estimator_out']
#     #
#     # for i in range(len(bindings)):
#     #     context.set_tensor_address(names[i], bindings[i])
#     #
#     # handle = torch.cuda.current_stream().cuda_stream
#     # context.execute_async_v3(stream_handle=handle)

#     # Create a list of bindings
#     bindings = [int(x.data_ptr()), int(mask.data_ptr()), int(mu.data_ptr()), int(t.data_ptr()), int(spks.data_ptr()), int(cond.data_ptr()), int(ret.data_ptr())]

#     # Execute the inference
#     context.execute_v2(bindings=bindings)

#     torch.cuda.synchronize()

#     return ret


# def test_calculate_value(estimator, onnx_file, trt_file, dummy_input, args):
#     torch_output = estimator.forward(**dummy_input).cpu().detach().numpy()
#     onnx_output = calculate_onnx(onnx_file, **dummy_input)
#     tensorrt_output = calculate_tensorrt(trt_file, **dummy_input).cpu().detach().numpy()
#     atol = 2e-3  # Absolute tolerance
#     rtol = 1e-4  # Relative tolerance

#     print(f"args.export_half: {args.export_half}, args.model_dir: {args.model_dir}")
#     print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

#     print("torch_output diff with onnx_output: ", )
#     print(f"compare with atol: {atol}, rtol: {rtol} ", np.allclose(torch_output, onnx_output, atol, rtol))
#     print(f"max diff value: ", np.max(np.fabs(torch_output - onnx_output)))
#     print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

#     print("torch_output diff with tensorrt_output: ")
#     print(f"compare with atol: {atol}, rtol: {rtol} ", np.allclose(torch_output, tensorrt_output, atol, rtol))
#     print(f"max diff value: ", np.max(np.fabs(torch_output - tensorrt_output)))
#     print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

#     print("onnx_output diff with tensorrt_output: ")
#     print(f"compare with atol: {atol}, rtol: {rtol} ", np.allclose(onnx_output, tensorrt_output, atol, rtol))
#     print(f"max diff value: ", np.max(np.fabs(onnx_output - tensorrt_output)))
#     print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")


def get_args():
    parser = argparse.ArgumentParser(description='Export your model for deployment')
    parser.add_argument('--model_dir', type=str, default='pretrained_models/CosyVoice-300M', help='Local path to the model directory')
    parser.add_argument('--export_half', type=str, choices=['True', 'False'], default='False', help='Export with half precision (FP16)')
    # parser.add_argument('--trt_max_len', type=int, default=8192, help='Export max len')
    parser.add_argument('--exec_export', type=str, choices=['True', 'False'], default='True', help='Exec export')
    
    args = parser.parse_args()
    args.export_half = args.export_half == 'True'
    args.exec_export = args.exec_export == 'True'
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
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
    out_channels = cosyvoice.model.flow.decoder.estimator.out_channels
    x = torch.rand((batch_size, out_channels, seq_len), dtype=dtype, device=device)
    mask = torch.ones((batch_size, 1, seq_len), dtype=dtype, device=device)
    mu = torch.rand((batch_size, out_channels, seq_len), dtype=dtype, device=device)
    t = torch.rand((batch_size, ), dtype=dtype, device=device)
    spks = torch.rand((batch_size, out_channels), dtype=dtype, device=device)
    cond = torch.rand((batch_size, out_channels, seq_len), dtype=dtype, device=device)

    onnx_file_name = 'estimator_fp32.onnx' if not args.export_half else 'estimator_fp16.onnx'
    onnx_file_path = os.path.join(args.model_dir, onnx_file_name)
    dummy_input = (x, mask, mu, t, spks, cond)

    estimator = estimator.to(dtype)

    if args.exec_export:
        torch.onnx.export(
            estimator,
            dummy_input,
            onnx_file_path,
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=['x', 'mask', 'mu', 't', 'spks', 'cond'],
            output_names=['estimator_out'],
            dynamic_axes={
                'x': {2: 'seq_len'},
                'mask': {2: 'seq_len'},
                'mu': {2: 'seq_len'},
                'cond': {2: 'seq_len'},
                'estimator_out': {2: 'seq_len'},
            }
        )

    # tensorrt_path = os.environ.get('tensorrt_root_dir')
    # if not tensorrt_path:
    #     raise EnvironmentError("Please set the 'tensorrt_root_dir' environment variable.")

    # if not os.path.isdir(tensorrt_path):
    #     raise FileNotFoundError(f"The directory {tensorrt_path} does not exist.")

    # trt_lib_path = os.path.join(tensorrt_path, "lib")
    # if trt_lib_path not in os.environ.get('LD_LIBRARY_PATH', ''):
    #     print(f"Adding TensorRT lib path {trt_lib_path} to LD_LIBRARY_PATH.")
    #     os.environ['LD_LIBRARY_PATH'] = f"{os.environ.get('LD_LIBRARY_PATH', '')}:{trt_lib_path}"

    # trt_file_name = 'estimator_fp32.plan' if not args.export_half else 'estimator_fp16.plan'
    # trt_file_path = os.path.join(args.model_dir, trt_file_name)

    # trtexec_bin = os.path.join(tensorrt_path, 'bin/trtexec')
    # trt_max_len = args.trt_max_len
    # trtexec_cmd = f"{trtexec_bin} --onnx={onnx_file_path} --saveEngine={trt_file_path} " \
    #               f"--minShapes=x:1x{out_channels}x1,mask:1x1x1,mu:1x{out_channels}x1,t:1,spks:1x{out_channels},cond:1x{out_channels}x1 " \
    #               f"--maxShapes=x:1x{out_channels}x{trt_max_len},mask:1x1x{trt_max_len},mu:1x{out_channels}x{trt_max_len},t:1,spks:1x{out_channels},cond:1x{out_channels}x{trt_max_len} " + \
    #               ("--fp16" if args.export_half else "")
    
    # print("execute ", trtexec_cmd)

    # if args.exec_export:
    #     os.system(trtexec_cmd)

    # dummy_input = {'x': x, 'mask': mask, 'mu': mu, 't': t, 'spks': spks, 'cond': cond}
    # test_calculate_value(estimator, onnx_file_path, trt_file_path, dummy_input, args)

if __name__ == "__main__":
    main()
