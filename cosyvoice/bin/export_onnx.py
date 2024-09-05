# Copyright (c) 2024 Antgroup Inc (authors: Zhoubofan, hexisyztem@icloud.com)
# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
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

from __future__ import print_function

import argparse
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
import os
import sys
import onnxruntime
import random
import torch
from tqdm import tqdm
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/../..'.format(ROOT_DIR))
sys.path.append('{}/../../third_party/Matcha-TTS'.format(ROOT_DIR))
from cosyvoice.cli.cosyvoice import CosyVoice


def get_dummy_input(batch_size, seq_len, out_channels, device):
    x = torch.rand((batch_size, out_channels, seq_len), dtype=torch.float32, device=device)
    mask = torch.ones((batch_size, 1, seq_len), dtype=torch.float32, device=device)
    mu = torch.rand((batch_size, out_channels, seq_len), dtype=torch.float32, device=device)
    t = torch.rand((batch_size), dtype=torch.float32, device=device)
    spks = torch.rand((batch_size, out_channels), dtype=torch.float32, device=device)
    cond = torch.rand((batch_size, out_channels, seq_len), dtype=torch.float32, device=device)
    return x, mask, mu, t, spks, cond


def get_args():
    parser = argparse.ArgumentParser(description='export your model for deployment')
    parser.add_argument('--model_dir',
                        type=str,
                        default='pretrained_models/CosyVoice-300M',
                        help='local path')
    args = parser.parse_args()
    print(args)
    return args


def main():
    args = get_args()
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')

    cosyvoice = CosyVoice(args.model_dir, load_jit=False, load_onnx=False)

    # 1. export flow decoder estimator
    estimator = cosyvoice.model.flow.decoder.estimator

    device = cosyvoice.model.device
    batch_size, seq_len = 1, 256
    out_channels = cosyvoice.model.flow.decoder.estimator.out_channels
    x, mask, mu, t, spks, cond = get_dummy_input(batch_size, seq_len, out_channels, device)
    torch.onnx.export(
        estimator,
        (x, mask, mu, t, spks, cond),
        '{}/flow.decoder.estimator.fp32.onnx'.format(args.model_dir),
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['x', 'mask', 'mu', 't', 'spks', 'cond'],
        output_names=['estimator_out'],
        dynamic_axes={
            'x': {0: 'batch_size', 2: 'seq_len'},
            'mask': {0: 'batch_size', 2: 'seq_len'},
            'mu': {0: 'batch_size', 2: 'seq_len'},
            'cond': {0: 'batch_size', 2: 'seq_len'},
            't': {0: 'batch_size'},
            'spks': {0: 'batch_size'},
            'estimator_out': {0: 'batch_size', 2: 'seq_len'},
        }
    )

    # 2. test computation consistency
    option = onnxruntime.SessionOptions()
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    option.intra_op_num_threads = 1
    providers = ['CUDAExecutionProvider' if torch.cuda.is_available() else 'CPUExecutionProvider']
    estimator_onnx = onnxruntime.InferenceSession('{}/flow.decoder.estimator.fp32.onnx'.format(args.model_dir),
                                                  sess_options=option, providers=providers)

    for _ in tqdm(range(10)):
        x, mask, mu, t, spks, cond = get_dummy_input(random.randint(1, 6), random.randint(16, 512), out_channels, device)
        output_pytorch = estimator(x, mask, mu, t, spks, cond)
        ort_inputs = {
            'x': x.cpu().numpy(),
            'mask': mask.cpu().numpy(),
            'mu': mu.cpu().numpy(),
            't': t.cpu().numpy(),
            'spks': spks.cpu().numpy(),
            'cond': cond.cpu().numpy()
        }
        output_onnx = estimator_onnx.run(None, ort_inputs)[0]
        torch.testing.assert_allclose(output_pytorch, torch.from_numpy(output_onnx).to(device), rtol=1e-2, atol=1e-4)


if __name__ == "__main__":
    main()
