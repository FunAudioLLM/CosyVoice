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
import torch
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/../..'.format(ROOT_DIR))
sys.path.append('{}/../../third_party/Matcha-TTS'.format(ROOT_DIR))
from cosyvoice.cli.cosyvoice import CosyVoice


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

    torch._C._jit_set_fusion_strategy([('STATIC', 1)])
    torch._C._jit_set_profiling_mode(False)
    torch._C._jit_set_profiling_executor(False)

    cosyvoice = CosyVoice(args.model_dir, load_jit=False, load_onnx=False)

    # 1. export llm text_encoder
    llm_text_encoder = cosyvoice.model.llm.text_encoder.half()
    script = torch.jit.script(llm_text_encoder)
    script = torch.jit.freeze(script)
    script = torch.jit.optimize_for_inference(script)
    script.save('{}/llm.text_encoder.fp16.zip'.format(args.model_dir))

    # 2. export llm llm
    llm_llm = cosyvoice.model.llm.llm.half()
    script = torch.jit.script(llm_llm)
    script = torch.jit.freeze(script, preserved_attrs=['forward_chunk'])
    script = torch.jit.optimize_for_inference(script)
    script.save('{}/llm.llm.fp16.zip'.format(args.model_dir))

    # 3. export flow encoder
    flow_encoder = cosyvoice.model.flow.encoder
    script = torch.jit.script(flow_encoder)
    script = torch.jit.freeze(script)
    script = torch.jit.optimize_for_inference(script)
    script.save('{}/flow.encoder.fp32.zip'.format(args.model_dir))


if __name__ == '__main__':
    main()
