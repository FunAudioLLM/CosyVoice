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
from cosyvoice.cli.cosyvoice import AutoModel
from cosyvoice.utils.file_utils import logging


def get_args():
    parser = argparse.ArgumentParser(description='export your model for deployment')
    parser.add_argument('--model_dir',
                        type=str,
                        default='pretrained_models/CosyVoice-300M',
                        help='local path')
    args = parser.parse_args()
    print(args)
    return args


def get_optimized_script(model, preserved_attrs=[]):
    script = torch.jit.script(model)
    if preserved_attrs != []:
        script = torch.jit.freeze(script, preserved_attrs=preserved_attrs)
    else:
        script = torch.jit.freeze(script)
    script = torch.jit.optimize_for_inference(script)
    return script


def main():
    args = get_args()
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')

    torch._C._jit_set_fusion_strategy([('STATIC', 1)])
    torch._C._jit_set_profiling_mode(False)
    torch._C._jit_set_profiling_executor(False)

    model = AutoModel(model_dir=args.model_dir)

    if model.__class__.__name__ == 'CosyVoice':
        # 1. export llm text_encoder
        llm_text_encoder = model.model.llm.text_encoder
        script = get_optimized_script(llm_text_encoder)
        script.save('{}/llm.text_encoder.fp32.zip'.format(args.model_dir))
        script = get_optimized_script(llm_text_encoder.half())
        script.save('{}/llm.text_encoder.fp16.zip'.format(args.model_dir))
        logging.info('successfully export llm_text_encoder')

        # 2. export llm llm
        llm_llm = model.model.llm.llm
        script = get_optimized_script(llm_llm, ['forward_chunk'])
        script.save('{}/llm.llm.fp32.zip'.format(args.model_dir))
        script = get_optimized_script(llm_llm.half(), ['forward_chunk'])
        script.save('{}/llm.llm.fp16.zip'.format(args.model_dir))
        logging.info('successfully export llm_llm')

        # 3. export flow encoder
        flow_encoder = model.model.flow.encoder
        script = get_optimized_script(flow_encoder)
        script.save('{}/flow.encoder.fp32.zip'.format(args.model_dir))
        script = get_optimized_script(flow_encoder.half())
        script.save('{}/flow.encoder.fp16.zip'.format(args.model_dir))
        logging.info('successfully export flow_encoder')
    elif model.__class__.__name__ == 'CosyVoice2':
        # 1. export flow encoder
        flow_encoder = model.model.flow.encoder
        script = get_optimized_script(flow_encoder)
        script.save('{}/flow.encoder.fp32.zip'.format(args.model_dir))
        script = get_optimized_script(flow_encoder.half())
        script.save('{}/flow.encoder.fp16.zip'.format(args.model_dir))
        logging.info('successfully export flow_encoder')
    else:
        raise ValueError('unsupported model type')


if __name__ == '__main__':
    main()
