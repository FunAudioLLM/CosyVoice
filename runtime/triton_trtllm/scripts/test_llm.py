
# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse

import numpy as np
import torch

import tensorrt_llm
from tensorrt_llm.logger import logger

from tensorrt_llm.runtime import ModelRunnerCpp
from transformers import AutoTokenizer


def parse_arguments(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_text',
        type=str,
        nargs='+',
        default=["Born in north-east France, Soyer trained as a"])
    parser.add_argument('--tokenizer_dir', type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument('--engine_dir', type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument('--log_level', type=str, default="debug")
    parser.add_argument('--kv_cache_free_gpu_memory_fraction', type=float, default=0.6)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top_k', type=int, default=50)
    parser.add_argument('--top_p', type=float, default=0.95)

    return parser.parse_args(args=args)


def parse_input(tokenizer,
                input_text=None,
                prompt_template=None):
    batch_input_ids = []
    for curr_text in input_text:
        if prompt_template is not None:
            curr_text = prompt_template.format(input_text=curr_text)
        input_ids = tokenizer.encode(
            curr_text)
        batch_input_ids.append(input_ids)

    batch_input_ids = [
        torch.tensor(x, dtype=torch.int32) for x in batch_input_ids
    ]

    logger.debug(f"Input token ids (batch_size = {len(batch_input_ids)}):")
    for i, input_ids in enumerate(batch_input_ids):
        logger.debug(f"Request {i}: {input_ids.tolist()}")

    return batch_input_ids


def main(args):
    runtime_rank = tensorrt_llm.mpi_rank()
    logger.set_level(args.log_level)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    prompt_template = "<|sos|>{input_text}<|task_id|>"
    end_id = tokenizer.convert_tokens_to_ids("<|eos1|>")

    batch_input_ids = parse_input(tokenizer=tokenizer,
                                  input_text=args.input_text,
                                  prompt_template=prompt_template)

    input_lengths = [x.size(0) for x in batch_input_ids]

    runner_kwargs = dict(
        engine_dir=args.engine_dir,
        rank=runtime_rank,
        max_output_len=1024,
        enable_context_fmha_fp32_acc=False,
        max_batch_size=len(batch_input_ids),
        max_input_len=max(input_lengths),
        kv_cache_free_gpu_memory_fraction=args.kv_cache_free_gpu_memory_fraction,
        cuda_graph_mode=False,
        gather_generation_logits=False,
    )

    runner = ModelRunnerCpp.from_dir(**runner_kwargs)

    with torch.no_grad():
        outputs = runner.generate(
            batch_input_ids=batch_input_ids,
            max_new_tokens=1024,
            end_id=end_id,
            pad_id=end_id,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            num_return_sequences=1,
            repetition_penalty=1.1,
            random_seed=42,
            streaming=False,
            output_sequence_lengths=True,
            output_generation_logits=False,
            return_dict=True,
            return_all_generated_tokens=False)
        torch.cuda.synchronize()
        output_ids, sequence_lengths = outputs["output_ids"], outputs["sequence_lengths"]
        num_output_sents, num_beams, _ = output_ids.size()
        assert num_beams == 1
        beam = 0
        batch_size = len(input_lengths)
        num_return_sequences = num_output_sents // batch_size
        assert num_return_sequences == 1
        for i in range(batch_size * num_return_sequences):
            batch_idx = i // num_return_sequences
            seq_idx = i % num_return_sequences
            inputs = output_ids[i][0][:input_lengths[batch_idx]].tolist()
            input_text = tokenizer.decode(inputs)
            print(f'Input [Text {batch_idx}]: \"{input_text}\"')
            output_begin = input_lengths[batch_idx]
            output_end = sequence_lengths[i][beam]
            outputs = output_ids[i][beam][output_begin:output_end].tolist()
            output_text = tokenizer.decode(outputs)
            print(f'Output [Text {batch_idx}]: \"{output_text}\"')
            logger.debug(str(outputs))


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
