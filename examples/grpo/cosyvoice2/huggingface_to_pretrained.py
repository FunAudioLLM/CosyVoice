
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
python3 hf2pretrained.py --hf-cosyvoice2-llm-path /workspace/rl-exp/checkpoint-400 --output-path /workspace/CosyVoice2-0.5B/llm-new.pt
"""
from argparse import ArgumentParser
import torch
from safetensors import safe_open
from transformers import AutoTokenizer


def get_args():
    parser = ArgumentParser()

    parser.add_argument(
        "--hf-cosyvoice2-llm-path",
        type=str,
        default=None,
        help="The RL trained CosyVoice2 model path in HuggingFace format",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="./llm.pt",
        help="The path to save the llm.pt",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    tokenizer = AutoTokenizer.from_pretrained(args.hf_cosyvoice2_llm_path)
    speech_start_idx = tokenizer.convert_tokens_to_ids("<|s_0|>")
    cosyvoice2_token_size = 6561 + 3
    llm_embedding_vocab_size = 2

    hf_tensors = {}
    with safe_open(f"{args.hf_cosyvoice2_llm_path}/model.safetensors", framework="pt", device="cpu") as f:
        for k in f.keys():
            if k.startswith("lm_head.bias"):
                # RL trained model disable bias for lm_head
                continue
            new_k = "llm.model." + k
            hf_tensors[new_k] = f.get_tensor(k)
            if k.startswith("lm_head"):
                hf_tensors["llm_decoder.weight"] = f.get_tensor(k)[speech_start_idx:speech_start_idx + cosyvoice2_token_size]
                hf_tensors["llm_decoder.bias"] = torch.zeros_like(hf_tensors["llm_decoder.weight"][:, 0])
            if k.startswith("model.embed_tokens"):
                hf_tensors["speech_embedding.weight"] = f.get_tensor(k)[speech_start_idx:speech_start_idx + cosyvoice2_token_size]
                hf_tensors["llm_embedding.weight"] = f.get_tensor(k)[speech_start_idx + cosyvoice2_token_size:speech_start_idx + cosyvoice2_token_size + llm_embedding_vocab_size]

        # use tie_word_embeddings=True
        hf_tensors["llm.model.model.embed_tokens.weight"] = hf_tensors["llm.model.model.embed_tokens.weight"][:151936]
        hf_tensors["llm.model.lm_head.weight"] = hf_tensors["llm.model.model.embed_tokens.weight"]

    torch.save(hf_tensors, args.output_path)
