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
Usage: Instruct TTS
  python3 infer.py \
    --token2wav-path /workspace/CosyVoice2-0.5B \
    --prompt-text "吃燕窝就选燕之屋，本节目由26年专注高品质燕窝的燕之屋冠名播出。豆奶牛奶换着喝，营养更均衡，本节目由豆本豆豆奶特约播出。" \
    --prompt-speech-path ./assets/prompt_audio.wav \
    --model-path ./transformers_cosyvoice2_llm \
    --input-text "用四川话说<|endofprompt|>扁担长，板凳宽，扁担绑在板凳上。吃葡萄不吐葡萄皮，不吃葡萄倒吐葡萄皮。"
"""
from cosyvoice.cli.cosyvoice import CosyVoice3
import sys
from argparse import ArgumentParser
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

sys.path.append("/workspace/CosyVoice/third_party/Matcha-TTS")


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--pretrained-cosyvoice3-path", 
        type=str, 
        default="/workspace/CosyVoice2-0.5B",
        help="Token2Wav path, default to %(default)r"
    )
    parser.add_argument(
        "--save-path", 
        type=str, 
        default='./transformers_cosyvoice3_llm', 
        help="The path to save the model"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    cosy3_model = CosyVoice3(args.pretrained_cosyvoice3_path)

    # text tokenizer
    tokenizer = AutoTokenizer.from_pretrained(f"{args.pretrained_cosyvoice3_path}/CosyVoice-BlankEN")
    
    llm = cosy3_model.model.llm.llm.model
    
    # speech token embedding (with sos/eos, etc), removing llm embedding
    speech_embedding = cosy3_model.model.llm.speech_embedding
    llm_decoder = cosy3_model.model.llm.llm_decoder
    
    special_tokens = {
            'eos_token': '<|endoftext|>',
            'pad_token': '<|endoftext|>',
            'additional_special_tokens': [
                '<|im_start|>', '<|im_end|>', '<|endofprompt|>',
                '[breath]', '<strong>', '</strong>', '[noise]', '[laughter]', '[cough]', '[clucking]', '[accent]', '[quick_breath]', 
                "<laughter>", "</laughter>", "[hissing]", "[sigh]", "[vocalized-noise]", "[lipsmack]", "[mn]", "<|endofsystem|>",
                "[AA]", "[AA0]", "[AA1]", "[AA2]", "[AE]", "[AE0]", "[AE1]", "[AE2]", "[AH]", "[AH0]", "[AH1]", "[AH2]",
                "[AO]", "[AO0]", "[AO1]", "[AO2]", "[AW]", "[AW0]", "[AW1]", "[AW2]", "[AY]", "[AY0]", "[AY1]", "[AY2]",
                "[B]", "[CH]", "[D]", "[DH]", "[EH]", "[EH0]", "[EH1]", "[EH2]", "[ER]", "[ER0]", "[ER1]", "[ER2]", "[EY]",
                "[EY0]", "[EY1]", "[EY2]", "[F]", "[G]", "[HH]", "[IH]", "[IH0]", "[IH1]", "[IH2]", "[IY]", "[IY0]", "[IY1]",
                "[IY2]", "[JH]", "[K]", "[L]", "[M]", "[N]", "[NG]", "[OW]", "[OW0]", "[OW1]", "[OW2]", "[OY]", "[OY0]",
                "[OY1]", "[OY2]", "[P]", "[R]", "[S]", "[SH]", "[T]", "[TH]", "[UH]", "[UH0]", "[UH1]", "[UH2]", "[UW]",
                "[UW0]", "[UW1]", "[UW2]", "[V]", "[W]", "[Y]", "[Z]", "[ZH]",
                "[a]", "[ai]", "[an]", "[ang]", "[ao]", "[b]", "[c]", "[ch]", "[d]", "[e]", "[ei]", "[en]", "[eng]", "[f]",
                "[g]", "[h]", "[i]", "[ian]", "[in]", "[ing]", "[iu]", "[ià]", "[iàn]", "[iàng]", "[iào]", "[iá]", "[ián]",
                "[iáng]", "[iáo]", "[iè]", "[ié]", "[iòng]", "[ióng]", "[iù]", "[iú]", "[iā]", "[iān]", "[iāng]", "[iāo]",
                "[iē]", "[iě]", "[iōng]", "[iū]", "[iǎ]", "[iǎn]", "[iǎng]", "[iǎo]", "[iǒng]", "[iǔ]", "[j]", "[k]", "[l]",
                "[m]", "[n]", "[o]", "[ong]", "[ou]", "[p]", "[q]", "[r]", "[s]", "[sh]", "[t]", "[u]", "[uang]", "[ue]",
                "[un]", "[uo]", "[uà]", "[uài]", "[uàn]", "[uàng]", "[uá]", "[uái]", "[uán]", "[uáng]", "[uè]", "[ué]", "[uì]",
                "[uí]", "[uò]", "[uó]", "[uā]", "[uāi]", "[uān]", "[uāng]", "[uē]", "[uě]", "[uī]", "[uō]", "[uǎ]", "[uǎi]",
                "[uǎn]", "[uǎng]", "[uǐ]", "[uǒ]", "[vè]", "[w]", "[x]", "[y]", "[z]", "[zh]", "[à]", "[ài]", "[àn]", "[àng]",
                "[ào]", "[á]", "[ái]", "[án]", "[áng]", "[áo]", "[è]", "[èi]", "[èn]", "[èng]", "[èr]", "[é]", "[éi]", "[én]",
                "[éng]", "[ér]", "[ì]", "[ìn]", "[ìng]", "[í]", "[ín]", "[íng]", "[ò]", "[òng]", "[òu]", "[ó]", "[óng]", "[óu]",
                "[ù]", "[ùn]", "[ú]", "[ún]", "[ā]", "[āi]", "[ān]", "[āng]", "[āo]", "[ē]", "[ēi]", "[ēn]", "[ēng]", "[ě]",
                "[ěi]", "[ěn]", "[ěng]", "[ěr]", "[ī]", "[īn]", "[īng]", "[ō]", "[ōng]", "[ōu]", "[ū]", "[ūn]", "[ǎ]", "[ǎi]",
                "[ǎn]", "[ǎng]", "[ǎo]", "[ǐ]", "[ǐn]", "[ǐng]", "[ǒ]", "[ǒng]", "[ǒu]", "[ǔ]", "[ǔn]", "[ǘ]", "[ǚ]", "[ǜ]"
            ]
        }
    tokenizer.add_special_tokens(special_tokens)

    original_tokenizer_vocab_size = len(tokenizer)
    cosyvoice3_token_size = 6561
    total_speech_tokens = cosyvoice3_token_size + 200

    new_tokens = [
        f"<|s_{i}|>" for i in range(total_speech_tokens)
    ] + [
        "<|sos|>", "<|eos|>", "<|task_id|>"
    ]
    num_added_tokens = tokenizer.add_tokens(new_tokens)

    speech_start_idx = tokenizer.convert_tokens_to_ids("<|s_0|>")

    speech_end_idx = tokenizer.convert_tokens_to_ids(
        f"<|s_{total_speech_tokens - 1}|>"
    ) + 1
    
    assert speech_start_idx != tokenizer.unk_token_id, "missing <|s_0|> in tokenizer"
    assert (speech_end_idx - speech_start_idx) == total_speech_tokens, (
        f"speech token span mismatch: got {speech_end_idx - speech_start_idx}, "
        f"expected {total_speech_tokens}"
    )

    llm.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=128)
    vocab_size = llm.get_input_embeddings().weight.shape[0]

    feature_size = speech_embedding.embedding_dim
    print(f'feature_size: {feature_size}, vocab_size: {vocab_size}')

    new_lm_head = torch.nn.Linear(
        in_features=feature_size, 
        out_features=vocab_size, 
        bias=False
    )

    control_source_tokens = {
        "sos": f"<|s_{cosyvoice3_token_size + 0}|>",
        "eos": f"<|s_{cosyvoice3_token_size + 1}|>",
        "task_id": f"<|s_{cosyvoice3_token_size + 2}|>",
    }
    alias_source_map = {
        "<|sos|>": control_source_tokens["sos"],
        "<|eos|>": control_source_tokens["eos"],
        "<|task_id|>": control_source_tokens["task_id"],
    }

    # output lm head
    with torch.no_grad():
        # set the weight and bias of the new lm_head to 0
        new_lm_head.weight.data.zero_()

        target_slice = slice(speech_start_idx, speech_end_idx)
        
        assert llm_decoder.weight.shape[0] == (target_slice.stop - target_slice.start), \
            f"dim mistach: llm_decoder {llm_decoder.weight.shape[0]} vs 目标切片 {target_slice.stop - target_slice.start}"

        new_lm_head.weight[target_slice] = llm_decoder.weight

        for alias_token, source_token in alias_source_map.items():
            alias_id = tokenizer.convert_tokens_to_ids(alias_token)
            source_id = tokenizer.convert_tokens_to_ids(source_token)
            assert alias_id != tokenizer.unk_token_id, f"missing alias token: {alias_token}"
            assert source_id != tokenizer.unk_token_id, f"missing source token: {source_token}"
            new_lm_head.weight[alias_id] = new_lm_head.weight[source_id]

    llm.lm_head = new_lm_head

    input_embeddings = llm.get_input_embeddings()

    with torch.no_grad():
        input_embeddings.weight[target_slice] = speech_embedding.weight

        for alias_token, source_token in alias_source_map.items():
            alias_id = tokenizer.convert_tokens_to_ids(alias_token)
            source_id = tokenizer.convert_tokens_to_ids(source_token)
            input_embeddings.weight[alias_id] = input_embeddings.weight[source_id]

    alias_eos_token_id = tokenizer.convert_tokens_to_ids("<|eos|>")
    real_eos_token_id = tokenizer.convert_tokens_to_ids(control_source_tokens["eos"])
    
    llm.generation_config.eos_token_id = [alias_eos_token_id, real_eos_token_id]
    llm.generation_config.pad_token_id = tokenizer.pad_token_id
    llm.generation_config.temperature = 1.0
    llm.generation_config.top_p = 0.8
    llm.generation_config.top_k = 25

    llm.config.eos_token_id = real_eos_token_id
    llm.config.vocab_size = vocab_size
    llm.config.tie_word_embeddings = False
    llm.config.use_bias = False
    llm.to(torch.bfloat16)
    llm.save_pretrained(args.save_path)

    TEMPLATE = (
        "{%- for message in messages %}"
        "{%- if message['role'] == 'user' %}"
        "{{- '<|sos|>' + message['content'] + '<|task_id|>' }}"
        "{%- elif message['role'] == 'assistant' %}"
        "{{- message['content']}}"
        "{%- endif %}"
        "{%- endfor %}"
    )

    tokenizer.chat_template = TEMPLATE
    tokenizer.save_pretrained(args.save_path)