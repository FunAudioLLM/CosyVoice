#!/usr/bin/env python3
# Copyright 2025 CosyVoice3 TRT-LLM Integration
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
Convert CosyVoice3 LLM to HuggingFace format with merged embeddings.

This script:
1. Loads CosyVoice3 model
2. Extends tokenizer vocab with speech tokens
3. Merges speech_embedding into embed_tokens of Qwen2
4. Replaces lm_head with llm_decoder using extended vocab
5. Saves model in HuggingFace format for TRT-LLM conversion

Usage:
    python scripts/convert_cosyvoice3_to_hf.py \
        --model-dir pretrained_models/Fun-CosyVoice3-0.5B \
        --output-dir pretrained_models/Fun-CosyVoice3-0.5B/hf_merged

Then convert to TRT-LLM:
    trtllm-build --checkpoint_dir <output_dir> --output_dir <trt_engines_dir> ...
"""
import argparse
import os
import sys
import logging

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'third_party/Matcha-TTS'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Convert CosyVoice3 to HuggingFace format with merged embeddings")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="pretrained_models/Fun-CosyVoice3-0.5B",
        help="Path to CosyVoice3 model directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for HuggingFace model (default: <model-dir>/hf_merged)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Output dtype for the model",
    )
    return parser.parse_args()


def load_cosyvoice3_model(model_dir: str):
    """Load CosyVoice3 model for weight extraction."""
    from hyperpyyaml import load_hyperpyyaml
    from cosyvoice.utils.class_utils import get_model_type

    hyper_yaml_path = os.path.join(model_dir, 'cosyvoice3.yaml')
    hf_llm_dir = os.path.join(model_dir, 'CosyVoice-BlankEN')

    if not os.path.exists(hyper_yaml_path):
        raise ValueError(f'{hyper_yaml_path} not found!')

    with open(hyper_yaml_path, 'r') as f:
        configs = load_hyperpyyaml(
            f,
            overrides={'qwen_pretrain_path': hf_llm_dir}
        )

    # Load LLM only
    llm = configs['llm']
    llm_weights_path = os.path.join(model_dir, 'llm.pt')
    llm.load_state_dict(torch.load(llm_weights_path, map_location='cpu'), strict=True)
    llm.eval()

    logger.info(f"Loaded CosyVoice3 LLM from {model_dir}")

    return llm, hf_llm_dir, configs


def get_speech_token_size(llm) -> int:
    """Determine speech token vocabulary size from the model."""
    # CosyVoice3LM has: speech_token_size + 200 in llm_decoder
    # speech_embedding has: speech_token_size + 200
    speech_embedding_size = llm.speech_embedding.num_embeddings
    # Use full embedding size (includes speech special tokens)
    return speech_embedding_size


def convert_cosyvoice3_to_hf(
    model_dir: str,
    output_dir: str,
    dtype: str = "bfloat16",
):
    """
    Convert CosyVoice3 LLM to HuggingFace format with merged embeddings.

    Merging architecture:
    - embed_tokens[0:original_vocab_size] = original text embeddings
    - embed_tokens[original_vocab_size:original_vocab_size+speech_token_size] = speech_embedding
    - lm_head[original_vocab_size:original_vocab_size+speech_token_size] = llm_decoder

    Args:
        model_dir: Path to CosyVoice3 model
        output_dir: Path to save HF model
        dtype: Data type for saving
    """
    logger.info(f"Loading CosyVoice3 model from {model_dir}")

    # 1. Load CosyVoice3 components
    cosyvoice3_llm, hf_llm_dir, configs = load_cosyvoice3_model(model_dir)

    # Extract key components
    qwen_model = cosyvoice3_llm.llm.model  # Qwen2ForCausalLM
    speech_embedding = cosyvoice3_llm.speech_embedding  # Embedding for speech tokens
    llm_decoder = cosyvoice3_llm.llm_decoder  # Linear for decoding to speech tokens

    speech_token_size = get_speech_token_size(cosyvoice3_llm)
    logger.info(f"Speech token size: {speech_token_size}")

    # 2. Load tokenizer and add CosyVoice3 text special tokens + speech tokens
    tokenizer = AutoTokenizer.from_pretrained(hf_llm_dir, trust_remote_code=True)
    base_vocab_size = len(tokenizer)
    logger.info(f"Base tokenizer vocab size: {base_vocab_size}")

    # IMPORTANT:
    # - In CosyVoice3, LLM speech special tokens (sos/eos/task_id/fill) are INSIDE speech_embedding,
    #   i.e. represented as <|s_6561|>, <|s_6562|>, <|s_6563|>, <|s_6564|>.
    # - But text-level special tokens like [cough]/[laughter] MUST exist in tokenizer
    #   (mirrors `CosyVoice3Tokenizer` from `cosyvoice/tokenizer/tokenizer.py`).
    special_tokens = {
        'eos_token': '<|endoftext|>',
        'pad_token': '<|endoftext|>',
        'additional_special_tokens': [
            '<|im_start|>', '<|im_end|>', '<|endofprompt|>',
            '[breath]', '<strong>', '</strong>', '[noise]',
            '[laughter]', '[cough]', '[clucking]', '[accent]',
            '[quick_breath]',
            "<laughter>", "</laughter>",
            "[hissing]", "[sigh]", "[vocalized-noise]",
            "[lipsmack]", "[mn]", "<|endofsystem|>",
            # Phoneme tokens (kept consistent with CosyVoice3Tokenizer)
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
            "[m]", "[n]", "[o]", "[ong]", "[ou]", "[p]", "[q]", "[r]",
            "[s]", "[sh]", "[t]", "[u]", "[uang]", "[ue]",
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
    text_vocab_size = len(tokenizer)
    logger.info(f"Tokenizer vocab after CosyVoice3 text special tokens: {text_vocab_size}")

    # Add speech tokens: <|s_0|>, <|s_1|>, ..., <|s_{embedding_size-1}|>
    # IMPORTANT: This range must match speech_embedding.num_embeddings (includes speech special tokens).
    actual_speech_tokens = speech_token_size  # Full embedding size (with speech special tokens)

    # replace <s_6561> to <|sos|>
    # replace <s_6562> to <|eos1|>
    # replace <s_6563> to <|task_id|>
    # replace <s_6564> to <|fill|>
    speech_tokens = [f"<|s_{i}|>" for i in range(actual_speech_tokens)]
    speech_tokens[6561] = "<|sos|>"
    speech_tokens[6562] = "<|eos1|>"
    speech_tokens[6563] = "<|task_id|>"
    speech_tokens[6564] = "<|fill|>"
    assert "<s_6561>" not in speech_tokens
    assert "<s_6562>" not in speech_tokens
    assert "<s_6563>" not in speech_tokens
    assert "<s_6564>" not in speech_tokens
    tokenizer.add_tokens(speech_tokens)

    new_vocab_size = len(tokenizer)
    logger.info(f"New tokenizer vocab size: {new_vocab_size}")
    logger.info(f"Added {new_vocab_size - base_vocab_size} tokens total (text special + speech tokens)")

    # 3. Resize embeddings in Qwen model
    # Align to 128 for TensorRT efficiency
    padded_vocab_size = ((new_vocab_size + 127) // 128) * 128
    qwen_model.resize_token_embeddings(padded_vocab_size)
    logger.info(f"Resized embeddings to: {padded_vocab_size}")

    # Speech tokens start after text vocab (base + CosyVoice3 text special tokens)
    speech_token_offset = text_vocab_size

    # 4. Copy speech_embedding into extended embed_tokens
    input_embeddings = qwen_model.get_input_embeddings()
    hidden_size = input_embeddings.weight.shape[1]

    logger.info(f"Hidden size: {hidden_size}")
    logger.info(f"speech_embedding shape: {speech_embedding.weight.shape}")
    logger.info(f"llm_decoder shape: {llm_decoder.weight.shape}")

    with torch.no_grad():
        # Copy speech_embedding weights into embed_tokens
        # Indices: [speech_token_offset, speech_token_offset + speech_token_size)
        src_size = min(speech_embedding.weight.shape[0], actual_speech_tokens)
        input_embeddings.weight[speech_token_offset:speech_token_offset + src_size] = \
            speech_embedding.weight[:src_size].to(input_embeddings.weight.dtype)

    logger.info(f"Copied speech_embedding to embed_tokens[{speech_token_offset}:{speech_token_offset + src_size}]")

    # 5. Create new lm_head with extended vocab and copy llm_decoder
    # Original lm_head: hidden_size -> original_vocab_size
    # New lm_head: hidden_size -> padded_vocab_size
    # llm_decoder: hidden_size -> speech_token_size

    # Create new lm_head
    has_bias = llm_decoder.bias is not None
    new_lm_head = torch.nn.Linear(
        in_features=hidden_size,
        out_features=padded_vocab_size,
        bias=has_bias
    )

    with torch.no_grad():
        # Initialize weights:
        # - Text part: copy from original lm_head (or zeros)
        # - Speech part: copy from llm_decoder
        # - Padding: zeros

        # Fill with zeros and -inf in bias (so text tokens are not generated)
        new_lm_head.weight.data.zero_()
        if has_bias:
            new_lm_head.bias.data.fill_(-float('inf'))

        # Copy original lm_head for text tokens (optional)
        original_lm_head = qwen_model.lm_head
        if original_lm_head is not None and original_lm_head.weight.shape[0] >= text_vocab_size:
            new_lm_head.weight[:text_vocab_size] = original_lm_head.weight[:text_vocab_size]
            if has_bias and original_lm_head.bias is not None:
                new_lm_head.bias[:text_vocab_size] = original_lm_head.bias[:text_vocab_size]

        # Copy llm_decoder for speech tokens
        decoder_size = min(llm_decoder.weight.shape[0], actual_speech_tokens)
        new_lm_head.weight[speech_token_offset:speech_token_offset + decoder_size] = \
            llm_decoder.weight[:decoder_size].to(new_lm_head.weight.dtype)

        if has_bias:
            new_lm_head.bias[speech_token_offset:speech_token_offset + decoder_size] = \
                llm_decoder.bias[:decoder_size].to(new_lm_head.bias.dtype)
        else:
            # If llm_decoder has no bias but we want it for text tokens
            pass

    # Replace lm_head
    qwen_model.lm_head = new_lm_head

    logger.info(f"Created new lm_head with shape: {new_lm_head.weight.shape}")
    logger.info(f"Copied llm_decoder to lm_head[{speech_token_offset}:{speech_token_offset + decoder_size}]")

    # 6. Update model configuration
    qwen_model.config.vocab_size = padded_vocab_size
    qwen_model.config.tie_word_embeddings = False  # Embeddings and lm_head are now different!

    # Set EOS token for generation (speech EOS lives inside speech_embedding as <|s_{base_speech_token_size+1}|>)
    base_speech_token_size = getattr(cosyvoice3_llm, "speech_token_size", 6561)
    eos_speech_idx = base_speech_token_size + 1
    eos_id = speech_token_offset + eos_speech_idx
    qwen_model.config.eos_token_id = eos_id

    # Generation settings
    qwen_model.generation_config.eos_token_id = eos_id
    qwen_model.generation_config.pad_token_id = eos_id
    qwen_model.generation_config.temperature = 0.8
    qwen_model.generation_config.top_p = 0.95
    qwen_model.generation_config.top_k = 25
    qwen_model.generation_config.repetition_penalty = 1.1
    qwen_model.generation_config.max_new_tokens = 2048

    # 7. Convert to target dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    target_dtype = dtype_map[dtype]
    qwen_model.to(target_dtype)

    # 8. Save model and tokenizer
    os.makedirs(output_dir, exist_ok=True)

    qwen_model.save_pretrained(output_dir)

    TEMPLATE = "{%- for message in messages %}{%- if message['role'] == 'user' %}{{- '<|sos|>' + message['content'] + '<|task_id|>' }}{%- elif message['role'] == 'assistant' %}{{- message['content']}}{%- endif %}{%- endfor %}"
    tokenizer.chat_template = TEMPLATE
    tokenizer.save_pretrained(output_dir)

    # Save metadata for TRT-LLM inference
    metadata = {
        "original_vocab_size": base_vocab_size,
        "text_vocab_size": text_vocab_size,
        "base_speech_token_size": base_speech_token_size,
        "embedding_size": actual_speech_tokens,
        "padded_vocab_size": padded_vocab_size,
        "eos_token_id": eos_id,
        "speech_token_offset": speech_token_offset,
        "dtype": dtype,
    }

    import json
    with open(os.path.join(output_dir, "cosyvoice3_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved HuggingFace model to {output_dir}")
    logger.info(f"Metadata: {metadata}")

    return output_dir, metadata


def main():
    args = parse_args()

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(args.model_dir, "hf_merged")

    convert_cosyvoice3_to_hf(
        model_dir=args.model_dir,
        output_dir=output_dir,
        dtype=args.dtype,
    )

    print("\n" + "=" * 70)
    print("✅ Conversion complete!")
    print("=" * 70)
    print(f"\nHuggingFace model saved to: {output_dir}")
    print("\nNext steps:")
    print("1. Convert to TRT-LLM weights:")
    print(f"   python -c \"from tensorrt_llm.models import QWenForCausalLM; ...")
    print("\n2. Build TRT-LLM engines:")
    print(f"   trtllm-build --checkpoint_dir <trt_weights_dir> --output_dir <trt_engines_dir> ...")
    print("=" * 70)


if __name__ == "__main__":
    main()