# Copyright (c) 2021 Mobvoi Inc. (authors: Binbin Zhang)
#               2024 Alibaba Inc (authors: Xiang Lyu, Zetao Hu)
#               2025 Alibaba Inc (authors: Xiang Lyu, Yabin Li)
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

import os
import json
import torch
import torchaudio
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s')


def read_lists(list_file):
    lists = []
    with open(list_file, 'r', encoding='utf8') as fin:
        for line in fin:
            lists.append(line.strip())
    return lists


def read_json_lists(list_file):
    lists = read_lists(list_file)
    results = {}
    for fn in lists:
        with open(fn, 'r', encoding='utf8') as fin:
            results.update(json.load(fin))
    return results


def load_wav(wav, target_sr, min_sr=16000):
    speech, sample_rate = torchaudio.load(wav, backend='soundfile')
    speech = speech.mean(dim=0, keepdim=True)
    if sample_rate != target_sr:
        assert sample_rate >= min_sr, 'wav sample rate {} must be greater than {}'.format(sample_rate, target_sr)
        speech = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)(speech)
    return speech


def convert_onnx_to_trt(trt_model, trt_kwargs, onnx_model, fp16):
    import tensorrt as trt
    logging.info("Converting onnx to trt...")
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 32)  # 4GB
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


# NOTE do not support bistream inference as only speech token embedding/head is kept
_COSYVOICE3_TEXT_SPECIAL_TOKENS = [
    '<|im_start|>', '<|im_end|>', '<|endofprompt|>',
    '[breath]', '<strong>', '</strong>', '[noise]', '[laughter]', '[cough]',
    '[clucking]', '[accent]', '[quick_breath]', '<laughter>', '</laughter>',
    '[hissing]', '[sigh]', '[vocalized-noise]', '[lipsmack]', '[mn]', '<|endofsystem|>',
    '[AA]', '[AA0]', '[AA1]', '[AA2]', '[AE]', '[AE0]', '[AE1]', '[AE2]',
    '[AH]', '[AH0]', '[AH1]', '[AH2]', '[AO]', '[AO0]', '[AO1]', '[AO2]',
    '[AW]', '[AW0]', '[AW1]', '[AW2]', '[AY]', '[AY0]', '[AY1]', '[AY2]',
    '[B]', '[CH]', '[D]', '[DH]', '[EH]', '[EH0]', '[EH1]', '[EH2]',
    '[ER]', '[ER0]', '[ER1]', '[ER2]', '[EY]', '[EY0]', '[EY1]', '[EY2]',
    '[F]', '[G]', '[HH]', '[IH]', '[IH0]', '[IH1]', '[IH2]',
    '[IY]', '[IY0]', '[IY1]', '[IY2]', '[JH]', '[K]', '[L]', '[M]', '[N]',
    '[NG]', '[OW]', '[OW0]', '[OW1]', '[OW2]', '[OY]', '[OY0]', '[OY1]', '[OY2]',
    '[P]', '[R]', '[S]', '[SH]', '[T]', '[TH]', '[UH]', '[UH0]', '[UH1]', '[UH2]',
    '[UW]', '[UW0]', '[UW1]', '[UW2]', '[V]', '[W]', '[Y]', '[Z]', '[ZH]',
    '[a]', '[ai]', '[an]', '[ang]', '[ao]', '[b]', '[c]', '[ch]', '[d]', '[e]',
    '[ei]', '[en]', '[eng]', '[f]', '[g]', '[h]', '[i]', '[ian]', '[in]', '[ing]',
    '[iu]', '[ià]', '[iàn]', '[iàng]', '[iào]', '[iá]', '[ián]', '[iáng]', '[iáo]',
    '[iè]', '[ié]', '[iòng]', '[ióng]', '[iù]', '[iú]', '[iā]', '[iān]', '[iāng]',
    '[iāo]', '[iē]', '[iě]', '[iōng]', '[iū]', '[iǎ]', '[iǎn]', '[iǎng]', '[iǎo]',
    '[iǒng]', '[iǔ]', '[j]', '[k]', '[l]', '[m]', '[n]', '[o]', '[ong]', '[ou]',
    '[p]', '[q]', '[r]', '[s]', '[sh]', '[t]', '[u]', '[uang]', '[ue]', '[un]',
    '[uo]', '[uà]', '[uài]', '[uàn]', '[uàng]', '[uá]', '[uái]', '[uán]', '[uáng]',
    '[uè]', '[ué]', '[uì]', '[uí]', '[uò]', '[uó]', '[uā]', '[uāi]', '[uān]',
    '[uāng]', '[uē]', '[uě]', '[uī]', '[uō]', '[uǎ]', '[uǎi]', '[uǎn]', '[uǎng]',
    '[uǐ]', '[uǒ]', '[vè]', '[w]', '[x]', '[y]', '[z]', '[zh]',
    '[à]', '[ài]', '[àn]', '[àng]', '[ào]', '[á]', '[ái]', '[án]', '[áng]', '[áo]',
    '[è]', '[èi]', '[èn]', '[èng]', '[èr]', '[é]', '[éi]', '[én]', '[éng]', '[ér]',
    '[ì]', '[ìn]', '[ìng]', '[í]', '[ín]', '[íng]', '[ò]', '[òng]', '[òu]',
    '[ó]', '[óng]', '[óu]', '[ù]', '[ùn]', '[ú]', '[ún]', '[ā]', '[āi]', '[ān]',
    '[āng]', '[āo]', '[ē]', '[ēi]', '[ēn]', '[ēng]', '[ě]', '[ěi]', '[ěn]',
    '[ěng]', '[ěr]', '[ī]', '[īn]', '[īng]', '[ō]', '[ōng]', '[ōu]', '[ū]',
    '[ūn]', '[ǎ]', '[ǎi]', '[ǎn]', '[ǎng]', '[ǎo]', '[ǐ]', '[ǐn]', '[ǐng]',
    '[ǒ]', '[ǒng]', '[ǒu]', '[ǔ]', '[ǔn]', '[ǘ]', '[ǚ]', '[ǜ]',
]

# Chat template for CosyVoice3 speculative vllm: <|sos|>{text}<|task_id|>{speech_tokens}
_COSYVOICE3_CHAT_TEMPLATE = (
    "{%- for message in messages %}"
    "{%- if message['role'] == 'user' %}{{- '<|sos|>' + message['content'] + '<|task_id|>' }}"
    "{%- elif message['role'] == 'assistant' %}{{- message['content']}}"
    "{%- endif %}{%- endfor %}"
)


def export_cosyvoice3_vllm_spec(model, model_path, hf_llm_dir, device):
    """Export CosyVoice3LM to HuggingFace format compatible with DSpark speculative decoding.

    Extends the Qwen2 tokenizer with CosyVoice3 text special tokens and speech tokens,
    then splices speech_embedding / llm_decoder into the model's embed_tokens / lm_head
    at offset `text_vocab_size`. This produces the same token-ID layout expected by the
    DSpark draft model (yuekai/cosyvoice3_llm_dspark): speech token N lives at token ID
    `text_vocab_size + N`.

    Mirrors export_cosyvoice2_vllm() in style: skips if model_path already exists.
    """
    if os.path.exists(model_path):
        return

    from transformers import AutoTokenizer
    import math

    dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(hf_llm_dir, trust_remote_code=True)
    tokenizer.add_special_tokens({
        'eos_token': '<|endoftext|>',
        'pad_token': '<|endoftext|>',
        'additional_special_tokens': _COSYVOICE3_TEXT_SPECIAL_TOKENS,
    })
    text_vocab_size = len(tokenizer)

    speech_token_size = model.speech_embedding.num_embeddings  # includes 200 special
    base_speech_token_size = model.speech_token_size            # e.g. 2512 or 6561

    speech_tokens = [f'<|s_{i}|>' for i in range(speech_token_size)]
    speech_tokens[base_speech_token_size + 0] = '<|sos|>'
    speech_tokens[base_speech_token_size + 1] = '<|eos1|>'
    speech_tokens[base_speech_token_size + 2] = '<|task_id|>'
    speech_tokens[base_speech_token_size + 3] = '<|fill|>'
    tokenizer.add_tokens(speech_tokens)
    tokenizer.chat_template = _COSYVOICE3_CHAT_TEMPLATE

    # Pad vocab to multiple of 128 for efficiency
    new_vocab = len(tokenizer)
    padded_vocab = math.ceil(new_vocab / 128) * 128

    qwen_model = model.llm.model
    qwen_model.resize_token_embeddings(padded_vocab)
    qwen_model.to(dtype).to(device)

    with torch.no_grad():
        # Splice speech_embedding into embed_tokens at [text_vocab_size:]
        src = min(model.speech_embedding.weight.shape[0], speech_token_size)
        qwen_model.get_input_embeddings().weight[text_vocab_size:text_vocab_size + src] = \
            model.speech_embedding.weight[:src].to(dtype)

        # Build new lm_head: text part zeroed (not generated), speech part = llm_decoder
        has_bias = model.llm_decoder.bias is not None
        new_lm_head = torch.nn.Linear(
            qwen_model.config.hidden_size, padded_vocab, bias=has_bias, device=device, dtype=dtype)
        new_lm_head.weight.data.zero_()
        if has_bias:
            new_lm_head.bias.data.fill_(float('-inf'))
        # Copy text part from original lm_head so text logits stay valid (optional but clean)
        orig_head = qwen_model.lm_head
        copy_text = min(orig_head.weight.shape[0], text_vocab_size)
        new_lm_head.weight[:copy_text] = orig_head.weight[:copy_text].to(dtype)
        # Copy llm_decoder into speech part
        dec_size = min(model.llm_decoder.weight.shape[0], speech_token_size)
        new_lm_head.weight[text_vocab_size:text_vocab_size + dec_size] = \
            model.llm_decoder.weight[:dec_size].to(dtype)
        if has_bias:
            new_lm_head.bias[text_vocab_size:text_vocab_size + dec_size] = \
                model.llm_decoder.bias[:dec_size].to(dtype)
        qwen_model.lm_head = new_lm_head

    eos_id = text_vocab_size + base_speech_token_size + 1
    qwen_model.config.vocab_size = padded_vocab
    qwen_model.config.tie_word_embeddings = False
    qwen_model.generation_config.eos_token_id = eos_id
    qwen_model.generation_config.pad_token_id = eos_id

    os.makedirs(model_path, exist_ok=True)
    qwen_model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    json.dump({
        'text_vocab_size': text_vocab_size,
        'base_speech_token_size': base_speech_token_size,
        'embedding_size': speech_token_size,
        'padded_vocab_size': padded_vocab,
        'eos_token_id': eos_id,
        'speech_token_offset': text_vocab_size,
    }, open(os.path.join(model_path, 'cosyvoice3_metadata.json'), 'w'), indent=2)


def export_cosyvoice2_vllm(model, model_path, device):
    if os.path.exists(model_path):
        return

    dtype = torch.bfloat16
    # lm_head
    use_bias = True if model.llm_decoder.bias is not None else False
    model.llm.model.lm_head = model.llm_decoder
    # embed_tokens
    embed_tokens = model.llm.model.model.embed_tokens
    model.llm.model.set_input_embeddings(model.speech_embedding)
    model.llm.model.to(device)
    model.llm.model.to(dtype)
    tmp_vocab_size = model.llm.model.config.vocab_size
    tmp_tie_embedding = model.llm.model.config.tie_word_embeddings
    del model.llm.model.generation_config.eos_token_id
    del model.llm.model.config.bos_token_id
    del model.llm.model.config.eos_token_id
    model.llm.model.config.vocab_size = model.speech_embedding.num_embeddings
    model.llm.model.config.tie_word_embeddings = False
    model.llm.model.config.use_bias = use_bias
    model.llm.model.save_pretrained(model_path)
    if use_bias is True:
        os.system('sed -i s@Qwen2ForCausalLM@CosyVoice2ForCausalLM@g {}/config.json'.format(os.path.abspath(model_path)))
    model.llm.model.config.vocab_size = tmp_vocab_size
    model.llm.model.config.tie_word_embeddings = tmp_tie_embedding
    model.llm.model.set_input_embeddings(embed_tokens)
