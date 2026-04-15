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
import os
import time
import threading
from typing import Generator, List
import numpy as np
from tqdm import tqdm
from hyperpyyaml import load_hyperpyyaml
from modelscope import snapshot_download
import torch
from cosyvoice.cli.frontend import CosyVoiceFrontEnd
from cosyvoice.cli.model import CosyVoiceModel, CosyVoice2Model, CosyVoice3Model
from cosyvoice.utils.file_utils import logging
from cosyvoice.utils.class_utils import get_model_type


class CosyVoice:

    def __init__(self, model_dir, load_jit=False, load_trt=False, fp16=False, trt_concurrent=1):
        self.model_dir = model_dir
        self.fp16 = fp16
        if not os.path.exists(model_dir):
            model_dir = snapshot_download(model_dir)
        hyper_yaml_path = '{}/cosyvoice.yaml'.format(model_dir)
        if not os.path.exists(hyper_yaml_path):
            raise ValueError('{} not found!'.format(hyper_yaml_path))
        with open(hyper_yaml_path, 'r') as f:
            configs = load_hyperpyyaml(f)
        assert get_model_type(configs) == CosyVoiceModel, 'do not use {} for CosyVoice initialization!'.format(model_dir)
        self.frontend = CosyVoiceFrontEnd(configs['get_tokenizer'],
                                          configs['feat_extractor'],
                                          '{}/campplus.onnx'.format(model_dir),
                                          '{}/speech_tokenizer_v1.onnx'.format(model_dir),
                                          '{}/spk2info.pt'.format(model_dir),
                                          configs['allowed_special'])
        self.sample_rate = configs['sample_rate']
        if torch.cuda.is_available() is False and (load_jit is True or load_trt is True or fp16 is True):
            load_jit, load_trt, fp16 = False, False, False
            logging.warning('no cuda device, set load_jit/load_trt/fp16 to False')
        self.model = CosyVoiceModel(configs['llm'], configs['flow'], configs['hift'], fp16)
        self.model.load('{}/llm.pt'.format(model_dir),
                        '{}/flow.pt'.format(model_dir),
                        '{}/hift.pt'.format(model_dir))
        if load_jit:
            self.model.load_jit('{}/llm.text_encoder.{}.zip'.format(model_dir, 'fp16' if self.fp16 is True else 'fp32'),
                                '{}/llm.llm.{}.zip'.format(model_dir, 'fp16' if self.fp16 is True else 'fp32'),
                                '{}/flow.encoder.{}.zip'.format(model_dir, 'fp16' if self.fp16 is True else 'fp32'))
        if load_trt:
            self.model.load_trt('{}/flow.decoder.estimator.{}.mygpu.plan'.format(model_dir, 'fp16' if self.fp16 is True else 'fp32'),
                                '{}/flow.decoder.estimator.fp32.onnx'.format(model_dir),
                                trt_concurrent,
                                self.fp16)
        del configs

    def list_available_spks(self):
        spks = list(self.frontend.spk2info.keys())
        return spks

    def add_zero_shot_spk(self, prompt_text, prompt_wav, zero_shot_spk_id):
        assert zero_shot_spk_id != '', 'do not use empty zero_shot_spk_id'
        model_input = self.frontend.frontend_zero_shot('', prompt_text, prompt_wav, self.sample_rate, '')
        del model_input['text']
        del model_input['text_len']
        self.frontend.spk2info[zero_shot_spk_id] = model_input
        return True

    def save_spkinfo(self):
        torch.save(self.frontend.spk2info, '{}/spk2info.pt'.format(self.model_dir))

    def inference_sft(self, tts_text, spk_id, stream=False, speed=1.0, text_frontend=True):
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            model_input = self.frontend.frontend_sft(i, spk_id)
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()

    def inference_zero_shot(self, tts_text, prompt_text, prompt_wav, zero_shot_spk_id='', stream=False, speed=1.0, text_frontend=True):
        prompt_text = self.frontend.text_normalize(prompt_text, split=False, text_frontend=text_frontend)
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            if (not isinstance(i, Generator)) and len(i) < 0.5 * len(prompt_text):
                logging.warning('synthesis text {} too short than prompt text {}, this may lead to bad performance'.format(i, prompt_text))
            model_input = self.frontend.frontend_zero_shot(i, prompt_text, prompt_wav, self.sample_rate, zero_shot_spk_id)
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()

    def inference_cross_lingual(self, tts_text, prompt_wav, zero_shot_spk_id='', stream=False, speed=1.0, text_frontend=True):
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            model_input = self.frontend.frontend_cross_lingual(i, prompt_wav, self.sample_rate, zero_shot_spk_id)
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()

    def inference_instruct(self, tts_text, spk_id, instruct_text, stream=False, speed=1.0, text_frontend=True):
        assert self.__class__.__name__ == 'CosyVoice', 'inference_instruct is only implemented for CosyVoice!'
        instruct_text = self.frontend.text_normalize(instruct_text, split=False, text_frontend=text_frontend)
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            model_input = self.frontend.frontend_instruct(i, spk_id, instruct_text)
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()

    def inference_vc(self, source_wav, prompt_wav, stream=False, speed=1.0):
        model_input = self.frontend.frontend_vc(source_wav, prompt_wav, self.sample_rate)
        start_time = time.time()
        for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
            speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
            logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
            yield model_output
            start_time = time.time()


class CosyVoice2(CosyVoice):

    def __init__(self, model_dir, load_jit=False, load_trt=False, load_vllm=False, fp16=False, trt_concurrent=1):
        self.model_dir = model_dir
        self.fp16 = fp16
        if not os.path.exists(model_dir):
            model_dir = snapshot_download(model_dir)
        hyper_yaml_path = '{}/cosyvoice2.yaml'.format(model_dir)
        if not os.path.exists(hyper_yaml_path):
            raise ValueError('{} not found!'.format(hyper_yaml_path))
        with open(hyper_yaml_path, 'r') as f:
            configs = load_hyperpyyaml(f, overrides={'qwen_pretrain_path': os.path.join(model_dir, 'CosyVoice-BlankEN')})
        assert get_model_type(configs) == CosyVoice2Model, 'do not use {} for CosyVoice2 initialization!'.format(model_dir)
        self.frontend = CosyVoiceFrontEnd(configs['get_tokenizer'],
                                          configs['feat_extractor'],
                                          '{}/campplus.onnx'.format(model_dir),
                                          '{}/speech_tokenizer_v2.onnx'.format(model_dir),
                                          '{}/spk2info.pt'.format(model_dir),
                                          configs['allowed_special'])
        self.sample_rate = configs['sample_rate']
        if torch.cuda.is_available() is False and (load_jit is True or load_trt is True or load_vllm is True or fp16 is True):
            load_jit, load_trt, load_vllm, fp16 = False, False, False, False
            logging.warning('no cuda device, set load_jit/load_trt/load_vllm/fp16 to False')
        self.model = CosyVoice2Model(configs['llm'], configs['flow'], configs['hift'], fp16)
        self.model.load('{}/llm.pt'.format(model_dir),
                        '{}/flow.pt'.format(model_dir),
                        '{}/hift.pt'.format(model_dir))
        if load_vllm:
            self.model.load_vllm('{}/vllm'.format(model_dir))
        if load_jit:
            self.model.load_jit('{}/flow.encoder.{}.zip'.format(model_dir, 'fp16' if self.fp16 is True else 'fp32'))
        if load_trt:
            self.model.load_trt('{}/flow.decoder.estimator.{}.mygpu.plan'.format(model_dir, 'fp16' if self.fp16 is True else 'fp32'),
                                '{}/flow.decoder.estimator.fp32.onnx'.format(model_dir),
                                trt_concurrent,
                                self.fp16)
        del configs

    def inference_instruct2(self, tts_text, instruct_text, prompt_wav, zero_shot_spk_id='', stream=False, speed=1.0, text_frontend=True):
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            model_input = self.frontend.frontend_instruct2(i, instruct_text, prompt_wav, self.sample_rate, zero_shot_spk_id)
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()


class CosyVoice3(CosyVoice2):

    def __init__(
        self,
        model_dir,
        load_trt=False,
        load_vllm=False,
        fp16=False,
        trt_concurrent=1,
        # llama.cpp parameters
        load_llama_cpp=False,
        gguf_model_path=None,
        speech_token_offset=None,  # auto-detect from GGUF vocab if None
    ):
        self.model_dir = model_dir
        self.fp16 = fp16
        self.gguf_model_path = gguf_model_path
        self._manual_speech_token_offset = speech_token_offset

        # match model's training context length
        self.llm_n_ctx = 32768
        # standard llama params
        self.llm_temperature = 0.8
        self.llm_top_p = 0.95
        self.llm_top_k = 25

        if load_llama_cpp and not gguf_model_path:
            raise ValueError('gguf_model_path must be provided when load_llama_cpp=True')
        if load_llama_cpp and not os.path.exists(gguf_model_path):
            raise FileNotFoundError('gguf_model_path not found: {}'.format(gguf_model_path))

        if not os.path.exists(model_dir):
            model_dir = snapshot_download(model_dir)
        hyper_yaml_path = '{}/cosyvoice3.yaml'.format(model_dir)
        if not os.path.exists(hyper_yaml_path):
            raise ValueError('{} not found!'.format(hyper_yaml_path))
        with open(hyper_yaml_path, 'r') as f:
            configs = load_hyperpyyaml(f, overrides={'qwen_pretrain_path': os.path.join(model_dir, 'CosyVoice-BlankEN')})
        assert get_model_type(configs) == CosyVoice3Model, 'do not use {} for CosyVoice3 initialization!'.format(model_dir)
        self.frontend = CosyVoiceFrontEnd(configs['get_tokenizer'],
                                          configs['feat_extractor'],
                                          '{}/campplus.onnx'.format(model_dir),
                                          '{}/speech_tokenizer_v3.onnx'.format(model_dir),
                                          '{}/spk2info.pt'.format(model_dir),
                                          configs['allowed_special'])
        self.sample_rate = configs['sample_rate']
        if torch.cuda.is_available() is False and (load_trt is True or fp16 is True):
            load_trt, fp16 = False, False
            logging.warning('no cuda device, set load_trt/fp16 to False')

        self.model = CosyVoice3Model(configs['llm'], configs['flow'], configs['hift'], fp16)

        # When using llama.cpp, skip loading PyTorch LLM weights to save VRAM
        self.model.load('{}/llm.pt'.format(model_dir),
                        '{}/flow.pt'.format(model_dir),
                        '{}/hift.pt'.format(model_dir),
                        load_llm=not load_llama_cpp)

        if load_vllm:
            self.model.load_vllm('{}/vllm'.format(model_dir))
        if load_trt:
            if self.fp16 is True:
                logging.warning('DiT tensorRT fp16 engine have some performance issue, use at caution!')
            self.model.load_trt('{}/flow.decoder.estimator.{}.mygpu.plan'.format(model_dir, 'fp16' if self.fp16 is True else 'fp32'),
                                '{}/flow.decoder.estimator.fp32.onnx'.format(model_dir),
                                trt_concurrent,
                                self.fp16)

        # Initialize llama.cpp if GGUF path provided
        self._llama_cpp_loaded = False
        if load_llama_cpp:
            self._init_speech_token_metadata()
            self._load_llama_cpp(gguf_model_path)
            logging.info('CosyVoice3 initialized with llama.cpp backend (gguf={})'.format(gguf_model_path))

        del configs

    # -------------------------------------------------------------------------
    # llama.cpp integration
    # -------------------------------------------------------------------------

    def _init_speech_token_metadata(self):
        """Initialize speech token ID constants for llama.cpp token mapping."""
        self.base_speech_token_size = 6561
        self.embedding_size = 6561 + 200
        self.speech_token_offset = 151936
        self.sos_speech_idx = self.base_speech_token_size + 0    # 6561
        self.eos_speech_idx = self.base_speech_token_size + 1    # 6562
        self.task_id_speech_idx = self.base_speech_token_size + 2  # 6563

    def _load_llama_cpp(self, gguf_model_path):
        """Load GGUF model via llama-cpp-python."""
        from llama_cpp import Llama

        self.llm_gguf = Llama(
            model_path=gguf_model_path,
            n_gpu_layers=-1,
            n_ctx=self.llm_n_ctx,
            logits_all=True,
            verbose=False,
            temperature=self.llm_temperature,
            top_p=self.llm_top_p,
            top_k=self.llm_top_k,
        )

        self.sos_token_id = self.speech_token_offset + self.sos_speech_idx
        self.eos_token_id = self.speech_token_offset + self.eos_speech_idx
        self.task_id_token_id = self.speech_token_offset + self.task_id_speech_idx

        self._llama_cpp_loaded = True

    def _sample_speech_token_constrained(self, logit_pos):
        """Sample next token constrained to speech tokens + EOS only.

        Uses manual logit extraction at the correct position.
        Fallback when built-in sample() produces text tokens.
        """
        logits = np.array(self.llm_gguf.scores[logit_pos], dtype=np.float32)
        n_vocab = len(logits)

        # Mask: only allow speech tokens [offset, offset+base_size) and EOS
        valid = np.full(n_vocab, False)
        s = self.speech_token_offset
        e = min(s + self.base_speech_token_size, n_vocab)
        valid[s:e] = True
        if self.eos_token_id < n_vocab:
            valid[self.eos_token_id] = True
        logits[~valid] = -np.inf

        logits = logits / max(self.llm_temperature, 1e-8)
        logits -= logits[valid].max()
        probs = np.exp(logits)
        probs /= probs.sum()

        if self.llm_top_k > 0:
            top_k = min(self.llm_top_k, int(np.sum(probs > 0)))
            if top_k > 0:
                threshold = np.sort(probs)[-top_k]
                probs[probs < threshold] = 0.0
                probs /= probs.sum()

        if self.llm_top_p < 1.0:
            sorted_idx = np.argsort(probs)[::-1]
            cum = np.cumsum(probs[sorted_idx])
            cutoff = np.searchsorted(cum, self.llm_top_p) + 1
            keep = sorted_idx[:cutoff]
            mask = np.zeros_like(probs)
            mask[keep] = probs[keep]
            probs = mask / mask.sum()

        return int(np.random.choice(n_vocab, p=probs))

    def _run_llama_cpp_inference(
        self,
        text_token_ids: List[int],
        prompt_text_token_ids: List[int],
        prompt_speech_tokens: List[int],
    ) -> List[int]:
        """
        Run llama.cpp inference to generate speech tokens.

        Uses pre-tokenized IDs from the CosyVoice frontend (same as PyTorch path).
        Format: [SOS] + prompt_text_ids + text_ids + [TASK_ID] + offset(prompt_speech_tokens)
        """
        all_text_ids = prompt_text_token_ids + text_token_ids
        prompt_speech_ids = [self.speech_token_offset + t for t in prompt_speech_tokens]
        input_ids = [self.sos_token_id] + all_text_ids + [self.task_id_token_id] + prompt_speech_ids

        self.llm_gguf.reset()
        self.llm_gguf.eval(input_ids)

        # Track position for constrained sampling fallback
        n_past = len(input_ids)

        speech_tokens = []
        raw_generated = []
        max_new_tokens = 2048

        for i in range(max_new_tokens):
            # Use built-in sample() (position-aware, like FastCosyVoice)
            next_token_id = self.llm_gguf.sample()

            # If built-in sample returns text token, retry with constrained sampling
            if (next_token_id != self.eos_token_id and
                not (self.speech_token_offset <= next_token_id < self.speech_token_offset + self.base_speech_token_size)):
                if i == 0:
                    logging.info('Built-in sample() returned text token {} on step 0, switching to constrained'.format(next_token_id))
                next_token_id = self._sample_speech_token_constrained(logit_pos=n_past - 1)

            raw_generated.append(next_token_id)

            if next_token_id == self.eos_token_id:
                break

            if self.speech_token_offset <= next_token_id < self.speech_token_offset + self.base_speech_token_size:
                speech_tokens.append(next_token_id - self.speech_token_offset)
            else:
                break

            self.llm_gguf.eval([next_token_id])
            n_past += 1

        return speech_tokens

    def _llama_cpp_job(
        self,
        text_token_ids: List[int],
        prompt_text_token_ids: List[int],
        prompt_speech_tokens: List[int],
        tokens_list: list,
        llm_end_flag: dict,
        tokens_lock: threading.Lock,
    ):
        """Thread target: generate all speech tokens via llama.cpp and fill shared tokens_list."""
        try:
            speech_tokens = self._run_llama_cpp_inference(
                text_token_ids=text_token_ids,
                prompt_text_token_ids=prompt_text_token_ids,
                prompt_speech_tokens=prompt_speech_tokens,
            )
            with tokens_lock:
                tokens_list.extend(speech_tokens)
        except Exception as e:
            logging.error('llama.cpp inference error: {}'.format(e), exc_info=True)
        finally:
            llm_end_flag['done'] = True

    # -------------------------------------------------------------------------
    # Overridden inference methods with llama.cpp support
    # -------------------------------------------------------------------------

    def _extract_token_ids(self, model_input):
        """Extract token ID lists from frontend model_input dict."""
        text_ids = model_input['text'].squeeze(0).tolist()
        prompt_text_ids = model_input.get('prompt_text', torch.zeros(1, 0, dtype=torch.int32)).squeeze(0).tolist()
        prompt_speech_ids = model_input.get('llm_prompt_speech_token', torch.zeros(1, 0, dtype=torch.int32)).squeeze(0).tolist()
        return text_ids, prompt_text_ids, prompt_speech_ids

    def inference_zero_shot(self, tts_text, prompt_text, prompt_wav, zero_shot_spk_id='', stream=False, speed=1.0, text_frontend=True):
        if not self._llama_cpp_loaded:
            yield from super().inference_zero_shot(tts_text, prompt_text, prompt_wav, zero_shot_spk_id, stream, speed, text_frontend)
            return

        # Generator text input: consume into string (llama.cpp needs full text upfront)
        if hasattr(tts_text, '__next__'):
            tts_text = ''.join(tts_text)
            logging.info('Consumed generator text: {}'.format(tts_text[:100]))

        prompt_text = self.frontend.text_normalize(prompt_text, split=False, text_frontend=text_frontend)
        for text_chunk in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            if (not isinstance(text_chunk, Generator)) and len(text_chunk) < 0.5 * len(prompt_text):
                logging.warning('synthesis text {} too short than prompt text'.format(text_chunk))

            model_input = self.frontend.frontend_zero_shot(text_chunk, prompt_text, prompt_wav, self.sample_rate, zero_shot_spk_id)
            start_time = time.time()
            logging.info('synthesis text {}'.format(text_chunk))

            text_ids, prompt_text_ids, prompt_speech_ids = self._extract_token_ids(model_input)

            if stream:
                tokens_list = []
                tokens_lock = threading.Lock()
                llm_end_flag = {'done': False}

                llm_thread = threading.Thread(
                    target=self._llama_cpp_job,
                    args=(text_ids, prompt_text_ids, prompt_speech_ids,
                          tokens_list, llm_end_flag, tokens_lock),
                    daemon=True
                )
                llm_thread.start()

                for model_output in self.model.tts_stream_external_llm(
                    tokens_list=tokens_list,
                    tokens_lock=tokens_lock,
                    llm_end_flag=llm_end_flag,
                    **{k: v for k, v in model_input.items() if k.startswith('flow') or k.startswith('prompt_speech')}
                ):
                    speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
                    logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                    yield model_output
                    start_time = time.time()

                llm_thread.join(timeout=5.0)
            else:
                speech_tokens = self._run_llama_cpp_inference(
                    text_token_ids=text_ids,
                    prompt_text_token_ids=prompt_text_ids,
                    prompt_speech_tokens=prompt_speech_ids,
                )
                model_output = self.model.tts_with_external_tokens(
                    tokens=speech_tokens,
                    speed=speed,
                    **{k: v for k, v in model_input.items() if k.startswith('flow') or k.startswith('prompt_speech')}
                )
                speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output

    def inference_cross_lingual(self, tts_text, prompt_wav, zero_shot_spk_id='', stream=False, speed=1.0, text_frontend=True):
        if not self._llama_cpp_loaded:
            yield from super().inference_cross_lingual(tts_text, prompt_wav, zero_shot_spk_id, stream, speed, text_frontend)
            return

        if hasattr(tts_text, '__next__'):
            tts_text = ''.join(tts_text)

        for text_chunk in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            model_input = self.frontend.frontend_cross_lingual(text_chunk, prompt_wav, self.sample_rate, zero_shot_spk_id)
            start_time = time.time()
            logging.info('synthesis text {}'.format(text_chunk))

            text_ids, prompt_text_ids, prompt_speech_ids = self._extract_token_ids(model_input)

            if stream:
                tokens_list = []
                tokens_lock = threading.Lock()
                llm_end_flag = {'done': False}

                llm_thread = threading.Thread(
                    target=self._llama_cpp_job,
                    args=(text_ids, prompt_text_ids, prompt_speech_ids,
                          tokens_list, llm_end_flag, tokens_lock),
                    daemon=True
                )
                llm_thread.start()

                for model_output in self.model.tts_stream_external_llm(
                    tokens_list=tokens_list,
                    tokens_lock=tokens_lock,
                    llm_end_flag=llm_end_flag,
                    **{k: v for k, v in model_input.items() if k.startswith('flow') or k.startswith('prompt_speech')}
                ):
                    speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
                    logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                    yield model_output
                    start_time = time.time()

                llm_thread.join(timeout=5.0)
            else:
                speech_tokens = self._run_llama_cpp_inference(
                    text_token_ids=text_ids,
                    prompt_text_token_ids=prompt_text_ids,
                    prompt_speech_tokens=prompt_speech_ids,
                )
                model_output = self.model.tts_with_external_tokens(
                    tokens=speech_tokens,
                    speed=speed,
                    **{k: v for k, v in model_input.items() if k.startswith('flow') or k.startswith('prompt_speech')}
                )
                speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output

    def inference_instruct2(self, tts_text, instruct_text, prompt_wav, zero_shot_spk_id='', stream=False, speed=1.0, text_frontend=True):
        if not self._llama_cpp_loaded:
            yield from super().inference_instruct2(tts_text, instruct_text, prompt_wav, zero_shot_spk_id, stream, speed, text_frontend)
            return

        if hasattr(tts_text, '__next__'):
            tts_text = ''.join(tts_text)

        for text_chunk in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            model_input = self.frontend.frontend_instruct2(text_chunk, instruct_text, prompt_wav, self.sample_rate, zero_shot_spk_id)
            start_time = time.time()
            logging.info('synthesis text {}'.format(text_chunk))

            text_ids, prompt_text_ids, prompt_speech_ids = self._extract_token_ids(model_input)

            if stream:
                tokens_list = []
                tokens_lock = threading.Lock()
                llm_end_flag = {'done': False}

                llm_thread = threading.Thread(
                    target=self._llama_cpp_job,
                    args=(text_ids, prompt_text_ids, prompt_speech_ids,
                          tokens_list, llm_end_flag, tokens_lock),
                    daemon=True
                )
                llm_thread.start()

                for model_output in self.model.tts_stream_external_llm(
                    tokens_list=tokens_list,
                    tokens_lock=tokens_lock,
                    llm_end_flag=llm_end_flag,
                    **{k: v for k, v in model_input.items() if k.startswith('flow') or k.startswith('prompt_speech')}
                ):
                    speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
                    logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                    yield model_output
                    start_time = time.time()

                llm_thread.join(timeout=5.0)
            else:
                speech_tokens = self._run_llama_cpp_inference(
                    text_token_ids=text_ids,
                    prompt_text_token_ids=prompt_text_ids,
                    prompt_speech_tokens=prompt_speech_ids,
                )
                model_output = self.model.tts_with_external_tokens(
                    tokens=speech_tokens,
                    speed=speed,
                    **{k: v for k, v in model_input.items() if k.startswith('flow') or k.startswith('prompt_speech')}
                )
                speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output

def AutoModel(**kwargs):
    if not os.path.exists(kwargs['model_dir']):
        kwargs['model_dir'] = snapshot_download(kwargs['model_dir'])
    if os.path.exists('{}/cosyvoice.yaml'.format(kwargs['model_dir'])):
        return CosyVoice(**kwargs)
    elif os.path.exists('{}/cosyvoice2.yaml'.format(kwargs['model_dir'])):
        return CosyVoice2(**kwargs)
    elif os.path.exists('{}/cosyvoice3.yaml'.format(kwargs['model_dir'])):
        return CosyVoice3(**kwargs)
    else:
        raise TypeError('No valid model type found!')
