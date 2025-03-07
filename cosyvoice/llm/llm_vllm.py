# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Zhihao Du)
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
import time
import queue
import asyncio
import threading
from typing import List, Generator, AsyncGenerator
import torch
from cosyvoice.utils.file_utils import logging
from cosyvoice.llm.llm import Qwen2LM

# 启用vllm V1版本
import os
os.environ["VLLM_USE_V1"] = '1'
from vllm import ModelRegistry
from vllm import LLMEngine, AsyncLLMEngine, CompletionOutput
from vllm.engine.arg_utils import EngineArgs, AsyncEngineArgs
from vllm.sampling_params import SamplingParams

from cosyvoice.llm.vllm_use_cosyvoice2_model import CosyVoice2Model as CosyVoice2LLM
ModelRegistry.register_model("CosyVoice2Model", CosyVoice2LLM)

# EngineArgs
ENGINE_ARGS = {
    "block_size": 16,
    "swap_space": 0,
    # "enforce_eager": True,
    "gpu_memory_utilization": 0.4,
    "max_num_batched_tokens": 1024,
    "max_model_len": 1024,
    "max_num_seqs": 256,
    "disable_log_requests": True,
    "disable_log_stats": True,
    "dtype": "float16"
}

from vllm.sampling_params import RequestOutputKind
# SamplingParams
SAMPLING_PARAMS = {
    "temperature": 1,  # 不能低于0.8, 否则会生成非常多的空音频，或者无法正常生成语音Token
    "top_p": 1,       # 不能低于0.8, 否则会生成非常多的空音频，或者无法正常生成语音Token
    "top_k": 25,
    # "min_tokens": 80,       # 不支持设置最小的tokens数量设置，开启后vllm直接崩溃，无法启动
    # "presence_penalty": 1.0,    # 不支持设置
    # "frequency_penalty": 0.0,   # 不支持设置
    "max_tokens": 1024,
    "detokenize": False,          # 目前 vllm 0.7.3 v1版本中设置无效，待后续版本更新后减少计算
    "ignore_eos": False,
    "output_kind": RequestOutputKind.DELTA  # 设置为DELTA，如调整该参数，请同时调整llm_inference的处理代码
}

def tensor_to_list(tensor: torch.tensor):
    return tensor.view(-1).cpu().numpy().tolist()

class VllmQwen2LM(Qwen2LM):
    def __init__(
            self,
            model_dir,
            mix_ratio: List[int] = [5, 15],
    ):
        self.fp16 = False
        self.half = lambda: None
        self.mix_ratio = mix_ratio
        # ---------------------------------------------
        # vllm engine 的参数配置
        engine_args = AsyncEngineArgs(
            model=model_dir,
            **ENGINE_ARGS,
        )
        self.llm_engine: AsyncLLMEngine = AsyncLLMEngine.from_engine_args(engine_args)

        self.speech_token_size = 6564       # 6561 + 3
        self.llm_token_size = 151936        # llm  vocab_size
        self.sos_eos_token_id = self.speech_token_size + self.llm_token_size + 1
        self.task_token_id = self.sos_eos_token_id + 1
        self.zero_token_id = self.task_token_id + 1

        # 不能直接在同步函数正确的使用 异步的生成器函数，即使使用协程也会对vllm造成崩溃
        # 使用 queue 的方式，后台线程运行推理任务
        self.task_queue = queue.Queue()
        self.loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self.loop_thread.start()
        # 运行后台协程，用于处理任务队列中的任务
        # TODO: 目前只能单任务运行，多任务运行需要对 inference_processor 进行修改
        asyncio.run_coroutine_threadsafe(self.inference_processor(self.task_queue), self.loop)

    def _run_event_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    async def inference_processor(self, task_queue):
        while True:
            try:
                logging.debug(f"inference_processor")
                out_queue, prompt_token_ids, request_id, stop_token_ids, max_tokens = task_queue.get()
                sampling_params = SamplingParams(**SAMPLING_PARAMS)
                sampling_params.stop_token_ids = stop_token_ids or [6561]
                if max_tokens:
                    sampling_params.max_tokens = max_tokens
                async for output in self.llm_engine.generate(
                        {
                            "prompt_token_ids": prompt_token_ids,
                        },
                        sampling_params=sampling_params,
                        request_id=request_id or f"{time.time()}",
                ):
                    out_queue.put((output.outputs[0], output.finished))
            except Exception as e:
                logging.error(f"Error in inference_processor: {e}")

    def llm_inference(self, prompt_token_ids: List[int], request_id: str=None, stop_token_ids=None, max_tokens=None):
        # 使用 同步转异步 会导致vllm崩溃，目前选择 queue 的方式，后台线程运行推理任务
        # 提交推理任务到队列中
        out_queue = queue.Queue()
        self.task_queue.put((out_queue, prompt_token_ids, request_id, stop_token_ids, max_tokens))
        # 将 out_queue 的结果返回
        finished = False
        while not finished:
            (output, finished) = out_queue.get_nowait() if not out_queue.empty() else out_queue.get()
            yield output

    def inference(
            self,
            text: torch.Tensor,
            text_len: torch.Tensor,
            prompt_text: torch.Tensor,
            prompt_text_len: torch.Tensor,
            prompt_speech_token: torch.Tensor,
            prompt_speech_token_len: torch.Tensor,
            embedding: torch.Tensor,
            sampling: int = 25,
            max_token_text_ratio: float = 20,
            min_token_text_ratio: float = 2,
    ) -> Generator[torch.Tensor|int, None, None]:
        prompt_text = tensor_to_list(prompt_text + torch.tensor(6564))
        prompt_speech_token = tensor_to_list(prompt_speech_token)

        text = tensor_to_list(text + torch.tensor(6564))
        prompt_token_ids = [self.sos_eos_token_id] + prompt_text + text + \
                           [self.task_token_id] + prompt_speech_token
        max_tokens = len(text) * 20
        for output in self.llm_inference(
                prompt_token_ids,
                stop_token_ids=[6561],
                max_tokens=max_tokens,
        ):
            if output.token_ids[-1] == 6561:
                need_add_tokens = output.token_ids[:-1]
            else:
                need_add_tokens = output.token_ids
            # 单个token 循环处理比较耗时，建议是在model中进行批量（extend）处理，减少循环
            # yield need_add_tokens
            for token in need_add_tokens:
                yield token

    def inference_bistream(
            self,
            text: Generator,
            prompt_text: torch.Tensor,
            prompt_text_len: torch.Tensor,
            prompt_speech_token: torch.Tensor,
            prompt_speech_token_len: torch.Tensor,
            embedding: torch.Tensor,
            sampling: int = 25,
            max_token_text_ratio: float = 20,
            min_token_text_ratio: float = 2,
    ) -> Generator[torch.Tensor, None, None]:
        prompt_text = tensor_to_list(prompt_text + torch.tensor(6564))
        prompt_speech_token = tensor_to_list(prompt_speech_token)

        last_tokens = []
        prompt_token_ids = [self.sos_eos_token_id]
        text_tokens_cache = prompt_text
        for this_text in text:
            this_text = tensor_to_list(this_text + torch.tensor(6564))
            # text need tokens
            assert isinstance(this_text, list), "text need token ids List[int]."
            text_tokens_cache += this_text
            while len(prompt_speech_token) != 0:
                if len(text_tokens_cache) >= self.mix_ratio[0]:
                    text_input_token = text_tokens_cache[:self.mix_ratio[0]]
                    speech_input_token = prompt_speech_token[:self.mix_ratio[1]]
                    prompt_token_ids += text_input_token + speech_input_token
                    # reset the last cache
                    text_tokens_cache = text_tokens_cache[self.mix_ratio[0]:]
                    prompt_speech_token = prompt_speech_token[self.mix_ratio[1]:]
                else:
                    logging.info('not enough text token to decode, wait for more')
                    break
            if len(prompt_speech_token) == 0:
                if (len(last_tokens) > 0 and last_tokens[-1] == 6563) or len(prompt_token_ids) == 1:
                    logging.info('get fill token, need to append more text token')
                    if len(text_tokens_cache) >= self.mix_ratio[0]:
                        text_tokens_temp = text_tokens_cache[:self.mix_ratio[0]]
                        prompt_token_ids += text_tokens_temp
                        logging.info('append {} text token'.format(len(text_tokens_temp)))
                        text_tokens_cache = text_tokens_cache[self.mix_ratio[0]:]
                    else:
                        logging.info('not enough text token to decode, wait for more')
                        continue
                for output in self.llm_inference(prompt_token_ids, stop_token_ids=[6563]):
                    last_tokens = output.token_ids
                    if last_tokens[-1] == 6563:
                        need_add_tokens = last_tokens[:-1]
                    else:
                        need_add_tokens = last_tokens
                    # 单个token 循环处理比较耗时，建议是在model中进行批量（extend）处理，减少循环
                    # yield need_add_tokens
                    for token in need_add_tokens:
                        yield token
                    prompt_token_ids.extend(need_add_tokens)
        prompt_token_ids += text_tokens_cache + [self.task_token_id]
        logging.info('no more text token, decode until met eos')
        for output in self.llm_inference(prompt_token_ids, stop_token_ids=[6561]):
            if output.token_ids[-1] == 6561:
                need_add_tokens = output.token_ids[:-1]
            else:
                need_add_tokens = output.token_ids
            # 单个token 循环处理比较耗时，建议是在model中进行批量（extend）处理，减少循环
            # yield need_add_tokens
            for token in need_add_tokens:
                yield token
