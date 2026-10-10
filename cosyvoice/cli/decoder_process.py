# Copyright (c) 2026 shrut2702
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
"""Runs flow + hift in a separate process for streaming inference.

In streaming mode CosyVoice2Model.tts runs the LM on a background thread and
flow + hift on the calling thread. Both hold the GIL for much of their time, so
the LM slows down whenever a chunk is being decoded. With the decoder in its
own process the two no longer share an interpreter, and the decoder can also be
given its own GPU.

Opt in with CosyVoice2(model_dir, decoder_process=True, decoder_device=1).
"""
import os
import queue
import threading
import time
import traceback
import weakref
import multiprocessing as mp

import torch


def _build_decoder(model_dir, yaml_name, fp16, load_jit, load_trt, trt_concurrent):
    from hyperpyyaml import load_hyperpyyaml
    from cosyvoice.utils.class_utils import get_model_type
    with open(os.path.join(model_dir, yaml_name), 'r') as f:
        configs = load_hyperpyyaml(f, overrides={'qwen_pretrain_path': os.path.join(model_dir, 'CosyVoice-BlankEN')})
    model_cls = get_model_type(configs)
    # NOTE the llm stays in the parent process, only flow and hift are loaded here
    model = model_cls(None, configs['flow'], configs['hift'], fp16)
    model.load(None, '{}/flow.pt'.format(model_dir), '{}/hift.pt'.format(model_dir))
    if load_jit:
        model.load_jit('{}/flow.encoder.{}.zip'.format(model_dir, 'fp16' if fp16 is True else 'fp32'))
    if load_trt:
        model.load_trt('{}/flow.decoder.estimator.{}.mygpu.plan'.format(model_dir, 'fp16' if fp16 is True else 'fp32'),
                       '{}/flow.decoder.estimator.fp32.onnx'.format(model_dir),
                       trt_concurrent,
                       fp16)
    del configs
    return model


def _decoder_worker(model_dir, yaml_name, fp16, load_jit, load_trt, trt_concurrent, device, requests, replies):
    try:
        if device is not None:
            # NOTE must happen before any cuda call, so torch.device('cuda') in the model resolves to this card
            torch.cuda.set_device(device)
        model = _build_decoder(model_dir, yaml_name, fp16, load_jit, load_trt, trt_concurrent)
    except Exception:
        replies.put(('fatal', None, traceback.format_exc()))
        return
    replies.put(('ready', None, None))

    prompts = {}
    while True:
        message = requests.get()
        if message is None:
            return
        kind, uuid, payload = message
        if kind == 'start':
            prompts[uuid] = payload
            model.hift_cache_dict[uuid] = None
        elif kind == 'end':
            prompts.pop(uuid, None)
            model.hift_cache_dict.pop(uuid, None)
        elif kind == 'chunk':
            # NOTE a chunk for an ended uuid means the caller stopped early, drop it
            if uuid not in prompts:
                continue
            token, token_offset, finalize = payload
            prompt_token, prompt_feat, embedding = prompts[uuid]
            try:
                # same arguments as CosyVoice2Model.tts, stream chunks are decoded with stream=True and the last one without
                speech = model.token2wav(token=token,
                                         prompt_token=prompt_token,
                                         prompt_feat=prompt_feat,
                                         embedding=embedding,
                                         token_offset=token_offset,
                                         uuid=uuid,
                                         stream=not finalize,
                                         finalize=finalize)
                replies.put(('audio', uuid, (speech.cpu(), finalize)))
            except Exception:
                prompts.pop(uuid, None)
                model.hift_cache_dict.pop(uuid, None)
                replies.put(('error', uuid, traceback.format_exc()))
            if finalize and torch.cuda.is_available():
                torch.cuda.empty_cache()


def _shutdown(process, requests):
    if process.is_alive():
        try:
            requests.put(None)
        except Exception:
            pass
        process.join(timeout=10)
    if process.is_alive():
        process.terminate()
        process.join(timeout=5)


class DecoderProcess:
    """Parent side of the decoder process.

    One process serves every request. Each request gets its own reply queue,
    keyed by uuid, which a reader thread fills from the shared one.
    """

    def __init__(self, model_dir, yaml_name, fp16=False, load_jit=False, load_trt=False, trt_concurrent=1, device=None, timeout=600):
        # NOTE spawn rather than fork, a forked child cannot use the parent's cuda context
        ctx = mp.get_context('spawn')
        self._requests, self._replies = ctx.Queue(), ctx.Queue()
        self._process = ctx.Process(target=_decoder_worker,
                                    args=(model_dir, yaml_name, fp16, load_jit, load_trt, trt_concurrent, device, self._requests, self._replies),
                                    daemon=True)
        self._process.start()
        self._finalizer = weakref.finalize(self, _shutdown, self._process, self._requests)

        deadline = time.time() + timeout
        while True:
            try:
                kind, _, payload = self._replies.get(timeout=1.0)
                break
            except queue.Empty:
                if not self._process.is_alive():
                    raise RuntimeError('decoder process exited with code {} before it was ready'.format(self._process.exitcode))
                if time.time() > deadline:
                    self.close()
                    raise RuntimeError('decoder process not ready after {}s'.format(timeout))
        if kind == 'fatal':
            self.close()
            raise RuntimeError('decoder process failed to load:\n{}'.format(payload))

        self._lock = threading.Lock()
        self._sessions = {}
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()

    def start(self, uuid, prompt_token, prompt_feat, embedding):
        """Registers a request and returns the queue its audio arrives on."""
        replies = queue.Queue()
        with self._lock:
            if not self._process.is_alive():
                raise RuntimeError('decoder process is not running (exit code {})'.format(self._process.exitcode))
            self._sessions[uuid] = replies
        self._requests.put(('start', uuid, (prompt_token.cpu(), prompt_feat.cpu(), embedding.cpu())))
        return replies

    def send_chunk(self, uuid, token, token_offset, finalize):
        """Queues a chunk for the decoder process and returns without waiting for its audio."""
        self._requests.put(('chunk', uuid, (token.cpu(), token_offset, finalize)))

    def fail(self, uuid, message):
        """Reports an error raised on the parent side, e.g. by the llm thread."""
        with self._lock:
            replies = self._sessions.get(uuid)
        if replies is not None:
            replies.put(('error', message))

    def end(self, uuid):
        with self._lock:
            self._sessions.pop(uuid, None)
        if self._process.is_alive():
            self._requests.put(('end', uuid, None))

    def close(self):
        self._finalizer()

    def _read(self):
        while True:
            try:
                kind, uuid, payload = self._replies.get(timeout=1.0)
            except queue.Empty:
                if self._process.is_alive():
                    continue
                # NOTE wake every waiting request rather than leaving it blocked forever
                with self._lock:
                    sessions, self._sessions = self._sessions, {}
                for replies in sessions.values():
                    replies.put(('error', 'decoder process exited with code {}'.format(self._process.exitcode)))
                return
            with self._lock:
                replies = self._sessions.get(uuid)
            if replies is not None:
                replies.put((kind, payload))
