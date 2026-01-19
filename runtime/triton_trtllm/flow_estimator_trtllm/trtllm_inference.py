"""
Inference script for CosyVoice TensorRT-LLM model
"""

import argparse
import json
import os
from functools import wraps

import tensorrt as trt
import torch
import numpy as np
from cuda import cudart

import tensorrt_llm
from tensorrt_llm._utils import str_dtype_to_torch, trt_dtype_to_torch
from tensorrt_llm.logger import logger
from tensorrt_llm.plugin.plugin import CustomAllReduceHelper
from tensorrt_llm.runtime.session import Session, TensorInfo


def CUASSERT(cuda_ret):
    err = cuda_ret[0]
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(
            f"CUDA ERROR: {err}, error code reference: https://nvidia.github.io/cuda-python/module/cudart.html#cuda.cudart.cudaError_t"
        )
    if len(cuda_ret) > 1:
        return cuda_ret[1:]
    return None


class CosyVoiceDiTTRT(object):
    """TensorRT-LLM inference wrapper for CosyVoice DiT"""

    def __init__(self,
                 engine_dir,
                 debug_mode=True,
                 stream: torch.cuda.Stream = None):

        # Load config
        config_file = os.path.join(engine_dir, 'config.json')
        with open(config_file) as f:
            config = json.load(f)

        self.dtype = config['pretrained_config']['dtype']
        self.hidden_size = config['pretrained_config']['hidden_size']
        self.mel_dim = config['pretrained_config']['mel_dim']
        self.mu_dim = config['pretrained_config'].get('mu_dim', self.mel_dim)
        self.spk_dim = config['pretrained_config']['spk_dim']

        rank = tensorrt_llm.mpi_rank()
        world_size = config['pretrained_config']['mapping']['world_size']
        cp_size = config['pretrained_config']['mapping']['cp_size']
        tp_size = config['pretrained_config']['mapping']['tp_size']
        pp_size = config['pretrained_config']['mapping']['pp_size']
        assert pp_size == 1

        self.mapping = tensorrt_llm.Mapping(
            world_size=world_size,
            rank=rank,
            cp_size=cp_size,
            tp_size=tp_size,
            pp_size=1,
            gpus_per_node=1  # Single GPU for now
        )

        local_rank = rank % self.mapping.gpus_per_node
        self.device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(self.device)
        CUASSERT(cudart.cudaSetDevice(local_rank))

        self.stream = stream
        if self.stream is None:
            self.stream = torch.cuda.Stream(self.device)
        torch.cuda.set_stream(self.stream)

        # Load engine
        engine_file = os.path.join(engine_dir, f"rank{rank}.engine")
        logger.info(f'Loading engine from {engine_file}')
        with open(engine_file, "rb") as f:
            engine_buffer = f.read()

        assert engine_buffer is not None
        self.session = Session.from_serialized_engine(engine_buffer)

        self.debug_mode = debug_mode
        self.inputs = {}
        self.outputs = {}
        self.buffer_allocated = False

        # Expected tensor names for Phase 2
        # Inputs: x, mu, t, spks, cond
        # Output: output (predicted noise)
        expected_tensor_names = ['x', 'mu', 't', 'spks', 'cond', 'output']

        if self.mapping.tp_size > 1:
            self.buffer, self.all_reduce_workspace = CustomAllReduceHelper.allocate_workspace(
                self.mapping,
                CustomAllReduceHelper.max_workspace_size_auto(self.mapping.tp_size)
            )
            self.inputs['all_reduce_workspace'] = self.all_reduce_workspace
            expected_tensor_names += ['all_reduce_workspace']

        found_tensor_names = [
            self.session.engine.get_tensor_name(i)
            for i in range(self.session.engine.num_io_tensors)
        ]

        logger.info(f"Expected tensor names: {expected_tensor_names}")
        logger.info(f"Found tensor names: {found_tensor_names}")

        if not self.debug_mode and set(expected_tensor_names) != set(found_tensor_names):
            logger.error(
                f"The following expected tensors are not found: {set(expected_tensor_names).difference(set(found_tensor_names))}"
            )
            logger.error(
                f"Those tensors in engine are not expected: {set(found_tensor_names).difference(set(expected_tensor_names))}"
            )
            raise RuntimeError("Tensor names in engine are not the same as expected.")

        if self.debug_mode:
            self.debug_tensors = list(set(found_tensor_names) - set(expected_tensor_names))

    def _tensor_dtype(self, name):
        """Return torch dtype given tensor name"""
        dtype = trt_dtype_to_torch(self.session.engine.get_tensor_dtype(name))
        return dtype

    def _setup(self, batch_size, seq_len):
        """Allocate output buffers"""
        for i in range(self.session.engine.num_io_tensors):
            name = self.session.engine.get_tensor_name(i)
            if self.session.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                # Get output shapes
                if name == 'output':
                    # Phase 2 output: predicted noise [batch, mel_dim, seq_len]
                    shape = [batch_size, self.mel_dim, seq_len]
                else:
                    shape = list(self.session.engine.get_tensor_shape(name))
                    shape[0] = batch_size

                self.outputs[name] = torch.empty(
                    shape,
                    dtype=self._tensor_dtype(name),
                    device=self.device
                )

        self.buffer_allocated = True

    def cuda_stream_guard(func):
        """Sync external stream and set current stream to the one bound to the session."""
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            external_stream = torch.cuda.current_stream()
            if external_stream != self.stream:
                external_stream.synchronize()
                torch.cuda.set_stream(self.stream)
            ret = func(self, *args, **kwargs)
            if external_stream != self.stream:
                self.stream.synchronize()
                torch.cuda.set_stream(external_stream)
            return ret
        return wrapper

    @cuda_stream_guard
    def forward(self, x: torch.Tensor, mu: torch.Tensor, t: torch.Tensor,
                spks: torch.Tensor, cond: torch.Tensor):
        """
        Forward pass of CosyVoice DiT

        Args:
            x: Noised mel-spec [batch, mel_dim, seq_len]
            mu: Text embeddings [batch, mu_dim, seq_len]
            t: Timestep [batch]
            spks: Speaker embeddings [batch, spk_dim]
            cond: Conditional audio [batch, mel_dim, seq_len]

        Returns:
            output: Predicted noise [batch, mel_dim, seq_len]
        """
        batch_size = x.shape[0]
        seq_len = x.shape[2]

        self._setup(batch_size, seq_len)
        if not self.buffer_allocated:
            raise RuntimeError('Buffer not allocated, please call setup first!')

        # Prepare inputs
        inputs = {
            'x': x.to(str_dtype_to_torch(self.dtype)),
            'mu': mu.to(str_dtype_to_torch(self.dtype)),
            't': t.float(),  # Timestep is always float32
            'spks': spks.to(str_dtype_to_torch(self.dtype)),
            'cond': cond.to(str_dtype_to_torch(self.dtype))
        }

        self.inputs.update(**inputs)
        self.session.set_shapes(self.inputs)

        # Run inference
        ok = self.session.run(self.inputs, self.outputs, self.stream.cuda_stream)

        if not ok:
            raise RuntimeError('Executing TRT engine failed!')

        if self.debug_mode:
            torch.cuda.synchronize()
            print("\n=== Debug: Input Stats ===")
            for k, v in self.inputs.items():
                if isinstance(v, torch.Tensor):
                    print(f"{k:20s}: shape={str(tuple(v.shape)):30s} mean={v.float().mean().item():10.6f} std={v.float().std().item():10.6f}")

            print("\n=== Debug: Output Stats ===")
            for k, v in self.outputs.items():
                if isinstance(v, torch.Tensor):
                    print(f"{k:20s}: shape={str(tuple(v.shape)):30s} mean={v.float().mean().item():10.6f} std={v.float().std().item():10.6f}")

        return self.outputs['output']