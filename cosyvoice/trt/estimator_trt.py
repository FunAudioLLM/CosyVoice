import os
import torch
import tensorrt as trt
import logging
import threading


_min_shape = [(2, 80, 4), (2, 1, 4), (2, 80, 4), (2,), (2, 80), (2, 80, 4)]

_opt_shape = [(2, 80, 193), (2, 1, 193), (2, 80, 193), (2,), (2, 80), (2, 80, 193)]

_max_shape = [(2, 80, 6800), (2, 1, 6800), (2, 80, 6800), (2,), (2, 80), (2, 80, 6800)]


class EstimatorTRT:
    def __init__(self, path_prefix: str, device: torch.device, fp16: bool = True):
        self.lock = threading.Lock()
        self.device = device
        with torch.cuda.device(device):
            self.input_names = ["x", "mask", "mu", "t", "spks", "cond"]
            self.output_name = "estimator_out"

            onnx_path = path_prefix + ".fp32.onnx"
            precision = ".fp16" if fp16 else ".fp32"
            trt_path = path_prefix + precision +".plan"

            self.fp16 = fp16
            self.logger = trt.Logger(trt.Logger.INFO)
            self.trt_runtime = trt.Runtime(self.logger)

            save_trt = not os.environ.get("NOT_SAVE_TRT", "0") == "1"

            if os.path.exists(trt_path):
                self.engine = self._load_trt(trt_path)
            else:
                self.engine = self._convert_onnx_to_trt(onnx_path, trt_path, save_trt)

            self.context = self.engine.create_execution_context()

    def _convert_onnx_to_trt(
        self, onnx_path: str, trt_path: str, save_trt: bool = True
    ):
        logging.info("Converting onnx to trt...")

        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        builder = trt.Builder(self.logger)
        network = builder.create_network(network_flags)
        parser = trt.OnnxParser(network, self.logger)
        config = builder.create_builder_config()

        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 33) # 8GB
        if (self.fp16):
            config.set_flag(trt.BuilderFlag.FP16)

        profile = builder.create_optimization_profile()

        # load onnx model
        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                exit(1)

        # set input shapes
        for i in range(len(self.input_names)):
            profile.set_shape(
                self.input_names[i], _min_shape[i], _opt_shape[i], _max_shape[i]
            )

        tensor_dtype = trt.DataType.HALF if self.fp16 else trt.DataType.FLOAT

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
        if save_trt:
            with open(trt_path, "wb") as f:
                f.write(engine_bytes)
            print("trt engine saved to {}".format(trt_path))

        engine = self.trt_runtime.deserialize_cuda_engine(engine_bytes)
        return engine

    def _load_trt(self, trt_path: str):
        logging.info("Found trt engine, loading...")

        with open(trt_path, "rb") as f:
            engine_bytes = f.read()
        engine = self.trt_runtime.deserialize_cuda_engine(engine_bytes)
        return engine

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        mu: torch.Tensor,
        t: torch.Tensor,
        spks: torch.Tensor,
        cond: torch.Tensor,
    ):
        with self.lock:
            with torch.cuda.device(self.device):
                self.context.set_input_shape("x", (2, 80, x.size(2)))
                self.context.set_input_shape("mask", (2, 1, x.size(2)))
                self.context.set_input_shape("mu", (2, 80, x.size(2)))
                self.context.set_input_shape("t", (2,))
                self.context.set_input_shape("spks", (2, 80))
                self.context.set_input_shape("cond", (2, 80, x.size(2)))
                # run trt engine
                self.context.execute_v2(
                    [
                        x.contiguous().data_ptr(),
                        mask.contiguous().data_ptr(),
                        mu.contiguous().data_ptr(),
                        t.contiguous().data_ptr(),
                        spks.contiguous().data_ptr(),
                        cond.contiguous().data_ptr(),
                        x.data_ptr(),
                    ]
                )
                return x

    def __call__(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        mu: torch.Tensor,
        t: torch.Tensor,
        spks: torch.Tensor,
        cond: torch.Tensor,
    ):
        return self.forward(x, mask, mu, t, spks, cond)
