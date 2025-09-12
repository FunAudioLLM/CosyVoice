# Copyright      2022  Xiaomi Corp.        (authors: Fangjun Kuang)
#                2023  Nvidia              (authors: Yuekai Zhang)
#                2023  Recurrent.ai        (authors: Songtao Shi)
# See LICENSE for clarification regarding multiple authors
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
This script supports to load dataset from huggingface and sends it to the server
for decoding, in parallel.

Usage:
num_task=2

# For offline F5-TTS
python3 client_grpc.py \
    --server-addr localhost \
    --model-name f5_tts \
    --num-tasks $num_task \
    --huggingface-dataset yuekai/seed_tts \
    --split-name test_zh \
    --log-dir ./log_concurrent_tasks_${num_task}

# For offline Spark-TTS-0.5B
python3 client_grpc.py \
    --server-addr localhost \
    --model-name spark_tts \
    --num-tasks $num_task \
    --huggingface-dataset yuekai/seed_tts \
    --split-name wenetspeech4tts \
    --log-dir ./log_concurrent_tasks_${num_task}
"""

import argparse
import asyncio
import json
import queue  # Added
import uuid  # Added
import functools  # Added

import os
import time
import types
from pathlib import Path

import numpy as np
import soundfile as sf
import tritonclient
import tritonclient.grpc.aio as grpcclient_aio  # Renamed original import
import tritonclient.grpc as grpcclient_sync  # Added sync client import
from tritonclient.utils import np_to_triton_dtype, InferenceServerException  # Added InferenceServerException


# --- Added UserData and callback ---
class UserData:
    def __init__(self):
        self._completed_requests = queue.Queue()
        self._first_chunk_time = None
        self._start_time = None

    def record_start_time(self):
        self._start_time = time.time()

    def get_first_chunk_latency(self):
        if self._first_chunk_time and self._start_time:
            return self._first_chunk_time - self._start_time
        return None


def callback(user_data, result, error):
    if user_data._first_chunk_time is None and not error:
        user_data._first_chunk_time = time.time()  # Record time of first successful chunk
    if error:
        user_data._completed_requests.put(error)
    else:
        user_data._completed_requests.put(result)
# --- End Added UserData and callback ---


def write_triton_stats(stats, summary_file):
    with open(summary_file, "w") as summary_f:
        model_stats = stats["model_stats"]
        # write a note, the log is from triton_client.get_inference_statistics(), to better human readability
        summary_f.write(
            "The log is parsing from triton_client.get_inference_statistics(), to better human readability. \n"
        )
        summary_f.write("To learn more about the log, please refer to: \n")
        summary_f.write("1. https://github.com/triton-inference-server/server/blob/main/docs/user_guide/metrics.md \n")
        summary_f.write("2. https://github.com/triton-inference-server/server/issues/5374 \n\n")
        summary_f.write(
            "To better improve throughput, we always would like let requests wait in the queue for a while, and then execute them with a larger batch size. \n"
        )
        summary_f.write(
            "However, there is a trade-off between the increased queue time and the increased batch size. \n"
        )
        summary_f.write(
            "You may change 'max_queue_delay_microseconds' and 'preferred_batch_size' in the model configuration file to achieve this. \n"
        )
        summary_f.write(
            "See https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#delayed-batching for more details. \n\n"
        )
        for model_state in model_stats:
            if "last_inference" not in model_state:
                continue
            summary_f.write(f"model name is {model_state['name']} \n")
            model_inference_stats = model_state["inference_stats"]
            total_queue_time_s = int(model_inference_stats["queue"]["ns"]) / 1e9
            total_infer_time_s = int(model_inference_stats["compute_infer"]["ns"]) / 1e9
            total_input_time_s = int(model_inference_stats["compute_input"]["ns"]) / 1e9
            total_output_time_s = int(model_inference_stats["compute_output"]["ns"]) / 1e9
            summary_f.write(
                f"queue time {total_queue_time_s:<5.2f} s, compute infer time {total_infer_time_s:<5.2f} s, compute input time {total_input_time_s:<5.2f} s, compute output time {total_output_time_s:<5.2f} s \n"  # noqa
            )
            model_batch_stats = model_state["batch_stats"]
            for batch in model_batch_stats:
                batch_size = int(batch["batch_size"])
                compute_input = batch["compute_input"]
                compute_output = batch["compute_output"]
                compute_infer = batch["compute_infer"]
                batch_count = int(compute_infer["count"])
                assert compute_infer["count"] == compute_output["count"] == compute_input["count"]
                compute_infer_time_ms = int(compute_infer["ns"]) / 1e6
                compute_input_time_ms = int(compute_input["ns"]) / 1e6
                compute_output_time_ms = int(compute_output["ns"]) / 1e6
                summary_f.write(
                    f"execuate inference with batch_size {batch_size:<2} total {batch_count:<5} times, total_infer_time {compute_infer_time_ms:<9.2f} ms, avg_infer_time {compute_infer_time_ms:<9.2f}/{batch_count:<5}={compute_infer_time_ms / batch_count:.2f} ms, avg_infer_time_per_sample {compute_infer_time_ms:<9.2f}/{batch_count:<5}/{batch_size}={compute_infer_time_ms / batch_count / batch_size:.2f} ms \n"  # noqa
                )
                summary_f.write(
                    f"input {compute_input_time_ms:<9.2f} ms, avg {compute_input_time_ms / batch_count:.2f} ms, "  # noqa
                )
                summary_f.write(
                    f"output {compute_output_time_ms:<9.2f} ms, avg {compute_output_time_ms / batch_count:.2f} ms \n"  # noqa
                )


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--server-addr",
        type=str,
        default="localhost",
        help="Address of the server",
    )

    parser.add_argument(
        "--server-port",
        type=int,
        default=8001,
        help="Grpc port of the triton server, default is 8001",
    )

    parser.add_argument(
        "--reference-audio",
        type=str,
        default=None,
        help="Path to a single audio file. It can't be specified at the same time with --manifest-dir",
    )

    parser.add_argument(
        "--reference-text",
        type=str,
        default="",
        help="",
    )

    parser.add_argument(
        "--target-text",
        type=str,
        default="",
        help="",
    )

    parser.add_argument(
        "--huggingface-dataset",
        type=str,
        default="yuekai/seed_tts",
        help="dataset name in huggingface dataset hub",
    )

    parser.add_argument(
        "--split-name",
        type=str,
        default="wenetspeech4tts",
        choices=["wenetspeech4tts", "test_zh", "test_en", "test_hard"],
        help="dataset split name, default is 'test'",
    )

    parser.add_argument(
        "--manifest-path",
        type=str,
        default=None,
        help="Path to the manifest dir which includes wav.scp trans.txt files.",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="f5_tts",
        choices=[
            "f5_tts",
            "spark_tts",
            "cosyvoice2"],
        help="triton model_repo module name to request",
    )

    parser.add_argument(
        "--num-tasks",
        type=int,
        default=1,
        help="Number of concurrent tasks for sending",
    )

    parser.add_argument(
        "--log-interval",
        type=int,
        default=5,
        help="Controls how frequently we print the log.",
    )

    parser.add_argument(
        "--compute-wer",
        action="store_true",
        default=False,
        help="""True to compute WER.
        """,
    )

    parser.add_argument(
        "--log-dir",
        type=str,
        required=False,
        default="./tmp",
        help="log directory",
    )

    # --- Added arguments ---
    parser.add_argument(
        "--mode",
        type=str,
        default="offline",
        choices=["offline", "streaming"],
        help="Select offline or streaming benchmark mode."
    )
    parser.add_argument(
        "--chunk-overlap-duration",
        type=float,
        default=0.1,
        help="Chunk overlap duration for streaming reconstruction (in seconds)."
    )

    parser.add_argument(
        "--use-spk2info-cache",
        type=bool,
        default=False,
        help="Use spk2info cache for reference audio.",
    )

    return parser.parse_args()


def load_audio(wav_path, target_sample_rate=16000):
    assert target_sample_rate == 16000, "hard coding in server"
    if isinstance(wav_path, dict):
        waveform = wav_path["array"]
        sample_rate = wav_path["sampling_rate"]
    else:
        waveform, sample_rate = sf.read(wav_path)
    if sample_rate != target_sample_rate:
        from scipy.signal import resample

        num_samples = int(len(waveform) * (target_sample_rate / sample_rate))
        waveform = resample(waveform, num_samples)
    return waveform, target_sample_rate


def prepare_request_input_output(
    protocol_client,  # Can be grpcclient_aio or grpcclient_sync
    waveform,
    reference_text,
    target_text,
    sample_rate=16000,
    padding_duration: int = None,  # Optional padding for offline mode
    use_spk2info_cache: bool = False
):
    """Prepares inputs for Triton inference (offline or streaming)."""
    assert len(waveform.shape) == 1, "waveform should be 1D"
    lengths = np.array([[len(waveform)]], dtype=np.int32)

    # Apply padding only if padding_duration is provided (for offline)
    if padding_duration:
        duration = len(waveform) / sample_rate
        # Estimate target duration based on text length ratio (crude estimation)
        # Avoid division by zero if reference_text is empty
        if reference_text:
            estimated_target_duration = duration / len(reference_text) * len(target_text)
        else:
            estimated_target_duration = duration  # Assume target duration similar to reference if no text

        # Calculate required samples based on estimated total duration
        required_total_samples = padding_duration * sample_rate * (
            (int(estimated_target_duration + duration) // padding_duration) + 1
        )
        samples = np.zeros((1, required_total_samples), dtype=np.float32)
        samples[0, : len(waveform)] = waveform
    else:
        # No padding for streaming or if padding_duration is None
        samples = waveform.reshape(1, -1).astype(np.float32)

    # Common input creation logic
    inputs = [
        protocol_client.InferInput("reference_wav", samples.shape, np_to_triton_dtype(samples.dtype)),
        protocol_client.InferInput(
            "reference_wav_len", lengths.shape, np_to_triton_dtype(lengths.dtype)
        ),
        protocol_client.InferInput("reference_text", [1, 1], "BYTES"),
        protocol_client.InferInput("target_text", [1, 1], "BYTES"),
    ]
    inputs[0].set_data_from_numpy(samples)
    inputs[1].set_data_from_numpy(lengths)

    input_data_numpy = np.array([reference_text], dtype=object)
    input_data_numpy = input_data_numpy.reshape((1, 1))
    inputs[2].set_data_from_numpy(input_data_numpy)

    input_data_numpy = np.array([target_text], dtype=object)
    input_data_numpy = input_data_numpy.reshape((1, 1))
    inputs[3].set_data_from_numpy(input_data_numpy)

    outputs = [protocol_client.InferRequestedOutput("waveform")]
    if use_spk2info_cache:
        inputs = inputs[-1:]
    return inputs, outputs


def run_sync_streaming_inference(
    sync_triton_client: tritonclient.grpc.InferenceServerClient,
    model_name: str,
    inputs: list,
    outputs: list,
    request_id: str,
    user_data: UserData,
    chunk_overlap_duration: float,
    save_sample_rate: int,
    audio_save_path: str,
):
    """Helper function to run the blocking sync streaming call."""
    start_time_total = time.time()
    user_data.record_start_time()  # Record start time for first chunk latency calculation

    # Establish stream
    sync_triton_client.start_stream(callback=functools.partial(callback, user_data))

    # Send request
    sync_triton_client.async_stream_infer(
        model_name,
        inputs,
        request_id=request_id,
        outputs=outputs,
        enable_empty_final_response=True,
    )

    # Process results
    audios = []
    while True:
        try:
            result = user_data._completed_requests.get()  # Add timeout
            if isinstance(result, InferenceServerException):
                print(f"Received InferenceServerException: {result}")
                sync_triton_client.stop_stream()
                return None, None, None  # Indicate error
            # Get response metadata
            response = result.get_response()
            final = response.parameters["triton_final_response"].bool_param
            if final is True:
                break

            audio_chunk = result.as_numpy("waveform").reshape(-1)
            if audio_chunk.size > 0:  # Only append non-empty chunks
                audios.append(audio_chunk)
            else:
                print("Warning: received empty audio chunk.")

        except queue.Empty:
            print(f"Timeout waiting for response for request id {request_id}")
            sync_triton_client.stop_stream()
            return None, None, None  # Indicate error

    sync_triton_client.stop_stream()
    end_time_total = time.time()
    total_request_latency = end_time_total - start_time_total
    first_chunk_latency = user_data.get_first_chunk_latency()

    # Reconstruct audio using cross-fade (from client_grpc_streaming.py)
    actual_duration = 0
    if audios:
        # Only spark_tts model uses cross-fade
        if model_name == "spark_tts":
            cross_fade_samples = int(chunk_overlap_duration * save_sample_rate)
            fade_out = np.linspace(1, 0, cross_fade_samples)
            fade_in = np.linspace(0, 1, cross_fade_samples)
            reconstructed_audio = None

            # Simplified reconstruction based on client_grpc_streaming.py
            if not audios:
                print("Warning: No audio chunks received.")
                reconstructed_audio = np.array([], dtype=np.float32)  # Empty array
            elif len(audios) == 1:
                reconstructed_audio = audios[0]
            else:
                reconstructed_audio = audios[0][:-cross_fade_samples]  # Start with first chunk minus overlap
                for i in range(1, len(audios)):
                    # Cross-fade section
                    cross_faded_overlap = (audios[i][:cross_fade_samples] * fade_in +
                                           audios[i - 1][-cross_fade_samples:] * fade_out)
                    # Middle section of the current chunk
                    middle_part = audios[i][cross_fade_samples:-cross_fade_samples]
                    # Concatenate
                    reconstructed_audio = np.concatenate([reconstructed_audio, cross_faded_overlap, middle_part])
                # Add the last part of the final chunk
                reconstructed_audio = np.concatenate([reconstructed_audio, audios[-1][-cross_fade_samples:]])

            if reconstructed_audio is not None and reconstructed_audio.size > 0:
                actual_duration = len(reconstructed_audio) / save_sample_rate
                # Save reconstructed audio
                sf.write(audio_save_path, reconstructed_audio, save_sample_rate, "PCM_16")
            else:
                print("Warning: No audio chunks received or reconstructed.")
                actual_duration = 0  # Set duration to 0 if no audio
        else:
            reconstructed_audio = np.concatenate(audios)
            print(f"reconstructed_audio: {reconstructed_audio.shape}")
            actual_duration = len(reconstructed_audio) / save_sample_rate
            # Save reconstructed audio
            sf.write(audio_save_path, reconstructed_audio, save_sample_rate, "PCM_16")

    else:
        print("Warning: No audio chunks received.")
        actual_duration = 0

    return total_request_latency, first_chunk_latency, actual_duration


async def send_streaming(
    manifest_item_list: list,
    name: str,
    server_url: str,  # Changed from sync_triton_client
    protocol_client: types.ModuleType,
    log_interval: int,
    model_name: str,
    audio_save_dir: str = "./",
    save_sample_rate: int = 16000,
    chunk_overlap_duration: float = 0.1,
    padding_duration: int = None,
    use_spk2info_cache: bool = False,
):
    total_duration = 0.0
    latency_data = []
    task_id = int(name[5:])
    sync_triton_client = None  # Initialize client variable

    try:  # Wrap in try...finally to ensure client closing
        print(f"{name}: Initializing sync client for streaming...")
        sync_triton_client = grpcclient_sync.InferenceServerClient(url=server_url, verbose=False)  # Create client here

        print(f"{name}: Starting streaming processing for {len(manifest_item_list)} items.")
        for i, item in enumerate(manifest_item_list):
            if i % log_interval == 0:
                print(f"{name}: Processing item {i}/{len(manifest_item_list)}")

            try:
                waveform, sample_rate = load_audio(item["audio_filepath"], target_sample_rate=16000)
                reference_text, target_text = item["reference_text"], item["target_text"]

                inputs, outputs = prepare_request_input_output(
                    protocol_client,
                    waveform,
                    reference_text,
                    target_text,
                    sample_rate,
                    padding_duration=padding_duration,
                    use_spk2info_cache=use_spk2info_cache
                )
                request_id = str(uuid.uuid4())
                user_data = UserData()

                audio_save_path = os.path.join(audio_save_dir, f"{item['target_audio_path']}.wav")

                total_request_latency, first_chunk_latency, actual_duration = await asyncio.to_thread(
                    run_sync_streaming_inference,
                    sync_triton_client,
                    model_name,
                    inputs,
                    outputs,
                    request_id,
                    user_data,
                    chunk_overlap_duration,
                    save_sample_rate,
                    audio_save_path
                )

                if total_request_latency is not None:
                    print(f"{name}: Item {i} - First Chunk Latency: {first_chunk_latency:.4f}s, Total Latency: {total_request_latency:.4f}s, Duration: {actual_duration:.4f}s")
                    latency_data.append((total_request_latency, first_chunk_latency, actual_duration))
                    total_duration += actual_duration
                else:
                    print(f"{name}: Item {i} failed.")

            except FileNotFoundError:
                print(f"Error: Audio file not found for item {i}: {item['audio_filepath']}")
            except Exception as e:
                print(f"Error processing item {i} ({item['target_audio_path']}): {e}")
                import traceback
                traceback.print_exc()

    finally:  # Ensure client is closed
        if sync_triton_client:
            try:
                print(f"{name}: Closing sync client...")
                sync_triton_client.close()
            except Exception as e:
                print(f"{name}: Error closing sync client: {e}")

    print(f"{name}: Finished streaming processing. Total duration synthesized: {total_duration:.4f}s")
    return total_duration, latency_data


async def send(
    manifest_item_list: list,
    name: str,
    triton_client: tritonclient.grpc.aio.InferenceServerClient,
    protocol_client: types.ModuleType,
    log_interval: int,
    model_name: str,
    padding_duration: int = None,
    audio_save_dir: str = "./",
    save_sample_rate: int = 16000,
    use_spk2info_cache: bool = False,
):
    total_duration = 0.0
    latency_data = []
    task_id = int(name[5:])

    print(f"manifest_item_list: {manifest_item_list}")
    for i, item in enumerate(manifest_item_list):
        if i % log_interval == 0:
            print(f"{name}: {i}/{len(manifest_item_list)}")
        waveform, sample_rate = load_audio(item["audio_filepath"], target_sample_rate=16000)
        reference_text, target_text = item["reference_text"], item["target_text"]

        inputs, outputs = prepare_request_input_output(
            protocol_client,
            waveform,
            reference_text,
            target_text,
            sample_rate,
            padding_duration=padding_duration,
            use_spk2info_cache=use_spk2info_cache
        )
        sequence_id = 100000000 + i + task_id * 10
        start = time.time()
        response = await triton_client.infer(model_name, inputs, request_id=str(sequence_id), outputs=outputs)

        audio = response.as_numpy("waveform").reshape(-1)
        actual_duration = len(audio) / save_sample_rate

        end = time.time() - start

        audio_save_path = os.path.join(audio_save_dir, f"{item['target_audio_path']}.wav")
        sf.write(audio_save_path, audio, save_sample_rate, "PCM_16")

        latency_data.append((end, actual_duration))
        total_duration += actual_duration

    return total_duration, latency_data


def load_manifests(manifest_path):
    with open(manifest_path, "r") as f:
        manifest_list = []
        for line in f:
            assert len(line.strip().split("|")) == 4
            utt, prompt_text, prompt_wav, gt_text = line.strip().split("|")
            utt = Path(utt).stem
            # gt_wav = os.path.join(os.path.dirname(manifest_path), "wavs", utt + ".wav")
            if not os.path.isabs(prompt_wav):
                prompt_wav = os.path.join(os.path.dirname(manifest_path), prompt_wav)
            manifest_list.append(
                {
                    "audio_filepath": prompt_wav,
                    "reference_text": prompt_text,
                    "target_text": gt_text,
                    "target_audio_path": utt,
                }
            )
    return manifest_list


def split_data(data, k):
    n = len(data)
    if n < k:
        print(f"Warning: the length of the input list ({n}) is less than k ({k}). Setting k to {n}.")
        k = n

    quotient = n // k
    remainder = n % k

    result = []
    start = 0
    for i in range(k):
        if i < remainder:
            end = start + quotient + 1
        else:
            end = start + quotient

        result.append(data[start:end])
        start = end

    return result


async def main():
    args = get_args()
    url = f"{args.server_addr}:{args.server_port}"

    # --- Client Initialization based on mode ---
    triton_client = None
    protocol_client = None
    if args.mode == "offline":
        print("Initializing gRPC client for offline mode...")
        # Use the async client for offline tasks
        triton_client = grpcclient_aio.InferenceServerClient(url=url, verbose=False)
        protocol_client = grpcclient_aio
    elif args.mode == "streaming":
        print("Initializing gRPC client for streaming mode...")
        # Use the sync client for streaming tasks, handled via asyncio.to_thread
        # We will create one sync client instance PER TASK inside send_streaming.
        # triton_client = grpcclient_sync.InferenceServerClient(url=url, verbose=False) # REMOVED: Client created per task now
        protocol_client = grpcclient_sync  # protocol client for input prep
    else:
        raise ValueError(f"Invalid mode: {args.mode}")
    # --- End Client Initialization ---

    if args.reference_audio:
        args.num_tasks = 1
        args.log_interval = 1
        manifest_item_list = [
            {
                "reference_text": args.reference_text,
                "target_text": args.target_text,
                "audio_filepath": args.reference_audio,
                "target_audio_path": "test",
            }
        ]
    elif args.huggingface_dataset:
        import datasets

        dataset = datasets.load_dataset(
            args.huggingface_dataset,
            split=args.split_name,
            trust_remote_code=True,
        )
        manifest_item_list = []
        for i in range(len(dataset)):
            manifest_item_list.append(
                {
                    "audio_filepath": dataset[i]["prompt_audio"],
                    "reference_text": dataset[i]["prompt_text"],
                    "target_audio_path": dataset[i]["id"],
                    "target_text": dataset[i]["target_text"],
                }
            )
    else:
        manifest_item_list = load_manifests(args.manifest_path)

    num_tasks = min(args.num_tasks, len(manifest_item_list))
    manifest_item_list = split_data(manifest_item_list, num_tasks)

    os.makedirs(args.log_dir, exist_ok=True)

    tasks = []
    start_time = time.time()
    for i in range(num_tasks):
        # --- Task Creation based on mode ---
        if args.mode == "offline":
            task = asyncio.create_task(
                send(
                    manifest_item_list[i],
                    name=f"task-{i}",
                    triton_client=triton_client,
                    protocol_client=protocol_client,
                    log_interval=args.log_interval,
                    model_name=args.model_name,
                    audio_save_dir=args.log_dir,
                    padding_duration=1,
                    save_sample_rate=16000 if args.model_name == "spark_tts" else 24000,
                    use_spk2info_cache=args.use_spk2info_cache,
                )
            )
        elif args.mode == "streaming":
            task = asyncio.create_task(
                send_streaming(
                    manifest_item_list[i],
                    name=f"task-{i}",
                    server_url=url,  # Pass URL instead of client
                    protocol_client=protocol_client,
                    log_interval=args.log_interval,
                    model_name=args.model_name,
                    audio_save_dir=args.log_dir,
                    padding_duration=10,
                    save_sample_rate=16000 if args.model_name == "spark_tts" else 24000,
                    chunk_overlap_duration=args.chunk_overlap_duration,
                    use_spk2info_cache=args.use_spk2info_cache,
                )
            )
        # --- End Task Creation ---
        tasks.append(task)

    ans_list = await asyncio.gather(*tasks)

    end_time = time.time()
    elapsed = end_time - start_time

    total_duration = 0.0
    latency_data = []
    for ans in ans_list:
        if ans:
            total_duration += ans[0]
            latency_data.extend(ans[1])  # Use extend for list of lists
        else:
            print("Warning: A task returned None, possibly due to an error.")

    if total_duration == 0:
        print("Total synthesized duration is zero. Cannot calculate RTF or latency percentiles.")
        rtf = float('inf')
    else:
        rtf = elapsed / total_duration

    s = f"Mode: {args.mode}\n"
    s += f"RTF: {rtf:.4f}\n"
    s += f"total_duration: {total_duration:.3f} seconds\n"
    s += f"({total_duration / 3600:.2f} hours)\n"
    s += f"processing time: {elapsed:.3f} seconds ({elapsed / 3600:.2f} hours)\n"

    # --- Statistics Reporting based on mode ---
    if latency_data:
        if args.mode == "offline":
            # Original offline latency calculation
            latency_list = [chunk_end for (chunk_end, chunk_duration) in latency_data]
            if latency_list:
                latency_ms = sum(latency_list) / float(len(latency_list)) * 1000.0
                latency_variance = np.var(latency_list, dtype=np.float64) * 1000.0
                s += f"latency_variance: {latency_variance:.2f}\n"
                s += f"latency_50_percentile_ms: {np.percentile(latency_list, 50) * 1000.0:.2f}\n"
                s += f"latency_90_percentile_ms: {np.percentile(latency_list, 90) * 1000.0:.2f}\n"
                s += f"latency_95_percentile_ms: {np.percentile(latency_list, 95) * 1000.0:.2f}\n"
                s += f"latency_99_percentile_ms: {np.percentile(latency_list, 99) * 1000.0:.2f}\n"
                s += f"average_latency_ms: {latency_ms:.2f}\n"
            else:
                s += "No latency data collected for offline mode.\n"

        elif args.mode == "streaming":
            # Calculate stats for total request latency and first chunk latency
            total_latency_list = [total for (total, first, duration) in latency_data if total is not None]
            first_chunk_latency_list = [first for (total, first, duration) in latency_data if first is not None]

            s += "\n--- Total Request Latency ---\n"
            if total_latency_list:
                avg_total_latency_ms = sum(total_latency_list) / len(total_latency_list) * 1000.0
                variance_total_latency = np.var(total_latency_list, dtype=np.float64) * 1000.0
                s += f"total_request_latency_variance: {variance_total_latency:.2f}\n"
                s += f"total_request_latency_50_percentile_ms: {np.percentile(total_latency_list, 50) * 1000.0:.2f}\n"
                s += f"total_request_latency_90_percentile_ms: {np.percentile(total_latency_list, 90) * 1000.0:.2f}\n"
                s += f"total_request_latency_95_percentile_ms: {np.percentile(total_latency_list, 95) * 1000.0:.2f}\n"
                s += f"total_request_latency_99_percentile_ms: {np.percentile(total_latency_list, 99) * 1000.0:.2f}\n"
                s += f"average_total_request_latency_ms: {avg_total_latency_ms:.2f}\n"
            else:
                s += "No total request latency data collected.\n"

            s += "\n--- First Chunk Latency ---\n"
            if first_chunk_latency_list:
                avg_first_chunk_latency_ms = sum(first_chunk_latency_list) / len(first_chunk_latency_list) * 1000.0
                variance_first_chunk_latency = np.var(first_chunk_latency_list, dtype=np.float64) * 1000.0
                s += f"first_chunk_latency_variance: {variance_first_chunk_latency:.2f}\n"
                s += f"first_chunk_latency_50_percentile_ms: {np.percentile(first_chunk_latency_list, 50) * 1000.0:.2f}\n"
                s += f"first_chunk_latency_90_percentile_ms: {np.percentile(first_chunk_latency_list, 90) * 1000.0:.2f}\n"
                s += f"first_chunk_latency_95_percentile_ms: {np.percentile(first_chunk_latency_list, 95) * 1000.0:.2f}\n"
                s += f"first_chunk_latency_99_percentile_ms: {np.percentile(first_chunk_latency_list, 99) * 1000.0:.2f}\n"
                s += f"average_first_chunk_latency_ms: {avg_first_chunk_latency_ms:.2f}\n"
            else:
                s += "No first chunk latency data collected (check for errors or if all requests failed before first chunk).\n"
    else:
        s += "No latency data collected.\n"
    # --- End Statistics Reporting ---

    print(s)
    if args.manifest_path:
        name = Path(args.manifest_path).stem
    elif args.split_name:
        name = args.split_name
    elif args.reference_audio:
        name = Path(args.reference_audio).stem
    else:
        name = "results"  # Default name if no manifest/split/audio provided
    with open(f"{args.log_dir}/rtf-{name}.txt", "w") as f:
        f.write(s)

    # --- Statistics Fetching using temporary Async Client ---
    # Use a separate async client for fetching stats regardless of mode
    stats_client = None
    try:
        print("Initializing temporary async client for fetching stats...")
        stats_client = grpcclient_aio.InferenceServerClient(url=url, verbose=False)
        print("Fetching inference statistics...")
        # Fetching for all models, filtering might be needed depending on server setup
        stats = await stats_client.get_inference_statistics(model_name="", as_json=True)
        print("Fetching model config...")
        metadata = await stats_client.get_model_config(model_name=args.model_name, as_json=True)

        write_triton_stats(stats, f"{args.log_dir}/stats_summary-{name}.txt")

        with open(f"{args.log_dir}/model_config-{name}.json", "w") as f:
            json.dump(metadata, f, indent=4)

    except Exception as e:
        print(f"Could not retrieve statistics or config: {e}")
    finally:
        if stats_client:
            try:
                print("Closing temporary async stats client...")
                await stats_client.close()
            except Exception as e:
                print(f"Error closing async stats client: {e}")
    # --- End Statistics Fetching ---


if __name__ == "__main__":
    # asyncio.run(main()) # Use TaskGroup for better exception handling if needed
    async def run_main():
        try:
            await main()
        except Exception as e:
            print(f"An error occurred in main: {e}")
            import traceback
            traceback.print_exc()

    asyncio.run(run_main())
