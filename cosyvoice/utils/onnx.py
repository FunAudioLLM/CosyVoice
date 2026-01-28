import onnxruntime
import torch, random
from torch import nn
import os
import whisper
import numpy as np
import torchaudio.compliance.kaldi as kaldi
import torch.nn.functional as F


class SpeechTokenExtractor():
    def __init__(self, model_path):
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        option = onnxruntime.SessionOptions()
        option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        option.intra_op_num_threads = 1
        self.speech_tokenizer_session = onnxruntime.InferenceSession(model_path,
                                                                     sess_options=option,
                                                                     providers=[("CUDAExecutionProvider", {'device_id': self.local_rank})])

    def inference(self, feat, feat_lengths, device):
        ort_out = self.speech_tokenizer_session.run(None,
                                                    {self.speech_tokenizer_session.get_inputs()[0].name:
                                                    feat.detach().cpu().numpy(),
                                                    self.speech_tokenizer_session.get_inputs()[1].name:
                                                    feat_lengths.detach().cpu().numpy()})
        speech_token, speech_token_embedding = ort_out[0], ort_out[1]
        return torch.tensor(speech_token).to(device), (feat_lengths / 2).to(torch.int32).to(device)


class EmbeddingExtractor():
    def __init__(self, model_path):
        option = onnxruntime.SessionOptions()
        option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        option.intra_op_num_threads = 1
        self.max_len = 10 * 16000
        self.campplus_session = onnxruntime.InferenceSession(model_path,
                                                             sess_options=option,
                                                             providers=["CPUExecutionProvider"])

    def inference(self, speech):
        if speech.shape[1] > self.max_len:
            start_index = random.randint(0, speech.shape[1] - self.max_len)
            speech = speech[:, start_index: start_index + self.max_len]
        feat = kaldi.fbank(speech,
                           num_mel_bins=80,
                           dither=0,
                           sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)
        embedding = self.campplus_session.run(None,
                                              {self.campplus_session.get_inputs()[0].name: feat.unsqueeze(dim=0).cpu().numpy()})[0].flatten().tolist()
        return torch.tensor(embedding).to(speech.device)

# singleton mode, only initialized once
onnx_path = os.environ.get('onnx_path')
if onnx_path is not None:
    embedding_extractor, online_feature = EmbeddingExtractor(model_path=os.path.join(onnx_path, 'campplus.onnx')), True
else:
    embedding_extractor, online_feature = None, False