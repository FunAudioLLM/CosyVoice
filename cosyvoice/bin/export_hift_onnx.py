# Export the conv-only path of (Causal)HiFTGenerator.decode for TRT fp16.
#
# Split point:
#   PyTorch (kept):  f0_predictor -> sine source -> STFT(s)
#                    conv_pre (causal, takes 1 or 2 args by finalize flag)
#                    iSTFT, finalize-truncate, audio_limit clamp
#   TRT (this export): leaky_relu + ups + (reflection_pad on last) + source_downs
#                      + source_resblocks + resblocks (Snake act) + conv_post
#                      + exp/sin to magnitude/phase  --  the dense GPU work
#
# Inputs to the engine: x_post_conv_pre (B, base_channels, T_x), s_stft (B, n_fft+2, T_stft)
# Outputs: magnitude (B, n_fft//2+1, T_out), phase  same shape
import argparse, os, sys, random
import torch
import torch.nn as nn
import torch.nn.functional as F
import onnxruntime
from tqdm import tqdm

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{ROOT}/../..')
sys.path.append(f'{ROOT}/../../third_party/Matcha-TTS')

from cosyvoice.cli.cosyvoice import AutoModel


def _strip_weight_norm(module: nn.Module):
    """Remove weight_norm regardless of legacy hook or new parametrize API."""
    from torch.nn.utils import remove_weight_norm as _legacy
    from torch.nn.utils.parametrize import remove_parametrizations
    for m in module.modules():
        # New parametrize style (PyTorch >=2.4)
        if hasattr(m, 'parametrizations') and 'weight' in getattr(m, 'parametrizations', {}):
            try:
                remove_parametrizations(m, 'weight', leave_parametrized=True)
                continue
            except Exception:
                pass
        # Legacy hook style
        for hook in list(getattr(m, '_forward_pre_hooks', {}).values()):
            if hook.__class__.__name__ == 'WeightNorm':
                try:
                    _legacy(m, 'weight')
                except Exception:
                    pass
                break


class HiftDecoderConvBlock(nn.Module):
    """The pure-conv post-conv_pre path of (Causal)HiFTGenerator.decode."""

    def __init__(self, hift):
        super().__init__()
        self.ups = hift.ups
        self.source_downs = hift.source_downs
        self.source_resblocks = hift.source_resblocks
        self.resblocks = hift.resblocks
        self.conv_post = hift.conv_post
        self.reflection_pad = hift.reflection_pad
        self.lrelu_slope = hift.lrelu_slope
        self.num_upsamples = hift.num_upsamples
        self.num_kernels = hift.num_kernels
        self.n_fft_half_p1 = hift.istft_params['n_fft'] // 2 + 1

    def forward(self, x: torch.Tensor, s_stft: torch.Tensor):
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, self.lrelu_slope)
            # ups[i] is CausalConv1dUpsample (CausalHiFTGenerator) or ConvTranspose1d.
            # Both can be invoked with single arg; default empty cache hits zero-pad path.
            x = self.ups[i](x)
            if i == self.num_upsamples - 1:
                x = self.reflection_pad(x)
            si = self.source_downs[i](s_stft)
            si = self.source_resblocks[i](si)
            x = x + si
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs = xs + self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        magnitude = torch.exp(x[:, :self.n_fft_half_p1, :])
        phase = torch.sin(x[:, self.n_fft_half_p1:, :])
        return magnitude, phase


def _probe_shapes(hift, device):
    # Build a dummy input by running the PyTorch path and snapshotting tensors at split points.
    # T_x = mel chunk length post conv_pre (causal pad shrinks input by causal_padding).
    # Use a representative chunk size: 25 tokens * 2 mel_ratio = 50 mel frames; conv_pre w/ pad=3 keeps T.
    dummy_mel = torch.randn(1, 80, 80, device=device, dtype=torch.float32)
    # f0 -> source -> STFT path mirrors CausalHiFTGenerator.inference (needs float64 f0 predictor)
    hift.f0_predictor.to(torch.float64)
    f0 = hift.f0_predictor(dummy_mel.to(torch.float64), finalize=True).to(dummy_mel)
    s = hift.f0_upsamp(f0[:, None]).transpose(1, 2)
    s, _, _ = hift.m_source(s)
    s = s.transpose(1, 2)
    # decode() preamble:
    x = hift.conv_pre(dummy_mel)
    s_real, s_imag = hift._stft(s.squeeze(1))
    s_stft = torch.cat([s_real, s_imag], dim=1)
    return x, s_stft


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', default='pretrained_models/Fun-CosyVoice3-0.5B')
    args = parser.parse_args()

    print(f'[export] loading {args.model_dir} ...', flush=True)
    auto = AutoModel(model_dir=args.model_dir, load_trt=False, load_vllm=False, fp16=False)
    hift = auto.model.hift
    device = next(hift.parameters()).device

    print('[export] removing weight_norm on hift (new+legacy APIs) ...', flush=True)
    _strip_weight_norm(hift)
    hift.eval()

    block = HiftDecoderConvBlock(hift).eval().to(device)

    print('[export] probing tensor shapes via PyTorch fwd ...', flush=True)
    x_dummy, s_stft_dummy = _probe_shapes(hift, device)
    print(f'         x={tuple(x_dummy.shape)}  s_stft={tuple(s_stft_dummy.shape)}', flush=True)

    onnx_path = os.path.join(args.model_dir, 'hift.decoder.fp32.onnx')
    print(f'[export] torch.onnx.export -> {onnx_path}', flush=True)
    torch.onnx.export(
        block,
        (x_dummy, s_stft_dummy),
        onnx_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['x', 's_stft'],
        output_names=['magnitude', 'phase'],
        dynamic_axes={
            'x': {2: 'T_x'},
            's_stft': {2: 'T_stft'},
            'magnitude': {2: 'T_out'},
            'phase': {2: 'T_out'},
        },
    )

    # Sanity check: run via onnxruntime and compare to PyTorch.
    print('[export] sanity check via onnxruntime CUDA EP ...', flush=True)
    sess = onnxruntime.InferenceSession(
        onnx_path,
        providers=['CUDAExecutionProvider' if torch.cuda.is_available() else 'CPUExecutionProvider'],
    )
    # Sanity-check on the actual probed shapes (the only ones for which we know
    # the exact T_stft / T_x relationship; the source-downs Conv1d ratios make
    # arbitrary T_x impossible to test with random stub tensors).
    out_pt = block(x_dummy, s_stft_dummy)
    out_ort = sess.run(None, {'x': x_dummy.cpu().numpy(), 's_stft': s_stft_dummy.cpu().numpy()})
    for name, pt, ort in zip(['magnitude', 'phase'], out_pt, out_ort):
        ort_t = torch.from_numpy(ort).to(device)
        diff = (pt - ort_t).abs()
        print(f'  ort vs torch  {name}: max_abs={diff.max().item():.3e} mean_abs={diff.mean().item():.3e} '
              f'shape={tuple(ort_t.shape)}')

    print(f'[export] done. ONNX size = {os.path.getsize(onnx_path) / 1e6:.1f} MB', flush=True)


if __name__ == '__main__':
    main()
