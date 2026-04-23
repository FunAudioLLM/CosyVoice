"""Find the exact (T_x_post_conv_pre, T_stft) relationship for several T_mel
inputs, so we can build a TRT optimization profile that satisfies the
internal-Add shape constraints."""
import sys, torch
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import AutoModel


def main():
    auto = AutoModel(model_dir='pretrained_models/Fun-CosyVoice3-0.5B',
                     load_trt=False, load_vllm=False, fp16=False)
    hift = auto.model.hift
    device = next(hift.parameters()).device
    hift.f0_predictor.to(torch.float64)

    print('T_mel | T_x_post_conv_pre | T_stft')
    pairs = []
    for T_mel in [25, 50, 80, 100, 150, 200, 300, 500]:
        mel = torch.randn(1, 80, T_mel, device=device)
        f0 = hift.f0_predictor(mel.to(torch.float64), finalize=True).to(mel)
        s = hift.f0_upsamp(f0[:, None]).transpose(1, 2)
        s, _, _ = hift.m_source(s)
        s = s.transpose(1, 2)
        x = hift.conv_pre(mel)
        s_real, s_imag = hift._stft(s.squeeze(1))
        s_stft = torch.cat([s_real, s_imag], dim=1)
        print(f'{T_mel:>5} | {x.shape[2]:>17} | {s_stft.shape[2]:>6}')
        pairs.append((T_mel, x.shape[2], s_stft.shape[2]))

    # Fit linear: T_stft = a * T_x + b
    import statistics
    if len(pairs) >= 2:
        xs = [p[1] for p in pairs]
        ys = [p[2] for p in pairs]
        slope = (ys[-1] - ys[0]) / (xs[-1] - xs[0])
        intercept = ys[0] - slope * xs[0]
        print(f'\nFit:  T_stft = {slope:.4f} * T_x + {intercept:.4f}')


if __name__ == '__main__':
    main()
