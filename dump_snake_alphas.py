"""Dump trained Snake alpha values from hift.pt to see if 1/alpha actually
overflows fp16 (max 65504). If all alphas >= 1.5e-5, fp16 overflow can't be
the bug, and R6's saturated audio has a different cause."""
import sys, torch
import numpy as np

ckpt = sys.argv[1] if len(sys.argv) > 1 else 'pretrained_models/Fun-CosyVoice3-0.5B/hift.pt'
sd = torch.load(ckpt, map_location='cpu', weights_only=True)
print(f'loaded {ckpt}: {len(sd)} keys')

alpha_keys = [k for k in sd.keys() if k.endswith('.alpha')]
print(f'\nfound {len(alpha_keys)} Snake alpha tensors\n')

all_vals = []
problem_count = 0
for k in alpha_keys[:10]:  # sample first 10
    a = sd[k].abs()
    inv_a_max = (1.0 / (a + 1e-9)).max().item()
    inv_a_min = (1.0 / (a + 1e-9)).min().item()
    print(f'  {k:60s}  shape={tuple(a.shape)}  alpha [min,max]=[{a.min().item():.4e}, {a.max().item():.4e}]  1/alpha max={inv_a_max:.2e}')
    all_vals.append(a.flatten())

all_vals = torch.cat([sd[k].abs().flatten() for k in alpha_keys])
inv_all = 1.0 / (all_vals + 1e-9)
print(f'\n=== overall stats over {len(all_vals)} alpha values ===')
print(f'  alpha min={all_vals.min().item():.4e}  max={all_vals.max().item():.4e}  mean={all_vals.mean().item():.4e}')
print(f'  1/alpha max={inv_all.max().item():.4e}  fp16 max=65504')
unsafe = (inv_all > 65504).sum().item()
print(f'  values where 1/alpha > 65504 (fp16 overflow): {unsafe} / {len(all_vals)}')
print(f'  values where 1/alpha > 6500  (close to limit): {(inv_all > 6500).sum().item()}')
