"""Dump ONNX node names + op types, focus on Snake-related ops, to find a
narrower keyword set than 'sin,pow,reciprocal,div' for the TRT fp32 override."""
import sys, onnx
from collections import Counter

p = sys.argv[1] if len(sys.argv) > 1 else 'pretrained_models/Fun-CosyVoice3-0.5B/hift.decoder.fp32.onnx'
m = onnx.load(p)
g = m.graph

print(f'graph: {len(g.node)} nodes')
op_counts = Counter(n.op_type for n in g.node)
print('\nop type counts (top 20):')
for op, c in op_counts.most_common(20):
    print(f'  {op:>20} : {c}')

# Snake decomposes to: Mul (alpha*x) -> Sin -> Pow(2) -> Add(alpha+eps) -> Reciprocal -> Mul -> Add(x+...)
# Look at node names containing 'Snake' or 'activation' (PyTorch module names)
print('\nnodes with "Snake" or "activation" in name (first 40):')
relevant = [n for n in g.node if 'snake' in n.name.lower() or 'activation' in n.name.lower()]
for n in relevant[:40]:
    inputs = [i for i in n.input if not i.startswith('onnx::')][:2]
    print(f'  {n.op_type:>15}  {n.name}  inputs={inputs}')

# Show a sample of Reciprocal nodes
print('\nall Reciprocal nodes:')
for n in g.node:
    if n.op_type == 'Reciprocal':
        print(f'  {n.name}  input={list(n.input)[:1]}')

# All ops with "alpha" related inputs (initializers contain 'alpha')
alpha_initializers = {init.name for init in g.initializer if 'alpha' in init.name.lower()}
print(f'\n{len(alpha_initializers)} initializers with "alpha" in name (first 5):')
for n in list(alpha_initializers)[:5]:
    print(f'  {n}')

# Find ops whose inputs reference an alpha initializer (these are the Snake math ops)
print('\nfirst 5 nodes whose input references alpha:')
alpha_consumers = []
for n in g.node:
    if any(i in alpha_initializers for i in n.input):
        alpha_consumers.append(n)
for n in alpha_consumers[:5]:
    print(f'  {n.op_type:>15}  {n.name}')
print(f'... total alpha consumers: {len(alpha_consumers)}')
