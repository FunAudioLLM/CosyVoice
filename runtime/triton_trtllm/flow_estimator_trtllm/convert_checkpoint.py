"""
Convert CosyVoice PyTorch checkpoint to TensorRT-LLM format
"""

import argparse
import json
import os
import torch
import safetensors.torch
from tensorrt_llm import str_dtype_to_torch


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pytorch_ckpt',
                        type=str,
                        required=True,
                        help='Path to PyTorch checkpoint (.pt or .pth)')
    parser.add_argument('--output_dir',
                        type=str,
                        default='tllm_checkpoint',
                        help='Output directory for TensorRT-LLM checkpoint')
    parser.add_argument('--dtype',
                        type=str,
                        default='float16',
                        choices=['float32', 'float16', 'bfloat16'])
    parser.add_argument('--hidden_size', type=int, default=1024)
    parser.add_argument('--mel_dim', type=int, default=80)
    parser.add_argument('--spk_dim', type=int, default=80)
    parser.add_argument('--num_blocks', type=int, default=22,
                        help='Number of DiT blocks to convert (default: 22)')
    return parser.parse_args()


def get_embedding_weight_mapping():
    """Embedding weights mapping"""
    return {
        # TimestepEmbedding
        'time_embed.time_mlp.0.weight': 'time_embed.time_mlp.weight',
        'time_embed.time_mlp.0.bias': 'time_embed.time_mlp.bias',
        'time_embed.time_mlp.2.weight': 'time_embed.time_mlp2.weight',
        'time_embed.time_mlp.2.bias': 'time_embed.time_mlp2.bias',

        # InputEmbedding - projection layer
        'input_embed.proj.weight': 'input_embed.proj.weight',
        'input_embed.proj.bias': 'input_embed.proj.bias',

        # InputEmbedding - CausalConvPositionEmbedding
        'input_embed.conv_pos_embed.conv1.0.weight': 'input_embed.conv_pos_embed.conv1.weight',
        'input_embed.conv_pos_embed.conv1.0.bias': 'input_embed.conv_pos_embed.conv1.bias',
        'input_embed.conv_pos_embed.conv2.0.weight': 'input_embed.conv_pos_embed.conv2.weight',
        'input_embed.conv_pos_embed.conv2.0.bias': 'input_embed.conv_pos_embed.conv2.bias',
    }


def get_block_weight_mapping(block_idx):
    """
    Get weight mapping for a single DiTBlock

    PyTorch → TensorRT-LLM mapping for transformer_blocks[block_idx]
    """
    pt_prefix = f'transformer_blocks.{block_idx}'
    trt_prefix = f'transformer_blocks.{block_idx}'  # Keep same index in Phase 3

    mapping = {
        # AdaLayerNorm modulation (6 * hidden_size outputs)
        f'{pt_prefix}.attn_norm.linear.weight': f'{trt_prefix}.attn_norm_modulation.weight',
        f'{pt_prefix}.attn_norm.linear.bias': f'{trt_prefix}.attn_norm_modulation.bias',

        # Attention: Q, K, V need to be concatenated
        # Will be handled separately in convert_weights()

        # Attention output projection
        f'{pt_prefix}.attn.to_out.0.weight': f'{trt_prefix}.attn.dense.weight',
        f'{pt_prefix}.attn.to_out.0.bias': f'{trt_prefix}.attn.dense.bias',

        # Feed-Forward
        f'{pt_prefix}.ff.ff.0.0.weight': f'{trt_prefix}.ff.fc.weight',
        f'{pt_prefix}.ff.ff.0.0.bias': f'{trt_prefix}.ff.fc.bias',
        f'{pt_prefix}.ff.ff.2.weight': f'{trt_prefix}.ff.proj.weight',
        f'{pt_prefix}.ff.ff.2.bias': f'{trt_prefix}.ff.proj.bias',
    }

    return mapping


def get_final_layer_mapping():
    """Get weight mapping for FinalLayer"""
    return {
        # AdaLayerNormZero_Final modulation (2 * hidden_size outputs)
        'norm_out.linear.weight': 'final_layer.norm_out_modulation.weight',
        'norm_out.linear.bias': 'final_layer.norm_out_modulation.bias',

        # Output projection
        'proj_out.weight': 'final_layer.proj_out.weight',
        'proj_out.bias': 'final_layer.proj_out.bias',
    }


def convert_weights(pytorch_ckpt_path, dtype='float16'):
    """
    Convert PyTorch weights to TensorRT-LLM format

    Args:
        pytorch_ckpt_path: Path to PyTorch checkpoint
        dtype: Target dtype for weights
    Returns:
        Dictionary of converted weights
    """
    print(f"Loading PyTorch checkpoint from: {pytorch_ckpt_path}")

    # Load PyTorch checkpoint, full flow model weights
    pytorch_weights = torch.load(pytorch_ckpt_path, map_location='cpu')

    # get estimator weights only
    estimator_keys = [k for k in pytorch_weights if 'decoder.estimator' in k]
    # remove the first 18 chars (decoder.estimator)
    estimator_weights = {k[18:]: pytorch_weights[k] for k in estimator_keys}


    # Convert weights
    trt_weights = {}
    torch_dtype = str_dtype_to_torch(dtype)

    # ========== Convert Embeddings ==========
    print("\n=== Converting Embedding Weights ===")
    embedding_mapping = get_embedding_weight_mapping()

    for pt_name, trt_name in embedding_mapping.items():
        if pt_name in estimator_weights:
            weight = estimator_weights[pt_name].to(torch_dtype)

            # Handle Conv1d weights: add trailing dimension
            if 'conv' in pt_name and 'weight' in pt_name and weight.ndim == 3:
                weight = weight.unsqueeze(-1)
                print(f"  ✓ {pt_name:60s} -> {trt_name:60s} {tuple(weight.shape)} (Conv1d)")
            else:
                print(f"  ✓ {pt_name:60s} -> {trt_name:60s} {tuple(weight.shape)}")

            trt_weights[trt_name] = weight.contiguous()
        else:
            print(f"  ✗ Missing: {pt_name}")

    # ========== Convert ALL Transformer Blocks ==========
    print(f"\n=== Converting all DiTBlocks ===")

    for block_idx in range(22):
        print(f"\n--- Block {block_idx} ---")
        block_mapping = get_block_weight_mapping(block_idx)

        pt_prefix = f'transformer_blocks.{block_idx}'
        trt_prefix = f'transformer_blocks.{block_idx}'

        # Handle QKV concatenation
        q_weight_name = f'{pt_prefix}.attn.to_q.weight'
        k_weight_name = f'{pt_prefix}.attn.to_k.weight'
        v_weight_name = f'{pt_prefix}.attn.to_v.weight'

        if all(name in estimator_weights for name in [q_weight_name, k_weight_name, v_weight_name]):
            # Concatenate Q, K, V weights
            q_weight = estimator_weights[q_weight_name].to(torch_dtype)
            k_weight = estimator_weights[k_weight_name].to(torch_dtype)
            v_weight = estimator_weights[v_weight_name].to(torch_dtype)

            qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
            trt_weights[f'{trt_prefix}.attn.qkv.weight'] = qkv_weight.contiguous()

            print(f"  ✓ QKV weights concatenated -> {trt_prefix}.attn.qkv.weight {tuple(qkv_weight.shape)}")

            # Concatenate Q, K, V biases
            q_bias_name = f'{pt_prefix}.attn.to_q.bias'
            k_bias_name = f'{pt_prefix}.attn.to_k.bias'
            v_bias_name = f'{pt_prefix}.attn.to_v.bias'

            if all(name in estimator_weights for name in [q_bias_name, k_bias_name, v_bias_name]):
                q_bias = estimator_weights[q_bias_name].to(torch_dtype)
                k_bias = estimator_weights[k_bias_name].to(torch_dtype)
                v_bias = estimator_weights[v_bias_name].to(torch_dtype)

                qkv_bias = torch.cat([q_bias, k_bias, v_bias], dim=0)
                trt_weights[f'{trt_prefix}.attn.qkv.bias'] = qkv_bias.contiguous()

                print(f"  ✓ QKV biases concatenated -> {trt_prefix}.attn.qkv.bias {tuple(qkv_bias.shape)}")
        else:
            print(f"  ✗ Missing Q/K/V weights for block {block_idx}")

        # Convert other block weights
        for pt_name, trt_name in block_mapping.items():
            if pt_name in estimator_weights:
                weight = estimator_weights[pt_name].to(torch_dtype)
                trt_weights[trt_name] = weight.contiguous()
                print(f"  ✓ {pt_name:60s} -> {trt_name:60s} {tuple(weight.shape)}")
            else:
                print(f"  ✗ Missing: {pt_name}")

    # ========== Convert FinalLayer ==========
    print("\n=== Converting FinalLayer Weights ===")
    final_mapping = get_final_layer_mapping()

    for pt_name, trt_name in final_mapping.items():
        if pt_name in estimator_weights:
            weight = estimator_weights[pt_name].to(torch_dtype)
            trt_weights[trt_name] = weight.contiguous()
            print(f"  ✓ {pt_name:60s} -> {trt_name:60s} {tuple(weight.shape)}")
        else:
            print(f"  ✗ Missing: {pt_name}")

    print(f"\n✅ Converted {len(trt_weights)} weights total")

    return trt_weights


def save_config(args):
    """Save TensorRT-LLM config.json"""
    config = {
        'architecture': 'DiT',
        'dtype': args.dtype,
        'hidden_size': 1024,
        'mel_dim': 80,
        'mu_dim': 80,
        'spk_dim': 80,
        'num_hidden_layers': 22,
        'num_attention_heads': 16,
        'ff_mult': 2,
        'max_position_embeddings': 1000,
        'mapping': {
            'world_size': 1,
            'tp_size': 1,
            'cp_size': 1,
            'pp_size': 1,
        }
    }

    os.makedirs(args.output_dir, exist_ok=True)
    config_path = os.path.join(args.output_dir, 'config.json')

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n💾 Saved config to: {config_path}")
    return config


def main():
    args = parse_arguments()

    print("="*80)
    print("CosyVoice PyTorch -> TensorRT-LLM Checkpoint Conversion")
    print("="*80)
    print(f"  Input:  {args.pytorch_ckpt}")
    print(f"  Output: {args.output_dir}")
    print(f"  Dtype:  {args.dtype}")

    # Save config
    config = save_config(args)

    # Convert weights
    trt_weights = convert_weights(args.pytorch_ckpt, args.dtype)

    # Save weights as safetensors
    weights_path = os.path.join(args.output_dir, 'rank0.safetensors')
    safetensors.torch.save_file(trt_weights, weights_path)
    print(f"💾 Saved weights to: {weights_path}")

    print("\n" + "="*80)
    print("✅ Conversion complete!")
    print("="*80)
    print(f"\nCheckpoint saved to: {args.output_dir}/")
    print(f"  - config.json")
    print(f"  - rank0.safetensors")


if __name__ == '__main__':
    main()
