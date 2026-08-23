# Native LoRA adaptation

CosyVoice can adapt its language-model stage with native LoRA adapters without
adding a PEFT dependency or changing the base checkpoint.

## Training

Use the existing LLM training entry point and enable LoRA:

```bash
PYTHONPATH=.:third_party/Matcha-TTS \
python cosyvoice/bin/train.py \
  --train_engine torch_ddp \
  --config path/to/config.yaml \
  --train_data path/to/train.data.list \
  --cv_data path/to/dev.data.list \
  --model llm \
  --checkpoint path/to/llm.pt \
  --model_dir experiments/lora \
  --tensorboard_dir tensorboard/lora \
  --lora --lora_rank 16 --lora_alpha 32 --lora_dropout 0.05
```

LoRA freezes the base model, injects adapters into the attention and MLP
projections, and keeps the speech embedding and decoder heads trainable. The
optimizer receives only trainable parameters.

To continue from an adapter checkpoint, use `--lora_checkpoint`. When using
`--train_engine torch_ddp`, the adapter checkpoint contains only trainable
adapter/head weights plus `epoch` and `step`, so the original base checkpoint is
still required.

## Inference

Construct the matching base model, inject the same rank and alpha, and load the
adapter state with `cosyvoice.utils.lora.load_lora_state_dict`. The base model
remains available for fallback, comparison, or a later merge operation.
