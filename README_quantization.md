# CosyVoice 模型量化指南

本指南提供了使用不同量化方法对CosyVoice模型进行量化的步骤。

## 准备工作

首先，您需要安装相应的量化库。我们推荐使用bitsandbytes进行量化，它的兼容性最好：

```bash
pip install bitsandbytes
```

## 量化模型

### 1. 使用 BitsAndBytes 量化 (推荐)

BitsAndBytes是一种简单易用的量化方法，适合快速尝试，兼容性最好。

```bash
python quant_cosyvoice_bnb.py --model_dir pretrained_models/CosyVoice2-0.5B --output_dir pretrained_models/CosyVoice2-0.5B-bnb --bits 8
```

参数说明：
- `--model_dir`: 原始模型目录
- `--output_dir`: 量化后模型保存目录
- `--bits`: 量化位数 (4 或 8)，建议先尝试8位

### 2. 使用简化的量化方法

我们提供了一个简化的量化脚本，它使用bitsandbytes库对模型进行量化，但采用了更直接的方法：

```bash
python quant_cosyvoice_gptq.py --model_dir pretrained_models/CosyVoice2-0.5B --output_dir pretrained_models/CosyVoice2-0.5B-quantized --bits 8
```

参数说明：
- `--model_dir`: 原始模型目录
- `--output_dir`: 量化后模型保存目录
- `--bits`: 量化位数 (4 或 8)
- `--block_size`: 量化块大小 (默认32)

## 使用量化后的模型

量化完成后，您可以使用以下命令测试量化后的模型：

```bash
python cosyvoice_2_demo.py --model_dir pretrained_models/CosyVoice2-0.5B-bnb
```

## 简单量化方法

如果上述方法都遇到问题，所有脚本都包含了一个简单的备选量化方法，它不依赖于特定的量化库，而是使用简单的权重量化技术。这种方法虽然不如专业量化库精确，但兼容性最好。

## 注意事项

1. 量化会导致模型质量略有下降，但通常不会显著影响语音合成质量
2. 4位量化可以显著减小模型大小，但可能会导致更多的质量损失
3. 如果遇到问题，建议先尝试8位量化，再尝试4位量化
4. 量化过程可能需要较长时间，请耐心等待

## 故障排除

如果在量化过程中遇到问题：

1. 首先尝试BitsAndBytes方法，它的兼容性最好
2. 如果出现内存错误，尝试在更大内存的机器上运行
3. 如果所有方法都失败，使用脚本中的简单量化方法
4. 确保您的Python环境干净，没有冲突的库版本 