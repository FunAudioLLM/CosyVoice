import sys
sys.path.append('third_party/Matcha-TTS')
import torch
import os
import shutil
from cosyvoice.cli.cosyvoice import CosyVoice2
import argparse

# 首先检查是否安装了必要的库
try:
    import auto_gptq
    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
    from auto_gptq.utils.peft_utils import get_gptq_peft_model
    print("成功导入auto-gptq库")
except ImportError as e:
    print(f"导入auto-gptq库失败: {e}")
    print("请安装auto-gptq库: pip install auto-gptq")
    sys.exit(1)
except Exception as e:
    print(f"auto-gptq库版本不兼容: {e}")
    print("请尝试安装兼容的版本: pip install auto-gptq==0.4.2 transformers==4.30.0")
    print("或者使用bitsandbytes方法: python quant_cosyvoice_bnb.py")
    sys.exit(1)

# 解析命令行参数
parser = argparse.ArgumentParser(description='使用GPTQ量化CosyVoice模型')
parser.add_argument('--model_dir', type=str, default='pretrained_models/CosyVoice2-0.5B',
                    help='原始模型目录路径')
parser.add_argument('--output_dir', type=str, default='pretrained_models/CosyVoice2-0.5B-gptq',
                    help='量化后模型保存目录')
parser.add_argument('--bits', type=int, default=8, choices=[2, 3, 4, 8],
                    help='量化位数 (2, 3, 4, 或 8)')
parser.add_argument('--group_size', type=int, default=128,
                    help='量化组大小')
parser.add_argument('--desc_act', action='store_true',
                    help='是否使用描述激活')
args = parser.parse_args()

# 创建输出目录
os.makedirs(args.output_dir, exist_ok=True)

# 复制原始模型目录中除了llm.pt之外的所有文件
print(f"复制模型文件从 {args.model_dir} 到 {args.output_dir}")
for file_name in os.listdir(args.model_dir):
    if not file_name.endswith('.pt') and not file_name.endswith(r'.backup') and not file_name.startswith(r'flow.'):
        src_path = os.path.join(args.model_dir, file_name)
        dst_path = os.path.join(args.output_dir, file_name)
        if os.path.isfile(src_path):
            shutil.copy2(src_path, dst_path)
            print(f"复制文件: {src_path} -> {dst_path}")

# 加载原始模型
print("加载原始模型...")
cosyvoice2 = CosyVoice2(args.model_dir, load_jit=False, load_trt=False, fp16=False)

# 提取LLM部分
original_model = cosyvoice2.model.llm
original_model.eval()

# 保存模型配置
if hasattr(original_model, 'config'):
    config_path = os.path.join(args.output_dir, 'config.json')
    if hasattr(original_model.config, 'to_json_file'):
        original_model.config.to_json_file(config_path)
        print(f"保存模型配置到: {config_path}")

# 设置GPTQ量化配置
quantize_config = BaseQuantizeConfig(
    bits=args.bits,                # 量化位数
    group_size=args.group_size,    # 量化组大小
    desc_act=args.desc_act,        # 是否使用描述激活
)

# 准备校准数据
# 这里使用一些简单的文本作为校准数据
# 在实际应用中，应该使用更多样化的数据
calibration_data = [
    "这是一个用于校准的示例文本，包含一些常见的中文词汇和句子结构。",
    "语音合成技术可以将文本转换为自然流畅的语音，广泛应用于各种场景。",
    "人工智能的发展日新月异，语音技术是其中重要的一环。",
    "这是一个测试句子，用于模型量化校准。",
    "欢迎使用CosyVoice语音合成系统，它可以生成自然、流畅的语音。"
]

# 创建一个简单的数据集
class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, texts, tokenizer):
        self.encodings = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        
    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}
    
    def __len__(self):
        return len(self.encodings.input_ids)

# 获取tokenizer
tokenizer = original_model.tokenizer if hasattr(original_model, 'tokenizer') else None

if tokenizer is None:
    print("警告: 无法获取tokenizer，将使用默认校准方法")
    examples = [""] * 5  # 使用空字符串作为默认校准数据
else:
    # 创建校准数据集
    dataset = SimpleDataset(calibration_data, tokenizer)
    examples = [{"input_ids": item["input_ids"]} for item in dataset]

print(f"开始使用GPTQ进行{args.bits}位量化...")






# 使用GPTQ量化模型
try:
    # 对于不同的模型架构，可能需要调整这里的代码
    quantized_model = AutoGPTQForCausalLM.from_pretrained(
        original_model,
        quantize_config=quantize_config,
    )
    
    # 执行量化
    quantized_model.quantize(examples)
    
    # 保存量化后的模型
    quantized_model_path = os.path.join(args.output_dir, 'llm.pt')
    quantized_model.save_pretrained(args.output_dir)
    print(f"量化模型已保存到: {args.output_dir}")
    
    print("量化完成！请使用以下命令测试量化后的模型:")
    print(f"python cosyvoice_2_demo.py --model_dir {args.output_dir}")
    
except Exception as e:
    print(f"量化过程中出错: {e}")
    print("尝试使用替代方法...")
    

    # 如果上面的方法失败，尝试使用更通用的方法
    try:
        from transformers import AutoModelForCausalLM
        from optimum.gptq import GPTQConfig, load_quantized_model
        
        print("使用optimum-gptq进行量化...")
        
        # 配置GPTQ
        gptq_config = GPTQConfig(
            bits=args.bits,
            group_size=args.group_size,
            desc_act=args.desc_act,
        )
        
        # 量化模型
        quantized_model = load_quantized_model(
            original_model,
            gptq_config,
            calibration_data,
        )
        
        # 保存量化后的模型
        quantized_model.save_pretrained(args.output_dir)
        print(f"量化模型已保存到: {args.output_dir}")
        
    except Exception as e2:
        print(f"替代方法也失败: {e2}")
        print("建议尝试使用其他量化工具，如bitsandbytes或llama.cpp") 
    
    