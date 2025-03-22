import sys
sys.path.append('third_party/Matcha-TTS')
import torch
import os
import shutil
from cosyvoice.cli.cosyvoice import CosyVoice2
import argparse

# 首先检查是否安装了awq
try:
    import awq
except ImportError:
    print("请先安装awq库: pip install awq")
    sys.exit(1)

# 解析命令行参数
parser = argparse.ArgumentParser(description='使用AWQ量化CosyVoice模型')
parser.add_argument('--model_dir', type=str, default='pretrained_models/CosyVoice2-0.5B',
                    help='原始模型目录路径')
parser.add_argument('--output_dir', type=str, default='pretrained_models/CosyVoice2-0.5B-awq',
                    help='量化后模型保存目录')
parser.add_argument('--bits', type=int, default=4, choices=[4, 8],
                    help='量化位数 (4 或 8)')
parser.add_argument('--group_size', type=int, default=128,
                    help='量化组大小')
args = parser.parse_args()

# 创建输出目录
os.makedirs(args.output_dir, exist_ok=True)

# 复制原始模型目录中除了llm.pt之外的所有文件
print(f"复制模型文件从 {args.model_dir} 到 {args.output_dir}")
for file_name in os.listdir(args.model_dir):
    if not file_name.endswith('.pt') and not file_name.endswith('.backup'):
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

# 准备校准数据
# 这里使用一些简单的文本作为校准数据
calibration_data = [
    "这是一个用于校准的示例文本，包含一些常见的中文词汇和句子结构。",
    "语音合成技术可以将文本转换为自然流畅的语音，广泛应用于各种场景。",
    "人工智能的发展日新月异，语音技术是其中重要的一环。",
    "这是一个测试句子，用于模型量化校准。",
    "欢迎使用CosyVoice语音合成系统，它可以生成自然、流畅的语音。"
]

print(f"开始使用AWQ进行{args.bits}位量化...")

# 使用AWQ量化模型
try:
    from awq import AutoAWQForCausalLM
    
    # 获取tokenizer
    tokenizer = original_model.tokenizer if hasattr(original_model, 'tokenizer') else None
    
    if tokenizer is None:
        print("警告: 无法获取tokenizer，AWQ可能无法正常工作")
        # 尝试从transformers加载tokenizer
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B")
            print("使用Qwen2-7B的tokenizer作为替代")
        except:
            print("无法加载替代tokenizer，将尝试继续...")
    
    # 使用AWQ量化
    quantized_model = AutoAWQForCausalLM.from_pretrained(
        original_model,
        tokenizer=tokenizer,
    )
    
    # 执行量化
    quantized_model.quantize(
        tokenizer=tokenizer,
        quant_config={
            "bits": args.bits,
            "group_size": args.group_size,
            "zero_point": True,
            "q_group_size": 128,
        },
        calib_data=calibration_data,
    )
    
    # 保存量化后的模型
    quantized_model.save_quantized(args.output_dir)
    print(f"量化模型已保存到: {args.output_dir}")
    
    print("量化完成！请使用以下命令测试量化后的模型:")
    print(f"python cosyvoice_2_demo.py --model_dir {args.output_dir}")
    
except Exception as e:
    print(f"AWQ量化过程中出错: {e}")
    print("尝试使用替代方法...")
    
    # 如果上面的方法失败，尝试使用optimum-awq
    try:
        from transformers import AutoModelForCausalLM
        from optimum.awq import AWQConfig, load_quantized_model
        
        print("使用optimum-awq进行量化...")
        
        # 配置AWQ
        awq_config = AWQConfig(
            bits=args.bits,
            group_size=args.group_size,
            zero_point=True,
        )
        
        # 量化模型
        quantized_model = load_quantized_model(
            original_model,
            awq_config,
            calibration_data,
        )
        
        # 保存量化后的模型
        quantized_model.save_pretrained(args.output_dir)
        print(f"量化模型已保存到: {args.output_dir}")
        
    except Exception as e2:
        print(f"替代方法也失败: {e2}")
        print("建议尝试使用其他量化工具，如bitsandbytes或llama.cpp") 