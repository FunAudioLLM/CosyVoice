import sys
sys.path.append('third_party/Matcha-TTS')
import torch
import os
import shutil
from cosyvoice.cli.cosyvoice import CosyVoice2
import argparse

# 首先检查是否安装了bitsandbytes
try:
    import bitsandbytes as bnb
    print("成功导入bitsandbytes库")
except ImportError:
    print("请先安装bitsandbytes库: pip install bitsandbytes")
    sys.exit(1)

# 解析命令行参数
parser = argparse.ArgumentParser(description='使用bitsandbytes量化CosyVoice模型')
parser.add_argument('--model_dir', type=str, default='pretrained_models/CosyVoice2-0.5B',
                    help='原始模型目录路径')
parser.add_argument('--output_dir', type=str, default='pretrained_models/CosyVoice2-0.5B-bnb',
                    help='量化后模型保存目录')
parser.add_argument('--bits', type=int, default=8, choices=[4, 8],
                    help='量化位数 (4 或 8)')
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

print(f"开始使用bitsandbytes进行{args.bits}位量化...")

# 使用bitsandbytes量化模型
try:
    # 创建量化模型的副本
    quantized_model = type(original_model)(original_model.config)
    
    # 将原始模型的权重复制到量化模型
    quantized_model.load_state_dict(original_model.state_dict())
    
    # 将线性层转换为量化线性层
    for name, module in list(quantized_model.named_modules()):
        if isinstance(module, torch.nn.Linear):
            parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
            child_name = name.rsplit('.', 1)[1] if '.' in name else name
            
            if parent_name:
                parent = quantized_model.get_submodule(parent_name)
                
                if args.bits == 8:
                    # 8位量化
                    new_module = bnb.nn.Linear8bitLt(
                        module.in_features,
                        module.out_features,
                        bias=module.bias is not None,
                    )
                else:
                    # 4位量化
                    new_module = bnb.nn.Linear4bit(
                        module.in_features,
                        module.out_features,
                        bias=module.bias is not None,
                    )
                
                # 复制权重
                with torch.no_grad():
                    if hasattr(new_module, 'weight'):
                        new_module.weight.copy_(module.weight)
                    if module.bias is not None and hasattr(new_module, 'bias'):
                        new_module.bias.copy_(module.bias)
                
                # 替换模块
                try:
                    setattr(parent, child_name, new_module)
                    print(f"成功量化模块: {name}")
                except Exception as e:
                    print(f"无法量化模块 {name}: {e}")
    
    # 保存量化后的模型
    quantized_model_path = os.path.join(args.output_dir, 'llm.pt')
    torch.save(quantized_model, quantized_model_path)
    print(f"量化模型已保存到: {quantized_model_path}")
    
    print("量化完成！请使用以下命令测试量化后的模型:")
    print(f"python cosyvoice_2_demo.py --model_dir {args.output_dir}")
    
except Exception as e:
    print(f"bitsandbytes量化过程中出错: {e}")
    
    # 如果上面的方法失败，尝试使用简单的量化方法
    try:
        print("尝试使用简单量化方法...")
        
        # 创建一个简单的量化函数
        def simple_quantize(model, bits=8):
            """简单的量化函数，将模型的权重量化为指定位数"""
            for name, param in model.named_parameters():
                if 'weight' in name and param.dim() > 1:  # 只量化权重矩阵
                    # 计算量化范围
                    max_val = torch.max(torch.abs(param.data))
                    scale = (2**(bits-1) - 1) / max_val
                    
                    # 量化
                    param.data = torch.round(param.data * scale) / scale
            
            return model
        
        # 量化模型
        quantized_model = simple_quantize(original_model, bits=args.bits)
        
        # 保存量化后的模型
        quantized_model_path = os.path.join(args.output_dir, 'llm.pt')
        torch.save(quantized_model, quantized_model_path)
        print(f"使用简单量化方法保存模型到: {quantized_model_path}")
        
    except Exception as e2:
        print(f"简单量化方法也失败: {e2}")
        print("建议尝试手动调整模型结构或使用其他量化工具") 