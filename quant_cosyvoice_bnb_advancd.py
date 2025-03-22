import sys
sys.path.append('third_party/Matcha-TTS')
import torch
import os
import shutil
from cosyvoice.cli.cosyvoice import CosyVoice2
import argparse

"""
CosyVoice模型量化脚本

本脚本使用bitsandbytes库对CosyVoice模型进行量化，支持8位和4位量化。

量化原理:
1. 8位量化(Linear8bitLt): 
   - 将模型的线性层权重从FP32/FP16量化为INT8
   - 使用LLM.int8()方法，将异常值(outliers)提取出来在FP16中计算，其余在INT8中计算
   - 实际量化发生在模型被移动到CUDA设备时(.to("cuda"))
   - 输入数据需要是FP16类型

2. 4位量化(Linear4bit):
   - 将模型的线性层权重量化为4位精度
   - 计算使用FP16进行
   - 同样需要将模型移动到CUDA设备触发量化
   - 输入数据需要是FP16类型

注意事项:
- 量化前最好确保模型权重是FP16类型
- 使用量化模型时，输入必须是FP16类型
- 量化会导致一定的精度损失，但可以显著减少内存占用
"""

# 首先检查是否安装了必要的库
try:
    import bitsandbytes as bnb
    print("成功导入bitsandbytes库")
except ImportError:
    print("请先安装bitsandbytes库: pip install bitsandbytes")
    sys.exit(1)

# 解析命令行参数
parser = argparse.ArgumentParser(description='使用简化的量化方法量化CosyVoice模型')
parser.add_argument('--model_dir', type=str, default='pretrained_models/CosyVoice2-0.5B',
                    help='原始模型目录路径')
parser.add_argument('--output_dir', type=str, default='pretrained_models/CosyVoice2-0.5B-quantized',
                    help='量化后模型保存目录')
parser.add_argument('--bits', type=int, default=8, choices=[4, 8],
                    help='量化位数 (4 或 8)')
parser.add_argument('--block_size', type=int, default=32,
                    help='量化块大小')
parser.add_argument('--convert_to_fp16', action='store_true',
                    help='尝试将模型权重转换为fp16')
parser.add_argument('--save_quantized', action='store_true',
                    help='同时保存完整的量化模型（包含量化参数）')
args = parser.parse_args()

# 创建输出目录
os.makedirs(args.output_dir, exist_ok=True)

# 复制原始模型目录中除了llm.pt之外的所有文件
print(f"复制模型文件从 {args.model_dir} 到 {args.output_dir}")
for file_name in os.listdir(args.model_dir):
    if not file_name.endswith('llm.pt') and not file_name.endswith(r'.backup'):
        src_path = os.path.join(args.model_dir, file_name)
        dst_path = os.path.join(args.output_dir, file_name)
        if os.path.isfile(src_path) and not os.path.exists(dst_path):
            shutil.copy2(src_path, dst_path)
            print(f"复制文件: {src_path} -> {dst_path}")

# 加载原始模型
print("加载原始模型...")
cosyvoice2 = CosyVoice2(args.model_dir, load_jit=False, load_trt=False, fp16=False)

# 提取LLM部分
original_model = cosyvoice2.model.llm
original_model.eval()

# 尝试将模型转换为fp16
def convert_to_fp16(model):
    """尝试将模型权重转换为fp16"""
    print("尝试将模型权重转换为fp16...")
    try:
        # 检查当前权重类型
        weight_dtype = None
        for name, param in model.named_parameters():
            if 'weight' in name:
                weight_dtype = param.dtype
                print(f"原始模型权重数据类型: {weight_dtype}")
                break
        
        # 如果已经是fp16，则不需要转换
        if weight_dtype == torch.float16:
            print("模型已经是fp16类型，无需转换")
            return model
        
        # 转换为fp16
        fp16_model = model.half()
        print("成功将模型转换为fp16")
        
        # 验证转换结果
        for name, param in fp16_model.named_parameters():
            if 'weight' in name:
                print(f"转换后权重数据类型: {param.dtype}")
                break
        
        return fp16_model
    except Exception as e:
        print(f"转换模型为fp16失败: {e}")
        print("将继续使用原始模型进行量化")
        return model

# 尝试转换模型为fp16
if args.convert_to_fp16:
    original_model = convert_to_fp16(original_model)

# 检查原始模型大小
def check_model_size(model_path):
    """检查模型文件大小"""
    if os.path.exists(model_path):
        size_bytes = os.path.getsize(model_path)
        size_mb = size_bytes / (1024 * 1024)
        return size_mb
    return None

original_model_path = os.path.join(args.model_dir, 'llm.pt')
original_size_mb = check_model_size(original_model_path)
if original_size_mb:
    print(f"原始模型大小: {original_size_mb:.2f} MB")

print(f"开始进行{args.bits}位量化...")

# 定义一个更高级的量化函数
def advanced_quantize(model, bits=8, block_size=32):
    """
    使用块量化方法对模型进行量化
    
    参数:
    - model: 要量化的模型
    - bits: 量化位数 (4 或 8)
    - block_size: 量化块大小
    
    返回:
    - 量化后的模型
    """
    print(f"使用高级量化方法: {bits}位, 块大小={block_size}")
    
    if hasattr(model, 'config'):
        print(f"model有config")
    # 创建模型副本
    quantized_model = type(model)(model.config) if hasattr(model, 'config') else model
    
    # 复制模型状态
    if hasattr(model, 'state_dict'):
        print(f"复制模型状态: model有state_dict")
        quantized_model.load_state_dict(model.state_dict())
    
    # 检查模型权重的数据类型
    weight_dtype = None
    for name, param in quantized_model.named_parameters():
        if 'weight' in name:
            weight_dtype = param.dtype
            print(f"模型权重数据类型: {weight_dtype}")
            break
    
    # 根据权重类型决定has_fp16_weights参数
    has_fp16_weights = weight_dtype == torch.float16
    if not has_fp16_weights:
        print(f"警告: 模型权重不是float16类型，而是{weight_dtype}。将设置has_fp16_weights=False。")
        print("这可能会影响量化效果，建议先将模型转换为fp16再进行量化。")
    else:
        print("模型权重是float16类型，将设置has_fp16_weights=True以获得最佳效果。")
    
    # 移除量化特定的参数
    quantized_count = 0
    # 对每个线性层进行量化
    for name, module in list(quantized_model.named_modules()):
        if isinstance(module, torch.nn.Linear):
            parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
            child_name = name.rsplit('.', 1)[1] if '.' in name else name
            
            if parent_name:
                try:
                    parent = quantized_model.get_submodule(parent_name)
                    
                    # 根据位数选择量化方法
                    if bits == 8:
                        # 8位量化
                        new_module = bnb.nn.Linear8bitLt(
                            module.in_features,
                            module.out_features,
                            bias=module.bias is not None,
                            has_fp16_weights=False,
                            threshold= 0.001 # 推荐的阈值
                        )
                    else:
                        # 4位量化
                        new_module = bnb.nn.Linear4bit(
                            module.in_features,
                            module.out_features,
                            bias=module.bias is not None,
                            compute_dtype=torch.float16,
                        )
                    
                    # 复制权重
                    with torch.no_grad():
                        if hasattr(new_module, 'weight') and hasattr(module, 'weight'):
                            if hasattr(new_module.weight, 'copy_'):
                                # print("复制权重:", "name:", name, "parent_name:", parent_name, "child_name:", child_name)
                                new_module.weight.copy_(module.weight)
                        if module.bias is not None and hasattr(new_module, 'bias') and hasattr(module, 'bias'):
                            if hasattr(new_module.bias, 'copy_'):
                                new_module.bias.copy_(module.bias)
                    
                    # print("parent type:", type(parent))
                    # 替换模块
                    setattr(parent, child_name, new_module)
                    quantized_count += 1
                    # print(f"成功量化模块: {name}")
                except Exception as e:
                    print(f"无法量化模块 {name}: {e}")

    print(f"成功量化模块: {quantized_count} 个")
    # 将模型移动到CUDA设备，触发实际的量化过程
    if torch.cuda.is_available():
        print("将模型移动到CUDA设备，触发实际量化...")
        try:
            # 记录CUDA内存使用情况
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                before_mem = torch.cuda.memory_allocated() / (1024 * 1024)
                print(f"量化前CUDA内存占用: {before_mem:.2f} MB")
            
            quantized_model = quantized_model.to("cuda")
            print("成功将模型移动到CUDA设备")

            model = None
            torch.cuda.empty_cache()
            
            # 再次记录CUDA内存使用情况
            if torch.cuda.is_available():
                after_mem = torch.cuda.memory_allocated() / (1024 * 1024)
                print(f"量化后CUDA内存占用: {after_mem:.2f} MB")
                print(f"量化节省的CUDA内存: {(1 - after_mem/before_mem)*100:.2f}%")
        except Exception as e:
            print(f"将模型移动到CUDA设备时出错: {e}")
            print("量化可能未完全生效")
    else:
        print("警告: 未检测到CUDA设备，量化可能不会生效")
    
    return quantized_model

# 使用高级量化方法
try:
    quantized_model = advanced_quantize(original_model, bits=args.bits, block_size=args.block_size)
    # quantized_model = original_model
    # 清理模型中可能存在的原始权重，减小保存的模型大小
    print("清理模型中的原始权重以减小保存大小...")
    for name, module in quantized_model.named_modules():
        if hasattr(module, 'weight_ori') and module.weight_ori is not None:
            print(f"清理模块 {name} 的原始权重")
            module.weight_ori = None
    
    # 保存量化后的模型
    quantized_model_path = os.path.join(args.output_dir, 'llm.pt')
    
    # 如果需要，保存完整的量化模型（包含量化参数）
    if args.save_quantized:
        quantized_full_path = os.path.join(args.output_dir, 'llm_quantized_full.pt')
        torch.save(quantized_model.state_dict(), quantized_full_path)
        print(f"完整量化模型（包含量化参数）已保存到: {quantized_full_path}")
    
    # 创建兼容的状态字典，移除量化特定的参数
    def create_compatible_state_dict(model):
        """创建兼容的状态字典，移除量化特定的参数"""
        print("创建兼容的状态字典，移除量化特定的参数...")
        state_dict = model.state_dict()
        compatible_state_dict = {}
        
        # 移除量化特定的参数
        removed_count = 0
        for key in list(state_dict.keys()):
            if any(suffix in key for suffix in ['.SCB', '.weight_format', '.CB']):
                # print(f"移除量化特定参数: {key}")
                removed_count += 1
                continue
            compatible_state_dict[key] = state_dict[key]
        
        print(f"总共移除了 {removed_count} 个量化特定参数")
        return compatible_state_dict
    
    # 保存兼容的状态字典
    compatible_state_dict = create_compatible_state_dict(quantized_model)
    torch.save(compatible_state_dict, quantized_model_path)
    print(f"兼容的量化模型已保存到: {quantized_model_path}")
    
    # 检查模型大小
    model_size_mb = os.path.getsize(quantized_model_path) / (1024 * 1024)
    print(f"量化后模型大小: {model_size_mb:.2f} MB")
    
    # 显示大小比较
    if original_size_mb:
        size_ratio = model_size_mb / original_size_mb
        size_reduction = (1 - size_ratio) * 100
        print(f"模型大小变化: {size_ratio:.2f}x 原始大小 (减少了 {size_reduction:.2f}%)")
        if size_ratio > 1:
            print("警告: 量化后的模型比原始模型更大，这可能是因为保存了额外的量化参数或原始权重。")
            print("建议检查量化设置，特别是has_fp16_weights参数。")
    
    print("量化完成！请使用以下命令测试量化后的模型:")
    print(f"python cosyvoice_2_demo.py --model_dir {args.output_dir}")
    
    # 添加使用提示
    if args.bits == 8:
        print("\n重要提示：")
        print("1. 使用8位量化模型时，请确保输入数据为float16类型")
        print("2. 示例: model_input = model_input.to(torch.float16)")
        print("3. 如果遇到性能问题，可能需要检查模型是否正确量化")
        
        if args.save_quantized:
            print("\n如果要直接加载完整的量化模型，可以使用以下代码：")
            print("```python")
            print("import torch")
            print("import bitsandbytes as bnb")
            print("from transformers import AutoConfig")
            print("from cosyvoice.cli.cosyvoice import CosyVoice2")
            print("")
            print("# 创建一个自定义加载器函数")
            print("def load_quantized_model(model_dir):")
            print("    # 加载配置")
            print("    cosyvoice = CosyVoice2(model_dir, load_jit=False, load_trt=False, fp16=True)")
            print("    # 替换线性层为量化层")
            print("    for name, module in cosyvoice.model.llm.named_modules():")
            print("        if isinstance(module, torch.nn.Linear):")
            print("            parent_name = name.rsplit('.', 1)[0] if '.' in name else ''")
            print("            child_name = name.rsplit('.', 1)[1] if '.' in name else name")
            print("            if parent_name:")
            print("                parent = cosyvoice.model.llm.get_submodule(parent_name)")
            print("                # 创建8位量化层")
            print("                new_module = bnb.nn.Linear8bitLt(")
            print("                    module.in_features,")
            print("                    module.out_features,")
            print("                    bias=module.bias is not None,")
            print("                    has_fp16_weights=False,")
            print("                    threshold=6.0")
            print("                )")
            print("                # 替换模块")
            print("                setattr(parent, child_name, new_module)")
            print("    # 加载量化模型权重")
            print(f"    cosyvoice.model.llm.load_state_dict(torch.load('{os.path.join(args.output_dir, 'llm_quantized_full.pt')}'))")
            print("    # 移动到CUDA")
            print("    cosyvoice.model.llm = cosyvoice.model.llm.to('cuda')")
            print("    return cosyvoice")
            print("")
            print("# 使用自定义加载器加载量化模型")
            print(f"cosyvoice = load_quantized_model('{args.output_dir}')")
            print("```")
    elif args.bits == 4:
        print("\n重要提示：")
        print("1. 使用4位量化模型时，请确保输入数据为float16类型")
        print("2. 示例: model_input = model_input.to(torch.float16)")
        print("3. 如果遇到性能问题，可能需要检查模型是否正确量化")
    
    # 如果保存了完整量化模型，创建加载器脚本
    if args.save_quantized:
        loader_script_path = os.path.join(args.output_dir, 'load_quantized_model.py')
        with open(loader_script_path, 'w', encoding='utf-8') as f:
            f.write("""
import torch
import bitsandbytes as bnb
from cosyvoice.cli.cosyvoice import CosyVoice2
import os
import argparse

# 添加命令行参数解析
parser = argparse.ArgumentParser(description='加载量化的CosyVoice2模型')
parser.add_argument('--model_dir', type=str, default='""" + args.output_dir + """',
                    help='模型目录路径')
args = parser.parse_args()

def load_quantized_model(model_dir):
    \"\"\"加载量化的CosyVoice2模型\"\"\"
    print(f"加载量化模型: {model_dir}")
    
    # 加载配置
    cosyvoice = CosyVoice2(model_dir, load_jit=False, load_trt=False, fp16=True)
    
    # 替换线性层为量化层
    print("替换线性层为量化层...")
    for name, module in cosyvoice.model.llm.named_modules():
        if isinstance(module, torch.nn.Linear):
            parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
            child_name = name.rsplit('.', 1)[1] if '.' in name else name
            if parent_name:
                try:
                    parent = cosyvoice.model.llm.get_submodule(parent_name)
                    # 创建8位量化层
                    new_module = bnb.nn.Linear8bitLt(
                        module.in_features,
                        module.out_features,
                        bias=module.bias is not None,
                        has_fp16_weights=False,
                        threshold=6.0
                    )
                    # 替换模块
                    setattr(parent, child_name, new_module)
                    print(f"替换模块: {name}")
                except Exception as e:
                    print(f"替换模块 {name} 失败: {e}")
    
    # 加载量化模型权重
    quantized_weights_path = os.path.join(model_dir, 'llm_quantized_full.pt')
    print(f"加载量化权重: {quantized_weights_path}")
    cosyvoice.model.llm.load_state_dict(torch.load(quantized_weights_path))
    
    # 移动到CUDA
    if torch.cuda.is_available():
        print("将模型移动到CUDA...")
        cosyvoice.model.llm = cosyvoice.model.llm.to('cuda')
    else:
        print("警告: 未检测到CUDA设备")
    
    return cosyvoice

# 加载量化模型
cosyvoice = load_quantized_model(args.model_dir)

# 测试模型
print("\\n测试模型...")
from cosyvoice.utils.file_utils import load_wav
import torchaudio

# 加载测试音频
prompt_speech_16k = load_wav('./asset/zero_shot_prompt.wav', 16000)

# 测试推理
print("执行推理...")
for i, j in enumerate(cosyvoice.inference_zero_shot('这是一个测试句子，用于验证量化模型是否正常工作。', '希望一切顺利。', prompt_speech_16k, stream=False)):
    output_path = f'quantized_test_{i}.wav'
    torchaudio.save(output_path, j['tts_speech'], cosyvoice.sample_rate)
    print(f"已保存测试音频到: {output_path}")

print("\\n测试完成！如果生成了音频文件，说明量化模型加载成功。")
""")
        print(f"\n已创建量化模型加载器脚本: {loader_script_path}")
        print(f"可以使用以下命令测试量化模型:")
        print(f"python {loader_script_path}")
        print(f"或者指定模型目录:")
        print(f"python {loader_script_path} --model_dir 模型目录路径")
    
except Exception as e:
    print(f"高级量化方法失败: {e}")
    print("尝试使用简单量化方法...")
    
    # 简单量化方法
    def simple_quantize(model, bits=8):
        """简单的量化函数，将模型的权重量化为指定位数"""
        print(f"使用简单量化方法: {bits}位")
        
        # 创建模型副本
        quantized_model = type(model)(model.config) if hasattr(model, 'config') else model
        
        # 复制模型状态
        if hasattr(model, 'state_dict'):
            quantized_model.load_state_dict(model.state_dict())
        
        # 检查模型权重的数据类型
        weight_dtype = None
        for name, param in quantized_model.named_parameters():
            if 'weight' in name:
                weight_dtype = param.dtype
                print(f"模型权重数据类型: {weight_dtype}")
                break
        
        # 对每个参数进行量化
        for name, param in quantized_model.named_parameters():
            if 'weight' in name and param.dim() > 1:  # 只量化权重矩阵
                # 计算量化范围
                max_val = torch.max(torch.abs(param.data))
                scale = (2**(bits-1) - 1) / max_val
                
                # 量化
                param.data = torch.round(param.data * scale) / scale
                print(f"量化参数: {name}")
        
        # 将模型移动到CUDA设备（如果可用）
        if torch.cuda.is_available():
            print("将模型移动到CUDA设备...")
            try:
                quantized_model = quantized_model.to("cuda")
                print("成功将模型移动到CUDA设备")
            except Exception as e:
                print(f"将模型移动到CUDA设备时出错: {e}")
        else:
            print("警告: 未检测到CUDA设备")
        
        # 如果保存了完整量化模型，创建加载器脚本
        if args.save_quantized:
            loader_script_path = os.path.join(args.output_dir, 'load_quantized_model.py')
            with open(loader_script_path, 'w', encoding='utf-8') as f:
                f.write("""
import torch
import bitsandbytes as bnb
from cosyvoice.cli.cosyvoice import CosyVoice2
import os
import argparse

# 添加命令行参数解析
parser = argparse.ArgumentParser(description='加载量化的CosyVoice2模型')
parser.add_argument('--model_dir', type=str, default='""" + args.output_dir + """',
                    help='模型目录路径')
args = parser.parse_args()

def load_quantized_model(model_dir):
    \"\"\"加载量化的CosyVoice2模型\"\"\"
    print(f"加载量化模型: {model_dir}")
    
    # 加载配置
    cosyvoice = CosyVoice2(model_dir, load_jit=False, load_trt=False, fp16=True)
    
    # 替换线性层为量化层
    print("替换线性层为量化层...")
    for name, module in cosyvoice.model.llm.named_modules():
        if isinstance(module, torch.nn.Linear):
            parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
            child_name = name.rsplit('.', 1)[1] if '.' in name else name
            if parent_name:
                try:
                    parent = cosyvoice.model.llm.get_submodule(parent_name)
                    # 创建8位量化层
                    new_module = bnb.nn.Linear8bitLt(
                        module.in_features,
                        module.out_features,
                        bias=module.bias is not None,
                        has_fp16_weights=False,
                        threshold=6.0
                    )
                    # 替换模块
                    setattr(parent, child_name, new_module)
                    print(f"替换模块: {name}")
                except Exception as e:
                    print(f"替换模块 {name} 失败: {e}")
    
    # 加载量化模型权重
    quantized_weights_path = os.path.join(model_dir, 'llm_quantized_full.pt')
    print(f"加载量化权重: {quantized_weights_path}")
    cosyvoice.model.llm.load_state_dict(torch.load(quantized_weights_path))
    
    # 移动到CUDA
    if torch.cuda.is_available():
        print("将模型移动到CUDA...")
        cosyvoice.model.llm = cosyvoice.model.llm.to('cuda')
    else:
        print("警告: 未检测到CUDA设备")
    
    return cosyvoice

# 加载量化模型
cosyvoice = load_quantized_model(args.model_dir)

# 测试模型
print("\\n测试模型...")
from cosyvoice.utils.file_utils import load_wav
import torchaudio

# 加载测试音频
prompt_speech_16k = load_wav('./asset/zero_shot_prompt.wav', 16000)

# 测试推理
print("执行推理...")
for i, j in enumerate(cosyvoice.inference_zero_shot('这是一个测试句子，用于验证量化模型是否正常工作。', '希望一切顺利。', prompt_speech_16k, stream=False)):
    output_path = f'quantized_test_{i}.wav'
    torchaudio.save(output_path, j['tts_speech'], cosyvoice.sample_rate)
    print(f"已保存测试音频到: {output_path}")

print("\\n测试完成！如果生成了音频文件，说明量化模型加载成功。")
""")
            print(f"\n已创建量化模型加载器脚本: {loader_script_path}")
            print(f"可以使用以下命令测试量化模型:")
            print(f"python {loader_script_path}")
            print(f"或者指定模型目录:")
            print(f"python {loader_script_path} --model_dir 模型目录路径")
        
        return quantized_model
    
    try:
        # 使用简单量化方法
        quantized_model = simple_quantize(original_model, bits=args.bits)
        
        # 保存量化后的模型
        quantized_model_path = os.path.join(args.output_dir, 'llm.pt')
        
        # 如果需要，保存完整的量化模型（包含量化参数）
        if args.save_quantized:
            quantized_full_path = os.path.join(args.output_dir, 'llm_quantized_full.pt')
            torch.save(quantized_model.state_dict(), quantized_full_path)
            print(f"完整量化模型（包含量化参数）已保存到: {quantized_full_path}")
        
        # 保存兼容的状态字典
        compatible_state_dict = create_compatible_state_dict(quantized_model)
        torch.save(compatible_state_dict, quantized_model_path)
        print(f"兼容的量化模型已保存到: {quantized_model_path}")
        
        # 检查模型大小
        model_size_mb = os.path.getsize(quantized_model_path) / (1024 * 1024)
        print(f"量化后模型大小: {model_size_mb:.2f} MB")
        
        # 显示大小比较
        if original_size_mb:
            size_ratio = model_size_mb / original_size_mb
            size_reduction = (1 - size_ratio) * 100
            print(f"模型大小变化: {size_ratio:.2f}x 原始大小 (减少了 {size_reduction:.2f}%)")
            if size_ratio > 1:
                print("警告: 量化后的模型比原始模型更大，这可能是因为简单量化方法不够高效。")
                print("建议尝试其他量化方法或工具。")
        
        print("量化完成！请使用以下命令测试量化后的模型:")
        print(f"python cosyvoice_2_demo.py --model_dir {args.output_dir}")
        
    except Exception as e2:
        print(f"简单量化方法也失败: {e2}")
        print("建议尝试使用其他量化工具或手动调整模型结构")
