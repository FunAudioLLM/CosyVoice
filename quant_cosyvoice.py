import sys
sys.path.append('third_party/Matcha-TTS')
import torch
import os
import shutil
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.llm.llm import Qwen2LM


# 原始模型路径
model_dir = 'pretrained_models/CosyVoice2-0.5B'
original_llm_path = os.path.join(model_dir, 'llm.pt')

# 备份原始模型
backup_path = os.path.join(model_dir, 'llm.pt.backup')
if not os.path.exists(backup_path):
    print(f"备份原始模型到: {backup_path}")
    shutil.copy(original_llm_path, backup_path)

# 创建量化后的模型目录
quantized_model_dir = 'pretrained_models/CosyVoice2-0.5B-quantized'
os.makedirs(quantized_model_dir, exist_ok=True)

# 复制原始模型目录中除了llm.pt之外的所有文件
for file_name in os.listdir(model_dir):
    if not file_name.endswith('.pt') and not file_name.endswith(r'.backup') and not file_name.startswith(r'flow.'):
        src_path = os.path.join(model_dir, file_name)
        dst_path = os.path.join(quantized_model_dir, file_name)
        if os.path.isfile(src_path):
            shutil.copy2(src_path, dst_path)
            print(f"复制文件: {src_path} -> {dst_path}")

# 使用CosyVoice2类加载模型
print("加载原始模型...")
cosyvoice2 = CosyVoice2(model_dir, load_jit=False, load_trt=False, fp16=False)

# 提取LLM部分
original_model = cosyvoice2.model.llm
original_model.eval()  # 设置为评估模式

print("开始量化模型...")

# 创建一个新的量化模型，只量化线性层
# 使用更保守的量化设置
quantized_model = torch.quantization.quantize_dynamic(
    original_model, 
    {torch.nn.Linear},  # 只量化线性层
    dtype=torch.qint8,
    inplace=False  # 不要修改原始模型
)

# 保存量化后的模型到新目录
quantized_model_path = os.path.join(quantized_model_dir, 'llm.pt')
print(f"保存量化模型到: {quantized_model_path}")

# 使用torch.save保存整个模型，而不仅仅是state_dict
torch.save(quantized_model, quantized_model_path)

print(f"量化完成！请使用以下命令测试量化后的模型:")
print(f"python cosyvoice_2_demo.py --model_dir {quantized_model_dir}")
print("如果出现问题，可以继续使用原始模型。")


"""
# 方案2: 如果需要量化嵌入层，可以尝试以下代码（取消注释使用）
# 注意：这需要PyTorch 1.13或更高版本

# import torch.ao.quantization as quantization
# from torch.ao.quantization import float_qparams_weight_only_qconfig
# from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx

# # 为嵌入层设置特殊配置
# qconfig_dict = {
#     'object_type': [
#         (torch.nn.Embedding, float_qparams_weight_only_qconfig),
#         (torch.nn.Linear, torch.quantization.get_default_qconfig('fbgemm'))
#     ]
# }

# # 使用FX图模式量化
# prepared_model = prepare_fx(model, qconfig_dict)
# # 如果有校准数据，可以在这里运行校准
# quantized_model = convert_fx(prepared_model)

# # 保存量化后的模型
# torch.save(quantized_model.state_dict(), out_model)
"""
