import torch

def gpu_diagnostics():
    if torch.cuda.is_available():
        print("GPU 诊断报告:")
        print("="*40)
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            total_memory = props.total_memory / (1024 * 1024)
            reserved_memory = torch.cuda.memory_reserved(i) / (1024 * 1024)
            allocated_memory = torch.cuda.memory_allocated(i) / (1024 * 1024)
            free_memory = total_memory - allocated_memory

            print(f"GPU {i}: {props.name}")
            print(f"  总显存      : {round(total_memory, 2)} MB")
            print(f"  已保留显存  : {round(reserved_memory, 2)} MB")
            print(f"  已分配显存  : {round(allocated_memory, 2)} MB")
            print(f"  空闲显存    : {round(free_memory, 2)} MB")
            print("="*40)
    else:
        print("未找到 GPU，使用 CPU")

if __name__ == "__main__":
    gpu_diagnostics()
