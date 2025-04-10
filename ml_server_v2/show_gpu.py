import subprocess

def get_gpu_info():
    try:
        # 运行 nvidia-smi 命令并捕获输出
        result = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used", "--format=csv,noheader,nounits"])
        gpu_info = result.decode("utf-8").strip().split('\n')
        
        # 解析输出，将每个 GPU 的信息存储为字典
        gpu_data = []
        for line in gpu_info:
            line = line.strip().split(',')
            gpu_data.append({
                "index": int(line[0]),
                "name": line[1],
                "gpu_utilization": float(line[2]),
                "memory_utilization": float(line[3]),
                "total_memory": int(line[4]),
                "free_memory": int(line[5]),
                "used_memory": int(line[6])
            })
        
        return gpu_data
    except Exception as e:
        return str(e)

# 获取 GPU 信息
gpu_info = get_gpu_info()

# 打印 GPU 信息
for gpu in gpu_info:
    print(f"GPU {gpu['index']} - Name: {gpu['name']}")
    print(f"GPU Utilization: {gpu['gpu_utilization']}%")
    print(f"Memory Utilization: {gpu['memory_utilization']}%")
    print(f"Total Memory: {gpu['total_memory']} MB")
    print(f"Free Memory: {gpu['free_memory']} MB")
    print(f"Used Memory: {gpu['used_memory']} MB")
    print("\n")
