'''
from mpi4py import MPI
import subprocess

def get_gpu_temperature():
    # 运行 nvidia-smi 命令获取 GPU 温度
    result = subprocess.check_output(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"])
    temperature = int(result.decode("utf-8").strip())
    return temperature

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# 指定主节点的 rank
master_rank = 0

# 选择主节点并采集 GPU 温度
if rank == master_rank:
    # 指定要运行程序的服务器列表（IP地址或主机名）
    servers = ["hadoop2", "hadoop4"]

    # 在主节点上获取 GPU 温度
    gpu_temperatures = {}
    for server in servers:
        ssh_command = f"ssh {server} 'nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits'"
        result = subprocess.check_output(ssh_command, shell=True)
        temperature = int(result.decode("utf-8").strip())
        gpu_temperatures[server] = temperature

    # 打印每台服务器的 GPU 温度
    for server, temperature in gpu_temperatures.items():
        print(f"{server} - GPU Temperature: {temperature}°C")

# 其他从节点等待主节点广播的数据
else:
    # 使用 MPI 广播从主节点接收的 GPU 温度数据
    gpu_temperatures = comm.bcast(None, root=master_rank)

    # 打印本节点的 GPU 温度
    print(f"Process {rank} - GPU Temperature: {gpu_temperatures[rank]}°C")
    '''

from mpi4py import MPI
import subprocess

def get_gpu_info():
    # 运行 nvidia-smi 命令获取 GPU 参数
    result = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used", "--format=csv,noheader,nounits"])
    gpu_info = result.decode("utf-8").strip().split('\n')
    
    # 解析输出，将每个 GPU 的信息存储为字典
    gpu_data = []
    for line in gpu_info:
        line = line.strip().split(',')
        gpu_data.append({
            "index": int(line[0]),
            "name": line[1],
            "temperature": int(line[2]),
            "gpu_utilization": float(line[3]),
            "memory_utilization": float(line[4]),
            "total_memory": int(line[5]),
            "free_memory": int(line[6]),
            "used_memory": int(line[7])
        })
    
    return gpu_data

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# 指定主节点的 rank
master_rank = 0

# 选择主节点并获取 GPU 参数
if rank == master_rank:
    # 指定要运行程序的服务器列表（IP地址或主机名）
    servers = ["hadoop2", "hadoop4"]

    # 在主节点上获取 GPU 参数
    gpu_info = {}
    for server in servers:
        ssh_command = f"ssh {server} 'nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv,noheader,nounits'"
        result = subprocess.check_output(ssh_command, shell=True).decode("utf-8").strip().split('\n')
        
        server_gpu_info = []
        for line in result:
            line = line.strip().split(',')
            server_gpu_info.append({
                "index": int(line[0]),
                "name": line[1],
                "temperature": int(line[2]),
                "gpu_utilization": float(line[3]),
                "memory_utilization": float(line[4]),
                "total_memory": int(line[5]),
                "free_memory": int(line[6]),
                "used_memory": int(line[7])
            })
        
        gpu_info[server] = server_gpu_info

    # 广播 GPU 参数数据到其他从节点
    gpu_info = comm.bcast(gpu_info, root=master_rank)

# 其他从节点等待主节点广播的数据
else:
    # 使用 MPI 广播从主节点接收的 GPU 参数数据
    gpu_info = comm.bcast(None, root=master_rank)

# 打印每个节点的 GPU 参数数据
for server, server_gpu_data in gpu_info.items():
    print(f"Server: {server}")
    for gpu in server_gpu_data:
        print(f"GPU {gpu['index']} - Name: {gpu['name']}")
        print(f"Temperature: {gpu['temperature']}°C")
        print(f"GPU Utilization: {gpu['gpu_utilization']}%")
        print(f"Memory Utilization: {gpu['memory_utilization']}%")
        print(f"Total Memory: {gpu['total_memory']} MB")
        print(f"Free Memory: {gpu['free_memory']} MB")
        print(f"Used Memory: {gpu['used_memory']} MB")
        print("\n")

