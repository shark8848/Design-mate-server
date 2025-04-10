import psutil

def get_cpu_usage():
    cpu_percentages = psutil.cpu_percent(percpu=True)
    return cpu_percentages

def get_memory_usage():
    memory = psutil.virtual_memory()
    return {
        'total': memory.total,
        'available': memory.available,
        'used': memory.used,
        'percent': memory.percent
    }

def get_swap_usage():
    swap = psutil.swap_memory()
    return {
        'total': swap.total,
        'used': swap.used,
        'free': swap.free,
        'percent': swap.percent
    }

def get_disk_usage():
    disk = psutil.disk_usage('/')
    return {
        'total': disk.total,
        'used': disk.used,
        'free': disk.free,
        'percent': disk.percent
    }

# 示例用法
cpu_usage = get_cpu_usage()
memory_usage = get_memory_usage()
swap_usage = get_swap_usage()
disk_usage = get_disk_usage()

print('CPU Usage:', cpu_usage)
print('Memory Usage:', memory_usage)
print('Swap Usage:', swap_usage)
print('Disk Usage:', disk_usage)


import time

while True:
    cpu_percentages = psutil.cpu_percent(percpu=True, interval=1)
    print("CPU Usage:", cpu_percentages)
    time.sleep(1)
