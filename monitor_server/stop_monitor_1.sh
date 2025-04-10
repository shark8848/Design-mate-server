#!/bin/bash

# 指定存放多个进程ID的文件路径
pid_path="./pid.txt"

# 检查 pid 文件是否存在
if [ -f "$pid_path" ]; then
    # 读取进程ID，并按行进行处理
    while IFS= read -r pid; do
        # 检查当前进程是否正在运行
        if ps -p $pid > /dev/null; then
            # 使用 kill 命令终止程序
            kill -9 $pid
            echo "Process with pid $pid successfully terminated."
        else
            echo "No running process found with pid $pid. It might have already been terminated."
        fi
    done < "$pid_path"

    # 删除 pid 文件
    rm $pid_path
else
    echo "Pid file $pid_path does not exist. No processes to terminate."
fi

