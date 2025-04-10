#!/bin/bash
  
# 指定 Python 程序的进程ID的路径
pid_path="./pid.txt"

# 检查 pid 文件是否存在
if [ -f "$pid_path" ]; then
    # 读取进程ID
    pid=$(cat $pid_path)

    # 检查进程是否正在运行
    if ps -p $pid > /dev/null; then
        # 使用 kill 命令终止程序
        kill -9 $pid
        echo "Websocket server successfully shut down."
    else
        echo "No running process found with pid $pid. It might have already been terminated."
    fi

    # 删除 pid 文件
    rm $pid_path
else
    echo "Pid file $pid_path does not exist. The Websocket server may not be running."
fi
