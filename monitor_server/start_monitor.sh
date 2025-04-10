#!/bin/bash

# 指定 Python 程序的路径
#program_path="./webSocketServer_MQ.py"
program_path="./monitorServer_MQ_tokenAuth.py"

# 定义一个函数来运行 Python 程序
run_program() {
    # 使用 nohup 和 & 在后台运行 Python 程序，并将输出重定向到一个日志文件
    nohup python3 $program_path > output.log 2>&1 &
    # 保存 Python 程序的 PID，以便稍后在需要重启程序时使用
    echo $! > pid.txt
    echo "Monitor server successfully launched."
}

# 检查 pid 文件是否存在，如果存在，检查对应的进程是否存在
if [ -e pid.txt ]; then
    pid=$(cat pid.txt)
    if ps -p $pid > /dev/null; then
        echo "Monitor server is already running."
        exit 1
    else
        run_program
    fi
else
    run_program
fi

# 将脚本转入后台运行
disown
