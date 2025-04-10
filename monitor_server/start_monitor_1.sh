#!/bin/bash

# 指定 Python 程序的路径
monitor_server_path="./monitorServer_MQ_tokenAuth.py"
data_collector_path="./MonitorDataCollector.py"
pid_file="./pid.txt"

# 检查监控服务器程序是否正在运行
check_process_running() {
    local process_pid=$1
    if ps -p $process_pid > /dev/null; then
        return 0
    else
        return 1
    fi
}

# 检查监控服务器程序是否已经在运行
if [ -f "$pid_file" ]; then
    while read -r pid; do
        if check_process_running $pid; then
            echo "Process with PID $pid is already running."
        else
            echo "Process with PID $pid is not running. Starting..."
            nohup python3 $monitor_server_path > monitor_output.log 2>&1 &
            echo $! >> "$pid_file"
            echo "Monitor server successfully launched."
        fi
    done < "$pid_file"
else
    echo "Pid file $pid_file does not exist. Starting monitor server..."
    nohup python3 $monitor_server_path > monitor_output.log 2>&1 &
    echo $! > "$pid_file"
    echo "Monitor server successfully launched."
fi

# 运行数据采集程序并放入后台运行
if ! pgrep -f "$data_collector_path" >/dev/null; then
    nohup python3 $data_collector_path > data_collector_output.log 2>&1 &
    echo $! >> "$pid_file"
    echo "Data collector successfully launched."
else
    echo "Data collector is already running."
fi

# 解除与终端的关联
#disown

# 退出脚本
exit 0

