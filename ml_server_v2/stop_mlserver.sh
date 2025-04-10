# Author: sunhy,2023/3/12
# Copyright (c) 2023 APOCO Corporation
# All rights reserved.

#!/bin/bash

# 设置pid文件路径
pid_file="./pids.txt"

# 读取pid文件并发送SIGTERM信号到所有进程
while read pid
do
#    echo "killed" $pid
    kill -TERM "$pid"
done < "$pid_file"
echo "All ml_microservices have been stopped!"
>"$pid_file"
