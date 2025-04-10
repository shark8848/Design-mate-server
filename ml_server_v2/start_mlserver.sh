# Author: sunhy,2023/3/12
# Copyright (c) 2023 APOCO Corporation
# All rights reserved.

#!/bin/bash

# 设置nameService清单文件路径
name_services_file="./ml_services.txt"

# 设置pid文件路径
pid_file="./pids.txt"

# 检查pid文件是否为空，如果不为空则停止所有微服务
if [ -s "$pid_file" ]
then
    ./stop_mlserver.sh
fi
# 从nameService清单文件中读取微服务名称并启动进程
> "$pid_file"
echo "---------------------------------------------------------------------"
echo "# Apoco Intelligent Analytics Multi-Job Frame Server v1.0           #"
echo "# Requires python 3.7 + ,nameko 2.11.2                              #"
echo "# Copyright (c) 2023 apoco. All rights reserved.                    #"
echo "---------------------------------------------------------------------"
while read name
do
    nameko run "$name" --broker amqp://guest:guest@192.168.1.36 >> ./log/nameko.log 2>&1 &
    echo "$!" >> "$pid_file"
    echo "$(date +"%Y-%m-%d %H:%M:%S") : PID - $! ML_microservice $name has been started "
done < "$name_services_file"
echo "All ml_microservices have been started!"
