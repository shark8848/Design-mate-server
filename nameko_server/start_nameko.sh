#!/bin/bash

# 设置nameService清单文件路径
#name_services_file="./name_services.txt"

# 设置pid文件路径
#pid_file="./pids.txt"
#./stop_nameko.sh
# 从nameService清单文件中读取微服务名称并启动进程
#> "$pid_file"

#while read name
#do
#    nameko run "$name" --broker amqp://guest:guest@192.168.1.19 >> ./log/nameko.log 2>&1 &
#    echo "$!" >> "$pid_file"
#    echo start microservice $name
#done < "$name_services_file"
#echo "All microservices have been started!"

#----------------------------------------------------------------------

# Author: sunhy,2023/3/12
# Copyright (c) 2023 APOCO Corporation
# All rights reserved.

#!/bin/bash

# 设置nameService清单文件路径
name_services_file="./name_services.txt"

# 设置pid文件路径
pid_file="./pids.txt"

# 检查pid文件是否为空，如果不为空则停止所有微服务
if [ -s "$pid_file" ]
then
    ./stop_nameko.sh
fi
# 从nameService清单文件中读取微服务名称并启动进程
> "$pid_file"
echo "---------------------------------------------------------------------"
echo "# Apoco Intelligent Analytics nameko Server v1.0                    #"
echo "# Requires python 3.7 + ,nameko 2.11.2                              #"
echo "# Copyright (c) 2023 apoco. All rights reserved.                    #"
echo "---------------------------------------------------------------------"
while read name
do
    #nameko run "$name" --broker amqp://guest:guest@10.8.0.181 >> ./log/nameko.log 2>&1 &
    #nameko run "$name" --broker amqp://guest:guest@10.8.0.181 2>&1 &
    nameko run "$name" --broker amqp://guest:guest@10.8.0.181 &
    echo "$!" >> "$pid_file"
    echo "$(date +"%Y-%m-%d %H:%M:%S") : PID - $! nameko services $name has been started "
done < "$name_services_file"
echo "All nameko services have been started!"
