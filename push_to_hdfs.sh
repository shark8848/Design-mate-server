#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <filter_pkl>"
    exit 1
fi

# Set the filter_pkl variable based on the argument
filter_pkl=$1

# Set the root directory
root_dir="/home/apoco/apoco-intelligent-analysis"

# Create the tar archive with specified exclusions
tar cvf python_lib.tar \
	--exclude '*.log' \
	--exclude '*.pyc' \
	--exclude '*.tar' \
	--exclude '*.bak' \
	--exclude '*.json' \
	--exclude '*bak*' \
	--exclude 'horovod*' \
	--exclude '*.xls' \
	--exclude '*.pdf' \
	--exclude '*.xlsx' \
	--exclude '*event*' \
	--exclude '*.png' \
	--exclude '*net_logs' \
	--exclude '*.h5' \
	--exclude '*.kreas' \
	--exclude '*net_model*' \
	--exclude '*html*' \
	--exclude '*.out' \
	--exclude '*check_point*' \
	--exclude "$filter_pkl" \
	apocolib config database flask_server ml_server_v2 nameko_server websocket_server monitor_server > /dev/null

zip --quiet python_lib.zip -r ml_server_v2 apocolib config database flask_server ml_server monitor_server nameko_server websocket_server -i *.py

# Put the tar archive in HDFS
hdfs dfs -put -f python_lib.tar /dependency/
hdfs dfs -put -f python_lib.zip /dependency/

# Copy the tar archive and house_model.engine to the remote server
scp python_lib.tar 192.168.1.39:$root_dir
scp ./ml_server_v2/net_model/house_model.engine 192.168.1.39:$root_dir/ml_server_v2/net_model

# Extract the tar archive on the remote server
ssh root@192.168.1.39 "cd $root_dir; tar -xf python_lib.tar"
