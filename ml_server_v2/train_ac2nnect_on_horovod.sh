#!/bin/bash

# 检查模型是否正在训练
if pgrep -x "python3" | grep "AC2NNetTrainer_horovod.py" > /dev/null; then
    echo "Model is already being trained. Cannot train another model simultaneously."
    exit 1
fi

# 根据$1参数选择训练模式
train_mode="$1"
if [ "$train_mode" != "retraining" ] && [ "$train_mode" != "continuing_training" ]; then
    echo "Invalid train mode. Please use 'retraining' or 'continuing_training'."
    exit 1
fi

cd ..
sh push_to_hdfs.sh *.pkl
cd ml_server_v2

#scp AC2NNetTrainer_horovod.py 192.168.1.39:/home/apoco/apoco-intelligent-analysis/ml_server_v2
# 运行训练脚本
# GPU_MEMORY_LIMIT 申请gpu的内存大小
# HOROVOD_TIMELINE timeline 文件，可以下载使用chrome:/tracing 查看，分析性能问题
# dropout_rate=0.412661, l2_lambda=0.005099, batch_size=41, learning_rate=0.004491
GPU_MEMORY_LIMIT=8192 MLIR_CRASH_REPRODUCER_DIRECTORY=./temp HOROVOD_TIMELINE=./net_logs/timeline.json horovodrun -np 2 -H hadoop2:1,hadoop4:1 \
        --log-level INFO \
        --mpi-threads-disable \
	--network-interfaces eno1 \
	--num-nccl-streams 4 \
        --verbose \
        --autotune  \
        --autotune-log-file ./log/autotune.log \
	--output-filename ./log/horovod_run_log \
	python AC2NNetTrainer_horovod.py --train_mode "$train_mode" \
        --gpu_memory_limit 4096 \
        --dataset_size 1000 \
        --dataset_dir ./datasets \
        --create_dataset_method reload \
        --dropout_rate 0.41 \
        --l2_lambda 0.005 \
        --learning_rate 0.004 \
        --epochs 10 \
        --batch_size 40 \
        --patience 15 \
	--net_log_dir ./net_logs \
	--verbose 1 \
	--hdfs_url http://192.168.1.19:9870 \
	--hdfs_user root \
        --hdfs_model_dir /net_model \
	--check_point_file ./check_point/checkpoint.h5
