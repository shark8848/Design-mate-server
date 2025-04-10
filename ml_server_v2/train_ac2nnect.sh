#!/bin/bash
# 检查模型是否正在训练
#retraining/continuing_training
if pgrep -x "python3" | grep "AC2NNetTrainer.py" > /dev/null; then
    echo "Model is already being trained. Cannot train another model simultaneously."
    exit 1
fi
# '--train_mode', type=str, required=True, choices=['retraining', 'continuing_training'], help='Train mode'
# '--dataset_size', type=int, required=True, help='DataSet size'
# '--create_dataset_method', type=str, required=True, choices=['reload','create'], help='Create dataset method'
# '--dropout_rate', type=float, required=True, help='Dropout rate'
# '--l2_lambda', type=float, required=True, help='L2 regularization lambda'
# '--learning_rate', type=float, required=True, help='Learning rate'
# '--epochs', type=int, required=True, help='Number of epochs'
# '--batch_size', type=int, required=True, help='Batch size'
python3 AC2NNetTrainer_v2.py \
        --train_mode continuing_training \
        --dataset_size 1000 \
        --create_dataset_method reload \
        --dropout_rate 0.2 \
        --l2_lambda 0.025 \
        --learning_rate 0.001 \
        --epochs 200 \
        --batch_size 64
