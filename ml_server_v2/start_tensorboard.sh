pkill -f "tensorboard"
tensorboard --logdir net_logs --host 10.8.0.181 --port 8000 &
