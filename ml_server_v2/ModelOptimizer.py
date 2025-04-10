import logging,os
import numpy as np
import pyswarms as ps
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
#from AC2NNetTrainer import HouseModelTrainer, loss
from AC2NNetTrainer_horovod import *
import subprocess

class ModelOptimizer:
    def __init__(self):
        self.options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
        self.bounds = (
            [0.0, 0.0, 10, 1e-6],  # 参数下界
            [0.5, 0.01, 100, 1e-2],  # 参数上界
        )
        self.dataset_size=1000
        self.create_dataset_method='reload'
        self.log_dir='./net_logs'
        self.epochs=100
        self.check_point_file='./check_point/checkpoint.h5'
        self.verbose=0
        self.patience=15
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler('./log/optimizer.log')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def objective(self, params, **kwargs):
        dropout_rate, l2_lambda, batch_size, learning_rate = params[0][0], params[0][1], params[0][2], params[0][3]
        batch_size = np.round(batch_size).astype(int)

        trainer = HouseModelTrainer(train_mode='retraining')
        trainer.generate_dataset(size=self.dataset_size, model=self.create_dataset_method)
        trainer.split_data()
        trainer.preprocess_data()
        trainer.create_model(dropout_rate=dropout_rate, l2_lambda=l2_lambda)
        
        # 创建 TensorBoard 回调
        self.log_dir = os.path.join(self.log_dir, time.strftime("%Y%m%d-%H%M%S"))
        tensorboard_callback = TensorBoard(log_dir=self.log_dir, histogram_freq=1, write_graph=True, write_images=True)
        # 创建激活日志记录器回调
        activation_logger = ActivationLogger(self.log_dir, trainer.x_val_scaled)
        # mpi ,要求模型序列化
        model_bytes = trainer.compile_model(optimizer= tf.optimizers.Adam(learning_rate * 2),loss=loss)
        val_loss = trainer.train_model(model_bytes = model_bytes, epochs = self.epochs, batch_size = batch_size, initial_lr = learning_rate, ckpt_file= self.check_point_file, verbose = self.verbose, callbacks=[tensorboard_callback, activation_logger], patience=self.patience)
        # 将结果记录到日志
        self.logger.info(f"dropout_rate: {dropout_rate}, l2_lambda: {l2_lambda}, batch_size: {batch_size}, learning_rate: {learning_rate}")
        self.logger.info(f"val_loss: {val_loss}")

        return val_loss

    def optimize_model(self):
        # 使用 TensorBoard 来记录优化过程中的相关数据
        tensorboard_callback = TensorBoard(log_dir='./net_logs')

        # 采用 PSO 粒子群算法进行超参优化
        self.logger.info(f"n_particles=20, dimensions=4, bounds={self.bounds} , options={self.options} ,iters=10 ")

        optimizer = ps.single.GlobalBestPSO(n_particles=20, dimensions=4, bounds=self.bounds, options=self.options)
        best_position, best_fitness = optimizer.optimize(self.objective, iters=10, callbacks=[tensorboard_callback])

        # 将最佳参数记录到日志
        best_params = tuple(best_fitness)
        self.logger.info(f"best_position: {best_position}, best_fitness: {best_fitness}")
        self.logger.info("最佳参数：dropout_rate=%.6f, l2_lambda=%.6f, batch_size=%d, learning_rate=%.6f" % best_params)

        return best_position, best_params

def main():
    optimizer = ModelOptimizer()
    best_position, best_fitness = optimizer.optimize_model()
    print(f"best_position: {best_position}，best_fitness: {best_fitness}")
    # 现有环境变量
    env = os.environ.copy()

    # 添加或修改新环境变量
    env["GPU_MEMORY_LIMIT"] = "4096"
    env["HOROVOD_TIMELINE"] = "./net_logs/timeline.json"
    env["MLIR_CRASH_REPRODUCER_DIRECTORY"] = "./temp"

    # 构建要执行的命令
    horovodrun_command = [
        "horovodrun", "-np", "2", "-H", "hadoop2:1,hadoop4:1",
        "--log-level", "INFO",
        "--mpi-threads-disable",
        "--num-nccl-streams", "4",
        "--verbose",
        "--autotune",
        "--autotune-log-file", "./log/autotune.log",
        "--output-filename", "./log/horovod_run_log",
        "python3", "AC2NNetTrainer_horovod.py",
        "--train_mode", "retraining",
        "--dataset_size", "1000",
        "--create_dataset_method", "reload",
        "--dropout_rate", str(best_fitness[0]),
        "--l2_lambda", str(best_fitness[1]),
        "--learning_rate", str(best_fitness[3]),
        "--epochs", "100",
        "--batch_size", str(np.round(best_fitness[2]).astype(int)),
        "--patience", "15",
        "--net_log_dir", "./net_logs",
        "--verbose", "1",
        "--hdfs_url", "http://192.168.1.19:9870",
        "--hdfs_user", "root",
        "--check_point_file", "./check_point/checkpoint.h5"
    ]

    # 启动子进程，捕获标准输出和标准错误
    proc = subprocess.Popen(
        horovodrun_command,
        env=env,  # 设置环境变量
        stdout=subprocess.PIPE,  # 捕获标准输出
        stderr=subprocess.PIPE,  # 捕获标准错误
        universal_newlines=True,  # 以文本模式处理输出
        shell=False,  # 不使用 shell 执行
        bufsize=1,  # 设置缓冲区大小以实现实时输出
        close_fds=True,  # 关闭不必要的文件描述符
    )

    # 逐行读取标准输出和标准错误并打印
    for line in proc.stdout:
        print("Standard Output:", line, end="")

    for line in proc.stderr:
        print("Standard Error:", line, end="")

    # 等待子进程完成
    proc.wait()

    # 子进程完成后，获取返回码
    return_code = proc.returncode
    print("Return Code:", return_code)

if __name__ == '__main__':
    main()
