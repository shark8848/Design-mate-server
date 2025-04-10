import logging
import numpy as np
import pyswarms as ps
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
from AC2NNetTrainer import HouseModelTrainer, loss
import subprocess

class ModelOptimizer:
    def __init__(self):
        self.options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
        self.bounds = (
            [0.0, 0.0, 10, 1e-6],  # 参数下界
            [0.5, 0.01, 100, 1e-2],  # 参数上界
        )
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler('./log/optimizer.log')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def objective(self, params, **kwargs):
        dropout_rate, l2_lambda, batch_size, learning_rate = params[0][0], params[0][1], params[0][2], params[0][3]
        trainer = HouseModelTrainer(train_mode='retraining')
        trainer.generate_dataset()
        trainer.split_data()
        trainer.preprocess_data()
        trainer.create_model(dropout_rate=dropout_rate, l2_lambda=l2_lambda)
        trainer.compile_model(optimizer=Adam(learning_rate=learning_rate), loss=loss)
        batch_size = np.round(batch_size).astype(int)
        
        # 创建 TensorBoard 回调
        tensorboard_callback = TensorBoard(log_dir='./net_logs')

        # 训练模型并将 TensorBoard 回调添加到训练过程中
        val_loss = trainer.train_model(epochs=200, batch_size=batch_size, verbose=0, callbacks=[tensorboard_callback])

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
        best_position, best_fitness = optimizer.optimize(self.objective, iters=100, callbacks=[tensorboard_callback])

        # 将最佳参数记录到日志
        best_params = tuple(best_fitness)
        self.logger.info(f"best_position: {best_position}, best_fitness: {best_fitness}")
        self.logger.info("最佳参数：dropout_rate=%.6f, l2_lambda=%.6f, batch_size=%d, learning_rate=%.6f" % best_params)

        return best_position, best_params

def main():
    optimizer = ModelOptimizer()
    best_position, best_fitness = optimizer.optimize_model()
    print(f"best_position: {best_position}，best_fitness: {best_fitness}")
        # 构建训练命令行参数
    train_command = [
        "python3",
        "AC2NNetTrainer.py",
        "--train_mode", "retraining",
        "--dataset_size", "1000",
        "--create_dataset_method", "reload",
        "--dropout_rate", str(best_fitness[0]),
        "--l2_lambda", str(best_fitness[1]),
        "--learning_rate", str(best_fitness[3]),
        "--epochs", "200",
        "--batch_size", str(np.round(best_fitness[2]).astype(int))
    ]

    # 调用训练脚本
    subprocess.run(train_command)
    

if __name__ == '__main__':
    main()
