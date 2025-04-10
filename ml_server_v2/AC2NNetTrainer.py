import numpy as np
import time
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import TensorBoard
from keras.layers import Flatten
from keras.layers import Reshape
from keras.utils import custom_object_scope
from tensorflow.keras.layers import Layer
from tensorflow.keras import regularizers
import MagicalDatasetProducer_v2 as mdsp
import dataSetBaseParamters as ds_bp
from SpaceStandard import *
import tensorflow as tf
import pdb
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
from skopt import gp_minimize
from tensorflow.keras.optimizers import Adam
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import shutil
from datetime import datetime


from apocolib import RpcProxyPool
pool = RpcProxyPool.RpcProxyPool()


np.set_printoptions(threshold=np.inf)

_ss = SpaceStandard()
max_k_room = _ss.get_max_k_room()
min_k_room = _ss.get_min_k_room()
max_k_house = _ss.get_max_k_house()
min_k_house = _ss.get_min_k_house()
max_num_rooms = _ss.get_max_num_rooms()
base_loss_weight = 1.0  # 调整这个权重以改变基础损失的重要性
penalty_weight = 0.3  # 调整这个权重以改变惩罚项的重要性
#dropout_rate = 0.005  # 增加Dropout比率
#l2_lambda = 0.0005  # L2正则化的lambda参数
#Best parameters: dropout_rate=0.184121, l2_lambda=0.009572, batch_size=23, learning_rate=0.003022
#Best parameters: dropout_rate=0.236804, l2_lambda=0.008009, batch_size=57, learning_rate=0.000519
dropout_rate = 0.15  # 增加Dropout比率
l2_lambda = 0.001  # L2正则化的lambda参数


def loss(y_true, y_pred):

    def compute_penalty(room_cost, room_k):
        if tf.reduce_all(tf.equal(room_cost, 0)):  # 使用 tf.reduce_all 将张量转换为布尔类型标量
            room_k = 0
            penalty = tf.maximum(0.0, room_k)
        else:
            penalty = tf.maximum(0.0, room_k - max_k_room) + tf.maximum(0.0, min_k_room - room_k)

        return penalty

    mse = tf.keras.losses.MeanSquaredError()
    base_loss = mse(y_true, y_pred)

    # 提取约束信息
    room_k_values = y_pred[:, 1:-1:2]  # 获取所有房间的 k 值，使用索引切片 :-1 排除平均 k 值(倒数第1个值)

    room_k_penalty = tf.map_fn(lambda x: compute_penalty(x[0], x[1]), (y_pred[:, ::2], room_k_values), fn_output_signature=tf.float32)

    house_avg_k_value = y_pred[:, -1]  # 对应最后一列的值为整体平均k值
    average_k_penalty = tf.maximum(0.0, house_avg_k_value - max_k_house) + tf.maximum(0.0, min_k_house - house_avg_k_value)

    total_loss = base_loss_weight * base_loss + penalty_weight * (tf.reduce_mean(room_k_penalty) + average_k_penalty)
    #total_loss = base_loss_weight * base_loss + penalty_weight * (constraint_loss + average_k_penalty)
    #total_loss = base_loss_weight * base_loss #+ penalty_weight * (constraint_loss + average_k_penalty)
    return total_loss * 100

# 用于优化输出
class optimizeOutputClipLayer(tf.keras.layers.Layer):
    # 
    min_val = 0
    max_val = 1

    # 想要的输出范围
    desired_min = min_k_house
    desired_max = max_k_house

    # 将输出范围也进行缩放
    scaled_min = (desired_min - min_val) / (max_val - min_val)
    scaled_max = (desired_max - min_val) / (max_val - min_val)

    def __init__(self, **kwargs):
        super(optimizeOutputClipLayer, self).__init__(**kwargs)

    def call(self, inputs):
        majority = inputs[:, :-1]  # 保留除最后一个值以外的所有值
        last_one = tf.clip_by_value(inputs[:, -1:], self.scaled_min, self.scaled_max)  # 将最后一个值限制在min - max 之间
        output = tf.concat([majority, last_one], axis=-1)  # 拼接这两部分
        #tf.print("output ",output)
        return output

    def get_config(self):
        config = super(optimizeOutputClipLayer, self).get_config()
        return config

class ConstraintLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ConstraintLayer, self).__init__(**kwargs)
        self.optimize_output_clip_layer = optimizeOutputClipLayer()

    def call(self, inputs):
        #pdb.set_trace()
        room_cost = inputs[:, :-2:2]  # 获取每个 room 的 cost，形状为 (batch_size, 12)
        room_k = inputs[:, 1:-2:2]  # 获取每个 room 的 k，形状为 (batch_size, 12)

        # 扩展维度以匹配 room_k 的形状
        room_cost_expanded = tf.expand_dims(room_cost, axis=-1)

        # Adjust room_k to 0 when room_cost is 0
        room_k = tf.where(tf.reduce_all(tf.equal(room_cost_expanded, 0)), tf.zeros_like(room_k), room_k)

        majority = inputs[:, :-2]  # 保留除最后两个值以外的所有值
        last_two = self.optimize_output_clip_layer(inputs[:, -2:])  # 对最后两个值进行处理
        output = tf.concat([majority, last_two], axis=-1)  # 拼接这两部分

        #tf.print("ConstraintLayer input ", inputs,summarize=-1)
        #tf.print("ConstraintLayer output ", output,summarize=-1)

        return output

class BackpropLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(BackpropLayer, self).__init__(**kwargs)

    def call(self, inputs):
        #pdb.set_trace()
        room_cost = inputs[:, :-2:2]  # 获取每个 room 的 cost，形状为 (batch_size, 12)
        room_k = inputs[:, 1:-2:2]  # 获取每个 room 的 k，形状为 (batch_size, 12)

        # Adjust room_k to 0 when room_cost is 0
        room_k = tf.where(tf.reduce_all(tf.equal(room_cost, 0)), tf.zeros_like(room_k), room_k)

        majority = inputs[:, :-2]  # 保留除最后两个值以外的所有值
        last_two = inputs[:, -2:]  # 获取最后两个值
        output = tf.concat([majority, last_two], axis=-1)  # 拼接这两部分

        #tf.print("BackpropLayer input ", inputs, summarize=-1)
        #tf.print("BackpropLayer output ", output, summarize=-1)

        return output

class HouseModelTrainer:

    def __init__(self,train_mode="continuing_training"): # retraining/continuing_training

        self.model_dir = "./net_model"
        self.model_files = ["house_model.h5", "model_weights.h5", "x_scaler_scale.npy", "x_scaler_min.npy", "y_scaler_scale.npy", "y_scaler_min.npy"]
        self.x_scaler = MinMaxScaler()
        self.y_scaler = MinMaxScaler()
        self.model = None
        self.houses_x = None
        self.houses_y = None
        self.train_mode= train_mode

    def load_model(self):

        f_model = f"{self.model_dir}/{self.model_files[0]}"
        if os.path.exists(f_model):
            with custom_object_scope({'optimizeOutputClipLayer': optimizeOutputClipLayer ,'ConstraintLayer': ConstraintLayer,'BackpropLayer': BackpropLayer, 'loss':loss }):
                self.model = tf.keras.models.load_model(f_model)
                print(f"\rload model file {f_model} successfully")
                return True
        print(f"\rload model file {f_model} failed")

        return False

    def generate_dataset(self,size=1000,model='reload'):
        #self.houses_x, self.houses_y = mdsp.generate_dataset(200)
        #self.houses_x, self.houses_y = mdsp.generate_dataset(size=100,model='create') # create/reload
        #self.houses_x, self.houses_y = mdsp.generate_dataset(size=size,model=model) # create/reload
        rpc_proxy = pool.get_connection()
        self.houses_x, self.houses_y = rpc_proxy.DataSetService.get_dataset(size=size,model=model)
        pool.put_connection(rpc_proxy)

    def split_data(self, test_size=0.2, validation_size=0.2):

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.houses_x, self.houses_y, test_size=test_size)
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(self.x_train, self.y_train, test_size=validation_size)

    def preprocess_data(self):

        self.x_train_scaled = self.x_scaler.fit_transform(self.x_train)
        self.x_val_scaled = self.x_scaler.transform(self.x_val)
        self.x_test_scaled = self.x_scaler.transform(self.x_test)
        self.y_train_scaled = self.y_scaler.fit_transform(self.y_train)
        self.y_val_scaled = self.y_scaler.transform(self.y_val)
        self.y_test_scaled = self.y_scaler.transform(self.y_test)

    def create_model(self,dropout_rate = 0.025,l2_lambda = 0.01):

        #dropout_rate = 0.025  # 增加Dropout比率
        #l2_lambda = 0.001  # L2正则化的lambda参数
        x_length = len(self.houses_x[0])
        y_length = len(self.houses_y[0])

        # reload model
        load = False
        if self.train_mode == "continuing_training":
            load = self.load_model()

        if not load:
            self.model = Sequential()

            self.model.add(Dense(64, activation='relu', input_dim=x_length,
                            kernel_regularizer=regularizers.l2(l2_lambda), bias_regularizer=regularizers.l2(l2_lambda)))
            self.model.add(Dropout(dropout_rate))
            self.model.add(Dense(128, activation='relu', 
                            kernel_regularizer=regularizers.l2(l2_lambda), bias_regularizer=regularizers.l2(l2_lambda)))
            self.model.add(Dropout(dropout_rate))
            self.model.add(Dense(256, activation='relu', 
                            kernel_regularizer=regularizers.l2(l2_lambda), bias_regularizer=regularizers.l2(l2_lambda)))
            self.model.add(Dropout(dropout_rate))
            self.model.add(Dense(128, activation='relu', 
                            kernel_regularizer=regularizers.l2(l2_lambda), bias_regularizer=regularizers.l2(l2_lambda)))
            self.model.add(Dropout(dropout_rate))
            self.model.add(Dense(64, activation='relu', 
                            kernel_regularizer=regularizers.l2(l2_lambda), bias_regularizer=regularizers.l2(l2_lambda)))
            self.model.add(Dropout(dropout_rate))
            # 硬约束层，修正空房间和无外墙房间 0值问题
            self.model.add(Dense(y_length, activation='relu'))  # 将输入形状改为 y_length = (26,)
            self.model.add(ConstraintLayer())
            self.model.add(Dense(y_length, activation='relu'))
            self.model.add(BackpropLayer())  # 添加自定义反向传播层 ,
            self.model.add(Dense(y_length, activation='relu'))

    # 编译模型
    def compile_model(self, optimizer='adam', loss='mse'):

        #self.model.compile(optimizer=optimizer, loss=loss ,metrics=['accuracy'])
        #self.model.compile(optimizer=optimizer, loss=loss ,metrics=['accuracy', tf.keras.metrics.RootMeanSquaredError(), "mae"])
        self.model.compile(optimizer=optimizer, loss=loss ,metrics=[tf.keras.metrics.RootMeanSquaredError(), "mae"])

    # 在模型训练中添加早停法
    def train_model(self, epochs=200, batch_size=50, verbose=1, callbacks=None):

        if callbacks is None:
            callbacks = []
        # 添加早停法回调
        early_stopping = EarlyStopping(monitor='val_loss', patience=30)  # patience参数是能够容忍多少个epoch内都没有improvement
        callbacks.append(early_stopping)

        history = self.model.fit(self.x_train_scaled, self.y_train_scaled, epochs=epochs, batch_size=batch_size, verbose=verbose, 
                callbacks=callbacks, validation_data=(self.x_val_scaled, self.y_val_scaled))

        return min(history.history['val_loss'])

    # 评估模型
    def evaluate_model(self):

        test_loss = self.model.evaluate(self.x_test_scaled, self.y_test_scaled)

        rmse_metric = tf.keras.metrics.RootMeanSquaredError()
        y_pred = self.model.predict(self.x_test_scaled)
        rmse_value = rmse_metric(self.y_test_scaled, y_pred)

        mae_metric = tf.keras.metrics.MeanAbsoluteError()
        mae_value = mae_metric(self.y_test_scaled, y_pred)

        #print("Test loss:", round(test_loss[0], 6), "Test accuracy:", round(test_loss[1], 6), "RMSE:", round(rmse_value.numpy(), 6), "MAE:", round(mae_value.numpy(), 6))
        print("Test loss: ", round(test_loss[0], 6), " RMSE: ", round(rmse_value.numpy(), 6), " MAE: ", round(mae_value.numpy(), 6))
        '''
        test_loss = self.model.evaluate(self.x_test_scaled, self.y_test_scaled)
        rmse_metric = tf.keras.metrics.RootMeanSquaredError()
        y_pred = self.model.predict(self.x_test_scaled)
        rmse_value = rmse_metric(self.y_test_scaled, y_pred)
        print("Test loss:", round(test_loss[0], 6), "Test accuracy:", round(test_loss[1], 6), "RMSE:", round(rmse_value.numpy(), 6))
        '''
        y_true = self.y_test_scaled
        y_pred = self.model.predict(self.x_test_scaled)

        #print("Input:")
        #tf.print(self.x_test_scaled)
        #print("True Output:")
        #print(y_true)
        #print("Predicted Output:")
        #tf.print(y_pred)
        #print("Test loss:", round(test_loss[0], 6), "Test accuracy:", round(test_loss[1], 6), "RMSE:", round(rmse_value.numpy(), 6))
        #test_loss = self.model.evaluate(self.x_test_scaled, self.y_test_scaled)
        #print("Test loss:", round(test_loss[0], 6), "Test accuracy:",round(test_loss[1], 6))

    def save_model(self):

        #model_dir = "./net_model"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = os.path.join(self.model_dir, timestamp)

        # Check if model files exist, if so, move them to a new directory
        #model_files = ["house_model.h5", "x_scaler_scale.npy", "x_scaler_min.npy", "y_scaler_scale.npy", "y_scaler_min.npy"]

        if any(os.path.exists(os.path.join(self.model_dir, file)) for file in self.model_files):
            os.makedirs(backup_dir, exist_ok=True)  # Create backup directory with timestamp
            for file in self.model_files:
                src_file_path = os.path.join(self.model_dir, file)
                if os.path.exists(src_file_path):
                    shutil.move(src_file_path, backup_dir)  # Move the file to backup directory

        # Save the new model files
        self.model.save(os.path.join(self.model_dir, "house_model.h5"))
        np.save(os.path.join(self.model_dir, "x_scaler_scale.npy"), self.x_scaler.scale_)
        np.save(os.path.join(self.model_dir, "x_scaler_min.npy"), self.x_scaler.min_)
        np.save(os.path.join(self.model_dir, "y_scaler_scale.npy"), self.y_scaler.scale_)
        np.save(os.path.join(self.model_dir, "y_scaler_min.npy"), self.y_scaler.min_)

    def process_weights(self,output_file):
        with open(output_file, 'w') as f:
            for layer in self.model.layers:
                layer_name = layer.name
                layer_weights = layer.get_weights()
                f.write(f"Layer: {layer_name}\n")
                f.write("Weights:\n")
                for i, weight in enumerate(layer_weights):
                    f.write(f"Weights {i + 1}:\n")
                    f.write(f"{weight}\n")
                f.write('\n')


class ActivationLogger(tf.keras.callbacks.Callback):
    def __init__(self, log_dir, x_val_scaled):
        super(ActivationLogger, self).__init__()
        self.log_dir = log_dir
        self.x_val_scaled = x_val_scaled

    def on_epoch_end(self, epoch, logs=None):
        layer_outputs = [layer.output for layer in self.model.layers]
        activation_model = tf.keras.models.Model(inputs=self.model.input, outputs=layer_outputs)
        activations = activation_model.predict(self.x_val_scaled)

        for i, activation in enumerate(activations):
            with open(f'{self.log_dir}/activation_{i}_epoch_{epoch}.npy', 'wb') as f:
                np.save(f, activation)

def objective(params):
    dropout_rate, l2_lambda, batch_size, learning_rate = params

    trainer = HouseModelTrainer()
    trainer.generate_dataset(size=1000,model='reload')
    trainer.split_data()
    trainer.preprocess_data()
    trainer.create_model(dropout_rate=dropout_rate, l2_lambda=l2_lambda)
    trainer.compile_model(optimizer=Adam(learning_rate=learning_rate), loss=loss)
    val_loss = trainer.train_model(epochs=100, batch_size=batch_size, verbose=1)
    return val_loss


def search_best_parameters():
    space = [(0.0, 0.5),  # dropout_rate
             (0.0, 0.01),  # l2_lambda
             (10, 100),  # batch_size
             (1e-6, 1e-2, 'log-uniform')]  # learning_rate

    res = gp_minimize(objective, space, n_calls=10, random_state=0)

    print("Best parameters: dropout_rate=%.6f, l2_lambda=%.6f, batch_size=%d, learning_rate=%.6f" % tuple(res.x))


import argparse

def main(args):
    # trainer = HouseModelTrainer(train_mode="retraining") #retraining/continuing_training
    trainer = HouseModelTrainer(train_mode=args.train_mode) #retraining/continuing_training
    trainer.generate_dataset(size=args.dataset_size, model=args.create_dataset_method)
    trainer.split_data()
    trainer.preprocess_data()
    trainer.create_model(dropout_rate=args.dropout_rate, l2_lambda=args.l2_lambda)

    # 创建TensorBoard回调
    log_dir = f'./net_logs/{time.strftime("%Y%m%d-%H%M%S")}'
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True)

    # 创建激活日志记录器回调
    activation_logger = ActivationLogger(log_dir, trainer.x_val_scaled)

    #trainer.compile_model(optimizer='adam', loss=loss)
    trainer.compile_model(optimizer=Adam(learning_rate=args.learning_rate), loss=loss)
    trainer.train_model(args.epochs, args.batch_size, 1, callbacks=[tensorboard_callback, activation_logger])
    trainer.evaluate_model()
    trainer.save_model()
    trainer.process_weights('./net_model/model_weights.h5')

if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='House Model Trainer')

    # 添加命令行参数
    parser.add_argument('--train_mode', type=str, required=True, choices=['retraining', 'continuing_training'], help='Train mode')
    parser.add_argument('--dataset_size', type=int, required=True, help='DataSet size')
    parser.add_argument('--create_dataset_method', type=str, required=True, choices=['reload','create'], help='Create dataset method')
    parser.add_argument('--dropout_rate', type=float, required=True, help='Dropout rate')
    parser.add_argument('--l2_lambda', type=float, required=True, help='L2 regularization lambda')
    parser.add_argument('--learning_rate', type=float, required=True, help='Learning rate')
    parser.add_argument('--epochs', type=int, required=True, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, required=True, help='Batch size')

    # 解析命令行参数
    args = parser.parse_args()

    # 调用主函数
    main(args)
