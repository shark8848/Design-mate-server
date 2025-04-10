import GPUtil
import argparse
import horovod.tensorflow.keras as hvd
import onnxmltools
import tf2onnx
from apocolib import RpcProxyPool
import hdfs
from datetime import datetime
import shutil
import os
from skopt import gp_minimize
import pdb
from BuildingElement import *
from BuildingSpaceBase import *
from SpaceStandard import *
import dataSetBaseParamters as ds_bp
import MagicalDatasetProducer_v2 as mdsp
from apocolib.dataset_io import load_dataset
from apocolib.GPUConfigurator import GPUConfigurator
import numpy as np
import time
import h5py
import io
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
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
import sys
sys.path.append("..")


# tf.config.run_functions_eagerly(True)
# tf.compat.v1.enable_eager_execution()


# gpu_memory_limit = int(os.getenv('GPU_MEMORY_LIMIT', '4096'))
# # 配置gpu 及内存使用大小
# configurator = GPUConfigurator(use_gpu=True, gpu_memory_limit=gpu_memory_limit)
# configurator.configure_gpu()
# tf.device(configurator.select_device())

# tf.get_logger().setLevel(tf.compat.v1.logging.INFO)
# tf.get_logger().addHandler(tf.get_logger().addHandler('./log/tensorflow.log'))

# from tensorflow.keras.optimizers import Adam


# 初始化 Horovod
hvd.init()
# 设置 GPU 可见性
# os.environ["CUDA_VISIBLE_DEVICES"] = str(hvd.local_rank())


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
# dropout_rate = 0.005  # 增加Dropout比率
# l2_lambda = 0.0005  # L2正则化的lambda参数
# Best parameters: dropout_rate=0.184121, l2_lambda=0.009572, batch_size=23, learning_rate=0.003022
# Best parameters: dropout_rate=0.236804, l2_lambda=0.008009, batch_size=57, learning_rate=0.000519
dropout_rate = 0.15  # 增加Dropout比率
l2_lambda = 0.001  # L2正则化的lambda参数


def loss(y_true, y_pred):

    min_avg_cost = 50
    max_avg_cost = 200
    min_avg_k = 0.5
    max_avg_k = 1.8
    # 定义基本的均方误差损失
    mse = tf.keras.losses.MeanSquaredError()
    # 排除整层的成本和平均K值  sum((y_pred - y_true)**2) / n
    base_loss = mse(y_true[:, :-2], y_pred[:, :-2])

    # 提取整层的总成本和平均K值
    avg_cost = y_true[:, -2]  # 提取整层的总成本
    avg_k = y_true[:, -1]  # 提取整层的平均K值

    # 计算整层的平衡惩罚
    avg_cost_balance_penalty = tf.maximum(
        0.0, avg_cost - max_avg_cost) + tf.maximum(0.0, min_avg_cost - avg_cost)
    avg_k_balance_penalty = tf.maximum(
        0.0, avg_k - max_avg_k) + tf.maximum(0.0, min_avg_k - avg_k)

    # 计算总损失
    total_loss = base_loss_weight * base_loss + penalty_weight * \
        (avg_cost_balance_penalty + avg_k_balance_penalty)

    return total_loss


def serialize_model(model):
    """Serialize model into byte array."""
    bio = io.BytesIO()
    with h5py.File(bio, 'w') as f:
        model.save(f)

    return bio.getvalue()


def serialize_modelfile(filepath):
    with h5py.File(filepath, 'r') as f:
        dataset = next(iter(f.values()))
        data = dataset[()]
    return data.tobytes()


def deserialize_model(model_bytes, load_model_fn, custom_objects):
    """Deserialize model from byte array."""
    bio = io.BytesIO(model_bytes)
    with h5py.File(bio, 'r') as f:
        return load_model_fn(f, custom_objects=custom_objects)


# 用于优化输出
@tf.keras.saving.register_keras_serializable('opt_ClipLayer')
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
        last_one = tf.clip_by_value(
            inputs[:, -1:], self.scaled_min, self.scaled_max)  # 将最后一个值限制在min - max 之间
        output = tf.concat([majority, last_one], axis=-1)  # 拼接这两部分
        # tf.print("output ",output)
        return output

    def get_config(self):
        config = super(optimizeOutputClipLayer, self).get_config()
        return config


if (tf.keras.saving.get_registered_object('opt_ClipLayer>optimizeOutputClipLayer') == optimizeOutputClipLayer):
    pass
    #ml_logger.info('optimizeOutputClipLayer registered')


@tf.keras.saving.register_keras_serializable('opt_ClipLayer')
class ConstraintLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ConstraintLayer, self).__init__(**kwargs)
        self.optimize_output_clip_layer = optimizeOutputClipLayer()

    def call(self, inputs):
        # pdb.set_trace()
        room_cost = inputs[:, :-2:2]  # 获取每个 room 的 cost，形状为 (batch_size, 12)
        room_k = inputs[:, 1:-2:2]  # 获取每个 room 的 k，形状为 (batch_size, 12)

        # 扩展维度以匹配 room_k 的形状
        room_cost_expanded = tf.expand_dims(room_cost, axis=-1)

        # Adjust room_k to 0 when room_cost is 0
        room_k = tf.where(tf.reduce_all(
            tf.equal(room_cost_expanded, 0)), tf.zeros_like(room_k), room_k)

        majority = inputs[:, :-2]  # 保留除最后两个值以外的所有值
        last_two = self.optimize_output_clip_layer(
            inputs[:, -2:])  # 对最后两个值进行处理
        output = tf.concat([majority, last_two], axis=-1)  # 拼接这两部分

        return output

    def get_config(self):
        config = super(ConstraintLayer, self).get_config()
        return config


if (tf.keras.saving.get_registered_object('opt_ClipLayer>ConstraintLayer') == ConstraintLayer):
    pass
    #ml_logger.info('ConstraintLayer registered')


@tf.keras.saving.register_keras_serializable('opt_ClipLayer')
class BackpropLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(BackpropLayer, self).__init__(**kwargs)

    def call(self, inputs):

        return self.correct_data_tensor(inputs)

    def get_config(self):
        config = super(BackpropLayer, self).get_config()
        return config

    def correct_house_data(self, house_data):

        # 分割成 12 个 house_data
        house_data_list = tf.split(house_data, 12, axis=1)

        corrected_house_data_list = []
        for house_data in house_data_list:
            corrected_house_data = self.correct_1_house_data(house_data)
            # 打印 house_data 的形状 和 corrected_house_data 的形状
            # tf.print("~~~~~~~~~~~ house_data.shape: ", house_data.shape,
            #         "corrected_house_data.shape: ", corrected_house_data.shape)
            corrected_house_data_list.append(corrected_house_data)

        # 拼接所有处理过的 house_data
        corrected_house_data_all = tf.concat(corrected_house_data_list, axis=1)

        # tf.print("~~~~~~~~~~~ corrected_house_data_all.shape: ",
        #        corrected_house_data_all.shape)

        return corrected_house_data_all

    def correct_1_house_data(self, house_data):

        # 截取 house_data 的前 24 位，即 12 个 room 的 cost && k
        # head_data = house_data[:24]
        # 尾部 2 位为 avg_cost 和 avg_k
        tensor_var = house_data[:, :24]
        tail_data = house_data[:, -2:]

        # 找到所有为0的位置
        condition = tf.equal(tensor_var[:, 0::2], 0)
        indices_to_zero = tf.where(condition)
        # 将需要修改的数据的坐标生成，即在原有的坐标的列上加 1
        indices_to_zero_var = tf.stack(
            [indices_to_zero[:, 0], indices_to_zero[:, 1]*2 + 1], axis=1)
        tensor_var = tf.tensor_scatter_nd_update(tensor_var, indices_to_zero_var, tf.zeros(
            tf.shape(indices_to_zero_var)[0], dtype=tf.float32))  # tf.int32))

        # print("!!!!!!!!!!! tensor_var.shape ",tensor_var.shape,"tail_data.shape ",tail_data.shape)
        # print(tail_data.shape)
        # 将头部数据和尾部数据拼接起来
        house_data = tf.concat([tensor_var, tail_data], axis=1)

        return house_data

    def correct_publicspace_data(self, publicspace_data):
        # 检查每个奇数位是否为 0 ，如果为0，则相邻的偶数位也设置为 0
        condition = tf.equal(publicspace_data[:, 0::2], 0)
        indices_to_zero = tf.where(condition)
        indices_to_zero_var = tf.stack(
            [indices_to_zero[:, 0], indices_to_zero[:, 1]*2 + 1], axis=1)
        publicspace_data = tf.tensor_scatter_nd_update(publicspace_data, indices_to_zero_var, tf.zeros(
            tf.shape(indices_to_zero_var)[0], dtype=tf.float32))  # tf.int32))
        # print("publicspace_data,after correct ", publicspace_data)
        return publicspace_data

    # 修正输出函数
    def correct_data_tensor(self, data_tensor):

        # tf.print(
        #    "--------------------------correct_data_tensor begin----------------------------------------")

        # 总长度为 330 = 12*(12*2+2) + 4*2 + 4*2 +2
        # 将数据截取为 4部分，分别是12个house（每个26位），4 个staircase(每个2位)，4个 corridor（每个2位），最后2位 avg_cost 和 avg_k
        # 单个 house 的长度为 House.HOUSE_FEATURES
        # 单个 staircase 的长度为 Staircase.STAIRCASE_FEATURES
        # 单个 corridor 的长度为 Corridor.CORRIDOR_FEATURES
        # tf.print("data_tensor ",data_tensor,"data_tensor.shape ",tf.shape(data_tensor))
        # tf.print("@@@@@@------ data_tensor.shape -------@@@@@@",
        #         tf.shape(data_tensor))

        houses_tensor = data_tensor[:, :12*(12*2+2)]
        staircase_tensor = data_tensor[:, 12*(12*2+2):12*(12*2+2)+4*2]
        corridor_tensor = data_tensor[:, 12*(12*2+2)+4*2:12*(12*2+2)+4*2+4*2]

        # 对每个 house 进行修正

        houses_tensor = self.correct_house_data(houses_tensor)

        # 对 staircase 进行修正
        staircase_tensor = self.correct_publicspace_data(staircase_tensor)

        # 对 corridor 进行修正
        corridor_tensor = self.correct_publicspace_data(corridor_tensor)

        tail_tensor = data_tensor[:, -2:]
 
        # 将修正后的数据拼接起来
        data_tensor = tf.concat(
            [houses_tensor, staircase_tensor, corridor_tensor, tail_tensor], axis=-1)

        return data_tensor


if (tf.keras.saving.get_registered_object('opt_ClipLayer>BackpropLayer') == BackpropLayer):
    pass
    #ml_logger.info('BackpropLayer registered')


class HouseModelTrainer:

    def __init__(self, train_mode="continuing_training",
                 hdfs_host='http://192.168.1.19:9870', hdfs_user='root', hdfs_model_dir='/net_model'):  # retraining/continuing_training

        self.model_dir = "./net_model"
        # self.model_files = ["house_model.keras", "model_weights.keras", "x_scaler_scale.npy", "x_scaler_min.npy", "y_scaler_scale.npy", "y_scaler_min.npy"]
        self.model_files = ["house_model.h5", "model_weights.h5", "x_scaler_scale.npy", "x_scaler_min.npy",
                            "y_scaler_scale.npy", "y_scaler_min.npy", "house_model.onnx", "house_model.engine"]
        self.x_scaler = MinMaxScaler()
        self.y_scaler = MinMaxScaler()
        self.model = None
        self.houses_x = None
        self.houses_y = None
        self.train_mode = train_mode

        self.hdfs_host = hdfs_host
        self.hdfs_user = hdfs_user
        self.hdfs_model_dir = hdfs_model_dir
        self.hdfs_model_files = ["house_model.h5", "model_weights.h5", "x_scaler_scale.npy", "x_scaler_min.npy",
                                 "y_scaler_scale.npy", "y_scaler_min.npy", "house_model.onnx", "house_model.engine"]

    def load_model(self):

        f_model = f"{self.model_dir}/{self.model_files[0]}"
        if os.path.exists(f_model):
            custom_objects = {'optimizeOutputClipLayer': optimizeOutputClipLayer,
                              'ConstraintLayer': ConstraintLayer, 'BackpropLayer': BackpropLayer, 'loss': loss}

            self.model = tf.keras.models.load_model(
                f_model, custom_objects=custom_objects)

            ml_logger.info(f"\rload model file {f_model} successfully")
            return True
        ml_logger.info(f"\rload model file {f_model} failed")

        return False

    def load_hdfs_model(self):

        client = hdfs.InsecureClient(self.hdfs_host, user=self.hdfs_user)
        f_model = f"{self.hdfs_model_dir}/{self.hdfs_model_files[0]}"
        ml_logger.info("f_model == ", f_model)

        # 读取HDFS上的模型文件内容
        if client.status(f_model, strict=False):
            # 将模型内容保存到本地临时文件
            temp_file = './temp/temp_model.h5'
            f_d = client.download(
                f_model, temp_file, overwrite=True, n_threads=10, temp_dir='/tmp')
            ml_logger.info(f"client.download {f_model} - {f_d}")
            if not f_d:
                return False

            custom_object = {'optimizeOutputClipLayer': optimizeOutputClipLayer,
                             'ConstraintLayer': ConstraintLayer,
                             'BackpropLayer': BackpropLayer,
                             'loss': loss}

            self.model = tf.keras.models.load_model(
                temp_file, custom_objects=custom_object)

            ml_logger.info(
                f"\rload model file {f_model} successfully from hdfs")
            return True

        ml_logger.info(f"\rload model file {f_model} failed from hdfs")
        return False

    def generate_dataset(self, size=1000, model='reload'):  # 不再使用此函数
        # self.houses_x, self.houses_y = mdsp.generate_dataset(200)
        # self.houses_x, self.houses_y = mdsp.generate_dataset(size=100,model='create') # create/reload
        # self.houses_x, self.houses_y = mdsp.generate_dataset(size=size,model=model) # create/reload
        rpc_proxy = pool.get_connection()
        self.houses_x, self.houses_y = rpc_proxy.DataSetService.get_dataset(
            size=size, model=model)
        pool.put_connection(rpc_proxy)

    # 从本地文件(pkl)读取数据集,pkl 是通过 DataSetGenerator 生成
    def generate_dataset_v2(self, dataset_dir):

        self.houses_x = tf.cast(load_dataset(
            f'{dataset_dir}/data_set_x.pkl'), dtype=tf.float64)
        self.houses_x = tf.where(tf.math.is_nan(
            self.houses_x), 0.0, self.houses_x)
        self.houses_x = self.houses_x.numpy()
        self.houses_y = tf.cast(load_dataset(
            f'{dataset_dir}/data_set_y.pkl'), dtype=tf.float64)
        self.houses_y = tf.where(tf.math.is_nan(
            self.houses_y), 0.0, self.houses_y)
        self.houses_y = self.houses_y.numpy()

        ml_logger.info(
            f"self.houses_x shape {self.houses_x.shape} ,self.houses_y shape {self.houses_y.shape} ")

    def split_data(self, test_size=0.2, validation_size=0.2):

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.houses_x, self.houses_y, test_size=test_size)
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(
            self.x_train, self.y_train, test_size=validation_size)

    def preprocess_data(self):

        self.x_train_scaled = self.x_scaler.fit_transform(self.x_train)
        self.x_val_scaled = self.x_scaler.transform(self.x_val)
        self.x_test_scaled = self.x_scaler.transform(self.x_test)
        self.y_train_scaled = self.y_scaler.fit_transform(self.y_train)
        self.y_val_scaled = self.y_scaler.transform(self.y_val)
        self.y_test_scaled = self.y_scaler.transform(self.y_test)

    def create_model(self, dropout_rate=0.025, l2_lambda=0.01):

        # dropout_rate = 0.025  # 增加Dropout比率
        # l2_lambda = 0.001  # L2正则化的lambda参数
        x_length = len(self.houses_x[0])
        y_length = len(self.houses_y[0])

        ml_logger.info(
            f"create model , x.length {x_length} y.length {y_length} ")

        # reload model
        load = False
        if self.train_mode == "continuing_training":
            # load = self.load_model()
            load = self.load_hdfs_model()

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
            # 将输入形状改为 y_length
            # self.model.add(Dense(y_length, activation='relu'))
            # self.model.add(ConstraintLayer())
            self.model.add(Dense(y_length, activation='relu'))
            # 修正0问题
            self.model.add(BackpropLayer())  # 添加自定义反向传播层 ,
            self.model.add(Dense(y_length, activation='relu'))

    # 编译模型

    def compile_model(self, optimizer='adam', loss='mse'):

        # optimizer = tf.optimizers.Adam(0.001 * hvd.size())
        # self.model.compile(optimizer=optimizer, loss=loss, metrics=[tf.keras.metrics.RootMeanSquaredError(), "mae"])
        # self.model.compile(optimizer=optimizer, loss=loss ,metrics=['accuracy'])
        # self.model.compile(optimizer=optimizer, loss=loss ,metrics=['accuracy', tf.keras.metrics.RootMeanSquaredError(), "mae"])
        optimizer = hvd.DistributedOptimizer(optimizer)
        self.model.compile(optimizer=optimizer, loss=loss, metrics=[
                           tf.keras.metrics.RootMeanSquaredError(), "mae"])
        # 模型序列化
        return (serialize_model(self.model))

    def train_model(self, model_bytes=[], epochs=200, batch_size=50, initial_lr=0.001,
                    ckpt_file='./check_point/checkpoint.h5', verbose=1, callbacks=None, patience=30):

        # 在模型训练中添加早停法
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=patience)  # patience参数是能够容忍多少个epoch内都没有improvement

        append_callbacks = [
            # 从rank 0进程广播初始变量状态到所有其他进程。这是为了确保在使用随机权重启动训练或从检查点还原训练时，所有的工作进程都有一致的初始化
            hvd.callbacks.BroadcastGlobalVariablesCallback(root_rank=0),
            # 在每个epoch结束时，将所有工作进程的指标（metrics）进行平均，以确保在分布式训练中得到一致的度量结果。这对于监控和记录训练进程非常重要
            hvd.callbacks.MetricAverageCallback(),
            # 在训练的最初几个epoch内逐步增加学习率。这种策略有助于防止使用较大的学习率启动训练时的不稳定性，尤其是在分布式环境中。
            # 通常，学习率在前几个epoch中会逐渐增加，然后保持不变
            hvd.callbacks.LearningRateWarmupCallback(
                initial_lr=initial_lr * hvd.size(), warmup_epochs=5, verbose=verbose),
            # 监视指定的验证指标（val_exp_rmspe）并在指标在一定连续epoch内没有改善时，减小学习率。patience参数指定了在多少个epoch内没有改善后降低学习率
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', patience=10, verbose=verbose),
            # 监视指定的验证指标（val_exp_rmspe）并在指标在一定连续epoch内没有改善时停止训练。mode参数指定了监视的模式，min表示指标要降低才算改善
            early_stopping,
            # 监控训练中是否存在NaN（不是一个数字）的情况。如果训练中出现NaN值，它会终止训练，以防止模型的数值不稳定性。
            tf.keras.callbacks.TerminateOnNaN(),
            # ModelCheckpoint 回调
            ModelCheckpoint(
                filepath=ckpt_file,  # ckpt_dir,
                verbose=verbose,
                save_weights_only=True,
                monitor='val_loss',
                mode='min',
                save_best_only=True)
            # ModelCheckpoint(ckpt_file, monitor='val_loss',save_best_only=False) #, save_best_only=True, mode='min')
        ]
        if callbacks is not None:
            callbacks.append(append_callbacks)
        else:
            callbacks = append_callbacks

        custom_objects = {'optimizeOutputClipLayer': optimizeOutputClipLayer,
                          'ConstraintLayer': ConstraintLayer,
                          'BackpropLayer': BackpropLayer,
                          'loss': loss}

        self.model = deserialize_model(
            model_bytes, hvd.load_model, custom_objects)
        history = self.model.fit(self.x_train_scaled, self.y_train_scaled, epochs=epochs, batch_size=batch_size,  # verbose=verbose,
                                 verbose=1 if hvd.rank() == 0 else 0,  # 只在第一个进程上显示输出, rank = 0
                                 callbacks=callbacks, validation_data=(self.x_val_scaled, self.y_val_scaled))

        # 如果任何一个节点触发早停，停止所有节点的训练
        if early_stopping.stopped_epoch is not None:
            ml_logger.info(
                "Early stopping triggered on node {}. Stopping all nodes.".format(hvd.rank()))
            hvd.allreduce(tf.constant(1, dtype=tf.int32), op=None)  # 发送停止信号

        return min(history.history['val_loss'])

    # 评估模型
    def evaluate_model(self):

        test_loss = self.model.evaluate(self.x_test_scaled, self.y_test_scaled)

        rmse_metric = tf.keras.metrics.RootMeanSquaredError()
        y_pred = self.model.predict(self.x_test_scaled)
        rmse_value = rmse_metric(self.y_test_scaled, y_pred)

        mae_metric = tf.keras.metrics.MeanAbsoluteError()
        mae_value = mae_metric(self.y_test_scaled, y_pred)

        # print("Test loss:", round(test_loss[0], 6), "Test accuracy:", round(test_loss[1], 6), "RMSE:", round(rmse_value.numpy(), 6), "MAE:", round(mae_value.numpy(), 6))
        ml_logger.info("Test loss: ", round(test_loss[0], 6), " RMSE: ", round(
            rmse_value.numpy(), 6), " MAE: ", round(mae_value.numpy(), 6))
        y_true = self.y_test_scaled
        y_pred = self.model.predict(self.x_test_scaled)


    def save_model(self, hdfs_url='http://192.168.1.19:9870', hdfs_user='root'):
        def upload_to_hadoop(hdfs_path, local_path):
            client = hdfs.InsecureClient(hdfs_url, hdfs_user)
            client.upload(hdfs_path, local_path, cleanup=True, overwrite=True)
            ml_logger.info(f"Upload file {local_path} to HDFS successfully.")

        def backup_to_local(model_dir, model_files, backup_dir):
            if any(os.path.exists(os.path.join(model_dir, file)) for file in model_files):
                # Create backup directory with timestamp
                os.makedirs(backup_dir, exist_ok=True)
                for file in model_files:
                    src_file_path = os.path.join(model_dir, file)
                    if os.path.exists(src_file_path):
                        # Move the file to backup directory
                        shutil.move(src_file_path, backup_dir)
                ml_logger.info(
                    f"Model-Files backup to LOCAL at {backup_dir} successfully.")

        def backup_to_hdfs(hdfs_model_dir, model_files, hdfs_backup_dir):
            client = hdfs.InsecureClient(hdfs_url, hdfs_user)
            if any(client.status(os.path.join(hdfs_model_dir, file), strict=False) for file in model_files):
                # Create backup directory in HDFS with timestamp
                hdfs_backup_path = os.path.join(
                    hdfs_backup_dir, 'backup_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
                client.makedirs(hdfs_backup_path)
                for file in model_files:
                    hdfs_file_path = os.path.join(hdfs_model_dir, file)
                    if client.status(hdfs_file_path, strict=False):
                        hdfs_backup_file_path = os.path.join(
                            hdfs_backup_path, file)
                        # Copy the file to HDFS backup directory
                        with client.read(hdfs_file_path) as src_file, client.write(hdfs_backup_file_path) as dest_file:
                            for chunk in src_file:
                                dest_file.write(chunk)
                ml_logger.info(
                    f"Model-Files backup to HDFS at {hdfs_backup_path} successfully.")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = os.path.join(self.model_dir, timestamp)

        # Move files to a new directory
        backup_to_local(self.model_dir, self.model_files, backup_dir)
        backup_to_hdfs(self.hdfs_model_dir,
                       self.model_files, self.hdfs_model_dir)

        # Loop through all files and save them
        for i, f in enumerate(self.model_files):
            local_path = os.path.join(self.model_dir, f)

            if i == 0:
                # save model file
                self.model.save(local_path)
                self.model.save("./net_model/tf_model", "tf")
                # save_options = tf.saved_model.SaveOptions(namespace_whitelist=['IO'])
                # tf.saved_model.save(self.model, "./net_model/tf_model")
                onnx_model, _ = tf2onnx.convert.from_keras(self.model)
                onnxmltools.utils.save_model(
                    onnx_model, './net_model/house_model.onnx')
                ml_logger.info("convert to house_model.onnx successfully")
                import subprocess
                # 定义命令
                command = [
                    'trtexec',              # 命令名称
                    '--onnx=./net_model/house_model.onnx',   # ONNX模型文件路径
                    '--saveEngine=./net_model/house_model.engine',  # 保存的TensorRT引擎文件路径
                    # '--useCudaGraph',
                    '--useSpinWait',
                    '--fp16'                # 使用FP16精度
                ]
                # 使用subprocess运行命令
                subprocess.run(command)

            elif i == 1:
                # save weights file
                self.process_weights(local_path)
            elif i == 2:
                np.save(local_path, self.x_scaler.scale_)
            elif i == 3:
                np.save(local_path, self.x_scaler.min_)
            elif i == 4:
                np.save(local_path, self.y_scaler.scale_)
            elif i == 5:
                np.save(local_path, self.y_scaler.min_)

            # Upload to Hadoop
            hdfs_path = os.path.join(self.hdfs_model_dir, f)

            upload_to_hadoop(hdfs_path, local_path)

    def process_weights(self, output_file):
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
        activation_model = tf.keras.models.Model(
            inputs=self.model.input, outputs=layer_outputs)
        activations = activation_model.predict(self.x_val_scaled)

        for i, activation in enumerate(activations):
            with open(f'{self.log_dir}/activation_{i}_epoch_{epoch}.npy', 'wb') as f:
                np.save(f, activation)


def main(args):

    gpu_memory_limit = max(4096, min(args.gpu_memory_limit, 8192))

    # 获取系统中 GPU 的内存信息
    gpus = GPUtil.getGPUs()
    if gpus:
        available_gpu_memory = gpus[0].memoryFree  # 假设只有一块 GPU，获取其可用内存大小
        warning_threshold = 0.9  # 警告阈值，即 90%

        if available_gpu_memory * warning_threshold < gpu_memory_limit:
            ml_logger.info(
                "Error: Available GPU memory is less than 90% of the specified limit. Exiting.")
            sys.exit(1)

    # gpu_memory_limit = int(os.getenv('GPU_MEMORY_LIMIT', '4096'))

    # 配置gpu 及内存使用大小
    configurator = GPUConfigurator(
        use_gpu=True, gpu_memory_limit=gpu_memory_limit)
    configurator.configure_gpu()
    tf.device(configurator.select_device())

    # retraining/continuing_training
    trainer = HouseModelTrainer(
        train_mode=args.train_mode,
        hdfs_host=args.hdfs_url,
        hdfs_user=args.hdfs_user,
        hdfs_model_dir=args.hdfs_model_dir
    )

    # trainer.generate_dataset(size=args.dataset_size, model=args.create_dataset_method)
    trainer.generate_dataset_v2(args.dataset_dir)
    trainer.split_data()
    trainer.preprocess_data()
    trainer.create_model(dropout_rate=args.dropout_rate,
                         l2_lambda=args.l2_lambda)

    # 创建TensorBoard回调
    log_dir = os.path.join(args.net_log_dir, time.strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = TensorBoard(
        log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True)

    # 创建激活日志记录器回调
    activation_logger = ActivationLogger(log_dir, trainer.x_val_scaled)

    # trainer.compile_model(optimizer='adam', loss=loss)
    # trainer.compile_model(optimizer=HvdAdam(learning_rate=args.learning_rate), loss=loss)

    # mpi ,要求模型序列化
    model_bytes = trainer.compile_model(optimizer=tf.optimizers.Adam(
        args.learning_rate * hvd.size()), loss=loss)

    ml_logger.info(
        f"############################################### hvd.rank {hvd.rank()} training model ###############################################")
    if trainer.train_model(model_bytes=model_bytes,
                           epochs=args.epochs,
                           batch_size=args.batch_size,
                           initial_lr=args.learning_rate,
                           ckpt_file=args.check_point_file,
                           verbose=args.verbose,
                           callbacks=[tensorboard_callback, activation_logger],
                           patience=args.patience):

        ml_logger.info(
            f"############################################### hvd.rank {hvd.rank()} evaluate model ###############################################")
        # eval model
        trainer.evaluate_model()

        ml_logger.info(
            f"############################################### hvd.rank {hvd.rank()} saving model ###############################################")
        # 仅在server rank 在上保存模型
        if hvd.rank() == 0:
            trainer.save_model(args.hdfs_url, args.hdfs_user)
    else:
        ml_logger.info("train model failed")


if __name__ == "__main__":

    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='House Model Trainer')

    # 添加命令行参数
    parser.add_argument('--train_mode', type=str, required=True,
                        choices=['retraining', 'continuing_training'], help='Train mode')
    parser.add_argument('--gpu_memory_limit', type=int,
                        required=True, help='size of limited gpu memory')
    parser.add_argument('--dataset_size', type=int,
                        required=True, help='DataSet size')
    parser.add_argument('--dataset_dir', type=str,
                        required=True, help='dir of DataSet')
    parser.add_argument('--create_dataset_method', type=str, required=True,
                        choices=['reload', 'create'], help='Create dataset method')
    parser.add_argument('--dropout_rate', type=float,
                        required=True, help='Dropout rate')
    parser.add_argument('--l2_lambda', type=float,
                        required=True, help='L2 regularization lambda')
    parser.add_argument('--learning_rate', type=float,
                        required=True, help='Learning rate')
    parser.add_argument('--epochs', type=int, required=True,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int,
                        required=True, help='Batch size')
    parser.add_argument('--patience', type=int,
                        required=True, help='Number of patience')
    parser.add_argument('--net_log_dir', type=str,
                        required=True, help='Dir of net logs')
    parser.add_argument('--verbose', type=int, required=True, help='verbose')
    parser.add_argument('--hdfs_url', type=str,
                        required=True, help='url of hdfs')
    parser.add_argument('--hdfs_user', type=str,
                        required=True, help='user of hdfs')
    parser.add_argument('--hdfs_model_dir', type=str,
                        required=True, help='dir of hdfs model')
    parser.add_argument('--check_point_file', type=str,
                        required=True, help='check point file path')

    # 解析命令行参数
    args = parser.parse_args()

    # 打印参数名和值
    for arg, value in args.__dict__.items():
        ml_logger.info(f"Argument: {arg}, Value: {value}")

    # 调用主函数
    main(args)
