import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import pdb
from nameko.rpc import rpc
import pickle
import time
import sys
import os
import tensorflow as tf

sys.path.append("..")
from apocolib.GPUConfigurator import GPUConfigurator
from BuildingElement import Floor
gpu_memory_limit = int(os.getenv('GPU_MEMORY_LIMIT', '4096'))
# 配置gpu 及内存使用大小
configurator = GPUConfigurator(use_gpu=True, gpu_memory_limit=gpu_memory_limit)
configurator.configure_gpu()
tf.device(configurator.select_device())

engine_file_path = './net_model/house_model.engine'
input_shape = (1,Floor.FEATURES_LEN)

trt_logger = trt.Logger(trt.Logger.INFO)#trt.Logger(trt.Logger.INFO) #VERBOSE)) #WARNING))

class tensorRT_Service:

    name = "tensorRT_Service"

    def __init__(self):

        # 加载TensorRT引擎文件
        with open(engine_file_path, "rb") as engine_file:
            self.runtime = trt.Runtime(trt_logger) 
            self.engine = self.runtime.deserialize_cuda_engine(engine_file.read())

        # 创建执行上下文
        self.context = self.engine.create_execution_context()

        # 准备输入数据
        #self.input_name = input_name
        self.input_shape = input_shape

        # 获取输出维度的数量
        self.output_dims = self.engine.get_binding_shape(1)
        #self.output_dims = self.engine.get_tensor_shape()
        self.output_size = 1
        for dim in self.output_dims:
            self.output_size *= dim

        # 创建CUDA内存和推理流
        self.d_input = cuda.mem_alloc(self.input_shape[0] * self.input_shape[1] * 4)  # Assuming float32 input
        self.d_output = cuda.mem_alloc(self.output_size*4) #float32 need 4bytes
        self.stream = cuda.Stream()

        # 初始化输出数据为None
        self.output_data = None

    @rpc
    def infer(self, input_data):

        start_time = time.perf_counter()  # 记录开始时间

        input_data = pickle.loads(input_data) # 反序列化
        
        print("Infer input_data  ",input_data)
        print("Infer input_data shape ",input_data.shape )

        if not isinstance(input_data, np.ndarray):
            raise ValueError("input_data should be a NumPy array.")

        # 将输入数据传输到GPU内存

        cuda.memcpy_htod_async(self.d_input, input_data.astype(np.float32), self.stream)

        # 设置输入维度
        bindings = [int(self.d_input), int(self.d_output)]
        # 执行推理
        result = self.context.execute_async_v2([int(self.d_input), int(self.d_output)], self.stream.handle,None)
        #print("reslut ",result)
        self.output_data = np.empty(self.output_size, dtype=np.float32)  # 初始化输出数据数组
        cuda.memcpy_dtoh_async(self.output_data, self.d_output, self.stream)
        self.stream.synchronize()

        # 转换输出数据为float64类型并处理空字符串
        converted_output = np.zeros(self.output_data.shape, dtype=np.float32)
        for i in range(self.output_data.size):
            original_value = self.output_data[i]
            if isinstance(original_value, str):
                if original_value.strip():  # 非空字符串
                    converted_value = float(original_value)
                else:  # 空字符串
                    converted_value = 0.0
            else:  # 值不是字符串，保持原样
                converted_value = original_value
            converted_output[i] = converted_value

        #print("converted_output ",converted_output)
        end_time = time.perf_counter()  # 记录结束时间
        elapsed_time = end_time - start_time  # 计算执行时间
        print(f"tensorRT - infer : {elapsed_time:.2f} seconds")

        return pickle.dumps(converted_output) # 返回序列化数据
