import tensorflow as tf
import time
import traceback,inspect

class GPUConfigurator:
    def __init__(self, use_gpu=True, gpu_memory_limit=None):
        self.use_gpu = use_gpu
        self.gpu_memory_limit = gpu_memory_limit
    
    def configure_gpu(self):
        if self.use_gpu:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            #print("gpus :",gpus )
            if gpus:
                try:
                    for gpu in gpus:
                        if self.gpu_memory_limit:
                            tf.config.experimental.set_virtual_device_configuration(
                                gpu,
                                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=self.gpu_memory_limit)]
                            )
                            #print("use gpu :",gpu ,"gpu_memory_limit :",self.gpu_memory_limit)
                        else:
                            tf.config.experimental.set_memory_growth(gpu, True)
                            #print("use gpu :",gpu ,"gpu_memory_limit :",None)
                except RuntimeError as e:
                    print(e)
            else:
                print("No GPU devices available.")
    
    def select_device(self):
        if self.use_gpu and tf.config.experimental.list_physical_devices('GPU'):
            #print("use gpu : /GPU:0, gpu_memory_limit :",self.gpu_memory_limit)
            return '/GPU:0'
        else:
            #print("use cpu : /CPU:0 ")
            return '/CPU:0'

def main():
    # 创建一个GPUConfigurator实例并配置GPU使用
    configurator = GPUConfigurator(use_gpu=True, gpu_memory_limit=4096)
    configurator.configure_gpu()
    for i in range(100):
    # 创建一个简单的TensorFlow会话以验证配置
        with tf.device('/GPU:0'):
            a = tf.constant([1.0, 2.0, 3.0])
            b = tf.constant([4.0, 5.0, 6.0])
            c = a + b

            print("Result using GPU in iteration", i + 1, ":", c)

       # 每次迭代后休眠10秒
        time.sleep(10)

if __name__ == "__main__":
    main()
