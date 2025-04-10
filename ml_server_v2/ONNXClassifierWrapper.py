import numpy as np
import tensorflow as tf
import tensorrt as trt

import pycuda.driver as cuda
import pycuda.autoinit

# For ONNX:

class ONNXClassifierWrapper():
    '''
    	file: tensorrt 文件路径， number_classes:[BATCH_SIZE, 1000], target_dtype: 识别精度
    '''
    def __init__(self, file, num_classes, target_dtype = np.float32):
        
        self.target_dtype = target_dtype
        self.num_classes = num_classes
        self.load(file)
        
        self.stream = None
      
    def load(self, file):
        f = open(file, "rb")
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING)) 

        engine = runtime.deserialize_cuda_engine(f.read())
        self.context = engine.create_execution_context()
        
    def allocate_memory(self, batch):
        self.output = np.empty(self.num_classes, dtype = self.target_dtype) # Need to set both input and output precisions to FP16 to fully enable FP16

        # Allocate device memory
        self.d_input = cuda.mem_alloc(1 * batch.nbytes)
        self.d_output = cuda.mem_alloc(1 * self.output.nbytes)

        self.bindings = [int(self.d_input), int(self.d_output)]

        self.stream = cuda.Stream()
        
    def predict(self, batch): # result gets copied into output
        if self.stream is None:
            self.allocate_memory(batch)
            
        # Transfer input data to device
        cuda.memcpy_htod_async(self.d_input, batch, self.stream)
        # Execute model
        self.context.execute_async_v2(self.bindings, self.stream.handle, None)
        # Transfer predictions back
        cuda.memcpy_dtoh_async(self.output, self.d_output, self.stream)
        # Syncronize threads
        self.stream.synchronize()
        
        return self.output

def convert_onnx_to_engine(onnx_filename, engine_filename , max_workspace_size = 1 << 30):

    logger = trt.Logger(trt.Logger.VERBOSE) #WARNING
    trt.init_libnvinfer_plugins(logger, namespace="")
    builder = trt.Builder(logger)

    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, max_workspace_size) # 1 MiB

    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    success = parser.parse_from_file(onnx_filename)
    for idx in range(parser.num_errors):
        print(parser.get_error(idx))

    if not success:
        return False
    

    serialized_engine = builder.build_serialized_network(network, config)

    # 判断 serialized_engine 是否为空
    if serialized_engine is None or len(serialized_engine) == 0:
        print("Serialized engine is empty")
        return False
    with open(engine_filename, 'wb') as f:
        f.write(serialized_engine)

    return True

if __name__ == '__main__':

    convert_onnx_to_engine("./net_model/house_model.onnx","./net_model/house_model.engine")
