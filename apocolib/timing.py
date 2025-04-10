import time
import inspect
from apocolib.MlLogger import mlLogger as ml_logger

def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time

        # 获取调用函数的文件名和类名
        frame = inspect.currentframe().f_back
        file_name = frame.f_code.co_filename
        class_name = None

        # 如果函数是类方法，则获取类名
        if 'self' in frame.f_locals:
            class_name = frame.f_locals['self'].__class__.__name__

        # 格式化时间，精确到小数点后4位
        elapsed_time_formatted = f"{elapsed_time:.4f}"

        # 构造日志信息
        # log_message = f"{file_name} - {class_name} - {func.__name__} Execution Time: {elapsed_time_formatted} seconds"
        log_message = f"{class_name} - {func.__name__} Execution Time: {elapsed_time_formatted} s"
        #ml_logger.debug(log_message)

        return result

    return wrapper
