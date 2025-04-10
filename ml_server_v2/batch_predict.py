import sys
import time
import schedule
import json
import threading

sys.path.append("..")
from apocolib import RpcProxyPool

pool = RpcProxyPool.RpcProxyPool()
files_list = "files.list"  # 文件名列表

class Predictor:
    def __init__(self):
        self.is_running = True

    def predict(self, filename):
        rpc_proxy = pool.get_connection()
        errorCode, msg = rpc_proxy.AC2NNetPredicterService.predict(filename)
        pool.put_connection(rpc_proxy)

    def batch_predict(self):
        with open(files_list, "r") as f:
            filenames = f.read().splitlines()
        for filename in filenames:
            self.predict(filename)
            print(f'start predict {filename} successfully')
            #time.sleep(8)

    def get_predict_queue_info(self):
        rpc_proxy = pool.get_connection()
        data = rpc_proxy.AC2NNetPredicterService.get_predict_queue_info()
        pool.put_connection(rpc_proxy)
        return data

def print_result(data):
    lines = json.dumps(data, indent=4).splitlines()
    for line in lines:
        print(line)

if __name__ == "__main__":
    predictor = Predictor()
    predictor.batch_predict()
