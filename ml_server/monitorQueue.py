import argparse
import sys
sys.path.append("..")
from apocolib import RpcProxyPool
from apocolib.MQueueManager import MQueueManager

pool = RpcProxyPool.RpcProxyPool()

if __name__ == "__main__":
    rpc_proxy = pool.get_connection()
    result = rpc_proxy.AC2NNetPredicterService.get_predict_queue_info()
    pool.put_connection(rpc_proxy)
    
    print("---------------------------------------------------------------------------------------------")
    print("Queue Info:")
    print(f"Current Size: {result['current_size']}")
    print(f"Max Size: {result['max_size']}")
    
    processes = result['processes']
    print("\nProcesses:")
    for process in processes:
        print("---------------------------------------------------------------------------------------------")
        print(f"PID: {process['pid']}")
        print(f"Input File: {process['input_file']}")
        print(f"Output File: {process['output_file']}")
        print(f"Queue Name: {process['queue_name']}")
        print(f"Task ID: {process['task_id']}")
    
    mq_manager = MQueueManager()
    mq_status = mq_manager.get_all_queue_status()
    print("\nMQ Status:")
    for queue_name, status in mq_status.items():
        print(f"Queue Name: {queue_name} | Status: {status}")

