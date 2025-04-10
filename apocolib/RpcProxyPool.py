# coding=utf-8
from nameko.standalone.rpc import ClusterRpcProxy
#import settings
import threading
import queue
from nameko.standalone.rpc import (ClusterProxy, ClusterRpcProxy)
import sys

sys.path.append("..")
#from apocolib import apocoIAServerConfigurationManager as confMng
from apocolib.ConfigManager import ConfigManager,ConfigInitException
config = ConfigManager().config_mq

print(config)

def synchronized(func):

    func.__lock__ = threading.Lock()

    def lock_func(*args, **kwargs):
        with func.__lock__:
            return func(*args, **kwargs)
    return lock_func


class RpcProxyPool:
    queue = queue.Queue()

    @synchronized
    def get_connection(self) -> ClusterProxy:
        if self.queue.empty():
            conn = self.create_connection()
            self.queue.put(conn)
        return self.queue.get()

    def init_rpc_proxy(self):
        return ClusterRpcProxy(config)

    @synchronized
    def create_connection(self) -> ClusterProxy:
        _rpc_proxy: ClusterRpcProxy = self.init_rpc_proxy()
        rpc_proxy: ClusterProxy = _rpc_proxy.start()

        return rpc_proxy

    @synchronized
    def put_connection(self, conn: ClusterProxy) -> bool:
        if isinstance(conn, ClusterProxy):
            self.queue.put(conn)
            return True
        return False
