import redis
import yaml
import threading

import sys
sys.path.append("..")
#from apocolib.apocolog4p import apoLogger as apolog
from apocolib import apocoIAServerConfigurationManager as iaConMg

class RedisPool:
    """
    Redis连接池
    """
    def __init__(self, host, port, password, db, max_connections):
        self.pool = redis.ConnectionPool(
            host=host,
            port=port,
            password=password,
            db=db,
            max_connections=max_connections
        )
        self.lock = threading.Lock()  # 添加锁

    def get_connection(self):
        """
        获取连接
        """
        self.lock.acquire()  # 获取锁
        try:
            #apolog.info('get redis connection')
            return redis.Redis(connection_pool=self.pool)
        except redis.RedisError as e:
            #apolog.error(f"获取Redis连接失败：{str(e)}")
            raise e
        finally:
            self.lock.release()  # 释放锁

    def release_connection(self, conn):
        """
        释放连接
        """
        self.lock.acquire()  # 获取锁
        try:
            if conn:
                conn.close()
                #apolog.info('released redis connection')
        except redis.RedisError as e:
            #apolog.error(f"释放Redis连接失败：{str(e)}")
            raise e
        finally:
            self.lock.release()  # 释放锁

class redisConnectionPool:

    @staticmethod
    def pool(): 
        try:
            redis_pool = RedisPool(
                host = iaConMg.redis_host,
                port = iaConMg.redis_port,
                password = iaConMg.redis_password,
                db=0,
                max_connections = 20
            )
            return redis_pool
        except Exception as e:
            raise e
            #apolog.error(f"发生异常：{str(e)}")
        return None
