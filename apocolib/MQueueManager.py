import redis
import argparse
import sys
import time
sys.path.append("..")
from apocolib.RedisPool import redisConnectionPool as rcPool
from apocolib.MlLogger import mlLogger as ml_logger
#import apocolib.redisConnectionPool as rcPool
#import apocolib.mlLogger as ml_logger

class MQueueManager:
    def __init__(self, mq_size=8):

        self.mq_size = mq_size
        self.redis_conn = rcPool.pool().get_connection()

        # 检查是否已经进行过初始化
        if not self.redis_conn.exists('initialized'):
            # 如果还没有进行过初始化，那么设置队列状态并添加初始化标志
            for i in range(1, self.mq_size):
                self.redis_conn.hset('queues', f'queue_{i}', 'free')
            self.redis_conn.set('initialized', 'true')

            ml_logger.info("MQ initialized,size {self.mq_size}")

        self.lock_key = 'queue_lock'
        self.lock_timeout = 10


    def acquire_lock(self):
        """获取分布式锁"""
        while True:
            acquired = self.redis_conn.set(self.lock_key, 'locked', nx=True, ex=self.lock_timeout)
            if acquired:
                return True
            else:
                time.sleep(0.1)

    def release_lock(self):
        """释放分布式锁"""
        self.redis_conn.delete(self.lock_key)

    def allocate_queue(self):
        """为任务分配空闲队列"""
        if self.acquire_lock():
            try:
                for queue_name in self.redis_conn.hkeys('queues'):
                    if self.redis_conn.hget('queues', queue_name) == b'free':
                        self.redis_conn.hset('queues', queue_name, 'busy')
                        ml_logger.info("allocate_queue {}".format(queue_name.decode()))
                        return queue_name.decode()
                ml_logger.error("allocate_queue error. Have no free queue")
                return None
            finally:
                self.release_lock()

    def release_queue(self, queue_name):
        """释放已使用的队列"""
        if self.acquire_lock():
            try:
                if self.redis_conn.hget('queues', queue_name) == b'busy':
                    self.redis_conn.hset('queues', queue_name, 'free')
                    return True
                else:
                    return False
            finally:
                self.release_lock()

    '''
    def allocate_queue(self):
        """为任务分配空闲队列"""
        for queue_name in self.redis_conn.hkeys('queues'):
            if self.redis_conn.hget('queues', queue_name) == b'free':
                self.redis_conn.hset('queues', queue_name, 'busy')
                ml_logger.info("allocate_queue {}".format(queue_name.decode()))
                return queue_name.decode()
        ml_logger.error("allocate_queue error.Have no free queue")
        return None

    def release_queue(self, queue_name):
        """释放已使用的队列"""
        if self.redis_conn.hget('queues', queue_name) == b'busy':
            self.redis_conn.hset('queues', queue_name, 'free')
            return True
        else:
            return False
    '''

    def close_connection(self):
        """关闭Redis连接"""
        if self.redis_conn:
                rcPool.pool().release_connection(self.redis_conn)

    def reset_initialized(redis_pool):
        redis_conn = redis_pool.get_connection()
        redis_conn.delete('initialized')
        redis_pool.release_connection(redis_conn)

    def get_all_queue_status(self):
        """获取所有队列的名称和状态"""
        queue_status = {}
        queues = self.redis_conn.hkeys('queues')
        for queue_name in queues:
            queue_name = queue_name.decode()
            status = self.redis_conn.hget('queues', queue_name).decode()
            queue_status[queue_name] = status
        return queue_status

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Manage MQ queues.')
    parser.add_argument('-r', '--reinitialize', action='store_true',
                        help='Reinitialize all queues')
    parser.add_argument('-a', '--release-all', action='store_true',
                        help='Release all queues')
    parser.add_argument('-i', '--release-id', type=int,
                        help='Release queue with specified id')

    args = parser.parse_args()

    mq_manager = MQueueManager(8)

    if args.reinitialize:
        mq_manager.redis_conn.delete('initialized')
        mq_manager.redis_conn.delete('queues')
        mq_manager = MQueueManager(8)  # Reinitialize
        ml_logger.info("MQ reinitialized,size {self.mq_size}")

    elif args.release_all:
        for i in range(1, mq_manager.mq_size):
            mq_manager.release_queue(f'queue_{i}')
        ml_logger.info("All queues released")

    elif args.release_id is not None:
        mq_manager.release_queue(f'queue_{args.release_id}')
        ml_logger.info(f"Queue_{args.release_id} released")
    else:
        print("Please provide an argument. Use -h for help.")
