from nameko.rpc import rpc,RpcProxy
import sys
import jwt
import datetime
sys.path.append("..")
#from apocolib.apocolog4p import apoLogger as apolog
from apocolib.NamekoLogger import namekoLogger as nameko_logger
from apocolib import apocoIAServerConfigurationManager as iaConMg
from apocolib.RedisPool import redisConnectionPool as rcPool

import redis
import nanomsg
import msgpack

class CollectorRegistryService:

    name = 'CollectorRegistryService'

    def __init__(self):

        self.SECRET_KEY = iaConMg.mail_jwt_secret_key
        self.JWT_EXPIRATION_DELTA = iaConMg.mail_jwt_expiration_delta
        self.redis_conn = None

    def register_collector(self, server_name, public_ip, private_ip):
        try:

            #--------- saved into redis ------------------#
            """
            将采集器信息注册到Redis中
            """
            collector_info = {
                'server_name': server_name,
                'public_ip': public_ip,
                'private_ip': private_ip
            }
            redis_conn = rcPool.pool().get_connection()
            # 将采集器信息以哈希表的形式存储到Redis中
            redis_key = 'collectors'
            self.redis_conn.hset(redis_key, server_name, msgpack.packb(collector_info))
            redis_conn.expire(token, self.JWT_EXPIRATION_DELTA)
        except Exception as e:
            nameko_logger.error(f'reset passord over email failed {email} error.{str(e)}')
            return -1, f'reset passord over email failed {email} error.{str(e)}'
        finally:
            if redis_conn: 
                rcPool.pool().release_connection(redis_conn)

        print("Collector registered successfully!")

    def unregister_collector(self, server_name):
        """
        从Redis中注销采集器信息
        """
        redis_key = 'collectors'
        self.redis_client.hdel(redis_key, server_name)

        print("Collector unregistered successfully!")
