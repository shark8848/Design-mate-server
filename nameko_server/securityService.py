# securityService.py

import yaml
from nameko.rpc import rpc
from sqlalchemy import create_engine, Column, Integer, String, Enum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import hashlib
import sys
import redis
import base64
import jwt
import datetime

sys.path.append("..")
#from apocolib.apocolog4p import apoLogger as apolog
from apocolib.NamekoLogger import namekoLogger as nameko_logger
from apocolib import apocoIAServerConfigurationManager as iaConMg
from apocolib import sqliteSession as sqlSession
from apocolib.RedisPool import redisConnectionPool as rcPool
from urpfModel import User


class securityService:
    name = 'securityService'

    SECRET_KEY = iaConMg.mail_jwt_secret_key
    JWT_EXPIRATION_DELTA = iaConMg.mail_jwt_expiration_delta
    algorithms = iaConMg.algorithms
    # 初始化 Redis 客户端
    redis_host = iaConMg.redis_host
    redis_port = iaConMg.redis_port
    redis_password = iaConMg.redis_password
    jwt_redis = redis.Redis(host=redis_host, port=redis_port, password=redis_password)

    @rpc
    def authenticate(self, token):
#        clientip = request.headers.get('X-Forwarded-For', request.remote_addr)
#        nameko_logger.info(f'token_required ,client ip {clientip}')
        redis_conn = None
        try:
            '''
            if 'Authorization' in request.headers:
                nameko_logger.info(f"Authorization {request.headers['Authorization']}")
                token = request.headers['Authorization'].split(' ')[1]
'''
            if not token:
                return -1,{}, 'Token is missing!'

            nameko_logger.info(f' receive token {token} ')
            
            data = jwt.decode(token, self.SECRET_KEY, algorithms=[self.algorithms]) # 过期或者无效，在此过程抛出异常,ExpiredSignatureError,InvalidTokenError
            nameko_logger.info(f'token_required jwt.decode {data}')
#            expiry_date = datetime.datetime.fromtimestamp(data['exp']).strftime('%Y-%m-%d %H:%M:%S')

            #---------------
            # Validate the validity of a token on the Redis server.
            # 从Redis中获取用户名
            redis_conn = rcPool.pool().get_connection()
            #vat =  self.jwt_redis.get(token).decode('utf-8')
            vat = redis_conn.get(token).decode('utf-8')
            if vat is None :
                return -3,{}, 'Token has expired or Invalid token!'
            #---------------
            return 0,data,'Token is valid'
        except jwt.ExpiredSignatureError:
            return -1,{},'Token has expired!'
        except jwt.InvalidTokenError:
            return -1,{},'Invalid token!'
        except Exception as e:
            nameko_logger.error(f"errorCode: -2,msg:{str(e)}")
            return -2,{},'read token failed!'
        finally:
            if redis_conn:
                rcPool.pool().release_connection(redis_conn)

    @rpc
    def deleteToken(self,token):
        try:
            if self.jwt_redis.delete(token):
                return 0,'delete token successfully'
            else:
                return -1,'delete token failed'
        except Exception as e:
            nameko_logger.error(f"errorCode: -2,msg:{str(e)}")
            return -2,'delete token failed'
