# generalMessageService.py

import yaml
from nameko.rpc import rpc,RpcProxy
from sqlalchemy import create_engine, Column, Integer, String, Enum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import hashlib
import sys
import redis
import json
import base64
import jwt
import datetime
import ssl

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

sys.path.append("..")
#from apocolib.apocolog4p import apoLogger as apolog
from apocolib.NamekoLogger import namekoLogger as nameko_logger
from apocolib import apocoIAServerConfigurationManager as iaConMg
from apocolib import sqliteSession as sqlSession
from apocolib.RedisPool import redisConnectionPool as rcPool
from urpfModel import User

class generalMessageService:

    name = 'generalMessageService'

    SECRET_KEY = iaConMg.mail_jwt_secret_key
    JWT_EXPIRATION_DELTA = iaConMg.mail_jwt_expiration_delta

    sender_mail = iaConMg.sender_mail
    sender_password = iaConMg.sender_password
    smtp_server = iaConMg.smtp_server
    smtp_port = iaConMg.smtp_port
    smtp_ssl_port = iaConMg.smtp_ssl_port
    smtp_server_token = iaConMg.smtp_server_token

    # 初始化 Redis 客户端
#    redis_host = iaConMg.redis_host
#    redis_port = iaConMg.redis_port
#    redis_password = iaConMg.redis_password
#    jwt_redis = redis.Redis(host=redis_host, port=redis_port, password=redis_password)

    # call usersService

    usrService = RpcProxy("usersService")

    @rpc
    def reset_password_over_email(self,email):

        redis_conn = None

        try:

            res,msg = self.usrService.email_is_registered(email)

            if res != 0:
                return res,msg

            # gen token,saved it in redis

            expiration_time = datetime.datetime.utcnow() + datetime.timedelta(seconds= self.JWT_EXPIRATION_DELTA)
            token = jwt.encode( {'email': email, 'exp': expiration_time}, self.SECRET_KEY)

            nameko_logger.info(f'create new token for reset password ,{email}: {token}')

            #--------- saved into redis ------------------#
            redis_conn = rcPool.pool().get_connection()
            redis_conn.set(token, email)
            redis_conn.expire(token, self.JWT_EXPIRATION_DELTA)
            #self.jwt_redis.set(token, email)
            #self.jwt_redis.expire(token, self.JWT_EXPIRATION_DELTA)
            #---------------------------------------------#

            # send mail
            message = MIMEMultipart()
            message["From"] = self.sender_mail
            message["To"] = email
            message["Subject"] = "APOCO DesignMate : Password reset link"

            text = f"Please click the following link to reset your password,expiration time is {expiration_time}:\n\n"
            link = f"http://ai.apoco.com.cn/password-reset/{token}"
            html = f'<p>Please click the following link to reset your password: <a href="{link}">{link}</a></p>'

            message.attach(MIMEText(text, "plain"))
            message.attach(MIMEText(html, "html"))

            nameko_logger.info(f' from {self.sender_mail} to {email} over {self.smtp_server} {self.smtp_port} {self.smtp_ssl_port}')
            
            context = ssl.create_default_context()
            context.set_ciphers('HIGH:!aNULL:!eNULL:!EXPORT:!DES:!MD5:!PSK:!RC4:!SSLv2:!SSLv3')

            with smtplib.SMTP_SSL(self.smtp_server, self.smtp_ssl_port ,context=context) as smtp:

                smtp.login(self.sender_mail, self.sender_password)
                smtp.send_message(message)

            return 0,'send Password reset mail success'

        except Exception as e:
            nameko_logger.error(f'reset passord over email failed {email} error.{str(e)}')
            return -1, f'reset passord over email failed {email} error.{str(e)}'
        finally:
            if redis_conn: 
                rcPool.pool().release_connection(redis_conn)
