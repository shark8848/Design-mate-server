import yaml
import sys
import datetime
import json
import re

sys.path.append("..")
#from apocolib.apocolog4p import apoLogger as apolog
from apocolib import AESCipher

config_file = '../config/AESC_SERVER.yaml'
key_file = '../config/server_key.yaml'
key = None
#print('\n--------------------------------------------------------------------')
#print('# system init ......')
try:
    with open(key_file) as file:
        config = yaml.safe_load(file)
        key = config['SERVER_KEY']
#        print('SERVER_KEY: ********** ') #{key}')
except Exception as e:
    #apolog.error(f"open config file {key_file} failed ,{str(e)}")
    #raise ConfigInitException(f"open key file {key_file} failed ,{str(e)}")
    print(f"open key file {key_file} failed ,{str(e)}")

if key is not None:

    aesc = AESCipher.AESCipher(key)

    try:
        #print('\r# loading config ......')

        with open(config_file) as file:
            config = yaml.safe_load(file)

        # redis 
        redis_host = config['redis']['redis_host']
        redis_port = config['redis']['redis_port']
        #解密
        redis_password = aesc.decrypt(config['redis']['redis_password'])
        #sqlite
        db_url = config['sqlite']['db_url']

        # flask
        flask_host = config['server']['flask_host']
        flask_port = config['server']['flask_port']
        flask_pid_file = config['server']['flask_pid_file']

        #解密
        flask_jwt_secret_key = aesc.decrypt(config['server']['flask_jwt_secret_key'])

        flask_jwt_expiration_delta = config['server']['flask_jwt_expiration_delta']

        #rabbitmq
        cmq = config['mq']['config_mq']
        config_mq = {'AMQP_URI': aesc.decrypt(cmq['AMQP_URI'])}
        #print("config_mq ",config_mq)

        #dir
        job_data_root_dir = config['dir']['job_data_root_dir']
        service_data_root_dir = config['dir']['service_data_root_dir']

        #mail:
        mail_jwt_secret_key = aesc.decrypt(config['mail']['mail_jwt_secret_key'])

        mail_jwt_expiration_delta = config['mail']['mail_jwt_expiration_delta']
        sender_mail = config['mail']['sender_mail']
        #解密
        sender_password = aesc.decrypt(config['mail']['sender_password'])

        smtp_server = config['mail']['smtp_server']
        smtp_port = config['mail']['smtp_port']
        smtp_ssl_port = config['mail']['smtp_ssl_port']

        smtp_server_token = config['mail']['smtp_server_token']
        #token
        algorithms = config['token']['algorithms']

    except Exception as e:
        #apolog.error(f"open config file {config_file} failed ,{str(e)}")
        #raise ConfigInitException(f"open config file {config_file} failed ,{str(e)}")
        print(f"open config file {config_file} failed ,{str(e)}")
else:
    #raise ConfigInitException(f"The key in the file {key_file} is None ")
    print(f"The key in the file {key_file} is None ")
    #apolog.error("The key in the file {key_file} is None ")

class ConfigInitException(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)
