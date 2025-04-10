import sys
import yaml
import json
import AESCipher
import re
import os


# 加密函数
def encrypt_config(config, key):
    AESCipherObj = AESCipher.AESCipher(key)
    # 获取redis_password的值并加密
    if 'redis' in config and 'redis_password' in config['redis']:
        redis_password = config['redis']['redis_password']
        encrypted_redis_password = AESCipherObj.encrypt(redis_password)
        config['redis']['redis_password'] = encrypted_redis_password

    # 获取flask_jwt_secret_key的值并加密
    if 'server' in config and 'flask_jwt_secret_key' in config['server']:
        flask_jwt_secret_key = config['server']['flask_jwt_secret_key']
        encrypted_flask_jwt_secret_key = AESCipherObj.encrypt(flask_jwt_secret_key)
        config['server']['flask_jwt_secret_key'] = encrypted_flask_jwt_secret_key

    # 获取config_mq的值并加密
    if 'mq' in config and 'config_mq' in config['mq']:
        temp = config['mq']['config_mq']
        config_mq = temp['AMQP_URI']
        encrypted_amqp_url = AESCipherObj.encrypt(config_mq)
        d = {}
        d['AMQP_URI'] = encrypted_amqp_url
        config['mq']['config_mq'] = d

    # 获取mail_jwt_secret_key的值并加密
    if 'mail' in config and 'mail_jwt_secret_key' in config['mail']:
        mail_jwt_secret_key = config['mail']['mail_jwt_secret_key']
        encrypted_mail_jwt_secret_key = AESCipherObj.encrypt(mail_jwt_secret_key)
        config['mail']['mail_jwt_secret_key'] = encrypted_mail_jwt_secret_key

    # 获取sender_password的值并加密
    if 'mail' in config and 'sender_password' in config['mail']:
        sender_password = config['mail']['sender_password']
        encrypted_sender_password = AESCipherObj.encrypt(sender_password)
        config['mail']['sender_password'] = encrypted_sender_password

    return config


def main():
    # 读取命令行参数
    if len(sys.argv) < 3:
        print('Usage: python encrypt_config.py input.yaml output.yaml')
        return

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    key_file = sys.argv[3]

    # 判断输入文件是否存在
    if not os.path.isfile(input_file):
        print(f'Input file "{input_file}" does not exist.')
        return

    # 读取配置文件
    try:
        with open(input_file, 'r') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(f'Error loading YAML file "{input_file}": {e}')
        return

    # 加密配置项
    key = None
    try:
        with open(key_file, 'r') as f:
            data = yaml.safe_load(f)
            key = data['SERVER_KEY']
    except yaml.YAMLError as e:
        print(f'Error loading YAML file "{input_file}": {e}')
        return

    if key is None or key == '':
        print('SERVER_KEY is None!')
        return

    config = encrypt_config(config, key)

    # 回写加密后的
    try:
        with open(output_file, 'w') as f:
            yaml.safe_dump(config, f, default_flow_style=False)
    except Exception as e:
        print(f'Error writing encrypted config to "{output_file}": {e}')
        return

    print(f'Successfully encrypted config and wrote it to "{output_file}".')

if __name__ == '__main__':
    main()
