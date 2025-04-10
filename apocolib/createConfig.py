import yaml
from getpass import getpass

import yaml
from getpass import getpass

# 配置信息
config = {
    'redis': {
        'redis_host': input('请输入redis主机地址（默认10.8.0.181）：') or '10.8.0.181',
        'redis_port': int(input('请输入redis端口号（默认6379）：') or '6379'),
        'redis_password': getpass('请输入redis密码：'),
    },
    'sqlite': {
        'db_url': input('请输入sqlite数据库url（默认/home/apoco/ai_micro_services/apoco_intelligent_inalytics/database/apoco_ai_masterdb.db）：') or 'sqlite:////home/apoco/ai_micro_services/apoco_intelligent_inalytics/database/apoco_ai_masterdb.db',
    },
    'server': {
        'flask_host': input('请输入flask主机地址（默认10.8.0.181）：') or '10.8.0.181',
        'flask_port': [int(p) for p in input('请输入flask端口号（多个端口请用逗号分隔，默认5000,5001,5002,5003）：').split(',') if p.strip()] or [5000, 5001, 5002, 5003],
        'flask_pid_file': input('请输入flask pid文件路径（默认pid.txt）：') or 'pid.txt',
        'flask_jwt_secret_key': getpass('请输入flask jwt secret key：'),
        'flask_jwt_expiration_delta': int(input('请输入flask jwt过期时间（秒，默认86400）：') or '86400'),
    },
    'mq': {
        'config_mq': {'AMQP_URI': input('请输入MQ连接地址（默认amqp://guest:guest@10.8.0.181）：') or 'amqp://guest:guest@10.8.0.181'}
    },
    'dir': {
        'job_data_root_dir': input('请输入job数据根目录（默认./json）：') or './json',
        'service_data_root_dir': input('请输入service数据根目录（默认./json）：') or './json',
    },
    'mail': {
        'mail_jwt_secret_key': getpass('请输入mail jwt secret key：'),
        'mail_jwt_expiration_delta': int(input('请输入mail jwt过期时间（秒，默认1800）：') or '1800'),
        'sender_mail': input('请输入发件人邮箱（默认113162985@qq.com）：') or '113162985@qq.com',
        'sender_password': getpass('请输入发件人邮箱密码：'),
        'smtp_server': input('请输入smtp服务器地址（默认smtp.qq.com）：') or 'smtp.qq.com',
        'smtp_port': input('请输入smtp服务器端口号（默认25）：') or '25',
        'smtp_ssl_port': input('请输入smtp服务器SSL端口号（默认465）：') or '465',
        'smtp_server_token': input('请输入smtp服务器token（默认qq_smtp_server_token,please_update_it）：') or 'qq_smtp_server_token,please_update_it',
    },
    'token': {
        'algorithms': input('请输入token算法（默认HS256）：') or 'HS256',
    }
}

# 将配置信息写入yaml文件
with open('config.yaml', 'w') as f:
    # 使用默认流样式
    yaml.dump(config, f, default_flow_style=False)

# 输出yaml文件内容
with open('config.yaml', 'r') as f:
    print(f.read())
