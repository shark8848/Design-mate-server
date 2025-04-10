import yaml
import sys
sys.path.append("..")
from apocolib import AESCipher

class ConfigManager:
    def __init__(self, key_file='/home/apoco/apoco-intelligent-analysis/config/server_key.yaml', 
            config_file='/home/apoco/apoco-intelligent-analysis/config/AESC_SERVER.yaml'):
        self.key_file = key_file
        self.config_file = config_file
        self.key = None
        self.aesc = None
        self.load_key()

    def load_key(self):
        try:
            with open(self.key_file) as file:
                config = yaml.safe_load(file)
                self.key = config.get('SERVER_KEY', None)
        except Exception as e:
            raise ConfigInitException(f"Open key file {self.key_file} failed: {str(e)}")

        if self.key is not None:
            self.aesc = AESCipher.AESCipher(self.key)
            self.load_config()
        else:
            raise ConfigInitException(f"The key in the file {self.key_file} is None")

    def load_config(self):
        try:
            with open(self.config_file) as file:
                config = yaml.safe_load(file)

            # Load configuration parameters using self.aesc for decryption
            self.redis_host = config['redis']['redis_host']
            self.redis_port = config['redis']['redis_port']
            self.redis_password = self.aesc.decrypt(config['redis']['redis_password'])
            self.db_url = config['sqlite']['db_url']
            self.flask_host = config['server']['flask_host']
            self.flask_port = config['server']['flask_port']
            self.flask_pid_file = config['server']['flask_pid_file']
            self.flask_jwt_secret_key = self.aesc.decrypt(config['server']['flask_jwt_secret_key'])
            self.flask_jwt_expiration_delta = config['server']['flask_jwt_expiration_delta']
            self.config_mq = {'AMQP_URI': self.aesc.decrypt(config['mq']['config_mq']['AMQP_URI'])}
            self.job_data_root_dir = config['dir']['job_data_root_dir']
            self.service_data_root_dir = config['dir']['service_data_root_dir']
            self.mail_jwt_secret_key = self.aesc.decrypt(config['mail']['mail_jwt_secret_key'])
            self.mail_jwt_expiration_delta = config['mail']['mail_jwt_expiration_delta']
            self.sender_mail = config['mail']['sender_mail']
            self.sender_password = self.aesc.decrypt(config['mail']['sender_password'])
            self.smtp_server = config['mail']['smtp_server']
            self.smtp_port = config['mail']['smtp_port']
            self.smtp_ssl_port = config['mail']['smtp_ssl_port']
            self.smtp_server_token = config['mail']['smtp_server_token']
            self.algorithms = config['token']['algorithms']

        except Exception as e:
            raise ConfigInitException(f"Open config file {self.config_file} failed: {str(e)}")

class ConfigInitException(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

# Example usage
if __name__ == "__main__":
    key_file = '../config/server_key.yaml'
    config_file = '../config/AESC_SERVER.yaml'
    #config = ConfigManager(key_file, config_file)
    config = ConfigManager()
    print(config.redis_host)
    print(config.db_url)
    print(config.flask_host)
