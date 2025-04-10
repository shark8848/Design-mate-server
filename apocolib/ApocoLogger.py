import sys
from loguru import logger
import datetime


class ApocoLogger:

    def __init__(self, name, path, rotation):
        self.name = name
        self.path = path
        self.rotation = rotation

    def getLogger(self):
        log_path = f"{self.path}_{datetime.datetime.now().strftime('%Y%m%d')}.log"
        lg = logger.bind(name=self.name)
        lg.add(log_path, encoding='utf-8', level='DEBUG',
               format='{time:YYYYMMDD HH:mm:ss} - {module}.{function}:{line} - {level} - {message}',
               filter=lambda record: record["extra"]["name"] == self.name,
               rotation=self.rotation)
        return lg

def getFlaskLogger():
    return ApocoLogger('flask', './log/flask_server', '1 day').getLogger()

def getNamekoLogger():
    return ApocoLogger('nameko','./log/nameko_server','1 day').getLogger()

def getMlLogger():
    return ApocoLogger('ml','./log/ml_server','1 day').getLogger()

def getWssLogger():
    return ApocoLogger('wss','./log/websocket_server','1 day').getLogger()

if __name__ == '__main__':

    lg_a = ApocoLogger.getFlaskLogger()
    lg_b = ApocoLogger.getNamekoLogger()
    lg_c = ApocoLogger.getWssLogger()

    lg_a.info('test log a')
    lg_b.info('test log b')
    lg_c.info('test log c')
