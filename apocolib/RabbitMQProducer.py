import pika
import apocolib.MQueueManager
import sys
sys.path.append("..")
from apocolib.MlLogger import mlLogger as ml_logger

class RabbitMQProducer:
    def __init__(self, queue_manager=None, host='localhost', port =5672):
        self.queue_manager = queue_manager
        self.host = host
        self.port = port
        self.queue_name = None

    def connect(self, queue_name):
        credentials = pika.PlainCredentials('guest', 'guest')
        parameters = pika.ConnectionParameters(self.host, self.port, '/', credentials)
        self.connection = pika.BlockingConnection(parameters)
        self.channel = self.connection.channel()
        self.queue_name = queue_name
        self.channel.queue_declare(queue=queue_name, passive=True)
        print("queue_name ",self.queue_name)
        #return queue_name

    def publish(self, message):

        if self.queue_name is None and self.queue_manager is not None: # 2023.5.30
            self.queue_name = self.queue_manager.allocate_queue()

        #ml_logger.info("queue_name is {}".format(self.queue_name))

        if self.queue_name is not None:

            if not hasattr(self, 'connection'):
                self.connect(self.queue_name)
                #ml_logger.info("connected queue ",self.queue_name)
                ml_logger.info("connected queue {}".format(self.queue_name))
            self.channel.basic_publish(exchange='', routing_key=self.queue_name, body=message)
            #self.close()
            #self.queue_manager.release_queue(self.queue_name)
        else:
            ml_logger.error("No available queue.")
            raise Exception("No available queue.")

    def get_queue_name(self):
        return "{}".format(self.queue_name)

    def set_queue_name(self,queue_name):
        self.queue_name = queue_name

    def close(self):

        if hasattr(self, 'connection'):
            self.connection.close()
            del self.connection

        self.queue_manager.release_queue(self.queue_name)
