import psutil
import json
import time
import asyncio
import aio_pika
import sys
import re
import os
sys.path.append("..")
from apocolib.RabbitMQProducer import RabbitMQProducer as RabbitMQProducer

class MonitorDataCollector:
    def __init__(self, rabbitmq_host='localhost', rabbitmq_port=5672, rabbitmq_queue='system'):
        self.rmqp = RabbitMQProducer()
        self.rmqp.set_queue_name(rabbitmq_queue)
        self.rmqp.connect(queue_name=rabbitmq_queue)
        #self.keywords =['nameko','flask_jwt_server']
        self.keywords = self.load_keywords()

    def load_keywords(self):
        keywords = []
        with open('./keywords.txt', 'r') as file:
            for line in file:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "#" in line:
                    keyword = line.split("#", 1)[0].strip()
                else:
                    keyword = line
                keywords.append(keyword)
        return keywords

    def publish_message(self, message):
        self.rmqp.publish(message)

    async def collect_and_publish_data(self, interval, data_type, data_func):
        while True:
            data = await data_func()
            data['type'] = data_type
            data['timestamp'] = time.strftime("%Y-%m-%d %H:%M:%S")
            message = json.dumps(data)
            print(f" publish message - {message}")
            self.publish_message(message)
            await asyncio.sleep(interval)

    async def collect_cpu_usage(self):
        return {
            'usage': psutil.cpu_percent(interval=None),
            'cores': psutil.cpu_percent(percpu=True)
        }

    async def collect_swap_usage(self):
        swap_usage = psutil.swap_memory()
        return {
            'total': swap_usage.total,
            'used': swap_usage.used,
            'free': swap_usage.free,
            'percent': swap_usage.percent
        }

    async def collect_memory_usage(self):
        memory_usage = psutil.virtual_memory()
        return {
            'total': memory_usage.total,
            'available': memory_usage.available,
            'used': memory_usage.used,
            'percent': memory_usage.percent
        }  
    async def collect_disk_usage(self):
        disk_usage = psutil.disk_usage('/')
        return {
            'total': disk_usage.total,
            'used': disk_usage.used,
            'free': disk_usage.free,
            'percent': disk_usage.percent
        }

    async def collect_process_info(self): #, keywords=None):
        process_list = []
        for process in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_info']):

            cmdline = " ".join(process.cmdline())

            #if self.keywords and not any(keyword.lower() in process.name().lower() for keyword in self.keywords):
            if self.keywords and not any(keyword.lower() in cmdline.lower() for keyword in self.keywords):
                continue

            process_info = {
                'pid': process.pid,
                #'name': process.name(),
                'cmdline':cmdline,
                'cpu_usage': process.cpu_percent(interval=0.1),
                'memory_usage': process.memory_info().rss
            }

            if self.keywords:
                for keyword in self.keywords:
                    pattern = re.compile(rf"({re.escape(keyword)}\s*.*)", re.IGNORECASE)
                    match = pattern.search(cmdline)
                    if match:
                        process_info['cmdline'] = match.group(1)
                        #print(match.group(1))
                        break

            process_list.append(process_info)
        #print("processes ",process_list)
        sorted_process_list = sorted(process_list, key=lambda x: self.get_sort_key(x['cmdline'], self.keywords))
        #print(sorted_process_list)
        return {'processes': sorted_process_list}
        #return {'processes': process_list}

    def get_sort_key(self, cmdline, keywords):
        for i, keyword in enumerate(keywords):
            if keyword.lower() in cmdline.lower():
                return (i, cmdline)
        return (len(keywords), cmdline)

    async def run(self):
        tasks = [
            asyncio.create_task(self.collect_and_publish_data(5, 'cpu', self.collect_cpu_usage)),
            asyncio.create_task(self.collect_and_publish_data(10, 'swap', self.collect_swap_usage)),
            asyncio.create_task(self.collect_and_publish_data(10, 'memory', self.collect_memory_usage)),
            asyncio.create_task(self.collect_and_publish_data(10, 'disk', self.collect_disk_usage)),
            asyncio.create_task(self.collect_and_publish_data(10, 'process', self.collect_process_info))
        ]
        await asyncio.gather(*tasks)

if __name__ == '__main__':
    keyword = 'MonitorDataCollector.py'
    current_cmdline = " ".join(psutil.Process(os.getpid()).cmdline())

    for process in psutil.process_iter(['pid', 'cmdline']):
        if process.pid != os.getpid() and keyword in " ".join(process.cmdline()):
            print("Another instance of the MonitorDataCollector program is already running. Exiting...")
            exit()

    monitor = MonitorDataCollector()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(monitor.run())

