import asyncio
import websockets
import pika
import json
from threading import Thread

class WebSocketServer:
    def __init__(self, ip="localhost", port=8086, rabbitmq_host='localhost',rabbitmq_port=5672,queue_size=8):
        self.ip = ip
        self.port = port
        self.rabbitmq_host = rabbitmq_host
        self.rabbitmq_port = rabbitmq_port
        self.queue_size = queue_size
        self.loop = asyncio.new_event_loop()   # New event loop
        self.queues = {f'queue_{i}': asyncio.Queue(loop=self.loop) for i in range(1, self.queue_size)}  # Create 8 queues

    def start_rabbitmq_consumer(self):
        credentials = pika.PlainCredentials('guest', 'guest')
        parameters = pika.ConnectionParameters(self.rabbitmq_host, self.rabbitmq_port, '/', credentials)
        connection = pika.BlockingConnection(parameters)
        channel = connection.channel()

        args = {"x-max-length": 200000, "x-message-ttl": 3600000}  # Set max length and message TTL to 1 hour

        # Declare 8 queues
        for queue_name in self.queues.keys():
            channel.queue_declare(queue=queue_name, arguments=args)

            def callback(ch, method, properties, body, queue_name=queue_name):
                asyncio.run_coroutine_threadsafe(self.queues[queue_name].put(body.decode()), self.loop)

            channel.basic_consume(queue=queue_name, on_message_callback=callback, auto_ack=True)

        channel.start_consuming()

    async def start(self):
        async def echo(websocket, path):
            queue_name = path.lstrip('/')  # Remove the leading '/'
            while True:
                if queue_name in self.queues:
                    message = await self.queues[queue_name].get()
                    await websocket.send(message)
                else:
                    await websocket.send(json.dumps({"error": f"Invalid queue name: {queue_name}"}))

        server = await websockets.serve(echo, self.ip, self.port, loop=self.loop)   # Set the loop for the server

        try:
            await server.wait_closed()
        except asyncio.CancelledError:
            server.close()
            await server.wait_closed()

if __name__ == "__main__":
    server = WebSocketServer(ip="192.168.1.19", port=8086, rabbitmq_host='localhost',rabbitmq_port = 5672,queue_size=8)
    Thread(target=server.start_rabbitmq_consumer).start()
    server.loop.run_until_complete(server.start())   # Use the event loop
    server.loop.run_forever()   # Use the event loop
