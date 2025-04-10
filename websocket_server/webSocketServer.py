import asyncio
import websockets
import json
from collections import deque
from threading import Thread
import argparse

class WebSocketServer:
    def __init__(self, ip="localhost", port=8086, max_size=100000):
        self.message_queues = {
            "system": deque(maxlen=max_size),
            "AC2NNetPredicter": deque(maxlen=max_size),
            "AC2NNetTrainer": deque(maxlen=max_size),
            "NamekoService": deque(maxlen=max_size),
            "flaskServer": deque(maxlen=max_size)
        }
        self.ip = ip
        self.port = port
        self.max_size = max_size

    def push_message(self, app_name, message):
        if app_name not in self.message_queues:
            self.message_queues[app_name] = deque(maxlen=self.max_size)
        self.message_queues[app_name].append(message)

    async def start(self):
        async def echo(websocket, path):
            app_name = path.strip("/")
            try:
                while True:
                    queue = self.message_queues.get(app_name)
                    if queue is not None:
                        while len(queue) > 0:
                            message = queue.popleft()
                            await websocket.send(json.dumps({app_name: message}))
                    await asyncio.sleep(0.01)
            except websockets.exceptions.ConnectionClosed:
                print('Client disconnected, cleaning up...')
            finally:
                await websocket.close()

        server = await websockets.serve(echo, self.ip, self.port)
        try:
            await server.wait_closed()
        except asyncio.CancelledError:
            server.close()
            await server.wait_closed()

def external_push_message(app_name, message):
    server.push_message(app_name, message)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='WebSocket Server')
    parser.add_argument('ip', type=str, help='The IP address of the server')
    parser.add_argument('port', type=int, help='The port of the server')
    parser.add_argument('max_size', type=int, help='The maximum size of the queue')
    args = parser.parse_args()

    server = WebSocketServer(args.ip, args.port, args.max_size)
    Thread(target=asyncio.run, args=(server.start(),)).start()
