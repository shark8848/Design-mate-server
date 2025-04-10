import asyncio
import websockets
import pika
import json
import jwt
from threading import Thread
import redis
import base64
import jwt
import datetime
import sys
import pdb
import ssl
sys.path.append("..")
from apocolib import apocoIAServerConfigurationManager as iaConMg
from apocolib.RedisPool import redisConnectionPool as rcPool
from apocolib.wssLogger import wssLogger as wss_logger
from apocolib import RpcProxyPool


class WebSocketServer:
    def __init__(self, ip="localhost", port=8086, rabbitmq_host='localhost',rabbitmq_port=5672,queue_size=8):
        self.ip = ip
        self.port = port
        self.rabbitmq_host = rabbitmq_host
        self.rabbitmq_port = rabbitmq_port
        self.queue_size = queue_size
        self.loop = asyncio.new_event_loop()   # New event loop
        #self.queues = {f'queue_{i}': asyncio.Queue(loop=self.loop) for i in range(1, self.queue_size)}  # Create 8 queues
        self.queues = {f'queue_{i}': asyncio.Queue() for i in range(1, self.queue_size)}  # Create 8 queues
        print("queues ",self.queues)
        self.SECRET_KEY = iaConMg.flask_jwt_secret_key
        #self.JWT_EXPIRATION_DELTA = iaConMg.flask_jwt_expiration_delta
        # rpc pool
        self.pool = RpcProxyPool.RpcProxyPool()

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

    #def authenticate(self, token):
    def authenticate(self,user,token,client_ip):
#        pdb.set_trace()
        redis_conn = None
        try:
            data = jwt.decode(token, self.SECRET_KEY, algorithms=['HS256'])
            current_user = data['username']
            expiry_date = datetime.datetime.fromtimestamp(data['exp']).strftime('%Y-%m-%d %H:%M:%S')
            print(f"user :{current_user},expiry_date :{expiry_date}ip :{client_ip}")
            wss_logger.info(f"user :{current_user},expiry_date :{expiry_date}, ip :{client_ip}")
            ''' 
            if data['clientip'] != client_ip:
                wss_logger.error(f"invalid client ip . token_ip: {data['clientip']} request_ip = {client_ip}")
                return False
                '''
                

            redis_conn = rcPool.pool().get_connection()
            username =  redis_conn.get(token).decode('utf-8')
            #apolog.info(f" request user '{current_user}' ,username in redis is '{username}' ")
            #print("username ----:",username)
            if username is None or current_user != username:
                return False

        except jwt.ExpiredSignatureError:
            return False
        except jwt.InvalidTokenError:
            return False
        except Exception as e:
            return False
        finally:
            if redis_conn: 
                rcPool.pool().release_connection(redis_conn)

        return True

    async def start(self):
        async def echo(websocket, path):
            client_ip = websocket.request_headers.get('X-Forwarded-For')
            #client_ip = websocket.remote_address[0]
            print(f"Request client ip {client_ip} ")
            wss_logger.info(f"Request client ip {client_ip} ")

            queue_name = path.lstrip('/')  # Remove the leading '/'
            authenticated = False

            # 进行身份认证
            user = None
            token = None
            try:
                recv_auth_data = await asyncio.wait_for(websocket.recv(), timeout=10)  # 设置10秒超时时间
                data = json.loads(recv_auth_data)
                user = data.get("user")
                token = data.get("token") 
                authenticated = True
            except (json.JSONDecodeError, asyncio.TimeoutError) as e:
                print("un receive auth message ,close the connection")
                await websocket.send(json.dumps({"TimeoutError": "Un receive auth message in 10s. wss refuse the connection request."}))
                wss_logger.error("un receive auth message ,close the connection")
                await websocket.close()
                return

            authenticated = self.authenticate(user,token,client_ip)
            print("user:",user,"token:",token,"ip:",client_ip)
            wss_logger.info(f"user: {user},token: {token},ip: {client_ip}")

            if not authenticated:
                await websocket.send(json.dumps({"error": "Authentication failed"}))
                await websocket.close()
                print(f"Request client ip {client_ip} ,Authentication failed,disconnect it")
                wss_logger.info(f"Request client ip {client_ip} ,Authentication failed,disconnect it")

                return

            print(f"access client ip {client_ip} ")
            wss_logger.info(f"access client ip {client_ip} ")

            while True:
                if queue_name in self.queues:
                    message = await self.queues[queue_name].get()
                    await websocket.send(message)
                else:
                    await websocket.send(json.dumps({"error": f"Invalid queue name: {queue_name}"}))
                    await websocket.close()
                    wss_logger.error(f"Invalid queue name: {queue_name}")

        ctx = ssl.SSLContext(ssl.PROTOCOL_SSLv23)
        ctx.load_cert_chain("./cert/ws.apoco.com.cn_bundle.pem",keyfile="./cert/ws.apoco.com.cn.key",password=None)

        server = await websockets.serve(echo, self.ip, self.port, ssl = ctx, loop=self.loop)

        #server = await websockets.serve(echo, self.ip, self.port, loop=self.loop)   # Set the loop for the server

        try:
            await server.wait_closed()
        except asyncio.CancelledError:
            server.close()
            await server.wait_closed()

if __name__ == "__main__":
    server = WebSocketServer(ip="192.168.1.36", port=8086, rabbitmq_host='localhost',rabbitmq_port = 5672,queue_size=8)
    Thread(target=server.start_rabbitmq_consumer).start()
    server.loop.run_until_complete(server.start())   # Use the event loop
    server.loop.run_forever()   # Use the event loop
