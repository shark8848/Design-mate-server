import asyncio
import websockets
import json
import argparse
import requests
import base64
import ssl
from requests.auth import HTTPBasicAuth

async def consumer(app_name,username,passowrd):
    #uri = f"ws://10.8.0.181:8087/{app_name}"
    #uri = f"ws://192.168.1.19:8087/{app_name}"
    uri = f"wss://monitor.apoco.com.cn/{app_name}"
    #uri = f"ws://192.168.1.19:8086/"
    username,token = login(username,passowrd)
    print("user:",username,"token:",token)
    if token is None:
        return
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    async with websockets.connect(uri,ssl=ssl_context) as websocket:
        try:
            # 连接建立后发送消息
            auth_message = {"user": username, "token": token}
            await websocket.send(json.dumps(auth_message))            

            while True:
                message = await websocket.recv()
                #print(f"Received message: {message}")
                print(message)
        except websockets.exceptions.ConnectionClosed:
            print('Server disconnected, cleaning up...')

def login(username, password):
    credentials = f"{username}:{password}"
    encoded_credentials = base64.b64encode(credentials.encode()).decode()
    headers = {'Content-Type': 'application/x-www-form-urlencoded','Authorization': f'Basic {encoded_credentials}'}

    url = 'https://ai.apoco.com.cn/login'
    #url = 'http://10.8.0.181:5000/login'

    try:
        response = requests.post(url, headers=headers)
        response.raise_for_status()  # 检查请求是否成功
        data = response.json()
        token = data.get('accessToken')
        return username, token
    except requests.exceptions.RequestException as e:
        print(f'Login failed: {e}')
        return None, None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='WebSocket Client')
    parser.add_argument('app_name', type=str, help='The name of the app to connect to')
    parser.add_argument('username', type=str, help='user name')
    parser.add_argument('password', type=str, help='password')
    args = parser.parse_args()

    asyncio.run(consumer(args.app_name, args.username, args.password))
