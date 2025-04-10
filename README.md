
# Design-mate-server 部署指南

## 运行环境要求
- **操作系统**: Ubuntu 20.04.6 LTS
- **Python**: 3.8.10 或更高版本
- **部署目录**: `/home/apoco`

---

## 快速部署步骤

### 1. 下载部署文件
```bash
cd /home/apoco
git clone https://github.com/shark8848/Design-mate-server.git
```

### 2. 创建 Python 虚拟环境
```bash
cd Design-mate-server
python3 -m venv venv
source venv/bin/activate
```

### 3. 安装依赖包
```bash
pip install -r requirements.txt
```

---

## 服务依赖安装

### 1. Redis 安装与配置
```bash
# 安装 Redis
sudo apt install redis-server

# 配置密码访问
sudo sed -i 's/# requirepass foobared/requirepass apoco2022/g' /etc/redis/redis.conf
sudo systemctl restart redis-server
```

### 2. RabbitMQ 安装与配置
```bash
# 添加仓库
curl -fsSL https://github.com/rabbitmq/signing-keys/releases/download/2.0/rabbitmq-release-signing-key.asc | sudo gpg --dearmor -o /usr/share/keyrings/rabbitmq-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/rabbitmq-archive-keyring.gpg] https://dl.bintray.com/rabbitmq-erlang/debian $(lsb_release -cs) erlang" | sudo tee /etc/apt/sources.list.d/erlang.list
echo "deb [signed-by=/usr/share/keyrings/rabbitmq-archive-keyring.gpg] https://dl.bintray.com/rabbitmq/debian $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/rabbitmq.list

# 安装服务
sudo apt update
sudo apt install -y rabbitmq-server

# 启用管理插件
sudo rabbitmq-plugins enable rabbitmq_management
sudo systemctl restart rabbitmq-server

# 访问管理界面
echo "访问 http://localhost:15672 (默认账号: guest/guest)"
```

---

## 系统配置

### 1. 参数加密配置
```bash
cd Design-mate-server/config
chmod +x encript_config.sh
./encript_config.sh  # 按提示输入参数
```

---

## 服务启动顺序

### 1. 启动 API 接口服务
```bash
cd Design-mate-server/flask_server
chmod +x *.sh
./start_flask.sh
./show_flask.sh  # 检查是否显示5个flask_jwt_server进程
```

### 2. 启动 Nameko 微服务
```bash
cd Design-mate-server/nameko_server
chmod +x *.sh
./start_nameko.sh
./show_services.sh  # 检查服务状态
```

### 3. 启动 ML 服务
```bash
cd Design-mate-server/ml_server
chmod +x *.sh
./start_mlserver.sh
./show_services.sh
```

### 4. 启动 WebSocket 服务
```bash
cd Design-mate-server/websocket_server
chmod +x *.sh
./start_wss.sh
```

### 5. 启动监控服务
```bash
cd Design-mate-server/monitor_server
chmod +x *.sh
./start_monitor_1.sh
```

---

## 前端部署

### 1. 下载前端代码
```bash
cd /home/apoco
git clone -b dev [https://github.com/shark8848/Design-mate-portal.git](https://github.com/shark8848/Design-mate-portal.git)
```

### 2. Apache 配置示例
```apacheconf
# /etc/apache2/sites-available/aiportal.conf
<VirtualHost *:80>
    ServerName portal.xxxx.com.cn
    DocumentRoot /home/apoco/Design-mate-portal/dist

    # API反向代理
    ProxyPass /api http://localhost:5000
    ProxyPassReverse /api http://localhost:5000

    # WebSocket代理
    ProxyPass /ws/ ws://localhost:8000/
    ProxyPassReverse /ws/ ws://localhost:8000/

    <Directory "/home/apoco/Design-mate-portal/dist">
        Options Indexes FollowSymLinks
        AllowOverride All
        Require all granted
    </Directory>
</VirtualHost>
```

---

## Nginx 反向代理配置

### 1. 安装 Nginx
```bash
sudo apt install nginx
```

### 2. 配置文件示例
```nginx
# /etc/nginx/conf.d/aiportal.conf
upstream ai_servers {
    server localhost:5000;
    server localhost:5001;
}

server {
    listen 80;
    server_name portal.xxxxx.com.cn;

    # 静态文件服务
    location / {
        root /home/apoco/Design-mate-portal/dist;
        try_files $uri $uri/ /index.html;
    }

    # API负载均衡
    location /api {
        proxy_pass http://ai_servers;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    # WebSocket代理
    location /ws {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }

    # 监控服务
    location /monitor {
        proxy_pass http://localhost:9000;
    }
}
```

### 3. 重启 Nginx
```bash
sudo nginx -t && sudo systemctl reload nginx
```

---

## 邮件服务独立部署

### 1. 下载脚本
```bash
mkdir mail_service && cd mail_service
curl -O https://gitlab.apoco.com.cn/ai/apoco-intelligent-analysis/-/raw/develop-sunhy/fetch_file.sh
curl -O https://gitlab.apoco.com.cn/ai/apoco-intelligent-analysis/-/raw/develop-sunhy/file_list.txt
chmod +x fetch_file.sh
```

### 2. 运行服务
```bash
./fetch_file.sh  # 自动下载依赖文件
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 mail_service.py
```

---

> **部署验证**
> 1. 访问 `http://aiportal.apoco.com.cn`
> 2. 检查各服务状态：
>    ```bash
>    # 检查进程
>    ps aux | grep -E 'flask|nameko|mlserver|wss|monitor'
>    
>    # 检查端口监听
>    sudo netstat -tulpn | grep -E '5000|8000|9000'
>    ```

```
