
运行环境要求
Ubuntu 20.04.6 LTS，Python 3.8.10 及以上
创建部署目录，如 /home/apoco
1、下载部署文件
git clone  https://gitlab.apoco.com.cn/ai/apoco-intelligent-analysis.git
在部署环境下会创建一个完整的文件包目录

2、创建python 虚拟隔离环境
cd apoco-intelligent-analysis
python3 -m venv venv
source venv/bin/activate
3、安装依赖包

pip install -r requirements.txt
4、安装redis 并配置
1）安装 sudo apt install redis-server
2）配置访问密码
在 Redis 中配置访问密码是为了增加安全性，以确保只有知道密码的用户可以访问 Redis 服务器。
以下是在 Redis 中配置访问密码的步骤：
打开 Redis 配置文件：通常，Redis 的配置文件位于 /etc/redis/redis.conf。你可以使用任何文本编辑器打开它，例如 nano 或 vim：
（1）vi /etc/redis/redis.conf
在配置文件中找到以下行：
# requirepass foobared
这是 Redis 访问密码的设置。foobared 是默认密码，通常是被注释掉的（以 # 开头）。你需要将其修改为你想要设置的密码。例如：
requirepass apoco2022
（2）重启 Redis 服务：配置更改后，需要重启 Redis 服务才能使密码生效。
sudo systemctl restart redis-server

5、安装rabbitmq

1）在开始安装RabbitMQ之前，确保系统已经安装了一些必要的依赖项。
sudo apt update
sudo apt install -y curl gnupg
2）添加RabbitMQ APT存储库：
curl -fsSL https://github.com/rabbitmq/signing-keys/releases/download/2.0/rabbitmq-release-signing-key.asc | sudo gpg --dearmor -o /usr/share/keyrings/rabbitmq-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/rabbitmq-archive-keyring.gpg] https://dl.bintray.com/rabbitmq-erlang/debian $(lsb_release -cs) erlang" | sudo tee /etc/apt/sources.list.d/erlang.list
echo "deb [signed-by=/usr/share/keyrings/rabbitmq-archive-keyring.gpg] https://dl.bintray.com/rabbitmq/debian $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/rabbitmq.list
3）安装RabbitMQ服务器：
sudo apt update
sudo apt install -y rabbitmq-server
4）启动RabbitMQ服务：
安装完成后，RabbitMQ将自动启动，并且在系统启动时自动运行。你可以使用以下命令来检查RabbitMQ的状态：
sudo systemctl status rabbitmq-server
如果看到"Active: active (running)"，表示RabbitMQ已经成功启动。
5）如果你希望使用RabbitMQ的Web管理界面，你还需要启用RabbitMQ Management插件。
sudo rabbitmq-plugins enable rabbitmq_management
接下来，重启RabbitMQ服务以使更改生效：
sudo systemctl restart rabbitmq-server
现在，你可以在浏览器中访问http://localhost:15672/，使用默认的用户名和密码（guest/guest）登录RabbitMQ管理界面。
6）增加远程访问授权
在 /etc/rabbitmq/目录下创建或者编辑文件 rabbitmq.conf 增加如下内容并保存。然后重启服务。


6、配置基础参数
在 apoco-intelligent-analysis/config 运行 encript_config.sh 配置基础参数。


安装上述提示输入对应的参数。输入完成后，系统自动输出对应信息：

出现以下信息表示配置成功。


7、按顺序启动系统服务
1）启动api 接口服务
在 apoco-intelligent-analysis/flask_server 运行 ./start_flask.sh

运行./show_flask.sh 检查，如果有5个flask_jwt_server 进程表示启动成功。
2）启动nameko_server 微服务
在 apoco-intelligent-analysis/nameko_server 运行 ./start_nameko.sh

运行./show_services.sh 检查


3）启动ml_server 微服务 
在 apoco-intelligent-analysis/ml_server 运行 ./start_mlserver.sh

运行./show_services.sh 检查


3）启动 消息服务器 websocket_server 
在apoco-intelligent-analysis/websocket_server 下执行 ./start_wss.sh


4）启动监控服务器 monitor_server
在apoco-intelligent-analysis/monitor_server 下执行 ./start_monitor_1.sh


8、部署aiportal
1）下载部署代码
在 /home/apoco/下执行
git clone -b dev https://gitlab.apoco.com.cn/ai/apoco-intelligent-analysis-admin.git
在部署环境下会创建一个完整的文件包目录 apoco-intelligent-analysis-admin

2）配置webserver，以apache 为例
（1）配置 虚拟机文件

（2）配置端口：

（3）配置 api 代理，在 apache2.conf 文件尾部增加如下内容



9、nginx 配置反向代理
1）安装nginx，完成后，在nginx conf.d 目录下创建一个文件，如 aiportal.conf，在该文件中配置如下内容
2）配置aiportal.apoco.com.cn的反向代理。

3) 配置 ai_servers 的负载均衡



4）配置消息服务的反向代理

5）配置监控服务的反向代理


10、邮件服务的独立部署（具备访问互联网的服务器）
1）创建虚拟隔离环境，略。
2）下载脚本 
curl -O https://gitlab.apoco.com.cn/ai/apoco-intelligent-analysis/-/raw/develop-sunhy/fetch_file.sh
curl -O https://gitlab.apoco.com.cn/ai/apoco-intelligent-analysis/-/raw/develop-sunhy/file_list.txt
3）执行 ./fetch_file.sh,将自动下载所需文件，下载的内容如下

4）运行 邮件服务 


