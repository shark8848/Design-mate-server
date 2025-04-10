ps -ef | grep 'usersService' | grep -v grep | awk '{print $2}' | xargs -r kill -9
#### users information manager
nohup nameko run usersService --broker amqp://guest:guest@192.168.1.19 >> ./log/nameko.log 2>&1 &
echo `date '+%Y-%m-%d %H:%M:%S %A'` " nameko serivce [usersService service] start!" >> ./log/nameko.log
