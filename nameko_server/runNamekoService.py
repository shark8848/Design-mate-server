import eventlet.wsgi
eventlet.monkey_patch()

from nameko.runners import ServiceRunner
from nameko.testing.utils import get_container
import usersService
import organizationsService
from flask import Flask

app = Flask(__name__)

config = { "AMQP_URI": "amqp://guest:guest@192.168.1.19" }
class usersService:
    name = "usersService"

class organizationsService:
    name = "organizationsService"

if __name__ == '__main__':
    runner = ServiceRunner(config=config)
    runner.add_service(usersService)
    runner.add_service(organizationsService)
    runner.start()
    eventlet.wsgi.server(eventlet.listen(('192.168.1.19', 8888)), app)
