import sqlite3
from nameko.rpc import rpc,RpcProxy
import sys
from collections import OrderedDict
sys.path.append("..")
from apocolib.NamekoLogger import namekoLogger as nameko_logger
from apocolib import sqliteSession as sqlSession
from ml_server.DataModel import *
class get_wall_materialsService:
    name="get_wall_materialsService"
    securityService = RpcProxy('securityService')
    @rpc
    def get_wall_materials(self):
        try:
            with sqlSession.sqliteSession().getSession() as session:
                walls = session.query(ImParameters).order_by(ImParameters.id.asc()).all()
                if walls is None:
                    return -1,[],'get_wall,no data found'
                else:
                    return 0,[{
                        'id':int(wall_item.id),
                        'type_number':wall_item.type_number,
                        'level':wall_item.level,
                        'description':wall_item.description,
                        'thickness':wall_item.thickness,
                        'K':wall_item.K,
                        'price':wall_item.price
                        } for wall_item in walls],'get_wall successfully'
        except Exception as e:
            nameko_logger.error(f'get_wall error.{str(e)}')
            return -1, [],f'get_wall error.{str(e)}'

