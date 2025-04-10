import sqlite3
from nameko.rpc import rpc,RpcProxy
import sys
from collections import OrderedDict
sys.path.append("..")
from apocolib.NamekoLogger import namekoLogger as nameko_logger
from apocolib import sqliteSession as sqlSession
from ml_server.DataModel import *
class get_glass_materialsService:
    name = "get_glass_materialsService"
    securityService = RpcProxy('securityService')
    @rpc
    def get_glass_materials(self):
        try:
            with sqlSession.sqliteSession().getSession() as session:
                glasses = session.query(Glass).order_by(Glass.id.asc()).all()
                if glasses is None:
                    return -1,[],'get_glass,no data found'
                else:
                   # glass_dict=OrderedDict()
                   # for glass_item in glasses:
                   return 0,[{
                        'id':glass_item.id,
                        'S_D':glass_item.S_D,
                        'thickness':glass_item.thickness,
                        'silver_plated':glass_item.silver_plated,
                        'hollow_material': glass_item.hollow_material,
                        #'t_price': glass_item.t_price,
                        #'t_K':glass_item.t_K,
                        'price': glass_item.t_price,
                        'K':glass_item.t_K,
                        'description':glass_item.description
                        } for glass_item in glasses],'get_glass successfully'
                    #sorted_glass_dict={str(key):glass_dict[key] for key in sorted(glass_dict)}
                    #sorted_glass_dict=dict(sorted(glass_dict.items(),key=lambda x:int(x[0])))
                    #sorted_glass_dict = {key:glass_dict[key] for key in sorted(glass_dict)}
                    #return 0,glass_dict,'get_glass successfully'
        except Exception as e:
            nameko_logger.error(f'get_glass error.{str(e)}')
            return -1, [],f'get_glass error.{str(e)}'
