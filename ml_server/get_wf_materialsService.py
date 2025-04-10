import sqlite3
from nameko.rpc import rpc,RpcProxy
import sys
from collections import OrderedDict
sys.path.append("..")
from apocolib.NamekoLogger import namekoLogger as nameko_logger
from apocolib import sqliteSession as sqlSession
from ml_server.DataModel import *
class get_wf_materialsService:
    name="get_wf_materialsService"
    securityService = RpcProxy('securityService')
    @rpc
    def get_wf_materials(self):
        try:
            with sqlSession.sqliteSession().getSession() as session:
                wf = session.query(WfMaterials).order_by(WfMaterials.id.asc()).all()
                if wf is None:
                    return -1,[],'get_wf,no data found'
                else:
                    return 0,[{
                        'id':wf_item.id,
                        'wf_material':wf_item.wf_material,
                        'profile_section':wf_item.profile_section,
                        'wf_type':wf_item.wf_type,
                        'window_area':wf_item.window_area,
                        'wf_area':wf_item.wf_area,
                        'window_frame_area_ratio':wf_item.window_frame_area_ratio,
                        'K':wf_item.K,
                        'price':wf_item.price,
                        'description':wf_item.description
                        } for wf_item in wf],'get_wf successfully' 
                    #for wf in wf}
                    #sorted_wf_dict = {key:wf_dict[key] for key in sorted(wf_dict)}
                   # return 0,wf_dict,'get_wf successfully'
        except Exception as e:
            nameko_logger.error(f'get_wf error.{str(e)}')
            return -1, [],f'get_wf error.{str(e)}'
