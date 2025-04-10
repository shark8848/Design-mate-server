# -*- coding: utf-8 -*-
"""
Program: buildingSpaceCompositionInformationService.py
Description: 提供buildSpaceCompositionInformation 的微服务(包括查询、新增、修改等服务)，供flask api 接口调用.
Author: Sunhy
Version: 1.0
Date: 2023-02-24
"""

from nameko.rpc import rpc,RpcProxy
#from nameko.service import Service
import json,sys,jsonschema
#from jsonschema import Draft7Validator
import verifyJsonFileLoader

sys.path.append("..")
#from apocolib.apocolog4p import apoLogger as apolog
from apocolib.NamekoLogger import namekoLogger as nameko_logger
from apocolib.jsonHandler import jsonHandler
import pdb

class buildingSpaceSimpleService:

    name = "buildingSpaceSimpleService"
    buildingSpaceSimple_json = "./json/buildingSpaceSimple.json"
    buildingSpaceSimple_schema = "./json/json_schema/buildingSpaceSimpleSchema.json"
    buildingSpaceSimpleTemplate_schema = "./json/json_schema/buildingSpaceSimpleTemplateSchema.json"
    loader = None

    #jhandler = None

    '''
    json_handler = JsonHandler('data.json')
    search_results = json_handler.search(['organizationCode', 'projectId', 'buildingSpaceId'], ['org_code_001', 'proj_001', 'bs_001'])
    insert_result = json_handler.insert({'organizationCode': 'org_code_001', 'projectId': 'proj_002', 'buildingSpaceId': 'bs_002', 'data': 'new_data'})
    delete_result = json_handler.delete(['organizationCode', 'projectId'], ['org_code_001', 'proj_001'])
    update_result = json_handler.update(['organizationCode', 'projectId', 'buildingSpaceId'], ['org_code_001', 'proj_001', 'bs_001'], {'data': 'updated_data'})

    '''

    def __init__(self):
        nameko_logger.info("debug 0")
        #self.jhander = jsonHandler(self.buildingSpaceSimple_json)
        nameko_logger.info("debug 1")

# 查询所有配置好的空间的组合的详细信息，包括详细的户型信息、楼梯间、公共通道等。
    @rpc
    def getAllBuildingSpaceCompositionInformation(self,projectInfo): 
        # ORG && PROJ {"organizationCode":organizationCode,"projectId":projectId}
        pdb.set_trace()
        nameko_logger.info("before call -: getAllBuildingSpaceCompositionInformation")
        nameko_logger.info(f" request params {projectInfo['organizationCode']}-{projectInfo['projectId']}")

        jh = jsonHandler(self.buildingSpaceSimple_json)

        result = jh.search(['organizationCode', 'projectId','buildingSpaceId'], [projectInfo['organizationCode'],projectInfo['projectId'],projectInfo['buildingSpaceId']])

        return result

# 查询某一个配置好的空间的组合的详细信息如某户型详细的户型信息。
#    @rpc
#    def getOneBuildingSpaceCompositionInformation(self,buildingSpaceInfo):


#    @rpc
#    def addOneBuildingSpaceCompositionInformation(self,buildingSpaceInfo):

# 编辑修改一个空间的组合的详细信息如某户型详细的户型信息。
#    @rpc
#    def editOneBuildingSpaceCompositionInformation(self,buildingSpaceInfo):

# 根据条件删除一个空间的组合的详细信息如某户型详细的户型信息。
#    @rpc
#    def delOneBuildingSpaceCompositionInformation(self,buildingSpaceInfo):
