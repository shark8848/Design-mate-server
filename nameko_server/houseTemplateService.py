# -*- coding: utf-8 -*-
"""
Program: houseTemplateService.py
Description: 提供模板管理 的微服务(包括查询、新增、修改等服务)，供flask api 接口调用.
Author: Sunhy
Version: 1.0
Date: 2023-07-01
"""

from nameko.rpc import rpc,RpcProxy
import json,sys,jsonschema
import verifyJsonFileLoader
from houseModel import HouseTemplate
from datetime import datetime
import pdb

sys.path.append("..")
from apocolib.NamekoLogger import namekoLogger as nameko_logger
from apocolib import sqliteSession as sqlSession

class houseTemplateService:

    name = "houseTemplateService"
    houseTemplate_schema = "../json/json_schema/houseTemplateSchema.json"
    loader = None

    def __init__(self):

#        self.loader = verifyJsonFileLoader.verifyJsonFileLoader(self.buildingSpaceCompositionInformation_json,
#                self.buildingSpaceCompositionInformation_schema)

        self.tmpLoader = verifyJsonFileLoader.verifyJsonFileLoader(None,
                self.houseTemplate_schema)
    # 新增一个模板
    @rpc
    def addHouseTemplate(self,houseTemplate):

        #pdb.set_trace()

        # 使用模板校验json 格式是否有效
        res,msg = self.tmpLoader.jsonDataIsValid(houseTemplate)
        if res != 0:
            return res,msg
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        f_path = f'../json/templates/ht_{timestamp}.json'
        res = self.tmpLoader.dumpJsonFile(f_path,houseTemplate)
        if res == 0:
            return self.registerHouseTemplate(houseTemplate,f_path)
        else:
            return -1,'saveHouseTemplateToJsonFile faild'

    def registerHouseTemplate(self,houseTemplate,f_path):
        #pdb.set_trace()
        nameko_logger.info(houseTemplate)
        templateName = None
        try:
            organizationCode = houseTemplate['organizationCode']
            projectId = houseTemplate['projectId']
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
            templateId = organizationCode + projectId + timestamp
            buildingId = None
            templateName = houseTemplate["houseTemplate"]["templateName"]
            templateJson = f_path
            status = 1

            with sqlSession.sqliteSession().getSession() as session:

                h_t = HouseTemplate(
                    organizationCode = organizationCode,
                    projectId = projectId,
                    templateId = templateId,
                    buildingId = buildingId,
                    templateName = templateName,
                    templateJson = templateJson,
                    status = status
                )
                session.add(h_t)
                session.commit()
                return 0,f'add houseTemplate {templateName} successfully'

        except Exception as e:
            nameko_logger.error(f'add houseTemplate {templateName} error.{str(e)}')
            #raise Exception(f'add user {userId} error.{str(e)}')
            return -1, f'add houseTemplate {templateName} failed: {str(e)}'
        return -1, 'add houseTemplate failed'

    # 查询模板列表
    @rpc
    def getHouseTemplates(self,projectInfo): 
        #pdb.set_trace()
        # ORG && PROJ {"organizationCode":organizationCode,"projectId":projectId}
        organizationCode = None
        nameko_logger.info(projectInfo)
        try:
            organizationCode = projectInfo["organizationCode"]
            projectId = projectInfo["projectId"]
            with sqlSession.sqliteSession().getSession() as session:
                houseTemplates = []
                if projectId:
                    # 查询指定 projectId 的记录
                    houseTemplates = session.query(HouseTemplate).filter(
                        HouseTemplate.organizationCode == organizationCode,
                        HouseTemplate.projectId == projectId,
                        HouseTemplate.status == 1
                    ).all()
                else:
                    # 查询满足其他条件下的所有记录
                    houseTemplates = session.query(HouseTemplate).filter(
                    HouseTemplate.organizationCode == organizationCode,
                    HouseTemplate.status == 1
                ).all()

                #houseTemplates = session.query(HouseTemplate).filter_by(organizationCode=organizationCode,status=1).all()
                if houseTemplates is None:
                    return -1,[],'get_HouseTemplates {organizationCode} ,no data found'
                else:
                    return 0,[{
                        'organizationCode': houseTemplate.organizationCode,
                        'projectId': houseTemplate.projectId,
                        'templateId': houseTemplate.templateId,
                        'buildingId': houseTemplate.buildingId,
                        'templateName': houseTemplate.templateName,
                        'templateJson': houseTemplate.templateJson,
                        'status': houseTemplate.status
                    } for houseTemplate in houseTemplates],f'get_HouseTemplates {organizationCode} successfully'
        except Exception as e:
            nameko_logger.error(f'get_HouseTemplates {organizationCode} error.{str(e)}')
            return -1, [],f'get_HouseTemplates {organizationCode} error.{str(e)}'

        return -1,[],'get_HouseTemplates failed'

    # 查询某一个模板内容
    @rpc
    def getHouseTemplateJson(self,houseTemplateFile):

        houseTemplate  = self.tmpLoader.loadJsonFile(houseTemplateFile)
        if houseTemplate is not None:
            return 0,houseTemplate,'getHouseTemplateJson successfully'
        else:
            return -1,{}, f'getHouseTemplateJson failed ,file {houseTemplateFile} data is none'
'''
    # 修改指定模板
    @rpc
    def editOneHouseTemplate(self,houseTemplate):
        return []

    # 删除指定模板
    @rpc
    def delOneHouseTemplate(self,houseTemplate):
        return []


        '''
