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

class buildingSpaceCompositionInformationService:

    name = "buildingSpaceCompositionInformationService"
    buildingSpaceCompositionInformation_json = "./json/buildingSpaceCompositionInformation2.json"
    buildingSpaceCompositionInformation_schema = "./json/json_schema/buildingSpaceCompositionSchema2.json"
    buildingSpaceCompositionInformationTemplate_schema = "./json/json_schema/buildingSpaceCompositionTemplateSchema2.json"
    loader = None

    def __init__(self):

        self.loader = verifyJsonFileLoader.verifyJsonFileLoader(self.buildingSpaceCompositionInformation_json,
                self.buildingSpaceCompositionInformation_schema)

        self.tmpLoader = verifyJsonFileLoader.verifyJsonFileLoader(None,
                self.buildingSpaceCompositionInformationTemplate_schema)

# 查询所有配置好的空间的组合的详细信息，包括详细的户型信息、楼梯间、公共通道等。
    @rpc
    def getAllBuildingSpaceCompositionInformation(self,projectInfo): 
        # ORG && PROJ {"organizationCode":organizationCode,"projectId":projectId}

        nameko_logger.info("before call -: getAllBuildingSpaceCompositionInformation")
        nameko_logger.info(f" request params {projectInfo['organizationCode']}-{projectInfo['projectId']}")

        data = self.loader.loadJsonFile(self.buildingSpaceCompositionInformation_json)

        if data is None:
            nameko_logger.error(f"file '{self.buildingSpaceCompositionInformation_json}' is empty")
            return -1, data, f"file '{self.buildingSpaceCompositionInformation_json}' is empty"

        elif data == 'ERROR_OBJECTDATA_IN_JSON_FILE':
            return -1, None, f"file '{self.buildingSpaceCompositionInformation_json},ERROR_OBJECTDATA_IN_JSON_FILE"
        '''
        else:
            nameko_logger.error(f"file '{self.buildingSpaceCompositionInformation_json}' ,Invalid JSON object")
            return -2, None, f"file '{self.buildingSpaceCompositionInformation_json}' ,Invalid JSON object"i
        '''

        try:
            organizations = data["organizations"]

            for i,org in enumerate(organizations):
                if org["organizationCode"] == projectInfo["organizationCode"]:
                    projects = org["projects"]

                    if projectInfo["projectId"] is None or projectInfo["projectId"] == '' :
                        return 0,org["projects"],f"get {projectInfo['organizationCode']}'s BuildingSpaceCompositionInformation successfully"
                    else:
                        for j,prj in enumerate(projects):
                            if prj["projectId"] == projectInfo["projectId"]:
                                #return 0,org["projects"][i]["buildingSpace"],f"get {projectInfo['organizationCode']}-{projectInfo['projectId']}'s BuildingSpaceCompositionInformation successfully"
                                #return 0,org[i]["projects"][j]["buildingSpace"],f"get {projectInfo['organizationCode']}-{projectInfo['projectId']}'s BuildingSpaceCompositionInformation successfully"
                                return 0,prj["buildingSpace"],f"get {projectInfo['organizationCode']}-{projectInfo['projectId']}'s BuildingSpaceCompositionInformation successfully"
                        else:
                            return 0,None, f"{projectInfo['organizationCode']}-{projectInfo['projectId']} have no data"
            else:
                return 0,None, f"{projectInfo['organizationCode']}-{projectInfo['projectId']} have no data"
        except Exception as e:
            return -1,None,f"get {projectInfo['organizationCode']}'s BuildingSpaceCompositionInformation failed,{str(e)}"

# 查询某一个配置好的空间的组合的详细信息如某户型详细的户型信息。
    @rpc
    def getOneBuildingSpaceCompositionInformation(self,buildingSpaceInfo):

        # load json 数据文件
        buildingSpaceCompositionInformation_info = self.loader.loadJsonFile(self.buildingSpaceCompositionInformation_json)

        # 检查文件是否存在、是否格式符合要求、是否存在内容
        if buildingSpaceCompositionInformation_info == 'ERROR_FILE_NOT_FOUND': # 文件不存在，则创建
            #create file && append
            return -1,'ERROR_FILE_NOT_FOUND'
        elif buildingSpaceCompositionInformation_info == 'ERROR_OBJECTDATA_IN_JSON_FILE': # 文件json 格式不符合 schema 要求
            return -1,'ERROR_OBJECTDATA_IN_JSON_FILE'
        elif buildingSpaceCompositionInformation_info is None :
            return -1,'ERROR_FILE_IS_EMPTY'

        # 从传入参数json 报文中获取 organizationCode\projectId\buildingSpaceId
        organizationCode = buildingSpaceInfo["organizationCode"]
        projectId = buildingSpaceInfo["projectId"]
        buildingSpaceId = buildingSpaceInfo["buildingSpaceId"]
        #检索并定位buildingSpaceId
        organizations = buildingSpaceCompositionInformation_info["organizations"]
        try:
            # 组织机构循环查找
            for i,org in enumerate(organizations):
                if org["organizationCode"] == organizationCode:
                    projects = org["projects"]
                    # 项目循环查找
                    for j,prj in enumerate(projects):
                        if prj["projectId"] == projectId:
                            buildingSpaces = prj["buildingSpace"]
                            # 建筑空间循环查找
                            for k,bs in enumerate(buildingSpaces):
                                # 查找成功
                                if bs["buildingSpaceId"] == buildingSpaceId:

                                    data = { 
                                            "organizationCode":organizationCode,
                                            "projectId": projectId,
                                            "buildingSpace": bs 
                                    }

                                    return 0,data, "getOneBuildingSpaceCompositionInformation successfully"
                            else:
                                return -1,{},f"getOneBuildingSpaceCompositionInformation failed,invalid buildingSpacesid '{buildingSpaceId}' "
                    else:
                        return -1,{},f"getOneBuildingSpaceCompositionInformation failed,invalid projectId '{projectId}' "
            else:
                return -1,{},f"getOneBuildingSpaceCompositionInformation failed,invalid organizationCode '{organizationCode}' "
        except AttributeError as e:
            nameko_logger.error(f"Error occurred while trying to select a building space from json file '{self.buildingSpaceCompositionInformation_json}'. '{str(e)}'")
            return -2,{},f"Error occurred while trying to select a building space from json file '{self.buildingSpaceCompositionInformation_json}'. '{str(e)}'"

    # 新增一个空间的组合的详细信息如某户型详细的户型信息。
    # 远程代理，用于校验工程信息是否正常
    prjService =  RpcProxy("projectsService")

    @rpc
    def addOneBuildingSpaceCompositionInformation(self,buildingSpaceInfo):

        
        nameko_logger.info(f"input buildingSpaceInfo is {str(buildingSpaceInfo)}")

        res,msg = self.tmpLoader.jsonDataIsValid(buildingSpaceInfo)
        if res != 0:
            return res,msg

        #--------------------
        # 检查组织何项目信息是否正确
        organizationCode = buildingSpaceInfo["organizationCode"]
        projectId = buildingSpaceInfo["projectId"]

        res = self.prjService.getProjectBuildingsIndexFileName(organizationCode,projectId) 
        if res is None:
            return -1,'ERROR_INVALID_PROJECT_INFO'

        #--------------------

        buildingSpaceCompositionInformation_info = self.loader.loadJsonFile(self.buildingSpaceCompositionInformation_json)

        if buildingSpaceCompositionInformation_info == 'ERROR_FILE_NOT_FOUND': # 文件不存在，则创建
            #create file && append
            return -1,'ERROR_FILE_NOT_FOUND'
        elif buildingSpaceCompositionInformation_info == 'ERROR_OBJECTDATA_IN_JSON_FILE': # 文件json格式不符合 schema 要求
            return -1,'ERROR_OBJECTDATA_IN_JSON_FILE'


        organizations = buildingSpaceCompositionInformation_info["organizations"]

        try:

            for i,org in enumerate(organizations):
                if org["organizationCode"] == buildingSpaceInfo["organizationCode"]:
                    projects = org["projects"]
                    for j,prj in enumerate(projects):
                        if prj["projectId"] == buildingSpaceInfo["projectId"]:
                            buildingSpaceCompositionInformation_info["organizations"][i]["projects"][j]["buildingSpace"].append(buildingSpaceInfo["buildingSpace"])

                            res,msg = self.loader.jsonDataIsValid(buildingSpaceCompositionInformation_info)
                            if res == 0:
                                return self.loader.dumpJsonFile(self.buildingSpaceCompositionInformation_json,buildingSpaceCompositionInformation_info), "addOneBuildingSpaceCompositionInformation successfully"
                            else:
                                return res,msg

                    else: #无项目信息
                        appendStr = {
                                    "projectId": buildingSpaceInfo["projectId"],
                                    "buildingSpace": [buildingSpaceInfo["buildingSpace"]] # 转换为arrary
                                 }
                        buildingSpaceCompositionInformation_info["organizations"][i]["projects"].append(appendStr)
                        res,msg = self.loader.jsonDataIsValid(buildingSpaceCompositionInformation_info)
                        if res == 0:
                            return self.loader.dumpJsonFile(self.buildingSpaceCompositionInformation_json,buildingSpaceCompositionInformation_info), "addOneBuildingSpaceCompositionInformation successfully"
                        else:
                            return res,msg

            else: #未曾有该组织的信息
                appendStr = {
                            "organizationCode": buildingSpaceInfo["organizationCode"],
                            "projects": [
                                {
                                    "projectId": buildingSpaceInfo["projectId"],
                                    "buildingSpace": [buildingSpaceInfo["buildingSpace"]]
                                 }
                            ]
                         }
                
                buildingSpaceCompositionInformation_info["organizations"].append(appendStr)

                res,msg = self.loader.jsonDataIsValid(buildingSpaceCompositionInformation_info)
                if res == 0:
                    return self.loader.dumpJsonFile(self.buildingSpaceCompositionInformation_json,buildingSpaceCompositionInformation_info), "addOneBuildingSpaceCompositionInformation successfully"
                else:
                    return res,msg

        except AttributeError as e:
            nameko_logger.error(f"Error occurred while trying to insert an object into the json file. '{buildingSpaceInfo}','{str(e)}'")
            return -1, f"Error occurred while trying to insert an object into the json file. '{buildingSpaceInfo}'"

# 编辑修改一个空间的组合的详细信息如某户型详细的户型信息。
    @rpc
    def editOneBuildingSpaceCompositionInformation(self,buildingSpaceInfo):

        nameko_logger.info(f"input buildingSpaceInfo is {str(buildingSpaceInfo)}")
        #  校验参数格式是否正确
        res,msg = self.tmpLoader.jsonDataIsValid(buildingSpaceInfo)
        if res != 0:
            return res,msg

        # load json 数据文件
        buildingSpaceCompositionInformation_info = self.loader.loadJsonFile(self.buildingSpaceCompositionInformation_json)

        # 检查文件是否存在、是否格式符合要求、是否存在内容
        if buildingSpaceCompositionInformation_info == 'ERROR_FILE_NOT_FOUND': # 文件不存在，则创建
            #create file && append
            return -1,'ERROR_FILE_NOT_FOUND'
        elif buildingSpaceCompositionInformation_info == 'ERROR_OBJECTDATA_IN_JSON_FILE': # 文件json 格式不符合 schema 要求
            return -1,'ERROR_OBJECTDATA_IN_JSON_FILE'
        elif buildingSpaceCompositionInformation_info is None :
            return -1,'ERROR_FILE_IS_EMPTY'

        # 从传入参数json 报文中获取 organizationCode\projectId\buildingSpaceId
        organizationCode = buildingSpaceInfo["organizationCode"]
        projectId = buildingSpaceInfo["projectId"]
        buildingSpaceId = buildingSpaceInfo["buildingSpace"]["buildingSpaceId"]
#        buildingSpace = buildingSpaceCompositionInformation_info["buildingSpace"]
        #nameko_logger.info(str(buildingSpace))
        #检索并定位buildingSpaceId
        organizations = buildingSpaceCompositionInformation_info["organizations"]
        try:
            # 组织机构循环查找
            for i,org in enumerate(organizations):
                if org["organizationCode"] == organizationCode:
                    projects = org["projects"]
                    # 项目循环查找
                    for j,prj in enumerate(projects):
                        if prj["projectId"] == projectId:
                            buildingSpaces = prj["buildingSpace"]
                            # 建筑空间循环查找
                            for k,bs in enumerate(buildingSpaces):
                                # 查找成功
                                if bs["buildingSpaceId"] == buildingSpaceId:
                                    # 修改 buildingSpace 内容
                                    buildingSpaceCompositionInformation_info["organizations"][i]["projects"][j]["buildingSpace"][k] = buildingSpaceInfo["buildingSpace"]
                                    # 校验修改后的格式
                                    res,msg = self.loader.jsonDataIsValid(buildingSpaceCompositionInformation_info)
                                    if res == 0:
                                        # dump 写入文件，并返回
                                        return self.loader.dumpJsonFile(self.buildingSpaceCompositionInformation_json,buildingSpaceCompositionInformation_info), "editBuildingSpaceCompositionInformation successfully"
                                    else: # 校验格式错误
                                        return res,msg
                            else:
                                return -1,f"editBuildingSpaceCompositionInformation failed,invalid buildingSpacesid '{buildingSpaceId}' "
                    else:
                        return -1,f"editBuildingSpaceCompositionInformation failed,invalid projectId '{projectId}' "
            else:
                return -1,f"editBuildingSpaceCompositionInformation failed,invalid organizationCode '{organizationCode}' "
        except AttributeError as e:
            nameko_logger.error(f"Error occurred while trying to edit json file '{self.buildingSpaceCompositionInformation_json}'. '{str(e)}'")
            return -2,f"Error occurred while trying to edit json file '{self.buildingSpaceCompositionInformation_json}'. '{str(e)}'"

# 根据条件删除一个空间的组合的详细信息如某户型详细的户型信息。
    @rpc
    def delOneBuildingSpaceCompositionInformation(self,buildingSpaceInfo):

        # load json 数据文件
        buildingSpaceCompositionInformation_info = self.loader.loadJsonFile(self.buildingSpaceCompositionInformation_json)

        # 检查文件是否存在、是否格式符合要求、是否存在内容
        if buildingSpaceCompositionInformation_info == 'ERROR_FILE_NOT_FOUND': # 文件不存在，则创建
            #create file && append
            return -1,'ERROR_FILE_NOT_FOUND'
        elif buildingSpaceCompositionInformation_info == 'ERROR_OBJECTDATA_IN_JSON_FILE': # 文件json 格式不符合 schema 要求
            return -1,'ERROR_OBJECTDATA_IN_JSON_FILE'
        elif buildingSpaceCompositionInformation_info is None :
            return -1,'ERROR_FILE_IS_EMPTY'

        # 从传入参数json 报文中获取 organizationCode\projectId\buildingSpaceId
        organizationCode = buildingSpaceInfo["organizationCode"]
        projectId = buildingSpaceInfo["projectId"]
        buildingSpaceId = buildingSpaceInfo["buildingSpaceId"]
        #检索并定位buildingSpaceId
        organizations = buildingSpaceCompositionInformation_info["organizations"]
        try:
            # 组织机构循环查找
            for i,org in enumerate(organizations):
                if org["organizationCode"] == organizationCode:
                    projects = org["projects"]
                    # 项目循环查找
                    for j,prj in enumerate(projects):
                        if prj["projectId"] == projectId:
                            buildingSpaces = prj["buildingSpace"]
                            # 建筑空间循环查找
                            for k,bs in enumerate(buildingSpaces):
                                # 查找成功
                                if bs["buildingSpaceId"] == buildingSpaceId:
                                    # delete buildingSpace 内容
                                    buildingSpaceCompositionInformation_info["organizations"][i]["projects"][j]["buildingSpace"].remove(bs)
                                    # 校验修改后的格式
                                    res,msg = self.loader.jsonDataIsValid(buildingSpaceCompositionInformation_info)
                                    if res == 0:
                                        # dump 写入文件，并返回
                                        return self.loader.dumpJsonFile(self.buildingSpaceCompositionInformation_json,buildingSpaceCompositionInformation_info), "delOneBuildingSpaceCompositionInformation successfully"
                                    else: # 校验格式错误
                                        return res,msg
                            else:
                                return -1,f"delOneBuildingSpaceCompositionInformation failed,invalid buildingSpacesid '{buildingSpaceId}' "
                    else:
                        return -1,f"delOneBuildingSpaceCompositionInformation failed,invalid projectId '{projectId}' "
            else:
                return -1,f"delOneBuildingSpaceCompositionInformation failed,invalid organizationCode '{organizationCode}' "
        except AttributeError as e:
            nameko_logger.error(f"Error occurred while trying to delete a building space from json file '{self.buildingSpaceCompositionInformation_json}'. '{str(e)}'")
            return -2,f"Error occurred while trying to delete a building space from json file '{self.buildingSpaceCompositionInformation_json}'. '{str(e)}'"
