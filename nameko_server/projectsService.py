# -*- coding: utf-8 -*-
"""
Program: projectsService.py
Description: 提供projectsService 的微服务(包括查询、新增、修改等服务)，供flask api 接口调用.
Author: Sunhy
Version: 1.0
Date: 2023-03-05
"""

from nameko.rpc import rpc,RpcProxy
#from nameko.service import Service
import json,sys,jsonschema
#from jsonschema import Draft7Validator
import verifyJsonFileLoader
import jsonDataValidator
sys.path.append("..")
from apocolib.NamekoLogger import namekoLogger as nameko_logger
#from apocolib.apocolog4p import apoLogger as apolog

class projectsService:

    name = "projectsService"
    # 存储所有组织机构的project 信息的json文件
    projectsInformation_json = "./json/projectsInformation.json"
    # projectsInformation_json 文件的数据的 schema ,在维护整个数据时，打开文件时对文件数据进行校验,避免文件数据格式错误
    projects_schema = "./json/json_schema/projectsSchema.json"
    # 单个组织机构的信息的 数据 schema ,用户在组织机构信息维护过程中校验数据,比如新增、和修改
    projectTemplate_schema = "./json/json_schema/projectTemplateSchema.json"
    loader = None
    jdValidator = None

    def __init__(self):
        self.loader = verifyJsonFileLoader.verifyJsonFileLoader(self.projectsInformation_json,
                self.projects_schema)
        self.jdValidator = jsonDataValidator.jsonDataValidator(self.projectTemplate_schema)

# 查询所有组织机构详细信息。
    @rpc
    def getAllProjectsInformation(self):
        nameko_logger.info("before call -: getAllProjectsInformation")
        data = self.loader.loadJsonFile(self.projectsInformation_json)
        nameko_logger.info("jsonfile data loaded")

        if data is None:
            nameko_logger.error(f"file '{self.projectsInformation_json}' is empty")
            return -1, None, f"file '{self.projectsInformation_json}' is empty"

        elif data != 'ERROR_OBJECTDATA_IN_JSON_FILE':
            return 0, data, "getAllProjectsInformation successfully"
        else:
            nameko_logger.error(f"file '{self.projectsInformation_json}' ,Invalid JSON object")
            return -2, None, f"file '{self.projectsInformation_json}' ,Invalid JSON object"

# 查询某一个组织机构的详细信息。
    @rpc
    def getOneProjectInformation(self,organizationCode,projectId):

        if (organizationCode is None ) or (organizationCode == '') :
            return -1,None,'organizationCode is null'

        projectsInformation_info  = self.loader.loadJsonFile(self.projectsInformation_json)

        # 检查文件是否存在、是否格式符合要求、是否存在内容
        if projectsInformation_info == 'ERROR_FILE_NOT_FOUND': # 文件不存在，则创建
            #create file && append
            return -1,None,'ERROR_FILE_NOT_FOUND'
        elif projectsInformation_info == 'ERROR_OBJECTDATA_IN_JSON_FILE': # 文件json 格式不符合 schema 要求
            return -1,None,'ERROR_OBJECTDATA_IN_JSON_FILE'
        elif projectsInformation_info is None :
            return -1,None,'ERROR_FILE_IS_EMPTY'

        #检索并定位organizationCode \ projectId

        organizations = projectsInformation_info["organizations"]

#        for i,org in enumerate(organizations):
        for org in organizations:
            if org["organizationCode"] == organizationCode:
                projects = org["projects"]
                #如何查询条件中 projectId 为空，则返回该 organizationCode 下所有项目信息,否则，则再匹配 projectId 检索 
                if (projectId is None) or (projectId == '') :
                    return 0,projects,f"get '{organizationCode}' ProjectsInformation successfully"
                else:
                    for j,prg in enumerate(projects):
                        if prg["projectId"] == projectId:
                            return 0,projects[j],f"get '{organizationCode}' ProjectsInformation '{projectId}' successfully"

        return -2,None,"no data found"

# 检索查询 project 的索引文件,用于projectBuildingsService rpc调用
#---------------
    @rpc
    def getProjectBuildingsIndexFileName(self,organizationCode,projectId):
        res,data,msg = self.getOneProjectInformation(organizationCode,projectId)
        nameko_logger.info(f"---- '{str(data)}' ---- '{msg}'")
        if res == 0:
            return data["projectBuildingsIndexFile"]
        else:
            return None

#---------------
# 在某组织机构下新增一个项目的详细信息。
    orgService =  RpcProxy("organizationsService")

    @rpc
    def addOneProjectInformation(self,projectInfo):

        # 校验输入json data 是否符合
        nameko_logger.info(f"projectInfo  '{projectInfo}' ")
        res,msg = self.jdValidator.jsonDataIsValid(projectInfo)
        nameko_logger.info(f"return '{res}' msg '{msg}'")
        if res != 0:
           return res,msg

        organizationCode = projectInfo["organizationCode"]

        if self.orgService.isAvalidOrganization(organizationCode) == False:
            return -1,f"unknown organizationCode '{organizationCode}'"

        projectId = projectInfo["projects"]["projectId"]
        nameko_logger.info(f"organizationCode '{organizationCode}',projectId '{projectId}' ")
        # 入参中无文件名定义,按组织机构代码+项目id，进行生成。
        prgIndexFile = './json/' + organizationCode+'-'+projectId+'-Index.json'
        projectInfo["projects"]["projectBuildingsIndexFile"] = prgIndexFile

        # 校验输入json data 是否符合
#        nameko_logger.info(f"projectInfo  '{projectInfo}' ")
#        res,msg = self.jdValidator.jsonDataIsValid(projectInfo)
#        nameko_logger.info(f"return '{res}' msg '{msg}'")
#        if res != 0:
#            return res,msg

        # load 系统projectsInformation 文件
        projectsInformation_info  = self.loader.loadJsonFile(self.projectsInformation_json)

        # 检查文件是否存在、是否格式符合要求、是否存在内容
        if projectsInformation_info == 'ERROR_FILE_NOT_FOUND': # 文件不存在，则创建
            #create file && append
            return -1,'ERROR_FILE_NOT_FOUND'
        elif projectsInformation_info == 'ERROR_OBJECTDATA_IN_JSON_FILE': # 文件json 格式不符合 schema 要求
            return -1,'ERROR_OBJECTDATA_IN_JSON_FILE'
        elif projectsInformation_info is None :
            return -1,'ERROR_FILE_IS_EMPTY'

        #检索并定位organizationCode \ projectId

        organizations = projectsInformation_info["organizations"]

        for i,org in enumerate(organizations):
            if org["organizationCode"] == organizationCode:
                projects = org["projects"]
                for j,prg in enumerate(projects):
                    if prg["projectId"] == projectId:
                        return -1, f"Duplicate projectId '{projectId}'"
                else:# 在组织机构下直接添加项目
                    try:
                        projectsInformation_info["organizations"][i]["projects"].append(projectInfo["projects"])
                        res,msg = self.loader.jsonDataIsValid(projectsInformation_info)
                        if res == 0:
                            nameko_logger.info(f"prgIndexFile," 'prgIndexFile')
                            #create a prgIndexFile
                            with open(prgIndexFile, "x") as f:
                                pass
                            return self.loader.dumpJsonFile(self.projectsInformation_json,projectsInformation_info), "addOneProjectInformation successfully"
                        else:
                            return res,msg
                    except AttributeError as e:
                        nameko_logger.error(f"Error occurred while trying to insert an object into the json file. '{projectInfo}','{str(e)}'")
                        return -1, f"Error occurred while trying to insert an object into the json file. '{projectInfo}'"
        else: #在projectsInformation 中没有该组织，则直接添加包括组织+项目在内的所有信息.
            try:
                #将其中的project元素转换为数组，解决与schema匹配问题
                projectInfo["projects"] = [projectInfo["projects"]]
                projectsInformation_info["organizations"].append(projectInfo)
                res,msg = self.loader.jsonDataIsValid(projectsInformation_info)
                if res == 0:
                    nameko_logger.info(f"prgIndexFile," 'prgIndexFile')
                    #create a prgIndexFile
                    with open(prgIndexFile, "x") as f:
                        pass
                    return self.loader.dumpJsonFile(self.projectsInformation_json,projectsInformation_info), "addOneProjectInformation successfully"
                else:
                    return res,msg
            except AttributeError as e:
                nameko_logger.error(f"Error occurred while trying to insert an object into the json file. '{projectInfo}','{str(e)}'")
                return -1, f"Error occurred while trying to insert an object into the json file. '{projectInfo}'"


# 修改某组织机构下的某个项目的详细信息。
    @rpc
    def editOneProjectInformation(self,projectInfo):

        # 校验输入json data 是否符合
        nameko_logger.info(f"projectInfo  '{projectInfo}' ")
        res,msg = self.jdValidator.jsonDataIsValid(projectInfo)
        nameko_logger.info(f"return '{res}' msg '{msg}'")
        if res != 0:
            return res,msg

        organizationCode = projectInfo["organizationCode"]
        projectId = projectInfo["projects"]["projectId"]
        nameko_logger.info(f"organizationCode '{organizationCode}',projectId '{projectId}' ")

        # load 系统projectsInformation 文件
        projectsInformation_info  = self.loader.loadJsonFile(self.projectsInformation_json)

        # 检查文件是否存在、是否格式符合要求、是否存在内容
        if projectsInformation_info == 'ERROR_FILE_NOT_FOUND': # 文件不存在，则创建
            #create file && append
            return -1,'ERROR_FILE_NOT_FOUND'
        elif projectsInformation_info == 'ERROR_OBJECTDATA_IN_JSON_FILE': # 文件json 格式不符合 schema 要求
            return -1,'ERROR_OBJECTDATA_IN_JSON_FILE'
        elif projectsInformation_info is None :
            return -1,'ERROR_FILE_IS_EMPTY'

        #检索并定位organizationCode \ projectId

        organizations = projectsInformation_info["organizations"]

        for i,org in enumerate(organizations):
            if org["organizationCode"] == organizationCode:
                projects = org["projects"]
                for j,prg in enumerate(projects):
                    if prg["projectId"] == projectId:
                        try:
                            #projectsInformation_info["organizations"][i]["projects"][j] = projectInfo["projects"]
                            projectsInformation_info["organizations"][i]["projects"][j].update(projectInfo["projects"])
                            res,msg = self.loader.jsonDataIsValid(projectsInformation_info)
                            if res == 0:
                                return self.loader.dumpJsonFile(self.projectsInformation_json,projectsInformation_info), "editOneProjectInformation successfully"
                            else:
                                return res,msg
                        except AttributeError as e:
                            nameko_logger.error(f"Error occurred while trying edit insert an object in the json file. '{projectInfo}','{str(e)}'")
                            return -1, f"Error occurred while trying to edit an object in the json file. '{projectInfo}'"
                else:
                    return -1,f"unknown organizationCode '{organizationCode}', projectId '{projectId}'"
        else:
            return -1,f"unknown organizationCode '{organizationCode}', projectId '{projectId}'"

# 删除某组织机构下的某个项目的详细信息。
    @rpc
    def delOneProjectInformation(self,projectInfo):

        organizationCode = projectInfo["organizationCode"]
        projectId = projectInfo["projectId"]
        nameko_logger.info(f"organizationCode '{organizationCode}',projectId '{projectId}' ")
        if ( organizationCode is None ) or ( organizationCode == '' ) or ( projectId is None ) or ( projectId == '' ):
            return -1,"invalid parameter organizationCode '{organizationCode}' , projectId '{projectId}'"

        # load 系统projectsInformation 文件
        projectsInformation_info  = self.loader.loadJsonFile(self.projectsInformation_json)

        # 检查文件是否存在、是否格式符合要求、是否存在内容
        if projectsInformation_info == 'ERROR_FILE_NOT_FOUND': # 文件不存在，则创建
            #create file && append
            return -1,'ERROR_FILE_NOT_FOUND'
        elif projectsInformation_info == 'ERROR_OBJECTDATA_IN_JSON_FILE': # 文件json 格式不符合 schema 要求
            return -1,'ERROR_OBJECTDATA_IN_JSON_FILE'
        elif projectsInformation_info is None :
            return -1,'ERROR_FILE_IS_EMPTY'

        #检索并定位organizationCode \ projectId

        organizations = projectsInformation_info["organizations"]

        for i,org in enumerate(organizations):
            if org["organizationCode"] == organizationCode:
                projects = org["projects"]
                for j,prg in enumerate(projects):
                    if prg["projectId"] == projectId:
                        try:
                            #删除当前project
                            projectsInformation_info["organizations"][i]["projects"].remove(prg)
                            #校验删除后的内容格式
                            res,msg = self.loader.jsonDataIsValid(projectsInformation_info)
                            if res == 0:
                                #dump 到文件中
                                return self.loader.dumpJsonFile(self.projectsInformation_json,projectsInformation_info), "delOneProjectInformation successfully"
                            else:
                                return res,msg
                        except AttributeError as e:
                            nameko_logger.error(f"Error occurred while trying to delete an object in the json file. '{projectInfo}','{str(e)}'")
                            return -1, f"Error occurred while trying to delete an object in the json file. '{projectInfo}'"
                else:
                    return -1,f"unknown organizationCode '{organizationCode}', projectId '{projectId}'"
        else:
            return -1,f"unknown organizationCode '{organizationCode}', projectId '{projectId}'"
