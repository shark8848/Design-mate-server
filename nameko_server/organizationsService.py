# -*- coding: utf-8 -*-
"""
Program: orgazinationsService.py
Description: 提供orgazinationsService 的微服务(包括查询、新增、修改等服务)，供flask api 接口调用.
Author: Sunhy
Version: 1.0
Date: 2023-03-05
"""

from nameko.rpc import rpc
#from nameko.service import Service
import json,sys,jsonschema
#from jsonschema import Draft7Validator
import verifyJsonFileLoader
import jsonDataValidator
sys.path.append("..")
#from apocolib.apocolog4p import apoLogger as apolog
from apocolib.NamekoLogger import namekoLogger as nameko_logger

class organizationsService:

    name = "organizationsService"
    # 存储所有组织机构信息的json文件
    organizationsInformation_json = "./json/organizationsInformation.json"
    # organizationsInformation_json 文件的数据的 schema ,在维护整个数据时，打开文件时对文件数据进行校验,避免文件数据格式错误
    organizations_schema = "./json/json_schema/organizationsSchema.json"
    # 单个组织机构的信息的 数据 schema ,用户在组织机构信息维护过程中校验数据,比如新增、和修改
    organizationTemplate_schema = "./json/json_schema/organizationTemplateSchema.json"
    loader = None
    jdValidator = None

    def __init__(self):
        self.loader = verifyJsonFileLoader.verifyJsonFileLoader(self.organizationsInformation_json,
                self.organizations_schema)
        self.jdValidator = jsonDataValidator.jsonDataValidator(self.organizationTemplate_schema)

# 查询所有组织机构详细信息。
    @rpc
    def getAllOrganizationsInformation(self):
        nameko_logger.info("before call -: getAllOrganizationsInformation")
        data = self.loader.loadJsonFile(self.organizationsInformation_json)
        nameko_logger.info("jsonfile data loaded")

        if data is None:
            nameko_logger.error(f"file '{self.organizationsInformation_json}' is empty")
            return -1, None, f"file '{self.organizationsInformation_json}' is empty"

        elif data != 'ERROR_OBJECTDATA_IN_JSON_FILE':
            return 0, data, "getAllOrganizationsInformation successfully"
        else:
            nameko_logger.error(f"file '{self.organizationsInformation_json}' ,Invalid JSON object")
            return -2, None, f"file '{self.organizationsInformation_json}' ,Invalid JSON object"

# 查询某一个组织机构的详细信息。
    @rpc
    def getOneOrganizationInformation(self,organizationCode):

        organizationsInformation_info  = self.loader.loadJsonFile(self.organizationsInformation_json)

        # 检查文件是否存在、是否格式符合要求、是否存在内容
        if organizationsInformation_info == 'ERROR_FILE_NOT_FOUND': # 文件不存在，则创建
            #create file && append
            return -1,None,'ERROR_FILE_NOT_FOUND'
        elif organizationsInformation_info == 'ERROR_OBJECTDATA_IN_JSON_FILE': # 文件json 格式不符合 schema 要求
            return -1,None,'ERROR_OBJECTDATA_IN_JSON_FILE'
        elif organizationsInformation_info is None :
            return -1,None,'ERROR_FILE_IS_EMPTY'

        #检索并定位organizationCode
        organizations = organizationsInformation_info["organizations"]
        i = 0
        for org in organizations:
            if org["organizationCode"] == organizationCode:
                return 0,organizations[i],f"getOneOrganizationInformation '{organizationCode}' successfully"
            else:
                i = i + 1

        return -2,None,"no data"

# 判断组织机构代码是否有效
    @rpc
    def isAvalidOrganization(self,organizationCode):
        res,data,msg = self.getOneOrganizationInformation(organizationCode)
        return(res == 0)

# 新增一个空间的组合的详细信息如某户型详细的户型信息。
    @rpc
    def addOneOrganizationInformation(self,organizationInfo):

        # 校验输入json data 是否符合
        nameko_logger.info(f"organizationInfo  '{organizationInfo}' ")
        res,msg = self.jdValidator.jsonDataIsValid(organizationInfo)
        nameko_logger.info(f"return '{res}' msg '{msg}'")
        if res != 0:
            return res,msg

        orgCode = organizationInfo["organizationCode"]

        nameko_logger.info(f"orgCode '{orgCode}' ")


        organizationsInformation_info  = self.loader.loadJsonFile(self.organizationsInformation_json)

        # 检查文件是否存在、是否格式符合要求、是否存在内容
        if organizationsInformation_info == 'ERROR_FILE_NOT_FOUND': # 文件不存在，则创建
            #create file && append
            return -1,None,'ERROR_FILE_NOT_FOUND'
        elif organizationsInformation_info == 'ERROR_OBJECTDATA_IN_JSON_FILE': # 文件json 格式不符合 schema 要求
            return -1,None,'ERROR_OBJECTDATA_IN_JSON_FILE'
        elif organizationsInformation_info is None :
            return -1,None,'ERROR_FILE_IS_EMPTY'

        organizations = organizationsInformation_info["organizations"]

        # 检查是否存在重复的 organizationCode
        for org in organizations:
            if org["organizationCode"] == orgCode:
                nameko_logger.error(f"Duplicate organizationCode '{orgCode}'")
                return -1, f"Duplicate organizationCode '{orgCode}'"

        else:
            try:
                organizationsInformation_info["organizations"].append(organizationInfo)
                res,msg = self.loader.jsonDataIsValid(organizationsInformation_info)
                if res == 0:
                    return self.loader.dumpJsonFile(self.organizationsInformation_json,organizationsInformation_info), "addOneOrganizationInformation successfully"
                else:
                    return res,msg
            except AttributeError as e:
                nameko_logger.error(f"Error occurred while trying to insert an object into the json file. '{organizationInfo}','{str(e)}'")
                return -1, f"Error occurred while trying to insert an object into the json file. '{organizationInfo}'"

# 编辑修改一个组织机构的详细信息。
    @rpc
    def editOneOrganizationInformation(self,organizationInfo):

        # 校验输入json data 是否符合
        nameko_logger.info(f"organizationInfo  '{organizationInfo}' ")
        res,msg = self.jdValidator.jsonDataIsValid(organizationInfo)
        nameko_logger.info(f"return '{res}' msg '{msg}'")
        if res != 0:
            return res,msg

        orgCode = organizationInfo["organizationCode"]

        nameko_logger.info(f"orgCode '{orgCode}' ")

        organizationsInformation_info  = self.loader.loadJsonFile(self.organizationsInformation_json)

        # 检查文件是否存在、是否格式符合要求、是否存在内容
        if organizationsInformation_info == 'ERROR_FILE_NOT_FOUND': # 文件不存在，则创建
            #create file && append
            return -1,None,'ERROR_FILE_NOT_FOUND'
        elif organizationsInformation_info == 'ERROR_OBJECTDATA_IN_JSON_FILE': # 文件json 格式不符合 schema 要求
            return -1,None,'ERROR_OBJECTDATA_IN_JSON_FILE'
        elif organizationsInformation_info is None :
            return -1,None,'ERROR_FILE_IS_EMPTY'

        organizations = organizationsInformation_info["organizations"]

        # 检查是否存在重复的 organizationCode
        i = 0
        for org in organizations:
            if org["organizationCode"] == orgCode:
                try:
                    organizationsInformation_info["organizations"][i] = organizationInfo
                    res,msg = self.loader.jsonDataIsValid(organizationsInformation_info)
                    if res == 0:
                        return self.loader.dumpJsonFile(self.organizationsInformation_json,organizationsInformation_info), "editOneOrganizationInformation successfully"
                    else:
                        return res,msg
                except AttributeError as e:
                    nameko_logger.error(f"Error occurred while trying to update an object into the json file. '{organizationInfo}','{str(e)}'")
            else:
                i = i + 1
        return -1,f"Error occurred while trying to update an object into the json file. '{orgCode}' is not found"


# 根据条件删除一个组织机构信息。
    @rpc
    def delOneOrganizationInformation(self,organizationCode):

        nameko_logger.info(f"organizationCode is '{organizationCode}' ")
        if organizationCode is None:
            return -1,"request parameter organizationCode is null"

        organizationsInformation_info  = self.loader.loadJsonFile(self.organizationsInformation_json)

        # 检查文件是否存在、是否格式符合要求、是否存在内容
        if organizationsInformation_info == 'ERROR_FILE_NOT_FOUND': # 文件不存在，则创建
            #create file && append
            return -1,None,'ERROR_FILE_NOT_FOUND'
        elif organizationsInformation_info == 'ERROR_OBJECTDATA_IN_JSON_FILE': # 文件json 格式不符合 schema 要求
            return -1,None,'ERROR_OBJECTDATA_IN_JSON_FILE'
        elif organizationsInformation_info is None :
            return -1,None,'ERROR_FILE_IS_EMPTY'

        organizations = organizationsInformation_info["organizations"]

        # 检查是否存在重复的 organizationCode
        i = 0
        for org in organizations:
            if org["organizationCode"] == organizationCode:
                try:
                    organizationsInformation_info["organizations"].remove(org)
                    res,msg = self.loader.jsonDataIsValid(organizationsInformation_info)
                    if res == 0:
                        return self.loader.dumpJsonFile(self.organizationsInformation_json,organizationsInformation_info), "delOneOrganizationInformation successfully"
                    else:
                        return res,msg
                except AttributeError as e:
                    nameko_logger.error(f"Error occurred while trying to delete an object into the json file. '{organizationCode}','{str(e)}'")
            else:
                i = i + 1
        else:
            return -1,f"Error occurred while trying to delete an object into the json file. '{organizationCode}' not found"
