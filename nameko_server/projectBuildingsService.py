# -*- coding: utf-8 -*-
"""
Program: projectBuildingsService.py
Description: 提供projectBuildingsService 的微服务(包括查询、新增、修改等服务)，供flask api 接口调用.
             主要实现某个项目的楼栋的详细信息的构件，包括增加楼层，楼层的建筑空间单元的构建组合。
Author: Sunhy
Version: 1.0
Date: 2023-03-04
"""
import os
import shutil
import datetime
from nameko.rpc import rpc,RpcProxy
#from nameko.service import Service
import json,sys,jsonschema
import jsonpatch
#from jsonschema import Draft7Validator
import verifyJsonFileLoader,jsonDataValidator
import indexFileRetriever as ifr
sys.path.append("..")
#from apocolib.apocolog4p import apoLogger as apolog
from apocolib.NamekoLogger import namekoLogger as nameko_logger
import pdb

class projectBuildingsService:

    name = "projectBuildingsService"
    # projectBuildingFilesIndex_json 文件的数据的 schema ,在维护整个数据时，打开文件时对文件数据进行校验,避免文件数据格式错误
    # 文件名是创建project时生成，文件名并存储在 ./json/projectsInformation.json里的 "projectBuildingsIndexFile" 属性值，
    # 同时创建空的 ./json/xxx-xxx-index.json 文件
    projectBuildingFilesIndex_schema = "./json/json_schema/projectBuildingFilesIndexSchema.json"

    # projectBuildingsIndexFile 文件 中定义的楼栋的 schema,
    projectBuildings_schema = "./json/json_schema/projectBuildingsSchema.json"

    # 用于编辑楼层时对楼层详细信息校验的 schema,
    buildingFloorTemplate_schema = "./json/json_schema/buildingFloorTemplateSchema.json"

    # 历史文件保存的相对路径
    historyFilesPath = "./json/history"

    # 远程代理，用于校验工程信息是否正常
    prjService =  RpcProxy("projectsService")
    #loader = None
    pbValidator = None
    bftValidator = None

    def __init__(self):
        self.ifLoader = verifyJsonFileLoader.verifyJsonFileLoader(None,self.projectBuildingFilesIndex_schema)
        self.pbLoader = verifyJsonFileLoader.verifyJsonFileLoader(None,self.projectBuildings_schema)
#        self.bfLoader = verifyJsonFileLoader.verifyJsonFileLoader(None,self.buildingFloorTemplate_schema)
        self.pbValidator = jsonDataValidator.jsonDataValidator(self.projectBuildings_schema)
        self.bftValidator = jsonDataValidator.jsonDataValidator(self.buildingFloorTemplate_schema)
#
# 新建一栋楼的信息,创建楼栋的基础信息。
# 主要逻辑：
# 1.在项目信息文件projectsInformation.json 中查找到对应的项目信息索引文件名;
# 2. 如果该项目没创建过楼栋，则该文件名无实际文件实体。同步创建该文件。写入该项目索引内容。如 orj001-proj001-Index.json
# 2.1 如果已经存在实体文件，则在索引文件中插入对应的 索引数据
# 例如 { "buildingId": buildingId,"buildingInfoFileName": buildingInfoFileName }
# 3. 为楼栋创建楼栋信息文件并写入楼栋信息。如：orj001-proj001-builing001.json
#
    @rpc
    def addOneProjectBuildingInformation(self,buildingInfo):

        # 校验输入json data 是否符合
        res,msg = self.pbValidator.jsonDataIsValid(buildingInfo)
        nameko_logger.info(f"return '{res}' msg '{msg}'")
        if res != 0:
            return res,msg

        organizationCode = buildingInfo["organizationCode"]
        projectId = buildingInfo["projectId"]
        indexFile = self.prjService.getProjectBuildingsIndexFileName(organizationCode,projectId)
        if indexFile is None:
            return -1,f"organizationCode '{organizationCode}' or projectId '{projectId}' not found"

        nameko_logger.info(f"index file '{indexFile}'")


        # load projectBuildingsIndexFile 文件
        projectBuildingsIndexFile_info  = self.ifLoader.loadJsonFile(indexFile)

        buildingId = buildingInfo["buildingId"] 
        buildingInfoFileName = "./json/{}-{}-{}.json".format(organizationCode, projectId, buildingId)

        # 检查文件是否存在、是否格式符合要求、是否存在内容
        if projectBuildingsIndexFile_info == 'ERROR_FILE_NOT_FOUND': # 文件不存在，则创建
            #create file && append
            return -1,'ERROR_FILE_NOT_FOUND'
        elif projectBuildingsIndexFile_info == 'ERROR_OBJECTDATA_IN_JSON_FILE': # 文件json 格式不符合 schema 要求
            return -1,'ERROR_OBJECTDATA_IN_JSON_FILE'
        elif (projectBuildingsIndexFile_info is None) or (projectBuildingsIndexFile_info == ''):

            nameko_logger.info(f"project '{buildingId}' file '{indexFile}' is not exist, we will create it ")

            pbIndexFile_json = {
                    "organizationCode": organizationCode,
                    "projects": {
                        "projectId": projectId,
                        "buildings": [
                            { "buildingId": buildingId, "buildingInfoFileName": buildingInfoFileName }
                        ]
                    }
            }
            try:
                res,msg = self.ifLoader.jsonDataIsValid(pbIndexFile_json)
                if res == 0:
                    res = self.ifLoader.dumpJsonFile(indexFile,pbIndexFile_json) 
                    if res == -1:
                        nameko_logger.error(f"Error occurred while trying to insert an object in the json file. '{indexFile}'")
                        return -1,f"Error occurred while trying to insert an object in the json file. '{indexFile}'"
                else:
                    nameko_logger.error(msg)
                    return res, msg
            except AttributeError as e:
                nameko_logger.error(f"Error occurred while trying to insert an object in the json file. '{indexFile}','{str(e)}'")
                return -1, f"Error occurred while trying to insert an object in the json file. '{indexFile}'"

        else: # insert into organization-project-Index.json
            try:

                bmg = { "buildingId": buildingId,"buildingInfoFileName": buildingInfoFileName }
                projectBuildingsIndexFile_info["projects"]["buildings"].append(bmg)
                res,msg = self.ifLoader.jsonDataIsValid(projectBuildingsIndexFile_info)
                if res == 0:
                    res = self.pbLoader.dumpJsonFile(indexFile,projectBuildingsIndexFile_info) 
                    if res != 0:
                        nameko_logger.info(f" insert object into '{indexFile}' failed, '{msg}' ")
                        return res,msg
                else:
                    nameko_logger.info(f" insert object into '{indexFile}' failed, '{msg}' ")
                    return res,msg

            except AttributeError as e:
                nameko_logger.error(f"Error occurred while trying to insert an object in the json file. '{indexFile}','{str(e)}'")
                return -1, f"Error occurred while trying to insert an object in the json file. '{indexFile}'"
            except Exception as e:
                nameko_logger.error(f"Error occurred while trying to insert an object in the json file. '{indexFile}','{str(e)}'")
                return -1, f"Error occurred while trying to insert an object in the json file. '{indexFile}'"

        # 将楼栋信息 dump 到 buildingInfoFileName

        try:
            res = self.pbLoader.dumpJsonFile(buildingInfoFileName,buildingInfo)
            if res == 0:
                return res,f"insert object into '{buildingInfoFileName}' successfully"
            else:
                nameko_logger.info(f" insert object into '{buildingInfoFileName}' failed, '{msg}' ")
                return res,msg
        except AttributeError as e:
            nameko_logger.error(f"Error occurred while trying to insert an object in the json file. '{buildingInfoFileName}','{str(e)}'")
            return -1, f"Error occurred while trying to insert an object in the json file. '{buildingInfoFileName}'"
        except Exception as e:
            nameko_logger.error(f"Error occurred while trying to insert an object in the json file. '{buildingInfoFileName}','{str(e)}'")
            return -1, f"Error occurred while trying to insert an object in the json file. '{buildingInfoFileName}'"

        return -2, f"addOneProjectBuildingInformation failed"
# ------------------------------------------------------------
    @rpc
    def editOneProjectBuildingInformation(self,buildingInfo):

        #pdb.set_trace()
        # 校验输入json data 是否符合
        res,msg = self.pbValidator.jsonDataIsValid(buildingInfo)
        nameko_logger.info(f"return '{res}' msg '{msg}'")
        if res != 0:
            return res,msg

        organizationCode = buildingInfo["organizationCode"]
        projectId = buildingInfo["projectId"]
        buildingId = buildingInfo["buildingId"] 

        buildingInfoFileName = "./json/{}-{}-{}.json".format(organizationCode, projectId, buildingId)
        # load buildingInfoFileName
        buildingInformation_info  = self.pbLoader.loadJsonFile(buildingInfoFileName)

        patchBuildingInformation = [
            { "op": "replace", "path": "/buildingName", "value": buildingInfo["buildingName"] },
            { "op": "replace", "path": "/buildingOrientation", "value": buildingInfo["buildingOrientation"] },
            { "op": "replace", "path": "/numberOfFloors", "value": buildingInfo["numberOfFloors"] },
            { "op": "replace", "path": "/buildingHeight", "value": buildingInfo["buildingHeight"] },
            { "op": "replace", "path": "/buildingVolume", "value": buildingInfo["buildingVolume"] },
            { "op": "replace", "path": "/buildingExternalSurfaceArea", "value": buildingInfo["buildingExternalSurfaceArea"] },
            { "op": "replace", "path": "/formFactor", "value": buildingInfo["formFactor"] },
            { "op": "replace", "path": "/northAngle", "value": buildingInfo["northAngle"] },
            { "op": "replace", "path": "/structuralType", "value": buildingInfo["structuralType"] },
            { "op": "replace", "path": "/designLimits", "value": buildingInfo["designLimits"] }
        ]

        nameko_logger.info(f"patch ********* '{patchBuildingInformation}' ****************")

        try:
            # 应用JSON Patch
            data = jsonpatch.apply_patch(buildingInformation_info,patchBuildingInformation)

            res,msg = self.pbLoader.jsonDataIsValid(data)

            if res == 0:
                return self.pbLoader.dumpJsonFile(buildingInfoFileName, data), "editOneProjectBuildingInformation successfully"
            else:
                return res,msg

        except json.JSONDecodeError as e: # the original data or new data is not valid JSON.
            return -1, f"Error occurred while trying to editOneProjectBuildingInformation.'{str(e)}'"
        except KeyError as e: #trying to get a non-existing key from the parsed JSON data.
            return -1, f"Error occurred while trying to editOneProjectBuildingInformation.'{str(e)}'"
        except jsonpatch.JsonPatchException as e: #there is an error in the JSON Patch, such as incorrect format or unable to apply the patch.
            return -1, f"Error occurred while trying to editOneProjectBuildingInformation.'{str(e)}'"
        except AssertionError as e: # the new data generated after applying the patch does not match the expected data.
            return -1, f"Error occurred while trying to editOneProjectBuildingInformation.'{str(e)}'"
        except AttributeError as e: #mistype the name of an attribute, or when the object's class doesn't have the attribute you are looking for.
            return -1, f"Error occurred while trying to editOneProjectBuildingInformation.'{str(e)}'"
        except Exception as e:
            return -1, f"Error occurred while trying to editOneProjectBuildingInformation.'{str(e)}'"

        return -1, f"Error occurred while trying to editOneProjectBuildingInformation. jsonfile- '{buildingInfoFileName}'"

#-----------------------------------------------------------
# 删除楼栋
# 删除项目中的楼栋索引；
# 如果该楼栋已经配置了详细的楼栋信息，则将楼栋的实例文件move 至 历史目录，并更名加上时间戳，如 wanke-proj002-B003.json.20240406XXXXXX
#----------------------------------------------------------
    @rpc
    def deleteOneProjectBuilding(self,buildingInfo):

        organizationCode = buildingInfo["organizationCode"]
        projectId = buildingInfo["projectId"]
        indexFile = self.prjService.getProjectBuildingsIndexFileName(organizationCode,projectId)
        if indexFile is None:
            return -1,f"organizationCode '{organizationCode}' or projectId '{projectId}' not found"

        nameko_logger.info(f"index file '{indexFile}'")

        # load projectBuildingsIndexFile 文件
        projectBuildingsIndexFile_info  = self.ifLoader.loadJsonFile(indexFile)

        buildingId = buildingInfo["buildingId"] 
        buildingInfoFileName = "./json/{}-{}-{}.json".format(organizationCode, projectId, buildingId)

        # 检查文件是否存在、是否格式符合要求、是否存在内容
        if projectBuildingsIndexFile_info == 'ERROR_FILE_NOT_FOUND': # 文件不存在，则创建
            #create file && append
            return -1,'ERROR_FILE_NOT_FOUND'
        elif projectBuildingsIndexFile_info == 'ERROR_OBJECTDATA_IN_JSON_FILE': # 文件json 格式不符合 schema 要求
            return -1,'ERROR_OBJECTDATA_IN_JSON_FILE'
        elif (projectBuildingsIndexFile_info is None) or (projectBuildingsIndexFile_info == ''):
            return -1,'ERROR_OBJECTDATA_DATA_IS_EMPTY'

        # 删除项目中的楼栋索引
        #检索并定位 projectId\building
#        pdb.set_trace()

        buildings = projectBuildingsIndexFile_info["projects"]["buildings"]

        for bld in buildings:
            if bld["buildingId"] == buildingId:
                try: 
                    #删除 building
                    projectBuildingsIndexFile_info["projects"]["buildings"].remove(bld)
                    #校验删除后的内容格式
                    res,msg = self.ifLoader.jsonDataIsValid(projectBuildingsIndexFile_info)
                    if res == 0:
                        #dump 到文件中
                        res = self.ifLoader.dumpJsonFile(indexFile,projectBuildingsIndexFile_info) 
                        if res != 0:
                            return res, f"delete building/project {buildingId}/{projectId}  from {indexFile} failed"
                        else:
                            break
                    else:
                        return res,msg
                except AttributeError as e:
                    nameko_logger.error(f"Error occurred while trying to delete an object in the json file. '{projectInfo}','{str(e)}'")
                    return -1, f"Error occurred while trying to delete an object in the json file. '{projectInfo}'"
        else:
            return -1,f"unknown buildingId {buildingId}, projectId {projectId}"

        # 转移文件到历史目录
        try:
            tempFileName = self.getHistoryFileName(buildingInfoFileName,self.historyFilesPath)
            shutil.move(buildingInfoFileName, tempFileName)
            return 0,f"delete building/project {buildingId}/{projectId}  from {indexFile} successfully,and the history file {buildingInfoFileName} be moved to {tempFileName}"
        except Exception as e:
            nameko_logger.error(f"Error occurred while trying to move {buildingInfoFileName} to {self.historyFilesPath}.{str(e)}")
            return -1,f"Error occurred while trying to move {buildingInfoFileName} to {self.historyFilesPath}.{str(e)}"

    def getHistoryFileName(self,fileName,path):

        # 生成时间戳
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        # 获取原始文件名和扩展名
        file_name, ext = os.path.splitext(os.path.basename(fileName))
        # 构造新的文件名
        new_file_name = f"{file_name}_{timestamp}{ext}"
        # 拼接新的文件路径
        dst_file_path = os.path.join(path, new_file_name)

        return dst_file_path

# 为楼栋添加楼层及楼层的空间
# 可一次性添加多个楼层
    @rpc
    def addFloorsInformationForBuilding(self,floorsInfo):

        # 校验输入json data 是否符合
#        nameko_logger.info(f"floorsInfo  '{floorsInfo}' ")
#        res,msg = self.pbValidator.jsonDataIsValid(floorsInfo)
#        nameko_logger.info(f"return '{res}' msg '{msg}'")
#        if res != 0:
#            return res,msg

        ''' json 示例
        {
            "organizationCode": "orj001",
            "projectId": "proj001",
            "buildingId": "B001",
            "floors": [
                {
                    "floorId": 1,
                    "height",3.0,
                    "buildingSpacesId": ["SPACE001", "SPACE002", "SPACE003", "SPACE004", "SPACE005", "SPACE006"]
                },
                {
                    "floorId": 2,
                    "height",3.0,
                    "buildingSpacesId": ["SPACE001", "SPACE002", "SPACE003", "SPACE004", "SPACE005", "SPACE006"]
                },
                {
                    "floorId": 3,
                    "height",3.0,
                    "buildingSpacesId": ["SPACE001", "SPACE002", "SPACE003", "SPACE004", "SPACE005", "SPACE006"]
                }
            ]
        }
        '''

        organizationCode = floorsInfo["organizationCode"]
        projectId = floorsInfo["projectId"]
        buildingId = floorsInfo["buildingId"]

        nameko_logger.info(f"organizationCode {organizationCode}-{projectId}-{buildingId}")

        res,bfFile = self.getBuildingFileName(organizationCode, projectId, buildingId)
        if res != 0:
            return res,bfFile

        # load bdFile
        buildingInformation_info  = self.pbLoader.loadJsonFile(bfFile)
        try:

            if "floors" in buildingInformation_info:
                buildingInformation_info["floors"].extend(floorsInfo["floors"])
            else:
                buildingInformation_info["floors"] = floorsInfo["floors"]

            res,msg = self.pbLoader.jsonDataIsValid(buildingInformation_info)

            if res == 0:
                return self.pbLoader.dumpJsonFile(bfFile, buildingInformation_info), "addFloorsInformationForBuilding successfully"
            else:
                return res,msg
        except AttributeError as e:
            nameko_logger.error(f"Error occurred while trying to addFloorsInformationForBuilding. jsonfile- '{bfFile}','{str(e)}'")
            return -1, f"Error occurred while trying to addFloorsInformationForBuilding. jsonfile- '{bfFile}'"
        except Exception as e:
            nameko_logger.error(f"Error occurred while trying to addFloorsInformationForBuilding. jsonfile- '{bfFile}','{str(e)}'")
            return -1, f"Error occurred while trying to addFloorsInformationForBuilding. jsonfile- '{bfFile}'"

        return -1, f"Error occurred while trying to addFloorsInformationForBuilding. jsonfile- '{bfFile}'"

# 编辑修改楼栋楼层的空间 使用 jsonpatch.JsonPatch 库
# 可一次性添加或者修改多个楼层,也可以同时提交混合的数据.
    @rpc
    def editFloorsInformationForBuilding(self,floorsInfo):

        ''' json 示例
        {
            "organizationCode": "orj001",
            "projectId": "proj001",
            "buildingId": "B001",
            "floors": [
                {
                    "floorId": 1,
                    "buildingSpacesId": ["SPACE001", "SPACE002", "SPACE003", "SPACE004", "SPACE005", "SPACE006"]
                },
                {
                    "floorId": 2,
                    "buildingSpacesId": ["SPACE001", "SPACE002", "SPACE003", "SPACE004", "SPACE005", "SPACE006"]
                },
                {
                    "floorId": 3,
                    "buildingSpacesId": ["SPACE001", "SPACE002", "SPACE003", "SPACE004", "SPACE005", "SPACE006"]
                }
            ]
        }
        '''

        organizationCode = floorsInfo["organizationCode"]
        projectId = floorsInfo["projectId"]
        buildingId = floorsInfo["buildingId"]

        nameko_logger.info(f"organizationCode {organizationCode}-{projectId}-{buildingId}")

        res,bfFile = self.getBuildingFileName(organizationCode, projectId, buildingId)
        if res != 0:
            return res,bfFile

        # load bdFile
        buildingInformation_info  = self.pbLoader.loadJsonFile(bfFile)
        # 提取floors，并修改成patch格式的json
        floors = buildingInformation_info["floors"]

        patchFloors = { "op": "replace", "path": "/floors", "value": floorsInfo["floors"] }

        nameko_logger.info(f"patchFloors ********* '{patchFloors}' ****************")

        try:
            # 生成JSON Patch
            patch = jsonpatch.JsonPatch.from_diff(floors, patchFloors["value"])

            # 应用JSON Patch
            patchData = patch.apply(floors)
            nameko_logger.info(f"patchData ********* '{patchData}' ****************")
            nameko_logger.info(f"floors ********* '{floors}' **************")

            # 验证修改是否正确
            assert patchData == patchFloors["value"]

            #buildingInformation_info["floors"].extend(patchData)
            buildingInformation_info["floors"] = patchData

            res,msg = self.pbLoader.jsonDataIsValid(buildingInformation_info)

            if res == 0:
                return self.pbLoader.dumpJsonFile(bfFile, buildingInformation_info), "editFloorsInformationForBuilding successfully"
            else:
                return res,msg

        except json.JSONDecodeError as e: # the original data or new data is not valid JSON.
            return -1, f"Error occurred while trying to editFloorsInformationForBuilding.'{str(e)}'"
        except KeyError as e: #trying to get a non-existing key from the parsed JSON data.
            return -1, f"Error occurred while trying to editFloorsInformationForBuilding.'{str(e)}'"
        except jsonpatch.JsonPatchException as e: #there is an error in the JSON Patch, such as incorrect format or unable to apply the patch.
            return -1, f"Error occurred while trying to editFloorsInformationForBuilding.'{str(e)}'"
        except AssertionError as e: # the new data generated after applying the patch does not match the expected data.
            return -1, f"Error occurred while trying to editFloorsInformationForBuilding.'{str(e)}'"
        except AttributeError as e: #mistype the name of an attribute, or when the object's class doesn't have the attribute you are looking for.
            return -1, f"Error occurred while trying to editFloorsInformationForBuilding.'{str(e)}'"
        except Exception as e:
            return -1, f"Error occurred while trying to editFloorsInformationForBuilding.'{str(e)}'"

        return -1, f"Error occurred while trying to addFloorsInformationForBuilding. jsonfile- '{bfFile}'"

# 检索楼栋文件名
    @rpc
    def getBuildingFileName(self,organizationCode,projectId,buildingId):
        # <------------- 检索到projectBuiling 文件名 ------------------>
        # 1.在文件 projectsInformation.json 文件中检索到项目的索引文件名，projectBuildingsIndexFile
        # 2. 在 projectBuildingsIndexFile 比如 org001-proj001-Index.json 中 
        # 根据 organizationCode、projectId、buildingId 、检索到对应的文件名 例如 org-proj001-B001.json

        indexFile = self.prjService.getProjectBuildingsIndexFileName(organizationCode,projectId)
        if indexFile is None:
            return -1,f"organizationCode '{organizationCode}' or projectId '{projectId}' not found"

        nameko_logger.info(f"index file  '{indexFile}'")

        # load projectBuildingsIndexFile 文件
        projectBuildingsIndexFile_info  = self.ifLoader.loadJsonFile(indexFile)
        # 检查文件是否存在、是否格式符合要求、是否存在内容
        if projectBuildingsIndexFile_info == 'ERROR_FILE_NOT_FOUND': # 文件不存在，则创建
            #create file && append
            return -1,'ERROR_FILE_NOT_FOUND'
        elif projectBuildingsIndexFile_info == 'ERROR_OBJECTDATA_IN_JSON_FILE': # 文件json 格式不符合 schema 要求
            return -1,'ERROR_OBJECTDATA_IN_JSON_FILE'
        elif (projectBuildingsIndexFile_info is None) or (projectBuildingsIndexFile_info == ''):
            return -1,'ERROR_FILE_IS_EMPTY'

        if projectBuildingsIndexFile_info["organizationCode"] != organizationCode:
            nameko_logger.error(f"Data consistency problem,'{indexFile}' organizationCode != '{organizationCode}'")
            return -3,'ERROR_FILE_DATA_CONSISTENCY_PROBLEM'

        nameko_logger.info(f"indexFile '{indexFile}'")

        nameko_logger.info(projectBuildingsIndexFile_info)

        projects = projectBuildingsIndexFile_info["projects"]

        for building in projects["buildings"]:
            if building["buildingId"] == buildingId:
                return 0, building["buildingInfoFileName"]
        else:
            return -2,'ERROR_NO_DATA_FOUND_IN_PROJECTS'


# 检索项目下所有楼栋的文件名
    @rpc
    def getOneProjectBuildingsFileName(self,organizationCode,projectId):

        indexFile = self.prjService.getProjectBuildingsIndexFileName(organizationCode,projectId)
        if indexFile is None:
            return -1,f"organizationCode '{organizationCode}' or projectId '{projectId}' not found"

        nameko_logger.info(f"index file  '{indexFile}'")

        # load projectBuildingsIndexFile 文件
        projectBuildingsIndexFile_info  = self.ifLoader.loadJsonFile(indexFile)
        # 检查文件是否存在、是否格式符合要求、是否存在内容
        if projectBuildingsIndexFile_info == 'ERROR_FILE_NOT_FOUND': # 文件不存在，则创建
            #create file && append
            return -1,'ERROR_FILE_NOT_FOUND'
        elif projectBuildingsIndexFile_info == 'ERROR_OBJECTDATA_IN_JSON_FILE': # 文件json 格式不符合 schema 要求
            return -1,'ERROR_OBJECTDATA_IN_JSON_FILE'
        elif (projectBuildingsIndexFile_info is None) or (projectBuildingsIndexFile_info == ''):
            return -1,'ERROR_FILE_IS_EMPTY'

        if projectBuildingsIndexFile_info["organizationCode"] != organizationCode:
            nameko_logger.error(f"Data consistency problem,'{indexFile}' organizationCode != '{organizationCode}'")
            return -3,'ERROR_FILE_DATA_CONSISTENCY_PROBLEM'

        return 0,projectBuildingsIndexFile_info["projects"]["buildings"]
#------------------------------------------------------------------------------------------------------------------------------
# 查询某一个项目下的某楼栋完整的详细信息。
    @rpc
    def getOneProjectBuildingInformation(self,buildingInfo):

        organizationCode = buildingInfo["organizationCode"]
        projectId = buildingInfo["projectId"]
        buildingId = buildingInfo["buildingId"]

        nameko_logger.info(f"organizationCode {organizationCode}-{projectId}-{buildingId}")

        res,bfFile = self.getBuildingFileName(organizationCode, projectId, buildingId)
        if res != 0:
            return res,{},'getOneProjectBuildingInformation failed'

        # load bdFile
        buildingInformation_info  = self.pbLoader.loadJsonFile(bfFile)
        return 0,buildingInformation_info,'getOneProjectBuildingInformation successfully'

# 查询某一个项目下所有楼栋完整的详细信息。
    @rpc
    def getOneProjectAllBuildingsInformation(self,buildingInfo):

        organizationCode = buildingInfo["organizationCode"]
        projectId = buildingInfo["projectId"]
#        buildingId = buildingInfo["buildingId"]

        nameko_logger.info(f"organizationCode {organizationCode}-{projectId}")

        res,bfFile = self.getOneProjectBuildingsFileName(organizationCode, projectId)
        if res != 0:
            return res,{},'getOneProjectAllBuildingsInformation failed'

        data = {}
        # load bdFile
        for bf in bfFile:
            data[bf["buildingId"]] =  self.pbLoader.loadJsonFile(bf["buildingInfoFileName"] )
        return 0,data,'getOneProjectAllBuildingsInformation successfully'
