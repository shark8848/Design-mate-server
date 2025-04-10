# -*- coding: utf-8 -*-
"""
Program: verifyJsonFileLoader.py
Description: 根据模板校验jsonfile
Author: Sunhy
Version: 1.0
Date: 2023-02-24
"""

import json,sys,jsonschema
from jsonschema import Draft7Validator
import fcntl
import os
sys.path.append("..")
#from apocolib.apocolog4p import apoLogger as apolog
from apocolib.NamekoLogger import namekoLogger as nameko_logger

class verifyJsonFileLoader:

    schema = None
    validator = None

    def __init__(self,json_file,schema_file):

        self.json_file = json_file
        self.schema_file = schema_file

        #nameko_logger.info("init buildingSpaceCompositionInformationService")
        try:

            with open(self.schema_file) as schFile:
                self.schema = json.load(schFile)
                #nameko_logger.info(f"file '{self.schema_file}' load successfully")

            self.validator = Draft7Validator(self.schema)
            #nameko_logger.info(f"file '{self.schema_file}'  Schema format validation succeefully")
        except jsonschema.exceptions.ValidationError as e:
            nameko_logger.error(f"file '{self.schema_file}' ,Invalid JSON object: {e.message}")
        except FileNotFoundError as e:
            nameko_logger.error(f"file '{self.schema_file}' not found")
        except jsonschema.exceptions.SchemaError as e:
            nameko_logger.error(f"file '{self.schema_file}' invalid Json schema, {e.message}")
        except json.JSONDecodeError as e:
            nameko_logger.error(f"file '{self.schema_file}' load failed,{e}")

# 加载配置数据文件
    def loadJsonFile(self,json_file):
        try:
            with open(json_file,"r") as f:
               json_data  = json.load(f)

            if json_data is not None:
                self.validator.validate(json_data)

            return json_data

        except jsonschema.exceptions.ValidationError as e:
            nameko_logger.error(f"file '{json_file}' ,Invalid JSON object: {e.message}")
            return 'ERROR_OBJECTDATA_IN_JSON_FILE'
        except FileNotFoundError as e:
            nameko_logger.error(f"file '{json_file}' not found")
            return 'ERROR_FILE_NOT_FOUND'
        except json.JSONDecodeError as e:
            nameko_logger.error(f"failed to load JSON data from file '{json_file}': {e}")
        except Exception as e:
            nameko_logger.error(f"failed to load JSON data from file '{json_file}': {e}")

        return None

# dump 配置数据文件
    def dumpJsonFile(self,json_file,data):
        try:
            with open(json_file,"w") as f:

                # 获取文件锁
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
               #json.dump(data,f,indent=4)
                json.dump(data,f,indent=4,ensure_ascii=False)

                # 释放文件锁
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

            return 0
        except IOError as e:
            nameko_logger.error(f"file '{json_file}' write error")
        except TypeError as e:
            nameko_logger.error(f"Data cannot be serialized to JSON string: {e}")
        return -1

# 校验 配置数据是否符合schema 格式要求
    def jsonDataIsValid(self,json_data):
        if json_data is not None:
            try:
                self.validator.validate(json_data)
                return 0, "data is valid"
            except jsonschema.exceptions.ValidationError as e:
                nameko_logger.error(f"Invalid JSON object: {e.message},error data: {e.instance}")
                return -1, f"Invalid JSON object: {e.message},error data: {e.instance}"
            except Exception as e:
                return -2, f"json data validate failed,{str(e)}"
        else:
            return -1, "json data is empty"

# 根据属性检索内容
    def searchInJson(json_data, property_name, property_value):
        """
        搜索json对象，查找是否存在符合条件的属性
        :param json_data: json对象
        :param property_name: 属性名称
        :param property_value: 属性值
        :return: True: 存在符合条件的属性; False: 不存在符合条件的属性
        """
        if isinstance(json_data, dict):
            if property_name in json_data and json_data[property_name] == property_value:
                return True
            for value in json_data.values():
                if search_json(value, property_name, property_value):
                    return True
        elif isinstance(json_data, list):
            for item in json_data:
                if search_json(item, property_name, property_value):
                    return True
        return False
