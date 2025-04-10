# -*- coding: utf-8 -*-
"""
Program: fileRetriever.py
Description: 从索引文件中检索到存储楼栋信息的文件名
Author: Sunhy
Version: 1.0
Date: 2023-02-24
"""

import json,sys,jsonschema
from jsonschema import Draft7Validator
sys.path.append("..")
#from apocolib.apocolog4p import apoLogger as apolog
from apocolib.NamekoLogger import namekoLogger as nameko_logger

class indexFileRetriever:

    def __init__(self):

        self.schema_file = "./json/json_schema/projectBuildingFilesIndexSchema.json"
        self.index_file = "./json/projectBuildingFilesIndex.json"
        self.validator = None
        self.schema = None

    def retrieveFileName(self,org_code,project_id,building_id):

        try:
            # load schema file
            with open(self.schema_file) as schFile:
                self.schema = json.load(schFile)
                nameko_logger.info(f"file '{self.schema_file}' load successfully")

            self.validator = Draft7Validator(self.schema)
            # 读取 JSON 文件并解析为 Python 对象
            with open(self.index_file) as indexFile:
                data = json.load(indexFile)
                self.validator.validate(data)

            # 检索文件名
            for project in data['projects']:
                if project['projectId'] == project_id:
                    for building in project['buildings']:
                        if building['buildingId'] == building_id:
                            return 0,building['buildingInfoFileName'],"retrieveFileName successfully"

            return -1,None,"no file"

        except jsonschema.exceptions.ValidationError as e:
            nameko_logger.error(f"file '{self.schema_file}' ,Invalid JSON object: {e.message}")
            return -1,None,f"file '{self.schema_file}' ,Invalid JSON object: {e.message}"
        except FileNotFoundError as e:
            nameko_logger.error(f"file '{self.schema_file}' not found")
            return -1,None,f"file '{self.schema_file}' not found"
        except jsonschema.exceptions.SchemaError as e:
            nameko_logger.error(f"file '{self.schema_file}' invalid Json schema, {e.message}")
            return -1,None,f"file '{self.schema_file}' invalid Json schema, {e.message}"
        except json.JSONDecodeError as e:
            nameko_logger.error(f"file '{self.schema_file}' load failed,{e}")
            return -1,None,f"file '{self.schema_file}' load failed,{e}"

        return -1,None,"no file"
