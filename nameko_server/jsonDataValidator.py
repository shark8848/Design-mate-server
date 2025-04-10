# -*- coding: utf-8 -*-
"""
Program: jsonDataValidator.py
Description: 根据模板校验jsondata
Author: Sunhy
Version: 1.0
Date: 2023-03-05
"""

import json,sys,jsonschema
from jsonschema import Draft7Validator
sys.path.append("..")
#from apocolib.apocolog4p import apoLogger as apolog
from apocolib.NamekoLogger import namekoLogger as nameko_logger

class jsonDataValidator:

    schema = None
    validator = None

    def __init__(self,schema_file):

        self.schema_file = schema_file
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

# 校验 json 数据是否符合schema 格式要求
    def jsonDataIsValid(self,json_data):
        if json_data is not None:
            try:
                self.validator.validate(json_data)
                return 0, "data is valid"
            except jsonschema.exceptions.ValidationError as e:
                nameko_logger.error(f"'{json_data}' ,Invalid JSON object: {e.message}")
                return -1, f"Invalid JSON object: {e.message}"
            except Exception as e:
                return -2, f"json data validate failed,{str(e)}"
        else:
            return -1, "json data is empty"
