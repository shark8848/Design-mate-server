import numpy as np
import time
from nameko.rpc import rpc
from nameko.standalone.rpc import ClusterRpcProxy
from nameko.standalone.events import event_dispatcher
from nameko.constants import NON_PERSISTENT, PERSISTENT
from nameko.events import EventDispatcher, event_handler, Publisher
from nameko.timer import timer
from enum import Enum
import sys
import json

sys.path.append("..")
#from apocolib.apocolog4p import apoLogger as apolog
from apocolib.MlLogger import mlLogger as ml_logger
from apocolib import apocoIAServerConfigurationManager as iaConMg
from nameko_server import jsonDataValidator,verifyJsonFileLoader
from nameko.events import EventDispatcher
from mlService import MLService


class MultiJobFrameService:
    name = "multiJobFrameService"
    #  Job schema, 用于存储所有job信息的校验
    jobs_schema = "./json/json_schema/jobsSchema.json"

    # Job Template schema,用于单个job 管理时信息的校验
    jobTemplate_schema = "./json/json_schema/jobTemplateSchema.json"

    dispatch = EventDispatcher()

    jobsValidator = None
    jobTempValidator = None

    def __init__(self):
        self.jobsValidator = jsonDataValidator.jsonDataValidator(self.jobs_schema)
        self.jobTempValidator = jsonDataValidator.jsonDataValidator(self.jobTemplate_schema)
        self.jobsLoader = verifyJsonFileLoader.verifyJsonFileLoader(None,self.jobs_schema)


    @rpc
    def submitJob(self, jobParams):
        
        ml_logger.info(f"submit job,{jobParams}")
        #校验job参数格式是否有效
        res,msg = self.jobTempValidator.jsonDataIsValid(jobParams)
        ml_logger.info(f"return '{res}' msg '{msg}'")
        if res != 0:
            return res,msg
        try:
            res,data,msg = MLService().runJob(jobParams)
            self.dispatch("jobSubmitted", {"jobId": data["jobId"], "jobParams": jobParams})
        except Exception as e:
            ml_logger.error(f"Error occurred while sumbit one job.'{str(e)}'")
            return -1, (f"Error occurred while sumbit one job.'{str(e)}'")

        return 0,"sumbmit job successfully"

    @rpc
    def getJobStatus(self, jobId):
        status = MLService().getJobStatus(jobId)
        return 0,{"jobId":jobId,"status":status},None

    @rpc
    def getJobList(self, organizationCode):
        res,data,msg = MLService().getJobList(organizationCode)
        return res,data,msg

    @rpc
    def getJobListiByPage(self, organizationCode,page,pageSize):
        res,data,msg = MLService().getJobList(organizationCode)
        if res != 0:
            return res,data,msg

        data = data[(page-1)*pageSize:page*pageSize]

        return res,data,msg

    @rpc
    def cancelJob(self, jobId):
        MLService().cancelJob(jobId)
        self.dispatch("jobCancelled", {"jobId": jobId})

    @timer(interval=60)
    def checkJobStatus(self):
        jobIds = MLService().getAllJobIds()
        for jobId in jobIds:
            status = MLService().getJobStatus(jobId)
            if status == "failed":
                self.dispatch("jobFailed", {"jobId": jobId})
            elif status == "completed":
                self.dispatch("jobCompleted", {"jobId": jobId})

