from nameko.rpc import RpcProxy
from nameko.events import EventDispatcher
from nameko.timer import timer
import threading
import random
import string
import sys
import datetime

sys.path.append("..")
#from apocolib.apocolog4p import apoLogger as apolog
from apocolib.MlLogger import mlLogger as ml_logger
from nameko_server import jsonDataValidator,verifyJsonFileLoader

class MLService:

    name = "mlService"
    #  Job schema, 用于存储所有job信息的校验
    jobs_schema = "./json/json_schema/jobsSchema.json"

    # Job Template schema,用于单个job 管理时信息的校验
    jobTemplate_schema = "./json/json_schema/jobTemplateSchema.json"

    jobs_json = "./json/jobs.json"

    jobsValidator = None
    jobTempValidator = None

    dispatch = EventDispatcher()
    jobStatus = {}
    eventHandler = RpcProxy("eventHandler")

    def __init__(self):

        self.result = []

        self.jobsValidator = jsonDataValidator.jsonDataValidator(self.jobs_schema)
        self.jobTempValidator = jsonDataValidator.jsonDataValidator(self.jobTemplate_schema)
        self.jobsLoader = verifyJsonFileLoader.verifyJsonFileLoader(self.jobs_json, self.jobs_schema)


    def runJob(self, new_job):
        jobId = self.generateJobId()
        self.jobStatus[jobId] = "running"
        # update jobParams
        # do something
        # write job info into jobs.json
        # do something
        # load 系统jobs 文件

        jobs_info  = self.jobsLoader.loadJsonFile(self.jobs_json)

         # 检查文件是否存在、是否格式符合要求、是否存在内容
        if jobs_info == 'ERROR_OBJECTDATA_IN_JSON_FILE': # 文件json 格式不符合 schema 要求
            return -1,{},'ERROR_OBJECTDATA_IN_JSON_FILE'
        elif jobs_info == 'ERROR_FILE_NOT_FOUND': # 文件不存在
            jobs_info = {"jobs":[]}
        elif jobs_info is None :
            jobs_info = {"jobs":[]}

        new_job["job_id"] = jobId
        new_job["status"] = "submitted"
        new_job["submit_time"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        jobs_info["jobs"].append(new_job)
        try:

            res,msg = self.jobsValidator.jsonDataIsValid(jobs_info)
            if res == 0:
                res = self.jobsLoader.dumpJsonFile(self.jobs_json,jobs_info)
                if res == -1:
                    ml_logger.error(f"Error occurred while trying to insert an object in the json file. '{self.jobs_json}'")
                    return -1,{},f"Error occurred while trying to insert an object in the json file. '{self.jobs_json}'"
            else:
                ml_logger.error(msg)
                return res, {},msg
        except AttributeError as e:
            ml_logger.error(f"Error occurred while trying to insert an object in the json file. '{self.jobs_json}','{str(e)}'")
            return -1, f"Error occurred while trying to insert an object in the json file. '{self.jobs_json}'"

        # Call machine learning calculation service
        result = self.calculate(new_job)

        if result == "error":
            self.jobStatus[jobId] = "failed"
        else:
            self.jobStatus[jobId] = "completed"

        #return -2, f"runJob failed"

        return 0,{"jobId":jobId,"status":self.jobStatus[jobId]},"run job successfully"

    def updateJobJson(self,jobId,status,msg):
        #do something
        jobs_info  = self.jobsLoader.loadJsonFile(self.jobs_json)

         # 检查文件是否存在、是否格式符合要求、是否存在内容
        if jobs_info == 'ERROR_OBJECTDATA_IN_JSON_FILE': # 文件json 格式不符合 schema 要求
            return -1,'ERROR_OBJECTDATA_IN_JSON_FILE'
        elif jobs_info == 'ERROR_FILE_NOT_FOUND': # 文件不存在
            return -1,'ERROR_FILE_NOT_FOUND'
        elif jobs_info is None :
            return -1,'ERROR_FILE_IS_EMPTY'

        for i,job in enumerate(jobs_info["jobs"]):
            if job["job_id"] == jobId:
                job["status"] = status
                job["error_massage"] = msg
                if status == 'completed' or status == 'cancelled':
                    job["end_time"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                jobs_info["jobs"][i] = job
                break

        try:
            res,msg = self.jobsValidator.jsonDataIsValid(jobs_info)
            if res == 0:
                res = self.jobsLoader.dumpJsonFile(self.jobs_json,jobs_info)
                if res == -1:
                    ml_logger.error(f"Error occurred while trying to insert an object in the json file. '{self.jobs_json}'")
                    return -1,{},f"Error occurred while trying to insert an object in the json file. '{self.jobs_json}'"
            else:
                ml_logger.error(msg)
                return res, {},msg
        except AttributeError as e:
            ml_logger.error(f"Error occurred while trying to insert an object in the json file. '{self.jobs_json}','{str(e)}'")
            return -1, f"Error occurred while trying to insert an object in the json file. '{self.jobs_json}'"

        return 0, "update jobsJson successfully"

    def getJobStatus(self, jobId):
        return self.jobStatus.get(jobId, "not_found")

    def getJobList(self,organizationCode):

        jobs_info  = self.jobsLoader.loadJsonFile(self.jobs_json)

         # 检查文件是否存在、是否格式符合要求、是否存在内容
        if jobs_info == 'ERROR_OBJECTDATA_IN_JSON_FILE': # 文件json 格式不符合 schema 要求
            return -1,{},'ERROR_OBJECTDATA_IN_JSON_FILE'
        elif jobs_info == 'ERROR_FILE_NOT_FOUND': # 文件不存在
            return -1,{},'ERROR_FILE_NOT_FOUND'
        elif jobs_info is None :
            return -1,{},'ERROR_FILE_IS_EMPTY'

        data = []
        for job in jobs_info["jobs"]:
            if job["organizationCode"] == organizationCode:
                data.append(job)

        return 0,data,'None'


    def cancelJob(self, jobId):
        if self.jobStatus.get(jobId) == "running":
            self.jobStatus[jobId] = "cancelled"

    def getAllJobIds(self):
        return list(self.jobStatus.keys())

    def generateJobId(self):
        # Generate unique job ID
        idLength = 20
        chars = string.ascii_uppercase + string.digits
        return ''.join(random.choice(chars) for _ in range(idLength))

    def calculate(self, jobParams):
        # Call machine learning calculation service asynchronously
        for i in range(10):
            t = threading.Thread(target=self._calculateJob, args=(jobParams, i))
            t.start()

    @timer(interval=60)
    def checkJobStatus(self):
        for jobId, status in self.jobStatus.items():
            if status == "cancelled":
                self.eventHandler.handleCancelledJob(jobId)
                del self.jobStatus[jobId]

    def _calculateJob(self, jobParams, jobIndex):
        ml_logger.info(f"Starting calculation job {jobIndex + 1}...")
        # Call machine learning calculation service synchronously
        # ...
        # Store result in self.result
        # ...
        ml_logger.info(f"Calculation job {jobIndex + 1} completed!")
