from nameko.events import event_handler
import sys

sys.path.append("..")
#from apocolib.apocolog4p import apoLogger as apolog
from apocolib.MlLogger import mlLogger as ml_logger
from mlService import MLService

class EventHandler:

    name = "eventHandler"

    @event_handler("multiJobFrameService", "jobSubmitted")
    def handleSubmittedJob(self, event):
        jobId = event["jobId"] #event.args[0]
        status = "submitted"
        MLService().updateJobJson(jobId,status,None)
        #ml_logger.info(f"Job {jobId} has been {status}")

    @event_handler("multiJobFrameService", "jobStarted")
    def handleStartedJob(self, event):
        #jobId = event.args[0]
        jobId = event["jobId"] #event.args[0]
        status = "started"
        MLService().updateJobJson(jobId,status,None)
        #ml_logger.info(f"Job {jobId} has been {status}")

    @event_handler("multiJobFrameService", "jobCancelled")
    def handleCancelledJob(self, event):
        #jobId = event.args[0]
        jobId = event["jobId"] #event.args[0]
        status = "cancelled"
        MLService().updateJobJson(jobId,status,None)
        #ml_logger.info(f"Job {jobId} has been {status}")

    @event_handler("multiJobFrameService", "jobCompleted")
    def handleCompletedJob(self, event):
        #jobId = event.args[0]
        jobId = event["jobId"] #event.args[0]
        status = "completed"
        MLService().updateJobJson(jobId,status,None)
        #ml_logger.info(f"Job {jobId} has been {status}")

