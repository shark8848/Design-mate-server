import os
import subprocess
from multiprocessing import cpu_count
from threading import Thread, Event
import time
from datetime import datetime

from queue import Queue
from nameko.rpc import rpc
import MagicalDatasetProducer_v2 as mdsp
import sys
import json
from DataModel import TaskHistory
sys.path.append("..")
from apocolib import sqliteSession as sqlSession
from apocolib.MlLogger import mlLogger as ml_logger
import pytz
from sqlalchemy import desc


class ProcessManager:
    def __init__(self):
        self.processes = Queue(maxsize=cpu_count() - 1)
        self._stop_event = Event()
        self._monitor_thread = Thread(target=self._monitor)
        self._monitor_thread.start()
        #self.china_tz = pytz.timezone('Asia/Shanghai')


    def _monitor(self):
        while not self._stop_event.is_set():
            if not self.processes.empty():
                p, file_name, out_file, queue_name, task_id = self.processes.queue[0] # Get the first item from the queue
                if p.poll() is not None: # if the process has finished
                    self.processes.get() # remove it from the queue
                    self._save_to_history(task_id, queue_name, file_name, out_file)
            time.sleep(1) # check every second

    def stop(self):
        self._stop_event.set()
        self._monitor_thread.join()

    def add_process(self, p, file_name, out_file, queue_name=None,task_id=None):
        t_id = task_id#next(self.task_id_generator)
        self.processes.put((p, file_name, out_file, queue_name, t_id))
        return {"task_id": t_id, "status": "processing", "input_file_name": file_name, "predicted_file_download_url": out_file}

    def is_queue_full(self):
        return self.processes.full()

    def is_file_in_queue(self, file_name):
        #processes = self.manager.processes.queue
        for _, process_file_name, _, _, _ in self.processes.queue:
            if process_file_name == file_name:
                return True
        return False


    def _save_to_history(self, task_id, queue_name, file_name, out_file):
        session = sqlSession.sqliteSession().getSession()

        try:
            completed_at = datetime.now(pytz.utc).astimezone(pytz.timezone('Asia/Shanghai'))
            task = TaskHistory(
                task_id=task_id,
                queue_name=queue_name,
                file_name=file_name,
                #out_file=out_file,
                out_file=f"{out_file};{out_file}.pdf", # exl;pdf
                status="completed",
                completed_at=completed_at
                #completed_at=datetime.utcnow()
                #datetime.now(pytz.utc).astimezone(pytz.timezone('Asia/Shanghai'))
            )
            session.add(task)
            session.commit()
        except Exception as e:
            session.rollback()
            ml_logger.error(str(e))
            raise e
        finally:
            session.close()

    def get_task_history(self):
        session = sqlSession.sqliteSession().getSession()
        try:
            #tasks = session.query(TaskHistory).all()
            tasks = session.query(TaskHistory).order_by(desc(TaskHistory.created_at)).all()
            history = []
            for task in tasks:
                history.append({
                    "task_id": task.task_id,
                    "queue_name": task.queue_name,
                    "file_name": task.file_name,
                    "out_file": task.out_file,
                    "status": task.status,
                    "created_at": task.created_at,
                    "completed_at": task.completed_at
                })
            return history
        finally:
            session.close()

class HousePredictorService:
    name = "AC2NNetPredicterService"
    base_url = "https://ai.apoco.com.cn/download/"
    manager = ProcessManager()

    @rpc
    def predict(self, file_name):

        if not os.path.exists(file_name):
            return -1, {"status": "error", "message": "File does not exist"}

        if self.manager.is_file_in_queue(file_name) is True:
            return -1, {"status": "error", "message": "File already in queue"}

        if self.manager.is_queue_full():
            return -1,{"status": "error", "message": "Queue is full"}

        out_file = mdsp.generate_excel_filename(file_name)
        r_out_file = f"{self.base_url}{os.path.basename(out_file)}"
        task_id = next(self._task_id_generator())
        time.sleep(3)  # 等待1秒

        ml_logger.info(f"python3 AC2NNetPredicter.py --loadFromFile {file_name} --outFile {out_file} --url {r_out_file} --taskId {task_id}")

        p = subprocess.Popen(["python3", "AC2NNetPredicter.py", "--loadFromFile", file_name, "--outFile", out_file, "--url", r_out_file, "--taskId", task_id])
        return 0, self.manager.add_process(p=p, file_name=file_name, out_file=r_out_file,queue_name=None,task_id=task_id)

    @rpc
    def get_predict_queue_info(self):
        queue_info = []
        for p, file_name, out_file, queue_name, task_id in self.manager.processes.queue:
            queue_info.append({
                "pid": p.pid,
                "input_file": file_name,
                "output_file": out_file,
                "queue_name": queue_name,
                "task_id": task_id
            })
        return {
            "current_size": self.manager.processes.qsize(),
            "max_size": self.manager.processes.maxsize,
            "processes": queue_info
        }
    # AC2NNetPredicter 在申请到消息队列后，获得对应的队列名,再更新队列中的queue_name，
    @rpc
    def assign_queue_name(self, task_id, queue_name):
        for i in range(self.manager.processes.qsize()):
            p, file_name, out_file, old_queue_name, tid = self.manager.processes.queue[i]
            if tid == task_id:
                self.manager.processes.queue[i] = (p, file_name, out_file, queue_name, tid)
                return True
        return False

    @rpc
    def get_task_history(self):
        return 0, self.manager.get_task_history()

    @rpc
    def get_predict_examples(self,file_path='./houses_json/training_dataset_json/examples/examples.list'):

        examples = []
        try:
            with open(file_path, 'r') as file:
                data = file.readlines()
            for line in data:
                examples.append(line.strip())
            return 0,examples
        except FileNotFoundError:
            ml_logger.error(f"Error: File '{file_path}' not found.")
        except IOError:
            ml_logger.error(f"Error: Unable to read file '{file_path}'.")
        except Exception as e:
            ml_logger.error(f"Error: An error occurred while reading the file '{file_path}': {str(e)}")

        return -1,examples

    @rpc
    def get_predict_json_content(self,file_path=None):
        examples = {}
        try:
            with open(file_path, 'r') as file:
                data = file.read()
                examples = json.loads(data)
            return 0, examples
        except FileNotFoundError:
            ml_logger.error(f"Error: File '{file_path}' not found.")
        except IOError:
            ml_logger.error(f"Error: Unable to read file '{file_path}'.")
        except json.JSONDecodeError as e:
            ml_logger.error(f"Error: Failed to decode JSON in file '{file_path}': {str(e)}")
        except Exception as e:
            ml_logger.error(f"Error: An error occurred while reading the file '{file_path}': {str(e)}")

        return -1, examples

    def _task_id_generator(self):
        while True:
            now = datetime.now()
            # 将微秒转换为毫秒
            milliseconds = int(now.microsecond / 1000)
            # 格式化年月日时分秒，然后将毫秒添加到字符串的末尾
            yield now.strftime("%Y%m%d%H%M%S") + "{:03d}".format(milliseconds)


