from nameko.rpc import rpc
import MagicalDatasetProducer_v2 as mdsp
import sys
sys.path.append("..")
from apocolib.MlLogger import mlLogger as ml_logger

class DataSetService:
    name = "DataSetService"
    houses_x = None
    houses_y = None

    def __init__(self, size=1000, model='reload'):
        self.size = size
        self.model = model
        self.load_dataset()

    def load_dataset(self):
        if DataSetService.houses_x is None or DataSetService.houses_y is None:
            print("Load dataset into memory")
            DataSetService.houses_x, DataSetService.houses_y = mdsp.generate_dataset(size=self.size, model=self.model)

    @rpc
    def get_dataset(self, size=1000, model='reload'):
        if model != 'reload':
            self.load_dataset()
        return DataSetService.houses_x, DataSetService.houses_y
