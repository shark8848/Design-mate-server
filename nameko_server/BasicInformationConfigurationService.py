from nameko.rpc import rpc
import json

class BasicInformationConfigurationService:

    name = "BasicInformationConfigurationService"
    bst_json = "./json/buildingStructureTypeInformation.json"
    ort_json = "./json/Orientation.json"
    bst_info = None
    ort_info = None

    def __init__(self):
        with open(self.bst_json) as f:
            self.bst_info = json.load(f)

        with open(self.ort_json) as f:
            self.ort_info = json.load(f)

    @rpc
    def getBuildingStructureTypeInformation(self):
        return self.bst_info

    @rpc
    def getOrientationInformation(self):
        return self.ort_info
