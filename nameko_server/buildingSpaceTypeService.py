from nameko.rpc import rpc
import json

class buildingSpaceTypeService:

    name = "buildingSpaceTypeService"
    buildingSpaceTypeList_json = "./json/buildingSpaceTypeList.json"
    @rpc
    def getBuildingSpaceTypeList(self):
    # get all buildingSpaceTypeList information,return all content of buildingSpaceTypeList.json
        with open(self.buildingSpaceTypeList_json) as f:
            buildingSpaceTypeList_info = json.load(f)
        return buildingSpaceTypeList_info
