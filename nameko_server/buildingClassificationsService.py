from nameko.rpc import rpc
import json

class buildingClassificationsService:

    name = "buildingClassificationsService"
    buildingClassifications_json = "./json/buildingClassificationsList.json"
    @rpc
    def getBuildingClassificationsList(self):
    # get all buildingClassificationsList information,return all content of ./json/buildingClassificationsList.json
        with open(self.buildingClassifications_json) as f:
            buildingClassifications_info = json.load(f)
        return buildingClassifications_info
