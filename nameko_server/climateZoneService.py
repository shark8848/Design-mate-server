from nameko.rpc import rpc
import json
import pdb

class climateZoneService:

    name = "climateZoneService"
    climateZone_json = "./json/climateZone.json"
    @rpc
    def getClimateZone(self):
        #pdb.set_trace()
        with open(self.climateZone_json) as f:
            climateZone_info = json.load(f)
        return climateZone_info
